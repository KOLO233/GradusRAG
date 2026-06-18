"""GradusRAG FastAPI 应用入口。

提供 RESTful API 和 SSE 流式接口。
所有组件在此组装，一个入口启动整个后端。
"""

from __future__ import annotations

import json
import logging
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

from src.core.settings import load_settings
from src.core.trace import TraceCollector
from src.query_classifier.classifier import create_classifier
from src.generation.pipeline import RAGPipeline
from src.generation.document_grader import DocumentGrader
from src.generation.query_rewriter import QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_search import HybridSearch
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.graph_builder.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI 实例
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GradusRAG",
    description="面向专业领域的多级查询理解与混合增强 RAG 问答系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 全局组件（启动时初始化）
# ---------------------------------------------------------------------------
settings = load_settings()
trace_collector = TraceCollector(
    trace_file=settings.observability.trace_file
    if settings.observability.trace_enabled
    else None
)

# Embedding
embedding_service = EmbeddingService(
    model_name=settings.embedding.model,
    device=settings.embedding.device,
    dimensions=settings.embedding.dimensions,
    api_key=settings.embedding.api_key,
    api_base_url=settings.embedding.api_base_url,
)

# Milvus
milvus_store = MilvusStore(
    host=settings.vector_store.host,
    port=settings.vector_store.port,
    collection=settings.vector_store.collection,
)

# Retriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import Reranker
from src.ingestion.graph_builder.graph_store import GraphStore

dense_retriever = DenseRetriever(embedding_service, milvus_store)
sparse_retriever = SparseRetriever(embedding_service, milvus_store)

# LLM（提前初始化，Reranker 需要）
llm_service = LLMService.from_settings(settings)

# Graph
graph_store = GraphStore(persist_path="data/knowledge_graph.json")
graph_retriever = GraphRetriever(graph_store, embedding_service) if settings.graph.enabled else None

# Reranker
reranker = Reranker(llm_service=llm_service)

hybrid_search = HybridSearch(
    settings=settings,
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    graph_retriever=graph_retriever,
    reranker=reranker,
)

# Graph Builder（摄取时自动构建图谱）
graph_builder = GraphBuilder(
    settings=settings.graph,
    llm_service=llm_service,
    graph_store=graph_store,
) if settings.graph.enabled else None

# LLM（已提前初始化）

# Generation components
classifier = create_classifier(settings.query_classifier)
grader = DocumentGrader(llm_service)
rewriter = QueryRewriter(llm_service)
generator = ResponseGenerator(llm_service, hybrid_search)

# Pipeline
pipeline = RAGPipeline(
    settings=settings,
    classifier=classifier,
    hybrid_search=hybrid_search,
    grader=grader,
    rewriter=rewriter,
    generator=generator,
    trace_collector=trace_collector,
)

# Ingestion
ingestion = IngestionPipeline(
    settings,
    embedding_service=embedding_service,
    milvus_store=milvus_store,
    graph_builder=graph_builder,
)

# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    stream: bool = Field(False, description="是否使用流式输出")


class QueryResponse(BaseModel):
    answer: str
    citations: list
    query_level: str
    query_type: str
    confidence: float
    retrieval_trace: dict
    generation_trace: dict


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="文件路径")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    llm: str
    embedding: str
    vector_store: str


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        llm=f"{settings.llm.provider}/{settings.llm.model}",
        embedding=f"{settings.embedding.provider}/{settings.embedding.model}",
        vector_store=f"{settings.vector_store.provider}/{settings.vector_store.host}:{settings.vector_store.port}",
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询接口（非流式）。"""
    try:
        response = await pipeline.run(request.question)
        return QueryResponse(
            answer=response.answer,
            citations=[c.to_dict() for c in response.citations],
            query_level=response.query_level,
            query_type=response.query_classification.query_type if response.query_classification else "",
            confidence=response.query_classification.confidence if response.query_classification else 0.0,
            retrieval_trace=response.retrieval_trace,
            generation_trace=response.generation_trace,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """查询接口（SSE 流式输出，实时推送 RAG 过程步骤）。"""

    import asyncio

    # 创建事件队列
    event_queue: asyncio.Queue = asyncio.Queue()

    def on_pipeline_event(event: dict):
        """Pipeline 回调：将事件推入队列。"""
        try:
            event_queue.put_nowait(event)
        except Exception:
            pass

    # 临时设置 pipeline 的事件回调
    original_callback = pipeline._event_callback
    pipeline._event_callback = on_pipeline_event

    async def event_generator():
        try:
            # 启动 pipeline（后台任务）
            async def run_pipeline():
                try:
                    result = await pipeline.run(request.question)
                    await event_queue.put({"type": "result", "data": result})
                except Exception as e:
                    await event_queue.put({"type": "error", "message": str(e)})
                finally:
                    await event_queue.put({"type": "done"})
                    pipeline._event_callback = original_callback

            import asyncio as aio
            task = aio.create_task(run_pipeline())

            # 从队列中读取事件并 yield
            while True:
                event = await event_queue.get()

                if event.get("type") == "done":
                    yield "data: [DONE]\n\n"
                    break
                elif event.get("type") == "error":
                    yield _sse({"type": "error", "message": event.get("message", "")})
                    break
                elif event.get("type") == "result":
                    # 最终结果
                    response = event["data"]
                    yield _sse({"type": "content", "text": response.answer})
                    yield _sse({
                        "type": "citation",
                        "citations": [c.to_dict() for c in response.citations],
                    })
                else:
                    # Pipeline 步骤事件（实时推送）
                    yield _sse({"type": "step", "step": event})

        except Exception as e:
            logger.error(f"Stream query failed: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/documents/ingest")
async def ingest_document(request: IngestRequest):
    """摄取文档到知识库。"""
    try:
        chunks, parents = ingestion.ingest_file(request.file_path)
        return {
            "status": "ok",
            "chunks": len(chunks),
            "parents": len(parents),
            "file": request.file_path,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile):
    """上传并摄取文档到知识库。

    支持格式：.pdf, .docx, .doc, .xlsx, .xls, .txt, .html, .md
    """
    import tempfile
    import shutil

    # 检查文件类型
    supported = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt", ".html", ".htm", ".md", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported)}",
        )

    # 保存到临时文件
    upload_dir = Path("data/documents")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 摄取
        chunks, parents = ingestion.ingest_file(str(file_path))

        # 如果有 embedding + milvus，写入向量库
        if ingestion._embedding and ingestion._store:
            total, _ = ingestion.ingest_file_to_milvus(str(file_path))
            return {
                "status": "ok",
                "file": file.filename,
                "chunks": total,
                "parents": len(parents),
                "message": f"成功上传并入库 {total} 个分块",
            }

        return {
            "status": "ok",
            "file": file.filename,
            "chunks": len(chunks),
            "parents": len(parents),
            "message": f"成功上传 {len(chunks)} 个分块（未入库向量库）",
        }

    except Exception as e:
        # 清理失败的文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """列出已入库的文档。"""
    upload_dir = Path("data/documents")
    if not upload_dir.exists():
        return {"documents": []}

    docs = []
    for f in sorted(upload_dir.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            stat = f.stat()
            docs.append({
                "filename": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "suffix": f.suffix.lower(),
            })
    return {"documents": docs}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """删除文档及所有关联数据（向量、图谱、BM25 统计）。"""
    file_path = Path("data/documents") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    cleanup = {"file_deleted": False, "vectors_deleted": 0, "graph_nodes_removed": 0, "bm25_reset": False}

    # 1. 从 Milvus 删除向量
    try:
        cleanup["vectors_deleted"] = milvus_store.delete_by_filename(filename)
    except Exception:
        pass

    # 2. 从知识图谱中移除该文件来源的实体和关系
    try:
        entities_before = len(graph_store.get_all_entities())
        graph_store.remove_by_source(filename)
        entities_after = len(graph_store.get_all_entities())
        cleanup["graph_nodes_removed"] = entities_before - entities_after
        graph_store.save()
    except Exception:
        pass

    # 3. 重置 BM25 统计（如果删除的是最后一个文件）
    try:
        remaining = list(Path("data/documents").glob("*"))
        remaining = [f for f in remaining if f.is_file() and not f.name.startswith(".")]
        if not remaining:
            # 没有剩余文档，清空 BM25 状态
            bm25_path = Path("data/bm25_state.json")
            if bm25_path.exists():
                bm25_path.unlink()
            cleanup["bm25_reset"] = True
    except Exception:
        pass

    # 4. 删除文件
    file_path.unlink()
    cleanup["file_deleted"] = True

    return {
        "status": "ok",
        "file": filename,
        "cleanup": cleanup,
        "message": f"已删除 {filename}（{cleanup['vectors_deleted']} 条向量, {cleanup['graph_nodes_removed']} 个图谱节点）",
    }


@app.get("/api/documents/stats")
async def document_stats():
    """获取文档库统计信息。"""
    upload_dir = Path("data/documents")
    file_count = 0
    total_size = 0
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                file_count += 1
                total_size += f.stat().st_size

    graph_stats = graph_store.stats()

    return {
        "file_count": file_count,
        "total_size_kb": round(total_size / 1024, 1),
        "graph_entities": graph_stats.get("entity_count", 0),
        "graph_relations": graph_stats.get("relation_count", 0),
    }


@app.get("/api/traces")
async def get_traces(limit: int = 20):
    return {"traces": trace_collector.get_recent_traces(limit=limit)}


@app.get("/api/traces/{query_id}")
async def get_trace(query_id: str):
    trace = trace_collector.get_trace(query_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.to_dict()


@app.get("/api/graph/stats")
async def graph_stats():
    """获取知识图谱统计信息。"""
    return graph_store.stats()


@app.post("/api/graph/rebuild")
async def rebuild_graph():
    """从已有 Milvus 数据重建知识图谱。

    用于：首次启用图谱、图谱数据丢失后恢复、更新实体抽取逻辑后重建。
    """
    if graph_builder is None:
        raise HTTPException(status_code=400, detail="Graph is disabled in settings")

    try:
        # 从 Milvus 获取所有文档分块
        client = milvus_store._get_client()
        all_docs = client.query(
            milvus_store._collection,
            output_fields=["chunk_id", "text"],
            limit=16384,
        )

        if not all_docs:
            return {"status": "ok", "message": "No documents in Milvus", "entities": 0, "relations": 0}

        # 转为 Chunk 对象
        from src.core.types import Chunk
        chunks = []
        for doc in all_docs:
            chunks.append(Chunk(
                id=doc.get("chunk_id", ""),
                text=doc.get("text", ""),
                metadata={"source_path": "rebuilt"},
            ))

        # 清空旧图并重建
        graph_store._graph = None
        from pathlib import Path
        kg_path = Path("data/knowledge_graph.json")
        if kg_path.exists():
            kg_path.unlink()

        import asyncio
        stats = asyncio.run(graph_builder.build_from_chunks(chunks))

        return {
            "status": "ok",
            "chunks_processed": stats.get("processed_chunks", 0),
            "entities": stats.get("total_entities", 0),
            "relations": stats.get("total_relations", 0),
        }
    except Exception as e:
        logger.error(f"Graph rebuild failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/entities")
async def graph_entities(keyword: str = ""):
    """搜索或列出实体。"""
    if keyword:
        return {"entities": graph_store.search_entities(keyword)}
    return {"entities": graph_store.get_all_entities()}


@app.get("/api/graph/neighbors/{entity_name}")
async def graph_neighbors(entity_name: str, hops: int = 1):
    """获取实体的多跳邻居。"""
    result = graph_store.get_neighbors(entity_name, hops=hops)
    return result


@app.get("/api/graph/data")
async def graph_data():
    """获取完整图数据（用于前端可视化）。"""
    return {
        "nodes": [
            {"id": e["name"], "label": e["name"], "type": e.get("entity_type", "")}
            for e in graph_store.get_all_entities()
        ],
        "edges": [
            {"source": r["source"], "target": r["target"], "label": r.get("relation_type", "")}
            for r in graph_store.get_all_relations()
        ],
    }


@app.get("/api/bm25/stats")
async def bm25_stats():
    """获取 BM25 倒排索引统计信息。"""
    stats = embedding_service.bm25_index_stats()
    return {
        "index_tokens": stats["tokens"],
        "index_documents": stats["documents"],
        "total_postings": stats["total_postings"],
        "has_index": stats["tokens"] > 0,
    }


@app.post("/api/bm25/rebuild")
async def rebuild_bm25_index():
    """从 Milvus 数据重建 BM25 倒排索引。

    用于：首次启用倒排索引、索引数据丢失后恢复。
    """
    try:
        client = milvus_store._get_client()
        all_docs = client.query(
            milvus_store._collection,
            output_fields=["chunk_id", "text", "filename", "source_path",
                           "page", "chunk_index", "chunk_level",
                           "parent_chunk_id", "root_chunk_id"],
            limit=16384,
        )

        if not all_docs:
            return {"status": "ok", "message": "No documents in Milvus", "indexed": 0}

        # 清空旧索引
        embedding_service._inverted_index.clear()
        embedding_service._doc_meta.clear()

        # 重建
        chunk_ids = [d["chunk_id"] for d in all_docs]
        texts = [d["text"] for d in all_docs]
        metas = [{k: v for k, v in d.items() if k != "text"} for d in all_docs]

        embedding_service.bm25_increment_add(texts)
        embedding_service.bm25_index_add(chunk_ids, texts, metas)

        stats = embedding_service.bm25_index_stats()
        return {
            "status": "ok",
            "indexed": stats["documents"],
            "tokens": stats["tokens"],
        }
    except Exception as e:
        logger.error(f"BM25 index rebuild failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
