"""命令行查询脚本。

用法：
    python scripts/query.py "什么是机器学习？"
    python scripts/query.py --level L3 "为什么会出现梯度消失？"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GradusRAG 查询")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--level", default=None, help="强制指定查询级别 (L1/L2/L3/L4)")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM（纯检索模式）")
    args = parser.parse_args()

    settings = load_settings()
    print(f"Settings: LLM={settings.llm.provider}/{settings.llm.model}")
    print(f"Embedding: {settings.embedding.model}")
    print(f"Milvus: {settings.vector_store.host}:{settings.vector_store.port}")
    print(f"Query: {args.question}")
    print("=" * 60)

    # 初始化组件
    trace_collector = TraceCollector()
    classifier = create_classifier(settings.query_classifier)

    # LLM（可选）
    llm_service = None
    if not args.no_llm:
        try:
            llm_service = LLMService.from_settings(settings)
        except Exception as e:
            print(f"Warning: LLM init failed ({e}), using fallback mode")

    grader = DocumentGrader(llm_service) if llm_service else DocumentGrader()
    rewriter = QueryRewriter(llm_service) if llm_service else QueryRewriter()

    # 检索器
    embedding = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    dense_retriever = DenseRetriever(embedding, store)

    # Sparse Retriever（BM25）
    from src.retrieval.sparse_retriever import SparseRetriever
    sparse_retriever = SparseRetriever(embedding, store)

    # Graph Retriever（知识图谱）
    from src.retrieval.graph_retriever import GraphRetriever
    from src.ingestion.graph_builder.graph_store import GraphStore
    graph_store = GraphStore(persist_path="data/knowledge_graph.json")
    graph_retriever = GraphRetriever(graph_store) if settings.graph.enabled else None

    # Reranker（Cross-Encoder / LLM 重排序）
    from src.retrieval.reranker import Reranker
    reranker = Reranker(llm_service=llm_service)

    hybrid_search = HybridSearch(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )

    # 生成器（Self-RAG 需要 hybrid_search）
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

    # 执行查询
    response = await pipeline.run(args.question)

    # 输出结果
    print(f"\n查询级别: {response.query_level}")
    if response.query_classification:
        print(f"分类置信度: {response.query_classification.confidence:.2f}")
        print(f"查询类型: {response.query_classification.query_type}")

    print(f"\n{'='*60}")
    print("回答：")
    print(response.answer)

    if response.citations:
        print(f"\n{'='*60}")
        print("引用来源：")
        for c in response.citations:
            page_str = f" (p.{c.page})" if c.page else ""
            print(f"  [{c.index}] {c.source}{page_str} - score: {c.score:.4f}")

    print(f"\n{'='*60}")
    print(f"检索追踪: {response.retrieval_trace}")


if __name__ == "__main__":
    asyncio.run(main())
