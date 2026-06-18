"""GradusRAG 一键全流程脚本。

自动完成：检查环境 → 创建集合 → 批量入库 → 建图谱 → 建倒排索引 → 生成测试集 → 运行评估

用法：
    python scripts/run_all.py                          # 完整流程
    python scripts/run_all.py --skip-ingest            # 跳过入库（已有数据）
    python scripts/run_all.py --skip-eval              # 跳过评估（只要入库）
    python scripts/run_all.py --eval-only              # 只跑评估
    python scripts/run_all.py --count 30               # 生成 30 条测试集
    python scripts/run_all.py --clean                  # 清空所有数据重新开始
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
SUPPORTED_EXTS = {".md", ".pdf", ".docx", ".txt", ".html"}


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def step(n: int, text: str):
    print(f"\n[{n}] {text}")


def check_milvus():
    """检查 Milvus 是否可连接。"""
    import httpx
    # Milvus 健康检查端口是 9091，不是 19530（19530 是 gRPC 数据端口）
    for port in [9091, 19530]:
        try:
            r = httpx.get(f"http://localhost:{port}/healthz", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    # 最后尝试用 pymilvus 直接连接
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        client.close()
        return True
    except Exception:
        return False


def get_docs():
    """获取文档目录下的所有支持文件。"""
    docs = []
    for f in sorted(DOCS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith("."):
            docs.append(f)
    return docs


async def run_full_pipeline(args):
    from src.core.settings import load_settings

    settings = load_settings()

    # =========================================================================
    # Step 1: 环境检查
    # =========================================================================
    step(1, "检查环境")

    docs = get_docs()
    print(f"  文档目录: {DOCS_DIR}")
    print(f"  找到 {len(docs)} 个文档:")
    for d in docs:
        print(f"    - {d.name}")

    milvus_ok = check_milvus()
    print(f"  Milvus 连接: {'OK' if milvus_ok else 'FAILED'}")

    if not milvus_ok:
        print("\n  Milvus 未启动，请先执行:")
        print("    docker compose up -d")
        print("  等待 30 秒后重试。")
        return

    print(f"  LLM: {settings.llm.provider}/{settings.llm.model}")
    print(f"  Embedding: {settings.embedding.provider}/{settings.embedding.model}")

    # =========================================================================
    # Step 2: 创建 Milvus 集合
    # =========================================================================
    step(2, "初始化 Milvus 集合")

    from src.retrieval.milvus_store import MilvusStore
    milvus = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    milvus.init_collection(dense_dim=settings.embedding.dimensions)
    print(f"  集合 '{settings.vector_store.collection}' 就绪")

    if args.clean:
        banner("清空所有数据")
        milvus._get_client().drop_collection(settings.vector_store.collection)
        milvus.init_collection(dense_dim=settings.embedding.dimensions)
        for f in ["data/bm25_state.json", "data/bm25_inverted_index.json", "data/knowledge_graph.json"]:
            p = Path(f)
            if p.exists():
                p.unlink()
                print(f"  已删除: {f}")
        print("  数据已清空")

    # =========================================================================
    # Step 3: 批量文档入库
    # =========================================================================
    if not args.skip_ingest and not args.eval_only:
        step(3, "批量文档入库（分块 + 向量化 + 图谱 + 倒排索引）")

        from src.libs.llm_service import LLMService
        from src.libs.embedding_service import EmbeddingService
        from src.ingestion.pipeline import IngestionPipeline
        from src.ingestion.graph_builder.graph_builder import GraphBuilder
        from src.ingestion.graph_builder.graph_store import GraphStore

        embedding_service = EmbeddingService(
            model_name=settings.embedding.model,
            device=settings.embedding.device,
            dimensions=settings.embedding.dimensions,
            api_key=settings.embedding.api_key,
            api_base_url=settings.embedding.api_base_url,
        )
        llm_service = LLMService.from_settings(settings)
        graph_store = GraphStore(persist_path="data/knowledge_graph.json")
        graph_builder = GraphBuilder(
            settings=settings.graph,
            llm_service=llm_service,
            graph_store=graph_store,
        ) if settings.graph.enabled else None

        pipeline = IngestionPipeline(
            settings,
            embedding_service=embedding_service,
            milvus_store=milvus,
            graph_builder=graph_builder,
        )

        total_chunks = 0
        total_parents = 0
        t0 = time.monotonic()

        for i, doc_path in enumerate(docs, 1):
            print(f"  [{i}/{len(docs)}] {doc_path.name}...", end=" ", flush=True)
            try:
                c, p = pipeline.ingest_file_to_milvus(str(doc_path))
                total_chunks += c
                total_parents += p
                print(f"OK ({c} chunks)")
            except Exception as e:
                print(f"FAILED: {e}")

        elapsed = time.monotonic() - t0
        print(f"\n  入库完成: {total_chunks} chunks, {total_parents} parents, 耗时 {elapsed:.1f}s")

        # 查看图谱统计
        if graph_builder:
            stats = graph_store.stats()
            print(f"  图谱: {stats['entity_count']} 实体, {stats['relation_count']} 关系")

        # 查看倒排索引统计
        idx_stats = embedding_service.bm25_index_stats()
        print(f"  倒排索引: {idx_stats['tokens']} tokens, {idx_stats['documents']} docs")

    # =========================================================================
    # Step 4: 验证检索
    # =========================================================================
    if not args.eval_only:
        step(4, "验证检索功能")

        from src.libs.llm_service import LLMService
        from src.libs.embedding_service import EmbeddingService
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.hybrid_search import HybridSearch
        from src.retrieval.reranker import Reranker
        from src.core.types import ProcessedQuery

        embedding_service = EmbeddingService(
            model_name=settings.embedding.model,
            device=settings.embedding.device,
            dimensions=settings.embedding.dimensions,
            api_key=settings.embedding.api_key,
            api_base_url=settings.embedding.api_base_url,
        )
        llm_service = LLMService.from_settings(settings)

        dense = DenseRetriever(embedding_service, milvus)
        sparse = SparseRetriever(embedding_service, milvus)
        reranker = Reranker(llm_service=llm_service)

        # 图检索（可选）
        graph_retriever = None
        if settings.graph.enabled:
            from src.ingestion.graph_builder.graph_store import GraphStore
            from src.retrieval.graph_retriever import GraphRetriever
            graph_store = GraphStore(persist_path="data/knowledge_graph.json")
            graph_retriever = GraphRetriever(graph_store, embedding_service)

        hybrid = HybridSearch(
            settings=settings,
            dense_retriever=dense,
            sparse_retriever=sparse,
            graph_retriever=graph_retriever,
            reranker=reranker,
        )

        test_query = ProcessedQuery(
            original_query="什么是糖尿病？",
            classified_level="L1",
            keywords=["糖尿病"],
        )
        result = await hybrid.search(test_query, top_k=3)
        print(f"  测试查询: '什么是糖尿病？'")
        print(f"  检索到 {len(result.results)} 个结果:")
        for r in result.results[:3]:
            source = r.metadata.get("filename", "unknown")
            print(f"    [{r.score:.4f}] {source}: {r.text[:60]}...")

    # =========================================================================
    # Step 5: 生成测试集 + 评估
    # =========================================================================
    if not args.skip_eval:
        step(5, "生成测试集并运行评估")

        from src.libs.llm_service import LLMService
        from scripts.generate_test_set import TestSetGenerator
        from src.evaluation.evaluator import Evaluator
        from scripts.run_evaluation import build_pipeline, print_report

        llm_service = LLMService.from_settings(settings)

        # 生成测试集
        test_set_file = "auto_test_set.json"
        if not args.eval_only:
            generator = TestSetGenerator(llm_service)
            test_cases = await generator.generate_from_milvus(milvus, count=args.count)
            if test_cases:
                generator.save(test_cases, test_set_file)
            else:
                print("  测试集生成失败，使用种子测试集")
                test_set_file = "seed_test_set.json"

        # 运行评估
        print(f"\n  开始评估 ({test_set_file})...")
        rag_pipeline = build_pipeline(settings)
        evaluator = Evaluator(pipeline=rag_pipeline, llm_service=llm_service)
        result = evaluator.evaluate(test_set_file=test_set_file, max_cases=args.count)

        print_report(result)

        # 保存结果
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"eval_{ts}.json"
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  结果已保存: {result_path}")

    banner("全部完成")


def main():
    parser = argparse.ArgumentParser(description="GradusRAG 一键全流程")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过文档入库")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--eval-only", action="store_true", help="只跑评估（跳过入库和验证）")
    parser.add_argument("--count", type=int, default=50, help="测试集数量（默认 50）")
    parser.add_argument("--clean", action="store_true", help="清空所有数据重新开始")
    args = parser.parse_args()

    asyncio.run(run_full_pipeline(args))


if __name__ == "__main__":
    main()
