"""检索诊断脚本。

对指定问题执行检索，打印返回的 Top-K 结果，帮助定位检索质量问题。

用法：
    python scripts/diagnose_retrieval.py "什么是糖尿病？"
    python scripts/diagnose_retrieval.py "比较监督学习和无监督学习" --top-k 10
    python scripts/diagnose_retrieval.py "为什么会出现梯度消失" --level L3
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import ProcessedQuery
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.query_classifier.classifier import create_classifier


async def diagnose(question: str, level: str = None, top_k: int = 5):
    settings = load_settings()

    # 如果没指定级别，自动分类
    if level is None:
        classifier = create_classifier(settings.query_classifier)
        classification = classifier.classify(question)
        level = classification.level
        print(f"自动分类: {level} (置信度: {classification.confidence:.0%})")
    else:
        print(f"手动指定级别: {level}")

    # 构建检索器
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    milvus = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    llm_service = LLMService.from_settings(settings)

    dense = DenseRetriever(embedding_service, milvus)
    sparse = SparseRetriever(embedding_service, milvus)
    reranker = Reranker(llm_service=llm_service)

    graph_retriever = None
    if settings.graph.enabled and level in ("L3", "L4"):
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

    # 执行检索
    from src.libs.text_utils import extract_keywords
    keywords = extract_keywords(question)
    print(f"关键词: {keywords}")

    query = ProcessedQuery(
        original_query=question,
        classified_level=level,
        keywords=keywords,
    )

    print(f"\n{'='*70}")
    print(f"查询: {question}")
    print(f"级别: {level}")
    print(f"{'='*70}")

    result = await hybrid.search(query, top_k=top_k)

    # 打印各路结果
    print(f"\n--- Dense 检索 ({len(result.dense_results)} 条) ---")
    for i, r in enumerate(result.dense_results[:5], 1):
        fname = r.metadata.get("filename", "?")
        print(f"  [{i}] score={r.score:.4f}  {fname}  |  {r.text[:80]}...")

    print(f"\n--- Sparse 检索 ({len(result.sparse_results)} 条) ---")
    for i, r in enumerate(result.sparse_results[:5], 1):
        fname = r.metadata.get("filename", "?")
        print(f"  [{i}] score={r.score:.4f}  {fname}  |  {r.text[:80]}...")

    if result.graph_results:
        print(f"\n--- Graph 检索 ({len(result.graph_results)} 条) ---")
        for i, r in enumerate(result.graph_results[:3], 1):
            center = r.metadata.get("center_entity", "?")
            print(f"  [{i}] score={r.score:.4f}  center={center}  |  {r.text[:80]}...")

    print(f"\n--- RRF 融合后 Top-{top_k} ---")
    for i, r in enumerate(result.results, 1):
        fname = r.metadata.get("filename", "?")
        src = r.retrieval_source
        print(f"  [{i}] score={r.score:.4f}  source={src}  {fname}")
        print(f"       {r.text[:120]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索诊断")
    parser.add_argument("question", help="查询问题")
    parser.add_argument("--top-k", type=int, default=5, help="返回数量")
    parser.add_argument("--level", type=str, help="指定级别 (L1/L2/L3/L4)")
    args = parser.parse_args()

    asyncio.run(diagnose(args.question, level=args.level, top_k=args.top_k))
