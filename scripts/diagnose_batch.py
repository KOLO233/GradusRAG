"""批量检索诊断。

对种子测试集中的每道题执行检索，打印 Top-3 结果，定位问题。

用法：
    python scripts/diagnose_batch.py
    python scripts/diagnose_batch.py --max 5
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.types import ProcessedQuery
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.libs.llm_service import LLMService
from src.libs.text_utils import extract_keywords


async def diagnose_batch(max_cases: int = 10):
    settings = load_settings()

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

    hybrid = HybridSearch(
        settings=settings,
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
    )

    # 加载测试集
    test_set_path = Path(__file__).resolve().parent.parent / "data" / "test_sets" / "seed_test_set.json"
    test_cases = json.loads(test_set_path.read_text(encoding="utf-8"))

    for i, tc in enumerate(test_cases[:max_cases]):
        question = tc["question"]
        level = tc["expected_level"]
        ref_answer = tc["expected_answer"]

        keywords = extract_keywords(question)
        query = ProcessedQuery(
            original_query=question,
            classified_level=level,
            keywords=keywords,
        )

        result = await hybrid.search(query, top_k=3)

        print(f"\n{'='*70}")
        print(f"[{i+1}] {question}")
        print(f"    级别: {level}  关键词: {keywords}")
        print(f"    参考答案前80字: {ref_answer[:80]}...")
        print(f"    检索结果 ({len(result.results)} 条):")
        for j, r in enumerate(result.results[:3], 1):
            fname = r.metadata.get("filename", "?")
            print(f"      [{j}] score={r.score:.4f}  {fname}")
            print(f"          {r.text[:100]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(diagnose_batch(args.max))
