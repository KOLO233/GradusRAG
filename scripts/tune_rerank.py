"""Rerank 混合比例调优脚本。

测试不同 alpha（Cross-Encoder 权重）对 MRR 的影响，
找出最优混合比例。

用法：
    python scripts/tune_rerank.py
    python scripts/tune_rerank.py --max-cases 20
"""

import argparse
import asyncio
import json
import logging
import sys
import time
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
from src.retrieval.fusion import rrf_fuse
from src.evaluation.metrics import hit_rate_at_k, mrr
from src.evaluation.test_set import TestSetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_alpha(alpha: float, test_cases, dense, sparse, settings, max_cases=20):
    """测试某个 alpha 值下的 MRR 和 Hit Rate。"""
    from src.retrieval.reranker import Reranker

    # 创建一个自定义 reranker 使用指定 alpha
    reranker = Reranker()
    reranker._blend_alpha = alpha  # 直接设置混合比例

    hit_rates = []
    mrr_scores = []

    cases = test_cases[:max_cases] if max_cases > 0 else test_cases

    for tc in cases:
        query = ProcessedQuery(
            original_query=tc.question,
            classified_level=tc.expected_level,
            keywords=tc.question.split(),
        )

        try:
            # Dense + Sparse retrieval
            dense_results = await dense.retrieve(query=tc.question, top_k=20)
            sparse_results = await sparse.retrieve(keywords=tc.question.split(), top_k=20, query_text=tc.question)

            ranking_lists = []
            if dense_results:
                ranking_lists.append(dense_results)
            if sparse_results:
                ranking_lists.append(sparse_results)

            if not ranking_lists:
                hit_rates.append(0.0)
                mrr_scores.append(0.0)
                continue

            # RRF fusion
            fused = rrf_fuse(ranking_lists, k=settings.retrieval.rrf_k, top_k=20)

            # Rerank with specified alpha
            reranked = await reranker.rerank(query=tc.question, results=fused, top_k=5)

            # Evaluate
            retrieved_ids = [r.chunk_id for r in reranked]
            ground_truth = tc.ground_truth_chunks or []

            hr = hit_rate_at_k(retrieved_ids, ground_truth, k=5)
            mrr_score = mrr(retrieved_ids, ground_truth, k=10)
            hit_rates.append(hr)
            mrr_scores.append(mrr_score)

        except Exception as e:
            logger.debug(f"  Error for '{tc.question[:30]}': {e}")
            hit_rates.append(0.0)
            mrr_scores.append(0.0)

    avg_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    return avg_hr, avg_mrr


async def main():
    parser = argparse.ArgumentParser(description="Rerank 混合比例调优")
    parser.add_argument("--max-cases", type=int, default=20, help="每种 alpha 测试的题数")
    parser.add_argument("--test-set", default="formal_test_set.json", help="测试集")
    args = parser.parse_args()

    settings = load_settings()

    # 加载测试集
    test_mgr = TestSetManager()
    test_cases = test_mgr.load(args.test_set)
    if not test_cases:
        logger.error("No test cases found")
        return

    logger.info(f"Test set: {len(test_cases)} cases, testing with {args.max_cases}")

    # 按级别均匀采样
    from collections import defaultdict
    by_level = defaultdict(list)
    for tc in test_cases:
        by_level[tc.expected_level].append(tc)
    per_level = max(args.max_cases // 4, 1)
    sampled = []
    for level in ["L1", "L2", "L3", "L4"]:
        sampled.extend(by_level.get(level, [])[:per_level])
    test_cases = sampled
    logger.info(f"Sampled {len(test_cases)} cases: {per_level} per level")

    # 初始化检索组件
    embedding_service = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    milvus_store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    dense = DenseRetriever(embedding_service, milvus_store)
    sparse = SparseRetriever(embedding_service, milvus_store)

    # 测试不同 alpha 值
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for alpha in alphas:
        logger.info(f"Testing alpha={alpha:.1f}...")
        t0 = time.monotonic()
        hr, mrr_score = await test_alpha(alpha, test_cases, dense, sparse, settings, args.max_cases)
        elapsed = time.monotonic() - t0
        results.append({"alpha": alpha, "hit_rate": hr, "mrr": mrr_score})
        logger.info(f"  alpha={alpha:.1f}: Hit@5={hr:.2%}, MRR={mrr_score:.2%} ({elapsed:.0f}s)")

    # 打印结果表格
    print(f"\n{'='*60}")
    print(f"RERANK BLEND ALPHA TUNING ({args.max_cases} cases)")
    print(f"{'='*60}")
    print(f"{'Alpha (CE weight)':>20} {'RRF weight':>12} {'Hit@5':>10} {'MRR':>10}")
    print(f"{'-'*60}")
    for r in results:
        alpha = r["alpha"]
        rrf_w = 1 - alpha
        print(f"{alpha:>18.1f}     {rrf_w:>10.1f} {r['hit_rate']:>9.2%} {r['mrr']:>9.2%}")
    print(f"{'='*60}")

    # 找最优
    best_mrr = max(results, key=lambda x: x["mrr"])
    best_hr = max(results, key=lambda x: x["hit_rate"])
    print(f"\nBest MRR:  alpha={best_mrr['alpha']:.1f} → MRR={best_mrr['mrr']:.2%}")
    print(f"Best Hit:  alpha={best_hr['alpha']:.1f} → Hit@5={best_hr['hit_rate']:.2%}")

    # 保存
    output = {"results": results, "best_mrr": best_mrr, "best_hr": best_hr}
    Path("results").mkdir(exist_ok=True)
    Path("results/rerank_tuning.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to results/rerank_tuning.json")


if __name__ == "__main__":
    asyncio.run(main())
