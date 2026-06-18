"""RRF (Reciprocal Rank Fusion) 融合算法。

将 Dense 和 Sparse 检索的排序列表融合为统一排序。
参考 MODULAR-RAG-MCP-SERVER 的 RRFFusion 实现。

公式: RRF_score(d) = sum( 1 / (k + rank(d)) )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_K = 60


def rrf_score(rank: int, k: int = DEFAULT_K) -> float:
    """计算单个排名位置的 RRF 分数。"""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    return 1.0 / (k + rank)


def rrf_fuse(
    ranking_lists: List[List[RetrievalResult]],
    k: int = DEFAULT_K,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """使用 RRF 融合多个排序列表。

    Args:
        ranking_lists: 多个检索结果列表，每个按相关性降序排列
        k: RRF 平滑常数（默认 60）
        top_k: 返回的最大结果数

    Returns:
        融合后的 RetrievalResult 列表，按 RRF 分数降序排列
    """
    non_empty = [lst for lst in ranking_lists if lst]
    if not non_empty:
        return []

    if len(non_empty) == 1:
        results = non_empty[0]
        return results[:top_k] if top_k else results

    # 计算 RRF 分数
    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, RetrievalResult] = {}

    for ranking_list in non_empty:
        for rank, result in enumerate(ranking_list, start=1):
            cid = result.chunk_id
            contribution = 1.0 / (k + rank)

            if cid not in rrf_scores:
                rrf_scores[cid] = 0.0
                chunk_data[cid] = result

            rrf_scores[cid] += contribution

    # 构建融合结果
    fused = [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=chunk_data[cid].text,
            metadata=chunk_data[cid].metadata.copy(),
            retrieval_source="fused",
        )
        for cid, score in rrf_scores.items()
    ]

    # 按 RRF 分数降序排序，相同分数按 chunk_id 排序保证稳定性
    fused.sort(key=lambda r: (-r.score, r.chunk_id))

    if top_k:
        fused = fused[:top_k]

    logger.debug(f"RRF fusion: {len(non_empty)} lists → {len(fused)} results")
    return fused


def weighted_rrf_fuse(
    ranking_lists: List[List[RetrievalResult]],
    weights: Optional[List[float]] = None,
    k: int = DEFAULT_K,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """加权 RRF 融合。

    Args:
        ranking_lists: 多个检索结果列表
        weights: 每个列表的权重（默认均匀权重）
        k: RRF 平滑常数
        top_k: 返回的最大结果数
    """
    non_empty = []
    non_empty_weights = []

    for i, lst in enumerate(ranking_lists):
        if lst:
            non_empty.append(lst)
            w = weights[i] if weights else 1.0
            non_empty_weights.append(w)

    if not non_empty:
        return []

    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, RetrievalResult] = {}

    for ranking_list, weight in zip(non_empty, non_empty_weights):
        for rank, result in enumerate(ranking_list, start=1):
            cid = result.chunk_id
            contribution = weight * (1.0 / (k + rank))

            if cid not in rrf_scores:
                rrf_scores[cid] = 0.0
                chunk_data[cid] = result

            rrf_scores[cid] += contribution

    fused = [
        RetrievalResult(
            chunk_id=cid,
            score=score,
            text=chunk_data[cid].text,
            metadata=chunk_data[cid].metadata.copy(),
            retrieval_source="fused",
        )
        for cid, score in rrf_scores.items()
    ]

    fused.sort(key=lambda r: (-r.score, r.chunk_id))

    if top_k:
        fused = fused[:top_k]

    return fused
