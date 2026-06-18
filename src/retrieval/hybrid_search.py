"""混合检索引擎。

协调 Dense、Sparse、Graph 三路检索，通过 RRF 融合，再经 Reranker 精排。
根据查询级别动态选择检索组合。

设计参考 MODULAR-RAG-MCP-SERVER 的 HybridSearch + SuperMew 的 Milvus 混合检索。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.types import ProcessedQuery, RetrievalResult, HybridSearchResult
from src.core.settings import Settings
from src.core.trace import TraceContext
from src.retrieval.fusion import rrf_fuse

logger = logging.getLogger(__name__)


class HybridSearch:
    """混合检索引擎。

    按查询级别动态组合检索器：
    - L1: Dense + Sparse → RRF
    - L2: Dense + Sparse → RRF → Rerank
    - L3: Dense + Sparse + Graph → RRF → Rerank
    - L4: Dense + Sparse + Graph → RRF → Rerank (多轮)
    """

    def __init__(
        self,
        settings: Settings,
        dense_retriever=None,
        sparse_retriever=None,
        graph_retriever=None,
        reranker=None,
    ):
        self._settings = settings
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._graph = graph_retriever
        self._reranker = reranker

    async def search(
        self,
        query: ProcessedQuery,
        top_k: Optional[int] = None,
        trace: Optional[TraceContext] = None,
    ) -> HybridSearchResult:
        """执行混合检索。

        Args:
            query: 处理后的查询对象
            top_k: 最终返回的结果数
            trace: 追踪上下文

        Returns:
            包含融合结果和各路原始结果的 HybridSearchResult
        """
        effective_top_k = top_k or self._settings.retrieval.fusion_top_k
        level = query.classified_level

        logger.info(f"HybridSearch: level={level}, query='{query.original_query[:50]}...'")

        # ---- Step 1: 确定各路检索是否启用 ----
        # "NAIVE" 级别：仅 Dense 检索，用于 NaiveRAG 基线对比实验
        if level == "NAIVE":
            use_dense = True
            use_sparse = False
            use_graph = False
            use_rerank = False
        else:
            use_dense = True
            use_sparse = True
            use_graph = self._settings.graph.enabled and level in ("L3", "L4")
            use_rerank = self._settings.rerank.enabled and level in ("L2", "L3", "L4")

        # ---- Step 2: 并行执行各路检索 ----
        dense_results: List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        graph_results: List[RetrievalResult] = []

        if use_dense and self._dense:
            t0 = time.monotonic()
            try:
                # HyDE: 用假设文档做 Dense 检索（语义更接近答案）
                # 否则用重写后的查询或原始查询
                dense_query = query.hyde_doc or query.rewritten_query or query.original_query
                dense_results = await self._dense.retrieve(
                    query=dense_query,
                    top_k=self._settings.retrieval.dense_top_k,
                )

                # L2 增强：如果使用了重写查询（Step-Back），同时用原始查询
                # 做一次 Dense 检索，合并结果。L2 比较类查询需要同时检索到
                # 涉及 A 和 B 的文档，Step-Back 查询可能偏向通用层面。
                if (query.classified_level == "L2"
                        and query.rewritten_query
                        and dense_query != query.original_query):
                    original_dense = await self._dense.retrieve(
                        query=query.original_query,
                        top_k=self._settings.retrieval.dense_top_k,
                    )
                    # 合并去重（保留 RRF 分数更高的）
                    seen = {r.chunk_id for r in dense_results}
                    for r in original_dense:
                        if r.chunk_id not in seen:
                            dense_results.append(r)
                            seen.add(r.chunk_id)
                    if trace:
                        trace.record_stage("l2_dense_augment", {
                            "original_count": len(original_dense),
                            "merged_count": len(dense_results),
                        })

                elapsed = (time.monotonic() - t0) * 1000
                if trace:
                    trace.record_stage("dense_retrieval", {
                        "result_count": len(dense_results),
                        "query": dense_query[:100],
                        "hyde_used": bool(query.hyde_doc),
                    }, elapsed_ms=elapsed)
            except Exception as e:
                logger.error(f"Dense retrieval failed: {e}")
                if trace:
                    trace.record_stage("dense_retrieval", {"error": str(e)})

        if use_sparse and self._sparse:
            t0 = time.monotonic()
            try:
                search_query = query.rewritten_query or query.original_query
                sparse_results = await self._sparse.retrieve(
                    keywords=query.keywords,
                    top_k=self._settings.retrieval.sparse_top_k,
                    query_text=search_query,
                )
                elapsed = (time.monotonic() - t0) * 1000
                if trace:
                    trace.record_stage("sparse_retrieval", {
                        "result_count": len(sparse_results),
                        "keywords": query.keywords[:10],
                    }, elapsed_ms=elapsed)
            except Exception as e:
                logger.error(f"Sparse retrieval failed: {e}")
                if trace:
                    trace.record_stage("sparse_retrieval", {"error": str(e)})

        if use_graph and self._graph:
            t0 = time.monotonic()
            try:
                graph_results = await self._graph.retrieve(
                    query=query.original_query,
                    top_k=self._settings.retrieval.dense_top_k,
                )
                elapsed = (time.monotonic() - t0) * 1000
                if trace:
                    trace.record_stage("graph_retrieval", {
                        "result_count": len(graph_results),
                    }, elapsed_ms=elapsed)
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")
                if trace:
                    trace.record_stage("graph_retrieval", {"error": str(e)})

        # ---- Step 3: RRF 融合 ----
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)
        if graph_results:
            ranking_lists.append(graph_results)

        t0 = time.monotonic()
        if not ranking_lists:
            logger.warning("All retrieval paths returned empty results")
            return HybridSearchResult(results=[], fusion_method="none")

        fused = rrf_fuse(ranking_lists, k=self._settings.retrieval.rrf_k, top_k=effective_top_k * 2)
        fusion_elapsed = (time.monotonic() - t0) * 1000

        if trace:
            trace.record_stage("rrf_fusion", {
                "input_lists": len(ranking_lists),
                "fused_count": len(fused),
                "rrf_k": self._settings.retrieval.rrf_k,
            }, elapsed_ms=fusion_elapsed)

        # ---- Step 4: Rerank (可选) ----
        rerank_applied = False
        if use_rerank and self._reranker and fused:
            t0 = time.monotonic()
            try:
                fused = await self._reranker.rerank(
                    query=query.original_query,
                    results=fused,
                    top_k=self._settings.rerank.top_k,
                )
                rerank_applied = True
                elapsed = (time.monotonic() - t0) * 1000
                if trace:
                    trace.record_stage("rerank", {
                        "model": self._settings.rerank.model,
                        "top_k": self._settings.rerank.top_k,
                        "result_count": len(fused),
                    }, elapsed_ms=elapsed)
            except Exception as e:
                logger.error(f"Rerank failed, using fused results: {e}")
                if trace:
                    trace.record_stage("rerank", {"error": str(e)})

        # 截断到最终 top_k
        final = fused[:effective_top_k]

        return HybridSearchResult(
            results=final,
            dense_results=dense_results,
            sparse_results=sparse_results,
            graph_results=graph_results,
            fusion_method="rrf",
            rerank_applied=rerank_applied,
        )
