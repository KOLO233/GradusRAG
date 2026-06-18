"""重排序器。

两段式检索架构的精排阶段：粗排（Dense+Sparse+RRF）→ 精排（Reranker）。

支持两种后端：
1. Cross-Encoder：本地模型，精度高但需要 PyTorch
2. LLM Rerank：用 LLM API 打分，不需要本地模型

当 Cross-Encoder 不可用时自动降级为 LLM Rerank。

优化策略：
- LLM 只看 500 字符（不是 200），给足上下文
- Rerank 分数与 RRF 分数混合，避免极端重排
- 如果 LLM 给分太均匀（标准差 < 0.5），说明它无法区分，回退原始顺序
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

RERANK_PROMPT = """你是一个文档相关性评估专家。
请对以下每个文档片段与查询的相关性进行打分（0-10分）。

评分标准：
- 9-10分：直接包含答案，高度相关
- 7-8分：包含相关信息，部分有用
- 5-6分：间接相关，可作为补充
- 3-4分：略有提及，但不直接回答
- 0-2分：完全不相关

查询：{query}

文档片段：
{documents}

请仅输出一个 JSON 数组，包含每个文档的分数，顺序与输入一致。
例如：[8, 5, 9, 3, 7]
不要输出其他任何内容。"""


class Reranker:
    """重排序器。

    自动选择可用的后端：Cross-Encoder > LLM Rerank > 直接返回。

    LLM Rerank 优化：
    - 片段长度 500 字符（给 LLM 足够上下文）
    - 分数与 RRF 原始分数混合（alpha=0.6），避免极端重排
    - 如果 LLM 给分方差太低，说明它无法区分文档，直接回退原始顺序
    """

    def __init__(
        self,
        llm_service=None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        force_llm_rerank: bool = False,
    ):
        self._llm = llm_service
        self._model_name = model_name
        self._cross_encoder = None
        self._backend = "auto"
        self._force_llm = force_llm_rerank  # 强制使用 LLM rerank（消融实验用）
        self._min_score_std = 0.5            # LLM 给分方差阈值，低于此值回退原始顺序
        self._blend_alpha = 0.6              # LLM 分数混合权重（0.6=LLM为主，0.4=RRF为主）

    def _try_load_cross_encoder(self) -> bool:
        """尝试加载 Cross-Encoder 模型。"""
        if self._cross_encoder is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._model_name)
            logger.info(f"Loaded Cross-Encoder: {self._model_name}")
            return True
        except Exception as e:
            logger.debug(f"Cross-Encoder not available, falling back to LLM: {e}")
            return False

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """对检索结果重排序。

        Args:
            query: 查询文本
            results: 待重排的检索结果
            top_k: 返回的最大结果数

        Returns:
            重排后的 RetrievalResult 列表
        """
        if not results:
            return []

        logger.info(f"Reranker: backend={self._backend}, llm={'yes' if self._llm else 'no'}, results={len(results)}")

        # 优先尝试 Cross-Encoder（唯一可靠的 rerank 方式）
        if self._backend in ("auto", "cross_encoder") and self._try_load_cross_encoder():
            return self._cross_encoder_rerank(query, results, top_k)

        # 强制 LLM rerank（消融实验用，展示 rerank 的影响）
        if self._force_llm and self._llm:
            logger.info("Force LLM rerank (ablation mode)")
            return await self._llm_rerank(query, results, top_k)

        # 无 Cross-Encoder 时，直接返回 RRF 原始排序
        logger.info("No Cross-Encoder available, returning RRF order")
        return results[:top_k]

    def _cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Cross-Encoder 重排序（与 RRF 分数混合，避免极端重排）。"""
        pairs = [(query, r.text[:1024]) for r in results]
        ce_scores = self._cross_encoder.predict(pairs)

        # 归一化 Cross-Encoder 分数到 0-1
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        ce_norm = [(s - ce_min) / ce_range for s in ce_scores]

        # 归一化 RRF 分数到 0-1
        rrf_scores = [r.score for r in results]
        rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0
        rrf_norm = [(s - rrf_min) / rrf_range for s in rrf_scores]

        # 混合：Cross-Encoder 15% + RRF 85%（保守策略，RRF 为主）
        # alpha 越小越保守，消融实验显示 CE 重排会降低 MRR，
        # 因此给 CE 很小的权重，仅在 CE 分数差异显著时微调排序
        alpha = 0.15
        blended = []
        for i, r in enumerate(results):
            mixed = alpha * ce_norm[i] + (1 - alpha) * rrf_norm[i]
            blended.append((r, mixed))

        blended.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in blended[:top_k]:
            reranked.append(RetrievalResult(
                chunk_id=result.chunk_id,
                score=score,
                text=result.text,
                metadata={**result.metadata, "rerank_score": score, "blend_alpha": alpha},
                retrieval_source=result.retrieval_source,
            ))

        logger.debug(f"Cross-Encoder reranked {len(results)} → {len(reranked)} (alpha={alpha})")
        return reranked

    async def _llm_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """LLM 重排序（带分数混合和回退机制）。

        流程：
        1. LLM 对每个 chunk 打分（看 500 字符，不是 200）
        2. 检查 LLM 给分方差：如果太均匀（std < 0.5），说明 LLM 无法区分，回退原始顺序
        3. 将 LLM 分数与 RRF 原始分数混合（alpha=0.6），避免极端重排
        4. 按混合分数排序，返回 top_k
        """
        # 保存原始 RRF 分数（用于混合）
        rrf_scores = [r.score for r in results]
        rrf_max = max(rrf_scores) if rrf_scores else 1.0
        rrf_min = min(rrf_scores) if rrf_scores else 0.0
        rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

        # 构造文档列表（500 字符，不是 200）
        doc_lines = []
        for i, r in enumerate(results):
            snippet = r.text[:500].replace("\n", " ")
            doc_lines.append(f"[{i}] {snippet}")

        prompt = RERANK_PROMPT.format(
            query=query,
            documents="\n\n".join(doc_lines),
        )

        try:
            response = await self._llm.ainvoke(prompt)
            llm_scores = self._parse_scores(response, len(results))

            # 检查 LLM 给分方差
            score_std = self._std_dev(llm_scores)
            if score_std < self._min_score_std:
                logger.info(f"LLM rerank scores too uniform (std={score_std:.2f}), falling back to RRF order")
                return results[:top_k]

            # 归一化 LLM 分数到 0-1
            llm_max = max(llm_scores)
            llm_min = min(llm_scores)
            llm_range = llm_max - llm_min if llm_max > llm_min else 1.0

            # 混合分数：alpha * llm_norm + (1-alpha) * rrf_norm
            alpha = self._blend_alpha
            blended = []
            for i, r in enumerate(results):
                llm_norm = (llm_scores[i] - llm_min) / llm_range
                rrf_norm = (r.score - rrf_min) / rrf_range
                mixed_score = alpha * llm_norm + (1 - alpha) * rrf_norm
                blended.append((r, mixed_score))

            blended.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for result, score in blended[:top_k]:
                reranked.append(RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=score,
                    text=result.text,
                    metadata={**result.metadata, "rerank_score": score, "blend_alpha": alpha},
                    retrieval_source=result.retrieval_source,
                ))

            logger.debug(f"LLM reranked {len(results)} → {len(reranked)} (std={score_std:.2f}, alpha={alpha})")
            return reranked

        except Exception as e:
            logger.error(f"LLM rerank failed, returning original order: {e}")
            return results[:top_k]

    @staticmethod
    def _parse_scores(response: str, expected_count: int) -> List[float]:
        """解析 LLM 返回的分数数组。"""
        import re
        import json

        # 尝试提取 JSON 数组
        match = re.search(r'\[[\d\s,\.]+\]', response)
        if match:
            try:
                scores = json.loads(match.group())
                if len(scores) == expected_count:
                    return [float(s) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass

        # 降级：均匀分数
        logger.warning(f"Could not parse LLM rerank scores, using uniform scores")
        return [5.0] * expected_count

    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """计算标准差。"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
