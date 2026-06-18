"""查询重写器。

当文档评分不通过时，通过查询重写改善检索效果。
支持两种策略：
- Step-Back：生成更通用的退步问题（适合 L2 比较类查询）
- HyDE：生成假设性文档并用其向量检索（适合 L3/L4 推理类查询）

参考 SuperMew 的 rewrite_question_node 实现。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STEP_BACK_PROMPT = """你是一个查询优化专家。
用户的问题可能太具体，导致检索效果不好。
请生成一个更通用、更宏观的"退步问题"，帮助检索到更相关的文档。

原始问题：{query}

请直接输出优化后的退步问题，不要输出其他内容。"""

HYDE_PROMPT = """你是一个领域专家。
请根据用户问题，生成一段可能包含答案的假设性文档。
这段文档不需要是真实的，但应该包含与答案相关的关键词和概念。
它将被用于向量检索，所以要包含尽可能多的答案相关术语。

用户问题：{query}

请直接输出假设性文档内容（200-300字），不要输出其他内容。"""


class QueryRewriter:
    """查询重写器。

    根据查询级别自动选择重写策略：
    - L1: 不重写（简单查询不需要）
    - L2: Step-Back（退步到更通用的问题）
    - L3/L4: HyDE（生成假设文档用于向量检索）

    Example:
        >>> rewriter = QueryRewriter(llm_service)
        >>> result = await rewriter.rewrite("A药和B药的区别是什么？", level="L2")
        >>> print(result["strategy"])         # "step_back"
        >>> result = await rewriter.rewrite("为什么会出现梯度消失？", level="L3")
        >>> print(result["strategy"])         # "hyde"
    """

    def __init__(self, llm_service=None):
        self._llm = llm_service

    async def rewrite(
        self,
        query: str,
        strategy: str = "auto",
        level: str = "L1",
    ) -> Dict:
        """重写查询。

        Args:
            query: 原始查询
            strategy: 重写策略 ("step_back" / "hyde" / "both" / "auto")
            level: 查询级别 (L1/L2/L3/L4)，auto 模式下用于选择策略

        Returns:
            {
                "strategy": 使用的策略,
                "rewritten_query": 重写后的查询（Step-Back 时为退步问题，HyDE 时为原始查询）,
                "step_back_question": 退步问题（如有）,
                "hypothetical_doc": 假设性文档（如有，HyDE 时用于向量检索）,
            }
        """
        if self._llm is None:
            logger.warning("No LLM configured, returning original query")
            return {
                "strategy": "none",
                "rewritten_query": query,
                "step_back_question": "",
                "hypothetical_doc": "",
            }

        # 自动选择策略
        if strategy == "auto":
            if level in ("L3", "L4"):
                strategy = "hyde"
            elif level == "L2":
                strategy = "step_back"
            else:
                # L1 不重写
                return {
                    "strategy": "none",
                    "rewritten_query": query,
                    "step_back_question": "",
                    "hypothetical_doc": "",
                }

        result = {
            "strategy": strategy,
            "rewritten_query": query,
            "step_back_question": "",
            "hypothetical_doc": "",
        }

        if strategy in ("step_back", "both"):
            try:
                prompt = STEP_BACK_PROMPT.format(query=query)
                step_back = await self._llm.ainvoke(prompt)
                result["step_back_question"] = step_back.strip()
                result["rewritten_query"] = step_back.strip()
                logger.debug(f"Step-back: '{step_back.strip()[:60]}...'")
            except Exception as e:
                logger.error(f"Step-back rewrite failed: {e}")

        if strategy in ("hyde", "both"):
            try:
                prompt = HYDE_PROMPT.format(query=query)
                hyde = await self._llm.ainvoke(prompt)
                result["hypothetical_doc"] = hyde.strip()
                # HyDE 模式下，rewritten_query 保持原始查询不变
                # 假设文档单独存储，由 HybridSearch 做向量检索
                logger.debug(f"HyDE doc generated: '{hyde.strip()[:60]}...'")
            except Exception as e:
                logger.error(f"HyDE rewrite failed: {e}")

        return result
