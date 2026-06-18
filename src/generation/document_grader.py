"""文档相关性评分器。

判断检索到的文档是否与用户问题相关。
参考 SuperMew 的 grade_documents_node 实现。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

GRADE_PROMPT = """你是一个文档相关性评估器。
请判断以下检索到的文档是否与用户问题相关。

用户问题：{question}

检索到的文档：
{context}

如果文档包含与问题相关的关键词或语义信息，回答 "yes"。
如果不包含，回答 "no"。
仅输出 "yes" 或 "no"，不要输出其他内容。"""


class DocumentGrader:
    """文档相关性评分器。

    使用 LLM 对检索结果进行二分类（yes/no）。
    评分不通过时触发查询重写和重新检索。

    Example:
        >>> grader = DocumentGrader(llm_service)
        >>> score = await grader.grade("什么是机器学习？", "机器学习是...")
        >>> print(score)  # "yes" or "no"
    """

    def __init__(self, llm_service=None):
        self._llm = llm_service

    async def grade(self, question: str, context: str) -> str:
        """评估文档相关性。

        Args:
            question: 用户问题
            context: 检索到的文档上下文

        Returns:
            "yes" 或 "no"
        """
        if not context.strip():
            return "no"

        if self._llm is None:
            # 无 LLM 时默认通过
            logger.warning("No LLM configured, defaulting grade to 'yes'")
            return "yes"

        prompt = GRADE_PROMPT.format(question=question, context=context[:3000])

        try:
            response = await self._llm.ainvoke(prompt)
            score = response.strip().lower()
            if score not in ("yes", "no"):
                logger.warning(f"Unexpected grade response: {score}, defaulting to 'yes'")
                return "yes"
            return score
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return "yes"  # 出错时默认通过，不阻断流程
