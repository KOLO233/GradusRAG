"""Self-RAG 迭代机制。

实现真正的检索 → 生成 → 批判 → 迭代循环，而非 Prompt 模拟。

参考 LightRAG 的自适应检索思想和 SuperMew 的文档评分门控机制。

流程：
1. 初始检索 → 初步生成
2. 批判性评估（忠实度 + 完整性 + 准确性）
3. 如果评估不通过 → 生成改进查询 → 重新检索 → 重新生成
4. 重复直到评估通过或达到最大迭代次数
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ===========================================================================
# 批判评估 Prompt
# ===========================================================================

CRITIQUE_PROMPT = """你是一个严格的质量评估专家。请对以下回答进行批判性评估。

用户问题：{question}

检索到的参考资料：
{context}

当前回答：
{answer}

请从以下三个维度评估（每个维度给出 1、0 或 -1 分）：

1. **忠实度 (Faithfulness)**：回答中的核心事实是否能在参考资料中找到依据？
   - 1：核心事实都有参考资料支撑，推理合理
   - 0：大部分有依据，少量无法验证
   - -1：大部分核心事实无法从参考资料中验证

2. **完整性 (Completeness)**：回答是否覆盖了参考资料中与问题相关的关键信息？
   - 1：覆盖了所有关键信息
   - 0：覆盖了大部分，遗漏了一些
   - -1：遗漏了大部分关键信息

3. **准确性 (Accuracy)**：回答中的事实是否正确，推理是否基于参考资料？
   - 1：事实正确，推理有据
   - 0：基本正确，个别地方不够严谨
   - -1：存在事实错误或无依据的推测

请以 JSON 格式输出：
```json
{{
  "faithfulness": 1/0/-1,
  "completeness": 1/0/-1,
  "accuracy": 1/0/-1,
  "overall": 综合分数,
  "pass": true/false,
  "issues": ["具体问题描述1", "具体问题描述2"],
  "improvement_suggestions": ["具体改进建议1", "具体改进建议2"]
}}
```

评分标准：
- 三个维度总分 >= 2：通过
- 三个维度总分 < 2：不通过，需要改进"""

REFINE_QUERY_PROMPT = """基于以下批判反馈，生成一个改进的检索查询，以获取更准确的信息。

原始问题：{question}
批判反馈：{critique}
存在的问题：{issues}

请直接输出改进后的查询（一句话），不要输出其他内容。"""

REFINE_ANSWER_PROMPT = """你是一位资深领域专家。请基于以下信息，生成一个改进的回答。

原始问题：{question}

参考资料（包含新检索的内容）：
{context}

之前的回答：
{previous_answer}

需要改进的问题：
{issues}

改进建议：
{suggestions}

关键约束：
1. 以参考资料为主要依据，可以合理推理和补充
2. 每个事实必须标注来源 [1][2]...
3. 如果参考资料不足以解决问题，说明能确定的部分
4. 使用与问题相同的语言回答

请生成改进回答："""


@dataclass
class CritiqueResult:
    """批判评估结果。"""
    faithfulness: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    overall: float = 0.0
    passed: bool = False
    issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "overall": self.overall,
            "passed": self.passed,
            "issues": self.issues,
            "improvement_suggestions": self.improvement_suggestions,
        }


@dataclass
class SelfRAGIteration:
    """单次迭代记录。"""
    iteration: int
    query: str
    context: str
    answer: str
    critique: CritiqueResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "query": self.query[:100],
            "context_length": len(self.context),
            "answer_length": len(self.answer),
            "critique": self.critique.to_dict(),
        }


class SelfRAG:
    """Self-RAG 迭代推理引擎。

    实现真正的 检索→生成→批判→迭代 循环。

    上下文管理策略：
    - 每轮迭代后对上下文做压缩，防止无限增长
    - 优先保留新检索的内容（因为是针对批判反馈改进的查询结果）
    - 最大上下文长度由 max_context_length 控制

    Example:
        >>> self_rag = SelfRAG(llm_service, hybrid_search, max_iterations=3)
        >>> result = await self_rag.run("如果用ReLU替代Sigmoid会怎样？", "L4")
        >>> print(result["iterations"])  # 迭代记录
        >>> print(result["final_answer"])  # 最终回答
    """

    def __init__(
        self,
        llm_service=None,
        hybrid_search=None,
        max_iterations: int = 3,
        pass_threshold: float = 2.0,
        max_context_length: int = 6000,
    ):
        self._llm = llm_service
        self._search = hybrid_search
        self._max_iterations = max_iterations
        self._pass_threshold = pass_threshold
        self._max_context_length = max_context_length

    async def run(
        self,
        question: str,
        query=None,
        initial_context: str = "",
        initial_answer: str = "",
    ) -> Dict[str, Any]:
        """执行 Self-RAG 迭代推理。

        Args:
            question: 用户问题
            query: 处理后的查询对象（用于检索）
            initial_context: 初始上下文（如果已有检索结果）
            initial_answer: 初始回答（如果已有生成结果）

        Returns:
            {
                "final_answer": 最终回答,
                "iterations": 迭代记录列表,
                "total_iterations": 总迭代次数,
                "final_critique": 最终批判结果,
            }
        """
        iterations: List[SelfRAGIteration] = []
        current_context = initial_context
        current_answer = initial_answer
        current_query_text = question

        for i in range(self._max_iterations):
            logger.info(f"Self-RAG iteration {i + 1}/{self._max_iterations}")

            # Step 1: 如果没有初始回答，生成一个
            if not current_answer and self._llm:
                if not current_context and self._search and query:
                    # 执行检索
                    from src.core.types import ProcessedQuery
                    from src.libs.text_utils import extract_keywords
                    search_query = query if hasattr(query, 'original_query') else None
                    if search_query is None:
                        search_query = ProcessedQuery(
                            original_query=current_query_text,
                            classified_level="L4",
                            keywords=extract_keywords(current_query_text),
                        )
                    search_result = await self._search.search(query=search_query)
                    current_context = self._format_context(search_result.results)

                current_answer = await self._generate_answer(
                    question, current_context
                )

            if not current_answer:
                break

            # Step 2: 批判性评估
            critique = await self._critique(question, current_context, current_answer)

            iterations.append(SelfRAGIteration(
                iteration=i + 1,
                query=current_query_text,
                context=current_context,
                answer=current_answer,
                critique=critique,
            ))

            logger.info(
                f"  Critique: overall={critique.overall:.1f}, "
                f"pass={critique.passed}, issues={len(critique.issues)}"
            )

            # Step 3: 如果通过，返回
            if critique.passed or i == self._max_iterations - 1:
                return {
                    "final_answer": current_answer,
                    "iterations": iterations,
                    "total_iterations": i + 1,
                    "final_critique": critique,
                }

            # Step 4: 不通过 → 改进查询 → 重新检索 → 重新生成
            refined_query = await self._refine_query(
                question, critique, critique.issues
            )
            if refined_query:
                current_query_text = refined_query
                logger.info(f"  Refined query: {refined_query[:60]}...")

            # 重新检索
            if self._search:
                from src.core.types import ProcessedQuery
                from src.libs.text_utils import extract_keywords
                new_query = ProcessedQuery(
                    original_query=current_query_text,
                    classified_level="L4",
                    keywords=extract_keywords(current_query_text),
                )
                search_result = await self._search.search(query=new_query)
                new_context = self._format_context(search_result.results)

                # 合并上下文：新内容在前（更重要），旧内容在后
                if current_context:
                    merged = new_context + "\n\n---\n\n" + current_context
                else:
                    merged = new_context

                # 压缩：防止上下文无限增长
                current_context = self._compress_context(merged)

            # 重新生成（基于批判建议改进）
            current_answer = await self._refine_answer(
                question, current_context, current_answer,
                critique.issues, critique.improvement_suggestions,
            )

        return {
            "final_answer": current_answer,
            "iterations": iterations,
            "total_iterations": len(iterations),
            "final_critique": iterations[-1].critique if iterations else None,
        }

    async def _generate_answer(self, question: str, context: str) -> str:
        """生成回答。"""
        if not self._llm:
            return ""

        prompt = (
            "你是一位资深领域专家。请根据提供的参考资料回答问题。\n\n"
            "---指令---\n"
            "1. 以参考资料为主要依据，可以用自己的知识组织和补充，但不要与参考资料矛盾。\n"
            "2. 如果参考资料不足以完整回答，说明你能确定的部分。\n"
            "3. 引用时标注来源编号 [1][2]。\n"
            "4. 使用与问题相同的语言回答。\n\n"
            f"---参考资料---\n{context[:4000]}\n\n"
            f"---问题---\n{question}\n\n"
            "请给出详细的回答："
        )

        try:
            return await self._llm.ainvoke(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    async def _critique(
        self, question: str, context: str, answer: str
    ) -> CritiqueResult:
        """批判性评估。"""
        if not self._llm:
            return CritiqueResult(overall=8.0, passed=True)

        prompt = CRITIQUE_PROMPT.format(
            question=question,
            context=context[:3000],
            answer=answer[:2000],
        )

        try:
            response = await self._llm.ainvoke(prompt)
            return self._parse_critique(response)
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return CritiqueResult(overall=8.0, passed=True)

    def _parse_critique(self, response: str) -> CritiqueResult:
        """解析批判评估结果（1/0/-1 评分制）。"""
        import json
        import re

        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return CritiqueResult(overall=2.0, passed=True)

        try:
            data = json.loads(json_match.group())
            faith = int(data.get("faithfulness", 0))
            comp = int(data.get("completeness", 0))
            acc = int(data.get("accuracy", 0))
            overall = faith + comp + acc  # 总分范围 -3 到 3
            return CritiqueResult(
                faithfulness=float(faith),
                completeness=float(comp),
                accuracy=float(acc),
                overall=float(overall),
                passed=overall >= 2,  # 总分 >= 2 通过
                issues=data.get("issues", []),
                improvement_suggestions=data.get("improvement_suggestions", []),
            )
        except (json.JSONDecodeError, ValueError):
            return CritiqueResult(overall=2.0, passed=True)

    async def _refine_query(
        self, question: str, critique: CritiqueResult, issues: List[str]
    ) -> str:
        """基于批判反馈生成改进查询。"""
        if not self._llm:
            return ""

        prompt = REFINE_QUERY_PROMPT.format(
            question=question,
            critique=critique.to_dict(),
            issues=", ".join(issues) if issues else "无",
        )

        try:
            return (await self._llm.ainvoke(prompt)).strip()
        except Exception:
            return ""

    async def _refine_answer(
        self,
        question: str,
        context: str,
        previous_answer: str,
        issues: List[str],
        suggestions: List[str],
    ) -> str:
        """基于批判建议生成改进回答。"""
        if not self._llm:
            return previous_answer

        prompt = REFINE_ANSWER_PROMPT.format(
            question=question,
            context=context[:4000],
            previous_answer=previous_answer[:1500],
            issues=", ".join(issues) if issues else "无",
            suggestions=", ".join(suggestions) if suggestions else "无",
        )

        try:
            return (await self._llm.ainvoke(prompt)).strip()
        except Exception as e:
            logger.error(f"Refine answer failed: {e}")
            return previous_answer

    def _compress_context(self, context: str) -> str:
        """压缩上下文，防止无限增长。

        策略：按段落分割，保留前面的内容（新检索的排在前面），
        截断到 max_context_length。段落边界对齐，不会切断中间的片段。
        """
        if len(context) <= self._max_context_length:
            return context

        # 按段落分割（以 --- 分隔）
        paragraphs = context.split("\n\n---\n\n")

        # 从前往后保留，直到达到长度限制
        kept = []
        total = 0
        for para in paragraphs:
            para_len = len(para) + 5  # +5 for separator
            if total + para_len > self._max_context_length and kept:
                break
            kept.append(para)
            total += para_len

        compressed = "\n\n---\n\n".join(kept)
        if len(compressed) < len(context):
            logger.debug(
                f"Self-RAG context compressed: {len(context)} → {len(compressed)} chars "
                f"({len(kept)}/{len(paragraphs)} paragraphs kept)"
            )
        return compressed

    @staticmethod
    def _format_context(results) -> str:
        """格式化检索结果为上下文。"""
        if not results:
            return ""
        chunks = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("filename", "Unknown")
            page = r.metadata.get("page", "N/A")
            chunks.append(f"[{i}] {source} (Page {page}):\n{r.text}")
        return "\n\n---\n\n".join(chunks)
