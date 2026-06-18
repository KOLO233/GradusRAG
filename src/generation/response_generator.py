"""分级响应生成器。

根据查询级别（L1-L4）选择不同的生成策略：
- L1: 直接回答（最简单）
- L2: 结构化分析
- L3: 专家角色 Chain-of-Thought 推理
- L4: Self-RAG 迭代批判

这是 GradusRAG 的核心创新模块。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def _load_prompt(name: str) -> str:
    """从 prompts 目录加载模板。"""
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    # 回退到内置默认
    return _DEFAULT_PROMPTS.get(name, "")


_DEFAULT_PROMPTS = {
    "generate_l1.txt": (
        "你是一个专业的领域问答助手。请根据提供的参考资料回答用户问题。\n\n"
        "---参考资料---\n{context}\n\n"
        "---用户问题---\n{question}\n\n"
        "请提供详细、结构化的回答。包含相关事实、解释和例子。\n"
        "如果参考资料不足以完整回答，说明你能确定的部分和不确定的部分。\n"
        "引用时标注来源编号 [1][2]。"
    ),
    "generate_l2.txt": (
        "你是一个专业的领域分析助手。请根据提供的参考资料进行结构化对比分析。\n\n"
        "---参考资料---\n{context}\n\n"
        "---用户问题---\n{question}\n\n"
        "请从多个维度进行详细对比分析，每个维度提供支撑证据和具体例子。\n"
        "包含相关事实、解释和例子。引用时标注来源编号 [1][2]。\n\n"
        "## 分析\n\n"
        "## 综合结论"
    ),
    "generate_l3_cot.txt": (
        "你是一位资深的领域专家。请使用 Chain-of-Thought 方法逐步推理回答问题。\n\n"
        "---参考资料---\n{context}\n\n"
        "---用户问题---\n{question}\n\n"
        "请提供详细的逐步推理，每一步都有解释和依据。包含相关事实、机制解释和例子。\n"
        "引用时标注来源编号 [1][2]。\n\n"
        "## 问题理解\n"
        "## 逐步推理\n"
        "Step 1: ...\n"
        "## 结论"
    ),
    "generate_l4_selfrag.txt": (
        "你是一位资深领域专家，擅长分析推理。请根据提供的参考资料进行深入分析。\n\n"
        "---参考资料---\n{context}\n\n"
        "---用户问题---\n{question}\n\n"
        "请提供全面的分析，包含相关事实、推理、机制解释和例子。\n"
        "引用时标注来源编号 [1][2]。\n\n"
        "## 初始分析\n"
        "## 批判性评估\n"
        "## 精炼回答"
    ),
}


class ResponseGenerator:
    """分级响应生成器。

    根据查询级别自动选择 Prompt 模板和生成策略。
    L4 级别使用 Self-RAG 迭代引擎（真正的 检索→生成→批判→迭代 循环）。

    Example:
        >>> gen = ResponseGenerator(llm_service, hybrid_search)
        >>> answer = await gen.generate("什么是机器学习？", context, level="L1")
    """

    def __init__(self, llm_service=None, hybrid_search=None):
        self._llm = llm_service
        self._search = hybrid_search
        self._prompts = {
            "L1": _load_prompt("generate_l1.txt"),
            "L2": _load_prompt("generate_l2.txt"),
            "L3": _load_prompt("generate_l3_cot.txt"),
        }
        # Self-RAG 引擎（L4 专用）
        self._self_rag = None
        # 忠实度校验开关（None = 未加载，首次使用时从配置读取）
        self._faithfulness_verify = None

    def _load_verify_setting(self) -> bool:
        """从配置懒加载忠实度校验开关。"""
        if self._faithfulness_verify is None:
            try:
                from src.core.settings import load_settings
                settings = load_settings()
                self._faithfulness_verify = settings.generation.faithfulness_verify
            except Exception:
                self._faithfulness_verify = True
        return self._faithfulness_verify

    async def generate(
        self,
        question: str,
        context: str,
        level: str = "L1",
        query=None,
    ) -> str:
        """根据查询级别生成回答。

        Args:
            question: 用户问题
            context: 检索到的上下文
            level: 查询级别 (L1/L2/L3/L4)
            query: ProcessedQuery 对象（L4 Self-RAG 需要）

        Returns:
            生成的回答文本
        """
        if self._llm is None:
            return self._fallback_answer(question, context, level)

        # L4 使用 Self-RAG 迭代引擎
        if level == "L4":
            answer = await self._generate_with_self_rag(question, context, query)
        else:
            # L1-L3 使用对应 Prompt 直接生成
            prompt_template = self._prompts.get(level, self._prompts["L1"])

            # 智能截断：按段落边界截断，不切断中间片段
            max_context_len = 6000
            if len(context) > max_context_len:
                context = self._truncate_context(context, max_context_len)

            prompt = prompt_template.format(context=context, question=question)

            try:
                answer = await self._llm.ainvoke(prompt)
                logger.info(f"[{level}] Generated {len(answer)} chars for '{question[:40]}...'")
                answer = answer.strip()
            except Exception as e:
                logger.error(f"[{level}] Generation failed: {e}")
                return self._fallback_answer(question, context, level)

        # 忠实度校验：检查回答是否有上下文依据，去除无依据内容
        if self._load_verify_setting() and answer and context:
            try:
                verified = await self._verify_faithfulness(answer, context)
                if verified != answer:
                    logger.info(
                        f"[{level}] Faithfulness verify: "
                        f"{len(answer)} → {len(verified)} chars"
                    )
                answer = verified
            except Exception as e:
                logger.debug(f"[{level}] Faithfulness verify skipped: {e}")

        return answer

    async def _generate_with_self_rag(
        self,
        question: str,
        context: str,
        query=None,
    ) -> str:
        """使用 Self-RAG 迭代引擎生成 L4 回答。"""
        from src.generation.self_rag import SelfRAG
        from src.core.settings import load_settings

        settings = load_settings()
        max_iter = settings.generation.max_self_rag_iterations

        if self._self_rag is None:
            self._self_rag = SelfRAG(
                llm_service=self._llm,
                hybrid_search=self._search,
                max_iterations=max_iter,
                pass_threshold=2.0,
            )

        logger.info(f"[L4] Starting Self-RAG (max {max_iter} iterations)")

        result = await self._self_rag.run(
            question=question,
            query=query,
            initial_context=context,
            initial_answer="",  # 让 Self-RAG 从头生成
        )

        iterations = result.get("total_iterations", 1)
        final_critique = result.get("final_critique")
        overall_score = final_critique.overall if final_critique else 0.0

        logger.info(
            f"[L4] Self-RAG completed: {iterations} iterations, "
            f"final score: {overall_score:.1f}"
        )

        return result.get("final_answer", "")

    async def _verify_faithfulness(self, answer: str, context: str) -> str:
        """生成后忠实度校验：检查回答是否有上下文依据，去除无依据的内容。

        对于指令遵循能力较弱的模型（如 mimo），生成的回答可能包含
        模型自己"补充"的知识。这个方法用 LLM 自己来检查并修正。

        Args:
            answer: 生成的回答
            context: 检索到的上下文

        Returns:
            修正后的回答（去掉了无依据的内容）
        """
        if not self._llm or not answer or not context:
            return answer

        verify_prompt = f"""请检查以下回答，判断每个事实陈述是否能在参考资料中找到依据。

参考资料：
{context[:3000]}

回答：
{answer[:2000]}

要求：
1. 保留所有能在参考资料中找到依据的内容
2. 对于合理推理部分（基于事实的推断），如果是合理的逻辑延伸，保留
3. 仅删除明显编造或与参考资料矛盾的内容
4. 不要改变回答的整体结构和语气
5. 如果大部分内容都有依据，只做最小改动

请输出修正后的回答（保持原格式和语言）："""

        try:
            verified = await self._llm.ainvoke(verify_prompt)
            if verified and len(verified) > len(answer) * 0.3:
                return verified.strip()
        except Exception as e:
            logger.debug(f"Faithfulness verification failed (non-fatal): {e}")

        return answer

    @staticmethod
    def _truncate_context(context: str, max_length: int = 6000) -> str:
        """按段落边界智能截断上下文。

        上下文格式为 [1] source (Page X):\ntext\n\n---\n\n[2] ...
        按 --- 分割，保留完整的段落，不切断中间内容。
        """
        paragraphs = context.split("\n\n---\n\n")
        kept = []
        total = 0
        for para in paragraphs:
            sep_len = 5 if kept else 0  # "---\n\n" separator
            if total + len(para) + sep_len > max_length and kept:
                break
            kept.append(para)
            total += len(para) + sep_len

        result = "\n\n---\n\n".join(kept)
        if len(kept) < len(paragraphs):
            result += f"\n\n...({len(paragraphs) - len(kept)} 个片段因长度限制省略)"
        return result

    @staticmethod
    def _fallback_answer(question: str, context: str, level: str) -> str:
        """无 LLM 时的降级回答。"""
        if not context:
            return f"[{level}] 未找到与问题相关的参考资料，无法回答。"
        snippet = context[:500] + ("..." if len(context) > 500 else "")
        return (
            f"[{level}] 查询: {question}\n\n"
            f"基于检索到的参考资料，以下是相关内容：\n\n{snippet}\n\n"
            f"（注：LLM 未配置，以上为检索结果摘要，非生成答案）"
        )
