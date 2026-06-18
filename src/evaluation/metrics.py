"""评估指标实现。

支持的指标：
- Hit Rate@K：Top-K 中是否包含正确文档
- MRR (Mean Reciprocal Rank)：正确文档的平均倒数排名
- Faithfulness：回答是否忠于检索内容
- Relevancy：回答与问题的相关性
- Classification Accuracy：查询分类准确率
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def hit_rate_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5,
) -> float:
    """Hit Rate@K：Top-K 检索结果中是否命中正确文档。

    Args:
        retrieved_ids: 检索结果的 chunk_id 列表（按相关性排序）
        ground_truth_ids: 标注的正确 chunk_id 列表
        k: 截断位置

    Returns:
        1.0 如果命中，0.0 如果未命中
    """
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gt = set(ground_truth_ids)
    return 1.0 if top_k & gt else 0.0


def mrr(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 10,
) -> float:
    """MRR (Mean Reciprocal Rank)：正确文档的倒数排名。

    Args:
        retrieved_ids: 检索结果的 chunk_id 列表
        ground_truth_ids: 标注的正确 chunk_id 列表
        k: 截断位置

    Returns:
        1/rank 的值（rank 为第一个正确结果的位置，1-indexed）
    """
    if not ground_truth_ids:
        return 0.0
    gt = set(ground_truth_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gt:
            return 1.0 / (i + 1)
    return 0.0


def classification_accuracy(
    predicted_levels: List[str],
    expected_levels: List[str],
) -> float:
    """查询分类准确率。

    Args:
        predicted_levels: 预测的级别列表 (L1/L2/L3/L4)
        expected_levels: 标注的级别列表

    Returns:
        准确率 (0.0 ~ 1.0)
    """
    if not predicted_levels or not expected_levels:
        return 0.0
    correct = sum(1 for p, e in zip(predicted_levels, expected_levels) if p == e)
    return correct / len(expected_levels)


def faithfulness_llm(
    answer: str,
    context: str,
    llm_service=None,
) -> float:
    """Faithfulness：回答是否忠于检索内容（需要 LLM 评估）。

    评估标准区分两类内容：
    - 核心事实（数据、定义、因果关系等）：必须在参考资料中有直接依据
    - 合理推理（基于事实的推断、综合、总结）：只要推理逻辑合理即可

    使用 3 级评分（减少 LLM 方差）：
    - 1.0 = 忠实（核心事实有依据，推理合理）
    - 0.5 = 部分忠实（大部分核心事实有依据，少量无法验证或推理有瑕疵）
    - 0.0 = 不忠实（核心事实无依据，或编造了不存在的信息）

    Args:
        answer: 生成的回答
        context: 检索到的上下文
        llm_service: LLM 服务（可选，无则用规则评估）

    Returns:
        忠实度分数 (0.0 / 0.5 / 1.0)
    """
    if not answer or not context:
        return 0.0

    if llm_service is None:
        return _rule_based_faithfulness(answer, context)

    prompt = f"""You are a fact-checking expert. Evaluate the faithfulness of the "Answer" against the "Context".

Important: Distinguish between two types of content, using different standards:

**A. Core Facts** (data, definitions, names, causal relationships, specific conclusions):
   → Must have direct support in the context. Unsupported core facts = unfaithful.

**B. Reasonable Inference** (deductions, synthesis, summaries based on facts):
   → As long as the reasoning logically follows from context facts, consider it faithful.
   → Word-for-word correspondence with the context is NOT required.

---

Context:
{context[:3000]}

Answer:
{answer[:2000]}

---

Score:
- **1** (Faithful): Core facts are all supported by context; inferences are logically sound.
- **0** (Partially Faithful): Most core facts supported, but some cannot be verified; or minor logical gaps in reasoning.
- **-1** (Unfaithful): Core facts are not supported by context; or fabricated specific information (data, names, causal claims).

Output only a single number: 1, 0, or -1"""

    try:
        import asyncio
        response = llm_service.invoke(prompt)
        # 提取数字
        import re
        match = re.search(r'(-1|0|1)', response.strip())
        if match:
            score = int(match.group())
            return {1: 1.0, 0: 0.5, -1: 0.0}.get(score, 0.5)
    except Exception as e:
        logger.error(f"LLM faithfulness evaluation failed: {e}")

    return _rule_based_faithfulness(answer, context)


def relevancy_llm(
    question: str,
    answer: str,
    llm_service=None,
) -> float:
    """Relevancy：回答与问题的相关性（需要 LLM 评估）。

    使用 3 级评分（减少 LLM 方差）。

    Args:
        question: 用户问题
        answer: 生成的回答
        llm_service: LLM 服务（可选）

    Returns:
        相关性分数 (0.0 / 0.5 / 1.0)
    """
    if not question or not answer:
        return 0.0

    if llm_service is None:
        return _rule_based_relevancy(question, answer)

    prompt = f"""Does the "Answer" actually address the "Question"?

Question: {question}

Answer: {answer[:1000]}

Score:
- **1** (Relevant): The answer directly and completely addresses the question.
- **0** (Partially Relevant): The answer partially addresses the question or is related but not fully on-topic.
- **-1** (Irrelevant): The answer does not address the question at all (off-topic or refusal to answer).

Output only a single number: 1, 0, or -1"""

    try:
        import asyncio
        response = llm_service.invoke(prompt)
        import re
        match = re.search(r'(-1|0|1)', response.strip())
        if match:
            score = int(match.group())
            return {1: 1.0, 0: 0.5, -1: 0.0}.get(score, 0.5)
    except Exception as e:
        logger.error(f"LLM relevancy evaluation failed: {e}")

    return _rule_based_relevancy(question, answer)


def context_recall(
    context: str,
    reference_answer: str,
    llm_service=None,
) -> float:
    """Context Recall：检索到的上下文是否覆盖了参考答案中的关键信息。

    与 LightRAG/RAGAS 的 context_recall 对齐。
    衡量检索系统的完整性——是否找到了所有相关信息。

    Args:
        context: 检索到的上下文
        reference_answer: 标注的参考答案
        llm_service: LLM 服务（可选）

    Returns:
        召回率分数 (0.0 ~ 1.0)
    """
    if not context or not reference_answer:
        return 0.0

    if llm_service is None:
        return _rule_based_context_recall(context, reference_answer)

    prompt = f"""Does the retrieved context cover the key information in the reference answer?

Reference Answer:
{reference_answer[:1000]}

Retrieved Context:
{context[:2000]}

Score:
- **1** (Full Coverage): The context covers all key information in the reference answer.
- **0** (Partial Coverage): The context covers some but not all key information.
- **-1** (Minimal Coverage): The context covers almost none of the key information.

Output only a single number: 1, 0, or -1"""

    try:
        response = llm_service.invoke(prompt)
        import re
        match = re.search(r'(-1|0|1)', response.strip())
        if match:
            score = int(match.group())
            return {1: 1.0, 0: 0.5, -1: 0.0}.get(score, 0.5)
    except Exception as e:
        logger.error(f"LLM context_recall evaluation failed: {e}")

    return _rule_based_context_recall(context, reference_answer)


def context_precision(
    context: str,
    question: str,
    reference_answer: str,
    llm_service=None,
) -> float:
    """Context Precision：检索到的上下文是否干净、无噪声。

    与 LightRAG/RAGAS 的 context_precision 对齐。
    衡量检索系统的精确性——检索到的内容是否都相关。

    Args:
        context: 检索到的上下文
        question: 用户问题
        reference_answer: 标注的参考答案
        llm_service: LLM 服务（可选）

    Returns:
        精确率分数 (0.0 ~ 1.0)
    """
    if not context:
        return 0.0

    if llm_service is None:
        return _rule_based_context_precision(context, question, reference_answer)

    prompt = f"""Is the retrieved context clean and relevant to the question?

Question: {question}

Reference Answer: {reference_answer[:500]}

Retrieved Context:
{context[:2000]}

Score:
- **1** (Clean): All context is relevant to the question, no noise.
- **0** (Mostly Clean): Context is partially relevant with some noise.
- **-1** (Noisy): Most of the context is noise or irrelevant.

Output only a single number: 1, 0, or -1"""

    try:
        response = llm_service.invoke(prompt)
        import re
        match = re.search(r'(-1|0|1)', response.strip())
        if match:
            score = int(match.group())
            return {1: 1.0, 0: 0.5, -1: 0.0}.get(score, 0.5)
    except Exception as e:
        logger.error(f"LLM context_precision evaluation failed: {e}")

    return _rule_based_context_precision(context, question, reference_answer)


# ===========================================================================
# 规则基线评估（不需要 LLM）
# ===========================================================================

def _rule_based_faithfulness(answer: str, context: str) -> float:
    """规则基线 Faithfulness：基于关键词重叠度。"""
    import jieba

    ans_tokens = set(jieba.cut(answer))
    ctx_tokens = set(jieba.cut(context))

    # 去除停用词
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "与", "对", "中", "上", "下"}
    ans_tokens -= stop
    ctx_tokens -= stop

    if not ans_tokens:
        return 0.0

    overlap = ans_tokens & ctx_tokens
    return min(len(overlap) / len(ans_tokens), 1.0)


def _rule_based_relevancy(question: str, answer: str) -> float:
    """规则基线 Relevancy：基于问题关键词在回答中的覆盖率。"""
    import jieba
    import re

    # 提取问题关键词
    q_tokens = set(jieba.cut(question))
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "什么", "怎么", "为什么", "如何", "哪些"}
    q_tokens -= stop
    q_tokens = {t for t in q_tokens if len(t) >= 2}

    if not q_tokens:
        return 0.0

    a_lower = answer.lower()
    matched = sum(1 for t in q_tokens if t in a_lower)
    return min(matched / len(q_tokens), 1.0)


def _rule_based_context_recall(context: str, reference_answer: str) -> float:
    """规则基线 Context Recall：参考答案的关键信息在上下文中的覆盖率。"""
    import jieba

    ref_tokens = set(jieba.cut(reference_answer))
    ctx_tokens = set(jieba.cut(context))

    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "与", "对", "中"}
    ref_tokens -= stop
    ref_tokens = {t for t in ref_tokens if len(t) >= 2}
    ctx_tokens -= stop

    if not ref_tokens:
        return 0.0

    overlap = ref_tokens & ctx_tokens
    return min(len(overlap) / len(ref_tokens), 1.0)


def _rule_based_context_precision(context: str, question: str, reference_answer: str) -> float:
    """规则基线 Context Precision：上下文中与问题/答案相关内容的比例。"""
    import jieba

    # 合并问题和参考答案的关键词作为"相关"标准
    relevant_tokens = set(jieba.cut(question + reference_answer))
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "什么", "怎么", "为什么"}
    relevant_tokens -= stop
    relevant_tokens = {t for t in relevant_tokens if len(t) >= 2}

    ctx_tokens = set(jieba.cut(context))
    ctx_tokens -= stop
    ctx_tokens = {t for t in ctx_tokens if len(t) >= 2}

    if not ctx_tokens:
        return 0.0

    overlap = ctx_tokens & relevant_tokens
    return min(len(overlap) / len(ctx_tokens), 1.0)
