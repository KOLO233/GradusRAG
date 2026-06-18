"""四级查询分类器。

将用户问题分为 L1-L4 四个复杂度级别，并路由到对应的检索/生成策略。

级别定义（基于开题报告）：
- L1 显性事实：答案直接出现在文档中（"什么是X？"）
- L2 隐性事实：需要跨段落/文档推理（"比较A和B"）
- L3 可解释原理：需要领域知识推理（"为什么X会导致Y？"）
- L4 隐藏原理：需要深层推理和假设（"如果改变X会怎样？"）
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.core.types import QueryClassification
from src.core.settings import QueryClassifierSettings

logger = logging.getLogger(__name__)


# ===========================================================================
# 规则基线分类器
# ===========================================================================

# 各级别的匹配规则，每条规则有权重
# 高权重规则（模式匹配）优先于低权重规则（关键词匹配）
_LEVEL_RULES = [
    # ===== L1: 显性事实（定义、列举、描述）=====
    # 中文模式
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^什么是"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"是什么[？?。]?$"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^列举|^列出|^请列出"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"有哪些[？?。]?$"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^请?介绍一?下"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^描述一?下?"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"(的|有哪些)(主要)?(分类|类型|种类|形式)"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"^常见的.{2,10}(有哪些|包括)"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"的基本(流程|步骤|过程|概念|定义)"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"^请?简述|^简要说明|^简单说说"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"的(核心|基本|主要)(概念|定义|思想|特征|特点)是什么"},
    # 英文模式
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^what is (?!the (difference|distinction|relationship|reason|cause|mechanism))"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^what are\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^define\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^list\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^name the\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^who (is|was|are|were)\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^when (is|was|did)\b"},
    {"level": "L1", "weight": 3.0, "type": "pattern", "pattern": r"^where (is|was|are)\b"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"^describe\b"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"^what (are the|is the) (types|categories|classification|main) of\b"},
    {"level": "L1", "weight": 2.0, "type": "pattern", "pattern": r"^give (a |an )?(brief )?overview of\b"},
    # 中英文关键词
    {"level": "L1", "weight": 1.5, "type": "keyword",
     "keywords": ["定义", "概念", "含义", "意思", "介绍", "简述", "概述", "描述", "分类", "类型",
                  "what is", "define", "definition", "who is", "who was", "when was",
                  "where is", "list", "name", "describe", "overview"]},

    # ===== L2: 隐性事实（比较、对比、区别）=====
    # 中文模式
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"和.{1,15}(区别|异同|不同|对比|比较|优劣)"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"比较.{1,15}(与|和)"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"相比|相较|相对于"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"(分别|各自|各).{0,10}(适用于|用于|有什么|是什么|在哪|有什么区别|有什么特点|有什么优势)"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"和.{1,15}(分别|各自)"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"有什么(区别|不同|异同|差异|优劣)"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"的主要(区别|差异|不同|异同)是"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"哪个(更好|更优|更强|更适合|更有效)"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"^X和Y|^A和B|^两者"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"(优势|劣势|优缺点|利弊)"},
    # 英文模式
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"(what is|what's) the (difference|distinction) between"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"^compare\b"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"\bvs\.?\b"},
    {"level": "L2", "weight": 3.0, "type": "pattern", "pattern": r"^how (is|are|does) .+ different from"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"^which .+ (is|are) (better|worse|faster|slower|more effective)"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"what (are the|is the) (advantages|disadvantages|pros and cons)"},
    {"level": "L2", "weight": 2.0, "type": "pattern", "pattern": r"^how does .+ (compare|differ)"},
    # 关键词
    {"level": "L2", "weight": 1.5, "type": "keyword",
     "keywords": ["比较", "区别", "异同", "对比", "优劣", "不同", "相似", "差异", "分别",
                  "利弊", "优缺点", "各自",
                  "compare", "versus", "difference", "distinguish", "similar",
                  "different", "better", "worse", "advantage", "disadvantage"]},

    # ===== L3: 可解释原理（因果推理、机制解释）=====
    # 中文模式
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"^为什么"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"的原因是什么"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^解释.{2,15}(原理|机制|原因|工作原理)"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"(是|会|能)怎么(产生|形成|发生|导致)的"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"(是|会|能)如何(产生|形成|发生|导致|运作|工作)的"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"的(工作|运作|运行)(原理|机制|过程)是什么"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"(导致|造成|引起)X?(的|是因为)"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^怎样理解|^如何理解|^怎么理解"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"的根本原因|的本质是什么|的内在机理"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^解释一?下?.{2,15}(怎么|如何|为什么)"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"背后的(原因|原理|机制|逻辑)"},
    # 英文模式
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"^why\b"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"^explain (why|how)\b"},
    {"level": "L3", "weight": 3.0, "type": "pattern", "pattern": r"what (is|are) the (reason|cause|mechanism)"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^how does .+ work\b"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^what causes?\b"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^how (is|are) .+ (produced|formed|generated|caused)\b"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^what (is|are) the (underlying|fundamental) (mechanism|principle|reason)"},
    {"level": "L3", "weight": 2.0, "type": "pattern", "pattern": r"^explain the (mechanism|process|principle) (of|behind|behind)\b"},
    # 关键词
    {"level": "L3", "weight": 1.5, "type": "keyword",
     "keywords": ["为什么", "原因", "机理", "原理", "机制", "推导", "如何理解", "本质",
                  "导致", "造成", "引发", "产生",
                  "why", "mechanism", "cause", "reason", "explain", "how does",
                  "underlying", "fundamental", "principle"]},

    # ===== L4: 隐藏原理（假设推理、预测、推断）=====
    # 中文模式（去掉 ^ 锚定，匹配句中任意位置的假设词）
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"如果.{2,40}(会|将|可能|怎样|怎么办)"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"假设.{2,40}(会|那么|可能)"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"倘若.{2,40}(会|那么|可能)"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"若.{2,30}(会|则|那么|可能)"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"假如.{2,40}(会|那么|可能)"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"假定.{2,40}(会|那么|可能)"},
    # 新增：无"如果"但有假设语气的模式
    {"level": "L4", "weight": 2.5, "type": "pattern", "pattern": r"(替代|替换|改为|换成|移除|去掉|取消|增大|减小|翻倍).{2,20}(会|将|可能|怎样|如何|有什么)"},
    {"level": "L4", "weight": 2.5, "type": "pattern", "pattern": r"(完全|彻底|全部).{2,20}(替代|替换|移除).{2,20}(会|将|可能|怎样)"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"基于.{2,20}(推测|推断|预测|设想|假设)"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"试想|设想一?下?"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"(可能|会|将)有什么(影响|后果|变化|结果)"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"(能否|可以|是否能)通过.{2,20}(来|实现|达到)"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"在.{2,20}(情况下|条件下|假设下).{2,20}(会|可能|将)"},
    # 新增：反事实推理
    {"level": "L4", "weight": 2.5, "type": "pattern", "pattern": r"(没有|缺乏|缺少|去除|不存在).{2,20}(会|将|可能|怎样|如何)"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"(理论上|从理论上|从原理上).{2,20}(会|能|可以|可能)"},
    # 英文模式
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"^what would happen if\b"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"^what if\b"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"^suppose\b"},
    {"level": "L4", "weight": 3.0, "type": "pattern", "pattern": r"^imagine\b"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"^how would .+ change if\b"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"^if .+ (were|was|had) .+ (what would|how would)\b"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"^assuming .+ (what|how|would)\b"},
    {"level": "L4", "weight": 2.0, "type": "pattern", "pattern": r"^could .+ (lead to|result in|cause)\b"},
    # 关键词（扩充假设性词汇）
    {"level": "L4", "weight": 1.5, "type": "keyword",
     "keywords": ["假设", "推测", "预测", "假如", "推断", "设想", "倘若", "假定",
                  "会怎样", "会如何", "会有什么", "会产生什么", "会导致什么",
                  "替代", "替换", "移除后", "去掉后", "取消后",
                  "hypothetically", "suppose", "imagine", "what if", "predict",
                  "assuming", "theoretically"]},
]


class RuleClassifier:
    """基于规则的快速查询分类器。

    使用加权评分机制：
    - 模式匹配（高权重 3.0）：句式结构决定级别，如"什么是X"→L1
    - 关键词匹配（中权重 1.5）：领域关键词辅助判断
    - 普通关键词（低权重 1.0）：兜底判断

    这种设计避免了"什么是注意力机制"被误判为 L3 的问题，
    因为"什么是"的模式权重(3.0)高于"机制"的关键词权重(1.0)。
    """

    def classify(self, query: str) -> QueryClassification:
        """对查询进行加权评分分类。

        遍历所有规则，为每个级别累计分数，取最高分的级别。
        """
        query_lower = query.lower().strip()
        scores = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0}
        matched_reasons = {"L1": [], "L2": [], "L3": [], "L4": []}

        for rule in _LEVEL_RULES:
            level = rule["level"]
            weight = rule["weight"]

            if rule["type"] == "pattern":
                if re.search(rule["pattern"], query, re.IGNORECASE):
                    scores[level] += weight
                    matched_reasons[level].append(f"pattern:{rule['pattern']}")

            elif rule["type"] == "keyword":
                for kw in rule["keywords"]:
                    if kw in query_lower:
                        scores[level] += weight
                        matched_reasons[level].append(f"keyword:{kw}")
                        break  # 每条规则只匹配一次

        # 选最高分的级别
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]

        if best_score == 0:
            return QueryClassification(
                level="L1",
                confidence=0.5,
                query_type="factual",
                reasoning="No rule matched, default to L1",
            )

        # 计算置信度（归一化到 0.5-0.95）
        total = sum(scores.values())
        confidence = min(0.5 + 0.45 * (best_score / total), 0.95) if total > 0 else 0.5

        return QueryClassification(
            level=best_level,
            confidence=round(confidence, 2),
            query_type=self._infer_type(best_level),
            reasoning=f"Scored {best_score:.1f}: {', '.join(matched_reasons[best_level][:3])}",
        )

    def _infer_type(self, level: str) -> str:
        type_map = {"L1": "factual", "L2": "comparative", "L3": "causal", "L4": "hypothetical"}
        return type_map.get(level, "factual")


# ===========================================================================
# LLM 分类器
# ===========================================================================

CLASSIFICATION_PROMPT = """你是一个查询复杂度分类器。请将用户问题分为以下四个级别之一：

L1 (显性事实): 答案直接出现在文档中的简单事实查询。
  特征：定义、概念解释、列举、简单事实
  示例："什么是光合作用？" "列举Python的数据类型"

L2 (隐性事实): 需要跨段落或跨文档推理的事实查询。
  特征：比较、对比、区别、综合多个信息点
  示例："比较监督学习和无监督学习的区别" "A药和B药的副作用有何不同"

L3 (可解释原理): 需要领域知识进行因果推理的查询。
  特征：为什么、原因、机理、深层解释
  示例："为什么该患者服用A药后出现B症状？" "解释梯度消失的原因"

L4 (隐藏原理): 需要深层推理、假设或预测的查询。
  特征：假设性、预测性、推断性、反事实推理
  示例："如果将学习率增大10倍会怎样？" "假设该药物作用于C受体，可能的机制是什么？"

用户问题：{query}

请以 JSON 格式输出：
{{"level": "L1/L2/L3/L4", "confidence": 0.0-1.0, "query_type": "factual/comparative/causal/hypothetical", "reasoning": "分类理由"}}"""


class LLMClassifier:
    """基于 LLM 的精确查询分类器。

    使用 LLM 的结构化输出进行四级分类，准确率高但延迟较高。
    """

    def __init__(self, llm_func=None):
        """
        Args:
            llm_func: LLM 调用函数，签名为 (prompt: str) -> str
        """
        self._llm_func = llm_func

    async def classify(self, query: str) -> QueryClassification:
        """使用 LLM 对查询进行分类。"""
        if self._llm_func is None:
            logger.warning("LLM function not configured, falling back to rule-based")
            return RuleClassifier().classify(query)

        prompt = CLASSIFICATION_PROMPT.format(query=query)

        try:
            import json
            response = await self._llm_func(prompt)
            # 尝试解析 JSON 响应
            result = json.loads(response)
            return QueryClassification(
                level=result.get("level", "L1"),
                confidence=float(result.get("confidence", 0.5)),
                query_type=result.get("query_type", "factual"),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return RuleClassifier().classify(query)


# ===========================================================================
# 混合分类器（默认）
# ===========================================================================

class HybridClassifier:
    """混合查询分类器。

    流程：
    1. 规则快速预分类
    2. 如果置信度低于阈值，使用 LLM 精确确认
    3. 综合两个结果给出最终分类

    这是系统默认使用的分类器。
    """

    def __init__(
        self,
        settings: QueryClassifierSettings,
        llm_func=None,
    ):
        self._settings = settings
        self._rule_classifier = RuleClassifier()
        self._llm_classifier = LLMClassifier(llm_func=llm_func)

    async def classify(self, query: str) -> QueryClassification:
        """混合分类：规则 + LLM。"""
        # Step 1: 规则预分类
        rule_result = self._rule_classifier.classify(query)

        # 如果模式为纯规则，直接返回
        if self._settings.mode == "rule":
            return rule_result

        # Step 2: 检查置信度
        if rule_result.confidence >= self._settings.llm_threshold:
            logger.debug(
                f"Rule classifier confident enough "
                f"({rule_result.confidence:.2f} >= {self._settings.llm_threshold}), "
                f"level={rule_result.level}"
            )
            return rule_result

        # Step 3: LLM 精确分类
        if self._settings.mode in ("llm", "hybrid"):
            logger.debug(
                f"Rule confidence low ({rule_result.confidence:.2f}), "
                f"invoking LLM classifier"
            )
            llm_result = await self._llm_classifier.classify(query)

            # 以 LLM 结果为主，规则结果为参考
            if llm_result.confidence > rule_result.confidence:
                return llm_result

        return rule_result


def create_classifier(
    settings: QueryClassifierSettings,
    llm_func=None,
) -> HybridClassifier:
    """工厂函数：根据配置创建分类器。"""
    return HybridClassifier(settings=settings, llm_func=llm_func)
