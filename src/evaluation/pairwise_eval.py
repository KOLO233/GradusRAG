"""四维度 Pairwise 评估模块。

参考 LightRAG 论文的评估方法，从四个维度对比两个系统的回答质量：
1. 全面性 (Comprehensiveness)：回答是否覆盖了问题的所有方面
2. 多样性 (Diversity)：回答是否提供了多种视角、不同角度的信息
3. 启发性 (Empowerment)：回答是否帮助用户理解问题本质
4. 综合质量 (Overall)：综合考虑，哪个回答更好

使用 LLM-as-Judge 进行评估，每个维度输出 Winner + Reasoning。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 四维度定义
DIMENSIONS = ["comprehensiveness", "diversity", "empowerment", "overall"]

DIMENSION_NAMES_ZH = {
    "comprehensiveness": "全面性",
    "diversity": "多样性",
    "empowerment": "启发性",
    "overall": "综合质量",
}

DIMENSION_DESCRIPTIONS = {
    "comprehensiveness": "回答是否覆盖了问题的所有方面？是否遗漏了重要信息？",
    "diversity": "回答是否提供了多种视角、不同角度的信息？是否有丰富的解释和例子？",
    "empowerment": "回答是否帮助用户理解问题本质，能指导进一步学习和思考？",
    "overall": "综合考虑全面性、多样性、启发性，哪个回答整体质量更高？",
}


# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass
class DimensionResult:
    """单个维度的评估结果。"""
    winner: str          # "Answer 1" / "Answer 2" / "Tie"
    reason: str          # LLM 的判断理由


@dataclass
class PairwiseResult:
    """单条查询的 Pairwise 评估结果。"""
    query_id: str
    question: str
    level: str
    domain: str
    answer_1_system: str   # Answer 1 来自哪个系统
    answer_2_system: str   # Answer 2 来自哪个系统
    answer_1: str
    answer_2: str
    dimensions: Dict[str, DimensionResult] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "level": self.level,
            "domain": self.domain,
            "answer_1_system": self.answer_1_system,
            "answer_2_system": self.answer_2_system,
            "answer_1": self.answer_1[:500],
            "answer_2": self.answer_2[:500],
            "dimensions": {
                dim: {"winner": dr.winner, "reason": dr.reason}
                for dim, dr in self.dimensions.items()
            },
            "latency_ms": self.latency_ms,
        }


@dataclass
class PairwiseSummary:
    """Pairwise 评估汇总统计。"""
    total: int = 0
    # 每个维度的胜率统计
    dim_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # 按级别统计
    level_stats: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    answer_1_system: str = ""
    answer_2_system: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "answer_1_system": self.answer_1_system,
            "answer_2_system": self.answer_2_system,
            "dim_stats": self.dim_stats,
            "level_stats": self.level_stats,
        }

    def summary_table(self) -> str:
        """生成可读的汇总表格。"""
        lines = []
        lines.append(f"Pairwise 评估汇总: {self.answer_1_system} vs {self.answer_2_system} ({self.total} 题)")
        lines.append("=" * 70)

        # 总体统计
        header = f"{'维度':<16} {self.answer_1_system + ' 胜':>12} {'平局':>8} {self.answer_2_system + ' 胜':>12}"
        lines.append(header)
        lines.append("-" * 70)
        for dim in DIMENSIONS:
            stats = self.dim_stats.get(dim, {})
            w1 = stats.get("Answer 1", 0)
            tie = stats.get("Tie", 0)
            w2 = stats.get("Answer 2", 0)
            name = DIMENSION_NAMES_ZH.get(dim, dim)
            lines.append(f"{name:<14} {w1:>10} ({w1/max(self.total,1)*100:4.1f}%) {tie:>4} ({tie/max(self.total,1)*100:4.1f}%) {w2:>10} ({w2/max(self.total,1)*100:4.1f}%)")

        # 按级别统计（仅 overall 维度）
        lines.append("")
        lines.append("按级别统计（综合质量）：")
        lines.append(f"{'级别':<8} {'题数':>6} {self.answer_1_system + ' 胜':>12} {'平局':>8} {self.answer_2_system + ' 胜':>12}")
        lines.append("-" * 60)
        for level in ["L1", "L2", "L3", "L4"]:
            level_data = self.level_stats.get(level, {})
            overall = level_data.get("overall", {})
            w1 = overall.get("Answer 1", 0)
            tie = overall.get("Tie", 0)
            w2 = overall.get("Answer 2", 0)
            total_level = w1 + tie + w2
            if total_level > 0:
                lines.append(f"{level:<8} {total_level:>4}   {w1:>4} ({w1/total_level*100:4.1f}%) {tie:>4} ({tie/total_level*100:4.1f}%) {w2:>4} ({w2/total_level*100:4.1f}%)")

        return "\n".join(lines)


# ===========================================================================
# LLM Pairwise 评估 Prompt
# ===========================================================================

PAIRWISE_PROMPT_TEMPLATE = """你是一个专业的回答质量评估专家。请从以下四个维度比较两个回答的质量。

用户问题：{question}

Answer 1（来自 {system_1}）：
{answer_1}

Answer 2（来自 {system_2}）：
{answer_2}

---

请从以下四个维度进行评估（请注意：不要因为回答来自哪个系统而产生偏见，仅根据回答本身的质量评判）：

1. **全面性 (Comprehensiveness)**：回答是否覆盖了问题的所有方面？是否遗漏了重要信息？
2. **多样性 (Diversity)**：回答是否提供了多种视角、不同角度的信息？是否有丰富的解释和例子？
3. **启发性 (Empowerment)**：回答是否帮助用户理解问题本质，能指导进一步学习和思考？
4. **综合质量 (Overall)**：综合考虑全面性、多样性、启发性，哪个回答整体质量更高？

对于每个维度，请选择 "Answer 1"、"Answer 2" 或 "Tie"（持平），并简要说明理由。

请以 JSON 格式输出（不要输出其他内容）：
```json
{{
  "comprehensiveness": {{"winner": "Answer 1/Answer 2/Tie", "reason": "..."}},
  "diversity": {{"winner": "Answer 1/Answer 2/Tie", "reason": "..."}},
  "empowerment": {{"winner": "Answer 1/Answer 2/Tie", "reason": "..."}},
  "overall": {{"winner": "Answer 1/Answer 2/Tie", "reason": "..."}}
}}
```"""


def _parse_pairwise_response(response: str) -> Optional[Dict[str, Dict[str, str]]]:
    """解析 LLM 的 JSON 格式 Pairwise 评估结果。

    Args:
        response: LLM 的原始输出

    Returns:
        解析后的维度结果字典，解析失败返回 None
    """
    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接解析整个响应
        json_str = response.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试用正则逐个提取
        logger.warning(f"JSON 解析失败，尝试正则提取: {response[:200]}")
        return _fallback_parse(response)

    # 验证结构
    result = {}
    for dim in DIMENSIONS:
        if dim in data:
            entry = data[dim]
            winner = entry.get("winner", "Tie")
            reason = entry.get("reason", "")
            # 标准化 winner
            winner = _normalize_winner(winner)
            result[dim] = {"winner": winner, "reason": reason}
        else:
            result[dim] = {"winner": "Tie", "reason": "解析缺失"}

    return result


def _normalize_winner(winner: str) -> str:
    """标准化 winner 字段。"""
    w = winner.strip().lower()
    if "answer 1" in w or "answer1" in w or w == "1":
        return "Answer 1"
    elif "answer 2" in w or "answer2" in w or w == "2":
        return "Answer 2"
    else:
        return "Tie"


def _fallback_parse(response: str) -> Optional[Dict[str, Dict[str, str]]]:
    """当 JSON 解析失败时的兜底正则提取。"""
    result = {}
    for dim in DIMENSIONS:
        # 匹配 "dimension": {"winner": "...", "reason": "..."}
        pattern = rf'"{dim}".*?"winner"\s*:\s*"([^"]+)".*?"reason"\s*:\s*"([^"]*)"'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result[dim] = {
                "winner": _normalize_winner(match.group(1)),
                "reason": match.group(2)[:200],
            }
        else:
            result[dim] = {"winner": "Tie", "reason": "解析失败"}

    return result if result else None


# ===========================================================================
# 核心评估函数
# ===========================================================================

def pairwise_4d_eval(
    question: str,
    answer_1: str,
    answer_2: str,
    system_1_name: str = "System A",
    system_2_name: str = "System B",
    llm_service=None,
) -> Dict[str, DimensionResult]:
    """对两个回答进行四维度 Pairwise 评估。

    Args:
        question: 用户问题
        answer_1: 第一个系统的回答
        answer_2: 第二个系统的回答
        system_1_name: 第一个系统的名称（用于 Prompt，避免偏见可设为 "Answer 1"）
        system_2_name: 第二个系统的名称
        llm_service: LLM 服务实例

    Returns:
        四个维度的评估结果字典
    """
    if llm_service is None:
        logger.warning("No LLM service provided, returning Tie for all dimensions")
        return {dim: DimensionResult(winner="Tie", reason="无 LLM 服务") for dim in DIMENSIONS}

    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question,
        system_1=system_1_name,
        system_2=system_2_name,
        answer_1=answer_1[:3000],   # 截断避免超长
        answer_2=answer_2[:3000],
    )

    try:
        response = llm_service.invoke(prompt)
        parsed = _parse_pairwise_response(response)

        if parsed is None:
            logger.warning("Pairwise evaluation parse failed, returning Tie")
            return {dim: DimensionResult(winner="Tie", reason="解析失败") for dim in DIMENSIONS}

        return {
            dim: DimensionResult(
                winner=parsed.get(dim, {}).get("winner", "Tie"),
                reason=parsed.get(dim, {}).get("reason", ""),
            )
            for dim in DIMENSIONS
        }

    except Exception as e:
        logger.error(f"Pairwise LLM evaluation failed: {e}")
        return {dim: DimensionResult(winner="Tie", reason=f"评估异常: {e}") for dim in DIMENSIONS}


def run_pairwise_evaluation(
    test_cases: List[Any],
    pipeline_1,
    pipeline_2,
    system_1_name: str = "System A",
    system_2_name: str = "System B",
    llm_service=None,
    save_path: str = "",
    max_cases: int = 0,
) -> PairwiseSummary:
    """运行完整的 Pairwise 评估流程。

    对每个测试用例，分别用两个 pipeline 生成回答，然后用 LLM-as-Judge
    进行四维度对比评估。

    Args:
        test_cases: 测试用例列表（需要有 question, expected_level, domain 属性）
        pipeline_1: 第一个系统的 RAG Pipeline
        pipeline_2: 第二个系统的 RAG Pipeline
        system_1_name: 第一个系统名称
        system_2_name: 第二个系统名称
        llm_service: LLM 评估服务
        save_path: 增量保存路径
        max_cases: 最大评估用例数（0=全部）

    Returns:
        PairwiseSummary 汇总统计
    """
    import asyncio
    from collections import defaultdict

    if max_cases > 0:
        # 按级别均匀采样
        by_level = defaultdict(list)
        for tc in test_cases:
            by_level[getattr(tc, 'expected_level', 'L1')].append(tc)
        per_level = max(max_cases // 4, 1)
        sampled = []
        for level in ["L1", "L2", "L3", "L4"]:
            sampled.extend(by_level.get(level, [])[:per_level])
        test_cases = sampled

    # 加载已有进度
    completed: Dict[str, PairwiseResult] = {}
    if save_path and Path(save_path).exists():
        try:
            existing = json.loads(Path(save_path).read_text(encoding="utf-8"))
            for d in existing.get("details", []):
                qid = d.get("query_id", "")
                if qid:
                    completed[qid] = d
            logger.info(f"Pairwise: resumed {len(completed)} completed")
        except Exception:
            pass

    results: List[PairwiseResult] = list(completed.values())
    total = len(test_cases)

    for i, tc in enumerate(test_cases):
        question = tc.question if hasattr(tc, 'question') else tc.get("question", "")
        query_id = f"pw_{i:04d}"

        # 跳过已完成
        if query_id in completed:
            continue

        logger.info(f"Pairwise [{i+1}/{total}] {question[:50]}...")

        try:
            # 生成两个系统的回答
            t0 = time.monotonic()
            loop = asyncio.new_event_loop()
            resp_1 = loop.run_until_complete(pipeline_1.run(question))
            resp_2 = loop.run_until_complete(pipeline_2.run(question))
            elapsed = (time.monotonic() - t0) * 1000

            answer_1 = resp_1.answer
            answer_2 = resp_2.answer

            # LLM 四维度评估
            dim_results = pairwise_4d_eval(
                question=question,
                answer_1=answer_1,
                answer_2=answer_2,
                system_1_name=system_1_name,
                system_2_name=system_2_name,
                llm_service=llm_service,
            )

            result = PairwiseResult(
                query_id=query_id,
                question=question,
                level=getattr(tc, 'expected_level', 'L1'),
                domain=getattr(tc, 'domain', ''),
                answer_1_system=system_1_name,
                answer_2_system=system_2_name,
                answer_1=answer_1,
                answer_2=answer_2,
                dimensions=dim_results,
                latency_ms=elapsed,
            )
            results.append(result)

            # 增量保存
            if save_path:
                _save_incremental(save_path, results, system_1_name, system_2_name)

        except Exception as e:
            logger.error(f"Pairwise [{i+1}] failed: {e}")

    # 生成汇总
    summary = _compute_summary(results, system_1_name, system_2_name)
    return summary


def _compute_summary(
    results: List[PairwiseResult],
    system_1_name: str,
    system_2_name: str,
) -> PairwiseSummary:
    """从详细结果计算汇总统计。"""
    from collections import defaultdict

    summary = PairwiseSummary(
        total=len(results),
        answer_1_system=system_1_name,
        answer_2_system=system_2_name,
    )

    # 总体统计
    for dim in DIMENSIONS:
        summary.dim_stats[dim] = {"Answer 1": 0, "Tie": 0, "Answer 2": 0}

    # 按级别统计
    for level in ["L1", "L2", "L3", "L4"]:
        summary.level_stats[level] = {}
        for dim in DIMENSIONS:
            summary.level_stats[level][dim] = {"Answer 1": 0, "Tie": 0, "Answer 2": 0}

    for r in results:
        level = r.level if hasattr(r, 'level') else r.get("level", "L1")
        dims = r.dimensions if hasattr(r, 'dimensions') else r.get("dimensions", {})

        for dim in DIMENSIONS:
            if isinstance(dims.get(dim), DimensionResult):
                winner = dims[dim].winner
            elif isinstance(dims.get(dim), dict):
                winner = dims[dim].get("winner", "Tie")
            else:
                winner = "Tie"

            if winner in summary.dim_stats.get(dim, {}):
                summary.dim_stats[dim][winner] += 1
            if level in summary.level_stats and dim in summary.level_stats.get(level, {}):
                if winner in summary.level_stats[level][dim]:
                    summary.level_stats[level][dim][winner] += 1

    return summary


def _save_incremental(
    save_path: str,
    results: List[PairwiseResult],
    system_1_name: str,
    system_2_name: str,
):
    """增量保存评估结果到 JSON 文件。"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = _compute_summary(results, system_1_name, system_2_name)

    data = {
        "summary": summary.to_dict(),
        "details": [
            r.to_dict() if hasattr(r, 'to_dict') else r
            for r in results
        ],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
