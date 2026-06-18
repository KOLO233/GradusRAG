"""评估编排器。

编排完整的评估流程：加载测试集 → 运行 Pipeline → 计算指标 → 输出报告。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.core.types import EvalTestCase, EvaluationResult
from src.evaluation.metrics import (
    hit_rate_at_k,
    mrr,
    classification_accuracy,
    faithfulness_llm,
    relevancy_llm,
    context_recall,
    context_precision,
)
from src.evaluation.test_set import TestSetManager

logger = logging.getLogger(__name__)


def _run_async(coro):
    """运行异步协程，兼容已有事件循环的环境（FastAPI、Jupyter）。

    如果当前没有事件循环，用 asyncio.run()；
    如果已有循环（如 uvicorn），用 nest_asyncio 或新建线程执行。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # 没有运行中的循环，直接 run
        return asyncio.run(coro)
    else:
        # 已有循环（FastAPI/Jupyter），用 nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # nest_asyncio 不可用时，在新线程中用新事件循环执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()


class Evaluator:
    """评估编排器。

    Example:
        >>> evaluator = Evaluator(pipeline, llm_service)
        >>> result = evaluator.evaluate("golden_test_set.json")
        >>> print(f"Hit Rate: {result.hit_rate:.2%}")
        >>> print(f"MRR: {result.mrr:.2%}")
    """

    def __init__(
        self,
        pipeline=None,
        llm_service=None,
        test_set_manager: Optional[TestSetManager] = None,
    ):
        self._pipeline = pipeline
        self._llm = llm_service
        self._test_set_mgr = test_set_manager or TestSetManager()

    def evaluate(
        self,
        test_set_file: str = "formal_test_set.json",
        max_cases: int = 0,
        save_path: str = "",
    ) -> EvaluationResult:
        """运行完整评估。

        Args:
            test_set_file: 测试集文件名
            max_cases: 最大评估用例数（0=全部）
            save_path: 增量保存路径（每题保存一次，中断后可续跑）

        Returns:
            EvaluationResult 包含所有指标
        """
        test_cases = self._test_set_mgr.load(test_set_file)
        if not test_cases:
            logger.error(f"No test cases loaded from {test_set_file}")
            return EvaluationResult()

        if max_cases > 0:
            # 按级别均匀采样
            from collections import defaultdict
            by_level = defaultdict(list)
            for tc in test_cases:
                by_level[tc.expected_level].append(tc)
            per_level = max(max_cases // 4, 1)
            sampled = []
            for level in ["L1", "L2", "L3", "L4"]:
                sampled.extend(by_level.get(level, [])[:per_level])
            test_cases = sampled
            logger.info(f"Sampled {len(test_cases)} cases: {per_level} per level")

        logger.info(f"Evaluating {len(test_cases)} test cases...")

        predicted_levels = []
        expected_levels = []
        hit_rates = []
        mrr_scores = []
        faithfulness_scores = []
        relevancy_scores = []
        context_recall_scores = []
        context_precision_scores = []
        details = []

        # 加载已有进度（支持续跑）
        completed_questions = set()
        if save_path and Path(save_path).exists():
            try:
                existing = json.loads(Path(save_path).read_text(encoding="utf-8"))
                for d in existing.get("details", []):
                    q = d.get("question", "")
                    if q and "error" not in d:
                        completed_questions.add(q)
                        details.append(d)
                        predicted_levels.append(d.get("predicted_level", "L1"))
                        expected_levels.append(d.get("expected_level", "L1"))
                        if d.get("hit_rate") is not None:
                            hit_rates.append(d["hit_rate"])
                        if d.get("mrr") is not None:
                            mrr_scores.append(d["mrr"])
                        faithfulness_scores.append(d.get("faithfulness", 0))
                        relevancy_scores.append(d.get("answer_relevance", 0))
                        context_recall_scores.append(d.get("context_recall", 0))
                        context_precision_scores.append(d.get("context_precision", 0))
                logger.info(f"Resumed: {len(completed_questions)} already completed")
            except Exception:
                pass

        for i, tc in enumerate(test_cases):
            # 跳过已完成的题目
            if tc.question in completed_questions:
                continue

            logger.info(f"  [{i+1}/{len(test_cases)}] {tc.question[:50]}...")

            try:
                # 运行 Pipeline（兼容已有事件循环的环境）
                t0 = time.monotonic()
                response = _run_async(self._pipeline.run(tc.question))
                elapsed = (time.monotonic() - t0) * 1000

                # 记录分类结果
                predicted_levels.append(response.query_level)
                expected_levels.append(tc.expected_level)

                # 检索指标（用 chunk_id 匹配 ground truth）
                retrieved_ids = [
                    c.chunk_id for c in response.citations if c.chunk_id
                ]
                if tc.ground_truth_chunks:
                    hr = hit_rate_at_k(retrieved_ids, tc.ground_truth_chunks, k=5)
                    mrr_score = mrr(retrieved_ids, tc.ground_truth_chunks, k=10)
                    hit_rates.append(hr)
                    mrr_scores.append(mrr_score)
                else:
                    hr = None
                    mrr_score = None

                # 生成指标（与 LightRAG/RAGAS 对齐）
                context = "\n".join([c.text_snippet for c in response.citations])
                faith = faithfulness_llm(response.answer, context, self._llm)
                relev = relevancy_llm(tc.question, response.answer, self._llm)
                ctx_recall = context_recall(context, tc.expected_answer, self._llm)
                ctx_precision = context_precision(context, tc.question, tc.expected_answer, self._llm)
                faithfulness_scores.append(faith)
                relevancy_scores.append(relev)
                context_recall_scores.append(ctx_recall)
                context_precision_scores.append(ctx_precision)

                detail = {
                    "question": tc.question,
                    "expected_level": tc.expected_level,
                    "predicted_level": response.query_level,
                    "level_correct": response.query_level == tc.expected_level,
                    "hit_rate": hr,
                    "mrr": mrr_score,
                    "faithfulness": faith,
                    "answer_relevance": relev,
                    "context_recall": ctx_recall,
                    "context_precision": ctx_precision,
                    "elapsed_ms": elapsed,
                    "answer_preview": response.answer[:200],
                }
                details.append(detail)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                details.append({
                    "question": tc.question,
                    "expected_level": tc.expected_level,
                    "error": str(e),
                })

            # 增量保存
            if save_path:
                self._save_partial(save_path, details)

        # 汇总指标（与 LightRAG/RAGAS 对齐的 6 个指标）
        result = EvaluationResult(
            hit_rate=_avg(hit_rates) if hit_rates else 0.0,
            mrr=_avg(mrr_scores) if mrr_scores else 0.0,
            faithfulness=_avg(faithfulness_scores) if faithfulness_scores else 0.0,
            answer_relevance=_avg(relevancy_scores) if relevancy_scores else 0.0,
            context_recall=_avg(context_recall_scores) if context_recall_scores else 0.0,
            context_precision=_avg(context_precision_scores) if context_precision_scores else 0.0,
            classification_accuracy=classification_accuracy(predicted_levels, expected_levels),
            total_cases=len(test_cases),
            details=details,
        )

        logger.info(f"Evaluation complete:")
        logger.info(f"  Classification Accuracy: {result.classification_accuracy:.2%}")
        logger.info(f"  Hit Rate@5: {result.hit_rate:.2%}")
        logger.info(f"  MRR: {result.mrr:.2%}")
        logger.info(f"  Faithfulness: {result.faithfulness:.2%}")
        logger.info(f"  Answer Relevance: {result.answer_relevance:.2%}")
        logger.info(f"  Context Recall: {result.context_recall:.2%}")
        logger.info(f"  Context Precision: {result.context_precision:.2%}")

        return result

    @staticmethod
    def _save_partial(save_path: str, details: list):
        """增量保存评估结果。"""
        data = {"details": details, "completed": len(details)}
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def evaluate_by_level(
        self,
        test_set_file: str = "formal_test_set.json",
    ) -> Dict[str, EvaluationResult]:
        """按级别分别评估。"""
        test_cases = self._test_set_mgr.load(test_set_file)
        if not test_cases:
            return {}

        results = {}
        for level in ["L1", "L2", "L3", "L4"]:
            level_cases = self._test_set_mgr.filter_by_level(test_cases, level)
            if not level_cases:
                continue

            # 临时保存级别测试集
            level_file = f"_temp_{level}.json"
            self._test_set_mgr.save(level_file, level_cases)

            result = self.evaluate(level_file)
            results[level] = result

            # 清理临时文件
            temp_path = self._test_set_mgr._data_dir / level_file
            if temp_path.exists():
                temp_path.unlink()

        return results


def _avg(values: List[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0
