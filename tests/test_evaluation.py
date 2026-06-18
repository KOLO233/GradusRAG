"""评估模块测试。

验证指标计算和测试集管理的正确性。
"""

import json
import pytest
from pathlib import Path
from src.core.types import EvalTestCase
from src.evaluation.metrics import (
    hit_rate_at_k,
    mrr,
    classification_accuracy,
    _rule_based_faithfulness,
    _rule_based_relevancy,
)
from src.evaluation.test_set import TestSetManager


# ===========================================================================
# 指标测试
# ===========================================================================

class TestHitRate:

    def test_hit(self):
        assert hit_rate_at_k(["a", "b", "c"], ["b"], k=5) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["a", "b", "c"], ["d"], k=5) == 0.0

    def test_hit_at_k_boundary(self):
        # 正确答案在第3位，k=2 时未命中
        assert hit_rate_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
        # k=3 时命中
        assert hit_rate_at_k(["a", "b", "c"], ["c"], k=3) == 1.0

    def test_empty_ground_truth(self):
        assert hit_rate_at_k(["a", "b"], [], k=5) == 0.0

    def test_multiple_ground_truth(self):
        # 多个正确答案，命中任一即可
        assert hit_rate_at_k(["a", "b", "c"], ["x", "b", "y"], k=5) == 1.0


class TestMRR:

    def test_first_rank(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_rank(self):
        assert mrr(["a", "b", "c"], ["b"]) == 0.5

    def test_third_rank(self):
        assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1/3)

    def test_miss(self):
        assert mrr(["a", "b", "c"], ["d"]) == 0.0

    def test_multiple_ground_truth(self):
        # 第一个命中的是 "b"，排在第2位
        assert mrr(["a", "b", "c"], ["x", "b", "y"]) == 0.5


class TestClassificationAccuracy:

    def test_all_correct(self):
        assert classification_accuracy(["L1", "L2", "L3"], ["L1", "L2", "L3"]) == 1.0

    def test_all_wrong(self):
        assert classification_accuracy(["L1", "L1", "L1"], ["L2", "L3", "L4"]) == 0.0

    def test_partial(self):
        assert classification_accuracy(["L1", "L2", "L1"], ["L1", "L2", "L3"]) == pytest.approx(2/3)

    def test_empty(self):
        assert classification_accuracy([], []) == 0.0


class TestRuleBasedFaithfulness:

    def test_high_faithfulness(self):
        context = "机器学习是人工智能的子领域，通过数据自动学习规律"
        answer = "机器学习是人工智能的一个子领域"
        score = _rule_based_faithfulness(answer, context)
        assert score > 0.5

    def test_low_faithfulness(self):
        context = "机器学习是人工智能的子领域"
        answer = "今天天气很好适合出去玩"
        score = _rule_based_faithfulness(answer, context)
        assert score < 0.3

    def test_empty(self):
        assert _rule_based_faithfulness("", "context") == 0.0


class TestRuleBasedRelevancy:

    def test_high_relevancy(self):
        score = _rule_based_relevancy("什么是机器学习？", "机器学习是人工智能的子领域")
        assert score > 0.3

    def test_low_relevancy(self):
        score = _rule_based_relevancy("什么是机器学习？", "今天天气很好")
        assert score < 0.5

    def test_empty(self):
        assert _rule_based_relevancy("", "answer") == 0.0


# ===========================================================================
# 测试集管理测试
# ===========================================================================

class TestTestSetManager:

    def test_save_and_load(self, tmp_path):
        manager = TestSetManager(data_dir=tmp_path)
        cases = [
            EvalTestCase(question="Q1", expected_level="L1", category="定义"),
            EvalTestCase(question="Q2", expected_level="L2", category="比较"),
        ]
        manager.save("test.json", cases)

        loaded = manager.load("test.json")
        assert len(loaded) == 2
        assert loaded[0].question == "Q1"
        assert loaded[1].expected_level == "L2"

    def test_filter_by_level(self, tmp_path):
        manager = TestSetManager(data_dir=tmp_path)
        cases = [
            EvalTestCase(question="Q1", expected_level="L1"),
            EvalTestCase(question="Q2", expected_level="L2"),
            EvalTestCase(question="Q3", expected_level="L1"),
        ]
        l1_cases = manager.filter_by_level(cases, "L1")
        assert len(l1_cases) == 2

    def test_stats(self, tmp_path):
        manager = TestSetManager(data_dir=tmp_path)
        cases = [
            EvalTestCase(question="Q1", expected_level="L1", category="定义"),
            EvalTestCase(question="Q2", expected_level="L2", category="比较"),
            EvalTestCase(question="Q3", expected_level="L1", category="定义"),
        ]
        stats = manager.get_stats(cases)
        assert stats["total"] == 3
        assert stats["by_level"]["L1"] == 2
        assert stats["by_category"]["定义"] == 2

    def test_load_golden_test_set(self):
        """加载实际的 Golden Test Set。"""
        manager = TestSetManager()
        files = manager.list_files()
        assert "golden_test_set.json" in files

        test_set = manager.load("golden_test_set.json")
        assert len(test_set) >= 20

        stats = manager.get_stats(test_set)
        assert stats["by_level"]["L1"] >= 5
        assert stats["by_level"]["L2"] >= 5
        assert stats["by_level"]["L3"] >= 5
        assert stats["by_level"]["L4"] >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
