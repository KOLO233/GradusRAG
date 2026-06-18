"""GradusRAG 查询分类器测试。

验证规则分类器能正确识别 L1-L4 四级查询。
"""

import pytest
from src.query_classifier.classifier import RuleClassifier


@pytest.fixture
def classifier():
    return RuleClassifier()


class TestRuleClassifier:
    """规则分类器测试。"""

    # ---- L1 显性事实 ----

    def test_l1_definition(self, classifier):
        """定义类问题应分类为 L1。"""
        result = classifier.classify("什么是光合作用？")
        assert result.level == "L1"
        assert result.confidence >= 0.7

    def test_l1_list(self, classifier):
        """列举类问题应分类为 L1。"""
        result = classifier.classify("列举Python的基本数据类型")
        assert result.level == "L1"

    def test_l1_what_is(self, classifier):
        """什么是类问题应分类为 L1。"""
        result = classifier.classify("什么是机器学习？")
        assert result.level == "L1"

    # ---- L2 隐性事实 ----

    def test_l2_compare(self, classifier):
        """比较类问题应分类为 L2。"""
        result = classifier.classify("比较监督学习和无监督学习的区别")
        assert result.level == "L2"

    def test_l2_difference(self, classifier):
        """区别类问题应分类为 L2。"""
        result = classifier.classify("Python和Java有什么不同？")
        assert result.level == "L2"

    # ---- L3 可解释原理 ----

    def test_l3_why(self, classifier):
        """为什么类问题应分类为 L3。"""
        result = classifier.classify("为什么会出现梯度消失现象？")
        assert result.level == "L3"

    def test_l3_mechanism(self, classifier):
        """机理类问题应分类为 L3。"""
        result = classifier.classify("解释注意力机制的原理")
        assert result.level == "L3"

    def test_l3_why_mechanism(self, classifier):
        """为什么+机制应分类为 L3。"""
        result = classifier.classify("为什么注意力机制能提升模型性能？")
        assert result.level == "L3"

    def test_l1_definition_with_mechanism(self, classifier):
        """什么是+机制 应分类为 L1（定义类），不是 L3。"""
        result = classifier.classify("什么是注意力机制？")
        assert result.level == "L1"

    def test_l1_definition_with_principle(self, classifier):
        """什么是+原理 应分类为 L1（定义类）。"""
        result = classifier.classify("什么是反向传播原理？")
        assert result.level == "L1"

    # ---- L4 隐藏原理 ----

    def test_l4_hypothetical(self, classifier):
        """假设性问题应分类为 L4。"""
        result = classifier.classify("如果将学习率增大10倍会怎样？")
        assert result.level == "L4"

    def test_l4_prediction(self, classifier):
        """预测性问题应分类为 L4。"""
        result = classifier.classify("假设该药物作用于C受体，推测其可能的副作用")
        assert result.level == "L4"

    # ---- 默认行为 ----

    def test_default_to_l1(self, classifier):
        """无法匹配的问题默认归为 L1。"""
        result = classifier.classify("你好")
        assert result.level == "L1"
        assert result.confidence == 0.5

    # ---- 英文查询 ----

    def test_en_l1_what_is(self, classifier):
        """英文 What is 应分类为 L1。"""
        result = classifier.classify("What is machine learning?")
        assert result.level == "L1"

    def test_en_l2_compare(self, classifier):
        """英文 compare 应分类为 L2。"""
        result = classifier.classify("What is the difference between BERT and GPT?")
        assert result.level == "L2"

    def test_en_l3_why(self, classifier):
        """英文 why 应分类为 L3。"""
        result = classifier.classify("Why does gradient vanishing happen in deep networks?")
        assert result.level == "L3"

    def test_en_l4_what_if(self, classifier):
        """英文 what if 应分类为 L4。"""
        result = classifier.classify("What would happen if all activation functions were ReLU?")
        assert result.level == "L4"


class TestQueryClassification:
    """查询分类结果测试。"""

    def test_classification_has_required_fields(self, classifier):
        """分类结果应包含所有必要字段。"""
        result = classifier.classify("什么是深度学习？")
        assert hasattr(result, "level")
        assert hasattr(result, "confidence")
        assert hasattr(result, "query_type")
        assert hasattr(result, "reasoning")

    def test_classification_serializable(self, classifier):
        """分类结果应可序列化为字典。"""
        result = classifier.classify("什么是深度学习？")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "level" in d
        assert "confidence" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
