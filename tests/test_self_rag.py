"""Self-RAG 迭代机制测试。

验证批判评估、查询精炼、迭代循环的正确性。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.generation.self_rag import SelfRAG, CritiqueResult


class TestCritiqueParsing:

    def test_parse_valid_json(self):
        """解析标准 JSON 批判结果。"""
        rag = SelfRAG()
        response = '''
        ```json
        {
            "faithfulness": 8,
            "completeness": 7,
            "accuracy": 9,
            "overall": 8,
            "pass": true,
            "issues": ["缺少细节"],
            "improvement_suggestions": ["补充例子"]
        }
        ```
        '''
        result = rag._parse_critique(response)
        assert result.overall == 8.0
        assert result.passed is True
        assert "缺少细节" in result.issues

    def test_parse_low_score_pass_false(self):
        """低分应判定为不通过。"""
        rag = SelfRAG(pass_threshold=7.0)
        response = '{"faithfulness": 5, "completeness": 4, "accuracy": 6, "overall": 5, "pass": false, "issues": ["编造信息"], "improvement_suggestions": []}'
        result = rag._parse_critique(response)
        assert result.overall == 5.0
        assert result.passed is False

    def test_parse_invalid_json(self):
        """无效 JSON 应返回默认通过。"""
        rag = SelfRAG()
        result = rag._parse_critique("这不是 JSON")
        assert result.passed is True  # 降级默认通过


class TestSelfRAGLoop:

    def test_immediate_pass(self):
        """批判一次通过时应直接返回。"""
        mock_llm = AsyncMock()
        # 生成回答
        mock_llm.ainvoke.side_effect = [
            "初始回答内容",  # _generate_answer
            '{"faithfulness": 9, "completeness": 9, "accuracy": 9, "overall": 9, "pass": true, "issues": [], "improvement_suggestions": []}',  # _critique
        ]

        rag = SelfRAG(llm_service=mock_llm, max_iterations=3)
        result = asyncio.run(rag.run("测试问题", initial_context="测试上下文"))

        assert result["total_iterations"] == 1
        assert result["final_answer"] == "初始回答内容"
        assert result["final_critique"].passed is True

    def test_iterates_when_not_passing(self):
        """批判不通过时应迭代。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            "初始回答",  # _generate_answer (iteration 1)
            '{"faithfulness": 4, "completeness": 3, "accuracy": 5, "overall": 4, "pass": false, "issues": ["不够详细"], "improvement_suggestions": ["补充更多细节"]}',  # _critique (iteration 1)
            "改进的查询",  # _refine_query
            "改进后的回答",  # _refine_answer
            '{"faithfulness": 8, "completeness": 8, "accuracy": 8, "overall": 8, "pass": true, "issues": [], "improvement_suggestions": []}',  # _critique (iteration 2)
        ]

        rag = SelfRAG(llm_service=mock_llm, max_iterations=3)
        result = asyncio.run(rag.run("测试问题", initial_context="测试上下文"))

        assert result["total_iterations"] == 2
        assert result["final_answer"] == "改进后的回答"
        assert result["final_critique"].passed is True

    def test_stops_at_max_iterations(self):
        """达到最大迭代次数时应停止。"""
        mock_llm = AsyncMock()
        # 每次都不通过
        low_critique = '{"faithfulness": 3, "completeness": 3, "accuracy": 3, "overall": 3, "pass": false, "issues": ["差"], "improvement_suggestions": ["改进"]}'
        mock_llm.ainvoke.side_effect = [
            "回答1", low_critique, "查询2", "回答2",
            low_critique, "查询3", "回答3",
            low_critique,
        ]

        rag = SelfRAG(llm_service=mock_llm, max_iterations=3)
        result = asyncio.run(rag.run("测试问题", initial_context="上下文"))

        assert result["total_iterations"] == 3
        # 最后一次迭代的批判应该不通过
        assert result["final_critique"].passed is False

    def test_no_llm_returns_empty(self):
        """无 LLM 时应返回空结果。"""
        rag = SelfRAG(llm_service=None)
        result = asyncio.run(rag.run("测试问题"))
        assert result["final_answer"] == ""
        assert result["total_iterations"] == 0


class TestSelfRAGIntegration:

    def test_full_flow_mock(self):
        """完整流程测试（mock LLM + mock search）。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            "ReLU 的优势包括...",  # _generate_answer
            '{"faithfulness": 8, "completeness": 6, "accuracy": 7, "overall": 7, "pass": true, "issues": [], "improvement_suggestions": []}',
        ]

        rag = SelfRAG(llm_service=mock_llm, max_iterations=2)
        result = asyncio.run(rag.run(
            question="如果用ReLU替代Sigmoid会怎样？",
            initial_context="ReLU在正区间梯度恒为1...",
        ))

        assert result["total_iterations"] >= 1
        assert len(result["final_answer"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
