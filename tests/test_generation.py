"""生成模块测试。

验证 DocumentGrader、QueryRewriter、ResponseGenerator 的正确性。
使用 mock LLM，不依赖外部 API。
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.generation.document_grader import DocumentGrader
from src.generation.query_rewriter import QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.generation.pipeline import RAGPipeline
from src.core.types import RAGResponse, QueryClassification, ProcessedQuery, RetrievalResult


# ===========================================================================
# DocumentGrader 测试
# ===========================================================================

class TestDocumentGrader:

    def test_no_llm_defaults_yes(self):
        """无 LLM 时默认返回 yes。"""
        grader = DocumentGrader(llm_service=None)
        result = asyncio.run(grader.grade("什么是机器学习？", "机器学习是..."))
        assert result == "yes"

    def test_empty_context_returns_no(self):
        """空上下文应返回 no。"""
        mock_llm = MagicMock()
        grader = DocumentGrader(llm_service=mock_llm)
        result = asyncio.run(grader.grade("什么是机器学习？", ""))
        assert result == "no"

    def test_llm_returns_yes(self):
        """LLM 返回 yes 时应透传。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "yes"
        grader = DocumentGrader(llm_service=mock_llm)
        result = asyncio.run(grader.grade("问题", "相关文档"))
        assert result == "yes"

    def test_llm_returns_no(self):
        """LLM 返回 no 时应透传。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "no"
        grader = DocumentGrader(llm_service=mock_llm)
        result = asyncio.run(grader.grade("问题", "不相关文档"))
        assert result == "no"


# ===========================================================================
# QueryRewriter 测试
# ===========================================================================

class TestQueryRewriter:

    def test_no_llm_returns_original(self):
        """无 LLM 时返回原始查询。"""
        rewriter = QueryRewriter(llm_service=None)
        result = asyncio.run(rewriter.rewrite("什么是机器学习？"))
        assert result["rewritten_query"] == "什么是机器学习？"
        assert result["strategy"] == "none"

    def test_step_back_strategy(self):
        """Step-Back 策略应生成退步问题。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "人工智能的发展历程是什么？"
        rewriter = QueryRewriter(llm_service=mock_llm)

        result = asyncio.run(rewriter.rewrite("机器学习的历史？", strategy="step_back"))
        assert result["strategy"] == "step_back"
        assert result["step_back_question"] == "人工智能的发展历程是什么？"
        assert result["rewritten_query"] == "人工智能的发展历程是什么？"

    def test_hyde_strategy(self):
        """HyDE 策略应生成假设性文档。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "机器学习是人工智能的一个分支..."
        rewriter = QueryRewriter(llm_service=mock_llm)

        result = asyncio.run(rewriter.rewrite("什么是机器学习？", strategy="hyde"))
        assert result["strategy"] == "hyde"
        assert "机器学习" in result["hypothetical_doc"]


# ===========================================================================
# ResponseGenerator 测试
# ===========================================================================

class TestResponseGenerator:

    def test_no_llm_fallback(self):
        """无 LLM 时应返回降级回答。"""
        gen = ResponseGenerator(llm_service=None)
        answer = asyncio.run(gen.generate(
            question="什么是机器学习？",
            context="机器学习是人工智能的子领域。",
            level="L1",
        ))
        assert "参考资料" in answer
        assert "LLM 未配置" in answer

    def test_no_context_fallback(self):
        """无上下文时应返回提示。"""
        gen = ResponseGenerator(llm_service=None)
        answer = asyncio.run(gen.generate(
            question="什么是机器学习？",
            context="",
            level="L1",
        ))
        assert "未找到" in answer

    def test_different_levels_use_different_prompts(self):
        """不同级别应使用不同的 Prompt。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "测试回答"
        gen = ResponseGenerator(llm_service=mock_llm)

        for level in ["L1", "L2", "L3", "L4"]:
            answer = asyncio.run(gen.generate(
                question="测试问题",
                context="测试上下文",
                level=level,
            ))
            assert answer == "测试回答"

    def test_llm_called_with_context(self):
        """LLM 应收到包含上下文的 prompt。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "答案"
        gen = ResponseGenerator(llm_service=mock_llm)

        asyncio.run(gen.generate(
            question="什么是X？",
            context="X是Y的一种。",
            level="L1",
        ))

        # 验证 LLM 被调用
        mock_llm.ainvoke.assert_called_once()
        call_arg = mock_llm.ainvoke.call_args[0][0]
        assert "X是Y的一种" in call_arg
        assert "什么是X？" in call_arg


# ===========================================================================
# Pipeline 集成测试（无 LLM）
# ===========================================================================

class TestRAGPipelineIntegration:

    def test_pipeline_run_without_llm(self):
        """Pipeline 在无 LLM 时应能运行并返回降级结果。"""
        from src.core.settings import IngestionSettings, QueryClassifierSettings, \
            GraphSettings, GenerationSettings, EvaluationSettings, ObservabilitySettings, \
            RetrievalSettings, RerankSettings, LLMSettings, EmbeddingSettings, VectorStoreSettings

        # 构造最小 settings
        settings = MagicMock()
        settings.llm = LLMSettings(provider="openai", model="gpt-4o", temperature=0.0, max_tokens=4096)
        settings.embedding = EmbeddingSettings(provider="local", model="BAAI/bge-m3", dimensions=1024)
        settings.retrieval = RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60)
        settings.rerank = RerankSettings(enabled=False, provider="none", model="", top_k=5)
        settings.ingestion = IngestionSettings(chunk_size_l1=2500, chunk_size_l2=1024, chunk_size_l3=512, chunk_overlap=100, splitter="recursive")
        settings.query_classifier = QueryClassifierSettings(mode="rule", llm_threshold=0.7)
        settings.graph = GraphSettings(enabled=False, storage="networkx", entity_types=[], max_hops=3)
        settings.generation = GenerationSettings(l1_strategy="direct", l2_strategy="structured_cot", l3_strategy="expert_cot", l4_strategy="self_rag", max_self_rag_iterations=3)
        settings.evaluation = EvaluationSettings(enabled=False, framework="custom", metrics=[])
        settings.observability = ObservabilitySettings(log_level="INFO", trace_enabled=False, trace_file="")

        pipeline = RAGPipeline(settings=settings)

        result = asyncio.run(pipeline.run("什么是机器学习？"))

        assert isinstance(result, RAGResponse)
        assert result.query_level in ("L1", "L2", "L3", "L4")
        assert result.answer  # 应有回答（降级模式）


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
