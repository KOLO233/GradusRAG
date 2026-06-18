"""混合检索模块测试。

验证 SparseRetriever（BM25）和 ContextBuilder（Auto-merging）的正确性。
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from src.core.types import RetrievalResult
from src.retrieval.context_builder import ContextBuilder
from src.libs.embedding_service import EmbeddingService


# ===========================================================================
# BM25 Sparse Retriever 测试（通过 EmbeddingService 间接测试）
# ===========================================================================

class TestBM25Integration:
    """BM25 集成测试。"""

    def test_bm25_scores_ranking(self, tmp_path):
        """BM25 应给相关文档更高分数。"""
        service = EmbeddingService(state_path=tmp_path / "bm25.json")

        # 添加文档
        service.bm25_increment_add([
            "机器学习是人工智能的子领域",
            "深度学习是机器学习的子领域",
            "今天天气很好适合出去玩",
            "自然语言处理是人工智能的重要分支",
            "卷积神经网络用于计算机视觉任务",
        ])

        # 查询 "机器学习"
        query_tokens = service.tokenize("机器学习")

        docs = [
            "机器学习是人工智能的子领域",
            "深度学习是机器学习的子领域",
            "今天天气很好适合出去玩",
        ]

        scores = []
        doc_tokens_list = [service.tokenize(d) for d in docs]
        for dt in doc_tokens_list:
            scores.append(service.compute_bm25_score(query_tokens, dt))

        # 最相关的文档应得分最高
        assert scores[0] > scores[2]  # "机器学习..." > "今天天气..."
        assert scores[1] > scores[2]  # "深度学习是机器学习..." > "今天天气..."

    def test_bm25_increment_update(self, tmp_path):
        """增量更新后 BM25 排序应反映新文档。"""
        service = EmbeddingService(state_path=tmp_path / "bm25.json")

        service.bm25_increment_add(["文档A的内容"])
        service.bm25_increment_add(["文档B的内容"])

        assert service._total_docs == 2

        service.bm25_increment_remove(["文档A的内容"])
        assert service._total_docs == 1


# ===========================================================================
# ContextBuilder 测试
# ===========================================================================

def _make_result(chunk_id: str, text: str, parent_id: str = "", score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={
            "source_path": "test.pdf",
            "filename": "test.pdf",
            "page": 1,
            "parent_chunk_id": parent_id,
        },
    )


class TestContextBuilder:

    def test_build_basic(self):
        """基本上下文构建。"""
        builder = ContextBuilder()
        results = [
            _make_result("c1", "第一段内容"),
            _make_result("c2", "第二段内容"),
        ]
        context = builder.build(results)
        assert "[1]" in context
        assert "[2]" in context
        assert "第一段内容" in context

    def test_build_empty(self):
        """空结果应返回空字符串。"""
        builder = ContextBuilder()
        assert builder.build([]) == ""

    def test_build_truncation(self):
        """超长上下文应被截断。"""
        builder = ContextBuilder()
        long_text = "很长的内容" * 1000
        results = [_make_result("c1", long_text)]
        context = builder.build(results, max_length=100)
        assert len(context) <= 150  # 截断 + 省略号

    def test_auto_merge_threshold(self):
        """当同一父块的子块数 >= threshold 时应合并。"""
        parent_store = {
            "parent_l2_0": {
                "text": "这是L2父块的完整内容，包含两个子块。",
                "level": 2,
                "parent_id": "parent_l1_0",
                "root_id": "parent_l1_0",
                "filename": "test.pdf",
                "page": 1,
            },
        }
        builder = ContextBuilder(
            parent_store=parent_store,
            auto_merge_enabled=True,
            auto_merge_threshold=2,
        )

        # 两个子块属于同一父块
        results = [
            _make_result("c1", "子块1内容", parent_id="parent_l2_0"),
            _make_result("c2", "子块2内容", parent_id="parent_l2_0"),
        ]

        merged = builder._auto_merge(results)

        # 应该被合并为一个父块
        assert len(merged) == 1
        assert merged[0].chunk_id == "parent_l2_0"
        assert "L2父块" in merged[0].text

    def test_auto_merge_below_threshold(self):
        """子块数不足 threshold 时不合并。"""
        parent_store = {
            "parent_l2_0": {"text": "父块", "level": 2, "parent_id": "", "root_id": "", "filename": "", "page": 1},
        }
        builder = ContextBuilder(
            parent_store=parent_store,
            auto_merge_enabled=True,
            auto_merge_threshold=3,  # 需要 3 个才合并
        )

        results = [
            _make_result("c1", "子块1", parent_id="parent_l2_0"),
            _make_result("c2", "子块2", parent_id="parent_l2_0"),
        ]

        merged = builder._auto_merge(results)
        # 只有 2 个子块，不够 3 个，不合并
        assert len(merged) == 2

    def test_auto_merge_disabled(self):
        """关闭 auto-merging 时不应合并。"""
        parent_store = {
            "parent_l2_0": {"text": "父块", "level": 2, "parent_id": "", "root_id": "", "filename": "", "page": 1},
        }
        builder = ContextBuilder(
            parent_store=parent_store,
            auto_merge_enabled=False,
        )

        results = [
            _make_result("c1", "子块1", parent_id="parent_l2_0"),
            _make_result("c2", "子块2", parent_id="parent_l2_0"),
        ]

        # 通过 build 调用（会检查 auto_merge_enabled 开关）
        context = builder.build(results)
        # 两个子块都应保留（不合并），所以上下文中应包含两个子块的内容
        assert "子块1" in context
        assert "子块2" in context


# ===========================================================================
# 混合检索集成测试（mock）
# ===========================================================================

class TestHybridSearchIntegration:

    def test_hybrid_search_with_both_retrievers(self):
        """混合检索应同时调用 Dense 和 Sparse。"""
        from src.retrieval.hybrid_search import HybridSearch
        from src.core.types import ProcessedQuery
        from src.core.settings import RetrievalSettings, RerankSettings, GraphSettings
        from unittest.mock import AsyncMock

        mock_settings = MagicMock()
        mock_settings.retrieval = RetrievalSettings(
            dense_top_k=10, sparse_top_k=10, fusion_top_k=5, rrf_k=60
        )
        mock_settings.rerank = RerankSettings(enabled=False, provider="none", model="", top_k=5)
        mock_settings.graph = GraphSettings(enabled=False, storage="", entity_types=[], max_hops=3)

        # Mock Dense
        mock_dense = AsyncMock()
        mock_dense.retrieve.return_value = [
            _make_result("d1", "dense结果1", score=0.9),
            _make_result("d2", "dense结果2", score=0.8),
        ]

        # Mock Sparse
        mock_sparse = AsyncMock()
        mock_sparse.retrieve.return_value = [
            _make_result("s1", "sparse结果1", score=3.0),
            _make_result("d2", "dense结果2", score=2.5),  # 与 dense 重叠
        ]

        hybrid = HybridSearch(
            settings=mock_settings,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        query = ProcessedQuery(
            original_query="测试查询",
            classified_level="L1",
            keywords=["测试", "查询"],
        )

        result = asyncio.run(hybrid.search(query, top_k=5))

        # 应该有结果
        assert len(result.results) > 0
        # Dense 和 Sparse 都应被调用
        mock_dense.retrieve.assert_called_once()
        mock_sparse.retrieve.assert_called_once()
        # 融合方法应为 RRF
        assert result.fusion_method == "rrf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
