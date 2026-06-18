"""重排序和上下文优化测试。

验证 Reranker（LLM 重排序）和 ContextBuilder（压缩 + Auto-merging）。
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.types import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.context_builder import ContextBuilder


def _make_result(chunk_id: str, text: str, score: float = 0.5, parent_id: str = "") -> RetrievalResult:
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


# ===========================================================================
# Reranker 测试
# ===========================================================================

class TestReranker:

    def test_rerank_empty(self):
        """空结果应返回空列表。"""
        reranker = Reranker()
        results = asyncio.run(reranker.rerank("query", [], top_k=5))
        assert results == []

    def test_llm_rerank_reorders(self):
        """LLM 重排序应改变顺序。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "[3, 9, 5]"

        reranker = Reranker(llm_service=mock_llm)
        results = [
            _make_result("c1", "不太相关的内容", score=0.9),
            _make_result("c2", "非常相关的内容", score=0.5),
            _make_result("c3", "一般相关的内容", score=0.7),
        ]

        reranked = asyncio.run(reranker.rerank("测试查询", results, top_k=3))

        assert len(reranked) == 3
        # c2 应排第一（LLM 给了 9 分）
        assert reranked[0].chunk_id == "c2"
        assert reranked[0].metadata.get("rerank_score") == 0.9

    def test_llm_rerank_parse_error_fallback(self):
        """LLM 返回异常时应降级返回原始顺序。"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = "这不是一个有效的JSON"

        reranker = Reranker(llm_service=mock_llm)
        results = [
            _make_result("c1", "内容1", score=0.9),
            _make_result("c2", "内容2", score=0.5),
        ]

        reranked = asyncio.run(reranker.rerank("查询", results, top_k=2))
        # 降级：均匀分数，但应仍有结果
        assert len(reranked) == 2

    def test_rerank_no_llm_fallback(self):
        """无 LLM 时应直接返回原始顺序。"""
        reranker = Reranker(llm_service=None)
        results = [
            _make_result("c1", "内容1", score=0.9),
            _make_result("c2", "内容2", score=0.5),
        ]

        reranked = asyncio.run(reranker.rerank("查询", results, top_k=2))
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c1"  # 原始顺序

    def test_parse_scores(self):
        """分数解析测试。"""
        scores = Reranker._parse_scores("[8, 5, 9, 3, 7]", 5)
        assert scores == [8.0, 5.0, 9.0, 3.0, 7.0]

    def test_parse_scores_fallback(self):
        """无法解析时应返回均匀分数。"""
        scores = Reranker._parse_scores("无效输出", 3)
        assert scores == [5.0, 5.0, 5.0]


# ===========================================================================
# ContextBuilder 压缩测试
# ===========================================================================

class TestContextCompression:

    def test_deduplicate(self):
        """去重应保留分数最高的。"""
        results = [
            _make_result("c1", "内容A", score=0.5),
            _make_result("c1", "内容A", score=0.9),  # 重复，分数更高
            _make_result("c2", "内容B", score=0.7),
        ]
        deduped = ContextBuilder._deduplicate(results)
        assert len(deduped) == 2
        # c1 应保留分数 0.9 的那个
        c1_result = [r for r in deduped if r.chunk_id == "c1"][0]
        assert c1_result.score == 0.9

    def test_compress_removes_similar(self):
        """高度相似的文本应被去除。"""
        builder = ContextBuilder()
        results = [
            _make_result("c1", "机器学习是人工智能的子领域，通过数据自动学习规律", score=0.9),
            _make_result("c2", "机器学习是人工智能的子领域，通过数据自动学习模式", score=0.8),  # 高度相似
            _make_result("c3", "深度学习使用多层神经网络进行特征提取", score=0.7),  # 不同内容
        ]

        compressed = builder._compress(results)
        # c2 应被去除（与 c1 高度相似）
        ids = [r.chunk_id for r in compressed]
        assert "c1" in ids
        assert "c3" in ids

    def test_compress_short_text_no_change(self):
        """短文本不需要压缩。"""
        builder = ContextBuilder()
        results = [
            _make_result("c1", "完全不同的内容A", score=0.9),
            _make_result("c2", "另一个完全不同的话题B", score=0.8),
        ]
        compressed = builder._compress(results)
        assert len(compressed) == 2

    def test_jaccard_similarity_identical(self):
        """相同文本相似度应为 1.0。"""
        sim = ContextBuilder._jaccard_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_jaccard_similarity_different(self):
        """完全不同的文本相似度应接近 0。"""
        sim = ContextBuilder._jaccard_similarity("aaa bbb", "xxx yyy zzz")
        assert sim < 0.3

    def test_build_with_compression(self):
        """带压缩的上下文构建。"""
        builder = ContextBuilder()
        results = [
            _make_result("c1", "机器学习是人工智能的子领域，它通过数据来学习规律", score=0.9),
            _make_result("c2", "机器学习是人工智能的子领域，它通过数据来学习模式", score=0.8),
            _make_result("c3", "深度学习是机器学习的一个分支，使用多层神经网络", score=0.7),
        ]

        context = builder.build(results, compress=True)
        # 压缩后应该减少重复内容
        assert "机器学习" in context
        assert "深度学习" in context

    def test_auto_merge_with_parent_store(self):
        """Auto-merging 应正确合并子块到父块。"""
        parent_store = {
            "parent_l2": {
                "text": "这是父块的完整内容，包含两个子块的全部信息。",
                "level": 2,
                "parent_id": "",
                "root_id": "",
                "filename": "test.pdf",
                "page": 1,
            },
        }
        builder = ContextBuilder(
            parent_store=parent_store,
            auto_merge_enabled=True,
            auto_merge_threshold=2,
        )

        results = [
            _make_result("c1", "子块1", score=0.9, parent_id="parent_l2"),
            _make_result("c2", "子块2", score=0.8, parent_id="parent_l2"),
        ]

        context = builder.build(results)
        # 应包含父块内容
        assert "父块的完整内容" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
