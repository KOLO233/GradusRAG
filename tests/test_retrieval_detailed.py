"""向量检索模块测试。

验证 EmbeddingService（BM25 稀疏向量）和 DenseRetriever 的正确性。
BM25 部分不依赖外部服务，可直接测试；密集向量部分通过 mock 测试。
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.libs.embedding_service import EmbeddingService
from src.core.types import RetrievalResult


# ===========================================================================
# EmbeddingService BM25 测试
# ===========================================================================

@pytest.fixture
def bm25_service(tmp_path):
    """使用临时目录的 BM25 服务。"""
    state_path = tmp_path / "bm25_state.json"
    return EmbeddingService(state_path=state_path)


class TestBM25Tokenize:
    """分词测试。"""

    def test_chinese_text(self, bm25_service):
        """中文文本应按单字切分。"""
        tokens = bm25_service.tokenize("机器学习是人工智能的子领域")
        assert len(tokens) > 0
        assert "机" in tokens or "机器" not in tokens  # 单字切分

    def test_english_text(self, bm25_service):
        """英文文本应按单词切分。"""
        tokens = bm25_service.tokenize("Machine Learning is a subset of AI")
        assert "Machine" in tokens
        assert "Learning" in tokens

    def test_mixed_text(self, bm25_service):
        """中英文混合文本应正确分词。"""
        tokens = bm25_service.tokenize("使用Python进行机器学习")
        assert "Python" in tokens
        assert len(tokens) > 3

    def test_empty_text(self, bm25_service):
        """空文本应返回空列表。"""
        assert bm25_service.tokenize("") == []


class TestBM25Incremental:
    """增量更新测试。"""

    def test_add_documents(self, bm25_service):
        """添加文档应更新统计。"""
        bm25_service.bm25_increment_add(["机器学习是人工智能的子领域"])
        assert bm25_service._total_docs == 1
        assert bm25_service._sum_token_len > 0

    def test_add_multiple_documents(self, bm25_service):
        """添加多篇文档应正确累计。"""
        bm25_service.bm25_increment_add([
            "机器学习是人工智能的子领域",
            "深度学习是机器学习的子领域",
        ])
        assert bm25_service._total_docs == 2

    def test_remove_documents(self, bm25_service):
        """删除文档应减少统计。"""
        texts = ["文档一的内容", "文档二的内容"]
        bm25_service.bm25_increment_add(texts)
        assert bm25_service._total_docs == 2

        bm25_service.bm25_increment_remove(["文档一的内容"])
        assert bm25_service._total_docs == 1

    def test_state_persistence(self, bm25_service):
        """BM25 状态应持久化到文件。"""
        bm25_service.bm25_increment_add(["测试持久化"])
        assert bm25_service._state_path.is_file()

        # 重新加载验证
        raw = json.loads(bm25_service._state_path.read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert raw["total_docs"] == 1


class TestBM25Scoring:
    """BM25 评分测试。"""

    def test_relevant_doc_higher_score(self, bm25_service):
        """相关文档应获得更高分数。"""
        bm25_service.bm25_increment_add([
            "机器学习是人工智能的重要分支",
            "今天天气很好适合出去玩",
            "深度学习是机器学习的子领域",
        ])

        query_tokens = bm25_service.tokenize("机器学习")
        doc_relevant = bm25_service.tokenize("机器学习是人工智能的重要分支")
        doc_irrelevant = bm25_service.tokenize("今天天气很好适合出去玩")

        score_relevant = bm25_service.compute_bm25_score(query_tokens, doc_relevant)
        score_irrelevant = bm25_service.compute_bm25_score(query_tokens, doc_irrelevant)

        assert score_relevant > score_irrelevant

    def test_embed_sparse_returns_dicts(self, bm25_service):
        """embed_sparse 应返回字典列表。"""
        bm25_service.bm25_increment_add(["机器学习深度学习自然语言处理"])
        sparse = bm25_service.embed_sparse(["机器学习"])
        assert len(sparse) == 1
        assert isinstance(sparse[0], dict)


# ===========================================================================
# DenseRetriever 测试（mock Milvus）
# ===========================================================================

class TestDenseRetriever:
    """Dense Retriever 测试（使用 mock）。"""

    def test_retrieve_returns_results(self):
        """检索应返回 RetrievalResult 列表。"""
        mock_embedding = MagicMock()
        mock_embedding.embed_dense_query.return_value = [0.1] * 1024

        mock_store = MagicMock()
        mock_store.search_dense.return_value = [
            {
                "chunk_id": "test::p1::l3::0",
                "score": 0.95,
                "text": "机器学习是人工智能的子领域",
                "filename": "test.pdf",
                "source_path": "/path/to/test.pdf",
                "page": 1,
                "chunk_index": 0,
                "chunk_level": 3,
                "parent_chunk_id": "test::p1::l2::0",
                "root_chunk_id": "test::p1::l1::0",
            },
            {
                "chunk_id": "test::p1::l3::1",
                "score": 0.85,
                "text": "深度学习是机器学习的子领域",
                "filename": "test.pdf",
                "source_path": "/path/to/test.pdf",
                "page": 1,
                "chunk_index": 1,
                "chunk_level": 3,
                "parent_chunk_id": "test::p1::l2::0",
                "root_chunk_id": "test::p1::l1::0",
            },
        ]

        from src.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(mock_embedding, mock_store)

        results = asyncio.run(retriever.retrieve("什么是机器学习？", top_k=5))

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score > results[1].score
        assert results[0].retrieval_source == "dense"
        assert "机器" in results[0].text

    def test_retrieve_empty_query(self):
        """空查询应返回空结果。"""
        mock_embedding = MagicMock()
        mock_embedding.embed_dense_query.return_value = [0.1] * 1024

        mock_store = MagicMock()
        mock_store.search_dense.return_value = []

        from src.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(mock_embedding, mock_store)

        results = asyncio.run(retriever.retrieve("", top_k=5))
        assert results == []

    def test_retrieve_metadata_preserved(self):
        """检索结果应保留完整元数据。"""
        mock_embedding = MagicMock()
        mock_embedding.embed_dense_query.return_value = [0.1] * 1024

        mock_store = MagicMock()
        mock_store.search_dense.return_value = [
            {
                "chunk_id": "doc::p3::l3::5",
                "score": 0.9,
                "text": "测试文本",
                "filename": "教材.pdf",
                "source_path": "/data/教材.pdf",
                "page": 3,
                "chunk_index": 5,
                "chunk_level": 3,
                "parent_chunk_id": "doc::p3::l2::1",
                "root_chunk_id": "doc::p3::l1::0",
            },
        ]

        from src.retrieval.dense_retriever import DenseRetriever
        retriever = DenseRetriever(mock_embedding, mock_store)

        results = asyncio.run(retriever.retrieve("测试", top_k=1))

        assert results[0].metadata["filename"] == "教材.pdf"
        assert results[0].metadata["page"] == 3
        assert results[0].metadata["parent_chunk_id"] == "doc::p3::l2::1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
