"""Sparse Retriever — BM25 稀疏检索器。

基于 BM25 算法的关键词检索，解决专有名词精确匹配问题。
与 Dense Retriever 互补：Dense 理解语义，Sparse 精确匹配关键词。

优先使用倒排索引（O(K) 复杂度），回退到全量扫描。
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Optional

from src.core.types import RetrievalResult
from src.libs.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25 稀疏检索器。

    优先使用倒排索引（O(K) 复杂度），回退到全量扫描。

    Example:
        >>> retriever = SparseRetriever(embedding_service, milvus_store)
        >>> results = await retriever.retrieve(keywords=["机器", "学习"], top_k=5)
    """

    def __init__(self, embedding_service: EmbeddingService, milvus_store=None):
        self._embedding = embedding_service
        self._store = milvus_store

    async def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        query_text: str = "",
    ) -> List[RetrievalResult]:
        """执行 BM25 稀疏检索。

        优先使用倒排索引，回退到全量扫描。

        Args:
            keywords: 查询关键词列表
            top_k: 返回的最大结果数
            query_text: 原始查询文本（用于分词，如果 keywords 为空）

        Returns:
            RetrievalResult 列表，按 BM25 分数降序排列
        """
        # 如果没有关键词，从查询文本中提取
        if not keywords and query_text:
            keywords = self._embedding.tokenize(query_text)

        if not keywords:
            return []

        query_text_combined = " ".join(keywords)

        # 优先使用倒排索引（O(K) 复杂度）
        if self._embedding._inverted_index:
            return self._retrieve_from_index(query_text_combined, top_k)

        # 回退到全量扫描（兼容旧行为）
        return await self._retrieve_full_scan(keywords, top_k)

    def _retrieve_from_index(
        self, query_text: str, top_k: int
    ) -> List[RetrievalResult]:
        """使用倒排索引检索（快速路径）。"""
        raw_results = self._embedding.bm25_search(query_text, top_k=top_k)

        results = []
        for item in raw_results:
            results.append(RetrievalResult(
                chunk_id=item.get("chunk_id", ""),
                score=item.get("score", 0.0),
                text=item.get("text", ""),
                metadata={
                    "source_path": item.get("source_path", ""),
                    "filename": item.get("filename", ""),
                    "page": item.get("page", 0),
                    "chunk_index": item.get("chunk_index", 0),
                    "chunk_level": item.get("chunk_level", 3),
                    "parent_chunk_id": item.get("parent_chunk_id", ""),
                    "root_chunk_id": item.get("root_chunk_id", ""),
                },
                retrieval_source="sparse",
            ))

        logger.debug(
            f"Sparse retrieval (inverted index): '{query_text[:30]}' → {len(results)} results"
        )
        return results

    async def _retrieve_full_scan(
        self, keywords: List[str], top_k: int
    ) -> List[RetrievalResult]:
        """全量扫描检索（兼容路径，当倒排索引不可用时）。"""
        if self._store is None:
            logger.warning("No Milvus store configured for sparse retrieval")
            return []

        try:
            all_docs = self._store._get_client().query(
                self._store._collection,
                output_fields=["chunk_id", "text", "filename", "source_path",
                               "page", "chunk_index", "chunk_level",
                               "parent_chunk_id", "root_chunk_id"],
                limit=16384,
            )
        except Exception as e:
            logger.error(f"Failed to query documents for BM25: {e}")
            return []

        if not all_docs:
            return []

        query_tokens = self._embedding.tokenize(" ".join(keywords))
        scored_docs = []

        for doc in all_docs:
            text = doc.get("text", "")
            if not text:
                continue
            doc_tokens = self._embedding.tokenize(text)
            score = self._embedding.compute_bm25_score(query_tokens, doc_tokens)
            if score > 0:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:top_k]:
            results.append(RetrievalResult(
                chunk_id=doc.get("chunk_id", ""),
                score=score,
                text=doc.get("text", ""),
                metadata={
                    "source_path": doc.get("source_path", ""),
                    "filename": doc.get("filename", ""),
                    "page": doc.get("page", 0),
                    "chunk_index": doc.get("chunk_index", 0),
                    "chunk_level": doc.get("chunk_level", 3),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                },
                retrieval_source="sparse",
            ))

        logger.debug(
            f"Sparse retrieval (full scan): keywords={keywords[:5]}, "
            f"docs_scanned={len(all_docs)}, results={len(results)}"
        )
        return results
