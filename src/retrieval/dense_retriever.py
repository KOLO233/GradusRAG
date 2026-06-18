"""Dense Retriever — 密集向量检索器。

将用户查询向量化后在 Milvus 中进行语义检索。
"""

from __future__ import annotations

import logging
from typing import List

from src.core.types import RetrievalResult
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """密集向量检索器。

    流程：查询文本 → EmbeddingService 向量化 → MilvusStore 语义检索

    Example:
        >>> retriever = DenseRetriever(embedding_service, milvus_store)
        >>> results = await retriever.retrieve("什么是机器学习？", top_k=5)
        >>> for r in results:
        ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_store: MilvusStore,
    ):
        self._embedding = embedding_service
        self._store = milvus_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """执行密集向量检索。

        Args:
            query: 查询文本
            top_k: 返回的最大结果数

        Returns:
            RetrievalResult 列表，按相关性降序排列
        """
        # Step 1: 查询向量化
        query_vector = self._embedding.embed_dense_query(query)

        # Step 2: Milvus 语义检索
        raw_results = self._store.search_dense(
            query_vector=query_vector,
            top_k=top_k,
        )

        # Step 3: 转换为标准 RetrievalResult
        results = []
        for item in raw_results:
            result = RetrievalResult(
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
                retrieval_source="dense",
            )
            results.append(result)

        logger.debug(f"Dense retrieval: query='{query[:50]}' → {len(results)} results")
        return results
