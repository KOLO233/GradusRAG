"""Milvus 向量存储。

管理 Milvus 集合的创建、写入、查询和删除。
支持密集向量（HNSW 索引）和稀疏向量（SPARSE_INVERTED_INDEX）混合检索。

参考 SuperMew 的 MilvusManager 实现。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUERY_MAX_LIMIT = 16384


class MilvusStore:
    """Milvus 向量存储管理器。

    Example:
        >>> store = MilvusStore(host="localhost", port=19530, collection="gradusrag")
        >>> store.init_collection(dense_dim=1024)
        >>> store.insert(data_list)
        >>> results = store.search_dense(query_vector, top_k=10)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection: str = "gradusrag",
    ):
        self._host = host
        self._port = port
        self._collection = collection
        self._client = None

    def _get_client(self):
        """懒加载 Milvus 客户端。"""
        if self._client is None:
            try:
                from pymilvus import MilvusClient
                uri = f"http://{self._host}:{self._port}"
                self._client = MilvusClient(uri=uri)
                logger.info(f"Connected to Milvus at {uri}")
            except ImportError:
                raise ImportError(
                    "pymilvus is required. Install with: pip install pymilvus"
                )
        return self._client

    def init_collection(self, dense_dim: int = 1024, force_recreate: bool = False) -> None:
        """初始化 Milvus 集合。

        创建集合并建立索引。如果集合已存在且 force_recreate=False，则跳过。

        Args:
            dense_dim: 密集向量维度
            force_recreate: 是否强制重建集合
        """
        from pymilvus import DataType

        client = self._get_client()

        if force_recreate and client.has_collection(self._collection):
            client.drop_collection(self._collection)
            logger.info(f"Dropped existing collection: {self._collection}")

        if not client.has_collection(self._collection):
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("text", DataType.VARCHAR, max_length=4000)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("source_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("page", DataType.INT64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("chunk_level", DataType.INT64)
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 256},
            )

            client.create_collection(
                collection_name=self._collection,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"Created collection: {self._collection} (dim={dense_dim})")
        else:
            logger.info(f"Collection already exists: {self._collection}")

    def insert(self, data: List[Dict[str, Any]]) -> None:
        """批量写入数据到 Milvus。

        Args:
            data: 数据列表，每项包含 dense_embedding, text, chunk_id 等字段
        """
        if not data:
            return
        client = self._get_client()
        result = client.insert(self._collection, data)
        logger.info(f"Inserted {len(data)} records into {self._collection}")
        return result

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """密集向量检索。

        Args:
            query_vector: 查询向量
            top_k: 返回的最大结果数
            output_fields: 返回的字段列表

        Returns:
            检索结果列表
        """
        client = self._get_client()

        if output_fields is None:
            output_fields = [
                "text", "chunk_id", "filename", "source_path",
                "page", "chunk_index", "chunk_level",
                "parent_chunk_id", "root_chunk_id",
            ]

        results = client.search(
            collection_name=self._collection,
            data=[query_vector],
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "IP", "params": {"ef": 128}},
        )

        # 解析 Milvus 返回格式
        parsed = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                entity["score"] = hit.get("distance", 0.0)
                parsed.append(entity)

        return parsed

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除所有相关记录。

        Args:
            filename: 文件名

        Returns:
            删除的记录数
        """
        client = self._get_client()
        expr = f'filename == "{filename}"'
        result = client.delete(self._collection, filter=expr)
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info(f"Deleted {count} records for filename={filename}")
        return count

    def count(self) -> int:
        """获取集合中的记录总数。"""
        client = self._get_client()
        stats = client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息。"""
        client = self._get_client()
        if not client.has_collection(self._collection):
            return {"exists": False}
        stats = client.get_collection_stats(self._collection)
        return {"exists": True, **stats}
