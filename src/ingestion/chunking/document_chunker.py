"""文档分块适配器。

桥接 SlidingWindowChunker（纯文本切分）与 core.types（业务对象）。
将 ChunkNode 转换为标准的 Chunk 对象，注入元数据和层级关系。

参考 MODULAR-RAG-MCP-SERVER 的 DocumentChunker 适配器模式。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Tuple

from src.core.types import Chunk, Document
from src.core.settings import IngestionSettings
from src.ingestion.chunking.sliding_window import SlidingWindowChunker, ChunkNode

logger = logging.getLogger(__name__)


class DocumentChunker:
    """文档分块适配器。

    接收 Document 对象，使用 SlidingWindowChunker 切分，
    输出标准的 Chunk 对象列表（L3 叶子块）和父块信息。

    Example:
        >>> chunker = DocumentChunker(settings.ingestion)
        >>> chunks, parents = chunker.split_document(document)
        >>> # chunks: List[Chunk]  — L3 叶子块，用于向量化
        >>> # parents: Dict[str, str]  — {chunk_id: parent_text} 父块映射
    """

    def __init__(self, settings: IngestionSettings):
        self._settings = settings
        self._chunker = SlidingWindowChunker(settings)

    def split_document(self, document: Document) -> Tuple[List[Chunk], Dict[str, Dict]]:
        """将 Document 切分为 Chunk 对象。

        Args:
            document: 源文档

        Returns:
            (chunks, parent_store) 二元组:
            - chunks: L3 叶子块 Chunk 对象列表（写入向量库）
            - parent_store: L1/L2 父块信息字典（写入 DocStore）
              格式: {chunk_id: {"text": ..., "level": ..., "parent_id": ..., "root_id": ...}}
        """
        filename = document.metadata.get("filename", document.metadata.get("source_path", "unknown"))
        page = document.metadata.get("page", 0)
        doc_id = document.id

        # 三级分块
        nodes = self._chunker.chunk_text(
            text=document.text,
            filename=filename,
            page=page,
            doc_id=doc_id,
        )

        if not nodes:
            logger.warning(f"No chunks generated for document {document.id}")
            return [], {}

        # 分离 L1/L2（父块）和 L3（叶子块）
        parent_store: Dict[str, Dict] = {}
        leaf_chunks: List[Chunk] = []
        chunk_index = 0

        for node in nodes:
            if node.chunk_level in (1, 2):
                # 父块信息存入 DocStore（后续 Auto-merging 需要）
                parent_store[node.id] = {
                    "text": node.text,
                    "level": node.chunk_level,
                    "parent_id": node.parent_chunk_id,
                    "root_id": node.root_chunk_id,
                    "filename": node.metadata.get("filename", ""),
                    "page": node.metadata.get("page", 0),
                }
            else:
                # L3 叶子块转为标准 Chunk 对象
                chunk = self._node_to_chunk(
                    node=node,
                    document=document,
                    chunk_index=chunk_index,
                )
                leaf_chunks.append(chunk)
                chunk_index += 1

        logger.info(
            f"Split document '{filename}': "
            f"{len([n for n in nodes if n.chunk_level == 1])} L1 + "
            f"{len([n for n in nodes if n.chunk_level == 2])} L2 + "
            f"{len(leaf_chunks)} L3 leaves"
        )

        return leaf_chunks, parent_store

    def _node_to_chunk(
        self,
        node: ChunkNode,
        document: Document,
        chunk_index: int,
    ) -> Chunk:
        """将 ChunkNode 转换为标准 Chunk 对象。"""
        # 继承文档元数据并扩展
        metadata = document.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "chunk_level": node.chunk_level,
            "parent_chunk_id": node.parent_chunk_id,
            "root_chunk_id": node.root_chunk_id,
            "source_ref": document.id,
            "filename": node.metadata.get("filename", metadata.get("filename", "")),
            "page": node.metadata.get("page", metadata.get("page", 0)),
        })

        return Chunk(
            id=node.id,
            text=node.text,
            metadata=metadata,
            source_ref=document.id,
        )
