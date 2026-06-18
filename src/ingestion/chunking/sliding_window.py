"""三级滑动窗口分块器。

实现 L1/L2/L3 三级嵌套分块：
- L1（根块）：2500 tokens，主题级上下文，存入 DocStore
- L2（中间块）：1024 tokens，段落级上下文，存入 DocStore
- L3（叶子块）：512 tokens，精细检索单元，写入 VectorStore

关键设计：
- 三级嵌套：L1 内切 L2，L2 内切 L3，保证父子关系
- Leaf-only 向量化：仅 L3 写入向量库，减少冗余
- Auto-merging 支持：通过 parent_chunk_id / root_chunk_id 追溯层级

参考 SuperMew 的 document_loader.py 三级分块实现。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.settings import IngestionSettings

logger = logging.getLogger(__name__)


@dataclass
class ChunkNode:
    """分块节点，包含层级关系信息。

    Attributes:
        id: 唯一标识
        text: 文本内容
        chunk_level: 层级 (1/2/3)
        parent_chunk_id: 父块 ID
        root_chunk_id: 根块 ID
        metadata: 附加元数据
    """
    id: str
    text: str
    chunk_level: int
    parent_chunk_id: str = ""
    root_chunk_id: str = ""
    metadata: Dict = field(default_factory=dict)


class SlidingWindowChunker:
    """三级滑动窗口分块器。

    将文本递归切分为三级层次结构：
    L1 → L2 → L3，每级使用不同的 chunk_size。

    Example:
        >>> chunker = SlidingWindowChunker(settings.ingestion)
        >>> nodes = chunker.chunk_text(text, filename="doc.pdf", page=1)
        >>> l3_nodes = [n for n in nodes if n.chunk_level == 3]
        >>> print(f"L3 leaf chunks: {len(l3_nodes)}")
    """

    def __init__(self, settings: IngestionSettings):
        self._settings = settings
        self._separators = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

    def chunk_text(
        self,
        text: str,
        filename: str = "",
        page: int = 0,
        doc_id: str = "",
    ) -> List[ChunkNode]:
        """对文本执行三级分块。

        Args:
            text: 待分块的文本
            filename: 来源文件名
            page: 页码
            doc_id: 文档 ID

        Returns:
            所有层级的 ChunkNode 列表（L1 + L2 + L3）
        """
        if not text or not text.strip():
            return []

        all_nodes: List[ChunkNode] = []

        # ---- L1: 根块 ----
        l1_chunks = self._split(
            text,
            self._settings.chunk_size_l1,
            self._settings.chunk_overlap,
        )

        for l1_idx, l1_text in enumerate(l1_chunks):
            if not l1_text.strip():
                continue

            l1_id = self._build_id(filename, page, 1, l1_idx)
            l1_node = ChunkNode(
                id=l1_id,
                text=l1_text,
                chunk_level=1,
                parent_chunk_id="",
                root_chunk_id=l1_id,
                metadata={
                    "filename": filename,
                    "page": page,
                    "doc_id": doc_id,
                },
            )
            all_nodes.append(l1_node)

            # ---- L2: 中间块（在 L1 内部切分）----
            l2_chunks = self._split(
                l1_text,
                self._settings.chunk_size_l2,
                self._settings.chunk_overlap,
            )

            for l2_idx, l2_text in enumerate(l2_chunks):
                if not l2_text.strip():
                    continue

                l2_id = self._build_id(filename, page, 2, l1_idx * 100 + l2_idx)
                l2_node = ChunkNode(
                    id=l2_id,
                    text=l2_text,
                    chunk_level=2,
                    parent_chunk_id=l1_id,
                    root_chunk_id=l1_id,
                    metadata={
                        "filename": filename,
                        "page": page,
                        "doc_id": doc_id,
                    },
                )
                all_nodes.append(l2_node)

                # ---- L3: 叶子块（在 L2 内部切分）----
                l3_chunks = self._split(
                    l2_text,
                    self._settings.chunk_size_l3,
                    self._settings.chunk_overlap,
                )

                for l3_idx, l3_text in enumerate(l3_chunks):
                    if not l3_text.strip():
                        continue

                    l3_id = self._build_id(
                        filename, page, 3,
                        l1_idx * 10000 + l2_idx * 100 + l3_idx,
                    )
                    l3_node = ChunkNode(
                        id=l3_id,
                        text=l3_text,
                        chunk_level=3,
                        parent_chunk_id=l2_id,
                        root_chunk_id=l1_id,
                        metadata={
                            "filename": filename,
                            "page": page,
                            "doc_id": doc_id,
                        },
                    )
                    all_nodes.append(l3_node)

        l1_count = sum(1 for n in all_nodes if n.chunk_level == 1)
        l2_count = sum(1 for n in all_nodes if n.chunk_level == 2)
        l3_count = sum(1 for n in all_nodes if n.chunk_level == 3)
        logger.debug(
            f"Chunked '{filename}' p{page}: "
            f"L1={l1_count}, L2={l2_count}, L3={l3_count}"
        )

        return all_nodes

    def _split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """递归字符文本切分。

        按 separators 优先级尝试分割，保证不超过 chunk_size。
        """
        if len(text) <= chunk_size:
            return [text]

        return self._recursive_split(text, chunk_size, overlap, self._separators)

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        separators: List[str],
    ) -> List[str]:
        """递归切分实现。"""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        # 找到能用的分隔符
        separator = ""
        for sep in separators:
            if sep in text:
                separator = sep
                break

        if not separator:
            # 无分隔符，硬切
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap
            return chunks

        # 按分隔符切分
        splits = text.split(separator)
        chunks = []
        current = ""

        for part in splits:
            candidate = current + separator + part if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # 如果单个 part 就超长，递归用下一级分隔符
                if len(part) > chunk_size:
                    sub_seps = separators[separators.index(separator) + 1:]
                    if sub_seps:
                        sub_chunks = self._recursive_split(
                            part, chunk_size, overlap, sub_seps
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # 硬切
                        hard = self._recursive_split(part, chunk_size, overlap, [""])
                        chunks.extend(hard)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def _build_id(filename: str, page: int, level: int, index: int) -> str:
        """构建确定性的分块 ID。

        格式: {filename}::p{page}::l{level}::{index}
        """
        return f"{filename}::p{page}::l{level}::{index}"
