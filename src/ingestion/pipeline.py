"""文档摄取流水线。

编排完整的文档摄取流程：
  文件 → Loader 解析 → Chunker 分块 → Embedding 向量化 → Milvus 写入
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.types import Chunk, Document
from src.core.settings import Settings
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.chunking.document_chunker import DocumentChunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """文档摄取流水线。

    支持两种模式：
    1. 纯分块模式（不连 Milvus）：用于测试和调试
    2. 完整模式：分块 + 向量化 + 写入 Milvus + 构建知识图谱

    Example:
        >>> settings = load_settings()
        >>> pipeline = IngestionPipeline(settings)
        >>> # 纯分块
        >>> chunks, parents = pipeline.ingest_file("doc.pdf")
        >>> # 完整入库
        >>> pipeline_full = IngestionPipeline(settings, embedding_service=emb, milvus_store=store)
        >>> pipeline_full.ingest_file_to_milvus("doc.pdf")
    """

    def __init__(
        self,
        settings: Settings,
        embedding_service=None,
        milvus_store=None,
        graph_builder=None,
    ):
        self._settings = settings
        self._loaders = {
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".docx": None,  # 懒加载，避免 import 错误
            ".doc": None,
            ".xlsx": None,
            ".xls": None,
            ".txt": None,
            ".html": None,
            ".htm": None,
        }
        self._chunker = DocumentChunker(settings.ingestion)
        self._embedding = embedding_service
        self._store = milvus_store
        self._graph_builder = graph_builder

    def _get_loader(self, file_path: Path):
        """根据文件扩展名选择 Loader（懒加载）。"""
        ext = file_path.suffix.lower()
        if ext not in self._loaders:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self._loaders.keys())}"
            )

        loader = self._loaders[ext]
        if loader is None:
            loader = self._create_loader(ext)
            self._loaders[ext] = loader

        return loader

    def _create_loader(self, ext: str):
        """按扩展名懒创建 Loader。"""
        if ext in (".docx", ".doc"):
            from src.ingestion.loaders.word_loader import WordLoader
            return WordLoader()
        elif ext in (".xlsx", ".xls"):
            from src.ingestion.loaders.excel_loader import ExcelLoader
            return ExcelLoader()
        elif ext == ".txt":
            from src.ingestion.loaders.txt_loader import TxtLoader
            return TxtLoader()
        elif ext in (".html", ".htm"):
            from src.ingestion.loaders.html_loader import HtmlLoader
            return HtmlLoader()
        raise ValueError(f"No loader for {ext}")

    def ingest_file(
        self,
        file_path: str | Path,
    ) -> Tuple[List[Chunk], Dict[str, Dict]]:
        """摄取单个文件（纯分块，不写入向量库）。

        Args:
            file_path: 文件路径

        Returns:
            (chunks, parent_store) 二元组
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path.name}")

        loader = self._get_loader(file_path)
        documents = loader.load(file_path)
        logger.info(f"  Loaded {len(documents)} pages")

        all_chunks: List[Chunk] = []
        all_parents: Dict[str, Dict] = {}

        for doc in documents:
            chunks, parents = self._chunker.split_document(doc)
            all_chunks.extend(chunks)
            all_parents.update(parents)

        logger.info(
            f"  Total: {len(all_chunks)} leaf chunks, "
            f"{len(all_parents)} parent blocks"
        )

        return all_chunks, all_parents

    def ingest_directory(
        self,
        dir_path: str | Path,
        extensions: tuple = (".pdf", ".md", ".markdown"),
    ) -> Tuple[List[Chunk], Dict[str, Dict]]:
        """摄取目录下所有文件（纯分块）。"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_chunks: List[Chunk] = []
        all_parents: Dict[str, Dict] = {}

        files = sorted(dir_path.rglob("*"))
        target_files = [f for f in files if f.suffix.lower() in extensions]
        logger.info(f"Found {len(target_files)} files to ingest in {dir_path}")

        for file_path in target_files:
            try:
                chunks, parents = self.ingest_file(file_path)
                all_chunks.extend(chunks)
                all_parents.update(parents)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
                continue

        return all_chunks, all_parents

    def ingest_file_to_milvus(
        self,
        file_path: str | Path,
        batch_size: int = 50,
    ) -> Tuple[int, int]:
        """摄取单个文件并写入 Milvus。

        完整流程：加载 → 分块 → 向量化 → 写入 Milvus

        Args:
            file_path: 文件路径
            batch_size: 批量写入大小

        Returns:
            (chunk_count, parent_count)
        """
        if self._embedding is None or self._store is None:
            raise RuntimeError(
                "Embedding service and Milvus store are required for ingest_file_to_milvus. "
                "Pass them to IngestionPipeline constructor."
            )

        # Step 1: 分块
        chunks, parents = self.ingest_file(file_path)

        if not chunks:
            logger.warning(f"No chunks generated for {file_path}")
            return 0, 0

        # Step 2: 批量向量化并写入 Milvus
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            # 生成密集向量
            dense_vectors = self._embedding.embed_dense(texts)

            # 更新 BM25 统计
            self._embedding.bm25_increment_add(texts)

            # 更新倒排索引
            chunk_ids = [c.id for c in batch]
            metas = [c.metadata for c in batch]
            self._embedding.bm25_index_add(chunk_ids, texts, metas)

            # 构造 Milvus 数据
            data = []
            for chunk, vec in zip(batch, dense_vectors):
                data.append({
                    "dense_embedding": vec,
                    "text": chunk.text[:3999],  # Milvus VARCHAR 限制
                    "chunk_id": chunk.id,
                    "filename": chunk.metadata.get("filename", ""),
                    "source_path": chunk.metadata.get("source_path", ""),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "chunk_level": chunk.metadata.get("chunk_level", 3),
                    "parent_chunk_id": chunk.metadata.get("parent_chunk_id", ""),
                    "root_chunk_id": chunk.metadata.get("root_chunk_id", ""),
                })

            self._store.insert(data)
            logger.info(f"  Inserted batch {i // batch_size + 1}: {len(data)} records")

        logger.info(
            f"Ingestion to Milvus complete: {total} chunks from {Path(file_path).name}"
        )

        # Step 3: 构建知识图谱（如果配置了 GraphBuilder）
        if self._graph_builder is not None:
            try:
                import asyncio
                graph_stats = asyncio.run(
                    self._graph_builder.build_from_chunks(chunks)
                )
                logger.info(
                    f"Graph built: {graph_stats.get('total_entities', 0)} entities, "
                    f"{graph_stats.get('total_relations', 0)} relations"
                )
            except Exception as e:
                logger.error(f"Graph building failed (non-fatal): {e}")

        return total, len(parents)

    def ingest_directory_to_milvus(
        self,
        dir_path: str | Path,
        extensions: tuple = (".pdf", ".md", ".markdown"),
        batch_size: int = 50,
    ) -> Tuple[int, int]:
        """摄取目录下所有文件并写入 Milvus。"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        total_chunks = 0
        total_parents = 0

        files = sorted(dir_path.rglob("*"))
        target_files = [f for f in files if f.suffix.lower() in extensions]
        logger.info(f"Found {len(target_files)} files to ingest")

        for file_path in target_files:
            try:
                c, p = self.ingest_file_to_milvus(file_path, batch_size)
                total_chunks += c
                total_parents += p
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
                continue

        logger.info(f"All ingestion complete: {total_chunks} total chunks")
        return total_chunks, total_parents
