"""TXT 纯文本加载器。"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from src.core.types import Document
from src.ingestion.loaders.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class TxtLoader(BaseLoader):
    """纯文本加载器。

    按空行或固定行数切分为多个 Document。
    """

    def __init__(self, max_lines_per_doc: int = 200):
        self._max_lines = max_lines_per_doc

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        self._validate_file(file_path)

        logger.info(f"Loading TXT: {file_path.name}")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        filename = file_path.name

        # 按行数切分
        lines = text.split("\n")
        documents: List[Document] = []

        for i in range(0, len(lines), self._max_lines):
            chunk_lines = lines[i:i + self._max_lines]
            chunk_text = "\n".join(chunk_lines).strip()
            if not chunk_text:
                continue

            documents.append(Document(
                id=f"{file_hash}_p{i // self._max_lines + 1}",
                text=chunk_text,
                metadata={
                    "source_path": str(file_path),
                    "filename": filename,
                    "doc_type": "txt",
                    "doc_id": file_hash,
                    "page": i // self._max_lines + 1,
                    "title": filename,
                },
            ))

        logger.info(f"  Loaded {len(documents)} sections from {filename}")
        return documents
