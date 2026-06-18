"""Markdown 文档加载器。

将 Markdown 文件解析为 Document 对象，按标题分割为多个段落。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import List

from src.core.types import Document
from src.ingestion.loaders.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器。

    将 Markdown 文件按一级/二级标题分割为多个 Document。
    """

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        self._validate_file(file_path)

        text = file_path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        filename = file_path.name

        # 按一级或二级标题切分
        sections = self._split_by_headers(text)

        documents: List[Document] = []
        for i, (title, content) in enumerate(sections):
            if not content.strip():
                continue
            doc = Document(
                id=f"{file_hash}_s{i + 1}",
                text=f"## {title}\n\n{content}" if title else content,
                metadata={
                    "source_path": str(file_path),
                    "filename": filename,
                    "doc_type": "markdown",
                    "doc_id": file_hash,
                    "page": i + 1,
                    "title": title,
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} sections from {filename}")
        return documents

    @staticmethod
    def _split_by_headers(text: str) -> List[tuple]:
        """按 # 和 ## 标题切分 Markdown。"""
        pattern = r'^(#{1,2})\s+(.+)$'
        sections = []
        current_title = ""
        current_lines = []

        for line in text.split("\n"):
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # 保存上一段
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # 最后一段
        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return sections if sections else [("", text)]
