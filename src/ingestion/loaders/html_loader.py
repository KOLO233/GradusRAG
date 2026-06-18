"""HTML 加载器。

解析 HTML 文件，提取正文文本，去除标签和脚本。
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


class HtmlLoader(BaseLoader):
    """HTML 文档加载器。

    提取 HTML 正文文本，去除标签、脚本、样式。
    """

    def load(self, file_path: str | Path) -> List[Document]:
        file_path = Path(file_path)
        self._validate_file(file_path)

        logger.info(f"Loading HTML: {file_path.name}")

        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        file_hash = hashlib.sha256(raw_html.encode()).hexdigest()[:16]
        filename = file_path.name

        # 提取标题
        title_match = re.search(r'<title[^>]*>(.*?)</title>', raw_html, re.DOTALL | re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else filename

        # 清洗 HTML
        text = self._clean_html(raw_html)

        if not text.strip():
            return []

        return [Document(
            id=file_hash,
            text=f"## {title}\n\n{text}" if title else text,
            metadata={
                "source_path": str(file_path),
                "filename": filename,
                "doc_type": "html",
                "doc_id": file_hash,
                "page": 1,
                "title": title,
            },
        )]

    @staticmethod
    def _clean_html(html: str) -> str:
        """清洗 HTML，提取正文文本。"""
        # 移除 script 和 style
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # 移除 HTML 标签
        html = re.sub(r'<[^>]+>', ' ', html)
        # 解码 HTML 实体
        html = html.replace('&nbsp;', ' ').replace('&amp;', '&')
        html = html.replace('&lt;', '<').replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        # 清理空白
        html = re.sub(r'[ \t]+', ' ', html)
        html = re.sub(r'\n\s*\n', '\n\n', html)
        return html.strip()
