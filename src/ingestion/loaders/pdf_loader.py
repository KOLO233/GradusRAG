"""PDF 文档加载器。

使用 PyMuPDF (fitz) 解析 PDF，提取每页文本和元数据。
PyMuPDF 对中文支持好、速度快、纯本地运行。

输出：每页一个 Document 对象，text 为 Markdown 格式。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from src.core.types import Document
from src.ingestion.loaders.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """PDF 文档加载器。

    使用 PyMuPDF 解析 PDF，支持：
    - 中英文混合文本
    - 自动提取页码、标题
    - 文件哈希生成唯一 ID

    Example:
        >>> loader = PDFLoader()
        >>> docs = loader.load("data/documents/教材.pdf")
        >>> print(len(docs))  # 页数
        >>> print(docs[0].metadata["page"])  # 页码
    """

    def load(self, file_path: str | Path) -> List[Document]:
        """加载 PDF 文件，每页生成一个 Document。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 列表，每个对应 PDF 的一页
        """
        file_path = Path(file_path)
        self._validate_file(file_path)

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install it with: pip install pymupdf"
            )

        logger.info(f"Loading PDF: {file_path.name}")

        # 计算文件哈希作为文档 ID
        file_hash = self._compute_file_hash(file_path)
        filename = file_path.name

        doc = fitz.open(str(file_path))
        documents: List[Document] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or not text.strip():
                logger.debug(f"  Page {page_num + 1}: empty, skipping")
                continue

            # 清洗文本
            cleaned_text = self._clean_text(text)

            # 尝试提取页面标题（取前几个非空行）
            title = self._extract_title(cleaned_text)

            page_doc = Document(
                id=f"{file_hash}_p{page_num + 1}",
                text=cleaned_text,
                metadata={
                    "source_path": str(file_path),
                    "filename": filename,
                    "doc_type": "pdf",
                    "doc_id": file_hash,
                    "page": page_num + 1,  # 1-based 页码
                    "total_pages": len(doc),
                    "title": title,
                },
            )
            documents.append(page_doc)

        doc.close()

        logger.info(f"  Loaded {len(documents)} pages from {filename}")
        return documents

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """计算文件 SHA256 哈希（取前 16 位作为 ID）。"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗 PDF 提取的原始文本。

        - 去除多余空行
        - 修复断行（PDF 经常把一个段落拆成多行）
        - 去除页眉页脚常见噪声
        """
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # 跳过纯数字行（页码）
            if line.isdigit() and len(line) <= 3:
                continue
            # 跳过常见页眉页脚
            if line.lower().startswith(("page ", "第", "— ", "– ")):
                continue
            cleaned_lines.append(line)

        # 合并连续非空行为段落
        result_lines = []
        current_para = []

        for line in cleaned_lines:
            if line:
                current_para.append(line)
            else:
                if current_para:
                    result_lines.append(" ".join(current_para))
                    current_para = []
                    result_lines.append("")  # 保留段落间的空行

        if current_para:
            result_lines.append(" ".join(current_para))

        return "\n".join(result_lines).strip()

    @staticmethod
    def _extract_title(text: str) -> str:
        """从页面文本中提取标题（取第一个非空行）。"""
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) < 100:  # 标题通常较短
                return line
        return ""
