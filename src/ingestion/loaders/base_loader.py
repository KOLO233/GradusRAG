"""文档加载器抽象接口。

所有格式的 Loader 都继承此接口，实现可插拔设计。
参考 MODULAR-RAG-MCP-SERVER 的 BaseLoader 思路。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.core.types import Document


class BaseLoader(ABC):
    """文档加载器基类。

    子类只需实现 load() 方法，将原始文件解析为标准化的 Document 对象。
    Document.text 应为 Markdown 格式，metadata 必须包含 source_path。
    """

    @abstractmethod
    def load(self, file_path: str | Path) -> List[Document]:
        """加载文件并返回 Document 列表。

        Args:
            file_path: 文件路径

        Returns:
            Document 列表（PDF 通常每页一个 Document）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        ...

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        """验证文件存在且可读。"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
