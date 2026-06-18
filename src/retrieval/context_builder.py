"""上下文构建器。

负责将检索结果转化为 LLM 可用的上下文文本。
支持 Auto-merging（叶子块自动合并到父块）和上下文压缩。

参考 SuperMew 的 _auto_merge_documents 和 _merge_to_parent_level 实现。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器。

    功能：
    1. Auto-merging：当多个叶子块属于同一父块时，合并为父块
    2. 上下文格式化：将检索结果格式化为 LLM 可读的文本
    3. 上下文压缩：截断过长的上下文

    Example:
        >>> builder = ContextBuilder(parent_store=parents)
        >>> context = builder.build(results, max_length=6000)
    """

    def __init__(
        self,
        parent_store: Dict[str, Dict] = None,
        auto_merge_enabled: bool = True,
        auto_merge_threshold: int = 2,
        llm_service=None,
    ):
        self._parent_store = parent_store or {}
        self._auto_merge_enabled = auto_merge_enabled
        self._auto_merge_threshold = auto_merge_threshold
        self._llm = llm_service

    def build(
        self,
        results: List[RetrievalResult],
        max_length: int = 6000,
        compress: bool = False,
        query: str = "",
    ) -> str:
        """构建上下文文本。

        Args:
            results: 检索结果列表
            max_length: 最大字符数
            compress: 是否启用上下文压缩（去除冗余，提取关键句）
            query: 查询文本（压缩时用于判断相关性）

        Returns:
            格式化的上下文文本
        """
        if not results:
            return ""

        # Auto-merging
        if self._auto_merge_enabled and self._parent_store:
            results = self._auto_merge(results)

        # 去重（相同 chunk_id 只保留一个）
        results = self._deduplicate(results)

        # 上下文压缩
        if compress:
            results = self._compress(results, query)

        # 格式化
        chunks = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("filename", "Unknown")
            page = r.metadata.get("page", "N/A")
            chunks.append(f"[{i}] {source} (Page {page}):\n{r.text}")

        context = "\n\n---\n\n".join(chunks)

        # 截断
        if len(context) > max_length:
            context = context[:max_length] + "\n\n...(上下文已截断)"

        return context

    def _auto_merge(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Auto-merging：将属于同一父块的叶子块合并为父块。

        当 L3 叶子块中，有 threshold 个以上属于同一个 L2 父块时，
        用 L2 父块替换这些叶子块，提供更完整的上下文。

        两段合并：L3→L2，再 L2→L1。
        """
        # L3 → L2 合并
        merged, count_l3_l2 = self._merge_to_parent(results)
        # L2 → L1 合并
        merged, count_l2_l1 = self._merge_to_parent(merged)

        total = count_l3_l2 + count_l2_l1
        if total > 0:
            logger.debug(f"Auto-merged: {count_l3_l2} L3→L2 + {count_l2_l1} L2→L1")

        return merged

    def _merge_to_parent(
        self,
        results: List[RetrievalResult],
    ) -> Tuple[List[RetrievalResult], int]:
        """将子块合并到父块。

        Returns:
            (merged_results, merge_count)
        """
        # 按 parent_chunk_id 分组
        groups: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for r in results:
            parent_id = r.metadata.get("parent_chunk_id", "").strip()
            if parent_id:
                groups[parent_id].append(r)

        # 找到需要合并的父块（子块数量 >= threshold）
        merge_ids = set()
        for parent_id, children in groups.items():
            if len(children) >= self._auto_merge_threshold and parent_id in self._parent_store:
                merge_ids.add(parent_id)

        if not merge_ids:
            return results, 0

        # 执行合并
        merged = []
        merged_count = 0
        seen = set()

        for r in results:
            parent_id = r.metadata.get("parent_chunk_id", "").strip()

            if parent_id and parent_id in merge_ids and parent_id not in seen:
                # 用父块替换
                parent_data = self._parent_store[parent_id]
                parent_result = RetrievalResult(
                    chunk_id=parent_id,
                    score=r.score,
                    text=parent_data["text"],
                    metadata={
                        "source_path": parent_data.get("filename", ""),
                        "filename": parent_data.get("filename", ""),
                        "page": parent_data.get("page", 0),
                        "chunk_level": parent_data.get("level", 2),
                        "parent_chunk_id": parent_data.get("parent_id", ""),
                        "root_chunk_id": parent_data.get("root_id", ""),
                        "merged_from_children": True,
                        "merged_child_count": len(groups[parent_id]),
                    },
                    retrieval_source=r.retrieval_source,
                )
                merged.append(parent_result)
                seen.add(parent_id)
                merged_count += 1
            elif parent_id and parent_id in merge_ids and parent_id in seen:
                # 同一父块的其他子块，跳过（已被父块替代）
                continue
            else:
                # 不需要合并的块，保留
                merged.append(r)

        # 去重
        deduped = []
        seen_ids = set()
        for r in merged:
            if r.chunk_id not in seen_ids:
                deduped.append(r)
                seen_ids.add(r.chunk_id)

        return deduped, merged_count

    def update_parent_store(self, parent_store: Dict[str, Dict]):
        """更新父块存储（入库后调用）。"""
        self._parent_store.update(parent_store)

    @staticmethod
    def _deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """按 chunk_id 去重，保留分数最高的。"""
        seen = {}
        for r in results:
            cid = r.chunk_id
            if cid not in seen or r.score > seen[cid].score:
                seen[cid] = r
        return list(seen.values())

    def _compress(
        self,
        results: List[RetrievalResult],
        query: str = "",
    ) -> List[RetrievalResult]:
        """上下文压缩：去除重复信息，保留关键内容。

        策略：
        1. 去除文本高度重叠的块（Jaccard 相似度 > 0.7）
        2. 截断过长的块到关键段落
        """
        if len(results) <= 2:
            return results

        compressed = []
        seen_texts = []

        for r in results:
            text = r.text.strip()
            if not text:
                continue

            # 检查是否与已有文本高度重复
            is_duplicate = False
            for seen in seen_texts:
                if self._jaccard_similarity(text, seen) > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                # 截断过长文本到前 800 字符
                if len(text) > 800:
                    text = text[:800] + "..."
                    r = RetrievalResult(
                        chunk_id=r.chunk_id,
                        score=r.score,
                        text=text,
                        metadata=r.metadata,
                        retrieval_source=r.retrieval_source,
                    )
                compressed.append(r)
                seen_texts.append(text)

        if len(compressed) < len(results):
            logger.debug(
                f"Context compression: {len(results)} → {len(compressed)} chunks"
            )

        return compressed

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """计算两段文本的 Jaccard 相似度（基于字符 bigram）。"""
        def bigrams(text):
            return set(text[i:i+2] for i in range(len(text)-1))

        b1 = bigrams(text1)
        b2 = bigrams(text2)
        if not b1 or not b2:
            return 0.0
        intersection = b1 & b2
        union = b1 | b2
        return len(intersection) / len(union) if union else 0.0
