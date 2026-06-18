"""Graph Retriever — 知识图谱检索器。

参考 LightRAG 的 kg_query 思路：
1. 从查询中提取关键词/实体
2. 在图中定位相关实体
3. 多跳遍历获取关联实体和关系
4. 将子图转化为文本上下文

L3 使用多跳遍历（因果链），L4 使用单跳（事实锚点）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.types import RetrievalResult
from src.ingestion.graph_builder.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphRetriever:
    """知识图谱检索器。

    流程：
    1. 实体定位：从查询中提取关键词，在图中搜索匹配实体
    2. 子图遍历：从匹配实体出发，多跳遍历获取关联信息
    3. 上下文构建：将子图信息转化为文本

    实体匹配策略：
    - 精确匹配：1.0 分
    - 子串匹配：0.6-0.9 分（按重叠比例）
    - n-gram 相似度：0.3-0.7 分（捕获"梯度消失"≈"梯度弥散"这类近义词）
    - 阈值过滤：>= 0.4 才保留（从 0.5 降低，让更多近义词能匹配上）

    Example:
        >>> retriever = GraphRetriever(graph_store)
        >>> results = await retriever.retrieve("为什么会出现梯度消失？", hops=2)
    """

    def __init__(self, graph_store: GraphStore, embedding_service=None):
        self._store = graph_store
        self._embedding = embedding_service  # 可选，用于语义匹配

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,  # 从 10 改为 3，减少噪声
        hops: int = 2,
    ) -> List[RetrievalResult]:
        """执行图检索。

        Args:
            query: 查询文本
            top_k: 返回的最大结果数（默认 3，减少噪声）
            hops: 遍历跳数（L3 用 2，L4 用 1）

        Returns:
            RetrievalResult 列表，每个结果包含一段子图上下文
        """
        # Step 1: 提取查询关键词
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # Step 2: 在图中搜索匹配实体（带相关性评分）
        matched_entities = []
        for kw in keywords:
            found = self._store.search_entities(kw)
            for entity in found:
                # 计算实体匹配质量
                entity_name = entity["name"]
                match_score = self._compute_match_score(kw, entity_name)
                if match_score >= 0.4:  # 降低阈值，让更多近义词能匹配
                    entity["_match_score"] = match_score
                    entity["_match_keyword"] = kw
                    matched_entities.append(entity)

        if not matched_entities:
            logger.debug(f"Graph: no entities matched for keywords {keywords}")
            return []

        # 去重并按匹配质量排序（优先精确匹配的实体）
        seen_names = set()
        unique_entities = []
        for entity in sorted(matched_entities, key=lambda e: e["_match_score"], reverse=True):
            name = entity["name"]
            if name not in seen_names:
                seen_names.add(name)
                unique_entities.append(entity)

        # Step 3: 从匹配实体出发，多跳遍历
        all_subgraph_contexts = []
        seen_entities = set()

        for entity in unique_entities[:3]:  # 最多从 3 个高质量实体出发（从 5 改为 3）
            entity_name = entity["name"]
            if entity_name in seen_entities:
                continue
            seen_entities.add(entity_name)

            neighbors = self._store.get_neighbors(
                entity_name, hops=hops, max_neighbors=5  # 从 10 改为 5，减少噪声
            )

            # 过滤：只保留有实际关联的子图
            rel_count = len(neighbors.get("relations", []))
            if rel_count == 0:
                continue

            # Step 4: 将子图转化为文本上下文
            context_text = self._subgraph_to_text(entity_name, neighbors)
            if context_text:
                all_subgraph_contexts.append({
                    "center": entity_name,
                    "text": context_text,
                    "entity_count": len(neighbors.get("entities", [])),
                    "relation_count": rel_count,
                    "match_score": entity.get("_match_score", 0.5),
                })

        # 按综合分数排序（匹配质量 × 关联密度）
        all_subgraph_contexts.sort(
            key=lambda x: x["match_score"] * (x["entity_count"] * x["relation_count"]),
            reverse=True,
        )

        # 转换为 RetrievalResult
        results = []
        for i, ctx in enumerate(all_subgraph_contexts[:top_k]):
            # 分数 = 匹配质量 × 关联密度归一化
            max_score = max(
                (c["match_score"] * c["entity_count"] * c["relation_count"]
                 for c in all_subgraph_contexts),
                default=1
            )
            raw_score = ctx["match_score"] * ctx["entity_count"] * ctx["relation_count"]
            score = raw_score / max_score if max_score > 0 else 0.1
            results.append(RetrievalResult(
                chunk_id=f"graph::{ctx['center']}",
                score=score,
                text=ctx["text"],
                metadata={
                    "source_path": "knowledge_graph",
                    "filename": "knowledge_graph",
                    "center_entity": ctx["center"],
                    "entity_count": ctx["entity_count"],
                    "relation_count": ctx["relation_count"],
                    "match_score": ctx["match_score"],
                    "hops": hops,
                },
                retrieval_source="graph",
            ))

        logger.debug(
            f"Graph retrieval: keywords={keywords}, "
            f"matched={len(matched_entities)}, results={len(results)}"
        )
        return results

    @staticmethod
    def _compute_match_score(keyword: str, entity_name: str) -> float:
        """计算关键词与实体名的匹配质量（多维度相似度）。

        策略：
        1. 精确匹配 → 1.0
        2. 子串匹配 → 0.6-0.9（按重叠比例）
        3. n-gram Jaccard 相似度 → 0.3-0.7（捕获近义词，如"梯度消失"≈"梯度弥散"）

        Returns:
            0.0 ~ 1.0 的匹配分数
        """
        kw_lower = keyword.lower().strip()
        en_lower = entity_name.lower().strip()

        if not kw_lower or not en_lower:
            return 0.0

        # 精确匹配
        if kw_lower == en_lower:
            return 1.0

        # 子串匹配
        if kw_lower in en_lower:
            # 关键词是实体名的子串：关键词越长，匹配越精确
            return 0.6 + 0.3 * (len(kw_lower) / len(en_lower))
        if en_lower in kw_lower:
            return 0.6

        # n-gram Jaccard 相似度（字符级 bigram）
        def char_bigrams(s: str) -> set:
            return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

        kw_bigrams = char_bigrams(kw_lower)
        en_bigrams = char_bigrams(en_lower)
        intersection = kw_bigrams & en_bigrams
        union = kw_bigrams | en_bigrams

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Jaccard > 0.3 才认为有意义的匹配
        if jaccard >= 0.3:
            return 0.3 + 0.4 * jaccard  # 映射到 0.3-0.7 范围

        return 0.0

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词（中英文双语支持）。

        使用 src.libs.text_utils 的公共实现，与其他模块保持一致。
        """
        from src.libs.text_utils import extract_keywords
        return extract_keywords(query, min_length=2)

    def _subgraph_to_text(self, center: str, neighbors: Dict) -> str:
        """将子图信息转化为可读文本。

        格式参考 LightRAG 的实体/关系上下文格式。
        """
        entities = neighbors.get("entities", [])
        relations = neighbors.get("relations", [])

        if not entities and not relations:
            return ""

        lines = []
        lines.append(f"## 关于「{center}」的知识图谱信息")

        # 实体信息
        if entities:
            lines.append("\n### 相关实体")
            for e in entities:
                hop_str = f" (第{e['hop']}跳)" if e.get("hop", 1) > 1 else ""
                desc = f": {e['description']}" if e.get("description") else ""
                lines.append(f"- **{e['name']}** [{e.get('entity_type', '')}]{hop_str}{desc}")

        # 关系信息
        if relations:
            lines.append("\n### 关系链")
            for r in relations:
                hop_str = f" (第{r['hop']}跳)" if r.get("hop", 1) > 1 else ""
                desc = f": {r['description']}" if r.get("description") else ""
                lines.append(f"- **{r['source']}** →[{r.get('relation_type', '')}]→ **{r['target']}**{hop_str}{desc}")

        return "\n".join(lines)
