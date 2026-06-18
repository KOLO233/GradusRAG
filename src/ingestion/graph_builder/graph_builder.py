"""知识图谱构建流水线。

从文档分块中抽取实体和关系，构建知识图谱。
参考 LightRAG 的文档摄取 → 实体抽取 → 图构建流程。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.core.types import Chunk, Entity, Relation
from src.core.settings import GraphSettings
from src.ingestion.graph_builder.entity_extractor import EntityExtractor
from src.ingestion.graph_builder.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphBuilder:
    """知识图谱构建器。

    流程：
    1. 遍历所有文档分块
    2. 对每个分块调用 LLM 抽取实体和关系
    3. 将实体和关系写入 GraphStore

    Example:
        >>> builder = GraphBuilder(settings, llm_service, graph_store)
        >>> builder.build_from_chunks(chunks)
        >>> print(graph_store.stats())
    """

    def __init__(
        self,
        settings: GraphSettings,
        llm_service=None,
        graph_store: GraphStore = None,
    ):
        self._settings = settings
        self._extractor = EntityExtractor(
            llm_service=llm_service,
            entity_types=settings.entity_types,
        )
        self._store = graph_store or GraphStore()

    async def build_from_chunks(self, chunks: List[Chunk]) -> Dict:
        """从分块列表构建知识图谱。

        Args:
            chunks: 文档分块列表

        Returns:
            构建统计信息
        """
        total_entities = 0
        total_relations = 0
        processed = 0

        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")

        for chunk in chunks:
            try:
                entities, relations = await self._extractor.extract(chunk.text)

                # 为实体和关系添加来源信息
                for e in entities:
                    e.source_chunks = [chunk.id]
                for r in relations:
                    r.source_chunks = [chunk.id]

                # 写入图存储
                self._store.add_entities(entities)
                self._store.add_relations(relations)

                total_entities += len(entities)
                total_relations += len(relations)
                processed += 1

                if processed % 10 == 0:
                    logger.info(f"  Processed {processed}/{len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")
                continue

        # 保存图
        self._store.save()

        stats = {
            "processed_chunks": processed,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "graph_stats": self._store.stats(),
        }

        logger.info(
            f"Knowledge graph built: {total_entities} entities, "
            f"{total_relations} relations from {processed} chunks"
        )

        return stats

    @property
    def graph_store(self) -> GraphStore:
        return self._store
