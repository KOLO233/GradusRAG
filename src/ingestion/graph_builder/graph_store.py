"""知识图谱存储。

基于 NetworkX 的图存储实现，支持：
- 实体/关系的增删查
- 多跳邻居查询
- 图的序列化/反序列化
- 图统计数据

参考 LightRAG 的 BaseGraphStorage 接口设计。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.types import Entity, Relation

logger = logging.getLogger(__name__)


class GraphStore:
    """基于 NetworkX 的知识图谱存储。

    Example:
        >>> store = GraphStore()
        >>> store.add_entity(Entity(name="机器学习", entity_type="概念", description="..."))
        >>> store.add_relation(Relation(source="机器学习", target="人工智能", relation_type="属于"))
        >>> neighbors = store.get_neighbors("机器学习", hops=2)
    """

    def __init__(self, persist_path: Optional[str | Path] = None):
        self._persist_path = Path(persist_path) if persist_path else None
        self._graph = None  # 懒加载

    def _get_graph(self):
        if self._graph is None:
            try:
                import networkx as nx
                self._graph = nx.DiGraph()
                if self._persist_path and self._persist_path.exists():
                    self._load()
            except ImportError:
                raise ImportError("networkx is required. Install: pip install networkx")
        return self._graph

    # =========================================================================
    # 写入
    # =========================================================================

    def add_entity(self, entity: Entity) -> None:
        """添加实体到图中。如果已存在则更新描述。"""
        G = self._get_graph()
        if G.has_node(entity.name):
            # 合并描述
            existing = G.nodes[entity.name]
            if entity.description and entity.description not in existing.get("description", ""):
                existing["description"] = (
                    existing.get("description", "") + "；" + entity.description
                )
        else:
            G.add_node(
                entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_chunks=entity.source_chunks,
            )

    def add_relation(self, relation: Relation) -> None:
        """添加关系到图中。"""
        G = self._get_graph()
        # 确保源和目标实体存在
        if not G.has_node(relation.source):
            G.add_node(relation.source, entity_type="未知", description="")
        if not G.has_node(relation.target):
            G.add_node(relation.target, entity_type="未知", description="")

        # 添加边
        key = f"{relation.source}--{relation.relation_type}-->{relation.target}"
        G.add_edge(
            relation.source,
            relation.target,
            key=key,
            relation_type=relation.relation_type,
            description=relation.description,
            weight=relation.weight,
            source_chunks=relation.source_chunks,
        )

    def add_entities(self, entities: List[Entity]) -> None:
        """批量添加实体。"""
        for e in entities:
            self.add_entity(e)

    def add_relations(self, relations: List[Relation]) -> None:
        """批量添加关系。"""
        for r in relations:
            self.add_relation(r)

    # =========================================================================
    # 查询
    # =========================================================================

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """获取实体信息。"""
        G = self._get_graph()
        if G.has_node(name):
            data = dict(G.nodes[name])
            data["name"] = name
            return data
        return None

    def get_neighbors(
        self,
        entity_name: str,
        hops: int = 1,
        max_neighbors: int = 20,
    ) -> Dict[str, Any]:
        """获取实体的多跳邻居。

        Args:
            entity_name: 起始实体名
            hops: 跳数（1=直接邻居，2=二跳邻居）
            max_neighbors: 最大返回邻居数

        Returns:
            {
                "center": 起始实体,
                "entities": 邻居实体列表,
                "relations": 关系列表,
            }
        """
        G = self._get_graph()
        if not G.has_node(entity_name):
            return {"center": entity_name, "entities": [], "relations": []}

        visited: Set[str] = {entity_name}
        current_level = {entity_name}
        all_entities = []
        all_relations = []

        for hop in range(hops):
            next_level = set()
            for node in current_level:
                # 出边
                for _, target, data in G.out_edges(node, data=True):
                    if target not in visited and len(all_entities) < max_neighbors:
                        visited.add(target)
                        next_level.add(target)
                        all_entities.append({
                            "name": target,
                            "entity_type": G.nodes[target].get("entity_type", ""),
                            "description": G.nodes[target].get("description", ""),
                            "hop": hop + 1,
                        })
                        all_relations.append({
                            "source": node,
                            "target": target,
                            "relation_type": data.get("relation_type", ""),
                            "description": data.get("description", ""),
                            "hop": hop + 1,
                        })

                # 入边
                for source, _, data in G.in_edges(node, data=True):
                    if source not in visited and len(all_entities) < max_neighbors:
                        visited.add(source)
                        next_level.add(source)
                        all_entities.append({
                            "name": source,
                            "entity_type": G.nodes[source].get("entity_type", ""),
                            "description": G.nodes[source].get("description", ""),
                            "hop": hop + 1,
                        })
                        all_relations.append({
                            "source": source,
                            "target": node,
                            "relation_type": data.get("relation_type", ""),
                            "description": data.get("description", ""),
                            "hop": hop + 1,
                        })

            current_level = next_level
            if not current_level:
                break

        return {
            "center": entity_name,
            "entities": all_entities,
            "relations": all_relations,
        }

    def search_entities(self, keyword: str) -> List[Dict[str, Any]]:
        """按关键词搜索实体（模糊匹配）。"""
        G = self._get_graph()
        results = []
        for node in G.nodes():
            if keyword.lower() in node.lower():
                data = dict(G.nodes[node])
                data["name"] = node
                results.append(data)
        return results

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """获取所有实体。"""
        G = self._get_graph()
        return [
            {"name": n, **dict(G.nodes[n])}
            for n in G.nodes()
        ]

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """获取所有关系。"""
        G = self._get_graph()
        return [
            {"source": s, "target": t, **dict(G.edges[s, t])}
            for s, t in G.edges()
        ]

    # =========================================================================
    # 统计
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """获取图统计信息。"""
        G = self._get_graph()
        return {
            "entity_count": G.number_of_nodes(),
            "relation_count": G.number_of_edges(),
            "connected_components": 0,
        }

    def remove_by_source(self, filename: str) -> int:
        """移除来自指定文件的所有实体和关系。

        Args:
            filename: 文件名

        Returns:
            移除的实体数量
        """
        G = self._get_graph()
        nodes_to_remove = []

        for node in list(G.nodes()):
            data = G.nodes[node]
            chunks = data.get("source_chunks", [])
            if any(filename in chunk_id for chunk_id in chunks):
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            G.remove_node(node)

        # 清理孤立边（指向已删除节点的边）
        edges_to_remove = []
        for s, t, data in G.edges(data=True):
            chunks = data.get("source_chunks", [])
            if any(filename in chunk_id for chunk_id in chunks):
                edges_to_remove.append((s, t))

        for s, t in edges_to_remove:
            if G.has_edge(s, t):
                G.remove_edge(s, t)

        return len(nodes_to_remove)

    # =========================================================================
    # 持久化
    # =========================================================================

    def save(self, path: Optional[str | Path] = None) -> None:
        """保存图到 JSON 文件。"""
        save_path = Path(path) if path else self._persist_path
        if save_path is None:
            return

        G = self._get_graph()
        data = {
            "nodes": [
                {"name": n, **dict(G.nodes[n])}
                for n in G.nodes()
            ],
            "edges": [
                {"source": s, "target": t, **dict(G.edges[s, t])}
                for s, t in G.edges()
            ],
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Graph saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def _load(self) -> None:
        """从 JSON 文件加载图。"""
        if not self._persist_path or not self._persist_path.exists():
            return

        import networkx as nx

        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        G = self._get_graph()

        for node in data.get("nodes", []):
            name = node.pop("name")
            G.add_node(name, **node)

        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            G.add_edge(source, target, **edge)

        logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
