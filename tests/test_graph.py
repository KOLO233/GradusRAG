"""知识图谱模块测试。

验证 GraphStore（NetworkX 图存储）、EntityExtractor（规则抽取）、
GraphRetriever（图检索）的正确性。
"""

import asyncio
import pytest
from src.core.types import Entity, Relation
from src.ingestion.graph_builder.graph_store import GraphStore
from src.ingestion.graph_builder.entity_extractor import EntityExtractor
from src.retrieval.graph_retriever import GraphRetriever


# ===========================================================================
# GraphStore 测试
# ===========================================================================

class TestGraphStore:

    def test_add_entity(self):
        store = GraphStore()
        store.add_entity(Entity(name="机器学习", entity_type="概念", description="ML描述"))
        entity = store.get_entity("机器学习")
        assert entity is not None
        assert entity["entity_type"] == "概念"

    def test_add_relation(self):
        store = GraphStore()
        store.add_relation(Relation(source="机器学习", target="人工智能", relation_type="属于"))
        relations = store.get_all_relations()
        assert len(relations) == 1
        assert relations[0]["relation_type"] == "属于"

    def test_get_neighbors(self):
        store = GraphStore()
        store.add_entity(Entity(name="A", entity_type="概念"))
        store.add_entity(Entity(name="B", entity_type="技术"))
        store.add_entity(Entity(name="C", entity_type="方法"))
        store.add_relation(Relation(source="A", target="B", relation_type="包含"))
        store.add_relation(Relation(source="B", target="C", relation_type="使用"))

        # 1 跳邻居
        n1 = store.get_neighbors("A", hops=1)
        assert len(n1["entities"]) == 1
        assert n1["entities"][0]["name"] == "B"

        # 2 跳邻居
        n2 = store.get_neighbors("A", hops=2)
        assert len(n2["entities"]) == 2
        names = {e["name"] for e in n2["entities"]}
        assert "B" in names and "C" in names

    def test_search_entities(self):
        store = GraphStore()
        store.add_entity(Entity(name="梯度消失", entity_type="概念"))
        store.add_entity(Entity(name="梯度爆炸", entity_type="概念"))
        store.add_entity(Entity(name="机器学习", entity_type="概念"))

        results = store.search_entities("梯度")
        assert len(results) == 2

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "graph.json"

        store1 = GraphStore(persist_path=path)
        store1.add_entity(Entity(name="A", entity_type="概念", description="desc A"))
        store1.add_relation(Relation(source="A", target="B", relation_type="属于"))
        store1.save()

        store2 = GraphStore(persist_path=path)
        assert store2.get_entity("A") is not None
        assert len(store2.get_all_relations()) == 1

    def test_stats(self):
        store = GraphStore()
        store.add_entity(Entity(name="A", entity_type="概念"))
        store.add_entity(Entity(name="B", entity_type="技术"))
        store.add_relation(Relation(source="A", target="B", relation_type="属于"))

        stats = store.stats()
        assert stats["entity_count"] == 2
        assert stats["relation_count"] == 1

    def test_update_entity_description(self):
        store = GraphStore()
        store.add_entity(Entity(name="A", entity_type="概念", description="第一次描述"))
        store.add_entity(Entity(name="A", entity_type="概念", description="第二次描述"))

        entity = store.get_entity("A")
        assert "第一次描述" in entity["description"]
        assert "第二次描述" in entity["description"]


# ===========================================================================
# EntityExtractor 规则抽取测试
# ===========================================================================

class TestEntityExtractorRuleBased:

    def test_extract_technical_terms(self):
        extractor = EntityExtractor(llm_service=None)
        text = "深度学习是机器学习的子领域，使用神经网络进行训练。"
        entities, relations = extractor._rule_based_extract(text)

        names = {e.name for e in entities}
        assert any("学习" in n for n in names)

    def test_extract_empty_text(self):
        extractor = EntityExtractor(llm_service=None)
        entities, relations = extractor._rule_based_extract("")
        assert entities == []
        assert relations == []


# ===========================================================================
# GraphRetriever 测试
# ===========================================================================

class TestGraphRetriever:

    def test_retrieve_with_matching_entities(self):
        store = GraphStore()
        store.add_entity(Entity(name="梯度消失", entity_type="概念", description="深层网络训练问题"))
        store.add_entity(Entity(name="Sigmoid", entity_type="技术", description="激活函数"))
        store.add_relation(Relation(source="Sigmoid", target="梯度消失", relation_type="导致"))

        retriever = GraphRetriever(store)
        results = asyncio.run(retriever.retrieve("为什么会出现梯度消失？", hops=2))

        assert len(results) > 0
        assert results[0].retrieval_source == "graph"
        assert "梯度消失" in results[0].text

    def test_retrieve_no_match(self):
        store = GraphStore()
        store.add_entity(Entity(name="机器学习", entity_type="概念"))

        retriever = GraphRetriever(store)
        results = asyncio.run(retriever.retrieve("完全无关的查询XYZ", hops=1))

        assert results == []

    def test_retrieve_multi_hop(self):
        store = GraphStore()
        store.add_entity(Entity(name="卷积神经网络", entity_type="技术"))
        store.add_entity(Entity(name="图像识别", entity_type="应用"))
        store.add_entity(Entity(name="特征提取", entity_type="方法"))
        store.add_relation(Relation(source="卷积神经网络", target="图像识别", relation_type="用于"))
        store.add_relation(Relation(source="卷积神经网络", target="特征提取", relation_type="实现"))

        retriever = GraphRetriever(store)
        results = asyncio.run(retriever.retrieve("卷积神经网络是什么？", hops=2))

        assert len(results) > 0
        assert "卷积神经网络" in results[0].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
