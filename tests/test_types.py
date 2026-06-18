"""GradusRAG 核心类型测试。"""

import pytest
from src.core.types import (
    Document, Chunk, ChunkRecord,
    QueryClassification, ProcessedQuery,
    RetrievalResult, Entity, Relation,
    Citation, RAGResponse, EvalTestCase,
)


class TestDocument:
    def test_create_document(self):
        doc = Document(
            id="doc_001",
            text="# Title\n\nContent here.",
            metadata={"source_path": "test.pdf", "doc_type": "pdf"},
        )
        assert doc.id == "doc_001"
        assert doc.metadata["source_path"] == "test.pdf"

    def test_document_requires_source_path(self):
        with pytest.raises(ValueError, match="source_path"):
            Document(id="d1", text="text", metadata={})

    def test_document_serialization(self):
        doc = Document(id="d1", text="text", metadata={"source_path": "a.pdf"})
        d = doc.to_dict()
        assert d["id"] == "d1"
        restored = Document.from_dict(d)
        assert restored.id == doc.id


class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(
            id="c_001",
            text="First paragraph.",
            metadata={"source_path": "test.pdf", "chunk_index": 0, "chunk_level": "L3"},
            source_ref="doc_001",
        )
        assert chunk.metadata["chunk_level"] == "L3"
        assert chunk.source_ref == "doc_001"


class TestRetrievalResult:
    def test_create_result(self):
        r = RetrievalResult(
            chunk_id="c1", score=0.95, text="text",
            metadata={"source_path": "a.pdf"},
            retrieval_source="dense",
        )
        assert r.retrieval_source == "dense"

    def test_empty_chunk_id_raises(self):
        with pytest.raises(ValueError, match="chunk_id"):
            RetrievalResult(chunk_id="", score=0.5, text="text")


class TestEntity:
    def test_entity_id(self):
        e = Entity(name="高血压", entity_type="疾病")
        assert e.id == "疾病::高血压"


class TestRelation:
    def test_relation_id(self):
        r = Relation(source="A药", target="肝损伤", relation_type="导致")
        assert r.id == "A药--导致-->肝损伤"


class TestRAGResponse:
    def test_response_serialization(self):
        resp = RAGResponse(
            answer="这是答案",
            query_level="L1",
            citations=[Citation(index=1, source="a.pdf", score=0.9)],
        )
        d = resp.to_dict()
        assert d["query_level"] == "L1"
        assert len(d["citations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
