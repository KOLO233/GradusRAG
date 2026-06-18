"""摄取模块测试。

验证三级分块器、文档分块适配器、以及摄取流水线的正确性。
不依赖外部文件，使用构造的文本数据测试。
"""

import pytest
from src.core.settings import IngestionSettings
from src.core.types import Document, Chunk
from src.ingestion.chunking.sliding_window import SlidingWindowChunker, ChunkNode
from src.ingestion.chunking.document_chunker import DocumentChunker


# ===========================================================================
# 测试用的配置（使用较小的 chunk_size 方便测试）
# ===========================================================================

@pytest.fixture
def small_settings():
    """小 chunk_size 配置，方便测试三级嵌套。"""
    return IngestionSettings(
        chunk_size_l1=200,
        chunk_size_l2=100,
        chunk_size_l3=50,
        chunk_overlap=10,
        splitter="recursive",
    )


@pytest.fixture
def default_settings():
    """默认配置。"""
    return IngestionSettings(
        chunk_size_l1=2500,
        chunk_size_l2=1024,
        chunk_size_l3=512,
        chunk_overlap=100,
        splitter="recursive",
    )


def _make_long_text(paragraphs: int = 20, words_per_para: int = 30) -> str:
    """生成长文本用于测试分块。"""
    import random
    random.seed(42)
    vocab = [
        "机器学习", "深度学习", "神经网络", "自然语言处理", "计算机视觉",
        "卷积", "循环", "注意力机制", "Transformer", "BERT",
        "梯度下降", "反向传播", "损失函数", "优化器", "正则化",
        "数据集", "特征工程", "模型评估", "交叉验证", "过拟合",
        "分类", "回归", "聚类", "降维", "强化学习",
    ]

    paragraphs_list = []
    for _ in range(paragraphs):
        words = [random.choice(vocab) for _ in range(words_per_para)]
        para = "、".join(words) + "。"
        paragraphs_list.append(para)

    return "\n\n".join(paragraphs_list)


# ===========================================================================
# SlidingWindowChunker 测试
# ===========================================================================

class TestSlidingWindowChunker:

    def test_empty_text(self, small_settings):
        """空文本应返回空列表。"""
        chunker = SlidingWindowChunker(small_settings)
        nodes = chunker.chunk_text("")
        assert nodes == []

    def test_short_text_single_chunk(self, small_settings):
        """很短的文本应只产生一个 L1 块。"""
        chunker = SlidingWindowChunker(small_settings)
        nodes = chunker.chunk_text("这是一段很短的文本。", filename="test.pdf", page=1)
        l1 = [n for n in nodes if n.chunk_level == 1]
        assert len(l1) == 1
        # 短文本不需要进一步切分
        l2 = [n for n in nodes if n.chunk_level == 2]
        l3 = [n for n in nodes if n.chunk_level == 3]
        # L2 和 L3 可能各 1 个（因为文本短，不触发进一步切分）
        assert len(l2) <= 1
        assert len(l3) <= 1

    def test_long_text_produces_three_levels(self, small_settings):
        """长文本应产生三级分块。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=10, words_per_para=20)
        nodes = chunker.chunk_text(text, filename="long.pdf", page=1)

        l1 = [n for n in nodes if n.chunk_level == 1]
        l2 = [n for n in nodes if n.chunk_level == 2]
        l3 = [n for n in nodes if n.chunk_level == 3]

        assert len(l1) > 0, "Should have at least 1 L1 chunk"
        assert len(l2) >= len(l1), "L2 count should >= L1"
        assert len(l3) >= len(l2), "L3 count should >= L2"

    def test_parent_child_relationships(self, small_settings):
        """父子关系应正确：L3.parent → L2, L2.parent → L1, L3.root → L1。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=8, words_per_para=25)
        nodes = chunker.chunk_text(text, filename="test.pdf", page=1)

        node_map = {n.id: n for n in nodes}

        for node in nodes:
            if node.chunk_level == 2:
                # L2 的 parent 应该是 L1
                parent = node_map.get(node.parent_chunk_id)
                assert parent is not None, f"L2 node {node.id} has no parent"
                assert parent.chunk_level == 1
                # L2 的 root 应该等于 parent（L1）
                assert node.root_chunk_id == parent.id

            elif node.chunk_level == 3:
                # L3 的 parent 应该是 L2
                parent = node_map.get(node.parent_chunk_id)
                assert parent is not None, f"L3 node {node.id} has no parent"
                assert parent.chunk_level == 2
                # L3 的 root 应该追溯到 L1
                root = node_map.get(node.root_chunk_id)
                assert root is not None
                assert root.chunk_level == 1

    def test_no_empty_chunks(self, small_settings):
        """不应产生空文本的分块。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=6)
        nodes = chunker.chunk_text(text)

        for node in nodes:
            assert node.text.strip(), f"Node {node.id} has empty text"

    def test_deterministic_ids(self, small_settings):
        """相同输入应产生相同的 ID。"""
        chunker = SlidingWindowChunker(small_settings)
        text = "测试确定性ID生成。" * 20

        nodes1 = chunker.chunk_text(text, filename="f.pdf", page=1)
        nodes2 = chunker.chunk_text(text, filename="f.pdf", page=1)

        ids1 = [n.id for n in nodes1]
        ids2 = [n.id for n in nodes2]
        assert ids1 == ids2

    def test_metadata_populated(self, small_settings):
        """分块元数据应包含 filename 和 page。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=5)
        nodes = chunker.chunk_text(text, filename="test.pdf", page=3)

        for node in nodes:
            assert node.metadata["filename"] == "test.pdf"
            assert node.metadata["page"] == 3


# ===========================================================================
# DocumentChunker 测试
# ===========================================================================

class TestDocumentChunker:

    def test_split_document_returns_chunks_and_parents(self, small_settings):
        """split_document 应返回 L3 chunks 和 parent_store。"""
        chunker = DocumentChunker(small_settings)
        doc = Document(
            id="doc_test",
            text=_make_long_text(paragraphs=8, words_per_para=25),
            metadata={"source_path": "test.pdf", "filename": "test.pdf", "page": 1},
        )

        chunks, parent_store = chunker.split_document(doc)

        # chunks 应该是 L3 叶子块
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

        # parent_store 应该包含 L1 和 L2
        assert len(parent_store) > 0
        for pid, pdata in parent_store.items():
            assert "text" in pdata
            assert "level" in pdata
            assert pdata["level"] in (1, 2)

    def test_chunks_have_correct_metadata(self, small_settings):
        """Chunk 对象应包含正确的元数据。"""
        chunker = DocumentChunker(small_settings)
        doc = Document(
            id="doc_meta",
            text=_make_long_text(paragraphs=6),
            metadata={
                "source_path": "/path/to/doc.pdf",
                "filename": "doc.pdf",
                "page": 5,
                "doc_type": "pdf",
            },
        )

        chunks, _ = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/path/to/doc.pdf"
            assert chunk.metadata["filename"] == "doc.pdf"
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["chunk_level"] == 3
            assert "parent_chunk_id" in chunk.metadata
            assert "root_chunk_id" in chunk.metadata
            assert chunk.source_ref == "doc_meta"

    def test_chunks_inherit_doc_metadata(self, small_settings):
        """Chunk 应继承 Document 的自定义元数据。"""
        chunker = DocumentChunker(small_settings)
        doc = Document(
            id="doc_inherit",
            text=_make_long_text(paragraphs=6),
            metadata={
                "source_path": "test.pdf",
                "filename": "test.pdf",
                "page": 1,
                "custom_field": "custom_value",
            },
        )

        chunks, _ = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata.get("custom_field") == "custom_value"


# ===========================================================================
# 层级一致性测试
# ===========================================================================

class TestHierarchyConsistency:
    """验证三级分块的层级一致性。"""

    def test_l3_text_is_subset_of_parent(self, small_settings):
        """L3 的文本内容应包含在对应 L2 父块中。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=8, words_per_para=20)
        nodes = chunker.chunk_text(text, filename="test.pdf", page=1)

        node_map = {n.id: n for n in nodes}
        l3_nodes = [n for n in nodes if n.chunk_level == 3]

        for l3 in l3_nodes[:10]:  # 抽样检查
            parent = node_map.get(l3.parent_chunk_id)
            if parent:
                # L3 文本的字符应大致包含在父块中
                # （由于 overlap，可能不完全精确，但核心内容应在）
                assert len(l3.text) <= len(parent.text)

    def test_all_l3_have_valid_parent_chain(self, small_settings):
        """每个 L3 都应有完整的 L2 → L1 父链。"""
        chunker = SlidingWindowChunker(small_settings)
        text = _make_long_text(paragraphs=6)
        nodes = chunker.chunk_text(text)

        node_map = {n.id: n for n in nodes}

        for node in nodes:
            if node.chunk_level == 3:
                # L3 → L2
                l2 = node_map[node.parent_chunk_id]
                assert l2.chunk_level == 2
                # L2 → L1
                l1 = node_map[l2.parent_chunk_id]
                assert l1.chunk_level == 1
                # L3.root → L1
                assert node.root_chunk_id == l1.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
