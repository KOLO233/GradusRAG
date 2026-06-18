"""GradusRAG RRF 融合算法测试。"""

import pytest
from src.core.types import RetrievalResult
from src.retrieval.fusion import rrf_fuse, weighted_rrf_fuse, rrf_score


def _make_result(chunk_id: str, score: float = 0.0, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"Text for {chunk_id}",
        metadata={"source_path": f"doc_{chunk_id}.pdf"},
    )


class TestRRFScore:
    """RRF 单项分数计算测试。"""

    def test_top_rank(self):
        """排名第一的文档应获得最高 RRF 分数。"""
        assert rrf_score(1, k=60) == pytest.approx(1.0 / 61)

    def test_decreasing_scores(self):
        """排名越高分数越高。"""
        s1 = rrf_score(1, k=60)
        s2 = rrf_score(2, k=60)
        s3 = rrf_score(3, k=60)
        assert s1 > s2 > s3

    def test_custom_k(self):
        """不同 k 值应产生不同分数。"""
        assert rrf_score(1, k=20) > rrf_score(1, k=60)


class TestRRFFuse:
    """RRF 融合测试。"""

    def test_empty_lists(self):
        """空列表融合应返回空结果。"""
        assert rrf_fuse([]) == []

    def test_single_list(self):
        """单列表融合应直接返回该列表。"""
        results = [_make_result("a", 0.9), _make_result("b", 0.8)]
        fused = rrf_fuse([results])
        assert len(fused) == 2

    def test_basic_fusion(self):
        """两个列表融合应正确合并。"""
        dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
        sparse = [_make_result("b", 5.0), _make_result("c", 4.0)]

        fused = rrf_fuse([dense, sparse])
        chunk_ids = [r.chunk_id for r in fused]

        # b 在两个列表中都出现，应排在最前
        assert chunk_ids[0] == "b"
        assert len(fused) == 3

    def test_top_k(self):
        """top_k 参数应限制返回数量。"""
        dense = [_make_result(f"d{i}", 0.9 - i * 0.1) for i in range(10)]
        sparse = [_make_result(f"s{i}", 10 - i) for i in range(10)]

        fused = rrf_fuse([dense, sparse], top_k=5)
        assert len(fused) == 5

    def test_deterministic(self):
        """相同输入应产生相同输出。"""
        dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
        sparse = [_make_result("b", 5.0), _make_result("c", 4.0)]

        r1 = rrf_fuse([dense, sparse])
        r2 = rrf_fuse([dense, sparse])
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


class TestWeightedRRFFuse:
    """加权 RRF 融合测试。"""

    def test_weight_increases_ranking(self):
        """增加权重应提升该列表中文档的排名。

        数据分析 (k=60):
        等权: a = 1/61 = 0.0164, b = 1/62 + 1/62 = 0.0323, c = 1/61 = 0.0164
        加权 dense=10: a = 10/61 = 0.164, b = 10/62 + 1/62 = 0.177, c = 1/61 = 0.0164
        加权 dense=15: a = 15/61 = 0.246, b = 15/62 + 1/62 = 0.258 → b 仍赢
        需要让 a(dense第一) 的权重足够大，使其超过 b(两路都有) 的累积分数。
        """
        dense = [_make_result("a", 0.9), _make_result("b", 0.8)]
        sparse = [_make_result("c", 5.0), _make_result("b", 4.0)]

        # 等权时 b 排第一（两路累积优势）
        fused_equal = weighted_rrf_fuse([dense, sparse], weights=[1.0, 1.0])
        assert fused_equal[0].chunk_id == "b"

        # 高权重 dense 时 a 应超过 b
        # a: 50/61 = 0.820, b: 50/62 + 1/62 = 0.823 → 差距极小
        # 用 100: a: 100/61=1.639, b: 100/62+1/62=1.629 → a 赢
        fused_weighted = weighted_rrf_fuse([dense, sparse], weights=[100.0, 1.0])
        assert fused_weighted[0].chunk_id == "a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
