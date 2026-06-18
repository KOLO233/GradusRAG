"""测试集管理。

管理 Golden Test Set 的加载、保存和查询。
测试集格式：JSON 文件，每条包含问题、期望级别、期望答案、标注的 chunk_id 等。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.types import EvalTestCase

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "test_sets"


class TestSetManager:
    """测试集管理器。

    Example:
        >>> manager = TestSetManager()
        >>> test_set = manager.load("golden_test_set.json")
        >>> print(f"Loaded {len(test_set)} test cases")
        >>> l3_cases = manager.filter_by_level(test_set, "L3")
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_TEST_SET_PATH
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filename: str) -> List[EvalTestCase]:
        """加载测试集。"""
        path = self._data_dir / filename
        if not path.exists():
            logger.warning(f"Test set not found: {path}")
            return []

        data = json.loads(path.read_text(encoding="utf-8"))
        test_cases = []
        for item in data:
            test_cases.append(EvalTestCase(
                question=item.get("question", ""),
                expected_answer=item.get("expected_answer", ""),
                expected_level=item.get("expected_level", "L1"),
                ground_truth_chunks=item.get("ground_truth_chunks", []),
                category=item.get("category", ""),
            ))

        logger.info(f"Loaded {len(test_cases)} test cases from {filename}")
        return test_cases

    def save(self, filename: str, test_cases: List[EvalTestCase]) -> None:
        """保存测试集。"""
        path = self._data_dir / filename
        data = [tc.to_dict() for tc in test_cases]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(test_cases)} test cases to {filename}")

    def filter_by_level(
        self,
        test_cases: List[EvalTestCase],
        level: str,
    ) -> List[EvalTestCase]:
        """按级别过滤测试用例。"""
        return [tc for tc in test_cases if tc.expected_level == level]

    def filter_by_category(
        self,
        test_cases: List[EvalTestCase],
        category: str,
    ) -> List[EvalTestCase]:
        """按类别过滤测试用例。"""
        return [tc for tc in test_cases if tc.category == category]

    def get_stats(self, test_cases: List[EvalTestCase]) -> Dict:
        """获取测试集统计信息。"""
        level_counts = {}
        category_counts = {}
        for tc in test_cases:
            level_counts[tc.expected_level] = level_counts.get(tc.expected_level, 0) + 1
            if tc.category:
                category_counts[tc.category] = category_counts.get(tc.category, 0) + 1

        return {
            "total": len(test_cases),
            "by_level": level_counts,
            "by_category": category_counts,
        }

    def list_files(self) -> List[str]:
        """列出所有测试集文件。"""
        return [f.name for f in self._data_dir.glob("*.json")]
