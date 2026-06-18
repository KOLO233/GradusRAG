"""全链路追踪收集器。

记录查询处理的每个阶段的状态、耗时、输入输出，用于调试、可视化和评估。
设计参考 MODULAR-RAG-MCP-SERVER 的 TraceContext。
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    """单个阶段的追踪记录。"""
    stage_name: str
    data: Dict[str, Any]
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "timestamp": self.timestamp,
            **self.data,
        }


@dataclass
class TraceContext:
    """查询级别的追踪上下文。

    使用方式:
        trace = TraceContext(query_id="q_001", query="什么是光合作用？")
        trace.record_stage("query_classification", {"level": "L1", "confidence": 0.95}, elapsed_ms=12.5)
        trace.record_stage("dense_retrieval", {"result_count": 10}, elapsed_ms=150.0)
        trace.finish()
    """
    query_id: str
    query: str
    query_level: str = ""
    stages: List[StageRecord] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: float = 0.0,
    ) -> None:
        """记录一个处理阶段。"""
        record = StageRecord(
            stage_name=stage_name,
            data=data,
            elapsed_ms=elapsed_ms,
        )
        self.stages.append(record)
        logger.debug(
            f"Trace [{self.query_id}] {stage_name}: "
            f"{elapsed_ms:.1f}ms, data_keys={list(data.keys())}"
        )

    def finish(self) -> None:
        """标记追踪结束。"""
        self.end_time = time.time()
        self.total_elapsed_ms = (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        self.finish()
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_level": self.query_level,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "stage_count": len(self.stages),
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata,
        }


class TraceCollector:
    """追踪收集器，负责存储和查询追踪记录。

    支持内存存储和文件持久化（JSONL 格式）。
    """

    def __init__(self, trace_file: Optional[str] = None):
        self._traces: Dict[str, TraceContext] = {}
        self._completed: List[TraceContext] = []
        self._trace_file = Path(trace_file) if trace_file else None

        if self._trace_file:
            self._trace_file.parent.mkdir(parents=True, exist_ok=True)

    def start_trace(self, query_id: str, query: str) -> TraceContext:
        """开始一个新的追踪。"""
        ctx = TraceContext(query_id=query_id, query=query)
        self._traces[query_id] = ctx
        return ctx

    def get_trace(self, query_id: str) -> Optional[TraceContext]:
        """获取追踪上下文。"""
        return self._traces.get(query_id)

    def complete_trace(self, query_id: str) -> Optional[Dict[str, Any]]:
        """完成追踪并持久化。"""
        ctx = self._traces.pop(query_id, None)
        if ctx is None:
            return None

        ctx.finish()
        self._completed.append(ctx)

        result = ctx.to_dict()

        if self._trace_file:
            try:
                with self._trace_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to persist trace: {e}")

        return result

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的追踪记录。"""
        return [t.to_dict() for t in self._completed[-limit:]]
