"""InsightFlow 执行追踪。

为 Agent 节点执行提供结构化追踪，包括计时、输入输出快照和错误捕获。
使用装饰器模式，与 LangGraph 节点函数透明集成。

核心概念:
- TraceSpan: 单个 Agent 节点的执行记录（时间、输入输出快照、状态）
- PipelineTrace: 一次完整 InsightFlow 执行的追踪记录（包含多个 TraceSpan）
- PipelineTracer: 追踪收集器（管理 TraceSpan 的生命周期）
- @trace_node: 装饰器，自动包裹 Agent 节点函数并记录执行数据
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class TraceSpan:
    """单个 Agent 节点执行的追踪记录。

    Attributes:
        node_name: Agent 节点名称（如 "scout", "analyst"）
        start_time: 执行开始时的 Unix 时间戳
        end_time: 执行结束时的 Unix 时间戳
        duration_ms: 执行耗时（毫秒）
        status: "success" 或 "error"
        input_snapshot: 关键输入状态字段的摘要
        output_snapshot: 关键输出状态字段的摘要
        error: 若 status 为 "error" 时的错误信息
        metadata: 额外元数据（tool 调用计数等）
    """

    node_name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "running"
    input_snapshot: dict[str, Any] = field(default_factory=dict)
    output_snapshot: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化的 dict。"""
        return {
            "node_name": self.node_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "input_snapshot": self.input_snapshot,
            "output_snapshot": self.output_snapshot,
            "error": self.error,
            "metadata": self.metadata,
            "token_usage": self.token_usage,
        }

    def to_otlp(self, trace_id: str) -> dict[str, Any]:
        """转换为 OpenTelemetry span 格式（OTLP JSON）。

        允许在 Jaeger、Grafana Tempo 或任何 OTLP 兼容后端中
        可视化 InsightFlow 的执行过程。

        Args:
            trace_id: InsightFlow 级别的追踪标识。

        Returns:
            符合 OTLP span schema 的 dict。
        """
        import uuid as _uuid

        return {
            "traceId": trace_id,
            "spanId": _uuid.uuid4().hex[:16],
            "parentSpanId": "",
            "name": f"insightflow.{self.node_name}",
            "kind": "SERVER",
            "startTimeUnixNano": int(self.start_time * 1e9),
            "endTimeUnixNano": int(self.end_time * 1e9),
            "attributes": {
                "agent.name": self.node_name,
                "agent.status": self.status,
                "agent.duration_ms": round(self.duration_ms, 2),
                **{
                    f"input.{k}": str(v)
                    for k, v in self.input_snapshot.items()
                },
                **{
                    f"output.{k}": str(v)
                    for k, v in self.output_snapshot.items()
                },
                **{
                    f"token.{k}": v
                    for k, v in self.token_usage.items()
                },
            },
            "status": {
                "code": "OK" if self.status == "success" else "ERROR",
                "message": self.error or "",
            },
            "events": [],
        }


@dataclass
class PipelineTrace:
    """一次完整 InsightFlow 执行的追踪记录。

    Attributes:
        trace_id: 此追踪的唯一标识
        pipeline_start: InsightFlow 开始时的 Unix 时间戳
        pipeline_end: InsightFlow 结束时的 Unix 时间戳
        total_duration_ms: 总耗时（毫秒）
        spans: 各 Agent 节点的 span 列表
        config: InsightFlow 配置快照
        summary: 汇总统计
    """

    trace_id: str
    pipeline_start: float = 0.0
    pipeline_end: float = 0.0
    total_duration_ms: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化的 dict。"""
        return {
            "trace_id": self.trace_id,
            "pipeline_start": self.pipeline_start,
            "pipeline_end": self.pipeline_end,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "config": self.config,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Tracer 单例
# ---------------------------------------------------------------------------

_active_tracer: PipelineTracer | None = None


class PipelineTracer:
    """收集和管理一次 InsightFlow 执行中的所有 TraceSpan。

    用法:
        tracer = PipelineTracer()
        tracer.start(config={"model": "gpt-4o-mini"})
        # ... Agent 节点执行，@trace_node 自动记录 span ...
        tracer.finish()
        trace = tracer.get_trace()
    """

    def __init__(self) -> None:
        self._trace: PipelineTrace | None = None
        self._spans: list[TraceSpan] = []

    def start(self, config: dict[str, Any] | None = None) -> str:
        """开始新的 InsightFlow 追踪。

        Args:
            config: 可选的配置快照。

        Returns:
            此次执行的 trace_id。
        """
        global _active_tracer
        _active_tracer = self

        trace_id = uuid.uuid4().hex[:16]
        self._trace = PipelineTrace(
            trace_id=trace_id,
            pipeline_start=time.time(),
            config=config or {},
        )
        self._spans = []
        logger.info("Pipeline trace started: %s", trace_id)
        return trace_id

    def add_span(self, span: TraceSpan) -> None:
        """记录一个已完成的 trace span。

        Args:
            span: 要记录的 TraceSpan。
        """
        self._spans.append(span)
        logger.info(
            "Trace span: %s | %.0fms | %s",
            span.node_name,
            span.duration_ms,
            span.status,
        )

    def finish(self) -> PipelineTrace:
        """结束 InsightFlow 追踪并计算汇总统计。

        Returns:
            已完成并计算好 summary 的 PipelineTrace。
        """
        if self._trace is None:
            raise RuntimeError("Tracer was not started. Call start() first.")

        self._trace.pipeline_end = time.time()
        self._trace.total_duration_ms = (
            (self._trace.pipeline_end - self._trace.pipeline_start) * 1000
        )
        self._trace.spans = list(self._spans)

        # Compute summary statistics
        durations = [s.duration_ms for s in self._spans]
        statuses = [s.status for s in self._spans]

        self._trace.summary = {
            "total_spans": len(self._spans),
            "successful_spans": statuses.count("success"),
            "failed_spans": statuses.count("error"),
            "total_agent_time_ms": round(sum(durations), 2),
            "avg_span_duration_ms": round(
                sum(durations) / len(durations), 2
            ) if durations else 0,
            "slowest_node": max(
                self._spans, key=lambda s: s.duration_ms
            ).node_name if self._spans else None,
            "fastest_node": min(
                self._spans, key=lambda s: s.duration_ms
            ).node_name if self._spans else None,
            "node_durations": {
                s.node_name: round(s.duration_ms, 2) for s in self._spans
            },
        }

        global _active_tracer
        _active_tracer = None

        logger.info(
            "Pipeline trace finished: %s | %.0fms total | %d spans",
            self._trace.trace_id,
            self._trace.total_duration_ms,
            len(self._spans),
        )

        return self._trace

    def get_trace(self) -> PipelineTrace | None:
        """Return the current trace (may be None if not started)."""
        return self._trace


def get_tracer() -> PipelineTracer | None:
    """Return the currently active PipelineTracer, or None."""
    return _active_tracer


def reset_tracer() -> None:
    """Reset the global tracer singleton (useful for testing)."""
    global _active_tracer
    _active_tracer = None


# ---------------------------------------------------------------------------
# State snapshot helpers
# ---------------------------------------------------------------------------

# Fields to include in snapshots (exclude large objects like DataFrame)
_SNAPSHOT_FIELDS = [
    "data_path",
    "analysis_task",
    "current_agent",
    "iteration",
]

_OUTPUT_FIELDS = [
    "current_agent",
    "iteration",
]


def _snapshot_state(state: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Create a lightweight snapshot of the state dict.

    Only includes specified fields and summarizes large objects.

    Args:
        state: The full AgentState dict.
        fields: List of field names to include.

    Returns:
        A serializable dict with the selected fields.
    """
    snap: dict[str, Any] = {}
    for key in fields:
        value = state.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            snap[key] = value
        elif isinstance(value, dict):
            snap[key] = {"_type": "dict", "_keys": list(value.keys()), "_len": len(value)}
        elif isinstance(value, list):
            snap[key] = {"_type": "list", "_len": len(value)}
        else:
            snap[key] = {"_type": type(value).__name__, "_repr": repr(value)[:100]}

    # Always note if dataframe is present
    df = state.get("dataframe")
    if df is not None:
        try:
            snap["_dataframe_shape"] = list(df.shape)
        except Exception:
            snap["_dataframe_shape"] = "unknown"

    return snap


# ---------------------------------------------------------------------------
# @trace_node decorator
# ---------------------------------------------------------------------------


def trace_node(node_name: str) -> Callable:
    """Decorator that wraps a LangGraph node function with tracing.

    Automatically captures execution timing, input/output state snapshots,
    and error information. The wrapped function behaves identically to the
    original — the decorator is fully transparent.

    装饰器：为 LangGraph 节点函数添加执行追踪。
    自动捕获执行时间、输入输出状态快照和错误信息。
    装饰后的函数行为与原函数完全一致（透明包裹）。

    Args:
        node_name: Name to identify this node in traces (e.g., "scout").

    Returns:
        A decorator function.

    Usage:
        @trace_node("scout")
        def scout_node(state: AgentState) -> dict:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            tracer = get_tracer()

            # If no tracer is active, just call the function directly
            if tracer is None:
                return func(state, *args, **kwargs)

            start_time = time.time()
            input_snap = _snapshot_state(state, _SNAPSHOT_FIELDS)

            span = TraceSpan(
                node_name=node_name,
                start_time=start_time,
                input_snapshot=input_snap,
            )

            try:
                result = func(state, *args, **kwargs)

                end_time = time.time()
                span.end_time = end_time
                span.duration_ms = (end_time - start_time) * 1000
                span.status = "success"

                # Build output snapshot from the partial state returned
                if isinstance(result, dict):
                    output_snap: dict[str, Any] = {}
                    for key in _OUTPUT_FIELDS:
                        if key in result:
                            val = result[key]
                            if isinstance(val, (str, int, float, bool)):
                                output_snap[key] = val
                            else:
                                output_snap[key] = {"_type": type(val).__name__}

                    # Count messages returned
                    msgs = result.get("messages", [])
                    output_snap["_messages_returned"] = len(msgs) if isinstance(msgs, list) else 0

                    # Count errors returned
                    errs = result.get("errors", [])
                    output_snap["_errors_returned"] = len(errs) if isinstance(errs, list) else 0

                    # Note specific output fields
                    for key in (
                        "data_profile",
                        "cleaning_plan",
                        "analysis_results",
                        "charts",
                        "report",
                    ):
                        if key in result:
                            val = result[key]
                            if isinstance(val, dict):
                                output_snap[key] = {
                                    "_type": "dict",
                                    "_keys": list(val.keys()),
                                }
                            elif isinstance(val, list):
                                output_snap[key] = {"_type": "list", "_len": len(val)}
                            elif isinstance(val, str):
                                output_snap[key] = {"_type": "str", "_len": len(val)}

                    span.output_snapshot = output_snap

                    # Check if the agent itself reported errors
                    if result.get("errors"):
                        span.metadata["agent_errors"] = result["errors"]

                tracer.add_span(span)
                return result

            except Exception as exc:
                end_time = time.time()
                span.end_time = end_time
                span.duration_ms = (end_time - start_time) * 1000
                span.status = "error"
                span.error = str(exc)
                tracer.add_span(span)
                raise  # Re-raise so the original error handling works

        return wrapper

    return decorator
