"""Tests for the observability tracer module."""

import time
import pytest
from insightflow.observability.tracer import (
    PipelineTracer,
    TraceSpan,
    get_tracer,
    reset_tracer,
    trace_node,
)


@pytest.fixture(autouse=True)
def clean_tracer():
    """Reset the global tracer before and after each test."""
    reset_tracer()
    yield
    reset_tracer()


class TestTraceSpan:
    def test_basic_creation(self):
        span = TraceSpan(node_name="scout", start_time=time.time())
        assert span.node_name == "scout"
        assert span.status == "running"
        assert span.error is None

    def test_to_dict(self):
        span = TraceSpan(
            node_name="analyst",
            start_time=1000.0,
            end_time=1001.5,
            duration_ms=1500.0,
            status="success",
            input_snapshot={"data_path": "test.csv"},
            output_snapshot={"current_agent": "analyst"},
        )
        d = span.to_dict()
        assert d["node_name"] == "analyst"
        assert d["duration_ms"] == 1500.0
        assert d["status"] == "success"


class TestPipelineTracer:
    def test_start_and_finish(self):
        tracer = PipelineTracer()
        trace_id = tracer.start(config={"model": "test"})
        assert len(trace_id) == 16
        assert get_tracer() is tracer

        trace = tracer.finish()
        assert trace.trace_id == trace_id
        assert trace.total_duration_ms >= 0
        assert get_tracer() is None

    def test_add_span(self):
        tracer = PipelineTracer()
        tracer.start()
        span = TraceSpan(
            node_name="scout",
            start_time=time.time(),
            end_time=time.time() + 0.1,
            duration_ms=100.0,
            status="success",
        )
        tracer.add_span(span)
        trace = tracer.finish()
        assert len(trace.spans) == 1
        assert trace.summary["total_spans"] == 1

    def test_summary_statistics(self):
        tracer = PipelineTracer()
        tracer.start()

        for i, (name, dur) in enumerate([("scout", 100), ("analyst", 300), ("reporter", 200)]):
            tracer.add_span(TraceSpan(
                node_name=name,
                start_time=time.time(),
                end_time=time.time() + dur / 1000,
                duration_ms=float(dur),
                status="success",
            ))

        trace = tracer.finish()
        assert trace.summary["slowest_node"] == "analyst"
        assert trace.summary["fastest_node"] == "scout"
        assert trace.summary["total_agent_time_ms"] == 600.0

    def test_finish_without_start_raises(self):
        tracer = PipelineTracer()
        with pytest.raises(RuntimeError, match="not started"):
            tracer.finish()


class TestTraceNodeDecorator:
    def test_decorator_with_tracer(self):
        tracer = PipelineTracer()
        tracer.start()

        @trace_node("test_node")
        def my_node(state):
            time.sleep(0.001)
            return {"current_agent": "test_node", "messages": [{"role": "test", "content": "ok"}]}

        state = {"data_path": "test.csv", "analysis_task": "test", "iteration": 0}
        result = my_node(state)

        assert result["current_agent"] == "test_node"

        trace = tracer.finish()
        assert len(trace.spans) == 1
        assert trace.spans[0].node_name == "test_node"
        assert trace.spans[0].status == "success"
        assert trace.spans[0].duration_ms > 0

    def test_decorator_without_tracer(self):
        """When no tracer is active, decorator should be transparent."""
        @trace_node("test_node")
        def my_node(state):
            return {"current_agent": "test_node"}

        result = my_node({"data_path": "test.csv"})
        assert result["current_agent"] == "test_node"

    def test_decorator_captures_error(self):
        tracer = PipelineTracer()
        tracer.start()

        @trace_node("failing_node")
        def bad_node(state):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            bad_node({"data_path": "test.csv"})

        trace = tracer.finish()
        assert trace.spans[0].status == "error"
        assert "test error" in trace.spans[0].error

    def test_decorator_preserves_function_name(self):
        @trace_node("my_node")
        def original_func(state):
            return {}

        assert original_func.__name__ == "original_func"
