"""Tests for the unified JSON parser (insightflow.utils.json_parser).

Covers all extraction strategies, schema validation, and pre-built schemas.
"""

import json

import pytest

from insightflow.utils.json_parser import (
    ANALYST_RESULTS_SCHEMA,
    CLEANER_PLAN_SCHEMA,
    SCOUT_PROFILE_SCHEMA,
    extract_json,
    extract_json_with_schema,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for the extract_json function."""

    def test_direct_json(self):
        """Strategy 1: direct json.loads on clean JSON."""
        data = {"key": "value", "number": 42}
        result = extract_json(json.dumps(data))
        assert result == data

    def test_json_in_fenced_block(self):
        """Strategy 2: JSON inside ```json ... ``` block."""
        inner = '{"findings": [1, 2, 3]}'
        text = f"Here is the result:\n```json\n{inner}\n```\nEnd."
        result = extract_json(text)
        assert result == {"findings": [1, 2, 3]}

    def test_json_with_surrounding_text(self):
        """Strategy 3: JSON object with surrounding prose."""
        data = {"summary": "hello"}
        text = f"The analysis shows {json.dumps(data)} as expected."
        result = extract_json(text)
        assert result["summary"] == "hello"

    def test_json_array_extraction(self):
        """Strategy 3 with expect_list: extract JSON array."""
        data = [1, 2, 3]
        text = f"Results: {json.dumps(data)}"
        result = extract_json(text, expect_list=True)
        assert result == [1, 2, 3]

    def test_default_on_failure(self):
        """Strategy 4: return default when all strategies fail."""
        result = extract_json("this is not json at all", default={"fallback": True})
        assert result == {"fallback": True}

    def test_raises_without_default(self):
        """ValueError when all strategies fail and no default provided."""
        with pytest.raises(ValueError, match="Failed to extract JSON"):
            extract_json("this is not json")

    def test_empty_string_with_default(self):
        """Empty string returns default."""
        result = extract_json("", default={"empty": True})
        assert result == {"empty": True}

    def test_none_input_with_default(self):
        """None input returns default."""
        result = extract_json(None, default={"none": True})
        assert result == {"none": True}

    def test_nested_json(self):
        """Nested JSON objects are preserved."""
        data = {"outer": {"inner": {"deep": 42}}}
        result = extract_json(json.dumps(data))
        assert result["outer"]["inner"]["deep"] == 42


# ---------------------------------------------------------------------------
# extract_json_with_schema
# ---------------------------------------------------------------------------


class TestExtractJsonWithSchema:
    """Tests for the extract_json_with_schema function."""

    def test_fills_missing_fields(self):
        """Missing fields are filled from schema defaults."""
        text = '{"summary": "test"}'
        result = extract_json_with_schema(text, ANALYST_RESULTS_SCHEMA)
        assert result["summary"] == "test"
        assert result["findings"] == []
        assert result["statistics"] == {}
        assert result["data_quality_note"] == ""

    def test_preserves_existing_fields(self):
        """Existing fields are preserved."""
        data = {
            "summary": "analysis complete",
            "findings": [{"finding": "revenue up", "confidence": "high"}],
            "statistics": {"mean": 42.0},
        }
        result = extract_json_with_schema(json.dumps(data), ANALYST_RESULTS_SCHEMA)
        assert result["summary"] == "analysis complete"
        assert len(result["findings"]) == 1
        assert result["statistics"]["mean"] == 42.0

    def test_strict_type_enforcement(self):
        """With strict=True, type mismatches are replaced with defaults."""
        data = {"summary": 123, "findings": "not a list"}
        result = extract_json_with_schema(
            json.dumps(data), ANALYST_RESULTS_SCHEMA, strict=True
        )
        assert result["summary"] == ""  # was int, expected str
        assert result["findings"] == []  # was str, expected list

    def test_empty_input_returns_defaults(self):
        """Completely empty input returns all schema defaults."""
        result = extract_json_with_schema("", CLEANER_PLAN_SCHEMA)
        assert result["strategy"] == []
        assert result["overall_notes"] == ""

    def test_scout_profile_schema(self):
        """SCOUT_PROFILE_SCHEMA fills all expected fields."""
        result = extract_json_with_schema("{}", SCOUT_PROFILE_SCHEMA)
        assert "basic_info" in result
        assert "column_details" in result
        assert "quality_issues" in result
        assert result["quality_issues"] == []


# ---------------------------------------------------------------------------
# DataFrameStore tests (in data_explorer)
# ---------------------------------------------------------------------------


class TestDataFrameStore:
    """Tests for the DataFrameStore cache."""

    def test_store_stats(self):
        """Store returns stats about cached entries."""
        from insightflow.data_mcp.data_explorer import DataFrameStore

        store = DataFrameStore(max_size=3, ttl_seconds=60)
        stats = store.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 3

    def test_store_clear(self):
        """Store.clear() removes all entries."""
        from insightflow.data_mcp.data_explorer import DataFrameStore
        import pandas as pd

        store = DataFrameStore(max_size=5, ttl_seconds=60)
        df = pd.DataFrame({"a": [1, 2, 3]})
        store._put("test", df)
        assert store.stats()["size"] == 1
        store.clear()
        assert store.stats()["size"] == 0


# ---------------------------------------------------------------------------
# Permission checker tests
# ---------------------------------------------------------------------------


class TestPermissionChecker:
    """Tests for the RBAC permission system."""

    def test_scout_role_scopes(self):
        """Scout role gets data:read and data:query scopes."""
        from insightflow.data_mcp.permissions import PermissionChecker

        checker = PermissionChecker.for_role("scout")
        assert "data:read" in checker.scopes
        assert "data:query" in checker.scopes
        assert "data:write" not in checker.scopes

    def test_check_allowed_tool(self):
        """Tools within granted scopes pass the check."""
        from insightflow.data_mcp.permissions import PermissionChecker

        checker = PermissionChecker.for_role("scout")
        assert checker.check("load_csv") is True
        assert checker.check("profile") is True
        assert checker.check("safe_query") is True

    def test_check_denied_tool(self):
        """Tools outside granted scopes are denied."""
        from insightflow.data_mcp.permissions import PermissionChecker

        checker = PermissionChecker.for_role("scout")
        assert checker.check("fill_missing") is False
        assert checker.check("remove_outliers") is False

    def test_admin_has_all_scopes(self):
        """Admin role has all scopes."""
        from insightflow.data_mcp.permissions import PermissionChecker

        checker = PermissionChecker.for_role("admin")
        assert checker.check("load_csv") is True
        assert checker.check("fill_missing") is True
        assert checker.check("remove_outliers") is True
        assert checker.check("safe_query") is True

    def test_filter_tools(self):
        """filter_tools removes tools outside the role's permissions."""
        from insightflow.data_mcp.permissions import PermissionChecker
        from unittest.mock import MagicMock

        checker = PermissionChecker.for_role("analyst")

        # Create mock tools
        tools = []
        for name in ["load_csv", "get_schema", "fill_missing", "safe_query"]:
            tool = MagicMock()
            tool.name = name
            tools.append(tool)

        filtered = checker.filter_tools(tools)
        filtered_names = [t.name for t in filtered]

        assert "load_csv" in filtered_names
        assert "get_schema" in filtered_names
        assert "safe_query" in filtered_names
        assert "fill_missing" not in filtered_names  # analyst has no data:write

    def test_reporter_has_no_tools(self):
        """Reporter role has no scopes, all tools filtered."""
        from insightflow.data_mcp.permissions import PermissionChecker
        from unittest.mock import MagicMock

        checker = PermissionChecker.for_role("reporter")
        tool = MagicMock()
        tool.name = "load_csv"
        assert checker.check("load_csv") is False


# ---------------------------------------------------------------------------
# ResilientLLMClient tests
# ---------------------------------------------------------------------------


class TestResilientLLMClient:
    """Tests for the resilient LLM client."""

    def test_successful_call(self):
        """Normal calls pass through to the underlying LLM."""
        from insightflow.llm.resilient_client import ResilientLLMClient
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "response"

        client = ResilientLLMClient(mock_llm, max_retries=3)
        result = client.invoke("messages")

        assert result == "response"
        assert client._total_calls == 1

    def test_retry_on_retryable_error(self):
        """Retryable errors trigger exponential backoff retries."""
        from insightflow.llm.resilient_client import ResilientLLMClient

        call_count = 0

        class FakeLLM:
            def invoke(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise TimeoutError("simulated timeout")
                return "success"

        client = ResilientLLMClient(FakeLLM(), max_retries=3, base_delay=0.01)
        result = client.invoke("messages")

        assert result == "success"
        assert call_count == 3

    def test_circuit_breaker_opens(self):
        """Circuit breaker opens after threshold consecutive failures."""
        from insightflow.llm.resilient_client import (
            CircuitBreakerOpenError,
            ResilientLLMClient,
        )

        class FailingLLM:
            def invoke(self, messages, **kwargs):
                raise TimeoutError("always fails")

        client = ResilientLLMClient(
            FailingLLM(),
            max_retries=1,
            base_delay=0.01,
            circuit_breaker_threshold=2,
        )

        # First call: fails, increments failure count
        with pytest.raises(Exception):
            client.invoke("msg")

        # Second call: fails again, opens circuit
        with pytest.raises(Exception):
            client.invoke("msg")

        # Third call: circuit is open
        with pytest.raises(CircuitBreakerOpenError):
            client.invoke("msg")

    def test_non_retryable_error_propagates(self):
        """Non-retryable errors propagate immediately without retrying."""
        from insightflow.llm.resilient_client import ResilientLLMClient

        class BadLLM:
            def invoke(self, messages, **kwargs):
                raise ValueError("bad input")  # Not retryable

        client = ResilientLLMClient(BadLLM(), max_retries=3, base_delay=0.01)

        with pytest.raises(ValueError, match="bad input"):
            client.invoke("messages")

        assert client._total_calls == 1  # Only one call, no retries

    def test_get_stats(self):
        """Stats tracking works correctly."""
        from insightflow.llm.resilient_client import ResilientLLMClient
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "ok"

        client = ResilientLLMClient(mock_llm)
        client.invoke("msg1")
        client.invoke("msg2")

        stats = client.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 0
        assert stats["circuit_state"] == "closed"


# ---------------------------------------------------------------------------
# TokenTracker tests
# ---------------------------------------------------------------------------


class TestTokenTracker:
    """Tests for the token/cost tracker."""

    def test_record_with_usage_metadata(self):
        """Records tokens from LangChain-style usage_metadata."""
        from insightflow.llm.token_tracker import TokenTracker
        from unittest.mock import MagicMock

        tracker = TokenTracker(model="qwen-max")

        mock_response = MagicMock()
        mock_response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        mock_response.response_metadata = {}

        delta = tracker.record("scout", mock_response)

        assert delta is not None
        assert delta.prompt_tokens == 100
        assert delta.completion_tokens == 50

        summary = tracker.get_summary()
        assert summary["scout"]["prompt_tokens"] == 100
        assert summary["scout"]["call_count"] == 1
        assert summary["total"]["total_tokens"] == 150

    def test_record_manual(self):
        """Manual token recording works."""
        from insightflow.llm.token_tracker import TokenTracker

        tracker = TokenTracker(model="qwen-max")
        tracker.record_manual("analyst", 200, 100)

        usage = tracker.get_agent_usage("analyst")
        assert usage is not None
        assert usage.total_tokens == 300

    def test_cost_estimation(self):
        """Cost is estimated based on model pricing."""
        from insightflow.llm.token_tracker import TokenTracker
        from unittest.mock import MagicMock

        tracker = TokenTracker(model="qwen-max")

        mock_response = MagicMock()
        mock_response.usage_metadata = {
            "input_tokens": 1000,
            "output_tokens": 1000,
        }
        mock_response.response_metadata = {}

        tracker.record("test", mock_response)

        cost = tracker.get_total_cost()
        assert cost > 0

    def test_no_usage_metadata_returns_none(self):
        """Responses without usage_metadata return None."""
        from insightflow.llm.token_tracker import TokenTracker
        from unittest.mock import MagicMock

        tracker = TokenTracker()
        mock_response = MagicMock(spec=[])  # No usage_metadata attribute

        delta = tracker.record("test", mock_response)
        assert delta is None

    def test_print_summary(self):
        """Summary string is properly formatted."""
        from insightflow.llm.token_tracker import TokenTracker

        tracker = TokenTracker(model="qwen-max")
        tracker.record_manual("scout", 100, 50)
        tracker.record_manual("analyst", 200, 100)

        summary = tracker.print_summary()
        assert "scout" in summary
        assert "analyst" in summary
        assert "TOTAL" in summary


# ---------------------------------------------------------------------------
# Error propagation tests
# ---------------------------------------------------------------------------


class TestErrorPropagator:
    """Tests for the structured error propagation system."""

    def test_no_errors_continues(self):
        """No errors means the pipeline continues."""
        from insightflow.errors import ErrorPropagator

        propagator = ErrorPropagator()
        should_continue, reason = propagator.should_continue([])
        assert should_continue is True

    def test_fatal_error_aborts(self):
        """Fatal errors (e.g., Scout failure) abort the pipeline."""
        from insightflow.errors import AgentError, ErrorPropagator

        propagator = ErrorPropagator()
        errors = [
            AgentError(
                agent_name="scout",
                error_type="api_error",
                message="API down",
                severity="fatal",
                recoverable=False,
            )
        ]
        should_continue, reason = propagator.should_continue(errors)
        assert should_continue is False
        assert "scout" in reason

    def test_degraded_error_continues(self):
        """Non-fatal degraded errors allow the pipeline to continue."""
        from insightflow.errors import AgentError, ErrorPropagator

        propagator = ErrorPropagator()
        errors = [
            AgentError(
                agent_name="visualizer",
                error_type="tool_error",
                message="chart failed",
                severity="degraded",
                recoverable=True,
            )
        ]
        should_continue, reason = propagator.should_continue(errors)
        assert should_continue is True

    def test_too_many_degraded_errors_aborts(self):
        """3+ degraded errors abort the pipeline."""
        from insightflow.errors import AgentError, ErrorPropagator

        propagator = ErrorPropagator()
        errors = [
            AgentError(
                agent_name=f"agent_{i}",
                error_type="runtime_error",
                message=f"failed {i}",
                severity="degraded",
            )
            for i in range(3)
        ]
        should_continue, reason = propagator.should_continue(errors)
        assert should_continue is False

    def test_health_status(self):
        """Health status correctly reflects error state."""
        from insightflow.errors import AgentError, ErrorPropagator

        propagator = ErrorPropagator()

        assert propagator.get_health_status([]) == "healthy"

        assert propagator.get_health_status([
            AgentError("visualizer", "err", "msg", severity="degraded"),
        ]) == "degraded"

        assert propagator.get_health_status([
            AgentError("scout", "err", "msg", severity="fatal"),
        ]) == "fatal"

    def test_fallback_context(self):
        """Fallback contexts provide sensible defaults."""
        from insightflow.errors import ErrorPropagator

        propagator = ErrorPropagator()

        fallback = propagator.get_fallback_context("visualizer")
        assert "charts" in fallback
        assert fallback["charts"] == []

        fallback = propagator.get_fallback_context("cleaner_execute")
        assert "cleaning_plan" in fallback

    def test_agent_error_from_exception(self):
        """AgentError.from_exception auto-classifies severity."""
        from insightflow.errors import AgentError

        err = AgentError.from_exception("scout", RuntimeError("boom"))
        assert err.severity == "fatal"
        assert err.recoverable is False

        err2 = AgentError.from_exception("visualizer", RuntimeError("oops"))
        assert err2.severity == "degraded"
        assert err2.recoverable is True


# ---------------------------------------------------------------------------
# DataFrameContext tests
# ---------------------------------------------------------------------------


class TestDataFrameContext:
    """Tests for the session-scoped DataFrame context."""

    def test_load_and_access(self):
        """Load a DataFrame and access it."""
        import pandas as pd
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ctx.load(df, label="test")

        assert ctx.has_data
        assert ctx.shape == (3, 2)
        assert ctx.version == 1

    def test_apply_creates_version(self):
        """Applying a modification creates a new version."""
        import pandas as pd
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx.load(df)

        ctx.apply("double", lambda d: d * 2)
        assert ctx.version == 2

    def test_rollback(self):
        """Rollback reverts to a previous version."""
        import pandas as pd
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx.load(df, label="original")

        ctx.apply("modify", lambda d: d.assign(a=[10, 20, 30]))
        assert ctx.df["a"].iloc[0] == 10

        ctx.rollback()
        assert ctx.version == 1
        assert ctx.df["a"].iloc[0] == 1

    def test_history(self):
        """Modification history is tracked."""
        import pandas as pd
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        df = pd.DataFrame({"a": [1, 2, 3]})
        ctx.load(df, label="initial")
        ctx.apply("step1", lambda d: d)
        ctx.apply("step2", lambda d: d)

        history = ctx.get_history()
        assert len(history) == 3
        assert history[0]["label"] == "initial"
        assert history[1]["label"] == "step1"
        assert history[2]["label"] == "step2"

    def test_quick_profile(self):
        """Quick profile generates basic stats without LLM."""
        import pandas as pd
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ctx.load(df)

        profile = ctx.quick_profile()
        assert "shape" in profile
        assert "columns" in profile
        assert "a" in profile["columns"]
        assert profile["columns"]["a"]["mean"] == 2.0

    def test_no_data_raises(self):
        """Accessing df without loading raises ValueError."""
        from insightflow.context.dataframe_context import DataFrameContext

        ctx = DataFrameContext()
        with pytest.raises(ValueError, match="No DataFrame loaded"):
            _ = ctx.df
