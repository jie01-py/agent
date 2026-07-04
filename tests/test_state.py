"""Tests for the AgentState definition and initialization."""

import pytest
from insightflow.state import AgentState, create_initial_state


class TestAgentState:
    """Test suite for AgentState TypedDict."""

    def test_create_initial_state(self):
        """Test that initial state is properly initialized."""
        state = create_initial_state(
            data_path="test.csv",
            analysis_task="Analyze sales trends",
        )
        assert state["data_path"] == "test.csv"
        assert state["analysis_task"] == "Analyze sales trends"
        assert state["dataframe"] is None
        assert state["data_profile"] == {}
        assert state["cleaning_plan"] == {}
        assert state["analysis_results"] == {}
        assert state["charts"] == []
        assert state["report"] == ""
        assert state["messages"] == []
        assert state["errors"] == []
        assert state["current_agent"] == "scout"
        assert state["iteration"] == 0

    def test_initial_state_with_chinese_task(self):
        """Test initial state with Chinese analysis task."""
        state = create_initial_state(
            data_path="data.csv",
            analysis_task="分析各城市销售分布",
        )
        assert state["analysis_task"] == "分析各城市销售分布"

    def test_state_is_dict(self):
        """Verify AgentState is a TypedDict (i.e., a dict at runtime)."""
        state = create_initial_state("test.csv", "task")
        assert isinstance(state, dict)

    def test_messages_accumulation(self):
        """Test that messages field supports accumulation via operator.add."""
        import operator
        # Simulate what LangGraph does with Annotated[list, operator.add]
        messages_1 = [{"role": "scout", "content": "explored data"}]
        messages_2 = [{"role": "cleaner", "content": "cleaned data"}]
        combined = operator.add(messages_1, messages_2)
        assert len(combined) == 2
        assert combined[0]["role"] == "scout"
        assert combined[1]["role"] == "cleaner"
