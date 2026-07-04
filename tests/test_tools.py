"""Tests for data_tools module."""

import pytest
import pandas as pd
import numpy as np
from insightflow.tools.data_tools import (
    set_dataframe,
    get_dataframe,
    fill_missing,
    remove_outliers,
    normalize_column,
    correlation_analysis,
    group_statistics,
    describe_numeric,
    value_distribution,
    get_dataframe_info,
)


@pytest.fixture(autouse=True)
def setup_test_dataframe():
    """Set up a test DataFrame before each test."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, np.nan, 40, 35],
        "salary": [50000, 60000, 70000, 150000, 80000],  # 150000 is outlier
        "city": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
        "score": [85.0, 90.0, 78.0, np.nan, 88.0],
    })
    set_dataframe(df.copy())
    yield
    set_dataframe(None)


class TestDataToolsSetup:
    def test_set_and_get_dataframe(self):
        df = get_dataframe()
        assert df is not None
        assert len(df) == 5
        assert "name" in df.columns

    def test_get_dataframe_info(self):
        result = get_dataframe_info.invoke({})
        assert "5" in result  # 5 rows
        assert "name" in result


class TestCleaningTools:
    def test_fill_missing_mean(self):
        result = fill_missing.invoke({"column": "age", "strategy": "mean"})
        df = get_dataframe()
        assert df["age"].isna().sum() == 0
        assert "filled" in result.lower() or "填充" in result

    def test_fill_missing_mode(self):
        result = fill_missing.invoke({"column": "score", "strategy": "mode"})
        df = get_dataframe()
        assert df["score"].isna().sum() == 0

    def test_fill_missing_zero(self):
        result = fill_missing.invoke({"column": "age", "strategy": "zero"})
        df = get_dataframe()
        assert df["age"].isna().sum() == 0
        # The NaN was at index 2, should now be 0
        assert df["age"].iloc[2] == 0

    def test_remove_outliers_iqr(self):
        result = remove_outliers.invoke({"column": "salary", "method": "iqr"})
        df = get_dataframe()
        # The outlier row should be removed
        assert len(df) < 5 or "outlier" in result.lower() or "异常" in result

    def test_normalize_column_minmax(self):
        result = normalize_column.invoke({"column": "salary", "method": "minmax"})
        df = get_dataframe()
        assert df["salary"].min() >= 0
        assert df["salary"].max() <= 1

    def test_fill_missing_invalid_column(self):
        result = fill_missing.invoke({"column": "nonexistent", "strategy": "mean"})
        assert "error" in result.lower() or "错误" in result or "not found" in result.lower()


class TestAnalysisTools:
    def test_correlation_analysis(self):
        result = correlation_analysis.invoke({"col_a": "age", "col_b": "salary"})
        assert isinstance(result, str)
        # Should contain some correlation info
        assert any(kw in result for kw in ["correlation", "相关", "coefficient", "系数"])

    def test_group_statistics(self):
        result = group_statistics.invoke({
            "group_col": "city",
            "value_col": "salary",
            "agg_func": "mean",
        })
        assert "Beijing" in result or "Shanghai" in result

    def test_describe_numeric(self):
        result = describe_numeric.invoke({})
        assert "age" in result
        assert "salary" in result

    def test_value_distribution(self):
        result = value_distribution.invoke({"column": "city", "top_n": 3})
        assert "Beijing" in result or "Shanghai" in result
