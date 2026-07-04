"""Tests for the MCP data explorer server tools."""

import json
import pytest
import pandas as pd
from pathlib import Path
from insightflow.data_mcp.data_explorer import (
    load_csv,
    get_schema,
    sample_rows,
    profile,
    safe_query,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.0, 92.0, 78.0, 88.0, 95.0],
        "city": ["Beijing", "Shanghai", "Beijing", "Guangzhou", "Shanghai"],
        "rating": [4.5, None, 3.8, 4.2, None],
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestLoadCsv:
    def test_basic_load(self, sample_csv):
        result = load_csv(sample_csv)
        assert "5" in result  # 5 rows
        assert "id" in result
        assert "name" in result

    def test_nonexistent_file(self):
        result = load_csv("/nonexistent/path.csv")
        assert "error" in result.lower() or "Error" in result


class TestGetSchema:
    def test_schema_info(self, sample_csv):
        result = get_schema(sample_csv)
        assert "id" in result
        assert "name" in result
        assert "score" in result


class TestSampleRows:
    def test_sample_default(self, sample_csv):
        result = sample_rows(sample_csv)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sample_n(self, sample_csv):
        result = sample_rows(sample_csv, n=3)
        assert isinstance(result, str)


class TestProfile:
    def test_profile_output(self, sample_csv):
        result = profile(sample_csv)
        assert "score" in result
        # Should contain stats info
        assert any(kw in result for kw in ["mean", "avg", "std", "min", "max"])


class TestSafeQuery:
    def test_valid_query(self, sample_csv):
        # First load the CSV
        load_csv(sample_csv)
        result = safe_query(sample_csv, "df[df['score'] > 85]")
        assert "error" not in result.lower() or "Error" not in result

    def test_blocked_mutation(self, sample_csv):
        load_csv(sample_csv)
        result = safe_query(sample_csv, "df.to_csv('evil.csv')")
        assert "error" in result.lower() or "blocked" in result.lower() or "拒绝" in result

    def test_blocked_import(self, sample_csv):
        load_csv(sample_csv)
        result = safe_query(sample_csv, "__import__('os')")
        assert "error" in result.lower() or "blocked" in result.lower() or "拒绝" in result
