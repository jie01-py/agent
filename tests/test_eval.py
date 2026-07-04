"""Tests for the evaluation metrics module."""

import os
import pytest
from insightflow.eval.metrics import (
    AgentScore,
    evaluate_analysis,
    evaluate_charts,
    evaluate_cleaning,
    evaluate_pipeline,
    evaluate_profile,
    evaluate_report,
)


class TestEvaluateProfile:
    """Test suite for Scout agent profile evaluation."""

    def test_empty_profile(self):
        score = evaluate_profile({})
        assert score.agent_name == "scout"
        assert score.overall_score == 0.0

    def test_error_profile(self):
        score = evaluate_profile({"error": "something went wrong"})
        assert score.overall_score == 0.0

    def test_complete_profile(self):
        profile = {
            "basic_info": {"rows": 200, "columns": 11},
            "column_details": {"col1": {"dtype": "int64"}},
            "numeric_summary": {"price": {"mean": 100, "std": 50}, "qty": {"mean": 3}},
            "quality_issues": ["missing values in rating", "outliers in price"],
            "categorical_summary": {"city": {"Beijing": 30, "Shanghai": 25}},
        }
        score = evaluate_profile(profile)
        assert score.overall_score >= 0.7
        assert score.dimensions["field_coverage"] > 0.7
        assert score.dimensions["quality_issues_found"] > 0.5

    def test_partial_profile(self):
        profile = {
            "basic_info": {"rows": 100},
        }
        score = evaluate_profile(profile)
        assert 0.0 < score.overall_score < 0.7

    def test_mcp_style_profile(self):
        """Test with profile format from the MCP data_explorer tool."""
        profile = {
            "descriptive_stats": {"price": {"mean": 100}, "qty": {"mean": 3}},
            "missing_values": {"rating": {"null_count": 30, "null_pct": 15.0}},
            "categorical_distributions": {"city": {"Beijing": 30}},
        }
        score = evaluate_profile(profile)
        assert score.overall_score > 0.4
        assert score.dimensions["numeric_stats_present"] > 0.5


class TestEvaluateCleaning:
    """Test suite for Cleaner agent cleaning plan evaluation."""

    def test_empty_plan(self):
        score = evaluate_cleaning({})
        assert score.overall_score == 0.0

    def test_good_plan(self):
        plan = {
            "strategy": [
                {"column": "rating", "action": "fill_missing",
                 "params": {"strategy": "mean"}, "reason": "15% missing"},
                {"column": "price", "action": "remove_outliers",
                 "params": {"method": "iqr"}, "reason": "extreme values"},
                {"column": "discount", "action": "fill_missing",
                 "params": {"strategy": "zero"}, "reason": "10% missing"},
            ],
            "overall_notes": "Standard cleaning for e-commerce data",
        }
        score = evaluate_cleaning(plan)
        assert score.overall_score > 0.7
        assert score.dimensions["has_strategy"] >= 1.0
        assert score.dimensions["actions_valid"] == 1.0

    def test_invalid_actions(self):
        plan = {
            "strategy": [
                {"column": "x", "action": "unknown_action", "params": {}, "reason": "test"},
            ],
        }
        score = evaluate_cleaning(plan)
        assert score.dimensions["actions_valid"] == 0.0

    def test_cross_reference_with_profile(self):
        plan = {
            "strategy": [
                {"column": "rating", "action": "fill_missing",
                 "params": {"strategy": "mean"}, "reason": "missing"},
            ],
        }
        profile = {
            "missing_values": {"rating": {"null_count": 30, "null_pct": 15.0}},
        }
        score = evaluate_cleaning(plan, data_profile=profile)
        assert score.dimensions["addresses_issues"] > 0.5


class TestEvaluateAnalysis:
    """Test suite for Analyst agent analysis results evaluation."""

    def test_empty_results(self):
        score = evaluate_analysis({})
        assert score.overall_score == 0.0

    def test_failed_analysis(self):
        score = evaluate_analysis({"summary": "分析失败: timeout"})
        assert score.overall_score == 0.0

    def test_good_results(self):
        results = {
            "summary": "电商销售数据显示，服装鞋帽类产品销售量最高，北京和上海是主要销售城市",
            "findings": [
                {"finding": "服装类销量领先", "evidence": "平均订单量 3.2 件/单", "confidence": "high"},
                {"finding": "价格与评分正相关", "evidence": "Pearson r = 0.42", "confidence": "medium"},
                {"finding": "支付宝是最常用支付方式", "evidence": "占比 35%", "confidence": "high"},
            ],
            "statistics": {"total_revenue": 150000, "avg_order": 250},
            "data_quality_note": None,
        }
        score = evaluate_analysis(results)
        assert score.overall_score > 0.7
        assert score.dimensions["findings_count"] > 0.5
        assert score.dimensions["evidence_quality"] > 0.5
        assert score.dimensions["structure"] >= 0.75


class TestEvaluateCharts:
    """Test suite for Visualizer agent chart evaluation."""

    def test_no_charts(self):
        score = evaluate_charts([])
        assert score.overall_score == 0.0

    def test_charts_nonexistent_files(self):
        score = evaluate_charts(["bar_chart.png", "line_trend.png", "pie_category.png"])
        assert score.dimensions["chart_count"] == 1.0  # 3 charts is ideal
        assert score.dimensions["files_exist"] == 0.0  # files don't exist
        assert score.dimensions["type_diversity"] > 0.5

    def test_charts_with_existing_files(self, tmp_path):
        # Create dummy chart files
        for name in ["bar_sales.png", "line_trend.png", "scatter_price.png"]:
            (tmp_path / name).write_text("dummy")
        paths = [str(tmp_path / name) for name in
                 ["bar_sales.png", "line_trend.png", "scatter_price.png"]]
        score = evaluate_charts(paths)
        assert score.dimensions["files_exist"] == 1.0
        assert score.overall_score > 0.8


class TestEvaluateReport:
    """Test suite for Reporter agent report evaluation."""

    def test_empty_report(self):
        score = evaluate_report("")
        assert score.overall_score == 0.0

    def test_failed_report(self):
        score = evaluate_report("# 报告生成失败\n\n生成报告时发生错误: timeout")
        assert score.overall_score == 0.0

    def test_good_report(self):
        report = """# 数据分析报告：电商销售分析

## 1. 概述
本报告分析了200条电商订单数据，重点关注产品类别、城市分布和价格特征。

## 2. 数据概况
数据包含11列，200行记录。

## 3. 数据清洗
对rating列执行均值填充，对price列执行IQR异常值移除。

## 4. 分析发现
服装鞋帽类产品销售量最高，占总销售额的35.2%。

| 类别 | 平均价格 | 销售量 |
|------|---------|--------|
| 服装 | ¥250 | 800 |

价格与评分的相关系数为0.42，呈中度正相关。

## 5. 可视化
![图表1](bar_sales.png)
![图表2](line_trend.png)

## 6. 结论与建议
建议重点关注服装和电子产品类别。
"""
        score = evaluate_report(report, charts=["bar_sales.png", "line_trend.png"])
        assert score.overall_score > 0.6
        assert score.dimensions["structure_complete"] > 0.8
        assert score.dimensions["chart_references"] > 0.5


class TestEvaluatePipeline:
    """Test suite for pipeline-level evaluation."""

    def test_empty_scores(self):
        result = evaluate_pipeline([])
        assert result["pipeline_score"] == 0.0

    def test_good_pipeline(self):
        scores = [
            AgentScore("scout", 0.85, {"field_coverage": 0.9}, "good"),
            AgentScore("cleaner", 0.80, {"has_strategy": 0.8}, "good"),
            AgentScore("analyst", 0.75, {"findings_count": 0.8}, "good"),
            AgentScore("visualizer", 0.70, {"chart_count": 0.7}, "good"),
            AgentScore("reporter", 0.80, {"structure_complete": 0.9}, "good"),
        ]
        result = evaluate_pipeline(scores)
        assert result["pipeline_score"] > 0.7
        assert result["grade"] in ("A", "B", "C")
        assert result["agent_count"] == 5

    def test_with_errors(self):
        scores = [
            AgentScore("scout", 0.3, {}, "bad"),
            AgentScore("cleaner", 0.0, {}, "error"),
        ]
        result = evaluate_pipeline(scores, errors=["[scout] timeout", "[cleaner] failed"])
        assert result["pipeline_score"] < 0.3
        assert result["grade"] == "F"


class TestAgentScore:
    """Test suite for AgentScore dataclass."""

    def test_grade_property(self):
        assert AgentScore("test", 0.95).grade == "A"
        assert AgentScore("test", 0.85).grade == "B"
        assert AgentScore("test", 0.75).grade == "C"
        assert AgentScore("test", 0.65).grade == "D"
        assert AgentScore("test", 0.50).grade == "F"

    def test_to_dict(self):
        score = AgentScore("scout", 0.85, {"coverage": 0.9}, "good profile")
        d = score.to_dict()
        assert d["agent_name"] == "scout"
        assert d["overall_score"] == 0.85
        assert "coverage" in d["dimensions"]
