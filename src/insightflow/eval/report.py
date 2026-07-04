"""InsightFlow 评估报告生成。

运行所有 Agent 级别的评估，汇总为统一的评估报告。

报告包含:
- 各 Agent 的评分和等级
- 整体 InsightFlow 评分
- 改进建议
- 导出为 JSON 或 Markdown 格式
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insightflow.eval.metrics import (
    AgentScore,
    evaluate_analysis,
    evaluate_charts,
    evaluate_cleaning,
    evaluate_pipeline,
    evaluate_profile,
    evaluate_report,
)


@dataclass
class EvaluationReport:
    """一次 InsightFlow 执行的完整评估报告。

    Attributes:
        scores: 各 Agent 评分列表
        pipeline_metrics: InsightFlow 级别指标 dict
        recommendations: 改进建议列表
        timestamp: 评估执行时间
    """

    scores: list[AgentScore] = field(default_factory=list)
    pipeline_metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化的 dict。"""
        return {
            "timestamp": self.timestamp,
            "scores": [s.to_dict() for s in self.scores],
            "pipeline_metrics": self.pipeline_metrics,
            "recommendations": self.recommendations,
        }


def _generate_recommendations(scores: list[AgentScore]) -> list[str]:
    """Generate improvement recommendations based on agent scores.

    根据各 Agent 评分生成改进建议。

    Args:
        scores: List of AgentScore objects.

    Returns:
        A list of recommendation strings (in Chinese).
    """
    recommendations: list[str] = []

    for score in scores:
        if score.overall_score >= 0.8:
            continue

        name = score.agent_name

        if name == "scout":
            if score.dimensions.get("field_coverage", 1.0) < 0.6:
                recommendations.append(
                    "Scout: 数据画像字段覆盖不足，建议优化 Scout Agent 的 System Prompt，"
                    "明确要求输出 basic_info、quality_issues 等必要字段"
                )
            if score.dimensions.get("quality_issues_found", 1.0) < 0.5:
                recommendations.append(
                    "Scout: 质量问题识别不充分，建议在数据探索工具中增加自动化的异常检测逻辑"
                )

        elif name == "cleaner":
            if score.dimensions.get("has_strategy", 1.0) < 0.5:
                recommendations.append(
                    "Cleaner: 清洗策略生成不完整，建议提供更详细的数据画像作为清洗决策的输入"
                )
            if score.dimensions.get("addresses_issues", 1.0) < 0.5:
                recommendations.append(
                    "Cleaner: 清洗计划未充分针对已知质量问题，建议在 Prompt 中强调优先处理缺失值和异常值"
                )

        elif name == "analyst":
            if score.dimensions.get("findings_count", 1.0) < 0.5:
                recommendations.append(
                    "Analyst: 分析发现数量不足，建议增加迭代次数或扩展工具集以支持更深入的分析"
                )
            if score.dimensions.get("evidence_quality", 1.0) < 0.5:
                recommendations.append(
                    "Analyst: 分析结论缺少数据支撑，建议在 System Prompt 中要求每个 finding 必须附带 evidence"
                )

        elif name == "visualizer":
            if score.dimensions.get("chart_count", 1.0) < 0.5:
                recommendations.append(
                    "Visualizer: 图表数量不足，建议在 Prompt 中明确要求生成 3-5 张图表"
                )
            if score.dimensions.get("files_exist", 1.0) < 0.8:
                recommendations.append(
                    "Visualizer: 部分图表文件未成功保存，请检查输出目录权限和磁盘空间"
                )

        elif name == "reporter":
            if score.dimensions.get("structure_complete", 1.0) < 0.6:
                recommendations.append(
                    "Reporter: 报告结构不完整，建议在 System Prompt 中更明确地定义各章节标题"
                )
            if score.dimensions.get("data_richness", 1.0) < 0.5:
                recommendations.append(
                    "Reporter: 报告缺少数据支撑，建议要求报告中包含具体的数值和百分比"
                )

    if not recommendations:
        recommendations.append("所有 Agent 表现良好，无需额外优化")

    return recommendations


def generate_evaluation_report(state: dict[str, Any]) -> EvaluationReport:
    """从最终的 InsightFlow 状态生成完整的评估报告。

    这是主要入口点。运行所有 Agent 级别的评估，
    计算 InsightFlow 指标，生成改进建议。

    Args:
        state: InsightFlow 执行后的最终 AgentState dict。

    Returns:
        完整的 EvaluationReport。
    """
    import time as _time

    scores: list[AgentScore] = []

    # 评估各 Agent 的输出
    scores.append(evaluate_profile(state.get("data_profile", {})))
    scores.append(evaluate_cleaning(
        state.get("cleaning_plan", {}),
        state.get("data_profile"),
    ))
    scores.append(evaluate_analysis(state.get("analysis_results", {})))
    scores.append(evaluate_charts(state.get("charts", [])))
    scores.append(evaluate_report(
        state.get("report", ""),
        state.get("charts"),
    ))

    # 计算 InsightFlow 级别指标
    pipeline_metrics = evaluate_pipeline(
        agent_scores=scores,
        errors=state.get("errors", []),
    )

    # 生成改进建议
    recommendations = _generate_recommendations(scores)

    return EvaluationReport(
        scores=scores,
        pipeline_metrics=pipeline_metrics,
        recommendations=recommendations,
        timestamp=_time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def export_evaluation_json(report: EvaluationReport, output_path: str) -> str:
    """将评估报告导出为 JSON 文件。

    Args:
        report: 要导出的 EvaluationReport。
        output_path: JSON 文件路径。

    Returns:
        输出文件路径。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def export_evaluation_markdown(report: EvaluationReport, output_path: str) -> str:
    """将评估报告导出为 Markdown 文档。

    Markdown 报告包含:
    - 整体 InsightFlow 评分和等级
    - 各 Agent 评分表（含维度）
    - 各 Agent 详细评价
    - 改进建议

    Args:
        report: 要导出的 EvaluationReport。
        output_path: Markdown 文件路径。

    Returns:
        输出文件路径。
    """
    lines: list[str] = []

    # 标题
    lines.append("# InsightFlow Pipeline 评估报告")
    lines.append("")
    lines.append(f"> 评估时间: {report.timestamp}")
    lines.append("")

    # InsightFlow 概览
    pm = report.pipeline_metrics
    grade_emoji = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(pm.get("grade", "F"), "⚪")
    lines.append("## 整体评估")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| InsightFlow 评分 | **{pm.get('pipeline_score', 0):.1%}** {grade_emoji} |")
    lines.append(f"| 评级 | **{pm.get('grade', 'N/A')}** |")
    lines.append(f"| Agent 平均分 | {pm.get('avg_agent_score', 0):.1%} |")
    lines.append(f"| 评估 Agent 数 | {pm.get('agent_count', 0)} |")
    lines.append(f"| 错误率 | {pm.get('error_rate', 0):.1%} |")
    lines.append("")

    # Agent 评分表
    lines.append("## 各 Agent 评分")
    lines.append("")
    lines.append("| Agent | 评分 | 等级 | 维度得分 |")
    lines.append("|-------|------|------|---------|")

    for score in report.scores:
        dim_str = ", ".join(
            f"{k}: {v:.0%}" for k, v in score.dimensions.items()
        )
        grade_emoji_single = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(score.grade, "⚪")
        lines.append(
            f"| {score.agent_name} | {score.overall_score:.1%} | "
            f"{score.grade} {grade_emoji_single} | {dim_str} |"
        )
    lines.append("")

    # 详细评价
    lines.append("## 详细评价")
    lines.append("")
    for score in report.scores:
        lines.append(f"### {score.agent_name.title()} Agent")
        lines.append(f"- **评分**: {score.overall_score:.1%} ({score.grade})")
        lines.append(f"- **评价**: {score.notes}")
        lines.append(f"- **维度详情**:")
        for dim_name, dim_score in score.dimensions.items():
            bar = "█" * int(dim_score * 10) + "░" * (10 - int(dim_score * 10))
            lines.append(f"  - {dim_name}: {bar} {dim_score:.0%}")
        lines.append("")

    # 改进建议
    lines.append("## 改进建议")
    lines.append("")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
