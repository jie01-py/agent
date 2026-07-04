"""Quality evaluation metrics for InsightFlow agent outputs.

Each evaluation function takes the agent's output data and returns an
AgentScore with an overall score (0-1), per-dimension breakdown, and notes.

评估采用规则驱动（rule-based）而非 LLM 评估，确保结果确定性和执行速度。

评估函数:
- evaluate_profile: 评估 Scout Agent 生成的数据画像质量
- evaluate_cleaning: 评估 Cleaner Agent 生成的清洗计划质量
- evaluate_analysis: 评估 Analyst Agent 生成的分析结果质量
- evaluate_charts: 评估 Visualizer Agent 生成的图表质量
- evaluate_report: 评估 Reporter Agent 生成的报告质量
- evaluate_pipeline: 评估整体 InsightFlow 执行质量
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentScore:
    """单个 Agent 输出的质量评分。

    Attributes:
        agent_name: Agent 名称（如 "scout", "analyst"）
        overall_score: 加权平均分（0.0 - 1.0）
        dimensions: 各维度评分 {dimension_name: score}
        notes: 可读的评估说明（中文）
    """

    agent_name: str
    overall_score: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化的 dict。"""
        return {
            "agent_name": self.agent_name,
            "overall_score": round(self.overall_score, 3),
            "dimensions": {k: round(v, 3) for k, v in self.dimensions.items()},
            "notes": self.notes,
        }

    @property
    def grade(self) -> str:
        """根据总分返回字母等级。"""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"


def _clamp(value: float) -> float:
    """将值限制在 [0.0, 1.0] 范围内。"""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Scout Agent 评估
# ---------------------------------------------------------------------------

def evaluate_profile(data_profile: dict[str, Any]) -> AgentScore:
    """评估 Scout Agent 的数据画像质量。

    评分维度:
    - field_coverage (0.30): 画像是否包含所有预期的顶级字段
    - quality_issues_found (0.25): 是否识别出质量问题
    - numeric_stats_present (0.25): 是否包含数值统计信息
    - categorical_summary (0.20): 是否包含分类分布信息

    Args:
        data_profile: AgentState 中的 data_profile dict。

    Returns:
        Scout Agent 的 AgentScore。
    """
    if not data_profile or data_profile.get("error"):
        return AgentScore(
            agent_name="scout",
            overall_score=0.0,
            dimensions={"field_coverage": 0, "quality_issues_found": 0,
                        "numeric_stats_present": 0, "categorical_summary": 0},
            notes="数据画像为空或包含错误",
        )

    dimensions: dict[str, float] = {}
    notes_parts: list[str] = []

    # Dimension 1: field_coverage (0.30 weight)
    expected_fields = ["basic_info", "column_details", "numeric_summary",
                       "quality_issues", "categorical_summary"]
    # Also accept alternative field names from the MCP profile tool
    alt_fields = ["descriptive_stats", "missing_values", "categorical_distributions"]

    found_fields = set(data_profile.keys())
    matched = sum(1 for f in expected_fields if f in found_fields)
    alt_matched = sum(1 for f in alt_fields if f in found_fields)
    coverage = (matched + alt_matched) / len(expected_fields)
    dimensions["field_coverage"] = _clamp(coverage)

    if coverage >= 0.8:
        notes_parts.append("画像字段覆盖完整")
    elif coverage >= 0.5:
        notes_parts.append("画像字段部分覆盖")
    else:
        notes_parts.append("画像字段覆盖不足")

    # Dimension 2: quality_issues_found (0.25 weight)
    quality_issues = data_profile.get("quality_issues", [])
    missing_values = data_profile.get("missing_values", {})

    issues_count = len(quality_issues) if isinstance(quality_issues, list) else 0
    missing_count = sum(1 for v in missing_values.values()
                        if isinstance(v, dict) and v.get("null_count", 0) > 0)

    if issues_count > 0 or missing_count > 0:
        dimensions["quality_issues_found"] = _clamp(min((issues_count + missing_count) / 3, 1.0))
        notes_parts.append(f"识别到 {issues_count + missing_count} 个质量问题")
    else:
        # Check if there are any indicators of quality analysis
        raw = str(data_profile)
        has_quality_keywords = any(kw in raw for kw in ["missing", "缺失", "null", "质量", "quality"])
        dimensions["quality_issues_found"] = 0.5 if has_quality_keywords else 0.2
        notes_parts.append("未发现明确的质量问题标记")

    # Dimension 3: numeric_stats_present (0.25 weight)
    numeric_stats = data_profile.get("numeric_summary") or data_profile.get("descriptive_stats", {})
    if isinstance(numeric_stats, dict) and len(numeric_stats) > 0:
        dimensions["numeric_stats_present"] = _clamp(min(len(numeric_stats) / 3, 1.0))
        notes_parts.append(f"包含 {len(numeric_stats)} 个数值列的统计信息")
    else:
        dimensions["numeric_stats_present"] = 0.1
        notes_parts.append("缺少数值统计信息")

    # Dimension 4: categorical_summary (0.20 weight)
    cat_summary = data_profile.get("categorical_summary") or data_profile.get("categorical_distributions", {})
    if isinstance(cat_summary, dict) and len(cat_summary) > 0:
        dimensions["categorical_summary"] = _clamp(min(len(cat_summary) / 3, 1.0))
        notes_parts.append(f"包含 {len(cat_summary)} 个分类列的分布信息")
    else:
        dimensions["categorical_summary"] = 0.1
        notes_parts.append("缺少分类分布信息")

    # Compute weighted overall score
    weights = {"field_coverage": 0.30, "quality_issues_found": 0.25,
               "numeric_stats_present": 0.25, "categorical_summary": 0.20}
    overall = sum(dimensions[k] * weights[k] for k in weights)

    return AgentScore(
        agent_name="scout",
        overall_score=_clamp(overall),
        dimensions=dimensions,
        notes="；".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Cleaner Agent 评估
# ---------------------------------------------------------------------------

def evaluate_cleaning(
    cleaning_plan: dict[str, Any],
    data_profile: dict[str, Any] | None = None,
) -> AgentScore:
    """评估 Cleaner Agent 的清洗计划质量。

    评分维度:
    - has_strategy (0.25): 策略列表是否存在且有内容
    - actions_valid (0.25): 所有操作是否为已识别的有效类型
    - params_complete (0.25): 每个策略条目是否有必要参数
    - addresses_issues (0.25): 清洗计划是否针对已知质量问题

    Args:
        cleaning_plan: AgentState 中的 cleaning_plan dict。
        data_profile: 可选的 data_profile，用于交叉验证。

    Returns:
        Cleaner Agent 的 AgentScore。
    """
    if not cleaning_plan or cleaning_plan.get("error"):
        return AgentScore(
            agent_name="cleaner",
            overall_score=0.0,
            dimensions={"has_strategy": 0, "actions_valid": 0,
                        "params_complete": 0, "addresses_issues": 0},
            notes="清洗计划为空或包含错误",
        )

    dimensions: dict[str, float] = {}
    notes_parts: list[str] = []

    strategy = cleaning_plan.get("strategy", [])
    if not isinstance(strategy, list):
        strategy = []

    # Dimension 1: has_strategy (0.25 weight)
    if len(strategy) > 0:
        dimensions["has_strategy"] = _clamp(min(len(strategy) / 3, 1.0))
        notes_parts.append(f"清洗计划包含 {len(strategy)} 项操作")
    else:
        dimensions["has_strategy"] = 0.0
        notes_parts.append("清洗计划为空")

    # Dimension 2: actions_valid (0.25 weight)
    valid_actions = {"fill_missing", "remove_outliers", "normalize_column",
                     "drop_duplicates", "convert_type"}
    if strategy:
        valid_count = sum(1 for s in strategy if s.get("action") in valid_actions)
        dimensions["actions_valid"] = valid_count / len(strategy)
        notes_parts.append(f"{valid_count}/{len(strategy)} 项操作类型有效")
    else:
        dimensions["actions_valid"] = 0.0

    # Dimension 3: params_complete (0.25 weight)
    if strategy:
        complete_count = 0
        for s in strategy:
            has_column = bool(s.get("column"))
            has_action = bool(s.get("action"))
            has_params = bool(s.get("params"))
            has_reason = bool(s.get("reason"))
            if has_column and has_action and (has_params or has_reason):
                complete_count += 1
        dimensions["params_complete"] = complete_count / len(strategy)
        notes_parts.append(f"{complete_count}/{len(strategy)} 项操作参数完整")
    else:
        dimensions["params_complete"] = 0.0

    # Dimension 4: addresses_issues (0.25 weight)
    if data_profile and strategy:
        # Check if cleaning actions target columns with known issues
        profile_issues = data_profile.get("quality_issues", [])
        missing_values = data_profile.get("missing_values", {})

        issue_columns: set[str] = set()
        if isinstance(profile_issues, list):
            for issue in profile_issues:
                if isinstance(issue, dict) and "column" in issue:
                    issue_columns.add(issue["column"])
        if isinstance(missing_values, dict):
            for col, info in missing_values.items():
                if isinstance(info, dict) and info.get("null_count", 0) > 0:
                    issue_columns.add(col)

        cleaned_columns = {s.get("column", "") for s in strategy}
        if issue_columns:
            overlap = cleaned_columns & issue_columns
            dimensions["addresses_issues"] = _clamp(len(overlap) / max(len(issue_columns), 1))
            notes_parts.append(f"清洗计划覆盖了 {len(overlap)}/{len(issue_columns)} 个问题列")
        else:
            dimensions["addresses_issues"] = 0.6  # No known issues, give moderate score
            notes_parts.append("数据画像中未明确标记问题列")
    else:
        dimensions["addresses_issues"] = 0.5
        notes_parts.append("无法交叉验证清洗计划与数据画像")

    weights = {"has_strategy": 0.25, "actions_valid": 0.25,
               "params_complete": 0.25, "addresses_issues": 0.25}
    overall = sum(dimensions[k] * weights[k] for k in weights)

    return AgentScore(
        agent_name="cleaner",
        overall_score=_clamp(overall),
        dimensions=dimensions,
        notes="；".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Analyst Agent 评估
# ---------------------------------------------------------------------------

def evaluate_analysis(analysis_results: dict[str, Any]) -> AgentScore:
    """评估 Analyst Agent 的分析结果质量。

    评分维度:
    - has_summary (0.20): 是否有有意义的摘要
    - findings_count (0.25): 发现的数量和质量
    - evidence_quality (0.30): 发现是否有数据支撑的证据
    - structure (0.25): 输出是否符合预期的 JSON 结构

    Args:
        analysis_results: AgentState 中的 analysis_results dict。

    Returns:
        Analyst Agent 的 AgentScore。
    """
    if not analysis_results or analysis_results.get("summary", "").startswith("分析失败"):
        return AgentScore(
            agent_name="analyst",
            overall_score=0.0,
            dimensions={"has_summary": 0, "findings_count": 0,
                        "evidence_quality": 0, "structure": 0},
            notes="分析结果为空或分析失败",
        )

    dimensions: dict[str, float] = {}
    notes_parts: list[str] = []

    # Dimension 1: has_summary (0.20 weight)
    summary = analysis_results.get("summary", "")
    if isinstance(summary, str) and len(summary) > 20:
        dimensions["has_summary"] = _clamp(min(len(summary) / 100, 1.0))
        notes_parts.append(f"摘要长度 {len(summary)} 字符")
    elif isinstance(summary, str) and len(summary) > 0:
        dimensions["has_summary"] = 0.4
        notes_parts.append("摘要过短")
    else:
        dimensions["has_summary"] = 0.0
        notes_parts.append("缺少摘要")

    # Dimension 2: findings_count (0.25 weight)
    findings = analysis_results.get("findings", [])
    if isinstance(findings, list) and len(findings) > 0:
        dimensions["findings_count"] = _clamp(min(len(findings) / 4, 1.0))
        notes_parts.append(f"包含 {len(findings)} 项发现")
    else:
        dimensions["findings_count"] = 0.1
        notes_parts.append("缺少结构化发现")

    # Dimension 3: evidence_quality (0.30 weight)
    if isinstance(findings, list) and len(findings) > 0:
        evidence_scores: list[float] = []
        for f in findings:
            if not isinstance(f, dict):
                evidence_scores.append(0.0)
                continue
            score = 0.0
            if f.get("finding"):
                score += 0.3
            if f.get("evidence") and len(str(f["evidence"])) > 10:
                score += 0.4
            if f.get("confidence") in ("high", "medium", "low"):
                score += 0.3
            evidence_scores.append(score)
        dimensions["evidence_quality"] = sum(evidence_scores) / len(evidence_scores)
        notes_parts.append(f"证据质量平均分 {dimensions['evidence_quality']:.2f}")
    else:
        dimensions["evidence_quality"] = 0.0

    # Dimension 4: structure (0.25 weight)
    expected_keys = {"summary", "findings", "statistics", "data_quality_note"}
    present_keys = set(analysis_results.keys()) & expected_keys
    dimensions["structure"] = _clamp(len(present_keys) / len(expected_keys))
    notes_parts.append(f"结构完整度 {len(present_keys)}/{len(expected_keys)}")

    weights = {"has_summary": 0.20, "findings_count": 0.25,
               "evidence_quality": 0.30, "structure": 0.25}
    overall = sum(dimensions[k] * weights[k] for k in weights)

    return AgentScore(
        agent_name="analyst",
        overall_score=_clamp(overall),
        dimensions=dimensions,
        notes="；".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Visualizer Agent 评估
# ---------------------------------------------------------------------------

def evaluate_charts(charts: list[str]) -> AgentScore:
    """评估 Visualizer Agent 的图表输出质量。

    评分维度:
    - chart_count (0.30): 生成的图表数量（目标: 3-5 张）
    - files_exist (0.40): 图表文件是否实际存在于磁盘
    - type_diversity (0.30): 使用的图表类型多样性

    Args:
        charts: AgentState 中的 charts 列表（文件路径列表）。

    Returns:
        Visualizer Agent 的 AgentScore。
    """
    dimensions: dict[str, float] = {}
    notes_parts: list[str] = []

    if not charts:
        return AgentScore(
            agent_name="visualizer",
            overall_score=0.0,
            dimensions={"chart_count": 0, "files_exist": 0, "type_diversity": 0},
            notes="未生成任何图表",
        )

    # Dimension 1: chart_count (0.30 weight)
    count = len(charts)
    if 3 <= count <= 5:
        dimensions["chart_count"] = 1.0
    elif count >= 1:
        dimensions["chart_count"] = _clamp(count / 3)
    else:
        dimensions["chart_count"] = 0.0
    notes_parts.append(f"生成 {count} 张图表")

    # Dimension 2: files_exist (0.40 weight)
    existing = sum(1 for path in charts if os.path.exists(path))
    dimensions["files_exist"] = existing / count if count > 0 else 0.0
    notes_parts.append(f"{existing}/{count} 个图表文件存在")

    # Dimension 3: type_diversity (0.30 weight)
    chart_types: set[str] = set()
    for path in charts:
        basename = os.path.basename(path).lower()
        for ct in ("bar", "line", "scatter", "hist", "pie", "box"):
            if ct in basename:
                chart_types.add(ct)
    if chart_types:
        dimensions["type_diversity"] = _clamp(len(chart_types) / 3)
        notes_parts.append(f"使用了 {len(chart_types)} 种图表类型: {', '.join(sorted(chart_types))}")
    else:
        dimensions["type_diversity"] = 0.3
        notes_parts.append("无法从文件名推断图表类型多样性")

    weights = {"chart_count": 0.30, "files_exist": 0.40, "type_diversity": 0.30}
    overall = sum(dimensions[k] * weights[k] for k in weights)

    return AgentScore(
        agent_name="visualizer",
        overall_score=_clamp(overall),
        dimensions=dimensions,
        notes="；".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Reporter Agent 评估
# ---------------------------------------------------------------------------

def evaluate_report(report: str, charts: list[str] | None = None) -> AgentScore:
    """评估 Reporter Agent 生成的报告质量。

    评分维度:
    - length_adequate (0.20): 报告长度是否充分
    - structure_complete (0.30): 是否包含所有预期章节
    - chart_references (0.20): 是否通过 markdown 图片引用了图表
    - data_richness (0.30): 报告是否包含数据驱动的内容

    Args:
        report: AgentState 中的 report 字符串。
        charts: 可选的图表路径列表，用于交叉验证。

    Returns:
        Reporter Agent 的 AgentScore。
    """
    dimensions: dict[str, float] = {}
    notes_parts: list[str] = []

    if not report or report.startswith("# 报告生成失败"):
        return AgentScore(
            agent_name="reporter",
            overall_score=0.0,
            dimensions={"length_adequate": 0, "structure_complete": 0,
                        "chart_references": 0, "data_richness": 0},
            notes="报告为空或生成失败",
        )

    # Dimension 1: length_adequate (0.20 weight)
    length = len(report)
    if length >= 2000:
        dimensions["length_adequate"] = 1.0
    elif length >= 500:
        dimensions["length_adequate"] = _clamp(length / 2000)
    else:
        dimensions["length_adequate"] = _clamp(length / 500) * 0.5
    notes_parts.append(f"报告长度 {length} 字符")

    # Dimension 2: structure_complete (0.30 weight)
    expected_sections = ["概述", "数据概况", "数据清洗", "分析发现", "可视化", "结论"]
    # Also accept English equivalents
    alt_sections = ["overview", "data overview", "cleaning", "analysis",
                    "visualization", "conclusion"]

    report_lower = report.lower()
    found_sections = 0
    for section in expected_sections:
        if section in report:
            found_sections += 1
    if found_sections < 3:
        for section in alt_sections:
            if section in report_lower:
                found_sections += 1

    dimensions["structure_complete"] = _clamp(found_sections / len(expected_sections))
    notes_parts.append(f"检测到 {found_sections}/{len(expected_sections)} 个预期章节")

    # Dimension 3: chart_references (0.20 weight)
    import re
    image_refs = re.findall(r"!\[.*?\]\(.*?\)", report)
    chart_ref_count = len(image_refs)

    if charts:
        expected_refs = len(charts)
        if expected_refs > 0:
            dimensions["chart_references"] = _clamp(chart_ref_count / expected_refs)
        else:
            dimensions["chart_references"] = 0.5 if chart_ref_count > 0 else 0.3
    else:
        dimensions["chart_references"] = _clamp(min(chart_ref_count / 3, 1.0))
    notes_parts.append(f"引用了 {chart_ref_count} 张图表")

    # Dimension 4: data_richness (0.30 weight)
    data_indicators = [
        len(re.findall(r"\d+\.?\d*%", report)),           # percentages
        len(re.findall(r"\|.*\|.*\|", report)),            # table rows
        len(re.findall(r"\d{4,}", report)),                 # large numbers (data values)
        len(re.findall(r"(?:相关|平均|总计|占比|增长率|同比|环比)", report)),  # analytical terms
    ]
    richness_score = _clamp(sum(min(d / 5, 1.0) for d in data_indicators) / len(data_indicators))
    dimensions["data_richness"] = richness_score
    notes_parts.append(f"数据丰富度评分 {richness_score:.2f}")

    weights = {"length_adequate": 0.20, "structure_complete": 0.30,
               "chart_references": 0.20, "data_richness": 0.30}
    overall = sum(dimensions[k] * weights[k] for k in weights)

    return AgentScore(
        agent_name="reporter",
        overall_score=_clamp(overall),
        dimensions=dimensions,
        notes="；".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# InsightFlow 整体评估
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    agent_scores: list[AgentScore],
    trace_summary: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """根据各 Agent 评分计算 InsightFlow 整体评估。

    Args:
        agent_scores: 各 Agent 的 AgentScore 列表。
        trace_summary: 可选的 PipelineTracer 追踪摘要 dict。
        errors: 可选的 AgentState 错误字符串列表。

    Returns:
        InsightFlow 级别指标 dict:
        - pipeline_score: 整体加权评分（0-1）
        - agent_count: 评估的 Agent 数量
        - avg_agent_score: Agent 平均分
        - error_rate: 出错的 Agent 比例
        - total_duration_ms: 总耗时
        - iteration_count: 质检迭代次数
        - grade: InsightFlow 字母等级
    """
    if not agent_scores:
        return {
            "pipeline_score": 0.0,
            "agent_count": 0,
            "avg_agent_score": 0.0,
            "error_rate": 1.0,
            "grade": "F",
        }

    scores = [s.overall_score for s in agent_scores]
    avg_score = sum(scores) / len(scores)

    # Pipeline score: 70% agent average + 30% execution quality
    execution_quality = 1.0
    if errors:
        execution_quality = _clamp(1.0 - len(errors) * 0.2)

    if trace_summary:
        failed = trace_summary.get("failed_spans", 0)
        total = trace_summary.get("total_spans", 1)
        execution_quality *= _clamp(1.0 - failed / max(total, 1))

    pipeline_score = 0.7 * avg_score + 0.3 * execution_quality

    # Letter grade
    if pipeline_score >= 0.9:
        grade = "A"
    elif pipeline_score >= 0.8:
        grade = "B"
    elif pipeline_score >= 0.7:
        grade = "C"
    elif pipeline_score >= 0.6:
        grade = "D"
    else:
        grade = "F"

    return {
        "pipeline_score": round(pipeline_score, 3),
        "agent_count": len(agent_scores),
        "avg_agent_score": round(avg_score, 3),
        "error_rate": round(len(errors or []) / max(len(agent_scores), 1), 3),
        "total_duration_ms": trace_summary.get("total_agent_time_ms", 0) if trace_summary else 0,
        "grade": grade,
    }
