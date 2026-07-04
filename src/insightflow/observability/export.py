"""Export pipeline trace data in multiple formats.

Supports JSON, Markdown, and HTML export of PipelineTrace objects.
The HTML export includes a standalone visual timeline (Gantt-style)
with a dark theme suitable for portfolio presentations.

支持将 PipelineTrace 数据导出为 JSON、Markdown 和 HTML 格式。
HTML 导出包含独立的可视化时间线（甘特图风格），采用深色主题。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from insightflow.observability.tracer import PipelineTrace, TraceSpan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_trace_json(trace: PipelineTrace, output_path: str | Path) -> Path:
    """Export a pipeline trace as a structured JSON file.

    Args:
        trace: The PipelineTrace to export.
        output_path: Destination file path.

    Returns:
        The resolved output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = trace.to_dict()
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    output_path.write_text(content, encoding="utf-8")
    logger.info("Trace exported to JSON: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as a human-readable string."""
    if ts == 0.0:
        return "N/A"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _snapshot_to_md(snapshot: dict[str, Any]) -> str:
    """Render a snapshot dict as a Markdown bullet list."""
    if not snapshot:
        return "_No data_\n"
    lines: list[str] = []
    for key, value in snapshot.items():
        if isinstance(value, dict):
            lines.append(f"- **{key}**: `{json.dumps(value, ensure_ascii=False)}`")
        else:
            lines.append(f"- **{key}**: `{value}`")
    return "\n".join(lines) + "\n"


def export_trace_markdown(trace: PipelineTrace, output_path: str | Path) -> Path:
    """Export a pipeline trace as a human-readable Markdown document.

    Includes trace metadata, summary statistics, a span duration table,
    per-span input/output details in collapsible sections, and error details.

    Args:
        trace: The PipelineTrace to export.
        output_path: Destination file path.

    Returns:
        The resolved output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # Title
    sections.append("# Pipeline Execution Trace\n")

    # Metadata
    sections.append("## Trace Metadata\n")
    sections.append(f"| Field | Value |")
    sections.append(f"|-------|-------|")
    sections.append(f"| **Trace ID** | `{trace.trace_id}` |")
    sections.append(f"| **Start Time** | {_format_timestamp(trace.pipeline_start)} |")
    sections.append(f"| **End Time** | {_format_timestamp(trace.pipeline_end)} |")
    sections.append(f"| **Total Duration** | {trace.total_duration_ms:.2f} ms |")
    sections.append("")

    # Summary
    if trace.summary:
        sections.append("## Summary Statistics\n")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total Spans | {trace.summary.get('total_spans', 0)} |")
        sections.append(f"| Successful | {trace.summary.get('successful_spans', 0)} |")
        sections.append(f"| Failed | {trace.summary.get('failed_spans', 0)} |")
        sections.append(
            f"| Total Agent Time | {trace.summary.get('total_agent_time_ms', 0):.2f} ms |"
        )
        sections.append(
            f"| Avg Span Duration | {trace.summary.get('avg_span_duration_ms', 0):.2f} ms |"
        )
        sections.append(f"| Slowest Node | {trace.summary.get('slowest_node', 'N/A')} |")
        sections.append(f"| Fastest Node | {trace.summary.get('fastest_node', 'N/A')} |")
        sections.append("")

    # Span table
    if trace.spans:
        sections.append("## Execution Spans\n")
        sections.append("| # | Node | Duration (ms) | Status | Start Offset (ms) |")
        sections.append("|---|------|--------------|--------|-------------------|")
        for i, span in enumerate(trace.spans, 1):
            offset = (span.start_time - trace.pipeline_start) * 1000
            status_icon = "SUCCESS" if span.status == "success" else "ERROR"
            sections.append(
                f"| {i} | **{span.node_name}** | {span.duration_ms:.2f} "
                f"| {status_icon} | {offset:.2f} |"
            )
        sections.append("")

    # Per-span details
    if trace.spans:
        sections.append("## Span Details\n")
        for i, span in enumerate(trace.spans, 1):
            sections.append(
                f"<details>\n<summary><b>Span {i}: {span.node_name}</b> "
                f"({span.duration_ms:.2f} ms, {span.status})</summary>\n"
            )
            sections.append(f"### Input Snapshot\n")
            sections.append(_snapshot_to_md(span.input_snapshot))
            sections.append(f"\n### Output Snapshot\n")
            sections.append(_snapshot_to_md(span.output_snapshot))
            if span.error:
                sections.append(f"\n### Error\n")
                sections.append(f"```\n{span.error}\n```\n")
            if span.metadata:
                sections.append(f"\n### Metadata\n")
                sections.append(f"```json\n{json.dumps(span.metadata, indent=2, ensure_ascii=False, default=str)}\n```\n")
            sections.append("</details>\n")

    # Error summary
    error_spans = [s for s in trace.spans if s.status == "error"]
    if error_spans:
        sections.append("## Error Details\n")
        sections.append(f"**{len(error_spans)} span(s) failed during execution.**\n")
        for span in error_spans:
            sections.append(f"### {span.node_name}\n")
            sections.append(f"```\n{span.error}\n```\n")

    content = "\n".join(sections)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Trace exported to Markdown: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def _build_gantt_bars(trace: PipelineTrace) -> str:
    """Build CSS-based Gantt chart bars for the timeline visualization.

    Each bar is positioned proportionally based on its start offset relative
    to the total pipeline duration, and its width is proportional to the
    span's duration.

    Args:
        trace: The completed PipelineTrace.

    Returns:
        HTML string containing the Gantt bar elements.
    """
    if not trace.spans or trace.total_duration_ms <= 0:
        return '<div class="gantt-empty">No spans recorded</div>'

    total_ms = trace.total_duration_ms
    bars: list[str] = []

    for i, span in enumerate(trace.spans):
        offset_ms = (span.start_time - trace.pipeline_start) * 1000
        left_pct = (offset_ms / total_ms) * 100
        width_pct = max((span.duration_ms / total_ms) * 100, 0.5)  # min visible width
        color = "var(--color-success)" if span.status == "success" else "var(--color-error)"
        bar_id = f"span-{i}"

        bars.append(
            f'<div class="gantt-row">'
            f'<div class="gantt-label" title="{span.node_name}">'
            f'<span class="node-name">{span.node_name}</span>'
            f'<span class="node-duration">{span.duration_ms:.0f}ms</span>'
            f'</div>'
            f'<div class="gantt-track">'
            f'<div class="gantt-bar" id="{bar_id}" '
            f'style="left:{left_pct:.2f}%;width:{width_pct:.2f}%;background:{color};" '
            f'title="{span.node_name}: {span.duration_ms:.2f}ms ({span.status})" '
            f'data-index="{i}">'
            f'<span class="bar-label">{span.node_name}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

    return "\n".join(bars)


def _build_span_rows(trace: PipelineTrace) -> str:
    """Build HTML table rows for the span details table.

    Args:
        trace: The completed PipelineTrace.

    Returns:
        HTML string containing table row elements.
    """
    rows: list[str] = []
    for i, span in enumerate(trace.spans):
        status_class = "status-success" if span.status == "success" else "status-error"
        status_text = "Success" if span.status == "success" else "Error"
        offset_ms = (span.start_time - trace.pipeline_start) * 1000

        input_keys = ", ".join(span.input_snapshot.keys()) if span.input_snapshot else "-"
        output_keys = ", ".join(span.output_snapshot.keys()) if span.output_snapshot else "-"

        error_cell = ""
        if span.error:
            error_cell = f'<span class="error-text" title="{_html_escape(span.error)}">ERROR</span>'
        else:
            error_cell = '<span class="dim-text">-</span>'

        rows.append(
            f'<tr>'
            f'<td class="num">{i + 1}</td>'
            f'<td class="node-cell"><strong>{span.node_name}</strong></td>'
            f'<td class="num">{span.duration_ms:.2f}</td>'
            f'<td class="num">{offset_ms:.2f}</td>'
            f'<td><span class="{status_class}">{status_text}</span></td>'
            f'<td class="keys-cell">{input_keys}</td>'
            f'<td class="keys-cell">{output_keys}</td>'
            f'<td>{error_cell}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _build_summary_cards(summary: dict[str, Any]) -> str:
    """Build HTML summary statistic cards.

    Args:
        summary: The trace summary dict.

    Returns:
        HTML string containing card elements.
    """
    if not summary:
        return ""

    cards = [
        ("Total Spans", summary.get("total_spans", 0), ""),
        ("Successful", summary.get("successful_spans", 0), "var(--color-success)"),
        ("Failed", summary.get("failed_spans", 0), "var(--color-error)"),
        ("Total Agent Time", f"{summary.get('total_agent_time_ms', 0):.0f}", "ms"),
        ("Avg Span Duration", f"{summary.get('avg_span_duration_ms', 0):.0f}", "ms"),
        ("Slowest Node", summary.get("slowest_node", "N/A"), ""),
        ("Fastest Node", summary.get("fastest_node", "N/A"), ""),
    ]

    html_cards: list[str] = []
    for label, value, unit in cards:
        color_attr = ""
        if isinstance(unit, str) and unit.startswith("var("):
            color_attr = f' style="color:{unit}"'
            unit = ""
        unit_html = f'<span class="card-unit">{unit}</span>' if unit else ""
        html_cards.append(
            f'<div class="summary-card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value"{color_attr}>{value}{unit_html}</div>'
            f'</div>'
        )

    return "\n".join(html_cards)


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for attribute values and text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def export_trace_html(trace: PipelineTrace, output_path: str | Path) -> Path:
    """Export a pipeline trace as a standalone HTML file with visual timeline.

    Produces a self-contained HTML document with inline CSS featuring:
    - Dark navy theme (#1a1a2e background)
    - Gantt-style timeline visualization with proportional bars
    - Color-coded success/error status indicators
    - Span details table
    - Summary statistics cards

    No external dependencies are required to view the output.

    Args:
        trace: The PipelineTrace to export.
        output_path: Destination file path.

    Returns:
        The resolved output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gantt_bars = _build_gantt_bars(trace)
    span_rows = _build_span_rows(trace)
    summary_cards = _build_summary_cards(trace.summary)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipeline Trace - {trace.trace_id}</title>
<style>
:root {{
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-tertiary: #0f3460;
    --accent: #e94560;
    --color-success: #00d4aa;
    --color-error: #e94560;
    --color-warning: #f0a500;
    --text-primary: #e8e8e8;
    --text-secondary: #a0a0b8;
    --text-dim: #6c6c8a;
    --border-color: #2a2a4a;
    --font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
}}

*, *::before, *::after {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: var(--font-family);
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}}

/* ---- Header ---- */

.trace-header {{
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
}}

.trace-header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    color: var(--text-primary);
}}

.trace-header .subtitle {{
    color: var(--text-secondary);
    font-size: 0.9rem;
}}

.trace-meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.25rem;
}}

.meta-item {{
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}}

.meta-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
}}

.meta-value {{
    font-size: 0.95rem;
    font-family: var(--font-mono);
    color: var(--text-primary);
}}

/* ---- Summary Cards ---- */

.summary-section {{
    margin-bottom: 2rem;
}}

.summary-section h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
}}

.summary-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    transition: border-color 0.2s;
}}

.summary-card:hover {{
    border-color: var(--accent);
}}

.card-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}}

.card-value {{
    font-size: 1.5rem;
    font-weight: 700;
    font-family: var(--font-mono);
}}

.card-unit {{
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-left: 0.15rem;
}}

/* ---- Gantt Timeline ---- */

.timeline-section {{
    margin-bottom: 2rem;
}}

.timeline-section h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.gantt-container {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    overflow-x: auto;
}}

.gantt-row {{
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
}}

.gantt-row:last-child {{
    margin-bottom: 0;
}}

.gantt-label {{
    width: 160px;
    min-width: 160px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-right: 1rem;
    font-size: 0.85rem;
}}

.node-name {{
    font-weight: 600;
    color: var(--text-primary);
}}

.node-duration {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-dim);
}}

.gantt-track {{
    flex: 1;
    position: relative;
    height: 28px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 6px;
    overflow: hidden;
}}

.gantt-bar {{
    position: absolute;
    top: 2px;
    height: 24px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    padding: 0 0.5rem;
    min-width: 4px;
    transition: opacity 0.2s;
    cursor: default;
}}

.gantt-bar:hover {{
    opacity: 0.85;
    filter: brightness(1.15);
}}

.bar-label {{
    font-size: 0.7rem;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}}

.gantt-empty {{
    text-align: center;
    color: var(--text-dim);
    padding: 2rem;
    font-style: italic;
}}

/* ---- Axis ---- */

.gantt-axis {{
    display: flex;
    justify-content: space-between;
    padding-left: 160px;
    margin-top: 0.5rem;
    font-size: 0.7rem;
    font-family: var(--font-mono);
    color: var(--text-dim);
}}

/* ---- Span Table ---- */

.table-section {{
    margin-bottom: 2rem;
}}

.table-section h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.table-wrapper {{
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}}

thead {{
    background: var(--bg-tertiary);
}}

th {{
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-color);
}}

td {{
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border-color);
    vertical-align: middle;
}}

tr:last-child td {{
    border-bottom: none;
}}

tr:hover td {{
    background: rgba(255, 255, 255, 0.02);
}}

td.num {{
    text-align: right;
    font-family: var(--font-mono);
    font-size: 0.825rem;
}}

.node-cell {{
    font-weight: 600;
}}

.status-success {{
    color: var(--color-success);
    font-weight: 600;
    font-size: 0.8rem;
}}

.status-error {{
    color: var(--color-error);
    font-weight: 600;
    font-size: 0.8rem;
}}

.error-text {{
    color: var(--color-error);
    font-weight: 600;
    font-size: 0.8rem;
    cursor: help;
}}

.dim-text {{
    color: var(--text-dim);
}}

.keys-cell {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* ---- Footer ---- */

.trace-footer {{
    text-align: center;
    padding: 1.5rem 0;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border-color);
}}

.trace-footer a {{
    color: var(--accent);
    text-decoration: none;
}}

/* ---- Responsive ---- */

@media (max-width: 768px) {{
    body {{
        padding: 1rem;
    }}
    .trace-meta {{
        grid-template-columns: 1fr 1fr;
    }}
    .summary-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .gantt-label {{
        width: 100px;
        min-width: 100px;
    }}
    .gantt-axis {{
        padding-left: 100px;
    }}
}}
</style>
</head>
<body>

<!-- Header -->
<div class="trace-header">
    <h1>Pipeline Execution Trace</h1>
    <div class="subtitle">InsightFlow Multi-Agent Pipeline &mdash; Execution Report</div>
    <div class="trace-meta">
        <div class="meta-item">
            <span class="meta-label">Trace ID</span>
            <span class="meta-value">{trace.trace_id}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Start Time</span>
            <span class="meta-value">{_format_timestamp(trace.pipeline_start)}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">End Time</span>
            <span class="meta-value">{_format_timestamp(trace.pipeline_end)}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Total Duration</span>
            <span class="meta-value">{trace.total_duration_ms:.2f} ms</span>
        </div>
    </div>
</div>

<!-- Summary Cards -->
<div class="summary-section">
    <h2>Summary</h2>
    <div class="summary-grid">
        {summary_cards}
    </div>
</div>

<!-- Gantt Timeline -->
<div class="timeline-section">
    <h2>Execution Timeline</h2>
    <div class="gantt-container">
        {gantt_bars}
        <div class="gantt-axis">
            <span>0 ms</span>
            <span>{trace.total_duration_ms / 4:.0f} ms</span>
            <span>{trace.total_duration_ms / 2:.0f} ms</span>
            <span>{trace.total_duration_ms * 3 / 4:.0f} ms</span>
            <span>{trace.total_duration_ms:.0f} ms</span>
        </div>
    </div>
</div>

<!-- Span Details Table -->
<div class="table-section">
    <h2>Span Details</h2>
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Node</th>
                    <th style="text-align:right">Duration (ms)</th>
                    <th style="text-align:right">Offset (ms)</th>
                    <th>Status</th>
                    <th>Input Keys</th>
                    <th>Output Keys</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
                {span_rows}
            </tbody>
        </table>
    </div>
</div>

<!-- Footer -->
<div class="trace-footer">
    Generated by InsightFlow Observability Module
</div>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Trace exported to HTML: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# OTLP (OpenTelemetry) export
# ---------------------------------------------------------------------------


def export_trace_otlp(trace: PipelineTrace, output_path: str | Path) -> Path:
    """Export a pipeline trace in OpenTelemetry (OTLP) JSON format.

    Each agent span is converted to an OTLP span with attributes for
    agent name, status, duration, and token usage. The output can be
    imported into Jaeger, Grafana Tempo, or any OTLP-compatible backend.

    Args:
        trace: The PipelineTrace to export.
        output_path: File path for the OTLP JSON output.

    Returns:
        The Path where the file was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    otlp_spans = [span.to_otlp(trace.trace_id) for span in trace.spans]

    otlp_payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": {
                        "service.name": "insightflow",
                        "service.version": "0.2.0",
                        "pipeline.trace_id": trace.trace_id,
                    }
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "insightflow.tracer",
                            "version": "0.2.0",
                        },
                        "spans": otlp_spans,
                    }
                ],
            }
        ],
        "metadata": {
            "pipeline_start": trace.pipeline_start,
            "pipeline_end": trace.pipeline_end,
            "total_duration_ms": round(trace.total_duration_ms, 2),
            "config": trace.config,
            "summary": trace.summary,
        },
    }

    output_path.write_text(
        json.dumps(otlp_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("OTLP trace exported to: %s (%d spans)", output_path, len(otlp_spans))
    return output_path


# ---------------------------------------------------------------------------
# Convenience: export all formats
# ---------------------------------------------------------------------------


def export_all(trace: PipelineTrace, output_dir: str | Path) -> dict[str, Path]:
    """Export a pipeline trace in all supported formats.

    Creates JSON, Markdown, HTML, and OTLP files in the specified directory,
    named using the trace ID.

    Args:
        trace: The PipelineTrace to export.
        output_dir: Directory where export files will be written.

    Returns:
        A dict mapping format names to their output Paths:
        ``{"json": Path, "markdown": Path, "html": Path, "otlp": Path}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = f"trace_{trace.trace_id}"

    results: dict[str, Path] = {}
    results["json"] = export_trace_json(trace, output_dir / f"{base}.json")
    results["markdown"] = export_trace_markdown(trace, output_dir / f"{base}.md")
    results["html"] = export_trace_html(trace, output_dir / f"{base}.html")
    results["otlp"] = export_trace_otlp(trace, output_dir / f"{base}_otlp.json")

    logger.info(
        "All trace formats exported to %s (%d files)",
        output_dir,
        len(results),
    )

    return results
