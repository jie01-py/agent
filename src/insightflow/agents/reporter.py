"""
报告撰写员代理（Reporter Agent）-- 将系统所有结果汇总为最终报告。

收集系统状态中的数据画像、清洗计划、分析结果和图表路径，
通过单次 LLM 调用（无需工具）生成完整的 Markdown 分析报告。

报告结构：
1. 概述
2. 数据概况
3. 数据清洗
4. 分析发现
5. 可视化
6. 结论与建议
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from insightflow.config import get_chat_model
from insightflow.llm.resilient_client import ResilientLLMClient
from insightflow.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 InsightFlow中的报告撰写员（Reporter Agent）。
你的任务是：将数据画像、清洗记录、分析结果和可视化图表汇总为一份完整的分析报告。

报告格式（Markdown）：
# 数据分析报告：[分析主题]

## 1. 概述
简要说明分析目标、数据来源和主要发现。

## 2. 数据概况
数据的基本信息和质量状况。

## 3. 数据清洗
执行的清洗操作和原因。

## 4. 分析发现
详细的分析结果和解读。

## 5. 可视化
引用生成的图表文件路径。

## 6. 结论与建议
基于分析结果的结论和建议。

要求：
- 语言清晰、专业
- 用数据支撑每个结论
- 适当使用表格展示关键数据
- 图表用 ![图表标题](路径) 引用
"""


def _format_json_block(obj: Any, label: str = "") -> str:
    """将任意对象序列化为 JSON 文本块，失败时回退到 str()。

    参数:
        obj: 要序列化的对象（通常是字典）。
        label: 可选标签，保留供后续扩展使用。

    返回:
        JSON 格式的字符串，或在序列化失败时返回 str(obj)。
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _format_chart_references(charts: list[str]) -> str:
    """为生成的图表文件构建 Markdown 图片引用。

    参数:
        charts: 图表图片文件路径列表。

    返回:
        包含 Markdown 图片引用的字符串，无图表时返回提示信息。
    """
    if not charts:
        return "（未生成图表）"

    lines: list[str] = []
    for i, path in enumerate(charts, start=1):
        # 从文件名中提取标题
        filename = os.path.basename(path)
        title = Path(filename).stem.replace("_", " ")
        lines.append(f"![图表{i}: {title}]({path})")

    return "\n\n".join(lines)


def reporter_node(state: AgentState) -> dict:
    """LangGraph 节点函数：报告撰写员。

    收集系统所有输出（数据画像、清洗计划、分析结果、图表路径），
    通过单次 LLM 调用生成完整的 Markdown 报告并保存到输出目录。

    参数:
        state: 当前系统状态，包含 ``analysis_task``、``data_profile``、
               ``cleaning_plan``、``analysis_results`` 和 ``charts``。

    返回:
        部分状态字典，包含以下键：
        - report: 生成的 Markdown 报告字符串
        - messages: 本代理的日志消息
        - current_agent: "reporter"
    """
    try:
        # ----------------------------------------------------------------
        # 1. 初始化 LLM（报告生成无需工具）
        # ----------------------------------------------------------------
        llm = get_chat_model()
        resilient_llm = ResilientLLMClient(llm)

        # ----------------------------------------------------------------
        # 2. 收集系统所有输出
        # ----------------------------------------------------------------
        analysis_task: str = state.get("analysis_task", "")
        data_profile: dict[str, Any] = state.get("data_profile", {})
        cleaning_plan: dict[str, Any] = state.get("cleaning_plan", {})
        analysis_results: dict[str, Any] = state.get("analysis_results", {})
        charts: list[str] = state.get("charts", [])

        # 将各部分格式化为提示词文本
        profile_text = _format_json_block(data_profile)
        cleaning_text = _format_json_block(cleaning_plan)
        analysis_text = _format_json_block(analysis_results)
        charts_text = _format_chart_references(charts)

        # ----------------------------------------------------------------
        # 3. 构建提示词并调用 LLM
        # ----------------------------------------------------------------
        human_input = (
            f"分析主题: {analysis_task}\n\n"
            f"=== 数据画像 ===\n{profile_text}\n\n"
            f"=== 清洗计划 ===\n{cleaning_text}\n\n"
            f"=== 分析结果 ===\n{analysis_text}\n\n"
            f"=== 图表文件 ===\n{charts_text}\n\n"
            f"请根据以上信息，生成一份完整的 Markdown 格式数据分析报告。"
        )

        # 手动构建消息列表，而非使用 `prompt | llm` 链式调用，
        # 因为 ResilientLLMClient 不是 LangChain Runnable。
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_input),
        ]
        result = resilient_llm.invoke(messages)
        report_content: str = result.content if hasattr(result, "content") else str(result)

        # ----------------------------------------------------------------
        # 4. 保存报告到输出目录
        # ----------------------------------------------------------------
        config = state.get("config", {})
        output_dir = Path(config.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "analysis_report.md"
        report_path.write_text(report_content, encoding="utf-8")
        logger.info("Report saved to: %s", report_path)

        return {
            "report": report_content,
            "messages": [
                {
                    "role": "reporter",
                    "content": (
                        f"报告生成完成。已保存至 {report_path}。"
                        f"报告长度: {len(report_content)} 字符。"
                    ),
                }
            ],
            "current_agent": "reporter",
        }

    except Exception as exc:
        logger.exception("Reporter agent failed: %s", exc)
        return {
            "report": f"# 报告生成失败\n\n生成报告时发生错误: {exc}",
            "messages": [
                {
                    "role": "reporter",
                    "content": f"报告生成失败: {exc}",
                }
            ],
            "errors": [f"[reporter] {exc}"],
            "current_agent": "reporter",
        }
