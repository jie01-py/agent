"""
数据可视化专家代理（Visualizer Agent）-- 根据分析结果和数据特征生成图表。

审查分析发现和数据特征，选择合适的图表类型，使用 ``create_chart`` 工具
及辅助工具（``get_dataframe_info``、``value_distribution``、``group_statistics``）
生成 3-5 张清晰、美观的可视化图表。

图表文件路径从工具执行结果中提取，存入 state 供 Reporter 代理在最终报告中引用。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from insightflow.config import get_chat_model
from insightflow.llm.resilient_client import ResilientLLMClient
from insightflow.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 InsightFlow中的数据可视化专家（Visualizer Agent）。
你的任务是：
1. 根据分析结果和数据特征，选择最合适的图表类型
2. 生成清晰、美观的可视化图表

图表选择原则：
- 比较类数据 → 柱状图 (bar)
- 趋势类数据 → 折线图 (line)
- 关系类数据 → 散点图 (scatter)
- 分布类数据 → 直方图 (hist) 或 箱线图 (box)
- 占比类数据 → 饼图 (pie)

每张图表请确保：
- 有清晰的标题（中文）
- 坐标轴有标签
- 选择合适的图表类型
- 生成 3-5 张图表覆盖关键发现
"""

_CHART_PATH_PATTERN = re.compile(r"Chart saved to:\s*(.+)")


def _extract_chart_paths(agent_result: dict) -> list[str]:
    """从代理执行记录中提取图表文件路径。

    搜索 agent 执行过程中的中间步骤，找到 ``create_chart`` 工具生成的
    图表文件路径。

    参数:
        agent_result: agent.invoke() 返回的完整结果字典，包含带有工具调用信息的消息。

    返回:
        生成的图表图片文件路径列表。
    """
    paths: list[str] = []

    messages = agent_result.get("messages", [])

    for msg in messages:
        if hasattr(msg, 'additional_kwargs'):
            tool_calls = msg.additional_kwargs.get('tool_calls', [])
            for tool_call in tool_calls:
                if tool_call.get('function', {}).get('name') == 'create_chart':
                    args_str = tool_call.get('function', {}).get('arguments', '{}')
                    try:
                        args = json.loads(args_str)
                        output_path = args.get('output_path', '')
                        if output_path and output_path not in paths:
                            paths.append(output_path)
                    except json.JSONDecodeError:
                        continue

    if not paths:
        final_output = messages[-1].content if messages else ""
        matches = _CHART_PATH_PATTERN.findall(str(final_output))
        for match in matches:
            path = match.strip()
            if path and path not in paths:
                paths.append(path)

    return paths


def visualizer_node(state: AgentState) -> dict:
    """LangGraph 节点函数：数据可视化专家。

    使用带有图表创建工具的 Agent，根据分析结果和数据画像生成 3-5 张可视化图表，
    并从工具执行结果中提取图表文件路径。

    参数:
        state: 当前系统状态，应包含 ``analysis_results``、``data_profile``
               以及清洗后的 ``dataframe``。

    返回:
        部分状态字典，包含以下键：
        - charts: 生成的图表图片文件路径列表
        - messages: 本代理的日志消息
        - current_agent: "visualizer"
    """
    try:
        llm = get_chat_model()
        resilient_llm = ResilientLLMClient(llm)

        from insightflow.tools.data_tools import (
            create_chart,
            get_dataframe_info,
            group_statistics,
            value_distribution,
        )

        tools = [create_chart, get_dataframe_info, value_distribution, group_statistics]

        analysis_results: dict[str, Any] = state.get("analysis_results", {})
        data_profile: dict[str, Any] = state.get("data_profile", {})

        analysis_text = json.dumps(analysis_results, ensure_ascii=False, indent=2, default=str)
        profile_text = json.dumps(data_profile, ensure_ascii=False, indent=2, default=str)

        config = state.get("config", {})
        output_dir = config.get("output_dir", "output")

        human_input = (
            f"请根据以下信息生成 3-5 张可视化图表：\n\n"
            f"分析结果：\n{analysis_text}\n\n"
            f"数据画像：\n{profile_text}\n\n"
            f"图表输出目录: {output_dir}\n"
            f"请确保每张图表的 output_path 以 '{output_dir}/' 开头。\n"
            f"选择最能展示关键发现的图表类型，确保标题和标签使用中文。"
        )

        agent = create_agent(
            model=resilient_llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=human_input)
                ]
            },
            config={"recursion_limit": 100},
        )

        chart_paths: list[str] = _extract_chart_paths(result)

        return {
            "charts": chart_paths,
            "messages": [
                {
                    "role": "visualizer",
                    "content": (
                            f"可视化完成。共生成 {len(chart_paths)} 张图表"
                            + (f": {', '.join(chart_paths)}" if chart_paths else "。")
                    ),
                }
            ],
            "current_agent": "visualizer",
        }

    except Exception as exc:
        logger.exception("Visualizer agent failed: %s", exc)
        return {
            "charts": [],
            "messages": [
                {
                    "role": "visualizer",
                    "content": f"可视化生成失败: {exc}",
                }
            ],
            "errors": [f"[visualizer] {exc}"],
            "current_agent": "visualizer",
        }
