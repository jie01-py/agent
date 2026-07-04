"""
数据分析师代理（Analyst Agent）-- 使用统计工具对清洗后的数据进行分析。

调用统计工具（correlation_analysis、group_statistics、describe_numeric、
value_distribution、get_dataframe_info）生成结构化分析结果，
直接回答用户的分析问题。

结果包括摘要、各项发现（含置信度）、原始统计数据以及数据质量备注。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from insightflow.config import get_chat_model
from insightflow.llm.resilient_client import ResilientLLMClient
from insightflow.state import AgentState
from insightflow.utils.json_parser import extract_json_with_schema, ANALYST_RESULTS_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 InsightFlow中的数据分析师（Analyst Agent）。
你的任务是：
1. 理解用户的分析需求
2. 使用统计工具对清洗后的数据进行分析
3. 给出结构化的分析结果

分析时请注意：
- 使用适当的统计方法（相关性分析、分组统计、描述统计等）
- 对结果进行解释，不要只给数字
- 如果数据质量仍存在问题，在结果中标注出来
- 分析结果应直接回答用户的问题

输出格式：JSON 格式
{
  "summary": "一句话总结",
  "findings": [{"finding": "发现", "evidence": "数据支撑", "confidence": "high/medium/low"}],
  "statistics": {...},
  "data_quality_note": "数据质量备注（如有）"
}
"""


def analyst_node(state: AgentState) -> dict:
    """LangGraph 节点函数：数据分析师。

    调用带有统计工具的 Agent，结合用户的原始问题和数据画像，对清洗后的数据
    进行分析，生成包含发现、统计数据和置信度的结构化结果。

    参数:
        state: 当前系统状态，应包含 ``analysis_task``、``data_profile``
               以及清洗后的 ``dataframe``。

    返回:
        部分状态字典，包含以下键：
        - analysis_results: 结构化分析结果字典
        - messages: 本代理的日志消息
        - current_agent: "analyst"
    """
    try:
        # ----------------------------------------------------------------
        # 1. 初始化 LLM 和统计工具
        # ----------------------------------------------------------------
        llm = get_chat_model()
        resilient_llm = ResilientLLMClient(llm)

        from insightflow.tools.data_tools import (
            correlation_analysis,
            describe_numeric,
            get_dataframe_info,
            group_statistics,
            value_distribution,
        )

        tools = [
            correlation_analysis,
            group_statistics,
            describe_numeric,
            value_distribution,
            get_dataframe_info,
        ]

        # ----------------------------------------------------------------
        # 2. 上下文刷新：对清洗后的 DataFrame 重新画像
        # ----------------------------------------------------------------
        analysis_task: str = state.get("analysis_task", "")
        data_profile: dict[str, Any] = state.get("data_profile", {})
        profile_text = json.dumps(data_profile, ensure_ascii=False, indent=2, default=str)

        from insightflow.context import get_context
        try:
            ctx = get_context()
            if ctx.has_data:
                fresh_profile = ctx.quick_profile()
                # 用最新的画像替代 Scout 阶段的旧画像
                profile_text = json.dumps(fresh_profile, ensure_ascii=False, indent=2, default=str)
        except ValueError:
            # 回退到原始画像
            pass

        human_input = (
            f"用户的分析需求：{analysis_task}\n\n"
            f"数据画像参考：\n{profile_text}\n\n"
            f"请使用统计工具对数据进行分析，并以 JSON 格式输出结果。"
            f"确保分析结果直接回答用户的问题。"
        )

        # ----------------------------------------------------------------
        # 3. 构建并调用 Agent（新 API）
        # ----------------------------------------------------------------
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

        agent_output: str = result["messages"][-1].content

        # ----------------------------------------------------------------
        # 4. 使用 Schema 解析分析结果
        # ----------------------------------------------------------------
        analysis_results: dict[str, Any] = extract_json_with_schema(
            agent_output, ANALYST_RESULTS_SCHEMA
        )

        # ----------------------------------------------------------------
        # 5. 检查数据质量标记
        # ----------------------------------------------------------------
        quality_note = analysis_results.get("data_quality_note")
        quality_flag = ""
        if quality_note and quality_note.strip():
            quality_flag = f" 数据质量提示: {quality_note}"

        summary: str = analysis_results.get("summary", "分析完成")
        findings_count: int = len(analysis_results.get("findings", []))

        return {
            "analysis_results": analysis_results,
            "messages": [
                {
                    "role": "analyst",
                    "content": (
                        f"分析完成。摘要: {summary}。"
                        f"共发现 {findings_count} 项关键发现。{quality_flag}"
                    ),
                }
            ],
            "current_agent": "analyst",
        }

    except Exception as exc:
        logger.exception("Analyst agent failed: %s", exc)
        return {
            "analysis_results": {"summary": f"分析失败: {exc}", "findings": [], "statistics": {}},
            "messages": [
                {
                    "role": "analyst",
                    "content": f"数据分析失败: {exc}",
                }
            ],
            "errors": [f"[analyst] {exc}"],
            "current_agent": "analyst",
        }
