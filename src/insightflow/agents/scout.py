"""
数据侦察员代理（Scout Agent）-- InsightFlow中的第一个代理。

负责：
1. 加载用户指定的 CSV 数据文件
2. 使用 MCP 工具探索数据结构、质量和特征
3. 生成结构化的数据画像（data profile）JSON

通过 MCP 数据探索工具（load_csv、get_schema、sample_rows、profile、
safe_query）全面了解数据，再交给下游的清洗和分析师代理处理。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from insightflow.config import get_chat_model
from insightflow.context import get_context
from insightflow.llm.resilient_client import ResilientLLMClient
from insightflow.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 InsightFlow中的数据侦察员（Scout Agent）。
你的任务是：
1. 加载用户指定的 CSV 数据文件
2. 全面探索数据的结构、质量和特征
3. 生成一份结构化的数据画像（data profile）

你需要使用数据探索工具来了解数据的全貌。重点关注：
- 数据的基本信息（行数、列数、数据类型）
- 缺失值情况（哪些列有缺失，缺失比例）
- 异常值线索（数值列的极端值）
- 分类列的分布情况
- 数值列的统计摘要

输出格式要求：将数据画像以 JSON 格式返回，包含以下字段：
- basic_info: {rows, columns, memory_usage}
- column_details: 每列的 {dtype, non_null_count, null_count, unique_count}
- numeric_summary: 数值列的统计摘要
- quality_issues: 发现的数据质量问题列表
- categorical_summary: 分类列的 top-5 值分布
"""


def scout_node(state: AgentState) -> dict:
    """LangGraph 节点函数：数据侦察员。

    加载 CSV 数据，使用 MCP 工具探索数据结构和质量，生成结构化数据画像，
    并将 DataFrame 注册到共享模块中供后续代理使用。

    参数:
        state: 当前系统状态，至少包含 ``data_path`` 和 ``analysis_task``。

    返回:
        部分状态字典，包含以下键：
        - data_profile: 结构化数据画像字典
        - messages: 本代理的日志消息
        - current_agent: 当前代理名称 ("scout")
    """
    try:
        # ----------------------------------------------------------------
        # 1. 初始化 LLM 和工具
        # ----------------------------------------------------------------
        llm = get_chat_model()
        resilient_llm = ResilientLLMClient(llm)

        from insightflow.data_mcp.mcp_bridge import create_sync_mcp_tools

        tools = create_sync_mcp_tools(role="scout")

        # ----------------------------------------------------------------
        # 2. 准备输入上下文
        # ----------------------------------------------------------------
        data_path: str = state.get("data_path", "")
        analysis_task: str = state.get("analysis_task", "")

        human_input = (
            f"请加载并分析以下数据文件：{data_path}\n"
            f"用户的分析目标是：{analysis_task}\n"
            f"请使用工具全面探索数据，然后输出 JSON 格式的数据画像。"
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

        # 提取最后一条消息作为 Agent 输出
        agent_output: str = result["messages"][-1].content

        # ----------------------------------------------------------------
        # 4. 从 Agent 响应中解析数据画像
        # ----------------------------------------------------------------
        from insightflow.utils.json_parser import (
            SCOUT_PROFILE_SCHEMA,
            extract_json_with_schema,
        )

        data_profile: dict[str, Any] = extract_json_with_schema(
            agent_output, SCOUT_PROFILE_SCHEMA
        )

        # ----------------------------------------------------------------
        # 5. 加载 CSV 到 pandas 并注册到 DataFrame 上下文
        # ----------------------------------------------------------------
        df = pd.read_csv(data_path)

        ctx = get_context()
        ctx.load(df, label="scout_load")

        # ----------------------------------------------------------------
        # 6. 返回部分状态
        # ----------------------------------------------------------------
        # 注意：DataFrame 不存入 state，以保持 state 的 msgpack 可序列化性
        # （供 LangGraph checkpointer 使用）。DataFrame 存放在 DataFrameContext 中，
        # 可通过 get_dataframe() 访问。
        return {
            "data_profile": data_profile,
            "messages": [
                {
                    "role": "scout",
                    "content": (
                        f"数据侦察完成。已加载 {df.shape[0]} 行 x {df.shape[1]} 列的数据。"
                        f"生成了数据画像，发现 {len(data_profile.get('quality_issues', []))} 个质量问题。"
                    ),
                }
            ],
            "current_agent": "scout",
        }

    except Exception as exc:
        logger.exception("Scout agent failed: %s", exc)
        return {
            "data_profile": {"error": str(exc)},
            "messages": [
                {
                    "role": "scout",
                    "content": f"数据侦察失败: {exc}",
                }
            ],
            "errors": [f"[scout] {exc}"],
            "current_agent": "scout",
        }
