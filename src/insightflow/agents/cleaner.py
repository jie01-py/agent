"""
数据清洁工代理（Cleaner Agent）-- 负责数据质量改进。

分为两个 LangGraph 节点：

1. ``cleaner_plan_node``  -- 分析数据画像，生成结构化清洗计划（JSON），
   说明要清洗哪些列、执行什么操作以及原因。

2. ``cleaner_execute_node`` -- 读取清洗计划，逐步调用对应的数据工具函数
   （fill_missing、remove_outliers、normalize_column）执行清洗。

这种拆分设计支持在计划和执行之间加入人工审核环节。
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
from insightflow.utils.json_parser import CLEANER_PLAN_SCHEMA, extract_json_with_schema

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 InsightFlow中的数据清洁工（Cleaner Agent）。
你的任务是：
1. 根据数据画像中暴露的质量问题，制定清洗策略
2. 执行清洗操作

制定清洗策略时，请输出 JSON 格式的清洗计划：
{
  "strategy": [
    {"column": "列名", "action": "操作类型", "params": {}, "reason": "原因"}
  ],
  "overall_notes": "整体说明"
}

可用操作类型：fill_missing, remove_outliers, normalize_column
strategy 参数：
- fill_missing: strategy (mean/median/mode/drop/zero)
- remove_outliers: method (iqr/zscore), threshold (float)
- normalize_column: method (minmax/standard)

规则：
- 只处理有问题的列，不要过度清洗
- 优先保留数据，只有在必要时才删除行
- 解释每个清洗决策的原因
"""


def cleaner_plan_node(state: AgentState) -> dict:
    """LangGraph 节点函数：清洗计划生成。

    LLM 审查 Scout 代理提供的数据画像，生成结构化 JSON 清洗计划，
    说明需要执行哪些清洗操作。

    参数:
        state: 当前系统状态，必须包含 ``data_profile``。

    返回:
        部分状态字典，包含以下键：
        - cleaning_plan: 结构化清洗计划字典
        - messages: 本阶段的日志消息
        - current_agent: "cleaner_plan"
    """
    try:
        llm = get_chat_model()
        resilient_llm = ResilientLLMClient(llm)

        from insightflow.tools.data_tools import (
            describe_numeric,
            get_dataframe_info,
        )

        tools = [get_dataframe_info, describe_numeric]

        data_profile = state.get("data_profile", {})
        profile_text = json.dumps(data_profile, ensure_ascii=False, indent=2, default=str)

        human_input = (
            f"以下是数据画像：\n{profile_text}\n\n"
            f"请根据画像中暴露的质量问题，制定一份 JSON 格式的清洗计划。"
            f"只输出 JSON，不要添加其他说明。"
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

        raw_output: str = result["messages"][-1].content

        cleaning_plan: dict[str, Any] = extract_json_with_schema(
            raw_output, CLEANER_PLAN_SCHEMA
        )

        return {
            "cleaning_plan": cleaning_plan,
            "messages": [
                {
                    "role": "cleaner",
                    "content": (
                        f"清洗计划已生成。共 {len(cleaning_plan.get('strategy', []))} 项清洗操作。"
                    ),
                }
            ],
            "current_agent": "cleaner_plan",
        }

    except Exception as exc:
        logger.exception("Cleaner plan phase failed: %s", exc)
        return {
            "cleaning_plan": {"strategy": [], "error": str(exc)},
            "messages": [
                {
                    "role": "cleaner",
                    "content": f"清洗计划生成失败: {exc}",
                }
            ],
            "errors": [f"[cleaner_plan] {exc}"],
            "current_agent": "cleaner_plan",
        }


def _execute_single_step(step: dict[str, Any]) -> str:
    """执行单个清洗步骤，调度到对应的数据工具函数。

    参数:
        step: 包含 ``column``、``action`` 和 ``params`` 键的字典。

    返回:
        描述操作结果的字符串。
    """
    from insightflow.tools.data_tools import (
        fill_missing,
        normalize_column,
        remove_outliers,
    )

    column: str = step.get("column", "")
    action: str = step.get("action", "")
    params: dict[str, Any] = step.get("params", {})

    if action == "fill_missing":
        strategy = params.get("strategy", "mean")
        return fill_missing.invoke({"column": column, "strategy": strategy})

    if action == "remove_outliers":
        method = params.get("method", "iqr")
        threshold = params.get("threshold", 1.5)
        return remove_outliers.invoke(
            {"column": column, "method": method, "threshold": float(threshold)}
        )

    if action == "normalize_column":
        method = params.get("method", "minmax")
        return normalize_column.invoke({"column": column, "method": method})

    return f"Unknown action '{action}' for column '{column}'. Skipped."


def cleaner_execute_node(state: AgentState) -> dict:
    """LangGraph 节点函数：清洗计划执行。

    遍历清洗计划中的 strategy 列表，逐步调用对应的数据工具函数执行清洗。
    所有步骤完成后，从共享模块中获取更新后的 DataFrame。

    参数:
        state: 当前系统状态，必须包含 ``cleaning_plan``
               以及先前通过 ``set_dataframe`` 设置的 DataFrame。

    返回:
        部分状态字典，包含以下键：
        - messages: 每个清洗步骤的执行日志
        - current_agent: "cleaner_execute"
    """
    try:
        from insightflow.tools.data_tools import get_dataframe, set_dataframe

        cleaning_plan: dict[str, Any] = state.get("cleaning_plan", {})
        strategy_list: list[dict[str, Any]] = cleaning_plan.get("strategy", [])

        execution_log: list[str] = []

        for i, step in enumerate(strategy_list, start=1):
            column = step.get("column", "unknown")
            action = step.get("action", "unknown")
            reason = step.get("reason", "")

            result_text = _execute_single_step(step)
            log_entry = f"[Step {i}] {action} on '{column}': {result_text}"
            execution_log.append(log_entry)
            logger.info(log_entry)

        df = get_dataframe()

        if df is not None:
            set_dataframe(df)

        return {
            # DataFrame 不存入 state（不支持 msgpack 序列化）。
            # 存放在 DataFrameContext 中，可通过 get_dataframe() 访问。
            "messages": [
                {
                    "role": "cleaner",
                    "content": (
                        f"清洗执行完成。共执行 {len(execution_log)} 步操作：\n"
                        + "\n".join(execution_log)
                    ),
                }
            ],
            "current_agent": "cleaner_execute",
        }

    except Exception as exc:
        logger.exception("Cleaner execute phase failed: %s", exc)
        return {
            "messages": [
                {
                    "role": "cleaner",
                    "content": f"清洗执行失败: {exc}",
                }
            ],
            "errors": [f"[cleaner_execute] {exc}"],
            "current_agent": "cleaner_execute",
        }
