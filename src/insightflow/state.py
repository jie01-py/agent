"""InsightFlow 多 Agent 流程的共享状态定义。

AgentState 是核心数据结构，在 LangGraph 工作流的所有 Agent 之间流转。
每个 Agent 读取和写入 AgentState 的特定字段，实现结构化的 Agent 间通信。

v2: 新增配置传递、quality_history（收敛检测）、session_id（DataFrame 上下文隔离）
和 token_usage 追踪。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """LangGraph 流程中的共享状态。

    设计思路:
    - 使用 TypedDict 获得静态类型检查和 IDE 支持
    - `messages` 通过 operator.add 自动追加，各 Agent 的消息是追加而非覆盖
    - `errors` 同样用 operator.add 做错误日志累积
    - `iteration` 记录质量检查循环的当前轮次
    - `config` 把流程配置带在图里传递（v2）
    - `quality_history` 记录每轮迭代的质量评分（v2）
    - `session_id` 关联 DataFrameContext 做隔离（v2）
    - `token_usage` 累积各 Agent 的 token 追踪数据（v2）

    Fields:
        data_path: 输入 CSV 文件的路径
        dataframe: 当前 DataFrame（由 Cleaner 修改）
        data_profile: Scout 生成的结构化数据画像
        analysis_task: 分析目标的自然语言描述
        cleaning_plan: Cleaner 生成的清洗策略字典
        analysis_results: Analyst 的统计分析结果
        charts: 生成的图表文件路径列表
        report: Reporter 生成的最终 Markdown 报告
        messages: Agent 间通信日志（累积）
        errors: 错误日志（累积）
        current_agent: 当前正在执行的 Agent 节点名称
        iteration: 质量检查循环的当前迭代次数
        trace_id: 执行追踪的唯一标识
        config: 流程配置字典（max_iterations、output_dir 等）
        quality_history: 每轮迭代的质量评分列表
        session_id: DataFrameContext 隔离用的会话标识
        token_usage: 各 Agent 的 token 用量追踪字典
    """

    # 数据流
    data_path: str
    dataframe: Any  # pd.DataFrame（不可序列化，TypedDict 用 Any）
    data_profile: dict[str, Any]

    # 分析流
    analysis_task: str
    cleaning_plan: dict[str, Any]
    analysis_results: dict[str, Any]
    charts: list[str]
    report: str

    # 元信息
    messages: Annotated[list[dict[str, str]], operator.add]
    errors: Annotated[list[str], operator.add]
    current_agent: str
    iteration: int
    trace_id: str

    # --- v2 新增 ---
    config: dict[str, Any]
    quality_history: list[float]
    session_id: str
    token_usage: dict[str, Any]


def create_initial_state(
    data_path: str,
    analysis_task: str,
    config: dict[str, Any] | None = None,
) -> AgentState:
    """创建流程的初始状态。

    Args:
        data_path: 要分析的 CSV 文件路径。
        analysis_task: 分析目标的自然语言描述。
        config: 可选的流程配置字典，为 None 时使用默认值。

    Returns:
        初始化好的 AgentState 字典。
    """
    import uuid

    default_config = {
        "max_iterations": 2,
        "quality_threshold": 0.6,
        "output_dir": "output",
        "chart_format": "png",
        "chart_dpi": 150,
        "human_review": True,
        "verbose": True,
    }
    if config:
        default_config.update(config)

    return AgentState(
        data_path=data_path,
        dataframe=None,
        data_profile={},
        analysis_task=analysis_task,
        cleaning_plan={},
        analysis_results={},
        charts=[],
        report="",
        messages=[],
        errors=[],
        current_agent="scout",
        iteration=0,
        trace_id="",
        config=default_config,
        quality_history=[],
        session_id=uuid.uuid4().hex[:8],
        token_usage={},
    )
