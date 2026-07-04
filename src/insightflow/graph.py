"""InsightFlow 的 LangGraph 工作流定义。

本模块定义主执行图，负责编排所有 Agent。
图实现了以下状态机:

    - 线性流程: Scout → Cleaner Plan → Cleaner Execute → Analyst → Visualizer → Reporter
    - 条件边: Analyst 可根据质量评分路由回 Cleaner 重新清洗
    - 健康检查边: 每个 Agent 执行后检查是否有致命错误
    - 人工介入: 清洗计划需用户批准后才执行
    - 迭代上限: 质量检查循环受 config.max_iterations 限制
    - 收敛检测: 如果质量不再提升，提前停止重新清洗

图的拓扑结构 (Mermaid):

    ```mermaid
    flowchart TD
        START --> scout
        scout --> health_check_1{healthy?}
        health_check_1 -->|yes| cleaner_plan
        health_check_1 -->|fatal| END
        cleaner_plan --> human_review
        human_review --> cleaner_execute
        cleaner_execute --> analyst
        analyst --> quality_gate{quality >= threshold?}
        quality_gate -->|yes| visualizer
        quality_gate -->|no & iter < max & improving| cleaner_plan
        quality_gate -->|no & not improving| visualizer
        visualizer --> reporter
        reporter --> END
    ```

v2 变更:
- 结构化质量评分替代脆弱的关键词匹配
- 配置通过 AgentState 传递（不再硬编码魔法数字）
- 错误传播，支持 fatal/degraded 路由
- 基于 quality_history 的收敛检测
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from insightflow.agents import (
    analyst_node,
    cleaner_execute_node,
    cleaner_plan_node,
    reporter_node,
    scout_node,
    visualizer_node,
)
from insightflow.errors import AgentError, ErrorPropagator
from insightflow.observability.tracer import PipelineTracer, trace_node
from insightflow.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 错误传播器实例（整个 InsightFlow 运行期间共享）
# ---------------------------------------------------------------------------

_error_propagator = ErrorPropagator()


# ---------------------------------------------------------------------------
# 条件路由函数 —— 决定图的分支走向
# ---------------------------------------------------------------------------


def _check_data_quality(state: AgentState) -> Literal["quality_ok", "needs_reclean"]:
    """用结构化评分来评估分析结果，替代原来的关键词匹配。

    用 eval/metrics.py 的多维度评分替代了旧的关键词检查。
    质量阈值从 config 读取（不硬编码），收敛检测避免在质量不再提升时白白重洗。

    决策逻辑:
    1. 用 evaluate_analysis() 给分析结果打分
    2. 跟 config.quality_threshold 比较
    3. 检查 config.max_iterations 迭代上限
    4. 收敛检测: 当前分数 <= 上轮分数就提前停止
    5. 返回路由决策

    Args:
        state: 当前流程状态，包含 analysis_results 和 config。

    Returns:
        "quality_ok" 继续到 Visualizer，"needs_reclean" 回到 Cleaner。
    """
    results = state.get("analysis_results", {})
    iteration = state.get("iteration", 0)
    config = state.get("config", {})
    quality_history = state.get("quality_history", [])

    max_iterations = config.get("max_iterations", 2)
    quality_threshold = config.get("quality_threshold", 0.6)

    # --- 结构化质量评分（替代关键词匹配）---
    from insightflow.eval.metrics import evaluate_analysis

    score = evaluate_analysis(results)
    current_score = score.overall_score

    logger.info(
        "Quality check: score=%.2f (threshold=%.2f), iteration=%d/%d",
        current_score,
        quality_threshold,
        iteration + 1,
        max_iterations,
    )

    # 质量达标 —— 继续往下走
    if current_score >= quality_threshold:
        logger.info("Quality OK (%.2f >= %.2f), proceeding to visualizer", current_score, quality_threshold)
        return "quality_ok"

    # 超过迭代上限 —— 不再重洗了
    if iteration >= max_iterations:
        logger.info(
            "Quality below threshold (%.2f < %.2f) but iteration limit reached (%d/%d), "
            "proceeding with degraded quality",
            current_score, quality_threshold, iteration + 1, max_iterations,
        )
        return "quality_ok"

    # 收敛检测: 质量不再提升就停
    if quality_history and current_score <= quality_history[-1]:
        logger.info(
            "Quality not improving (%.2f <= %.2f previous), stopping re-clean loop",
            current_score, quality_history[-1],
        )
        return "quality_ok"

    # 质量不达标且还有改善空间 —— 回去重洗
    logger.info(
        "Quality below threshold (%.2f < %.2f), routing back to cleaner (iteration %d)",
        current_score, quality_threshold, iteration + 1,
    )
    return "needs_reclean"


def _increment_iteration(state: AgentState) -> dict[str, Any]:
    """迭代计数加一，同时记录质量分数用于收敛检测。"""
    current = state.get("iteration", 0)
    quality_history = list(state.get("quality_history", []))

    # 记录当前分析的质量分数
    results = state.get("analysis_results", {})
    from insightflow.eval.metrics import evaluate_analysis

    score = evaluate_analysis(results)
    quality_history.append(score.overall_score)

    logger.info(
        "Incrementing iteration: %d -> %d (quality_history: %s)",
        current,
        current + 1,
        [f"{s:.2f}" for s in quality_history],
    )
    return {
        "iteration": current + 1,
        "quality_history": quality_history,
    }


# ---------------------------------------------------------------------------
# 健康检查 —— 错误传播路由
# ---------------------------------------------------------------------------


def _check_health(state: AgentState) -> Literal["healthy", "fatal"]:
    """检查累积错误，判断流程能否继续。

    用 ErrorPropagator 对错误分类，看有没有致命故障需要中止流程。

    Args:
        state: 当前流程状态，包含 errors 列表。

    Returns:
        "healthy" 继续执行，"fatal" 中止。
    """
    raw_errors = state.get("errors", [])
    if not raw_errors:
        return "healthy"

    # 把错误字符串解析成 AgentError 对象，方便结构化评估
    agent_errors: list[AgentError] = []
    for err_str in raw_errors:
        # 错误格式: "[agent_name] message"
        if err_str.startswith("[") and "]" in err_str:
            bracket_end = err_str.index("]")
            agent_name = err_str[1:bracket_end]
            message = err_str[bracket_end + 2:]
            agent_errors.append(AgentError(
                agent_name=agent_name,
                error_type="runtime_error",
                message=message,
                severity="fatal" if agent_name in {"scout"} else "degraded",
                recoverable=agent_name not in {"scout"},
            ))

    should_continue, reason = _error_propagator.should_continue(agent_errors)

    if not should_continue:
        logger.error("Health check FAILED: %s", reason)
        return "fatal"

    logger.info("Health check: degraded but continuing (%s)", reason)
    return "healthy"


# ---------------------------------------------------------------------------
# 人工介入节点
# ---------------------------------------------------------------------------


def _human_review_node(state: AgentState) -> dict[str, Any]:
    """清洗计划的人工审核检查点。

    支持两种模式:
    - **Web 模式** (``config.review_mode == "web"``): 用 LangGraph 的
      ``interrupt()`` 暂停图执行。清洗计划作为 interrupt 值发送，
      用户决策通过 ``Command(resume=...)`` 接收。需要在 ``compile_graph()``
      时传入 checkpointer。
    - **CLI 模式**（默认）: 用 ``input()`` 做交互式终端提示。

    展示 Cleaner 生成的清洗计划，用户可以批准、修改或拒绝。

    Args:
        state: 当前流程状态，包含 cleaning_plan。

    Returns:
        更新后的状态，清洗计划可能被修改。
    """
    config = state.get("config", {})
    review_mode = config.get("review_mode", "cli")

    # 配置禁用了人工审核就跳过
    if not config.get("human_review", True):
        return {
            "messages": [{"role": "human_review", "content": "人工审核已禁用 (auto模式)"}]
        }

    plan = state.get("cleaning_plan", {})

    if not plan or not plan.get("strategy"):
        logger.info("No cleaning plan found, skipping human review")
        return {"messages": [{"role": "human_review", "content": "跳过人工审核：无清洗计划"}]}

    # ------------------------------------------------------------------
    # Web 模式: 用 LangGraph interrupt() 实现暂停/恢复
    # ------------------------------------------------------------------
    if review_mode == "web":
        from langgraph.types import interrupt

        # 构造给前端的 interrupt 数据
        interrupt_payload = {
            "type": "human_review",
            "cleaning_plan": plan,
            "data_profile": state.get("data_profile", {}),
            "message": "请审核清洗策略，批准后将继续执行清洗。",
        }

        # interrupt() 在这里暂停图执行
        # 通过 Command(resume=value) 恢复时，返回该 value
        review_result = interrupt(interrupt_payload)

        # review_result 是前端通过 Command(resume=...) 传回来的
        # 预期格式: {"approved": bool, "feedback": str}
        if isinstance(review_result, dict):
            approved = review_result.get("approved", True)
            feedback = review_result.get("feedback", "")
        else:
            approved = bool(review_result)
            feedback = ""

        if not approved:
            logger.info("User rejected cleaning plan via web review")
            return {
                "cleaning_plan": {},
                "messages": [{
                    "role": "human_review",
                    "content": f"用户拒绝了清洗计划，跳过清洗步骤。反馈: {feedback}" if feedback else "用户拒绝了清洗计划，跳过清洗步骤",
                }],
            }

        logger.info("Cleaning plan approved via web review")
        return {
            "messages": [{
                "role": "human_review",
                "content": f"清洗计划已批准 (Web审核) {'| 反馈: ' + feedback if feedback else ''}",
            }]
        }

    # ------------------------------------------------------------------
    # CLI 模式: 交互式终端提示
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🔍 清洗策略审核 (Cleaning Plan Review)")
    print("=" * 60)

    strategy = plan.get("strategy", [])
    notes = plan.get("overall_notes", "无")

    print(f"\n整体说明: {notes}\n")
    print("清洗步骤:")
    print("-" * 40)

    for i, step in enumerate(strategy, 1):
        print(f"  {i}. [{step.get('action', 'unknown')}] 列: {step.get('column', '?')}")
        print(f"     参数: {json.dumps(step.get('params', {}), ensure_ascii=False)}")
        print(f"     原因: {step.get('reason', '未说明')}")
        print()

    print("-" * 40)
    print("选项: [y] 批准执行  [n] 跳过清洗  [s] 跳过审核直接执行")

    try:
        choice = input("\n请选择 (y/n/s): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # 非交互模式: 自动批准
        choice = "y"

    if choice == "n":
        logger.info("User rejected cleaning plan")
        return {
            "cleaning_plan": {},
            "messages": [{"role": "human_review", "content": "用户拒绝了清洗计划，跳过清洗步骤"}],
        }

    logger.info("Cleaning plan approved (choice=%s)", choice)
    return {
        "messages": [{"role": "human_review", "content": f"清洗计划已批准 (选择: {choice})"}]
    }


# ---------------------------------------------------------------------------
# 图构建 —— 定义节点、边和条件路由
#
# 拓扑概览:
#   START → scout → [健康检查] → cleaner_plan → human_review → cleaner_execute
#   → analyst → [质量门控] → visualizer → reporter → END
#
#   质量门控: 评分达标走 visualizer，不达标且未收敛则走 increment_iteration → cleaner_plan 循环
#   健康检查: fatal 直接走 END，healthy 继续
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """构建 InsightFlow LangGraph 工作流。

    Returns:
        编译好的 StateGraph，可直接执行。

    节点:
        - scout: 数据探索与画像
        - cleaner_plan: 生成清洗策略
        - human_review: 用户审批清洗计划
        - cleaner_execute: 执行已批准的清洗操作
        - analyst: 统计分析
        - visualizer: 图表生成
        - reporter: 报告合成

    条件边:
        - scout 之后: 健康检查（fatal → END，healthy → 继续）
        - analyst 之后: 结构化质量门控
    """
    graph = StateGraph(AgentState)

    # --- 添加节点（Agent 节点用 tracing 包装）---
    graph.add_node("scout", trace_node("scout")(scout_node))
    graph.add_node("cleaner_plan", trace_node("cleaner_plan")(cleaner_plan_node))
    graph.add_node("human_review", _human_review_node)
    graph.add_node("cleaner_execute", trace_node("cleaner_execute")(cleaner_execute_node))
    graph.add_node("analyst", trace_node("analyst")(analyst_node))
    graph.add_node("visualizer", trace_node("visualizer")(visualizer_node))
    graph.add_node("reporter", trace_node("reporter")(reporter_node))
    graph.add_node("increment_iteration", _increment_iteration)

    # --- 线性边（固定顺序连接）---
    graph.add_edge(START, "scout")
    graph.add_edge("cleaner_plan", "human_review")
    graph.add_edge("human_review", "cleaner_execute")
    graph.add_edge("cleaner_execute", "analyst")
    graph.add_edge("visualizer", "reporter")
    graph.add_edge("reporter", END)

    # --- scout 之后的健康检查（致命错误 → 中止）---
    graph.add_conditional_edges(
        "scout",
        _check_health,
        {
            "healthy": "cleaner_plan",
            "fatal": END,
        },
    )

    # --- analyst 之后的结构化质量门控 ---
    # 评分达标 → visualizer；不达标 → increment_iteration（再循环回 cleaner_plan）
    graph.add_conditional_edges(
        "analyst",
        _check_data_quality,
        {
            "quality_ok": "visualizer",
            "needs_reclean": "increment_iteration",
        },
    )

    # 循环回来: increment_iteration → cleaner_plan（重新清洗一轮）
    graph.add_edge("increment_iteration", "cleaner_plan")

    return graph


def compile_graph(
    interrupt_before: list[str] | None = None,
    checkpointer: Any | None = None,
):
    """构建并编译 InsightFlow 图。

    Args:
        interrupt_before: 需要在执行前暂停的节点名列表。
                         传入 ``["human_review"]`` 可启用 Web 模式的人工审核。
        checkpointer: LangGraph checkpointer（如 ``MemorySaver``）。
                     使用 ``interrupt_before`` 时**必须传入**，因为
                     LangGraph 暂停时需要持久化状态。

    Returns:
        编译后的 LangGraph 图（CompiledStateGraph）。
    """
    graph = build_graph()
    return graph.compile(
        interrupt_before=interrupt_before,
        checkpointer=checkpointer,
    )


def get_mermaid_diagram() -> str:
    """生成 InsightFlow 工作流的 Mermaid 图。

    Returns:
        Mermaid 流程图字符串。
    """
    return """```mermaid
flowchart TD
    START((START)) --> scout[🔍 Scout Agent<br/>数据侦察]
    scout --> health_check{⚕️ Health Check<br/>错误检查}
    health_check -->|healthy| cleaner_plan[📋 Cleaner Plan<br/>制定清洗策略]
    health_check -->|fatal| END
    cleaner_plan --> human_review{👤 Human Review<br/>人工审核}
    human_review -->|批准| cleaner_execute[🧹 Cleaner Execute<br/>执行清洗]
    human_review -->|拒绝| analyst
    cleaner_execute --> analyst[📊 Analyst Agent<br/>统计分析]
    analyst -->|质量评分 >= 阈值| visualizer[📈 Visualizer Agent<br/>数据可视化]
    analyst -->|质量不足 &<br/>迭代 < max &<br/>仍在改善| increment[🔄 Increment<br/>迭代+1]
    analyst -->|质量不足但<br/>不再改善| visualizer
    increment --> cleaner_plan
    visualizer --> reporter[📝 Reporter Agent<br/>生成报告]
    reporter --> END((END))

    style scout fill:#e3f2fd,stroke:#1565c0
    style health_check fill:#fff9c4,stroke:#f9a825
    style cleaner_plan fill:#fff3e0,stroke:#e65100
    style human_review fill:#fce4ec,stroke:#c62828
    style cleaner_execute fill:#fff3e0,stroke:#e65100
    style analyst fill:#e8f5e9,stroke:#2e7d32
    style visualizer fill:#f3e5f5,stroke:#6a1b9a
    style reporter fill:#fce4ec,stroke:#ad1457
    style increment fill:#fff9c4,stroke:#f9a825
```"""


# ---------------------------------------------------------------------------
# 便捷运行入口
# ---------------------------------------------------------------------------


def run_pipeline(
    data_path: str,
    analysis_task: str,
    *,
    verbose: bool = True,
    enable_trace: bool = True,
    enable_eval: bool = True,
    output_dir: str = "output",
    max_iterations: int | None = None,
    human_review: bool = True,
) -> AgentState:
    """运行完整的 InsightFlow 流程，可选开启追踪和评估。

    这是执行数据分析流程的主入口。
    支持结构化执行追踪（可观测性）和输出质量评估（eval），
    两者可独立开关。

    Args:
        data_path: 要分析的 CSV 文件路径。
        analysis_task: 分析目标的自然语言描述。
        verbose: 是否打印进度信息。
        enable_trace: 是否启用执行追踪（计时、快照）。
        enable_eval: 是否完成后跑质量评估。
        output_dir: 追踪/评估导出和报告的保存目录。
        max_iterations: 覆盖质量检查的最大迭代次数（默认取 config）。
        human_review: 是否启用清洗计划的人工审核。

    Returns:
        所有 Agent 执行完毕后的最终 AgentState。
    """
    from pathlib import Path

    from insightflow.context.dataframe_context import new_context
    from insightflow.llm.token_tracker import TokenTracker
    from insightflow.state import create_initial_state

    # 设置日志
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # 构建配置字典，通过 state 传递给各 Agent
    config = {
        "max_iterations": max_iterations or 2,
        "quality_threshold": 0.6,
        "output_dir": output_dir,
        "human_review": human_review,
        "verbose": verbose,
    }

    # 创建初始状态（含配置）
    initial_state = create_initial_state(data_path, analysis_task, config=config)

    # 创建会话级 DataFrame 上下文
    ctx = new_context(session_id=initial_state["session_id"])

    # 初始化 token 追踪器
    token_tracker = TokenTracker()

    # 初始化执行追踪（如果启用）
    tracer: PipelineTracer | None = None
    if enable_trace:
        tracer = PipelineTracer()
        trace_id = tracer.start(config={
            "data_path": data_path,
            "analysis_task": analysis_task,
            "output_dir": output_dir,
            "session_id": initial_state["session_id"],
        })
        initial_state["trace_id"] = trace_id

    # 编译并运行图
    app = compile_graph()

    if verbose:
        print("=" * 60)
        print("🚀 InsightFlow Starting (v2)")
        print("=" * 60)
        print(f"📁 数据文件: {data_path}")
        print(f"📋 分析任务: {analysis_task}")
        print(f"🆔 会话 ID: {initial_state['session_id']}")
        if enable_trace:
            print(f"🔍 追踪 ID: {initial_state.get('trace_id', 'N/A')}")
        if enable_eval:
            print(f"📊 质量评估: 已启用")
        print(f"🔄 最大迭代: {config['max_iterations']}")
        print("=" * 60)

    # 执行图
    final_state = app.invoke(initial_state)

    # 完成追踪
    trace = None
    if tracer:
        trace = tracer.finish()

    # 运行质量评估
    eval_report = None
    if enable_eval:
        from insightflow.eval.report import generate_evaluation_report

        eval_report = generate_evaluation_report(final_state)

    # 导出结果
    if enable_trace and trace:
        from insightflow.observability.export import export_all

        export_all(trace, output_dir)

    if enable_eval and eval_report:
        from insightflow.eval.report import (
            export_evaluation_json,
            export_evaluation_markdown,
        )

        eval_dir = Path(output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        export_evaluation_json(eval_report, str(eval_dir / "evaluation.json"))
        export_evaluation_markdown(eval_report, str(eval_dir / "evaluation.md"))

    if verbose:
        print("\n" + "=" * 60)
        print("✅ InsightFlow Completed (v2)")
        print("=" * 60)
        print(f"📊 处理消息数: {len(final_state.get('messages', []))}")
        print(f"📈 生成图表数: {len(final_state.get('charts', []))}")
        print(f"🔄 迭代次数: {final_state.get('iteration', 0)}")
        quality_history = final_state.get("quality_history", [])
        if quality_history:
            print(f"📉 质量收敛: {' -> '.join(f'{s:.2f}' for s in quality_history)}")
        if final_state.get("errors"):
            print(f"⚠️  错误数: {len(final_state['errors'])}")
        if final_state.get("report"):
            print(f"📝 报告已生成")

        # 打印 DataFrame 上下文历史
        ctx_history = ctx.get_history()
        if ctx_history:
            print(f"\n📦 DataFrame 上下文历史:")
            for record in ctx_history:
                print(f"   v{record['version']} [{record['label']}] shape={record['shape']}")

        # 打印追踪摘要
        if trace:
            print("\n" + "-" * 60)
            print("🔍 执行追踪摘要")
            print("-" * 60)
            print(f"   追踪 ID: {trace.trace_id}")
            print(f"   总耗时: {trace.total_duration_ms:.0f}ms")
            print(f"   Agent 节点数: {trace.summary.get('total_spans', 0)}")
            print(f"   成功: {trace.summary.get('successful_spans', 0)} | "
                  f"失败: {trace.summary.get('failed_spans', 0)}")
            print(f"   最慢节点: {trace.summary.get('slowest_node', 'N/A')}")
            print(f"   最快节点: {trace.summary.get('fastest_node', 'N/A')}")
            node_durations = trace.summary.get("node_durations", {})
            if node_durations:
                print("   各节点耗时:")
                for node, dur in node_durations.items():
                    bar = "█" * int(dur / max(max(node_durations.values()), 1) * 20)
                    print(f"     {node:<20s} {bar} {dur:.0f}ms")

        # 打印评估摘要
        if eval_report:
            pm = eval_report.pipeline_metrics
            grade_emoji = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
            print("\n" + "-" * 60)
            print("📊 质量评估摘要")
            print("-" * 60)
            grade = pm.get("grade", "N/A")
            emoji = grade_emoji.get(grade, "⚪")
            print(f"   InsightFlow 评分: {pm.get('pipeline_score', 0):.1%} {emoji} (等级: {grade})")
            print(f"   Agent 平均分: {pm.get('avg_agent_score', 0):.1%}")
            print("   各 Agent 评分:")
            for score in eval_report.scores:
                bar = "█" * int(score.overall_score * 10) + "░" * (10 - int(score.overall_score * 10))
                print(f"     {score.agent_name:<14s} {bar} {score.overall_score:.0%} ({score.grade})")
            if eval_report.recommendations:
                print("   改进建议:")
                for rec in eval_report.recommendations[:3]:
                    print(f"     → {rec[:80]}...")

        print("=" * 60)

    return final_state
