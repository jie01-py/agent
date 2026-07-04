"""InsightFlow 后端服务 — 多 Agent 数据分析系统。

提供以下 REST API 功能：
- CSV 文件上传
- InsightFlow 执行（后台线程异步运行）
- 实时状态轮询
- 结果获取（报告、追踪、评估、图表）
- 文件下载

启动方式：
    python -m insightflow.server
    # 或者
    uvicorn insightflow.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

# InsightFlow 各阶段定义，用于进度跟踪（名称、中文标签、权重）
STAGES = [
    ("upload", "上传数据", 5),
    ("scout", "数据侦察", 15),
    ("health_check", "健康检查", 2),
    ("cleaner_plan", "制定清洗策略", 15),
    ("human_review", "审核清洗策略", 2),
    ("cleaner_execute", "执行清洗", 10),
    ("analyst", "统计分析", 20),
    ("visualizer", "数据可视化", 15),
    ("reporter", "生成报告", 15),
    ("done", "完成", 1),
]

STAGE_NAMES = [s[0] for s in STAGES]
STAGE_PROGRESS = {s[0]: s[2] for s in STAGES}


@dataclass
class SessionInfo:
    """记录单次 InsightFlow 执行会话的全部状态。"""

    session_id: str
    status: str = "idle"  # idle | running | completed | failed | waiting_review
    current_stage: str = "upload"
    progress: int = 0  # 0-100
    data_path: str = ""
    data_filename: str = ""
    analysis_task: str = ""
    output_dir: str = ""
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    # 人工审核相关字段
    review_approved: bool | None = None
    review_feedback: str = ""

    # 用于 interrupt/resume 的 checkpoint 信息
    thread_id: str = ""
    checkpoint_ns: str = ""
    checkpointer: Any = None  # MemorySaver 实例，支持 interrupt/resume 机制

    # 运行完成后填充的结果字段
    report: str = ""
    messages: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    quality_history: list[float] = field(default_factory=list)
    data_profile: dict[str, Any] = field(default_factory=dict)
    cleaning_plan: dict[str, Any] = field(default_factory=dict)
    analysis_results: dict[str, Any] = field(default_factory=dict)

    def to_status_dict(self) -> dict:
        """返回状态摘要，供前端轮询使用。"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "current_stage_label": dict((s[0], s[1]) for s in STAGES).get(
                self.current_stage, self.current_stage
            ),
            "progress": self.progress,
            "data_filename": self.data_filename,
            "analysis_task": self.analysis_task,
            "error": self.error,
            "duration_ms": self.duration_ms,
            # 等待审核时，把清洗计划也带上
            "cleaning_plan": self.cleaning_plan if self.status == "waiting_review" else None,
            "review_feedback": self.review_feedback,
        }

    def to_results_dict(self) -> dict:
        """返回完整结果，包括报告、追踪、评估等全部字段。"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "report": self.report,
            "charts": self.charts,
            "trace": self.trace,
            "evaluation": self.evaluation,
            "iteration": self.iteration,
            "quality_history": self.quality_history,
            "data_profile": self.data_profile,
            "cleaning_plan": self.cleaning_plan,
            "analysis_results": self.analysis_results,
            "messages": self.messages,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


# 内存中的会话存储（单实例部署够用了）
_sessions: dict[str, SessionInfo] = {}
_sessions_lock = threading.Lock()

# 上传和输出用的临时目录
_base_tmp = Path(tempfile.gettempdir()) / "insightflow_web"
_base_tmp.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 通过监控 tracer 来跟踪进度
# ---------------------------------------------------------------------------

# Agent 节点名到 InsightFlow 阶段名的映射
_NODE_TO_STAGE = {
    "scout": "scout",
    "cleaner_plan": "cleaner_plan",
    "human_review": "human_review",
    "cleaner_execute": "cleaner_execute",
    "analyst": "analyst",
    "visualizer": "visualizer",
    "reporter": "reporter",
}

# 与 tracer span 对应的有序阶段列表
_SPAN_STAGES = ["scout", "cleaner_plan", "cleaner_execute", "analyst", "visualizer", "reporter"]


def _monitor_tracer(session: SessionInfo, stop_event: threading.Event) -> None:
    """后台线程：监控 tracer span 来更新会话进度。

    检查 PipelineTracer 中已完成的 span，判断当前正在执行哪个阶段。
    当某个 span 完成时，序列中的下一个阶段变为活跃状态。

    参数：
        session: 要更新的 SessionInfo 对象。
        stop_event: 用来通知线程停止的 Event。
    """
    last_span_count = 0

    while not stop_event.is_set():
        stop_event.wait(timeout=0.5)

        from insightflow.observability.tracer import get_tracer

        tracer = get_tracer()
        if tracer is None:
            continue

        spans = tracer._spans
        span_count = len(spans)

        if span_count > last_span_count:
            last_span_count = span_count

            # 找到最后一个已完成的阶段
            last_completed = spans[-1].node_name
            last_stage = _NODE_TO_STAGE.get(last_completed, last_completed)

            # 序列中的下一个阶段才是当前活跃阶段
            try:
                completed_idx = _SPAN_STAGES.index(last_completed)
                if completed_idx + 1 < len(_SPAN_STAGES):
                    next_stage = _SPAN_STAGES[completed_idx + 1]
                    session.current_stage = next_stage
                else:
                    session.current_stage = "reporter"  # 最后一个阶段
            except ValueError:
                session.current_stage = last_stage

            # 根据已完成 span 累计计算进度
            completed_stages = set()
            for span in spans:
                stage = _NODE_TO_STAGE.get(span.node_name)
                if stage:
                    completed_stages.add(stage)

            progress = 0
            for stage_name, _, weight in STAGES:
                if stage_name in completed_stages:
                    progress += weight
            session.progress = min(progress, 95)  # 最多到 95%，真正完成时才到 100


# ---------------------------------------------------------------------------
# InsightFlow 执行（在后台线程中运行）
# ---------------------------------------------------------------------------


def _run_pipeline_thread(session: SessionInfo, human_review: bool = False) -> None:
    """在后台线程中执行 InsightFlow 分析流程。

    当 ``human_review`` 为 True 时，InsightFlow 会在 ``_human_review_node`` 中
    通过 LangGraph 的 ``interrupt()`` 暂停执行，需要 MemorySaver checkpointer
    来保存暂停时的状态。

    参数：
        session: 用于更新进度和结果的 SessionInfo 对象。
        human_review: 是否启用人工审核（interrupt 模式）。
    """
    # 启动 tracer 监控线程，实时更新进度
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_tracer,
        args=(session, stop_monitor),
        daemon=True,
    )
    monitor_thread.start()

    try:
        session.status = "running"
        session.start_time = time.time()
        session.current_stage = "scout"
        session.progress = 5

        from langgraph.checkpoint.memory import MemorySaver

        from insightflow.context.dataframe_context import new_context
        from insightflow.graph import compile_graph
        from insightflow.state import create_initial_state

        # 构建运行配置 — review_mode="web" 会在 human_review_node 里触发 interrupt()
        config = {
            "max_iterations": 2,
            "quality_threshold": 0.6,
            "output_dir": session.output_dir,
            "chart_format": "png",
            "chart_dpi": 150,
            "human_review": human_review,
            "review_mode": "web" if human_review else "cli",
            "verbose": True,
        }

        # 创建初始状态
        initial_state = create_initial_state(
            data_path=session.data_path,
            analysis_task=session.analysis_task,
            config=config,
        )

        # 创建会话级别的 DataFrameContext（线程局部存储）。
        # new_context 会把上下文绑定到当前线程，后续同一线程内的节点可以通过
        # get_context() 拿到这个 DataFrame，无需显式传递。
        ctx = new_context(session_id=initial_state["session_id"])

        # 初始化 tracer，用于追踪执行过程
        from insightflow.observability.tracer import PipelineTracer

        tracer = PipelineTracer()
        trace_id = tracer.start(config={
            "data_path": session.data_path,
            "analysis_task": session.analysis_task,
            "session_id": session.session_id,
        })
        initial_state["trace_id"] = trace_id

        # 用 thread_id 作为 checkpoint 的键
        session.thread_id = session.session_id

        # 创建 MemorySaver checkpointer（用于 interrupt/resume 时持久化状态）
        checkpointer = MemorySaver() if human_review else None
        session.checkpointer = checkpointer

        # 编译图并绑定 checkpointer（不需要 interrupt_before，
        # 因为 interrupt() 是在 _human_review_node 内部主动调用的）
        app = compile_graph(checkpointer=checkpointer)

        # 运行配置，带上 thread_id 用于 checkpoint 定位
        run_config = {"configurable": {"thread_id": session.thread_id}}

        # 执行图
        final_state = app.invoke(initial_state, config=run_config)

        # 到这里 InsightFlow 要么跑完了、要么在 interrupt 处暂停了。
        # 赶紧停掉监控线程，免得它覆盖 current_stage。
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        # ------------------------------------------------------------------
        # interrupt 检测逻辑：
        # graph_state.tasks 是一个 PregelTask 元组，每个 task 可能带有
        # 'interrupts' 属性（interrupt 列表）。只要任一 task 存在非空
        # 的 interrupts，就说明图在某个节点处暂停了，等待人工介入。
        # ------------------------------------------------------------------
        if human_review and checkpointer is not None:
            graph_state = app.get_state(run_config)

            has_interrupt = False
            for task in graph_state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    has_interrupt = True
                    break

            if has_interrupt:
                # 从当前图状态中提取清洗计划
                current_values = graph_state.values if hasattr(graph_state, 'values') else {}
                cleaning_plan = current_values.get("cleaning_plan", {})
                data_profile = current_values.get("data_profile", {})

                session.status = "waiting_review"
                session.current_stage = "human_review"
                session.progress = 30
                session.data_profile = _safe_json(data_profile)
                session.cleaning_plan = _safe_json(cleaning_plan)
                session.messages = current_values.get("messages", [])
                logger.info("InsightFlow 暂停等待人工审核，会话 %s", session.session_id)
                return  # 退出线程，等待 /api/review 来恢复

        # 没有 interrupt，说明 InsightFlow 跑完了
        _process_final_state(session, final_state)

        # 导出追踪和评估数据
        _export_trace_and_eval(session, tracer, final_state)

    except Exception as exc:
        session.end_time = time.time()
        session.duration_ms = (session.end_time - session.start_time) * 1000
        session.status = "failed"
        session.error = str(exc)
        logger.exception("InsightFlow 执行失败，会话 %s: %s", session.session_id, exc)

    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)


def _export_trace_and_eval(session: SessionInfo, tracer, final_state: dict) -> None:
    """导出追踪和评估数据，并回载到会话对象中。

    完成 tracer 记录，调用 export_all 写出追踪数据，再生成评估报告
    （JSON + Markdown），最后把生成的文件读回 session.trace 和
    session.evaluation，方便前端直接获取。

    参数：
        session: 当前会话信息。
        tracer: PipelineTracer 实例。
        final_state: InsightFlow 最终状态。
    """
    try:
        trace = tracer.finish()

        from insightflow.observability.export import export_all

        export_all(trace, session.output_dir)

        from insightflow.eval.report import (
            export_evaluation_json,
            export_evaluation_markdown,
            generate_evaluation_report,
        )

        eval_report = generate_evaluation_report(final_state)
        eval_dir = Path(session.output_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        export_evaluation_json(eval_report, str(eval_dir / "evaluation.json"))
        export_evaluation_markdown(eval_report, str(eval_dir / "evaluation.md"))

        # 把写好的 trace/eval 文件重新加载到会话中
        trace_files = list(eval_dir.glob("trace_*.json"))
        if trace_files:
            with open(trace_files[0], encoding="utf-8") as f:
                session.trace = json.load(f)

        eval_file = eval_dir / "evaluation.json"
        if eval_file.exists():
            with open(eval_file, encoding="utf-8") as f:
                session.evaluation = json.load(f)
    except Exception as trace_err:
        logger.warning("导出追踪/评估数据失败: %s", trace_err)


def _resume_pipeline_thread(session: SessionInfo, approved: bool, feedback: str = "") -> None:
    """恢复人工审核后的 InsightFlow 执行流程。

    复用首次运行时的 MemorySaver checkpointer，通过 ``Command(resume=...)``
    把用户的审核决定传回图中。

    如果质量门触发了重新清洗，图会再次循环经过 ``human_review`` 节点并
    调用 ``interrupt()``。这里会检测第二次 interrupt 并将会话状态
    重新设为 ``waiting_review``。

    之前因为没传 checkpointer 就调 ``invoke(None, ...)`` 导致
    ``EmptyInputError``，这里已修复。

    参数：
        session: SessionInfo 对象。
        approved: 是否批准了清洗计划。
        feedback: 用户对清洗计划的反馈意见。
    """
    # 启动 tracer 监控线程，恢复期间也实时更新进度
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_tracer,
        args=(session, stop_monitor),
        daemon=True,
    )
    monitor_thread.start()

    try:
        session.status = "running"
        session.review_approved = approved
        session.review_feedback = feedback
        session.current_stage = "cleaner_execute"
        session.progress = 35

        import pandas as pd
        from langgraph.types import Command

        from insightflow.context.dataframe_context import new_context
        from insightflow.graph import compile_graph
        from insightflow.observability.tracer import PipelineTracer

        # 重新创建 DataFrameContext（线程局部存储）。
        # 因为恢复是在新线程里执行的，原线程的 DataFrameContext 在这边
        # 访问不到，所以必须重新 new 一个并加载数据。
        ctx = new_context(session_id=session.session_id)
        df = pd.read_csv(session.data_path)
        ctx.load(df, label="resume_load")

        # 初始化 tracer，追踪恢复阶段的执行
        tracer = PipelineTracer()
        tracer.start(config={
            "phase": "resume",
            "session_id": session.session_id,
        })

        # 复用首次运行时的 checkpointer
        checkpointer = session.checkpointer
        if checkpointer is None:
            raise RuntimeError("找不到 checkpointer，无法恢复 InsightFlow 执行")

        # 用同一个 checkpointer 重新编译图
        app = compile_graph(checkpointer=checkpointer)

        # 运行配置，带上同一个 thread_id
        run_config = {"configurable": {"thread_id": session.thread_id}}

        # 用用户的审核决定来恢复 interrupt 处暂停的图
        resume_value = {"approved": approved, "feedback": feedback}
        resume_command = Command(resume=resume_value)

        logger.info(
            "恢复 InsightFlow 执行，会话 %s: approved=%s",
            session.session_id, approved,
        )

        # invoke(Command(resume=...), config=...) 从 checkpoint 处恢复执行
        final_state = app.invoke(resume_command, config=run_config)

        # 停掉监控线程，防止它覆盖 current_stage
        stop_monitor.set()
        monitor_thread.join(timeout=2)

        # ------------------------------------------------------------------
        # 检测是否有新的 interrupt：
        # 质量门可能判定清洗不合格，触发重新清洗，图会再次循环经过
        # human_review 节点并调用 interrupt()。这里检测到后就会话
        # 重新回到 waiting_review 状态，等待用户再次审核。
        # ------------------------------------------------------------------
        graph_state = app.get_state(run_config)
        has_interrupt = False
        for task in graph_state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                has_interrupt = True
                break

        if has_interrupt:
            # InsightFlow 又暂停了，等待下一轮人工审核
            current_values = graph_state.values if hasattr(graph_state, 'values') else {}
            cleaning_plan = current_values.get("cleaning_plan", {})
            data_profile = current_values.get("data_profile", {})

            session.status = "waiting_review"
            session.current_stage = "human_review"
            session.cleaning_plan = _safe_json(cleaning_plan)
            session.data_profile = _safe_json(data_profile)
            session.messages = current_values.get("messages", [])
            logger.info(
                "InsightFlow 再次暂停等待人工审核，会话 %s",
                session.session_id,
            )
            return  # 等待 /api/review 再次恢复

        # InsightFlow 跑完了，处理最终状态
        _process_final_state(session, final_state)

        # 导出恢复阶段的追踪和评估数据
        _export_trace_and_eval(session, tracer, final_state)

    except Exception as exc:
        session.end_time = time.time()
        session.duration_ms = (session.end_time - session.start_time) * 1000
        session.status = "failed"
        session.error = str(exc)
        logger.exception("InsightFlow 恢复执行失败，会话 %s: %s", session.session_id, exc)

    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=2)


def _process_final_state(session: SessionInfo, final_state: dict) -> None:
    """InsightFlow 跑完后，从最终状态中提取各项结果。

    参数：
        session: SessionInfo 对象。
        final_state: InsightFlow 执行结束后的 AgentState。
    """
    session.end_time = time.time()
    session.duration_ms = (session.end_time - session.start_time) * 1000
    session.progress = 100
    session.current_stage = "done"

    # 从最终状态中提取各项结果
    session.report = final_state.get("report", "")
    session.messages = final_state.get("messages", [])
    session.errors = final_state.get("errors", [])
    session.charts = final_state.get("charts", [])
    session.iteration = final_state.get("iteration", 0)
    session.quality_history = final_state.get("quality_history", [])
    session.data_profile = _safe_json(final_state.get("data_profile", {}))
    session.cleaning_plan = _safe_json(final_state.get("cleaning_plan", {}))
    session.analysis_results = _safe_json(final_state.get("analysis_results", {}))

    # 加载追踪数据
    trace_dir = Path(session.output_dir)
    trace_files = list(trace_dir.glob("trace_*.json"))
    if trace_files:
        with open(trace_files[0], encoding="utf-8") as f:
            session.trace = json.load(f)

    # 加载评估数据
    eval_file = trace_dir / "evaluation.json"
    if eval_file.exists():
        with open(eval_file, encoding="utf-8") as f:
            session.evaluation = json.load(f)

    if session.errors:
        session.status = "completed"  # 完成了，但有一些警告
    else:
        session.status = "completed"


def _safe_json(obj: Any) -> Any:
    """确保对象能被 JSON 序列化，不行的话就降级为 str()。"""
    try:
        json.dumps(obj, default=str)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InsightFlow",
    description="多 Agent 数据分析 InsightFlow API",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（前端页面）
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """返回单页前端应用。"""
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


class RunRequest(BaseModel):
    """启动 InsightFlow 运行的请求体。"""

    analysis_task: str
    data_filename: str = ""
    human_review: bool = True  # 默认启用人工审核


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """上传 CSV 文件，返回文件引用信息。

    参数：
        file: 上传的 CSV 文件。

    返回：
        包含 file_id 和 filename 的 JSON。
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件")

    file_id = uuid.uuid4().hex[:12]
    upload_dir = _base_tmp / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / f"{file_id}_{file.filename}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(save_path),
        "size": len(content),
    }


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    """启动一次新的 InsightFlow 执行。

    需要一个已上传的文件路径和分析任务描述。
    InsightFlow 在后台线程中运行，通过 /api/status 轮询进度。

    参数：
        req: RunRequest，包含 analysis_task 和 data_filename。

    返回：
        包含 session_id 的 JSON。
    """
    # 查找已上传的文件
    upload_dir = _base_tmp / "uploads"
    matching = list(upload_dir.glob(f"*_{req.data_filename}"))
    if not matching:
        # 找不到精确匹配的，就试试 uploads 目录下的任意 CSV
        all_csvs = list(upload_dir.glob("*.csv"))
        if not all_csvs:
            raise HTTPException(status_code=400, detail="没找到已上传的 CSV 文件，请先上传。")
        data_path = str(all_csvs[-1])
    else:
        data_path = str(matching[-1])

    session_id = uuid.uuid4().hex[:8]
    output_dir = str(_base_tmp / "outputs" / session_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    session = SessionInfo(
        session_id=session_id,
        data_path=data_path,
        data_filename=req.data_filename,
        analysis_task=req.analysis_task,
        output_dir=output_dir,
    )

    with _sessions_lock:
        _sessions[session_id] = session

    # 在后台线程中启动 InsightFlow
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(session, req.human_review),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id, "status": "running"}


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """轮询 InsightFlow 执行的当前状态。

    参数：
        session_id: 会话标识符。

    返回：
        包含 status、progress、current_stage 等信息的 JSON。
    """
    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_status_dict()


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """获取已完成执行的完整结果。

    参数：
        session_id: 会话标识符。

    返回：
        包含 report、trace、evaluation、charts 等的 JSON。
    """
    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status == "running":
        return {"status": "running", "message": "InsightFlow 仍在执行中"}

    return session.to_results_dict()


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """下载结果文件（报告、图表、追踪导出等）。

    参数：
        session_id: 会话标识符。
        filename: 要下载的文件名。

    返回：
        文件响应。
    """
    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 在输出目录中查找文件
    output_dir = Path(session.output_dir)

    # 先尝试直接匹配
    target = output_dir / filename
    if target.exists() and target.is_file():
        return FileResponse(
            path=str(target),
            filename=filename,
            media_type="application/octet-stream",
        )

    # 直接匹配找不到就递归搜索
    for found in output_dir.rglob(filename):
        if found.is_file():
            return FileResponse(
                path=str(found),
                filename=filename,
                media_type="application/octet-stream",
            )

    raise HTTPException(status_code=404, detail=f"文件 '{filename}' 未找到")


@app.get("/api/sessions")
async def list_sessions():
    """列出所有 InsightFlow 会话。

    返回：
        会话摘要的 JSON 数组。
    """
    with _sessions_lock:
        return [s.to_status_dict() for s in _sessions.values()]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其输出文件。

    参数：
        session_id: 会话标识符。
    """
    with _sessions_lock:
        session = _sessions.pop(session_id, None)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 清理输出目录
    output_dir = Path(session.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    return {"status": "deleted", "session_id": session_id}


class ReviewRequest(BaseModel):
    """人工审核决定的请求体。"""

    approved: bool
    feedback: str = ""


@app.post("/api/review/{session_id}")
async def submit_review(session_id: str, req: ReviewRequest):
    """提交人工审核决定（批准或拒绝清洗计划）。

    参数：
        session_id: 会话标识符。
        req: ReviewRequest，包含 approved 决定和可选的 feedback。

    返回：
        更新后的状态 JSON。
    """
    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "waiting_review":
        raise HTTPException(status_code=400, detail="Session is not waiting for review")

    # 在后台线程中恢复 InsightFlow 执行
    thread = threading.Thread(
        target=_resume_pipeline_thread,
        args=(session, req.approved, req.feedback),
        daemon=True,
    )
    thread.start()

    action = "approved" if req.approved else "rejected"
    logger.info("审核提交，会话 %s: %s", session_id, action)

    return {
        "status": "review_submitted",
        "approved": req.approved,
        "message": f"清洗计划已{ '批准' if req.approved else '拒绝' }，继续执行..."
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    """启动 InsightFlow Web 服务器。"""
    import uvicorn

    print()
    print("=" * 60)
    print("  InsightFlow Web Server")
    print("  http://localhost:8002")
    print("=" * 60)
    print()

    uvicorn.run(
        "insightflow.server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
