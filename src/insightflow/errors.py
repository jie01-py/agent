"""InsightFlow 结构化错误传播。

实现分级错误模型：Agent 故障按严重程度和可恢复性分类。
下游 Agent 和图编排器可以查询错误状态，决定继续、降级还是中止。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

# 这些 Agent 失败就是致命的 —— 下游没有它们跑不了
FATAL_AGENTS = frozenset({"scout"})

# 这些 Agent 失败可以降级 —— 流程能跳过它们继续跑，只是输出质量会下降
DEGRADABLE_AGENTS = frozenset({"cleaner_execute", "visualizer"})

# 其余 Agent（cleaner_plan、analyst、reporter）属于"重要"级别 ——
# 失败不会立即导致中止，但会显著降低输出质量。


@dataclass
class AgentError:
    """Agent 故障的结构化记录。

    Attributes:
        agent_name: 出错的 Agent 名称（如 "scout"、"analyst"）。
        error_type: 错误类别。
        message: 可读的错误描述。
        severity: "fatal" | "degraded" | "warning"
        recoverable: 流程是否可以继续运行。
        timestamp: 错误的 Unix 时间戳。
        metadata: 附加上下文（堆栈、工具名等）。
    """

    agent_name: str
    error_type: str  # "timeout", "api_error", "parse_error", "tool_error", "config_error"
    message: str
    severity: Literal["fatal", "degraded", "warning"] = "warning"
    recoverable: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "agent_name": self.agent_name,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_exception(
        cls,
        agent_name: str,
        exc: Exception,
        *,
        error_type: str = "runtime_error",
    ) -> AgentError:
        """从异常创建 AgentError，自动分类严重程度。"""
        severity = "fatal" if agent_name in FATAL_AGENTS else (
            "degraded" if agent_name in DEGRADABLE_AGENTS else "warning"
        )
        recoverable = agent_name not in FATAL_AGENTS

        return cls(
            agent_name=agent_name,
            error_type=error_type,
            message=str(exc),
            severity=severity,
            recoverable=recoverable,
        )


# ---------------------------------------------------------------------------
# 错误传播器
# ---------------------------------------------------------------------------


class ErrorPropagator:
    """评估累积错误并决定流程走向。

    用法::

        propagator = ErrorPropagator()
        should_continue, reason = propagator.should_continue(errors)
        if not should_continue:
            # 中止流程
    """

    def should_continue(
        self,
        errors: list[AgentError],
    ) -> tuple[bool, str]:
        """根据累积错误判断流程是否继续。

        Args:
            errors: 目前收集到的 AgentError 列表。

        Returns:
            元组 (是否继续, 原因说明)。
        """
        if not errors:
            return True, "No errors."

        # fatal 错误直接中止
        for err in errors:
            if err.severity == "fatal":
                reason = (
                    f"Fatal error in '{err.agent_name}' "
                    f"({err.error_type}): {err.message}"
                )
                logger.error("Pipeline aborted: %s", reason)
                return False, reason

        # 统计 degraded 错误数量
        degraded_count = sum(1 for e in errors if e.severity == "degraded")
        if degraded_count >= 3:
            reason = (
                f"Too many degraded errors ({degraded_count}). "
                "Pipeline output quality would be unacceptable."
            )
            logger.warning("Pipeline aborted: %s", reason)
            return False, reason

        return True, f"Continuing with {len(errors)} non-fatal error(s)."

    def get_health_status(
        self,
        errors: list[AgentError],
    ) -> Literal["healthy", "degraded", "fatal"]:
        """根据累积错误返回简单的健康状态。

        供图的条件边做路由判断用。
        """
        if not errors:
            return "healthy"
        for err in errors:
            if err.severity == "fatal":
                return "fatal"
        return "degraded"

    def get_fallback_context(
        self,
        failed_agent: str,
    ) -> dict[str, Any]:
        """为下游 Agent 生成兜底状态。

        当某个 Agent 失败但流程继续（降级模式）时，
        下游 Agent 需要一些最基本的状态才能工作。
        这个方法提供合理的默认值。

        Args:
            failed_agent: 出错的 Agent 名称。

        Returns:
            部分状态字典，作为兜底上下文注入。
        """
        fallbacks: dict[str, dict[str, Any]] = {
            "cleaner_plan": {
                "cleaning_plan": {
                    "strategy": [],
                    "overall_notes": "清洗计划生成失败，跳过清洗步骤",
                },
            },
            "cleaner_execute": {
                "cleaning_plan": {
                    "strategy": [],
                    "overall_notes": "清洗执行失败，使用原始数据继续分析",
                },
            },
            "visualizer": {
                "charts": [],
            },
            "analyst": {
                "analysis_results": {
                    "summary": "分析失败，无法生成统计结果",
                    "findings": [],
                    "statistics": {},
                    "data_quality_note": "分析阶段出错",
                },
            },
        }
        return fallbacks.get(failed_agent, {})
