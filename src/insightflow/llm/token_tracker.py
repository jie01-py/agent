"""InsightFlow 中的 LLM Token 和成本追踪。

按 Agent 记录 LangChain ``usage_metadata`` 中的 token 用量，
根据模型定价计算预估成本。结果以摘要 dict 形式提供，
可集成到 tracer 和 InsightFlow 输出中。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token 用量数据
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """单个 Agent 或整个 InsightFlow 的 token 用量记录。

    Attributes:
        prompt_tokens: 消耗的输入/prompt token 数。
        completion_tokens: 生成的输出/completion token 数。
        total_tokens: 总 token 数（prompt + completion）。
        estimated_cost_usd: 基于模型定价的预估成本（美元）。
        call_count: 记录的 LLM 调用次数。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    call_count: int = 0

    def add(self, prompt: int, completion: int, cost: float = 0.0) -> None:
        """累加一次用量记录。"""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.estimated_cost_usd += cost
        self.call_count += 1

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 兼容的 dict。"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "call_count": self.call_count,
        }


# ---------------------------------------------------------------------------
# 定价表（美元 / 1K tokens）
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "qwen-max": {"input": 0.002, "output": 0.006},
    "qwen-plus": {"input": 0.0004, "output": 0.0012},
    "qwen-turbo": {"input": 0.0002, "output": 0.0006},
    # OpenAI 模型（近似价格）
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

DEFAULT_PRICING = {"input": 0.002, "output": 0.006}


def _compute_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """根据 token 用量计算预估成本（美元）。"""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# TokenTracker 追踪器
# ---------------------------------------------------------------------------


class TokenTracker:
    """线程安全的 token 用量追踪器。

    按 Agent 记录 token 用量并提供汇总统计。
    通过锁保证线程安全，支持并发 Agent 执行。

    用法::

        tracker = TokenTracker(model="qwen-max")
        tracker.record("scout", response_message)
        tracker.record("analyst", response_message)
        summary = tracker.get_summary()
    """

    def __init__(self, model: str = "qwen-max") -> None:
        self._model = model
        self._usage: dict[str, TokenUsage] = {}
        self._total = TokenUsage()
        self._lock = threading.Lock()

    def record(self, agent_name: str, response: Any) -> TokenUsage | None:
        """从 LLM 响应中记录 token 用量。

        从 LangChain AIMessage 中提取 ``usage_metadata``（如存在），
        并累加到按 Agent 和总计的追踪中。

        Args:
            agent_name: 发起 LLM 调用的 Agent 名称。
            response: LLM 响应（AIMessage 或类似对象）。

        Returns:
            本次调用的 TokenUsage 增量，若无用量数据则返回 None。
        """
        usage_meta = self._extract_usage(response)
        if usage_meta is None:
            return None

        prompt_tokens = usage_meta.get("input_tokens", 0) or usage_meta.get("prompt_tokens", 0) or 0
        completion_tokens = usage_meta.get("output_tokens", 0) or usage_meta.get("completion_tokens", 0) or 0
        cost = _compute_cost(prompt_tokens, completion_tokens, self._model)

        with self._lock:
            if agent_name not in self._usage:
                self._usage[agent_name] = TokenUsage()
            self._usage[agent_name].add(prompt_tokens, completion_tokens, cost)
            self._total.add(prompt_tokens, completion_tokens, cost)

        delta = TokenUsage()
        delta.add(prompt_tokens, completion_tokens, cost)
        return delta

    def record_manual(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """手动记录 token 用量（当 usage_metadata 不可用时）。"""
        cost = _compute_cost(prompt_tokens, completion_tokens, self._model)
        with self._lock:
            if agent_name not in self._usage:
                self._usage[agent_name] = TokenUsage()
            self._usage[agent_name].add(prompt_tokens, completion_tokens, cost)
            self._total.add(prompt_tokens, completion_tokens, cost)

    def get_summary(self) -> dict[str, Any]:
        """返回按 Agent 和总计的 token 用量汇总。

        Returns:
            包含各 Agent TokenUsage dict 和 "total" 条目的 dict。
        """
        with self._lock:
            summary: dict[str, Any] = {}
            for agent_name, usage in self._usage.items():
                summary[agent_name] = usage.to_dict()
            summary["total"] = self._total.to_dict()
            return summary

    def get_agent_usage(self, agent_name: str) -> TokenUsage | None:
        """返回指定 Agent 的 token 用量。"""
        with self._lock:
            return self._usage.get(agent_name)

    def get_total_cost(self) -> float:
        """返回总预估成本（美元）。"""
        with self._lock:
            return self._total.estimated_cost_usd

    def print_summary(self) -> str:
        """返回格式化的 token 用量摘要字符串。"""
        with self._lock:
            lines = ["Token Usage Summary:"]
            lines.append("-" * 50)
            for agent_name, usage in self._usage.items():
                lines.append(
                    f"  {agent_name:<16s} | "
                    f"in: {usage.prompt_tokens:>6d} | "
                    f"out: {usage.completion_tokens:>6d} | "
                    f"cost: ${usage.estimated_cost_usd:.4f} | "
                    f"calls: {usage.call_count}"
                )
            lines.append("-" * 50)
            lines.append(
                f"  {'TOTAL':<16s} | "
                f"in: {self._total.prompt_tokens:>6d} | "
                f"out: {self._total.completion_tokens:>6d} | "
                f"cost: ${self._total.estimated_cost_usd:.4f} | "
                f"calls: {self._total.call_count}"
            )
            return "\n".join(lines)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        """从 LangChain 响应中提取 usage metadata。

        兼容 AIMessage.usage_metadata（LangChain >= 0.2）和
        response_metadata.token_usage（旧格式）。
        """
        # LangChain >= 0.2: AIMessage 有 usage_metadata 属性
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            if meta and isinstance(meta, dict):
                return meta

        # 旧格式: response_metadata -> token_usage
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if isinstance(meta, dict):
                token_usage = meta.get("token_usage") or meta.get("usage")
                if token_usage and isinstance(token_usage, dict):
                    return token_usage

        # dict 格式（部分提供商返回）
        if isinstance(response, dict):
            if "usage_metadata" in response:
                return response["usage_metadata"]
            if "usage" in response:
                return response["usage"]

        return None
