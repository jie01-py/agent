"""弹性 LLM 客户端 —— 指数退避重试 + circuit breaker 熔断器模式。

封装任意 LangChain ChatModel，提供生产级弹性能力：
- **指数退避 + 随机抖动**：应对限速、超时、连接错误等瞬态故障
- **Circuit breaker 熔断器**：连续失败 N 次后熔断，防止 LLM API 宕机时级联崩溃
- **三种熔断状态**：closed（正常）→ open（拒绝请求）→ half-open（探测恢复）
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常类
# ---------------------------------------------------------------------------


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出，拒绝所有请求。"""

    def __init__(self, failures: int, cooldown_remaining: float):
        self.failures = failures
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker open after {failures} failures. "
            f"Retry in {cooldown_remaining:.1f}s."
        )


class MaxRetriesExceededError(Exception):
    """所有重试机会耗尽时抛出。"""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted. "
            f"Last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# 可重试错误检测
# ---------------------------------------------------------------------------

# 表示瞬态故障的错误类型名称，值得重试
_RETRYABLE_ERROR_NAMES = frozenset({
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIError",
})


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否属于可重试的瞬态错误。"""
    # 按类名判断（兼容不同 LLM 提供商的 SDK）
    exc_name = type(exc).__name__
    if exc_name in _RETRYABLE_ERROR_NAMES:
        return True

    # 检查错误信息中常见的 HTTP 状态码
    msg = str(exc).lower()
    if any(code in msg for code in ("429", "502", "503", "504", "timeout")):
        return True

    # ConnectionError 和 TimeoutError 始终视为可重试
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    return False


# ---------------------------------------------------------------------------
# Circuit breaker 熔断状态
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = "closed"        # 正常运行
    OPEN = "open"            # 拒绝所有请求
    HALF_OPEN = "half_open"  # 探测是否恢复


# ---------------------------------------------------------------------------
# ResilientLLMClient
# ---------------------------------------------------------------------------


class ResilientLLMClient:
    """封装 LangChain ChatModel，加入 retry 重试和 circuit breaker 熔断逻辑。

    用法::

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="qwen-max", ...)
        client = ResilientLLMClient(llm, max_retries=3)
        response = client.invoke(messages)

    客户端会追踪连续失败次数，达到 ``circuit_breaker_threshold`` 后打开熔断器。
    熔断器打开时，所有请求立即抛出 ``CircuitBreakerOpenError``。
    等待 ``circuit_cooldown_seconds`` 后进入 half-open 状态，允许一个探测请求。
    探测成功则关闭熔断器；探测失败则重新打开。

    Args:
        llm: LangChain ChatModel（需要有 ``invoke`` 方法）。
        max_retries: 每次调用最大重试次数。
        base_delay: 指数退避的基础延迟（秒）。
        max_delay: 延迟上限（秒）。
        circuit_breaker_threshold: 连续失败多少次后触发熔断。
        circuit_cooldown_seconds: 熔断后等待多久进入 half-open 探测。
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_cooldown_seconds: float = 60.0,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._cb_threshold = circuit_breaker_threshold
        self._cb_cooldown = circuit_cooldown_seconds

        # circuit breaker 状态
        self._cb_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._cb_opened_at: float | None = None

        # 统计数据
        self._total_calls = 0
        self._total_retries = 0
        self._total_failures = 0

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """调用 LLM，带 retry 重试和 circuit breaker 熔断保护。

        Args:
            messages: 要发送的消息（格式同底层 LLM）。
            **kwargs: 传递给 LLM invoke 方法的额外参数。

        Returns:
            LLM 响应结果。

        Raises:
            CircuitBreakerOpenError: 熔断器处于 open 状态。
            MaxRetriesExceededError: 所有重试机会已耗尽。
        """
        # 检查熔断器状态
        self._check_circuit()

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            self._total_calls += 1
            if attempt > 0:
                self._total_retries += 1

            try:
                result = self._llm.invoke(messages, **kwargs)
                self._on_success()
                return result

            except Exception as exc:
                last_error = exc

                if not _is_retryable(exc):
                    # 不可重试的错误直接向上抛出
                    self._on_failure()
                    raise

                if attempt < self._max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d, retryable), "
                        "retrying in %.1fs: %s: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    time.sleep(delay)
                else:
                    self._on_failure()

        # 所有重试机会已耗尽
        assert last_error is not None
        raise MaxRetriesExceededError(self._max_retries + 1, last_error)

    def bind_tools(self, tools: list, **kwargs: Any) -> ResilientLLMClient:
        """代理 ``bind_tools`` 到 LLM，返回重新包装后的 ResilientLLMClient。

        LangGraph 的 ``create_agent`` 在初始化时会调用 ``model.bind_tools(tools)``。
        此方法确保返回的模型仍保留 retry 重试和 circuit breaker 熔断保护。

        Returns:
            新的 ResilientLLMClient，封装了已绑定 tool 的 LLM。
        """
        bound_llm = self._llm.bind_tools(tools, **kwargs)
        return ResilientLLMClient(
            bound_llm,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            circuit_breaker_threshold=self._cb_threshold,
            circuit_cooldown_seconds=self._cb_cooldown,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> ResilientLLMClient:
        """代理 ``with_structured_output`` 并重新包装。

        Returns:
            新的 ResilientLLMClient，封装了结构化输出的 LLM。
        """
        bound_llm = self._llm.with_structured_output(schema, **kwargs)
        return ResilientLLMClient(
            bound_llm,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            circuit_breaker_threshold=self._cb_threshold,
            circuit_cooldown_seconds=self._cb_cooldown,
        )

    def with_config(self, config: dict | None = None, **kwargs: Any) -> ResilientLLMClient:
        """代理 ``with_config`` 并重新包装。

        Returns:
            新的 ResilientLLMClient，封装了已配置的 LLM。
        """
        bound_llm = self._llm.with_config(config, **kwargs) if config else self._llm.with_config(**kwargs)
        return ResilientLLMClient(
            bound_llm,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            circuit_breaker_threshold=self._cb_threshold,
            circuit_cooldown_seconds=self._cb_cooldown,
        )

    def get_stats(self) -> dict[str, Any]:
        """返回客户端统计信息。"""
        return {
            "total_calls": self._total_calls,
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
            "circuit_state": self._cb_state.value,
            "consecutive_failures": self._consecutive_failures,
        }

    def reset_circuit(self) -> None:
        """手动重置 circuit breaker 到 closed 状态。"""
        self._cb_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._cb_opened_at = None
        logger.info("Circuit breaker manually reset to CLOSED")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _check_circuit(self) -> None:
        """检查熔断器状态，若处于 open 则拒绝请求。"""
        if self._cb_state == CircuitState.CLOSED:
            return

        if self._cb_state == CircuitState.OPEN:
            assert self._cb_opened_at is not None
            elapsed = time.time() - self._cb_opened_at
            if elapsed >= self._cb_cooldown:
                # 冷却时间已过，转入 half-open 探测
                self._cb_state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker: OPEN -> HALF_OPEN (cooldown elapsed, probing)"
                )
                return
            else:
                remaining = self._cb_cooldown - elapsed
                raise CircuitBreakerOpenError(
                    self._consecutive_failures, remaining
                )

        # HALF_OPEN: 放行一个请求作为探测
        # （自然处理——若失败，_on_failure 会重新打开熔断器）

    def _on_success(self) -> None:
        """调用成功时的处理 —— 重置失败追踪。"""
        if self._cb_state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker: HALF_OPEN -> CLOSED (probe succeeded)")
        self._cb_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._cb_opened_at = None

    def _on_failure(self) -> None:
        """调用失败时的处理 —— 累加失败计数。"""
        self._total_failures += 1
        self._consecutive_failures += 1

        if self._consecutive_failures >= self._cb_threshold:
            self._cb_state = CircuitState.OPEN
            self._cb_opened_at = time.time()
            logger.warning(
                "Circuit breaker: -> OPEN after %d consecutive failures",
                self._consecutive_failures,
            )

    def _compute_delay(self, attempt: int) -> float:
        """计算指数退避延迟（带随机抖动）。

        delay = min(base * 2^attempt + jitter, max_delay)
        """
        exponential = self._base_delay * (2 ** attempt)
        jitter = random.uniform(0, self._base_delay)
        return min(exponential + jitter, self._max_delay)
