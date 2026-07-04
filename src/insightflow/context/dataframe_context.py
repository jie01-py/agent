"""会话级 DataFrame 上下文 —— 支持版本管理和回滚。

替代 ``data_tools.py`` 中模块级全局 ``_current_df``，
提供线程安全、会话作用域的上下文管理器：

- **版本管理**：每次修改都创建新的版本快照
- **回滚**：可恢复到之前的版本（如质检后重新清洗效果不佳时）
- **修改历史**：完整的变更审计轨迹
- **并发隔离**：线程本地存储，多个 InsightFlow 实例互不干扰
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 线程本地上下文注册表
# ---------------------------------------------------------------------------

_context_registry = threading.local()


def get_context() -> DataFrameContext:
    """返回当前线程的 DataFrameContext。

    Raises:
        ValueError: 当前线程未设置上下文。
    """
    ctx = getattr(_context_registry, "current", None)
    if ctx is None:
        raise ValueError(
            "No DataFrameContext active in this thread. "
            "Call new_context() or set_context(ctx) first."
        )
    return ctx


def set_context(ctx: DataFrameContext) -> None:
    """设置当前线程的 DataFrameContext。"""
    _context_registry.current = ctx


def new_context(session_id: str | None = None) -> DataFrameContext:
    """创建新的 DataFrameContext，注册到当前线程并返回。

    Args:
        session_id: 可选的会话标识，省略则自动生成。

    Returns:
        绑定到当前线程的全新 DataFrameContext。
    """
    ctx = DataFrameContext(session_id=session_id)
    set_context(ctx)
    return ctx


# ---------------------------------------------------------------------------
# 版本记录
# ---------------------------------------------------------------------------


@dataclass
class _VersionRecord:
    """DataFrame 版本快照的内部记录。"""

    version: int
    label: str
    shape: tuple[int, int]
    timestamp: float
    # 存储 DataFrame 副本用于回滚
    df_snapshot: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# DataFrameContext
# ---------------------------------------------------------------------------


class DataFrameContext:
    """会话级 DataFrame 上下文，支持版本管理和隔离。

    替代模块级全局可变 DataFrame，支持并发 InsightFlow、修改历史和回滚。

    用法::

        ctx = new_context()
        ctx.load(df, label="raw")
        ctx.apply("fill_missing", lambda df: df.fillna(0))
        ctx.rollback()  # 撤销 fillna

    Attributes:
        session_id: 此上下文会话的唯一标识。
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:8]
        self._df: pd.DataFrame | None = None
        self._history: list[_VersionRecord] = []
        self._version: int = 0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """返回当前 DataFrame，若未加载则抛出异常。"""
        if self._df is None:
            raise ValueError(
                f"No DataFrame loaded in context '{self.session_id}'. "
                "Call load() first."
            )
        return self._df

    @property
    def version(self) -> int:
        """当前版本号（0 = 未加载数据）。"""
        return self._version

    @property
    def has_data(self) -> bool:
        """是否已加载 DataFrame。"""
        return self._df is not None

    @property
    def shape(self) -> tuple[int, int] | None:
        """当前 DataFrame 的形状，未加载则为 None。"""
        return self._df.shape if self._df is not None else None

    # ------------------------------------------------------------------
    # 核心操作
    # ------------------------------------------------------------------

    def load(self, df: pd.DataFrame, label: str = "initial") -> None:
        """加载 DataFrame 到上下文，创建版本 1。

        Args:
            df: 要加载的 DataFrame（存储副本）。
            label: 此版本的可读标签。
        """
        self._df = df.copy()
        self._version = 1
        record = _VersionRecord(
            version=self._version,
            label=label,
            shape=df.shape,
            timestamp=time.time(),
            df_snapshot=df.copy(),
        )
        self._history = [record]
        logger.info(
            "Context '%s': loaded DataFrame v%d (%d rows x %d cols) [%s]",
            self.session_id,
            self._version,
            df.shape[0],
            df.shape[1],
            label,
        )

    def apply(
        self,
        operation: str,
        func: Callable[[pd.DataFrame], pd.DataFrame | None],
        **kwargs: Any,
    ) -> str:
        """对 DataFrame 执行修改操作并记录到历史。

        *func* 接收当前 DataFrame，应返回修改后的 DataFrame
        （或返回 None 表示原地修改）。

        Args:
            operation: 操作的可读名称（如 "fill_missing"）。
            func: 接收 DataFrame 并返回修改版本的 callable
                  （返回 None 表示原地修改）。
            **kwargs: 额外元数据。

        Returns:
            描述操作结果的字符串。
        """
        if self._df is None:
            raise ValueError("No DataFrame loaded.")

        shape_before = self._df.shape
        result = func(self._df)

        if result is not None:
            self._df = result

        self._version += 1
        record = _VersionRecord(
            version=self._version,
            label=operation,
            shape=self._df.shape,
            timestamp=time.time(),
            df_snapshot=self._df.copy(),
        )
        self._history.append(record)

        rows_diff = self._df.shape[0] - shape_before[0]
        desc = (
            f"Applied '{operation}': {shape_before} -> {self._df.shape}"
        )
        if rows_diff != 0:
            desc += f" ({rows_diff:+d} rows)"

        logger.info("Context '%s': v%d %s", self.session_id, self._version, desc)
        return desc

    def rollback(self, versions: int = 1) -> str:
        """回滚到之前的版本。

        Args:
            versions: 回滚的版本数（默认 1）。

        Returns:
            回滚操作描述。

        Raises:
            ValueError: 回滚会越过版本 1。
        """
        target_version = max(1, self._version - versions)
        if target_version >= self._version:
            return "No rollback needed."

        target_record = None
        for record in self._history:
            if record.version == target_version:
                target_record = record
                break

        if target_record is None or target_record.df_snapshot is None:
            raise ValueError(
                f"Cannot rollback to version {target_version}: snapshot not available."
            )

        self._df = target_record.df_snapshot.copy()
        old_version = self._version
        self._version = target_version

        # 截断历史到目标版本
        self._history = [r for r in self._history if r.version <= target_version]

        desc = (
            f"Rolled back from v{old_version} to v{target_version} "
            f"('{target_record.label}')"
        )
        logger.info("Context '%s': %s", self.session_id, desc)
        return desc

    # ------------------------------------------------------------------
    # 查询辅助方法
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict[str, Any]]:
        """返回修改历史（dict 列表）。

        每个 dict 包含: version, label, shape, timestamp。
        DataFrame 快照不包含在内（体积太大）。
        """
        return [
            {
                "version": r.version,
                "label": r.label,
                "shape": list(r.shape),
                "timestamp": r.timestamp,
            }
            for r in self._history
        ]

    def get_dataframe_copy(self) -> pd.DataFrame:
        """返回当前 DataFrame 的副本。

        这是安全获取 DataFrame 的方式 —— 对返回副本的修改不会影响上下文。
        """
        return self.df.copy()

    def quick_profile(self) -> dict[str, Any]:
        """生成轻量级数据画像，不涉及 LLM 调用。

        返回包含基础统计、空值计数和列信息的 dict。
        适用于 Agent 之间的上下文刷新（如给 Analyst 提供清洗后数据的新画像，
        而非 Scout 阶段的旧画像）。
        """
        df = self.df
        total_rows = len(df)
        profile: dict[str, Any] = {
            "shape": {"rows": total_rows, "columns": len(df.columns)},
            "columns": {},
        }

        for col in df.columns:
            col_info: dict[str, Any] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(int(df[col].isna().sum()) / total_rows * 100, 2)
                if total_rows
                else 0.0,
                "unique_count": int(df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["mean"] = round(float(df[col].mean()), 4) if total_rows else None
                col_info["std"] = round(float(df[col].std()), 4) if total_rows else None
                col_info["min"] = float(df[col].min()) if total_rows else None
                col_info["max"] = float(df[col].max()) if total_rows else None

            profile["columns"][col] = col_info

        return profile

    def __repr__(self) -> str:
        shape_str = f"{self._df.shape}" if self._df is not None else "None"
        return (
            f"DataFrameContext(session='{self.session_id}', "
            f"version={self._version}, shape={shape_str})"
        )
