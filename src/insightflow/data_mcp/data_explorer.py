"""
data_explorer.py - MCP Server for CSV data exploration.

A FastMCP-based server that exposes data exploration capabilities over the
Model Context Protocol (MCP). Goes beyond simple tool registration to
demonstrate the full MCP protocol surface:

- **Tools**: Interactive operations (load, query, profile)
- **Resources**: URI-addressable data endpoints (schema, profile, sample)
- **Prompts**: Server-side prompt templates with domain expertise
- **DataFrameStore**: TTL-based LRU cache replacing the naive dict cache

Run standalone:
    python -m insightflow.mcp.data_explorer
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("data-explorer")

# ---------------------------------------------------------------------------
# DataFrameStore — TTL-based LRU cache (replaces naive _cache dict)
# ---------------------------------------------------------------------------


class DataFrameStore:
    """In-memory DataFrame cache with TTL expiry and LRU eviction.

    Replaces the old module-level ``_cache: dict`` that never expired
    and could grow unbounded. This store:
    - Evicts entries older than ``ttl_seconds`` on access
    - Limits total entries via ``max_size`` (LRU eviction when full)
    - Tracks access counts for observability

    Args:
        max_size: Maximum number of cached DataFrames.
        ttl_seconds: Time-to-live for each entry in seconds.
    """

    def __init__(self, max_size: int = 10, ttl_seconds: int = 300) -> None:
        self._store: OrderedDict[str, tuple[pd.DataFrame, float, int]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, path: str) -> pd.DataFrame:
        """Return the cached DataFrame for *path*, loading from disk if needed.

        On access, the entry is moved to the end (most recently used) and
        expired entries are evicted.
        """
        self.evict_expired()

        if path in self._store:
            df, ts, count = self._store.pop(path)
            # Check TTL
            if time.time() - ts > self._ttl:
                # Expired — reload from disk
                df = self._load_from_disk(path)
                ts = time.time()
                count = 0
            self._store[path] = (df, ts, count + 1)
            return df

        # Not cached — load and cache
        df = self._load_from_disk(path)
        self._put(path, df)
        return df

    def _put(self, path: str, df: pd.DataFrame) -> None:
        """Add to cache, evicting LRU if at capacity."""
        if len(self._store) >= self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.info("DataFrameStore: evicted LRU entry '%s'", evicted_key)
        self._store[path] = (df, time.time(), 0)

    def evict_expired(self) -> int:
        """Remove all entries past their TTL. Returns count of evicted entries."""
        now = time.time()
        expired = [
            key for key, (_, ts, _) in self._store.items()
            if now - ts > self._ttl
        ]
        for key in expired:
            del self._store[key]
        return len(expired)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "entries": {
                path: {
                    "shape": list(df.shape),
                    "age_seconds": round(time.time() - ts, 1),
                    "access_count": count,
                }
                for path, (df, ts, count) in self._store.items()
            },
        }

    @staticmethod
    def _load_from_disk(path: str) -> pd.DataFrame:
        """Load a CSV from disk."""
        return pd.read_csv(path)


# Singleton store instance
_store = DataFrameStore(max_size=10, ttl_seconds=300)


def _load(path: str) -> pd.DataFrame:
    """Return a cached DataFrame for *path*, loading from disk on first access."""
    return _store.get(path)


# ---------------------------------------------------------------------------
# MCP Tools (existing, now using DataFrameStore)
# ---------------------------------------------------------------------------


@mcp.tool()
def load_csv(path: str) -> str:
    """Load a CSV file into memory and return basic metadata.

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.

    Returns
    -------
    str
        JSON string containing:
        - row_count: number of rows
        - column_count: number of columns
        - columns: list of column names
        - memory_usage_kb: approximate memory footprint in KB
        - head: first 3 rows as a preview
    """
    try:
        df = _load(path)
        info: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "head": df.head(3).to_dict(orient="records"),
        }
        return json.dumps(info, ensure_ascii=False, default=str, indent=2)
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    except pd.errors.EmptyDataError:
        return json.dumps({"error": f"File is empty: {path}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def get_schema(path: str) -> str:
    """Return schema information for every column in the CSV.

    For each column the following are reported:
    - dtype (pandas data type)
    - non_null_rate (fraction of values that are not null, 0-1)
    - unique_count (number of distinct values)

    Parameters
    ----------
    path:
        Filesystem path to the CSV file (must have been loaded via load_csv
        or will be loaded automatically).

    Returns
    -------
    str
        JSON string mapping column names to their schema details.
    """
    try:
        df = _load(path)
        schema: dict[str, dict[str, Any]] = {}
        total_rows = len(df)
        for col in df.columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "non_null_rate": round(int(df[col].notna().sum()) / total_rows, 4)
                if total_rows
                else 0.0,
                "unique_count": int(df[col].nunique()),
            }
        return json.dumps(schema, ensure_ascii=False, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def sample_rows(path: str, n: int = 5) -> str:
    """Return *n* randomly sampled rows from the CSV as formatted text.

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    n:
        Number of random rows to return. Clamped to [1, row_count].

    Returns
    -------
    str
        JSON string containing the sampled rows (list of dicts).
    """
    try:
        df = _load(path)
        n = max(1, min(n, len(df)))
        sample = df.sample(n=n, random_state=42)
        return json.dumps(
            {"sampled_rows": sample.to_dict(orient="records"), "count": n},
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool()
def profile(path: str) -> str:
    """Generate a comprehensive data profile for the CSV.

    The profile includes:
    - Descriptive statistics (mean, std, min, max, quartiles) for numeric columns
    - Missing value summary per column (count and percentage)
    - Value distribution for categorical columns (top 5 most frequent values)

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.

    Returns
    -------
    str
        JSON string with the full profile report.
    """
    try:
        df = _load(path)
        total_rows = len(df)
        profile_report: dict[str, Any] = {}

        # --- Descriptive statistics for numeric columns ---
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            profile_report["descriptive_stats"] = desc
        else:
            profile_report["descriptive_stats"] = "(no numeric columns)"

        # --- Missing value summary ---
        missing: dict[str, dict[str, Any]] = {}
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            missing[col] = {
                "null_count": null_count,
                "null_pct": round(null_count / total_rows * 100, 2)
                if total_rows
                else 0.0,
            }
        profile_report["missing_values"] = missing

        # --- Value distribution for categorical columns (top 5) ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        distributions: dict[str, dict[str, int]] = {}
        for col in cat_cols:
            vc = df[col].value_counts().head(5)
            distributions[col] = {str(k): int(v) for k, v in vc.items()}
        profile_report["categorical_distributions"] = distributions or "(no categorical columns)"

        return json.dumps(profile_report, ensure_ascii=False, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP Resources — URI-addressable data endpoints
# ---------------------------------------------------------------------------
# Resources are the second pillar of MCP (alongside Tools and Prompts).
# Unlike Tools (which perform actions), Resources provide read-only access
# to data via URIs. Clients can subscribe to resource changes and benefit
# from MIME type negotiation. This is the "REST" side of MCP.


@mcp.resource("data://csv/{path}/schema")
def resource_schema(path: str) -> str:
    """Schema of a CSV file as a resource.

    Clients can read this resource to understand the data structure
    without invoking a tool. Supports subscription for change detection.
    """
    return get_schema(path)


@mcp.resource("data://csv/{path}/profile")
def resource_profile(path: str) -> str:
    """Comprehensive data profile as a resource.

    Provides a full statistical profile including descriptive stats,
    missing value analysis, and categorical distributions.
    """
    return profile(path)


@mcp.resource("data://csv/{path}/sample/{n}")
def resource_sample(path: str, n: int = 5) -> str:
    """Sampled rows from a CSV as a resource."""
    return sample_rows(path, int(n))


@mcp.resource("data://csv/{path}/info")
def resource_info(path: str) -> str:
    """Basic metadata (row count, columns, memory) as a resource."""
    return load_csv(path)


@mcp.resource("cache://stats")
def resource_cache_stats() -> str:
    """DataFrameStore cache statistics as a resource.

    Useful for monitoring cache hit rates and memory usage.
    """
    return json.dumps(_store.stats(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP Prompts — server-side prompt templates
# ---------------------------------------------------------------------------
# Prompts are the third pillar of MCP. The server provides structured
# prompt templates that encode domain expertise about how to best use
# its tools. Clients discover and invoke these prompts, receiving
# pre-crafted message sequences. This means the server knows the best
# way to guide analysis of its data.


@mcp.prompt()
def data_analysis_prompt(task: str, path: str) -> list[dict]:
    """Structured prompt for data analysis.

    The data_explorer server knows the best way to guide analysis of CSV data.
    This prompt template provides a structured starting point that includes
    tool usage guidance and output format expectations.

    Parameters
    ----------
    task:
        Natural language description of what to analyze.
    path:
        Filesystem path to the CSV file.
    """
    return [
        {
            "role": "system",
            "content": (
                "你是一个数据分析专家。你需要使用以下工具来探索和分析 CSV 数据：\n"
                "- load_csv: 加载数据，获取基本信息\n"
                "- get_schema: 查看每列的数据类型和质量\n"
                "- sample_rows: 随机采样查看数据样本\n"
                "- profile: 获取完整的数据画像（统计摘要、缺失值、分布）\n"
                "- safe_query: 执行安全的 pandas 查询（只读）\n\n"
                "分析流程建议：\n"
                "1. 先用 load_csv 了解数据规模\n"
                "2. 用 get_schema 和 sample_rows 理解数据结构\n"
                "3. 用 profile 获取完整画像\n"
                "4. 根据分析目标用 safe_query 做深入查询\n\n"
                "输出要求：结构化 JSON，包含 basic_info, column_details, "
                "numeric_summary, quality_issues, categorical_summary"
            ),
        },
        {
            "role": "user",
            "content": f"请分析数据文件 {path}，分析目标是：{task}",
        },
    ]


@mcp.prompt()
def cleaning_strategy_prompt(profile_json: str) -> list[dict]:
    """Prompt template for generating data cleaning strategies.

    Given a data profile, guides the LLM to produce a structured cleaning plan
    with specific actions for each problematic column.

    Parameters
    ----------
    profile_json:
        JSON string of the data profile (from the profile tool or resource).
    """
    return [
        {
            "role": "system",
            "content": (
                "你是数据清洗专家。根据提供的数据画像，制定结构化的清洗计划。\n\n"
                "可用操作：\n"
                "- fill_missing: 填充缺失值 (strategy: mean/median/mode/drop/zero)\n"
                "- remove_outliers: 移除异常值 (method: iqr/zscore, threshold: float)\n"
                "- normalize_column: 标准化 (method: minmax/standard)\n\n"
                "原则：\n"
                "- 只处理有问题的列，不要过度清洗\n"
                "- 优先保留数据，只有在必要时才删除行\n"
                "- 解释每个决策的原因\n\n"
                "输出格式：JSON {\"strategy\": [...], \"overall_notes\": \"...\"}"
            ),
        },
        {
            "role": "user",
            "content": f"以下是数据画像：\n{profile_json}\n\n请制定清洗计划。",
        },
    ]


@mcp.prompt()
def query_builder_prompt(question: str, path: str) -> list[dict]:
    """Prompt template for building safe pandas queries.

    Given a natural language question about the data, guides the LLM to
    construct an appropriate safe_query expression.

    Parameters
    ----------
    question:
        Natural language question about the data.
    path:
        Filesystem path to the CSV file.
    """
    return [
        {
            "role": "system",
            "content": (
                "你需要使用 safe_query 工具来回答关于数据的问题。\n"
                "safe_query 接受一个 pandas 表达式，变量 `df` 代表 DataFrame。\n\n"
                "规则：\n"
                "- 只使用只读操作（不能修改数据）\n"
                "- 不能用 import, exec, eval\n"
                "- 不能用分号（只允许单表达式）\n\n"
                "常用模式：\n"
                "- df[df['col'] > value]  — 过滤\n"
                "- df.groupby('col')['val'].sum()  — 分组聚合\n"
                "- df[['a', 'b']].sort_values('a')  — 选择+排序\n"
                "- df.describe()  — 统计摘要"
            ),
        },
        {
            "role": "user",
            "content": f"数据文件: {path}\n问题: {question}\n请构造合适的查询表达式。",
        },
    ]


# ---------------------------------------------------------------------------
# safe_query helpers
# ---------------------------------------------------------------------------

# Patterns that indicate a mutation attempt — these are rejected.
_UNSAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(assign|drop|rename|replace|fillna|update|insert|append)\s*\(", re.I),
    re.compile(r"\b(del|pop|iloc\[|loc\[)\s*[=]", re.I),
    re.compile(r"\bto_csv|to_excel|to_sql|to_pickle|to_parquet|to_json|to_hdf\b", re.I),
    re.compile(r"\bimport\b", re.I),
    re.compile(r"\bexec\s*\(|eval\s*\(|compile\s*\(", re.I),
    re.compile(r"\b(open|__import__|getattr|setattr|delattr)\s*\(", re.I),
    re.compile(r";", re.I),  # disallow statement chaining
    re.compile(r"__", re.I),  # disallow dunder access
]


def _is_safe(expression: str) -> bool:
    """Return True if *expression* looks like a read-only pandas query."""
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(expression):
            return False
    return True


@mcp.tool()
def safe_query(path: str, expression: str) -> str:
    """Execute a safe, read-only pandas expression on the loaded DataFrame.

    The variable ``df`` is bound to the DataFrame in the evaluation scope.
    Only read-only operations are permitted; any expression that appears to
    modify data, perform I/O, or access unsafe builtins is rejected.

    Security model: regex-based pattern matching + restricted builtins namespace.
    The expression is evaluated in a sandboxed scope with only safe Python
    builtins available.

    Examples of valid expressions::

        df[df['price'] > 100]
        df.groupby('category')['sales'].sum()
        df[['name', 'age']].sort_values('age', ascending=False).head(10)
        df.describe()

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    expression:
        A pandas expression string. The DataFrame is available as ``df``.

    Returns
    -------
    str
        JSON string with the query result (up to 100 rows) or an error message.
    """
    try:
        if not _is_safe(expression):
            return json.dumps(
                {"error": "Expression rejected: potentially unsafe operation detected."},
                ensure_ascii=False,
            )

        df = _load(path)

        # Build a restricted evaluation namespace.
        safe_builtins: dict[str, Any] = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "range": range,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }
        eval_globals: dict[str, Any] = {"__builtins__": safe_builtins, "df": df, "pd": pd}

        result = eval(expression, eval_globals)  # noqa: S307

        # Convert result to a serialisable form.
        if isinstance(result, pd.DataFrame):
            rows = min(len(result), 100)
            output: dict[str, Any] = {
                "result_type": "DataFrame",
                "row_count": len(result),
                "columns": list(result.columns),
                "data": result.head(rows).to_dict(orient="records"),
            }
            if len(result) > 100:
                output["truncated"] = True
                output["note"] = "Only the first 100 rows are shown."
        elif isinstance(result, pd.Series):
            output = {
                "result_type": "Series",
                "name": result.name,
                "data": result.head(100).to_dict(),
            }
        else:
            output = {"result_type": type(result).__name__, "value": result}

        return json.dumps(output, ensure_ascii=False, default=str, indent=2)
    except SyntaxError as exc:
        return json.dumps(
            {"error": f"Syntax error in expression: {exc}"}, ensure_ascii=False
        )
    except Exception as exc:
        return json.dumps({"error": f"Query failed: {exc}"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
