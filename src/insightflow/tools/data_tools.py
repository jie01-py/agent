"""LangChain tool，用于数据清洗、分析和可视化。

这些 tool 操作由 DataFrameContext 管理的共享 DataFrame。
调用前通过 ``set_dataframe(df)`` 加载数据，
调用后通过 ``get_dataframe()`` 获取当前状态。
"""

from __future__ import annotations

import json
import os
import platform

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

# ---------------------------------------------------------------------------
# DataFrame 管理 —— 会话级上下文（v2）
# ---------------------------------------------------------------------------
# 旧的全局 _current_df 已被 DataFrameContext 替代，
# 提供版本管理、回滚和线程安全隔离。
# 旧 API（set_dataframe / get_dataframe）保留为兼容封装。


def _get_ctx():
    """获取当前 DataFrameContext，必要时自动创建（降级处理）。"""
    try:
        from insightflow.context import get_context
        return get_context()
    except ValueError:
        # 降级: 为向后兼容自动创建上下文
        from insightflow.context import new_context
        return new_context()


def set_dataframe(df: pd.DataFrame) -> None:
    """设置共享 DataFrame（兼容封装）。

    v2 中委托给会话级 DataFrameContext，
    提供版本管理和线程安全隔离。
    """
    ctx = _get_ctx()
    if df is not None:
        ctx.load(df, label="set_dataframe")
    else:
        # 重置: 创建新上下文
        from insightflow.context import new_context
        new_context()


def get_dataframe() -> pd.DataFrame | None:
    """返回当前 DataFrame（兼容封装）。"""
    try:
        ctx = _get_ctx()
        return ctx.get_dataframe_copy() if ctx.has_data else None
    except ValueError:
        return None


def _require_df() -> pd.DataFrame:
    """返回共享 DataFrame，若未设置则抛出异常。"""
    ctx = _get_ctx()
    if not ctx.has_data:
        raise ValueError(
            "DataFrame has not been set. Call set_dataframe(df) first."
        )
    return ctx.df


# ---------------------------------------------------------------------------
# matplotlib 中文字体配置
# ---------------------------------------------------------------------------

def _configure_chinese_font() -> None:
    """配置 matplotlib 以正确渲染中文字符。"""
    system = platform.system()
    if system == "Windows":
        # Prefer SimHei, fall back to Microsoft YaHei
        for font_name in ("SimHei", "Microsoft YaHei"):
            try:
                plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get(
                    "font.sans-serif", []
                )
                break
            except Exception:
                continue
        else:
            plt.rcParams["font.sans-serif"] = ["sans-serif"]
    else:
        # On macOS / Linux try common CJK fonts, then generic fallback
        for font_name in ("SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"):
            plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams.get(
                "font.sans-serif", []
            )
            break
        else:
            plt.rcParams["font.sans-serif"] = ["sans-serif"]

    plt.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


# ===================================================================
# 清洗 TOOL
# ===================================================================

@tool
def fill_missing(column: str, strategy: str = "mean") -> str:
    """填充或删除当前 DataFrame 某列中的缺失值。

    支持的策略: "mean", "median", "mode", "drop", "zero"。
    返回操作描述及受影响的值数量。
    """
    try:
        df = _require_df()

        if column not in df.columns:
            return f"Error: column '{column}' not found in DataFrame."

        missing_before = int(df[column].isna().sum())
        if missing_before == 0:
            return f"Column '{column}' has no missing values."

        strategy = strategy.lower().strip()

        if strategy == "drop":
            filtered = df.dropna(subset=[column]).reset_index(drop=True)
            try:
                ctx = _get_ctx()
                ctx.apply(
                    f"fill_missing(drop, {column})",
                    lambda _: filtered,
                )
            except ValueError:
                set_dataframe(filtered)
            return (
                f"Dropped {missing_before} rows with missing values in "
                f"column '{column}'. DataFrame now has {len(filtered)} rows."
            )

        if strategy == "mean":
            fill_value = df[column].mean()
        elif strategy == "median":
            fill_value = df[column].median()
        elif strategy == "mode":
            fill_value = df[column].mode().iloc[0]
        elif strategy == "zero":
            fill_value = 0
        else:
            return (
                f"Error: unknown strategy '{strategy}'. "
                "Use one of: mean, median, mode, drop, zero."
            )

        df[column] = df[column].fillna(fill_value)
        # 将修改后的 DataFrame 更新到上下文
        try:
            ctx = _get_ctx()
            ctx.apply(
                f"fill_missing({strategy}, {column})",
                lambda _: df,
            )
        except ValueError:
            set_dataframe(df)

        return (
            f"Filled {missing_before} missing values in column '{column}' "
            f"using strategy '{strategy}' (fill_value={fill_value})."
        )
    except Exception as e:
        return f"Error in fill_missing: {e}"


@tool
def remove_outliers(column: str, method: str = "iqr", threshold: float = 1.5) -> str:
    """根据数值列从 DataFrame 中移除异常值行。

    方法: "iqr"（四分位距）或 "zscore"（Z-score > 3）。
    返回移除了多少行异常值的描述。
    """
    try:
        df = _require_df()

        if column not in df.columns:
            return f"Error: column '{column}' not found in DataFrame."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: column '{column}' is not numeric."

        method = method.lower().strip()
        original_len = len(df)

        if method == "iqr":
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (df[column] >= lower) & (df[column] <= upper)
        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                return f"Column '{column}' has zero variance; no outliers to remove."
            z_scores = ((df[column] - mean) / std).abs()
            mask = z_scores <= 3.0
        else:
            return (
                f"Error: unknown method '{method}'. Use 'iqr' or 'zscore'."
            )

        global_result_df = df[mask].reset_index(drop=True)
        removed = original_len - len(global_result_df)

        # 通过版本管理将修改更新到上下文
        try:
            ctx = _get_ctx()
            ctx.apply(
                f"remove_outliers({method}, {column})",
                lambda _: global_result_df,
            )
        except ValueError:
            set_dataframe(global_result_df)

        return (
            f"Removed {removed} outlier rows from column '{column}' "
            f"using method '{method}' (threshold={threshold}). "
            f"DataFrame now has {len(global_result_df)} rows."
        )
    except Exception as e:
        return f"Error in remove_outliers: {e}"


@tool
def normalize_column(column: str, method: str = "minmax") -> str:
    """对 DataFrame 中的数值列进行归一化。

    方法: "minmax"（缩放到 [0, 1]）或 "standard"（z-score 标准化）。
    返回所应用的归一化描述。
    """
    try:
        df = _require_df()

        if column not in df.columns:
            return f"Error: column '{column}' not found in DataFrame."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: column '{column}' is not numeric."

        method = method.lower().strip()

        if method == "minmax":
            col_min = df[column].min()
            col_max = df[column].max()
            if col_max == col_min:
                return (
                    f"Column '{column}' has constant value; "
                    "min-max normalization skipped."
                )
            df[column] = (df[column] - col_min) / (col_max - col_min)
            try:
                ctx = _get_ctx()
                ctx.apply(f"normalize(minmax, {column})", lambda _: df)
            except ValueError:
                set_dataframe(df)
            return (
                f"Applied min-max normalization to column '{column}' "
                f"(min={col_min}, max={col_max}). Values now in [0, 1]."
            )

        if method == "standard":
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                return (
                    f"Column '{column}' has zero variance; "
                    "standard normalization skipped."
                )
            df[column] = (df[column] - mean) / std
            try:
                ctx = _get_ctx()
                ctx.apply(f"normalize(standard, {column})", lambda _: df)
            except ValueError:
                set_dataframe(df)
            return (
                f"Applied standard (z-score) normalization to column '{column}' "
                f"(mean={mean:.4f}, std={std:.4f})."
            )

        return (
            f"Error: unknown method '{method}'. Use 'minmax' or 'standard'."
        )
    except Exception as e:
        return f"Error in normalize_column: {e}"


# ===================================================================
# 分析 TOOL
# ===================================================================

@tool
def correlation_analysis(col_a: str, col_b: str) -> str:
    """计算两个数值列之间的 Pearson 相关系数。

    返回系数值及定性解读（弱/中等/强相关）。
    """
    try:
        df = _require_df()

        for c in (col_a, col_b):
            if c not in df.columns:
                return f"Error: column '{c}' not found in DataFrame."
            if not pd.api.types.is_numeric_dtype(df[c]):
                return f"Error: column '{c}' is not numeric."

        corr = df[col_a].corr(df[col_b])

        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = "positive" if corr >= 0 else "negative"

        return (
            f"Pearson correlation between '{col_a}' and '{col_b}': "
            f"{corr:.4f} ({strength} {direction} correlation)."
        )
    except Exception as e:
        return f"Error in correlation_analysis: {e}"


@tool
def group_statistics(group_col: str, value_col: str, agg_func: str = "mean") -> str:
    """按分组列计算值列的分组统计。

    支持的聚合函数: "mean", "sum", "count", "min", "max"。
    返回格式化的分组结果表。
    """
    try:
        df = _require_df()

        for c in (group_col, value_col):
            if c not in df.columns:
                return f"Error: column '{c}' not found in DataFrame."

        agg_func = agg_func.lower().strip()
        valid_funcs = ("mean", "sum", "count", "min", "max")
        if agg_func not in valid_funcs:
            return (
                f"Error: unknown agg_func '{agg_func}'. "
                f"Use one of: {', '.join(valid_funcs)}."
            )

        grouped = df.groupby(group_col)[value_col].agg(agg_func)
        result_df = grouped.reset_index()
        result_df.columns = [group_col, f"{agg_func}({value_col})"]

        # 格式化为可读的表格
        table_lines = []
        header = f"| {group_col} | {agg_func}({value_col}) |"
        separator = "|---|---|"
        table_lines.append(header)
        table_lines.append(separator)
        for _, row in result_df.iterrows():
            val = row.iloc[1]
            if isinstance(val, float):
                val = f"{val:.4f}"
            table_lines.append(f"| {row.iloc[0]} | {val} |")

        return (
            f"Grouped statistics ({agg_func}) of '{value_col}' by '{group_col}':\n"
            + "\n".join(table_lines)
        )
    except Exception as e:
        return f"Error in group_statistics: {e}"


@tool
def describe_numeric() -> str:
    """返回 DataFrame 中所有数值列的描述性统计。

    包含 count、mean、std、min、max 和四分位数。
    """
    try:
        df = _require_df()

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "No numeric columns found in the DataFrame."

        desc = numeric_df.describe()
        result = json.loads(desc.to_json(orient="split"))

        columns = result["columns"]
        data = result["data"]
        index = result["index"]

        # Build formatted output
        header = "| Statistic | " + " | ".join(str(c) for c in columns) + " |"
        sep = "|---|" + "|".join("---" for _ in columns) + "|"
        rows = []
        for idx_name, row_vals in zip(index, data):
            formatted = []
            for v in row_vals:
                if v is None:
                    formatted.append("N/A")
                elif isinstance(v, float):
                    formatted.append(f"{v:.4f}")
                else:
                    formatted.append(str(v))
            rows.append(f"| {idx_name} | " + " | ".join(formatted) + " |")

        return (
            f"Descriptive statistics for {len(columns)} numeric columns:\n"
            + "\n".join([header, sep] + rows)
        )
    except Exception as e:
        return f"Error in describe_numeric: {e}"


@tool
def value_distribution(column: str, top_n: int = 10) -> str:
    """返回某列的值计数和百分比。

    显示 top_n 个最频繁的值及其计数和百分比。
    """
    try:
        df = _require_df()

        if column not in df.columns:
            return f"Error: column '{column}' not found in DataFrame."

        counts = df[column].value_counts().head(top_n)
        total = len(df)
        unique = df[column].nunique()

        lines = []
        lines.append(
            f"Value distribution for '{column}' "
            f"(showing top {min(top_n, unique)} of {unique} unique values, "
            f"total rows: {total}):"
        )
        lines.append("")
        lines.append("| Value | Count | Percentage |")
        lines.append("|---|---|---|")
        for val, cnt in counts.items():
            pct = cnt / total * 100
            lines.append(f"| {val} | {cnt} | {pct:.2f}% |")

        return "\n".join(lines)
    except Exception as e:
        return f"Error in value_distribution: {e}"


# ===================================================================
# 可视化 TOOL
# ===================================================================

@tool
def create_chart(
    chart_type: str,
    x_col: str,
    y_col: str,
    title: str,
    output_path: str = "",
    group_col: str = "",
) -> str:
    """从当前 DataFrame 创建图表并保存为文件。

    图表类型: "bar", "line", "scatter", "hist", "pie", "box"。
    设置 group_col 可创建分组图表。
    返回保存的图表图片文件路径。
    """
    try:
        df = _require_df()

        chart_type = chart_type.lower().strip()
        valid_types = ("bar", "line", "scatter", "hist", "pie", "box")
        if chart_type not in valid_types:
            return (
                f"Error: unknown chart_type '{chart_type}'. "
                f"Use one of: {', '.join(valid_types)}."
            )

        for c in (x_col, y_col):
            if c and c not in df.columns:
                return f"Error: column '{c}' not found in DataFrame."

        if group_col and group_col not in df.columns:
            return f"Error: group column '{group_col}' not found in DataFrame."

        # Determine output path
        if not output_path:
            safe_title = "".join(
                ch if ch.isalnum() or ch in ("-", "_", " ") else "_"
                for ch in title
            ).replace(" ", "_")
            output_path = f"{chart_type}_{safe_title}.png"

        # Ensure parent directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if group_col:
                grouped = df.groupby([x_col, group_col])[y_col].mean().unstack()
                grouped.plot(kind="bar", ax=ax)
            else:
                ax.bar(df[x_col].astype(str), df[y_col], color="steelblue")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line":
            if group_col:
                for name, group_df in df.groupby(group_col):
                    ax.plot(
                        group_df[x_col], group_df[y_col],
                        marker="o", label=str(name),
                    )
                ax.legend(title=group_col)
            else:
                ax.plot(df[x_col], df[y_col], marker="o", color="steelblue")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "scatter":
            if group_col:
                for name, group_df in df.groupby(group_col):
                    ax.scatter(
                        group_df[x_col], group_df[y_col],
                        label=str(name), alpha=0.7,
                    )
                ax.legend(title=group_col)
            else:
                ax.scatter(df[x_col], df[y_col], color="steelblue", alpha=0.7)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif chart_type == "hist":
            if group_col:
                for name, group_df in df.groupby(group_col):
                    ax.hist(
                        group_df[y_col], bins=20, alpha=0.5, label=str(name),
                    )
                ax.legend(title=group_col)
            else:
                ax.hist(df[y_col], bins=20, color="steelblue", edgecolor="white")
            ax.set_xlabel(y_col)
            ax.set_ylabel("Frequency")

        elif chart_type == "pie":
            if group_col:
                pie_data = df.groupby(group_col)[y_col].sum()
            else:
                pie_data = df[x_col].value_counts().head(10)
            ax.pie(
                pie_data.values,
                labels=pie_data.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")

        elif chart_type == "box":
            if group_col:
                box_data = []
                labels = []
                for name, group_df in df.groupby(group_col):
                    box_data.append(group_df[y_col].dropna().values)
                    labels.append(str(name))
                ax.boxplot(box_data, labels=labels, patch_artist=True)
            else:
                ax.boxplot(df[y_col].dropna(), vert=True, patch_artist=True)
                ax.set_xticks([1])
                ax.set_xticklabels([y_col])
            ax.set_ylabel(y_col)

        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return f"Chart saved to: {output_path}"
    except Exception as e:
        return f"Error in create_chart: {e}"


# ===================================================================
# 实用 TOOL
# ===================================================================

@tool
def get_dataframe_info() -> str:
    """返回当前 DataFrame 的概览: 形状、列名、数据类型和前几行。

    用于了解已加载数据的结构和内容。
    """
    try:
        df = _require_df()

        info = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "missing": int(df[col].isna().sum()),
                    "unique": int(df[col].nunique()),
                }
                for col in df.columns
            ],
            "head": json.loads(df.head(5).to_json(orient="records", default_handler=str)),
        }

        return json.dumps(info, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error in get_dataframe_info: {e}"
