"""InsightFlow 的 DataFrame 上下文管理。"""

from insightflow.context.dataframe_context import (
    DataFrameContext,
    get_context,
    set_context,
    new_context,
)

__all__ = ["DataFrameContext", "get_context", "set_context", "new_context"]
