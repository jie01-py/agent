"""
mcp_bridge.py - MCP-to-LangChain adapter for the InsightFlow pipeline.

This module bridges the data_explorer MCP server and LangChain agents. It
provides three entry points with RBAC permission filtering:

1. ``mcp_session()`` - async context manager that launches the MCP server as a
   subprocess, connects over stdio, discovers tools, filters by role
   permissions, and yields them as LangChain ``BaseTool`` instances.

2. ``create_mcp_tools()`` - convenience coroutine that returns adapted tools
   together with the session context so callers can manage lifecycle.

3. ``create_sync_mcp_tools()`` - synchronous fallback that wraps the
   data_explorer functions directly (no subprocess, no MCP protocol) with
   role-based permission filtering.

v2 changes:
- Permission-aware tool filtering via RBAC role system
- Resource discovery support (list_resources from MCP server)
- Prompt discovery support (list_prompts from MCP server)
- Role parameter on all entry points for scoped tool access

Typical async usage::

    async with mcp_session(role="scout") as tools:
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke({"input": "Profile sales.csv"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

# ============================================================================
# 关键修复：解决外部 mcp 包与本地 insightflow.mcp 模块的命名冲突
# ============================================================================
# 当此文件被导入时，Python 可能已经将 'mcp' 绑定到了本地的 insightflow.mcp
# 模块。我们需要彻底清理所有相关的模块缓存，然后重新导入外部包。
# ============================================================================

# 1. 收集所有需要删除的模块键
_modules_to_remove = []
for module_name in list(sys.modules.keys()):
    # 如果模块名是 'mcp' 或以 'mcp.' 开头
    if module_name == 'mcp' or module_name.startswith('mcp.'):
        mod = sys.modules[module_name]
        # 检查它是否是本地模块（文件路径包含 insightflow）
        if hasattr(mod, '__file__') and mod.__file__:
            if 'insightflow' in mod.__file__:
                _modules_to_remove.append(module_name)

# 2. 删除所有冲突的模块引用
for module_name in _modules_to_remove:
    del sys.modules[module_name]

# 3. 现在可以安全导入外部的 mcp 包了
from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from insightflow.data_mcp.permissions import PermissionChecker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_EXPLORER_PATH: str = str(
    Path(__file__).resolve().parent / "data_explorer.py"
)


def _build_server_params() -> StdioServerParameters:
    """Build stdio transport parameters pointing at the data_explorer server."""
    return StdioServerParameters(
        command=sys.executable,
        args=[_DATA_EXPLORER_PATH],
        env=None,
    )


def _mcp_tool_to_langchain(
    session: ClientSession,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain ``StructuredTool``.

    The MCP tool's JSON-schema ``input_schema`` is translated into a Pydantic
    model so that LangChain can validate arguments before dispatch.

    Parameters
    ----------
    session:
        An active ``ClientSession`` connected to the MCP server.
    tool_name:
        The MCP tool's name (e.g. ``"load_csv"``).
    tool_description:
        Human-readable description surfaced by the MCP server.
    input_schema:
        JSON Schema dict describing the tool's parameters.

    Returns
    -------
    StructuredTool
        A LangChain tool whose ``arun`` delegates to ``session.call_tool``.
    """
    # Build a Pydantic model from the MCP input schema so LangChain has
    # proper arg validation.
    args_schema = _json_schema_to_pydantic(tool_name, input_schema)

    async def _call(**kwargs: Any) -> str:
        result = await session.call_tool(tool_name, arguments=kwargs)
        # MCP returns a list of TextContent blocks; join their text fields.
        if hasattr(result, "content"):
            parts: list[str] = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return str(result)

    return StructuredTool(
        name=tool_name,
        description=tool_description or f"Call MCP tool '{tool_name}'.",
        coroutine=_call,
        args_schema=args_schema,
    )


def _json_schema_to_pydantic(name: str, schema: dict[str, Any]) -> type:
    """Dynamically create a Pydantic v2 model from a JSON Schema dict.

    This is intentionally lightweight - we only need enough structure for
    LangChain to understand the argument names, types, and defaults.

    Parameters
    ----------
    name:
        Base name used to name the generated model (``<name>Input``).
    schema:
        A JSON Schema object with ``properties`` and optional ``required``.

    Returns
    -------
    type
        A Pydantic ``BaseModel`` subclass.
    """
    from pydantic import Field, create_model

    properties: dict[str, Any] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        py_type = _json_type_to_python(field_schema.get("type", "string"))
        description = field_schema.get("description", "")
        if field_name in required:
            field_definitions[field_name] = (py_type, Field(..., description=description))
        else:
            default = field_schema.get("default", None)
            # Make optional fields accept None so callers can omit them.
            optional_type = py_type | None  # type: ignore[valid-type]
            field_definitions[field_name] = (
                optional_type,
                Field(default=default, description=description),
            )

    return create_model(f"{name}Input", **field_definitions)


def _json_type_to_python(json_type: str) -> type:
    """Map a JSON Schema type string to a Python type."""
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


# ---------------------------------------------------------------------------
# Async context manager: full MCP subprocess bridge
# ---------------------------------------------------------------------------


@asynccontextmanager
async def mcp_session(role: str | None = None) -> AsyncIterator[list[BaseTool]]:
    """Launch the data_explorer MCP server and yield adapted LangChain tools.

    Usage::

        async with mcp_session(role="scout") as tools:
            agent = create_react_agent(llm, tools)
            ...

    The server process is automatically started when entering the context
    and torn down on exit.

    Args:
        role: Optional RBAC role name (e.g., "scout", "analyst"). If provided,
              tools are filtered to only include those permitted for the role.

    Yields
    ------
    list[BaseTool]
        LangChain tools backed by the live MCP connection, filtered by role.
    """
    server_params = _build_server_params()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover tools advertised by the MCP server.
            tool_list = await session.list_tools()

            langchain_tools: list[BaseTool] = []
            for tool in tool_list.tools:
                lc_tool = _mcp_tool_to_langchain(
                    session=session,
                    tool_name=tool.name,
                    tool_description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                )
                langchain_tools.append(lc_tool)

            # Apply RBAC permission filtering if role is specified
            if role:
                checker = PermissionChecker.for_role(role)
                langchain_tools = checker.filter_tools(langchain_tools)
                logger.info(
                    "MCP session: role='%s', tools after filtering: %s",
                    role,
                    [t.name for t in langchain_tools],
                )

            # Discover resources and prompts (for observability / future use)
            try:
                resources = await session.list_resources()
                logger.info(
                    "MCP server resources: %s",
                    [r.uri for r in resources.resources] if hasattr(resources, 'resources') else [],
                )
            except Exception:
                pass  # Resources not supported by all servers

            try:
                prompts = await session.list_prompts()
                logger.info(
                    "MCP server prompts: %s",
                    [p.name for p in prompts.prompts] if hasattr(prompts, 'prompts') else [],
                )
            except Exception:
                pass

            yield langchain_tools


# ---------------------------------------------------------------------------
# High-level async factory
# ---------------------------------------------------------------------------


async def create_mcp_tools() -> tuple[list[BaseTool], Any]:
    """Create LangChain tools connected to a live MCP data_explorer server.

    Returns
    -------
    tuple[list[BaseTool], Any]
        A two-element tuple:
        - The list of LangChain ``BaseTool`` instances.
        - The async context manager (``mcp_session``) that the caller must
          keep alive for the duration of tool usage and explicitly close
          when done.

    Example::

        tools, ctx = await create_mcp_tools()
        # ... use tools ...
        await ctx.__aexit__(None, None, None)  # or use aexit for cleanup
    """
    ctx = mcp_session()
    tools = await ctx.__aenter__()
    return tools, ctx


# ---------------------------------------------------------------------------
# Synchronous fallback: direct in-process wrappers (no MCP subprocess)
# ---------------------------------------------------------------------------


def create_sync_mcp_tools(role: str | None = None) -> list[BaseTool]:
    """Create synchronous LangChain tools that call data_explorer directly.

    Instead of spawning an MCP server subprocess, this function imports the
    tool functions from ``data_explorer`` and wraps them as plain LangChain
    ``StructuredTool`` instances. This is ideal for:

    - Unit tests and demos
    - Environments where subprocess creation is restricted
    - Quick local experimentation without MCP protocol overhead

    Args:
        role: Optional RBAC role name. If provided, only tools permitted
              for the role are returned.

    Returns:
        LangChain tools mirroring the MCP server's tool surface, filtered
        by role permissions.
    """
    # Import the underlying implementations.
    from insightflow.data_mcp.data_explorer import (
        get_schema,
        load_csv,
        profile,
        safe_query,
        sample_rows,
    )

    # ---- load_csv --------------------------------------------------------

    def _load_csv(path: str) -> str:
        """Load a CSV file and return basic metadata (row count, columns, etc.)."""
        return load_csv(path)

    load_csv_tool = StructuredTool.from_function(
        func=_load_csv,
        name="load_csv",
        description=(
            "Load a CSV file into memory and return basic metadata including "
            "row count, column count, column names, and memory usage."
        ),
    )

    # ---- get_schema ------------------------------------------------------

    def _get_schema(path: str) -> str:
        """Return column dtypes, non-null rate, and unique value counts."""
        return get_schema(path)

    get_schema_tool = StructuredTool.from_function(
        func=_get_schema,
        name="get_schema",
        description=(
            "Return schema information for every column: dtype, non-null "
            "rate, and unique value count."
        ),
    )

    # ---- sample_rows -----------------------------------------------------

    def _sample_rows(path: str, n: int = 5) -> str:
        """Return n randomly sampled rows from the CSV."""
        return sample_rows(path, n)

    sample_rows_tool = StructuredTool.from_function(
        func=_sample_rows,
        name="sample_rows",
        description="Return n randomly sampled rows from the loaded CSV file.",
    )

    # ---- profile ---------------------------------------------------------

    def _profile(path: str) -> str:
        """Generate a comprehensive data profile with stats and distributions."""
        return profile(path)

    profile_tool = StructuredTool.from_function(
        func=_profile,
        name="profile",
        description=(
            "Generate a comprehensive data profile: descriptive statistics "
            "for numeric columns, missing value summary, and value "
            "distributions (top 5) for categorical columns."
        ),
    )

    # ---- safe_query ------------------------------------------------------

    def _safe_query(path: str, expression: str) -> str:
        """Execute a safe, read-only pandas query expression on the DataFrame."""
        return safe_query(path, expression)

    safe_query_tool = StructuredTool.from_function(
        func=_safe_query,
        name="safe_query",
        description=(
            "Execute a safe, read-only pandas query expression on the loaded "
            "DataFrame. The variable 'df' refers to the DataFrame. Only "
            "read-only operations are permitted; mutations and I/O are rejected."
        ),
    )

    all_tools = [
        load_csv_tool,
        get_schema_tool,
        sample_rows_tool,
        profile_tool,
        safe_query_tool,
    ]

    # Apply RBAC permission filtering if role is specified
    if role:
        checker = PermissionChecker.for_role(role)
        filtered = checker.filter_tools(all_tools)
        logger.info(
            "create_sync_mcp_tools: role='%s', %d/%d tools after filtering",
            role,
            len(filtered),
            len(all_tools),
        )
        return filtered

    return all_tools
