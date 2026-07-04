"""RBAC (Role-Based Access Control) for MCP tools in the InsightFlow pipeline.

Implements a permission model where each MCP tool has a required scope and
risk level. Agents receive filtered tool lists based on their granted scopes,
and high-risk operations require explicit confirmation.

This demonstrates deep understanding of MCP beyond simple tool registration:
- Tools are not blindly exposed to all agents
- Each agent role gets a permission set matched to its responsibilities
- The bridge layer filters tools before the agent sees them

MCP 工具权限管控 —— 基于角色的访问控制（RBAC）。
每个工具定义所需权限域和风险等级，Agent 只能看到被授权的工具。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPermission:
    """Permission metadata for a single MCP tool.

    Attributes:
        required_scope: The scope string required to use this tool
                        (e.g., "data:read", "data:write", "data:admin").
        risk_level: How destructive the tool could be.
        requires_confirmation: Whether the tool needs explicit user approval
                               before execution (for interactive settings).
    """

    required_scope: str
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = False


# Registry: tool_name -> ToolPermission
TOOL_PERMISSIONS: dict[str, ToolPermission] = {
    # --- Read-only tools (low risk) ---
    "load_csv": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "get_schema": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "sample_rows": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "profile": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),

    # --- Query tool (medium risk — eval sandbox) ---
    "safe_query": ToolPermission(
        required_scope="data:query",
        risk_level="medium",
    ),

    # --- Write/mutation tools (medium-high risk) ---
    "fill_missing": ToolPermission(
        required_scope="data:write",
        risk_level="medium",
        requires_confirmation=True,
    ),
    "remove_outliers": ToolPermission(
        required_scope="data:write",
        risk_level="high",
        requires_confirmation=True,
    ),
    "normalize_column": ToolPermission(
        required_scope="data:write",
        risk_level="medium",
    ),

    # --- Visualization (low risk, but writes files) ---
    "create_chart": ToolPermission(
        required_scope="output:write",
        risk_level="low",
    ),

    # --- Analysis tools (read-only, low risk) ---
    "correlation_analysis": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "group_statistics": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "describe_numeric": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "value_distribution": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
    "get_dataframe_info": ToolPermission(
        required_scope="data:read",
        risk_level="low",
    ),
}


# ---------------------------------------------------------------------------
# Role presets — pre-defined scope sets for each agent role
# ---------------------------------------------------------------------------

ROLE_SCOPES: dict[str, set[str]] = {
    "scout": {"data:read", "data:query"},
    "cleaner": {"data:read", "data:write"},
    "analyst": {"data:read", "data:query"},
    "visualizer": {"data:read", "output:write"},
    "reporter": set(),  # Reporter uses no tools
    "admin": {"data:read", "data:write", "data:query", "data:admin", "output:write"},
}


# ---------------------------------------------------------------------------
# PermissionChecker
# ---------------------------------------------------------------------------


class PermissionChecker:
    """Checks tool access based on granted scopes.

    Usage::

        checker = PermissionChecker.for_role("scout")
        if checker.check("load_csv"):
            # agent can use load_csv
            ...
        filtered_tools = checker.filter_tools(all_tools)

    Args:
        granted_scopes: Set of scope strings this checker allows.
    """

    def __init__(self, granted_scopes: set[str]) -> None:
        self._scopes: set[str] = set(granted_scopes)

    @classmethod
    def for_role(cls, role: str) -> PermissionChecker:
        """Create a PermissionChecker with the scopes for a predefined role.

        Args:
            role: One of the keys in ROLE_SCOPES (e.g., "scout", "analyst").

        Returns:
            A PermissionChecker with the role's scopes.
        """
        scopes = ROLE_SCOPES.get(role, set())
        return cls(granted_scopes=scopes)

    @property
    def scopes(self) -> set[str]:
        """Return the set of granted scopes."""
        return set(self._scopes)

    def check(self, tool_name: str) -> bool:
        """Check if a tool is accessible with the current scopes.

        Args:
            tool_name: The name of the tool to check.

        Returns:
            True if the tool's required scope is in the granted set,
            or if the tool has no registered permission (permissive default).
        """
        perm = TOOL_PERMISSIONS.get(tool_name)
        if perm is None:
            # Unknown tools are allowed by default (could be tightened)
            return True
        return perm.required_scope in self._scopes

    def check_risk(self, tool_name: str, max_risk: Literal["low", "medium", "high"] = "high") -> bool:
        """Check if a tool's risk level is within the allowed maximum.

        Args:
            tool_name: The name of the tool.
            max_risk: Maximum allowed risk level.

        Returns:
            True if the tool's risk is <= max_risk.
        """
        risk_order = {"low": 0, "medium": 1, "high": 2}
        perm = TOOL_PERMISSIONS.get(tool_name)
        if perm is None:
            return True
        return risk_order.get(perm.risk_level, 2) <= risk_order.get(max_risk, 2)

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation before execution."""
        perm = TOOL_PERMISSIONS.get(tool_name)
        return perm.requires_confirmation if perm else False

    def filter_tools(self, tools: list[Any]) -> list[Any]:
        """Filter a list of LangChain tools based on permissions.

        Args:
            tools: List of LangChain BaseTool instances.

        Returns:
            Filtered list containing only tools the checker allows.
        """
        filtered = [t for t in tools if self.check(t.name)]
        excluded_count = len(tools) - len(filtered)
        if excluded_count > 0:
            excluded_names = [t.name for t in tools if not self.check(t.name)]
            logger.info(
                "PermissionChecker filtered out %d tool(s): %s (granted scopes: %s)",
                excluded_count,
                excluded_names,
                self._scopes,
            )
        return filtered

    def get_tool_metadata(self, tool_name: str) -> dict[str, Any] | None:
        """Return permission metadata for a tool, or None if unregistered."""
        perm = TOOL_PERMISSIONS.get(tool_name)
        if perm is None:
            return None
        return {
            "scope": perm.required_scope,
            "risk_level": perm.risk_level,
            "requires_confirmation": perm.requires_confirmation,
        }
