"""统一的 JSON 提取器 —— 多种 fallback 策略。

替代原来分散在 Scout / Cleaner / Analyst 中的重复 JSON 解析逻辑。
本模块提供单一、经过测试的实现，额外支持 fenced code block 提取
和 schema 校验。
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Fenced code-block 模式: ```json ... ``` 或 ``` ... ```
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def extract_json(
    text: str,
    *,
    default: dict[str, Any] | None = None,
    expect_list: bool = False,
) -> dict[str, Any] | list[Any]:
    """从 LLM 输出中提取 JSON，支持多种 fallback 策略。

    策略（按顺序）:
    1. 直接 ``json.loads`` 整个文本
    2. 从 fenced code block 中提取（```` ```json ... ``` ````)
    3. 查找最外层的 ``{ }`` 或 ``[ ]`` 括号对
    4. 返回 *default* 或抛出 ``ValueError``

    Args:
        text: 应包含 JSON 的 LLM 原始输出。
        default: 所有策略失败时返回此值（而非抛异常）。
        expect_list: 为 True 时，括号搜索优先使用 ``[ ]``。

    Returns:
        解析后的 JSON dict（若 *expect_list* 为 True 则可能是 list）。

    Raises:
        ValueError: 所有策略失败且 *default* 为 None。
    """
    if text is None:
        text = ""

    # --- 策略 1: 直接解析 ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # --- 策略 2: fenced code block ---
    for match in _FENCED_JSON_RE.finditer(text):
        fragment = match.group(1).strip()
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    # --- 策略 3: 最外层括号对 ---
    open_char = "[" if expect_list else "{"
    close_char = "]" if expect_list else "}"

    start = text.find(open_char)
    end = text.rfind(close_char)

    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 也尝试另一种括号类型作为补充
    alt_open = "{" if expect_list else "["
    alt_close = "}" if expect_list else "]"

    start = text.find(alt_open)
    end = text.rfind(alt_close)
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # --- 策略 4: fallback ---
    if default is not None:
        return default

    raise ValueError(
        f"Failed to extract JSON from LLM output (length={len(text)}). "
        f"First 200 chars: {text[:200]!r}"
    )


def extract_json_with_schema(
    text: str,
    schema: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """提取 JSON 并根据简易 schema 校验/填充。

    *schema* 是一个简化的规格 dict：
    - 键为期望的字段名。
    - 值为以下之一：
      - Python 类型（``str``, ``int``, ``list``, ``dict`` 等）用于类型检查
      - 元组 ``(type, default_value)`` 同时指定类型和默认值

    缺失字段会从 schema 默认值填充。若 *strict* 为 True，
    类型不匹配也会被替换为默认值。

    Args:
        text: LLM 原始输出。
        schema: 简化的 schema 规格 dict。
        strict: 为 True 时强制类型检查，不匹配则替换为默认值。

    Returns:
        包含所有 schema 字段的 dict（缺失的从默认值填充）。

    Example::

        schema = {
            "summary": (str, ""),
            "findings": (list, []),
            "statistics": (dict, {}),
            "data_quality_note": (str, ""),
        }
        result = extract_json_with_schema(llm_output, schema)
    """
    # 用空 dict 作为默认值，确保提取始终返回 dict
    raw = extract_json(text, default={})
    if not isinstance(raw, dict):
        raw = {"raw_value": raw}

    result: dict[str, Any] = dict(raw)

    for field_name, spec in schema.items():
        if isinstance(spec, tuple) and len(spec) == 2:
            expected_type, default_value = spec
        else:
            expected_type = spec
            default_value = None

        if field_name not in result:
            # 填充缺失字段
            result[field_name] = default_value
        elif strict and expected_type is not None:
            # 强制类型检查
            if not isinstance(result[field_name], expected_type):
                result[field_name] = default_value

    return result


# ---------------------------------------------------------------------------
# InsightFlow 各 Agent 的预设 schema
# ---------------------------------------------------------------------------

SCOUT_PROFILE_SCHEMA: dict[str, Any] = {
    "basic_info": (dict, {}),
    "column_details": (dict, {}),
    "numeric_summary": (dict, {}),
    "quality_issues": (list, []),
    "categorical_summary": (dict, {}),
}

CLEANER_PLAN_SCHEMA: dict[str, Any] = {
    "strategy": (list, []),
    "overall_notes": (str, ""),
}

ANALYST_RESULTS_SCHEMA: dict[str, Any] = {
    "summary": (str, ""),
    "findings": (list, []),
    "statistics": (dict, {}),
    "data_quality_note": (str, ""),
}
