"""Minimal governed-context helpers for non-Semantier checkouts."""

from __future__ import annotations

import re
from typing import Any

_ANALYTICS_PATTERNS = (
    re.compile(
        r"(analytics|analysis|profit|margin|revenue|cost|forecast|trend|sql|query|dashboard|kpi)",
        re.IGNORECASE,
    ),
    re.compile(r"(成本|利润率|趋势|营收|收入|毛利|分析|报表|指标|查询|数据)"),
)


def is_analytics_query_message(user_message: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _ANALYTICS_PATTERNS)


def build_governed_runtime_context_prompt(source: Any, user_message: str) -> str | None:
    if not is_analytics_query_message(user_message):
        return None
    return (
        "Governed analytics intent detected. Prefer governed query surfaces and "
        "avoid shell or code-execution detours when answering this request."
    )
