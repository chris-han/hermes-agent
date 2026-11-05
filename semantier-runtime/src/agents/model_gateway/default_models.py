"""Platform-owned signup preferences; not a model deployment or tariff authority.

The recommendation is a logical product preference. A funded account becomes
eligible only after an independently qualified per-tenant deployment and price.
"""
from __future__ import annotations

PLATFORM_RECOMMENDATION_VERSION = "2026-11-04.v1"
RECOMMENDED = "qwen3.5-plus"
RECOMMENDATION_MODE = "platform_recommended"
MODELS = (
    ("qwen3.5-plus", "Qwen3.5-Plus", "balanced", "均衡能力，适合日常工作与文档分析"),
    ("deepseek-v4.1-flash-1017", "DeepSeek Flash", "efficient", "适合常规对话和轻量任务"),
    ("qwen3.8-max", "Qwen3.8-Max", "advanced", "适合复杂任务，消耗额度可能更快"),
)


def normalize_preference(value):
    """Validate one bounded public preference without giving it runtime rights."""
    if value in (None, "", RECOMMENDATION_MODE):
        return RECOMMENDATION_MODE
    if isinstance(value, str) and value in {item[0] for item in MODELS}:
        return value
    raise ValueError("Unsupported model preference")


def resolved_model(preference):
    selected = normalize_preference(preference)
    return RECOMMENDED if selected == RECOMMENDATION_MODE else selected


def public_catalog():
    return {
        "version": PLATFORM_RECOMMENDATION_VERSION,
        "default_preference": RECOMMENDATION_MODE,
        "recommended_model": RECOMMENDED,
        "models": [
            {"id": key, "label": label, "purpose": purpose, "description_zh": zh}
            for key, label, purpose, zh in MODELS
        ],
        "notice": "Selection is a preference; availability and prices are verified at activation.",
    }
