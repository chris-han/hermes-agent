from __future__ import annotations

TOOLSET_NAME = "governed_data_engineering"


def _schema(description: str, properties: dict, required: tuple[str, ...] = ()):
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


CSV_INPUT_PROPERTIES = {
    "csv_path": {
        "type": "string",
        "description": "Explicit CSV artifact path supplied by the user or workspace runtime context.",
    },
    "encoding": {"type": "string", "default": "utf-8-sig"},
    "delimiter": {"type": "string", "default": ","},
}


PROFILE_TABLE_SCHEMA = _schema(
    "Profile an explicit CSV table artifact with deterministic, dependency-light diagnostics.",
    {
        **CSV_INPUT_PROPERTIES,
        "target": {"type": "string", "description": "Optional target column for leakage and class-balance hints."},
        "sample_limit": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
        "max_rows": {"type": "integer", "minimum": 1, "default": 50000},
    },
    ("csv_path",),
)


QUALITY_GATE_SCHEMA = _schema(
    "Run explicit quality rules against a CSV table artifact and return PASS/FAIL.",
    {
        **CSV_INPUT_PROPERTIES,
        "rules": {
            "type": "object",
            "description": (
                "Quality rules. Supports dataset.min_rows/max_rows/no_duplicate_rows "
                "and columns.<name>.required/non_null/unique/min/max/enum/regex."
            ),
        },
        "max_rows": {"type": "integer", "minimum": 1, "default": 50000},
    },
    ("csv_path", "rules"),
)


SAMPLE_SIZE_CHECK_SCHEMA = _schema(
    "Run rough sample-size sufficiency heuristics. This is not a formal power analysis.",
    {
        "task": {"type": "string", "enum": ["classification", "regression", "detection"]},
        "n": {"type": "integer", "minimum": 0},
        "features": {"type": "integer", "minimum": 0, "default": 0},
        "classes": {"type": "integer", "minimum": 0, "default": 0},
        "per_class": {"type": "array", "items": {"type": "integer", "minimum": 0}},
    },
    ("task", "n"),
)


DATA_FEASIBILITY_SCHEMA = _schema(
    "Combine four explicit data-readiness answers into a deterministic feasibility summary.",
    {
        "project": {"type": "string"},
        "data_support": {"type": "string", "enum": ["ok", "warn", "insufficient"]},
        "quality": {"type": "string", "enum": ["ok", "warn", "insufficient"]},
        "scale": {"type": "string", "enum": ["ok", "warn", "insufficient"]},
        "feature_value": {"type": "string", "enum": ["ok", "warn", "insufficient"]},
        "evidence": {
            "type": "object",
            "description": "Optional evidence strings keyed by data_support, quality, scale, feature_value.",
        },
    },
    ("data_support", "quality", "scale", "feature_value"),
)


COMPANY_LAKEHOUSE_REFRESH_SCHEMA = _schema(
    (
        "Refresh the governed lakehouse for a target Semantier company or organization. "
        "Use company_id/organization_id and company_type/dataset_type when the active governed "
        "session context is not already bound. Supports DEMO and REAL datasets."
    ),
    {
        "organization_id": {"type": "string"},
        "company_id": {
            "type": "string",
            "description": "Alias for organization_id when targeting a specific governed company.",
        },
        "dataset_type": {
            "type": "string",
            "description": "Governed organization dataset type. Examples: DEMO or REAL.",
        },
        "company_type": {"type": "string", "description": "Alias for dataset_type. Examples: DEMO or REAL."},
        "active_dataset_version_id": {"type": "string"},
        "dataset_version_id": {
            "type": "string",
            "description": "Alias for active_dataset_version_id when targeting a versioned company dataset.",
        },
        "dataset_key": {
            "type": "string",
            "description": "Governed demo dataset key. Used as the refresh dataset identifier when no active_dataset_version_id exists.",
        },
        "requested_by": {"type": "string"},
        "idempotency_key": {"type": "string"},
        "db_path": {"type": "string"},
        "sim_json": {"type": "string"},
        "output_dir": {"type": "string"},
        "replace": {"type": "boolean", "default": True},
    },
    ("requested_by", "idempotency_key"),
)
