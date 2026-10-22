from __future__ import annotations

TOOLSET_NAME = "real_company_onboarding"

CONTEXT_PROPERTIES = {
    "organization_id": {"type": "string"},
    "dataset_type": {
        "type": "string",
        "description": "Governed organization dataset type. Use DEMO for demo organizations; use REAL only for uploaded real-company source data.",
    },
    "active_dataset_version_id": {"type": "string"},
}

STATUS_SCHEMA = {
    "description": (
        "Read the governed REAL company setup status from Semantier core. "
        "Use automate_excel for file mechanics and governed_data_engineering for advisory data diagnostics."
    ),
    "parameters": {
        "type": "object",
        "properties": CONTEXT_PROPERTIES,
        "required": ["organization_id"],
    },
}

WRITE_SCHEMA = {
    "description": (
        "Run a deterministic REAL company setup stage through Semantier core after file mechanics "
        "and advisory diagnostics have been handled by automate_excel and governed_data_engineering where needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **CONTEXT_PROPERTIES,
            "requested_by": {"type": "string"},
            "idempotency_key": {"type": "string"},
            "member_role": {"type": "string"},
            "can_change_settings": {"type": "boolean", "default": False},
        },
        "required": ["organization_id", "active_dataset_version_id", "requested_by", "idempotency_key"],
    },
}

READ_ONLY_DATASET_SCHEMA = {
    "description": (
        "Read governed REAL company setup diagnostics from Semantier core. "
        "For table profiling or quality gates over explicit artifacts, use governed_data_engineering."
    ),
    "parameters": {
        "type": "object",
        "properties": CONTEXT_PROPERTIES,
        "required": ["organization_id"],
    },
}
