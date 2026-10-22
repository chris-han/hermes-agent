"""Compatibility wrapper for installed console entrypoints.

See ``docs/canonical/architecture.md`` for the runtime boundary; this module exists only
to preserve historical import paths used by generated launch scripts.
"""

from agents.launcher import (
    run_agent_cli,
    run_api,
    run_gateway_cli,
    run_hermes_cli,
    run_industry_sim,
    run_runtime_cli,
    run_smb_sim,
)

__all__ = [
    "run_hermes_cli",
    "run_api",
    "run_agent_cli",
    "run_gateway_cli",
    "run_runtime_cli",
    "run_industry_sim",
    "run_smb_sim",
]
