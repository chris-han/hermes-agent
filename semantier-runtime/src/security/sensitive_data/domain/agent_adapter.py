from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from security.sensitive_data.rendering.output_governance import (
    govern_simulated_agent_output,
)
from security.sensitive_data.rendering.workshop_school_output import (
    govern_workshop_school_output,
)
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class GovernedAgentExecution:
    governed_response: dict[str, Any]
    execution_boundary: dict[str, Any]
    authority_objects: list[dict[str, Any]]


class LocalSimulatedGovernedAgentAdapter:
    adapter_ref = "local_simulated_sensitive_governance_agent_v1"

    def execute(
        self,
        *,
        run_id: str,
        assessment_id: str,
        assessment_hash: str,
        transformed_input: str,
        transformed_input_hash: str,
        execution_boundary: dict[str, Any],
        purpose: str | None = None,
        source_evidence: dict[str, Any] | None = None,
        findings: list[dict[str, Any]] | None = None,
        decisions: list[dict[str, Any]] | None = None,
    ) -> GovernedAgentExecution:
        computed_hash = canonical_hash({"transformed_input": transformed_input})
        if computed_hash != transformed_input_hash:
            raise ValueError("transformed_input_hash mismatch")
        boundary = {
            **execution_boundary,
            "adapter_ref": self.adapter_ref,
            "transformed_input_hash": transformed_input_hash,
            "received_input_hash": computed_hash,
            "raw_input_received": False,
        }
        if purpose and purpose.startswith("workshop_") and source_evidence:
            governed_response, authority_objects = govern_workshop_school_output(
                run_id=run_id,
                assessment_id=assessment_id,
                assessment_hash=assessment_hash,
                purpose=purpose,
                source_evidence=source_evidence,
                findings=findings or [],
                decisions=decisions or [],
                execution_boundary=boundary,
            )
            return GovernedAgentExecution(
                governed_response=governed_response,
                execution_boundary=boundary,
                authority_objects=authority_objects,
            )
        output = govern_simulated_agent_output(
            run_id=run_id,
            assessment_id=assessment_id,
            assessment_hash=assessment_hash,
            execution_boundary=boundary,
        )
        return GovernedAgentExecution(
            governed_response=output.governed_response,
            execution_boundary=boundary,
            authority_objects=output.authority_objects,
        )
