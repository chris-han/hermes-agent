"""Stage-aware routing over the existing ChangeCandidate target/failure taxonomy."""

from contracts.candidate_stage_binding import CandidateStageBindingV1


class CandidateStageRoutingService:
    OWNER_BY_STAGE = {
        "DESCRIPTIVE": "descriptive-ontology-owner",
        "NORMATIVE": "normative-semantics-owner",
        "EPISTEMIC": "epistemic-evaluator-owner",
        "DECISION": "decision-formation-owner",
        "COMMITMENT": "commitment-policy-owner",
        "ENVIRONMENT": "environment-owner",
    }

    def bind(self, *, target: str, failure_class: str, owning_stage: str) -> CandidateStageBindingV1:
        if target == "WORKFLOW":
            raise ValueError("WORKFLOW_TARGET_ROUTE_FORBIDDEN")
        owner_ref = self.OWNER_BY_STAGE.get(owning_stage)
        if owner_ref is None:
            raise ValueError("UNKNOWN_OWNING_STAGE")
        return CandidateStageBindingV1(
            target=target,
            failure_class=failure_class,
            owning_stage=owning_stage,
            owner_ref=owner_ref,
        )

    @staticmethod
    def assert_route(binding: CandidateStageBindingV1, *, requested_owner_ref: str) -> None:
        if binding.owner_ref != requested_owner_ref:
            raise ValueError("UNAUTHORIZED_STAGE_ROUTE")
        if binding.owning_stage == "COMMITMENT" and requested_owner_ref != "commitment-policy-owner":
            raise ValueError("UNAUTHORIZED_STAGE_ROUTE")
