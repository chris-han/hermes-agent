"""Deterministic governed contraction from competency candidates to F_C."""
from contracts.capability_frontier import (
    CapabilityContractionAssessmentV1, CapabilityExclusionV1, CapabilityFrontierEntryV1,
    CapabilityFrontierV1, CapabilityRouteSelectionV1,
)
from contracts.runtime_competency_projection import RuntimeCompetencyProjectionV1
from contracts.semantic_kernel import WorkingContext
from eos.artifact_hashing import content_hash

def _digest(v): return "sha256:"+content_hash(v)

class CapabilityFrontierService:
    @staticmethod
    def assessment_from_owners(*, competency_key: str, working_context: WorkingContext,
                               required_evidence_kinds: tuple[str, ...] = (),
                               authorization=None, required_operation: str | None = None,
                               availability_basis=None, available: bool | None = None,
                               precondition_basis=None, preconditions_satisfied: bool | None = None):
        """Project existing owner evidence into a non-authoritative contraction assessment."""
        from contracts.ontology_building import ArtifactPinV1
        basis = []
        evidence_sufficient = None
        if required_evidence_kinds:
            evidence = tuple(getattr(working_context, "evidence", ()))
            present = {pin.artifact_kind for pin in evidence}
            evidence_sufficient = all(kind in present for kind in required_evidence_kinds)
            basis.extend(evidence)
        authorization_sufficient = None
        if required_operation is not None:
            authorization_sufficient = bool(
                authorization is not None
                and authorization.operation == required_operation
                and authorization.organization_id == working_context.organization_id
                and authorization.workspace_id == working_context.workspace_id
            )
            if authorization is not None:
                basis.extend(authorization.governing_pins)
                basis.append(ArtifactPinV1(artifact_kind="authorization_resolver",
                    artifact_ref=authorization.resolver_ref, artifact_hash=authorization.resolver_hash))
        if available is not None:
            if availability_basis is None:
                raise ValueError("AVAILABILITY_GATE_REQUIRES_OWNER_BASIS")
            basis.append(availability_basis)
        if preconditions_satisfied is not None:
            if precondition_basis is None:
                raise ValueError("PRECONDITION_GATE_REQUIRES_OWNER_BASIS")
            basis.append(precondition_basis)
        unique = {(p.artifact_kind,p.artifact_ref,p.artifact_hash):p for p in basis}
        if any(v is not None for v in (evidence_sufficient,authorization_sufficient,available,preconditions_satisfied)) and not unique:
            raise ValueError("CONTRACTION_ASSESSMENT_REQUIRES_OWNER_BASIS")
        return CapabilityContractionAssessmentV1(
            competency_key=competency_key, evidence_sufficient=evidence_sufficient,
            authorization_sufficient=authorization_sufficient, available=available,
            preconditions_satisfied=preconditions_satisfied,
            basis_refs=tuple(unique[k] for k in sorted(unique)),
        )

    @staticmethod
    def contract(*, working_context: WorkingContext,
                 candidates: tuple[RuntimeCompetencyProjectionV1,...],
                 assessments: tuple[CapabilityContractionAssessmentV1,...]=()):
        by_key={x.competency_key:x for x in assessments}
        if len(by_key)!=len(assessments):
            raise ValueError("DUPLICATE_CONTRACTION_ASSESSMENT")
        eligible=[]; excluded=[]
        for c in sorted(candidates,key=lambda x:x.competency_key):
            if (c.organization_id,c.workspace_id,c.working_context_ref,c.working_context_hash)!=(
                working_context.organization_id,working_context.workspace_id,
                working_context.value_ref,working_context.value_hash):
                raise ValueError("CAPABILITY_CANDIDATE_CONTEXT_MISMATCH")
            entry=dict(competency_key=c.competency_key,projection_ref=c.projection_ref,
                       projection_hash=c.projection_hash,decision_contract_ref=c.decision_contract_ref,
                       decision_contract_hash=c.decision_contract_hash)
            disposition=None; reasons=[]
            if c.applicability=="NOT_APPLICABLE":
                disposition="NOT_APPLICABLE"; reasons.append("TRIAGE_NOT_APPLICABLE")
            elif c.applicability=="UNRESOLVED":
                disposition="UNRESOLVED_APPLICABILITY"; reasons.append("TRIAGE_UNRESOLVED")
            else:
                a=by_key.get(c.competency_key)
                if a:
                    gates=(
                        ("evidence_sufficient","EVIDENCE_INSUFFICIENT"),
                        ("authorization_sufficient","AUTHORITY_INSUFFICIENT"),
                        ("available","UNAVAILABLE"),
                        ("preconditions_satisfied","PRECONDITION_UNSATISFIED"),
                    )
                    for field,code in gates:
                        if getattr(a,field) is False:
                            disposition=code; reasons.append(code); break
            if disposition:
                excluded.append(CapabilityExclusionV1(**entry,disposition=disposition,reason_codes=tuple(reasons)))
            else:
                eligible.append(CapabilityFrontierEntryV1(**entry))
        refs=tuple(sorted(x.projection_ref for x in candidates))
        policy_hash=_digest({"policy":"capability-contraction","version":"v1",
            "order":["applicability","evidence","authority","availability","preconditions"],
            "missing_optional_gate":"not_evaluated"})
        body=dict(organization_id=working_context.organization_id,workspace_id=working_context.workspace_id,
            working_context_ref=working_context.value_ref,working_context_hash=working_context.value_hash,
            candidate_projection_refs=refs,eligible_capabilities=tuple(sorted(eligible,key=lambda x:x.competency_key)),
            excluded_capabilities=tuple(sorted(excluded,key=lambda x:x.competency_key)),
            contraction_policy_hash=policy_hash)
        draft=CapabilityFrontierV1.model_construct(frontier_ref="capability-frontier:v1:pending",
            frontier_hash="sha256:"+"0"*64,**body)
        digest=_digest(draft.hash_payload())
        return CapabilityFrontierV1.model_validate({**body,"frontier_ref":"capability-frontier:v1:"+digest[7:],
            "frontier_hash":digest})

    @staticmethod
    def select(frontier: CapabilityFrontierV1):
        n=len(frontier.eligible_capabilities)
        if n==0: disposition="NO_ADMISSIBLE_CAPABILITY"; selected=None
        elif n==1: disposition="SELECTED"; selected=frontier.eligible_capabilities[0]
        else: disposition="ROUTE_UNRESOLVED"; selected=None
        return CapabilityRouteSelectionV1(frontier_ref=frontier.frontier_ref,
            frontier_hash=frontier.frontier_hash,disposition=disposition,selected=selected)
