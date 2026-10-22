"""Governed derived Capability Frontier contracts."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash

_SHA=r"^sha256:[0-9a-f]{64}$"
def _digest(v): return "sha256:"+content_hash(v)

class _Frozen(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)

Disposition=Literal["NOT_APPLICABLE","UNRESOLVED_APPLICABILITY","EVIDENCE_INSUFFICIENT","AUTHORITY_INSUFFICIENT","UNAVAILABLE","PRECONDITION_UNSATISFIED"]

class CapabilityFrontierEntryV1(_Frozen):
    competency_key:str
    projection_ref:str
    projection_hash:str=Field(pattern=_SHA)
    decision_contract_ref:str
    decision_contract_hash:str=Field(pattern=_SHA)

class CapabilityExclusionV1(CapabilityFrontierEntryV1):
    disposition:Disposition
    reason_codes:tuple[str,...]=()

class CapabilityFrontierV1(_Frozen):
    schema_version:Literal["capability_frontier.v1"]="capability_frontier.v1"
    frontier_ref:str
    frontier_hash:str=Field(pattern=_SHA)
    organization_id:str
    workspace_id:str
    working_context_ref:str
    working_context_hash:str=Field(pattern=_SHA)
    candidate_projection_refs:tuple[str,...]
    eligible_capabilities:tuple[CapabilityFrontierEntryV1,...]
    excluded_capabilities:tuple[CapabilityExclusionV1,...]
    contraction_policy_ref:Literal["capability-contraction-policy:v1"]="capability-contraction-policy:v1"
    contraction_policy_hash:str=Field(pattern=_SHA)
    authority_class:Literal["DERIVED_NON_AUTHORITY"]="DERIVED_NON_AUTHORITY"

    @model_validator(mode="after")
    def invariants(self):
        refs=self.candidate_projection_refs
        if refs != tuple(sorted(refs)) or len(refs)!=len(set(refs)):
            raise ValueError("candidate projection refs must be canonical and unique")
        eligible=tuple(x.projection_ref for x in self.eligible_capabilities)
        excluded=tuple(x.projection_ref for x in self.excluded_capabilities)
        if set(eligible)&set(excluded) or set(eligible)|set(excluded)!=set(refs):
            raise ValueError("frontier must account for every candidate exactly once")
        if self.eligible_capabilities != tuple(sorted(self.eligible_capabilities,key=lambda x:x.competency_key)):
            raise ValueError("eligible capabilities must be canonical")
        if self.excluded_capabilities != tuple(sorted(self.excluded_capabilities,key=lambda x:x.competency_key)):
            raise ValueError("excluded capabilities must be canonical")
        digest=_digest(self.hash_payload())
        if (self.frontier_ref,self.frontier_hash)!=("capability-frontier:v1:"+digest[7:],digest):
            raise ValueError("capability frontier identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json",exclude={"frontier_ref","frontier_hash"})

class CapabilityRouteSelectionV1(_Frozen):
    schema_version:Literal["capability_route_selection.v1"]="capability_route_selection.v1"
    frontier_ref:str
    frontier_hash:str=Field(pattern=_SHA)
    disposition:Literal["SELECTED","NO_ADMISSIBLE_CAPABILITY","ROUTE_UNRESOLVED"]
    selected:CapabilityFrontierEntryV1|None=None

    @model_validator(mode="after")
    def consistent(self):
        if (self.disposition=="SELECTED") != (self.selected is not None):
            raise ValueError("route selection disposition/selection mismatch")
        return self

class CapabilityContractionAssessmentV1(_Frozen):
    """Non-authoritative gate assessment bound to exact owner evidence pins."""
    competency_key: str
    evidence_sufficient: bool | None = None
    authorization_sufficient: bool | None = None
    available: bool | None = None
    preconditions_satisfied: bool | None = None
    basis_refs: tuple["ArtifactPinV1", ...] = ()

    @model_validator(mode="after")
    def gates_require_basis(self):
        if any(value is not None for value in (
            self.evidence_sufficient, self.authorization_sufficient,
            self.available, self.preconditions_satisfied,
        )) and not self.basis_refs:
            raise ValueError("contraction gate assessment requires exact owner basis")
        return self

from contracts.ontology_building import ArtifactPinV1
CapabilityContractionAssessmentV1.model_rebuild()
