"""Exact internal advisory commitments; never independent legal certification."""
from datetime import datetime
from typing import Literal
from pydantic import ConfigDict, Field, model_validator
from contracts.semantic_kernel import _ContentAddressed, DecisionRecord
from contracts.ontology_building import ArtifactPinV1
from contracts.evidence_location import SourceIdentity, EvidenceRef
from eos.artifact_hashing import content_hash

POLICY_REF = 'policy:tender-owner-internal-advisory:v1'
POLICY_HASH = 'sha256:' + content_hash({'policy': POLICY_REF, 'scope': 'OWNER_ADOPTED_INTERNAL_ADVISORY',
    'permission': 'current_governed_organization_owner', 'independent_legal_validation': 'NOT_ESTABLISHED',
    'validity': 'adoption_until_revocation', 'binding_legal_conclusion': False})

def ontology_contract_hash(ontology):
    """Compatibility preserves all semantic, scope, provenance and version fields."""
    from contracts.semantic_kernel import Ontology
    ontology = Ontology.model_validate(ontology.model_dump(mode='json'))
    return 'sha256:' + content_hash(ontology.model_dump(mode='json', exclude={'artifact_ref','artifact_hash','recorded_at'}))


class TenderOwnerAdoptionProposalV1(_ContentAddressed):
    # Preserve persisted payloads and content hashes while avoiding BaseModel.schema.
    model_config = ConfigDict(serialize_by_alias=True)
    ARTIFACT_KIND = 'tender_owner_adoption_proposal'
    schema_version: Literal['tender_owner_adoption_proposal.v1'] = 'tender_owner_adoption_proposal.v1'
    organization_id: str
    workspace_id: str
    basis_kind: Literal['OWNER_ADOPTED_INTERNAL_ADVISORY'] = 'OWNER_ADOPTED_INTERNAL_ADVISORY'
    policy_ref: Literal['policy:tender-owner-internal-advisory:v1'] = POLICY_REF
    policy_hash: str = POLICY_HASH
    source_identity: SourceIdentity
    canonical_source_ir: ArtifactPinV1
    source_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    candidate: ArtifactPinV1
    schema_: ArtifactPinV1 = Field(alias='schema')
    plan: ArtifactPinV1
    rule_ref: str = Field(min_length=1)
    canonical_pattern_ref: str = Field(min_length=1)
    canonical_pattern_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    pattern_retrieval_snapshot: ArtifactPinV1
    ontology_contract_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    context: ArtifactPinV1
    ontology: ArtifactPinV1
    legal_edition: None = None
    legal_effective_from: None = None
    jurisdiction: None = None
    unknowns: tuple[str, ...] = ('LEGAL_EDITION_NOT_ESTABLISHED', 'LEGAL_EFFECTIVITY_NOT_ESTABLISHED', 'JURISDICTION_NOT_ESTABLISHED')
    independent_legal_validation: Literal['NOT_ESTABLISHED'] = 'NOT_ESTABLISHED'
    binding_legal_conclusion: Literal[False] = False

    @model_validator(mode='after')
    def exact_scope(self):
        scope = (self.organization_id, self.workspace_id)
        if self.policy_hash != POLICY_HASH:
            raise ValueError('OWNER_ADOPTION_POLICY_MISMATCH')
        if self.unknowns != ('LEGAL_EDITION_NOT_ESTABLISHED', 'LEGAL_EFFECTIVITY_NOT_ESTABLISHED', 'JURISDICTION_NOT_ESTABLISHED'):
            raise ValueError('OWNER_ADOPTION_LEGAL_UNKNOWNS_REQUIRED')
        if (self.source_identity.organization_id, self.source_identity.workspace_id) != scope:
            raise ValueError('OWNER_ADOPTION_SCOPE_MISMATCH')
        for evidence in self.source_evidence:
            if ((evidence.organization_id, evidence.workspace_id) != scope or
                evidence.source_identity_ref != self.source_identity.source_identity_ref or
                evidence.selector.selector_hash() != evidence.selector_hash):
                raise ValueError('OWNER_ADOPTION_SOURCE_EVIDENCE_MISMATCH')
        for name, kind in (('canonical_source_ir','canonical_source_ir'), ('candidate','normative_semantic_candidate_set'),
            ('schema_','normative_schema_contract'), ('pattern_retrieval_snapshot','runtime_semantic_retrieval_snapshot'), ('plan','compiled_rule_plan'), ('context','context_graph'), ('ontology','ontology')):
            if getattr(self,name).artifact_kind != kind:
                raise ValueError('OWNER_ADOPTION_PIN_KIND_MISMATCH')
        return self

class TenderOwnerAdoptionV1(_ContentAddressed):
    ARTIFACT_KIND = 'tender_owner_adoption'
    schema_version: Literal['tender_owner_adoption.v1'] = 'tender_owner_adoption.v1'
    organization_id: str
    workspace_id: str
    proposal: ArtifactPinV1
    source_decision: ArtifactPinV1
    interpretation_decision: ArtifactPinV1
    actor_id: str
    adopted_at: datetime

class TenderOwnerRevocationV1(_ContentAddressed):
    ARTIFACT_KIND = 'tender_owner_revocation'
    schema_version: Literal['tender_owner_revocation.v1'] = 'tender_owner_revocation.v1'
    organization_id: str
    workspace_id: str
    adoption: ArtifactPinV1
    predecessor: ArtifactPinV1
    decision: ArtifactPinV1
    actor_id: str
    revoked_at: datetime


def validate_frozen(*, adoption, proposal, source_decision, interpretation_decision):
    """Pure exact-payload verification, without current membership or source reads."""
    proposal = TenderOwnerAdoptionProposalV1.model_validate(proposal.model_dump(mode='json'))
    adoption = TenderOwnerAdoptionV1.model_validate(adoption.model_dump(mode='json'))
    if adoption.proposal != proposal.pin() or (adoption.organization_id,adoption.workspace_id) != (proposal.organization_id,proposal.workspace_id):
        raise ValueError('OWNER_ADOPTION_PROPOSAL_MISMATCH')
    for operation, decision, pin, subject in (
        ('SOURCE_ADOPTION', source_decision, adoption.source_decision, proposal.canonical_source_ir),
        ('INTERPRETATION_ADOPTION', interpretation_decision, adoption.interpretation_decision, proposal.candidate)):
        decision = DecisionRecord.model_validate(decision.model_dump(mode='json'))
        if (decision.pin() != pin or decision.purpose != 'CONSEQUENTIAL_DECISION' or decision.result_code != operation
            or decision.subject != subject or decision.context != proposal.context or decision.ontology != proposal.ontology
            or decision.actor_id != adoption.actor_id or decision.recorded_at != adoption.adopted_at
            or (decision.organization_id,decision.workspace_id) != (proposal.organization_id,proposal.workspace_id)
            or decision.reasoner_identity != POLICY_REF or decision.reasoning.reasoning_version != POLICY_HASH
            or decision.reasoning.diagnostics != (proposal.model_dump_json(),)
            or decision.authorization.operation != operation):
            raise ValueError('OWNER_ADOPTION_DECISION_MISMATCH')
    return adoption


def validate_revocation(*, adoption, proposal, revocation, decision):
    revocation = TenderOwnerRevocationV1.model_validate(revocation.model_dump(mode='json'))
    decision = DecisionRecord.model_validate(decision.model_dump(mode='json'))
    if (revocation.adoption != adoption.pin() or revocation.predecessor != adoption.pin()
        or revocation.decision != decision.pin() or revocation.revoked_at < adoption.adopted_at
        or (revocation.organization_id,revocation.workspace_id) != (proposal.organization_id,proposal.workspace_id)
        or decision.purpose != 'CONSEQUENTIAL_DECISION' or decision.result_code != 'REVOKE_OWNER_ADOPTION'
        or decision.subject != adoption.pin() or decision.actor_id != revocation.actor_id
        or decision.recorded_at != revocation.revoked_at or decision.context != proposal.context or decision.ontology != proposal.ontology
        or decision.reasoner_identity != POLICY_REF or decision.reasoning.reasoning_version != POLICY_HASH
        or decision.reasoning.diagnostics != (proposal.model_dump_json(),)
        or decision.authorization.operation != 'REVOKE_OWNER_ADOPTION'):
        raise ValueError('OWNER_REVOCATION_DECISION_MISMATCH')
    return revocation
