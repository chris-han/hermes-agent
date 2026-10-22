"""Lossless replay surface and conservative compact decision view, version two."""
from __future__ import annotations

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from contracts.decision_local_projection import DecisionContractV1, DecisionLocalProjectionV1
from contracts.decision_materiality_policy import DecisionMaterialityPolicyV1
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from contracts.semantic_kernel import ContextGraph, DecisionRecord, Evidence, Ontology, WorkingContext
from contracts.observed_constraint_hypothesis import EffectiveRestrictionRuntimeContextV1, RuntimeSemanticRetrievalSnapshotV1
from eos.artifact_hashing import content_hash


def digest(value) -> str:
    return 'sha256:' + content_hash(value)


COMPILER_BASIS = {
    'compiler': 'bounded_decision_local_projection', 'version': 'v2',
    'representation': 'selected assertion labels and exact source spans with projection-local provenance handles',
    'materiality_policy_hash': DecisionMaterialityPolicyV1().policy_hash,
}
BOUNDED_COMPILER = ArtifactPinV1(artifact_kind='decision_projection_compiler',
    artifact_ref='bounded-decision-projection-compiler:v2:' + digest(COMPILER_BASIS)[7:],
    artifact_hash=digest(COMPILER_BASIS))


class DecisionLocalProjectionV2(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal['decision_local_projection.v2'] = 'decision_local_projection.v2'
    bounded_compiler: ArtifactPinV1 = BOUNDED_COMPILER
    legacy: DecisionLocalProjectionV1
    context: ContextGraph
    working_context: WorkingContext
    ontology: Ontology
    evidence: tuple[Evidence, ...]
    source_region: RuntimeSemanticSourceRegionV1
    decision_contract: DecisionContractV1
    runtime_context: EffectiveRestrictionRuntimeContextV1
    retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1
    canonical_pattern_ref: str
    governed_rules: tuple[dict, ...]
    proposed_context: ContextGraph
    admission_decision: DecisionRecord
    unselected_evidence: tuple[Evidence, ...] = ()
    policy_hash: str
    model_input_hash: str
    omission_manifest_hash: str
    projection_hash: str

    @model_validator(mode='after')
    def verify(self):
        legacy = self.legacy
        if self.bounded_compiler != BOUNDED_COMPILER:
            raise ValueError('UNTRUSTED_BOUNDED_COMPILER')
        if (self.context.pin(), self.working_context.pin(), self.ontology.pin()) != (
                legacy.context_graph, legacy.working_context, legacy.ontology):
            raise ValueError('REPLAY_PIN_MISMATCH')
        from services.decision_local_projection_service import DecisionLocalProjectionService
        from services.working_context_selection_policy import working_context_selection_policy
        compiler = DecisionLocalProjectionService.compiler_pin()
        if (legacy.compiler_ref, legacy.compiler_hash) != (compiler.artifact_ref, compiler.artifact_hash):
            raise ValueError('UNTRUSTED_LEGACY_COMPILER')
        if self.working_context.selection_policy != working_context_selection_policy():
            raise ValueError('UNTRUSTED_WORKING_CONTEXT_SELECTOR')
        contract = self.decision_contract
        if (legacy.decision_contract_ref, legacy.decision_contract_hash, legacy.semantic_predicates,
                legacy.deterministic_constraints) != (contract.contract_ref, contract.contract_hash,
                contract.semantic_predicates, contract.deterministic_constraints):
            raise ValueError('DECISION_CONTRACT_BINDING_MISMATCH')
        if legacy.decision_subject_ref != self.source_region.source_region_ref or legacy.decision_subject_ref not in self.working_context.selected_node_ids:
            raise ValueError('DECISION_SUBJECT_BINDING_MISMATCH')
        projected_ids = [a.assertion_id for a in legacy.evidence_projection]
        if len(projected_ids) != len(set(projected_ids)) or set(projected_ids) != set(legacy.selected_assertion_ids):
            raise ValueError('ASSERTION_SELECTION_MISMATCH')
        if self.context.predecessor != self.proposed_context.pin() or self.context.admission_decision != self.admission_decision.pin():
            raise ValueError('ADMISSION_REPLAY_BINDING_MISMATCH')
        if self.admission_decision.subject != self.proposed_context.pin() or self.admission_decision.context != self.proposed_context.pin():
            raise ValueError('ADMISSION_SUBJECT_MISMATCH')
        if self.admission_decision.ontology != self.ontology.pin():
            raise ValueError('ADMISSION_ONTOLOGY_MISMATCH')
        scope = (legacy.organization_id, legacy.workspace_id)
        if any((item.organization_id, item.workspace_id) != scope
               for item in (self.context, self.working_context, self.ontology, self.proposed_context, self.admission_decision, *self.evidence, *self.unselected_evidence)):
            raise ValueError('FOREIGN_SCOPE')
        if self.context.state != 'ADMITTED' or self.context.ontology != self.ontology.pin():
            raise ValueError('ADMITTED_ONTOLOGY_BINDING_REQUIRED')
        if self.working_context.context_graph != self.context.pin():
            raise ValueError('WORKING_CONTEXT_BINDING_MISMATCH')
        actual = {e.artifact_ref: e.pin() for e in self.evidence}
        expected = {e.artifact_ref: e for e in legacy.evidence}
        if actual != expected or len(actual) != len(self.evidence):
            raise ValueError('EVIDENCE_CLOSURE_MISMATCH')
        if set(self.working_context.selected_assertion_ids) != set(legacy.selected_assertion_ids):
            raise ValueError('ASSERTION_SELECTION_MISMATCH')
        all_evidence = {e.artifact_ref: e.pin() for e in (*self.evidence, *self.unselected_evidence)}
        if all_evidence != {p.artifact_ref: p for p in self.context.evidence} or len(all_evidence) != len(self.evidence) + len(self.unselected_evidence):
            raise ValueError('FULL_EVIDENCE_CLOSURE_MISMATCH')
        assertions = {a.assertion_id: a for a in self.context.assertions}
        nodes = {n.node_id: n for n in self.context.nodes}
        for item in legacy.evidence_projection:
            assertion = assertions.get(item.assertion_id)
            if assertion is None or (assertion.subject_ref, assertion.predicate, assertion.object_ref,
                    assertion.evidence) != (item.subject_ref, item.predicate, item.object_ref, item.evidence):
                raise ValueError('ASSERTION_BINDING_MISMATCH')
            if (nodes[item.subject_ref].label, nodes[item.object_ref].label) != (item.subject_label, item.object_label):
                raise ValueError('ASSERTION_LABEL_MISMATCH')
        source = {(a.evidence_ref.source_identity_ref, a.evidence_ref.selector_hash,
                   a.evidence_ref.evidence_role): a for a in self.source_region.evidence}
        if set(source) != {(e.source_ref, e.selector_hash, e.evidence_role) for e in self.evidence}:
            raise ValueError('SOURCE_EVIDENCE_CLOSURE_MISMATCH')
        for evidence in self.evidence:
            anchor = source[(evidence.source_ref, evidence.selector_hash, evidence.evidence_role)]
            ref = anchor.evidence_ref
            if (ref.organization_id, ref.workspace_id) != scope or ref.selector != evidence.selector:
                raise ValueError('SOURCE_SELECTOR_SCOPE_MISMATCH')
            if evidence.source_hash != anchor.locator.source_hash or tuple(evidence.provenance) != (self.source_region.source_binding.canonical_source_ir,):
                raise ValueError('SOURCE_PROVENANCE_BINDING_MISMATCH')
            if anchor.text_hash != 'sha256:' + sha256(ref.selector.exact_quote.encode()).hexdigest():
                raise ValueError('SOURCE_TEXT_HASH_MISMATCH')
        if {p.artifact_ref: p for p in self.working_context.evidence} != expected:
            raise ValueError('WORKING_CONTEXT_EVIDENCE_CLOSURE_MISMATCH')
        selected_evidence = {p.artifact_ref: p for item in legacy.evidence_projection for p in item.evidence}
        if selected_evidence != expected:
            raise ValueError('ASSERTION_EVIDENCE_CLOSURE_MISMATCH')
        requirements = {(p.predicate_id, r.dependency_id): r
                        for p in legacy.semantic_predicates for r in p.material_requirements}
        resolutions = {(r.predicate_id, r.dependency_id): r for r in legacy.material_dependency_resolutions}
        if requirements.keys() != resolutions.keys() or len(resolutions) != len(legacy.material_dependency_resolutions):
            raise ValueError('DEPENDENCY_REQUIREMENT_MISMATCH')
        for key, requirement in requirements.items():
            resolution = resolutions[key]
            matches = {e.artifact_ref: e.pin() for e in self.evidence
                       if e.evidence_role == requirement.required_evidence_role}
            satisfied = len(matches) >= requirement.min_count
            if (resolution.status, resolution.required_evidence_role, resolution.min_count) != (
                    'SATISFIED' if satisfied else 'MISSING', requirement.required_evidence_role, requirement.min_count):
                raise ValueError('DEPENDENCY_STATUS_MISMATCH')
            if {p.artifact_ref: p for p in resolution.matched_evidence} != (matches if satisfied else {}):
                raise ValueError('DEPENDENCY_EVIDENCE_CLOSURE_MISMATCH')
        from services.decision_local_projection_service import DecisionLocalProjectionService
        if legacy.world_mode != self.context.world.mode or legacy.world_time_basis != DecisionLocalProjectionService._world_time_basis(self.context):
            raise ValueError('WORLD_TIME_BASIS_MISMATCH')
        if self.policy_hash != DecisionMaterialityPolicyV1().policy_hash:
            raise ValueError('UNTRUSTED_MATERIALITY_POLICY')
        if self.model_input_hash != digest(self.model_input_payload()):
            raise ValueError('MODEL_VIEW_HASH_MISMATCH')
        if self.omission_manifest_hash != digest(self.omission_manifest()):
            raise ValueError('OMISSION_MANIFEST_HASH_MISMATCH')
        if self.projection_hash != digest(self.hash_payload()):
            raise ValueError('PROJECTION_HASH_MISMATCH')
        return self

    def ordered_evidence(self):
        anchors = {(a.evidence_ref.source_identity_ref, a.evidence_ref.selector_hash,
                    a.evidence_ref.evidence_role): a for a in self.source_region.evidence}
        return sorted(self.evidence, key=lambda e: (
            e.source_ref,
            tuple(anchors[(e.source_ref, e.selector_hash, e.evidence_role)].locator.ordinal_path),
            e.artifact_ref))

    def model_input_payload(self) -> dict:
        legacy = self.legacy
        ordered = self.ordered_evidence()
        handles = {e.artifact_ref: f'e{i}' for i, e in enumerate(ordered)}
        # One closure handle per assertion prevents repeated full pin expansion.
        assertions = []
        for i, a in enumerate(legacy.evidence_projection):
            assertions.append({'ref': f'a{i}', 'subject_ref': a.subject_ref,
                'subject_label': a.subject_label, 'predicate': a.predicate,
                'object_ref': a.object_ref, 'object_label': a.object_label,
                'evidence_set': f'closure:a{i}'})
        dependencies = []
        for dep in legacy.material_dependency_resolutions:
            row = dep.model_dump(mode='json')
            row['matched_evidence'] = [handles[p.artifact_ref] for p in dep.matched_evidence]
            dependencies.append(row)
        return {'decision_subject_ref': legacy.decision_subject_ref,
                'decision_contract_ref': legacy.decision_contract_ref,
                'decision_contract_hash': legacy.decision_contract_hash,
                'semantic_predicates': legacy.model_input_payload()['semantic_predicates'],
                'assertions': assertions,
                'source_excerpts': [{'ref': handles[e.artifact_ref], 'role': e.evidence_role,
                                     'text': e.selector.exact_quote} for e in ordered],
                'material_dependency_resolutions': dependencies,
                'deterministic_constraints': [p.model_dump(mode='json') for p in legacy.deterministic_constraints],
                'world_mode': legacy.world_mode, 'world_time_basis': legacy.world_time_basis}

    def omission_manifest(self) -> list[dict]:
        # All evidence text and all selected assertions remain visible. Only their
        # verbose provenance expansion is represented by compact local handles.
        rows = [{'artifact_ref': e.artifact_ref, 'artifact_hash': e.artifact_hash,
                 'reason': 'DUPLICATE_PROVENANCE_EXPANSION', 'represented_by': f'e{i}'}
                for i, e in enumerate(self.ordered_evidence())]
        selected = set(self.legacy.selected_assertion_ids)
        rows.extend({'artifact_ref': a.assertion_id, 'artifact_hash': digest(a.model_dump(mode='json')),
                     'reason': 'NON_MATERIAL_TO_DECISION_CONTRACT'}
                    for a in sorted(self.context.assertions, key=lambda a: a.assertion_id)
                    if a.assertion_id not in selected)
        rows.extend({'artifact_ref': e.artifact_ref, 'artifact_hash': e.artifact_hash,
                     'reason': 'NON_MATERIAL_TO_DECISION_CONTRACT'}
                    for e in sorted(self.unselected_evidence, key=lambda e: e.artifact_ref))
        return rows

    def provenance_handles(self) -> dict:
        result = {f'e{i}': e.pin().model_dump(mode='json') for i, e in enumerate(self.ordered_evidence())}
        for i, a in enumerate(self.legacy.evidence_projection):
            result[f'a{i}'] = {'assertion_id': a.assertion_id, 'assertion_hash': digest(a.model_dump(mode='json'))}
            result[f'closure:a{i}'] = [p.model_dump(mode='json') for p in a.evidence]
        return result

    @property
    def projection_ref(self) -> str:
        return 'decision-local-projection:v2:' + self.projection_hash[7:]

    def hash_payload(self) -> dict:
        value = self.model_dump(mode='json')
        value.pop('projection_hash')
        return value

    @classmethod
    def create(cls, **basis):
        draft = cls.model_construct(**basis, policy_hash=DecisionMaterialityPolicyV1().policy_hash,
            model_input_hash='', omission_manifest_hash='', projection_hash='')
        draft = draft.model_copy(update={'model_input_hash': digest(draft.model_input_payload()),
            'omission_manifest_hash': digest(draft.omission_manifest())})
        return cls.model_validate({**draft.model_dump(), 'projection_hash': digest(draft.hash_payload())})
