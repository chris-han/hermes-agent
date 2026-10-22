"""Versioned bounded-view adapter over the real runtime-bound Tender compiler."""
from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.semantic_kernel import ContextGraph, DecisionRecord, Evidence, Ontology
from services.tender_decision_local_projection_service import TenderDecisionLocalProjectionService
from storage.ontology_build_store import OntologyBuildStore


class BoundedDecisionProjectionService:
    def __init__(self, *, build_store: OntologyBuildStore,
                 tender_compiler: TenderDecisionLocalProjectionService):
        self.store = build_store
        self.tender_compiler = tender_compiler

    def compile_runtime_bound(self, **inputs) -> DecisionLocalProjectionV2:
        legacy = self.tender_compiler.compile_runtime_bound(**inputs)
        wc = inputs['working_context']

        def resolve(pin, contract):
            value = self.store.get(organization_id=wc.organization_id,
                workspace_id=wc.workspace_id, artifact_ref=pin.artifact_ref, contract_type=contract)
            if value is None or value.pin() != pin:
                raise ValueError('UNRESOLVABLE_REPLAY_PIN')
            return value

        context = resolve(legacy.context_graph, ContextGraph)
        constraints, semantic_type, family = self.tender_compiler._runtime_binding(
            runtime_context=inputs['runtime_context'], retrieval_snapshot=inputs['retrieval_snapshot'],
            canonical_pattern_ref=inputs['canonical_pattern_ref'])
        from contracts.decision_local_projection import MaterialDependencyRequirementV1
        contract = self.tender_compiler.decision_contract(issue_type=semantic_type, family_id=family,
            deterministic_constraints=constraints, material_requirements=tuple(
                MaterialDependencyRequirementV1(dependency_id=d.dependency_id,
                    required_evidence_role=d.required_evidence_role, min_count=d.min_count)
                for d in inputs['source_region'].material_dependencies))
        rules = tuple(dict(self.tender_compiler._governed_rule_resolver(ref))
                      for ref in inputs['retrieval_snapshot'].ordered_governed_rule_version_refs)
        return DecisionLocalProjectionV2.create(legacy=legacy,
            context=context, working_context=wc,
            decision_contract=contract, runtime_context=inputs['runtime_context'],
            retrieval_snapshot=inputs['retrieval_snapshot'],
            canonical_pattern_ref=inputs['canonical_pattern_ref'], governed_rules=rules,
            proposed_context=resolve(context.predecessor, ContextGraph),
            admission_decision=resolve(context.admission_decision, DecisionRecord),
            unselected_evidence=tuple(resolve(p, Evidence) for p in context.evidence if p not in legacy.evidence),
            ontology=resolve(legacy.ontology, Ontology),
            evidence=tuple(resolve(p, Evidence) for p in legacy.evidence),
            source_region=inputs['source_region'])
