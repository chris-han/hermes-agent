"""Resolve the exact reviewed teaching policy through native normative owners."""
from hashlib import sha256
from pathlib import Path
import json
from eos.artifact_hashing import content_hash

from contracts.expense_reimbursement import ExpenseTeachingPolicyV1
from contracts.normative_schema import NormativeSchemaProposalEvidenceV1
from contracts.normative_semantics import NormativeSemanticProposalEvidenceV1, NormativeSemanticCorrectionReviewV1
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_validation import CompiledRulePlanV1
from services.normative_schema_production_service import NormativeSchemaProductionService
from services.normative_semantic_production_service import NormativeSemanticProductionService
from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler
from services.teaching_candidate_scope import require_teaching_scope
from storage.ontology_build_store import ontology_build_store_path


class ExpenseTeachingPolicyService:
    def __init__(self, *, ctx, store, source_registry):
        self.ctx, self.store, self.sources = ctx, store, source_registry
        self.scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise PermissionError('EXPENSE_POLICY_WORKSPACE_MISMATCH')

    def read(self, pin):
        require_teaching_scope(self.ctx, **self.scope)
        if pin.artifact_kind != 'expense_teaching_policy':
            raise ValueError('EXPENSE_POLICY_PIN_KIND_MISMATCH')
        policy = self.store.get(**self.scope, artifact_ref=pin.artifact_ref, contract_type=ExpenseTeachingPolicyV1)
        if policy is None or policy.policy_hash != pin.artifact_hash:
            raise ValueError('EXPENSE_POLICY_PIN_MISMATCH')
        schema, adequacy = NormativeSchemaProductionService.read_persisted(store=self.store, **self.scope,
            schema_pin=ArtifactPinV1.model_validate(policy.schema_pin))
        candidate = NormativeSemanticProductionService.read_persisted(store=self.store, **self.scope,
            candidate_pin=ArtifactPinV1.model_validate(policy.candidate_pin))
        plan = self.store.get(**self.scope, artifact_ref=policy.plan_pin['artifact_ref'], contract_type=CompiledRulePlanV1)
        if (plan is None or plan.plan_hash != policy.plan_pin['artifact_hash'] or
            not schema.teaching_candidate or not candidate.teaching_candidate or not plan.teaching_candidate or
            (candidate.normative_schema_ref, candidate.normative_schema_hash) != (schema.schema_ref, schema.schema_hash) or
            (candidate.source_identity_ref, candidate.source_hash) != (policy.source_identity_ref, policy.source_hash)):
            raise ValueError('EXPENSE_POLICY_LINEAGE_MISMATCH')
        expected = OntologyRulePlanCompiler(compiler_ref=plan.compiler_ref, compiler_hash=plan.compiler_hash).compile(
            candidate=candidate, validation_contract_ref=plan.validation_contract_ref,
            validation_contract_hash=plan.validation_contract_hash,
            competency_refs=plan.rules[0].competency_refs, compiled_at=plan.compiled_at, teaching_context=self.ctx)
        if expected != plan:
            raise ValueError('EXPENSE_POLICY_COMPILATION_MISMATCH')
        defined = {fact.fact_ref for fact in schema.fact_definitions or ()}
        if not set(policy.field_bindings).issubset(defined) or not set(policy.disposition_effects).issubset(defined):
            raise ValueError('EXPENSE_POLICY_FACT_BINDING_MISMATCH')
        _, identity = self.sources.resolve_source_identity_lineage(**self.scope, source_identity_ref=policy.source_identity_ref)
        source = (Path(self.ctx.workspace_root) / identity.artifact_ref).resolve()
        if not source.is_relative_to(Path(self.ctx.workspace_root).resolve()):
            raise PermissionError('EXPENSE_POLICY_SOURCE_SCOPE_MISMATCH')
        if identity.source_hash != policy.source_hash or 'sha256:' + sha256(source.read_bytes()).hexdigest() != policy.source_hash:
            raise ValueError('EXPENSE_POLICY_SOURCE_PIN_MISMATCH')
        receipt = self.store.get(**self.scope, artifact_ref=schema.proposal_evidence_ref, contract_type=NormativeSchemaProposalEvidenceV1)
        semantic_receipt = self.store.get(**self.scope, artifact_ref=candidate.proposal_evidence_ref, contract_type=NormativeSemanticProposalEvidenceV1)
        review = self.store.get(**self.scope, artifact_ref=candidate.source_review_ref, contract_type=NormativeSemanticCorrectionReviewV1)
        if (receipt is None or semantic_receipt is None or review is None
            or semantic_receipt.evidence_hash != candidate.proposal_evidence_hash
            or review.review_hash != candidate.source_review_hash):
            raise ValueError('EXPENSE_POLICY_PROVENANCE_MISSING')
        ir_path = (Path(self.ctx.workspace_root) / candidate.canonical_source_ir_ref).resolve()
        if not ir_path.is_relative_to(Path(self.ctx.workspace_root).resolve()):
            raise PermissionError('EXPENSE_POLICY_SOURCE_SCOPE_MISMATCH')
        ir = json.loads(ir_path.read_text())
        if 'sha256:' + content_hash(ir) != candidate.canonical_source_ir_hash:
            raise ValueError('EXPENSE_POLICY_CANONICAL_PIN_MISMATCH')
        return dict(policy=policy, schema=schema, candidate=candidate, plan=plan,
                    source_identity=identity, source_bytes=source.read_bytes(), source_receipt=receipt,
                    semantic_receipt=semantic_receipt, semantic_review=review, canonical_source=ir)

    def current(self):
        pins = self.store.list_artifact_pins(**self.scope, schema_version='expense_teaching_policy.v1')
        policies = [self.store.get(**self.scope, artifact_ref=pin.artifact_ref, contract_type=ExpenseTeachingPolicyV1) for pin in pins]
        if not policies:
            from services.workshop_teaching_materials import load_expense_teaching_bundle
            return load_expense_teaching_bundle(self.ctx)
        policy = max(policies, key=lambda item: (item.created_at, item.policy_hash))
        return self.read(ArtifactPinV1(artifact_kind='expense_teaching_policy', artifact_ref=policy.policy_ref, artifact_hash=policy.policy_hash))


from contextlib import contextmanager


@contextmanager
def expense_policy_owner(ctx, bound):
    """Open native policy owners only for the authenticated active workspace."""
    from storage.ontology_build_store import OntologyBuildStore
    from storage.knowledge_builder_store import KnowledgeBuilderStore
    if ctx is None or not ctx.authenticated:
        raise PermissionError('EXPENSE_AUTHENTICATION_REQUIRED')
    if Path(ctx.hermes_home).resolve() != bound.workspace_home.resolve():
        raise PermissionError('EXPENSE_POLICY_WORKSPACE_MISMATCH')
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    sources = KnowledgeBuilderStore(workspace_root=ctx.workspace_root)
    try:
        yield ExpenseTeachingPolicyService(ctx=ctx, store=store, source_registry=sources)
    finally:
        store.close()
        sources.close()
