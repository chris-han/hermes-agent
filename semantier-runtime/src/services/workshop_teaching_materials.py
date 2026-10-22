"""Read launcher-installed, public teaching inputs with frozen producer provenance.

This is distribution of one reviewed compiled teaching example, not admission,
compilation in another tenant, or access to the originating private workspace.
"""
from hashlib import sha256
import json
from pathlib import Path
from contracts.expense_reimbursement import ExpenseTeachingPolicyV1
from contracts.evidence_location import SourceIdentity
from contracts.normative_schema import NormativeSchemaContractV1, NormativeSchemaProposalEvidenceV1
from contracts.normative_semantics import (NormativeSemanticCandidateSetV1,
    NormativeSemanticProposalEvidenceV1, NormativeSemanticCorrectionReviewV1)
from contracts.ontology_validation import CompiledRulePlanV1


_BUNDLE_MANIFEST_SHA256 = "c2479101ed4011f0df41bcfb932e887f4c7ef84890d545dc729b5d9db6732236"


def load_expense_teaching_bundle(ctx):
    if not ctx.authenticated:
        raise PermissionError('TEACHING_MATERIAL_AUTHENTICATION_REQUIRED')
    root = Path(ctx.hermes_home).resolve() / 'demo-materials'
    manifest_path = root / 'expense-policy-manifest.json'
    if not manifest_path.is_file():
        return None
    for name in ('expense-policy-manifest.json', 'expense-policy-prepared.json', '财务报销政策.pdf'):
        if not (root / name).resolve().is_relative_to(Path(ctx.hermes_home).resolve()):
            raise PermissionError('TEACHING_MATERIAL_PATH_ESCAPE')
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != _BUNDLE_MANIFEST_SHA256:
        raise ValueError('TEACHING_BUNDLE_MANIFEST_HASH_MISMATCH')
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema_version') != 'workshop_expense_bundle.v1' or manifest.get('teaching_only') is not True:
        raise ValueError('TEACHING_BUNDLE_CONTRACT_MISMATCH')
    raw = (root / 'expense-policy-prepared.json').read_bytes()
    source = (root / '财务报销政策.pdf').read_bytes()
    if sha256(raw).hexdigest() != manifest['snapshot_sha256'] or sha256(source).hexdigest() != manifest['source_sha256']:
        raise ValueError('TEACHING_BUNDLE_HASH_MISMATCH')
    payload = json.loads(raw)
    models = dict(policy=ExpenseTeachingPolicyV1, schema=NormativeSchemaContractV1,
        candidate=NormativeSemanticCandidateSetV1, plan=CompiledRulePlanV1,
        source_identity=SourceIdentity, source_receipt=NormativeSchemaProposalEvidenceV1,
        semantic_receipt=NormativeSemanticProposalEvidenceV1, semantic_review=NormativeSemanticCorrectionReviewV1)
    prepared = {key: model.model_validate(payload[key]) for key, model in models.items()}
    policy, schema, candidate, plan = (prepared[key] for key in ('policy', 'schema', 'candidate', 'plan'))
    if (not all(item.teaching_candidate for item in (policy, schema, candidate, plan))
        or policy.source_hash != 'sha256:' + manifest['source_sha256']
        or policy.schema_pin['artifact_hash'] != schema.schema_hash
        or policy.candidate_pin['artifact_hash'] != candidate.candidate_set_hash
        or policy.plan_pin['artifact_hash'] != plan.plan_hash):
        raise ValueError('TEACHING_BUNDLE_LINEAGE_MISMATCH')
    return dict(**prepared, canonical_source=payload['canonical_source'], source_bytes=source)
