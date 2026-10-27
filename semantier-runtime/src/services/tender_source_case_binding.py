"""Owner-local descriptive binding helpers; no legal or source-admission authority."""
from pathlib import Path
from hashlib import sha256
import json
from contracts.tender_legal_compliance import TenderSourceCaseBindingV1,TenderSourceSubclaimV1
from contracts.legal_evidence_recovery import LegalEvidenceResultV1
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_pin
from storage.runtime_semantic_hypothesis_artifact_store import RuntimeSemanticHypothesisArtifactStore

LEGAL_GAPS=('LEGAL_EDITION_NOT_ESTABLISHED','LEGAL_EFFECTIVITY_NOT_ESTABLISHED','JURISDICTION_NOT_ESTABLISHED')

def frozen(binding,recovered):
    """Pure verification of pinned native snapshots; never query now, parse or write."""
    binding=TenderSourceCaseBindingV1.model_validate(binding.model_dump(mode='json'))
    recovered=LegalEvidenceResultV1.model_validate(recovered.model_dump(mode='json',by_alias=True))
    req=recovered.request
    if (binding.organization_id,binding.workspace_id,binding.recovery_run_id,binding.recovery_hash)!=(req.organization_id,req.workspace_id,req.run_id,recovered.result_hash):
        raise ValueError('SOURCE_CASE_RECOVERY_PIN_MISMATCH')
    sources={s.identity.source_identity_ref:s for s in recovered.sources}
    if set(sources)!={d.identity.source_identity_ref for d in binding.documents}:
        raise ValueError('SOURCE_CASE_DOCUMENT_POPULATION_MISMATCH')
    for d in binding.documents:
        source=sources[d.identity.source_identity_ref]
        if (d.identity,d.canonical_source_ir,d.normalized_document_ref)!=(source.identity,source.regions[0].source_binding.canonical_source_ir,source.normalized_document_ref):
            raise ValueError('SOURCE_CASE_VERSION_PIN_MISMATCH')
        text=''.join(str(b.content) for b in source.canonical_document.blocks)
        if d.role in ('ORIGINAL','CORRECTION','REPLACEMENT') and binding.case_id not in text:
            raise ValueError('SOURCE_CASE_ID_MISMATCH')
        if d.role in ('ORIGINAL','REPLACEMENT') and f'采购包{binding.package_id}' not in ''.join(text.split()):
            raise ValueError('SOURCE_CASE_PACKAGE_MISMATCH')
    for e in (*binding.precedence_evidence,*(e for c in binding.subclaims for e in c.evidence)):
        source=sources[e.source_identity_ref];evidence_ref_pin(e)
        from semantier.knowledge_feature_packs.canonical_source_ir_adapter import resolve_unique_mapping
        block_id=e.provenance.get('source_block_id')
        blocks=[b for b in source.canonical_document.blocks if b.block_id==block_id]
        if len(blocks)!=1:raise ValueError('SOURCE_CASE_SELECTOR_UNRESOLVED:missing_block')
        block=blocks[0];quote=e.selector.exact_quote
        if not isinstance(block.content,str) or not quote or block.content.count(quote)!=1:
            raise ValueError('SOURCE_CASE_SELECTOR_UNRESOLVED:missing_or_ambiguous_quote')
        locator=resolve_unique_mapping(source.canonical_document,block.block_id)
        if (e.provenance.get('source_locator')!=locator.model_dump(mode='json',by_alias=True)
            or e.provenance.get('source_hash')!=source.identity.source_hash):
            raise ValueError('SOURCE_CASE_SELECTOR_UNRESOLVED:locator_mismatch')
    corrections=set(binding.correction_refs)
    if corrections != {e.source_identity_ref for e in binding.precedence_evidence}:
        raise ValueError('SOURCE_CASE_PRECEDENCE_POPULATION_MISMATCH')
    return binding


def _workspace(owner,ws):
    root=owner.workspace_root.resolve()
    if root.name!=ws or root.parent.name!='workspaces':
        raise ValueError('SOURCE_CASE_WORKSPACE_REQUIRED')
    return root


def verify_live(owner,binding,*,actor_context,finding=None):
    owner.authority.require_current_actor(actor_context=actor_context)
    if (binding.organization_id,binding.workspace_id)!=(actor_context.organization_id,actor_context.workspace_id):
        raise ValueError('SOURCE_CASE_SCOPE_MISMATCH')
    workspace=_workspace(owner,binding.workspace_id)
    artifacts=RuntimeSemanticHypothesisArtifactStore(workspace,organization_id=binding.organization_id,workspace_id=binding.workspace_id)
    recovered=artifacts.get_legal_evidence_result(run_id=binding.recovery_run_id,result_hash=binding.recovery_hash)
    frozen(binding,recovered)
    from services.legal_evidence_recovery_service import LegalEvidenceRecoveryService
    current=LegalEvidenceRecoveryService(runtime_root=workspace.parent.parent,db_path=owner.db_path).recover(recovered.request,observations=recovered.observations,produced_at=recovered.produced_at)
    if current!=recovered:
        raise ValueError('SOURCE_CASE_LIVE_RECOVERY_MISMATCH')
    # JSON API responses remain exact governed artifacts; HTML is a declared /data/content derivative.
    from contracts.ontology_workbench_context import WorkspaceArtifactV1
    for d in binding.documents:
        if d.derivation_input is None:continue
        pin=d.derivation_input
        raw=owner.store.get(organization_id=binding.organization_id,workspace_id=binding.workspace_id,artifact_ref=pin.artifact_ref,contract_type=WorkspaceArtifactV1)
        if raw is None or raw.artifact_hash!=pin.artifact_hash or pin.artifact_kind!='workspace_artifact':
            raise ValueError('SOURCE_CASE_DERIVATION_PIN_MISMATCH')
        def read(ref):
            path=(workspace/ref).resolve()
            if Path(ref).is_absolute() or not path.is_relative_to(workspace):raise ValueError('SOURCE_CASE_DERIVATION_SCOPE')
            return path.read_bytes()
        original=read(raw.byte_owner_ref)
        if 'sha256:'+sha256(original).hexdigest()!=raw.content_hash or json.loads(original)['data']['content'].encode()!=read(d.identity.artifact_ref):
            raise ValueError('SOURCE_CASE_DERIVATION_MISMATCH')
    if finding is not None:
        matches=[d for d in binding.documents if d.identity.source_identity_ref==finding.source_identity_ref and d.identity.source_hash==finding.source_hash]
        if len(matches)!=1:raise ValueError('SOURCE_CASE_FINDING_MISMATCH')
    return binding


def construct(owner,*,actor_context,recovered,documents,subclaims,missing_premises,**expression):
    owner.authority.require_current_actor(actor_context=actor_context)
    recovered=LegalEvidenceResultV1.model_validate(recovered.model_dump(mode='json',by_alias=True))
    if (recovered.request.organization_id,recovered.request.workspace_id)!=(actor_context.organization_id,actor_context.workspace_id):
        raise ValueError('SOURCE_CASE_SCOPE_MISMATCH')
    claims=tuple(TenderSourceSubclaimV1(subclaim_id=key,evidence=tuple(evidence),source_traceability='AVAILABLE' if evidence else 'UNKNOWN',missing_premises=tuple(missing_premises[key])) for key,evidence in sorted(subclaims.items()))
    binding=TenderSourceCaseBindingV1.create(organization_id=actor_context.organization_id,workspace_id=actor_context.workspace_id,prepared_by=actor_context.actor_id,recovery_run_id=recovered.request.run_id,recovery_hash=recovered.result_hash,documents=documents,subclaims=claims,**expression)
    frozen(binding,recovered)
    # Existing source/result owners only. Validate current native sources before retaining the source snapshot.
    workspace=_workspace(owner,actor_context.workspace_id)
    from services.legal_evidence_recovery_service import LegalEvidenceRecoveryService
    current=LegalEvidenceRecoveryService(runtime_root=workspace.parent.parent,db_path=owner.db_path).recover(recovered.request,observations=recovered.observations,produced_at=recovered.produced_at)
    if current!=recovered:raise ValueError('SOURCE_CASE_LIVE_RECOVERY_MISMATCH')
    RuntimeSemanticHypothesisArtifactStore(workspace,organization_id=actor_context.organization_id,workspace_id=actor_context.workspace_id).put_legal_evidence_result(recovered)
    return verify_live(owner,binding,actor_context=actor_context)
