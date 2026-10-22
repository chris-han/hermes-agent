"""Scoped DEVELOPMENT caller and durable source-only QuerySpec composition."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import json
import time
from uuid import uuid4

from contracts.source_text_query import SourceTextQueryRequest
from services import source_text_query_engine as engine
from services.legal_evidence_recovery_service import LegalEvidenceRecoveryService
from contracts.legal_evidence_recovery import LegalEvidenceRequestV1
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import exact_snapshot_text
from storage.runtime_semantic_hypothesis_artifact_store import RuntimeSemanticHypothesisArtifactStore

COMPOSITION = 'source-text-query-runtime/1'

def now():
    return datetime.now(timezone.utc).isoformat()


def _verify_export(packet):
    """Offline: hashes, scope, closure, exact sets and source membership only."""
    payload = dict(packet); claimed = payload.pop('sha256')
    if engine.digest(payload) != claimed:
        raise ValueError('export hash mismatch')
    if packet.get('complete_dependency_closure') != 'UNESTABLISHED' or packet.get('review_state') != 'UNREVIEWED' or packet.get('legal_applicability') != 'NOT_ASSESSED':
        raise ValueError('non-authority disposition mismatch')
    req = SourceTextQueryRequest.model_validate(packet['start']['request'])
    start, end = packet['start'], packet['end']
    if (end['attempt_id'], end['start_sha256']) != (start['attempt_id'], engine.digest(start)):
        raise ValueError('missing or mismatched END')
    if end['output_sha256'] != engine.digest(packet['output']):
        raise ValueError('output hash mismatch')
    if start['composition'] != COMPOSITION or start['manifest_sha256'] != req.manifest_sha256:
        raise ValueError('composition mismatch')
    for value in (start['produced_at'], end['produced_at']):
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError('aware UTC required')
    if packet['output']['state'].startswith('COMPLETE'):
        if not engine.verify_certificate(req.spec, req.spec_sha256, packet['graph'], packet['documents'], packet['output']):
            raise ValueError('invalid certificate')
    elif packet['output'].get('certificate') is not None:
        raise ValueError('incomplete output has certificate')
    if not packet['sources'] and packet['output']['state'] == 'EXECUTION_FAILURE':
        return {'state': 'EXECUTION_FAILURE', 'output_sha256': end['output_sha256'], 'attempt_id': start['attempt_id'], 'valid': True}
    if set(packet['sources']) != {p.logical_source_id for p in req.sources} or set(packet['documents']) != set(packet['sources']):
        raise ValueError('incomplete source manifest')
    for pin in req.sources:
        source = packet['sources'][pin.logical_source_id]
        from contracts.legal_evidence_recovery import LegalEvidenceSourceV1
        LegalEvidenceSourceV1.model_validate({'normalized_document_ref':pin.normalized_document_ref,
            'identity':source['identity'],'canonical_document':source['canonical'],'regions':source['regions']})
        identity = source['identity']
        if (identity['organizationId'], identity['workspaceId'], identity['sourceIdentityRef'], identity['sourceHash'], identity['sourceVersion']) != (req.organization_id, req.workspace_id, pin.source_identity_ref, 'sha256:'+pin.raw_sha256, pin.edition):
            raise ValueError('source identity mismatch')
        raw = base64.b64decode(source['raw_base64'], validate=True)
        if engine.sha(raw) != pin.raw_sha256:
            raise ValueError('raw hash mismatch')
        from contracts.canonical_source_ir import CanonicalSourceDocument
        from eos.artifact_hashing import content_hash
        canonical = source['canonical']
        if content_hash(canonical) != pin.canonical_sha256:
            raise ValueError('canonical hash mismatch')
        # Read retained IR and the explicit identity mapping; never reparse raw.
        text, mapping = exact_snapshot_text(CanonicalSourceDocument.model_validate(canonical))
        document = packet['documents'][pin.logical_source_id]
        if text != document['text'] or engine.sha(text.encode()) != pin.text_sha256:
            raise ValueError('source text mismatch')
        if source['mapping'] != mapping:
            raise ValueError('source mapping mismatch')
    return {'state': packet['output']['state'], 'output_sha256': end['output_sha256'],
            'attempt_id': start['attempt_id'], 'valid': True}


def verify_source_text_export(packet):
    try:
        return _verify_export(packet)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('incomplete exported closure') from exc


class SourceTextQueryService:
    def __init__(self, *, runtime_root, db_path, authority_resolver):
        self._root = Path(runtime_root).resolve()
        self._db = Path(db_path)
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError('production authority resolver required')
        self._authority = authority_resolver

    def _scope(self, actor_context, boundary):
        self._authority.require_current_actor(actor_context=actor_context)
        workspace = (self._root/'workspaces'/actor_context.workspace_id).resolve()
        if boundary.user_id != actor_context.actor_id or boundary.workspace_id != actor_context.workspace_id or Path(boundary.paths.hermes_home).resolve() != workspace:
            raise ValueError('execution boundary scope mismatch')
        if not boundary.policy.require_boundary or boundary.policy.provider_fallback_enabled:
            raise ValueError('governed execution boundary required')
        from gateway.execution_boundary import DiskIOBoundaryGuard
        DiskIOBoundaryGuard(boundary).resolve_read_path(workspace)
        return RuntimeSemanticHypothesisArtifactStore(self._root,
            organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)

    def _sources(self, request, boundary):
        # Reuse the current recovery owner, with no semantic observations.
        from contracts.legal_evidence_recovery import LegalEvidenceQueryV1
        from contracts.ontology_building import ArtifactPinV1
        # Source recovery query fields describe access, never answer selectors.
        query = LegalEvidenceQueryV1(query_id=0, title='DEVELOPMENT pinned source text', clause_content='source snapshot')
        access = LegalEvidenceRequestV1(organization_id=request.organization_id,
            workspace_id=request.workspace_id, run_id=request.request_id, classification='DEVELOPMENT',
            question='DEVELOPMENT pinned source text', query=query,
            corpus=ArtifactPinV1(artifact_kind='source_only_corpus',artifact_ref='source-text:'+request.manifest_sha256,artifact_hash='sha256:'+request.manifest_sha256),
            configuration=ArtifactPinV1(artifact_kind='evidence_recovery_configuration',artifact_ref=COMPOSITION,artifact_hash='sha256:'+engine.digest(COMPOSITION)),
            normalized_document_refs=tuple(p.normalized_document_ref for p in request.sources))
        recovered = LegalEvidenceRecoveryService(runtime_root=self._root,db_path=self._db).recover(access, observations=())
        docs, sources = {}, {}
        from eos.artifact_hashing import content_hash
        for pin, source in zip(request.sources, recovered.sources, strict=True):
            identity=source.identity
            canonical=source.canonical_document.model_dump(mode='json',by_alias=True)
            text,mapping=exact_snapshot_text(source.canonical_document)
            if (identity.source_identity_ref,identity.source_hash,identity.source_version,content_hash(canonical),engine.sha(text.encode())) != (pin.source_identity_ref,'sha256:'+pin.raw_sha256,pin.edition,pin.canonical_sha256,pin.text_sha256):
                raise ValueError('source/raw/edition/canonical pin mismatch')
            original=(Path(boundary.paths.hermes_home)/identity.artifact_ref).resolve()
            if not any(original.is_relative_to(Path(p).resolve()) for p in boundary.policy.allowed_read_roots):
                raise ValueError('source outside execution boundary')
            from gateway.execution_boundary import DiskIOBoundaryGuard
            raw=DiskIOBoundaryGuard(boundary).resolve_read_path(original).read_bytes()
            docs[pin.logical_source_id]={'source_path':pin.logical_source_id,'text':text,
                'source_text_sha256':pin.text_sha256,'raw_source_sha256':pin.raw_sha256,'declared_snapshot_edition':pin.edition}
            sources[pin.logical_source_id]={'identity':identity.model_dump(mode='json',by_alias=True),
                'canonical':canonical,'mapping':mapping,'raw_base64':base64.b64encode(raw).decode(),
                'regions':[r.model_dump(mode='json',by_alias=True) for r in source.regions]}
        return docs,sources

    def invoke(self, request, *, actor_context, boundary):
        request=SourceTextQueryRequest.model_validate(request.model_dump(mode='json'))
        store=self._scope(actor_context,boundary)
        if (request.organization_id,request.workspace_id)!=(actor_context.organization_id,actor_context.workspace_id):
            raise ValueError('request outside actor scope')
        with store.source_text_attempt_lock(request.request_id, boundary=boundary):
            previous=store.read_source_text_attempt(request.request_id, boundary=boundary)
            if previous:
                if previous['start']['request']!=request.model_dump(mode='json'):
                    raise ValueError('REQUEST_IDENTITY_CONFLICT')
                if 'end' not in previous:
                    raise ValueError('ABANDONED_ATTEMPT_MISSING_END')
                verify_source_text_export(previous)
                return previous
            started=time.monotonic()
            start={'attempt_id':uuid4().hex,'produced_at':now(),'composition':COMPOSITION,
                   'actor_id':actor_context.actor_id,'session_id':boundary.session_id,'request':request.model_dump(mode='json'),
                   'native_pin':'701aa6a755899b8b63ce38ee971827b3e2c092e7','pyoxigraph':'0.5.11',
                   'implementation':{p:engine.fsha(Path(__file__).resolve().parents[1]/p) for p in ('services/source_text_query_engine.py','services/source_text_query_service.py','contracts/source_text_query.py','storage/runtime_semantic_hypothesis_artifact_store.py','semantier/knowledge_feature_packs/canonical_source_ir_adapter.py')},
                   'manifest_sha256':request.manifest_sha256}
            store.write_source_text_stage(request.request_id,'start',start,boundary=boundary)
            docs,sources,graph={},{},{}
            try:
                docs,sources=self._sources(request,boundary)
                graph=engine.graph_builder(docs)
                output=engine.execute(request.spec,request.spec_sha256,graph,docs)
            except Exception as exc:
                output={'state':'EXECUTION_FAILURE','execution_state':'FAILED','certificate':None,
                        'reason':type(exc).__name__+': '+str(exc),
                        'cost':{'backend_calls':0,'charged_records':0,'elapsed_seconds':0,'returned_query_bytes':0}}
            end={'attempt_id':start['attempt_id'],'start_sha256':engine.digest(start),
                 'produced_at':now(),'elapsed_seconds':time.monotonic()-started,
                 'output_sha256':engine.digest(output)}
            packet={'start':start,'end':end,'documents':docs,'sources':sources,'graph':graph,'output':output,
                    'complete_dependency_closure':'UNESTABLISHED','review_state':'UNREVIEWED','legal_applicability':'NOT_ASSESSED'}
            packet['sha256']=engine.digest(packet)
            store.write_source_text_stage(request.request_id,'end',packet,boundary=boundary)
            return packet

    def export(self, request_id, *, actor_context, boundary):
        store=self._scope(actor_context,boundary)
        packet=store.read_source_text_attempt(request_id,boundary=boundary)
        if packet is None or 'end' not in packet:
            raise ValueError('MISSING_END')
        verify_source_text_export(packet)
        return packet


def invoke_development_source_text_query(service, request, *, actor_context, boundary):
    """In-process command adapter; caller context is revalidated against auth DB."""
    return service.invoke(request,actor_context=actor_context,boundary=boundary)
