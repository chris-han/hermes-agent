"""Documentary association over existing source, query and report owners.

Only the explicit command invokes retrieval. Read/export validates retained data.
"""
from hashlib import sha256
import json

from contracts.tender_review import (TenderAuthoredQueryV1, DocumentaryOccurrenceV1,
    DocumentaryLawV1, TenderDocumentaryCandidateV1)
from semantier.knowledge_feature_packs.evidence_selector_builder import build_evidence_selector, evidence_ref_for_block
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref


def read_authored_query(owner, ctx, pin):
    from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
    artifact = owner._artifacts.resolve(ctx=ctx, artifact_pin=pin)
    if artifact.media_type != 'text/plain' or artifact.byte_size > 262144:
        raise ValueError('AUTHORED_QUERY_ARTIFACT_REQUIRED')
    blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=owner._artifacts._max_bytes)
    with blobs.exact_get(byte_owner_ref=artifact.byte_owner_ref, expected_hash=artifact.content_hash,
                         expected_size=artifact.byte_size) as source:
        query = TenderAuthoredQueryV1.model_validate_json(source.read())
    if (query.request.organization_id, query.request.workspace_id) != (ctx.organization_id, ctx.workspace_id):
        raise ValueError('FOREIGN_DOCUMENTARY_QUERY')
    return query


def occurrence(identity, document, ir_pin, block, start, end, quote, *, location=None):
    if not isinstance(block.content, str) or not 0 <= start < end <= len(block.content) or block.content[start:end] != quote:
        raise ValueError('DOCUMENTARY_RANGE_MISMATCH')
    if location is not None and location.get('kind') == 'html_text_range':
        return DocumentaryOccurrenceV1(identity=identity, canonical_source_ir_pin=ir_pin,
            evidence=None, location=location, exact_quote=quote, quote_hash='sha256:'+sha256(quote.encode()).hexdigest())
    selector = build_evidence_selector(document, block).model_copy(update={
        'exact_quote': quote, 'normalized_quote': None,
        'prefix': block.content[max(0, start-48):start], 'suffix': block.content[end:end+48]})
    evidence = evidence_ref_for_block(document, block, source_identity=identity, created_at=identity.created_at, selector=selector)
    evidence = evidence.model_copy(update={'provenance': {**evidence.provenance, 'start_offset': start, 'end_offset': end}})
    if location is None:
        resolved = resolve_evidence_ref(evidence, document, artifact_ref=ir_pin.artifact_ref, selector_hash=evidence.selector_hash)
        if resolved.status not in ('exact', 'relocated_exact'):
            raise ValueError('DOCUMENTARY_EXACT_LOCATION_UNAVAILABLE')
        location = resolved.location
    return DocumentaryOccurrenceV1(identity=identity, canonical_source_ir_pin=ir_pin,
        evidence=evidence, location=location, exact_quote=quote, quote_hash='sha256:'+sha256(quote.encode()).hexdigest())


def prepare_candidate(owner, ctx, command, target):
    """Validate all material inputs before dispatch; no query execution here."""
    query = read_authored_query(owner, ctx, command.query_artifact_pin)
    _, prep, identity, document, _ = target
    items = owner._items(identity, document, prep)
    if not any(item['subject'] == command.subject.model_dump(mode='json') for item in items):
        raise ValueError('DOCUMENTARY_SUBJECT_MISMATCH')
    block = next(b for b in document.blocks if b.block_id == command.subject.source_block_ref)
    tender = occurrence(identity, document, prep.canonical_source_ir_pin, block,
        command.start_offset, command.end_offset, command.exact_quote)
    originals = {}
    from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
    blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=owner._artifacts._max_bytes)
    pins = {p.logical_source_id:p for p in query.request.sources}
    for original in query.originals:
        prepared = owner._preparation.resolve(ctx=ctx, artifact_pin=original.artifact_pin,
            preparation_pin=original.preparation_pin, purpose='DESIGN_SOURCE')
        source, canonical = owner._preparation.resolve_design_source(ctx=ctx,
            artifact_pin=original.artifact_pin, preparation_pin=original.preparation_pin)
        artifact = owner._artifacts.resolve(ctx=ctx, artifact_pin=original.artifact_pin)
        if (source.source_identity_ref, source.source_hash, prepared.canonical_source_ir_pin) != (original.source_identity_ref, original.source_hash, original.canonical_source_ir_pin):
            raise ValueError('DOCUMENTARY_ORIGINAL_IDENTITY_MISMATCH')
        if source.source_hash != 'sha256:'+pins[original.logical_source_id].raw_sha256:
            raise ValueError('DOCUMENTARY_ORIGINAL_HASH_MISMATCH')
        # Check actual original custody before a costly dispatch, not hashes alone.
        with blobs.exact_get(byte_owner_ref=artifact.byte_owner_ref, expected_hash=artifact.content_hash,
                             expected_size=artifact.byte_size) as stream:
            stream.read(1)
        originals[original.logical_source_id] = (original, prepared, source, canonical)
    return query, tender, originals


def attach_candidate(command, prepared, packet):
    from services.original_html_source import select_occurrence
    from services.source_text_query_service import verify_source_text_export
    verify_source_text_export(packet)
    if packet['output']['state'] not in ('COMPLETE_EMPTY', 'COMPLETE_NONEMPTY'):
        raise ValueError('DOCUMENTARY_RETRIEVAL_INCOMPLETE')
    query, tender, originals = prepared
    laws = []
    for article in packet['output']['articles']:
        original, prep, source, document = originals[article['source_path']]
        if source.media_type != 'text/html':
            raise ValueError('DOCUMENTARY_LAW_ORIGINAL_UNAVAILABLE')
        projection = document.assets[0]['packet']
        selected = select_occurrence(projection, article['text'])
        if selected['state'] != 'exact':
            raise ValueError('DOCUMENTARY_LAW_RANGE_NOT_UNIQUE')
        start, end = selected['start'], selected['end']
        location = dict(kind='html_text_range', startOffset=start, endOffset=end, exactText=article['text'],
            sourceHash=source.source_hash, projectionHash='sha256:'+projection['sha256'],
            quoteHash='sha256:'+selected['quote_sha256'], mappingVersion=projection['version'])
        # HTML preparation owns one full projection text block.
        blocks = [b for b in document.blocks if isinstance(b.content,str) and b.content == projection['text']]
        if len(blocks) != 1:
            raise ValueError('DOCUMENTARY_HTML_CANONICAL_MISMATCH')
        selected_source = occurrence(source, document, prep.canonical_source_ir_pin, blocks[0],
            start, end, article['text'], location=location)
        laws.append(DocumentaryLawV1(article=article['article'], logical_source_id=article['source_path'],
            edition=article['edition'], start=article['start'], end=article['end'], original=original, occurrence=selected_source))
    return TenderDocumentaryCandidateV1(schema_version='tender_documentary_candidate.v1',
        query_artifact_pin=command.query_artifact_pin, authored_query=query, tender=tender,
        retrieval=packet, laws=tuple(laws), applicability='UNKNOWN', review_state='UNREVIEWED',
        association='EXPLICIT_NOMINATION_NOT_APPLICABILITY')
