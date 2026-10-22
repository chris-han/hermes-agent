"""Reviewed law-provision discovery with exact offline source reconstruction.

The authenticated caller owns tenant-admin authorization. This core owner never
infers authority from source text; review events attest only their explicit scope.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re

from contracts.canonical_source_ir import CanonicalSourceDocument, content_hash as block_hash
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.knowledge_feature_pack import HumanDecision, HumanLabel, LearningEventEnvelope, LearningEventType
from contracts.law_provision_witness import (LawProvisionSpanV1, LawProvisionWitnessV1,
    LawProvisionContrastV1, LawProvisionWitnessManifestV1)
from contracts.source_semantic_witness import SemanticExpectation, WitnessSourceBinding
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
from semantier.knowledge_feature_packs.falkordb_backend import InMemoryGraphBackend
from storage.atomic_json_artifact import publish_immutable_json
from storage.human_grounding_store import HumanGroundingStore
from services.knowledge_relation_taxonomy_service import KnowledgeRelationTaxonomyService

_ARTICLE = re.compile(r'^(第[一二三四五六七八九十百]+条)(?:\s|$)')
_CHAPTER = re.compile(r'^第[一二三四五六七八九十百]+章(?:\s|$)')
_FOOTER = re.compile(r'^-\s*\d+\s*-$')
_HASH = re.compile(r'^sha256:[0-9a-f]{64}$')


def _text_hash(text):
    return 'sha256:' + sha256(text.encode('utf-8')).hexdigest()


class LawProvisionWitnessService:
    def __init__(self, workspace_root: str | Path, *, source_identity: SourceIdentity,
                 document: CanonicalSourceDocument, graph_intent: GraphIntentBrief,
                 profile_id: str, profile_hash: str):
        self.root = Path(workspace_root).resolve()
        self.source = source_identity
        self.document = document
        self.intent = graph_intent
        self.profile_id = profile_id
        self.profile_hash = profile_hash
        self.taxonomy = KnowledgeRelationTaxonomyService().load_canonical()
        self.grounding = HumanGroundingStore(self.root / 'eos.db', graph_backend=InMemoryGraphBackend())
        self._validate_source()

    def _validate_source(self):
        binding = self.intent.source_binding
        if not self.intent.compiler_eligible:
            raise ValueError('accepted GraphIntent review required')
        if (binding.organization_id != self.source.organization_id
            or binding.workspace_id != self.source.workspace_id
            or binding.source_identity_ref != self.source.source_identity_ref
            or binding.source_hash != self.source.source_hash
            or self.document.source_identity_ref != self.source.source_identity_ref):
            raise ValueError('source/tenant binding mismatch')
        if block_hash(self.document.model_dump(mode='json', by_alias=True)) != binding.canonical_source_ir_hash:
            raise ValueError('CanonicalSourceIR hash mismatch')
        if len({b.block_id for b in self.document.blocks}) != len(self.document.blocks):
            raise ValueError('duplicate source block identity')
        if [b.ordinal for b in self.document.blocks] != sorted({b.ordinal for b in self.document.blocks}):
            raise ValueError('source block order is ambiguous')
        for block in self.document.blocks:
            if not isinstance(block.content, str) or block.content_hash != block_hash(block.content):
                raise ValueError('source block content/hash mismatch')
        if not self.profile_id.isascii() or not _HASH.fullmatch(self.profile_hash):
            raise ValueError('profile pin invalid')

    def _path(self, kind, digest):
        if not _HASH.fullmatch(digest):
            raise ValueError('invalid provision artifact hash')
        path = (self.root / 'artifacts/law-provision-witness/v1' / kind / (digest[7:] + '.json')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('provision artifact escapes governed workspace')
        return path

    def _persist(self, kind, artifact):
        publish_immutable_json(self._path(kind, artifact.content_hash), artifact.model_dump(mode='json'),
                               conflict_error='LAW_PROVISION_IMMUTABLE_CONFLICT')
        return artifact

    def _load(self, kind, digest):
        cls = {'spans': LawProvisionSpanV1, 'witnesses': LawProvisionWitnessV1,
               'contrasts': LawProvisionContrastV1, 'manifests': LawProvisionWitnessManifestV1}[kind]
        path = self._path(kind, digest)
        if not path.is_file():
            raise ValueError('provision artifact not found')
        value = cls.model_validate_json(path.read_text(encoding='utf-8'))
        if value.content_hash != digest:
            raise ValueError('provision artifact pin mismatch')
        return value

    def _article_members(self, blocks, start, end):
        """Collapse native overlapping anchors only after lossless reconstruction.

        The full article is the evidence member. Its paragraph/item projections
        remain in the pinned IR and must reconstruct that parent without loss.
        Flat paragraph-based documents retain their original membership policy.
        """
        root = blocks[start]
        if root.structure.get('sectionType') != 'article':
            return blocks[start:end]
        stable = root.structure.get('stableLocator', '')
        if not re.fullmatch(r'article_\d+', stable):
            raise ValueError('native article locator missing')
        paragraphs, item_locators = [], set()
        for child in blocks[start + 1:end]:
            locator = child.structure.get('stableLocator', '')
            if child.block_type == 'legal_paragraph':
                expected = f'{stable}.paragraph_{len(paragraphs) + 1:03d}'
                if child.structure.get('sectionType') != 'paragraph' or locator != expected:
                    raise ValueError('native article paragraph membership/order mismatch')
                paragraphs.append(child.content)
            elif child.block_type == 'legal_item' and paragraphs:
                expected = f'{stable}.paragraph_{len(paragraphs):03d}.item_001'
                if (child.structure.get('sectionType') != 'item' or locator != expected
                    or locator in item_locators or child.content != paragraphs[-1]):
                    raise ValueError('native article item projection mismatch')
                item_locators.add(locator)
            else:
                raise ValueError('native article has unsupported overlapping source block')
            mapped = [loc for mapping in self.document.source_map.mappings
                      if child.block_id in mapping.block_ids for loc in mapping.source_locators]
            if len(mapped) != 1 or mapped[0].section_id != locator:
                raise ValueError('native article child source locator mismatch')
        reconstructed = '\n'.join([root.content.splitlines()[0], *paragraphs])
        if not paragraphs or reconstructed != root.content:
            raise ValueError('native article parent/paragraph text mismatch')
        return [root]

    def _spans(self, article_label=None):
        self._validate_source()
        blocks = self.document.blocks
        starts = [i for i, block in enumerate(blocks) if _ARTICLE.match(block.content)]
        labels = [_ARTICLE.match(blocks[i].content).group(1) for i in starts]
        if not starts or len(labels) != len(set(labels)):
            raise ValueError('article boundaries absent or ambiguous')
        binding = WitnessSourceBinding(sourceIdentitySchemaVersion='semantier.source_identity.v1',
            sourceIdentityRef=self.source.source_identity_ref, sourceHash=self.source.source_hash,
            canonicalSourceIrRef=self.intent.source_binding.canonical_source_ir_ref,
            canonicalSourceIrHash=self.intent.source_binding.canonical_source_ir_hash)
        result = []
        for n, start in enumerate(starts):
            if article_label is not None and labels[n] != article_label:
                continue
            end = starts[n + 1] if n + 1 < len(starts) else len(blocks)
            members, texts = [], []
            seen_chapter = False
            for block in self._article_members(blocks, start, end):
                if block.block_type not in {'paragraph', 'legal_article'}:
                    raise ValueError('unsupported article block type')
                use = 'page_furniture' if _FOOTER.fullmatch(block.content) else (
                    'section_heading' if _CHAPTER.match(block.content) else 'provision_text')
                if use == 'section_heading':
                    seen_chapter = True
                if seen_chapter and use == 'provision_text':
                    raise ValueError('ambiguous chapter/article boundary')
                evidence = evidence_ref_for_block(self.document, block, source_identity=self.source,
                    evidence_role='law_provision:' + use, created_at=self.source.created_at)
                if evidence.provenance['source_hash'] != self.source.source_hash:
                    raise ValueError('source locator hash mismatch')
                evidence = evidence.model_copy(update={'provenance': {**evidence.provenance,
                    'canonical_source_ir_ref': binding.canonical_source_ir_ref,
                    'canonical_source_ir_hash': binding.canonical_source_ir_hash}})
                members.append(dict(block_id=block.block_id, ordinal=block.ordinal,
                    block_content_hash=block.content_hash, use=use,
                    evidence=evidence.model_dump(mode='json')))
                if use == 'provision_text':
                    texts.append(block.content)
            result.append(LawProvisionSpanV1.create(dict(schema_version='law_provision_span.v1',
                source_role='law_provisions', boundary_policy='cn_article_blocks.v1',
                organization_id=self.source.organization_id, workspace_id=self.source.workspace_id,
                source_binding=binding.model_dump(mode='json'),
                graph_intent_ref=self.intent.graph_intent_id, graph_intent_hash=self.intent.graph_intent_hash,
                profile_id=self.profile_id, profile_hash=self.profile_hash, article_label=labels[n],
                members=members, complete_text_hash=_text_hash('\n'.join(texts)))))
        return tuple(result)

    def propose_spans(self):
        return tuple(self._persist('spans', span) for span in self._spans())

    def reconstruct(self, span: LawProvisionSpanV1) -> str:
        # Rebuild from all pinned blocks: callers cannot shorten, merge or reorder an article.
        canonical = next((candidate for candidate in self._spans(span.article_label)
                          if candidate.article_label == span.article_label), None)
        if canonical != span:
            raise ValueError('provision boundary/membership/source pin mismatch')
        by_id = {b.block_id: b for b in self.document.blocks}
        return '\n'.join(by_id[m.block_id].content for m in span.members if m.use == 'provision_text')

    def _review(self, kind, digest, *, actor_ref, reviewer_kind, justification):
        if reviewer_kind not in {'human', 'delegated_ai'} or not justification.strip():
            raise ValueError('explicit reviewer kind and justification required')
        if not actor_ref or not actor_ref.isascii():
            raise ValueError('governed actor reference required')
        artifact = self._load(kind, digest)
        if kind == 'spans':
            span = artifact
            self.reconstruct(span)
            scope = ['provision_boundaries', 'law_provisions_source_role']
        else:
            witness = self._load('witnesses', artifact.positive_witness_hash) if kind == 'contrasts' else artifact
            span = self._load('spans', witness.span_hash)
            self._verify_span_review(witness)
            self.reconstruct(span)
            scope = ['hard_negative_distinction'] if kind == 'contrasts' else ['semantic_expectation']
        seed = block_hash(dict(kind=kind, digest=digest, actor=actor_ref,
                               reviewer_kind=reviewer_kind, justification=justification))
        event_id = 'event:law-provision-review:' + seed[7:]
        prior = self.grounding.get_learning_event(event_id)
        if prior is not None:
            return self._verify_review(kind, digest, event_id)
        history = self.grounding.list_events(digest)
        prior_hash = None
        if history:
            prior_hash = self.grounding.get_learning_event(history[-1]['event_id'])['event_hash']
        event = LearningEventEnvelope(event_id=event_id, case_id=digest, prior_event_hash=prior_hash,
            event_type=LearningEventType.HUMAN_ACCEPT, occurred_at=datetime.now(timezone.utc),
            source_document_hash=self.source.source_hash,
            source_anchor_refs=tuple(m.evidence.evidence_ref for m in span.members),
            graph_version=self.intent.graph_intent_id, rule_version='law-provision-review.v1',
            evaluation_corpus_version='law-provision-witness.v1', actor_ref=actor_ref,
            human_label=HumanLabel(decision=HumanDecision.ACCEPT, certainty='high',
                reason_code='law_provision_scoped_review', justification=justification),
            downstream_outcome=dict(artifact_kind=kind, artifact_hash=digest,
                organization_id=self.source.organization_id, workspace_id=self.source.workspace_id,
                graph_intent_hash=self.intent.graph_intent_hash,
                reviewer_kind=reviewer_kind, review_scope=scope,
                establishes_legal_effect=False, grants_gate1_admission=False))
        self.grounding.append_event(event)
        return self._verify_review(kind, digest, event_id)

    def _verify_review(self, kind, digest, event_ref):
        event = self.grounding.get_learning_event(event_ref)
        if event is None:
            raise ValueError('committed provision review missing')
        outcome = event.get('downstream_outcome') or {}
        scope = (['provision_boundaries', 'law_provisions_source_role'] if kind == 'spans' else
                 ['hard_negative_distinction'] if kind == 'contrasts' else ['semantic_expectation'])
        if (event['event_type'] != LearningEventType.HUMAN_ACCEPT.value
            or not event.get('actor_ref') or event['case_id'] != digest
            or event['source_document_hash'] != self.source.source_hash
            or event['graph_version'] != self.intent.graph_intent_id
            or event['rule_version'] != 'law-provision-review.v1'
            or outcome != dict(artifact_kind=kind, artifact_hash=digest,
                organization_id=self.source.organization_id, workspace_id=self.source.workspace_id,
                graph_intent_hash=self.intent.graph_intent_hash,
                reviewer_kind=outcome.get('reviewer_kind'), review_scope=scope,
                establishes_legal_effect=False, grants_gate1_admission=False)
            or outcome.get('reviewer_kind') not in {'human', 'delegated_ai'}):
            raise ValueError('provision review source/scope binding mismatch')
        artifact = self._load(kind, digest)
        witness = self._load('witnesses', artifact.positive_witness_hash) if kind == 'contrasts' else artifact
        span = artifact if kind == 'spans' else self._load('spans', witness.span_hash)
        label = event.get('human_label') or {}
        seed = block_hash(dict(kind=kind, digest=digest, actor=event['actor_ref'],
            reviewer_kind=outcome['reviewer_kind'], justification=label.get('justification')))
        if (event_ref != 'event:law-provision-review:' + seed[7:]
            or label.get('decision') != HumanDecision.ACCEPT.value
            or event['source_anchor_refs'] != [m.evidence.evidence_ref for m in span.members]):
            raise ValueError('provision review actor/anchor/decision mismatch')
        return event

    def review_span(self, span_hash, *, actor_ref, reviewer_kind, justification):
        return self._review('spans', span_hash, actor_ref=actor_ref,
                            reviewer_kind=reviewer_kind, justification=justification)

    def discover(self, span_hash, semantic_expectation: SemanticExpectation, *, span_review_ref=None):
        if span_review_ref is None:
            raise ValueError('explicit committed span review required')
        span = self._load('spans', span_hash)
        self.reconstruct(span)
        review = self._verify_review('spans', span_hash, span_review_ref)
        return self._persist('witnesses', LawProvisionWitnessV1.create(dict(
            schema_version='law_provision_witness.v1', source_role='law_provisions',
            span_hash=span_hash, span_review_ref=span_review_ref,
            span_review_hash=review['event_hash'],
            semantic_expectation=semantic_expectation.model_dump(mode='json'))))

    def _verify_span_review(self, witness):
        event = self._verify_review('spans', witness.span_hash, witness.span_review_ref)
        if event['event_hash'] != witness.span_review_hash:
            raise ValueError('span review hash mismatch')

    def review_witness(self, witness_hash, *, actor_ref, reviewer_kind, justification):
        return self._review('witnesses', witness_hash, actor_ref=actor_ref,
                            reviewer_kind=reviewer_kind, justification=justification)

    def replay(self, witness_hash, *, review_event_ref):
        witness = self._load('witnesses', witness_hash)
        self._verify_review('witnesses', witness_hash, review_event_ref)
        self._verify_span_review(witness)
        return witness, self.reconstruct(self._load('spans', witness.span_hash))

    def propose_contrast(self, positive_witness_hash, *, scenario, rejected_interpretation, expected_distinction):
        witness = self._load('witnesses', positive_witness_hash)
        self._verify_span_review(witness)
        self.reconstruct(self._load('spans', witness.span_hash))
        return self._persist('contrasts', LawProvisionContrastV1.create(dict(
            schema_version='law_provision_contrast.v1', construction_kind='review_only_counterexample',
            authority_promotion_allowed=False, positive_witness_hash=positive_witness_hash,
            scenario=scenario, rejected_interpretation=rejected_interpretation,
            expected_distinction=expected_distinction)))

    def review_contrast(self, contrast_hash, *, actor_ref, reviewer_kind, justification):
        return self._review('contrasts', contrast_hash, actor_ref=actor_ref,
                            reviewer_kind=reviewer_kind, justification=justification)

    def propose_manifest(self, witness_kind, witness_hashes):
        if witness_kind not in {'positive_exemplar', 'hard_negative'}:
            raise ValueError('unsupported provision witness kind')
        witnesses = [self._load('witnesses' if witness_kind == 'positive_exemplar' else 'contrasts', h)
                     for h in witness_hashes]
        if not witnesses:
            raise ValueError('empty provision manifest')
        positive = witnesses[0] if witness_kind == 'positive_exemplar' else self._load('witnesses', witnesses[0].positive_witness_hash)
        span = self._load('spans', positive.span_hash)
        return self._persist('manifests', LawProvisionWitnessManifestV1.create(dict(
            schema_version='law_provision_witness_manifest.v1', witness_kind=witness_kind, state='proposed',
            predecessor_manifest_hash=None, graph_intent_ref=self.intent.graph_intent_id,
            graph_intent_hash=self.intent.graph_intent_hash, source_binding=span.source_binding.model_dump(mode='json'),
            taxonomy_ref=self.taxonomy.taxonomy_ref, taxonomy_hash=self.taxonomy.taxonomy_hash,
            profile_id=self.profile_id, profile_hash=self.profile_hash,
            entries=[dict(witness_hash=h, review_event_ref=None, review_event_hash=None) for h in witness_hashes])))

    def accept_manifest(self, proposed_hash, review_event_refs):
        proposal = self._load('manifests', proposed_hash)
        if proposal.state != 'proposed' or len(review_event_refs) != len(proposal.entries):
            raise ValueError('exact proposed manifest and ordered reviews required')
        kind = 'witnesses' if proposal.witness_kind == 'positive_exemplar' else 'contrasts'
        entries = []
        for entry, ref in zip(proposal.entries, review_event_refs, strict=True):
            review = self._verify_review(kind, entry.witness_hash, ref)
            entries.append(dict(witness_hash=entry.witness_hash, review_event_ref=ref,
                                review_event_hash=review['event_hash']))
        payload = proposal.model_dump(mode='json')
        payload.update(state='accepted', predecessor_manifest_hash=proposed_hash, entries=entries)
        accepted = LawProvisionWitnessManifestV1.create(payload)
        self._verify_manifest(accepted)
        return self._persist('manifests', accepted)

    def _verify_manifest(self, manifest):
        if (manifest.state != 'accepted' or manifest.graph_intent_ref != self.intent.graph_intent_id
            or manifest.graph_intent_hash != self.intent.graph_intent_hash
            or manifest.profile_id != self.profile_id or manifest.profile_hash != self.profile_hash
            or manifest.taxonomy_ref != self.taxonomy.taxonomy_ref or manifest.taxonomy_hash != self.taxonomy.taxonomy_hash):
            raise ValueError('accepted provision manifest pin mismatch')
        proposed = self._load('manifests', manifest.predecessor_manifest_hash)
        expected = manifest.model_dump(mode='json')
        expected.update(state='proposed', predecessor_manifest_hash=None,
            entries=[dict(witness_hash=e.witness_hash, review_event_ref=None, review_event_hash=None) for e in manifest.entries])
        if LawProvisionWitnessManifestV1.create(expected) != proposed:
            raise ValueError('provision manifest predecessor mismatch')
        kind = 'witnesses' if manifest.witness_kind == 'positive_exemplar' else 'contrasts'
        span_hashes, text_hashes, positives = [], [], []
        for entry in manifest.entries:
            event = self._verify_review(kind, entry.witness_hash, entry.review_event_ref)
            if event['event_hash'] != entry.review_event_hash:
                raise ValueError('ordered provision review hash mismatch')
            artifact = self._load(kind, entry.witness_hash)
            witness = artifact if kind == 'witnesses' else self._load('witnesses', artifact.positive_witness_hash)
            self._verify_span_review(witness)
            span = self._load('spans', witness.span_hash)
            self.reconstruct(span)
            if span.source_binding != manifest.source_binding:
                raise ValueError('provision manifest source mismatch')
            span_hashes.append(span.content_hash)
            text_hashes.append(span.complete_text_hash)
            positives.append(witness)
        if kind == 'witnesses' and (len(set(span_hashes)) != len(span_hashes) or len(set(text_hashes)) != len(text_hashes)):
            raise ValueError('duplicate provision evidence is not a distinct positive')
        return tuple(positives)

    def qualify(self, positive_manifest_hash, hard_negative_manifest_hash):
        positive = self._load('manifests', positive_manifest_hash)
        negative = self._load('manifests', hard_negative_manifest_hash)
        if positive.witness_kind != 'positive_exemplar' or len(positive.entries) != 32:
            raise ValueError('law provision qualification requires exactly 32 positives')
        if negative.witness_kind != 'hard_negative' or len(negative.entries) != 6:
            raise ValueError('law provision qualification requires exactly 6 hard negatives')
        witnesses = self._verify_manifest(positive)
        self._verify_manifest(negative)
        positive_hashes = {e.witness_hash for e in positive.entries}
        contrasts = [self._load('contrasts', e.witness_hash) for e in negative.entries]
        if any(c.positive_witness_hash not in positive_hashes for c in contrasts):
            raise ValueError('contrast is not bound to a qualified positive')
        if len({c.scenario for c in contrasts}) != 6:
            raise ValueError('duplicate hard-negative scenario')
        composed = [w for w in witnesses if w.semantic_expectation.representation_kind == 'composed_frame_set']
        if not composed:
            raise ValueError('composed positive representative required')
        # This result is reconstructed, never accepted as caller-supplied gate authority.
        return dict(schema_version='accepted_law_provision_gate.v1',
            profile_id=self.profile_id, profile_hash=self.profile_hash,
            taxonomy_ref=self.taxonomy.taxonomy_ref, taxonomy_hash=self.taxonomy.taxonomy_hash,
            positive_manifest_ref='law-provision-manifest:' + positive.content_hash[7:],
            positive_manifest_hash=positive.content_hash, positive_count=32,
            hard_negative_manifest_ref='law-provision-manifest:' + negative.content_hash[7:],
            hard_negative_manifest_hash=negative.content_hash, hard_negative_count=6,
            representative_positive_witness_refs=['law-provision-witness:'+w.content_hash[7:]
                for w in (witnesses[0], composed[0])],
            hard_negative_witness_refs=['law-provision-contrast:'+c.content_hash[7:] for c in contrasts],
            material_mismatches=[])
