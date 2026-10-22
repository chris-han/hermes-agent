"""Controlled policy source/version engineering through existing workspace owners.

No active-policy pointer is written. Publication creates source/evidence and a
rebuild event; builds and evaluation records are non-authoritative observations.
"""
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
from contracts.policy_revision import PolicyRevisionRecordV1 as Record
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash
from services.teaching_candidate_scope import require_teaching_scope


def digest(text):
    return 'sha256:' + sha256(text.encode()).hexdigest()


def controlled_amendment(source):
    if ('v1.0' not in source or '## 第 3 条【开票超过一年】' not in source
        or '一年按公历一周年计算，不按固定 365 天计算' not in source):
        raise ValueError('POLICY_AMENDMENT_BASIS_MISMATCH')
    return source.replace('v1.0', 'v2.0（DEVELOPMENT；自 2026-10-16 起生效）', 1).replace('一年', '两年').replace('一周年', '两周年').replace('固定 365 天', '固定 730 天')


def markdown_blocks(source):
    """Exact character spans and stable logical clause ids; never line anchors."""
    matches = list(re.finditer(r'^## 第 (\d+) 条.*$', source, re.M))
    cuts = [(0, 'introduction')] + [(m.start(), 'clause:' + m.group(1)) for m in matches]
    blocks = []
    for index, (start, key) in enumerate(cuts):
        end = cuts[index + 1][0] if index + 1 < len(cuts) else len(source)
        text = source[start:end]
        blocks.append(dict(block_id=key, start=start, end=end, text=text, block_hash=digest(text)))
    return blocks


def deadline_outcome(issued, submitted, years):
    """Execute the existing core calendar predicate, independently of case labels."""
    from contracts.ontology_validation import CompiledPredicateV1
    from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
    try:
        if not issued or date.fromisoformat(issued) > date.fromisoformat(submitted):
            return 'UNKNOWN'
    except (ValueError, TypeError):
        return 'UNKNOWN'
    predicate = CompiledPredicateV1(predicate_id='deadline', operator='CALENDAR_YEARS_AFTER',
        fact_ref='submitted', comparison_fact_ref='issued', expected_value=years,
        ir_node_ref='clause:3', evidence_refs=('source-bound-deadline',))
    result = DeterministicRuleEvaluator._evaluate_predicate(predicate,
        dict(issued=issued, submitted=submitted), optional_absence_is_failure=False)
    return {'SATISFIED': 'exception_review_required', 'FAILED': 'allowed'}.get(result.result, 'UNKNOWN')


class PolicyRevisionService:
    def __init__(self, *, ctx, store, policy_owner):
        self.ctx, self.store, self.policy_owner = ctx, store, policy_owner
        self.scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        from storage.ontology_build_store import ontology_build_store_path
        if store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise PermissionError('POLICY_WORKSPACE_MISMATCH')
        self._access()

    def _access(self):
        require_teaching_scope(self.ctx, **self.scope)

    def _record(self, kind, data):
        return Record.create(**self.scope, actor_ref='user:' + self.ctx.user_id,
            kind=kind, produced_at=datetime.now(timezone.utc), data=data)

    def _list(self, kind):
        values = [self.read(pin.artifact_ref) for pin in self.store.list_artifact_pins(
            **self.scope, schema_version='policy_revision_record.v1')]
        return [value for value in values if value.kind == kind]

    def read(self, ref, expected_hash=None):
        self._access()
        value = self.store.get(**self.scope, artifact_ref=ref, contract_type=Record)
        if value is None or (expected_hash and expected_hash != value.artifact_hash):
            raise ValueError('POLICY_RECORD_PIN_MISMATCH')
        return value

    def _assets(self):
        root = Path(self.ctx.hermes_home) / 'demo-materials'
        # Shared launcher-managed assets are the authoring seed, never active policy.
        reference_text = (root / 'expense-policy-version-reference.json').read_text()
        if digest(reference_text) != 'sha256:67864c3aa7fdf8b64289ad38258b8b3d5e8c26f57f36b8d4748e6f1d60dc6369':
            raise ValueError('POLICY_REFERENCE_HASH_MISMATCH')
        reference = json.loads(reference_text)
        v1 = (root / '财务报销政策.md').read_text()
        if digest(v1) != reference['v1_hash'] or digest(controlled_amendment(v1)) != reference['v2_hash']:
            raise ValueError('POLICY_SEED_HASH_MISMATCH')
        return v1, reference

    @staticmethod
    def pin(value):
        return dict(artifact_ref=value.artifact_ref, artifact_hash=value.artifact_hash)

    def _operation(self, key, request_hash, record, verify=None, others=()):
        pin = self.store.put_many_with_operation_result(**self.scope,
            actor_ref='policy-revision-stream', operation=record.kind,
            request_id=key, request_hash=request_hash, artifacts=(*others, record),
            result_pin=ArtifactPinV1(artifact_kind='policy_revision', **self.pin(record)),
            verify_before_commit=verify)
        return self.read(pin.artifact_ref, pin.artifact_hash)

    def initialize(self):
        sources = self._list('source')
        if sources:
            return min(sources, key=lambda value: value.data['version'])
        v1, reference = self._assets()
        prepared = self.policy_owner.current()
        if prepared is None:
            raise ValueError('POLICY_K1_NOT_PREPARED')
        seed = self._record('source', dict(version=1, markdown=v1, source_hash=digest(v1),
            parent=None, effective_on='2026-10-15', blocks=markdown_blocks(v1),
            normative_pdf_hash=prepared['policy'].source_hash,
            active_k1_pin=prepared['policy'].candidate_pin, reference=reference))
        return self._operation('initial-source', digest(v1), seed)

    def consumer_source_revision(self, *, parent_ref, parent_hash, markdown, effective_on, session_ref):
        """Retain an authored DEVELOPMENT source branch without publishing policy."""
        self._access()
        parent=self.read(parent_ref,parent_hash)
        if parent.kind not in ('source','draft') or len(markdown.encode())>100_000:
            raise ValueError('SOURCE_REVISION_PARENT_OR_SIZE_INVALID')
        if parent.kind=='draft' and parent.data.get('session_ref')!=session_ref:
            raise PermissionError('SOURCE_REVISION_PARENT_SESSION_MISMATCH')
        if date.fromisoformat(effective_on).isoformat()!=effective_on:
            raise ValueError('SOURCE_REVISION_EFFECTIVITY_INVALID')
        source=self._record('draft',dict(lane='source_change',session_ref=session_ref,
            version=parent.data['version']+1,parent=self.pin(parent),markdown=markdown,source_hash=digest(markdown),
            effective_on=effective_on,blocks=markdown_blocks(markdown),authorship='ASSISTANT_AUTHORED',completeness='NOT_QUALIFIED'))
        self.store.put(source)
        return source

    def register_procurement_source(self, *, raw_json, session_ref, effective_on, profile_id=None):
        """Retain controlled source bytes/mapping without expense policy context."""
        from services.source_consumer_verification import procurement_source_content
        self._access()
        if date.fromisoformat(effective_on).isoformat()!=effective_on:
            raise ValueError('SOURCE_REVISION_EFFECTIVITY_INVALID')
        text,blocks=procurement_source_content(raw_json)
        import base64
        data=dict(lane='source_change',domain='procurement',session_ref=session_ref,
            version=1,parent=None,markdown=text,source_hash=digest(text),
            effective_on=effective_on,blocks=blocks,source_document_hash='sha256:'+sha256(raw_json).hexdigest(),
            original_json_b64=base64.b64encode(raw_json).decode(),
            authorship='ASSISTANT_AUTHORED',completeness='NOT_QUALIFIED')
        if profile_id is not None:
            from contracts.source_change_consumer_verification import AMOUNT_PROFILE
            if profile_id!=AMOUNT_PROFILE:raise ValueError('PROCUREMENT_SOURCE_PROFILE_INVALID')
            data['profile_id']=profile_id
        record=self._record('draft',data)
        return self._operation('procurement-source:'+session_ref,digest(json.dumps(data,sort_keys=True)),record)

    @staticmethod
    def canonical_consumer_source(source):
        """Exact Markdown blocks through the existing canonical source contracts."""
        from services.source_consumer_verification import canonical_source_for_record
        return canonical_source_for_record(source)


    def head(self):
        return max(self._list('source'), key=lambda value: value.data['version'])

    def context(self):
        baseline = self.initialize()
        sources = sorted(self._list('source'), key=lambda value: value.data['version'])
        builds = sorted(self._list('build'), key=lambda value: value.produced_at)
        runs = sorted(self._list('eval_run'), key=lambda value: value.produced_at)
        reports = sorted(self._list('eval_report'), key=lambda value: value.produced_at)
        drafts = sorted((r for r in self._list('draft') if r.data.get('domain')!='procurement'), key=lambda value: value.produced_at)
        return dict(sources=[self.present(s) for s in sources],
            draft=drafts[-1].model_dump(mode='json') if drafts else None,
            amendment=controlled_amendment(baseline.data['markdown']),
            builds=[v.model_dump(mode='json') for v in builds],
            eval_runs=[v.model_dump(mode='json') for v in runs],
            eval_reports=[v.model_dump(mode='json') for v in reports],
            active_k1_pin=baseline.data['active_k1_pin'],
            activation=dict(available=False, reason='DEVELOPMENT_NOT_PRODUCTION_AUTHORITY'),
            formal_reference='NOT_QUALIFIED')

    def present(self, source):
        value = source.model_dump(mode='json')
        value['data']['spans'] = [dict(evidence_ref=digest(source.artifact_ref + ':' + b['block_id'] + ':' + b['block_hash']),
            quote=b['text'], anchor_ref=b['block_id'], presentation_location=dict(kind='markdown_block',
                blockId=b['block_id'], blockHash=b['block_hash'], sourceHash=source.data['source_hash'],
                startOffset=b['start'], endOffset=b['end'], exactText=b['text'])) for b in source.data['blocks']]
        from contracts.evidence_location import EvidenceRef, EvidenceSelector, SourceIdentity
        identity = SourceIdentity(schema_version='semantier.source_identity.v1',
            source_identity_ref=source.artifact_ref, **self.scope, source_hash=source.data['source_hash'],
            media_type='text/markdown', original_name='Policy-v' + str(source.data['version']) + '.md',
            artifact_ref=source.artifact_ref, source_version=str(source.data['version']),
            supersedes_source_identity_ref=(source.data.get('parent') or {}).get('artifact_ref'),
            created_at=source.produced_at)
        value['source_identity'] = identity.model_dump(mode='json', by_alias=True)
        for span in value['data']['spans']:
            selector = EvidenceSelector(schema_version='semantier.evidence_selector.v1', selector_kind='composite',
                exact_quote=span['quote'], prefix=None, suffix=None, normalized_quote=None,
                structural_path=[span['anchor_ref']], table=None, source_element_ref=span['anchor_ref'])
            span['canonical_evidence'] = EvidenceRef(schema_version='semantier.evidence_ref.v1',
                evidence_ref=span['evidence_ref'], **self.scope, source_identity_ref=source.artifact_ref,
                selector_schema_version='semantier.evidence_selector.v1', selector_hash=selector.selector_hash(),
                selector=selector, evidence_role='source_clause', provenance=dict(source_hash=source.data['source_hash'],
                    source_version=source.data['version'], origin='controlled_development_source'),
                created_at=source.produced_at).model_dump(mode='json', by_alias=True)
        return value

    def save_draft(self, *, parent_ref, parent_hash, markdown, source_hash):
        self.initialize()
        parent = self.read(parent_ref, parent_hash)
        if parent.kind != 'source' or self.head().artifact_ref != parent_ref:
            raise ValueError('POLICY_STALE_PARENT')
        if digest(markdown) != source_hash:
            raise ValueError('POLICY_SOURCE_HASH_MISMATCH')
        if len(markdown.encode()) > 100_000:
            raise ValueError('POLICY_SOURCE_TOO_LARGE')
        record = self._record('draft', dict(parent=self.pin(parent), markdown=markdown,
            source_hash=source_hash, blocks=markdown_blocks(markdown)))
        self.store.put(record)
        return record.model_dump(mode='json')

    def publish(self, *, parent_ref, parent_hash, markdown, source_hash):
        self.initialize()
        parent = self.read(parent_ref, parent_hash)
        if parent.kind != 'source' or digest(markdown) != source_hash:
            raise ValueError('POLICY_SOURCE_HASH_MISMATCH')
        baseline = self.initialize()
        # Only the frozen DEVELOPMENT treatment or an unchanged source is eligible.
        if markdown not in (parent.data['markdown'], controlled_amendment(baseline.data['markdown'])):
            raise ValueError('POLICY_AMENDMENT_OUTSIDE_CONTROLLED_SCOPE')
        request_hash = 'sha256:' + content_hash(dict(parent=self.pin(parent), source_hash=source_hash))
        previous = self.store.get_operation_result(**self.scope, actor_ref='policy-revision-stream',
            operation='publication', request_id=parent_ref, request_hash=request_hash, artifact_kind='policy_revision')
        if previous:
            event = self.read(previous.artifact_ref)
            self.rebuild(event.artifact_ref, event.artifact_hash)
            return self.context()
        if self.head().artifact_ref != parent_ref:
            raise ValueError('POLICY_STALE_PARENT')
        source = self._record('source', dict(version=parent.data['version'] + 1, markdown=markdown,
            source_hash=source_hash, parent=self.pin(parent), effective_on='2026-10-16',
            blocks=markdown_blocks(markdown)))
        event = self._record('publication', dict(source=self.pin(source), parent=self.pin(parent),
            idempotency_key=source_hash, status='PUBLISHED'))
        def verify():
            self._access()
            existing = [s for s in self._list('source') if s.artifact_ref != source.artifact_ref]
            if max(existing, key=lambda s: s.data['version']).artifact_ref != parent_ref:
                raise ValueError('POLICY_STALE_PARENT')
        event = self._operation(parent_ref, request_hash, event, verify, (source,))
        self.rebuild(event.artifact_ref, event.artifact_hash)
        return self.context()

    def rebuild(self, event_ref, event_hash):
        event = self.read(event_ref, event_hash)
        if event.kind != 'publication':
            raise ValueError('POLICY_PUBLICATION_EVENT_REQUIRED')
        source = self.read(**self._read_args(event.data['source']))
        prior = self.store.get_operation_result(**self.scope, actor_ref='policy-revision-stream',
            operation='build', request_id=source.artifact_ref, request_hash=source.artifact_hash,
            artifact_kind='policy_revision')
        if prior:
            return self.read(prior.artifact_ref).model_dump(mode='json')
        try:
            graph, native = self._build_graph(source)
            data = dict(status='SUCCEEDED', source=self.pin(source), event=self.pin(event),
                graph=graph, semantica=native,
                delta='NO_SEMANTIC_CHANGE' if source.data['source_hash'] == self.read(**self._read_args(event.data['parent'])).data['source_hash'] else 'SEMANTIC',
                authority_delta='unavailable', active_k1_unchanged=True)
        except Exception as exc:
            failure = self._record('build', dict(status='FAILED_RETRYABLE', source=self.pin(source),
                event=self.pin(event), error=type(exc).__name__ + ':' + str(exc), active_k1_unchanged=True))
            self.store.put(failure)
            return failure.model_dump(mode='json')
        result = self._record('build', data)
        return self._operation(source.artifact_ref, source.artifact_hash, result).model_dump(mode='json')

    @staticmethod
    def _read_args(pin):
        return dict(ref=pin['artifact_ref'], expected_hash=pin['artifact_hash'])

    def _build_graph(self, source):
        from semantier.knowledge_feature_packs.semantica_adapter import (
            InProcessSemanticaAdapter, build_semantica_candidate_graph,
        )
        elements = [dict(text=b['text'], family_id=1, anchor=dict(anchor_id=b['block_id'],
            document_hash=source.data['source_hash'], document_part='paragraph', paragraph_index=i,
            start_offset=0, end_offset=len(b['text']), exact_text=b['text'], local_text_hash=b['block_hash']))
            for i, b in enumerate(source.data['blocks'])]
        native_records = []
        def retain(raw, raw_hash):
            record = self._record('native_extraction', dict(source=self.pin(source),
                native_hash=raw_hash, raw_output=raw))
            self.store.put(record)
            native_records.append(record)
            return record.artifact_ref
        native = build_semantica_candidate_graph(parsed_document=dict(
            document_hash=source.data['source_hash'], elements=elements),
            extraction_run_id=source.artifact_ref, adapter=InProcessSemanticaAdapter(artifact_writer=retain))
        # The closed deadline treatment is lowered deterministically; unrestricted
        # native entities/relations remain proposals with UNKNOWN applicability.
        graph = graph_for_source(source)
        return graph, dict(provider=native.provider_ref, commit=native.provider_commit,
            native_hash=native.native_artifact_hash, native_pin=self.pin(native_records[-1]),
            native_candidate=native.graph.model_dump(mode='json'),
            interpreted_relationships='UNKNOWN', authority='proposal_only',
            deadline_lowering='controlled-calendar-deadline/1')

    def evaluate(self, *, source_ref, source_hash, build_ref, build_hash, reference_complete=True):
        started_at = datetime.now(timezone.utc)
        from benchmark.knowledge_evaluation.graph.contracts import GraphBenchmarkCaseV1
        from benchmark.knowledge_evaluation.graph.metrics import evaluate_graph_case
        source = self.read(source_ref, source_hash)
        build = self.read(build_ref, build_hash)
        if (source.kind != 'source' or build.kind != 'build' or build.data['status'] != 'SUCCEEDED'
            or build.data['source'] != self.pin(source)):
            raise ValueError('POLICY_EVAL_PIN_MISMATCH')
        baseline = self.initialize()
        reference = baseline.data['reference']
        # Reference nodes/edges and temporal labels are frozen independently of candidate output.
        from benchmark.knowledge_evaluation.graph.contracts import GraphPredictionV1
        ref_years = 2 if source.data['source_hash'] == reference['v2_hash'] else 1
        refgraph = GraphPredictionV1(kind='candidate', graph_ref=source.artifact_ref,
            nodes=[dict(node_id=b['block_id'], label=b['text'].splitlines()[0], entity_type='policy_clause',
                source_anchor_refs=(b['block_id'],), topology_state='candidate') for b in source.data['blocks']],
            edges=[dict(edge_id='deadline:exception', subject_id='clause:3', predicate='requires', object_id='clause:4',
                threshold=ref_years, condition='CALENDAR_YEARS_AFTER', exception='finance_exception_review',
                scope=source.data['effective_on'], source_anchor_refs=('clause:3','clause:4'),
                extraction_run_ref=source.artifact_ref, topology_state='candidate')]).model_dump(mode='json')
        candidate = build.data['graph']
        case = GraphBenchmarkCaseV1(case_id=source.artifact_ref, source_document_hashes=(source.data['source_hash'],),
            challenge_tags=('deadline', 'exception', 'temporal', 'development'),
            reference=dict(reference_release_id='development:' + digest(json.dumps(reference, sort_keys=True))[7:],
                reference_release_hash=digest(json.dumps(reference, sort_keys=True)),
                nodes=refgraph['nodes'], edges=refgraph['edges']),
            prediction=candidate, annotation_version='controlled-development/1', adjudication_state='single')
        scored = evaluate_graph_case(case)
        years = candidate['edges'][0]['threshold']
        temporal = [dict(**row, observed=deadline_outcome(row['invoice_date'], row['submitted_on'], years),
            expected=row['v2'] if ref_years == 2 else row['v1'],
            applicability='NOT_APPLICABLE' if years == 2 and row['submitted_on'] < source.data['effective_on'] else 'DEVELOPMENT_HYPOTHESIS',
            decision_binding='K1') for row in reference['cases']]
        failures = sum(row['observed'] != row['expected'] for row in temporal)
        # Critical fidelity is checked separately from averages and from missing-reference status.
        failures += int(candidate['edges'] != refgraph['edges'])
        failures += int({n['node_id'] for n in candidate['nodes']} != {b['block_id'] for b in source.data['blocks']})
        metrics = {group:[dict(m) for m in values] for group,values in scored.metric_groups.items()}
        data = dict(source=self.pin(source), build=self.pin(build), status='COMPLETED',
            started_at=started_at.isoformat(), completed_at=datetime.now(timezone.utc).isoformat(),
            disposition='PASS' if failures == 0 and reference_complete else 'INSUFFICIENT_REFERENCE' if not reference_complete else 'FAIL_CRITICAL',
            reference_status=reference['reference_status'] if reference_complete else 'INSUFFICIENT_REFERENCE',
            reference_hash=digest(json.dumps(reference, sort_keys=True)), temporal=temporal, metrics=metrics,
            critical_regressions=failures, findings=self.present(source)['data']['spans'],
            evaluator=dict(name='benchmark.knowledge_evaluation.graph.metrics.evaluate_graph_case',
                hash=digest(Path(__import__('benchmark.knowledge_evaluation.graph.metrics', fromlist=['x']).__file__).read_text()),
                implementation_hash=digest(Path(__file__).read_text()),
                temporal='services.deterministic_rule_evaluator.CALENDAR_YEARS_AFTER'),
            judge=dict(verdict='ABSTAIN', reason='NO_QUALIFIED_INTERPRETED_REFERENCE', model_calls=0),
            formal_qualification='NOT_QUALIFIED', cost=dict(provider_calls=0, usd=0), authority_granted=False)
        report = self._record('eval_report', data)
        run = self._record('eval_run', dict(source=self.pin(source), build=self.pin(build), report=self.pin(report), status='COMPLETED'))
        from benchmark.knowledge_evaluation.canonical_evidence import CanonicalEvidenceProducer
        from eos.evaluation_control_plane import EvaluationSubjectManifest, EvaluationRun, CriticalGateResult
        from storage.evaluation_control_plane_store import EvaluationControlPlaneStore
        canonical_store = EvaluationControlPlaneStore(self.store._path)
        try:
            subject = EvaluationSubjectManifest(subject_id=source.artifact_ref, subject_type='knowledge_runtime',
                runtime_commit=__import__('os').environ.get('SEMANTIER_RUNTIME_COMMIT', 'UNAVAILABLE'), environment_image_digest=digest(__import__('sys').version),
                knowledge_version=build.artifact_hash, policy_bundle_version=source.data['source_hash'])
            canonical = CanonicalEvidenceProducer(canonical_store).complete(subject=subject,
                run=EvaluationRun(run_id=run.artifact_ref, benchmark_family_ref='knowledge-evaluation-stack.v2',
                    scenario_ref='policy-version-engineering.v1', subject_manifest_ref=subject.to_record()['content_hash'],
                    environment_manifest_ref=digest(__import__('sys').version), evaluator_bundle_ref=data['evaluator']['hash'],
                    started_at=started_at.isoformat(), completed_at=run.produced_at.isoformat(), validity='VALID'),
                evidence_id='policy-evaluation:' + run.artifact_hash[7:],
                scorecard=dict(not_evaluable=True, engineering_disposition=data['disposition'], metrics=metrics),
                critical_gate_results=[CriticalGateResult(gate_id='policy-version-fidelity',
                    status='FAIL' if failures else 'PASS', dimension='grounding',
                    message='Controlled DEVELOPMENT source/graph/temporal comparison; formal reference unqualified.',
                    artifact_refs=[report.artifact_ref])], artifact_refs=[report.artifact_ref],
                expected_pins=dict(source=source.artifact_hash, build=build.artifact_hash),
                observed_pins=dict(source=source.artifact_hash, build=build.artifact_hash))
        finally:
            canonical_store.close()
        self.store.put_many((report, run))
        return dict(run=run.model_dump(mode='json'), report=report.model_dump(mode='json'), canonical_evaluation=dict(run=canonical.run,evidence=canonical.evidence))

    def register_procurement_incident_profile(self, *, base_source, supplement_text, session_ref, profile_id):
        """Explicit synthetic amendment, never an organizational policy adoption."""
        import base64
        from contracts.source_change_consumer_verification import INCIDENT_PROFILE
        from services.source_consumer_verification import raw_hash, procurement_profile_content
        self._access()
        if profile_id!=INCIDENT_PROFILE or raw_hash(supplement_text)!= 'sha256:8fc931d2abac59e76b6550c22612f6cfe50d34cb25539af7ed17f17890d98cda':
            raise ValueError('INCIDENT_PROFILE_REFERENCE_MISMATCH')
        if self.read(base_source.artifact_ref,base_source.artifact_hash)!=base_source or base_source.data.get('session_ref')!=session_ref:
            raise PermissionError('INCIDENT_BASE_SOURCE_SCOPE_MISMATCH')
        supplement=self._record('draft',dict(lane='incident_supplement',domain='procurement',session_ref=session_ref,profile_id=profile_id,
            text_b64=base64.b64encode(supplement_text).decode(),document_hash=raw_hash(supplement_text),
            markdown=supplement_text.decode(),source_hash=raw_hash(supplement_text),
            authorship='ASSISTANT_AUTHORED',completeness='NOT_QUALIFIED',status='PROPOSED_FOR_REVIEW'))
        self.store.put(supplement)
        data=dict(base_source.data,profile_id=profile_id,version=2,base_source=base_source.model_dump(mode='json'),
                  supplement=supplement.model_dump(mode='json'))
        text,blocks=procurement_profile_content(data)
        data.update(markdown=text,blocks=blocks,source_hash=digest(text))
        record=self._record('draft',data)
        return self._operation('incident-profile:'+session_ref,digest(json.dumps(data,sort_keys=True)),record)

    def retain_synthetic_incident_witness(self, *, source, session_ref, witness):
        from contracts.source_change_consumer_verification import SyntheticIncidentWitnessV1,INCIDENT_PROFILE
        self._access()
        witness=SyntheticIncidentWitnessV1.model_validate(witness)
        if self.read(source.artifact_ref,source.artifact_hash)!=source or source.data.get('session_ref')!=session_ref or source.data.get('profile_id')!=INCIDENT_PROFILE:
            raise PermissionError('INCIDENT_WITNESS_SOURCE_SCOPE_MISMATCH')
        if witness.source_pin.model_dump(mode='json')!=dict(artifact_kind='policy_revision',**self.pin(source)):
            raise ValueError('INCIDENT_WITNESS_SOURCE_PIN_MISMATCH')
        record=self._record('consumer_candidate',dict(lane='incident_witness',session_ref=session_ref,
            witness=witness.model_dump(mode='json'),authorship='ASSISTANT_AUTHORED',completeness='NOT_QUALIFIED',
            institutional_authentication='NOT_ESTABLISHED'))
        self.store.put(record)
        return record



def graph_for_source(source):
    from benchmark.knowledge_evaluation.graph.contracts import GraphPredictionV1
    years = 2 if '两周年' in source.data['markdown'] else 1
    spans = source.data['blocks']
    nodes = [dict(node_id=b['block_id'], label=b['text'].splitlines()[0], entity_type='policy_clause',
        source_anchor_refs=(b['block_id'],), topology_state='candidate') for b in spans]
    edge = dict(edge_id='deadline:exception', subject_id='clause:3', predicate='requires', object_id='clause:4',
        threshold=years, condition='CALENDAR_YEARS_AFTER', exception='finance_exception_review',
        scope=source.data['effective_on'], source_anchor_refs=('clause:3','clause:4'),
        extraction_run_ref=source.artifact_ref, topology_state='candidate')
    return GraphPredictionV1(kind='candidate', graph_ref=source.artifact_ref, graph_version=str(source.data['version']),
        nodes=nodes, edges=[edge]).model_dump(mode='json')
