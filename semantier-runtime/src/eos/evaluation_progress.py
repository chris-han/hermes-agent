"""Pure, deterministic evaluation reading. Never certify, dispatch or ingest."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from math import isfinite
from contracts.evaluation_progress import EvaluationProgressV1
from eos.artifact_hashing import content_hash

# Exact registered engineering intake sources. No caller-provided path is resolved.
ARCHIVES = {
 'harness-bench-source-stop': ('docs/research/logs/harness-bench-runtime-v1/result.json', '5120ef1ee4fe8ee8d421c1e0376467fc0ded0ed349b8f53ba853c9c7e96038eb'),
 'contractnli-repeatability': ('docs/research/logs/contractnli-adjacency-repeatability-v1/reduction.json', 'fc23c050174e77b781447420f5546eeb745723df8993e781f8af11c890ff19b2'),
 'native-mapping-denial': ('docs/research/logs/harness-native-action-mapping-v2/reduction-1.json', 'fc21097758ea3cd24d51df81cd346673b33888d09b4883b55369f6f82f422bd3'),
}

METRIC_SOURCE_PIN = dict(artifact_kind='metric_catalog', artifact_ref='docs/derived/learning-change-evaluation-metrics-v1.md', artifact_hash='sha256:595acc8f1f538348a2dd4d028c19d556d180e723ca97570c0faa28512ecb60f6')

METRIC_IDS = ('INT-01', 'INT-02', 'INT-03', 'OUT-01', 'OUT-02', 'OUT-03', 'OUT-04', 'CTX-01', 'CTX-02', 'CTX-03', 'CTX-04', 'SKL-01', 'SKL-02', 'SKL-03', 'EXE-01', 'EXE-02', 'EXE-03', 'EXE-04', 'SAF-01', 'SAF-02', 'SAF-03', 'SAF-04', 'REL-01', 'REL-02', 'REL-03', 'REL-04', 'USR-01', 'USR-02', 'USR-03', 'USR-04', 'EFF-01', 'EFF-02', 'EFF-03', 'EFF-04', 'EFF-05', 'GEN-01', 'GEN-02', 'GEN-03', 'GEN-04', 'LRN-01', 'LRN-02', 'LRN-03', 'LRN-04', 'LRN-05', 'MET-01', 'MET-02', 'MET-03', 'MET-04', 'MET-05')

def selected_profile_coverage() -> dict[str, Any]:
    """Catalog coverage, not metric readiness or measured efficacy."""
    entries = []
    for metric in METRIC_IDS:
        reason = 'No qualified selected-profile measurement producer or bound case/reference/instrument manifest.'
        if metric.startswith('USR-'): reason = 'No actual human observation bound to this profile.'
        if metric.startswith('EFF-'): reason = 'Complete actor, simulator and auxiliary accounting unavailable for this profile.'
        if metric.startswith('MET-'): reason = 'Instrument qualification and probability/calibration basis unavailable.'
        applicability = 'applicable' if metric.startswith(('MET-', 'SAF-', 'REL-', 'USR-', 'EFF-')) else 'unresolved'
        entries.append(dict(metric_id=metric, applicability=applicability, measurement_status='not_measured', value=None, reason=reason,
                            evidence_refs=['docs/derived/learning-change-evaluation-metrics-v1.md'], instrument_qualification=None, catalog_pin=METRIC_SOURCE_PIN))
    return dict(scope='selected_profile', profile_id='learning-change-review-engineering.v1', checked=0, unmeasured=len(entries), entries=entries,
                limitations=['49 catalog entries are not a metric-readiness denominator.'])


def source_readiness() -> list[dict[str, Any]]:
    ref = 'docs/derived/learning-change-evaluation-metrics-v1.md'
    return [
        dict(source='WorkBuddy Office v1.0', role='task_and_native_reference_candidate', source_ref=ref+'#11-public-benchmark-candidates--source-suitability-2026-10-19', release='625b2233093ae4f23e76be28c1f341d41cc70373', score_profiles=['Rule-only','Rule–Judge'], execution='not_run', adapter='unverified', certification=None, reason='Rights/assets and installed-agent adapter unqualified; score profiles are incompatible.'),
        dict(source='JobBench', role='professional_task_candidate', source_ref=ref+'#11-public-benchmark-candidates--source-suitability-2026-10-19', release=None, execution='not_run', adapter='unverified', certification=None, reason='Dataset revision unresolved; main/easy variants and source families are correlated.'),
        dict(source='Jev-RAG', role='grounding_instrument_diagnostic_candidate', source_ref=ref+'#12-jev-rag-claim-verification-supplement--2026-10-19', release='204349a675d43c8d5696b66e74ff9aa7aff9260a', reference_role='source_derived_labels', prediction_role='Jev_predictions_not_independent_gold', execution='not_run', adapter='unverified', certification=None, reason='Reference transformation/grouping unqualified; missing annotation is not proven truth or semantic UNKNOWN.'),
    ]


def _finish(payload: dict) -> dict:
    validated = EvaluationProgressV1.model_validate(payload).model_dump()
    validated['content_hash'] = content_hash({k:v for k,v in validated.items() if k != 'content_hash'})
    return validated


def diagnostic_progress(source_id: str, repository_root: Path) -> dict:
    path, expected = ARCHIVES[source_id]
    root = repository_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root): raise ValueError('archive escaped registered root')
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != expected: raise ValueError('archive hash mismatch')
    data = json.loads(raw)
    pin = dict(artifact_kind='diagnostic_archive', artifact_ref=path, artifact_hash='sha256:'+expected)
    if source_id == 'harness-bench-source-stop':
        counts = dict(assigned=data['assigned_executions'], scored=data['scored_attempts'], actor_failures=data['failed_actor_executions'], valid=data['valid_executions'])
    elif source_id == 'contractnli-repeatability':
        groups = list(data['denominators'].values())
        pairs = data['analysis']['pairs']
        counts = dict(assigned=data['assigned'], started=sum(x['starts'] for x in groups), forwarded=sum(x['sends'] for x in groups), not_run=sum(x['not_run'] for x in groups), ties=sum(x['joint']=='TIE' for x in pairs), unassessable=sum(x['joint']=='UNASSESSABLE' for x in pairs), pair_blocks=data['analysis']['pair_blocks'])
    else:
        counts = {**data['denominators'], **data['counters']}
    objects = [dict(object_id=source_id+':run', kind='run', availability='available', native_verdict=data['disposition'], reason='Original diagnostic disposition; no certification implied.', source_pins=[pin])]
    for kind in ['output','revision','path','contribution']:
        objects.append(dict(object_id=source_id+':'+kind, kind=kind, availability='unassessed', reason='No independently qualified '+kind+' assessment in registered reduction.', source_pins=[pin]))
    source_fields = {'disposition': data['disposition']}
    if source_id == 'native-mapping-denial':
        source_fields['attempts'] = data['attempts']
        objects[1].update(availability='available', native_verdict=data['attempts'][0]['final'], reason='Native loop completed, but actual task_complete=false and source_observed=false.')
    elif source_id == 'contractnli-repeatability':
        source_fields['denominators'] = data['denominators']
        source_fields['analysis.decision'] = data['analysis']['decision']
    else:
        source_fields.update({k:data[k] for k in ['assigned_executions','scored_attempts','stop_gate','cost_per_verified_completion','host_agent_overhead']})
    excerpts = _root_excerpts(raw, set(source_fields), pin)
    return _finish(dict(source_excerpts=excerpts,source_fields=source_fields,source_id=source_id,source_owner='diagnostic_archive',produced_at=data.get('produced_at'),source_pins=[pin],objects=objects,counts=counts,metric_coverage=selected_profile_coverage()['entries'],source_readiness=source_readiness(),limitations=['Historical diagnostic only; no model, human, learning or current-composition efficacy claim.','Missing measurements remain unknown. Counts overlap and are not a partition.','Protected references and raw model outputs are not exported.'] + data.get('controls_limits',[])))


def canonical_progress(store, *, run_id: str, evidence_id: str, organization_id: str, workspace_id: str | None) -> dict:
    """Caller must first establish scoped ownership through BenchmarkReadModelStore."""
    run, evidence = store.get_run(run_id), store.get_evidence(evidence_id)
    if evidence['run_ref'] != run_id: raise ValueError('canonical evidence/run mismatch')
    pins = []
    for record, ref in [(run,run_id),(evidence,evidence_id)]:
        digest = content_hash({k:v for k,v in record.items() if k!='content_hash'})
        if digest != record['content_hash']: raise ValueError('canonical record hash mismatch')
        pins.append(dict(artifact_kind=record['schema_version'],artifact_ref=ref,artifact_hash='sha256:'+digest))
    allowed_metrics = {'current_round_completion', 'cumulative_obligation_pass_rate', 'prior_capability_regression_count', 'critical_gate_failure_count', 'case_count', 'decision_accuracy', 'fixture_count', 'precision', 'recall', 'f1', 'scored_attempts', 'assigned_executions'}
    source_fields = dict(validity=run['validity'], certification_result=evidence['certification_result'], scorecard={k:v for k,v in evidence['scorecard'].items() if k in allowed_metrics and (type(v) in (int,float,bool) or v is None)})
    return _finish(dict(source_fields=source_fields,source_id=run_id,source_owner='canonical',organization_id=organization_id,workspace_id=workspace_id,produced_at=run['completed_at'],source_pins=pins,
      objects=[dict(object_id=run_id+':'+kind,kind=kind,availability='available' if kind=='run' else 'unassessed',native_verdict=evidence['certification_result'] if kind=='run' else None,reason='Owner-native certification retained.' if kind=='run' else 'Explicit qualified assessment not supplied.',source_pins=pins) for kind in ['run','output','revision','path','contribution']],counts=dict(assigned=None,scored=None),metric_coverage=selected_profile_coverage()['entries'],source_readiness=source_readiness(),limitations=['Native certification is separate from execution mode and admission.','Protected evaluator artifacts omitted; only exact allowed record identities exported.']))


def comparable_delta(before, after, before_basis: dict, after_basis: dict) -> dict:
    keys = ('protocol','task_set','reference','evaluator','model','runtime','hermes','skill','config','environment','scale','release','score_profile')
    missing = [k for k in keys if before_basis.get(k) is None or after_basis.get(k) is None]
    mismatch = [k for k in keys if before_basis.get(k) != after_basis.get(k)]
    status = 'unknown' if missing else 'not_comparable' if mismatch else 'compatible'
    measured = type(before) in (int,float) and type(after) in (int,float) and isfinite(before) and isfinite(after)
    return dict(status=status, reasons=missing or mismatch,delta=after-before if status=='compatible' and measured else None)


def explicit_lineage(nodes: list[str], edges: list[tuple[str,str]], path: list[str]) -> dict:
    if len(nodes)!=len(set(nodes)): raise ValueError('duplicate node IDs')
    children = {n:[] for n in nodes}
    for a,b in set(edges):
        if a not in children or b not in children: raise ValueError('partial lineage endpoint')
        children[a].append(b)
    done, active = set(), set()
    def visit(n):
        if n in active: raise ValueError('lineage cycle')
        if n in done: return
        active.add(n)
        for child in children[n]: visit(child)
        active.remove(n); done.add(n)
    for node in nodes: visit(node)
    if any(n not in children for n in path) or any((a,b) not in edges for a,b in zip(path,path[1:])): raise ValueError('path missing explicit edge')
    return dict(path=path,off_path_parents=sorted({a for a,b in edges if b in path and a not in path}),contribution_count=None,limitation='Association only; no inferred causal contribution.')


def explicit_contributions(records: list[dict]) -> list[dict]:
    """Deduplicate only exact documented family/artifact identity, never semantics."""
    from contracts.artifact_pin import ArtifactPinV1
    unique = {}
    for record in records:
        if set(record) != {'family_id', 'artifact_pin', 'maturity', 'limitation'} or not record['family_id']:
            raise ValueError('explicit family, exact pin, maturity and limitation required')
        pin = ArtifactPinV1.model_validate(record['artifact_pin'])
        identity = (record['family_id'], pin.artifact_ref, pin.artifact_hash)
        if identity in unique and unique[identity] != record:
            raise ValueError('conflicting contribution metadata')
        unique[identity] = record
    return [unique[key] for key in sorted(unique)]


def _root_excerpts(raw: bytes, allowed: set[str], pin: dict) -> list[dict]:
    """Locate exact original top-level JSON values after whole-byte integrity check."""
    text = raw.decode('utf-8')
    decoder = json.JSONDecoder()
    i = text.index('{') + 1
    excerpts = []
    while True:
        while text[i].isspace(): i += 1
        if text[i] == '}': break
        key, i = decoder.raw_decode(text, i)
        while text[i].isspace(): i += 1
        if text[i] != ':': raise ValueError('invalid object delimiter')
        i += 1
        while text[i].isspace(): i += 1
        start = i
        _, i = decoder.raw_decode(text, i)
        if key in allowed:
            value = text[start:i]
            excerpts.append(dict(source_pin=pin, json_pointer='/'+key.replace('~','~0').replace('/','~1'),start_byte=len(text[:start].encode('utf-8')),end_byte=len(text[:i].encode('utf-8')),exact_text=value,excerpt_hash='sha256:'+sha256(value.encode('utf-8')).hexdigest()))
        while text[i].isspace(): i += 1
        if text[i] == '}': break
        if text[i] != ',': raise ValueError('invalid object delimiter')
        i += 1
    return excerpts
