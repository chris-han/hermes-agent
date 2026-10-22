"""Governed persistence and cold retrieval of exact paired-evaluation companions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Callable
from types import SimpleNamespace

from pydantic import BaseModel, TypeAdapter
from contracts.ontology_building import (ArtifactPinV1, OntologyBuildManifestV1, ExploratoryGraphSnapshotV1,
    CanonicalEvidenceAdequacyAssessmentV1, CompetencyContractV1)
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_paired_evaluation import (ArtifactPayloadPinV1, DeterministicRuleObligationV1,
    EvaluatorCodeUnitV1, ObservedFactsV1, OntologyPairedObligationSuiteV1,
    OntologyPairedEvaluationReportV1, OntologyPairedEvaluationRunV1,
    OntologyPairedEvaluationResultV1, PairedEvaluationSubjectV1,
    PairedEvaluationPolicyV1, PairedEvaluatorBundleV1)
from contracts.ontology_seed_alignment import (SourceGroundedSeedDeltaV1, OntologySeedAlignmentCommandV1,
    INPUT_PIN_FIELDS)
from contracts.ontology_seed_application import SeedSelectionReceiptV1
from contracts.ontology_validation import CompiledRulePlanV1, RuntimeSemanticEvaluationBundleV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.runtime_learning_paired_evaluation import (
    RuntimeRepairObligationSuiteV1,
    RuntimeRepairPairV1,
    RuntimeRepairPairedComparisonViewV1,
)
from eos.artifact_hashing import content_hash
from services.ontology_audit_service import OntologyAuditService
from services.ontology_build_coordinator import OntologyBuildCoordinator
from services.ontology_paired_evaluation_service import OntologyPairedEvaluationService, _plan_hash
from storage.ontology_build_store import OntologyBuildStore, _IDENTITY_FIELDS

# This archives the owned executor/contract/hash implementation closure, not the
# Python interpreter or third-party dependencies. GET never reads these files.
_CODE_MODULES = ('services.ontology_paired_comparison_service', 'services.ontology_paired_evaluation_service', 'services.deterministic_rule_evaluator',
    'contracts.ontology_paired_evaluation', 'contracts.ontology_validation', 'eos.artifact_hashing')
_RUNTIME_REPAIR_MODULE = 'contracts.runtime_learning_paired_evaluation'
_CONTRACT_MODULES = ('contracts.ontology_building', 'contracts.ontology_build_inputs',
    'contracts.ontology_paired_evaluation', 'contracts.ontology_seed_alignment',
    'contracts.ontology_seed_application', 'contracts.ontology_seed_state',
    'contracts.ontology_hub_snapshot', 'contracts.ontology_seeds',
    'contracts.ontology_validation', 'contracts.normative_semantics', 'contracts.validation_contract')
_CONTRACT_MODULES = (*_CONTRACT_MODULES, _RUNTIME_REPAIR_MODULE)


def _contract(schema):
    for name in _CONTRACT_MODULES:
        for value in vars(import_module(name)).values():
            if isinstance(value, type) and issubclass(value, BaseModel):
                field = value.model_fields.get('schema_version')
                if field is not None and field.default == schema:
                    return value
    raise ValueError(f'unsupported comparison closure artifact: {schema}')


def _load(store, tenant, ref, digest=None, cls=None, payload_hash=None):
    raw = store.get_raw(organization_id=tenant[0], workspace_id=tenant[1], artifact_ref=ref)
    if raw is None:
        raise LookupError(f'missing pinned comparison artifact: {ref}')
    schema = raw['schema_version']
    if schema not in _IDENTITY_FIELDS:
        raise ValueError('unsupported comparison artifact identity')
    value = (cls or _contract(schema)).model_validate(raw['payload'])
    payload = value.model_dump(mode='json')
    ref_field, hash_field = _IDENTITY_FIELDS[schema]
    if (payload.get('schema_version'), payload.get(ref_field), payload.get(hash_field),
            payload.get('organization_id'), payload.get('workspace_id')) != (
            schema, ref, raw['artifact_hash'], *tenant):
        raise ValueError('comparison artifact row/payload identity mismatch')
    if digest is not None and raw['artifact_hash'] != digest:
        raise ValueError('comparison artifact exact hash mismatch')
    if payload_hash is not None and 'sha256:' + content_hash(payload) != payload_hash:
        raise ValueError('comparison artifact full payload hash mismatch')
    return value


def _pinned(store, tenant, pin, cls=None):
    return _load(store, tenant, pin.artifact_ref, pin.artifact_hash, cls, getattr(pin, 'payload_hash', None))


def _child(manifest, kind):
    found = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
    if len(found) != 1:
        raise ValueError(f'comparison requires one exact manifest child: {kind}')
    return found[0]


def _pin(value, ref, digest):
    return ArtifactPayloadPinV1.of(value, ref, digest)


@dataclass(frozen=True)
class PairedComparisonView:
    result: OntologyPairedEvaluationResultV1
    evaluator_bundle: PairedEvaluatorBundleV1
    suite: OntologyPairedObligationSuiteV1 | None
    audits: tuple
    inputs: tuple[BaseModel, ...]

    @property
    def comparison_ref(self):
        return self.result.report.report_ref

    @property
    def comparison_hash(self):
        return self.result.report.report_hash

    def to_dict(self):
        return dict(comparison_ref=self.comparison_ref, comparison_hash=self.comparison_hash,
            **self.result.model_dump(mode='json'), evaluator_bundle=self.evaluator_bundle.model_dump(mode='json'),
            suite=self.suite.model_dump(mode='json') if self.suite else None,
            audits=[item.model_dump(mode='json') for item in self.audits],
            inputs=[item.model_dump(mode='json') for item in self.inputs])


class OntologyPairedComparisonService:
    def __init__(self, *, clock: Callable[[], datetime] | None = None):
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _pair(self, store, tenant, baseline_ref, baseline_hash, successor_ref, successor_hash):
        baseline = _load(store, tenant, baseline_ref, baseline_hash, OntologyBuildManifestV1)
        successor = _load(store, tenant, successor_ref, successor_hash, OntologyBuildManifestV1)
        if (successor.predecessor_build_ref, successor.predecessor_build_hash) != (baseline_ref, baseline_hash):
            raise ValueError('comparison successor does not pin exact baseline')
        from services.ontology_seed_application_service import OntologySeedApplicationService
        OntologySeedApplicationService()._validate_source_first_baseline(store,
            SimpleNamespace(organization_id=tenant[0], workspace_id=tenant[1]), baseline)
        audits = []
        snapshots = []
        envelopes = []
        for manifest in (baseline, successor):
            assessment = _pinned(store, tenant, _child(manifest, 'canonical_evidence_adequacy_assessment'), CanonicalEvidenceAdequacyAssessmentV1)
            rebuilt = OntologyBuildCoordinator().project_manifest(build_ref=manifest.build_ref,
                organization_id=tenant[0], workspace_id=tenant[1], assurance_profile=manifest.assurance_profile,
                assessment=assessment, additional_artifacts=tuple(pin for pin in manifest.child_artifacts
                    if pin.artifact_kind != 'canonical_evidence_adequacy_assessment'), created_at=manifest.created_at,
                predecessor_build_ref=manifest.predecessor_build_ref, predecessor_build_hash=manifest.predecessor_build_hash)
            if rebuilt != manifest:
                raise ValueError('comparison manifest original hash basis mismatch')
            envelope = _pinned(store, tenant, _child(manifest, 'ontology_build_inputs'), OntologyBuildInputEnvelopeV1)
            envelopes.append(envelope)
            archived = {item.artifact_ref for item in envelope.audit_inputs()}
            # Validate every actual model before packaging raw audit payloads.
            for child in manifest.child_artifacts:
                if child.artifact_ref not in archived:
                    _pinned(store, tenant, child)
            audits.append(OntologyAuditService().export(store=store, organization_id=tenant[0],
                workspace_id=tenant[1], build_ref=manifest.build_ref))
            snapshots.append(_pinned(store, tenant, _child(manifest, 'exploratory_graph_snapshot'), ExploratoryGraphSnapshotV1))
        if envelopes[0].model_dump(exclude={'build_ref','envelope_ref','envelope_hash'}) != envelopes[1].model_dump(exclude={'build_ref','envelope_ref','envelope_hash'}):
            raise ValueError('comparison archived source/intent inputs differ')
        delta = _pinned(store, tenant, _child(successor, 'source_grounded_seed_delta'), SourceGroundedSeedDeltaV1)
        command = _pinned(store, tenant, _child(successor, 'ontology_seed_alignment'), OntologySeedAlignmentCommandV1)
        selection = _pinned(store, tenant, _child(successor, 'seed_selection_receipt'), SeedSelectionReceiptV1)
        if (delta.alignment_ref, delta.alignment_hash, delta.inputs) != (command.alignment_ref, command.alignment_hash, command.inputs):
            raise ValueError('comparison delta/command input mismatch')
        for name, (ref, digest) in INPUT_PIN_FIELDS.items():
            _pinned(store, tenant, getattr(command.inputs, name))
        if (selection.baseline_manifest != command.inputs.baseline_manifest or
                selection.imported_snapshot != command.inputs.seed_snapshot or
                selection.imported_profile != command.inputs.seed_profile):
            raise ValueError('comparison selected seed does not bind alignment inputs')
        subject = PairedEvaluationSubjectV1(baseline_manifest=_pin(baseline,'build_ref','build_hash'),
            successor_manifest=_pin(successor,'build_ref','build_hash'),
            baseline_snapshot=_pin(snapshots[0],'snapshot_ref','snapshot_hash'),
            successor_snapshot=_pin(snapshots[1],'snapshot_ref','snapshot_hash'),
            alignment_delta=_pin(delta,'delta_ref','delta_hash'))
        if (command.inputs.baseline_manifest.artifact_ref, command.inputs.baseline_manifest.artifact_hash,
                command.inputs.baseline_manifest.payload_hash) != (
                subject.baseline_manifest.artifact_ref, subject.baseline_manifest.artifact_hash, subject.baseline_manifest.payload_hash):
            raise ValueError('comparison alignment baseline mismatch')
        if (snapshots[1].predecessor_snapshot_ref, snapshots[1].predecessor_snapshot_hash,
                snapshots[1].triggering_ref) != (snapshots[0].snapshot_ref, snapshots[0].snapshot_hash, delta.delta_ref):
            raise ValueError('comparison snapshot predecessor/delta mismatch')
        for snapshot, envelope in zip(snapshots, envelopes):
            from contracts.graph_intent_brief import GraphIntentBrief
            binding = GraphIntentBrief.from_dict(envelope.graph_intent).source_binding
            if (snapshot.source_identity_ref,snapshot.source_hash,snapshot.canonical_source_ir_ref,
                    snapshot.canonical_source_ir_hash,snapshot.build_ref) != (
                    binding.source_identity_ref,binding.source_hash,binding.canonical_source_ir_ref,
                    binding.canonical_source_ir_hash,envelope.build_ref):
                raise ValueError('comparison snapshot does not bind archived canonical source')
        snapshot = snapshots[1]
        payload = snapshot.model_dump(mode='json',exclude={'snapshot_ref','snapshot_hash'})
        payload['created_at'] = snapshot.created_at.isoformat()
        digest = content_hash(payload)
        if (snapshot.snapshot_ref,snapshot.snapshot_hash) != ('exploratory-snapshot:'+digest,'sha256:'+digest):
            raise ValueError('comparison successor snapshot original hash basis mismatch')
        if (snapshot.nodes,snapshot.edges,snapshot.created_at) != (
                (*snapshots[0].nodes,*delta.added_nodes),(*snapshots[0].edges,*delta.added_edges),command.reviewed_at):
            raise ValueError('comparison successor differs from exact archived delta')
        for kind, ref_field, hash_field, prefix, exclude in (
            ('candidate_graph_lineage_projection','lineage_ref','lineage_hash','candidate-graph-lineage',{'schema_version','created_at'}),
            ('candidate_graph_diff_projection','diff_ref','diff_hash','candidate-graph-diff',{'schema_version','created_at'})):
            artifact = _pinned(store,tenant,_child(successor,kind))
            material = artifact.model_dump(mode='json',exclude={ref_field,hash_field,*exclude})
            if not exclude: material['created_at'] = artifact.created_at.isoformat()
            digest = content_hash(material)
            if (getattr(artifact,ref_field),getattr(artifact,hash_field),artifact.created_at) != (
                    prefix+':'+digest,'sha256:'+digest,command.reviewed_at):
                raise ValueError('comparison diagnostic artifact original hash basis mismatch')
        return baseline, successor, snapshots, delta, subject, tuple(audits)

    def _suite_inputs(self, store, tenant, suite_ref, suite_hash):
        if (suite_ref is None) != (suite_hash is None):
            raise ValueError('comparison suite ref/hash must be paired')
        if suite_ref is None:
            return None, (), (), (), None
        suite = _load(store, tenant, suite_ref, suite_hash, OntologyPairedObligationSuiteV1)
        facts = {}; plans = ({}, {})
        for case in suite.cases:
            if not isinstance(case, DeterministicRuleObligationV1):
                continue
            facts[case.observed_facts_ref] = _load(store, tenant, case.observed_facts_ref,
                case.observed_facts_hash, ObservedFactsV1)
            for side, ref, digest in ((0,case.baseline_plan_ref,case.baseline_plan_hash),
                                     (1,case.successor_plan_ref,case.successor_plan_hash)):
                if ref is not None:
                    plan = _load(store, tenant, ref, digest, CompiledRulePlanV1)
                    if _plan_hash(plan) != digest:
                        raise ValueError('comparison rule plan original hash basis mismatch')
                    plans[side][ref] = plan
        competency = (_load(store, tenant, suite.competency_contract_ref, suite.competency_contract_hash,
            CompetencyContractV1, suite.competency_contract_content_hash) if suite.competency_contract_ref else None)
        return suite, tuple(facts.values()), tuple(plans[0].values()), tuple(plans[1].values()), competency

    def evaluate(self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
            baseline_ref: str, baseline_hash: str, successor_ref: str, successor_hash: str,
            suite_ref: str | None = None, suite_hash: str | None = None) -> PairedComparisonView:
        tenant = (organization_id, workspace_id)
        baseline, successor, snapshots, delta, subject, _ = self._pair(store, tenant,
            baseline_ref, baseline_hash, successor_ref, successor_hash)
        suite, facts, baseline_plans, successor_plans, competency = self._suite_inputs(store, tenant, suite_ref, suite_hash)
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != timezone.utc.utcoffset(now) or now < max(baseline.created_at, successor.created_at):
            raise ValueError('comparison time must be UTC and cannot predate its builds')
        earliest = [baseline.created_at,successor.created_at]
        if suite: earliest.append(suite.reviewed_at)
        earliest.extend(item.frozen_at for item in facts)
        earliest.extend(item.compiled_at for item in (*baseline_plans,*successor_plans))
        if now < max(earliest):
            raise ValueError('comparison cannot predate frozen evaluation inputs')
        units = []
        for name in _CODE_MODULES:
            module = import_module(name)
            code = Path(module.__file__).read_bytes()
            units.append(EvaluatorCodeUnitV1(code_ref='python-module:'+name,
                code_hash='sha256:'+sha256(code).hexdigest(), code_bytes=code))
        bundle = PairedEvaluatorBundleV1.create(organization_id=organization_id, workspace_id=workspace_id,
            code_units=tuple(units), policy=PairedEvaluationPolicyV1(required_case_ids=tuple(case.case_id for case in suite.cases) if suite else ()),
            frozen_at=now)
        result = OntologyPairedEvaluationService().evaluate_pair(baseline_manifest=baseline,
            successor_manifest=successor, baseline_snapshot=snapshots[0], successor_snapshot=snapshots[1],
            alignment_delta=delta, subject=subject, suite=suite, competency_contract=competency,
            evaluator_bundle=bundle, trusted_code_hashes={unit.code_ref:unit.code_hash for unit in units},
            observed_facts=facts, baseline_plans=baseline_plans, successor_plans=successor_plans, evaluated_at=now)
        artifacts = (bundle, *(run for run in (result.baseline_run,result.successor_run) if run is not None),
            *result.runtime_traces, result.report)
        store.put_many(artifacts)
        return self.get(store=store, organization_id=organization_id, workspace_id=workspace_id,
            comparison_ref=result.report.report_ref, comparison_hash=result.report.report_hash)

    def get(self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
            comparison_ref: str, comparison_hash: str | None = None) -> PairedComparisonView:
        tenant = (organization_id, workspace_id)
        report = _load(store, tenant, comparison_ref, comparison_hash, OntologyPairedEvaluationReportV1)
        subject = report.subject
        baseline, successor, snapshots, delta, actual_subject, audits = self._pair(store, tenant,
            subject.baseline_manifest.artifact_ref, subject.baseline_manifest.artifact_hash,
            subject.successor_manifest.artifact_ref, subject.successor_manifest.artifact_hash)
        if subject != actual_subject:
            raise ValueError('comparison report exact subject payload mismatch')
        bundle = _load(store, tenant, report.evaluator_ref, report.evaluator_hash, PairedEvaluatorBundleV1)
        suite, facts, baseline_plans, successor_plans, competency = self._suite_inputs(store, tenant, report.suite_ref, report.suite_hash)
        if suite:
            snap = snapshots[0]
            if (suite.source_identity_ref,suite.source_hash,suite.canonical_source_ir_ref,suite.canonical_source_ir_hash) != (
                    snap.source_identity_ref,snap.source_hash,snap.canonical_source_ir_ref,snap.canonical_source_ir_hash):
                raise ValueError('comparison suite source binding mismatch')
            for manifest in (baseline,successor):
                intent = _child(manifest,'graph_intent_brief')
                if (intent.artifact_ref,intent.artifact_hash) != (suite.graph_intent_ref,suite.graph_intent_hash):
                    raise ValueError('comparison suite GraphIntent mismatch')
            if bundle.policy.required_case_ids != tuple(case.case_id for case in suite.cases):
                raise ValueError('comparison policy does not cover exact reviewed cases')
        elif (report.baseline_run_ref is not None or report.verdict != 'INCONCLUSIVE' or
                report.reason_code != 'MISSING_REVIEWED_OBLIGATIONS'):
            raise ValueError('comparison without suite cannot claim measured behavior')
        runs = []
        for side, ref, digest in (('baseline',report.baseline_run_ref,report.baseline_run_hash),
                                 ('successor',report.successor_run_ref,report.successor_run_hash)):
            run = _load(store, tenant, ref, digest, OntologyPairedEvaluationRunV1) if ref else None
            if run and (run.side,run.suite_ref,run.suite_hash,run.evaluator_ref,run.evaluator_hash,run.evaluated_at) != (
                    side,report.suite_ref,report.suite_hash,report.evaluator_ref,report.evaluator_hash,report.evaluated_at):
                raise ValueError('comparison run input binding mismatch')
            runs.append(run)
        traces = {}
        for run in runs:
            for case in run.cases if run else ():
                if case.runtime_trace_ref:
                    traces[case.runtime_trace_ref] = _load(store,tenant,case.runtime_trace_ref,
                        case.runtime_trace_hash,RuntimeSemanticEvaluationBundleV1,case.runtime_trace_payload_hash)
        result = OntologyPairedEvaluationResultV1(baseline_run=runs[0],successor_run=runs[1],report=report,
            runtime_traces=tuple(traces.values()))
        return PairedComparisonView(result,bundle,suite,audits,(*facts,*baseline_plans,*successor_plans,
            *((competency,) if competency else ())))

    def evaluate_runtime_repair_pair(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        subject: RuntimeRepairPairV1, suite_pin, request_id: str, request_hash: str,
    ) -> RuntimeRepairPairedComparisonViewV1:
        """Persist a runtime-repair paired result through the existing artifact store.

        The evaluator runs before publication.  All inputs are resolved from
        governed artifact pins; this method does not synthesize seed-pair
        lineage or alter the legacy ``evaluate`` path.
        """
        tenant = (organization_id, workspace_id)
        if not request_id or not request_hash:
            raise ValueError("runtime repair comparison requires request identity")
        subject = RuntimeRepairPairV1.model_validate(subject.model_dump(mode="json"))
        if (subject.organization_id, subject.workspace_id) != tenant:
            raise PermissionError("runtime repair pair crosses tenant scope")
        baseline = _load(store, tenant, subject.predecessor_candidate_set.artifact_ref,
                         subject.predecessor_candidate_set.artifact_hash, NormativeSemanticCandidateSetV1)
        successor = _load(store, tenant, subject.successor_candidate_set.artifact_ref,
                          subject.successor_candidate_set.artifact_hash, NormativeSemanticCandidateSetV1)
        suite = None
        if suite_pin is not None:
            suite_pin = ArtifactPinV1.model_validate(suite_pin.model_dump(mode="json"))
            suite = _load(store, tenant, suite_pin.artifact_ref, suite_pin.artifact_hash, RuntimeRepairObligationSuiteV1)
        facts = []
        baseline_plans = []
        successor_plans = []
        if suite is not None:
            for case in suite.cases:
                facts.append(_load(store, tenant, case.observed_facts_pin.artifact_ref,
                    case.observed_facts_pin.artifact_hash, ObservedFactsV1))
                if case.baseline_plan_pin is not None:
                    baseline_plans.append(_load(store, tenant, case.baseline_plan_pin.artifact_ref,
                        case.baseline_plan_pin.artifact_hash, CompiledRulePlanV1))
                if case.successor_plan_pin is not None:
                    successor_plans.append(_load(store, tenant, case.successor_plan_pin.artifact_ref,
                        case.successor_plan_pin.artifact_hash, CompiledRulePlanV1))
        oracle_raw = _load(store, tenant, suite.oracle_manifest.artifact_ref,
                           suite.oracle_manifest.artifact_hash, None) if suite is not None else None
        oracle_payload = oracle_raw.model_dump(mode="json") if isinstance(oracle_raw, BaseModel) else (oracle_raw or {})
        children = oracle_payload.get("child_artifacts", ()) if isinstance(oracle_payload, dict) else ()
        def child(kind):
            values = [item for item in children if item.get("artifact_kind") == kind]
            return values[0] if len(values) == 1 else None
        evaluator_child = child("paired_evaluator_bundle")
        environment_child = child("runtime_repair_environment") or child("evaluation_environment")
        if evaluator_child is None and suite is not None and suite.oracle_manifest.artifact_kind == "paired-evaluator-bundle":
            evaluator_child = suite.oracle_manifest.model_dump(mode="json")
        if evaluator_child is None or environment_child is None:
            raise ValueError("runtime repair oracle manifest lacks evaluator/environment pins")
        evaluator = _load(store, tenant, evaluator_child["artifact_ref"], evaluator_child["artifact_hash"], PairedEvaluatorBundleV1)
        environment = __import__("contracts.ontology_building", fromlist=["ArtifactPinV1"]).ArtifactPinV1.model_validate(environment_child)
        result = OntologyPairedEvaluationService().evaluate_runtime_repair_pair(
            subject=subject, suite=suite, baseline_candidate_set=baseline,
            successor_candidate_set=successor, baseline_plans=tuple(baseline_plans),
            successor_plans=tuple(successor_plans), observed_facts=tuple(facts),
            evaluator_bundle=evaluator,
            trusted_code_hashes={unit.code_ref: unit.code_hash for unit in evaluator.code_units},
            environment_pin=environment, evaluated_at=self._clock(),
        )
        pair_pin = ArtifactPinV1(artifact_kind="runtime-repair-pair", artifact_ref=subject.pair_ref, artifact_hash=subject.pair_hash)
        evaluator_pin = ArtifactPinV1(artifact_kind="paired-evaluator-bundle", artifact_ref=evaluator.evaluator_ref, artifact_hash=evaluator.evaluator_hash)
        baseline_run_pin = ArtifactPinV1(artifact_kind="runtime-repair-run", artifact_ref=result.baseline_run.run_ref, artifact_hash=result.baseline_run.run_hash) if result.baseline_run else None
        successor_run_pin = ArtifactPinV1(artifact_kind="runtime-repair-run", artifact_ref=result.successor_run.run_ref, artifact_hash=result.successor_run.run_hash) if result.successor_run else None
        view_values = {
            "schema_version": "runtime_repair_paired_comparison_view.v1", "organization_id": tenant[0], "workspace_id": tenant[1],
            "subject": subject, "baseline_run_pin": baseline_run_pin, "successor_run_pin": successor_run_pin,
            "suite_pin": suite_pin, "evaluator_bundle_pin": evaluator_pin, "environment_pin": environment,
            "runtime_trace_pins": tuple(ArtifactPinV1(artifact_kind="runtime-semantic-evaluation", artifact_ref=trace.bundle_ref, artifact_hash=trace.bundle_hash) for trace in result.runtime_traces),
            "metric_results": tuple(result.report.metric_results), "verdict": result.report.verdict,
            "reason_code": result.report.reason_code,
            "audit_input_pins": (
                pair_pin, *(
                    item for item in (
                        subject.predecessor_manifest, subject.successor_manifest,
                        subject.predecessor_snapshot, subject.successor_snapshot,
                        subject.predecessor_candidate_set, subject.successor_candidate_set,
                        subject.accepted_review, subject.learning, subject.attribution,
                        subject.repair_input, subject.repair_impact, subject.repair_step_receipt,
                        subject.validation_contract, subject.source_identity,
                        subject.canonical_source_ir, subject.graph_intent,
                    )
                ),
                *((suite_pin,) if suite_pin else ()), evaluator_pin, environment,
            ),
            "evaluated_at": result.report.evaluated_at,
        }
        digest = content_hash(TypeAdapter(dict[str, object]).dump_python(view_values, mode="json"))
        view = RuntimeRepairPairedComparisonViewV1(**view_values, comparison_ref="runtime-repair-comparison:" + digest, comparison_hash="sha256:" + digest)
        artifacts = (subject, result.report, *(run for run in (result.baseline_run, result.successor_run) if run is not None), evaluator, *result.runtime_traces, view)
        store.put_many(tuple(artifact for artifact in artifacts if artifact is not None))
        return view

    def get_runtime_repair_pair(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        comparison_pin,
    ) -> RuntimeRepairPairedComparisonViewV1:
        """Cold-read a stored runtime-repair view without invoking evaluation."""
        pin = ArtifactPinV1.model_validate(comparison_pin.model_dump(mode="json"))
        if pin.artifact_kind != "runtime-repair-comparison":
            raise ValueError("runtime repair comparison pin kind mismatch")
        view = _load(store, (organization_id, workspace_id), pin.artifact_ref,
                      pin.artifact_hash, RuntimeRepairPairedComparisonViewV1)
        def verify_raw(item):
            raw = store.get_raw(organization_id=organization_id, workspace_id=workspace_id,
                                artifact_ref=item.artifact_ref)
            if raw is None or raw["artifact_hash"] != item.artifact_hash:
                raise ValueError("runtime repair closure pin mismatch")
        for item in (view.subject.predecessor_manifest, view.subject.successor_manifest,
                     view.subject.predecessor_snapshot, view.subject.successor_snapshot,
                     view.subject.predecessor_candidate_set, view.subject.successor_candidate_set,
                     view.subject.accepted_review, view.subject.learning, view.subject.attribution,
                     view.subject.repair_input, view.subject.repair_impact, view.subject.repair_step_receipt,
                     view.subject.validation_contract, view.subject.source_identity,
                     view.subject.canonical_source_ir, view.subject.graph_intent,
                     view.evaluator_bundle_pin, view.environment_pin, *view.runtime_trace_pins):
            verify_raw(item)
        return view


__all__ = ['OntologyPairedComparisonService','PairedComparisonView']
