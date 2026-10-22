"""Paired comparison persistence for A0-to-A1 certification successors."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from pathlib import Path

from contracts.ontology_building import OntologyBuildManifestV1, ExploratoryGraphSnapshotV1
from contracts.ontology_paired_evaluation import (
    ArtifactPayloadPinV1,
    DeterministicRuleObligationV1,
    EvaluatorCodeUnitV1,
    ObservedFactsV1,
    OntologyPairedEvaluationReportV1,
    OntologyPairedEvaluationRunV1,
    OntologyPairedObligationSuiteV1,
    OntologyPairedEvaluationResultV1,
    PairedEvaluationPolicyV1,
    PairedEvaluationSubjectV1,
    PairedEvaluatorBundleV1,
)
from contracts.ontology_validation import CompiledRulePlanV1, RuntimeSemanticEvaluationBundleV1
from services.ontology_paired_comparison_service import PairedComparisonView
from services.ontology_paired_evaluation_service import OntologyPairedEvaluationService, _plan_hash
from storage.ontology_build_store import OntologyBuildStore


_CODE_MODULES = (
    "services.ontology_certification_comparison_service",
    "services.ontology_paired_evaluation_service",
    "services.deterministic_rule_evaluator",
    "contracts.ontology_paired_evaluation",
)


def _payload_pin(artifact, ref_field: str, hash_field: str) -> ArtifactPayloadPinV1:
    return ArtifactPayloadPinV1.of(artifact, ref_field, hash_field)


def _child(manifest: OntologyBuildManifestV1, kind: str):
    found = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
    if len(found) != 1:
        raise ValueError(f"certification comparison requires one exact manifest child: {kind}")
    return found[0]


class OntologyCertificationComparisonService:
    def __init__(self, *, clock=None):
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _load(store: OntologyBuildStore, tenant: tuple[str, str], ref: str, cls, digest: str | None = None):
        value = store.get(
            organization_id=tenant[0], workspace_id=tenant[1], artifact_ref=ref, contract_type=cls
        )
        if value is None:
            raise LookupError(f"missing pinned certification comparison artifact: {ref}")
        if digest is not None:
            _, hash_field = {
                OntologyBuildManifestV1: ("build_ref", "build_hash"),
                OntologyPairedObligationSuiteV1: ("suite_ref", "suite_hash"),
                ObservedFactsV1: ("facts_ref", "facts_hash"),
                CompiledRulePlanV1: ("plan_ref", "plan_hash"),
                PairedEvaluatorBundleV1: ("evaluator_ref", "evaluator_hash"),
                OntologyPairedEvaluationRunV1: ("run_ref", "run_hash"),
                OntologyPairedEvaluationReportV1: ("report_ref", "report_hash"),
                RuntimeSemanticEvaluationBundleV1: ("bundle_ref", "bundle_hash"),
            }[cls]
            if getattr(value, hash_field) != digest:
                raise ValueError("certification comparison exact hash mismatch")
        return value

    def _subject(
        self,
        *,
        store: OntologyBuildStore,
        tenant: tuple[str, str],
        baseline: OntologyBuildManifestV1,
        successor: OntologyBuildManifestV1,
    ) -> tuple[PairedEvaluationSubjectV1, ExploratoryGraphSnapshotV1, ExploratoryGraphSnapshotV1]:
        if successor.assurance_profile != "A1_INTERNAL_POLICY_MVL":
            raise ValueError("certification comparison successor must be A1")
        _child(successor, "ontology_certification_readiness")
        current = successor
        seen = {current.build_ref}
        found_baseline = False
        while current.predecessor_build_ref is not None:
            if (current.predecessor_build_ref, current.predecessor_build_hash) == (
                baseline.build_ref, baseline.build_hash
            ):
                found_baseline = True
                break
            predecessor = self._load(
                store,
                tenant,
                current.predecessor_build_ref,
                OntologyBuildManifestV1,
                current.predecessor_build_hash,
            )
            if predecessor.build_ref in seen:
                raise ValueError("certification comparison lineage contains a cycle")
            seen.add(predecessor.build_ref)
            current = predecessor
        if not found_baseline:
            raise ValueError("certification comparison baseline is not an exact ancestor")
        baseline_pin = _child(baseline, "exploratory_graph_snapshot")
        successor_pin = _child(successor, "exploratory_graph_snapshot")
        baseline_snapshot, _ = store.get_verified_exploratory_snapshot(
            organization_id=tenant[0], workspace_id=tenant[1], pin=baseline_pin
        )
        successor_snapshot, _ = store.get_verified_exploratory_snapshot(
            organization_id=tenant[0], workspace_id=tenant[1], pin=successor_pin
        )
        current_manifest = successor
        current_snapshot = successor_snapshot
        seen_snapshots = {current_snapshot.snapshot_ref}
        while current_manifest.build_ref != baseline.build_ref:
            if current_manifest.predecessor_build_ref is None:
                raise ValueError("certification comparison snapshot ancestry ended before baseline")
            predecessor_manifest = self._load(
                store,
                tenant,
                current_manifest.predecessor_build_ref,
                OntologyBuildManifestV1,
                current_manifest.predecessor_build_hash,
            )
            predecessor_pin = _child(predecessor_manifest, "exploratory_graph_snapshot")
            predecessor_snapshot, _ = store.get_verified_exploratory_snapshot(
                organization_id=tenant[0], workspace_id=tenant[1], pin=predecessor_pin
            )
            if (
                current_snapshot.predecessor_snapshot_ref,
                current_snapshot.predecessor_snapshot_hash,
            ) != (predecessor_snapshot.snapshot_ref, predecessor_snapshot.snapshot_hash):
                raise ValueError("certification comparison snapshot lineage is discontinuous")
            if predecessor_snapshot.snapshot_ref in seen_snapshots:
                raise ValueError("certification comparison snapshot lineage contains a cycle")
            seen_snapshots.add(predecessor_snapshot.snapshot_ref)
            current_manifest = predecessor_manifest
            current_snapshot = predecessor_snapshot
        if (current_snapshot.snapshot_ref, current_snapshot.snapshot_hash) != (
            baseline_snapshot.snapshot_ref, baseline_snapshot.snapshot_hash
        ):
            raise ValueError("certification comparison snapshot baseline mismatch")
        if (
            baseline_snapshot.source_identity_ref,
            baseline_snapshot.source_hash,
            baseline_snapshot.canonical_source_ir_ref,
            baseline_snapshot.canonical_source_ir_hash,
        ) != (
            successor_snapshot.source_identity_ref,
            successor_snapshot.source_hash,
            successor_snapshot.canonical_source_ir_ref,
            successor_snapshot.canonical_source_ir_hash,
        ):
            raise ValueError("certification comparison source/IR basis changed")
        baseline_intent = _child(baseline, "graph_intent_brief")
        successor_intent = _child(successor, "graph_intent_brief")
        if (baseline_intent.artifact_ref, baseline_intent.artifact_hash) != (
            successor_intent.artifact_ref, successor_intent.artifact_hash
        ):
            raise ValueError("certification comparison GraphIntent changed")
        return (
            PairedEvaluationSubjectV1(
                baseline_manifest=_payload_pin(baseline, "build_ref", "build_hash"),
                successor_manifest=_payload_pin(successor, "build_ref", "build_hash"),
                baseline_snapshot=_payload_pin(baseline_snapshot, "snapshot_ref", "snapshot_hash"),
                successor_snapshot=_payload_pin(successor_snapshot, "snapshot_ref", "snapshot_hash"),
                alignment_delta=None,
            ),
            baseline_snapshot,
            successor_snapshot,
        )

    def _suite_inputs(
        self,
        *,
        store: OntologyBuildStore,
        tenant: tuple[str, str],
        successor: OntologyBuildManifestV1,
        suite_ref: str | None,
        suite_hash: str | None,
    ):
        if (suite_ref is None) != (suite_hash is None):
            raise ValueError("comparison suite ref/hash must be paired")
        if suite_ref is None:
            pin = _child(successor, "ontology_paired_obligation_suite")
            suite_ref, suite_hash = pin.artifact_ref, pin.artifact_hash
        suite = self._load(store, tenant, suite_ref, OntologyPairedObligationSuiteV1, suite_hash)
        facts: dict[str, ObservedFactsV1] = {}
        successor_plans: dict[str, CompiledRulePlanV1] = {}
        for case in suite.cases:
            if not isinstance(case, DeterministicRuleObligationV1):
                continue
            facts[case.observed_facts_ref] = self._load(
                store, tenant, case.observed_facts_ref, ObservedFactsV1, case.observed_facts_hash
            )
            if case.baseline_plan_ref is not None:
                raise ValueError("A0 certification baseline must not acquire an executable plan")
            if case.successor_plan_ref is not None:
                plan = self._load(
                    store, tenant, case.successor_plan_ref, CompiledRulePlanV1, case.successor_plan_hash
                )
                if _plan_hash(plan) != case.successor_plan_hash:
                    raise ValueError("certification comparison plan semantic hash mismatch")
                successor_plans[plan.plan_ref] = plan
        return suite, tuple(facts.values()), tuple(successor_plans.values())

    def evaluate(
        self,
        *,
        store: OntologyBuildStore,
        organization_id: str,
        workspace_id: str,
        baseline_ref: str,
        baseline_hash: str,
        successor_ref: str,
        successor_hash: str,
        suite_ref: str | None = None,
        suite_hash: str | None = None,
    ) -> PairedComparisonView:
        tenant = (organization_id, workspace_id)
        baseline = self._load(store, tenant, baseline_ref, OntologyBuildManifestV1, baseline_hash)
        successor = self._load(store, tenant, successor_ref, OntologyBuildManifestV1, successor_hash)
        subject, baseline_snapshot, successor_snapshot = self._subject(
            store=store, tenant=tenant, baseline=baseline, successor=successor
        )
        suite, facts, successor_plans = self._suite_inputs(
            store=store, tenant=tenant, successor=successor,
            suite_ref=suite_ref, suite_hash=suite_hash,
        )
        intent = _child(baseline, "graph_intent_brief")
        if (
            suite.source_identity_ref, suite.source_hash,
            suite.canonical_source_ir_ref, suite.canonical_source_ir_hash,
            suite.graph_intent_ref, suite.graph_intent_hash,
        ) != (
            baseline_snapshot.source_identity_ref, baseline_snapshot.source_hash,
            baseline_snapshot.canonical_source_ir_ref, baseline_snapshot.canonical_source_ir_hash,
            intent.artifact_ref, intent.artifact_hash,
        ):
            raise ValueError("certification comparison suite basis mismatch")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != timezone.utc.utcoffset(now):
            raise ValueError("certification comparison clock must be UTC")
        units = []
        for name in _CODE_MODULES:
            module = import_module(name)
            code = Path(module.__file__).read_bytes()
            units.append(
                EvaluatorCodeUnitV1(
                    code_ref="python-module:" + name,
                    code_hash="sha256:" + sha256(code).hexdigest(),
                    code_bytes=code,
                )
            )
        policy = PairedEvaluationPolicyV1(required_case_ids=tuple(case.case_id for case in suite.cases))
        bundle = PairedEvaluatorBundleV1.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            code_units=tuple(units),
            policy=policy,
            frozen_at=now,
        )
        result = OntologyPairedEvaluationService().evaluate_pair(
            baseline_manifest=baseline,
            successor_manifest=successor,
            baseline_snapshot=baseline_snapshot,
            successor_snapshot=successor_snapshot,
            alignment_delta=None,
            subject=subject,
            suite=suite,
            competency_contract=None,
            evaluator_bundle=bundle,
            trusted_code_hashes={unit.code_ref: unit.code_hash for unit in units},
            observed_facts=facts,
            baseline_plans=(),
            successor_plans=successor_plans,
            evaluated_at=now,
            allow_ancestor_lineage=True,
        )
        store.put_many(
            (
                bundle,
                result.baseline_run,
                result.successor_run,
                *result.runtime_traces,
                result.report,
            )
        )
        return PairedComparisonView(
            result=result,
            evaluator_bundle=bundle,
            suite=suite,
            audits=(),
            inputs=(*facts, *successor_plans),
        )

    def get(
        self,
        *,
        store: OntologyBuildStore,
        organization_id: str,
        workspace_id: str,
        comparison_ref: str,
        comparison_hash: str | None = None,
    ) -> PairedComparisonView:
        tenant = (organization_id, workspace_id)
        report = self._load(
            store, tenant, comparison_ref, OntologyPairedEvaluationReportV1, comparison_hash
        )
        baseline = self._load(
            store, tenant, report.subject.baseline_manifest.artifact_ref,
            OntologyBuildManifestV1, report.subject.baseline_manifest.artifact_hash,
        )
        successor = self._load(
            store, tenant, report.subject.successor_manifest.artifact_ref,
            OntologyBuildManifestV1, report.subject.successor_manifest.artifact_hash,
        )
        actual_subject, _, _ = self._subject(
            store=store, tenant=tenant, baseline=baseline, successor=successor
        )
        if actual_subject != report.subject:
            raise ValueError("certification comparison subject mismatch")
        bundle = self._load(store, tenant, report.evaluator_ref, PairedEvaluatorBundleV1, report.evaluator_hash)
        suite, facts, plans = self._suite_inputs(
            store=store, tenant=tenant, successor=successor,
            suite_ref=report.suite_ref, suite_hash=report.suite_hash,
        )
        baseline_run = self._load(
            store, tenant, report.baseline_run_ref, OntologyPairedEvaluationRunV1, report.baseline_run_hash
        ) if report.baseline_run_ref else None
        successor_run = self._load(
            store, tenant, report.successor_run_ref, OntologyPairedEvaluationRunV1, report.successor_run_hash
        ) if report.successor_run_ref else None
        traces: dict[str, RuntimeSemanticEvaluationBundleV1] = {}
        for run in (baseline_run, successor_run):
            for case in run.cases if run else ():
                if case.runtime_trace_ref:
                    traces[case.runtime_trace_ref] = self._load(
                        store, tenant, case.runtime_trace_ref, RuntimeSemanticEvaluationBundleV1,
                        case.runtime_trace_hash,
                    )
        result = OntologyPairedEvaluationResultV1(
            baseline_run=baseline_run,
            successor_run=successor_run,
            report=report,
            runtime_traces=tuple(traces.values()),
        )
        return PairedComparisonView(
            result=result,
            evaluator_bundle=bundle,
            suite=suite,
            audits=(),
            inputs=(*facts, *plans),
        )


__all__ = ["OntologyCertificationComparisonService"]
