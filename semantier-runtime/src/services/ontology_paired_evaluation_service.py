"""Pure same-input paired evaluation over exact governed ontology artifacts."""
from __future__ import annotations

from contracts.ontology_building import ArtifactPinV1, CompetencyContractV1, ExploratoryGraphSnapshotV1, OntologyBuildManifestV1
from contracts.ontology_paired_evaluation import (
    ArtifactPayloadPinV1, DeterministicRuleObligationV1, EvidenceTraceObligationV1,
    ObservedFactsV1, OntologyPairedEvaluationReportV1, OntologyPairedEvaluationResultV1,
    OntologyPairedEvaluationRunV1, OntologyPairedObligationSuiteV1, PairedCaseResultV1,
    PairedEvaluationSubjectV1, PairedEvaluatorBundleV1,
)
from contracts.ontology_seed_alignment import SourceGroundedSeedDeltaV1
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.runtime_learning_paired_evaluation import (
    RuntimeRepairMetricResultV1,
    RuntimeRepairObligationSuiteV1,
    RuntimeRepairPairV1,
    RuntimeRepairPairedEvaluationReportV1,
    RuntimeRepairPairedEvaluationResultV1,
    RuntimeRepairPairedEvaluationRunV1,
)
from eos.artifact_hashing import content_hash
from services.deterministic_rule_evaluator import DeterministicRuleEvaluator


def _exact(pin: ArtifactPayloadPinV1, artifact, ref_field: str, hash_field: str) -> None:
    if (pin.artifact_ref, pin.artifact_hash, pin.payload_hash) != (
        getattr(artifact, ref_field), getattr(artifact, hash_field),
        "sha256:" + content_hash(artifact.model_dump(mode="json")),
    ):
        raise ValueError("artifact payload pin mismatch")


def _plan_hash(plan: CompiledRulePlanV1) -> str:
    basis = {
        "candidate_set_ref": plan.candidate_set_ref, "candidate_set_hash": plan.candidate_set_hash,
        "validation_contract_ref": plan.validation_contract_ref,
        "validation_contract_hash": plan.validation_contract_hash,
        "compiler_ref": plan.compiler_ref, "compiler_hash": plan.compiler_hash,
        "semantic_profile": plan.semantic_profile,
        "rules": [rule.model_dump(mode="json") for rule in plan.rules],
    }
    if plan.precedences:
        basis["precedences"] = [item.model_dump(mode="json") for item in plan.precedences]
    timestamped = "sha256:" + content_hash(dict(basis, compiled_at=plan.compiled_at.isoformat()))
    if plan.plan_hash == timestamped:
        return timestamped
    # Historical plans retain their original identity scheme. The caller also
    # verifies the exact payload pin, including its production timestamp.
    return "sha256:" + content_hash(basis)


class OntologyPairedEvaluationService:
    @staticmethod
    def _runtime_address(cls, *, prefix: str, values: dict[str, object], ref_field: str, hash_field: str):
        probe = cls.model_construct(**{**values, ref_field: "pending", hash_field: "sha256:" + "0" * 64})
        digest = content_hash(probe.model_dump(mode="json", exclude={ref_field, hash_field}))
        return cls(**{**values, ref_field: f"{prefix}:{digest}", hash_field: f"sha256:{digest}"})

    @staticmethod
    def _runtime_pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)

    def evaluate_runtime_repair_pair(
        self, *, subject: RuntimeRepairPairV1,
        suite: RuntimeRepairObligationSuiteV1 | None,
        baseline_candidate_set: NormativeSemanticCandidateSetV1,
        successor_candidate_set: NormativeSemanticCandidateSetV1,
        baseline_plans: tuple[CompiledRulePlanV1, ...],
        successor_plans: tuple[CompiledRulePlanV1, ...],
        observed_facts: tuple[ObservedFactsV1, ...],
        evaluator_bundle: PairedEvaluatorBundleV1,
        trusted_code_hashes: dict[str, str],
        environment_pin: ArtifactPinV1,
        evaluated_at,
    ) -> RuntimeRepairPairedEvaluationResultV1:
        """Evaluate a repair pair without persistence or side effects.

        Runtime-repair evaluation deliberately has a separate owner from the
        legacy seed-pair path.  Inputs are validated before the deterministic
        evaluator is called, and missing reviewed obligations remain explicit
        INCONCLUSIVE/UNMEASURED output.
        """
        subject = RuntimeRepairPairV1.model_validate(subject.model_dump(mode="json"))
        baseline_candidate_set = NormativeSemanticCandidateSetV1.model_validate(
            baseline_candidate_set.model_dump(mode="json")
        )
        successor_candidate_set = NormativeSemanticCandidateSetV1.model_validate(
            successor_candidate_set.model_dump(mode="json")
        )
        evaluator_bundle = PairedEvaluatorBundleV1.model_validate(
            evaluator_bundle.model_dump(mode="json")
        )
        baseline_plans = tuple(CompiledRulePlanV1.model_validate(item.model_dump(mode="json")) for item in baseline_plans)
        successor_plans = tuple(CompiledRulePlanV1.model_validate(item.model_dump(mode="json")) for item in successor_plans)
        observed_facts = tuple(ObservedFactsV1.model_validate(item.model_dump(mode="json")) for item in observed_facts)
        if (baseline_candidate_set.organization_id, baseline_candidate_set.workspace_id) != (
            subject.organization_id, subject.workspace_id
        ) or (successor_candidate_set.organization_id, successor_candidate_set.workspace_id) != (
            subject.organization_id, subject.workspace_id
        ) or (evaluator_bundle.organization_id, evaluator_bundle.workspace_id) != (
            subject.organization_id, subject.workspace_id
        ):
            raise PermissionError("runtime repair evaluation input crosses tenant scope")
        if (subject.predecessor_candidate_set.artifact_ref, subject.predecessor_candidate_set.artifact_hash) != (
            baseline_candidate_set.candidate_set_ref, baseline_candidate_set.candidate_set_hash
        ) or (subject.successor_candidate_set.artifact_ref, subject.successor_candidate_set.artifact_hash) != (
            successor_candidate_set.candidate_set_ref, successor_candidate_set.candidate_set_hash
        ):
            raise ValueError("runtime repair pair does not bind exact candidate sets")
        if (successor_candidate_set.predecessor_candidate_set_ref,
            successor_candidate_set.predecessor_candidate_set_hash) != (
                baseline_candidate_set.candidate_set_ref, baseline_candidate_set.candidate_set_hash
            ):
            raise ValueError("successor candidate set does not bind predecessor candidate set")
        if successor_candidate_set.triggering_ref != subject.repair_input.artifact_ref:
            raise ValueError("successor candidate set does not bind exact repair input")
        source_values = (
            (baseline_candidate_set.source_identity_ref, baseline_candidate_set.source_hash,
             baseline_candidate_set.canonical_source_ir_ref, baseline_candidate_set.canonical_source_ir_hash),
            (successor_candidate_set.source_identity_ref, successor_candidate_set.source_hash,
             successor_candidate_set.canonical_source_ir_ref, successor_candidate_set.canonical_source_ir_hash),
        )
        if source_values[0] != source_values[1] or source_values[0] != (
            subject.source_identity.artifact_ref, subject.source_identity.artifact_hash,
            subject.canonical_source_ir.artifact_ref, subject.canonical_source_ir.artifact_hash,
        ):
            raise ValueError("runtime repair pair does not preserve source/IR basis")
        if (evaluator_bundle.model_dump(mode="json").get("code_units") and
            {unit.code_ref: unit.code_hash for unit in evaluator_bundle.code_units} != trusted_code_hashes):
            raise ValueError("evaluator bundle does not match trusted executing code bundle")
        if not environment_pin.artifact_ref or not environment_pin.artifact_hash:
            raise ValueError("runtime repair evaluation requires an environment pin")
        tenant = (subject.organization_id, subject.workspace_id)
        facts_by_ref = {item.facts_ref: item for item in observed_facts}
        plans_by_side = {
            "baseline": {item.plan_ref: item for item in baseline_plans},
            "successor": {item.plan_ref: item for item in successor_plans},
        }

        def unmeasured_metrics(reason: str):
            if suite is None:
                return ()
            return tuple(RuntimeRepairMetricResultV1(
                schema_version="runtime_repair_metric_result.v1",
                case_id=case.case_id, dimension=case.dimension,
                expected_outcome_pin=case.expected_outcome_pin,
                baseline_observed_pin=None, successor_observed_pin=None,
                baseline_outcome="UNMEASURED", successor_outcome="UNMEASURED",
                reason_code=reason,
            ) for case in suite.cases)

        if suite is not None and (suite.organization_id, suite.workspace_id) != tenant:
            raise PermissionError("runtime repair obligation suite crosses tenant scope")
        if suite is not None and (
            (suite.source_identity.artifact_ref, suite.source_identity.artifact_hash,
             suite.canonical_source_ir.artifact_ref, suite.canonical_source_ir.artifact_hash,
             suite.graph_intent.artifact_ref, suite.graph_intent.artifact_hash)
            != (subject.source_identity.artifact_ref, subject.source_identity.artifact_hash,
                subject.canonical_source_ir.artifact_ref, subject.canonical_source_ir.artifact_hash,
                subject.graph_intent.artifact_ref, subject.graph_intent.artifact_hash)
        ):
            raise ValueError("runtime repair obligation suite does not bind pair source/intent")
        required_ids = {
            "cq.finance_approval_required", "cq.cancellable_usage_excluded",
            "cq.emergency_exception_boundary", "cq.request_order_identity",
            "cq.emergency_exception_ceiling_25001",
        }
        if suite is None:
            reason = "MISSING_REVIEWED_OBLIGATIONS"
        elif not required_ids.issubset({case.case_id for case in suite.cases}):
            reason = "MISSING_REQUIRED_OBLIGATION_CASE"
        elif not suite.cases:
            reason = "MISSING_REVIEWED_OBLIGATIONS"
        else:
            reason = None

        traces = []
        metrics = []
        if reason is None:
            for case in suite.cases:
                baseline_plan = (plans_by_side["baseline"].get(case.baseline_plan_pin.artifact_ref)
                                 if case.baseline_plan_pin is not None else None)
                successor_plan = (plans_by_side["successor"].get(case.successor_plan_pin.artifact_ref)
                                  if case.successor_plan_pin is not None else None)
                facts = facts_by_ref.get(case.observed_facts_pin.artifact_ref)
                if (baseline_plan is None or successor_plan is None or facts is None or
                    case.baseline_plan_pin is None or case.successor_plan_pin is None or
                    baseline_plan.plan_hash != case.baseline_plan_pin.artifact_hash or
                    successor_plan.plan_hash != case.successor_plan_pin.artifact_hash or
                    facts.facts_hash != case.observed_facts_pin.artifact_hash):
                    metrics.append(RuntimeRepairMetricResultV1(
                        schema_version="runtime_repair_metric_result.v1", case_id=case.case_id,
                        dimension=case.dimension, expected_outcome_pin=case.expected_outcome_pin,
                        baseline_observed_pin=None, successor_observed_pin=None,
                        baseline_outcome="UNMEASURED", successor_outcome="UNMEASURED",
                        reason_code="MISSING_EXECUTABLE_INPUT"))
                    continue
                baseline_trace = DeterministicRuleEvaluator().evaluate(
                    plan=baseline_plan, execution_epoch_ref=f"runtime-repair:{subject.pair_ref}:baseline:{case.case_id}",
                    observed_facts=facts.facts, target_evidence_refs=facts.evidence_refs, evaluated_at=evaluated_at)
                successor_trace = DeterministicRuleEvaluator().evaluate(
                    plan=successor_plan, execution_epoch_ref=f"runtime-repair:{subject.pair_ref}:successor:{case.case_id}",
                    observed_facts=facts.facts, target_evidence_refs=facts.evidence_refs, evaluated_at=evaluated_at)
                traces.extend((baseline_trace, successor_trace))
                baseline_outcome = "UNMEASURED" if baseline_trace.unsupported_semantics else (
                    "PASS" if baseline_trace.result == case.expected_outcome else "FAIL")
                successor_outcome = "UNMEASURED" if successor_trace.unsupported_semantics else (
                    "PASS" if successor_trace.result == case.expected_outcome else "FAIL")
                metrics.append(RuntimeRepairMetricResultV1(
                    schema_version="runtime_repair_metric_result.v1", case_id=case.case_id,
                    dimension=case.dimension, expected_outcome_pin=case.expected_outcome_pin,
                    baseline_observed_pin=self._runtime_pin("runtime-semantic-evaluation", baseline_trace.bundle_ref, baseline_trace.bundle_hash),
                    successor_observed_pin=self._runtime_pin("runtime-semantic-evaluation", successor_trace.bundle_ref, successor_trace.bundle_hash),
                    baseline_outcome=baseline_outcome, successor_outcome=successor_outcome,
                    reason_code="UNSUPPORTED_SEMANTICS" if baseline_outcome == "UNMEASURED" or successor_outcome == "UNMEASURED" else "EVALUATED"))
            if any(item.baseline_outcome == "UNMEASURED" or item.successor_outcome == "UNMEASURED" for item in metrics):
                reason, verdict = "UNMEASURED_REQUIRED_CASE", "INCONCLUSIVE"
            elif any(item.baseline_outcome == "PASS" and item.successor_outcome == "FAIL" for item in metrics):
                reason, verdict = "REGRESSION", "REGRESSED"
            elif any(item.baseline_outcome == "FAIL" and item.successor_outcome == "PASS" for item in metrics):
                reason, verdict = "REVIEWED_CASE_IMPROVED", "IMPROVED"
            else:
                reason, verdict = "COMPARABLE_CASES_UNCHANGED", "CHANGED"
        else:
            verdict = "INCONCLUSIVE"
            metrics = list(unmeasured_metrics(reason))

        suite_pin = ArtifactPinV1(artifact_kind="runtime-repair-obligation-suite", artifact_ref=suite.suite_ref, artifact_hash=suite.suite_hash) if suite else None
        pair_pin = ArtifactPinV1(artifact_kind="runtime-repair-pair", artifact_ref=subject.pair_ref, artifact_hash=subject.pair_hash)
        evaluator_pin = ArtifactPinV1(artifact_kind="paired-evaluator-bundle", artifact_ref=evaluator_bundle.evaluator_ref, artifact_hash=evaluator_bundle.evaluator_hash)
        def make_run(side, candidate, outcomes):
            pins = tuple(self._runtime_pin("runtime-repair-metric-result", f"{subject.pair_ref}:{side}:{item.case_id}", "sha256:" + content_hash(item.model_dump(mode="json"))) for item in outcomes)
            traces_for_side = tuple(trace for index, trace in enumerate(traces) if index % 2 == (0 if side == "baseline" else 1))
            return self._runtime_address(RuntimeRepairPairedEvaluationRunV1, prefix="runtime-repair-run", ref_field="run_ref", hash_field="run_hash", values={
                "schema_version":"runtime_repair_paired_evaluation_run.v1", "organization_id":tenant[0], "workspace_id":tenant[1], "side":side,
                "pair_pin":pair_pin, "candidate_set_pin":ArtifactPinV1(artifact_kind="normative-candidate-set", artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash),
                "suite_pin":suite_pin or pair_pin, "evaluator_bundle_pin":evaluator_pin, "environment_pin":environment_pin,
                "case_result_pins":pins, "runtime_trace_pins":tuple(self._runtime_pin("runtime-semantic-evaluation", trace.bundle_ref, trace.bundle_hash) for trace in traces_for_side), "evaluated_at":evaluated_at})
        baseline_run = successor_run = None
        if suite is not None:
            baseline_run = make_run("baseline", baseline_candidate_set, metrics)
            successor_run = make_run("successor", successor_candidate_set, metrics)
        report_values = {"schema_version":"runtime_repair_paired_evaluation_report.v1", "organization_id":tenant[0], "workspace_id":tenant[1], "pair_pin":pair_pin, "suite_pin":suite_pin, "evaluator_bundle_pin":evaluator_pin, "environment_pin":environment_pin, "baseline_run_pin":ArtifactPinV1(artifact_kind="runtime-repair-run", artifact_ref=baseline_run.run_ref, artifact_hash=baseline_run.run_hash) if baseline_run else None, "successor_run_pin":ArtifactPinV1(artifact_kind="runtime-repair-run", artifact_ref=successor_run.run_ref, artifact_hash=successor_run.run_hash) if successor_run else None, "metric_results":tuple(metrics), "verdict":verdict, "reason_code":reason, "evaluated_at":evaluated_at}
        report = self._runtime_address(RuntimeRepairPairedEvaluationReportV1, prefix="runtime-repair-report", ref_field="report_ref", hash_field="report_hash", values=report_values)
        result_values = {
            "schema_version": "runtime_repair_paired_evaluation_result.v1",
            "organization_id": tenant[0], "workspace_id": tenant[1],
            "baseline_run": baseline_run, "successor_run": successor_run,
            "report": report, "runtime_traces": tuple(traces),
        }
        return self._runtime_address(
            RuntimeRepairPairedEvaluationResultV1,
            prefix="runtime-repair-result", ref_field="result_ref", hash_field="result_hash",
            values=result_values,
        )

    def evaluate_pair(
        self, *, baseline_manifest: OntologyBuildManifestV1,
        successor_manifest: OntologyBuildManifestV1,
        baseline_snapshot: ExploratoryGraphSnapshotV1,
        successor_snapshot: ExploratoryGraphSnapshotV1,
        alignment_delta: SourceGroundedSeedDeltaV1 | None,
        subject: PairedEvaluationSubjectV1,
        suite: OntologyPairedObligationSuiteV1 | None,
        competency_contract: CompetencyContractV1 | None = None,
        evaluator_bundle: PairedEvaluatorBundleV1,
        trusted_code_hashes: dict[str, str],
        observed_facts: tuple[ObservedFactsV1, ...],
        baseline_plans: tuple[CompiledRulePlanV1, ...],
        successor_plans: tuple[CompiledRulePlanV1, ...],
        evaluated_at,
        allow_ancestor_lineage: bool = False,
    ) -> OntologyPairedEvaluationResultV1:
        baseline_manifest = OntologyBuildManifestV1.model_validate(baseline_manifest.model_dump(mode="json"))
        successor_manifest = OntologyBuildManifestV1.model_validate(successor_manifest.model_dump(mode="json"))
        baseline_snapshot = ExploratoryGraphSnapshotV1.model_validate(baseline_snapshot.model_dump(mode="json"))
        successor_snapshot = ExploratoryGraphSnapshotV1.model_validate(successor_snapshot.model_dump(mode="json"))
        subject = PairedEvaluationSubjectV1.model_validate(subject.model_dump(mode="json"))
        evaluator_bundle = PairedEvaluatorBundleV1.model_validate(evaluator_bundle.model_dump(mode="json"))
        suite = (OntologyPairedObligationSuiteV1.model_validate(suite.model_dump(mode="json"))
                 if suite is not None else None)
        alignment_delta = (SourceGroundedSeedDeltaV1.model_validate(alignment_delta.model_dump(mode="json"))
                           if alignment_delta is not None else None)
        observed_facts = tuple(ObservedFactsV1.model_validate(item.model_dump(mode="json"))
                               for item in observed_facts)
        baseline_plans = tuple(CompiledRulePlanV1.model_validate(item.model_dump(mode="json"))
                               for item in baseline_plans)
        successor_plans = tuple(CompiledRulePlanV1.model_validate(item.model_dump(mode="json"))
                                for item in successor_plans)
        tenant = (baseline_manifest.organization_id, baseline_manifest.workspace_id)
        supplied = (successor_manifest, baseline_snapshot, successor_snapshot, evaluator_bundle)
        if any((item.organization_id, item.workspace_id) != tenant for item in supplied):
            raise PermissionError("paired evaluation artifacts cross tenant scope")
        _exact(subject.baseline_manifest, baseline_manifest, "build_ref", "build_hash")
        _exact(subject.successor_manifest, successor_manifest, "build_ref", "build_hash")
        _exact(subject.baseline_snapshot, baseline_snapshot, "snapshot_ref", "snapshot_hash")
        _exact(subject.successor_snapshot, successor_snapshot, "snapshot_ref", "snapshot_hash")
        if alignment_delta is None:
            if subject.alignment_delta is not None:
                raise ValueError("alignment delta payload pin is unresolved")
        else:
            if subject.alignment_delta is None:
                raise ValueError("alignment delta payload pin is missing")
            _exact(subject.alignment_delta, alignment_delta, "delta_ref", "delta_hash")
            if (alignment_delta.organization_id, alignment_delta.workspace_id) != tenant:
                raise PermissionError("alignment delta crosses tenant scope")
        if not allow_ancestor_lineage:
            if (successor_manifest.predecessor_build_ref, successor_manifest.predecessor_build_hash) != (
                    baseline_manifest.build_ref, baseline_manifest.build_hash):
                raise ValueError("successor manifest does not bind exact predecessor")
            if (successor_snapshot.predecessor_snapshot_ref, successor_snapshot.predecessor_snapshot_hash) != (
                    baseline_snapshot.snapshot_ref, baseline_snapshot.snapshot_hash):
                raise ValueError("successor snapshot does not bind exact predecessor")
        source_basis = (baseline_snapshot.source_identity_ref, baseline_snapshot.source_hash,
                        baseline_snapshot.canonical_source_ir_ref, baseline_snapshot.canonical_source_ir_hash)
        if source_basis != (successor_snapshot.source_identity_ref, successor_snapshot.source_hash,
                            successor_snapshot.canonical_source_ir_ref, successor_snapshot.canonical_source_ir_hash):
            raise ValueError("paired snapshots do not share a common source/IR basis")
        for manifest, snapshot in ((baseline_manifest, baseline_snapshot), (successor_manifest, successor_snapshot)):
            if snapshot.build_ref != manifest.build_ref:
                raise ValueError("snapshot does not belong to evaluated manifest build")
            pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == "exploratory_graph_snapshot"]
            if len(pins) != 1 or (pins[0].artifact_ref, pins[0].artifact_hash) != (
                    snapshot.snapshot_ref, snapshot.snapshot_hash):
                raise ValueError("manifest does not pin exact evaluated snapshot")
        supplied_code = {unit.code_ref: unit.code_hash for unit in evaluator_bundle.code_units}
        if supplied_code != trusted_code_hashes:
            raise ValueError("evaluator bundle does not match trusted executing code bundle")
        def child_pin(manifest, kind):
            found = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
            return found[0] if len(found) == 1 else None

        baseline_graph = child_pin(baseline_manifest, "graph_intent_brief")
        successor_graph = child_pin(successor_manifest, "graph_intent_brief")
        if baseline_graph is None or successor_graph is None or (
                baseline_graph.artifact_ref, baseline_graph.artifact_hash) != (
                successor_graph.artifact_ref, successor_graph.artifact_hash):
            raise ValueError("paired manifests do not share exact GraphIntent")
        if alignment_delta is not None:
            baseline_input = alignment_delta.inputs.baseline_manifest
            snapshot_input = alignment_delta.inputs.baseline_snapshot
            if (baseline_input.artifact_ref, baseline_input.artifact_hash, baseline_input.payload_hash) != (
                    subject.baseline_manifest.artifact_ref, subject.baseline_manifest.artifact_hash,
                    subject.baseline_manifest.payload_hash) or (
                    snapshot_input.artifact_ref, snapshot_input.artifact_hash, snapshot_input.payload_hash) != (
                    subject.baseline_snapshot.artifact_ref, subject.baseline_snapshot.artifact_hash,
                    subject.baseline_snapshot.payload_hash):
                raise ValueError("alignment delta does not bind exact paired baseline")
            if successor_snapshot.triggering_ref != alignment_delta.delta_ref:
                raise ValueError("successor snapshot is not triggered by the exact alignment delta")
        if suite is None:
            report = OntologyPairedEvaluationReportV1.create(
                organization_id=tenant[0], workspace_id=tenant[1], subject=subject,
                suite_ref=None, suite_hash=None, evaluator_ref=evaluator_bundle.evaluator_ref,
                evaluator_hash=evaluator_bundle.evaluator_hash, verdict="INCONCLUSIVE",
                reason_code="MISSING_REVIEWED_OBLIGATIONS", behavioral_measured_count=0,
                behavioral_total_count=0, evidence_measured_count=0, evidence_total_count=0,
                evaluated_at=evaluated_at,
            )
            return OntologyPairedEvaluationResultV1(baseline_run=None, successor_run=None, report=report)
        if (suite.organization_id, suite.workspace_id) != tenant or source_basis != (
            suite.source_identity_ref, suite.source_hash, suite.canonical_source_ir_ref,
            suite.canonical_source_ir_hash,
        ):
            raise ValueError("reviewed obligation suite does not bind the common source")
        for manifest in (baseline_manifest, successor_manifest):
            graph_pin = child_pin(manifest, "graph_intent_brief")
            if graph_pin is None or (graph_pin.artifact_ref, graph_pin.artifact_hash) != (
                    suite.graph_intent_ref, suite.graph_intent_hash):
                raise ValueError("paired manifests do not pin the reviewed suite GraphIntent")
        if suite.competency_contract_ref is not None:
            if competency_contract is None:
                raise ValueError("reviewed suite competency contract is unresolved")
            if "sha256:" + content_hash(competency_contract.model_dump(mode="json")) != suite.competency_contract_content_hash:
                raise ValueError("reviewed suite competency content hash mismatch")
            for manifest in (baseline_manifest, successor_manifest):
                pin = child_pin(manifest, "competency_contract")
                if pin is None or (pin.artifact_ref, pin.artifact_hash) != (
                        suite.competency_contract_ref, suite.competency_contract_hash):
                    raise ValueError("paired manifests do not pin the reviewed competency contract")
        elif competency_contract is not None:
            raise ValueError("unrequested competency contract cannot alter paired evaluation")
        case_ids = {case.case_id for case in suite.cases}
        if not set(evaluator_bundle.policy.required_case_ids) <= case_ids:
            raise ValueError("evaluator policy requires cases absent from reviewed suite")
        critical_ids = {case.case_id for case in suite.cases if case.critical}
        if not critical_ids <= set(evaluator_bundle.policy.required_case_ids):
            raise ValueError("all critical reviewed cases must be required by evaluator policy")
        if any(case.execution_kind not in evaluator_bundle.supported_execution_kinds for case in suite.cases):
            raise ValueError("reviewed suite requests an evaluator-unsupported execution kind")
        facts_by_ref = {item.facts_ref: item for item in observed_facts}
        if len(facts_by_ref) != len(observed_facts):
            raise ValueError("observed fact identities must be unique")
        plans = {
            "baseline": {item.plan_ref: item for item in baseline_plans},
            "successor": {item.plan_ref: item for item in successor_plans},
        }
        if any(len(plans[side]) != len(values) for side, values in (
                ("baseline", baseline_plans), ("successor", successor_plans))):
            raise ValueError("rule plan identities must be unique")
        for item in (*observed_facts, *baseline_plans, *successor_plans):
            if (item.organization_id, item.workspace_id) != tenant:
                raise PermissionError("evaluation input crosses tenant scope")

        def require_plan_inputs(manifest, plan):
            expected = (
                ("compiled_rule_plan", plan.plan_ref, plan.plan_hash),
                ("normative_semantic_candidate_set", plan.candidate_set_ref, plan.candidate_set_hash),
                ("validation_contract", plan.validation_contract_ref, plan.validation_contract_hash),
            )
            for kind, ref, digest in expected:
                pin = child_pin(manifest, kind)
                if pin is None or (pin.artifact_ref, pin.artifact_hash) != (ref, digest):
                    raise ValueError("compiled plan is not bound to evaluated manifest semantic inputs")

        def evaluate_side(side: str, snapshot: ExploratoryGraphSnapshotV1):
            results = []
            traces = []
            manifest = baseline_manifest if side == "baseline" else successor_manifest
            for case in suite.cases:
                if isinstance(case, EvidenceTraceObligationV1):
                    witnesses = tuple(edge for edge in snapshot.edges if (
                        edge.relation_type, edge.source_node_id, edge.target_node_id
                    ) == (case.relation_type, case.source_node_id, case.target_node_id))
                    present = bool(witnesses) and any(
                        set(case.expected_evidence_refs) <= set(edge.evidence_refs) for edge in witnesses
                    )
                    results.append(PairedCaseResultV1(
                        case_id=case.case_id, execution_kind=case.execution_kind,
                        measurement_scope="evidence_navigation", critical=case.critical,
                        status="MEASURED", expected_result=case.expected_present,
                        actual_result=present, passed=present == case.expected_present,
                        reason_code="TRACE_COMPARED", witness_refs=tuple(
                            dict.fromkeys([*(edge.edge_id for edge in witnesses),
                                           *(ref for edge in witnesses for ref in edge.evidence_refs)])
                        ),
                    ))
                    continue
                assert isinstance(case, DeterministicRuleObligationV1)
                plan_ref = case.baseline_plan_ref if side == "baseline" else case.successor_plan_ref
                plan_hash = case.baseline_plan_hash if side == "baseline" else case.successor_plan_hash
                plan = plans[side].get(plan_ref) if plan_ref else None
                facts = facts_by_ref.get(case.observed_facts_ref)
                reason = None
                if plan is None:
                    reason = "MISSING_EXECUTABLE_PLAN"
                elif plan.plan_hash != plan_hash or plan.plan_hash != _plan_hash(plan):
                    raise ValueError("compiled plan exact hash mismatch")
                else:
                    require_plan_inputs(manifest, plan)
                if reason is None and len(plan.rules) != 1:
                    reason = "PLAN_RULE_COUNT_UNSUPPORTED"
                elif reason is None and (plan.rules[0].rule_id != case.rule_id or case.competency_id not in plan.rules[0].competency_refs):
                    raise ValueError("compiled plan does not bind reviewed rule/competency")
                elif reason is None and not set(case.expected_evidence_refs) <= {
                    *plan.rules[0].evidence_refs,
                    *(ref for predicate in (*plan.rules[0].predicates,
                                             *plan.rules[0].exception_predicates)
                      for ref in predicate.evidence_refs),
                    *(ref for group in (plan.rules[0].exception_groups or ()) for ref in group.evidence_refs),
                }:
                    raise ValueError("compiled plan lacks reviewed expected evidence")
                if facts is None:
                    if plan is not None:
                        reason = "MISSING_OBSERVED_FACTS"
                elif facts.facts_hash != case.observed_facts_hash:
                    raise ValueError("observed facts exact hash mismatch")
                if reason:
                    results.append(PairedCaseResultV1(
                        case_id=case.case_id, execution_kind=case.execution_kind,
                        measurement_scope="behavioral", critical=case.critical,
                        status="UNMEASURED", expected_result=case.expected_outcome,
                        reason_code=reason,
                    ))
                    continue
                bundle = DeterministicRuleEvaluator().evaluate(
                    plan=plan, execution_epoch_ref="paired-evaluation:" + content_hash({
                        "side": side, "case": case.case_id, "snapshot": snapshot.snapshot_ref,
                        "suite": suite.suite_ref, "evaluator": evaluator_bundle.evaluator_ref,
                    }), observed_facts=facts.facts, target_evidence_refs=facts.evidence_refs,
                    evaluated_at=evaluated_at,
                )
                unsupported = bool(bundle.unsupported_semantics)
                traces.append(bundle)
                results.append(PairedCaseResultV1(
                    case_id=case.case_id, execution_kind=case.execution_kind,
                    measurement_scope="behavioral", critical=case.critical,
                    status="UNMEASURED" if unsupported else "MEASURED",
                    expected_result=case.expected_outcome, actual_result=bundle.result,
                    passed=None if unsupported else bundle.result == case.expected_outcome,
                    reason_code="UNSUPPORTED_SEMANTICS" if unsupported else bundle.justification.reason_code,
                    witness_refs=bundle.source_evidence_refs, runtime_trace_ref=bundle.bundle_ref,
                    runtime_trace_hash=bundle.bundle_hash,
                    runtime_trace_payload_hash="sha256:" + content_hash(bundle.model_dump(mode="json")),
                ))
            pin = subject.baseline_manifest if side == "baseline" else subject.successor_manifest
            snapshot_pin = subject.baseline_snapshot if side == "baseline" else subject.successor_snapshot
            return OntologyPairedEvaluationRunV1.create(
                organization_id=tenant[0], workspace_id=tenant[1], side=side,
                manifest=pin, snapshot=snapshot_pin, suite_ref=suite.suite_ref,
                suite_hash=suite.suite_hash, evaluator_ref=evaluator_bundle.evaluator_ref,
                evaluator_hash=evaluator_bundle.evaluator_hash, cases=tuple(results),
                evaluated_at=evaluated_at,
            ), traces

        baseline_run, baseline_traces = evaluate_side("baseline", baseline_snapshot)
        successor_run, successor_traces = evaluate_side("successor", successor_snapshot)
        paired = list(zip(baseline_run.cases, successor_run.cases, suite.cases))
        behavioral = [(a, b, case) for a, b, case in paired if a.measurement_scope == "behavioral"]
        evidence = [(a, b) for a, b, _ in paired if a.measurement_scope == "evidence_navigation"]
        required = set(evaluator_bundle.policy.required_case_ids)
        required_unmeasured = any(case.case_id in required and (
            a.status != "MEASURED" or b.status != "MEASURED") for a, b, case in paired)
        regressions = [case for a, b, case in behavioral if a.passed is True and b.passed is False and case.critical]
        improvements = [case for a, b, case in behavioral if a.passed is False and b.passed is True]
        measured_behavior = sum(a.status == b.status == "MEASURED" for a, b, _ in behavioral)
        if required_unmeasured or not behavioral or measured_behavior == 0:
            verdict, reason = "INCONCLUSIVE", "MISSING_COMPARABLE_BEHAVIOR"
        elif regressions:
            verdict, reason = "REGRESSED", "CRITICAL_BEHAVIOR_REGRESSION"
        elif improvements:
            verdict, reason = "IMPROVED", "REVIEWED_BEHAVIOR_FAIL_TO_PASS"
        else:
            verdict, reason = "CHANGED", "COMPARABLE_BEHAVIOR_UNCHANGED"
        report = OntologyPairedEvaluationReportV1.create(
            organization_id=tenant[0], workspace_id=tenant[1], subject=subject,
            suite_ref=suite.suite_ref, suite_hash=suite.suite_hash,
            evaluator_ref=evaluator_bundle.evaluator_ref, evaluator_hash=evaluator_bundle.evaluator_hash,
            baseline_run_ref=baseline_run.run_ref, baseline_run_hash=baseline_run.run_hash,
            successor_run_ref=successor_run.run_ref, successor_run_hash=successor_run.run_hash,
            verdict=verdict, reason_code=reason, behavioral_measured_count=measured_behavior,
            behavioral_total_count=len(behavioral),
            evidence_measured_count=sum(a.status == b.status == "MEASURED" for a, b in evidence),
            evidence_total_count=len(evidence), evaluated_at=evaluated_at,
        )
        return OntologyPairedEvaluationResultV1(
            baseline_run=baseline_run, successor_run=successor_run, report=report,
            runtime_traces=tuple((*baseline_traces, *successor_traces)),
        )


__all__ = ["OntologyPairedEvaluationService"]
