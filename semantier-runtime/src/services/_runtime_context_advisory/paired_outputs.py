"""Pure retained verification for paired-evaluation output artifacts."""

from __future__ import annotations

from typing import Mapping

from contracts.ontology_building import (
    ArtifactPinV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from contracts.ontology_paired_evaluation import (
    OntologyPairedEvaluationReportV1,
    OntologyPairedEvaluationRunV1,
    OntologyPairedObligationSuiteV1,
    PairedEvaluationPolicyV1,
    PairedEvaluatorBundleV1,
)
from contracts.ontology_validation import (
    RuntimeSemanticEvaluationBundleV1,
)
from contracts.runtime_context_reconstruction import CapturedSourceV1
from eos.artifact_hashing import content_hash

from .build import (
    verify_exploratory_graph_snapshot_v1,
    verify_ontology_build_manifest_v1,
)
from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _payload,
    _pin_key,
    _require_supplied_source,
    _validate_closure,
)
from .compiler import _plan_value
from .paired_inputs import (
    _declared_ancestry,
    _observed_facts_value,
    _resolve,
    _suite_case_profile,
    verify_ontology_paired_obligation_suite_v1,
)
from .trial_build import _resolve_payload_pin


_CODE_REFS = (
    "python-module:services.ontology_paired_comparison_service",
    "python-module:services.ontology_paired_evaluation_service",
    "python-module:services.deterministic_rule_evaluator",
    "python-module:contracts.ontology_paired_evaluation",
    "python-module:contracts.ontology_validation",
    "python-module:eos.artifact_hashing",
)
_CASE_IDS = (
    "emergency_boundary",
    "selected_proxy",
    "independent_low_value",
)


def _is_lower_hex_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=ref,
        artifact_hash=digest,
    )


def _model_payload(source: CapturedSourceV1, contract: type, label: str):
    source, payload = _payload(source)
    value = contract.model_validate(payload)
    if value.model_dump(mode="json") != payload:
        raise ValueError(f"retained {label} encoding mismatch")
    return value


def _verify_declared(source, closure, verifier) -> None:
    ancestry = _declared_ancestry(source, closure)
    verifier(source=source, closure=ancestry)


def _first_occurrence(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _predicate_projection(predicate) -> dict:
    return {
        "predicate_id": predicate.predicate_id,
        "operator": predicate.operator,
        "fact_ref": predicate.fact_ref,
        "expected_value": predicate.expected_value,
        "ir_node_ref": predicate.ir_node_ref,
        "evidence_refs": list(predicate.evidence_refs),
    }


def _expected_trace_summary(trace, rule, facts):
    if len(trace.candidate_rule_evaluations) != 1:
        raise ValueError("paired runtime trace requires one candidate rule evaluation")
    evaluation = trace.candidate_rule_evaluations[0]
    if evaluation.rule_id != rule.rule_id:
        raise ValueError("runtime trace rule identity mismatch")
    if len(evaluation.predicate_results) != len(rule.predicates) or len(
        evaluation.exception_results
    ) != len(rule.exception_predicates):
        raise ValueError("runtime trace predicate cardinality mismatch")
    for stored, predicate in zip(
        evaluation.predicate_results, rule.predicates, strict=True
    ):
        if {
            **_predicate_projection(predicate),
            "result": stored.result,
            "observed_value": facts.facts.get(predicate.fact_ref),
        } != stored.model_dump(mode="json"):
            raise ValueError("runtime trace predicate member mismatch")
    for stored, predicate in zip(
        evaluation.exception_results, rule.exception_predicates, strict=True
    ):
        if {
            **_predicate_projection(predicate),
            "result": stored.result,
            "observed_value": facts.facts.get(predicate.fact_ref),
        } != stored.model_dump(mode="json"):
            raise ValueError("runtime trace exception member mismatch")

    results = (*evaluation.predicate_results, *evaluation.exception_results)
    unsupported = tuple(
        sorted({item.operator for item in results if item.result == "UNSUPPORTED"})
    )
    unknown = tuple(
        sorted(
            {
                item.fact_ref
                for item in evaluation.predicate_results
                if item.result == "UNKNOWN"
            }
        )
    )
    satisfied = tuple(
        item.predicate_id
        for item in evaluation.predicate_results
        if item.result == "SATISFIED"
    )
    failed = tuple(
        item.predicate_id
        for item in evaluation.predicate_results
        if item.result == "FAILED"
    )
    if unsupported:
        result, reason = "UNKNOWN", "UNSUPPORTED_SEMANTICS"
    elif unknown:
        result, reason = "UNKNOWN", "MISSING_MATERIAL_INPUT"
    elif failed:
        result, reason = "NO_MATCH", "RULE_CONDITIONS_NOT_SATISFIED"
    elif evaluation.exception_results and all(
        item.result == "SATISFIED" for item in evaluation.exception_results
    ):
        result, reason = "NO_MATCH", "EXPLICIT_EXCEPTION_APPLIES"
    else:
        result, reason = "MATCH", "RULE_CONDITIONS_SATISFIED"
    rejection = (
        (reason,)
        if result == "NO_MATCH" and reason != "EXPLICIT_EXCEPTION_APPLIES"
        else ()
    )
    ir_refs = _first_occurrence(
        (
            rule.ir_node_ref,
            *(item.ir_node_ref for item in results),
        )
    )
    evidence_refs = _first_occurrence(
        (
            *rule.evidence_refs,
            *(ref for item in results for ref in item.evidence_refs),
        )
    )
    if (
        trace.result != result
        or evaluation.outcome != result
        or evaluation.rejection_reasons != rejection
        or trace.satisfied_predicates != satisfied
        or trace.failed_predicates != failed
        or trace.exception_checks != evaluation.exception_results
        or trace.unknown_inputs != unknown
        or trace.unsupported_semantics != unsupported
        or trace.competency_refs != rule.competency_refs
        or trace.ir_node_refs != ir_refs
        or trace.source_evidence_refs != evidence_refs
        or trace.justification.model_dump(mode="json")
        != {
            "schema_version": "runtime_justification.v1",
            "result": result,
            "reason_code": reason,
            "rule_refs": [rule.rule_id],
            "ir_node_refs": list(ir_refs),
            "competency_refs": list(rule.competency_refs),
            "source_evidence_refs": list(evidence_refs),
        }
    ):
        raise ValueError("runtime trace derived summary or justification mismatch")
    return unsupported


def _policy_value(source: CapturedSourceV1) -> PairedEvaluationPolicyV1:
    source, payload = _payload(source)
    policy = PairedEvaluationPolicyV1.model_validate(payload)
    if policy.model_dump(mode="json") != payload:
        raise ValueError("retained paired evaluation policy encoding mismatch")
    digest = policy.policy_hash.removeprefix("sha256:")
    _expect_identity(
        source,
        kind="paired_evaluation_policy",
        ref=f"paired-evaluation-policy:{digest}",
        digest=policy.policy_hash,
    )
    if policy.required_case_ids not in ((), _CASE_IDS):
        raise ValueError("paired evaluation policy case order is unsupported")
    return policy


def verify_paired_evaluation_policy_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify the exact schema-less policy member and leaf dependency shape."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _policy_value(source)
    return _expect_dependencies(source, (), closure)


def verify_paired_evaluator_bundle_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify the concrete six-module archive and its exact policy member."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    source, payload = _payload(source)
    bundle = PairedEvaluatorBundleV1.model_validate(payload)
    if bundle.model_dump(mode="json") != payload:
        raise ValueError("retained paired evaluator bundle encoding mismatch")
    _expect_identity(
        source,
        kind="paired_evaluator_bundle",
        ref=bundle.evaluator_ref,
        digest=bundle.evaluator_hash,
    )
    if tuple(unit.code_ref for unit in bundle.code_units) != _CODE_REFS:
        raise ValueError("paired evaluator archive module order mismatch")
    if bundle.supported_execution_kinds != ("evidence_trace", "deterministic_rule"):
        raise ValueError("paired evaluator supported-kind order mismatch")
    policy_pin = ArtifactPinV1(
        artifact_kind="paired_evaluation_policy",
        artifact_ref=(
            "paired-evaluation-policy:" + bundle.policy_hash.removeprefix("sha256:")
        ),
        artifact_hash=bundle.policy_hash,
    )
    dependencies = _expect_dependencies(source, (policy_pin,), closure)
    policy_source = closure[_pin_key(policy_pin)]
    verify_paired_evaluation_policy_v1(source=policy_source, closure=closure)
    if _policy_value(policy_source) != bundle.policy:
        raise ValueError("paired evaluator bundle policy member mismatch")
    return dependencies


def _trace_value(source, closure):
    source, payload = _payload(source)
    trace = RuntimeSemanticEvaluationBundleV1.model_validate(payload)
    if trace.model_dump(mode="json") != payload:
        raise ValueError("retained runtime semantic evaluation encoding mismatch")
    basis = {
        "execution_epoch_ref": trace.execution_epoch_ref,
        "plan_ref": trace.plan_ref,
        "plan_hash": trace.plan_hash,
        "result": trace.result,
        "observed_facts": trace.observed_facts,
        "target_evidence_refs": trace.target_evidence_refs,
        "candidate_rule_evaluations": [
            item.model_dump(mode="json") for item in trace.candidate_rule_evaluations
        ],
        "justification": trace.justification.model_dump(mode="json"),
    }
    digest = content_hash(basis)
    if (
        trace.bundle_ref != f"runtime-semantic-evaluation:{digest}"
        or trace.bundle_hash != f"sha256:{digest}"
    ):
        raise ValueError("runtime semantic evaluation native identity mismatch")
    _expect_identity(
        source,
        kind="runtime_semantic_evaluation_bundle",
        ref=trace.bundle_ref,
        digest=trace.bundle_hash,
    )
    plan_pin = _pin("compiled_rule_plan", trace.plan_ref, trace.plan_hash)
    facts_pins = tuple(
        pin
        for pin in source.dependencies
        if pin.artifact_kind == "paired_observed_facts"
    )
    if len(facts_pins) != 1:
        raise ValueError("runtime trace must declare one paired observed-facts input")
    dependencies = _expect_dependencies(source, (plan_pin, facts_pins[0]), closure)
    plan_source = _resolve(closure, plan_pin)
    plan_ancestry = _declared_ancestry(plan_source, closure)
    plan, _plan_dependencies = _plan_value(plan_source, plan_ancestry)
    facts_source = _resolve(closure, facts_pins[0])
    facts_ancestry = _declared_ancestry(facts_source, closure)
    facts, _case, _root, _facts_dependencies = _observed_facts_value(
        facts_source, facts_ancestry
    )
    if (
        (trace.organization_id, trace.workspace_id)
        != (plan.organization_id, plan.workspace_id)
        or (trace.organization_id, trace.workspace_id)
        != (facts.organization_id, facts.workspace_id)
        or trace.observed_facts != facts.facts
        or trace.target_evidence_refs != facts.evidence_refs
        or len(plan.rules) != 1
        or not trace.execution_epoch_ref.startswith("paired-evaluation:")
        or not _is_lower_hex_digest(
            trace.execution_epoch_ref.removeprefix("paired-evaluation:")
        )
    ):
        raise ValueError("runtime trace plan, facts, scope, or epoch mismatch")
    unsupported = _expected_trace_summary(trace, plan.rules[0], facts)
    return trace, plan, facts, unsupported, dependencies


def verify_runtime_semantic_evaluation_bundle_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one retained trace from stored members without reevaluating predicates."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _trace, _plan, _facts, _unsupported, dependencies = _trace_value(source, closure)
    return dependencies


def _manifest_plan_pin(manifest, side_cases, side):
    child_plans = tuple(
        pin
        for pin in manifest.child_artifacts
        if pin.artifact_kind == "compiled_rule_plan"
    )
    suite_plans = {
        (
            getattr(case, f"{side}_plan_ref"),
            getattr(case, f"{side}_plan_hash"),
        )
        for case in side_cases
    }
    if (
        len(child_plans) != 1
        or len(suite_plans) != 1
        or (child_plans[0].artifact_ref, child_plans[0].artifact_hash)
        != next(iter(suite_plans))
    ):
        raise ValueError("paired run manifest and suite side plan mismatch")
    return child_plans[0]


def _run_value(source, closure):
    source, payload = _payload(source)
    run = OntologyPairedEvaluationRunV1.model_validate(payload)
    if run.model_dump(mode="json") != payload:
        raise ValueError("retained paired evaluation run encoding mismatch")
    _expect_identity(
        source,
        kind="ontology_paired_evaluation_run",
        ref=run.run_ref,
        digest=run.run_hash,
    )
    manifest_source, manifest = _resolve_payload_pin(
        closure,
        run.manifest,
        kind="ontology_build_manifest",
        contract=OntologyBuildManifestV1,
        label="paired run manifest",
    )
    snapshot_source, snapshot = _resolve_payload_pin(
        closure,
        run.snapshot,
        kind="exploratory_graph_snapshot",
        contract=ExploratoryGraphSnapshotV1,
        label="paired run snapshot",
    )
    suite_pin = _pin("ontology_paired_obligation_suite", run.suite_ref, run.suite_hash)
    evaluator_pin = _pin(
        "paired_evaluator_bundle", run.evaluator_ref, run.evaluator_hash
    )
    suite_source = _resolve(closure, suite_pin)
    bundle_source = _resolve(closure, evaluator_pin)
    _verify_declared(manifest_source, closure, verify_ontology_build_manifest_v1)
    _verify_declared(snapshot_source, closure, verify_exploratory_graph_snapshot_v1)
    _verify_declared(suite_source, closure, verify_ontology_paired_obligation_suite_v1)
    _verify_declared(bundle_source, closure, verify_paired_evaluator_bundle_v1)
    suite = _model_payload(
        suite_source, OntologyPairedObligationSuiteV1, "paired obligation suite"
    )
    bundle = _model_payload(
        bundle_source, PairedEvaluatorBundleV1, "paired evaluator bundle"
    )
    suite_cases = _suite_case_profile(suite)
    if len(run.cases) != len(suite_cases):
        raise ValueError("paired run case cardinality mismatch")
    _manifest_plan_pin(manifest, suite_cases, run.side)
    snapshot_children = tuple(
        pin
        for pin in manifest.child_artifacts
        if pin.artifact_kind == "exploratory_graph_snapshot"
    )
    graph_children = tuple(
        pin
        for pin in manifest.child_artifacts
        if pin.artifact_kind == "graph_intent_brief"
    )
    if (
        len(snapshot_children) != 1
        or (snapshot_children[0].artifact_ref, snapshot_children[0].artifact_hash)
        != (snapshot.snapshot_ref, snapshot.snapshot_hash)
        or len(graph_children) != 1
        or (graph_children[0].artifact_ref, graph_children[0].artifact_hash)
        != (suite.graph_intent_ref, suite.graph_intent_hash)
        or (run.organization_id, run.workspace_id)
        != (manifest.organization_id, manifest.workspace_id)
        or (run.organization_id, run.workspace_id)
        != (snapshot.organization_id, snapshot.workspace_id)
        or (run.organization_id, run.workspace_id)
        != (suite.organization_id, suite.workspace_id)
        or (run.organization_id, run.workspace_id)
        != (bundle.organization_id, bundle.workspace_id)
        or run.evaluated_at != bundle.frozen_at
    ):
        raise ValueError("paired run subject, suite, or evaluator scope mismatch")

    trace_dependencies = []
    for case, expected in zip(run.cases, suite_cases, strict=True):
        if (
            case.case_id != expected.case_id
            or case.execution_kind != expected.execution_kind
            or case.measurement_scope != "behavioral"
            or case.critical != expected.critical
            or case.expected_result != expected.expected_outcome
        ):
            raise ValueError("paired run case does not match reviewed suite order")
        if case.runtime_trace_ref is None:
            if (
                case.execution_kind != "deterministic_rule"
                or case.status != "UNMEASURED"
                or case.reason_code
                not in {
                    "MISSING_EXECUTABLE_PLAN",
                    "MISSING_OBSERVED_FACTS",
                    "PLAN_RULE_COUNT_UNSUPPORTED",
                }
                or case.actual_result is not None
                or case.passed is not None
                or case.witness_refs
                or case.runtime_trace_hash is not None
                or case.runtime_trace_payload_hash is not None
            ):
                raise ValueError("unsupported deterministic no-trace producer shape")
            continue
        trace_pin = _pin(
            "runtime_semantic_evaluation_bundle",
            case.runtime_trace_ref,
            case.runtime_trace_hash,
        )
        trace_source = _resolve(closure, trace_pin)
        trace_ancestry = _declared_ancestry(trace_source, closure)
        trace, plan, facts, unsupported, _dependencies = _trace_value(
            trace_source, trace_ancestry
        )
        side_ref = getattr(expected, f"{run.side}_plan_ref")
        side_hash = getattr(expected, f"{run.side}_plan_hash")
        expected_epoch = "paired-evaluation:" + content_hash(
            {
                "side": run.side,
                "case": case.case_id,
                "snapshot": run.snapshot.artifact_ref,
                "suite": run.suite_ref,
                "evaluator": run.evaluator_ref,
            }
        )
        if (
            (plan.plan_ref, plan.plan_hash) != (side_ref, side_hash)
            or (facts.facts_ref, facts.facts_hash)
            != (expected.observed_facts_ref, expected.observed_facts_hash)
            or case.runtime_trace_payload_hash
            != "sha256:" + content_hash(trace.model_dump(mode="json"))
            or case.witness_refs != trace.source_evidence_refs
            or case.actual_result != trace.result
            or trace.execution_epoch_ref != expected_epoch
            or trace.evaluated_at != run.evaluated_at
        ):
            raise ValueError("paired run trace triple or member relationship mismatch")
        if unsupported:
            expected_shape = ("UNMEASURED", None, "UNSUPPORTED_SEMANTICS")
        else:
            expected_shape = (
                "MEASURED",
                trace.result == expected.expected_outcome,
                trace.justification.reason_code,
            )
        if (case.status, case.passed, case.reason_code) != expected_shape:
            raise ValueError("paired run trace result projection mismatch")
        trace_dependencies.append(trace_pin)

    dependencies = _expect_dependencies(
        source,
        (
            _pin(
                "ontology_build_manifest",
                run.manifest.artifact_ref,
                run.manifest.artifact_hash,
            ),
            _pin(
                "exploratory_graph_snapshot",
                run.snapshot.artifact_ref,
                run.snapshot.artifact_hash,
            ),
            suite_pin,
            evaluator_pin,
            *trace_dependencies,
        ),
        closure,
    )
    return run, manifest, snapshot, suite, bundle, dependencies


def verify_ontology_paired_evaluation_run_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one side's exact native run, retained traces, and input bindings."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _run, _manifest, _snapshot, _suite, _bundle, dependencies = _run_value(
        source, closure
    )
    return dependencies


def _report_verdict(report, policy, suite, baseline, successor):
    paired = tuple(zip(baseline.cases, successor.cases, suite.cases, strict=True))
    behavioral = tuple(
        (left, right, case)
        for left, right, case in paired
        if left.measurement_scope == "behavioral"
    )
    evidence = tuple(
        (left, right)
        for left, right, _case in paired
        if left.measurement_scope == "evidence_navigation"
    )
    if any(
        left.measurement_scope != right.measurement_scope
        for left, right, _case in paired
    ):
        raise ValueError("paired report run measurement scopes disagree")
    required = set(policy.required_case_ids)
    required_unmeasured = any(
        case.case_id in required
        and (left.status != "MEASURED" or right.status != "MEASURED")
        for left, right, case in paired
    )
    regressions = tuple(
        case
        for left, right, case in behavioral
        if left.passed is True and right.passed is False and case.critical
    )
    improvements = tuple(
        case
        for left, right, case in behavioral
        if left.passed is False and right.passed is True
    )
    measured_behavior = sum(
        left.status == right.status == "MEASURED" for left, right, _case in behavioral
    )
    measured_evidence = sum(
        left.status == right.status == "MEASURED" for left, right in evidence
    )
    if required_unmeasured or not behavioral or measured_behavior == 0:
        verdict, reason = "INCONCLUSIVE", "MISSING_COMPARABLE_BEHAVIOR"
    elif regressions:
        verdict, reason = "REGRESSED", "CRITICAL_BEHAVIOR_REGRESSION"
    elif improvements:
        verdict, reason = "IMPROVED", "REVIEWED_BEHAVIOR_FAIL_TO_PASS"
    else:
        verdict, reason = "CHANGED", "COMPARABLE_BEHAVIOR_UNCHANGED"
    actual = (
        report.behavioral_measured_count,
        report.behavioral_total_count,
        report.evidence_measured_count,
        report.evidence_total_count,
        report.verdict,
        report.reason_code,
    )
    expected = (
        measured_behavior,
        len(behavioral),
        measured_evidence,
        len(evidence),
        verdict,
        reason,
    )
    if actual != expected:
        raise ValueError("paired report count or verdict projection mismatch")


def verify_ontology_paired_evaluation_report_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify the exact report subject, runs, counts, and native verdict priority."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    source, payload = _payload(source)
    report = OntologyPairedEvaluationReportV1.model_validate(payload)
    if report.model_dump(mode="json") != payload:
        raise ValueError("retained paired evaluation report encoding mismatch")
    _expect_identity(
        source,
        kind="ontology_paired_evaluation_report",
        ref=report.report_ref,
        digest=report.report_hash,
    )
    if report.subject.alignment_delta is not None:
        raise ValueError("bounded paired report does not support alignment delta")
    subject_pins = (
        (
            report.subject.baseline_manifest,
            "ontology_build_manifest",
            OntologyBuildManifestV1,
            "baseline manifest",
        ),
        (
            report.subject.successor_manifest,
            "ontology_build_manifest",
            OntologyBuildManifestV1,
            "successor manifest",
        ),
        (
            report.subject.baseline_snapshot,
            "exploratory_graph_snapshot",
            ExploratoryGraphSnapshotV1,
            "baseline snapshot",
        ),
        (
            report.subject.successor_snapshot,
            "exploratory_graph_snapshot",
            ExploratoryGraphSnapshotV1,
            "successor snapshot",
        ),
    )
    resolved_subject = []
    dependencies = []
    for pin, kind, contract, label in subject_pins:
        retained, value = _resolve_payload_pin(
            closure, pin, kind=kind, contract=contract, label=label
        )
        verifier = (
            verify_ontology_build_manifest_v1
            if kind == "ontology_build_manifest"
            else verify_exploratory_graph_snapshot_v1
        )
        _verify_declared(retained, closure, verifier)
        resolved_subject.append(value)
        dependencies.append(_pin(kind, pin.artifact_ref, pin.artifact_hash))
    baseline_manifest, successor_manifest, baseline_snapshot, successor_snapshot = (
        resolved_subject
    )
    evaluator_pin = _pin(
        "paired_evaluator_bundle", report.evaluator_ref, report.evaluator_hash
    )
    evaluator_source = _resolve(closure, evaluator_pin)
    _verify_declared(evaluator_source, closure, verify_paired_evaluator_bundle_v1)
    evaluator = _model_payload(
        evaluator_source, PairedEvaluatorBundleV1, "paired evaluator bundle"
    )
    dependencies.append(evaluator_pin)
    scope = (report.organization_id, report.workspace_id)
    source_basis = (
        baseline_snapshot.source_identity_ref,
        baseline_snapshot.source_hash,
        baseline_snapshot.canonical_source_ir_ref,
        baseline_snapshot.canonical_source_ir_hash,
    )
    if (
        scope != (baseline_manifest.organization_id, baseline_manifest.workspace_id)
        or scope
        != (successor_manifest.organization_id, successor_manifest.workspace_id)
        or scope != (baseline_snapshot.organization_id, baseline_snapshot.workspace_id)
        or scope
        != (successor_snapshot.organization_id, successor_snapshot.workspace_id)
        or scope != (evaluator.organization_id, evaluator.workspace_id)
        or report.evaluated_at != evaluator.frozen_at
        or source_basis
        != (
            successor_snapshot.source_identity_ref,
            successor_snapshot.source_hash,
            successor_snapshot.canonical_source_ir_ref,
            successor_snapshot.canonical_source_ir_hash,
        )
        or (
            successor_manifest.predecessor_build_ref,
            successor_manifest.predecessor_build_hash,
        )
        != (baseline_manifest.build_ref, baseline_manifest.build_hash)
        or (
            successor_snapshot.predecessor_snapshot_ref,
            successor_snapshot.predecessor_snapshot_hash,
        )
        != (baseline_snapshot.snapshot_ref, baseline_snapshot.snapshot_hash)
    ):
        raise ValueError("paired report subject lineage or scope mismatch")
    for manifest, snapshot in (
        (baseline_manifest, baseline_snapshot),
        (successor_manifest, successor_snapshot),
    ):
        snapshot_children = tuple(
            pin
            for pin in manifest.child_artifacts
            if pin.artifact_kind == "exploratory_graph_snapshot"
        )
        if (
            snapshot.build_ref != manifest.build_ref
            or len(snapshot_children) != 1
            or (
                snapshot_children[0].artifact_ref,
                snapshot_children[0].artifact_hash,
            )
            != (snapshot.snapshot_ref, snapshot.snapshot_hash)
        ):
            raise ValueError("paired report manifest-snapshot mismatch")

    if report.suite_ref is None:
        if (
            report.baseline_run_ref is not None
            or report.successor_run_ref is not None
            or evaluator.policy.required_case_ids
            or (
                report.behavioral_measured_count,
                report.behavioral_total_count,
                report.evidence_measured_count,
                report.evidence_total_count,
                report.verdict,
                report.reason_code,
            )
            != (
                0,
                0,
                0,
                0,
                "INCONCLUSIVE",
                "MISSING_REVIEWED_OBLIGATIONS",
            )
        ):
            raise ValueError("suite-less paired report control shape mismatch")
        return _expect_dependencies(source, tuple(dependencies), closure)

    suite_pin = _pin(
        "ontology_paired_obligation_suite", report.suite_ref, report.suite_hash
    )
    suite_source = _resolve(closure, suite_pin)
    _verify_declared(suite_source, closure, verify_ontology_paired_obligation_suite_v1)
    suite = _model_payload(
        suite_source, OntologyPairedObligationSuiteV1, "paired obligation suite"
    )
    if evaluator.policy.required_case_ids != tuple(
        case.case_id for case in suite.cases
    ) or source_basis != (
        suite.source_identity_ref,
        suite.source_hash,
        suite.canonical_source_ir_ref,
        suite.canonical_source_ir_hash,
    ):
        raise ValueError("paired report suite or policy member mismatch")
    dependencies.append(suite_pin)
    run_pins = (
        _pin(
            "ontology_paired_evaluation_run",
            report.baseline_run_ref,
            report.baseline_run_hash,
        ),
        _pin(
            "ontology_paired_evaluation_run",
            report.successor_run_ref,
            report.successor_run_hash,
        ),
    )
    run_values = []
    for pin in run_pins:
        run_source = _resolve(closure, pin)
        run_ancestry = _declared_ancestry(run_source, closure)
        run, _manifest, _snapshot, _suite, _bundle, _run_dependencies = _run_value(
            run_source, run_ancestry
        )
        run_values.append(run)
        dependencies.append(pin)
    baseline_run, successor_run = run_values
    if (
        (baseline_run.side, successor_run.side) != ("baseline", "successor")
        or baseline_run.manifest != report.subject.baseline_manifest
        or successor_run.manifest != report.subject.successor_manifest
        or baseline_run.snapshot != report.subject.baseline_snapshot
        or successor_run.snapshot != report.subject.successor_snapshot
        or (baseline_run.suite_ref, baseline_run.suite_hash)
        != (report.suite_ref, report.suite_hash)
        or (successor_run.suite_ref, successor_run.suite_hash)
        != (report.suite_ref, report.suite_hash)
        or (baseline_run.evaluator_ref, baseline_run.evaluator_hash)
        != (report.evaluator_ref, report.evaluator_hash)
        or (successor_run.evaluator_ref, successor_run.evaluator_hash)
        != (report.evaluator_ref, report.evaluator_hash)
        or not (
            baseline_run.evaluated_at
            == successor_run.evaluated_at
            == report.evaluated_at
        )
    ):
        raise ValueError("paired report run identity or saved time mismatch")
    _report_verdict(report, evaluator.policy, suite, baseline_run, successor_run)
    return _expect_dependencies(source, tuple(dependencies), closure)


__all__ = [
    "verify_ontology_paired_evaluation_report_v1",
    "verify_ontology_paired_evaluation_run_v1",
    "verify_paired_evaluation_policy_v1",
    "verify_paired_evaluator_bundle_v1",
    "verify_runtime_semantic_evaluation_bundle_v1",
]
