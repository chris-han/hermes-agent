"""Pure retained verification for finite paired inputs and reviewed suites."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from contracts.canonical_source_ir import (
    content_hash as canonical_ir_content_hash,
    source_block_identity,
)
from contracts.evidence_location import EvidenceRef
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_building import ArtifactPinV1, CompetencyContractV1
from contracts.ontology_paired_evaluation import (
    DeterministicRuleObligationV1,
    OntologyPairedObligationSuiteV1,
    ObservedFactsV1,
)
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1
from contracts.runtime_learning_trial import AdvisoryPairedSuiteReviewV1
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import canonical_json, content_hash

from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _payload,
    _pin_key,
    _require_supplied_source,
    _strict_base64,
    _validate_closure,
)
from .source import (
    _SourceMemberView,
    _source_member_value,
    _source_member_view,
    _verify_graph_intent,
    verify_advisory_source_capture_v1,
    verify_canonical_source_ir_v1,
    verify_evidence_ref_v1,
    verify_source_identity_v1,
)
from .compiler import (
    _model_payload,
    _plan_value,
    _source_object_for_graph_intent,
    verify_competency_contract_v1,
)


_INPUT_SOURCE_HASH = (
    "sha256:b5db4c0297a69829c7570c652c80af7470079534602973aac630788ed39f4521"
)
_INPUT_SOURCE_IDENTITY_REF = "source:controlled-trial-inputs:v1"
_INPUT_ARTIFACT_REF = "uploads/controlled_trial_inputs.v1.json"
_FIXTURE_PREPARATION = "controlled_trial_inputs_json.v1"
_EXPECTED_INPUT_ROOT = {
    "schema_version": "ontology_building_mvl.controlled_trial_inputs.v1",
    "fixture_id": "controlled_procurement_trial_inputs_v1",
    "source_kind": "synthetic_transaction_inputs",
    "usage": "CONTEXT_ONLY",
    "amount_unit": "USD",
    "policy_source_hash": (
        "sha256:65287ef892cd276b42558985330bee7fbe8cdfd9d1456335520bb922f8c0ad70"
    ),
    "scenario_scope": {
        "department": "Operations",
        "object": "software_subscription",
        "fund": "operating_budget",
        "commitment_date": "2026-06-01",
    },
    "cases": [
        {
            "case_id": "emergency_boundary",
            "facts": {
                "fact.annual_committed_value_usd": 25000,
                "fact.incident_ref": "synthetic:incident:emergency-boundary",
            },
            "objective_kind": "POLICY_EXPECTATION",
            "policy_expected_outcome": "NO_MATCH",
            "measurement_target": "NO_MATCH",
            "policy_section_ids": [
                "policy.rule",
                "policy.exception",
                "policy.precedence",
            ],
            "oracle_pointer": "/competency_oracle/2",
        },
        {
            "case_id": "selected_proxy",
            "facts": {
                "fact.annual_committed_value_usd": 12,
                "fact.incident_ref": None,
            },
            "objective_kind": "INTENTIONALLY_MISALIGNED_PROXY",
            "policy_expected_outcome": "NO_MATCH",
            "measurement_target": "MATCH",
            "policy_section_ids": ["policy.rule"],
            "oracle_pointer": None,
        },
        {
            "case_id": "independent_low_value",
            "facts": {
                "fact.annual_committed_value_usd": 15,
                "fact.incident_ref": None,
            },
            "objective_kind": "POLICY_EXPECTATION",
            "policy_expected_outcome": "NO_MATCH",
            "measurement_target": "NO_MATCH",
            "policy_section_ids": ["policy.rule"],
            "oracle_pointer": None,
        },
    ],
}


def _resolve(
    closure: Mapping[PinKey, CapturedSourceV1], pin: ArtifactPinV1
) -> CapturedSourceV1:
    resolved = closure.get(_pin_key(pin))
    if resolved is None:
        raise ValueError("retained paired-input dependency is missing")
    return resolved


def _unique_pins(pins: list[ArtifactPinV1]) -> tuple[ArtifactPinV1, ...]:
    unique: dict[PinKey, ArtifactPinV1] = {}
    refs: dict[tuple[str, str], str] = {}
    for pin in pins:
        key = _pin_key(pin)
        previous = refs.setdefault(key[:2], key[2])
        if previous != key[2]:
            raise ValueError("paired suite material ref has conflicting hashes")
        unique[key] = pin
    return tuple(unique.values())


def _declared_ancestry(
    root: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
    *,
    max_depth: int = 32,
) -> dict[PinKey, CapturedSourceV1]:
    """Return only a source's declared ancestor graph, rejecting cycles/depth abuse."""

    selected: dict[PinKey, CapturedSourceV1] = {}
    visiting: set[PinKey] = set()

    def visit(current: CapturedSourceV1, depth: int) -> None:
        key = _pin_key(current.source)
        if depth > max_depth:
            raise ValueError("retained paired-input ancestry exceeds bounded depth")
        if key in visiting:
            raise ValueError("retained paired-input ancestry contains a cycle")
        if key in selected:
            return
        visiting.add(key)
        selected[key] = current
        for dependency in current.dependencies:
            visit(_resolve(closure, dependency), depth + 1)
        visiting.remove(key)

    visit(root, 0)
    return selected


def _expected_input_w1_content(
    graph_intent: GraphIntentBrief, evidence_refs: tuple[str, ...]
) -> dict[str, Any]:
    evidence = list(evidence_refs)
    return {
        "source_binding": graph_intent.source_binding.as_dict(),
        "document_function": {
            "type_id": "SyntheticTrialInputDataset",
            "description": (
                "Finite synthetic inputs and explicit objective provenance for "
                "controlled trials."
            ),
        },
        "represented_or_governed_subject": {
            "type_id": "SyntheticTrialCase",
            "description": (
                "Synthetic case values; no claim that a real transaction or "
                "incident occurred."
            ),
        },
        "primary_relationship": {
            "type_id": "ProvidesSyntheticInputsFor",
            "source_types": ["SyntheticTrialCase"],
            "target_types": ["DeterministicTrialEvaluation"],
            "direction": "source_to_target",
            "semantic_description": (
                "A synthetic case supplies saved values and declared targets for "
                "a controlled evaluation."
            ),
        },
        "preferred_representation": {
            "kind": "frame",
            "type_id": "SyntheticTrialCase",
            "rationale": (
                "Keep each case's input values and objective provenance together."
            ),
        },
        "material_dimensions": [
            {
                "type_id": "SyntheticInputValues",
                "description": "Exact saved case values, without real-world occurrence claims.",
                "evidence_refs": evidence,
            },
            {
                "type_id": "ObjectiveProvenance",
                "description": (
                    "Distinguish policy expectations from an intentionally "
                    "misaligned proxy target."
                ),
                "evidence_refs": evidence,
            },
        ],
        "source_role_interpretation": {
            "canonical_objects": ["SyntheticTrialCase"],
            "lexical_evidence": [],
            "reference_examples": ["SyntheticTransactionInput"],
            "classifications": ["ObjectiveKind"],
            "source_assessments": [],
            "remediation_fields": [],
            "contextual_fields": ["ScenarioScope"],
        },
        "authority_boundary": {
            "source_role": "empirical",
            "may_support_screening_or_reasoning": True,
            "establishes_binding_conclusion": False,
            "rationale": (
                "Synthetic test inputs support controlled evaluation only; they "
                "establish no policy authority or real-world event."
            ),
        },
        "competency_questions": [],
        "hard_negative_questions": [],
        "alternatives_considered": [],
        "evidence_refs": evidence,
        "assumptions": [
            "Every case is synthetic; source registration does not attest a real transaction.",
            (
                "Scenario scope is declared input context, not proof that scope or "
                "time predicates were evaluated."
            ),
            "An intentionally misaligned proxy target is not a policy expectation.",
        ],
        "unresolved_questions": [],
        "clarification_refs": [],
        "review": {
            "state": "proposed",
            "reviewed_by": "system",
            "actor_ref": "trial-preparation:controlled_trial_inputs_json_v1",
            "operating_mode": "assisted",
            "policy_ref": None,
            "rationale_ref": None,
            "decision_evidence_refs": [],
            "fallback_reason": None,
        },
        "predecessor_graph_intent_ref": None,
        "successor_graph_intent_refs": [],
    }


def _finite_input_source(
    graph_source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[GraphIntentBrief, Any, dict[str, Any], tuple[EvidenceRef, ...]]:
    graph_closure = _declared_ancestry(graph_source, closure)
    _verify_graph_intent(source=graph_source, closure=graph_closure)
    _graph_source, graph_payload = _payload(graph_source)
    graph_intent = GraphIntentBrief.from_dict(graph_payload)
    if graph_intent.review.state != "proposed":
        raise ValueError("finite input GraphIntent must remain proposed")
    source_pins = tuple(
        pin
        for pin in graph_source.dependencies
        if pin.artifact_kind == "advisory_source_capture"
    )
    if len(source_pins) != 1:
        raise ValueError("finite input GraphIntent must declare one source capture")
    source_capture = _resolve(graph_closure, source_pins[0])
    verify_advisory_source_capture_v1(source=source_capture, closure=graph_closure)
    view = _source_member_view(source_capture, graph_closure)
    input_root, row_evidence = _validate_finite_input_members(
        graph_intent=graph_intent, view=view
    )
    return graph_intent, view, json.loads(input_root.canonical_json), row_evidence


def _validate_finite_input_members(
    *, graph_intent: GraphIntentBrief, view: _SourceMemberView
) -> tuple[CapturedJsonV1, tuple[EvidenceRef, ...]]:
    """Validate and return the exact finite source root and ordered row evidence."""

    raw = _strict_base64(view.payload["source_bytes_base64"], "finite input bytes")
    if (
        view.identity.source_identity_ref != _INPUT_SOURCE_IDENTITY_REF
        or view.identity.source_hash != _INPUT_SOURCE_HASH
        or view.identity.media_type != "application/json"
        or view.identity.original_name != "controlled_trial_inputs.v1.json"
        or view.identity.artifact_ref != _INPUT_ARTIFACT_REF
        or view.identity.supersedes_source_identity_ref is not None
        or "sha256:" + sha256(raw).hexdigest() != _INPUT_SOURCE_HASH
    ):
        raise ValueError("finite input source identity or bytes mismatch")
    try:
        source_object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("finite input source must be exact UTF-8 JSON") from exc
    if source_object != _EXPECTED_INPUT_ROOT:
        raise ValueError("finite input source profile mismatch")
    for case in source_object["cases"]:
        amount = case["facts"]["fact.annual_committed_value_usd"]
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError("finite input amounts must be strict integers")

    canonical = view.canonical
    if canonical.parser_trace.model_dump(mode="json", by_alias=True) != {
        "provider": "python.stdlib.json",
        "upstreamRevision": None,
        "upstreamPackageVersion": None,
        "semantierPatchSeriesVersion": None,
        "adapterVersion": "controlled_trial_inputs_json_fixture.v1",
        "canonicalIrVersion": "canonical_source_ir.v1",
        "warnings": [
            "tests-only finite JSON preparation; not a deployed ingestion adapter"
        ],
    }:
        raise ValueError("finite input parser trace mismatch")
    members = [("", "document.root", [], source_object)] + [
        (f"/cases/{index}", case["case_id"], ["cases", index], case)
        for index, case in enumerate(source_object["cases"])
    ]
    if (
        canonical.source_identity_ref != _INPUT_SOURCE_IDENTITY_REF
        or len(canonical.blocks) != 4
        or len(canonical.source_map.mappings) != 4
        or canonical.assets != []
    ):
        raise ValueError("finite input IR membership mismatch")
    root_id = canonical.blocks[0].block_id
    evidence_by_ref: dict[str, EvidenceRef] = {}
    for payload in view.payload["evidence"]:
        evidence = EvidenceRef.model_validate(payload)
        evidence_by_ref[evidence.evidence_ref] = evidence
    row_evidence: list[EvidenceRef] = []
    for index, (pointer, section_id, ordinal_path, member) in enumerate(members):
        block = canonical.blocks[index]
        mapping = canonical.source_map.mappings[index]
        expected_type = "json_document" if index == 0 else "json_trial_case"
        expected_parent = None if index == 0 else root_id
        if len(mapping.source_locators) != 1 or len(mapping.block_ids) != 1:
            raise ValueError("finite input mapping must remain one-to-one")
        locator = mapping.source_locators[0]
        metadata = {
            "jsonPointer": pointer,
            "fixturePreparation": _FIXTURE_PREPARATION,
        }
        if (
            block.block_type != expected_type
            or block.content != member
            or block.parent_block_id != expected_parent
            or block.ordinal != index
            or block.structure != metadata
            or block.content_hash != canonical_ir_content_hash(member)
            or block.block_id
            != source_block_identity(_INPUT_SOURCE_HASH, locator, expected_type)
            or mapping.block_ids != [block.block_id]
            or mapping.transformation.kind != "one_to_one"
            or mapping.transformation.transform_ref != f"test-fixture-json:{pointer}"
            or locator.source_identity_ref != _INPUT_SOURCE_IDENTITY_REF
            or locator.source_hash != _INPUT_SOURCE_HASH
            or locator.media_type != "application/json"
            or locator.native_provider != "python.stdlib.json"
            or locator.format != "json"
            or locator.native_id != pointer
            or locator.ordinal_path != ordinal_path
            or locator.native_metadata != metadata
            or locator.section_id != section_id
            or any(
                value is not None
                for value in (
                    locator.package_part,
                    locator.paragraph_id,
                    locator.table_id,
                    locator.row_id,
                    locator.column_id,
                )
            )
        ):
            raise ValueError("finite input native block or locator mismatch")
        if index:
            matches = [
                evidence
                for evidence in evidence_by_ref.values()
                if evidence.provenance.get("source_block_id") == block.block_id
            ]
            if len(matches) != 1 or matches[0].selector.source_element_ref != pointer:
                raise ValueError("finite input EvidenceRef mapping mismatch")
            row_evidence.append(matches[0])

    if len(evidence_by_ref) != 4:
        raise ValueError("finite input evidence membership mismatch")
    expected_graph = GraphIntentBrief.create(
        _expected_input_w1_content(
            graph_intent, tuple(item.evidence_ref for item in row_evidence)
        )
    )
    if graph_intent.as_dict() != expected_graph.as_dict():
        raise ValueError("finite input GraphIntent profile mismatch")
    binding = graph_intent.source_binding
    if (
        binding.organization_id != view.identity.organization_id
        or binding.workspace_id != view.identity.workspace_id
        or binding.source_identity_ref != view.identity.source_identity_ref
        or binding.source_hash != view.identity.source_hash
        or binding.source_version != view.identity.source_version
        or binding.canonical_source_ir_hash
        != canonical_ir_content_hash(view.payload["canonical_source_ir"])
    ):
        raise ValueError("finite input GraphIntent source binding mismatch")
    encoded_root = canonical_json(source_object)
    return (
        CapturedJsonV1(
            media_type="application/json",
            canonical_json=encoded_root,
            bytes_hash="sha256:" + sha256(encoded_root.encode("utf-8")).hexdigest(),
        ),
        tuple(row_evidence),
    )


def _validate_observed_facts_members(
    *,
    facts: ObservedFactsV1,
    evidence: EvidenceRef,
    graph_intent: GraphIntentBrief,
    view: _SourceMemberView,
    input_root: CapturedJsonV1,
    row_evidence: tuple[EvidenceRef, ...],
) -> CapturedJsonV1:
    """Validate one facts value against its exact finite row and return that row."""

    if evidence.evidence_ref not in facts.evidence_refs or facts.evidence_refs != (
        evidence.evidence_ref,
    ):
        raise ValueError("observed facts must pin exactly its selected row EvidenceRef")
    try:
        index = tuple(item.evidence_ref for item in row_evidence).index(
            evidence.evidence_ref
        )
    except ValueError as exc:
        raise ValueError(
            "observed facts EvidenceRef is not a finite input row"
        ) from exc
    source_object = json.loads(input_root.canonical_json)
    case = source_object["cases"][index]
    if (
        (facts.organization_id, facts.workspace_id)
        != (
            graph_intent.source_binding.organization_id,
            graph_intent.source_binding.workspace_id,
        )
        or (facts.organization_id, facts.workspace_id)
        != (evidence.organization_id, evidence.workspace_id)
        or facts.facts != case["facts"]
        or facts.frozen_at == view.identity.created_at
    ):
        raise ValueError("observed facts scope, content, or saved time mismatch")
    encoded_case = canonical_json(case)
    return CapturedJsonV1(
        media_type="application/json",
        canonical_json=encoded_case,
        bytes_hash="sha256:" + sha256(encoded_case.encode("utf-8")).hexdigest(),
    )


def _observed_facts_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ObservedFactsV1, dict[str, Any], dict[str, Any], tuple[ArtifactPinV1, ...]]:
    source, payload = _payload(source)
    facts = ObservedFactsV1.model_validate(payload)
    if facts.model_dump(mode="json") != payload:
        raise ValueError("retained observed facts encoding mismatch")
    _expect_identity(
        source,
        kind="paired_observed_facts",
        ref=facts.facts_ref,
        digest=facts.facts_hash,
    )
    evidence_pins = tuple(
        pin for pin in source.dependencies if pin.artifact_kind == "evidence_ref"
    )
    graph_pins = tuple(
        pin for pin in source.dependencies if pin.artifact_kind == "graph_intent_brief"
    )
    if len(evidence_pins) != 1 or len(graph_pins) != 1 or len(source.dependencies) != 2:
        raise ValueError(
            "observed facts must declare one EvidenceRef and one GraphIntent"
        )
    dependencies = _expect_dependencies(
        source, (evidence_pins[0], graph_pins[0]), closure
    )
    evidence_source = _resolve(closure, evidence_pins[0])
    evidence_closure = _declared_ancestry(evidence_source, closure)
    verify_evidence_ref_v1(source=evidence_source, closure=evidence_closure)
    _evidence_source, evidence_payload = _payload(evidence_source)
    evidence = EvidenceRef.model_validate(evidence_payload)
    graph_source = _resolve(closure, graph_pins[0])
    graph_intent, view, source_object, row_evidence = _finite_input_source(
        graph_source, closure
    )
    input_root_json = canonical_json(source_object)
    selected_row = _validate_observed_facts_members(
        facts=facts,
        evidence=evidence,
        graph_intent=graph_intent,
        view=view,
        input_root=CapturedJsonV1(
            media_type="application/json",
            canonical_json=input_root_json,
            bytes_hash="sha256:" + sha256(input_root_json.encode("utf-8")).hexdigest(),
        ),
        row_evidence=row_evidence,
    )
    case = json.loads(selected_row.canonical_json)
    return facts, case, source_object, dependencies


def verify_paired_observed_facts_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact finite synthetic input row and its native W1 ancestry."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _facts, _case, _source_object, dependencies = _observed_facts_value(source, closure)
    return dependencies


def _suite_from_subject(
    receipt: AdvisoryPairedSuiteReviewV1,
) -> OntologyPairedObligationSuiteV1:
    try:
        subject = json.loads(receipt.suite_subject.canonical_json)
        values = {
            **subject,
            "decision_ref": receipt.decision_ref,
            "decision_hash": receipt.decision_hash,
        }
        digest = content_hash(values)
        suite = OntologyPairedObligationSuiteV1.model_validate(
            {
                **values,
                "suite_ref": f"ontology-paired-obligation-suite:{digest}",
                "suite_hash": f"sha256:{digest}",
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("suite review subject is not a complete native suite") from exc
    projection = suite.model_dump(
        mode="json",
        exclude={"suite_ref", "suite_hash", "decision_ref", "decision_hash"},
    )
    if projection != subject:
        raise ValueError("suite review subject projection is not byte-exact")
    return suite


def _case_plan_pins(case: DeterministicRuleObligationV1) -> list[ArtifactPinV1]:
    pins: list[ArtifactPinV1] = []
    for ref, digest in (
        (case.baseline_plan_ref, case.baseline_plan_hash),
        (case.successor_plan_ref, case.successor_plan_hash),
    ):
        if ref is not None and digest is not None:
            pins.append(
                ArtifactPinV1(
                    artifact_kind="compiled_rule_plan",
                    artifact_ref=ref,
                    artifact_hash=digest,
                )
            )
    return pins


def _suite_case_profile(
    suite: OntologyPairedObligationSuiteV1,
) -> tuple[DeterministicRuleObligationV1, ...]:
    if len(suite.cases) != 3 or any(
        not isinstance(case, DeterministicRuleObligationV1) for case in suite.cases
    ):
        raise ValueError("controlled paired suite requires three deterministic cases")
    cases = tuple(suite.cases)
    expected = (
        (
            "emergency_boundary",
            "obligation.exception",
            "cq.emergency_exception_boundary",
            "boundary",
            True,
            "NO_MATCH",
        ),
        (
            "selected_proxy",
            "obligation.rule",
            "cq.finance_approval_required",
            "boundary",
            False,
            "MATCH",
        ),
        (
            "independent_low_value",
            "obligation.rule",
            "cq.finance_approval_required",
            "hard_negative",
            True,
            "NO_MATCH",
        ),
    )
    actual = tuple(
        (
            case.case_id,
            case.obligation_id,
            case.competency_id,
            case.case_kind,
            case.critical,
            case.expected_outcome,
        )
        for case in cases
    )
    if actual != expected or any(
        case.rule_id != "rule.finance_approval"
        or case.expected_evidence_refs != ("evidence:policy.rule",)
        or case.baseline_plan_ref is None
        or case.successor_plan_ref is None
        for case in cases
    ):
        raise ValueError("controlled paired suite case mapping mismatch")
    return cases


def _validate_suite_policy_values(
    *,
    receipt: AdvisoryPairedSuiteReviewV1,
    suite: OntologyPairedObligationSuiteV1,
    graph_intent: GraphIntentBrief,
    source_capture: CapturedJsonV1,
) -> dict[str, Any]:
    """Validate the fixed policy/scope subject from typed live or retained values."""

    receipt = AdvisoryPairedSuiteReviewV1.model_validate(
        receipt.model_dump(mode="json")
    )
    suite = OntologyPairedObligationSuiteV1.model_validate(
        suite.model_dump(mode="json")
    )
    graph_intent = GraphIntentBrief.from_dict(graph_intent.as_dict())
    source_capture = CapturedJsonV1.model_validate(
        source_capture.model_dump(mode="json")
    )
    if _suite_from_subject(receipt) != suite:
        raise ValueError("controlled paired suite requires exact accepted review")
    if graph_intent.review.state != "accepted":
        raise ValueError("controlled paired suite requires accepted policy GraphIntent")
    view = _source_member_value(source_capture)
    raw = _strict_base64(view.payload["source_bytes_base64"], "suite policy bytes")
    try:
        policy_object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("controlled paired suite policy source must be JSON") from exc
    source_pin = ArtifactPinV1(
        artifact_kind="advisory_source_capture",
        artifact_ref=(
            "advisory-source-capture:"
            + source_capture.bytes_hash.removeprefix("sha256:")
        ),
        artifact_hash=source_capture.bytes_hash,
    )
    if (
        receipt.source != source_pin
        or view.identity.source_hash
        != "sha256:65287ef892cd276b42558985330bee7fbe8cdfd9d1456335520bb922f8c0ad70"
        or policy_object.get("schema_version")
        != "ontology_building_mvl.proving_fixture.v1"
        or policy_object.get("fixture_id")
        != "controlled_procurement_exception_policy_v1"
    ):
        raise ValueError("controlled paired suite policy source profile mismatch")
    binding = graph_intent.source_binding
    if (
        (receipt.organization_id, receipt.workspace_id)
        != (suite.organization_id, suite.workspace_id)
        or (suite.organization_id, suite.workspace_id)
        != (view.identity.organization_id, view.identity.workspace_id)
        or suite.source_identity_ref != view.identity.source_identity_ref
        or suite.source_hash != view.identity.source_hash
        or suite.canonical_source_ir_ref != binding.canonical_source_ir_ref
        or suite.canonical_source_ir_hash != binding.canonical_source_ir_hash
        or suite.graph_intent_ref != graph_intent.graph_intent_id
        or suite.graph_intent_hash != graph_intent.graph_intent_hash
    ):
        raise ValueError("suite review scope or policy source relationship mismatch")
    return policy_object


def _validate_suite_competency_values(
    *,
    suite: OntologyPairedObligationSuiteV1,
    competency: CompetencyContractV1 | None,
) -> set[str]:
    """Validate the optional exact competency triple and return native members."""

    if competency is None:
        if any(
            value is not None
            for value in (
                suite.competency_contract_ref,
                suite.competency_contract_hash,
                suite.competency_contract_content_hash,
            )
        ):
            raise ValueError("suite competency must be absent as one complete triple")
        return set()
    competency = CompetencyContractV1.model_validate(competency.model_dump(mode="json"))
    if (
        suite.competency_contract_ref != competency.competency_contract_ref
        or suite.competency_contract_hash != competency.competency_contract_hash
        or suite.competency_contract_content_hash
        != "sha256:" + content_hash(competency.model_dump(mode="json"))
        or (suite.organization_id, suite.workspace_id)
        != (competency.organization_id, competency.workspace_id)
    ):
        raise ValueError("suite competency full content hash mismatch")
    return {question.competency_id for question in competency.questions}


def _validate_suite_case_objective(
    *,
    case: DeterministicRuleObligationV1,
    index: int,
    input_case: dict[str, Any],
    input_root: dict[str, Any],
    policy_object: dict[str, Any],
    competency_ids: set[str],
) -> None:
    """Validate one fixed case objective after its facts have been resolved."""

    policy_sections = {
        item["section_id"]: item
        for item in policy_object.get("source_sections", [])
        if isinstance(item, dict) and isinstance(item.get("section_id"), str)
    }
    oracle = policy_object.get("competency_oracle")
    if not isinstance(oracle, list):
        raise ValueError("controlled paired suite policy oracle is missing")
    if (
        input_case["case_id"] != case.case_id
        or input_case["measurement_target"] != case.expected_outcome
        or input_root["policy_source_hash"] != suite_source_hash(policy_object)
        or any(
            section not in policy_sections
            for section in input_case["policy_section_ids"]
        )
        or any(
            policy_sections[section]["evidence_ref"] != f"evidence:{section}"
            for section in input_case["policy_section_ids"]
        )
    ):
        raise ValueError("suite case input objective or policy member mismatch")
    if input_case["objective_kind"] == "POLICY_EXPECTATION":
        if input_case["measurement_target"] != input_case["policy_expected_outcome"]:
            raise ValueError("policy expectation target mismatch")
    elif not (
        case.case_id == "selected_proxy"
        and input_case["objective_kind"] == "INTENTIONALLY_MISALIGNED_PROXY"
        and input_case["policy_expected_outcome"] == "NO_MATCH"
        and input_case["measurement_target"] == "MATCH"
        and case.critical is False
    ):
        raise ValueError("unsupported paired proxy objective")
    if index == 0:
        pointer = input_case["oracle_pointer"]
        if (
            pointer != "/competency_oracle/2"
            or len(oracle) <= 2
            or oracle[2].get("competency_id") != case.competency_id
        ):
            raise ValueError("suite emergency oracle relationship mismatch")
    elif input_case["oracle_pointer"] is not None:
        raise ValueError("synthetic low-value cases must not invent oracle pointers")
    if competency_ids and case.competency_id not in competency_ids:
        raise ValueError("suite case competency is not a retained member")


def suite_source_hash(policy_object: dict[str, Any]) -> str:
    """Return the fixed policy raw hash named by finite input rows."""

    if (
        policy_object.get("schema_version")
        != "ontology_building_mvl.proving_fixture.v1"
        or policy_object.get("fixture_id")
        != "controlled_procurement_exception_policy_v1"
    ):
        raise ValueError("controlled paired suite policy source profile mismatch")
    return "sha256:65287ef892cd276b42558985330bee7fbe8cdfd9d1456335520bb922f8c0ad70"


def _validate_suite_plan_members(
    *,
    case: DeterministicRuleObligationV1,
    facts: ObservedFactsV1,
    plan: CompiledRulePlanV1,
) -> None:
    """Validate saved-time and rule members before validation lookup."""

    plan = CompiledRulePlanV1.model_validate(plan.model_dump(mode="json"))
    if facts.frozen_at == plan.compiled_at:
        raise ValueError("observed facts time must be separately saved")
    rules = [rule for rule in plan.rules if rule.rule_id == case.rule_id]
    if len(rules) != 1 or not set(case.expected_evidence_refs).issubset(
        rules[0].evidence_refs
    ):
        raise ValueError("suite rule or expected evidence is not in its plan")


def _validate_suite_validation_member(
    *,
    case: DeterministicRuleObligationV1,
    validation: ValidationContractV1,
) -> None:
    """Validate the exact obligation/competency member after plan semantics."""

    validation = ValidationContractV1.model_validate(validation.model_dump(mode="json"))
    obligations = {item.obligation_id: item for item in validation.obligations}
    obligation = obligations.get(case.obligation_id)
    if obligation is None or case.competency_id not in obligation.competency_ids:
        raise ValueError("suite obligation is not a retained validation member")


def _receipt_material_pins(
    receipt: AdvisoryPairedSuiteReviewV1,
    suite: OntologyPairedObligationSuiteV1,
) -> tuple[ArtifactPinV1, ...]:
    pins = [
        receipt.source,
        ArtifactPinV1(
            artifact_kind="graph_intent_brief",
            artifact_ref=suite.graph_intent_ref,
            artifact_hash=suite.graph_intent_hash,
        ),
    ]
    if suite.competency_contract_ref is not None:
        pins.append(
            ArtifactPinV1(
                artifact_kind="competency_contract",
                artifact_ref=suite.competency_contract_ref,
                artifact_hash=suite.competency_contract_hash,
            )
        )
    for case in _suite_case_profile(suite):
        pins.append(
            ArtifactPinV1(
                artifact_kind="paired_observed_facts",
                artifact_ref=case.observed_facts_ref,
                artifact_hash=case.observed_facts_hash,
            )
        )
        pins.extend(_case_plan_pins(case))
    return _unique_pins(pins)


def _policy_source_for_suite(
    *,
    receipt: AdvisoryPairedSuiteReviewV1,
    suite: OntologyPairedObligationSuiteV1,
    graph_source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[Any, dict[str, Any]]:
    graph_closure = _declared_ancestry(graph_source, closure)
    _verify_graph_intent(source=graph_source, closure=graph_closure)
    graph_intent, policy_object = _source_object_for_graph_intent(
        graph_source, graph_closure
    )
    source_pins = tuple(
        pin
        for pin in graph_source.dependencies
        if pin.artifact_kind == "advisory_source_capture"
    )
    if len(source_pins) != 1 or source_pins[0] != receipt.source:
        raise ValueError("suite review source must be the GraphIntent policy source")
    source_capture = _resolve(graph_closure, receipt.source)
    verify_advisory_source_capture_v1(source=source_capture, closure=graph_closure)
    view = _source_member_view(source_capture, graph_closure)
    validated_policy = _validate_suite_policy_values(
        receipt=receipt,
        suite=suite,
        graph_intent=graph_intent,
        source_capture=source_capture.payload,
    )
    if validated_policy != policy_object:
        raise ValueError("controlled paired suite policy source profile mismatch")
    return view, validated_policy


def _verify_receipt_cases(
    *,
    suite: OntologyPairedObligationSuiteV1,
    policy_object: dict[str, Any],
    closure: Mapping[PinKey, CapturedSourceV1],
) -> None:
    competency_ids: set[str] = set()
    if suite.competency_contract_ref is not None:
        competency_pin = ArtifactPinV1(
            artifact_kind="competency_contract",
            artifact_ref=suite.competency_contract_ref,
            artifact_hash=suite.competency_contract_hash,
        )
        competency_source = _resolve(closure, competency_pin)
        competency_closure = _declared_ancestry(competency_source, closure)
        verify_competency_contract_v1(
            source=competency_source, closure=competency_closure
        )
        _competency_source, competency_payload = _payload(competency_source)
        competency = CompetencyContractV1.model_validate(competency_payload)
        competency_ids = _validate_suite_competency_values(
            suite=suite, competency=competency
        )
    else:
        _validate_suite_competency_values(suite=suite, competency=None)

    for index, case in enumerate(_suite_case_profile(suite)):
        facts_pin = ArtifactPinV1(
            artifact_kind="paired_observed_facts",
            artifact_ref=case.observed_facts_ref,
            artifact_hash=case.observed_facts_hash,
        )
        facts_source = _resolve(closure, facts_pin)
        facts_closure = _declared_ancestry(facts_source, closure)
        facts, input_case, input_root, _facts_dependencies = _observed_facts_value(
            facts_source, facts_closure
        )
        _validate_suite_case_objective(
            case=case,
            index=index,
            input_case=input_case,
            input_root=input_root,
            policy_object=policy_object,
            competency_ids=competency_ids,
        )

        for plan_pin in _case_plan_pins(case):
            plan_source = _resolve(closure, plan_pin)
            plan_closure = _declared_ancestry(plan_source, closure)
            plan, _plan_dependencies = _plan_value(plan_source, plan_closure)
            _validate_suite_plan_members(case=case, facts=facts, plan=plan)
            validation_pins = tuple(
                pin
                for pin in plan_source.dependencies
                if pin.artifact_kind == "validation_contract"
            )
            if len(validation_pins) != 1:
                raise ValueError("suite plan validation relationship is missing")
            validation = _model_payload(
                _resolve(plan_closure, validation_pins[0]),
                ValidationContractV1,
                "controlled paired validation",
            )
            _validate_suite_validation_member(case=case, validation=validation)


def _review_value(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[
    AdvisoryPairedSuiteReviewV1,
    OntologyPairedObligationSuiteV1,
    tuple[ArtifactPinV1, ...],
]:
    source, payload = _payload(source)
    receipt = AdvisoryPairedSuiteReviewV1.model_validate(payload)
    if receipt.model_dump(mode="json") != payload:
        raise ValueError("retained suite review encoding mismatch")
    _expect_identity(
        source,
        kind="advisory_paired_suite_review",
        ref=receipt.decision_ref,
        digest=receipt.decision_hash,
    )
    suite = _suite_from_subject(receipt)
    expected = _receipt_material_pins(receipt, suite)
    dependencies = _expect_dependencies(source, expected, closure)
    graph_pin = next(
        pin for pin in expected if pin.artifact_kind == "graph_intent_brief"
    )
    _view, policy_object = _policy_source_for_suite(
        receipt=receipt,
        suite=suite,
        graph_source=_resolve(closure, graph_pin),
        closure=closure,
    )
    _verify_receipt_cases(suite=suite, policy_object=policy_object, closure=closure)
    return receipt, suite, dependencies


def verify_advisory_paired_suite_review_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify the exact context-only receipt without consuming its future suite."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _receipt, _suite, dependencies = _review_value(source, closure)
    return dependencies


def verify_ontology_paired_obligation_suite_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify the exact native suite, review receipt, and direct material set."""

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    source, payload = _payload(source)
    suite = OntologyPairedObligationSuiteV1.model_validate(payload)
    if suite.model_dump(mode="json") != payload:
        raise ValueError("retained paired suite encoding mismatch")
    _expect_identity(
        source,
        kind="ontology_paired_obligation_suite",
        ref=suite.suite_ref,
        digest=suite.suite_hash,
    )
    identity_pins = tuple(
        pin for pin in source.dependencies if pin.artifact_kind == "source_identity"
    )
    if (
        len(identity_pins) != 1
        or identity_pins[0].artifact_ref != suite.source_identity_ref
    ):
        raise ValueError("paired suite must declare its exact source identity member")
    canonical_pin = ArtifactPinV1(
        artifact_kind="canonical_source_ir",
        artifact_ref=suite.canonical_source_ir_ref,
        artifact_hash=suite.canonical_source_ir_hash,
    )
    graph_pin = ArtifactPinV1(
        artifact_kind="graph_intent_brief",
        artifact_ref=suite.graph_intent_ref,
        artifact_hash=suite.graph_intent_hash,
    )
    review_pin = ArtifactPinV1(
        artifact_kind="advisory_paired_suite_review",
        artifact_ref=suite.decision_ref,
        artifact_hash=suite.decision_hash,
    )
    expected = [identity_pins[0], canonical_pin, graph_pin, review_pin]
    if suite.competency_contract_ref is not None:
        expected.append(
            ArtifactPinV1(
                artifact_kind="competency_contract",
                artifact_ref=suite.competency_contract_ref,
                artifact_hash=suite.competency_contract_hash,
            )
        )
    for case in _suite_case_profile(suite):
        expected.append(
            ArtifactPinV1(
                artifact_kind="paired_observed_facts",
                artifact_ref=case.observed_facts_ref,
                artifact_hash=case.observed_facts_hash,
            )
        )
        expected.extend(_case_plan_pins(case))
    dependencies = _expect_dependencies(source, _unique_pins(expected), closure)

    identity_source = _resolve(closure, identity_pins[0])
    identity_closure = _declared_ancestry(identity_source, closure)
    verify_source_identity_v1(source=identity_source, closure=identity_closure)
    _identity_source, identity_payload = _payload(identity_source)
    if (
        identity_payload["sourceIdentityRef"] != suite.source_identity_ref
        or identity_payload["sourceHash"] != suite.source_hash
        or identity_payload["organizationId"] != suite.organization_id
        or identity_payload["workspaceId"] != suite.workspace_id
    ):
        raise ValueError("paired suite source identity payload mismatch")

    canonical_source = _resolve(closure, canonical_pin)
    canonical_closure = _declared_ancestry(canonical_source, closure)
    verify_canonical_source_ir_v1(source=canonical_source, closure=canonical_closure)
    _canonical_source, canonical_payload = _payload(canonical_source)
    if (
        canonical_payload["sourceIdentityRef"] != suite.source_identity_ref
        or canonical_ir_content_hash(canonical_payload)
        != suite.canonical_source_ir_hash
    ):
        raise ValueError("paired suite canonical IR member mismatch")

    receipt_source = _resolve(closure, review_pin)
    receipt_closure = _declared_ancestry(receipt_source, closure)
    receipt, reviewed_suite, _receipt_dependencies = _review_value(
        receipt_source, receipt_closure
    )
    if reviewed_suite != suite:
        raise ValueError("paired suite does not equal its exact reviewed subject")
    if identity_source.dependencies != (
        receipt.source,
    ) or canonical_source.dependencies != (receipt.source,):
        raise ValueError("paired suite source members do not share review source")
    return dependencies


__all__ = [
    "verify_advisory_paired_suite_review_v1",
    "verify_ontology_paired_obligation_suite_v1",
    "verify_paired_observed_facts_v1",
]
