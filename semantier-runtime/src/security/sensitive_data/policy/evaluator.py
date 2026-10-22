from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from security.sensitive_data.authority.bundle_loader import ActivatedSensitiveGovernanceBundle
from security.sensitive_data.classification.ontology import BoundFinding
from semantier.sensitive_governance_contracts.enums import DisclosureAction
from semantier.sensitive_governance_contracts.hashing import canonical_hash

ROUTE_ORDER = {"PASS_AUTO": 0, "ESCALATE": 1, "FAIL_HARD": 2}
ACTION_ORDER = {
    DisclosureAction.ALLOW.value: 0,
    DisclosureAction.MASK.value: 1,
    DisclosureAction.TOKENIZE.value: 2,
    DisclosureAction.REDACT.value: 3,
    DisclosureAction.BLOCK.value: 4,
}


@dataclass(frozen=True)
class PolicyDecisionResult:
    policy_decision_ref: str
    finding_id: str
    finding_class: str
    institutional_route: str
    enforcement_action: DisclosureAction
    reason_codes: list[str]
    authority_refs: list[dict[str, Any]]
    reveal_eligibility: bool
    copy_eligibility: bool
    ttl_seconds: int | None
    policy_bundle_hash: str
    resolution_trace_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_decision_ref": self.policy_decision_ref,
            "finding_id": self.finding_id,
            "finding_class": self.finding_class,
            "institutional_route": self.institutional_route,
            "enforcement_action": self.enforcement_action.value,
            "reason_codes": self.reason_codes,
            "authority_refs": self.authority_refs,
            "reveal_eligibility": self.reveal_eligibility,
            "copy_eligibility": self.copy_eligibility,
            "ttl_seconds": self.ttl_seconds,
            "policy_bundle_hash": self.policy_bundle_hash,
            "resolution_trace_hash": self.resolution_trace_hash,
        }


class SensitivePolicyEvaluator:
    evaluator_version = "repo_sensitive_policy_evaluator_v1"

    def __init__(self, bundle: ActivatedSensitiveGovernanceBundle) -> None:
        self._bundle = bundle
        self._policy = bundle.policy_artifact
        unsupported = self._policy.get("unsupported_constructs")
        if unsupported:
            raise ValueError("sensitive governance policy bundle contains unsupported constructs")
        rules = self._policy.get("rules")
        if not isinstance(rules, list):
            raise ValueError("sensitive governance policy rules are missing")
        self._rules = {
            str(rule.get("class_id")): rule
            for rule in rules
            if isinstance(rule, dict) and rule.get("class_id")
        }

    def decide(
        self, findings: list[BoundFinding], *, destination: str,
        display_preferences: dict[str, str] | None = None,
    ) -> list[PolicyDecisionResult]:
        preferences = display_preferences or {}
        for class_id, requested in preferences.items():
            rule = self._rules.get(class_id)
            if rule is None or requested not in ACTION_ORDER:
                raise ValueError('invalid display preference')
            if ACTION_ORDER[requested] < ACTION_ORDER[str(rule['action'])]:
                raise ValueError('display preference weakens policy')
        decisions: list[PolicyDecisionResult] = []
        authority_refs = self._authority_refs()
        policy_bundle_hash = self._bundle.artifact_hashes[str(self._policy["artifact_id"])]
        for finding in findings:
            rule = self._rules.get(finding.finding_class.value)
            if rule is None:
                raise ValueError(f"missing T4 policy rule for {finding.finding_class.value}")
            route = str(rule.get("route") or "FAIL_HARD")
            action = DisclosureAction(str(rule.get("action") or DisclosureAction.BLOCK.value))
            reason_codes = [
                str(reason)
                for reason in rule.get("reason_codes", [])
                if isinstance(reason, str)
            ]
            if destination == "external_agent" and action == DisclosureAction.TOKENIZE:
                action = DisclosureAction.REDACT
                reason_codes = [*reason_codes, "external_destination_more_restrictive"]
            if finding.finding_class.value == "EXISTING_STATE_SECRET_MARKING" and destination != "local_simulated_agent":
                route = "FAIL_HARD"
                action = DisclosureAction.BLOCK
                reason_codes = [*reason_codes, "external_state_secret_transmission_blocked"]
            requested = preferences.get(finding.finding_class.value)
            if requested is not None:
                if ACTION_ORDER[requested] < ACTION_ORDER[action.value]:
                    raise ValueError('display preference weakens policy')
                if requested != action.value:
                    reason_codes = [*reason_codes, 'user_requested_additional_protection']
                action = DisclosureAction(requested)
            trace = {
                "finding_id": finding.finding_id,
                "class": finding.finding_class.value,
                "destination": destination,
                "route": route,
                "action": action.value,
                "policy_artifact_id": self._policy["artifact_id"],
            }
            if requested is not None:
                trace['requested_display_preference'] = requested
            decisions.append(
                PolicyDecisionResult(
                    policy_decision_ref="sg_policy_decision_" + canonical_hash(trace)[:24],
                    finding_id=finding.finding_id,
                    finding_class=finding.finding_class.value,
                    institutional_route=route,
                    enforcement_action=action,
                    reason_codes=reason_codes,
                    authority_refs=authority_refs,
                    reveal_eligibility=action in {DisclosureAction.MASK, DisclosureAction.TOKENIZE},
                    copy_eligibility=False,
                    ttl_seconds=30 if action in {DisclosureAction.MASK, DisclosureAction.TOKENIZE} else None,
                    policy_bundle_hash=policy_bundle_hash,
                    resolution_trace_hash=canonical_hash(trace),
                )
            )
        return decisions

    def aggregate_route(self, decisions: list[PolicyDecisionResult]) -> str:
        if not decisions:
            return "PASS_AUTO"
        return max((decision.institutional_route for decision in decisions), key=lambda route: ROUTE_ORDER[route])

    def aggregate_action(self, decisions: list[PolicyDecisionResult]) -> DisclosureAction:
        if not decisions:
            return DisclosureAction.ALLOW
        return max((decision.enforcement_action for decision in decisions), key=lambda action: ACTION_ORDER[action.value])

    def _authority_refs(self) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for artifact in self._bundle.authority_artifacts:
            for ref in artifact.get("references", []):
                if not isinstance(ref, dict):
                    continue
                refs.append(
                    {
                        "tier": artifact["tier"],
                        "artifact_id": artifact["artifact_id"],
                        "authority_ref": ref.get("authority_ref"),
                        "display_name_zh": ref.get("display_name_zh"),
                    }
                )
        return refs
