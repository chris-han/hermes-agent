"""Deterministic cross-domain resolver for Gate 3 (v8.2).

This module is pure and stateless:
  - no DB access
  - no live retrieval
  - no LLM/OCR/parser dependencies

Inputs are pinned runtime artifacts and policy contracts.
"""

from __future__ import annotations

from uuid import uuid4

from eos.artifact_hashing import content_hash

NON_BLOCKING_OUTCOMES = {"ALLOW", "ALLOW_WITH_DISCLOSURE", "ALLOW_WITH_LIMITS"}
VALID_OUTCOMES = NON_BLOCKING_OUTCOMES | {"ESCALATE", "BLOCK_ACTION", "BLOCK_EXPORT"}


def _projected_domains(projection_set: dict) -> set[str]:
    domains = set(projection_set.get("domains", []))
    if domains:
        return domains
    projections = projection_set.get("projections", [])
    return {p.get("domain") for p in projections if p.get("domain")}


def resolve_cross_domain(
    projection_set: dict,
    action_intent: str,
    xpolicy: dict,
    resolver_contract: dict,
    tiered_verification_contract: dict | None,
) -> dict:
    """Resolve cross-domain action admissibility.

    Policy shape (minimum supported):
      hard_block_actions: {action_intent: [reason_code, ...]}
      hard_block_exports: {action_intent: [reason_code, ...]}
      disclosure_required_actions: [action_intent, ...]
      limits_by_action: {action_intent: { ... limits ... }}
      escalation_actions: [action_intent, ...]
      required_domains_by_action: {action_intent: [domain, ...]}
    """
    if action_intent is None or action_intent == "":
        raise ValueError("action_intent is required")

    policy_version = xpolicy.get("policy_version") or xpolicy.get("version") or "unknown"
    resolver_version = (
        resolver_contract.get("resolver_version")
        or resolver_contract.get("version")
        or "unknown"
    )

    projected_domains = _projected_domains(projection_set)
    reasons: list[str] = []
    limits = None
    disclosure_required = False

    required_domains_map = xpolicy.get("required_domains_by_action", {})
    required_domains = set(required_domains_map.get(action_intent, []))
    if required_domains and not required_domains.issubset(projected_domains):
        missing = sorted(required_domains - projected_domains)
        reasons.append(f"missing_required_domains:{','.join(missing)}")
        outcome = "ESCALATE"
    else:
        hard_block_actions = xpolicy.get("hard_block_actions", {})
        hard_block_exports = xpolicy.get("hard_block_exports", {})

        if action_intent in hard_block_actions:
            reasons.extend(hard_block_actions.get(action_intent, []))
            outcome = "BLOCK_ACTION"
        elif action_intent in hard_block_exports:
            reasons.extend(hard_block_exports.get(action_intent, []))
            outcome = "BLOCK_EXPORT"
        elif action_intent in xpolicy.get("escalation_actions", []):
            reasons.append("policy_requires_escalation")
            outcome = "ESCALATE"
        else:
            disclosure_required = action_intent in xpolicy.get("disclosure_required_actions", [])
            limits = xpolicy.get("limits_by_action", {}).get(action_intent)
            if disclosure_required:
                outcome = "ALLOW_WITH_DISCLOSURE"
            elif limits:
                outcome = "ALLOW_WITH_LIMITS"
            else:
                outcome = "ALLOW"

    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"invalid outcome generated: {outcome!r}")

    blocked_action = outcome == "BLOCK_ACTION"
    blocked_export = outcome == "BLOCK_EXPORT"

    payload = {
        "xorder_id": f"xord_{uuid4().hex}",
        "action_intent": action_intent,
        "policy_version": policy_version,
        "resolver_version": resolver_version,
        "order_result": outcome,
        "disclosure_required": 1 if disclosure_required else 0,
        "limits": limits,
        "blocked_action": 1 if blocked_action else 0,
        "blocked_export": 1 if blocked_export else 0,
        "reason_codes": reasons,
        "projection_set_hash": content_hash(projection_set),
        "tvc_version": (tiered_verification_contract or {}).get("contract_version"),
    }
    payload["content_hash"] = content_hash(payload)
    return payload
