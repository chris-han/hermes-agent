from decimal import Decimal, InvalidOperation

# Patent evidence C001: tiered validation gate, including structural REA checks,
# semantic matrix validation, required-artifact / evidence checks, and projection
# scope checks referenced in the admissibility-governance method claims.

# Policy bundles with declarative rules (OPA-style)
POLICY_BUNDLES = {
    "rb_v1": {
        "require_invoice": True,
        "rules": {
            "min_resources": 2,
            "enforce_conservation": True,
            "enforce_currency_consistency": True,
            "enforce_counterparty_consistency": True
        }
    },
    "rb_v2": {
        "require_invoice": False,
        "rules": {
            "min_resources": 2,
            "enforce_conservation": True,
            "enforce_currency_consistency": False,
            "enforce_counterparty_consistency": True
        }
    },
}


def _to_decimal(x):
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError):
        return None


def _matrix_allows(semantic_matrix: dict, authority_domain: str, authority_level: str, semantic_tier: str) -> bool:
    allowed = semantic_matrix.get(authority_domain) or {}
    allowed_levels = set(allowed.get("tag_authority_level", []))
    allowed_tiers = set(allowed.get("semantic_tier", []))
    return authority_level in allowed_levels and semantic_tier in allowed_tiers


def _validate_structural_rea(justification: dict, policy_version="rb_v1"):
    """Internal REA structural checks used by strict v5 tiered validation."""
    if isinstance(policy_version, dict):
        policy = policy_version
    else:
        policy = POLICY_BUNDLES.get(policy_version, POLICY_BUNDLES["rb_v1"])
    rules = policy.get("rules", {})

    rea = justification.get("semantic_claim") or {}
    resources = rea.get("resources") or []
    duality_ref = (rea.get("duality") or {}).get("settles")
    counterparties = rea.get("counterparties") or {}

    if policy.get("require_invoice") and not duality_ref:
        return {
            "status": "EXCEPTION",
            "reasons": ["missing_invoice_reference"],
        }

    if len(resources) < rules.get("min_resources", 2):
        return {
            "status": "EXCEPTION",
            "reasons": ["insufficient_resources"],
        }

    total = Decimal("0")
    currencies = set()

    for r in resources:
        d = _to_decimal(r.get("delta"))
        if d is None:
            return {"status": "REJECT", "reasons": ["invalid_decimal"]}

        total += d

        c = r.get("currency")
        if c:
            currencies.add(c)

    if rules.get("enforce_conservation") and total != 0:
        return {"status": "REJECT", "reasons": ["not_conserved"]}

    if rules.get("enforce_currency_consistency") and len(currencies) > 1:
        return {"status": "REJECT", "reasons": ["currency_mismatch"]}

    if (
        rules.get("enforce_counterparty_consistency")
        and counterparties.get("invoice")
        and counterparties.get("payment")
        and counterparties.get("invoice") != counterparties.get("payment")
    ):
        return {"status": "REJECT", "reasons": ["counterparty_mismatch"]}

    return {"status": "ALLOW", "reasons": []}


def validate_tiered(
    justification: dict,
    policy_version="rb_v1",
    O_tag_v: dict | None = None,
    K_v: dict | None = None,
    Pi_v: dict | None = None,
    ConstraintBundle_v: dict | None = None,
    PrecedenceGraph_v: dict | None = None,
    ResolutionRules_v: dict | None = None,
) -> dict:
    """Tiered KGL-bound gate:
    validate_tiered(J_t, O_v, C_v, O_tag,v, K_v, Π_v, ConstraintBundle_v, PrecedenceGraph_v, ResolutionRules_v).

    This gate is strict: all governed artifacts must be supplied.
    """
    # Always run structural REA checks first
    structural_result = _validate_structural_rea(justification, policy_version)
    if structural_result["status"] != "ALLOW":
        return structural_result

    constraint_results = []
    required_artifacts = {
        "O_tag_v": O_tag_v,
        "K_v": K_v,
        "Pi_v": Pi_v,
        "ConstraintBundle_v": ConstraintBundle_v,
        "PrecedenceGraph_v": PrecedenceGraph_v,
        "ResolutionRules_v": ResolutionRules_v,
    }
    missing_artifacts = [name for name, value in required_artifacts.items() if value is None]
    if missing_artifacts:
        return {
            "status": "REJECT",
            "reasons": [f"missing_tiered_artifact:{name}" for name in missing_artifacts],
            "constraint_results": [
                {
                    "claim_id": None,
                    "constraint_id": f"missing_tiered_artifact:{name}",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": "compliance",
                    "reason": f"Required artifact missing: {name}",
                }
                for name in missing_artifacts
            ],
        }

    reasons = []
    pins = justification.get("pins") or {}
    claims = justification.get("claims") or []
    evidence = justification.get("evidence") or []
    evidence_types = {item.get("type") for item in evidence if item.get("type")}

    # Core governed pin presence
    required_pins = [
        "O_v",
        "C_v",
        "O_tag_v",
        "K_v",
        "Pi_v",
        "ConstraintBundle_v",
        "PrecedenceGraph_v",
        "ResolutionRules_v",
    ]
    for pin in required_pins:
        if not pins.get(pin):
            reasons.append(f"missing_pin:{pin}")
            constraint_results.append(
                {
                    "claim_id": None,
                    "constraint_id": f"pin_required:{pin}",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": "compliance",
                    "reason": f"Missing governance pin {pin}",
                }
            )

    # Claim-level semantic matrix validation
    semantic_matrix = (O_tag_v or {}).get("semantic_matrix", {})
    if not isinstance(semantic_matrix, dict):
        reasons.append("invalid_O_tag_v:semantic_matrix_missing")
    if not O_tag_v.get("version"):
        reasons.append("invalid_O_tag_v:version_missing")

    precedence_edges = PrecedenceGraph_v.get("edges", [])
    # Compatibility note: runtime enforcement here validates the semantic authority axis (T0..T6).
    # Lifecycle/maturity axis (C0..C3) is currently a spec-level model and not enforced in this validator.
    declared_tiers = {"T0", "T1", "T2", "T3", "T4", "T5", "T6"}
    for edge in precedence_edges:
        if edge.get("from") not in declared_tiers or edge.get("to") not in declared_tiers:
            reasons.append("invalid_precedence_graph:unknown_tier")

    rules = ResolutionRules_v.get("rules", [])
    scope_rules = set(ResolutionRules_v.get("scopes", []))
    if scope_rules and not {"execution", "projection"}.issubset(scope_rules):
        reasons.append("invalid_resolution_rules:missing_required_scope")

    for idx, claim in enumerate(claims):
        authority_domain = claim.get("authority_domain")
        authority_level = claim.get("tag_authority_level")
        semantic_tier = claim.get("semantic_tier")
        source_type = claim.get("source_type")

        if not authority_domain:
            reasons.append(f"missing_claim_field:{idx}:authority_domain")
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "claim_field_required:authority_domain",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": "compliance",
                    "reason": "Claim missing authority_domain",
                }
            )
            continue
        if not authority_level:
            reasons.append(f"missing_claim_field:{idx}:tag_authority_level")
        if not semantic_tier:
            reasons.append(f"missing_claim_field:{idx}:semantic_tier")
        if not source_type:
            reasons.append(f"missing_claim_field:{idx}:source_type")

        allowed = semantic_matrix.get(authority_domain)
        if not allowed:
            reasons.append(f"unknown_authority_domain:{authority_domain}")
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "authority_domain_known",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": authority_domain,
                    "reason": f"Unknown authority_domain {authority_domain}",
                }
            )
            continue

        allowed_levels = set(allowed.get("tag_authority_level", []))
        allowed_tiers = set(allowed.get("semantic_tier", []))

        if authority_level and authority_level not in allowed_levels:
            reasons.append(
                f"invalid_authority_level:{authority_domain}:{authority_level}"
            )
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "matrix_authority_level",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": authority_domain,
                    "reason": f"Invalid tag_authority_level {authority_level}",
                }
            )
        if semantic_tier and semantic_tier not in allowed_tiers:
            reasons.append(
                f"invalid_semantic_tier:{authority_domain}:{semantic_tier}"
            )
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "matrix_semantic_tier",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": authority_domain,
                    "reason": f"Invalid semantic_tier {semantic_tier}",
                }
            )

        # Hard-invalid combos from architecture
        if authority_domain in {"tax", "accounting"} and semantic_tier == "T6":
            reasons.append(f"invalid_combo:{authority_domain}:T6")
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "hard_invalid_combo",
                    "scope": "governance",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": authority_domain,
                    "reason": "Tax/accounting claims cannot be T6",
                }
            )
        if authority_domain == "management" and semantic_tier == "T2":
            reasons.append("invalid_combo:management:T2")
        if authority_domain == "internal_control" and semantic_tier == "T2":
            reasons.append("invalid_combo:internal_control:T2")
        if source_type == "agent" and semantic_tier in {"T0", "T2", "T3", "T4"}:
            reasons.append("agent_source_requires_escalation")
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "agent_source_escalation",
                    "scope": "governance",
                    "decision": "ESCALATE",
                    "tier": semantic_tier,
                    "authority_domain": authority_domain,
                    "reason": "Agent-origin claim requires escalation",
                }
            )

        proposed_target = claim.get("proposes")
        if proposed_target:
            target_domain = proposed_target.get("authority_domain")
            target_level = proposed_target.get("tag_authority_level")
            target_tier = proposed_target.get("semantic_tier")
            if authority_domain != "agent_suggestion" or semantic_tier != "T6":
                reasons.append("invalid_proposal_source_shape")
                constraint_results.append(
                    {
                        "claim_id": claim.get("id"),
                        "constraint_id": "proposal_source_shape",
                        "scope": "governance",
                        "decision": "REJECT",
                        "tier": "T2",
                        "authority_domain": authority_domain,
                        "reason": "Only T6 agent_suggestion claims may carry proposes",
                    }
                )
            elif not all([target_domain, target_level, target_tier]):
                reasons.append("invalid_proposal_target:missing_fields")
            elif not _matrix_allows(semantic_matrix, target_domain, target_level, target_tier):
                reasons.append("invalid_proposal_target:matrix")
                constraint_results.append(
                    {
                        "claim_id": claim.get("id"),
                        "constraint_id": "proposal_target_matrix",
                        "scope": "governance",
                        "decision": "REJECT",
                        "tier": "T2",
                        "authority_domain": target_domain,
                        "reason": "Proposed governed target is not allowed by semantic matrix",
                    }
                )
            else:
                constraint_results.append(
                    {
                        "claim_id": claim.get("id"),
                        "constraint_id": "proposal_target_valid",
                        "scope": "governance",
                        "decision": "ALLOW",
                        "tier": target_tier,
                        "authority_domain": target_domain,
                        "reason": "Proposal target is semantically valid",
                    }
                )

    # K_v checks: governed knowledge context
    required_k_keys = K_v.get("required_claim_fields", [])
    bindings = justification.get("bindings") or {}
    for key in required_k_keys:
        if not bindings.get(key):
            reasons.append(f"missing_required_field:{key}")
            constraint_results.append(
                {
                    "claim_id": None,
                    "constraint_id": f"required_binding:{key}",
                    "scope": "execution",
                    "decision": "REJECT",
                    "tier": "T4",
                    "authority_domain": "internal_control",
                    "reason": f"Missing required field {key}",
                }
            )

    forbidden_types = set(K_v.get("forbidden_resource_types", []))
    rea = justification.get("semantic_claim") or {}
    for r in rea.get("resources", []):
        if r.get("type") in forbidden_types:
            reasons.append(f"forbidden_resource_type:{r.get('type')}")
            constraint_results.append(
                {
                    "claim_id": None,
                    "constraint_id": f"forbidden_resource_type:{r.get('type')}",
                    "scope": "execution",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": "compliance",
                    "reason": f"Resource type forbidden: {r.get('type')}",
                }
            )

    # Pi_v checks: projection bundle consistency
    expected_bundle = Pi_v.get("bundle_id")
    if expected_bundle and pins.get("Pi_v") != expected_bundle:
        reasons.append(
            f"projection_bundle_mismatch:expected={expected_bundle}"
            f",got={pins.get('Pi_v')}"
        )
        constraint_results.append(
            {
                "claim_id": None,
                "constraint_id": "projection_bundle_match",
                "scope": "projection",
                "decision": "REJECT",
                "tier": "T4",
                "authority_domain": "operations",
                "reason": "Projection bundle mismatch",
            }
        )
    projection_views = Pi_v.get("views", {})
    domain_scopes = Pi_v.get("domain_scopes", {})
    if projection_views and not isinstance(projection_views, dict):
        reasons.append("invalid_Pi_v:views_not_dict")
    if domain_scopes and not isinstance(domain_scopes, dict):
        reasons.append("invalid_Pi_v:domain_scopes_not_dict")
    for claim in claims:
        proposed_target = claim.get("proposes") or {}
        target_domain = proposed_target.get("authority_domain") or claim.get("authority_domain")
        if target_domain in domain_scopes:
            continue
        if projection_views and target_domain not in projection_views and target_domain != "agent_suggestion":
            reasons.append(f"missing_projection_scope:{target_domain}")
            constraint_results.append(
                {
                    "claim_id": claim.get("id"),
                    "constraint_id": "projection_scope_required",
                    "scope": "projection",
                    "decision": "ESCALATE",
                    "tier": "T4",
                    "authority_domain": "operations",
                    "reason": f"No projection scope configured for {target_domain}",
                }
            )

    # ConstraintBundle-v declared required checks
    required_constraints = ConstraintBundle_v.get("required_bindings", [])
    for key in required_constraints:
        if not bindings.get(key):
            reasons.append(f"missing_constraint_binding:{key}")
            constraint_results.append(
                {
                    "claim_id": None,
                    "constraint_id": f"constraint_binding:{key}",
                    "scope": "execution",
                    "decision": "REJECT",
                    "tier": "T4",
                    "authority_domain": "internal_control",
                    "reason": f"Missing constraint binding {key}",
                }
            )
    for evidence_type in ConstraintBundle_v.get("required_evidence_types", []):
        if evidence_type not in evidence_types:
            reasons.append(f"missing_required_evidence:{evidence_type}")
            constraint_results.append(
                {
                    "claim_id": None,
                    "constraint_id": f"required_evidence:{evidence_type}",
                    "scope": "evidence",
                    "decision": "REJECT",
                    "tier": "T2",
                    "authority_domain": "compliance",
                    "reason": f"Missing required evidence type {evidence_type}",
                }
            )
    for domain, keys in (ConstraintBundle_v.get("domain_required_bindings", {}) or {}).items():
        domain_claims = [
            claim for claim in claims
            if claim.get("authority_domain") == domain
            or (claim.get("proposes") or {}).get("authority_domain") == domain
        ]
        if not domain_claims:
            continue
        for key in keys:
            if not bindings.get(key):
                reasons.append(f"missing_domain_binding:{domain}:{key}")
                constraint_results.append(
                    {
                        "claim_id": None,
                        "constraint_id": f"domain_binding:{domain}:{key}",
                        "scope": "execution",
                        "decision": "REJECT",
                        "tier": "T4",
                        "authority_domain": domain,
                        "reason": f"Missing binding {key} for authority domain {domain}",
                    }
                )

    # Governance consistency checks for precedence/rules presence
    if not PrecedenceGraph_v.get("edges"):
        reasons.append("invalid_precedence_graph:missing_edges")
    if not ResolutionRules_v.get("rules"):
        reasons.append("invalid_resolution_rules:missing_rules")

    if reasons:
        status = "EXCEPTION"
        if any(r.startswith("missing_tiered_artifact:") or r.startswith("invalid_combo:") for r in reasons):
            status = "REJECT"
        return {"status": status, "reasons": reasons, "constraint_results": constraint_results}

    if not constraint_results:
        constraint_results.append(
            {
                "claim_id": None,
                "constraint_id": "tiered_validation_passed",
                "scope": "execution",
                "decision": "ALLOW",
                "tier": "T4",
                "authority_domain": "internal_control",
                "reason": "Tiered validation passed",
            }
        )
    return {"status": "ALLOW", "reasons": [], "constraint_results": constraint_results}


def default_v5_artifacts() -> dict:
    return {
        "O_tag_v": {
            "version": "otag_v1",
            "semantic_matrix": {
                "factual_evidence": {
                    "tag_authority_level": ["descriptive"],
                    "semantic_tier": ["T0"],
                },
                "ontology": {
                    "tag_authority_level": ["descriptive", "operational"],
                    "semantic_tier": ["T1"],
                },
                "agent_suggestion": {
                    "tag_authority_level": ["descriptive"],
                    "semantic_tier": ["T6"],
                },
                "tax": {
                    "tag_authority_level": ["regulatory", "accounting"],
                    "semantic_tier": ["T2", "T3"],
                },
                "accounting": {
                    "tag_authority_level": ["accounting", "regulatory"],
                    "semantic_tier": ["T3", "T2"],
                },
                "management": {
                    "tag_authority_level": ["management", "operational"],
                    "semantic_tier": ["T5", "T4"],
                },
                "internal_control": {
                    "tag_authority_level": ["operational"],
                    "semantic_tier": ["T4"],
                },
                "compliance": {
                    "tag_authority_level": ["regulatory", "operational"],
                    "semantic_tier": ["T2", "T4"],
                },
                "risk": {
                    "tag_authority_level": ["operational", "regulatory"],
                    "semantic_tier": ["T4", "T2"],
                },
                "operations": {
                    "tag_authority_level": ["operational", "descriptive"],
                    "semantic_tier": ["T4", "T5"],
                },
            },
        },
        "K_v": {"version": "k_v1", "required_claim_fields": [], "forbidden_resource_types": []},
        "Pi_v": {"bundle_id": "pi_v1"},
        "ConstraintBundle_v": {"version": "cb_v1", "required_bindings": []},
        "PrecedenceGraph_v": {"version": "pg_v1", "edges": [{"from": "T2", "to": "T3"}]},
        "ResolutionRules_v": {"version": "rr_v1", "rules": [{"id": "default"}]},
    }


def validate_v5(justification: dict, policy_version="rb_v1", artifacts: dict | None = None) -> dict:
    """v5 runtime validation entrypoint (tiered-only)."""
    a = artifacts or default_v5_artifacts()
    j = dict(justification)
    pins = dict(j.get("pins") or {})
    pins.setdefault("O_v", "v2026-05-02")
    pins.setdefault("C_v", policy_version if isinstance(policy_version, str) else "rb_v1")
    pins.setdefault("O_tag_v", a["O_tag_v"].get("version"))
    pins.setdefault("K_v", a["K_v"].get("version"))
    pins.setdefault("Pi_v", a["Pi_v"].get("bundle_id"))
    pins.setdefault("ConstraintBundle_v", a["ConstraintBundle_v"].get("version"))
    pins.setdefault("PrecedenceGraph_v", a["PrecedenceGraph_v"].get("version"))
    pins.setdefault("ResolutionRules_v", a["ResolutionRules_v"].get("version"))
    j["pins"] = pins
    claims = list(j.get("claims") or [])
    if not claims:
        claims = [
            {
                "id": "claim:factual_evidence",
                "key": "factual_evidence.present",
                "value": True,
                "authority_domain": "factual_evidence",
                "tag_authority_level": "descriptive",
                "semantic_tier": "T0",
                "source_type": "system",
                "status": "ACTIVE",
            }
        ]
    j["claims"] = claims
    return validate_tiered(
        j,
        policy_version,
        O_tag_v=a["O_tag_v"],
        K_v=a["K_v"],
        Pi_v=a["Pi_v"],
        ConstraintBundle_v=a["ConstraintBundle_v"],
        PrecedenceGraph_v=a["PrecedenceGraph_v"],
        ResolutionRules_v=a["ResolutionRules_v"],
    )
