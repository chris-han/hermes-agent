"""Semantic Completion Service implementing PRD.

Enables users to record financial events via natural language in Weixin,
where the system interactively completes missing semantic requirements
(e.g., invoice) before validation and execution.

Flow:
  NL Input → Tag Extraction → Projection Π → Detect Missing Fields → Ask User
  → Receive Input → Update Draft → Build Justification → Validate → Commit
"""


from storage import hosted_postgres as hosted

import json
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso, init_db
from eos.explainability_surface import build_explainability_surface
from eos.jurisdiction_classifier import classify_proposal_jurisdiction
from eos.memory_adapter import SemantierMemoryAdapter
from eos.projection_exception_store import create_projection_exception
from eos.verification_artifacts import persist_explainability_surface
from eos.validation_service import validate_tiered as eos_validate_tiered
from eos.resolution_engine import run_resolution as eos_resolve
from eos.projection_engine import run_projection as project_rea_to_ledger
from eos.replay_binding_store import create_replay_binding
from memory.holographic_provider import InMemoryHolographicProvider
from storage import (
    DraftStore,
    UnifiedEventStore,
    UnifiedIdempotencyStore,
    SemanticDraft,
    SemanticDraftStatus,
    ValidationResult,
    ValidationStatus,
    get_event_store,
    get_idempotency_store,
)
from storage.governance_store import GovernanceStore


class SemanticCompletionStep(str, Enum):
    """Semantic completion flow steps."""
    EXTRACT_TAGS = "EXTRACT_TAGS"
    PROJECT_FIELDS = "PROJECT_FIELDS"
    DETECT_MISSING = "DETECT_MISSING"
    ASK_USER = "ASK_USER"
    RECEIVE_INPUT = "RECEIVE_INPUT"
    UPDATE_DRAFT = "UPDATE_DRAFT"
    BUILD_JUSTIFICATION = "BUILD_JUSTIFICATION"
    VALIDATE = "VALIDATE"
    COMMIT = "COMMIT"


# Required fields for financial transactions per policy
REQUIRED_FIELDS = {
    "amount": "Financial amount",
    "currency": "Currency code",
    "counterparty": "Transaction counterparty",
}

OPTIONAL_FIELDS = {
    "invoice_ref": "Invoice reference (for settlement)",
    "description": "Transaction description",
}

_FIELD_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "semantic_completion"


def _load_field_prompt(field_name: str) -> str:
    return (_FIELD_PROMPTS_DIR / f"{field_name}.md").read_text(encoding="utf-8").strip()

# Field query prompts
FIELD_PROMPTS = {
    "amount": _load_field_prompt("amount"),
    "currency": _load_field_prompt("currency"),
    "counterparty": _load_field_prompt("counterparty"),
    "invoice_ref": _load_field_prompt("invoice_ref"),
    "description": _load_field_prompt("description"),
}


class SemanticCompletionService:
    """Service for interactive semantic completion.

    Per PRD:
    - User claims are not evidence. Missing semantic requirements must be
      completed interactively before validation.
    - Governance context (O_v, C_v; K_v, Pi_v for tiered) is pinned at
      completion time for replay determinism (FR7).
    - v5 tiered gate is selected per conversation (FR8).
    - projection_context_hash is persisted after commit (FR9).
    """

    # Default ontology and policy bundle versions (overridable per conversation)
    DEFAULT_O_V = "v2026-05-02"
    DEFAULT_C_V = "rb_v1"
    DEFAULT_O_TAG_V = {
        "version": "otag_v1",
        "semantic_matrix": {
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
        },
    }
    DEFAULT_K_V = {"version": "k_v1", "required_claim_fields": [], "forbidden_resource_types": []}
    DEFAULT_PI_V = {"bundle_id": "pi_v1"}
    DEFAULT_CONSTRAINT_BUNDLE_V = {"version": "cb_v1", "required_bindings": []}
    DEFAULT_PRECEDENCE_GRAPH_V = {"version": "pg_v1", "edges": [{"from": "T2", "to": "T3"}]}
    GOVERNANCE_FAST_TRACK_ACTOR = "semantic_completion:auto_promote"
    DEFAULT_RESOLUTION_RULES_V = {
        "version": "rr_v1",
        "rules": [{"id": "default"}],
        "auto_promote_proposals": True,
    }

    def __init__(
        self,
        draft_store: DraftStore,
        idempotency_store: Optional[UnifiedIdempotencyStore] = None,
        event_store: Optional[UnifiedEventStore] = None,
    ):
        self.draft_store = draft_store
        self.idempotency_store = idempotency_store or get_idempotency_store()
        self.event_store = event_store
    
    def start_completion(
        self,
        org_id: str,
        user_input: str,
        channel: str = "weixin",
        gate_mode: str = "tiered",
        O_tag_v: Optional[Dict[str, Any]] = None,
        K_v: Optional[Dict[str, Any]] = None,
        Pi_v: Optional[Dict[str, Any]] = None,
        ConstraintBundle_v: Optional[Dict[str, Any]] = None,
        PrecedenceGraph_v: Optional[Dict[str, Any]] = None,
        ResolutionRules_v: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SemanticDraft, str]:
        """Start semantic completion flow.

        Args:
            org_id: Organization ID
            user_input: User's natural language input
            channel: Input channel (weixin, etc.)
            gate_mode: v5 expects 'tiered'; 'base' is rejected.
            K_v: Governed knowledge context dict for tiered gate (FR7).
            Pi_v: Projection bundle dict for tiered gate (FR7).

        Returns:
            (draft, next_question) - Draft and question to ask user
        """
        import hashlib

        conversation_id = str(uuid.uuid4())
        if gate_mode == "tiered":
            O_tag_v = O_tag_v or dict(self.DEFAULT_O_TAG_V)
            K_v = K_v or dict(self.DEFAULT_K_V)
            Pi_v = Pi_v or dict(self.DEFAULT_PI_V)
            ConstraintBundle_v = ConstraintBundle_v or dict(self.DEFAULT_CONSTRAINT_BUNDLE_V)
            PrecedenceGraph_v = PrecedenceGraph_v or dict(self.DEFAULT_PRECEDENCE_GRAPH_V)
            ResolutionRules_v = ResolutionRules_v or dict(self.DEFAULT_RESOLUTION_RULES_V)

        # Step 1: Extract tags from NL input
        tags = self._extract_tags(user_input)

        # FR7: Build and stamp governance pins
        pins = {
            "O_v": self.DEFAULT_O_V,
            "C_v": self.DEFAULT_C_V,
        }
        if K_v is not None:
            pins["K_v"] = K_v.get("version") or "unknown"
        if Pi_v is not None:
            pins["Pi_v"] = Pi_v.get("bundle_id") or "unknown"
        if O_tag_v is not None:
            pins["O_tag_v"] = O_tag_v.get("version") or "unknown"
        if ConstraintBundle_v is not None:
            pins["ConstraintBundle_v"] = ConstraintBundle_v.get("version") or "unknown"
        if PrecedenceGraph_v is not None:
            pins["PrecedenceGraph_v"] = PrecedenceGraph_v.get("version") or "unknown"
        if ResolutionRules_v is not None:
            pins["ResolutionRules_v"] = ResolutionRules_v.get("version") or "unknown"

        # Create draft
        draft = self.draft_store.create_draft(
            conversation_id=conversation_id,
            org_id=org_id,
            initial_payload={
                "channel": channel,
                "user_input": user_input,
                "tags": tags,
                "O_tag_v": O_tag_v,
                "K_v": K_v,
                "Pi_v": Pi_v,
                "ConstraintBundle_v": ConstraintBundle_v,
                "PrecedenceGraph_v": PrecedenceGraph_v,
                "ResolutionRules_v": ResolutionRules_v,
            }
        )
        
        # Store user input as first message
        self.draft_store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
        )
        
        # Update draft with tags and governance context
        draft.tags = tags
        draft.status = SemanticDraftStatus.PENDING_USER_INPUT
        draft.pins = pins
        draft.gate_mode = gate_mode
        draft = self.draft_store.update_draft(draft)
        
        # Step 2: Project fields and detect missing
        missing = self._detect_missing_fields(draft)
        draft.missing_fields = missing
        draft = self.draft_store.update_draft(draft)
        
        # Step 3: Ask for first missing field
        if missing:
            next_question = self._build_question(missing[0])
        else:
            # All required fields are present - ready for validation
            draft.status = SemanticDraftStatus.COMPLETE
            draft = self.draft_store.update_draft(draft)
            next_question = "✅ 已完成信息收集，准备验证 (Ready for validation)"
        
        # Store system message
        self.draft_store.add_message(
            conversation_id=conversation_id,
            role="system",
            content=next_question,
        )

        self._record_t6_capture_events(
            org_id=org_id,
            conversation_id=conversation_id,
            channel=channel,
            user_input=user_input,
            tags=tags,
            missing_fields=missing,
        )
        
        return draft, next_question

    def _record_t6_capture_events(
        self,
        *,
        org_id: str,
        conversation_id: str,
        channel: str,
        user_input: str,
        tags: List[Any],
        missing_fields: List[str],
    ) -> None:
        if self.event_store is None:
            return

        capture_event_id = f"{conversation_id}:t6_capture"
        capture_payload = {
            "conversation_id": conversation_id,
            "channel": channel,
            "intent_text": user_input,
            "tags": tags,
            "semantic_tier": "T6",
        }
        capture_event = self.event_store.append_event(
            event_id=capture_event_id,
            event_type="T6_CAPTURE",
            org_id=org_id,
            correlation_id=conversation_id,
            payload=capture_payload,
        )

        self.event_store.append_event(
            event_id=f"{conversation_id}:audit_recorded",
            event_type="AUDIT_RECORDED",
            org_id=org_id,
            correlation_id=conversation_id,
            payload={
                "conversation_id": conversation_id,
                "channel": channel,
                "captured_event_id": capture_event.event_id,
                "captured_event_hash": capture_event.content_hash,
                "missing_fields": missing_fields,
            },
        )
    
    def continue_completion(
        self,
        conversation_id: str,
        user_input: str,
    ) -> Tuple[SemanticDraft, str]:
        """Continue semantic completion with user input.
        
        Args:
            conversation_id: Conversation ID
            user_input: User's input
        
        Returns:
            (draft, next_question_or_status) - Updated draft and response
        """
        draft = self.draft_store.get_draft(conversation_id)
        if not draft:
            raise ValueError(f"Draft not found: {conversation_id}")
        
        # Store user message
        self.draft_store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
        )
        
        # Step 5: Receive input and update draft
        self._update_draft_from_input(draft, user_input)
        draft = self.draft_store.update_draft(draft)
        
        # Step 6: Check if more fields missing
        missing = self._detect_missing_fields(draft)
        draft.missing_fields = missing
        draft = self.draft_store.update_draft(draft)
        
        if missing:
            # Ask for next missing field
            next_question = self._build_question(missing[0])
            response = next_question
        else:
            # All required fields complete
            draft.status = SemanticDraftStatus.COMPLETE
            draft = self.draft_store.update_draft(draft)
            response = "✅ 已入账 (Recorded)"
        
        # Store system message
        self.draft_store.add_message(
            conversation_id=conversation_id,
            role="system",
            content=response,
        )
        
        return draft, response
    
    def build_justification(
        self,
        draft: SemanticDraft,
    ) -> Dict[str, Any]:
        """Build justification from complete draft.
        
        Step 7: Build Justification (J_t)
        
        Only called when draft is COMPLETE.
        """
        if draft.status != SemanticDraftStatus.COMPLETE:
            raise ValueError(f"Draft not complete: {draft.status}")

        # Build REA resources satisfying conservation law: sum(deltas) == 0
        try:
            amount_val = Decimal(str(draft.amount))
            credit_delta = str(-amount_val)
        except (InvalidOperation, TypeError):
            credit_delta = f"-{draft.amount}"

        semantic_claim = {
            "type": "financial_event",
            "resources": [
                {
                    "type": "expense",
                    "delta": str(draft.amount),
                    "currency": draft.currency,
                    "account": "expense",
                },
                {
                    "type": "cash",
                    "delta": credit_delta,
                    "currency": draft.currency,
                    "account": "cash",
                },
            ],
            "counterparties": {
                "payment": draft.counterparty,
                "invoice": draft.counterparty,
            }
        }

        if draft.invoice_ref:
            semantic_claim["duality"] = {
                "settles": draft.invoice_ref,
            }

        # FR7: Carry persisted governance pins forward into J_t
        pins = dict(draft.pins) if draft.pins else {}
        pins.setdefault("O_v", self.DEFAULT_O_V)
        pins.setdefault("C_v", self.DEFAULT_C_V)
        # Legacy keys for backward compat with existing tests
        pins.setdefault("ontology_version", pins["O_v"])
        pins.setdefault("projection_version", 1)

        # Build justification
        justification = {
            "org_id": draft.org_id,
            "intent_id": draft.conversation_id,
            "claims": [self._build_claim_from_tag(tag, draft, pins) for tag in draft.tags],
            "evidence": [
                {
                    "evidence_id": ref,
                    "type": "uploaded",
                } for ref in draft.evidence_refs
            ],
            "semantic_claim": semantic_claim,
            "bindings": {
                "amount": draft.amount,
                "currency": draft.currency,
                "counterparty": draft.counterparty,
            },
            "pins": pins,
        }
        
        # Compute content hash for determinism
        from eos.artifact_hashing import content_hash

        justification["content_hash"] = content_hash(justification)
        
        return justification

    def _build_claim_from_tag(
        self,
        raw_tag: Any,
        draft: SemanticDraft,
        pins: Dict[str, Any],
    ) -> Dict[str, Any]:
        ontology_pins = {
            "O_v": pins.get("O_v"),
            "O_tag_v": pins.get("O_tag_v"),
            "K_v": pins.get("K_v"),
        }
        if isinstance(raw_tag, dict):
            claim = {
                "id": raw_tag.get("id") or f"claim:{raw_tag.get('key', 'tag')}",
                "type": raw_tag.get("type", "tag"),
                "key": raw_tag.get("key", "tag"),
                "value": raw_tag.get("value", True),
                "authority_domain": raw_tag.get("authority_domain", "agent_suggestion"),
                "tag_authority_level": raw_tag.get("tag_authority_level", "descriptive"),
                "semantic_tier": raw_tag.get("semantic_tier", "T6"),
                "source_type": raw_tag.get("source_type", "agent"),
                "evidence_refs": list(raw_tag.get("evidence_refs", draft.evidence_refs)),
                "ontology_pins": dict(raw_tag.get("ontology_pins", ontology_pins)),
                "status": raw_tag.get("status", "PROPOSED"),
            }
            if "proposes" in raw_tag:
                claim["proposes"] = raw_tag["proposes"]
            return claim

        return {
            "id": f"claim:{raw_tag}",
            "type": "tag",
            "key": raw_tag,
            "value": True,
            "authority_domain": "agent_suggestion",
            "tag_authority_level": "descriptive",
            "semantic_tier": "T6",
            "source_type": "agent",
            "evidence_refs": list(draft.evidence_refs),
            "ontology_pins": ontology_pins,
            "status": "PROPOSED",
        }

    def _derive_projection_scope(
        self,
        authority_domain: str,
        Pi_v: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if not Pi_v:
            return None
        domain_scopes = Pi_v.get("domain_scopes", {})
        if isinstance(domain_scopes, dict) and authority_domain in domain_scopes:
            return domain_scopes[authority_domain]
        views = Pi_v.get("views", {})
        if isinstance(views, dict) and authority_domain in views:
            view = views.get(authority_domain) or {}
            return view.get("projection_scope", authority_domain)
        return Pi_v.get("projection_scope") or Pi_v.get("bundle_id")

    def _materialize_committed_claims(
        self,
        justification: Dict[str, Any],
        resolution_result: Dict[str, Any],
        Pi_v: Optional[Dict[str, Any]],
        ResolutionRules_v: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        auto_promote = bool((ResolutionRules_v or {}).get("auto_promote_proposals"))
        winning_ids = {
            claim.get("id")
            for claim in resolution_result.get("winning_claims", [])
        }
        committed_claims: List[Dict[str, Any]] = []

        for claim in justification.get("claims", []):
            proposal = dict(claim)
            if claim.get("id") in winning_ids and claim.get("semantic_tier") != "T6":
                proposal["status"] = "ACTIVE"
                proposal["projection_scope"] = (
                    proposal.get("projection_scope")
                    or self._derive_projection_scope(proposal.get("authority_domain", ""), Pi_v)
                )
            committed_claims.append(proposal)

            proposed_target = claim.get("proposes")
            if not proposed_target or claim.get("id") not in winning_ids or not auto_promote:
                continue

            promoted_claim = {
                "id": f"{claim.get('id')}:active",
                "type": claim.get("type", "tag"),
                "key": claim.get("key"),
                "value": claim.get("value"),
                "authority_domain": proposed_target.get("authority_domain"),
                "tag_authority_level": proposed_target.get("tag_authority_level"),
                "semantic_tier": proposed_target.get("semantic_tier"),
                "source_type": "governance",
                "evidence_refs": list(claim.get("evidence_refs", [])),
                "ontology_pins": dict(claim.get("ontology_pins", {})),
                "status": "ACTIVE",
                "projection_scope": self._derive_projection_scope(
                    proposed_target.get("authority_domain", ""),
                    Pi_v,
                ),
                "promoted_from": claim.get("id"),
                "promotion_context": {
                    "resolution_rules_version": (ResolutionRules_v or {}).get("version"),
                    "proposal_claim_id": claim.get("id"),
                },
            }
            committed_claims.append(promoted_claim)

        return committed_claims

    def _resolve_projection_bundle(
        self,
        draft: SemanticDraft,
    ) -> Optional[Dict[str, Any]]:
        payload = draft.payload or {}
        Pi_v = payload.get("Pi_v")
        if isinstance(Pi_v, dict) and (Pi_v.get("codes") or Pi_v.get("views")):
            return Pi_v

        gov_store = GovernanceStore()
        pinned_bundle_version = None
        if isinstance(Pi_v, dict):
            pinned_bundle_version = Pi_v.get("coa_bundle_version") or Pi_v.get("bundle_version")
        if pinned_bundle_version:
            pinned_bundle = gov_store.get_coa_bundle(str(pinned_bundle_version))
            if pinned_bundle:
                return pinned_bundle

        active_bundle = gov_store.get_active_coa_bundle()
        if active_bundle:
            return active_bundle

        active_coa_row = gov_store.get_active("coa_version")
        if not active_coa_row:
            return None
        return json.loads(active_coa_row.get("content_yaml", "{}"))

    def _govern_promoted_claims(
        self,
        *,
        draft: SemanticDraft,
        committed_claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Persist fast-track T5 promotions and mark higher-tier promotions for review."""
        governance_actions: List[Dict[str, Any]] = []
        gov_store = GovernanceStore()

        for claim in committed_claims:
            if "promoted_from" not in claim:
                continue

            target_tier = str(claim.get("semantic_tier") or "")
            target_domain = str(claim.get("authority_domain") or "unknown")
            claim_key = str(claim.get("key") or "unnamed")
            jurisdiction_review = classify_proposal_jurisdiction(
                authority_domain=target_domain,
                scope="organization" if target_domain in {"tax", "accounting", "compliance", "internal_control"} else "personal",
                affects_approval_logic=target_domain == "internal_control" or target_tier in {"T2", "T3", "T4"},
                affects_projection_meaning=target_tier == "T5",
                source_kind="t6_promotion",
            )
            target_tier = str(jurisdiction_review["target_tier"])
            claim["jurisdiction_review"] = jurisdiction_review

            if not jurisdiction_review["fast_track_allowed"]:
                claim["status"] = "REVIEW_REQUIRED"
                claim["governance_status"] = "REVIEW_REQUIRED"
                claim["context_only"] = True
                governance_actions.append(
                    {
                        "claim_id": claim.get("id"),
                        "decision": "REVIEW_REQUIRED",
                        "semantic_tier": target_tier,
                        "reason": jurisdiction_review["reason"],
                        "jurisdiction_review": jurisdiction_review,
                    }
                )
                continue

            candidate_name = f"scoped::{draft.org_id}::{target_domain}::{claim_key}"
            candidate_payload = {
                "org_id": draft.org_id,
                "name": candidate_name,
                "claim": {
                    "key": claim.get("key"),
                    "value": claim.get("value"),
                    "authority_domain": claim.get("authority_domain"),
                    "tag_authority_level": claim.get("tag_authority_level"),
                    "semantic_tier": claim.get("semantic_tier"),
                    "projection_scope": claim.get("projection_scope"),
                },
                "promotion_context": {
                    **dict(claim.get("promotion_context") or {}),
                    "conversation_id": draft.conversation_id,
                    "promoted_from": claim.get("promoted_from"),
                },
                "evidence_refs": list(claim.get("evidence_refs", [])),
                "ontology_pins": dict(claim.get("ontology_pins", {})),
            }
            payload_hash = content_hash(candidate_payload)
            row = gov_store.propose(
                name=candidate_name,
                category="policy_bundle",
                content_yaml=canonical_json(candidate_payload),
                content_hash=payload_hash,
                proposed_by=self.GOVERNANCE_FAST_TRACK_ACTOR,
                semantic_tier="T5",
                scoped_owner=f"org:{draft.org_id}",
                rollback_window_hours=24,
                rollback_strategy="Deactivate candidate and revert to prior scoped operating behavior",
                notes="Auto-promoted from semantic completion winning T6 proposal",
                jurisdiction_review=jurisdiction_review,
            )
            self._record_governance_event(
                org_id=draft.org_id,
                conversation_id=draft.conversation_id,
                channel=str((draft.payload or {}).get("channel") or "unknown"),
                event_type="GOVERNANCE_PROPOSED",
                payload={
                    "candidate_id": row["candidate_id"],
                    "semantic_tier": target_tier,
                    "channel": str((draft.payload or {}).get("channel") or "unknown"),
                    "transport_metadata": {"conversation_id": draft.conversation_id},
                },
            )

            if row["status"] == "PENDING":
                row = gov_store.fast_track_activate(
                    row["candidate_id"],
                    actor=self.GOVERNANCE_FAST_TRACK_ACTOR,
                    actor_roles=("delegated_curator",),
                    notes="Auto-promoted from semantic completion winning T6 proposal",
                )
                decision = "FAST_TRACK_ACTIVATED"
            elif row["status"] == "APPROVED":
                row = gov_store.activate(
                    row["candidate_id"],
                    actor_roles=("delegated_curator",),
                )
                decision = "ACTIVATED_EXISTING_APPROVAL"
            elif row["status"] == "ACTIVE":
                decision = "REUSED_ACTIVE"
            else:
                claim["status"] = "REVIEW_REQUIRED"
                claim["governance_status"] = row["status"]
                claim["context_only"] = True
                governance_actions.append(
                    {
                        "claim_id": claim.get("id"),
                        "decision": "REVIEW_REQUIRED",
                        "semantic_tier": target_tier,
                        "reason": f"existing_candidate_status:{row['status']}",
                        "candidate_id": row["candidate_id"],
                    }
                )
                continue

            claim["governance_candidate_id"] = row["candidate_id"]
            claim["governance_status"] = row["status"]
            self._record_governance_event(
                org_id=draft.org_id,
                conversation_id=draft.conversation_id,
                channel=str((draft.payload or {}).get("channel") or "unknown"),
                event_type="GOVERNANCE_ACTIVATED",
                payload={
                    "candidate_id": row["candidate_id"],
                    "semantic_tier": target_tier,
                    "channel": str((draft.payload or {}).get("channel") or "unknown"),
                    "transport_metadata": {"conversation_id": draft.conversation_id},
                },
            )
            governance_actions.append(
                {
                    "claim_id": claim.get("id"),
                    "decision": decision,
                    "semantic_tier": target_tier,
                    "candidate_id": row["candidate_id"],
                    "candidate_status": row["status"],
                    "jurisdiction_review": jurisdiction_review,
                }
            )

        return governance_actions

    def _record_governance_event(
        self,
        *,
        org_id: str,
        conversation_id: str,
        channel: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        if self.event_store is None:
            return
        self.event_store.append_event(
            event_id=f"{conversation_id}:{event_type.lower()}:{payload.get('candidate_id', 'na')}",
            event_type=event_type,
            org_id=org_id,
            stream_id=org_id,
            correlation_id=conversation_id,
            payload=payload,
        )
    
    def _validate_justification(
        self,
        J_t: Dict[str, Any],
        gate_mode: str = "tiered",
        O_tag_v: Optional[Dict[str, Any]] = None,
        K_v: Optional[Dict[str, Any]] = None,
        Pi_v: Optional[Dict[str, Any]] = None,
        ConstraintBundle_v: Optional[Dict[str, Any]] = None,
        PrecedenceGraph_v: Optional[Dict[str, Any]] = None,
        ResolutionRules_v: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Route J_t through strict tiered EOS validation gate."""
        pins = J_t.get("pins") or {}
        policy_version = pins.get("C_v", self.DEFAULT_C_V)
        if gate_mode != "tiered":
            result = {"status": "REJECT", "reasons": ["unsupported_gate_mode:base"], "constraint_results": []}
        else:
            result = eos_validate_tiered(
                J_t,
                policy_version,
                O_tag_v=O_tag_v,
                K_v=K_v,
                Pi_v=Pi_v,
                ConstraintBundle_v=ConstraintBundle_v,
                PrecedenceGraph_v=PrecedenceGraph_v,
                ResolutionRules_v=ResolutionRules_v,
            )

        status_map = {
            "ALLOW": ValidationStatus.ALLOW,
            "EXCEPTION": ValidationStatus.EXCEPTION,
            "REJECT": ValidationStatus.REJECT,
        }
        return ValidationResult(
            status=status_map.get(result["status"], ValidationStatus.REJECT),
            reasons=result.get("reasons", []),
            metadata={"constraint_results": result.get("constraint_results", [])},
        )

    def validate_draft(
        self,
        draft: SemanticDraft,
    ) -> ValidationResult:
        """Validate complete draft via v5 tiered EOS gate (FR8)."""
        if draft.status != SemanticDraftStatus.COMPLETE:
            return ValidationResult(
                status=ValidationStatus.REJECT,
                reasons=["draft_incomplete"],
            )
        J_t = self.build_justification(draft)
        O_tag_v = draft.payload.get("O_tag_v") if draft.payload else None
        K_v = draft.payload.get("K_v") if draft.payload else None
        Pi_v = draft.payload.get("Pi_v") if draft.payload else None
        ConstraintBundle_v = draft.payload.get("ConstraintBundle_v") if draft.payload else None
        PrecedenceGraph_v = draft.payload.get("PrecedenceGraph_v") if draft.payload else None
        ResolutionRules_v = draft.payload.get("ResolutionRules_v") if draft.payload else None
        return self._validate_justification(
            J_t,
            gate_mode=draft.gate_mode,
            O_tag_v=O_tag_v,
            K_v=K_v,
            Pi_v=Pi_v,
            ConstraintBundle_v=ConstraintBundle_v,
            PrecedenceGraph_v=PrecedenceGraph_v,
            ResolutionRules_v=ResolutionRules_v,
        )
    
    def commit_draft(
        self,
        draft: SemanticDraft,
        justification: Dict[str, Any],
        validation: Optional[ValidationResult] = None,
        resolution_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Commit a validated draft by persisting to the shared EOS store.

        Writes to eos.db (signals → workflows → rea_claims → justifications →
        validation_results), mirroring exactly how the Hermes pipeline commits.
        Enforces idempotency: re-submitting the same justification returns
        DUPLICATE without writing duplicate rows.
        
        After fact commit succeeds, invokes synchronous COA projection
        (project_rea_to_ledger). MVP: projection failures do not block commit;
        the fact remains committed with projection_status=EXCEPTION.

        Returns: { "result": "COMMITTED" | "DUPLICATE", "event_id": str, 
                   "idempotency_key": str, "projection_status": "PENDING"|"PROJECTED"|"EXCEPTION" }
        The returned event_id is the justification_id in eos.db.
        """
        idempotency_key = justification["content_hash"]
        policy_version = (justification.get("pins") or {}).get("C_v", self.DEFAULT_C_V)

        if self.idempotency_store.exists(idempotency_key):
            return {
                "result": "DUPLICATE",
                "event_id": None,
                "idempotency_key": idempotency_key,
                "projection_status": None,
            }

        signal_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())
        rea_id = str(uuid.uuid4())
        justification_id = str(uuid.uuid4())
        validation_id = str(uuid.uuid4())
        now = now_iso()
        rea_claim = justification.get("semantic_claim", {})
        Pi_v = draft.payload.get("Pi_v") if draft.payload else None
        ResolutionRules_v = draft.payload.get("ResolutionRules_v") if draft.payload else None
        committed_claims = self._materialize_committed_claims(
            justification,
            resolution_result or {"winning_claims": []},
            Pi_v,
            ResolutionRules_v,
        )
        governance_actions = self._govern_promoted_claims(
            draft=draft,
            committed_claims=committed_claims,
        )
        governance_artifacts = {
            "O_tag_v": (draft.payload or {}).get("O_tag_v"),
            "K_v": (draft.payload or {}).get("K_v"),
            "Pi_v": (draft.payload or {}).get("Pi_v"),
            "ConstraintBundle_v": (draft.payload or {}).get("ConstraintBundle_v"),
            "PrecedenceGraph_v": (draft.payload or {}).get("PrecedenceGraph_v"),
            "ResolutionRules_v": (draft.payload or {}).get("ResolutionRules_v"),
        }
        stored_justification = dict(justification)
        stored_justification["committed_claims"] = committed_claims
        stored_justification["governance_actions"] = governance_actions
        stored_justification["governance_artifacts"] = governance_artifacts
        if resolution_result is not None:
            stored_justification["resolution_result"] = resolution_result

        init_db()
        with get_conn() as conn:
            conn.execute(
                hosted.sql_query("INSERT INTO signals (signal_id, signal_type, payload_json, created_at)"
                " VALUES (?,?,?,?)", 'INSERT INTO signals (signal_id, signal_type, payload_json, created_at) VALUES (%s,%s,%s,%s)'),
                (
                    signal_id,
                    "weixin_completion",
                    json.dumps({
                        "org_id": draft.org_id,
                        "conversation_id": draft.conversation_id,
                        "tags": draft.tags,
                    }),
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("INSERT INTO workflows"
                " (workflow_id, signal_id, workflow_type, workflow_json, status, created_at)"
                " VALUES (?,?,?,?,?,?)", 'INSERT INTO workflows (workflow_id, signal_id, workflow_type, workflow_json, status, created_at) VALUES (%s,%s,%s,%s,%s,%s)'),
                (
                    workflow_id,
                    signal_id,
                    "record_financial_event",
                    json.dumps({"conversation_id": draft.conversation_id}),
                    "COMMITTED",
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("INSERT INTO rea_claims"
                " (rea_id, workflow_id, claim_json, claim_hash, created_at)"
                " VALUES (?,?,?,?,?)", 'INSERT INTO rea_claims (rea_id, workflow_id, claim_json, claim_hash, created_at) VALUES (%s,%s,%s,%s,%s)'),
                (
                    rea_id,
                    workflow_id,
                    json.dumps(rea_claim),
                    idempotency_key,
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("INSERT INTO justifications"
                " (justification_id, workflow_id, rea_id, justification_json,"
                "  content_hash, created_at)"
                " VALUES (?,?,?,?,?,?)", 'INSERT INTO justifications (justification_id, workflow_id, rea_id, justification_json,  content_hash, created_at) VALUES (%s,%s,%s,%s,%s,%s)'),
                (
                    justification_id,
                    workflow_id,
                    rea_id,
                    json.dumps(stored_justification),
                    idempotency_key,
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("INSERT INTO validation_results"
                " (validation_id, justification_id, status, result_json,"
                "  policy_bundle_version, created_at)"
                " VALUES (?,?,?,?,?,?)", 'INSERT INTO validation_results (validation_id, justification_id, status, result_json,  policy_bundle_version, created_at) VALUES (%s,%s,%s,%s,%s,%s)'),
                (
                    validation_id,
                    justification_id,
                    "ALLOW",
                    json.dumps(
                        {
                            "status": "ALLOW",
                            "reasons": list((validation.reasons if validation else []) or []),
                            "constraint_results": (
                                (validation.metadata or {}).get("constraint_results", [])
                                if validation else []
                            ),
                            "resolution_result": resolution_result,
                            "committed_claims": committed_claims,
                            "pins": justification.get("pins", {}),
                        }
                    ),
                    policy_version,
                    now,
                ),
            )
            conn.commit()

        # Fact commit succeeded. Now invoke synchronous projection (MVP).
        projection_status = "EXCEPTION"
        projection_ctx_hash: Optional[str] = None
        projection_ctx_ref: Optional[str] = None
        projection_result_payload: Optional[Dict[str, Any]] = None
        effective_projection_bundle_version: Optional[str] = (
            (Pi_v or {}).get("version")
            or (Pi_v or {}).get("bundle_id")
            or self.DEFAULT_PI_V.get("version")
            or self.DEFAULT_PI_V.get("bundle_id")
        )
        try:
            projection_bundle = self._resolve_projection_bundle(draft)

            if not projection_bundle:
                # No active COA bundle; projection deferred
                projection_status = "PENDING"
            else:
                projection_ctx_ref = f"ctx_{justification_id}"
                adapter = SemantierMemoryAdapter(InMemoryHolographicProvider())
                adapter_result = adapter.build_projection_context(
                    projection_context_ref=projection_ctx_ref,
                    org_id=draft.org_id,
                    rea_event=rea_claim,
                    event_id=justification_id,
                    authority_domain="accounting",
                    projection_bundle_id=effective_projection_bundle_version
                    or projection_bundle.get("version")
                    or projection_bundle.get("bundle_id")
                    or "default",
                    O_v=justification.get("pins", {}).get("O_v"),
                    O_tag_v=justification.get("pins", {}).get("O_tag_v"),
                    K_v=justification.get("pins", {}).get("K_v"),
                    ConstraintBundle_v=justification.get("pins", {}).get("ConstraintBundle_v"),
                    PrecedenceGraph_v=justification.get("pins", {}).get("PrecedenceGraph_v"),
                    ResolutionRules_v=justification.get("pins", {}).get("ResolutionRules_v"),
                    query_intent="semantic_completion_projection",
                    seed_context={
                        "justification_id": justification_id,
                        "O_v": justification.get("pins", {}).get("O_v"),
                        "C_v": justification.get("pins", {}).get("C_v"),
                        "K_v": justification.get("pins", {}).get("K_v"),
                        "Pi_v": justification.get("pins", {}).get("Pi_v"),
                        "projection_views": sorted(
                            list((projection_bundle.get("views") or {}).keys())
                        ) if projection_bundle.get("views") else [projection_bundle.get("bundle_id") or projection_bundle.get("name") or "default"],
                        "ontology_version": justification.get("pins", {}).get("ontology_version"),
                        "projection_version": justification.get("pins", {}).get("projection_version"),
                        "currency": draft.currency,
                        "dimensions": {
                            "department": (draft.payload or {}).get("department"),
                            "project": (draft.payload or {}).get("project"),
                            "vendor": draft.counterparty,
                            "counterparty": draft.counterparty,
                        },
                        "rea_event_type": "purchase",
                    },
                    created_by="system",
                )
                projection_ctx_hash = adapter_result.projection_context_hash

                proj_result = project_rea_to_ledger(
                    rea_id=rea_id,
                    rea_claim=rea_claim,
                    active_coa=projection_bundle,
                    projection_context_hash=projection_ctx_hash,
                    projection_context_ref=projection_ctx_ref,
                    semantic_claims=committed_claims,
                    pin_versions={
                        "projection_bundle_version": effective_projection_bundle_version,
                        "coa_bundle_version": projection_bundle.get("version") or projection_bundle.get("bundle_id"),
                        "knowledge_context_version": justification.get("pins", {}).get("K_v"),
                        "constraint_bundle_version": justification.get("pins", {}).get("ConstraintBundle_v"),
                        "tag_ontology_version": justification.get("pins", {}).get("O_tag_v"),
                        "precedence_graph_version": justification.get("pins", {}).get("PrecedenceGraph_v"),
                        "resolution_rules_version": justification.get("pins", {}).get("ResolutionRules_v"),
                    },
                )
                projection_result_payload = proj_result

                projection_status = proj_result.get("status", "EXCEPTION")
                if projection_status == "ALLOW":
                    projection_status = "PROJECTED"
        except Exception as exc:
            # Projection failed but fact remains committed (MVP requirement)
            projection_status = "EXCEPTION"
            projection_result_payload = {
                "status": "EXCEPTION",
                "reason": f"projection_runtime_error:{exc}",
                "projections": [],
            }

        if projection_status != "PROJECTED" and projection_bundle:
            try:
                create_projection_exception(
                    rea_id=rea_id,
                    justification_id=justification_id,
                    projection_context_ref=projection_ctx_ref,
                    projection_context_hash=projection_ctx_hash,
                    coa_version=projection_bundle.get("version")
                    or projection_bundle.get("bundle_id")
                    or projection_bundle.get("name"),
                    projection_bundle_version=effective_projection_bundle_version,
                    constraint_bundle_version=justification.get("pins", {}).get("ConstraintBundle_v"),
                    status=projection_status,
                    exception_payload={
                        "rea_id": rea_id,
                        "justification_id": justification_id,
                        "projection_status": projection_status,
                        "reason": (projection_result_payload or {}).get("reason", "projection_failed"),
                        "projection_context_ref": projection_ctx_ref,
                        "projection_context_hash": projection_ctx_hash,
                        "pins": {
                            "O_v": justification.get("pins", {}).get("O_v"),
                            "O_tag_v": justification.get("pins", {}).get("O_tag_v"),
                            "K_v": justification.get("pins", {}).get("K_v"),
                            "Pi_v": effective_projection_bundle_version,
                            "ConstraintBundle_v": justification.get("pins", {}).get("ConstraintBundle_v"),
                            "PrecedenceGraph_v": justification.get("pins", {}).get("PrecedenceGraph_v"),
                            "ResolutionRules_v": justification.get("pins", {}).get("ResolutionRules_v"),
                        },
                    },
                )
            except Exception:
                pass

        try:
            if projection_ctx_ref and projection_ctx_hash and resolution_result is not None:
                create_replay_binding(
                    event_id=justification_id,
                    pins=justification.get("pins", {}),
                    projection_context_ref=projection_ctx_ref,
                    projection_context_hash=projection_ctx_hash,
                    resolution_trace=(resolution_result.get("resolution_trace") or {}),
                    effects={
                        "projection_status": projection_status,
                        "projection_context_hash": projection_ctx_hash,
                    },
                    event_payload=stored_justification,
                )
                self._record_governance_event(
                    org_id=draft.org_id,
                    conversation_id=draft.conversation_id,
                    channel=str((draft.payload or {}).get("channel") or "unknown"),
                    event_type="REPLAY_BOUND",
                    payload={
                        "candidate_id": justification_id,
                        "semantic_tier": "T5",
                        "channel": str((draft.payload or {}).get("channel") or "unknown"),
                        "transport_metadata": {"projection_context_ref": projection_ctx_ref},
                    },
                )
        except Exception:
            # Do not break commit path on replay-binding persistence errors.
            pass

        explainability_result: Optional[Dict[str, Any]] = None
        if projection_status == "PROJECTED" and projection_ctx_ref and projection_ctx_hash and resolution_result is not None:
            explainability_result = build_explainability_surface(justification_id)
            if explainability_result.get("status") == "OK":
                persist_explainability_surface(
                    justification_id,
                    explainability_result["surface"],
                )
            with get_conn() as conn:
                validation_row = conn.execute(
                    hosted.sql_query("SELECT result_json FROM validation_results WHERE validation_id = ?", 'SELECT result_json FROM validation_results WHERE validation_id = %s'),
                    (validation_id,),
                ).fetchone()
                if validation_row is not None:
                    validation_payload = json.loads(validation_row["result_json"])
                    validation_payload["explainability_surface_status"] = explainability_result.get("status")
                    if explainability_result.get("status") == "OK":
                        validation_payload["explainability_surface"] = explainability_result.get("surface")
                    else:
                        validation_payload["explainability_surface_error"] = {
                            "reason": explainability_result.get("reason"),
                            "errors": explainability_result.get("errors", []),
                        }
                    conn.execute(
                        hosted.sql_query("UPDATE validation_results SET result_json = ? WHERE validation_id = ?", 'UPDATE validation_results SET result_json = %s WHERE validation_id = %s'),
                        (json.dumps(validation_payload, sort_keys=True), validation_id),
                    )
                    conn.commit()

        self.idempotency_store.register(idempotency_key)
        draft.status = SemanticDraftStatus.COMMITTED
        # FR9: persist projection_context_hash and projection_status for audit/replay
        draft.projection_context_hash = projection_ctx_hash
        draft.projection_status = projection_status
        self.draft_store.update_draft(draft)

        return {
            "result": "COMMITTED",
            "event_id": justification_id,
            "idempotency_key": idempotency_key,
            "projection_status": projection_status,
            "projection_context_hash": projection_ctx_hash,
            "committed_claims": committed_claims,
            "governance_actions": governance_actions,
            "explainability_surface_status": (explainability_result or {}).get("status"),
        }

    def complete_and_validate(
        self,
        conversation_id: str,
    ) -> Tuple[Optional[Dict[str, Any]], ValidationResult, Optional[Dict[str, Any]]]:
        """Validate a complete draft and commit if ALLOW.

        Returns: (justification, validation_result, commit_result)
          commit_result is None when validation is not ALLOW.
        """
        draft = self.draft_store.get_draft(conversation_id)
        if not draft:
            return None, ValidationResult(
                status=ValidationStatus.REJECT,
                reasons=["draft_not_found"],
            ), None

        if draft.status not in (SemanticDraftStatus.COMPLETE, SemanticDraftStatus.VALIDATED):
            return None, ValidationResult(
                status=ValidationStatus.REJECT,
                reasons=["draft_not_complete"],
            ), None

        # Temporarily set to COMPLETE so build_justification's guard passes
        # (draft may already be COMPLETE; this is a no-op in that case)
        draft.status = SemanticDraftStatus.COMPLETE
        justification = self.build_justification(draft)
        O_tag_v = draft.payload.get("O_tag_v") if draft.payload else None
        K_v = draft.payload.get("K_v") if draft.payload else None
        Pi_v = draft.payload.get("Pi_v") if draft.payload else None
        ConstraintBundle_v = draft.payload.get("ConstraintBundle_v") if draft.payload else None
        PrecedenceGraph_v = draft.payload.get("PrecedenceGraph_v") if draft.payload else None
        ResolutionRules_v = draft.payload.get("ResolutionRules_v") if draft.payload else None
        # FR8: use gate_mode stored on the draft
        validation = self._validate_justification(
            justification,
            gate_mode=draft.gate_mode,
            O_tag_v=O_tag_v,
            K_v=K_v,
            Pi_v=Pi_v,
            ConstraintBundle_v=ConstraintBundle_v,
            PrecedenceGraph_v=PrecedenceGraph_v,
            ResolutionRules_v=ResolutionRules_v,
        )

        if validation.is_allow():
            constraint_results = validation.metadata.get("constraint_results", [])
            resolution_result = eos_resolve(
                claims=justification.get("claims", []),
                constraint_results=constraint_results,
                precedence_graph=PrecedenceGraph_v or {},
                resolution_rules=ResolutionRules_v or {},
                scope="execution",
                pins=justification.get("pins", {}),
            )
            if resolution_result.get("execution_decision") != "ALLOW":
                draft.status = SemanticDraftStatus.VALIDATED
                self.draft_store.update_draft(draft)
                return justification, ValidationResult(
                    status=ValidationStatus.REJECT,
                    reasons=["resolution_rejected"],
                    metadata={"resolution_result": resolution_result},
                ), None
            commit_result = self.commit_draft(
                draft,
                justification,
                validation=validation,
                resolution_result=resolution_result,
            )
            commit_result["resolution_result"] = resolution_result
        else:
            draft.status = SemanticDraftStatus.VALIDATED
            self.draft_store.update_draft(draft)
            commit_result = None

        return justification, validation, commit_result
    
    # ==========================================
    # Helper methods
    # ==========================================
    
    def _extract_tags(self, user_input: str) -> List[str]:
        """Extract tags from natural language input.
        
        Stub: Could call Hermes for semantic tagging.
        For now, use simple keyword extraction.
        """
        keywords = [
            "书", "书籍", "报销", "支付", "转账", "收款",
            "发票", "收据", "账单", "交易",
        ]
        
        tags = []
        lower_input = user_input.lower()
        for keyword in keywords:
            if keyword in lower_input:
                tags.append(keyword)
        
        return tags or ["transaction"]
    
    def _detect_missing_fields(self, draft: SemanticDraft) -> List[str]:
        """Detect required but missing fields.
        
        Returns list of missing field names to ask for.
        """
        missing = []
        
        for field_name in REQUIRED_FIELDS.keys():
            value = getattr(draft, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)
        
        return missing
    
    def _build_question(self, field_name: str) -> str:
        """Build user-friendly question for a missing field."""
        return FIELD_PROMPTS.get(field_name, f"📎 请提供 {field_name}")
    
    def _update_draft_from_input(
        self,
        draft: SemanticDraft,
        user_input: str,
    ) -> None:
        """Extract and update draft fields from user input.
        
        Attempts to parse user input and update draft fields.
        Uses heuristics to detect field values.
        """
        lower_input = user_input.lower().strip()
        
        # Try to detect field being answered based on missing fields
        if draft.missing_fields:
            primary_missing = draft.missing_fields[0]
            
            if primary_missing == "amount":
                # Try to extract number
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', user_input)
                if numbers:
                    draft.amount = numbers[0]
            
            elif primary_missing == "currency":
                # Common currency codes
                upper_input = user_input.upper()
                for code in ["CNY", "USD", "EUR", "GBP", "JPY"]:
                    if code in upper_input:
                        draft.currency = code
                        break
                else:
                    # Use first 3 chars if uppercase
                    if len(user_input) >= 3 and user_input[:3].isupper():
                        draft.currency = user_input[:3]
            
            elif primary_missing == "counterparty":
                # Use entire input as counterparty
                draft.counterparty = user_input.strip()
            
            elif primary_missing == "invoice_ref":
                # Use as invoice reference
                draft.invoice_ref = user_input.strip()
        
        # Also extract any additional info from the input
        # (e.g., if user provides "100 USD" when asked for amount, update currency too)
        import re
        
        # Try to extract amounts like "100 CNY" or "100 CNY 出版社"
        match = re.search(r'(\d+(?:\.\d+)?)\s*([A-Z]{3})?', user_input.upper())
        if match:
            if not draft.amount:
                draft.amount = match.group(1)
            if match.group(2) and not draft.currency:
                draft.currency = match.group(2)
        
        # Extract non-numeric parts as potential counterparty
        # Remove amount and currency, keep the rest
        parts = user_input.split()
        remaining_parts = []
        for part in parts:
            # Skip numbers
            if re.match(r'^\d+(?:\.\d+)?$', part):
                continue
            # Skip common currency codes
            if part.upper() in ["CNY", "USD", "EUR", "GBP", "JPY"]:
                continue
            remaining_parts.append(part)
        
        if remaining_parts and not draft.counterparty:
            draft.counterparty = " ".join(remaining_parts).strip()
    
    def add_evidence(
        self,
        conversation_id: str,
        evidence_type: str,
        reference: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add evidence (e.g., uploaded invoice) to draft.
        
        Args:
            conversation_id: Conversation ID
            evidence_type: Type of evidence (e.g., "invoice", "receipt")
            reference: Reference or path to evidence
            metadata: Additional metadata
        
        Returns:
            Evidence ID
        """
        evidence_id = self.draft_store.add_evidence(
            conversation_id=conversation_id,
            evidence_type=evidence_type,
            reference=reference,
            metadata=metadata,
        )
        
        # Update draft with invoice_ref if it's an invoice
        draft = self.draft_store.get_draft(conversation_id)
        if evidence_type == "invoice" and not draft.invoice_ref:
            draft.invoice_ref = reference
            self.draft_store.update_draft(draft)
        
        return evidence_id
    
    def get_draft_status(
        self,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Get current draft status for user.
        
        Returns:
            Status dict with current state and any pending questions
        """
        draft = self.draft_store.get_draft(conversation_id)
        if not draft:
            return {"status": "not_found"}
        
        missing = self._detect_missing_fields(draft)
        
        return {
            "conversation_id": conversation_id,
            "status": draft.status.value,
            "tags": draft.tags,
            "amount": draft.amount,
            "currency": draft.currency,
            "counterparty": draft.counterparty,
            "invoice_ref": draft.invoice_ref,
            "evidence_count": len(draft.evidence_refs),
            "missing_fields": missing,
            "created_at": draft.created_at,
            "updated_at": draft.updated_at,
        }
