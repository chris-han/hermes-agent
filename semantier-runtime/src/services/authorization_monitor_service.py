"""Pure non-blocking explicit-delegation assessment; never dispatch or authority."""
from datetime import datetime
from contracts.learning_trace_evidence import AuthorizationMonitorAssessmentV1, DelegationBasisV1
from eos.artifact_hashing import content_hash


class AuthorizationMonitorService:
    def assess(self, *, basis: DelegationBasisV1, actions, completion, coverage,
               surface, recorded_at, gaps=()):
        if not isinstance(basis, DelegationBasisV1):
            raise TypeError("frozen delegation basis required")
        retained, seen, missing, violations = [], {}, list(gaps), []
        conflicted = set()
        for original in actions:
            row = dict(original)
            event_ref = row.get("event_ref")
            if not event_ref:
                missing.append("unbound_action")
            if event_ref in seen:
                if seen[event_ref] != row:
                    missing.append("conflicting_event_identity")
                    conflicted.add(event_ref)
                continue
            seen[event_ref] = row
            retained.append(row)
            if row.get("stage") not in {"TOOL_REPORTED", "EFFECT_CONFIRMED"}:
                missing.append("effect_or_execution_unconfirmed")
                continue
            reasons = []
            for field, allowed in (("operation", basis.allowed_operations),
                                   ("object_ref", basis.allowed_objects),
                                   ("recipient", basis.allowed_recipients)):
                if allowed is None:
                    continue
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    missing.append("missing_" + field)
                elif value not in allowed:
                    reasons.append(field + "_outside_delegation")
            if basis.stop_after_sequence is not None:
                seq = row.get("sequence")
                if type(seq) is not int:
                    missing.append("missing_action_sequence")
                elif seq > basis.stop_after_sequence:
                    reasons.append("after_stop_boundary")
            if basis.valid_from is not None or basis.valid_until is not None:
                when = row.get("occurred_at")
                if isinstance(when, str):
                    try:
                        when = datetime.fromisoformat(when.replace("Z", "+00:00"))
                    except ValueError:
                        when = None
                if not isinstance(when, datetime) or when.tzinfo is None:
                    missing.append("missing_valid_action_time")
                elif (basis.valid_from is not None and when < basis.valid_from
                      or basis.valid_until is not None and when > basis.valid_until):
                    reasons.append("outside_delegation_time")
            if reasons and event_ref:
                violations.append({"event_ref": event_ref, "stage": row["stage"], "reasons": reasons})
        violations = [v for v in violations if v["event_ref"] not in conflicted]
        missing = tuple(sorted(set(missing)))
        effective_coverage = "PARTIAL" if missing and coverage == "COMPLETE_FOR_DECLARED_SURFACE" else coverage
        status = ("OUT_OF_SCOPE" if violations else
                  "WITHIN_SCOPE" if effective_coverage == "COMPLETE_FOR_DECLARED_SURFACE" and not missing
                  else "UNKNOWN")
        return AuthorizationMonitorAssessmentV1(
            basis_hash="sha256:" + content_hash(basis.model_dump(mode="json")),
            snapshot_hash="sha256:" + content_hash({"actions": retained, "coverage": effective_coverage,
                                                   "gaps": missing, "surface": surface}),
            surface=surface, completion=completion, authorization_status=status,
            coverage=effective_coverage, actions=tuple(retained), violations=tuple(violations),
            gaps=missing, recorded_at=recorded_at)
