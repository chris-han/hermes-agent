from idempotency import IdempotencyStore
from events import ExecutionEventStore, ExecutionEventType
from approval import ApprovalStore
from projection import project
from domain import DomainEventStore
import os

idempotency_store = IdempotencyStore()
event_store = ExecutionEventStore()
approval_store = ApprovalStore()
domain_store = DomainEventStore()

REQUIRED_APPROVALS = int(os.getenv("SEMANTIER_REQUIRED_APPROVALS", "1"))


def authority_commit(intent, valid: bool):
    key = intent.idempotency_key()

    # ----------------------------
    # 1. validation
    # ----------------------------
    if not valid:
        event_store.append_event(
            ExecutionEventType.VALIDATION_FAILED,
            {"idempotency_key": key},
        )
        return "REJECTED"

    # ----------------------------
    # 2. intent created (always log)
    # ----------------------------
    event_store.append_event(
        ExecutionEventType.EXECUTION_INTENT_CREATED,
        {"idempotency_key": key, "action": intent.action},
    )

    # ----------------------------
    # 3. approval gate
    # ----------------------------
    approvals = approval_store.count_approvals(key)
    if approvals < REQUIRED_APPROVALS:
        event_store.append_event(
            ExecutionEventType.VALIDATION_PASSED,
            {"idempotency_key": key, "status": "PENDING_APPROVAL"},
        )
        return "PENDING_APPROVAL"

    # ----------------------------
    # 4. idempotency check
    # ----------------------------
    if idempotency_store.exists(key):
        event_store.append_event(
            ExecutionEventType.EXECUTION_DUPLICATE,
            {"idempotency_key": key},
        )
        return "DUPLICATE"

    # ----------------------------
    # 5. commit
    # ----------------------------
    event = project(intent)
    domain_store.append(event["event_type"], event["payload"])

    idempotency_store.register(key)

    event_store.append_event(
        ExecutionEventType.EXECUTION_COMMITTED,
        {"idempotency_key": key},
    )

    return "COMMITTED"