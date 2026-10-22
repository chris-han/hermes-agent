import os
from idempotency import IdempotencyStore
from approval import ApprovalStore

REQUIRED_APPROVALS = int(os.getenv("SEMANTIER_REQUIRED_APPROVALS", "1"))


def get_status(key: str):
    id_store = IdempotencyStore()
    approval_store = ApprovalStore()

    if id_store.exists(key):
        return {
            "status": "COMMITTED",
            "approvals": approval_store.count_approvals(key),
            "required": REQUIRED_APPROVALS,
            "approvers": approval_store.list_approvers(key),
        }

    approvals = approval_store.count_approvals(key)

    if approvals >= REQUIRED_APPROVALS:
        status = "APPROVED"
    elif approvals > 0:
        status = "PARTIALLY_APPROVED"
    else:
        status = "PENDING_APPROVAL"

    return {
        "status": status,
        "approvals": approvals,
        "required": REQUIRED_APPROVALS,
        "approvers": approval_store.list_approvers(key),
    }