from __future__ import annotations

import json
from typing import Any


def _error(exc: Exception) -> str:
    return json.dumps(
        {"status": "error", "error_code": str(exc) or exc.__class__.__name__},
        ensure_ascii=False,
        sort_keys=True,
    )


def extract_financial_document(args: dict[str, Any], **_kw: Any) -> str:
    try:
        from .service import extract_live, extract_replay

        mode = str(args.get("mode") or "live")
        result = extract_replay(args) if mode == "replay" else extract_live(args)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return _error(exc)


def evaluate_expense_reimbursement(args: dict[str, Any], **_kw: Any) -> str:
    try:
        from contracts.expense_reimbursement import ExpenseReimbursementRequest
        from plugins.document_extraction.storage import bound_session
        from services.expense_reimbursement_service import ExpenseReimbursementService

        request = ExpenseReimbursementRequest.model_validate(args)
        bound = bound_session(args)
        owner = ExpenseReimbursementService(bound)
        if request.action == 'evaluate':
            from plugins.governed_graph.tools import _session_context
            from services.expense_teaching_policy_service import expense_policy_owner
            with expense_policy_owner(_session_context(), bound) as policy_owner:
                result = ExpenseReimbursementService(bound, policy_owner=policy_owner).evaluate(
                    financial_ref=request.financial_ref, submitted_on=request.submitted_on,
                    applicant_fields=request.applicant_fields.model_dump())
        elif request.action == 'read_current':
            result = owner.current()
            if result is None:
                raise ValueError('EXPENSE_RESULT_NOT_FOUND')
        elif request.action == 'replay':
            result = owner.replay(request.result_ref, request.result_hash)
        else:
            result = owner.save_draft(request.result_ref, request.result_hash)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return _error(exc)
