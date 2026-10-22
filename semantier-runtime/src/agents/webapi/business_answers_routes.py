"""Authenticated source-bounded receivables commands and retained artifact reads."""
from fastapi import APIRouter, HTTPException, Request
from agents import business_answers
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.business_answers_receivables import ReceivablesRunRequest
from services.runtime_errors import RuntimeContextError

router = APIRouter(tags=["business-answers"])
BASE = "/api/business-answers/receivables"


def _call(http_request, operation, **kwargs):
    ctx = require_route_authorization(http_request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER, route_hint=str(http_request.url.path))
    try:
        value = operation(ctx=ctx, **kwargs)
        if value is None:
            raise HTTPException(status_code=404, detail="Business answer artifact not found")
        return value if isinstance(value, dict) else value.model_dump(mode="json")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="Business answer artifact not found") from exc
    except RuntimeContextError as exc:
        code = 403 if exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SCOPE"} else 409
        raise HTTPException(status_code=code, detail=exc.code) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(BASE + "/runs")
def create_receivables_run(body: ReceivablesRunRequest, request: Request):
    return _call(request, business_answers.create_run_projection, request=body)


@router.get(BASE + "/runs/{run_ref}")
def get_receivables_run(run_ref: str, request: Request):
    return _call(request, business_answers.get_run_projection, run_ref=run_ref)


@router.get(BASE + "/answers/{answer_ref}")
def get_receivables_answer(answer_ref: str, request: Request):
    return _call(request, business_answers.get_answer_projection, answer_ref=answer_ref)
