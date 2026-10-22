"""Standalone Jev API client (Python >=3.10; httpx >=0.28,<0.29).

This is a deliberately bounded text-rubric subset, not a transparent API proxy:
state is nonblank text or a strict JSON object/array (not top-level null);
instructions are nonblank text; descriptions are strings, plus Choice nulls.
Only TypeSafe and Cloudflare's fixed HTTPS endpoints are reachable via evaluate.
No credential discovery, environment proxies, redirects, or automatic retries.
"""
from __future__ import annotations

import json
import math
import re
import time

import httpx


MAX_QUESTIONS = 128
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024


class JevError(Exception):
    """Static display-safe category and optional HTTP status, never raw input.

    invalid_* identifies local validation or response/JSON schema failures;
    *_too_large identifies a byte budget failure; provider is success:false.
    retryable describes a transient failure, NOT an automatic retry promise or
    a guarantee that a billed inference did not already execute.
    """

    def __init__(self, category="request", *, status_code=None, retryable=False, provider_code=None):
        categories = (
            "request", "HTTP", "timeout", "network", "provider", "invalid_json",
            "invalid_backend", "invalid_token", "invalid_account_id", "invalid_model", "invalid_gateway_id",
            "invalid_timeout", "invalid_state", "invalid_questions", "invalid_payload",
            "invalid_response", "payload_too_large", "response_too_large",
        )
        category = category if type(category) is str and category in categories else "request"
        status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        self.category = category
        self.status_code = status_code
        self.retryable = retryable is True
        self.provider_code = (provider_code if type(provider_code) is int and
                              0 <= provider_code <= 999_999_999 else None)
        self.hint = ("Enable authentication on the target Cloudflare AI Gateway for Unified Billing; "
                     "configure its gateway_id and use a permitted Cloudflare API token."
                     if self.provider_code == 2049 else None)
        message = f"Jev {category} error"
        if status_code is not None:
            message += f" ({status_code})"
        super().__init__(message)


def safe_error_details(error):
    """Allowlisted diagnostics, reconstructed instead of formatting exceptions."""
    if not isinstance(error, JevError):
        return {}
    safe = JevError(error.category, status_code=error.status_code,
                    retryable=error.retryable, provider_code=error.provider_code)
    details = {"message": str(safe), "category": safe.category,
               "status_code": safe.status_code, "retryable": safe.retryable}
    if safe.provider_code is not None:
        details["provider_code"] = safe.provider_code
    if safe.hint is not None:
        details["hint"] = safe.hint
    return details


def _invalid_constant(_value):
    raise ValueError("Non-JSON constant")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _provider_code(data):
    """Extract only a bounded integer; never retain provider text."""
    errors = data.get("errors") if type(data) is dict else None
    if type(errors) is list:
        for error in errors:
            code = error.get("code") if type(error) is dict else None
            if type(code) is int and 0 <= code <= 999_999_999:
                return code
    return None


def _http_provider_code(response):
    body = bytearray()
    try:
        for chunk in response.iter_bytes(chunk_size=4096):
            if len(body) + len(chunk) > MAX_ERROR_BYTES:
                return None
            body.extend(chunk)
        return _provider_code(json.loads(body, parse_constant=_invalid_constant,
                                         object_pairs_hook=_unique_object))
    except (ValueError, RecursionError, httpx.HTTPError, OSError):
        return None


def validate_gateway_id(backend, gateway_id):
    """Validate non-secret routing before credential access or HTTP requests."""
    if gateway_id is not None and (
        backend != "cloudflare" or type(gateway_id) is not str or
        not 1 <= len(gateway_id) <= 64 or
        not re.fullmatch(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", gateway_id)
    ):
        raise JevError("invalid_gateway_id")


def _post(url, token, payload, timeout, *, gateway_id=None):
    """Internal HTTP seam for transport fixtures; not an endpoint override API."""
    body = bytearray()
    cloudflare = bool(re.fullmatch(r"https://api\.cloudflare\.com/client/v4/accounts/[0-9a-fA-F]{32}/ai/run", url))
    validate_gateway_id("cloudflare" if cloudflare else "typesafe", gateway_id)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "Accept": "application/json", "Accept-Encoding": "identity"}
    if cloudflare:
        headers.update({"cf-aig-collect-log": "false", "cf-aig-skip-cache": "true",
                        "cf-aig-max-attempts": "1"})
        if gateway_id is not None:
            headers["cf-aig-gateway-id"] = gateway_id
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout, trust_env=False) as http:
            with http.stream("POST", url, headers=headers, content=_encode_payload(payload)) as response:
                if not 200 <= response.status_code < 300:
                    status = response.status_code
                    raise JevError("HTTP", status_code=status, retryable=status == 429 or status >= 500,
                                   provider_code=_http_provider_code(response) if cloudflare else None)
                length = response.headers.get("Content-Length", "")
                if length.isascii() and length.isdigit() and (len(length) > 10 or int(length) > MAX_RESPONSE_BYTES):
                    raise JevError("response_too_large")
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise JevError("response_too_large")
                    body.extend(chunk)
    except httpx.TimeoutException:
        raise JevError("timeout", retryable=True) from None
    except (httpx.HTTPError, OSError):
        raise JevError("network", retryable=True) from None
    try:
        return json.loads(body, parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise JevError("invalid_json") from None


def _json_value(value, depth=0):
    """Accept JSON types without silently coercing tuples or object keys."""
    if depth > 64:
        return False
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        return not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    if type(value) is list:
        return all(_json_value(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(key, depth + 1)
                   and _json_value(item, depth + 1) for key, item in value.items())
    return False


def _encode_payload(payload):
    encoded = bytearray()
    try:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for part in encoder.iterencode(payload):
            chunk = part.encode("utf-8")
            if len(encoded) + len(chunk) > MAX_PAYLOAD_BYTES:
                raise JevError("payload_too_large")
            encoded.extend(chunk)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise JevError("invalid_payload") from None
    return bytes(encoded)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_questions(questions):
    if type(questions) is not dict or not 1 <= len(questions) <= MAX_QUESTIONS or not _json_value(questions):
        raise JevError("invalid_questions")
    for key, question in questions.items():
        if not _text(key) or not isinstance(question, dict):
            raise JevError("invalid_questions")
        kind = question.get("type")
        if kind not in ("noul", "choice", "score") or not _text(question.get("instructions")):
            raise JevError("invalid_questions")
        criteria = question.get("criteria")
        if kind == "noul":
            valid = "criteria" not in question or (
                isinstance(criteria, dict) and bool(criteria) and
                set(criteria) <= {"true", "false"} and
                all(isinstance(v, str) for v in criteria.values())
            )
        elif kind == "choice":
            valid = (isinstance(criteria, dict) and 2 <= len(criteria) <= 255 and
                     all(_text(k) and (v is None or isinstance(v, str)) for k, v in criteria.items()))
        else:
            valid = (isinstance(criteria, list) and len(criteria) >= 2 and
                     all(isinstance(v, str) for v in criteria))
        if not valid:
            raise JevError("invalid_questions")


def _number_in(value, lower, upper):
    return type(value) in (int, float) and lower <= value <= upper


def _validate_response(result, questions):
    if (not isinstance(result, dict) or not _text(result.get("model")) or
            not isinstance(result.get("usage"), dict) or
            not isinstance(result.get("answers"), dict)):
        raise JevError("invalid_response")
    for key in ("input_tokens", "output_tokens"):
        if key in result["usage"] and (
            type(result["usage"][key]) is not int or result["usage"][key] < 0
        ):
            raise JevError("invalid_response")
    for key, question in questions.items():
        answer = result["answers"].get(key)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise JevError("invalid_response")
        kind = question["type"]
        if kind == "noul":
            valid = _number_in(answer.get("noul"), 0, 1) and "confidence" not in answer
        elif kind == "choice":
            valid = isinstance(answer.get("choice"), str) and answer["choice"] in question["criteria"]
        else:
            valid = _number_in(answer.get("score"), 0, len(question["criteria"]) - 1)
        if not valid:
            raise JevError("invalid_response")
        if kind == "noul":
            continue
        if "confidence" in answer and not _number_in(answer["confidence"], 0, 1):
            raise JevError("invalid_response")
        if kind == "score" and "legend" in answer:
            legend = answer["legend"]
            if (not isinstance(legend, dict) or
                    set(legend) != {str(i) for i in range(len(question["criteria"]))} or
                    not all(isinstance(v, str) for v in legend.values())):
                raise JevError("invalid_response")
        if "probabilities" in answer:
            probabilities = answer["probabilities"]
            keys = (set(question["criteria"]) if kind == "choice" else
                    {str(i) for i in range(len(question["criteria"]))})
            if (not isinstance(probabilities, dict) or set(probabilities) != keys or
                    not all(_number_in(v, 0, 1) for v in probabilities.values()) or
                    not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=0.01)):
                raise JevError("invalid_response")


def evaluate(*, backend: str, token: str, state, questions: dict,
             account_id: str = "", model: str | None = None, timeout: float = 30,
             gateway_id: str | None = None) -> dict:
    """Evaluate once; retain model/answers/usage and append local metadata.

    backend: exactly 'typesafe' or 'cloudflare'. Cloudflare requires a 32-hex
    account_id. Defaults: jev-latest / typesafe/jev, respectively. Model override
    changes only the JSON model identifier, never the fixed endpoint.
    gateway_id is optional, Cloudflare-only, and must be 1..64 lowercase
    alphanumeric/underscore characters with single interior hyphens. Cloudflare
    requests disable gateway logging/caching and permit only one attempt.

    Limits: 1..128 questions; Choice 2..255 options; Score >=2 string levels;
    JSON depth <=64; UTF-8 request <=1 MiB; decoded response <=2 MiB.
    timeout is 1..120 seconds per HTTP I/O phase (not an overall deadline).

    Noul has no confidence. Optional Choice/Score probabilities and confidence
    are validated when present, never fabricated. Probability sum tolerance is
    0.01 for rounding. Missing requested answers always fail closed.
    Raises JevError with static categories/status and a bounded numeric provider
    code/static hint where available, never provider error text. HTTP error
    diagnostics parse at most 64 KiB; malformed/oversized bodies stay generic.
    """
    started = time.monotonic()
    if backend not in ("typesafe", "cloudflare"):
        raise JevError("invalid_backend")
    validate_gateway_id(backend, gateway_id)
    if not isinstance(token, str) or not re.fullmatch(r"[!-~]{1,8192}", token):
        raise JevError("invalid_token")
    if backend == "cloudflare" and (
        not isinstance(account_id, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", account_id)
    ):
        raise JevError("invalid_account_id")
    if model is not None and (
        not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/@-]{0,127}", model)
    ):
        raise JevError("invalid_model")
    if type(timeout) not in (int, float) or not 1 <= timeout <= 120:
        raise JevError("invalid_timeout")
    if (type(state) not in (str, dict, list) or
            (isinstance(state, str) and not state.strip()) or not _json_value(state)):
        raise JevError("invalid_state")
    _validate_questions(questions)
    if backend == "cloudflare":
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
        payload = {"model": model or "typesafe/jev", "input": {"state": state, "questions": questions}}
    else:
        url = "https://api.typesafe.ai/v1/systemone"
        payload = {"model": model or "jev-latest", "state": state, "questions": questions}
    _encode_payload(payload)
    result = (_post(url, token, payload, timeout) if gateway_id is None else
              _post(url, token, payload, timeout, gateway_id=gateway_id))
    if not isinstance(result, dict):
        raise JevError("invalid_response")
    if backend == "cloudflare":
        # Live /ai/run may wrap the provider response in two result envelopes.
        # Bound unwrapping and inspect failure flags at every layer.
        for depth in range(3):
            if not isinstance(result, dict):
                raise JevError("invalid_response")
            if result.get("success") is False:
                raise JevError("provider", provider_code=_provider_code(result))
            if "result" not in result:
                break
            if depth == 2 or any(key in result for key in ("answers", "model", "usage")):
                raise JevError("invalid_response")
            result = result["result"]
    _validate_response(result, questions)
    # Provider extensions must never become local status or authorization fields.
    return {"model": result["model"], "answers": result["answers"], "usage": result["usage"],
            "backend": backend, "latency_ms": (time.monotonic() - started) * 1000}
