"""Pinned native client adaptation; offline by default, exact operator specimen only."""
from __future__ import annotations
import base64
import copy
from dataclasses import dataclass
import hashlib
from types import ModuleType
import json
from pathlib import Path
import time
import httpx

MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
SOURCE_SHA256 = "413173b777c870ddb4b76b14383d70d419ecbbf366b1675d039bc8a4df3abdbe"
SOURCE = Path(__file__).parent / "upstream" / "client.py"
MAX_REQUEST = 1024 * 1024
MAX_RESPONSE = 2 * 1024 * 1024
SPECIMEN_SHA256 = "e160b92c55bdf0c16eb17668a2233fdb21aa3fe794b02c08afa7348627eac19d"
SPECIMEN_BYTES = 24077

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def upstream():
    source_bytes = SOURCE.read_bytes()
    if sha(source_bytes) != SOURCE_SHA256:
        raise ValueError("UPSTREAM_SOURCE_DRIFT")
    module = ModuleType("_jev_candidate_upstream")
    exec(compile(source_bytes, str(SOURCE), "exec"), module.__dict__)
    original = module._validate_questions
    def validate(questions):
        proxy = copy.deepcopy(questions)
        if isinstance(proxy, dict):
            for question in proxy.values():
                if isinstance(question, dict) and question.get("type") == "choice" and isinstance(question.get("criteria"), dict):
                    for key, description in question["criteria"].items():
                        if isinstance(description, dict) and set(description) == {"what"} and isinstance(description["what"], str):
                            question["criteria"][key] = description["what"]
        original(proxy)
        if not module._json_value(questions):
            raise module.JevError("invalid_questions")
    module._validate_questions = validate
    return module

@dataclass(frozen=True)
class PreparedRequest:
    body: bytes
    sha256: str
    model: str = MODEL
    endpoint: str = ENDPOINT

def prepare_request(raw):
    if type(raw) is not bytes or not raw or len(raw) > MAX_REQUEST:
        raise ValueError("REQUEST_BYTES_INVALID")
    module = upstream()
    try:
        payload = json.loads(raw, parse_constant=module._invalid_constant, object_pairs_hook=module._unique_object)
        if type(payload) is not dict or set(payload) != {"model", "state", "questions"} or payload["model"] != MODEL:
            raise ValueError("REQUEST_SHAPE_INVALID")
        state = payload["state"]
        if type(state) not in (str, dict, list) or (type(state) is str and not state.strip()) or not module._json_value(state):
            raise ValueError("STATE_INVALID")
        module._validate_questions(payload["questions"])
    except (ValueError, TypeError, RecursionError, module.JevError):
        raise ValueError("REQUEST_INVALID") from None
    return PreparedRequest(raw, sha(raw))

def _strict_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("DUPLICATE_JSON_KEY")
    return result

def _reject_constant(value):
    raise ValueError("NON_JSON_CONSTANT")

def redact_bytes(raw, credential):
    """Keep pre-redaction hash separately; sanitized bytes are never called raw."""
    value = raw
    if not credential:
        return value
    variants = {credential, json.dumps(credential, ensure_ascii=False)[1:-1],
                json.dumps(credential, ensure_ascii=True)[1:-1]}
    escaped = "".join("\\u%04x" % ord(c) for c in credential)
    variants.update({escaped, escaped.upper().replace("\\U", "\\u"), escaped.replace("\\", "\\\\")})
    for variant in sorted(variants, key=len, reverse=True):
        value = value.replace(variant.encode(), b"[CREDENTIAL_REMOVED]")
    # Catch mixed JSON escaping while preserving bytes unless a secret is present.
    try:
        decoded = json.loads(value, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
        changed = False
        def walk(item):
            nonlocal changed
            if isinstance(item, str):
                if credential in item:
                    changed = True
                    return item.replace(credential, "[CREDENTIAL_REMOVED]")
                return item
            if isinstance(item, list):
                return [walk(x) for x in item]
            if isinstance(item, dict):
                return {walk(k): walk(v) for k, v in item.items()}
            return item
        safe = walk(decoded)
        if changed:
            value = json.dumps(safe, ensure_ascii=True, separators=(",", ":")).encode()
    except (ValueError, RecursionError):
        if value:
            return b"[UNPARSEABLE_RESPONSE_WITHHELD]"
    return value

def safe_provider_value(value, credential):
    """Redact decoded provider strings, never JSON syntax or generated fields."""
    if isinstance(value, str):
        return value.replace(credential, "[CREDENTIAL_REMOVED]") if credential else value
    if isinstance(value, list):
        return [safe_provider_value(item, credential) for item in value]
    if isinstance(value, dict):
        return {safe_provider_value(key, credential): safe_provider_value(item, credential)
                for key, item in value.items()}
    return value

class _FixtureStream(httpx.SyncByteStream):
    def __init__(self, fixture, raw):
        self.fixture, self.raw = fixture, raw
    def __iter__(self):
        size = max(1, int(self.fixture.get("chunk_bytes", len(self.raw) or 1)))
        for start in range(0, len(self.raw), size):
            delay = float(self.fixture.get("chunk_delay_seconds", 0))
            if delay:
                time.sleep(delay)
            yield self.raw[start:start + size]
        if self.fixture.get("read_error"):
            raise httpx.ReadError("offline read error")
    def close(self):
        pass

def run_native(raw, credential, deadline, fixture):
    """Existing offline API; a missing fixture never enables provider access."""
    if not isinstance(fixture, dict):
        raise PermissionError("PAID_EXPERIMENT_AUTHORITY_UNAVAILABLE")
    return _run_native(raw, credential, deadline, fixture)


def run_specimen_native(raw, credential, deadline):
    """Only the user-run handoff may call this fixed, single-specimen transport."""
    if type(raw) is not bytes or len(raw) != SPECIMEN_BYTES or sha(raw) != SPECIMEN_SHA256:
        raise ValueError("SPECIMEN_REQUEST_MISMATCH")
    return _run_native(raw, credential, deadline, None)


def _run_native(raw, credential, deadline, fixture):
    if time.monotonic() >= deadline:
        raise TimeoutError("DEADLINE_EXPIRED")
    prepared = prepare_request(raw)
    payload = json.loads(raw)
    module = upstream()
    captured = bytearray()
    receipt = {"wire_request_sha256": prepared.sha256, "request_bytes": len(raw),
               "requested_model": MODEL, "actual_model": None, "usage": None,
               "http_status": None, "request_id": None, "attempts_observed": 0,
               "response_complete": False, "response_available": False,
               "retries": 0, "redirects": 0, "fallbacks": 0}
    returned = None
    def mock_handler(request):
        expected = fixture.get("expected_credential")
        if expected is not None and request.headers.get("Authorization") != "Bearer " + expected:
            raise RuntimeError("PROFILE_CREDENTIAL_MISMATCH")
        if fixture.get("block_seconds"):
            time.sleep(float(fixture["block_seconds"]))
        if fixture.get("network_error"):
            raise httpx.ConnectError("offline connect failure")
        if fixture.get("timeout"):
            raise httpx.ReadTimeout("offline phase timeout")
        body = base64.b64decode(fixture.get("body_b64", ""), validate=True)
        headers = {"x-typesafe-request-id": str(fixture.get("request_id", "offline"))}
        if "declared_length" in fixture:
            headers["Content-Length"] = str(fixture["declared_length"])
        return httpx.Response(int(fixture.get("status", 200)), headers=headers,
                              stream=_FixtureStream(fixture, body), request=request)
    class OneAttemptTransport(httpx.BaseTransport):
        def __init__(self, transport):
            self.transport = transport

        def handle_request(self, request):
            expected_headers = {
                "Authorization": "Bearer " + credential, "Content-Type": "application/json",
                "Accept": "application/json", "Accept-Encoding": "identity",
            }
            if (receipt["attempts_observed"] != 0 or request.method != "POST"
                    or request.content != raw or str(request.url) != ENDPOINT
                    or any(request.headers.get(key) != value for key, value in expected_headers.items())):
                raise RuntimeError("TRANSPORT_IDENTITY_DRIFT")
            if time.monotonic() >= deadline:
                raise TimeoutError("DEADLINE_EXPIRED")
            # Count immediately at the physical transport boundary, including errors.
            receipt["attempts_observed"] += 1
            return self.transport.handle_request(request)

        def close(self):
            self.transport.close()

    def post(url, token, outgoing, timeout, **kwargs):
        nonlocal returned
        if url != ENDPOINT or token != credential or outgoing != payload or kwargs:
            raise ValueError("PREPARED_REQUEST_DRIFT")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("DEADLINE_EXPIRED")
        transport = OneAttemptTransport(httpx.MockTransport(mock_handler) if fixture is not None
                                        else httpx.HTTPTransport(retries=0, trust_env=False))
        with httpx.Client(transport=transport, follow_redirects=False, trust_env=False,
                          timeout=min(remaining, 30)) as http:
            with http.stream("POST", ENDPOINT, content=raw, headers={
                "Authorization": "Bearer " + token, "Content-Type": "application/json",
                "Accept": "application/json", "Accept-Encoding": "identity",
            }) as response:
                receipt["http_status"] = response.status_code
                receipt["response_available"] = True
                request_id = response.headers.get("x-typesafe-request-id")
                if request_id and len(request_id) <= 256 and credential not in request_id:
                    receipt["request_id"] = request_id
                for chunk in response.iter_bytes():
                    remaining_bytes = MAX_RESPONSE - len(captured)
                    captured.extend(chunk[:remaining_bytes])
                    if len(chunk) > remaining_bytes:
                        raise module.JevError("response_too_large")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("DEADLINE_EXPIRED")
                receipt["response_complete"] = True
                if not 200 <= response.status_code < 300:
                    raise module.JevError("HTTP", status_code=response.status_code)
        try:
            returned = json.loads(captured, parse_constant=module._invalid_constant,
                                  object_pairs_hook=module._unique_object)
        except (ValueError, RecursionError):
            raise module.JevError("invalid_json") from None
        if not module._json_value(returned):
            raise module.JevError("invalid_response")
        return returned
    module._post = post
    started = time.monotonic()
    try:
        result = module.evaluate(backend="typesafe", token=credential, model=MODEL,
                                 state=payload["state"], questions=payload["questions"], timeout=30)
        if result["model"] != MODEL or any(key not in result["usage"] for key in ("input_tokens", "output_tokens")):
            raise module.JevError("invalid_response")
        receipt.update(outcome="VALID_RESPONSE", actual_model=result["model"], usage=result["usage"],
                       answers=result["answers"])
    except module.JevError as error:
        receipt["outcome"] = {"HTTP": "HTTP_ERROR", "invalid_json": "INVALID_JSON",
                              "invalid_response": "INVALID_RESPONSE", "response_too_large": "RESPONSE_TOO_LARGE"}.get(error.category, "LOCAL_ERROR")
        receipt["error_category"] = error.category
    except (TimeoutError, httpx.TimeoutException):
        receipt.update(outcome="DEADLINE_OR_TIMEOUT", error_category="timeout")
    except httpx.HTTPError:
        receipt.update(outcome="NETWORK_ERROR", error_category="network")
    except Exception:
        receipt.update(outcome="LOCAL_ERROR", error_category="local")
    finally:
        if isinstance(returned, dict) and module._json_value(returned):
            receipt["actual_model"] = safe_provider_value(returned.get("model"), credential)
            receipt["usage"] = safe_provider_value(returned.get("usage"), credential)
        original = bytes(captured)
        safe = redact_bytes(original, credential)
        if "answers" in receipt:
            receipt["answers"] = safe_provider_value(receipt["answers"], credential)
        receipt.update(response_retention=("EXACT_RAW" if safe == original else "WITHHELD_UNPARSEABLE" if safe == b"[UNPARSEABLE_RESPONSE_WITHHELD]" else "REDACTED"),
                       raw_response_sha256=sha(original) if receipt["response_available"] else None,
                       response_hash_scope=("FULL_RESPONSE" if receipt["response_complete"] else "RECEIVED_PREFIX") if receipt["response_available"] else "UNAVAILABLE",
                       retained_response_sha256=sha(safe), response_bytes_before_redaction=len(original),
                       credential_redaction_applied=safe != original, response_b64=base64.b64encode(safe).decode(),
                       elapsed_ms=round((time.monotonic()-started)*1000))
    return receipt
