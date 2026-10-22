"""Offline-only Qwen preparation for U0 reference-conformance fixtures."""
from __future__ import annotations
import base64, hashlib, importlib.metadata, json, time
from dataclasses import dataclass
from pathlib import Path
import httpx

MODEL = "qwen3.5-plus"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MAX_RESPONSE = 2 * 1024 * 1024

def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _strict(pairs):
    out = dict(pairs)
    if len(out) != len(pairs): raise ValueError("DUPLICATE_JSON_KEY")
    return out
def _constant(value): raise ValueError("NON_JSON_CONSTANT")

@dataclass(frozen=True)
class PreparedQwen:
    body: bytes
    sha256: str
    source_pins: dict
    sdk_versions: dict
    model: str = MODEL
    endpoint: str = ENDPOINT

def prepare_qwen(raw, *, hermes_root):
    if type(raw) is not bytes or not raw or len(raw) > 12000: raise ValueError("REQUEST_BYTES_INVALID")
    try: payload = json.loads(raw, object_pairs_hook=_strict, parse_constant=_constant)
    except (ValueError, TypeError, RecursionError): raise ValueError("REQUEST_INVALID") from None
    keys = {"model","messages","temperature","response_format","max_completion_tokens","enable_thinking"}
    if type(payload) is not dict or set(payload) != keys: raise ValueError("REQUEST_SHAPE_INVALID")
    if payload["model"] != MODEL or type(payload["messages"]) is not list or not payload["messages"]: raise ValueError("REQUEST_POLICY_INVALID")
    if payload["temperature"] != 0 or payload["response_format"] != {"type":"json_object"}: raise ValueError("REQUEST_POLICY_INVALID")
    if type(payload["max_completion_tokens"]) is not int or payload["max_completion_tokens"] != 502 or payload["enable_thinking"] is not False: raise ValueError("REQUEST_POLICY_INVALID")
    for m in payload["messages"]:
        if type(m) is not dict or set(m) != {"role","content"} or type(m["content"]) is not str: raise ValueError("MESSAGE_INVALID")
    root = Path(hermes_root)
    names = ('providers/base.py', 'providers/__init__.py',
             'plugins/model-providers/alibaba/__init__.py', 'agent/transports/chat_completions.py')
    pins = {name: _sha((root / name).read_bytes()) for name in names}
    captured = []
    def capture(request):
        if str(request.url) != ENDPOINT or 'authorization' in request.headers:
            raise ValueError('PREPARATION_ROUTE_DRIFT')
        captured.append(request.content)
        return httpx.Response(200, content=b'{}')
    sdk_exchange(payload, httpx.MockTransport(capture))
    if len(captured) != 1: raise ValueError('PREPARATION_ATTEMPT_COUNT')
    body = captured[0]
    wire = dict(payload); wire['max_tokens'] = wire.pop('max_completion_tokens'); wire['stream'] = False
    if json.loads(body) != wire or len(body) > 12000: raise ValueError('PREPARED_BODY_DRIFT')
    return PreparedQwen(body, _sha(body), pins, {name: importlib.metadata.version(name) for name in ('openai','httpx')})

def sdk_exchange(payload, transport, credential=None, timeout=30):
    """Existing Hermes profile/transport and SDK serializer; no route fallback."""
    import openai
    from openai._base_client import SyncAPIClient
    from openai.resources.chat.completions import Completions
    from providers import get_provider_profile
    from agent.transports.chat_completions import ChatCompletionsTransport
    profile = get_provider_profile('alibaba')
    if profile is None or profile.name != 'alibaba' or profile.api_mode != 'chat_completions' or profile.env_vars != ('DASHSCOPE_API_KEY',):
        raise PermissionError('HERMES_ALIBABA_PROFILE_DRIFT')
    kwargs = ChatCompletionsTransport().build_kwargs(
        MODEL, payload['messages'], provider_profile=profile,
        base_url=ENDPOINT.removesuffix('/chat/completions'), temperature=0,
        max_tokens=502, max_tokens_param_fn=lambda n: {'max_tokens':n},
        reasoning_config={'enabled':False}, extra_body_additions={'enable_thinking':False},
        request_overrides={'response_format':{'type':'json_object'},'stream':False})
    class SDK(SyncAPIClient):
        def _make_status_error(self, message, *, body, response):
            return openai.APIStatusError('U0_HTTP_STATUS', response=response, body=body)
    headers = {'Accept-Encoding':'identity'}
    if credential is not None: headers['Authorization']='Bearer '+credential
    with httpx.Client(transport=transport, follow_redirects=False, trust_env=False, timeout=timeout) as http:
        with SDK(version=openai.__version__,base_url=ENDPOINT.removesuffix('/chat/completions'),
                 max_retries=0,http_client=http,timeout=timeout,custom_headers=headers,
                 _strict_response_validation=False) as client:
            try:
                with Completions(client).with_streaming_response.create(**kwargs) as response:
                    return response.http_response.status_code, response.http_response.read()
            except openai.APIStatusError as error:
                return error.response.status_code, error.response.content

def run_qwen(raw, *, hermes_root, fixture, deadline):
    if not isinstance(fixture, dict): raise PermissionError("PAID_EXPERIMENT_AUTHORITY_UNAVAILABLE")
    if time.monotonic() >= deadline: raise TimeoutError("DEADLINE_EXPIRED")
    prepared = prepare_qwen(raw, hermes_root=hermes_root); captured = bytearray()
    receipt = {"wire_request_sha256":prepared.sha256,"requested_model":MODEL,"actual_model":None,"usage":None,"http_status":None,"attempts_observed":0,"response_complete":False,"response_available":False,"retries":0,"redirects":0,"fallbacks":0,"credential_redaction_applied":False}
    def handler(request):
        if fixture.get("block_seconds"): time.sleep(float(fixture["block_seconds"]))
        if time.monotonic() >= deadline: raise TimeoutError("DEADLINE_EXPIRED")
        if fixture.get("network_error"): raise httpx.ConnectError("offline connect failure")
        body = base64.b64decode(fixture.get("body_b64",""), validate=True)
        return httpx.Response(int(fixture.get("status",200)), content=body, request=request)
    class OneAttempt(httpx.BaseTransport):
        def __init__(self): self.inner = httpx.MockTransport(handler)
        def handle_request(self, request):
            if receipt["attempts_observed"] != 0: raise RuntimeError("MULTIPLE_PHYSICAL_ATTEMPTS")
            if request.method != "POST" or str(request.url) != ENDPOINT or request.content != prepared.body: raise RuntimeError("TRANSPORT_IDENTITY_DRIFT")
            if request.headers.get("Authorization") is not None: raise RuntimeError("OFFLINE_AUTH_HEADER_FORBIDDEN")
            receipt["attempts_observed"] = 1
            return self.inner.handle_request(request)
        def close(self): self.inner.close()
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError("DEADLINE_EXPIRED")
        with httpx.Client(transport=OneAttempt(), follow_redirects=False, trust_env=False, timeout=min(remaining,30)) as client:
            response = client.post(ENDPOINT, content=prepared.body, headers={"Content-Type":"application/json","Accept":"application/json"})
            receipt["http_status"] = response.status_code; receipt["response_available"] = True
            captured.extend(response.content[:MAX_RESPONSE]); receipt["response_complete"] = True
            if len(response.content) > MAX_RESPONSE or not 200 <= response.status_code < 300: raise ValueError("RESPONSE_INVALID")
        decoded = json.loads(captured, object_pairs_hook=_strict, parse_constant=_constant)
        usage = decoded.get("usage") if isinstance(decoded,dict) else None
        good = isinstance(usage,dict) and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] >= 0 and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] >= 0
        if decoded.get("model") != MODEL or not good: raise ValueError("RESPONSE_METADATA_INVALID")
        receipt.update(outcome="VALID_RESPONSE", actual_model=MODEL, usage=usage)
    except TimeoutError: raise
    except Exception: receipt["outcome"] = "FAILED"
    response = bytes(captured); h = _sha(response) if response else None
    receipt.update(response_b64=base64.b64encode(response).decode(), raw_response_sha256=h, retained_response_sha256=h)
    return receipt
