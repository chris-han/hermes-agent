"""Explicit platform-funded diagnostics for the captured LiteLLM admin listener.

This is an operator command, not a customer-key or wallet fallback. Supplier cost
remains unknown without a supplier tariff; unpriced exposure is never released.
"""
from dataclasses import dataclass
from decimal import Decimal
import json
from uuid import uuid4

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from contracts.model_gateway import canonical_hash, CHAT_FIELDS
from agents.model_gateway.physical_binding import extract_physical_binding
from agents.model_gateway.proxy_capture import AttemptCapture, _attempt, MAX_WIRE_BYTES
from billing.native_evidence import final_native_observation

PATHS = {"/v1/chat/completions", "/chat/completions", "/health/test_connection"}


@dataclass(frozen=True)
class DiagnosticPolicy:
    organization_id: str
    account_id: str
    reservation: str
    aliases: tuple
    native_model_id: str
    physical_binding: dict

    @property
    def provider(self):
        return self.physical_binding["provider_model_id"].split("/", 1)[0]

    @classmethod
    def from_dict(cls, value, origins):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("explicit complete diagnostic policy required")
        for name in ("organization_id", "account_id", "native_model_id"):
            if not isinstance(value[name], str) or not value[name] or len(value[name]) > 200:
                raise ValueError("bounded diagnostic identity required")
        amount = Decimal(value["reservation"])
        if not amount.is_finite() or not 0 < amount <= 1:
            raise ValueError("diagnostic reservation must be positive and at most CNY1")
        aliases = value["aliases"]
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a,str) and a and a.isascii() for a in aliases):
            raise ValueError("explicit diagnostic model aliases required")
        binding = value["physical_binding"]
        provider = str(binding.get("provider_model_id", "")).split("/", 1)[0]
        if provider not in {"dashscope", "azure"}:
            raise ValueError("unsupported diagnostic provider")
        extracted = extract_physical_binding({**binding, "model": binding.get("provider_model_id")}, provider)
        from urllib.parse import urlsplit
        endpoint = urlsplit(extracted["api_base"])
        origin = endpoint.scheme + "://" + endpoint.netloc
        if origins.get(origin) != provider:
            raise ValueError("diagnostic provider origin must be admitted")
        return cls(**{**value, "aliases": tuple(aliases), "physical_binding": extracted})

    def prepare(self, body, path):
        if not isinstance(body, dict):
            raise ValueError("JSON object required")
        if path == "/health/test_connection":
            if set(body) - {"litellm_params", "model_info", "mode"} or body.get("mode") not in (None, "chat"):
                raise ValueError("only pinned chat diagnostics are supported")
            params = body.get("litellm_params") or {}
            info = body.get("model_info") or {}
            if not isinstance(params,dict) or not isinstance(info,dict):
                raise ValueError("invalid diagnostic model parameters")
            if set(params) - {"model", "custom_llm_provider", "litellm_credential_name"}:
                raise ValueError("diagnostic credential or endpoint overrides are prohibited")
            if params.get("custom_llm_provider") not in (None, self.provider) or params.get("litellm_credential_name"):
                raise ValueError("use the pinned model credential")
            if set(info) - {"id"} or info.get("id") not in (None, self.native_model_id):
                raise ValueError("diagnostic deployment does not match")
            allowed = {*self.aliases, self.physical_binding["provider_model_id"]}
            if params.get("model") not in allowed:
                raise ValueError("Model is not enabled for platform diagnostics. Select: " + ", ".join(self.aliases))
            return {**body, "mode":"chat", "model_info":{"id":self.native_model_id},
                    "litellm_params":{**params, "model":self.physical_binding["provider_model_id"],
                                      "max_tokens":32, **({"enable_thinking":False} if self.provider == "dashscope" else {})}}
        unsupported = set(body) - (CHAT_FIELDS | {"litellm_trace_id"})
        if unsupported:
            raise ValueError("Unsupported diagnostic parameters: " + ", ".join(sorted(unsupported))[:200] + ". Endpoint, credentials and payer remain server-owned.")
        if body.get("model") not in self.aliases or not isinstance(body.get("messages"), list):
            raise ValueError("Model is not enabled for platform diagnostics. Select: " + ", ".join(self.aliases))
        cap = body.get("max_completion_tokens", body.get("max_tokens",256))
        if isinstance(cap,bool) or not isinstance(cap,int) or not 1 <= cap <= 32768:
            raise ValueError("max_tokens or max_completion_tokens must be an integer from 1 to 32768.")
        if "max_tokens" in body and "max_completion_tokens" in body:
            raise ValueError("choose one output cap")
        trace = body.get("litellm_trace_id")
        if trace is not None and (not isinstance(trace,str) or not trace.isascii() or not 1 <= len(trace) <= 200):
            raise ValueError("bounded diagnostic correlation required")
        prepared = dict(body)
        if "max_completion_tokens" not in prepared:
            prepared["max_tokens"] = cap
        if prepared.get("stream"):
            prepared["stream_options"] = {"include_usage": True}
        return prepared


def load_diagnostic_policies(value, origins):
    entries = value if isinstance(value, list) else [value]
    if not entries:
        raise ValueError("at least one explicit diagnostic policy required")
    policies = tuple(DiagnosticPolicy.from_dict(entry, origins) for entry in entries)
    names, deployments = set(), set()
    for policy in policies:
        route_names = {*policy.aliases, policy.physical_binding["provider_model_id"]}
        if names & route_names or policy.native_model_id in deployments:
            raise ValueError("ambiguous diagnostic alias, physical model or deployment")
        names.update(route_names)
        deployments.add(policy.native_model_id)
    return policies


class OperatorDiagnosticsMiddleware:
    def __init__(self, app, store, policy, authorize):
        self.app, self.store, self.authorize = app, store, authorize
        self.policies = (policy,) if isinstance(policy, DiagnosticPolicy) else tuple(policy or ())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in PATHS:
            return await self.app(scope,receive,send)
        wire = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            wire.extend(message.get("body",b""))
            if len(wire) > MAX_WIRE_BYTES:
                return await JSONResponse({"error":{"message":"private request exceeds wire bound"}},413)(scope,receive,send)
            if not message.get("more_body"):
                break
        def replay(payload):
            sent = False
            async def read():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type":"http.request","body":payload,"more_body":False}
                return await receive()
            return read
        try:
            body = json.loads(wire)
            # Existing facade-generated admissions stay on their original path.
            metadata = body.get("metadata",{}) if isinstance(body,dict) else {}
            if isinstance(metadata,dict) and metadata.get("semantier_attempt_id"):
                return await self.app(scope,replay(bytes(wire)),send)
            if len(wire) > 8192:
                return await JSONResponse({"error":{"message":"diagnostic request exceeds 8 KiB"}},413)(scope,receive,send)
            auth = await self.authorize(Request(scope,receive=replay(bytes(wire))))
            role = getattr(auth.user_role,"value",auth.user_role)
            if role != "proxy_admin" or not auth.user_id:
                raise HTTPException(403,"platform diagnostics require a proxy administrator")
            try:
                params = body.get("litellm_params", {}) if scope["path"] == "/health/test_connection" else body
                model = params.get("model") if isinstance(params, dict) else None
                p = next((policy for policy in self.policies if model in policy.aliases or
                    (scope["path"] == "/health/test_connection" and model == policy.physical_binding["provider_model_id"])), None)
                if p is None:
                    raise ValueError("Model is not enabled for platform diagnostics. Select: " +
                                     ", ".join(alias for policy in self.policies for alias in policy.aliases))
                body = p.prepare(body,scope["path"])
            except ValueError as exc:
                return await JSONResponse({"error":{"code":"diagnostic_validation_failed",
                    "message":str(exc)}},422)(scope,receive,send)
        except HTTPException as exc:
            return await JSONResponse({"error":{"message":str(exc.detail)}},exc.status_code)(scope,receive,send)
        except (ValueError,TypeError):
            return await JSONResponse({"error":{"message":"Use the configured MaaS model; diagnostics require a bounded chat request without endpoint, credential or payer overrides."}},422)(scope,receive,send)
        except Exception as exc:
            raw_status = getattr(exc, "status_code", getattr(exc, "code", 503))
            try:
                status = int(raw_status)
            except (TypeError, ValueError):
                status = 503
            status = status if 400 <= status <= 499 else 503
            return await JSONResponse({"error":{"message":"Diagnostic authentication failed or is unavailable."}},status)(scope,receive,send)
        try:
            admitted = await self.store.admit(organization_id=p.organization_id,account_id=p.account_id,
                operation="operator.diagnostic",idempotency_key=uuid4().hex,request_hash=canonical_hash(body),
                deployment_id=p.native_model_id,funding="CUSTOMER_CLOUD",currency="CNY",max_cost=p.reservation,
                attribution={"actor_user_id":auth.user_id,"purpose":"operator_diagnostic",
                    "provider":p.provider,"physical_binding":p.physical_binding,
                    "reservation_basis":"operator-authorized-exposure-not-supplier-price-v1"})
        except ValueError:
            return await JSONResponse({"error":{"message":"Platform diagnostic budget unavailable or exhausted; reconcile retained exposure before retrying."}},402)(scope,receive,send)
        invocation, attempt_id = admitted["invocation_id"], admitted["attempt_id"]
        await self.store.mark_dispatching(p.organization_id,invocation)
        capture = AttemptCapture(attempt_id,invocation_id=invocation,provider=p.provider,
                                 physical_binding=p.physical_binding,
                                 diagnostic_output_limit=32 if scope["path"] == "/health/test_connection" else
                                     body.get("max_completion_tokens",body.get("max_tokens")))
        token = _attempt.set(capture)
        if scope["path"] != "/health/test_connection":
            body["metadata"] = {"semantier_attempt_id":attempt_id,"semantier_invocation_id":invocation}
        async def traced_send(message):
            if message["type"] == "http.response.start":
                message = {**message,"headers":[*message.get("headers",[]),
                    (b"x-semantier-invocation-id",invocation.encode()),
                    (b"x-semantier-diagnostic-account",p.account_id.encode())]}
            await send(message)
        # Native auth caches parsed JSON on the shared ASGI scope. Once the
        # admitted body changes, invalidate that cache so the endpoint parses
        # the prepared wire (including output bounds and server metadata).
        payload = json.dumps(body).encode()
        prepared_scope = dict(scope)
        prepared_scope.pop("parsed_body", None)
        prepared_scope["headers"] = [
            (key, value) for key, value in scope.get("headers", [])
            if key.lower() != b"content-length"
        ] + [(b"content-length", str(len(payload)).encode())]
        try:
            await self.app(prepared_scope,replay(payload),traced_send)
        finally:
            _attempt.reset(token)
            if not capture.sends:
                await self.store.fail_before_send(p.organization_id,invocation)
            else:
                receipts = await self.store.read_raw_observations(p.organization_id,attempt_id)
                final = final_native_observation(receipts,p.provider)
                if final and final.get("raw_usage"):
                    await self.store.meter(p.organization_id,attempt_id,"diagnostic:"+attempt_id,
                                           p.provider,final["raw_usage"],None)
                    # Missing tariffs intentionally leave the hold pending reconciliation.
                    await self.store.settle(p.organization_id,invocation)
                else:
                    await self.store.mark_uncertain(p.organization_id,invocation)


async def authorize_proxy_admin(request):
    # Use the same production auth owner as the endpoint; never trust role headers.
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    return await user_api_key_auth(request,api_key=request.headers.get("authorization"),
        azure_api_key_header=None,anthropic_api_key_header=None,google_ai_studio_api_key_header=None,
        azure_apim_header=None,custom_litellm_key_header=None)
