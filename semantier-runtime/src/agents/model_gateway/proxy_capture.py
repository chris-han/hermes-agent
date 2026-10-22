"""Native HTTP receipt capture, loaded only by the private source Proxy service."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from urllib.parse import urlsplit

import httpx


@dataclass
class AttemptCapture:
    attempt_id: str
    sends: int = 0
    invocation_id: str | None = None
    provider: str | None = None
    physical_binding: dict | None = None


_attempt: ContextVar[AttemptCapture | None] = ContextVar('semantier_proxy_attempt', default=None)
MAX_WIRE_BYTES = 8 * 1024 * 1024


def native_usage(wire: bytes, content_type: str):
    """Read native counters; never substitute normalized SDK or estimated usage."""
    if 'amazon.eventstream' in content_type:
        from botocore.eventstream import EventStreamBuffer
        buffer = EventStreamBuffer()
        buffer.add_data(wire)
        usage, stopped = None, False
        for event in buffer:
            kind = event.headers.get(':event-type')
            body = json.loads(event.payload)
            if kind == 'messageStop':
                stopped = True
            if kind == 'metadata':
                usage = body.get('usage')
        return usage, stopped and bool(usage)
    if 'text/event-stream' in content_type:
        usage, done = None, False
        for line in wire.decode('utf-8').splitlines():
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                done = True
            elif payload:
                value = json.loads(payload)
                if value.get('usage'):
                    usage = value['usage']
        return usage, done and bool(usage)
    value = json.loads(wire)
    usage = value.get('usage')
    return usage, bool(usage)


class ReceiptStream(httpx.AsyncByteStream):
    def __init__(self, original, *, store, attempt, provider, content_type, status, request_id):
        self.original = original
        self.store, self.attempt, self.provider = store, attempt, provider
        self.content_type, self.status, self.request_id = content_type, status, request_id
        self.blocks, self.size, self.overflow, self.recorded = [], 0, False, False
        self.digest = hashlib.sha256()

    async def _record(self, exhausted):
        if self.recorded:
            return
        usage, complete = None, False
        response_model = None
        if not self.overflow:
            try:
                wire = b''.join(self.blocks)
                usage, complete = native_usage(wire, self.content_type)
                if 'text/event-stream' in self.content_type:
                    models = [json.loads(line[5:].strip()).get('model')
                              for line in wire.decode('utf-8').splitlines()
                              if line.startswith('data:') and line[5:].strip() not in ('', '[DONE]')]
                    response_model = next((m for m in reversed(models) if isinstance(m,str)), None)
                elif 'amazon.eventstream' not in self.content_type:
                    response_model = json.loads(wire).get('model')
                if not isinstance(response_model,str) or len(response_model)>200:
                    response_model = None
            except Exception:
                pass
        native_terminal = ('text/event-stream' in self.content_type or
                           'amazon.eventstream' in self.content_type)
        final = complete and self.status < 400 and (exhausted or native_terminal)
        state = 'FINAL' if final else 'PARTIAL' if usage else 'MISSING'
        digest = self.digest.hexdigest()
        await self.store.append_raw_observation(
            self.attempt.attempt_id, f'native:{self.attempt.sends}:{digest}', self.provider,
            usage, {'transport': 'httpx-native-wire-v1', 'wire_sha256': digest,
                    'content_type': self.content_type, 'http_status': self.status,
                    'response_model': response_model, 'response_model_basis': 'provider_reported' if response_model else 'UNRESOLVED',
                    'provider_request_id': self.request_id, 'body_limit_exceeded': self.overflow},
            self.attempt.sends, state)
        self.recorded = True

    async def __aiter__(self):
        exhausted = False
        try:
            async for block in self.original:
                self.digest.update(block)
                self.size += len(block)
                if self.size <= MAX_WIRE_BYTES:
                    self.blocks.append(block)
                else:
                    self.overflow = True
                    self.blocks.clear()
                yield block
            exhausted = True
        finally:
            await self._record(exhausted)

    async def aclose(self):
        try:
            await self.original.aclose()
        finally:
            await self._record(False)


class CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, store, origins, *, transport=None):
        self.store = store
        self.origins = dict(origins)
        self.native = transport or httpx.AsyncHTTPTransport(retries=0, trust_env=False)

    async def handle_async_request(self, request):
        origin = str(request.url.copy_with(path='/', query=None, fragment=None)).rstrip('/')
        provider = self.origins.get(origin)
        attempt = _attempt.get()
        if not provider or attempt is None or attempt.provider != provider:
            raise RuntimeError('model egress requires admitted attempt and matching allowed origin')
        if attempt.sends:
            raise RuntimeError('a physical send is already claimed for this attempt')
        from agents.model_gateway.physical_binding import verify_native_request
        verify_native_request(attempt.physical_binding,provider,request)
        await self.store.claim_native_send(attempt.attempt_id,attempt.invocation_id,provider)
        attempt.sends = 1
        response = await self.native.handle_async_request(request)
        response.stream = ReceiptStream(response.stream, store=self.store, attempt=attempt,
            provider=provider, content_type=response.headers.get('content-type', ''),
            status=response.status_code, request_id=response.headers.get('apim-request-id') or
            response.headers.get('x-amzn-requestid') or response.headers.get('x-request-id'))
        return response

    async def aclose(self):
        await self.native.aclose()


class AdmittedAttemptMiddleware:
    """The private listener receives only the facade's server-generated metadata."""
    def __init__(self, app, store):
        self.app, self.store = app, store

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') not in ('/v1/chat/completions', '/chat/completions'):
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            messages.append(message)
            size += len(message.get('body', b''))
            if size > MAX_WIRE_BYTES or message['type'] != 'http.request':
                raise ValueError('invalid private invocation body')
            if not message.get('more_body'):
                break
        body = json.loads(b''.join(m.get('body', b'') for m in messages))
        attempt_id = body.get('metadata', {}).get('semantier_attempt_id')
        if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id) > 200:
            raise ValueError('admitted attempt required')
        invocation_id = body.get('metadata', {}).get('semantier_invocation_id')
        if not isinstance(invocation_id,str):
            raise ValueError('admitted invocation required')
        admission = await self.store.capture_admission(attempt_id,invocation_id)
        token = _attempt.set(AttemptCapture(attempt_id,invocation_id=invocation_id,provider=admission['provider'],physical_binding=admission['attribution'].get('physical_binding')))
        iterator = iter(messages)
        async def replay():
            return next(iterator, None) or await receive()
        try:
            await self.app(scope, replay, send)
        finally:
            _attempt.reset(token)


def install_native_capture(store, origins):
    """Call in the running Proxy event loop before native inference is accepted."""
    import litellm
    from litellm.llms.custom_httpx.http_handler import configure_semantier_async_transport, AsyncHTTPHandler
    for origin in origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path not in ('', '/'):
            raise ValueError('exact model origin required')
    litellm.aclient_session = httpx.AsyncClient(transport=CaptureTransport(store, origins),
                                               trust_env=False, timeout=10.0)
    configure_semantier_async_transport(lambda: CaptureTransport(store, origins))
    handler = AsyncHTTPHandler(timeout=httpx.Timeout(10.0))
    for key in ('async_httpx_clienttimeout_' + str(httpx.Timeout(10.0)) + 'bedrock',
                'async_httpx_clientbedrock'):
        litellm.in_memory_llm_clients_cache.set_cache(key=key,value=handler,ttl=3600)
