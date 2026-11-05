"""Origin-bound synchronous HTTPX adapter for an explicitly selected H3 session."""
from urllib.parse import urlsplit
from importlib import import_module
import httpx


def _translate(error, request, *, reading=False):
    errors = import_module('curl_cffi.requests').exceptions
    if isinstance(error, errors.Timeout):
        kind = httpx.ReadTimeout if reading else httpx.TimeoutException
    elif isinstance(error, errors.RequestException):
        kind = httpx.ReadError if reading else httpx.ConnectError
    else:
        raise error
    raise kind('Cloud HTTP/3 transfer failed', request=request) from None


class _Stream(httpx.SyncByteStream):
    def __init__(self, response, request):
        self.response = response
        self.request = request

    def __iter__(self):
        try:
            yield from self.response.iter_content()
        except Exception as error:
            _translate(error, self.request, reading=True)

    def close(self):
        self.response.close()


class CloudHTTP3Transport(httpx.BaseTransport):
    def __init__(self, base_url, session):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'https' or parsed.username or parsed.password:
            raise ValueError('Cloud HTTP/3 requires an HTTPS origin')
        self.origin = (parsed.scheme, parsed.hostname, parsed.port or 443)
        self.session = session

    def handle_request(self, request):
        origin = (request.url.scheme, request.url.host, request.url.port or 443)
        if origin != self.origin:
            raise httpx.UnsupportedProtocol('Cloud HTTP/3 origin mismatch', request=request)
        try:
            response = self.session.request(method=request.method, url=str(request.url),
                headers=list(request.headers.multi_items()), data=request.read(),
                timeout=request.extensions.get('timeout', {}),
                cancel_event=request.extensions.get('semantier_h3_cancel'),
                # HTTPX owns decoding; curl must return the encoded response bytes.
                accept_encoding=None, stream=True, allow_redirects=False)
        except Exception as error:
            _translate(error, request)
        if response.http_version != 30:
            response.close()
            raise httpx.UnsupportedProtocol('Cloud HTTP/3 was not negotiated', request=request)
        try:
            return httpx.Response(response.status_code, headers=response.headers.multi_items(),
                stream=_Stream(response, request), extensions={'http_version': b'HTTP/3'})
        except BaseException:
            response.close()
            raise

    def close(self):
        self.session.close()


class _BodyBuffer:
    """Bound response bytes; a paused curl callback has consumed no bytes."""
    PAUSE = 0x10000001
    ERROR = 0xFFFFFFFF

    def __init__(self, capacity, resume, *, read_timeout=None):
        from collections import deque
        from threading import Condition
        if capacity <= 0:
            raise ValueError('Positive HTTP/3 buffer capacity required')
        self.capacity = capacity
        self.read_timeout = read_timeout
        self.resume = resume
        self.condition = Condition()
        self.chunks = deque()
        self.buffered_bytes = 0
        self.paused = False
        self.finished = False
        self.closed = False
        self.error = None

    def write(self, chunk):
        with self.condition:
            if self.closed or self.finished:
                return self.ERROR
            if len(chunk) > self.capacity:
                self.error = httpx.ReadError('Cloud HTTP/3 chunk exceeds buffer capacity')
                self.finished = True
                self.condition.notify_all()
                return self.ERROR
            if self.buffered_bytes + len(chunk) > self.capacity:
                self.paused = True
                return self.PAUSE
            self.chunks.append(bytes(chunk))
            self.buffered_bytes += len(chunk)
            self.condition.notify_all()
            return len(chunk)

    def read(self):
        import time
        resume = False
        deadline = None if self.read_timeout is None else time.monotonic() + self.read_timeout
        with self.condition:
            while not self.chunks and not self.finished and not self.closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise httpx.ReadTimeout('Cloud HTTP/3 response timed out')
                self.condition.wait(remaining)
            if not self.chunks:
                if self.error is not None and not self.closed:
                    raise self.error
                return None
            chunk = self.chunks.popleft()
            self.buffered_bytes -= len(chunk)
            if self.paused:
                self.paused = False
                resume = True
        if resume:
            self.resume()
        return chunk

    def finish(self, error=None):
        with self.condition:
            self.finished = True
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.chunks.clear()
            self.buffered_bytes = 0
            self.condition.notify_all()


class LoopCloudSession:
    """Own curl handles on one loop so closing a silent stream removes it now."""

    def __init__(self, *, cacert=None, native_ca=False, buffer_capacity=65536):
        import asyncio
        import threading
        self.cacert = cacert
        self.native_ca = native_ca
        self.buffer_capacity = buffer_capacity
        self.loop = asyncio.new_event_loop()
        self.closed = False
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       name='semantier-cloud-http3', daemon=True)
        self.thread.start()

    def request(self, *, method, url, headers, data, **kwargs):
        import asyncio
        import concurrent.futures
        import threading
        from curl_cffi import AsyncCurl, Curl, CurlHttpVersion, CurlOpt, CurlInfo
        if self.closed:
            raise httpx.ConnectError('Cloud HTTP/3 session is closed')
        ready = concurrent.futures.Future()
        done = threading.Event()
        timeout = kwargs.get('timeout') or {}
        body = _BodyBuffer(self.buffer_capacity, lambda: None,
                           read_timeout=timeout.get('read'))
        response = _LoopResponse(body, done)

        async def transfer():
            curl = Curl(cacert=self.cacert or '')
            multi = AsyncCurl(loop=self.loop)
            pending = None
            watchdog = None
            status = None
            response_headers = []
            def header(line):
                nonlocal status
                if line.startswith(b'HTTP/'):
                    parts = line.split()
                    status = int(parts[1])
                    response.http_version = 30 if parts[0] == b'HTTP/3' else 0
                    response_headers.clear()
                elif line in (b'\r\n', b'\n'):
                    if status is not None and status >= 200 and not ready.done():
                        response.status_code = status
                        response.headers = httpx.Headers(response_headers)
                        ready.set_result(response)
                elif b':' in line:
                    key, value = line.split(b':', 1)
                    response_headers.append((key.strip(), value.strip()))
                return len(line)
            def resume():
                if not done.is_set() and not body.closed:
                    curl.pause(0)
            body.resume = lambda: self.loop.call_soon_threadsafe(resume)
            try:
                curl.setopt(CurlOpt.URL, url.encode())
                curl.setopt(CurlOpt.CUSTOMREQUEST, method.encode())
                curl.setopt(CurlOpt.HTTP_VERSION, CurlHttpVersion.V3ONLY)
                curl.setopt(CurlOpt.PROXY, b'')
                curl.setopt(CurlOpt.FOLLOWLOCATION, 0)
                curl.setopt(CurlOpt.CONNECTTIMEOUT_MS, max(1, int(timeout.get('connect') * 1000))
                            if timeout.get('connect') is not None else 0)
                curl.setopt(CurlOpt.HTTPHEADER,
                    [f'{key}: {value}'.encode() for key, value in headers])
                if data or method not in ('GET', 'HEAD'):
                    curl.setopt(CurlOpt.POSTFIELDS, data)
                    curl.setopt(CurlOpt.POSTFIELDSIZE, len(data))
                if self.native_ca:
                    curl.setopt(CurlOpt.SSL_OPTIONS, 1 << 4)
                curl.setopt(CurlOpt.WRITEFUNCTION, body.write)
                curl.setopt(CurlOpt.HEADERFUNCTION, header)
                pending = multi.add_handle(curl)
                async def watch_headers():
                    # Read deadlines begin after the TLS/QUIC handshake and upload.
                    # Inspect native transfer state only from its owning event loop.
                    write_started = None
                    read_started = None
                    while not pending.done() and not ready.done():
                        await asyncio.sleep(.01)
                        if pending.done() or ready.done():
                            return
                        now = self.loop.time()
                        if not curl.getinfo(CurlInfo.APPCONNECT_TIME):
                            continue
                        if write_started is None:
                            write_started = now
                        uploading = curl.getinfo(CurlInfo.SIZE_UPLOAD_T) < len(data)
                        if not uploading and read_started is None:
                            read_started = now
                        phase = 'write' if uploading else 'read'
                        limit = timeout.get(phase)
                        started = write_started if uploading else read_started
                        if limit is not None and now - started >= limit:
                            error_type = httpx.WriteTimeout if uploading else httpx.ReadTimeout
                            ready.set_exception(error_type('Cloud HTTP/3 transfer timed out'))
                            multi.remove_handle(curl)
                            return
                watchdog = asyncio.create_task(watch_headers())
                await pending
                if not ready.done():
                    ready.set_exception(httpx.ConnectError('Cloud HTTP/3 response missing'))
                body.finish()
            except asyncio.CancelledError:
                body.close()
                if not ready.done():
                    ready.set_exception(httpx.ConnectError('Cloud HTTP/3 transfer cancelled'))
                raise
            except Exception as exc:
                # Native errors may contain request URLs or credential-bearing details.
                timed_out = getattr(exc, 'code', None) == 28
                error_type = httpx.ReadTimeout if timed_out else httpx.ReadError
                error = error_type('Cloud HTTP/3 transfer failed')
                if not ready.done():
                    connect_error = httpx.ConnectTimeout if timed_out else httpx.ConnectError
                    ready.set_exception(connect_error('Cloud HTTP/3 transfer failed'))
                body.finish(error)
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    await asyncio.gather(watchdog, return_exceptions=True)
                if pending is not None:
                    multi.remove_handle(curl)
                await multi.close()
                curl.close()
                done.set()
        response.task = asyncio.run_coroutine_threadsafe(transfer(), self.loop)
        try:
            cancel_event = kwargs.get('cancel_event')
            if cancel_event is None:
                return ready.result()
            while True:
                if cancel_event.is_set():
                    raise httpx.ConnectError('Cloud HTTP/3 transfer cancelled')
                try:
                    return ready.result(timeout=.05)
                except concurrent.futures.TimeoutError:
                    if ready.done():
                        raise
        except BaseException:
            response.close()
            raise

    def close(self):
        import asyncio
        if self.closed:
            return
        self.closed = True
        async def shutdown():
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Cloud HTTP/3 transport did not stop')
        self.loop.close()


class _LoopResponse:
    def __init__(self, body, done):
        self.body = body
        self.done = done
        self.task = None
        self.http_version = 0

    def iter_content(self):
        while (chunk := self.body.read()) is not None:
            yield chunk

    def close(self):
        self.body.close()
        if not self.done.is_set():
            self.task.cancel()
            if not self.done.wait(10):
                raise RuntimeError('Cloud HTTP/3 transfer did not close')


def cloud_http3_client(base_url, *, async_mode=False, settings=None):
    """Select only the configured Cloud origin; setup failures must propagate."""
    import sys
    if settings is None:
        from hermes_cli.cloud_device_client import _config
        settings = (_config().get('providers') or {}).get('semantier-cloud') or {}
    configured = settings.get('base_url')
    if not configured:
        return None
    def origin(value):
        parsed = urlsplit(value)
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)
    if origin(base_url) != origin(configured):
        return None
    version = settings.get('http_version', '3' if sys.platform == 'win32' else None)
    if version is None:
        return None
    if version != '3':
        raise ValueError('Unsupported Cloud HTTP version.')
    if urlsplit(base_url).scheme != 'https':
        raise ValueError('Cloud HTTP/3 requires HTTPS.')
    ca = settings.get('ssl_ca_cert')
    if ca is not None and (not isinstance(ca, str) or not ca.strip()):
        raise ValueError('Invalid Cloud CA certificate path.')
    # Resolve the native dependency before allocating a worker or returning a client.
    import_module('curl_cffi')
    session = LoopCloudSession(cacert=ca.strip() if ca else None,
                              native_ca=sys.platform == 'win32' and ca is None)
    try:
        transport = CloudHTTP3Transport(base_url, session)
        client_class = httpx.AsyncClient if async_mode else httpx.Client
        if async_mode:
            transport = AsyncCloudHTTP3Transport(transport)
        return client_class(transport=transport,
            trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(connect=15, read=None, write=15, pool=10))
    except BaseException:
        session.close()
        raise


class AsyncCloudHTTP3Transport(httpx.AsyncBaseTransport):
    def __init__(self, sync):
        self.sync = sync

    async def handle_async_request(self, request):
        import asyncio
        import threading
        payload = await request.aread()
        cancelled = threading.Event()
        extensions = {**request.extensions, 'semantier_h3_cancel': cancelled}
        sync_request = httpx.Request(request.method, request.url,
            headers=request.headers, content=payload, extensions=extensions)
        task = asyncio.create_task(asyncio.to_thread(self.sync.handle_request, sync_request))
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            results = await asyncio.gather(task, return_exceptions=True)
            if isinstance(results[0], httpx.Response):
                await asyncio.to_thread(results[0].close)
            raise
        return httpx.Response(response.status_code, headers=response.headers,
            stream=_AsyncStream(response.stream), extensions=response.extensions)

    async def aclose(self):
        import asyncio
        await asyncio.to_thread(self.sync.close)


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self, stream):
        self.stream = stream

    async def __aiter__(self):
        import asyncio
        iterator = iter(self.stream)
        sentinel = object()
        try:
            while True:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                yield chunk
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        import asyncio
        await asyncio.to_thread(self.stream.close)
