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

    def __init__(self, capacity, resume):
        from collections import deque
        from threading import Condition
        if capacity <= 0:
            raise ValueError('Positive HTTP/3 buffer capacity required')
        self.capacity = capacity
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
        resume = False
        with self.condition:
            while not self.chunks and not self.finished and not self.closed:
                self.condition.wait()
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
        from curl_cffi import AsyncCurl, Curl, CurlHttpVersion, CurlOpt
        if self.closed:
            raise httpx.ConnectError('Cloud HTTP/3 session is closed')
        ready = concurrent.futures.Future()
        done = threading.Event()
        body = _BodyBuffer(self.buffer_capacity, lambda: None)
        response = _LoopResponse(body, done)

        async def transfer():
            curl = Curl(cacert=self.cacert or '')
            multi = AsyncCurl(loop=self.loop)
            pending = None
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
                curl.setopt(CurlOpt.CONNECTTIMEOUT_MS, 15000)
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
                await pending
                if not ready.done():
                    ready.set_exception(httpx.ConnectError('Cloud HTTP/3 response missing'))
                body.finish()
            except asyncio.CancelledError:
                body.close()
                if not ready.done():
                    ready.set_exception(httpx.ConnectError('Cloud HTTP/3 transfer cancelled'))
                raise
            except Exception:
                # Native errors may contain request URLs or credential-bearing details.
                error = httpx.ReadError('Cloud HTTP/3 transfer failed')
                if not ready.done():
                    ready.set_exception(httpx.ConnectError('Cloud HTTP/3 transfer failed'))
                body.finish(error)
            finally:
                if pending is not None:
                    multi.remove_handle(curl)
                await multi.close()
                curl.close()
                done.set()
        response.task = asyncio.run_coroutine_threadsafe(transfer(), self.loop)
        try:
            return ready.result()
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
