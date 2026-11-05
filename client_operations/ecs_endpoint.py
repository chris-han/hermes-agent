"""Private fixed-scope HTTP storage API; the existing coordinator owns eligibility."""

import asyncio
import re
from fastapi import FastAPI, Request
from fastapi.responses import Response
from uvicorn.protocols.http.h11_impl import H11Protocol
from .artifact_io import OperationError
from .ecs_protocol import (
    PREFIX,
    MAX_JSON,
    authenticate,
    canonical,
    decode,
    fail,
    metadata_bytes,
    sha,
)


class BoundedHTTP(H11Protocol):
    """Bound connections before headers arrive; one request per connection."""

    def connection_made(self, transport):
        if len(self.connections) >= 16:
            transport.close()
            return
        super().connection_made(transport)
        self.header_prefix = b""
        self.header_done = False
        self.header_timer = self.loop.call_later(5, transport.close)

    def data_received(self, data):
        if not self.header_done:
            self.header_prefix += data[: 8193 - len(self.header_prefix)]
            marker = self.header_prefix.find(b"\r\n\r\n")
            if (marker < 0 and len(self.header_prefix) > 8192) or marker + 4 > 8192:
                self.transport.close()
                return
        super().data_received(data)
        if self.scope is not None and not self.header_done:
            self.header_done = True
            self.header_timer.cancel()

    def on_response_complete(self):
        self.transport.close()
        super().on_response_complete()

    def connection_lost(self, exc):
        timer = getattr(self, "header_timer", None)
        if timer:
            timer.cancel()
        if getattr(self, "transport", None) is not None:
            super().connection_lost(exc)


async def chunks(request):
    iterator = request.stream().__aiter__()
    while True:
        try:
            body = await asyncio.wait_for(anext(iterator), 30)
        except StopAsyncIteration:
            return
        except TimeoutError:
            fail(408, "ecs_body_idle_timeout")
        for offset in range(0, len(body), 1024 * 1024):
            yield body[offset : offset + 1024 * 1024]


def create_app(store):
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.api_route(PREFIX + "/{route:path}", methods=["GET", "PUT", "POST"])
    async def execute(request: Request, route: str):
        upload = None
        try:
            path = request.url.path
            if (
                request.scope["raw_path"] != path.encode("ascii")
                or request.scope["query_string"]
            ):
                fail(400, "ecs_route_invalid")
            pairs = request.scope["headers"]
            if len({name for name, _ in pairs}) != len(pairs):
                fail(400, "ecs_duplicate_header")
            headers = request.headers
            object_route = re.fullmatch(
                r"operations/([a-f0-9]{32})/objects/(0|[1-9][0-9]{0,2})", route
            )
            status_route = re.fullmatch(r"operations/([a-f0-9]{32})", route)
            mutation = re.fullmatch(
                r"operations/([a-f0-9]{32})/(authorize|takeover|head|complete|release|cleanup)",
                route,
            )
            if request.method == "PUT" and object_route:
                key = authenticate(
                    store.keys(),
                    headers,
                    "PUT",
                    path,
                    headers.get("x-semantier-body-sha256"),
                )
                body = decode(metadata_bytes(headers))
                ordinal = int(object_route[2])
                if body.get("operation_id") != object_route[1]:
                    fail(422, "ecs_route_binding_invalid")
                upload = await asyncio.to_thread(
                    store.begin_upload,
                    body,
                    ordinal,
                    key,
                    headers["x-semantier-request-id"],
                )
                if (
                    headers.get("content-length") != str(upload.row["size"])
                    or headers["x-semantier-body-sha256"] != upload.row["sha256"]
                ):
                    fail(422, "ecs_upload_binding_invalid")
                async for part in chunks(request):
                    await asyncio.to_thread(upload.write, part)
                result = await asyncio.to_thread(upload.finish)
            else:
                raw = bytearray()
                async for part in chunks(request):
                    if len(raw) + len(part) > MAX_JSON:
                        fail(413, "ecs_metadata_limit")
                    raw.extend(part)
                key = authenticate(
                    store.keys(), headers, request.method, path, sha(raw)
                )
                if request.method == "GET" and status_route:
                    if raw or metadata_bytes(headers):
                        fail(400, "ecs_get_body_invalid")
                    result = await asyncio.to_thread(
                        store.inspect, status_route[1], key
                    )
                elif request.method == "GET" and object_route:
                    if raw:
                        fail(400, "ecs_get_body_invalid")
                    body = decode(metadata_bytes(headers))
                    if body.get("operation_id") != object_route[1]:
                        fail(422, "ecs_route_binding_invalid")
                    result = await asyncio.to_thread(
                        store.object_facts, body, int(object_route[2]), key
                    )
                elif request.method == "POST" and (
                    route == "reservations/acquire" or mutation
                ):
                    if metadata_bytes(headers):
                        fail(400, "ecs_metadata_not_allowed")
                    body = decode(bytes(raw))
                    action = (
                        "acquire" if route == "reservations/acquire" else mutation[2]
                    )
                    if mutation and body.get("operation_id") != mutation[1]:
                        fail(422, "ecs_route_binding_invalid")
                    result = await asyncio.to_thread(
                        store.mutate,
                        action,
                        body,
                        key,
                        headers["x-semantier-request-id"],
                    )
                else:
                    fail(404, "ecs_route_missing")
            return Response(
                canonical(result),
                media_type="application/json",
                headers={"Cache-Control": "no-store", "Connection": "close"},
            )
        except OperationError as error:
            return Response(
                canonical(
                    dict(
                        error=error.code,
                        outcome=getattr(error, "mutation_outcome", "unknown"),
                    )
                ),
                status_code=error.status,
                media_type="application/json",
                headers={"Cache-Control": "no-store", "Connection": "close"},
            )
        finally:
            if upload and not upload.closed:
                await asyncio.to_thread(upload.abort)

    return app


def main():
    import argparse
    import os
    from pathlib import Path
    import stat
    import uvicorn
    from .ecs_filesystem import Root, absolute_directory
    from .ecs_protocol import fields
    from .ecs_storage import Storage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    parent = absolute_directory(args.config.parent)
    try:
        descriptor = os.open(
            args.config.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid not in (0, os.geteuid())
                or info.st_mode & 0o022
            ):
                fail(503, "ecs_configuration_permissions")
            config = decode(stream.read(MAX_JSON + 1))
    finally:
        os.close(parent)
    fields(
        config, ("endpoint_id", "principal", "tracks", "public_root", "private_root")
    )
    for name in ("public_root", "private_root"):
        fields(config[name], ("path", "device", "inode"))
    with Root(
        config["public_root"]["path"],
        {k: config["public_root"][k] for k in ("device", "inode")},
    ) as public:
        with Root(
            config["private_root"]["path"],
            {k: config["private_root"][k] for k in ("device", "inode")},
            private=True,
        ) as private:
            store = Storage(
                public,
                private,
                args.credentials,
                endpoint_id=config["endpoint_id"],
                principal=config["principal"],
                tracks=config["tracks"],
            )
            try:
                uvicorn.run(
                    create_app(store),
                    host=args.host,
                    port=args.port,
                    http=BoundedHTTP,
                    ws="none",
                    access_log=False,
                    server_header=False,
                )
            finally:
                store.close()


if __name__ == "__main__":
    main()
