"""Channel1 private storage adapter used by the existing publication coordinator."""

import base64
from contextlib import contextmanager
import json
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from .artifact_io import OperationError, file_facts, safe_file
from .ecs_protocol import PREFIX, canonical, decode, headers_for, hex_value, sha
from .ecs_storage import credentials
from .remote_publication import publish_objects_then_head


class ECSTarget:
    provider = "ecs"

    def __init__(
        self,
        endpoint,
        public_base,
        config_hash,
        key,
        auth_revision,
        qualification_operation,
    ):
        for value in (endpoint, public_base):
            url = urlsplit(value)
            if (
                url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in ("", "/")
                or (
                    url.scheme != "https"
                    and not (
                        url.scheme == "http"
                        and url.hostname in ("127.0.0.1", "localhost")
                    )
                )
            ):
                raise ValueError("private storage requires a pinned HTTPS origin")
        self.endpoint = endpoint.rstrip("/")
        self.base = public_base.rstrip("/")
        self.config_hash = hex_value(config_hash, 64)
        self.key, self.auth_revision, self.qualification_operation = (
            key,
            auth_revision,
            hex_value(qualification_operation),
        )
        self.guard = lambda: None

    @classmethod
    def from_spec(cls, spec):
        revision, keys = credentials(spec["credential_file"])
        if len(keys) != 1:
            raise OperationError(503, "ecs_coordinator_key_selection_required")
        return cls(
            spec["endpoint"],
            spec["public_base"],
            spec["config_hash"],
            next(iter(keys.values())),
            revision,
            spec.get("qualification_operation", "0" * 32),
        )

    @classmethod
    def configured(cls, spec, store):
        target = cls.from_spec(spec)
        with store.connection() as c:
            row = c.execute(
                "SELECT body FROM operations WHERE operation_id=%s AND status='channel1_storage_qualified'",
                (spec["qualification_operation"],),
            ).fetchone()
        expected = dict(
            endpoint=target.endpoint,
            config_hash=target.config_hash,
            public_base=target.base,
        )
        if not row or any(row["body"].get(k) != v for k, v in expected.items()):
            raise OperationError(503, "ecs_storage_qualification_missing")
        return target

    def request(
        self,
        method,
        route,
        body=None,
        *,
        metadata=b"",
        path=None,
        commit_guard=lambda: None,
    ):
        self.guard()
        raw = b"" if body is None else canonical(body)
        urlpath = PREFIX + "/" + route
        size = path.stat().st_size if path else len(raw)
        digest = file_facts(path)[1] if path else sha(raw)
        headers = headers_for(self.key, method, urlpath, digest, metadata=metadata)
        headers["Content-Length"] = str(size)

        def content():
            with path.open("rb") as stream:
                part = stream.read(1024 * 1024)
                while part:
                    following = stream.read(1024 * 1024)
                    if not following:
                        self.guard()
                        commit_guard()
                    yield part
                    part = following

        try:
            with httpx.Client(
                timeout=httpx.Timeout(30, connect=5),
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    method,
                    self.endpoint + urlpath,
                    content=content() if path else raw,
                    headers=headers,
                ) as response:
                    result = bytearray()
                    for chunk in response.iter_bytes():
                        if len(result) + len(chunk) > 2 * 1024 * 1024:
                            raise OperationError(503, "ecs_response_limit")
                        result.extend(chunk)
                    try:
                        value = decode(bytes(result), maximum=2 * 1024 * 1024)
                    except OperationError as error:
                        raise OperationError(
                            503, "ecs_response_outcome_unknown"
                        ) from error
                    if response.status_code != 200:
                        raise OperationError(
                            response.status_code,
                            value.get("error", "ecs_endpoint_refused"),
                            mutation_outcome=value.get("outcome", "unknown"),
                        )
        except httpx.HTTPError as error:
            raise OperationError(503, "ecs_transport_outcome_unknown") from error
        if (
            value.get("config_hash") != self.config_hash
            or value.get("target_id") != "channel1"
        ):
            raise OperationError(503, "ecs_endpoint_binding_changed")
        return value

    def status(self, lock):
        return self.request("GET", "operations/" + lock["operation_id"])

    def read(self, key):
        # Public reads carry no private credential. Paths are metadata only.
        safe_file(Path("/"), key)
        if not key.startswith(
            ("releases/channels/", "releases/channel-builds/")
        ) or not key.endswith(".json"):
            raise OperationError(422, "ecs_public_metadata_scope")
        self.guard()
        try:
            with httpx.Client(
                timeout=httpx.Timeout(30, connect=5),
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    "GET", self.base + "/" + key, headers={"Cache-Control": "no-cache"}
                ) as response:
                    if response.status_code == 404:
                        return None, None
                    if response.status_code != 200:
                        raise OperationError(503, "ecs_public_read_unavailable")
                    result = bytearray()
                    for part in response.iter_bytes():
                        if len(result) + len(part) > 4 * 1024 * 1024:
                            raise OperationError(503, "ecs_public_metadata_limit")
                        result.extend(part)
                    return bytes(result), response.headers.get("etag")
        except httpx.HTTPError as error:
            raise OperationError(503, "ecs_public_read_unavailable") from error

    def initialize(self, head):
        raw, _ = self.read("releases/channels/" + head["name"] + ".json")
        if raw is None or json.loads(raw) != head:
            raise OperationError(503, "ecs_initial_head_requires_operator_provisioning")

    def reservation(self, owner, track):
        body = dict(
            version=1,
            config_hash=self.config_hash,
            operation_id=owner["operation_id"],
            track=track,
            incarnation=self.key.incarnation,
        )
        body["reservation_hash"] = sha(
            canonical(
                {**body, "target_id": "channel1", "principal": self.key.principal}
            )
        )
        return dict(
            provider="ecs",
            endpoint=self.endpoint,
            config_hash=self.config_hash,
            auth_revision=self.auth_revision,
            incarnation=self.key.incarnation,
            operation_id=owner["operation_id"],
            reservation_hash=body["reservation_hash"],
            acquire=body,
            generation=None,
            intent=None,
            owner=owner,
        )

    def identity(self, owner):
        return {
            **owner,
            "ecs": dict(
                endpoint=self.endpoint,
                config_hash=self.config_hash,
                incarnation=self.key.incarnation,
                auth_revision=self.auth_revision,
            ),
        }

    def fence(self, lock):
        body = {
            name: lock[name]
            for name in (
                "config_hash",
                "operation_id",
                "reservation_hash",
                "generation",
                "incarnation",
            )
        }
        body["version"] = 1
        if lock.get("intent"):
            body["intent_hash"] = sha(canonical(lock["intent"]))
        return body

    def check_binding(self, lock):
        if (lock.get("endpoint"), lock.get("config_hash")) != (
            self.endpoint,
            self.config_hash,
        ):
            raise OperationError(503, "ecs_captured_endpoint_changed")

    @contextmanager
    def locked(self, *, owner, reservation, persist, guard):
        previous_guard = self.guard
        self.guard = guard
        try:
            with self._locked(owner=owner, reservation=reservation, persist=persist):
                yield reservation
        finally:
            self.guard = previous_guard

    @contextmanager
    def _locked(self, *, owner, reservation, persist):
        self.check_binding(reservation)
        if reservation["generation"] is None:
            result = self.request(
                "POST", "reservations/acquire", reservation["acquire"]
            )
            reservation["generation"] = result["generation"]
            persist(reservation)
        else:
            status = self.status(reservation)
            if status["state"] != "RELEASED":
                self.verify_reservation(reservation)
        try:
            yield reservation
        finally:
            status = self.status(reservation)
            if status["state"] == "COMPLETE":
                self.request(
                    "POST",
                    "operations/" + reservation["operation_id"] + "/release",
                    {**self.fence(reservation), "outcome": "confirmed"},
                )
            elif status["state"] in ("RESERVED", "AUTHORIZED"):
                # Failure after an upload cannot have crossed head intent while
                # persisted state remains AUTHORIZED. COMMITTING is always held.
                self.request(
                    "POST",
                    "operations/" + reservation["operation_id"] + "/release",
                    {**self.fence(reservation), "outcome": "aborted"},
                )

    def verify_reservation(self, lock):
        self.check_binding(lock)
        status = self.status(lock)
        if (
            status["generation"],
            status["incarnation"],
            status["result"]["global_generation"],
            status["result"]["active_operation"],
        ) != (
            lock["generation"],
            self.key.incarnation,
            lock["generation"],
            lock["operation_id"],
        ):
            raise OperationError(409, "ecs_stale_reservation")
        return status

    def unresolved(self, lock):
        return bool(lock and self.status(lock)["state"] != "RELEASED")

    def recover_reservation(self, lock, old_owner, *, persist):
        self.check_binding(lock)
        try:
            status = self.status(lock)
        except OperationError as error:
            if (
                error.status != 404
                or lock.get("generation") is not None
                or lock.get("intent")
            ):
                raise
            if lock["incarnation"] != self.key.incarnation:
                fresh = self.reservation(
                    self.identity(lock["owner"]), lock["acquire"]["track"]
                )
                fresh["variant_hash"] = lock["variant_hash"]
                lock.clear()
                lock.update(fresh)
                persist(lock)
            return lock
        if lock["generation"] is None:
            lock["generation"] = status["generation"]
            persist(lock)
        takeover = lock.get("takeover")
        if (
            takeover
            and status["incarnation"] == self.key.incarnation
            and status["generation"] == takeover["expected_global_generation"]
            and status["intent_hash"] == takeover["intent_hash"]
            and status["result"]["reservation_hash"] == lock["reservation_hash"]
        ):
            lock.update(
                generation=status["generation"],
                incarnation=status["incarnation"],
                auth_revision=self.auth_revision,
            )
            persist(lock)
            return lock
        if status["state"] == "RELEASED":
            return lock
        if old_owner.get("ecs", {}).get("incarnation") == self.key.incarnation:
            # Only the same live coordinator may resume unchanged credentials.
            return lock
        intent_hash = sha(canonical(lock["intent"])) if lock.get("intent") else None
        takeover = dict(
            version=1,
            config_hash=self.config_hash,
            operation_id=lock["operation_id"],
            reservation_hash=lock["reservation_hash"],
            old_generation=lock["generation"],
            old_incarnation=lock["incarnation"],
            new_incarnation=self.key.incarnation,
            expected_global_generation=status["result"]["global_generation"],
            auth_revision=self.auth_revision,
            intent_hash=intent_hash,
        )
        lock["takeover"] = takeover
        persist(lock)  # durable intent BEFORE POST
        result = self.request(
            "POST", "operations/" + lock["operation_id"] + "/takeover", takeover
        )
        lock.update(
            generation=result["generation"],
            incarnation=result["incarnation"],
            auth_revision=self.auth_revision,
        )
        persist(lock)
        return lock

    def files(self, release, root):
        from .package_graph import package_objects

        files = {
            item["key"]: safe_file(root, item["key"])
            for package in release["manifest"]["packages"]
            for item in package_objects(
                package, release["manifest"]["request"]["buildId"], root
            )
        }
        files[release["manifest_key"]] = safe_file(root, release["manifest_key"])
        return files

    def preflight(self, release, root, *, reservation=None):
        if reservation:
            self.verify_reservation(reservation)
        status = self.status(
            reservation or {"operation_id": self.qualification_operation}
        )
        files = self.files(release, root)
        from .ecs_storage import MARGIN

        if (
            status["result"]["free_bytes"]
            < sum(path.stat().st_size for path in files.values()) + MARGIN
        ):
            raise OperationError(413, "ecs_storage_margin")
        return dict(
            delete_keys=[],
            current_bytes=status["result"]["storage_bytes"],
            incoming_allocation_bytes=sum(
                path.stat().st_size for path in files.values()
            ),
            projected_bytes=status["result"]["storage_bytes"]
            + sum(path.stat().st_size for path in files.values()),
            missing_keys=sorted(files),
            storage_config_hash=self.config_hash,
        )

    def publication_intent(self, release, root, previous, intended, lock, capacity):
        def head(record):
            raw = canonical(record)
            return dict(body=base64.b64encode(raw).decode(), sha256=sha(raw))

        if lock.get("intent"):
            expected_head = lock["intent"]["expected_head"]
            if json.loads(base64.b64decode(expected_head["body"])) != previous:
                raise OperationError(409, "ecs_previous_head_changed")
        else:
            raw, _ = self.read("releases/channels/" + intended["name"] + ".json")
            if raw is None or json.loads(raw) != previous:
                raise OperationError(409, "ecs_previous_head_changed")
            expected_head = dict(body=base64.b64encode(raw).decode(), sha256=sha(raw))
        return dict(
            version=1,
            target_id="channel1",
            config_hash=self.config_hash,
            operation_id=lock["operation_id"],
            track=intended["name"],
            kind="release",
            release_id=release["manifest"]["request"]["buildId"],
            variant_hash=lock["variant_hash"],
            expected_head=expected_head,
            intended_head=head(intended),
            objects=[
                dict(key=key, size=path.stat().st_size, sha256=file_facts(path)[1])
                for key, path in sorted(self.files(release, root).items())
            ],
            cleanup=[],
            allocation_bytes=capacity["incoming_allocation_bytes"],
        )

    def promote(self, release, root, previous, record, *, reservation, authorize):
        lock = reservation
        status = self.status(lock)
        terminal = status["state"] == "RELEASED"
        if terminal:
            if status["result"]["terminal_outcome"] != "confirmed":
                raise OperationError(409, "ecs_aborted_before_head")
            if (
                status["generation"],
                status["incarnation"],
                status["result"]["reservation_hash"],
                status["intent_hash"],
            ) != (
                lock["generation"],
                lock["incarnation"],
                lock["reservation_hash"],
                sha(canonical(lock["intent"])),
            ):
                raise OperationError(409, "ecs_terminal_binding_changed")
        else:
            status = self.verify_reservation(lock)
        expected = self.publication_intent(
            release,
            root,
            previous,
            record,
            lock,
            {"incoming_allocation_bytes": lock["intent"]["allocation_bytes"]},
        )
        if expected != lock["intent"]:
            raise OperationError(409, "ecs_durable_intent_changed")
        route = "operations/" + lock["operation_id"]
        fence = self.fence(lock)
        if status["state"] == "RESERVED":
            authorize()
            self.request(
                "POST",
                route + "/authorize",
                {
                    **{k: v for k, v in fence.items() if k != "intent_hash"},
                    "intent": lock["intent"],
                },
            )
            status = self.status(lock)
        actual, _ = self.read("releases/channels/" + record["name"] + ".json")
        recovering = actual == canonical(record)
        if terminal and not recovering:
            raise OperationError(503, "ecs_terminal_head_changed")
        if actual not in (
            base64.b64decode(lock["intent"]["expected_head"]["body"]),
            canonical(record),
        ):
            raise OperationError(409, "ecs_head_conflict")
        if status["state"] == "COMMITTING":
            authorize()
            self.request("POST", route + "/head", fence)
            status = self.status(lock)
        files = self.files(release, root)
        ordinals = {
            row["key"]: index for index, row in enumerate(lock["intent"]["objects"])
        }

        def upload(key, path):
            authorize()
            self.request(
                "PUT",
                route + "/objects/" + str(ordinals[key]),
                metadata=canonical(fence),
                path=path,
                commit_guard=authorize,
            )

        def verify(key, path):
            result = self.request(
                "GET",
                route + "/objects/" + str(ordinals[key]),
                metadata=canonical(fence),
            )["result"]
            if not result or (result["size"], result["sha256"]) != file_facts(path):
                raise OperationError(422, "ecs_remote_object_integrity")

        publish_objects_then_head(
            files,
            recovering=recovering,
            assert_reservation=lambda: self.verify_reservation(lock),
            upload=upload,
            verify=verify,
            authorize=authorize,
            commit_head=lambda: self.request("POST", route + "/head", fence),
            read_head=lambda: self.read(
                "releases/channels/" + record["name"] + ".json"
            )[0],
            intended_bytes=canonical(record),
            readback_error="ecs_head_readback_unknown",
        )
        if status["state"] not in ("COMPLETE", "RELEASED"):
            self.request(
                "POST",
                route + "/complete",
                {**fence, "observed_head_sha256": sha(canonical(record))},
            )
        return {"maintenance_status": "none"}
