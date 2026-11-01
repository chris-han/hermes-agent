"""Durable, fenced channel1 storage; release eligibility stays in the coordinator."""

from contextlib import contextmanager
from datetime import UTC, datetime
import base64
import fcntl
import hmac
import os
import re
from pathlib import Path
import stat
import threading
from uuid import uuid4

from .ecs_filesystem import Root, absolute_directory
from .artifact_io import OperationError
from .ecs_protocol import (
    Key,
    MAX_HEAD,
    MAX_JSON,
    MAX_OBJECT_BYTES,
    MAX_OBJECTS,
    canonical,
    decode,
    fail,
    fields,
    hex_value,
    integer,
    sha,
)

SLOT = 2 * 1024 * 1024
STATE_CAP = 64 * 1024 * 1024
ADMISSION_CAP = 60 * 1024 * 1024
MARGIN = 64 * 1024 * 1024
FENCE = (
    "version",
    "config_hash",
    "operation_id",
    "reservation_hash",
    "generation",
    "incarnation",
)


def credentials(path):
    path = Path(path)
    parent = absolute_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid not in (0, os.geteuid())
                or info.st_mode & 0o077
                or info.st_size > MAX_JSON
            ):
                fail(503, "ecs_credentials_invalid")
            value = decode(stream.read(MAX_JSON + 1))
    finally:
        os.close(parent)
    fields(value, ("revision", "keys"))
    integer(value["revision"])
    if type(value["keys"]) is not list or not 1 <= len(value["keys"]) <= 2:
        fail(503, "ecs_credentials_invalid")
    keys = {}
    for row in value["keys"]:
        fields(row, ("key_id", "principal", "incarnation", "secret_hex"))
        try:
            key = Key(
                row["key_id"],
                row["principal"],
                row["incarnation"],
                bytes.fromhex(row["secret_hex"]),
            )
        except (ValueError, TypeError):
            fail(503, "ecs_credentials_invalid")
        if key.key_id in keys:
            fail(503, "ecs_credentials_invalid")
        keys[key.key_id] = key
    if len({(key.principal, key.incarnation) for key in keys.values()}) != 1:
        fail(503, "ecs_credentials_invalid")
    return value["revision"], keys


class Storage:
    def __init__(self, public, private, auth_file, *, endpoint_id, principal, tracks):
        if os.geteuid() == 0:
            fail(503, "ecs_nonroot_identity_required")
        if public.device != private.device:
            fail(503, "ecs_cross_device_staging")
        if (
            public.path == private.path
            or public.path.is_relative_to(private.path)
            or private.path.is_relative_to(public.path)
        ):
            fail(503, "ecs_private_storage_must_be_disjoint")
        info = os.fstat(private.fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            fail(503, "ecs_private_storage_permissions")
        self.public, self.private, self.auth_file = public, private, Path(auth_file)
        self.principal = principal
        self.tracks = set(tracks)
        hex_value(endpoint_id)
        if not self.tracks or any(
            not isinstance(track, str)
            or re.fullmatch("[a-z0-9][a-z0-9_-]{0,63}", track) is None
            for track in tracks
        ):
            fail(503, "ecs_track_scope_invalid")
        self.config_hash = sha(
            canonical(
                dict(
                    version=1,
                    endpoint_id=endpoint_id,
                    target_id="channel1",
                    principal=principal,
                    tracks=sorted(tracks),
                    public_root=dict(path=str(public.path), **public.identity),
                    private_root=dict(path=str(private.path), **private.identity),
                )
            )
        )
        self.mutex = threading.RLock()
        self.active_writers = set()
        self.fault = None
        self.lock_depth = 0
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        private = self.private
        auth_file = self.auth_file
        principal = self.principal
        with private.parent("endpoint.lock", create=True) as (parent, name):
            self.lifetime = os.open(
                name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=parent
            )
        try:
            fcntl.flock(self.lifetime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail(503, "ecs_endpoint_already_running")
        with private.parent("state.lock", create=True) as (parent, name):
            self.lockfd = os.open(
                name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=parent
            )
        with self.locked():
            revision, keys = credentials(auth_file)
            key = next(iter(keys.values()))
            if key.principal != principal:
                fail(503, "ecs_credential_principal_mismatch")
            if self._read("global") is None:
                self._write(
                    "global",
                    dict(
                        config_hash=self.config_hash,
                        generation=0,
                        active=None,
                        auth_revision=revision,
                        incarnation=key.incarnation,
                        auth_digest=self._auth_digest(revision, keys),
                        key_ids=sorted(keys),
                        migration=None,
                    ),
                )
            self.reload_credentials()
            self._startup()

    def close(self):
        for name in ("lockfd", "lifetime"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)

    @contextmanager
    def locked(self):
        with self.mutex:
            if not self.lock_depth:
                fcntl.flock(self.lockfd, fcntl.LOCK_EX)
            self.lock_depth += 1
            try:
                self.public.check()
                self.private.check()
                yield
            finally:
                self.lock_depth -= 1
                if not self.lock_depth:
                    fcntl.flock(self.lockfd, fcntl.LOCK_UN)

    @staticmethod
    def _auth_digest(revision, keys):
        return sha(
            canonical(
                dict(
                    revision=revision,
                    keys=[
                        dict(
                            key_id=k.key_id,
                            principal=k.principal,
                            incarnation=k.incarnation,
                            secret_sha256=sha(k.secret),
                        )
                        for k in sorted(keys.values(), key=lambda k: k.key_id)
                    ],
                )
            )
        )

    def _path(self, name):
        return "state/" + name + ".record"

    def _read(self, name):
        try:
            with self.private.reader(self._path(name)) as stream:
                header = stream.read(72)
                length = int(header[:8], 16)
                if (
                    os.fstat(stream.fileno()).st_size != SLOT
                    or not 0 < length <= SLOT - 72
                ):
                    raise ValueError("state size")
                body = stream.read(length)
                if header[8:].decode() != sha(body):
                    raise ValueError("state integrity")
                return decode(body, maximum=SLOT)
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeError, OperationError):
            fail(503, "ecs_state_integrity_unknown")

    def _write(self, name, value):
        body = canonical(value)
        if len(body) > SLOT - 72:
            fail(503, "ecs_operation_recovery_capacity")
        if self.state_bytes() + SLOT > STATE_CAP:
            fail(503, "ecs_state_recovery_capacity")
        temporary = "state-temp-" + uuid4().hex
        with self.private.creator(temporary) as stream:
            os.posix_fallocate(stream.fileno(), 0, SLOT)
            stream.write(f"{len(body):08x}".encode() + sha(body).encode() + body)
            stream.flush()
            os.fsync(stream.fileno())
        self.private.replace_from(self.private, temporary, self._path(name))

    def state_bytes(self):
        return sum(
            info.st_size
            for key, info in self.private.entries()
            if key.startswith(("state/", "state-temp-")) and stat.S_ISREG(info.st_mode)
        )

    def _operations(self):
        return [
            self._read(key.removeprefix("state/").removesuffix(".record"))
            for key, _ in self.private.entries()
            if key.startswith("state/op-") and key.endswith(".record")
        ]

    def reload_credentials(self):
        with self.locked():
            global_state = self._read("global")
            revision, keys = credentials(self.auth_file)
            key = next(iter(keys.values()))
            if global_state["config_hash"] != self.config_hash:
                fail(503, "ecs_storage_configuration_changed")
            if (
                key.principal != self.principal
                or revision < global_state["auth_revision"]
            ):
                fail(503, "ecs_auth_revision_rollback")
            digest = self._auth_digest(revision, keys)
            if revision == global_state["auth_revision"]:
                if digest != global_state["auth_digest"]:
                    fail(503, "ecs_auth_revision_required")
                return
            if key.incarnation == global_state["incarnation"] or set(keys) & set(
                global_state["key_ids"]
            ):
                fail(503, "ecs_replacement_requires_fresh_incarnation_key")
            previous = global_state["generation"]
            global_state["generation"] += 1
            operation = (
                self._read("op-" + global_state["active"])
                if global_state["active"]
                else None
            )
            global_state.update(
                auth_revision=revision,
                auth_digest=digest,
                key_ids=sorted(keys),
                incarnation=key.incarnation,
                migration=dict(
                    operation_id=global_state["active"],
                    old_generation=operation["generation"] if operation else previous,
                    old_incarnation=operation["incarnation"] if operation else None,
                    generation=global_state["generation"],
                    incarnation=key.incarnation,
                    auth_revision=revision,
                ),
            )
            self._write("global", global_state)

    def keys(self):
        revision, keys = credentials(self.auth_file)
        state = self._read("global")
        if revision < state["auth_revision"]:
            fail(401, "ecs_auth_revision_rollback")
        return keys

    def _actor(self, key):
        revision, keys = credentials(self.auth_file)
        installed = keys.get(key.key_id)
        state = self._read("global")
        if (
            installed is None
            or installed.principal != key.principal
            or installed.incarnation != key.incarnation
            or not hmac.compare_digest(installed.secret, key.secret)
        ):
            fail(401, "ecs_credential_revoked")
        if (
            revision != state["auth_revision"]
            or self._auth_digest(revision, keys) != state["auth_digest"]
        ):
            fail(503, "ecs_auth_reload_required")
        return state

    def _startup(self):
        state = self._read("global")
        active = [op for op in self._operations() if op["state"] != "RELEASED"]
        if len(active) > 1:
            fail(503, "ecs_active_operation_unknown")
        if not state["active"] and active:
            op = active[0]
            if op["state"] != "RESERVED" or op["generation"] != state["generation"] + 1:
                fail(503, "ecs_reservation_recovery_unknown")
            state.update(active=op["operation_id"], generation=op["generation"])
            self._write("global", state)
        if state["active"]:
            op = self._read("op-" + state["active"])
            if not op:
                fail(503, "ecs_active_operation_unknown")
            if op["state"] == "RELEASED":
                state.update(
                    active=None,
                    generation=max(state["generation"], op["generation"] + 1),
                )
                self._write("global", state)
            elif op["state"] == "COMMITTING":
                self._reconcile_head(op)
            self._recover_links(op)

    def _recover_links(self, op):
        changed = False
        for ordinal, records in op["stages"].items():
            row = op["intent"]["objects"][int(ordinal)]
            for record in records:
                if record["state"] != "LINKING":
                    continue
                actual = self.public.facts(row["key"])
                if actual is not None:
                    if (actual["size"], actual["sha256"]) != (
                        row["size"],
                        row["sha256"],
                    ):
                        fail(503, "ecs_link_outcome_unknown")
                    previous = op["objects"].get(ordinal)
                    created = actual == record["facts"] or previous == {
                        **actual,
                        "created": True,
                    }
                    op["objects"][ordinal] = {**actual, "created": created}
                    record["state"] = "LINKED"
                    changed = True
        if changed:
            self._write("op-" + op["operation_id"], op)

    def _response(self, op, result=None):
        return dict(
            version=1,
            target_id="channel1",
            config_hash=self.config_hash,
            operation_id=op["operation_id"],
            intent_hash=op.get("intent_hash"),
            generation=op["generation"],
            incarnation=op["incarnation"],
            state=op["state"],
            result=result,
        )

    def inspect(self, operation_id, key):
        """Observe recovery facts without spending journal capacity or changing state."""
        hex_value(operation_id)
        with self.locked():
            state = self._actor(key)
            op = self._read("op-" + operation_id)
            if op is None:
                fail(404, "ecs_operation_missing")
            result = dict(
                global_generation=state["generation"],
                auth_revision=state["auth_revision"],
                active_operation=state["active"],
                reservation_hash=op["reservation_hash"],
                intent=op.get("intent"),
                objects=op["objects"],
                stages=op["stages"],
                storage_bytes=sum(
                    info.st_size
                    for _, info in self.public.entries()
                    if stat.S_ISREG(info.st_mode)
                ),
                state_bytes=self.state_bytes(),
                terminal_outcome=op.get("terminal_outcome"),
            )
            volume = os.fstatvfs(self.public.fd)
            result["free_bytes"] = volume.f_bavail * volume.f_frsize
            if op.get("intent") and op["intent"]["kind"] == "release":
                current = self.public.read(self._head_key(op), MAX_HEAD)
                result["observed_head_sha256"] = (
                    sha(current) if current is not None else None
                )
            return self._response(op, result)

    def object_facts(self, body, ordinal, key):
        fields(body, FENCE, ("intent_hash",))
        if type(body["version"]) is not int or body["version"] != 1:
            fail(400, "ecs_version_invalid")
        with self.locked():
            self._actor(key)
            op = self._read("op-" + hex_value(body["operation_id"]))
            if not op or not op.get("intent"):
                fail(404, "ecs_operation_missing")
            if any(
                body.get(name) != op.get(name)
                for name in (
                    "config_hash",
                    "reservation_hash",
                    "generation",
                    "incarnation",
                    "intent_hash",
                )
            ):
                fail(409, "ecs_facts_binding_changed")
            integer(ordinal, maximum=len(op["intent"]["objects"]) - 1, minimum=0)
            row = op["intent"]["objects"][ordinal]
            return self._response(op, self.public.facts(row["key"]))

    def _fence(self, body, key, op):
        if type(body["version"]) is not int or body["version"] != 1:
            fail(400, "ecs_version_invalid")
        integer(body["generation"])
        state = self._actor(key)
        if (
            body["config_hash"] != self.config_hash
            or body["version"] != 1
            or body["incarnation"] != key.incarnation
            or op["incarnation"] != key.incarnation
            or body["reservation_hash"] != op["reservation_hash"]
            or body["generation"] != op["generation"]
            or op["generation"] != state["generation"]
            or state["active"] != op["operation_id"]
        ):
            fail(409, "ecs_stale_fence")
        if (
            op.get("intent_hash")
            and (
                body.get("intent_hash")
                or (sha(canonical(body["intent"])) if "intent" in body else None)
            )
            != op["intent_hash"]
        ):
            fail(409, "ecs_intent_changed")

    def _replay(self, action, body, request_id):
        digest = sha(canonical(body))
        for op in self._operations():
            prior = op["requests"].get(request_id)
            if prior:
                if prior["action"] != action or prior["body_sha256"] != digest:
                    fail(409, "ecs_request_id_changed")
                return prior.get("response")
        return None

    def mutate(self, action, body, key, request_id):
        if type(body.get("version")) is not int:
            fail(400, "ecs_version_invalid")
        hex_value(request_id)
        hex_value(body.get("operation_id"))
        with self.locked():
            self._actor(key)
            if body.get("config_hash") != self.config_hash or body.get("version") != 1:
                fail(409, "ecs_configuration_changed")
            op = self._read("op-" + body["operation_id"])
            if action == "acquire" and op and op["state"] == "RELEASED":
                fail(409, "ecs_terminal_operation")
            if action != "takeover" and body.get("incarnation") != key.incarnation:
                fail(403, "ecs_incarnation_mismatch")
            replay = self._replay(action, body, request_id)
            if replay is not None:
                return replay
            if action == "acquire":
                return self._acquire(body, key, request_id, op)
            if op is None:
                fail(404, "ecs_operation_missing")
            if action == "takeover":
                return self._takeover(body, key, request_id, op)
            extra = {
                "authorize": ("intent",),
                "head": (),
                "complete": ("observed_head_sha256",),
                "release": ("outcome",),
                "cleanup": ("ordinals",),
            }
            if action not in extra:
                fail(400, "ecs_action_invalid")
            fields(body, (*FENCE, *extra[action]), ("intent_hash",))
            if action == "release" and op["state"] == "RELEASED":
                if (
                    body["reservation_hash"] != op["reservation_hash"]
                    or body["generation"] != op["generation"]
                    or body["outcome"] != op.get("terminal_outcome")
                    or body.get("intent_hash") != op.get("intent_hash")
                ):
                    fail(409, "ecs_terminal_operation")
                return self._response(op)
            self._fence(body, key, op)
            if (
                action == "authorize"
                and op["state"] == "AUTHORIZED"
                and sha(canonical(body["intent"])) == op["intent_hash"]
            ):
                return self._response(op)
            if action == "head" and op["state"] in ("COMMITTED", "COMPLETE"):
                self._head(op, key)
                return self._response(op)
            if action == "complete" and op["state"] == "COMPLETE":
                if (
                    body["observed_head_sha256"]
                    != op["intent"]["intended_head"]["sha256"]
                ):
                    fail(409, "ecs_commit_unconfirmed")
                return self._response(op)
            if (
                len(op["requests"])
                >= 32 + len(op.get("intent", {}).get("objects", [])) * 24
            ):
                fail(503, "ecs_request_recovery_capacity")
            if action == "authorize":
                self._authorize(body, op)
            op["requests"][request_id] = dict(
                action=action, body_sha256=sha(canonical(body)), response=None
            )
            self._write("op-" + op["operation_id"], op)
            handlers = {
                "authorize": lambda: None,
                "head": lambda: self._head(op, key),
                "complete": lambda: self._complete(body, op),
                "release": lambda: self._release(body, op),
                "cleanup": lambda: self._cleanup(body, op),
            }
            handlers[action]()
            response = self._response(op)
            op["requests"][request_id] = dict(
                action=action, body_sha256=sha(canonical(body)), response=response
            )
            self._write("op-" + op["operation_id"], op)
            return response

    def _complete(self, body, op):
        if (
            op["state"] != "COMMITTED"
            or body["observed_head_sha256"] != op["intent"]["intended_head"]["sha256"]
        ):
            fail(409, "ecs_commit_unconfirmed")
        op["state"] = "COMPLETE"

    def _acquire(self, body, key, request_id, op):
        fields(
            body,
            (
                "version",
                "config_hash",
                "operation_id",
                "track",
                "incarnation",
                "reservation_hash",
            ),
        )
        if body["track"] not in self.tracks or body["incarnation"] != key.incarnation:
            fail(403, "ecs_scope_denied")
        expected = sha(
            canonical(
                {
                    **{k: v for k, v in body.items() if k != "reservation_hash"},
                    "target_id": "channel1",
                    "principal": key.principal,
                }
            )
        )
        if body["reservation_hash"] != expected:
            fail(422, "ecs_reservation_hash_invalid")
        state = self._read("global")
        if op:
            if (
                op["reservation_hash"] != expected
                or op["incarnation"] != key.incarnation
                or op["generation"] != state["generation"]
            ):
                fail(409, "ecs_reservation_changed")
            return self._response(op)
        if state["active"]:
            fail(409, "ecs_target_reserved")
        if self.state_bytes() >= ADMISSION_CAP:
            fail(503, "ecs_state_admission_capacity")
        generation = integer(state["generation"] + 1)
        op = dict(
            operation_id=body["operation_id"],
            track=body["track"],
            principal=key.principal,
            config_hash=self.config_hash,
            reservation_hash=expected,
            incarnation=key.incarnation,
            generation=generation,
            state="RESERVED",
            intent_hash=None,
            requests={},
            objects={},
            stages={},
            incarnations=1,
            produced_at=datetime.now(UTC).isoformat(),
        )
        response = self._response(op)
        op["requests"][request_id] = dict(
            action="acquire", body_sha256=sha(canonical(body)), response=response
        )
        self._write("op-" + op["operation_id"], op)
        state.update(active=op["operation_id"], generation=generation)
        self._write("global", state)
        return response

    def _authorize(self, body, op):
        intent = body["intent"]
        validate_intent(intent, op, self.config_hash)
        digest = sha(canonical(intent))
        if op["state"] == "AUTHORIZED" and op["intent_hash"] == digest:
            return
        if op["state"] != "RESERVED":
            fail(409, "ecs_authorization_state")
        if (
            os.fstatvfs(self.public.fd).f_bavail * os.fstatvfs(self.public.fd).f_frsize
            < intent["allocation_bytes"] + MARGIN
        ):
            fail(413, "ecs_storage_margin")
        op.update(state="AUTHORIZED", intent=intent, intent_hash=digest)

    def _head_key(self, op):
        return "releases/channels/" + op["track"] + ".json"

    def _reconcile_head(self, op):
        current = self.public.read(self._head_key(op), MAX_HEAD)
        intent = op["intent"]
        expected = unpack_head(intent["expected_head"])
        intended = unpack_head(intent["intended_head"])
        op["state"] = (
            "COMMITTED"
            if current == intended
            else "AUTHORIZED"
            if current == expected
            else "CONFLICT"
        )
        self._write("op-" + op["operation_id"], op)

    def _head(self, op, key):
        if op["state"] == "COMMITTING":
            self._reconcile_head(op)
        if op["state"] in ("COMMITTED", "COMPLETE"):
            if self.public.read(self._head_key(op), MAX_HEAD) != unpack_head(
                op["intent"]["intended_head"]
            ):
                op["state"] = "CONFLICT"
                self._write("op-" + op["operation_id"], op)
                fail(503, "ecs_committed_head_changed")
            return
        if op["state"] != "AUTHORIZED" or op["intent"]["kind"] != "release":
            fail(409, "ecs_head_state_invalid")
        for row in op["intent"]["objects"]:
            facts = self.public.facts(row["key"])
            if not facts or (facts["size"], facts["sha256"]) != (
                row["size"],
                row["sha256"],
            ):
                fail(422, "ecs_mandatory_object_unverified")
        if self.public.read(self._head_key(op), MAX_HEAD) != unpack_head(
            op["intent"]["expected_head"]
        ):
            fail(409, "ecs_head_conflict")
        temporary = "head-" + op["operation_id"]
        op["state"] = "COMMITTING"
        stage = op.setdefault("head_stage", dict(key=temporary, facts=None))
        if stage["key"] != temporary:
            fail(503, "ecs_head_stage_binding_changed")
        self._write("op-" + op["operation_id"], op)
        intended = unpack_head(op["intent"]["intended_head"])
        existing = self.private.facts(temporary)
        if existing is None:
            self.private.write_new(temporary, intended)
        elif (existing["size"], existing["sha256"]) != (
            len(intended),
            sha(intended),
        ) or (stage["facts"] is not None and existing != stage["facts"]):
            fail(503, "ecs_head_stage_integrity_unknown")
        stage["facts"] = self.private.facts(temporary)
        self._write("op-" + op["operation_id"], op)
        if self.fault:
            self.fault("before_head_replace")
        self._actor(key)
        self.public.replace_from(self.private, temporary, self._head_key(op))
        if self.fault:
            self.fault("after_head_replace")
        if self.public.read(self._head_key(op), MAX_HEAD) != unpack_head(
            op["intent"]["intended_head"]
        ):
            fail(503, "ecs_head_readback_unknown")
        op["state"] = "COMMITTED"

    def _release(self, body, op):
        if (op["state"] == "COMPLETE" and body["outcome"] == "confirmed") or (
            op["state"] in ("RESERVED", "AUTHORIZED")
            and body["outcome"] == "aborted"
            and not any(item[0] == op["operation_id"] for item in self.active_writers)
        ):
            op["state"] = "RELEASED"
            op["terminal_outcome"] = body["outcome"]
            # The operation tombstone is durable before the global reservation clears.
            self._write("op-" + op["operation_id"], op)
            state = self._read("global")
            state.update(active=None, generation=state["generation"] + 1)
            self._write("global", state)
        else:
            fail(409, "ecs_release_reconciliation_required")

    def _takeover(self, body, key, request_id, op):
        fields(
            body,
            (
                "version",
                "config_hash",
                "operation_id",
                "reservation_hash",
                "old_generation",
                "old_incarnation",
                "new_incarnation",
                "expected_global_generation",
                "auth_revision",
                "intent_hash",
            ),
        )
        state = self._actor(key)
        migration = state["migration"]
        if (
            not migration
            or migration["operation_id"] != op["operation_id"]
            or state["active"] != op["operation_id"]
            or body["reservation_hash"] != op["reservation_hash"]
            or body["intent_hash"] != op["intent_hash"]
            or body["new_incarnation"] != key.incarnation
            or body["auth_revision"] != state["auth_revision"]
            or body["expected_global_generation"] != state["generation"]
            or body["old_generation"] != migration["old_generation"]
            or body["old_incarnation"] != migration["old_incarnation"]
            or op["state"] == "RELEASED"
        ):
            fail(409, "ecs_takeover_fence_invalid")
        adopted = (op["generation"], op["incarnation"]) == (
            state["generation"],
            key.incarnation,
        )
        if adopted:
            return self._response(op)
        if not adopted:
            if (op["generation"], op["incarnation"]) != (
                body["old_generation"],
                body["old_incarnation"],
            ):
                fail(409, "ecs_takeover_changed")
            if op["incarnations"] >= 8:
                fail(503, "ecs_incarnation_capacity")
            if any(slot[0] == op["operation_id"] for slot in self.active_writers):
                fail(409, "ecs_writer_still_active")
            op.update(
                generation=state["generation"],
                incarnation=key.incarnation,
                incarnations=op["incarnations"] + 1,
            )
            if op["state"] == "COMMITTING":
                self._reconcile_head(op)
        response = self._response(op)
        op["requests"][request_id] = dict(
            action="takeover", body_sha256=sha(canonical(body)), response=response
        )
        self._write("op-" + op["operation_id"], op)
        return response

    def _cleanup(self, body, op):
        if op["state"] != "AUTHORIZED" or op["intent"]["kind"] != "cleanup":
            fail(409, "ecs_cleanup_state_invalid")
        ordinals = body["ordinals"]
        if (
            type(ordinals) is not list
            or len(ordinals) > 16
            or len(set(ordinals)) != len(ordinals)
        ):
            fail(400, "ecs_cleanup_ordinals_invalid")
        for ordinal in ordinals:
            integer(ordinal, maximum=len(op["intent"]["cleanup"]) - 1, minimum=0)
            row = op["intent"]["cleanup"][ordinal]
            expected = {
                k: v for k, v in row.items() if k not in ("key", "owner_operation_id")
            }
            owner = self._read("op-" + row["owner_operation_id"])
            if not owner or owner["state"] != "RELEASED":
                fail(409, "ecs_cleanup_owner_not_reconciled")
            owned = any(
                item["key"] == row["key"]
                and owner["objects"].get(str(index)) == {**expected, "created": True}
                for index, item in enumerate(owner.get("intent", {}).get("objects", []))
            )
            if not owned:
                fail(409, "ecs_cleanup_ownership_unproven")
            release = row["key"].split("/")[2]
            for track in self.tracks:
                raw = self.public.read("releases/channels/" + track + ".json", MAX_HEAD)
                if raw is not None:
                    try:
                        import json

                        head = json.loads(raw)
                        from hermes_cli.release_channels import validate_record

                        validate_record(head)
                    except (ValueError, UnicodeError):
                        fail(409, "ecs_cleanup_head_unknown")
                    if release in str(head):
                        fail(409, "ecs_cleanup_head_referenced")
            self.public.unlink(row["key"], expected)
            op.setdefault("cleaned", {})[str(ordinal)] = True
            self._write("op-" + op["operation_id"], op)

    def begin_upload(self, body, ordinal, key, request_id):
        with self.locked():
            fields(body, FENCE, ("intent_hash",))
            op = self._read("op-" + body["operation_id"])
            if not op:
                fail(404, "ecs_operation_missing")
            self._fence(body, key, op)
            if op["state"] != "AUTHORIZED":
                fail(409, "ecs_upload_not_authorized")
            integer(ordinal, maximum=len(op["intent"]["objects"]) - 1, minimum=0)
            hex_value(request_id)
            slot = (op["operation_id"], ordinal)
            if slot in self.active_writers:
                fail(409, "ecs_upload_active")
            records = op["stages"].setdefault(str(ordinal), [])
            prior = next(
                (
                    record
                    for records in op["stages"].values()
                    for record in records
                    if record["request_id"] == request_id
                ),
                None,
            )
            if prior and prior in records and prior["state"] == "LINKED":
                self.active_writers.add(slot)
                return Upload(self, body, ordinal, key, request_id, None)
            if prior:
                fail(409, "ecs_upload_request_already_recorded")
            if len(records) >= 8:
                fail(503, "ecs_upload_recovery_capacity")
            volume = os.fstatvfs(self.private.fd)
            if (
                volume.f_bavail * volume.f_frsize
                < op["intent"]["objects"][ordinal]["size"] + MARGIN
            ):
                fail(413, "ecs_storage_margin")
            stage = "upload-" + op["operation_id"] + "-" + uuid4().hex
            records.append(
                dict(
                    key=stage,
                    request_id=request_id,
                    generation=op["generation"],
                    state="RECEIVING",
                    facts=None,
                )
            )
            self._write("op-" + op["operation_id"], op)
            self.active_writers.add(slot)
            try:
                return Upload(self, body, ordinal, key, request_id, stage)
            except BaseException:
                self.active_writers.discard(slot)
                raise


def unpack_head(value):
    if value is None:
        return None
    fields(value, ("body", "sha256"))
    try:
        raw = base64.b64decode(value["body"], validate=True)
    except (ValueError, TypeError):
        fail(400, "ecs_head_encoding_invalid")
    if len(raw) > MAX_HEAD or sha(raw) != value["sha256"]:
        fail(422, "ecs_head_integrity_invalid")
    return raw


def validate_intent(intent, op, config_hash):
    if type(intent.get("version")) is not int or intent["version"] != 1:
        fail(400, "ecs_version_invalid")
    fields(
        intent,
        (
            "version",
            "target_id",
            "config_hash",
            "operation_id",
            "track",
            "kind",
            "release_id",
            "variant_hash",
            "expected_head",
            "intended_head",
            "objects",
            "cleanup",
            "allocation_bytes",
        ),
    )
    if any(
        intent[k] != v
        for k, v in dict(
            version=1,
            target_id="channel1",
            config_hash=config_hash,
            operation_id=op["operation_id"],
            track=op["track"],
        ).items()
    ):
        fail(422, "ecs_intent_binding_invalid")
    if intent["kind"] not in ("release", "probe", "cleanup"):
        fail(422, "ecs_intent_kind_invalid")
    if (
        type(intent["objects"]) is not list
        or len(intent["objects"]) > MAX_OBJECTS
        or type(intent["cleanup"]) is not list
        or len(intent["cleanup"]) > 16
    ):
        fail(413, "ecs_inventory_limit")
    keys = set()
    total = 0
    for row in intent["objects"]:
        fields(row, ("key", "size", "sha256"))
        integer(row["size"], MAX_OBJECT_BYTES)
        hex_value(row["sha256"], 64)
        Root.parts(row["key"])
        if row["key"].endswith((".msix", ".msixbundle")) and row["size"] > 480_000_000:
            fail(413, "ecs_native_cache_cap")
        if row["key"] in keys:
            fail(422, "ecs_duplicate_object")
        keys.add(row["key"])
        total += row["size"]
        if intent["kind"] == "release":
            hex_value(intent["release_id"])
            hex_value(intent["variant_hash"], 64)
            if not row["key"].startswith(
                "releases/channel-builds/" + intent["release_id"] + "/"
            ):
                fail(422, "ecs_object_scope_invalid")
        elif intent["kind"] == "probe":
            if (
                row["key"]
                != "releases/probes/channel1/" + op["operation_id"] + "/probe.bin"
                or row["size"] > 4096
            ):
                fail(422, "ecs_probe_scope_invalid")
        else:
            fail(422, "ecs_cleanup_cannot_upload")
    integer(intent["allocation_bytes"], maximum=128 * MAX_OBJECT_BYTES, minimum=0)
    if intent["allocation_bytes"] < total:
        fail(422, "ecs_allocation_incomplete")
    if intent["kind"] == "release":
        if not intent["objects"] or unpack_head(intent["intended_head"]) is None:
            fail(422, "ecs_release_head_required")
        unpack_head(intent["expected_head"])
    elif intent["expected_head"] is not None or intent["intended_head"] is not None:
        fail(422, "ecs_probe_cannot_change_head")
    if intent["kind"] != "cleanup" and intent["cleanup"]:
        fail(422, "ecs_cleanup_scope_invalid")
    for row in intent["cleanup"]:
        fields(
            row,
            ("key", "device", "inode", "type", "size", "sha256", "owner_operation_id"),
        )
        Root.parts(row["key"])
        hex_value(row["owner_operation_id"])
        hex_value(row["sha256"], 64)
        integer(row["device"])
        integer(row["inode"])
        integer(row["size"], minimum=0)
        if row["type"] != "regular" or not row["key"].startswith(
            ("releases/channel-builds/", "releases/probes/channel1/")
        ):
            fail(422, "ecs_cleanup_scope_invalid")


class Upload:
    def __init__(self, store, body, ordinal, key, request_id, stage):
        self.store, self.body, self.ordinal, self.key, self.request_id, self.stage = (
            store,
            dict(body),
            ordinal,
            key,
            request_id,
            stage,
        )
        self.opid = body["operation_id"]
        self.slot = (self.opid, ordinal)
        self.row = store._read("op-" + self.opid)["intent"]["objects"][ordinal]
        self.written = 0
        self.hasher = __import__("hashlib").sha256()
        self.closed = False
        self.context = store.private.creator(stage) if stage else None
        self.stream = self.context.__enter__() if self.context else None

    def write(self, body):
        if (
            self.closed
            or len(body) > 1024 * 1024
            or self.written + len(body) > self.row["size"]
        ):
            fail(413, "ecs_upload_length_invalid")
        if self.stream:
            self.stream.write(body)
        self.hasher.update(body)
        self.written += len(body)

    def finish(self):
        try:
            if self.closed:
                fail(409, "ecs_upload_closed")
            if self.stream:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self.context.__exit__(None, None, None)
            self.closed = True
            with self.store.locked():
                op = self.store._read("op-" + self.opid)
                self.store._fence(self.body, self.key, op)
                if (
                    self.written != self.row["size"]
                    or self.hasher.hexdigest() != self.row["sha256"]
                ):
                    fail(422, "ecs_upload_integrity_invalid")
                if self.stage is None:
                    actual = self.store.public.facts(self.row["key"])
                    owned = op["objects"].get(str(self.ordinal))
                    if (
                        not actual
                        or owned is None
                        or {k: v for k, v in owned.items() if k != "created"} != actual
                    ):
                        fail(409, "ecs_upload_retry_identity_changed")
                    return self.store._response(op, owned)
                record = next(
                    row
                    for row in op["stages"][str(self.ordinal)]
                    if row["key"] == self.stage
                )
                record.update(
                    state="LINKING", facts=self.store.private.facts(self.stage)
                )
                self.store._write("op-" + self.opid, op)
                created = self.store.public.publish(
                    self.store.private,
                    self.stage,
                    self.row["key"],
                    self.row["sha256"],
                    self.row["size"],
                )
                facts = self.store.public.facts(self.row["key"])
                # Only the staged inode can establish ownership after response loss.
                created = (
                    created
                    or facts == record["facts"]
                    or op["objects"].get(str(self.ordinal))
                    == {**facts, "created": True}
                )
                op["objects"][str(self.ordinal)] = {**facts, "created": created}
                record["state"] = "LINKED"
                self.store._write("op-" + self.opid, op)
                self.store.private.unlink(self.stage, record["facts"])
                return self.store._response(op, op["objects"][str(self.ordinal)])
        finally:
            self.store.active_writers.discard(self.slot)

    def abort(self):
        if not self.closed:
            if self.context:
                self.context.__exit__(None, None, None)
            self.closed = True
        self.store.active_writers.discard(self.slot)
