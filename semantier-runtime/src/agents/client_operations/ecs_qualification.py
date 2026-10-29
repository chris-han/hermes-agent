"""Existing operations journal owns channel1 inert storage qualification."""

import json
from pathlib import Path
from uuid import uuid4
import httpx
from .artifact_io import OperationError, immutable_write, safe_file
from .ecs_protocol import canonical, sha


def qualify(store, remote, actor, track, incoming_root):
    def authorize():
        if not {"release_read", "release_manage"} <= store.permissions(
            actor, track
        ) or not {"read", "publish"} <= store.target_permissions(actor, track).get(
            "channel1", set()
        ):
            raise OperationError(403, "explicit_platform_grant_required")

    authorize()
    binding = dict(
        endpoint=remote.endpoint,
        config_hash=remote.config_hash,
        public_base=remote.base,
        incarnation=remote.key.incarnation,
        auth_revision=remote.auth_revision,
    )

    def persist(operation, body, status):
        with store.connection() as c:
            c.execute(
                "UPDATE operations SET body=%s::jsonb,status=%s,updated_at=now() WHERE operation_id=%s AND user_id=%s",
                (json.dumps(body), status, operation, actor),
            )

    def reserve(kind):
        operation = uuid4().hex
        owner = dict(operation_id=operation, kind=kind)
        lock = remote.reservation(owner, track)
        body = {**binding, "remote_lock": lock}
        with store.connection() as c:
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (store.schema + ":ecs-qualification",),
            )
            pending = c.execute(
                "SELECT operation_id FROM operations WHERE status IN ('ecs_probe_pending','ecs_cleanup_pending') LIMIT 1"
            ).fetchone()
            if pending:
                raise OperationError(
                    503, "ecs_qualification_prior_outcome_requires_reconciliation"
                )
            c.execute(
                "INSERT INTO operations(operation_id,user_id,channel,idempotency_key,request_hash,body,status) VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)",
                (
                    operation,
                    actor,
                    track,
                    operation,
                    sha(canonical(body)),
                    json.dumps(body),
                    "ecs_" + kind + "_pending",
                ),
            )
        return lock, body

    before = remote.read("releases/channels/" + track + ".json")[0]
    lock, body = reserve("probe")
    operation = lock["operation_id"]
    raw = ("channel1 inert storage proof " + operation).encode()
    key = "releases/probes/channel1/" + operation + "/probe.bin"
    path = safe_file(Path(incoming_root), "qualifications/" + operation + "/probe.bin")
    immutable_write(path, raw)
    with remote.locked(
        owner=lock["owner"],
        reservation=lock,
        persist=lambda value: persist(
            operation, {**body, "remote_lock": value}, "ecs_probe_pending"
        ),
        guard=authorize,
    ):
        intent = dict(
            version=1,
            target_id="channel1",
            config_hash=remote.config_hash,
            operation_id=operation,
            track=track,
            kind="probe",
            release_id=None,
            variant_hash=None,
            expected_head=None,
            intended_head=None,
            objects=[dict(key=key, size=len(raw), sha256=sha(raw))],
            cleanup=[],
            allocation_bytes=len(raw),
        )
        lock["intent"] = intent
        body["remote_lock"] = lock
        persist(operation, body, "ecs_probe_pending")
        route = "operations/" + operation
        remote.request(
            "POST",
            route + "/authorize",
            {
                **{k: v for k, v in remote.fence(lock).items() if k != "intent_hash"},
                "intent": intent,
            },
        )
        result = remote.request(
            "PUT",
            route + "/objects/0",
            metadata=canonical(remote.fence(lock)),
            path=path,
            commit_guard=authorize,
        )
        identity = result["result"]
        with httpx.Client(
            timeout=httpx.Timeout(30, connect=5),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            with client.stream("GET", remote.base + "/" + key) as response:
                public = bytearray()
                for part in response.iter_bytes():
                    if len(public) + len(part) > 4096:
                        raise OperationError(503, "ecs_probe_public_limit")
                    public.extend(part)
                if response.status_code != 200 or bytes(public) != raw:
                    raise OperationError(503, "ecs_probe_public_readback_failed")
        body.update(
            probe_key=key,
            probe_identity=identity,
            public_sha256=sha(public),
            head_before_sha256=sha(before) if before else None,
        )
        persist(operation, body, "ecs_probe_pending")
    if remote.status(lock)["state"] != "RELEASED":
        raise OperationError(503, "ecs_probe_release_unknown")
    persist(operation, body, "ecs_probe_verified")
    cleanup, cleanup_body = reserve("cleanup")
    cleanup_id = cleanup["operation_id"]
    with remote.locked(
        owner=cleanup["owner"],
        reservation=cleanup,
        persist=lambda value: persist(
            cleanup_id, {**cleanup_body, "remote_lock": value}, "ecs_cleanup_pending"
        ),
        guard=authorize,
    ):
        row = {
            **{k: v for k, v in identity.items() if k != "created"},
            "key": key,
            "owner_operation_id": operation,
        }
        intent = dict(
            version=1,
            target_id="channel1",
            config_hash=remote.config_hash,
            operation_id=cleanup_id,
            track=track,
            kind="cleanup",
            release_id=None,
            variant_hash=None,
            expected_head=None,
            intended_head=None,
            objects=[],
            cleanup=[row],
            allocation_bytes=0,
        )
        cleanup["intent"] = intent
        cleanup_body["remote_lock"] = cleanup
        persist(cleanup_id, cleanup_body, "ecs_cleanup_pending")
        route = "operations/" + cleanup_id
        remote.request(
            "POST",
            route + "/authorize",
            {
                **{
                    k: v for k, v in remote.fence(cleanup).items() if k != "intent_hash"
                },
                "intent": intent,
            },
        )
        remote.request(
            "POST", route + "/cleanup", {**remote.fence(cleanup), "ordinals": [0]}
        )
    after = remote.read("releases/channels/" + track + ".json")[0]
    if after != before or remote.status(cleanup)["state"] != "RELEASED":
        raise OperationError(503, "ecs_qualification_final_state_unknown")
    cleanup_body.update(
        probe_operation=operation,
        probe_identity=identity,
        public_sha256=sha(raw),
        head_unchanged=True,
        probe_removed=True,
    )
    persist(cleanup_id, cleanup_body, "channel1_storage_qualified")
    return dict(qualification_operation=cleanup_id, **binding)
