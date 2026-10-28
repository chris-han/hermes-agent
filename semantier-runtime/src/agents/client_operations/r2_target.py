"""R2 publication target with separate private control custody."""
from contextlib import contextmanager, nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4
from xml.etree import ElementTree as ET

from .artifact_io import OperationError

from contracts.client_operations import canonical

MAX_BYTES = 10_000_000_000
MAX_RELEASES = 3
LOCK_KEY = "_control/publication.lock"
RETENTION_KEY = "_control/retention.json"
BUILD = re.compile(r"^releases/channel-builds/([a-f0-9]{32})/")


def capacity_plan(objects, manifests, protected, incoming_size=0, incoming_build=None, *, admission_order=None):
    """Conservative allocation includes duplicate/retry uploads and all unknown objects."""
    if type(incoming_size) is not int or incoming_size < 0:
        raise OperationError(422, "invalid_r2_allocation")
    groups = {}
    total = 0
    for key, size in objects.items():
        if type(size) is not int or size < 0:
            raise OperationError(503, "invalid_r2_inventory")
        total += size
        match = BUILD.match(key)
        if match:
            group = groups.setdefault(match[1], {"keys": [], "size": 0})
            group["keys"].append(key)
            group["size"] += size
    protected = set(protected) | ({incoming_build} if incoming_build else set())
    # Missing admission provenance still prevents deletion. It need not prevent
    # an allocation whose complete inventory fits without deleting anything.
    if len(set(groups) | protected) <= MAX_RELEASES and total + incoming_size < MAX_BYTES:
        return [], total + incoming_size
    admission_order = admission_order or {}
    def order(build):
        entry = admission_order.get(build, {})
        try:
            stamp = datetime.fromisoformat(entry['first_admitted_at'].replace('Z', '+00:00'))
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0 or entry['release_id'] != build:
                raise ValueError('invalid admission')
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            raise OperationError(503, 'r2_retention_order_unavailable') from error
        return stamp, build
    ordered = sorted(groups, key=order, reverse=True)
    keep = set(protected)
    for build in ordered:
        if len(keep) < MAX_RELEASES:
            keep.add(build)
    if len(keep) > MAX_RELEASES:
        raise OperationError(409, "r2_protected_release_count_exceeded")
    delete = []
    for build in reversed(ordered):
        if build not in keep or (total + incoming_size >= MAX_BYTES and build not in protected):
            delete.extend(groups[build]["keys"])
            total -= groups[build]["size"]
            keep.discard(build)
    if total + incoming_size >= MAX_BYTES:
        raise OperationError(413, "r2_storage_budget_exceeded")
    return delete, total + incoming_size


class R2Target:
    def __init__(self, transport, creds, base, bucket, *, control_target=None):
        self.transport, self.creds, self.base, self.bucket = transport, creds, base, bucket
        if control_target and (control_target.control_target or
                (control_target.base, control_target.bucket) == (base, bucket)):
            raise ValueError('R2 private control requires a separate non-nested bucket')
        self.control_target = control_target

    @classmethod
    def configured(cls, *, public_delivery=True):
        from agents.webapi.deps import ensure_repo_hermes_agent_on_path
        ensure_repo_hermes_agent_on_path()
        from scripts.releases import r2
        if any(not os.environ.get(key) for key in ('CLOUDFLARE_R2_ACCESS_KEY_ID','CLOUDFLARE_R2_SECRET_ACCESS_KEY','CLOUDFLARE_R2_ACCOUNT_ID','CLOUDFLARE_R2_BUCKET')):
            raise OperationError(503, 'delivery_target_credentials_unavailable')
        creds, base, bucket = r2.credentials()
        control = None
        if public_delivery:
            fields = ('CLOUDFLARE_R2_CONTROL_BUCKET', 'CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID',
                      'CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY')
            if any(not os.environ.get(key) for key in fields):
                raise OperationError(503, 'delivery_private_control_credentials_unavailable')
            control = cls(r2, {'access_key_id': os.environ[fields[1]], 'secret_key': os.environ[fields[2]]},
                          base, os.environ[fields[0]])
            r2.R2Scope.configured().bucket_url(base, control.bucket)
        return cls(r2, creds, base, bucket, control_target=control)

    def url(self, key):
        return self.transport.R2Scope.configured().object_url(self.base, self.bucket, key)

    def request(self, method, key, body=None, headers=None, query=""):
        if self.control_target and key.startswith('_control/'):
            return self.control_target.request(method, key, body, headers, query)
        if method == 'PUT' and key.startswith('releases/channels/'):
            headers = {**(headers or {}), 'Cache-Control': 'no-store', 'Content-Type': 'application/json'}
        try:
            return self.transport.signed_request(
                method, self.url(key) + query, creds=self.creds, now=self.transport.amz_timestamp(),
                body=body, body_hash=hashlib.sha256(body or b"").hexdigest(),
                content_length=len(body) if body is not None else None, extra_headers=headers)
        except self.transport.R2RequestError as error:
            if error.status in {404,412}:
                raise
            raise OperationError(503, 'delivery_target_unavailable') from error

    def listing(self, query):
        try:
            return self.transport.signed_request('GET', self.base+'/'+self.bucket+'?'+query,
                creds=self.creds, now=self.transport.amz_timestamp())
        except self.transport.R2RequestError as error:
            raise OperationError(503, 'delivery_target_unavailable') from error

    def read(self, key):
        try:
            result = self.request("GET", key)
            if len(result._body) > 4 * 1024 * 1024:
                raise OperationError(503, "r2_metadata_too_large")
            return result._body, result.header("etag")
        except self.transport.R2RequestError as error:
            if error.status == 404:
                return None, None
            raise

    @contextmanager
    def locked(self, owner=None, reservation=None):
        token = reservation or {"owner": owner or {"kind": "publisher"}, "id": uuid4().hex}
        if owner is not None and token.get("owner") != owner:
            raise OperationError(409, "r2_lock_intent_changed")
        body = canonical(token)
        try:
            self.request("PUT", LOCK_KEY, body, {"If-None-Match": "*"})
        except self.transport.R2RequestError as error:
            if error.status == 412:
                raise OperationError(409, "r2_publisher_busy") from error
            raise
        raw, _ = self.read(LOCK_KEY)
        if raw != body:
            raise OperationError(503, 'r2_lock_readback_failed')
        try:
            yield token
        finally:
            try:
                raw, etag = self.read(LOCK_KEY)
                if raw == body:
                    self.finish_upload_journal(token)
                    self.request("DELETE", LOCK_KEY, headers={"If-Match": etag})
            except self.transport.R2RequestError as error:
                raise OperationError(503, "r2_lock_release_unconfirmed") from error

    def upload_journal_key(self, reservation):
        token = reservation.get('id', '')
        if not re.fullmatch('[a-f0-9]{32}', token):
            raise OperationError(409, 'r2_upload_owner_invalid')
        return '_control/uploads/'+token+'.json'

    def journal_upload(self, reservation, key, upload_id):
        self.verify_reservation(reservation)
        journal_key = self.upload_journal_key(reservation)
        raw, _ = self.read(journal_key)
        journal = json.loads(raw) if raw else {'reservation': reservation, 'uploads': []}
        if journal.get('reservation') != reservation or not isinstance(upload_id, str) or not upload_id:
            raise OperationError(409, 'r2_upload_owner_invalid')
        match = BUILD.match(key)
        if not match or match[1] != reservation.get('owner', {}).get('release_id'):
            raise OperationError(409, 'r2_upload_scope_invalid')
        journal['uploads'].append({'key': key, 'upload_id': upload_id})
        body = canonical(journal)
        if len(body) > 16000:
            raise OperationError(413, 'r2_upload_journal_full')
        self.request('PUT', journal_key, body)
        if self.read(journal_key)[0] != body:
            raise OperationError(503, 'r2_upload_journal_unconfirmed')

    def pending_uploads(self):
        pending = self.listing('uploads=')
        root = ET.fromstring(pending._body)
        if root.findtext('{*}IsTruncated') == 'true':
            raise OperationError(503, 'r2_upload_inventory_incomplete')
        return [(row.findtext('{*}Key'), row.findtext('{*}UploadId')) for row in root.findall('{*}Upload')]

    def finish_upload_journal(self, reservation, *, abort=False):
        key = self.upload_journal_key(reservation)
        raw, etag = self.read(key)
        if raw is None:
            if self.pending_uploads():
                raise OperationError(409, 'r2_incomplete_upload_requires_recovery')
            return
        journal = json.loads(raw)
        if journal.get('reservation') != reservation:
            raise OperationError(409, 'r2_upload_owner_invalid')
        entries = journal.get('uploads')
        if not isinstance(entries, list) or len(entries) > 32:
            raise OperationError(503, 'r2_upload_journal_invalid')
        for item in entries:
            match = BUILD.match(item.get('key', ''))
            if (not match or match[1] != reservation.get('owner', {}).get('release_id')
                    or not isinstance(item.get('upload_id'), str) or not 0 < len(item['upload_id']) <= 2048):
                raise OperationError(503, 'r2_upload_journal_invalid')
        owned = {(item['key'], item['upload_id']) for item in entries}
        pending = set(self.pending_uploads())
        if not pending <= owned:
            raise OperationError(409, 'r2_unknown_upload_owner')
        if pending and not abort:
            raise OperationError(503, 'r2_upload_recovery_required')
        for upload_key, upload_id in sorted(pending):
            self.request('DELETE', upload_key, query='?'+self.transport.canonical_query({'uploadId': upload_id}))
        if self.pending_uploads():
            raise OperationError(503, 'r2_upload_abort_unconfirmed')
        self.request('DELETE', key, headers={'If-Match': etag})

    def verify_reservation(self, reservation):
        raw, _ = self.read(LOCK_KEY)
        if not reservation or raw != canonical(reservation):
            raise OperationError(409, 'r2_reservation_changed')

    def recover_owned_lock(self, reservation, owner, current_instance=None):
        raw, etag = self.read(LOCK_KEY)
        if raw is None:
            if not reservation:
                if self.pending_uploads():
                    raise OperationError(409, 'r2_unknown_upload_owner')
                return
            journal, _ = self.read(self.upload_journal_key(reservation))
            if journal is None:
                if self.pending_uploads():
                    raise OperationError(409, 'r2_unknown_upload_owner')
                return
        if not reservation or (raw is not None and raw != canonical(reservation)) or reservation.get('owner') != owner:
            raise OperationError(409, 'r2_lock_ownership_unresolved')
        import socket
        import psutil
        if owner.get('host') != socket.gethostname():
            raise OperationError(409, 'r2_lock_owner_unreachable')
        try:
            process = psutil.Process(owner['pid'])
            if process.create_time() == owner['start_time_seconds']:
                exact_instance = current_instance and all(owner.get(key) == value for key, value in current_instance.items())
                if not exact_instance:
                    raise OperationError(409, 'r2_lock_owner_still_running')
        except psutil.NoSuchProcess:
            pass
        # Unknown unfinished uploads require explicit transport-level recovery;
        # age alone never permits releasing a lock or abandoning their storage.
        self.finish_upload_journal(reservation, abort=True)
        self.inventory()
        if raw is not None:
            self.request('DELETE', LOCK_KEY, headers={'If-Match': etag})

    def retention(self):
        raw, _ = self.read(RETENTION_KEY)
        if raw is None:
            return {'schema': 1, 'admissions': {}, 'inflight': []}
        try:
            value = json.loads(raw)
            if value.get('schema') != 1 or not isinstance(value.get('admissions'), dict) or not isinstance(value.get('inflight'), list):
                raise ValueError('invalid retention projection')
            return value
        except (ValueError, TypeError, AttributeError) as error:
            raise OperationError(503, 'r2_retention_order_unavailable') from error

    def inventory(self):
        objects, manifests, heads = {}, {}, set()
        token = None
        while True:
            params = {"list-type": "2", "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            page = self.listing(self.transport.canonical_query(params))
            root = ET.fromstring(page._body)
            for item in root.findall("{*}Contents"):
                key = item.findtext("{*}Key")
                objects[key] = int(item.findtext("{*}Size"))
            token = root.findtext("{*}NextContinuationToken")
            if root.findtext("{*}IsTruncated") != "true":
                break
            if not token or len(objects) > 10000:
                raise OperationError(503, "r2_inventory_incomplete")
        if self.control_target:
            if any(key.startswith('_control/') for key in objects):
                # Public visibility is not control authorization. Preserve and
                # hold legacy records until an exact owned migration occurs.
                raise OperationError(503, 'r2_public_control_objects_require_migration')
            private_objects, _, _ = self.control_target.inventory()
            if any(not key.startswith('_control/') for key in private_objects):
                raise OperationError(503, 'r2_private_control_inventory_invalid')
            objects.update(private_objects)
        # An abandoned upload may still consume storage; never allocate through that uncertainty.
        if self.pending_uploads():
            raise OperationError(409, "r2_incomplete_upload_requires_recovery")
        for key in objects:
            if re.fullmatch(r"releases/channel-builds/[a-f0-9]{32}/build.json", key):
                raw, _ = self.read(key)
                manifest = json.loads(raw)
                build = BUILD.match(key)[1]
                if manifest.get("request", {}).get("buildId") != build or type(manifest["request"].get("sequence")) is not int:
                    raise OperationError(503, "r2_manifest_inventory_invalid")
                manifests[build] = manifest
            elif re.fullmatch(r"releases/channels/[a-z0-9-]+.json", key):
                raw, _ = self.read(key)
                record = json.loads(raw)
                if record.get("head"):
                    heads.add(record["head"]["buildId"])
        return objects, manifests, heads

    def initialize(self, record):
        key = "releases/channels/" + record["name"] + ".json"
        with self.locked():
            raw, _ = self.read(key)
            if raw is None:
                objects, manifests, heads = self.inventory()
                capacity_plan(objects, manifests, heads, len(canonical(record)), admission_order=self.retention()["admissions"])
                self.request("PUT", key, canonical(record), {"If-None-Match": "*"})
            elif any(json.loads(raw).get(field) != record.get(field) for field in ("name", "repository", "identity")):
                raise OperationError(503, "r2_channel_provenance_mismatch")
            # A DB-uncertain committed promotion is reconciled by promote, not overwritten here.

    def files(self, release, public_root):
        files = {}
        from .chunks import chunk_entries
        for package in release['manifest']['packages']:
            for chunk in chunk_entries(package['artifact'], release['manifest']['request']['buildId']):
                files[chunk['key']] = Path(public_root)/chunk['key']
            for item in [package['artifact'], package.get('distributionArchive'), package['feed']]:
                if item:
                    files[item['key']] = Path(public_root)/item['key']
        files[release['manifest_key']] = Path(public_root)/release['manifest_key']
        return files

    def preflight(self, release, public_root, *, reservation=None):
        # Read-only previews are advisory. Publication repeats this inventory
        # under its retained exclusive reservation before any mutation.
        with nullcontext(reservation) as held:
            if held:
                self.verify_reservation(held)
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            build = release['manifest']['request']['buildId']
            admission = release.get('retention_admission') or {
                'release_id': release['release_id'], 'first_admitted_at': release['received_at']}
            previous = projection['admissions'].get(build)
            if previous and previous != admission:
                raise OperationError(409, 'r2_first_admission_conflict')
            projection['admissions'][build] = admission
            missing = []
            for key, path in self.files(release, public_root).items():
                if key in objects:
                    self.transport.verify_remote_artifact(self.url(key), self.creds, self.transport.amz_timestamp(),
                        path.stat().st_size, self.transport.file_sha256(str(path)), algorithm='sha256', encoding='hex')
                else:
                    missing.append(key)
            # Deterministic bounded control/lock/head allowance, independent of
            # ephemeral lock ids; every actual object remains in inventory bytes.
            lock_bytes = objects.get(LOCK_KEY, 0)
            control = sum(size for key, size in objects.items() if key.startswith('_control/')) - lock_bytes
            allocation = max(0, sum(self.files(release, public_root)[key].stat().st_size for key in missing) + 65536 - control)
            incoming = max(0, allocation - lock_bytes)
            delete, projected = capacity_plan(objects, manifests, heads | set(projection['inflight']), incoming, build,
                                              admission_order=projection['admissions'])
            retained = {BUILD.match(key)[1] for key in objects if BUILD.match(key) and key not in delete} | {build}
            return {'delete_keys': sorted(delete), 'projected_bytes': projected, 'current_bytes': sum(objects.values()) - lock_bytes,
                    'incoming_allocation_bytes': allocation, 'retained_build_count': len(retained),
                    'protected_builds': sorted(heads | set(projection['inflight'])), 'max_bytes_exclusive': MAX_BYTES,
                    'missing_keys': sorted(missing), 'retention': projection}

    def promote(self, release, public_root, previous, record, *, reservation=None, authorize=lambda: None):
        key = 'releases/channels/'+record['name']+'.json'
        with nullcontext(reservation) if reservation else self.locked() as held:
            self.verify_reservation(held)
            raw, etag = self.read(key)
            actual = json.loads(raw) if raw else None
            recovering = actual == record
            if not recovering and actual != previous:
                raise OperationError(409, 'r2_channel_revision_conflict')
            files = self.files(release, public_root)
            if recovering:
                for file_key, file_path in files.items():
                    self.transport.verify_remote_artifact(self.url(file_key), self.creds, self.transport.amz_timestamp(),
                        file_path.stat().st_size, self.transport.file_sha256(str(file_path)), algorithm='sha256', encoding='hex')
            else:
                plan = self.preflight(release, public_root, reservation=held)
                authorize()
                projection = plan['retention']
                build = release['manifest']['request']['buildId']
                projection['inflight'] = sorted(set(projection['inflight']) | {build})
                if len(canonical(projection)) > 60000:
                    raise OperationError(413, 'r2_retention_projection_full')
                self.request('PUT', RETENTION_KEY, canonical(projection))
                for old in plan['delete_keys']:
                    self.request('DELETE', old)
                for file_key, file_path in files.items():
                    self.verify_reservation(held)
                    if file_key in plan['missing_keys']:
                        self.transport.put_object(self.creds, self.base, self.bucket, file_key, str(file_path),
                            self.transport.amz_timestamp(), self.transport.content_type_for(file_key), {'If-None-Match': '*'},
                            on_multipart_created=lambda upload_id, file_key=file_key: self.journal_upload(held, file_key, upload_id))
                    self.transport.verify_remote_artifact(self.url(file_key), self.creds, self.transport.amz_timestamp(),
                        file_path.stat().st_size, self.transport.file_sha256(str(file_path)), algorithm='sha256', encoding='hex')
                authorize()
                self.verify_reservation(held)
                self.request('PUT', key, canonical(record), {'If-Match': etag})
                readback, _ = self.read(key)
                if readback != canonical(record):
                    raise OperationError(503, 'r2_head_readback_failed')
            # A confirmed head remains Published even if subsequent maintenance fails.
            try:
                projection = self.retention()
                build = release['manifest']['request']['buildId']
                projection['inflight'] = [item for item in projection['inflight'] if item != build]
                self.request('PUT', RETENTION_KEY, canonical(projection))
                objects, manifests, heads = self.inventory()
                delete, _ = capacity_plan(objects, manifests, heads | set(projection['inflight']),
                                           admission_order=projection['admissions'])
                for old in delete:
                    self.request('DELETE', old)
                return {'maintenance_status': 'none'}
            except (OperationError, OSError, self.transport.R2RequestError):
                return {'maintenance_status': 'failed'}

    def cleanup(self, confirmed_build, authorize, *, reservation):
        """Retention-only recovery: never write a channel head."""
        with nullcontext(reservation):
            self.verify_reservation(reservation)
            objects, manifests, heads = self.inventory()
            projection = self.retention()
            projection['inflight'] = [build for build in projection['inflight'] if build != confirmed_build]
            delete, _ = capacity_plan(objects, manifests, heads | set(projection['inflight']),
                                       admission_order=projection['admissions'])
            authorize()
            self.request('PUT', RETENTION_KEY, canonical(projection))
            for key in delete:
                self.request('DELETE', key)
