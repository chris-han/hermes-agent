"""Private R2 publication target, reusing Hermes' qualified SigV4/multipart transport."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4
from xml.etree import ElementTree as ET

from .publication import OperationError, canonical

MAX_BYTES = 10_000_000_000
MAX_RELEASES = 3
LOCK_KEY = "_control/publication.lock"
BUILD = re.compile(r"^releases/channel-builds/([a-f0-9]{32})/")


def capacity_plan(objects, manifests, protected, incoming_size=0, incoming_build=None):
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
    ordered = sorted(groups, key=lambda b: (manifests.get(b, {}).get("request", {}).get("sequence", -1), b), reverse=True)
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
    def __init__(self, transport, creds, base, bucket):
        self.transport, self.creds, self.base, self.bucket = transport, creds, base, bucket

    @classmethod
    def configured(cls):
        from agents.webapi.deps import ensure_repo_hermes_agent_on_path
        ensure_repo_hermes_agent_on_path()
        from scripts.releases import r2
        creds, base, bucket = r2.credentials()
        return cls(r2, creds, base, bucket)

    def url(self, key):
        return self.transport.R2Scope.configured().object_url(self.base, self.bucket, key)

    def request(self, method, key, body=None, headers=None, query=""):
        return self.transport.signed_request(
            method, self.url(key) + query, creds=self.creds, now=self.transport.amz_timestamp(),
            body=body, body_hash=hashlib.sha256(body or b"").hexdigest(),
            content_length=len(body) if body is not None else None, extra_headers=headers)

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
    def locked(self):
        body = canonical({"owner": "publisher", "id": uuid4().hex})
        try:
            self.request("PUT", LOCK_KEY, body, {"If-None-Match": "*"})
        except self.transport.R2RequestError as error:
            if error.status == 412:
                raise OperationError(409, "r2_publisher_busy") from error
            raise
        try:
            yield
        finally:
            self.request("DELETE", LOCK_KEY)

    def inventory(self):
        objects, manifests, heads = {}, {}, set()
        token = None
        while True:
            params = {"list-type": "2", "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            page = self.transport.signed_request(
                "GET", self.base + "/" + self.bucket + "?" + self.transport.canonical_query(params),
                creds=self.creds, now=self.transport.amz_timestamp())
            root = ET.fromstring(page._body)
            for item in root.findall("{*}Contents"):
                key = item.findtext("{*}Key")
                objects[key] = int(item.findtext("{*}Size"))
            token = root.findtext("{*}NextContinuationToken")
            if root.findtext("{*}IsTruncated") != "true":
                break
            if not token or len(objects) > 10000:
                raise OperationError(503, "r2_inventory_incomplete")
        # An abandoned upload may still consume storage; never allocate through that uncertainty.
        pending = self.transport.signed_request(
            "GET", self.base + "/" + self.bucket + "?uploads=",
            creds=self.creds, now=self.transport.amz_timestamp())
        if ET.fromstring(pending._body).find("{*}Upload") is not None:
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
                capacity_plan(objects, manifests, heads, len(canonical(record)))
                self.request("PUT", key, canonical(record), {"If-None-Match": "*"})
            elif any(json.loads(raw).get(field) != record.get(field) for field in ("name", "repository", "identity")):
                raise OperationError(503, "r2_channel_provenance_mismatch")
            # A DB-uncertain committed promotion is reconciled by promote, not overwritten here.

    def promote(self, release, public_root, previous, record):
        key = "releases/channels/" + record["name"] + ".json"
        with self.locked():
            raw, etag = self.read(key)
            actual = json.loads(raw) if raw else None
            recovering = actual == record
            if not recovering and actual != previous:
                raise OperationError(409, "r2_channel_revision_conflict")
            manifest = release["manifest"]
            files = {}
            for package in manifest["packages"]:
                for item in [package["artifact"], package.get("distributionArchive"), package["feed"]]:
                    if item:
                        files[item["key"]] = Path(public_root) / item["key"]
            manifest_key = release["manifest_key"]
            files[manifest_key] = Path(public_root) / manifest_key
            if recovering:
                for file_key, file_path in files.items():
                    self.transport.verify_remote_artifact(
                        self.url(file_key), self.creds, self.transport.amz_timestamp(), file_path.stat().st_size,
                        self.transport.file_sha256(str(file_path)), algorithm="sha256", encoding="hex")
                return
            objects, manifests, heads = self.inventory()
            # Full allocation before multipart starts; old + new head bytes conservatively overlap.
            incoming = sum(path.stat().st_size for path in files.values()) + len(canonical(record))
            delete, _ = capacity_plan(objects, manifests, heads, incoming, manifest["request"]["buildId"])
            for old in delete:
                self.request("DELETE", old)
            for file_key, file_path in files.items():
                self.transport.put_object(self.creds, self.base, self.bucket, file_key, str(file_path),
                                          self.transport.amz_timestamp(), self.transport.content_type_for(file_key),
                                          {"If-None-Match": "*"})
                self.transport.verify_remote_artifact(
                    self.url(file_key), self.creds, self.transport.amz_timestamp(), file_path.stat().st_size,
                    self.transport.file_sha256(str(file_path)), algorithm="sha256", encoding="hex")
            self.request("PUT", key, canonical(record), {"If-Match": etag})
            readback, _ = self.read(key)
            if readback != canonical(record):
                raise OperationError(503, "r2_head_readback_failed")
            # Prune again after head transition; previous active build is no longer protected.
            objects, manifests, heads = self.inventory()
            delete, _ = capacity_plan(objects, manifests, heads)
            for old in delete:
                self.request("DELETE", old)
