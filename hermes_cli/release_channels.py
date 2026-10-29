"""R2 channel wire protocol. Names are data; no local channel registry exists."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_SEQUENCE = 2**32 - 1
MAX_METADATA = 4 * 1024 * 1024

_POLICIES = ("preview", "stable-release", "canary-release", "source-branch")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


class ChannelError(ValueError):
    """Invalid, unavailable, or untrusted channel metadata (never a fallback)."""


class ChannelNotFound(ChannelError):
    pass


def _match(pattern: str, value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value, re.ASCII) is None:
        raise ChannelError(f"Invalid {label}")
    return value


def validate_name(value: object) -> str:
    name = _match(r"[a-z0-9]+(?:-[a-z0-9]+)*", value, "channel name")
    if len(name) > 32 or name in _RESERVED:
        raise ChannelError("Invalid channel name")
    return name


def validate_repository(value: object) -> str:
    return _match(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value, "repository")


def require_sha256(value: object) -> str:
    return _match(r"[a-f0-9]{64}", value, "SHA256")


def require_commit(value: object) -> str:
    return _match(r"[a-f0-9]{40}", value, "commit")


def build_prefix(build_id: object) -> str:
    return "releases/channel-builds/" + _match(r"[a-f0-9]{32}", build_id, "build ID") + "/"


def channel_key(name: object) -> str:
    return f"releases/channels/{validate_name(name)}.json"


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def decode_json(body: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ChannelError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(body, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise ChannelError("Invalid channel JSON") from exc
    if not isinstance(value, dict):
        raise ChannelError("Channel metadata must be an object")
    return value


def public_base(value: str) -> str:
    parsed = urlsplit(value)
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c in value for c in "%\\;")
            or any(ord(c) <= 32 for c in value)
            or any(p in (".", "..") for p in parsed.path.split("/"))
            or not (parsed.scheme == "https" or (parsed.scheme == "http"
                    and parsed.hostname in ("127.0.0.1", "localhost", "::1")))):
        raise ChannelError("Invalid channel archive authority")
    return value.rstrip("/")


def artifact_key(value: object) -> str:
    key = _match(r"[A-Za-z0-9_./+-]+", value, "artifact key")
    if not key.startswith("releases/") or any(p in ("", ".", "..") for p in key.split("/")):
        raise ChannelError("Invalid artifact key")
    if any(p.split(".")[0].lower() in _RESERVED or p.endswith(".") for p in key.split("/")):
        raise ChannelError("Invalid artifact key")
    return key


def package_versions(sequence: int) -> tuple[str, str]:
    _integer(sequence, "sequence", 1, MAX_SEQUENCE)
    return f"0.0.{sequence}", f"0.{sequence // 65536}.{sequence % 65536}.0"


def _integer(value: object, label: str, minimum: int = 1, maximum: int = MAX_SEQUENCE + 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ChannelError(f"Invalid {label}")
    return value


def _schema(value: object) -> dict:
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
        raise ChannelError("Unsupported channel schema")
    return value


def validate_identity(value: object) -> dict:
    if not isinstance(value, dict):
        raise ChannelError("Invalid channel identity")
    _match(r"[a-f0-9]{16}", value.get("token"), "identity token")
    _match(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}", value.get("displayName"), "identity display name")
    for field in ("appNamePascal", "artifactNamePascal"):
        _match(r"[A-Za-z][A-Za-z0-9]{0,63}", value.get(field), field)
    _match(r"[a-z][a-z0-9.-]{2,127}", value.get("appId"), "app ID")
    _match(r"[A-Za-z][A-Za-z0-9.-]{2,49}", value.get("msixAppIdWithOrg"), "MSIX identity")
    _match(r"[a-z][a-z0-9-]{0,63}", value.get("cliName"), "CLI name")
    _match(r"[A-Za-z][A-Za-z0-9 ._-]{0,79}", value.get("windowsExecutableName"), "Windows executable")
    return value


def validate_request(value: object, *, repository: str | None = None,
                     base_url: str | None = None, policy: str = "preview") -> dict:
    request = _schema(value)
    build_prefix(request.get("buildId"))
    validate_name(request.get("channel"))
    actual_repo = validate_repository(request.get("repository"))
    if repository is not None and actual_repo.casefold() != repository.casefold():
        raise ChannelError("Channel repository authority mismatch")
    require_commit(request.get("commit"))
    if "controllerCommit" in request:
        require_commit(request["controllerCommit"])
    _match(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", request.get("sourceVersion"), "source version")
    versions = package_versions(request.get("sequence"))
    if policy == "preview":
        if (request.get("version"), request.get("windowsVersion")) != versions:
            raise ChannelError("Channel package version does not match sequence")
    else:
        _match(
            r"v[0-9]+\.[0-9]+\.[0-9]+(?:\+canary\.20[0-9]{6}T[0-9]{6}Z)?",
            request.get("releaseTag"),
            "release tag",
        )
        if request.get("version") != request["releaseTag"][1:]:
            raise ChannelError("Release package version mismatch")
        archive_ref = request.get("archiveRef")
        if archive_ref is not None:
            from scripts.releases.versioning import parse_attempt_ref
            parsed = parse_attempt_ref(archive_ref) if isinstance(archive_ref, str) else None
            if parsed is None:
                if archive_ref != request["releaseTag"]:
                    raise ChannelError("Invalid archive ref")
            elif parsed[0] != request["version"]:
                raise ChannelError("Archive ref does not name the release version")
        # Stable uses the Store quad with revision zero; canary uses its UTC
        # yy.mmdd.hh.mmss package version.
        pattern = r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" if policy == "canary-release" else r"[0-9]+\.[0-9]+\.[0-9]+\.0"
        quad = _match(pattern, request.get("windowsVersion"), "Windows version")
        if any(int(part) > 65535 for part in quad.split(".")):
            raise ChannelError("Invalid Windows version")
    validate_identity(request.get("identity"))
    values = request.get("bundleEnv")
    if not isinstance(values, dict):
        raise ChannelError("Invalid bundle environment")
    for key, item in values.items():
        _match(r"[A-Za-z_][A-Za-z0-9_]*", key, "bundle environment name")
        if item is not None and (not isinstance(item, str) or "\0" in item):
            raise ChannelError("Invalid bundle environment value")
    base = request.get("publicBase")
    if not isinstance(base, str) or public_base(base) != base:
        raise ChannelError("Invalid request archive authority")
    if base_url is not None and base != public_base(base_url):
        raise ChannelError("Request archive authority mismatch")
    return request


def _head(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ChannelError("Invalid channel head")
    prefix = build_prefix(value.get("buildId"))
    _integer(value.get("sequence"), "head sequence", maximum=MAX_SEQUENCE)
    if value.get("manifestKey") != prefix + "build.json":
        raise ChannelError("Invalid channel manifest key")
    require_sha256(value.get("sha256"))
    return value


def validate_record(value: object, *, name: str | None = None, repository: str | None = None) -> dict:
    record = _schema(value)
    if not {"name", "repository", "policy", "state", "revision", "nextSequence", "identity", "head"} <= record.keys():
        raise ChannelError("Missing channel record fields")
    validate_name(record.get("name"))
    if name is not None and record["name"] != name:
        raise ChannelError("Channel name does not match object key")
    actual_repo = validate_repository(record.get("repository"))
    if repository is not None and actual_repo.casefold() != repository.casefold():
        raise ChannelError("Channel repository authority mismatch")
    if record.get("policy") not in _POLICIES or record.get("state") not in ("active", "retired"):
        raise ChannelError("Invalid channel policy or state")
    _integer(record.get("revision"), "channel revision", maximum=2**53 - 1)
    _integer(record.get("nextSequence"), "next sequence")
    head = _head(record.get("head"))
    if head is not None and head["sequence"] >= record["nextSequence"]:
        raise ChannelError("Channel head exceeds allocation high-water")
    native_head = _head(record.get("nativeHead"))
    if native_head is not None:
        if (record["state"] != "active" or record["policy"] == "source-branch" or head is None
                or not head["sequence"] < native_head["sequence"] < record["nextSequence"]):
            raise ChannelError("Native head requires a legacy bridge and newer allocated sequence")
    if record["policy"] == "source-branch":
        delivery = record.get("delivery")
        if record.get("identity") is not None or head is not None or not isinstance(delivery, dict) or delivery.get("kind") != "source-branch":
            raise ChannelError("Source branch cannot carry native artifacts")
        branch = _match(r"[A-Za-z0-9][A-Za-z0-9_./-]*", delivery.get("branch"), "source branch")
        if ".." in branch or "//" in branch or branch.endswith(("/", ".", ".lock")):
            raise ChannelError("Invalid source branch")
    else:
        validate_identity(record.get("identity"))
    if record["state"] == "retired":
        validate_name(record.get("destination"))
        _match(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", record.get("minimumVersion"), "retirement minimum version")
        if _head(record.get("destinationHead")) is None:
            raise ChannelError("Retirement requires a pinned destination head")
        if type(record.get("receiverProtocol")) is not int or record["receiverProtocol"] != 1:
            raise ChannelError("Unsupported retirement receiver protocol")
        receiver = record.get("receiver")
        if not isinstance(receiver, dict) or receiver.get("kind") not in ("in-place", "discontinued"):
            raise ChannelError("Invalid retirement receiver kind")
        if _head(record.get("lastHead")) != head:
            raise ChannelError("Retirement last head mismatch")
    return record


@dataclass(frozen=True)
class ChannelResolution:
    requested: dict
    terminal: dict
    manifest: dict | None



class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ChannelError("Channel archive redirects are not permitted")


_RETRY_READS: ContextVar[bool] = ContextVar("release_channel_read_retries", default=False)


@contextmanager
def retrying_reads():
    """Let channel reads inside the block wait out a CDN blip with PM's bounded retries.

    Only an explicit ``hermes update`` opts in. Passive checks (``hermes --version``, the
    banner, the Desktop/dashboard update check) make one attempt: offline also reads as
    transient (DNS EAI_AGAIN, ENETUNREACH), so retrying there adds backoff to a synchronous
    status line, stretches a hung CDN from 30 s to ~2 min and logs a WARNING per retry.
    """
    token = _RETRY_READS.set(True)
    try:
        yield
    finally:
        _RETRY_READS.reset(token)


class ChannelReader:
    def __init__(self, base_url: str, repository: str | None = None, opener=None):
        self.base_url = public_base(base_url)
        self.repository = validate_repository(repository) if repository is not None else None
        self.opener = opener or build_opener(_NoRedirect()).open

    def read_bytes(self, key: str, sha256: str | None = None) -> bytes:
        url = self.base_url + "/" + artifact_key(key)
        if sha256 is not None:
            require_sha256(sha256)

        # Channel reads are idempotent and precede the PM download path, so an explicit update
        # uses the same bounded transient-failure policy instead of a second retry classifier.
        from pm.network import edge_block, retry_network

        def read() -> bytes:
            with self.opener(Request(url, headers={"Cache-Control": "no-cache"}), timeout=30) as response:
                if response.geturl() != url:
                    raise ChannelError("Channel archive redirects are not permitted")
                return response.read(MAX_METADATA + 1)

        try:
            body = retry_network(read) if _RETRY_READS.get() else read()
        except HTTPError as exc:
            if exc.code == 404:
                raise ChannelNotFound(f"Channel object not found: {key}") from exc
            block = edge_block(exc)
            raise ChannelError(f"Channel read unavailable: HTTP {exc.code}" + (f"; {block}" if block else "")) from exc
        except (OSError, URLError) as exc:
            raise ChannelError("Channel read unavailable") from exc
        if len(body) > MAX_METADATA:
            raise ChannelError("Channel metadata exceeds size limit")
        if sha256 is not None and hashlib.sha256(body).hexdigest() != sha256:
            raise ChannelError("Channel metadata SHA256 mismatch")
        return body

    def read_record(self, name: str) -> dict:
        return validate_record(decode_json(self.read_bytes(channel_key(name))), name=name, repository=self.repository)

    def resolve(self, name: str) -> ChannelResolution:
        requested = record = self.read_record(name)

        if record["state"] == "retired":
            if record["destination"] == name:
                raise ChannelError("Channel retirement cycle")
            record = self.read_record(record["destination"])
            if record["repository"].casefold() != requested["repository"].casefold():
                raise ChannelError("Retirement repository authority mismatch")
            if record["state"] != "active" or record["policy"] != "stable-release":
                raise ChannelError("Retirement requires a directly active stable-release destination")
        head = record["head"]
        if requested["state"] == "retired":
            # The first receiver stays pinned even when stable advances offline.
            head = requested["destinationHead"]
            if head is None or record["head"] is None or head["sequence"] > record["head"]["sequence"]:
                raise ChannelError("Invalid retirement destination head")
        manifest = None if head is None else validate_manifest(
            decode_json(self.read_bytes(head["manifestKey"], head["sha256"])), {**record, "head": head}, self.base_url)
        if requested["state"] == "retired":
            assert manifest is not None
            if manifest.get("receiverProtocol") != requested["receiverProtocol"]:
                raise ChannelError("Stable build has no supported retirement receiver")
            actual = tuple(map(int, manifest["request"]["sourceVersion"].split(".")))
            floor = tuple(map(int, requested["minimumVersion"].split(".")))
            if actual < floor:
                raise ChannelError("Destination does not meet retirement minimum version")
        return ChannelResolution(requested, record, manifest)


def _manifest_schema(value: object, allow_native: bool) -> dict:
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] not in (1, 2):
        raise ChannelError("Unsupported manifest schema")
    manifest = value
    native_manifest = manifest["schema"] == 2
    if native_manifest and not allow_native:
        raise ChannelError("Native manifest requires explicit receiver/admission capability")
    if "receiverProtocol" in manifest:
        _integer(manifest["receiverProtocol"], "receiver protocol", maximum=2**53 - 1)
    return manifest


def validate_manifest(value: object, record: dict, base_url: str, *, allow_native: bool = False) -> dict:
    manifest = _manifest_schema(value, allow_native)
    native_manifest = manifest["schema"] == 2
    request = validate_request(manifest.get("request"), repository=record["repository"],
                               base_url=base_url, policy=record["policy"])
    if request["identity"] != record["identity"]:
        raise ChannelError("Manifest channel identity mismatch")
    if request["channel"] != record["name"]:
        raise ChannelError("Manifest channel mismatch")
    head = record.get("head")
    if head is not None and any(request[field] != head[field] for field in ("buildId", "sequence")):
        raise ChannelError("Manifest does not match channel head")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ChannelError("Build manifest has no packages")
    prefixes = [build_prefix(request["buildId"])]
    if record["policy"] in {"stable-release", "canary-release"}:
        # The archive prefix is the attempt ref when the request names one; a
        # bare releaseTag fallback never holds attempt artifacts (fail closed).
        prefixes.append(f"releases/tag/{request.get('archiveRef') or request['releaseTag']}/")
    if native_manifest and len(packages) != 1:
        raise ChannelError("Native manifest requires one Windows x64 target")
    seen = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ChannelError("Invalid native package")
        target = tuple(package.get(field) for field in ("platform", "arch", "variant"))
        if target[0] not in ("darwin", "win32") or target[1] not in ("arm64", "x64") or target[2] != ("native-msix-assets" if native_manifest else "bundled"):
            raise ChannelError("Unsupported channel package target")
        if target[:2] in seen:
            raise ChannelError("Duplicate package target")
        seen.add(target[:2])
        artifact = package.get("artifact")
        if not isinstance(artifact, dict):
            raise ChannelError("Invalid package artifact")
        key = artifact_key(artifact.get("key"))
        if not any(key.startswith(prefix) for prefix in prefixes):
            raise ChannelError("Artifact outside admitted build namespace")
        require_sha256(artifact.get("sha256"))
        _integer(artifact.get("size"), "artifact size", maximum=2**53 - 1)
        identity_field, version_field, signing_field = (
            ("appId", "version", "teamId") if target[0] == "darwin"
            else ("msixAppIdWithOrg", "windowsVersion", "publisher"))
        if package.get("identity") != request["identity"][identity_field] or package.get("version") != request[version_field]:
            raise ChannelError("Native package identity or version mismatch")
        signing = package.get(signing_field)
        if not isinstance(signing, str) or not signing or any(ord(c) < 32 for c in signing):
            raise ChannelError("Missing native signing identity")
        if target[0] == "darwin":
            _match(r"[A-Z0-9]{10}", signing, "Apple team ID")
        feed = package.get("feed")
        if not isinstance(feed, dict) or feed.get("channel") != "stable":
            raise ChannelError("Channel native metadata requires a neutral stable feed")
        feed_key = artifact_key(feed.get("key"))
        if not any(feed_key.startswith(prefix) for prefix in prefixes):
            raise ChannelError("Native metadata outside admitted build namespace")
        if native_manifest:
            _validate_native_package(package, request)
        elif "native" in package:
            raise ChannelError("Native package requires manifest schema2")
    return manifest


def _validate_native_package(package: dict, request: dict) -> None:
    if package.get("platform") != "win32" or package.get("arch") != "x64":
        raise ChannelError("Native asset receiver requires Windows x64")
    native = _schema(package.get("native"))
    if (native.get("api") != "PackageManager.AddPackageByUriAsync"
            or native.get("digestMethod") != "IAppxDigestProvider.GetDigest"
            or native.get("digestEncoding") != "base64"):
        raise ChannelError("Unsupported native API/digest contract")
    graph = native.get("graph")
    if not isinstance(graph, dict):
        raise ChannelError("Missing native graph descriptor")
    key = artifact_key(graph.get("key"))
    if key != package["feed"]["key"] or not key.startswith(build_prefix(request["buildId"])) or not key.endswith("/native-graph.json"):
        raise ChannelError("Native graph descriptor binding mismatch")
    require_sha256(graph.get("sha256"))
    _integer(graph.get("size"), "graph descriptor size", maximum=MAX_METADATA)
    _integer(package["artifact"].get("size"), "native root size", maximum=480_000_000)
    if "chunks" in package["artifact"] or not package["artifact"]["key"].startswith(build_prefix(request["buildId"])):
        raise ChannelError("Native root cannot use legacy chunks or another build")


def _native_object_identity(row,kind,identity,publisher,version,arch,identities):
    key=row['key']
    extension='.msixbundle' if kind.endswith('bundle') else '.msix'
    if not key.endswith(extension):raise ChannelError('Native object extension mismatch')
    name=row.get('identity')
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9.-]{3,50}',name):raise ChannelError('Invalid native identity')
    resource=row.get('resourceId','')
    if kind=='asset-package':
        if name!=identity or not isinstance(resource,str) or not re.fullmatch('runtime[0-9]{4}',resource) or row.get('allowExecution') is not True:
            raise ChannelError('Invalid executable asset package binding')
    elif resource:raise ChannelError('Unexpected native resource identity')
    if (name,kind,resource) in identities:raise ChannelError('Ambiguous native package identity')
    identities.add((name,kind,resource))
    if kind.startswith('main-') and name!=identity:raise ChannelError('Main native identity mismatch')
    if kind.startswith('optional-') and (name==identity or not name.startswith(identity+'.')):
        raise ChannelError('Related native identity mismatch')
    if any(row.get(k)!=v for k,v in dict(publisher=publisher,version=version,arch='neutral' if kind.endswith('bundle') or kind=='asset-package' else arch).items()):
        raise ChannelError('Native identity/version/architecture mismatch')


def validate_native_graph(value, *, build_id, identity, publisher, version, arch):
    """Producer-side structural gate; never proof of signing or native execution.

    Remote URIs are derived from qualified base + immutable keys, never supplied
    independently. Every referenced outer/inner object has its own bytes/digest
    binding. Keep this gate separate from admission until real layout proof.
    """
    import base64
    if not isinstance(value, dict) or type(value.get('schema')) is not int or value['schema'] != 1 or value.get('transport') != 'windows-package-uri':
        raise ChannelError('Unsupported native graph')
    require_sha256(value.get('payloadInventorySha256'))
    prefix=build_prefix(build_id)
    objects=value.get('objects')
    if not isinstance(objects,list) or not objects or len(objects)>64:
        raise ChannelError('Invalid native object set')
    rows={}
    identities=set()
    for row in objects:
        if not isinstance(row,dict):raise ChannelError('Invalid native object')
        key=artifact_key(row.get('key'))
        if not key.startswith(prefix) or key in rows:raise ChannelError('Native object build/uniqueness mismatch')
        kind=row.get('kind')
        if kind not in ('main-package','optional-package','asset-package','main-bundle','optional-bundle'):
            raise ChannelError('Invalid native object kind')
        _native_object_identity(row,kind,identity,publisher,version,arch,identities)
        if type(row.get('size')) is not int or not 0<row['size']<=480_000_000:
            raise ChannelError('Native object exceeds cacheable cap')
        require_sha256(row.get('sha256'))
        digest=row.get('nativeDigest')
        try:
            decoded=base64.b64decode(digest,validate=True)
            if len(decoded)!=32 or base64.b64encode(decoded).decode()!=digest:raise ChannelError('Invalid native digest')
        except (ValueError,TypeError) as exc:raise ChannelError('Invalid native digest') from exc
        refs=row.get('references')
        if not isinstance(refs,list) or any(not isinstance(k,str) for k in refs) or len(set(refs))!=len(refs):
            raise ChannelError('Invalid native references')
        if kind.endswith('package') and refs:raise ChannelError('Package cannot introduce unbound dependencies')
        rows[key]=row
    root=value.get('root')
    if not isinstance(root,str) or root not in rows or rows[root]['kind'] not in ('main-package','main-bundle'):
        raise ChannelError('Invalid native graph root')
    if sum(r['kind']=='main-package' for r in objects)!=1 or sum(r['kind']=='main-bundle' for r in objects)>1:
        raise ChannelError('Ambiguous native main package')
    seen=set(); active=set()
    def visit(key):
        if key not in rows:raise ChannelError('Missing native object')
        if key in active:raise ChannelError('Native graph cycle')
        if key in seen:return
        active.add(key)
        for reference in rows[key]['references']:visit(reference)
        active.remove(key);seen.add(key)
    visit(root)
    if seen!=set(rows):raise ChannelError('Unreachable native graph object')
    return value
