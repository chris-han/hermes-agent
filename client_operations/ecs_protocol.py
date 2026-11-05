"""Protocol v1 for the fixed private channel1 storage boundary."""

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import json
import re
from uuid import uuid4

from .artifact_io import OperationError

PREFIX = "/v1/targets/channel1"
MAX_JSON = 256 * 1024
MAX_OBJECTS = 128
MAX_OBJECT_BYTES = 10_000_000_000
MAX_HEAD = 1024 * 1024


def fail(status, code, *, outcome="unknown"):
    error = OperationError(status, code)
    error.mutation_outcome = outcome
    raise error


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def decode(body, maximum=MAX_JSON):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        if len(body) > maximum:
            fail(413, "ecs_metadata_limit")
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
        if type(value) is not dict or canonical(value) != body:
            raise ValueError("canonical object required")
        return value
    except (ValueError, TypeError, UnicodeError) as error:
        raise OperationError(400, "ecs_metadata_invalid") from error


def fields(value, required, optional=()):
    if (
        type(value) is not dict
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        fail(400, "ecs_fields_invalid")
    return value


def integer(value, maximum=2**63 - 1, minimum=1):
    if type(value) is not int or not minimum <= value <= maximum:
        fail(400, "ecs_integer_invalid")
    return value


def hex_value(value, length=32):
    if (
        not isinstance(value, str)
        or re.fullmatch("[a-f0-9]{" + str(length) + "}", value) is None
    ):
        fail(400, "ecs_identifier_invalid")
    return value


@dataclass(frozen=True)
class Key:
    key_id: str
    principal: str
    incarnation: str
    secret: bytes = field(repr=False)

    def __post_init__(self):
        if (
            not re.fullmatch("[A-Za-z0-9._-]{1,64}", self.key_id)
            or not re.fullmatch("[A-Za-z0-9:._-]{1,128}", self.principal)
            or len(self.secret) != 32
        ):
            raise ValueError("Invalid installed storage credential")
        hex_value(self.incarnation)


def transcript(method, path, stamp, request_id, body_sha, metadata):
    return "\n".join(
        (method, path, stamp, request_id, body_sha, sha(metadata))
    ).encode()


def headers_for(
    key, method, path, body_sha, *, metadata=b"", now=None, request_id=None
):
    stamp = (now or datetime.now(UTC)).isoformat()
    request_id = request_id or uuid4().hex
    headers = {
        "x-semantier-key-id": key.key_id,
        "x-semantier-time": stamp,
        "x-semantier-request-id": request_id,
        "x-semantier-body-sha256": body_sha,
        "x-semantier-signature": hmac.new(
            key.secret,
            transcript(method, path, stamp, request_id, body_sha, metadata),
            hashlib.sha256,
        ).hexdigest(),
    }
    if metadata:
        headers["x-semantier-metadata"] = (
            base64.urlsafe_b64encode(metadata).decode().rstrip("=")
        )
    return headers


def metadata_bytes(headers):
    encoded = headers.get("x-semantier-metadata", "")
    try:
        if len(encoded) > 4096 or re.fullmatch("[A-Za-z0-9_-]*", encoded) is None:
            raise ValueError("metadata encoding")
        value = base64.urlsafe_b64decode(encoded + "=" * ((-len(encoded)) % 4))
        if base64.urlsafe_b64encode(value).decode().rstrip("=") != encoded:
            raise ValueError("metadata encoding")
        if value:
            decode(value)
        return value
    except ValueError as error:
        raise OperationError(400, "ecs_metadata_encoding_invalid") from error


def authenticate(keys, headers, method, path, body_sha, *, now=None):
    try:
        if (
            method not in {"GET", "PUT", "POST"}
            or not path.startswith(PREFIX + "/")
            or re.fullmatch("[A-Za-z0-9/_-]+", path) is None
        ):
            fail(400, "ecs_route_invalid")
        key = keys.get(headers.get("x-semantier-key-id"))
        if key is None:
            fail(401, "ecs_authentication_required")
        request_id = hex_value(headers.get("x-semantier-request-id"))
        expected = hex_value(headers.get("x-semantier-body-sha256"), 64)
        signature = hex_value(headers.get("x-semantier-signature"), 64)
        stamp = headers["x-semantier-time"]
        instant = datetime.fromisoformat(stamp)
        if (
            instant.utcoffset() is None
            or instant.utcoffset().total_seconds() != 0
            or abs(((now or datetime.now(UTC)) - instant).total_seconds()) > 60
        ):
            fail(401, "ecs_request_time_invalid")
        metadata = metadata_bytes(headers)
        actual = hmac.new(
            key.secret,
            transcript(method, path, stamp, request_id, expected, metadata),
            hashlib.sha256,
        ).hexdigest()
        if expected != body_sha or not hmac.compare_digest(signature, actual):
            fail(401, "ecs_signature_invalid")
        return key
    except (KeyError, ValueError, TypeError) as error:
        raise OperationError(401, "ecs_authentication_required") from error
