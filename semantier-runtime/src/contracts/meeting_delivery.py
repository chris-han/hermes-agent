"""Bounded meeting recipient authority; records are evidence, never self-issued grants."""

import hashlib
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

CONTRACT = "meeting-core-clarification.v1"
PURPOSE = "missing_normalized_slot"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("aware UTC timestamp required")
    return result.astimezone(timezone.utc)


def slot(value):
    if not isinstance(value, dict) or set(value) != {
        "start_time",
        "end_time",
        "timezone",
    }:
        raise ValueError("exact structured slot required")
    start, end = instant(value["start_time"]), instant(value["end_time"])
    ZoneInfo(value["timezone"])
    if end <= start:
        raise ValueError("positive slot duration required")
    return dict(
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        timezone=value["timezone"],
    )


def bot_generation(config):
    # No credential or reversible credential material enters a delivery association.
    return digest(
        {
            k: config.get(k)
            for k in (
                "app_id",
                "app_secret",
                "bot_open_id",
                "tenant_key",
                "updated_at",
                "verification_token",
                "encrypt_key",
            )
        }
    )
