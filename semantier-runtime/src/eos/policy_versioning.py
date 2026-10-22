from typing import Dict, Any
from datetime import datetime, timezone

# in-memory version store (replace with DB later)
POLICY_VERSIONS = {}


def set_policy(org_id: str, policy_type: str, policy: Dict[str, Any]):
    key = (org_id, policy_type)
    version_list = POLICY_VERSIONS.setdefault(key, [])

    version = len(version_list) + 1

    version_list.append({
        "version": version,
        "policy": policy,
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    return version


def get_policy(org_id: str, policy_type: str, version: int = None):
    key = (org_id, policy_type)
    versions = POLICY_VERSIONS.get(key, [])

    if not versions:
        return None

    if version is None:
        return versions[-1]

    for v in versions:
        if v["version"] == version:
            return v

    return None


def get_active_policies(org_id: str):
    market = get_policy(org_id, "market")
    org = get_policy(org_id, "org")

    return {
        "market": market,
        "org": org
    }


def record_policy_snapshot(org_id: str, event_id: str):
    policies = get_active_policies(org_id)

    return {
        "event_id": event_id,
        "org_id": org_id,
        "policy_snapshot": {
            "market_version": policies["market"]["version"] if policies["market"] else None,
            "org_version": policies["org"]["version"] if policies["org"] else None
        }
    }


def replay_with_policy(org_id: str, snapshot: Dict[str, Any]):
    market_v = snapshot["policy_snapshot"]["market_version"]
    org_v = snapshot["policy_snapshot"]["org_version"]

    market_policy = get_policy(org_id, "market", market_v)
    org_policy = get_policy(org_id, "org", org_v)

    return {
        "market": market_policy,
        "org": org_policy
    }
