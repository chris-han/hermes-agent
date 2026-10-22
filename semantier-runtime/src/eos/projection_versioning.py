from datetime import datetime, timezone

PROJECTION_VERSIONS = {}


def version_projection(org_id, ontology_version, context, policy):
    key = (org_id, ontology_version)
    versions = PROJECTION_VERSIONS.setdefault(key, [])

    v = len(versions) + 1

    record = {
        "version": v,
        "ontology_version": ontology_version,
        "context": context,
        "policy": policy,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    versions.append(record)

    return record


def get_projection(org_id, ontology_version, version=None):
    key = (org_id, ontology_version)
    versions = PROJECTION_VERSIONS.get(key, [])

    if not versions:
        return None

    if version is None:
        return versions[-1]

    for v in versions:
        if v["version"] == version:
            return v

    return None


def snapshot_projection(org_id, ontology_version, projection_record, event_id):
    return {
        "event_id": event_id,
        "org_id": org_id,
        "projection_snapshot": {
            "ontology_version": ontology_version,
            "projection_version": projection_record["version"]
        }
    }


def replay_projection(org_id, snapshot):
    ont_v = snapshot["projection_snapshot"]["ontology_version"]
    proj_v = snapshot["projection_snapshot"]["projection_version"]

    record = get_projection(org_id, ont_v, proj_v)

    if not record:
        return None

    return record["policy"]
