from datetime import datetime, timezone
import json
import hashlib

EVENT_STORE = []


def append_event(event):
    # compute a deterministic content hash for parity with sqlite store
    def _canonical_json(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def _content_hash(payload):
        canon = _canonical_json(payload)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    payload = {
        "org_id": event.get("org_id"),
        "ontology_version": event.get("ontology_version"),
        "projection_version": event.get("projection_version"),
        "policy": event.get("policy", {}),
        "justification": event.get("justification"),
        "execution": event.get("execution"),
        "effects": event.get("effects")
    }

    content_hash = _content_hash(payload)

    # prev_event_hash from last stored event for same org
    prev_hash = None
    for e in reversed(EVENT_STORE):
        if e.get("org_id") == event.get("org_id") and e.get("content_hash"):
            prev_hash = e.get("content_hash")
            break

    event_record = {
        "event_id": event.get("event_id"),
        "org_id": event.get("org_id"),
        "ontology_version": event.get("ontology_version"),
        "projection_version": event.get("projection_version"),
        "policy": event.get("policy"),
        "justification": event.get("justification"),
        "execution": event.get("execution"),
        "effects": event.get("effects"),
        "content_hash": content_hash,
        "prev_event_hash": event.get("prev_event_hash") or prev_hash,
        "signature": event.get("signature"),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    EVENT_STORE.append(event_record)

    return event_record


def get_event(event_id):
    for e in EVENT_STORE:
        if e["event_id"] == event_id:
            return e
    return None


def replay_event(event_id):
    event = get_event(event_id)
    if not event:
        return None

    return {
        "ontology_version": event["ontology_version"],
        "projection_version": event["projection_version"],
        "policy": event["policy"],
        "justification": event["justification"],
        "execution": event["execution"],
        "effects": event["effects"]
    }
