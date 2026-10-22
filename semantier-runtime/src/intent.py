import hashlib
import json
from models import ExecutionIntent

# Build Intent from φ
def hash_justification(phi: dict) -> str:
    raw = json.dumps(phi, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def build_intent(phi: dict) -> ExecutionIntent:
    action = phi.get("action", {})
    return ExecutionIntent(
        action=action,
        justification_hash=hash_justification(phi)
    )