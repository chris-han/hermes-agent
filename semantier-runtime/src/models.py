from pydantic import BaseModel
from typing import List, Dict, Any
import hashlib
import json

class ExecutionIntent(BaseModel):
    action: Dict[str, Any]
    justification_hash: str

    def idempotency_key(self) -> str:
        payload = {
            "action": self.action,
            "justification_hash": self.justification_hash
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
    
class Claim(BaseModel):
    type: str
    value: Any

class Evidence(BaseModel):
    ref: str
    hash: str

class Justification(BaseModel):
    claims: List[Claim]
    evidence: List[Evidence]
    semantic_type: str

class Action(BaseModel):
    name: str
    payload: Dict[str, Any]