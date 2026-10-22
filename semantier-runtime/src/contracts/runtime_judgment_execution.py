"""Content-addressed request, compatibility and response records for judgment."""
from typing import ClassVar, Literal
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import content_hash

_SHA = r'^sha256:[0-9a-f]{64}$'


class _Artifact(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    PREFIX: ClassVar[str]
    artifact_ref: str
    artifact_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    run_id: str

    @model_validator(mode='after')
    def identity(self):
        digest = content_hash(self.model_dump(mode='json', exclude={'artifact_ref','artifact_hash'}))
        if (self.artifact_ref,self.artifact_hash) != (self.PREFIX+digest,'sha256:'+digest):
            raise ValueError('runtime judgment artifact hash mismatch')
        return self

    @classmethod
    def create(cls, **payload):
        draft=cls.model_construct(artifact_ref='',artifact_hash='sha256:'+'0'*64,**payload)
        body=draft.model_dump(mode='json',exclude={'artifact_ref','artifact_hash'})
        digest=content_hash(body)
        return cls.model_validate(dict(**body,artifact_ref=cls.PREFIX+digest,artifact_hash='sha256:'+digest))


class RuntimeJudgmentRequestV1(_Artifact):
    PREFIX='runtime-judgment-request:'
    schema_version: Literal['runtime_judgment_request.v1']='runtime_judgment_request.v1'
    request_id: str = Field(min_length=1)
    judgment_request: dict
    execution_profile: dict


class RuntimeJudgmentAttemptV1(_Artifact):
    PREFIX='runtime-judgment-attempt:'
    schema_version: Literal['runtime_judgment_attempt.v1']='runtime_judgment_attempt.v1'
    execution_ref: str
    execution_hash: str = Field(pattern=_SHA)
    predecessor_attempt_ref: str | None = None
    predecessor_attempt_hash: str | None = Field(default=None,pattern=_SHA)
    compatibility: dict

    @model_validator(mode='after')
    def predecessor_pair(self):
        if (self.predecessor_attempt_ref is None) != (self.predecessor_attempt_hash is None):
            raise ValueError('judgment predecessor pins must be paired')
        return self


class RuntimeJudgmentResponseV1(_Artifact):
    PREFIX='runtime-judgment-response:'
    schema_version: Literal['runtime_judgment_response.v1']='runtime_judgment_response.v1'
    execution_ref: str
    execution_hash: str = Field(pattern=_SHA)
    predecessor_attempt_ref: str
    predecessor_attempt_hash: str = Field(pattern=_SHA)
    result: dict


PROFILE_FIELDS = frozenset({
    'profile_ref', 'provider', 'model', 'model_revision', 'tokenizer_revision',
    'template_revision', 'base_url', 'effective_context_limit', 'input_limit',
    'output_limit', 'required_output_reserve',
})


def profile_resolved(profile: dict) -> bool:
    names = ('profile_ref', 'provider', 'model', 'model_revision',
             'tokenizer_revision', 'template_revision', 'base_url')
    limits = ('effective_context_limit', 'input_limit', 'output_limit', 'required_output_reserve')
    return (set(profile) == PROFILE_FIELDS
            and all(isinstance(profile.get(key), str) and profile[key].strip() for key in names)
            and all(type(profile.get(key)) is int and profile[key] > 0 for key in limits))

def messages_texts(body):
    result = []
    for content in [body.get('system', ''), *(m.get('content', '') for m in body.get('messages', []))]:
        if isinstance(content, str) and content:
            result.append(content)
        elif isinstance(content, list):
            result.extend(block['text'] for block in content
                          if isinstance(block, dict) and block.get('type') == 'text'
                          and isinstance(block.get('text'), str))
    return Counter(result)



def verify_complete_request_compatibility(*, execution_profile, request, receipt):
    import json
    from hashlib import sha256
    from datetime import datetime, timedelta

    if receipt.get("schema_version") != "complete_request_compatibility.v1" or receipt.get("outcome") not in {"COMPATIBLE", "NO_COMPATIBLE_MODEL"}:
        raise ValueError("judgment compatibility shape mismatch")
    checked = datetime.fromisoformat(receipt["checked_at"].replace("Z", "+00:00"))
    if checked.utcoffset() != timedelta(0):
        raise ValueError("judgment compatibility timestamp must be UTC")
    if "request_body" in receipt and "sha256:" + sha256(receipt["request_body"].encode()).hexdigest() != receipt.get("request_hash"):
        raise ValueError("judgment serialized request hash mismatch")
    if receipt["outcome"] != "COMPATIBLE":
        if not receipt.get("reason"):
            raise ValueError("incompatible judgment requires a reason")
        return
    profile = receipt["profile"]
    authorized = [{key: value for key, value in item.items() if key != "api_key_env"}
                  for item in execution_profile.get("profiles", []) if isinstance(item, dict)]
    if not profile_resolved(profile) or profile not in authorized:
        raise ValueError("judgment profile is not authorized by its request")
    body = json.loads(receipt["request_body"])
    tokens = receipt.get("input_tokens")
    reserve = profile["required_output_reserve"]
    if (type(tokens) is not int or tokens < 0 or tokens > profile["input_limit"]
        or tokens + reserve > profile["effective_context_limit"] or reserve > profile["output_limit"]
        or body.get("max_tokens") != reserve or body.get("model") != profile["model"]):
        raise ValueError("judgment compatibility limits or model mismatch")
    endpoint = profile["base_url"].rstrip("/") + "/v1/messages"
    counting_basis = receipt.get("counting_basis")
    if counting_basis == "provider_count_api":
        count_body = receipt.get("count_response_body", "")
        if (not count_body
            or "sha256:" + sha256(count_body.encode()).hexdigest() != receipt.get("count_response_hash")
            or json.loads(count_body) != receipt.get("count_response")):
            raise ValueError("judgment count response hash mismatch")
        if (receipt.get("request_url") != endpoint
            or receipt.get("count_endpoint") != endpoint + "/count_tokens"
            or receipt.get("count_response", {}).get("input_tokens") != tokens):
            raise ValueError("judgment counting basis mismatch")
    elif counting_basis == "local_exact_tokenizer":
        if receipt.get("counted_request_hash") != receipt.get("request_hash"):
            raise ValueError("judgment counting basis mismatch")
        if any(receipt.get(field) is not None for field in (
            "count_endpoint", "count_response", "count_response_body", "count_response_hash"
        )):
            raise ValueError("judgment local count evidence mismatch")
    else:
        raise ValueError("judgment counting basis mismatch")
    texts = messages_texts(body)
    if (request["input_text"] not in texts or not any(request["instructions"] in text for text in texts)
        or body.get("output_config", {}).get("format", {}).get("schema") != request["json_schema"]):
        raise ValueError("judgment adapter request does not bind its projection and schema")
