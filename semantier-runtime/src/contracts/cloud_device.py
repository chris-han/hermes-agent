"""Versioned Cloud/native connection wire contract; no credential authority in hints."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

CONTRACT = 'semantier-cloud-device-connection/v1'
REVISION = '1.0.1'
Opaque = Annotated[str, Field(pattern=r'^[A-Za-z0-9_.:-]{1,128}$')]
Proof = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{43,128}$')]


class Wire(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    contract_version: Literal['semantier-cloud-device-connection/v1'] = CONTRACT


class DeviceStart(Wire):
    client_label: Annotated[str, Field(min_length=1, max_length=80, pattern=r'^[^\x00-\x1f\x7f]+$')]
    runtime_ref: Opaque
    profile_ref: Opaque
    generation: Annotated[int, Field(ge=1, le=2147483647)]
    possession_challenge: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
    idempotency_key: Opaque


class DeviceProof(Wire):
    proof: Proof


class DeviceContinue(Wire):
    account_id: Opaque | None = None
    product_id: Opaque | None = None
    term: Literal['monthly', 'annual'] | None = None
    language: Literal['en', 'zh'] = 'en'
    order_id: Opaque | None = None


class DeviceApprove(Wire):
    account_id: Opaque
    decision: Literal['approve', 'deny']


class DeviceClaim(DeviceProof):
    claim_id: Opaque
    account_id: Opaque
    runtime_ref: Opaque
    profile_ref: Opaque
    generation: Annotated[int, Field(ge=1, le=2147483647)]


class DeviceAck(DeviceClaim):
    installed_generation: Annotated[int, Field(ge=1, le=2147483647)]


class DeviceEnvelope(Wire):
    intent_ref: Opaque
    status: Literal['pending', 'slow_down', 'approved', 'claimed', 'installed', 'denied', 'expired', 'cancelled', 'revoked']
    expires_at: str
    retry_after: int = 5
    data: dict = Field(default_factory=dict)


class ClientCredential(BaseModel):
    model_config = ConfigDict(extra='forbid')
    token_id: str
    secret: str
    expires_at: str


class ApprovedBinding(BaseModel):
    account_id: str
    currency: Literal['CNY', 'USD']
    runtime_ref: str
    profile_ref: str
    generation: int


class ClaimedConnection(ApprovedBinding):
    model_config = ConfigDict(extra='forbid')
    credential: ClientCredential
    organization_id: str
    inference_origin: str
    billing_origin: str
    catalog_path: Literal['/api/cloud-provider/catalog']
    readiness_path: Literal['/api/cloud-provider/device/readiness']
    claim_id: str
    installation_expires_at: str
    renewal: Literal['reauthorization_required']


class EmptyData(BaseModel):
    model_config = ConfigDict(extra='forbid')


class StartData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    verification_uri: str
    renewal: Literal['reauthorization_required']


class InstallationReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    claim_id: str
    installed_generation: int


class BrowserData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    client_label: str
    profile_ref: str
    generation: int
    continuation: dict
    account_id: str | None
    currency: str | None
    accounts: list[dict] = Field(default_factory=list)


class StartEnvelope(DeviceEnvelope):
    data: StartData


class BrowserEnvelope(DeviceEnvelope):
    data: BrowserData


class PollEnvelope(DeviceEnvelope):
    data: ApprovedBinding | EmptyData


class ClaimEnvelope(DeviceEnvelope):
    data: ClaimedConnection | InstallationReceipt | EmptyData


class AckEnvelope(DeviceEnvelope):
    data: InstallationReceipt | EmptyData


class RevokeResponse(Wire):
    status: Literal['revoked']
    token_id: str


class RenewResponse(Wire):
    status: Literal['reauthorization_required']
    start_path: Literal['/api/cloud-provider/device/start']


class ReadinessResponse(Wire):
    account_id: str
    currency: Literal['CNY', 'USD']
    payment_entitlement_allowance: dict
    account_provisioning: dict | None
    catalog: dict
    device: dict | None
    as_of: str
    renewal: Literal['reauthorization_required']
