"""Semantier currency-safe billing values, independent of the legacy Nous USD contract."""
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, field_validator

from .base import JsonValue, Result
from .common import ProfileParams


class BillingReadParams(ProfileParams):
    schema_version: str | None = None


class Money(Result):
    amount: str = Field(pattern=r'^-?\d+(\.\d+)?$')
    currency: str = Field(pattern=r'^[A-Z]{3}$')


class BillingOrganization(Result):
    id: str
    display_name: str | None = None


class BillingCapabilities(Result):
    can_read_usage: bool
    can_charge: Literal[False]
    can_change_plan: Literal[False]
    can_auto_top_up: Literal[False] = False


class BillingUsage(Result):
    known_cost: Money | None = None
    status: str
    period: str | None = None
    unresolved_attempts: int | None = Field(default=None, ge=0)


class SemantierBillingState(Result):
    schema_version: Literal['semantier.billing.v1']
    backend: Literal['semantier']
    organization: BillingOrganization
    billing_account_id: str
    mode: str
    funding_source: str
    balances: list[Money]
    capabilities: BillingCapabilities
    usage: BillingUsage
    as_of: str = Field(pattern=r"^.+(?:Z|\+00:00)$")
    account: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator('as_of')
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        stamp = datetime.fromisoformat(value)
        if stamp.utcoffset() != timedelta(0):
            raise ValueError('Billing timestamp must be timezone-aware UTC.')
        return value


class SemantierSubscriptionState(Result):
    schema_version: Literal['semantier.billing.v1']
    backend: Literal['semantier']
    billing_account_id: str
    status: Literal['unavailable']
    capabilities: dict[str, bool]
    reason: str


class BillingCommerceCapabilities(BillingCapabilities):
    can_manage_billing: bool
    can_open_portal: bool


class BillingSubscription(Result):
    plan: str
    term: Literal['annual', 'monthly']
    status: Literal['active', 'expired']
    starts_at: str
    ends_at: str
    manual_renewal: Literal[True]

    _utc_period = field_validator('starts_at', 'ends_at')(SemantierBillingState.utc_timestamp.__func__)


class BillingCredits(Result):
    allowance: Money
    purchased: Money
    held: Money
    available: Money


class BillingCommerceUsage(Result):
    known_cost: Money
    period: Literal['lifetime']


class BillingPortal(Result):
    url: str | None

    @field_validator('url')
    @classmethod
    def trusted_origin(cls, value: str | None) -> str | None:
        from urllib.parse import urlsplit
        if value is None:
            return None
        parsed = urlsplit(value)
        if (parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {'', '/'} or not parsed.hostname
                or (parsed.scheme != 'https' and not (
                    parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}))):
            raise ValueError('Billing portal must be an HTTPS origin (HTTP loopback allowed).')
        return value


class BillingMerchant(Result):
    id: str
    display_name: str


class SemantierBillingV2(Result):
    schema_version: Literal['semantier.billing.v2']
    backend: Literal['semantier']
    organization: BillingOrganization
    billing_account_id: str
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    merchant: BillingMerchant | None
    environment: Literal['DEVELOPMENT', 'UNAVAILABLE']
    subscription: BillingSubscription | None
    credits: BillingCredits
    usage: BillingCommerceUsage
    capabilities: BillingCommerceCapabilities
    portal: BillingPortal
    as_of: str = Field(pattern=r'^.+(?:Z|\+00:00)$')

    _utc_timestamp = field_validator('as_of')(SemantierBillingState.utc_timestamp.__func__)
