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
