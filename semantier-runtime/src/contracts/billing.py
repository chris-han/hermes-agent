"""Currency-safe, deterministic billing values; identity remains outside billing."""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


class BillingError(ValueError):
    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        super().__init__(detail or code)


def decimal_amount(value: str | Decimal | int) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError('money must be a decimal string')
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError('invalid decimal amount') from None
    if not result.is_finite() or result < 0 or result.as_tuple().exponent < -12 or result >= Decimal('1e26'):
        raise ValueError('amount outside NUMERIC(38,12)')
    return result


def currency_code(value: str) -> str:
    if value not in ('USD', 'CNY'):
        raise ValueError('unsupported currency')
    return value


@dataclass(frozen=True)
class NormalizedUsage:
    provider: str
    input_total: int | None
    output_total: int | None
    uncached_input: int | None
    cache_read_input: int | None
    cache_write_input: int | None
    reasoning_subset: int | None
    completeness: str
    missing_fields: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()
    normalizer_version: str = 'native-usage.v1'
    inclusion_rules_version: str = 'provider-inclusion.v1'

    @property
    def total_tokens(self) -> int | None:
        if self.input_total is None or self.output_total is None:
            return None
        return self.input_total + self.output_total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {'missing_fields': list(self.missing_fields), 'unsupported_fields': list(self.unsupported_fields), 'total_tokens': self.total_tokens}
