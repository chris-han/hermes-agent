"""Pure native-schema normalization and artifact-pinned Decimal rating.

These four explicit schema families are authored protocol contracts, not a live
provider qualification or an assertion that every native usage field is supported.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from types import MappingProxyType
from typing import Mapping

from contracts.billing import NormalizedUsage, currency_code, decimal_amount

MILLION = Decimal(1000000)


def _counter(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('counter must be a non-negative integer')
    return value


def normalize_usage(provider: str, raw: dict) -> NormalizedUsage:
    provider = {'azure_openai': 'azure', 'aws_bedrock': 'bedrock', 'volcano': 'ark', 'alibaba': 'dashscope'}.get(provider, provider)
    if provider == 'bedrock':
        names = {'inputTokens', 'outputTokens', 'totalTokens', 'cacheReadInputTokens', 'cacheWriteInputTokens'}
        ordinary, output = _counter(raw.get('inputTokens')), _counter(raw.get('outputTokens'))
        read, write, reasoning = _counter(raw.get('cacheReadInputTokens', 0)), _counter(raw.get('cacheWriteInputTokens', 0)), None
        total = None if ordinary is None else ordinary + read + write
    elif provider in ('azure', 'ark', 'dashscope'):
        dash = provider == 'dashscope' and 'input_tokens' in raw
        ik, ok = ('input_tokens', 'output_tokens') if dash else ('prompt_tokens', 'completion_tokens')
        names = {ik, ok, 'total_tokens', 'prompt_tokens_details', 'completion_tokens_details', 'cached_tokens', 'input_tokens_details', 'output_tokens_details'}
        total, output = _counter(raw.get(ik)), _counter(raw.get(ok))
        details = raw.get('prompt_tokens_details') or raw.get('input_tokens_details') or {}
        out_details = raw.get('completion_tokens_details') or raw.get('output_tokens_details') or {}
        read = _counter(details.get('cached_tokens', raw.get('cached_tokens', 0)))
        write = _counter(details.get('cache_creation_tokens', 0))
        reasoning = _counter(out_details.get('reasoning_tokens'))
        ordinary = None if total is None else total - read - write
        if ordinary is not None and ordinary < 0:
            raise ValueError('cache subsets exceed input')
        if reasoning is not None and output is not None and reasoning > output:
            raise ValueError('reasoning subset exceeds output')
        nested_unknown = set(details) - {'cached_tokens', 'cache_creation_tokens'}
        nested_unknown |= set(out_details) - {'reasoning_tokens'}
    else:
        raise ValueError('unsupported native schema')
    unknown = set(raw) - names
    if provider != 'bedrock':
        unknown |= nested_unknown
    total_key = 'totalTokens' if provider == 'bedrock' else 'total_tokens'
    if total_key in raw and total is not None and output is not None and _counter(raw[total_key]) != total+output:
        unknown.add('total_tokens_mismatch')
    missing = tuple(k for k, v in [('input_total', total), ('output_total', output)] if v is None)
    complete = 'MISSING' if len(missing) == 2 else 'PARTIAL' if missing or unknown else 'PROVIDER_REPORTED'
    return NormalizedUsage(provider,total,output,ordinary,read,write,reasoning,complete,missing,tuple(sorted(unknown)))


@dataclass(frozen=True)
class PriceVersion:
    version_id: str
    currency: str
    rates: Mapping[str, str]
    min_input: int = 0
    max_input: int | None = None
    unit: int = 1000000
    fixed_fee: str = '0'
    provider: str = ''
    model: str = ''
    region: str = ''
    tier: str = ''
    effective_from: str = '1970-01-01T00:00:00Z'
    effective_to: str | None = None
    provenance: str = 'authored-fixture'

    def __post_init__(self):
        if not self.version_id or not self.provenance or self.unit <= 0 or self.min_input < 0 or (self.max_input is not None and self.max_input < self.min_input):
            raise ValueError('invalid price version')
        currency_code(self.currency)
        for value in self.rates.values():
            decimal_amount(value)
        decimal_amount(self.fixed_fee)
        for stamp in (self.effective_from, self.effective_to):
            if stamp and datetime.fromisoformat(stamp.replace('Z', '+00:00')).utcoffset() is None:
                raise ValueError('price timestamps require timezone')
        for key in ('effective_from','effective_to'):
            stamp=getattr(self,key)
            if stamp:
                object.__setattr__(self,key,datetime.fromisoformat(stamp.replace('Z','+00:00')).astimezone(timezone.utc).isoformat().replace('+00:00','Z'))
        if self.effective_to and self.effective_to<=self.effective_from: raise ValueError('empty price interval')
        object.__setattr__(self,'rates',MappingProxyType(dict(self.rates)))

    def to_dict(self):
        return {key: (dict(value) if key == 'rates' else value) for key,value in self.__dict__.items()}


def resolve_price(prices, input_total, *, at: str | None = None) -> PriceVersion:
    if input_total is None:
        raise ValueError('price unavailable: input total unknown')
    moment = datetime.fromisoformat(at.replace('Z','+00:00')) if at else None
    selected = [p for p in prices if p.min_input <= input_total and (p.max_input is None or input_total <= p.max_input)
                and (moment is None or (datetime.fromisoformat(p.effective_from.replace('Z','+00:00')) <= moment and
                     (p.effective_to is None or moment < datetime.fromisoformat(p.effective_to.replace('Z','+00:00')))))]
    if len(selected) != 1:
        raise ValueError('price ambiguous' if selected else 'price unavailable')
    return selected[0]


def rate_usage(usage: NormalizedUsage, price: PriceVersion) -> Decimal:
    if usage.completeness not in ('PROVIDER_REPORTED', 'RECONCILED'):
        raise ValueError('price cannot finalize incomplete native usage')
    resolve_price([price],usage.input_total)
    with localcontext() as ctx:
        ctx.prec = 50
        cost = decimal_amount(price.fixed_fee)
        for key in ('uncached_input','cache_read_input','cache_write_input','output_total'):
            quantity = getattr(usage,key)
            if quantity is None:
                raise ValueError('price cannot finalize missing counter')
            if quantity:
                if key not in price.rates:
                    raise ValueError('price unavailable for '+key)
                cost += Decimal(quantity)*decimal_amount(price.rates[key])/Decimal(price.unit)
        return cost.quantize(Decimal('.000000000001'),rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class FxQuote:
    quote_id: str
    base: str
    quote: str
    rate: str
    source: str
    observed_at: str
    purpose: str = 'reporting'

    def __post_init__(self):
        currency_code(self.base); currency_code(self.quote)
        if self.base == self.quote or not decimal_amount(self.rate) or not self.quote_id or not self.source:
            raise ValueError('invalid FX quote')
        if datetime.fromisoformat(self.observed_at.replace('Z','+00:00')).utcoffset() is None:
            raise ValueError('FX timestamps require timezone')
        object.__setattr__(self,'observed_at',datetime.fromisoformat(self.observed_at.replace('Z','+00:00')).astimezone(timezone.utc).isoformat().replace('+00:00','Z'))

    def convert(self, amount, source, target):
        amount=decimal_amount(amount)
        if source == target: return amount
        if (source,target)==(self.base,self.quote): return amount*Decimal(self.rate)
        if (source,target)==(self.quote,self.base): return amount/Decimal(self.rate)
        raise ValueError('FX orientation unavailable')


def _usd(amount,currency,fx):
    currency_code(currency)
    if currency=='USD': return decimal_amount(amount)
    if fx is None: raise ValueError('FX unavailable')
    return fx.convert(amount,currency,'USD')


def _text(value):
    return format(value,'f').rstrip('0').rstrip('.') if '.' in format(value,'f') else format(value,'f')


def standard_usd_rates(price,fx=None):
    return {key:_text(_usd(value,price.currency,fx)*MILLION/Decimal(price.unit)) for key,value in price.rates.items()}


def effective_usd_per_million(cost,currency,total_tokens,fx=None,*,complete=True):
    result={'known_cost':{'amount':format(decimal_amount(cost),'f') if cost is not None else None,'currency':currency},'usd_per_million':None,'reason':None,'complete':complete,'fx_quote_id':fx.quote_id if fx else None}
    if not complete: result['reason']='unpriced_or_partial_coverage'
    elif total_tokens is None or total_tokens<=0: result['reason']='zero_or_unknown_denominator'
    else: result['usd_per_million']=_text(_usd(cost,currency,fx)*MILLION/Decimal(total_tokens))
    return result


def capacity_report(cost,currency,utilized_tokens,capacity_tokens,period,allocation_policy,fx=None):
    if capacity_tokens<=0 or utilized_tokens<0 or utilized_tokens>capacity_tokens or not period or not allocation_policy:
        raise ValueError('invalid capacity allocation')
    return effective_usd_per_million(cost,currency,utilized_tokens,fx) | {'period':period,'allocation_policy':allocation_policy,'idle_fraction':_text(Decimal(capacity_tokens-utilized_tokens)/Decimal(capacity_tokens)),'basis':'allocated_capacity; not invoice or marginal cash cost'}


def realized_report(cost,currency,total_tokens,period,allocation_policy,fx=None,*,complete=True,tax_basis='excluded',discount_basis='gross',cost_status='RATED'):
    """Period-attributed economics; never asserts an invoice or useful-task equivalence."""
    if not period or not allocation_policy or cost_status not in ('ESTIMATED','RATED','RECONCILED'):
        raise ValueError('realized cost requires a pinned period and allocation basis')
    return effective_usd_per_million(cost,currency,total_tokens,fx,complete=complete) | {'period':period,'allocation_policy':allocation_policy,'tax_basis':tax_basis,'discount_basis':discount_basis,'cost_status':cost_status}
