"""Offline metering replay. No database, provider SDK, clock or live FX access."""
import hashlib
import json
from decimal import Decimal, localcontext

from billing.rating import FxQuote, PriceVersion, normalize_usage, rate_usage


def bundle_hash(bundle):
    payload={k:v for k,v in bundle.items() if k!='sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def rerate_bundle(bundle):
    if bundle.get('schema_version')!='semantier.metering-bundle.v1' or bundle_hash(bundle)!=bundle.get('sha256'):
        raise ValueError('metering bundle integrity mismatch')
    catalog={(p['version_id'],p['kind']):PriceVersion(**p['payload']) for p in bundle['prices']}
    quotes={q['quote_id']:FxQuote(**q['payload']) for q in bundle['fx_quotes']}
    rows=[]
    for obs in bundle['usage_observations']:
        original=obs['normalized']
        if original.get('normalizer_version')!='native-usage.v1' or original.get('inclusion_rules_version')!='provider-inclusion.v1':
            raise ValueError('unsupported pinned normalizer version')
        normalized=normalize_usage(original['provider'],obs['raw_usage'])
        if normalized.to_dict()!=original:
            # JSON turns tuple fields into lists; compare canonical JSON values.
            if json.dumps(normalized.to_dict(),sort_keys=True)!=json.dumps(original,sort_keys=True):
                raise ValueError('normalization mismatch')
        price=catalog.get((obs['price_version'],'customer'))
        cost=rate_usage(normalized,price) if price and normalized.completeness=='PROVIDER_REPORTED' else None
        retained=obs['customer_cost']
        if (cost is None)!=(retained is None) or (cost is not None and cost!=Decimal(retained)):
            raise ValueError('retained rating mismatch')
        supplier_price=catalog.get((obs.get('supplier_price_version'),'supplier'))
        native=None
        reported=None
        if supplier_price and normalized.completeness=='PROVIDER_REPORTED':
            native=rate_usage(normalized,supplier_price)
            with localcontext() as context:
                context.prec=50
                target=bundle['invocation']['currency']
                reported=native if supplier_price.currency==target else quotes[obs['fx_version']].convert(native,supplier_price.currency,target).quantize(Decimal('.000000000001'))
            if native!=Decimal(obs['supplier_native_cost']) or reported!=Decimal(obs['supplier_cost']):
                raise ValueError('supplier rating mismatch')
        rows.append({'event_id':obs['event_id'],'customer_cost':format(cost,'f') if cost is not None else None,'price_version':obs['price_version'],'currency':price.currency if price else None,'supplier_native_cost':format(native,'f') if native is not None else None,'supplier_cost':format(reported,'f') if reported is not None else None})
    return {'invocation_id':bundle['invocation']['invocation_id'],'bundle_sha256':bundle['sha256'],'ratings':rows,'posting_ids':sorted({p['id'] for p in bundle['postings']}),'network_access':'NOT_USED'}
