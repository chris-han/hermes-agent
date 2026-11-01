"""Pure, versioned source normalization and USD operational catalogue projection."""
from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import json
from billing.supplier_catalog import SupplierCatalog, _range
from contracts.billing import decimal_amount
from billing.supplier_pricing_document import normalize_api, parse_document

MAPPER = 'alibaba-native-map.v1'
RATE_KEYS = {'input_token':'input_cost_per_token', 'output_token':'output_cost_per_token',
    'input_token_cache':'cache_read_input_token_cost', 'input_token_cache_read':'cache_read_input_token_cost',
    'input_token_cache_creation_5m':'cache_creation_input_token_cost'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _rates(prices, band, fx):
    selected = [p for p in prices if p.get('time_band') == band and p.get('type') in RATE_KEYS]
    rates = {}
    with localcontext() as ctx:
        ctx.prec = 50
        for p in selected:
            if p.get('price_unit') != '每百万tokens':
                raise ValueError('UNSUPPORTED_UNIT')
            key = RATE_KEYS[p['type']]
            amount = decimal_amount(p['price'])
            value = float(fx.convert(amount, 'CNY', 'USD') / Decimal(1000000))
            if key in rates and rates[key] != value:
                raise ValueError('AMBIGUOUS_CACHE_RATE')
            rates[key] = value
    if not {'input_cost_per_token','output_cost_per_token'} <= rates.keys():
        raise ValueError('MISSING_BASE_PRICE')
    return rates


def project_native_map(snapshot, base_map, fx):
    """Preserve the entire pinned native map. No clock, IO or price activation."""
    if fx is None or {fx.base, fx.quote} != {'CNY','USD'}:
        raise ValueError('EXPLICIT_CNY_USD_FX_REQUIRED')
    if not isinstance(base_map, dict) or not base_map:
        raise ValueError('FULL_BASE_MAP_REQUIRED')
    if snapshot.get('api', {}).get('status') != 'COMPLETE':
        raise ValueError('COMPLETE_API_REQUIRED')
    c = SupplierCatalog.capture(snapshot['api']['raw'])
    if c.completeness != 'COMPLETE_FOR_QUERY':
        raise ValueError('COMPLETE_API_REQUIRED')
    reference=parse_document(snapshot["document"].get("raw_html", ""), snapshot["models"], snapshot["api"]["raw"])
    result = deepcopy(base_map)
    excluded, overlays = {}, []
    rows = normalize_api(c.raw)
    for model in snapshot['models']:
        try:
            match = snapshot.get('document', {}).get('matches', {}).get(model, {})
            if snapshot.get('document', {}).get('status') != 'COMPLETE' or match.get('status') != 'MATCH' or reference['matches'].get(model,{}).get('status')!='MATCH':
                raise ValueError('DOCUMENT_REFERENCE_UNAVAILABLE_OR_CONFLICT')
            source = [r for r in rows if r['model'] == model]
            if not source:
                raise ValueError('MODEL_UNAVAILABLE')
            known_types=set(RATE_KEYS)|{'input_token_batch','output_token_batch','input_token_batch_chat','output_token_batch_chat'}
            if any(p.get('type') not in known_types for r in source for p in r['prices']):
                raise ValueError('UNSUPPORTED_PRICE_DIMENSION')
            bands = {p.get('time_band') for r in source for p in r['prices']}
            # v1 preserves time rates but cannot publish them without an exact
            # separately captured official window. No guessed peak/offpeak hours.
            time_window=reference.get('time_windows',{}).get(model)
            if bands != {'standard'}:
                if (bands != {'peak','offpeak'} or not time_window
                        or time_window.get('timezone')!='Asia/Shanghai' or not time_window.get('source_ref')
                        or len(source)!=1 or source[0]['range']!='Default'):
                    raise ValueError('TIME_WINDOW_UNAVAILABLE')
            tiers = []
            for r in source:
                lo, hi = _range(r['range'])
                rates = _rates(r['prices'], 'standard' if bands=={'standard'} else 'peak', fx)
                if hi is None:
                    if len(source) != 1:
                        raise ValueError('AMBIGUOUS_TIER')
                    tiers = [rates]
                else:
                    tiers.append({'range':[max(0, lo-1), hi], **rates})
            ordered = sorted(tiers, key=lambda t:t.get('range',[0])[0])
            if any(ordered[i-1]['range'][1] != t['range'][0] for i,t in enumerate(ordered) if i):
                raise ValueError('NONCONTIGUOUS_TIERS')
            key = 'dashscope/' + model
            entry = deepcopy(result.get(key, {}))
            # Preserve capability/context metadata; replace only source rate
            # fields. Deployment custom rates remain in native router registry.
            for field in ('input_cost_per_token','output_cost_per_token','tiered_pricing',
                          'cache_read_input_token_cost','cache_creation_input_token_cost','off_peak_pricing'):
                entry.pop(field, None)
            entry.update(litellm_provider='dashscope', mode='chat', source='https://help.aliyun.com/zh/model-studio/model-pricing')
            # Native tier selection treats zero input as no tier. Retain the
            # first band's explicit rates so positive output never becomes free.
            entry.update({**{k:v for k,v in ordered[0].items() if k!='range'}, 'tiered_pricing':ordered}
                         if 'range' in ordered[0] else ordered[0])
            if bands=={'peak','offpeak'}:
                def utc_clock(value):
                    from datetime import datetime
                    parsed=datetime.strptime(value,'%H:%M')
                    return f'{(parsed.hour-8)%24:02d}:{parsed.minute:02d}'
                entry['off_peak_pricing']={**_rates(source[0]['prices'],'offpeak',fx),
                    'hours_utc':utc_clock(time_window['offpeak_start'])+'-'+utc_clock(time_window['offpeak_end'])}
            result[key] = entry
            overlays.append(key)
        except (ValueError, KeyError, TypeError) as exc:
            excluded[model] = str(exc) if isinstance(exc, ValueError) else 'UNSUPPORTED_SOURCE_SHAPE'
    if not overlays:
        raise ValueError('NO_ELIGIBLE_OVERLAYS')
    body = canonical(result)
    return {'reference_validation':reference, 'mapper_version':MAPPER, 'snapshot_version':snapshot['version_id'],
        'fx_quote':fx.__dict__, 'base_sha256':digest(base_map), 'map':result,
        'sha256':hashlib.sha256(body).hexdigest(),
        'native_revision':hashlib.sha1(b'blob %d\0' % len(body)+body, usedforsecurity=False).hexdigest(),
        'overlays':overlays, 'excluded':excluded, 'account_effective':False,
        'coverage':'PARTIAL' if excluded else 'COMPLETE_FOR_SELECTION',
        'native_scope':'ONLINE_TEXT_COMPLETIONS',
        'retained_not_native_qualified':['BatchFile','BatchChat','other modalities']}
