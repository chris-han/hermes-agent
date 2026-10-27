"""Pinned supplier catalog and pure list-price estimates, never settlement authority."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, localcontext
import hashlib
import json
import re
from zoneinfo import ZoneInfo

from contracts.billing import currency_code, decimal_amount


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _safe(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"authorization", "api_key", "password", "secret", "access_token", "headers"}:
                raise ValueError("SECRET_BEARING_CATALOG")
            _safe(child)
    elif isinstance(value, list):
        for child in value:
            _safe(child)


def _utc(value):
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.utcoffset() is None:
        raise ValueError("timezone required")
    return moment.astimezone(timezone.utc)


@dataclass(frozen=True)
class SupplierCatalog:
    """Serialized source is immutable; callers receive copies of the complete raw data."""
    raw_json: str
    version_id: str
    completeness: str
    inventory_scope: str

    @property
    def raw(self):
        return json.loads(self.raw_json)

    @classmethod
    def capture(cls, raw):
        _safe(raw)
        _utc(raw["collected_at"])
        encoded = _json(raw)
        if len(encoded.encode()) > 8 * 1024 * 1024:
            raise ValueError("CATALOG_SIZE_LIMIT")
        groups = {}
        complete = bool(raw.get("responses"))
        selected = False
        for response in raw.get("responses", []):
            query = response.get("query", {})
            selected |= bool(query.get("model"))
            key = _json({k: v for k, v in query.items() if k not in ("page_no", "page_size")})
            groups.setdefault(key, []).append(response)
        for pages in groups.values():
            totals, sizes, numbers, models = set(), set(), [], []
            for page in pages:
                body = page.get("body", {})
                output = body.get("output") or {}
                total, size, number = (output.get(k) for k in ("total", "page_size", "page_no"))
                if (page.get("http_status") != 200 or body.get("success") is not True
                        or any(type(n) is not int for n in (total, size, number))
                        or total < 0 or size <= 0 or number < 1
                        or page.get("query", {}).get("page_no") != number):
                    complete = False
                    continue
                totals.add(total); sizes.add(size); numbers.append(number)
                models.extend(output.get("models", []))
            if len(totals) != 1 or len(sizes) != 1:
                complete = False
                continue
            total, size = next(iter(totals)), next(iter(sizes))
            expected_count = max(1, (total + size - 1) // size)
            identities = [(m.get("model"), m.get("inference_provider")) for m in models]
            if (len(numbers) != expected_count or any(n != i for i, n in enumerate(sorted(numbers), 1))
                    or len(models) != total or len(set(identities)) != len(identities)):
                complete = False
        return cls(encoded, "sha256:" + hashlib.sha256(encoded.encode()).hexdigest(),
                   "COMPLETE_FOR_QUERY" if complete else "PARTIAL",
                   "SELECTED_MODELS" if selected else "QUERY_PARTITION")

    def to_dict(self):
        return {"schema_version": "supplier_catalog.v1", "raw": self.raw,
                "version_id": self.version_id, "completeness": self.completeness,
                "inventory_scope": self.inventory_scope}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != "supplier_catalog.v1":
            raise ValueError("CATALOG_SCHEMA")
        result = cls.capture(value["raw"])
        if result.to_dict() != value:
            raise ValueError("CATALOG_HASH_OR_METADATA_MISMATCH")
        return result


def _range(label):
    if label == "Default":
        return 0, None
    match = re.fullmatch(r"(?:(\d+)([km])<)?输入<=(\d+)([km])", label.lower())
    if not match:
        raise ValueError("UNMAPPED_INPUT_RANGE")
    lo, lo_unit, hi, hi_unit = match.groups()
    factor = {"k": 1000, "m": 1000000}
    return (int(lo) * factor[lo_unit] + 1 if lo else 0), int(hi) * factor[hi_unit]


def _band(moment, window):
    if not window or not window.get("source_ref"):
        raise ValueError("TIME_WINDOW_UNAVAILABLE")
    local = _utc(moment).astimezone(ZoneInfo(window["timezone"]))
    start = datetime.strptime(window["offpeak_start"], "%H:%M").time()
    end = datetime.strptime(window["offpeak_end"], "%H:%M").time()
    current = local.time().replace(tzinfo=None)
    offpeak = start <= current < end if start < end else current >= start or current < end
    return "offpeak" if offpeak else "peak"


def estimate_supplier(catalog, *, model, input_tokens, output_tokens, currency, currency_source,
                      region, mode="online", cache_kind="implicit", cache_read=0, cache_write=0,
                      time_window=None, at=None, until=None):
    """Caller supplies explicit provenance-bearing assumptions; no network or writes."""
    result = dict(schema_version="supplier_estimate.v1", amount=None, currency=currency,
                  cost_status="UNAVAILABLE", reason=None, catalog_version=catalog.version_id,
                  mapper_version="alibaba-catalog.v1", model=model, region=region, mode=mode,
                  currency_source=currency_source, effective_from=None, account_effective=False,
                  quality="UNMEASURED", time_window=time_window)
    try:
        if not currency or not currency_source or not region:
            raise ValueError("CURRENCY_OR_REGION_BASIS_UNAVAILABLE")
        currency_code(currency)
        raw = catalog.raw
        if (currency != "CNY" or region != "cn-beijing"
                or raw.get("endpoint_template") != "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/models"
                or any(p.get("query", {}).get("service_site") != "asia-pacific-china" for p in raw.get("responses", []))):
            raise ValueError("UNQUALIFIED_CURRENCY_OR_REGION_PARTITION")
        if catalog.completeness != "COMPLETE_FOR_QUERY":
            raise ValueError("CATALOG_PARTIAL")
        for count in (input_tokens, output_tokens, cache_read, cache_write):
            if type(count) is not int or count < 0:
                raise ValueError("INVALID_TOKEN_COUNT")
        if cache_read + cache_write > input_tokens:
            raise ValueError("CACHE_SUBSETS_EXCEED_INPUT")
        if mode not in ("online", "batch_file", "batch_chat") or cache_kind not in ("implicit", "explicit"):
            raise ValueError("UNSUPPORTED_MODE")
        if mode != "online" and (cache_read or cache_write):
            raise ValueError("BATCH_CACHE_COMBINATION_UNQUALIFIED")
        if cache_kind == "implicit" and cache_write:
            raise ValueError("IMPLICIT_CACHE_CREATION_UNQUALIFIED")
        matches = [m for p in catalog.raw["responses"] for m in p.get("body", {}).get("output", {}).get("models", [])
                   if m.get("model") == model]
        if len(matches) != 1:
            raise ValueError("MODEL_UNAVAILABLE_OR_AMBIGUOUS")
        tiers = []
        for tier in matches[0].get("prices", []):
            lo, hi = _range(tier["range_name"])
            if lo <= input_tokens and (hi is None or input_tokens <= hi):
                tiers.append(tier)
        if len(tiers) != 1:
            raise ValueError("PRICE_TIER_UNAVAILABLE_OR_AMBIGUOUS")
        prices = tiers[0]["prices"]
        bands = {p.get("time_band") for p in prices}
        band = "standard"
        if bands != {"standard"}:
            if not at:
                raise ValueError("BILLING_TIME_UNAVAILABLE")
            band = _band(at, time_window)
            if until:
                begin, end = _utc(at), _utc(until)
                if end < begin:
                    raise ValueError("INVALID_TIME_INTERVAL")
                # Endpoints alone miss windows crossed during a long interval.
                if end - begin >= timedelta(days=1) or _band(until, time_window) != band:
                    raise ValueError("TIME_BAND_CROSSED")
                cursor = begin
                while cursor < end:
                    if _band(cursor.isoformat(), time_window) != band:
                        raise ValueError("TIME_BAND_CROSSED")
                    cursor += timedelta(minutes=1)
        suffix = {"online": "", "batch_file": "_batch", "batch_chat": "_batch_chat"}[mode]
        quantities = {"input_token" + suffix: input_tokens - cache_read - cache_write,
                      "output_token" + suffix: output_tokens}
        if cache_read:
            quantities["input_token_cache_read" if cache_kind == "explicit" else "input_token_cache"] = cache_read
        if cache_write:
            quantities["input_token_cache_creation_5m"] = cache_write
        with localcontext() as ctx:
            ctx.prec = 50
            cost = Decimal(0)
            for kind, quantity in quantities.items():
                if not quantity:
                    continue
                rates = [p for p in prices if p["type"] == kind and p.get("time_band") == band]
                if len(rates) != 1 or rates[0].get("price_unit") != "每百万tokens":
                    raise ValueError("PRICE_OR_UNIT_UNAVAILABLE")
                cost += Decimal(quantity) * decimal_amount(rates[0]["price"]) / Decimal(1000000)
        result.update(amount=format(cost, "f"), cost_status="LIST_PRICE_ESTIMATE", time_band=band,
                      assumptions={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                   "cache_read": cache_read, "cache_write": cache_write, "cache_kind": cache_kind})
    except (ValueError, KeyError) as exc:
        result["reason"] = str(exc)
    return result
