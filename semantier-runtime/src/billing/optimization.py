"""Pure, explicitly hypothetical price comparisons; no transport or financial writer."""
from decimal import Decimal, localcontext
from datetime import datetime
from billing.rating import normalize_usage, rate_usage, resolve_price


def comparison_usage(provider, raw_usage, **versions):
    if not isinstance(raw_usage, dict):
        raise ValueError("USAGE_OBJECT_REQUIRED")
    for key in ("prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details"):
        if raw_usage.get(key) is not None and not isinstance(raw_usage[key], dict):
            raise ValueError("INVALID_USAGE_DETAILS")
    usage = normalize_usage(provider, raw_usage, **versions)
    if usage.completeness not in ("PROVIDER_REPORTED", "RECONCILED"):
        raise ValueError("INCOMPLETE_USAGE")
    return usage


def price_comparison(*, raw_usage, provider, baseline, candidate,
                     normalizer_version="native-usage.v1", inclusion_rules_version="provider-inclusion.v1",
                     kind="same_usage_estimate", basis="CUSTOMER", valuation_at=None):
    if kind not in {"same_usage_estimate", "what_if"} or basis not in {"CUSTOMER", "SUPPLIER"}:
        raise ValueError("explicit comparison kind and basis required")
    result = dict(kind=kind, basis=basis, amount=None, baseline_amount=None, difference=None,
                  percent=None, currency=baseline.currency if baseline else None, status="UNAVAILABLE",
                  quality="UNMEASURED", reason=None, actual=False, valuation_at=valuation_at,
                  baseline_price_version=baseline.version_id if baseline else None,
                  candidate_price_version=candidate.version_id if candidate else None,
                  normalizer_version=normalizer_version, inclusion_rules_version=inclusion_rules_version)
    try:
        if baseline is None or candidate is None:
            raise ValueError("PRICE_UNAVAILABLE")
        if baseline.currency != candidate.currency:
            raise ValueError("EXPLICIT_FX_BASIS_REQUIRED")
        usage = comparison_usage(provider, raw_usage, normalizer_version=normalizer_version,
                                inclusion_rules_version=inclusion_rules_version)
        if not valuation_at:
            raise ValueError("VALUATION_TIME_REQUIRED")
        moment = datetime.fromisoformat(valuation_at.replace("Z", "+00:00"))
        if moment.tzinfo is None or moment.utcoffset().total_seconds() != 0:
            raise ValueError("UTC_VALUATION_TIME_REQUIRED")
        resolve_price([baseline], usage.input_total, at=valuation_at)
        resolve_price([candidate], usage.input_total, at=valuation_at)
        with localcontext() as ctx:
            ctx.prec = 50
            old, new = rate_usage(usage, baseline), rate_usage(usage, candidate)
            difference = old - new
            result.update(amount=format(new, "f"), baseline_amount=format(old, "f"),
                          difference=format(difference, "f"),
                          percent=format(difference / old * Decimal(100), "f") if old else None,
                          status="ESTIMATE", normalized=usage.to_dict(),
                          assumptions="Same billed quantities; alternative output, quality and task effects unmeasured.")
    except ValueError as exc:
        result["reason"] = str(exc)
    return result


def catalog_comparison(*, catalog, raw_usage, provider, normalizer_version,
                       inclusion_rules_version, kind, assumptions):
    from billing.supplier_catalog import estimate_supplier
    result = dict(kind=kind, basis="SUPPLIER", actual=False, amount=None,
        baseline_amount=None, difference=None, percent=None, status="UNAVAILABLE",
        currency=assumptions.get("currency"), catalog_version=catalog.version_id,
        quality="UNMEASURED", assumptions=assumptions, reason=None)
    try:
        allowed = {"baseline_model", "candidate_model", "currency", "currency_source",
                   "region", "mode", "cache_kind", "cache_read", "cache_write", "time_window", "at", "until"}
        if set(assumptions) - allowed:
            raise ValueError("UNSUPPORTED_SUPPLIER_ASSUMPTIONS")
        usage = comparison_usage(provider, raw_usage, normalizer_version=normalizer_version,
                                inclusion_rules_version=inclusion_rules_version)
        common = {key: value for key, value in assumptions.items()
                  if key not in {"baseline_model", "candidate_model"}}
        if kind == "same_usage_estimate":
            common.update(cache_read=usage.cache_read_input, cache_write=usage.cache_write_input)
        elif kind == "what_if":
            common.setdefault("cache_read", usage.cache_read_input)
            common.setdefault("cache_write", usage.cache_write_input)
        else:
            raise ValueError("UNSUPPORTED_COMPARISON_KIND")
        common.update(input_tokens=usage.input_total, output_tokens=usage.output_total)
        old = estimate_supplier(catalog, model=assumptions["baseline_model"], **common)
        new = estimate_supplier(catalog, model=assumptions["candidate_model"], **common)
        result.update(baseline_estimate=old, candidate_estimate=new, normalized=usage.to_dict())
        if old["amount"] is None or new["amount"] is None:
            raise ValueError(old["reason"] or new["reason"])
        with localcontext() as context:
            context.prec = 50
            baseline, candidate = Decimal(old["amount"]), Decimal(new["amount"])
            difference = baseline - candidate
            result.update(amount=new["amount"], baseline_amount=old["amount"],
                difference=format(difference, "f"),
                percent=format(difference / baseline * 100, "f") if baseline else None,
                status="LIST_PRICE_ESTIMATE")
    except (ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)
    return result
