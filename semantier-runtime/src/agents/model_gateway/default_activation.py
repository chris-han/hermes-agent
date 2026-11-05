"""Idempotent platform-default activation for one already-funded personal account.

The payment transaction already writes credit, budget and the account-sync intent.
This worker never issues credit or moves money. Platform models are shared native
provider registrations, while LiteLLM Team/Key scopes remain strictly per account.
Only operator-pinned, tariff-qualified bindings may be materialized.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from dataclasses import asdict
from decimal import Decimal
from typing import Mapping

from agents.model_gateway.default_models import (
    PLATFORM_RECOMMENDATION_VERSION,
    resolved_model,
)
from billing.rating import PriceVersion
from billing.platform_pricing import latest_published_price
from contracts.model_gateway import Deployment

CONFIG_ENV = "SEMANTIER_PLATFORM_DEFAULT_MODELS_JSON"
REQUIRED = {
    "proxy_model", "native_model_id", "config_fingerprint",
    "provider_model_id", "provider",
    "region", "protocol", "max_cost", "capabilities",
    "credential_binding_id", "revision", "capability_profile",
}
OPTIONAL = {"supplier_price_version", "fx_version", "model_version", "version_status",
            "normalizer_version", "inclusion_rules_version", "tokens_per_minute"}


def configured_blueprints(env: Mapping[str, str] | None = None) -> dict:
    """Parse only an explicit operator-controlled, versioned service configuration."""
    raw = (os.environ if env is None else env).get(CONFIG_ENV, "")
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"version", "models"}:
        raise ValueError("Invalid platform default model configuration")
    if data["version"] != PLATFORM_RECOMMENDATION_VERSION or not isinstance(data["models"], dict):
        raise ValueError("Unrecognized platform model policy revision")
    from agents.model_gateway.default_models import MODELS
    allowed = {row[0] for row in MODELS}
    for key, profile in data["models"].items():
        if key not in allowed or not isinstance(profile, dict) or not REQUIRED <= set(profile) or set(profile) - REQUIRED - OPTIONAL:
            raise ValueError("Unsupported platform default model specification")
        if any(not isinstance(profile[k], str) or not profile[k] for k in REQUIRED - {"capabilities", "capability_profile"}):
            raise ValueError("Incomplete platform default model binding")
        caps = profile["capability_profile"]
        if (not isinstance(profile["capabilities"], list)
                or not profile["capabilities"]
                or not set(profile["capabilities"]) <= {"chat", "stream", "tools", "vision", "reasoning", "structured_output"}
                or len(profile["capabilities"]) != len(set(profile["capabilities"]))
                or not isinstance(caps, dict)
                or type(caps.get("context_window")) is not int
                or type(caps.get("max_output_tokens")) is not int
                or not 1 <= caps["context_window"] <= 10_000_000
                or not 1 <= caps["max_output_tokens"] <= min(caps["context_window"], 32768)):
            raise ValueError("Qualified model capabilities required")
        cost = Decimal(profile["max_cost"])
        if not cost.is_finite() or not Decimal("0") < cost <= Decimal("5"):
            raise ValueError("Default model per-call reserve exceeds bounded policy")
    return data["models"]


def price_qualified(price: PriceVersion, blueprint: dict, now: datetime) -> bool:
    """Reject missing, stale, wrong-provider or incompatible customer tariffs."""
    native_name = blueprint["provider_model_id"].split("/", 1)[-1]
    effective_from = datetime.fromisoformat(price.effective_from.replace("Z", "+00:00"))
    effective_to = datetime.fromisoformat(price.effective_to.replace("Z", "+00:00")) if price.effective_to else None
    return (
        price.currency == "CNY"
        and price.provider == blueprint["provider"]
        and price.model in (native_name, blueprint["provider_model_id"])
        and price.region == blueprint["region"]
        and effective_from <= now
        and (effective_to is None or now < effective_to)
        and {"uncached_input", "output_total"} <= set(price.rates)
        and price.provenance not in {"authored-fixture", ""}
        and (price.max_input is None or blueprint["capability_profile"]["context_window"] <= price.max_input)
    )


async def maybe_activate_funded_default(sync_store, proxy, claim, *, env=None, model_id=None):
    """Materialize at most one existing platform-pinned route; never mint money.

    The account-sync claim is the durable work intent. Binding its current route
    enqueues a successor sync revision in the same PG transaction; the old claim
    safely supersedes itself before LiteLLM Team/Key reconciliation.
    """
    if proxy is None or claim.get("currency") != "CNY":
        return False
    org, aid = claim["organization_id"], claim["account_id"]
    if not org.startswith("personal-"):
        return False
    blueprints = configured_blueprints(env)
    if not blueprints:
        return False
    user_id = org[len("personal-"):]
    from agents.model_gateway.store import GatewayStore
    from psycopg import sql
    store = GatewayStore(sync_store.dsn)
    async with await store._connection() as c:
        account = await (await c.execute(sql.SQL("""
            SELECT a.wallet,a.held,a.budget_spent,a.budget_limit,a.blocked,a.funding
            FROM {}.accounts a JOIN semantier_gateway.account_scopes s
              USING(organization_id,account_id,currency)
            JOIN semantier_gateway.account_access x USING(organization_id,account_id,currency)
            JOIN semantier_gateway.grants g ON g.organization_id=x.organization_id AND g.user_id=x.user_id
            JOIN semantier_identity.users u ON u.user_id=x.user_id
            WHERE a.organization_id=%s AND a.account_id=%s AND a.currency='CNY'
              AND s.billing_schema=%s AND s.funding=a.funding
              AND x.user_id=%s AND x.status='ACTIVE'
              AND g.permissions @> '["model:invoke"]'::jsonb
              AND u.payload_json::jsonb->>'organization_id'=%s
              AND u.payload_json::jsonb->>'membership_status'='active'
        """).format(sql.Identifier(sync_store.billing_schema)), (org, aid, sync_store.billing_schema, user_id, org))).fetchone()
        if not account or account["blocked"] or account["funding"] != "SEMANTIER_CREDITS":
            return False
        if (account["wallet"] <= account["held"]
                or account["budget_limit"] <= account["budget_spent"] + account["held"]):
            return False
        unexpired = await (await c.execute(sql.SQL("""
            SELECT 1 FROM {}.credit_grants
            WHERE organization_id=%s AND account_id=%s AND currency='CNY'
              AND NOT expired AND starts_at<=now()
              AND (expires_at IS NULL OR expires_at>now())
              AND remaining>held LIMIT 1
        """).format(sql.Identifier(sync_store.billing_schema)), (org, aid))).fetchone()
        if not unexpired:
            return False
        customer = await (await c.execute(
            "SELECT payload_json FROM semantier_identity.users WHERE user_id=%s", (user_id,)
        )).fetchone()
        if not customer:
            return False
        # Signup preference chooses the default, not the set of authorized
        # models. A funded account may use every separately qualified platform
        # model; no additional credit or model setting is required.
        selected = model_id or resolved_model(json.loads(customer["payload_json"]).get("model_preference"))
        if selected not in blueprints:
            return False
        blueprint = blueprints[selected]
        # Existing explicit/disabled routes are never overwritten or revived.
        prior = await (await c.execute("""
            SELECT 1 FROM semantier_gateway.deployments
            WHERE organization_id=%s AND body->>'account_id'=%s
              AND body->>'alias'=%s LIMIT 1
        """, (org, aid, selected))).fetchone()
        if prior:
            return False
        # The published Pricing pointer is the ONLY tariff selector.
        # No customer rate or version number is supplied by the model blueprint.
        await c.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(
            sql.Identifier(sync_store.billing_schema)))
        published = await latest_published_price(c, selected)
        if published is None:
            return False
        price = published["price"]
        if not price_qualified(price, blueprint, datetime.now(timezone.utc)):
            return False
        price_version = published["customer_price_version"]
        price_revision = published["publication_revision"]

    # Remote readback and immutable binding are intentionally outside the
    # account/billing transaction; no network request runs while PG is locked.
    deployment = Deployment(
        organization_id=org,
        account_id=aid,
        currency="CNY",
        funding="SEMANTIER_CREDITS",
        deployment_id="platform-default-" + selected,
        alias=selected,
        proxy_model=blueprint["proxy_model"],
        native_model_id=blueprint["native_model_id"],
        config_fingerprint=blueprint["config_fingerprint"],
        provider_model_id=blueprint["provider_model_id"],
        customer_price_version=price_version,
        supplier_price_version=blueprint.get("supplier_price_version"),
        fx_version=blueprint.get("fx_version"),
        provider=blueprint["provider"],
        region=blueprint["region"],
        protocol=blueprint["protocol"],
        credential_binding_id=blueprint["credential_binding_id"],
        revision=blueprint["revision"],
        max_cost=Decimal(blueprint["max_cost"]),
        capabilities=tuple(blueprint["capabilities"]),
        capability_profile=blueprint["capability_profile"],
        model_version=blueprint.get("model_version"),
        version_status=blueprint.get("version_status", "UNRESOLVED"),
        tokens_per_minute=blueprint.get("tokens_per_minute"),
        normalizer_version=blueprint.get("normalizer_version"),
        inclusion_rules_version=blueprint.get("inclusion_rules_version"),
    )
    snapshot = await proxy.operation("GET", "/model/info")
    await proxy.verify_binding(deployment, snapshot=snapshot)

    # Final custody check inside the binding owner serializes two worker
    # claims; an operator's explicit route always wins this race.
    async with await store._connection() as c:
        await c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("platform-default:" + org + ":" + aid,))
        fresh = await (await c.execute(sql.SQL("""
            SELECT wallet,held,budget_spent,budget_limit,blocked,funding
            FROM {}.accounts WHERE organization_id=%s AND account_id=%s
              AND currency='CNY' FOR UPDATE
        """).format(sql.Identifier(sync_store.billing_schema)), (org, aid))).fetchone()
        if (not fresh or fresh["blocked"] or fresh["funding"] != "SEMANTIER_CREDITS"
                or fresh["wallet"] <= fresh["held"]
                or fresh["budget_limit"] <= fresh["budget_spent"] + fresh["held"]):
            return False
        await c.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(
            sql.Identifier(sync_store.billing_schema)))
        current_publication = await latest_published_price(c, selected)
        if (current_publication is None
                or current_publication["customer_price_version"] != price_version
                or current_publication["publication_revision"] != price_revision):
            return False
        prior = await (await c.execute(
            "SELECT 1 FROM semantier_gateway.deployments WHERE organization_id=%s AND body->>'account_id'=%s AND body->>'alias'=%s LIMIT 1",
            (org, aid, selected),
        )).fetchone()
        if prior:
            return False
        # This is a platform-owned shared native registration, NOT a tenant-
        # managed model object. Tenant authority comes only from this route
        # and subsequent own-team/own-key verification.
        # Match GatewayStore.bind's canonical serialization. In particular,
        # NULL tokens_per_minute is not an override: it must be absent, since
        # the policy composer treats a present value as an explicit limit.
        payload = asdict(deployment)
        if payload["tokens_per_minute"] is None:
            del payload["tokens_per_minute"]
        payload["max_cost"] = str(deployment.max_cost)
        snapshot = {key: value for key, value in payload.items() if key != "enabled"}
        await c.execute("""
            INSERT INTO semantier_gateway.deployment_versions
              (organization_id,deployment_id,revision,body)
            VALUES (%s,%s,%s,%s::jsonb)
            ON CONFLICT DO NOTHING
        """, (org, deployment.deployment_id, deployment.revision, json.dumps(snapshot)))
        existing = await (await c.execute("""
            SELECT body FROM semantier_gateway.deployment_versions
            WHERE organization_id=%s AND deployment_id=%s AND revision=%s
        """, (org,deployment.deployment_id,deployment.revision))).fetchone()
        if not existing or existing["body"] != json.loads(json.dumps(snapshot)):
            raise ValueError("Immutable default deployment revision conflict")
        await c.execute("""
            INSERT INTO semantier_gateway.deployments(organization_id,deployment_id,body)
            VALUES (%s,%s,%s::jsonb)
        """, (org, deployment.deployment_id, json.dumps(payload)))
        from uuid import uuid4
        await c.execute("""
            INSERT INTO semantier_gateway.audit
              (event_id,produced_at,organization_id,actor_user_id,operation,target,revision)
            VALUES (%s,now(),%s,'server-provisioning','platform_default.bind',%s,%s)
        """, (uuid4().hex,org,deployment.deployment_id,deployment.revision))
        from agents.model_gateway.account_sync_store import enqueue_current
        await enqueue_current(c,org,aid,"CNY","platform_default_route",
                              billing_schema=sync_store.billing_schema)
    return True
