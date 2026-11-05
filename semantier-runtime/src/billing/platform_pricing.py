"""Canonical publication pointers for platform retail model token prices.

The existing billing.prices table owns every numerical rate and immutable
PriceVersion. This module only publishes an audited, monotone pointer for
eligible platform models, shared by Pricing display and tenant activation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from billing.rating import PriceVersion

REQUIRED_RATES = frozenset({"uncached_input", "cache_read_input", "cache_write_input", "output_total"})


def _model_ids():
    from agents.model_gateway.default_models import MODELS
    return frozenset(item[0] for item in MODELS)


def qualify_publication(model_id: str, price: PriceVersion, *, now: datetime | None = None):
    """Validate an operator-authored customer price, never infer seller markup."""
    if model_id not in _model_ids() or price.currency != "CNY":
        raise ValueError("unsupported platform retail pricing scope")
    if price.provider != "dashscope" or price.model not in {model_id, "dashscope/" + model_id}:
        raise ValueError("customer price does not match platform model")
    if not price.region or price.unit != 1_000_000 or not REQUIRED_RATES.issubset(price.rates):
        raise ValueError("complete per-million customer rates required")
    if not price.provenance or price.provenance == "authored-fixture":
        raise ValueError("verified customer price provenance required")
    if any(not Decimal(value).is_finite() or Decimal(value) < 0 for value in price.rates.values()):
        raise ValueError("invalid retail rate")
    moment = now or datetime.now(timezone.utc)
    start = datetime.fromisoformat(price.effective_from.replace("Z", "+00:00"))
    end = datetime.fromisoformat(price.effective_to.replace("Z", "+00:00")) if price.effective_to else None
    if start > moment or (end is not None and moment >= end):
        raise ValueError("customer price is not currently effective")
    return True


# Both the public Pricing page and default activation read this exact projection.
CURRENT_PUBLICATION_SQL = """
SELECT m.model_id,m.publication_revision,m.customer_price_version,
       p.payload,m.published_at
FROM platform_model_price_publications m
JOIN prices p ON p.version_id=m.customer_price_version AND p.kind='customer'
WHERE m.publication_revision=(
    SELECT max(revision.publication_revision)
    FROM platform_model_price_publications revision
    WHERE revision.model_id=m.model_id
)
ORDER BY m.model_id
"""


def eligible_publication(row, *, now=None):
    if row is None:
        return None
    try:
        price = PriceVersion(**row["payload"])
        qualify_publication(row["model_id"], price, now=now)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "model_id": row["model_id"],
        "publication_revision": row["publication_revision"],
        "customer_price_version": row["customer_price_version"],
        "price": price,
    }


async def latest_published_price(connection, model_id):
    if model_id not in _model_ids():
        raise ValueError("unsupported platform model")
    row = await (await connection.execute(
        "SELECT model_id,publication_revision,customer_price_version,payload,published_at "
        "FROM (" + CURRENT_PUBLICATION_SQL + ") published WHERE model_id=%s",
        (model_id,),
    )).fetchone()
    return eligible_publication(row)


async def list_published_prices(connection):
    rows = await (await connection.execute(CURRENT_PUBLICATION_SQL)).fetchall()
    visible = []
    for row in rows:
        item = eligible_publication(row)
        if not item:
            continue
        p = item["price"]
        visible.append({
            "model_id": item["model_id"],
            "publication_revision": item["publication_revision"],
            "customer_price_version": item["customer_price_version"],
            "currency": p.currency,
            "provider": p.provider,
            "region": p.region,
            "unit": p.unit,
            "rates": dict(p.rates),
            "effective_from": p.effective_from,
            "effective_to": p.effective_to,
        })
    return visible

async def wake_credited_personal_accounts(connection, model_id: str, *, billing_schema: str) -> int:
    """Reopen existing DONE sync claims after an approved model tariff publishes.

    The caller is already inside the publication transaction. A credit balance
    never grants access by itself: tenant identity, membership, scoped account
    rights, nonexpired grant, budget and absence of explicit routes all apply.
    The native worker must separately qualify the configured physical model.
    """
    from agents.model_gateway.account_sync_store import target_configuration
    from psycopg import sql

    if model_id not in _model_ids():
        raise ValueError("unsupported platform model")
    target, namespace = target_configuration()
    rows = await (await connection.execute(sql.SQL("""
        SELECT DISTINCT p.target_instance_id,p.environment_namespace,
               p.organization_id,p.account_id,p.currency,p.desired_revision
        FROM semantier_gateway.account_projections p
        JOIN {}.accounts a USING(organization_id,account_id,currency)
        JOIN semantier_gateway.account_scopes s
          USING(organization_id,account_id,currency)
        JOIN semantier_gateway.account_access x
          USING(organization_id,account_id,currency)
        JOIN semantier_gateway.grants g
          ON g.organization_id=x.organization_id AND g.user_id=x.user_id
        JOIN semantier_identity.users u ON u.user_id=x.user_id
        WHERE p.target_instance_id=%s AND p.environment_namespace=%s
          AND s.billing_schema=%s AND s.funding=a.funding
          AND p.organization_id='personal-'||u.user_id
          AND x.role='owner' AND x.status='ACTIVE'
          AND g.permissions @> '["model:invoke"]'::jsonb
          AND u.payload_json::jsonb->>'organization_id'=p.organization_id
          AND u.payload_json::jsonb->>'membership_status'='active'
          -- Signup's model preference is a default selection, never a
          -- restriction on the set of commercially eligible choices.
          AND a.currency='CNY' AND a.funding='SEMANTIER_CREDITS'
          AND NOT a.blocked AND a.wallet>a.held
          AND a.budget_limit>a.budget_spent+a.held
          AND EXISTS (
              SELECT 1 FROM {}.credit_grants credit
              WHERE credit.organization_id=a.organization_id
                AND credit.account_id=a.account_id AND credit.currency=a.currency
                AND NOT credit.expired AND credit.starts_at<=now()
                AND (credit.expires_at IS NULL OR credit.expires_at>now())
                AND credit.remaining>credit.held
          )
          AND NOT EXISTS (
              SELECT 1 FROM semantier_gateway.deployments d
              WHERE d.organization_id=p.organization_id
                AND d.body->>'account_id'=p.account_id
                AND d.body->>'alias'=%s
          )
        ORDER BY p.organization_id,p.account_id
        LIMIT 501
    """).format(sql.Identifier(billing_schema),sql.Identifier(billing_schema)),
        (target,namespace,billing_schema,model_id))).fetchall()
    if len(rows)>500:
        raise ValueError("bounded pricing activation batch exceeded; split publication by environment")
    activated=0
    for row in rows:
        scope=tuple(row[k] for k in (
            "target_instance_id","environment_namespace",
            "organization_id","account_id","currency",
        ))
        cursor=await connection.execute("""
            UPDATE semantier_gateway.account_sync_intents
            SET status='PENDING',available_at=now(),failures=0,
                lease_owner=NULL,lease_token=NULL,lease_until=NULL,
                last_error_code=NULL,completed_at=NULL
            WHERE target_instance_id=%s AND environment_namespace=%s
              AND organization_id=%s AND account_id=%s AND currency=%s
              AND revision=%s
              AND status IN ('DONE','RETRY_WAIT','NEEDS_ATTENTION',
                             'BLOCKED_DEPENDENCY','OWNERSHIP_CONFLICT')
            RETURNING event_id
        """, (*scope,row["desired_revision"]))
        if await cursor.fetchone():
            await connection.execute("""
                UPDATE semantier_gateway.account_projections
                SET status='PENDING',last_error_code=NULL,updated_at=now()
                WHERE target_instance_id=%s AND environment_namespace=%s
                  AND organization_id=%s AND account_id=%s AND currency=%s
                  AND desired_revision=%s
            """, (*scope,row["desired_revision"]))
            activated+=1
    return activated

