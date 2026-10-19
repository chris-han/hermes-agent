# Semantier development model gateway

This opt-in extension uses the existing custom-provider transport. It does not register
Semantier as Nous, replace a selected provider, or change qualified execution bindings.
Billing and model configuration belong to the selected Hermes profile. Configure an isolated
profile; never export its credential into a shared process environment.

```yaml
model:
  provider: custom
  default: configured-model-alias
  base_url: http://127.0.0.1:8000/v1
  api_key: ${SEMANTIER_MODEL_GATEWAY_TOKEN}
  context_length: 128000
billing:
  backend: semantier
semantier_gateway:
  enabled: true
  base_url: http://127.0.0.1:8000/v1
  api_key_env: SEMANTIER_MODEL_GATEWAY_TOKEN
auxiliary:
  title_generation:
    provider: custom
    model: configured-model-alias
    base_url: http://127.0.0.1:8000/v1
    api_key: ${SEMANTIER_MODEL_GATEWAY_TOKEN}
compression:
  enabled: false
```

Store `SEMANTIER_MODEL_GATEWAY_TOKEN` in that profile's `.env`, using a token issued through
real Semantier authentication with model access and `billing:read`. Credentials resolve from
the existing profile secret scope; the extension never mutates `os.environ`. Production
endpoints require HTTPS; plain HTTP is accepted only on loopback. Redirects are refused by
the billing reader. Set `billing.backend: nous` or omit it for existing Nous behavior.
`semantier_gateway.enabled: false` disables model attribution; a still-selected Semantier
billing backend then returns `provider_not_configured`, never a Nous fallback.

## Enabled caller boundary

The initial hosted profile admits primary chat (including its native tool follow-up calls)
and the existing `agent.title_generator.generate_title` helper only. `X-Semantier-Session-Id`,
`X-Semantier-Run-Id`, and `X-Semantier-Purpose` carry correlation; token authentication owns
user, organization, account, price and route authorization. Main chat's run ID is the native
task ID. Helper callers pass the owning session/run in `main_runtime`.

| Caller | Initial hosted profile |
| --- | --- |
| Primary native chat/tool loop | Enabled, purpose `chat` |
| Session-title helper | Enabled, purpose `title_generation` |
| Child agents / delegated main loop | Refused before dispatch (`subagent`) |
| Cron agent loop | Refused before dispatch (`cron`) |
| Compression, memory, vision, MoA, other shared auxiliary tasks | Refused on the configured gateway route |
| Plugin/tool internal auxiliary LLM | Refused on the configured gateway route unless it is the named title helper |
| Direct provider SDK, plugin-owned transport, arbitrary terminal/scripts | Outside this hosted profile; not claimed metered |
| Qualified Semantier structured/DLM compositions | Direct qualified transport only; generic gateway excluded by Semantier core |

The existing main and auxiliary SDKs disable hidden SDK retries. Native logical calls carry a stable Idempotency-Key (main native request identity, or a helper UUID).
Same-call transport retries retain that key; duplicate requests return invocation status instead
of dispatching another paid call. Reconstructing a chat response after caller interruption
is not claimed: a status envelope is not synthesized into model text. Gateway primary/auxiliary provider fallback
cannot escape to Nous or BYOK after a failure. This boundary does not constrain an arbitrary
user script or externally configured plugin transport. Do not advertise complete platform
metering or enable unsupported callers without separate integration evidence.

## Billing wire and Desktop

Both native canonical and legacy transports support `billing.state` and `subscription.state`.
Clients send `schema_version: semantier.billing.v1`; older clients receive a structured
`unsupported_version` refusal for Semantier rather than a dollar-shaped response. Nous
responses preserve their existing wire. Canonical reads authorize the transport's profile and
bind home, secrets and terminal scope off-loop. No legacy execution fallback is used.

Semantier account data is read from `/api/billing/state` and `/api/billing/subscription` on the
same configured origin. Amounts retain their decimal strings and explicit currency. Missing
wallets remain unavailable; subscription status, unpriced usage and pending reconciliation
are explicit. Commercial charge/plan/auto-top-up controls are absent and legacy mutation
RPCs refuse Semantier without opening Nous Remote Spending. `usage.bars` is unavailable for
Semantier because its existing contract is dollar-specific.

Settings → Billing retains the native settings primitives and active app locale. Billing
queries are keyed by socket generation/profile, removed and canceled on context changes,
and discarded when inactive. A changed backend or organization response replaces the entire
account snapshot. Out-of-band backend changes are observed on the existing polling/refetch
cycle; there is no invented push notification or client identity inference.

## Local evidence and connected acceptance

`scripts/run_tests.sh tests/agent/test_semantier_native_loop.py` exercises the real `AIAgent`
read_file loop and the real title helper through a deterministic external HTTP fixture.
`tests/gateway/test_semantier_billing.py` exercises canonical dispatch, profile-owned tokens,
schema negotiation and foreign-profile denial. These do not prove the real Semantier ledger.

Connected acceptance must replace only the external fixture behavior, using the real
Semantier gateway, token issuer, PostgreSQL ledger and adapter/Proxy. Use `AIAgent` with the
configured alias/endpoint/token, `enabled_toolsets=['file']`, a finite `max_iterations`,
`skip_context_files=True`, `skip_memory=True`, `skip_background_review=True`, and an isolated
workspace. A controlled nonstreaming witness may set `_disable_streaming=True`; streaming
requires its own native witness. Call `run_conversation(..., task_id=run_id)`, then
`generate_title(..., main_runtime={'provider':'custom','model':alias,'base_url':endpoint,
'api_key':token,'api_mode':'chat_completions','session_id':session_id,'run_id':run_id})`.
Inspect persisted attempts, raw usage, rating and account evidence. Real renderer/account
switch checks are required separately; mocked BillingApi fixtures are not connected acceptance.
