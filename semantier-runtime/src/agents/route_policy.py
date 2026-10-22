"""Route-policy matrix for the unified webapi gateway.

Single source of truth for per-route auth classification and for the
upstream-forwarding allowlist used by the embedded-API proxy.

See `docs/derived/gateway-unified-multitenant-design.md` section 8.1 and
`docs/operational/saas-multitenancy-roadmap.md` items P0.1 and P0.2.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from agents.webapi.plugin_extensions import (
    plugin_route_authz_class_entries,
    plugin_route_policy_entries,
)


class RoutePolicy(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    INTERNAL = "internal"


class RouteAuthorizationClass(str, Enum):
    SELF_ONLY = "self-only"
    TENANT_MEMBER = "tenant-member"
    TENANT_ADMIN = "tenant-admin"
    SYSTEM_OPERATOR = "system/operator"


# Mapping from `(method, path)` to required policy. Path matches FastAPI
# route declaration string verbatim (including `{param}` placeholders).
# Methods with the same policy share an entry via the special method "*".
#
# Observability/audit note:
# - Only implemented FastAPI routes belong in this executable map.
# - Target surfaces such as `/institutional-cases/{case_id}/trace`,
#   `/institutional-cases/{case_id}/evidence-export`, and
#   `/observability/institutional-maturity` must be added here in the same
#   change that implements their handlers and documents them in
#   docs/derived/gateway-unified-multitenant-design.md section 8.1.1.
# - Until then, they remain target contract surfaces, not callable gateway routes.
ROUTE_POLICY_MAP: dict[tuple[str, str], RoutePolicy] = {
    ('POST', '/api/billing/budget'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/prices'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/prices'): RoutePolicy.AUTHENTICATED,

    ('GET', '/api/model-gateway/invocations/{invocation_id}/evidence'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/admin/model-gateway/provider-fields'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/admin/model-gateway/model-cost-map'): RoutePolicy.AUTHENTICATED,

    ('GET', '/api/admin/model-gateway/context'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/admin/model-gateway/models/{model_id}/update'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/admin/model-gateway/models/{model_id}/block'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/admin/model-gateway/models/{model_id}/unblock'): RoutePolicy.AUTHENTICATED,

    ('GET', '/v1/models'): RoutePolicy.AUTHENTICATED,
    ('POST', '/v1/chat/completions'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/model-gateway/credentials'): RoutePolicy.AUTHENTICATED,
    ('DELETE', '/api/model-gateway/credentials/{token_id}'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/model-gateway/invocations/{invocation_id}'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/admin/model-gateway/models'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/admin/model-gateway/models'): RoutePolicy.AUTHENTICATED,
    ('DELETE', '/api/admin/model-gateway/models/{model_id}'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/admin/model-gateway/keys'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/admin/model-gateway/keys'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/admin/model-gateway/usage'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/commerce/catalog'): RoutePolicy.PUBLIC,
    ('GET', '/api/billing/commerce/accounts'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/commerce/quotes'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/commerce/orders'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/commerce/orders'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/commerce/orders/{order_id}'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/commerce/orders/{order_id}/test-payment'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/commerce/orders/{order_id}/payment-attempts'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/commerce/orders/{order_id}/payment-attempts/{attempt_id}/simulate'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/commerce/state'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/state'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/subscription'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/usage'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/billing/comparison'): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/billing/charges'): RoutePolicy.AUTHENTICATED,

    ("GET", "/api/tender-review/context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/owner-adoption-proposals"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/owner-adoptions"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/owner-adoptions/revoke"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/records"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-review/records"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-review/records/{record_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/reports"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/candidate-reports"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-review/reports/{report_ref}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-review/reports/{report_ref}/export"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-review/docx"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-review/docx/{derivative_ref}/download"): RoutePolicy.AUTHENTICATED,

    ("GET", "/api/tender-gold/packages"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-gold/packages/{package_id}"): RoutePolicy.AUTHENTICATED,
    ("PUT", "/api/tender-gold/packages/{package_id}/draft"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-gold/packages/{package_id}/cases/{case_id}/annotations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-gold/packages/{package_id}/submissions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-gold/packages/{package_id}/comparison"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-gold/packages/{package_id}/adjudications"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-gold/packages/{package_id}/qualifications"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-gold/packages/{package_id}/dataset"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/presentation-context"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/executions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/executions/decision"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/executions/signal-user-wait"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workshop/finance/assessments"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/workshop/classroom/readiness"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/workshop/classroom/stream"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workshop/classroom/start"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/memory/semantic-records"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/memory/project-states"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/memory/lineage"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/learning"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/learning/inspect"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/learning/graph"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/learning/context-snapshot"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/learning/action"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/learning/slash"): RoutePolicy.AUTHENTICATED,
    # --- public surfaces ---
    ("*", "/health"): RoutePolicy.PUBLIC,
    ("*", "/health/detailed"): RoutePolicy.PUBLIC,
    ("GET", "/gateway/channels"): RoutePolicy.PUBLIC,
    (
        "GET",
        "/auth/context",
    ): RoutePolicy.PUBLIC,  # session-aware, payload distinguishes
    ("POST", "/auth/password/register"): RoutePolicy.PUBLIC,
    ("DELETE", "/auth/feishu/link"): RoutePolicy.AUTHENTICATED,
    ("POST", "/messaging/feishu/pairing/approve-self"): RoutePolicy.AUTHENTICATED,
    # --- authenticated tenant surfaces ---
    ("POST", "/upload"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/runtime"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/graph/nodes"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/graph/edges"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/temporal/bounds"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/graph/search"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/graph/path"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/provenance/report"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/builder/upload"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/builder/preview"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/builder/build"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/schema-profiles"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/schema-candidates"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/graph-intents/propose"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/graph-intents/{graph_intent_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/graph-intents/{graph_intent_ref}/review"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/schema-candidates/{decision_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-candidates/induce"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/review"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/sample"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/feedback-retest"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/decide"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/extract"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/feedback"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/feedback-retest"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/schema-extract-runs/{extract_run_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/schema-extract-runs/{extract_run_id}/grounding-gate"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/runtime-expert"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/evaluation/learning-gate"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workbench/actions"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workbench-contexts/resolve"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/workbench/artifacts"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/etl/evaluate"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/etl/source-row"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/etl/source-profile"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/etl/cancel"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/etl/runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/etl/runs/{receipt_ref}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/etl/runs/{receipt_ref}/exports/{filename}"): RoutePolicy.AUTHENTICATED,

    ("POST", "/api/workbench/artifacts"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workbench/artifacts/select"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workbench/artifacts/prepare"): RoutePolicy.AUTHENTICATED,
    ("PUT", "/api/workbench/artifact-bindings/{lifecycle}/{role}"): RoutePolicy.AUTHENTICATED,
    ("DELETE", "/api/workbench/artifact-bindings/{lifecycle}/{role}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/workbench-contexts/artifact"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/admit"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}/workbench"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}/audit"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}/seed-state"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/seed-search"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/seed-ignore"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/seed-select"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}/seed-inputs"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/seed-align"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/ontology-builds/{build_ref}/comparisons"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/ontology-builds/{build_ref}/comparisons/{comparison_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/telemetry"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/candidate-spans"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/jd-reviews/{review_id}/feedback"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/expense-workbench/context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/evaluations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/drafts"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/replay"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/inferences"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/expense-workbench/policy/consumer-context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/consumer-candidates"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/consumer-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/expense-workbench/policy/consumer-run"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/expense-workbench/policy"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/drafts"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/publications"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/rebuilds"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/expense-workbench/policy/activate"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/hr-workbench/context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/sources"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/models"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/interpretations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/interpretation-evidence"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/remediation"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/hr-workbench/operations/{request_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/operations/{request_id}/stop"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/hr-workbench/runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/hr-workbench/runs/{run_ref}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/hr-workbench/runs/{run_ref}/export"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/jd-reviews"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/jd-reviews/{review_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/jd-reviews/{review_id}/successors"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/jd-reviews/{review_id}/versions/{version_id}/download"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/source-documents/{sourceIdentityRef:path}/content"): RoutePolicy.AUTHENTICATED,
    ("HEAD", "/api/contextgraph/source-documents/{sourceIdentityRef:path}/content"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/extraction-runs/{run_id}/boundary-actions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/learning-events"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/candidate-graph/{graph_version}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/evidence/resolve"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/workspace-domains"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/lineages"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/lineage-graph"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/contextgraph/lineages/{lineage_root_artifact_id}/versions"): RoutePolicy.AUTHENTICATED,
    ("DELETE", "/api/contextgraph/lineages/{lineage_root_artifact_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/lineages/retirement-preview"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/contextgraph/lineages/batch-delete"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/sources"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/source-representations"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/source-representations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/source/{source_id}/retire"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/source/{source_id}/retirement"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/source/{source_id}/retirement/{operation_id}/resume"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/source/{source_id}/retirement/preview"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/evidence/reground"): RoutePolicy.AUTHENTICATED,
    ("GET", "/sessions"): RoutePolicy.AUTHENTICATED,
    ("POST", "/sessions"): RoutePolicy.AUTHENTICATED,
    ("PATCH", "/sessions"): RoutePolicy.AUTHENTICATED,
    ("DELETE", "/sessions"): RoutePolicy.AUTHENTICATED,
    ("POST", "/sessions/{session_key:path}/branches"): RoutePolicy.AUTHENTICATED,
    ("GET", "/sessions/{session_key:path}/trajectory"): RoutePolicy.AUTHENTICATED,
    ("GET", "/sessions/{session_key}/trajectory"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/sessions/{session_key:path}/artifacts/{artifact_path:path}/raw",
    ): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/sessions/{session_key}/artifacts/{artifact_path:path}/raw",
    ): RoutePolicy.AUTHENTICATED,
    ("*", "/api/sessions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/search"): RoutePolicy.AUTHENTICATED,
    ("*", "/api/sessions/{session_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id:path}/activity"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/lineage"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/activity"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id:path}/log"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/log"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/latest-descendant"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/memory-snapshot"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id:path}/effective-context"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/effective-context"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sessions/{session_id}/messages"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/sessions/{session_id}/chat"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/sessions/{session_id}/chat/stream"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/sessions/send"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/memory/read"): RoutePolicy.AUTHENTICATED,
    ("*", "/api/jobs"): RoutePolicy.AUTHENTICATED,
    ("*", "/api/jobs/{job_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/jobs/{job_id}/{action}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/jobs/{job_id}/{action}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/plugins/kanban/board"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/plugins/kanban/tasks"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/plugins/kanban/tasks/{task_id}"): RoutePolicy.AUTHENTICATED,
    ("PATCH", "/api/plugins/kanban/tasks/{task_id}"): RoutePolicy.AUTHENTICATED,
    ("DELETE", "/api/plugins/kanban/tasks/{task_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/plugins/kanban/assignees"): RoutePolicy.AUTHENTICATED,
    ('POST', '/api/business-answers/receivables/runs'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/business-answers/receivables/runs/{run_ref}'): RoutePolicy.AUTHENTICATED,
    ('GET', '/api/business-answers/receivables/answers/{answer_ref}'): RoutePolicy.AUTHENTICATED,
    ("POST", "/company-dataset-imports/upload"): RoutePolicy.AUTHENTICATED,
    ("POST", "/company-dataset-imports/preview"): RoutePolicy.AUTHENTICATED,
    ("GET", "/company-data-context/setup-status"): RoutePolicy.AUTHENTICATED,
    ("GET", "/company-dataset-imports"): RoutePolicy.AUTHENTICATED,
    ("GET", "/company-dataset-imports/active-dataset"): RoutePolicy.AUTHENTICATED,
    ("GET", "/company-dataset-imports/{import_id}"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/company-dataset-imports/{import_id}/validate",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/company-dataset-imports/{import_id}/promote"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/sensitive-governance/assessments"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/sensitive-governance/assessments/{assessment_id}/execute",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/sensitive-governance/runs/{run_id}"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/reveal",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/copy",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/sensitive-governance/runs/{run_id}/replay"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/sensitive-governance/escalations/{escalation_id}/review",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/state-evolution/continuous/runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/state-evolution/continuous/runs/{run_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/runtime-context-scope-policies"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/runtime-context-scope-policies/{policy_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/runtime-context-admissions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/runtime-context-admissions/{admission_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys/preflight"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/prepare-authority"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/propose-competency"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/produce-normative-schema"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/produce-normative-semantics"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/propose-validation-cases"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/review-validation-cases"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/validate-source-semantics"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/review-competency"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/review-source-coverage"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys/{journey_id}/runtime-context-commands/{command_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys/{journey_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys/{journey_id}/episodes/{transition_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-journeys/{journey_id}/episodes/{transition_id}/decision-basis"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/advance"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/select-policy"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/select-document"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/form-runtime-context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/inspect"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/next-document"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/build-context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/discover-intent"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/review-intent"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/select-certification"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/certify-context"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/block"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/bind-runtime"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/pause"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/select-session"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/resume"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-journeys/{journey_id}/cancel"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/detections"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/graph-focus"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-document-review/reports/{report_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-document-review/runs/{run_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/tender-document-review/runs/{run_id}/replay"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/runs/{run_id}/labeled-docx"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/runs/{run_id}/report"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/runs/{run_id}/rerun"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/tender-document-review/runs/{run_id}/feedback"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/disposition",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/feedback",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/inventory"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/dashboard"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/legal-corpus/pipeline2-evidence-contract",
    ): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/legal-corpus/acceptance-evidence-export",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/acceptance-evidence-export",
    ): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/legal-corpus/acceptance-evidence-exports",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-corpus/sources"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/versions",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/versions/{version_id}/artifacts",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/versions/{version_id}/source-upload",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/manual-fallback",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/refresh-schedule",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/refresh-check",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-corpus/scan-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/scan-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/scan-runs/{scan_run_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/sources/due"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/sources/{source_id}/status"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/scan-items/{scan_item_id}/retry",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/legal-corpus/change-candidates"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/legal-corpus/change-candidates/{candidate_id}/impact",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/legal-corpus/alerts/{alert_id}/acknowledge",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-corpus/certifications"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-corpus/activations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-runtime/execute"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/legal-runtime/replay"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workflow-learning/candidates"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workflow-learning/candidates/verify"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/workflow-learning/meeting-installation"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/architecture-monitor/evaluations"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/scheduled-evaluations",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/architecture-monitor/schedules"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/architecture-monitor/schedules"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/architecture-monitor/reviews"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/architecture-monitor/reviews/{review_id}"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/decision",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/reopen",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/handoff",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/handoff/revoke",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/close",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/architecture-monitor/dashboard"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/architecture-monitor/inspector/reviews/{review_id}",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/metadata-drift/policies"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/metadata-drift/policies/{policy_id}/certify",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/metadata-drift/policies/{policy_id}/activate",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/metadata-drift/baselines"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/metadata-drift/observations"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/metadata-drift/recovery-verifications",
    ): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/metadata-drift/findings/{finding_id}/audit",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/metadata-drift/dashboard"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/governed-graph/projections/resolve",
    ): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/governed-graph/projections/{projection_id}",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/governed-graph/projections/{projection_id}/resolve",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/governed-graph/nodes/{node_id}"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/governed-graph/assertions/{assertion_id}",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/governed-graph/anchors/{anchor_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/governed-graph/search"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/governed-graph/query"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/governed-graph/conflicts/compare",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/governed-graph/replay/{object_ref}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/governed-graph/evidence-exports"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/governed-graph/commands"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/governed-graph/index-status/{source_ref}",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/governed-graph/index-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/activations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/activations"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/activations/{activation_id}/lifecycle",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/sources"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/discovery-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/discovery-runs/{run_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/feedback-deltas"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/extraction-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/extraction-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/extraction-capability"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/extraction-runs/{run_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/extraction-runs/{run_id}/occurrences"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/extraction-runs/{run_id}/ai-grounding-suggestions"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/assertion-candidates"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/assertion-candidates/{assertion_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/assertion-candidates/batch-grounding-events"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/assertion-candidates/{assertion_id}/graph-delta-preview"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/reference-concepts"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/reference-concepts/{concept_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/reference-concepts/{concept_id}/graph-delta-preview"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/reference-concepts/batch-grounding-events"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/grounding-events"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/evidence-reviews"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/evidence-reviews/{feedback_id}/regex-proposals"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}/relation-edits"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/release"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/graph-releases"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/graph-releases/batch-delete"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/lineage"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/projection"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/projection/coherence-analysis"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/compare"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/assertion-candidates/{assertion_id}/grounding-events"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/assertion-candidates/{assertion_id}/release"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/project-activate"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/retry-materialization"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/tender-packages/{package_id}"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/tender-packages"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/evidence-obligations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/evidence-obligations"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/evidence-obligations/{obligation_id}/resolve"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/canonical-terms"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/evaluation-datasets"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/knowledge/builder/evaluation-datasets/{dataset_id}",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/evaluation-runs"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/benchmark-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-runs"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/progress"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/evaluation-progress/{source_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}/layers"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/cases"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/challenge-slices"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/artifacts"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/builder/evaluation-results/{result_id}/rating",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/promotions"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/builder/read-model-rebuilds"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/knowledge/builder/runtime-semantic-index/{authority_version_id}",
    ): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/preferences/principal"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/preferences/role"): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/preferences/workspace-default",
    ): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/policy-rules"): RoutePolicy.AUTHENTICATED,
    ("POST", "/api/knowledge/policy-rules"): RoutePolicy.AUTHENTICATED,
    ("GET", "/api/knowledge/policy-rules/{candidate_id}"): RoutePolicy.AUTHENTICATED,
    (
        "GET",
        "/api/knowledge/policy-rules/{candidate_id}/audit",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/reviewer-edit",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/transition-request",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/approval",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/rejection",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/supersession",
    ): RoutePolicy.AUTHENTICATED,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/activation-request",
    ): RoutePolicy.AUTHENTICATED,
}


# Routes that are intentionally excluded from the documented-matrix
# coverage check (e.g. auth flows whose policy is route-specific and
# documented in the design doc rather than as a single AUTH/PUBLIC label,
# or runtime convenience endpoints that fan out to multiple sub-policies).
#
# Each entry below either has its own per-route gating (org-admin checks,
# system-auth helpers, signed webhook validation) or is a framework-managed
# public surface (docs, openapi). Bringing these into ROUTE_POLICY_MAP is
# tracked under P1.2 (tenant-aware authorization model).
ROUTE_POLICY_EXEMPT_PREFIXES: tuple[str, ...] = (
    # framework-managed public surfaces
    "/openapi.json",
    "/docs",
    "/redoc",
    # auth + identity flows (route-specific policy in design doc 8.5)
    "/auth/",
    # messaging gateways (per-route owner-binding)
    "/messaging/",
    # organization / membership (org-admin gating in handlers)
    "/organizations/",
    # workspace-bounded operations (separate auth coverage)
    "/runs/",
    "/run/",
    "/files",
    "/memory",
    "/terminal",
    # diagnostics + signing surfaces (separate gating)
    "/system/",
    "/gateway/",
    "/admin/",
    "/signals",
    "/explainability/",
    "/merkle/",
    "/replay/",
    "/replay-binding/",
    "/replay-resolution/",
    "/execute",
    # semantic workflow surfaces (provider-validated entrypoints)
    "/semantic/",
    # config + model introspection (covered by proxy allowlist where forwarded)
    "/api/start-hermes",
    "/api/model/info",
    "/api/config",
)

# Catch-all proxy routes are governed by FORWARDED_UPSTREAM_ALLOWLIST below
# rather than the per-route matrix, since their effective path is dynamic.
ROUTE_POLICY_PROXY_PATHS: frozenset[str] = frozenset(
    {
        "/v1/{upstream_path:path}",
        "/api/{upstream_path:path}",
    }
)


# P1.2 tenant-aware route authorization classes.
# This mapping is intentionally scoped to tenant-facing handlers where role
# semantics must be explicit and testable.
ROUTE_AUTHZ_CLASS_MAP: dict[tuple[str, str], RouteAuthorizationClass] = {
    ('POST', '/api/billing/budget'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/prices'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/prices'): RouteAuthorizationClass.TENANT_MEMBER,

    ('GET', '/api/model-gateway/invocations/{invocation_id}/evidence'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/admin/model-gateway/provider-fields'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/admin/model-gateway/model-cost-map'): RouteAuthorizationClass.TENANT_MEMBER,

    ('GET', '/api/admin/model-gateway/context'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/admin/model-gateway/models/{model_id}/update'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/admin/model-gateway/models/{model_id}/block'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/admin/model-gateway/models/{model_id}/unblock'): RouteAuthorizationClass.TENANT_MEMBER,

    ('GET', '/v1/models'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/v1/chat/completions'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/model-gateway/credentials'): RouteAuthorizationClass.TENANT_MEMBER,
    ('DELETE', '/api/model-gateway/credentials/{token_id}'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/model-gateway/invocations/{invocation_id}'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/admin/model-gateway/models'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/admin/model-gateway/models'): RouteAuthorizationClass.TENANT_MEMBER,
    ('DELETE', '/api/admin/model-gateway/models/{model_id}'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/admin/model-gateway/keys'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/admin/model-gateway/keys'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/admin/model-gateway/usage'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/commerce/accounts'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/commerce/quotes'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/commerce/orders'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/commerce/orders'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/commerce/orders/{order_id}'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/commerce/orders/{order_id}/test-payment'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/commerce/orders/{order_id}/payment-attempts'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/commerce/orders/{order_id}/payment-attempts/{attempt_id}/simulate'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/commerce/state'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/state'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/subscription'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/usage'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/billing/comparison'): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/billing/charges'): RouteAuthorizationClass.TENANT_MEMBER,

    ("GET", "/api/tender-review/context"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/owner-adoption-proposals"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/owner-adoptions"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/owner-adoptions/revoke"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/records"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/tender-review/records"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/tender-review/records/{record_ref}"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/reports"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/candidate-reports"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/tender-review/reports/{report_ref}"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/tender-review/reports/{report_ref}/export"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/tender-review/docx"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/tender-review/docx/{derivative_ref}/download"): RouteAuthorizationClass.SELF_ONLY,

    ("GET", "/api/tender-gold/packages"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-gold/packages/{package_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("PUT", "/api/tender-gold/packages/{package_id}/draft"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-gold/packages/{package_id}/cases/{case_id}/annotations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-gold/packages/{package_id}/submissions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-gold/packages/{package_id}/comparison"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-gold/packages/{package_id}/adjudications"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-gold/packages/{package_id}/qualifications"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-gold/packages/{package_id}/dataset"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/presentation-context"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/executions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/executions/decision"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/executions/signal-user-wait"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/memory/semantic-records"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/memory/project-states"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/memory/lineage"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workshop/finance/assessments"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/workshop/classroom/readiness"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/workshop/classroom/stream"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workshop/classroom/start"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/learning"): RouteAuthorizationClass.SELF_ONLY,
    # Result reads are personal; other kinds enforce TENANT_MEMBER in the handler.
    ("GET", "/api/learning/inspect"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/learning/graph"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/learning/context-snapshot"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/learning/action"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/learning/slash"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/ontology-builds"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workbench/actions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workbench-contexts/resolve"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/workbench/artifacts"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/etl/evaluate"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/etl/source-row"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/etl/source-profile"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/etl/cancel"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/etl/runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/etl/runs/{receipt_ref}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/etl/runs/{receipt_ref}/exports/{filename}"): RouteAuthorizationClass.TENANT_MEMBER,

    ("POST", "/api/workbench/artifacts"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/workbench/artifacts/select"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/workbench/artifacts/prepare"): RouteAuthorizationClass.SELF_ONLY,
    ("PUT", "/api/workbench/artifact-bindings/{lifecycle}/{role}"): RouteAuthorizationClass.SELF_ONLY,
    ("DELETE", "/api/workbench/artifact-bindings/{lifecycle}/{role}"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/workbench-contexts/artifact"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/admit"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/ontology-builds/{build_ref}/seed-state"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/seed-search"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/seed-ignore"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/seed-select"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/ontology-builds/{build_ref}/seed-inputs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/seed-align"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/ontology-builds/{build_ref}/comparisons"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/ontology-builds/{build_ref}/comparisons/{comparison_ref}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/runtime"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/graph/nodes"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/graph/edges"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/temporal/bounds"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/graph/search"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/graph/path"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/provenance/report"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/builder/upload"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/builder/preview"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/builder/build"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/schema-profiles"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/schema-candidates"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/graph-intents/propose"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/graph-intents/{graph_intent_ref}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/graph-intents/{graph_intent_ref}/review"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/schema-candidates/{decision_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-candidates/induce"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/review"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/sample"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/feedback-retest"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-candidates/{decision_id}/decide"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/extract"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/feedback"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-decisions/{schema_decision_id}/feedback-retest"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/schema-extract-runs/{extract_run_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/schema-extract-runs/{extract_run_id}/grounding-gate"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/runtime-expert"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/evaluation/learning-gate"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/telemetry"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/candidate-spans"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/jd-reviews/{review_id}/feedback"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/expense-workbench/context"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/evaluations"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/drafts"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/replay"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/inferences"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/expense-workbench/policy/consumer-context"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/consumer-candidates"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/consumer-runs"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/expense-workbench/policy/consumer-run"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/expense-workbench/policy"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/drafts"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/publications"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/rebuilds"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/expense-workbench/policy/activate"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/hr-workbench/context"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/sources"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/models"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/interpretations"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/interpretation-evidence"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/remediation"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/hr-workbench/operations/{request_id}"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/operations/{request_id}/stop"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/hr-workbench/runs"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/hr-workbench/runs/{run_ref}"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/hr-workbench/runs/{run_ref}/export"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/jd-reviews"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/jd-reviews/{review_id}"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/jd-reviews/{review_id}/successors"): RouteAuthorizationClass.SELF_ONLY,
    ("GET", "/api/jd-reviews/{review_id}/versions/{version_id}/download"): RouteAuthorizationClass.SELF_ONLY,
    # Personal session uploads/public teaching bytes enter SELF_ONLY; all other
    # source families enforce TENANT_MEMBER in the content handler.
    ("GET", "/api/contextgraph/source-documents/{sourceIdentityRef:path}/content"): RouteAuthorizationClass.SELF_ONLY,
    ("HEAD", "/api/contextgraph/source-documents/{sourceIdentityRef:path}/content"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/contextgraph/extraction-runs/{run_id}/boundary-actions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/learning-events"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/extraction-runs/{run_id}/candidate-graph/{graph_version}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/evidence/resolve"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/workspace-domains"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/lineages"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/lineage-graph"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/contextgraph/lineages/{lineage_root_artifact_id}/versions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("DELETE", "/api/contextgraph/lineages/{lineage_root_artifact_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/lineages/retirement-preview"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/contextgraph/lineages/batch-delete"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/sources"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/source-representations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/source-representations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/source/{source_id}/retire"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/source/{source_id}/retirement"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/source/{source_id}/retirement/{operation_id}/resume"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/source/{source_id}/retirement/preview"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/evidence/reground"): RouteAuthorizationClass.TENANT_MEMBER,
    # self-scoped profile and org selection operations
    ("GET", "/auth/settings"): RouteAuthorizationClass.SELF_ONLY,
    ("PATCH", "/auth/settings"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/auth/profile/complete"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/organizations/join"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/organizations/switch"): RouteAuthorizationClass.SELF_ONLY,
    # active-tenant member operations
    ("GET", "/organizations/me"): RouteAuthorizationClass.SELF_ONLY,
    (
        "GET",
        "/organizations/materialization-policy",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/organizations/knowledge-access"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/organizations/demo-profiles"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/organizations/demo-onboarding"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/organizations/real-company"): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/organizations/sharing"): RouteAuthorizationClass.TENANT_MEMBER,
    ('POST', '/api/business-answers/receivables/runs'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/business-answers/receivables/runs/{run_ref}'): RouteAuthorizationClass.TENANT_MEMBER,
    ('GET', '/api/business-answers/receivables/answers/{answer_ref}'): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/company-dataset-imports/upload"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/company-dataset-imports/preview"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/company-data-context/setup-status",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/company-dataset-imports"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/company-dataset-imports/active-dataset",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/company-dataset-imports/{import_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/company-dataset-imports/{import_id}/validate",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/legal-corpus/inventory"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/legal-corpus/dashboard"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/pipeline2-evidence-contract",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/acceptance-evidence-export",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/acceptance-evidence-exports",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/legal-corpus/scan-runs"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/scan-runs/{scan_run_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/legal-corpus/sources/due"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/sources/{source_id}/status",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workflow-learning/candidates"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workflow-learning/candidates/verify"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/workflow-learning/meeting-installation"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/legal-corpus/change-candidates"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/legal-corpus/change-candidates/{candidate_id}/impact",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/legal-corpus/alerts/{alert_id}/acknowledge",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/upload"): RouteAuthorizationClass.SELF_ONLY,
    # tenant-admin operations
    (
        "POST",
        "/organizations/materialization-policy",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/organizations/approve"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/organizations/remove"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/organizations/member-role"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/organizations/invite"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/company-dataset-imports/{import_id}/promote",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/acceptance-evidence-export",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/legal-corpus/sources"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/versions",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/versions/{version_id}/artifacts",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/versions/{version_id}/source-upload",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/manual-fallback",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/refresh-schedule",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/sources/{source_id}/refresh-check",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/legal-corpus/scan-runs"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/legal-corpus/scan-items/{scan_item_id}/retry",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/legal-corpus/certifications"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/legal-corpus/activations"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/legal-runtime/execute"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/legal-runtime/replay"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/architecture-monitor/evaluations"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/architecture-monitor/scheduled-evaluations",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/architecture-monitor/schedules"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/architecture-monitor/schedules"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/architecture-monitor/reviews"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/architecture-monitor/reviews/{review_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/decision",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/reopen",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/handoff",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/handoff/revoke",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/architecture-monitor/reviews/{review_id}/close",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/architecture-monitor/dashboard"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/architecture-monitor/inspector/reviews/{review_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/metadata-drift/policies"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/metadata-drift/policies/{policy_id}/certify",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/metadata-drift/policies/{policy_id}/activate",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/metadata-drift/baselines"): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/metadata-drift/observations"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/metadata-drift/recovery-verifications",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "GET",
        "/api/metadata-drift/findings/{finding_id}/audit",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/metadata-drift/dashboard"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/governed-graph/projections/resolve",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/governed-graph/projections/{projection_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/governed-graph/projections/{projection_id}/resolve",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/governed-graph/nodes/{node_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/governed-graph/assertions/{assertion_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/governed-graph/anchors/{anchor_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/governed-graph/search"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/governed-graph/query"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/governed-graph/conflicts/compare",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/governed-graph/replay/{object_ref}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/governed-graph/evidence-exports",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/governed-graph/commands"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/governed-graph/index-status/{source_ref}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/governed-graph/index-runs"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/knowledge/activations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/activations"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/activations/{activation_id}/lifecycle",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/preferences/principal",
    ): RouteAuthorizationClass.SELF_ONLY,
    ("POST", "/api/knowledge/preferences/role"): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/preferences/workspace-default",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    ("POST", "/api/knowledge/builder/sources"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/discovery-runs",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/builder/discovery-runs/{run_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/builder/feedback-deltas",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/extraction-runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/extraction-runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/extraction-capability"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/extraction-runs/{run_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/extraction-runs/{run_id}/occurrences"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/extraction-runs/{run_id}/ai-grounding-suggestions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/assertion-candidates"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/assertion-candidates/{assertion_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/assertion-candidates/batch-grounding-events"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/assertion-candidates/{assertion_id}/graph-delta-preview"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/reference-concepts"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/reference-concepts/{concept_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/reference-concepts/{concept_id}/graph-delta-preview"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/reference-concepts/batch-grounding-events"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/grounding-events"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/evidence-reviews"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/evidence-reviews/{feedback_id}/regex-proposals"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/candidate-graphs/{candidate_graph_id}/relation-edits"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/reference-concepts/{concept_id}/release"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/graph-releases"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/graph-releases/batch-delete"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/lineage"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/projection"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/projection/coherence-analysis"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/graph-releases/{graph_version}/compare"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/assertion-candidates/{assertion_id}/grounding-events"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/assertion-candidates/{assertion_id}/release"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/project-activate"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/graph-releases/{graph_version}/retry-materialization"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/tender-packages/{package_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/tender-packages"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/evidence-obligations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/evidence-obligations"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/evidence-obligations/{obligation_id}/resolve"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/canonical-terms",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/evaluation-datasets",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/builder/evaluation-datasets/{dataset_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/evaluation-runs",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/benchmark-runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/progress"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/evaluation-progress/{source_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-runs/{benchmark_orchestration_id}/layers"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/cases"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/challenge-slices"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/builder/benchmark-layer-runs/{evaluation_run_id}/artifacts"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/evaluation-results/{result_id}/rating",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/builder/promotions"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/builder/read-model-rebuilds",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/builder/runtime-semantic-index/{authority_version_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/knowledge/policy-rules"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/knowledge/policy-rules"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/policy-rules/{candidate_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/transition-request",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/knowledge/policy-rules/{candidate_id}/audit",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/reviewer-edit",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/approval",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/rejection",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/supersession",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/knowledge/policy-rules/{candidate_id}/activation-request",
    ): RouteAuthorizationClass.TENANT_ADMIN,
    (
        "POST",
        "/api/sensitive-governance/assessments",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/sensitive-governance/assessments/{assessment_id}/execute",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/sensitive-governance/runs/{run_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/reveal",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/copy",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/sensitive-governance/runs/{run_id}/replay",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/sensitive-governance/escalations/{escalation_id}/review",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/state-evolution/continuous/runs"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/state-evolution/continuous/runs/{run_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/runtime-context-scope-policies"): RouteAuthorizationClass.TENANT_ADMIN,
    ("GET", "/api/runtime-context-scope-policies/{policy_ref}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/runtime-context-admissions"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/runtime-context-admissions/{admission_ref}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys/preflight"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/prepare-authority"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/propose-competency"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/produce-normative-schema"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/produce-normative-semantics"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/propose-validation-cases"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/review-validation-cases"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/validate-source-semantics"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/review-competency"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/review-source-coverage"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys/{journey_id}/runtime-context-commands/{command_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys/{journey_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys/{journey_id}/episodes/{transition_id}"): RouteAuthorizationClass.TENANT_MEMBER,
    ("GET", "/api/tender-journeys/{journey_id}/episodes/{transition_id}/decision-basis"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/advance"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/select-policy"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/select-document"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/form-runtime-context"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/inspect"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/next-document"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/build-context"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/discover-intent"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/review-intent"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/select-certification"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/certify-context"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/block"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/bind-runtime"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/pause"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/select-session"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/resume"): RouteAuthorizationClass.TENANT_MEMBER,
    ("POST", "/api/tender-journeys/{journey_id}/cancel"): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/detections",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/graph-focus",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/tender-document-review/reports/{report_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/tender-document-review/runs/{run_id}",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "GET",
        "/api/tender-document-review/runs/{run_id}/replay",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/labeled-docx",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/report",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/rerun",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/disposition",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/feedback",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    (
        "POST",
        "/api/tender-document-review/runs/{run_id}/feedback",
    ): RouteAuthorizationClass.TENANT_MEMBER,
    # reserved operator-only diagnostics surface
    # IDENTITY EXCEPTION: SYSTEM_OPERATOR is resolved from the
    # SEMANTIER_OPERATOR_USER_IDS environment variable, not a governed authority
    # record in auth.db. This is intentional (bootstrap access before a governed
    # principal may exist) but diverges from Architecture Law 1. Do not use this
    # pattern for any other authorization class. See docs/derived/gateway-unified-multitenant-design.md
    # section 8.1.2 for the documented exception and future hardening path.
    ("GET", "/system/auth/state"): RouteAuthorizationClass.SYSTEM_OPERATOR,
}


# Upstream paths that the gateway is allowed to forward to the embedded
# Hermes API. Anything outside this allowlist must be rejected with 404,
# even for authenticated callers, to keep the proxy from becoming an
# implicit auth bypass into arbitrary embedded endpoints.
#
# Match policy: each entry is either an exact path (no trailing slash) or
# a prefix ending in `/` for subtree match.
FORWARDED_UPSTREAM_ALLOWLIST: frozenset[str] = frozenset(
    {
        # OpenAI-compatible surface
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/models",
        "/v1/models/",  # subtree: /v1/models/{model_id}
        # Embedded health
        "/health",
        "/health/detailed",
        # Hermes config introspection (read-only)
        "/api/config",
        "/api/config/defaults",
        "/api/config/schema",
        "/api/model/info",
        # Memory browser endpoints still owned by embedded Hermes.
        # /api/memory/read is Semantier-native because curated short paths must
        # resolve through the authenticated workspace boundary before disk IO.
        "/api/memory",
        "/api/memory/list",
        "/api/memory/search",
        "/api/memory/write",
        # Logs viewer (workspace-scoped at the embedded layer)
        "/api/logs",
    }
)


def forwarded_upstream_allowed(upstream_path: str) -> bool:
    """Return True if `upstream_path` matches the forwarding allowlist."""
    if not upstream_path.startswith("/"):
        upstream_path = "/" + upstream_path
    if upstream_path in FORWARDED_UPSTREAM_ALLOWLIST:
        return True
    for entry in FORWARDED_UPSTREAM_ALLOWLIST:
        if entry.endswith("/") and upstream_path.startswith(entry):
            return True
    return False


def classify_route(method: str, path: str) -> RoutePolicy | None:
    """Look up the declared policy for a route, or None if undeclared."""
    explicit = ROUTE_POLICY_MAP.get((method.upper(), path))
    if explicit is not None:
        return explicit
    wildcard = ROUTE_POLICY_MAP.get(("*", path))
    if wildcard is not None:
        return wildcard
    plugin_entries = plugin_route_policy_entries()
    plugin_explicit = plugin_entries.get((method.upper(), path))
    if plugin_explicit is not None:
        return RoutePolicy(plugin_explicit)
    plugin_wildcard = plugin_entries.get(("*", path))
    return RoutePolicy(plugin_wildcard) if plugin_wildcard is not None else None


def iter_declared_routes() -> Iterable[tuple[str, str, RoutePolicy]]:
    for (method, path), policy in ROUTE_POLICY_MAP.items():
        yield method, path, policy
    for (method, path), policy in plugin_route_policy_entries().items():
        yield method, path, RoutePolicy(policy)


def classify_authorization_class(
    method: str, path: str
) -> RouteAuthorizationClass | None:
    explicit = ROUTE_AUTHZ_CLASS_MAP.get((method.upper(), path))
    if explicit is not None:
        return explicit
    wildcard = ROUTE_AUTHZ_CLASS_MAP.get(("*", path))
    if wildcard is not None:
        return wildcard
    plugin_entries = plugin_route_authz_class_entries()
    plugin_explicit = plugin_entries.get((method.upper(), path))
    if plugin_explicit is not None:
        return RouteAuthorizationClass(plugin_explicit)
    plugin_wildcard = plugin_entries.get(("*", path))
    return (
        RouteAuthorizationClass(plugin_wildcard)
        if plugin_wildcard is not None
        else None
    )
