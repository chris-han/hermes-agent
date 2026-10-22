"""Aggregated Semantier web API router."""

from __future__ import annotations

from fastapi import APIRouter

from agents.webapi.architecture_monitor_routes import router as architecture_monitor_router
from agents.webapi.jd_review_routes import router as jd_review_router
from agents.webapi.hr_workbench_routes import router as hr_workbench_router
from agents.webapi.expense_workbench_routes import router as expense_workbench_router
from agents.webapi.auth_routes import router as auth_router
from agents.webapi.business_answers_routes import router as business_answers_router
from agents.webapi.company_dataset_routes import router as company_dataset_router
from agents.webapi.contextgraph_runtime_routes import router as contextgraph_runtime_router
from agents.webapi.contextgraph_builder_routes import router as contextgraph_builder_router
from agents.webapi.contextgraph_learning_gate_routes import (
    router as contextgraph_learning_gate_router,
)
from agents.webapi.embedded_proxy_routes import router as embedded_proxy_router
from agents.webapi.evidence_routes import router as evidence_router
from agents.webapi.execution_decision_routes import router as execution_decision_router
from agents.webapi.gateway_channel_routes import router as gateway_channel_router
from agents.webapi.governed_graph_routes import router as governed_graph_router
from agents.webapi.inventory_routes import router as inventory_router
from agents.webapi.job_routes import router as job_router
from agents.webapi.knowledge_activation_routes import (
    router as knowledge_activation_router,
)
from agents.webapi.knowledge_builder_routes import router as knowledge_builder_router
from agents.webapi.knowledge_source_retirement_routes import (
    router as knowledge_source_retirement_router,
)
from agents.webapi.contextgraph_lineage_routes import (
    router as contextgraph_lineage_router,
)
from agents.webapi.legal_corpus_routes import router as legal_corpus_router
from agents.webapi.memory_routes import router as memory_router
from agents.webapi.learning_workbench_routes import router as learning_workbench_router
from agents.webapi.metadata_drift_routes import router as metadata_drift_router
from agents.webapi.ontology_building_routes import router as ontology_building_router
from agents.webapi.ontology_workbench_context_routes import (
    router as ontology_workbench_context_router,
)
from agents.webapi.organization_routes import router as organization_router
from agents.webapi.policy_rule_routes import router as policy_rule_router
from agents.webapi.plugin_extensions import include_plugin_webapi_routers
from agents.webapi.session_routes import router as session_router
from agents.webapi.sensitive_governance_routes import (
    router as sensitive_governance_router,
)
from agents.webapi.state_evolution_routes import router as state_evolution_router
from agents.webapi.runtime_context_authority_routes import (
    router as runtime_context_authority_router,
)
from agents.webapi.system_routes import router as system_router
from agents.webapi.source_document_routes import (
    router as source_document_router,
)
from agents.webapi.system_skill_routes import router as system_skill_router
from agents.webapi.tender_document_review_routes import (
    router as tender_document_review_router,
)
from agents.webapi.tender_gold_routes import router as tender_gold_router
from agents.webapi.tender_review_routes import router as tender_review_router
from agents.webapi.tender_journey_routes import router as tender_journey_router
from agents.webapi.upload_routes import router as upload_router
from agents.webapi.workflow_learning_routes import router as workflow_learning_router
from agents.webapi.workshop_classroom_routes import router as workshop_classroom_router


router = APIRouter(tags=["webapi-gateway"])

from agents.webapi.model_gateway_routes import router as model_gateway_router
from agents.webapi.model_gateway_admin_routes import router as model_gateway_admin_router
from agents.webapi.billing_routes import router as billing_router
from agents.webapi.billing_commerce_routes import router as billing_commerce_router
router.include_router(model_gateway_router)
router.include_router(model_gateway_admin_router)
router.include_router(billing_router)
router.include_router(billing_commerce_router)

router.include_router(jd_review_router)
router.include_router(hr_workbench_router)
router.include_router(expense_workbench_router)
router.include_router(upload_router)
router.include_router(evidence_router)
router.include_router(execution_decision_router)
router.include_router(company_dataset_router)
router.include_router(business_answers_router)
router.include_router(contextgraph_runtime_router)
router.include_router(contextgraph_builder_router)
router.include_router(contextgraph_learning_gate_router)
router.include_router(auth_router)
router.include_router(organization_router)
router.include_router(session_router)
router.include_router(knowledge_activation_router)
router.include_router(knowledge_builder_router)
router.include_router(knowledge_source_retirement_router)
router.include_router(contextgraph_lineage_router)
router.include_router(policy_rule_router)
router.include_router(sensitive_governance_router)
router.include_router(state_evolution_router)
router.include_router(runtime_context_authority_router)
router.include_router(tender_document_review_router)
router.include_router(tender_journey_router)
router.include_router(tender_gold_router)
router.include_router(tender_review_router)
router.include_router(legal_corpus_router)
router.include_router(architecture_monitor_router)
router.include_router(metadata_drift_router)
router.include_router(ontology_building_router)
router.include_router(governed_graph_router)
router.include_router(gateway_channel_router)
router.include_router(system_skill_router)
router.include_router(source_document_router)
router.include_router(job_router)
router.include_router(inventory_router)
router.include_router(memory_router)
router.include_router(learning_workbench_router)
router.include_router(system_router)
router.include_router(workflow_learning_router)
router.include_router(workshop_classroom_router)
# Includes the shared artifact collection, preparation, selection and binding routes.
router.include_router(ontology_workbench_context_router)
from agents.webapi.etl_pipeline_routes import router as etl_pipeline_router
router.include_router(etl_pipeline_router)
include_plugin_webapi_routers(router)

# Keep the dynamic embedded proxy routes last. FastAPI checks routes in
# registration order, so `/api/{upstream_path:path}` must not shadow concrete
# `/api/*` routes above.
router.include_router(embedded_proxy_router)

__all__ = ["router"]
