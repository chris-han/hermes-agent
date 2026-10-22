"""Knowledge extraction request contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


KnowledgeBuilderSourceRole = Literal[
    "reference_sensitive_word_list",
    "runtime_tender_document",
    "legacy_unknown",
]
KnowledgeBuilderWorkflowKind = Literal[
    "reference_graph_build",
    "runtime_tender_evaluation",
    "legacy_candidate_assertion",
]


class KnowledgeBuilderExtractionRunRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["knowledge_builder_extraction_run_request.v2"] = Field(
        alias="schemaVersion"
    )
    source_id: str = Field(alias="sourceId", min_length=1)
    discovery_run_id: str = Field(alias="discoveryRunId", min_length=1)
    document_id: str = Field(alias="documentId", min_length=1)
    provider: str = "semantica"
    profile: str | None = None
    tender_package_id: str | None = Field(default=None, alias="tenderPackageId")
    canonical_ir_ref: str | None = Field(default=None, alias="canonicalIrRef")
    source_identity_ref: str | None = Field(default=None, alias="sourceIdentityRef")
    parser_trace_id: str | None = Field(default=None, alias="parserTraceId")
    provider_options: dict = Field(default_factory=dict, alias="providerOptions")
    source_role: KnowledgeBuilderSourceRole = Field(alias="sourceRole")
    workflow_kind: KnowledgeBuilderWorkflowKind = Field(alias="workflowKind")

    @model_validator(mode="after")
    def require_reference_graph_build_route(self) -> "KnowledgeBuilderExtractionRunRequestV2":
        if (
            self.source_role == "runtime_tender_document"
            or self.workflow_kind == "runtime_tender_evaluation"
        ):
            raise ValueError(
                "runtime tender documents must use tender document review evaluation routes"
            )
        if self.source_role != "reference_sensitive_word_list":
            raise ValueError("sourceRole must be reference_sensitive_word_list")
        if self.workflow_kind != "reference_graph_build":
            raise ValueError("workflowKind must be reference_graph_build")
        return self
