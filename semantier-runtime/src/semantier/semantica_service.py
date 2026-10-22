"""HTTP sidecar exposing the pinned in-process Semantica provider.

This is an execution wrapper, not an alternative extraction implementation:
all extraction is delegated to :class:`InProcessSemanticaAdapter`.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from semantier.knowledge_feature_packs.semantica_adapter import (
    NATIVE_EXTRACTION_CONTRACT,
    SEMANTICA_COMMIT,
    InProcessSemanticaAdapter,
)


app = FastAPI(title="Semantier pinned Semantica provider", version="1.0")


class ExtractionRequest(BaseModel):
    schema_version: str
    ontology_version: str
    elements: list[dict] = Field(default_factory=list)
    run_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "provider": "semantica",
        "executionMode": "pinned-in-process-provider-over-http",
        "providerCommit": SEMANTICA_COMMIT,
        "contract": NATIVE_EXTRACTION_CONTRACT,
    }


@app.post("/api/enrich/extract")
def extract(payload: ExtractionRequest) -> JSONResponse:
    if payload.schema_version != "sensitive_term_candidate_graph.v1":
        raise HTTPException(status_code=422, detail="unsupported schema_version")
    if payload.ontology_version != "tender_risk_ontology.v1":
        raise HTTPException(status_code=422, detail="unsupported ontology_version")
    request_id = f"semantica_{uuid4().hex}"
    run_id = payload.run_id or request_id
    adapter = InProcessSemanticaAdapter(
        artifact_writer=lambda _raw, digest: f"sha256:{digest}",
    )
    result = adapter.extract(elements=payload.elements, run_id=run_id)
    response = JSONResponse(result.structured_output)
    response.headers["x-request-id"] = request_id
    response.headers["x-semantica-commit"] = SEMANTICA_COMMIT
    return response
