from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from agents.webapi.deps import require_authenticated_context

router = APIRouter(tags=["webapi-gateway"])


def _normalize_memory_relative_path(input_path: str) -> str:
    relative_path = str(input_path or "").replace("\\", "/").strip()
    if not relative_path:
        raise HTTPException(status_code=400, detail="Path is required")
    if relative_path.startswith("/") or Path(relative_path).is_absolute():
        raise HTTPException(status_code=400, detail="Absolute paths are not allowed")
    if ".." in relative_path.split("/"):
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")
    if not relative_path.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only Markdown files are allowed")
    return relative_path


def _is_curated_memory_path(relative_path: str) -> bool:
    return relative_path in {"MEMORY.md", "USER.md", "memories/MEMORY.md", "memories/USER.md"}


def _resolve_memory_file_path(workspace_root: Path, input_path: str) -> tuple[str, Path]:
    relative_path = _normalize_memory_relative_path(input_path)
    storage_relative_path = (
        f"memories/{relative_path}" if relative_path in {"MEMORY.md", "USER.md"} else relative_path
    )
    resolved_root = workspace_root.expanduser().resolve()
    full_path = (resolved_root / storage_relative_path).resolve()
    try:
        full_path.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Resolved path is outside workspace") from exc
    return relative_path, full_path


@router.get("/api/memory/read")
async def read_memory_file(request: Request, path: str = Query(default="")) -> dict[str, str]:
    ctx = require_authenticated_context(request)
    relative_path, full_path = _resolve_memory_file_path(ctx.hermes_home, path)
    try:
        content = full_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if _is_curated_memory_path(relative_path):
            return {"path": relative_path, "content": ""}
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"path": relative_path, "content": content}



@router.get('/api/memory/project-states')
def memory_project_states(request: Request, as_of: str = Query(min_length=1)):
    """Read only persisted derived state at an explicit UTC-normalized query time."""
    from datetime import datetime
    from agents.webapi.learning_workbench_routes import _owners
    from services.memory_workbench_projection_service import MemoryWorkbenchProjectionService
    with _owners(request, '/api/memory/project-states') as (context, owners):
        return MemoryWorkbenchProjectionService(store=owners._store, learning=owners).project_states(
            context=context, as_of=datetime.fromisoformat(as_of.replace('Z', '+00:00')))


@router.get('/api/memory/lineage')
def memory_lineage(request: Request, kind: str = Query(min_length=1),
                   ref: str = Query(min_length=1), hash: str = Query(pattern=r'^sha256:[a-f0-9]{64}$')):
    """Exact read projection over existing artifact owners, never a mutation API."""
    from agents.webapi.learning_workbench_routes import _owners
    from contracts.ontology_building import ArtifactPinV1
    from services.memory_workbench_projection_service import MemoryWorkbenchProjectionService
    from storage.knowledge_extraction_store import KnowledgeExtractionStore
    from storage.knowledge_builder_store import KnowledgeBuilderStore
    from agents.webapi.knowledge_source_retirement_routes import _eos_path
    with _owners(request, '/api/memory/lineage') as (context, owners):
        sources = KnowledgeBuilderStore(_eos_path(), workspace_root=context.workspace_root)
        documents = KnowledgeExtractionStore(_eos_path())
        try:
            return MemoryWorkbenchProjectionService(store=owners._store, learning=owners, sources=sources, documents=documents).lineage(context=context,
                subject=ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=hash))
        finally:
            documents._conn.close()
            sources.close()


@router.get('/api/memory/semantic-records')
def memory_semantic_records(request: Request):
    from agents.webapi.learning_workbench_routes import _owners
    from services.memory_workbench_projection_service import MemoryWorkbenchProjectionService
    with _owners(request, '/api/memory/semantic-records') as (context, owners):
        return MemoryWorkbenchProjectionService(store=owners._store, learning=owners).semantic_records(context=context)
