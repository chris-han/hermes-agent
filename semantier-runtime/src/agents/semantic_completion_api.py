"""API endpoints for semantic completion (Weixin semantic completion flow).

Implements the PRD endpoints for multi-turn interactive semantic completion
where users provide financial event info incrementally with system guidance
on missing required fields.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List

from semantic_completion import SemanticCompletionService
from storage import (
    get_draft_store,
    get_event_store,
    SemanticDraftStatus,
)


router = APIRouter(prefix="/semantic", tags=["semantic-completion"])


# Canonical channels that Semantier exposes for user interaction surfaces.
SUPPORTED_CHANNELS = (
    "weixin",
    "feishu",
    "web_ui",
    "web_api",
)

CHANNEL_ALIASES = {
    "wx": "weixin",
    "wechat": "weixin",
    "fei": "feishu",
    "web": "web_ui",
    "webui": "web_ui",
    "ui": "web_ui",
    "api": "web_api",
    "http": "web_api",
    "rest": "web_api",
    "web-api": "web_api",
}


def normalize_channel(channel: str) -> str:
    raw = (channel or "").strip().lower()
    normalized = CHANNEL_ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_CHANNELS:
        supported = ", ".join(SUPPORTED_CHANNELS)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported channel '{channel}'. Supported channels: {supported}",
        )
    return normalized


class InitiateCompletionRequest(BaseModel):
    """Request to start semantic completion."""
    org_id: str
    user_input: str
    channel: str = "weixin"
    # FR8: v5 gate mode — tiered only
    gate_mode: str = "tiered"
    # FR7: Optional governed knowledge/projection context for tiered gate
    O_tag_v: Optional[dict] = None
    K_v: Optional[dict] = None
    Pi_v: Optional[dict] = None
    ConstraintBundle_v: Optional[dict] = None
    PrecedenceGraph_v: Optional[dict] = None
    ResolutionRules_v: Optional[dict] = None


class InitiateCompletionResponse(BaseModel):
    """Response from initiating completion."""
    conversation_id: str
    tags: List[str]
    next_question: str
    missing_fields: List[str]


class ContinueCompletionRequest(BaseModel):
    """Request to continue semantic completion."""
    conversation_id: str
    user_input: str


class ContinueCompletionResponse(BaseModel):
    """Response from continuing completion."""
    conversation_id: str
    status: str
    next_question: Optional[str] = None
    complete: bool
    missing_fields: List[str]


class DraftStatusResponse(BaseModel):
    """Response with draft status."""
    conversation_id: str
    status: str
    tags: List[str]
    amount: Optional[str]
    currency: Optional[str]
    counterparty: Optional[str]
    invoice_ref: Optional[str]
    evidence_count: int
    missing_fields: List[str]


class ValidateResponse(BaseModel):
    """Response from validation."""
    conversation_id: str
    status: str
    reasons: List[str]
    can_commit: bool
    committed: bool = False
    event_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    # FR9: Projection auditability
    projection_context_hash: Optional[str] = None
    projection_status: Optional[str] = None


# Initialize services
def get_completion_service() -> SemanticCompletionService:
    """Get semantic completion service."""
    draft_store = get_draft_store()
    return SemanticCompletionService(
        draft_store=draft_store,
        event_store=get_event_store(),
    )


@router.post("/initiate", response_model=InitiateCompletionResponse)
async def initiate_completion(request: InitiateCompletionRequest):
    """Initiate semantic completion for a user input.
    
    POST /semantic/initiate
    
    Starts a new conversation and asks for first missing required field.
    
    Request:
    {
      "org_id": "org_123",
      "user_input": "买了一本书，准备报销",
      "channel": "weixin"
    }
    
    Response:
    {
      "conversation_id": "conv_uuid",
      "tags": ["书", "报销"],
      "next_question": "📎 请输入金额 (Enter amount)",
      "missing_fields": ["amount", "currency", "counterparty", "invoice_ref"]
    }
    """
    service = get_completion_service()
    channel = normalize_channel(request.channel)

    draft, question = service.start_completion(
        org_id=request.org_id,
        user_input=request.user_input,
        channel=channel,
        gate_mode=request.gate_mode,
        O_tag_v=request.O_tag_v,
        K_v=request.K_v,
        Pi_v=request.Pi_v,
        ConstraintBundle_v=request.ConstraintBundle_v,
        PrecedenceGraph_v=request.PrecedenceGraph_v,
        ResolutionRules_v=request.ResolutionRules_v,
    )
    
    return InitiateCompletionResponse(
        conversation_id=draft.conversation_id,
        tags=draft.tags,
        next_question=question,
        missing_fields=draft.missing_fields,
    )


@router.get("/channels")
async def list_supported_channels():
    """List canonical gateway channels and accepted aliases."""
    return {
        "supported_channels": list(SUPPORTED_CHANNELS),
        "channel_aliases": CHANNEL_ALIASES,
    }


@router.post("/continue", response_model=ContinueCompletionResponse)
async def continue_completion(request: ContinueCompletionRequest):
    """Continue semantic completion with user input.
    
    POST /semantic/continue
    
    Processes user input, detects next missing field, asks follow-up question.
    
    Request:
    {
      "conversation_id": "conv_uuid",
      "user_input": "100"
    }
    
    Response:
    {
      "conversation_id": "conv_uuid",
      "status": "PENDING_USER_INPUT",
      "next_question": "📎 请输入币种 (Enter currency)",
      "complete": false,
      "missing_fields": ["currency", "counterparty", "invoice_ref"]
    }
    """
    service = get_completion_service()
    
    draft, response = service.continue_completion(
        conversation_id=request.conversation_id,
        user_input=request.user_input,
    )
    
    return ContinueCompletionResponse(
        conversation_id=draft.conversation_id,
        status=draft.status.value,
        next_question=response,
        complete=draft.status == SemanticDraftStatus.COMPLETE,
        missing_fields=draft.missing_fields,
    )


@router.get("/status/{conversation_id}", response_model=DraftStatusResponse)
async def get_draft_status(conversation_id: str):
    """Get current draft status.
    
    GET /semantic/status/{conversation_id}
    
    Returns current state of the draft without modifying it.
    """
    service = get_completion_service()
    status = service.get_draft_status(conversation_id)
    
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Draft not found")
    
    return DraftStatusResponse(**status)


@router.post("/evidence/{conversation_id}")
async def upload_evidence(
    conversation_id: str,
    file: UploadFile = File(...),
    evidence_type: str = Form(default="invoice"),
):
    """Upload evidence (e.g., invoice) for a draft.
    
    POST /semantic/evidence/{conversation_id}
    
    Multipart form:
      - file: The uploaded file
      - evidence_type: Type of evidence (invoice, receipt, etc.)
    
    Response:
    {
      "evidence_id": "evid_uuid",
      "evidence_type": "invoice",
      "reference": "/path/to/file",
      "message": "📎 已收到发票 (Invoice received)"
    }
    """
    service = get_completion_service()
    draft_store = get_draft_store()
    
    # Verify draft exists
    draft = draft_store.get_draft(conversation_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    # Store file (in real implementation, upload to storage service)
    import hashlib
    from pathlib import Path
    
    file_content = await file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    file_ref = f".semantier/evidence/{file_hash}_{file.filename}"
    
    # Ensure directory exists
    Path(file_ref).parent.mkdir(parents=True, exist_ok=True)
    
    # Save file
    with open(file_ref, "wb") as f:
        f.write(file_content)
    
    # Add evidence to draft
    evidence_id = service.add_evidence(
        conversation_id=conversation_id,
        evidence_type=evidence_type,
        reference=file_ref,
        metadata={
            "filename": file.filename,
            "size": len(file_content),
            "hash": file_hash,
        }
    )
    
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "reference": file_ref,
        "message": "📎 已收到发票 (Invoice received)",
    }


@router.post("/validate/{conversation_id}", response_model=ValidateResponse)
async def validate_draft(conversation_id: str):
    """Validate and prepare draft for commit.
    
    POST /semantic/validate/{conversation_id}
    
    Performs deterministic validation on the complete draft.
    
    Response:
    {
      "conversation_id": "conv_uuid",
      "status": "ALLOW",
      "reasons": [],
      "can_commit": true
    }
    
    Or on validation failure:
    {
      "conversation_id": "conv_uuid",
      "status": "EXCEPTION",
      "reasons": ["missing_invoice_reference"],
      "can_commit": false
    }
    """
    service = get_completion_service()

    justification, validation, commit_result = service.complete_and_validate(conversation_id)

    if not justification:
        raise HTTPException(status_code=404, detail="Draft not found")

    committed = commit_result is not None and commit_result.get("result") == "COMMITTED"
    duplicate = commit_result is not None and commit_result.get("result") == "DUPLICATE"

    return ValidateResponse(
        conversation_id=conversation_id,
        status=validation.status.value,
        reasons=validation.reasons,
        can_commit=validation.is_allow(),
        committed=committed or duplicate,
        event_id=commit_result.get("event_id") if commit_result else None,
        idempotency_key=commit_result.get("idempotency_key") if commit_result else None,
        projection_context_hash=commit_result.get("projection_context_hash") if commit_result else None,
        projection_status=commit_result.get("projection_status") if commit_result else None,
    )


@router.get("/messages/{conversation_id}")
async def get_messages(conversation_id: str):
    """Get conversation message history.
    
    GET /semantic/messages/{conversation_id}
    
    Returns all messages (user and system) in the conversation.
    """
    draft_store = get_draft_store()
    
    draft = draft_store.get_draft(conversation_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    messages = draft_store.get_messages(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "messages": messages,
    }


@router.get("/evidence/{conversation_id}")
async def get_evidence(conversation_id: str):
    """Get evidence list for a draft.
    
    GET /semantic/evidence/{conversation_id}
    
    Returns all evidence uploaded for this conversation.
    """
    draft_store = get_draft_store()
    
    draft = draft_store.get_draft(conversation_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    evidence = draft_store.get_evidence(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "evidence": evidence,
    }
