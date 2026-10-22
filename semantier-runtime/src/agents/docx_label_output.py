from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.auth_session import RequestContext
from plugins.tender_sensitive_term_detection_modeling.docx_output import (
    COMMENT_WRITER_VERSION,
    generate_labeled_docx,
    write_labeled_docx_bytes,
)


def generate_labeled_docx_derivative(
    *,
    ctx: RequestContext,
    run: dict[str, Any],
    created_by: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    return generate_labeled_docx(
        ctx=ctx,
        run=run,
        created_by=created_by,
        db_path=db_path,
    )
