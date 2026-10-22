---
name: financial-documents
description: Convert compatible canonical_document artifacts into invoice_contract.v1 proposals and validation findings.
metadata:
  semantier:
    route: procedural_only
---
# Financial Documents

Use this plugin after `document_extraction.extract_document_content` has produced a compatible canonical document artifact. `canonical_document.v1` and DOCX `canonical_document.v2` are accepted.

Call `extract_financial_document` with `canonical_document_artifact_ref`. Do not open source files directly, do not use unmanaged paths, and do not ask users to re-enter fields that the tool extracted with sufficient confidence and evidence.

The tool returns an invoice proposal, unresolved fields, validation findings, review tasks, and a persisted `invoice_contract.v1` artifact. Treat its status as a proposal only; Semantier core owns authoritative admissibility, approval, and accounting execution.

For replay, pass `mode: "replay"` and a pinned `financial_extraction_artifact_ref`. Replay must use stored artifacts only. Missing or invalid replay artifacts are fail-closed.

If the tool surface is not loaded, report that the `financial_documents` plugin must be installed or enabled in the active workspace.
