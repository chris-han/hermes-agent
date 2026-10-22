---
name: document-extraction
description: Extract governed uploaded documents into neutral canonical_document artifacts for downstream workflows.
metadata:
  semantier:
    route: procedural_only
---
# Document Extraction

Use this plugin when a workflow needs neutral document content from an uploaded PDF, image, DOCX, XLSX, HTML, XML, or JSON file.

Call `extract_document_content` with the governed `document_ref` from the active workspace session. If the gateway message only provides a Hermes document-cache path such as `/home/.../.hermes/cache/documents/...`, still call `extract_document_content` with that path; the tool imports cache files into governed session uploads before parsing. Do not open host paths, arbitrary URLs, terminal commands, generated Python, or user-supplied absolute paths as substitutes for the registered tool.

The tool returns neutral canonical document artifacts with pages, blocks, tables, key-value candidates, QR/barcode evidence, and persisted artifact references. Born-digital PDF/DOCX inputs use the qualified native evidence providers and emit `canonical_document.v2`: PyMuPDF RAWDICT observations for PDF and direct OOXML structure for DOCX. OCR is not qualified and is not used on this path. Provider-derived semantic roles are not source authority. Historical `canonical_document.v1` artifacts remain replay-compatible as stored evidence, but new PDF/DOCX live extraction does not recreate the legacy parser path. It does not decide invoice totals, tax identifiers, reimbursement eligibility, or accounting facts.

For replay, call `extract_document_content` with `mode: "replay"` and the pinned `normalized_document_artifact_ref` plus provider artifact references. Replay must use stored artifacts only; if pins or artifacts are missing, stop and report the replay failure.

If the tool surface is not loaded, report that the `document_extraction` plugin must be installed or enabled in the active workspace.
