---
name: expense-reimbursement
category: productivity
description: Use when a user wants to review an uploaded invoice against reimbursement policy, supply application details, or save a reimbursement review draft.
triggers:
  - user mentions reimbursement (报销)
  - user provides an invoice for reimbursement review
metadata:
  hermes:
    platform_compatibility: [cli, weixin, feishu, api_server, acp]
---

# Expense reimbursement review

Entry: a reimbursement review or draft request in the current authenticated session.
Exit: show the actual core result and its evidence, or the precise unavailable capability; save a draft only through the registered owner. A draft is not a submission, approval, payment, accounting entry, or formal policy activation.

## Uploaded invoice

Discover and invoke the registered tools by their exact names:

1. `extract_document_content` with the governed session `document_ref`.
2. `extract_financial_document` with `canonical_document_artifact_ref` set to the first tool's returned `normalized_document_artifact_ref`. Use the appropriate jurisdiction hint when known.
3. `evaluate_expense_reimbursement` with `action: evaluate`, the returned `financial_extraction_artifact_ref` as `financial_ref`, an explicit `submitted_on`, and known applicant fields (`name`, `department`, `purpose`).

Use the actual tool-returned references. Do not guess workspace/session IDs or paths, read the invoice through terminal, or replace unavailable tools with invented extraction results. Report a missing tool accurately. Authentication and session binding come from the runtime, not model arguments.

## Result and missing information

Keep invoice observations, user inputs, deterministic calculations, policy results, and model inference separate. Reuse reliable extracted amounts and dates; ask only for unresolved facts or fields required by the actual policy result. An expense purpose supplied by the user is not printed invoice evidence.

Present `evaluation_status` separately from `disposition`. `insufficient_basis` has no policy disposition: missing policy must not become allowed, rejected, or exception review by prompt inference. Show the actual missing basis and extraction findings. Only a successfully executed, applicable policy may provide `allowed`, `exception_review_required`, or `rejected`. Completing applicant fields does not independently clear a policy exception.

The core owns policy thresholds, required fields, exception semantics, and version pins. Do not add asset thresholds, vehicle approval rules, invoice replacement deadlines, tax deductibility, or procurement permissions from general knowledge. The teaching policy is not a real company's financial policy.

## Optional inference

For an already saved review in the current session, first call `evaluate_expense_reimbursement` with `action: read_current`. This reads the actual frozen result without parsing or evaluating again. Use its observations as the explicit input basis for the requested inference. The task UI can retain the completed native Chat turn with **Retain latest Chat inference / 保留最新 Chat 模型推测**; this preserves its exact read-result binding and unverified label.

A requested possible explanation or use may be helpful, but label the claim **模型推测 / Model inference**, unconfirmed, next to its text. Retain its actual input basis, generation receipt, provider/model when available, prompt/input pins and UTC time through the existing Chat/result lineage owner. If that retention capability is unavailable, state that limitation; do not claim replayable lineage. Do not invent confidence, policy quotes, EvidenceRefs or highlights. State “无对应政策条款 / No corresponding policy clause” when appropriate. Never overwrite invoice facts or policy disposition with inference.

## Drafts and recovery

Use `evaluate_expense_reimbursement` with `action: save_draft` and the exact `result_ref` and `result_hash`. Report only the returned artifact reference and actual draft status. Do not modify Markdown to claim submission or approval. With new user-provided fields, evaluate again and retain the prior result as history.

Use `action: replay` with the saved result pins to recover a frozen result. Replay verifies retained dependencies without model calls or extraction. Missing, changed, or cross-session artifacts are errors, not a reason to regenerate an allegedly historical result.
