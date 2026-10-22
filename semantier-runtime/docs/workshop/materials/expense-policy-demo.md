# Expense Policy Review teaching demo

Open the pinned **Expense Policy Review / 报销政策审阅** application in a password-authenticated DEVELOPMENT workspace: `/ontology-workbench2?taskIntent=expense-reimbursement`.

Launcher bootstrap installs the source-reviewed teaching policy, native schema, semantics, compiled plan and producer provenance in every authenticated workspace. Demo accounts need no individual grant or policy preparation. The public teaching bundle is not a production policy installation. See the [execution plan](../../plans/archive/2026-10-15-expense-policy-graph-lineage-pinned-demo-mvl-v1.md) for acceptance and the [evidence directory](../../research/logs/expense-policy-lineage-demo-v1/) for native modeling records.

1. Upload the unchanged [45.92 CNY invoice](北京象鲜科技有限公司_发票金额45.92元.pdf). Leave applicant, department and purpose empty for the first review.
2. Review the total, signed discount rows, missing fields and exception-review result. The displayed demonstration submission date is 2026-10-15; the invoice date is 2025-06-30.
3. Open the policy source and invoice source. Follow their exact highlighted passages, return to the result, and inspect the policy graph.
4. Enter fictional applicant details and a teaching purpose, then review again. The missing fields clear; the overdue exception remains. Save a draft.
5. Open global Chat for the same task. Ask it to read the current result using `evaluate_expense_reimbursement` with `action=read_current`, then suggest one possible expense purpose explicitly labeled as unverified model inference with no corresponding policy clause. Retain that completed inference and expand its actual input and generation lineage.
6. Save the draft again and refresh. The same frozen result and inference should return without recomputation.

The teaching [policy PDF](财务报销政策.pdf) is the normative source. Its Markdown file is only an authoring input. Exactly one calendar year is not overdue; the next day is. February 29 anniversaries clamp to February 28 in non-leap years. Completing a form never approves an exception.

The invoice is an original supplied document. The policy issuer and applicant entries are teaching constructs; the policy does not represent either invoice company. This journey does not verify invoice authenticity, authorize tax treatment, submit reimbursement, issue payment or post accounts. Source observations, user assertions, deterministic policy decisions and model inference remain separately labeled. Human learning benefit has not been measured.

Fresh UI replay re-executes the generated browser actions in a new password-authenticated context. Historical offline replay instead reads frozen artifacts and performs no model call, extraction, OCR or network access. These are separate checks.

All teaching files are available under **Workshop → 打开全部教学材料** (`demo-materials/`). The finance lesson offers direct invoice/policy PDF downloads and **打开报销政策演示**. Existing demo accounts without an organization can be prepared with `python scripts/prepare_workshop_demo_accounts.py`; existing organization associations and passwords are preserved. Quick real-password checks for the roster: `python scripts/verify_workshop_demo_accounts.py --output .artifacts/workshop-demo-check.json`.


## DEVELOPMENT policy version and Graph Eval

In the same pinned task, **Policy versions and Graph Eval** opens the source-version content inside the existing workbenches. Inspect Markdown v1, choose **Edit policy → Apply two-year amendment → Save policy draft**, then explicitly **Publish v2**. The controlled treatment changes the deadline to two calendar years, effective2026-10-16; other clauses remain intact. Saving a draft leaves K1 unchanged. Publication creates an immutable source, durable event and idempotent Semantica-backed candidate rebuild; it does not activate K2.

**Run Graph Eval** in EVALUATE executes and persists real structural/calendar evaluation. COMPARE links old/new clauses and the unchanged45.92CNY invoice's prospective deadline impact at2026-10-16. Open a finding for exact Markdown source focus and return; **Invoice review** preserves the original PDF evidence journey. **Replay historical K1** uses the existing pinned expense owner and retains the original exception-review result.

These are controlled DEVELOPMENT engineering expectations, not human-reviewed gold, institutionally admitted policy or generalized learning evidence. Canonical evaluation certification remains NOT_EVALUABLE even when engineering checks PASS; missing reference and interpreted residual UNKNOWN/ABSTAIN stay visible. Admission/activation is unavailable for this treatment. The original PDF remains the frozen K1 teaching basis. The immutable version sources are separate candidate evidence; no historical result is rewritten.
