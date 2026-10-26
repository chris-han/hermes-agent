---
name: tender-review-journey
description: Use when a user wants to set up, continue, or use a Tender review journey from governed policy source through repeated tender-document inspection.
author: Semantier Runtime
version: 1.9.0
category: governance
triggers:
  - set up tender review
  - use this policy for tender review
  - review this tender
  - continue tender review
  - why was this tender finding flagged
tags:
  - governance
  - tender
  - journey
  - source-first
metadata:
  hermes:
    source_tier: application
    governance: governed
    immutable: true
    selectable: true
  semantier:
    selection:
      intents:
        - all_phrases: [tender]
          any_phrases: [review, inspect, continue, policy]
        - all_phrases: [招投标审查]
        - all_phrases: [招投标检查]
        - all_phrases: [招投标政策]
      exclude_phrases: []
---

# Tender Review Journey

This is the selectable user-facing Tender journey. It orchestrates existing governed capabilities but owns no semantic compiler, certification, runtime authority, or inspection semantics.

## Authority and persistence

<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/agents/tender_journey_tool_api.py::TenderJourneyToolAPI.invoke -->
This skill is stateless. Never persist journey phase, blockers, artifact pins, or next action in Markdown, JSON, Chat history, localStorage, a URL, or skill-local state. Resolve the server-owned TenderJourney projection first. SQLite-backed TenderJourney state is authoritative for workflow continuity.

Do not infer organization/workspace authority from conversation. Use authenticated server context. Do not create or imply certification, admission, activation, A3 status, or legal conclusion from a journey phase.

## Journey behavior

1. Call `tender_journey_resolve` with `{}` on entry and after a new Chat session. The tool validates current actor/session authority, preserves a session selection, and otherwise selects a sole active or paused journey.
2. For `CHOICE_REQUIRED`, ask the user to choose from the returned journeys, then call `tender_journey_resolve` with that `journey_id`. For `NO_JOURNEY`, use `tender_journey_action` with `action: create` and a stable `arguments.command_id` when the user wants to start review setup.
3. Use `tender_journey_action` with the exact journey ID and arguments described by the returned `action_schemas`. Follow `next_action_code`; explain the next useful step in ordinary language. The tool derives identity/session and records the Chat surface. Reuse command/idempotency keys on retry. A stale command returns the current projection under `error.recovery`.
4. For policy setup, call `preflight` first. Continue intake only when it returns `READY`; report the returned blocker and recovery owner otherwise, and rerun after recovery. The check uses current DB-backed membership, governed source-artifact access and the Tender schema profile; it is not a cached permission grant. Then use `list_artifacts` to find an existing governed artifact or `list_uploads` to find an attachment in the current session. Follow `next_offset` when more uploads exist. Use `import_upload` with its exact reference/hash and the artifact collection revision, then `prepare_artifact` with purpose `DESIGN_SOURCE`. Pass the resulting artifact/preparation pins to `select_policy`, then use `discover_intent` and `build_context`. For material questions, call `read_intent`, obtain the user's resolution, and submit `review_intent` with the exact predecessor pin, selected question IDs and typed relation edit. Continue `build_context` from the reviewed successor when the projection permits it. These actions reuse the graph-intent specialist owner; they do not approve semantic authority. For certification, call `propose_competency` if no competency binding exists. The returned `certification_competency` contains source-backed questions and success conditions; unresolved proposals do not count as reviewed competency. Present material competency decisions to the user and submit `review_competency` only for their explicit choice, using the current exact competency pin, question ID, decision kind and rationale. Do not invent user confirmation; reviewed contracts are immutable successors, and a new proposal cannot reset an existing review. `review_source_coverage` records the user's explicit review of a selected, pinned set of legal sections and their semantic roles. Obtain that review before submitting; the resulting coverage describes only those sections, never an exhaustive corpus search. `prepare_authority_dependencies` takes that reviewed coverage pin and exact legal-source/version/artifact/section selections, including section hashes and reviewed roles, and returns the pinned dependency set while remaining in preparation. Do not infer approval or create reviewed coverage from conversation. Certification uses `certify_context` with existing reviewed prerequisite pins or `select_certification` with an existing owner receipt. Missing prerequisites remain blockers; report the specific missing preparation rather than claiming setup is complete.
<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
5. Ask the user to review GraphIntent only when the governed proposal reports material unresolved ambiguity. Otherwise continue safe work automatically.
<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/services/tender_journey_service.py::TenderJourneyService.bind_current_runtime_context -->
6. Treat bounded certification/readiness as distinct from runtime authority. When the server requires governed runtime binding, use the existing runtime authority owner; never manufacture a “ready” receipt.
<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/services/tender_journey_service.py::TenderJourneyService.bind_current_runtime_context -->
7. For runtime review, import the tender attachment in the same way and use `prepare_artifact` with purpose `RUNTIME_TARGET`, then `select_document`, `form_runtime_context`, and `inspect`. The shared production owners return the exact context/result pins. Use `bind_runtime` only when the projection requires it; it binds existing governed runtime authority without admitting or activating a build.
8. To explain saved findings, select the exact transition from the server-returned `inspection_episodes` and call `read_episode`. Use its saved context, source regions, findings and evidence pins; this read does not rerun inspection. After results review, use `next_document` for another tender under the same policy journey. `pause`, `resume`, and `cancel` use the current revision and retain immutable history.
<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
9. Chat explanations and Workbench exploration must refer to the same exact journey/build/runtime episode. Use the returned `workbench_url` to open the policy context or saved inspection. Episode links carry the exact transition and governed session references; the screen resolves their saved context/result and historical policy build while showing current journey progress separately. Use server pins, never `latest`.

<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
## Interrupt the user only when necessary

<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
Interrupt only for missing policy/runtime document, material GraphIntent ambiguity, required authority action, ambiguous journey selection, or material conflict/risk. Otherwise advance safe receipt-backed operations automatically.

## Recovery

<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/storage/tender_journey_store.py::TenderJourneyStore.replay_command -->
On stale revision, reload the server head and continue from its projected next action. Duplicate delivery must be idempotent. After restart or a new Chat, resume from SQLite; never reconstruct workflow state from transcript history.

On `UPLOAD_UNAVAILABLE`, call `list_uploads` again and copy the returned `upload_ref` and `upload_hash` exactly. The reference is canonically percent-encoded ASCII; the original filename is separate display metadata. Do not decode, repair, translate, or guess a reference, and do not substitute another parser or filesystem path for governed import. If the selected upload is absent, ask the user to select or upload it again.

## Source requirements and saved clarification

<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
When the selected Tender journey has produced normative semantics, read its saved `requirements_review` using `tender_journey_action` / `read_requirements`. Source semantic production attaches a bounded assessment automatically. Use `assess_requirements` for a selected producer candidate whose basis needs rechecking. Descriptive graph exploration continues without a clarification ceremony.

<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/services/tender_journey_service.py::TenderJourneyService.requirements_command -->
For a material unresolved question, copy the current review pin, question ID, dependency hash and journey revision into `review_requirement`. Save `answer`, `defer` or `reject` with the user's rationale; an accepted repair needs a persisted, reviewed candidate successor. An answer records clarification and rechecks the supported set; it does not admit or activate a rule. External source bytes remain unchanged.

<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
On a new Chat, reload the selected journey and saved answers before asking again. A stale revision reloads; an exact interrupted command reuses its key and committed receipt. Keep policy defects, internal runtime-document contradictions and established compliance findings distinct. Conditional/exception/time gaps remain UNKNOWN; continue independent advisory work.

<!-- runtime-owner: TOOL invariant-class: EXPLAINED_AND_ENFORCED enforced-by: src/services/tender_journey_service.py::TenderJourneyService.produce_normative_semantics -->
For an exactly selected prepared runtime document, call `produce_normative_semantics` with `source_role: RUNTIME_TARGET`. The existing normative schema supplies typed vocabulary only; the producer reads the target's own pinned source, persists a separate runtime candidate and attaches its requirements review. Copy the current revision and retain the command/idempotency keys on recovery. A target contradiction is neither a policy defect nor a proved bidder violation. Model unavailability does not prevent independent source inspection or advisory work.
