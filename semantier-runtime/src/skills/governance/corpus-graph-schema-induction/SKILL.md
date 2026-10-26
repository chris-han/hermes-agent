---
name: corpus-graph-schema-induction
description: Use only when the canonical knowledge-graph-building journey invokes the bounded corpus compiler capability, not for user journey selection.
author: Semantier Runtime
version: 1.0.0
category: governance
triggers:
  - internal corpus schema compiler
tags:
  - governance
  - graph-schema
  - provenance
metadata:
  hermes:
    source_tier: application
    governance: governed
    immutable: true
    selectable: false
---

# Corpus Graph Schema Induction

This is a non-selectable internal compiler capability invoked by `knowledge-graph-building`. Semantier core services retain semantic, persistence, review, and promotion authority.

## Procedure

1. Select the source through a governed upload, artifact, document reference, or registered repository fixture. Never accept an arbitrary host path from user prose.
2. Select an exact registered, repo-owned profile ID appropriate to the source kind. Do not infer a profile from a filename, content, or localized display label.
3. Record the intended use and concise competency questions that describe what the resulting graph must answer.
4. Ask the core induction service to parse the source, apply the data-only profile, and persist its evidence-backed candidate and decision envelope.
5. Present semantic roles, node/relation/event/frame proposals, representation decisions, ambiguities, evidence, and extraction hints for human review. Machine proposals are not accepted authority.
6. Record an accept, edit, or reject decision through the governed review service. An edit must carry the complete edited candidate and remains bound to its immutable original decision.
7. Execute the candidate's frozen samples or acceptance cases against the reviewed proposal, preserving source hashes, evidence anchors, candidate hashes, and review state.
8. Hand an accepted candidate to the governed schema/graph lifecycle. Do not silently activate, mutate historical artifacts, or bypass the independent persistence and projection trust gates.

Keep procedural wording in this skill. Keep profile selection, validation, candidate identity, review authority, persistence, replay, and handoff logic in Semantier core.


## Normative source review

<!-- runtime-owner: WORKFLOW invariant-class: WORKFLOW_HEURISTIC -->
When this source is modeled through a selected Tender journey, follow `tender-review-journey` → Source requirements and saved clarification. Rehydrate its saved review and answers through the shared authenticated actions. Keep descriptive graph work available; report unsupported context/exception semantics as UNKNOWN. This routing adds no generic requirements engine or rule activation.
