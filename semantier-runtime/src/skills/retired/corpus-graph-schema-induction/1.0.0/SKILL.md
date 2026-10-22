---
name: corpus-graph-schema-induction
description: Governed, profile-driven corpus graph-schema induction with competency questions, evidence-backed proposals, human review, and executable samples.
author: Semantier Runtime
version: 1.0.0
category: governance
triggers:
  - corpus graph schema induction
  - document semantic structure discovery
  - graph schema proposal
tags:
  - governance
  - graph-schema
  - provenance
metadata:
  hermes:
    source_tier: application
    governance: governed
    immutable: true
---

# Corpus Graph Schema Induction

Use this capability to turn a governed corpus source into a reviewable graph-schema candidate. It guides the journey; Semantier core services retain semantic, persistence, review, and promotion authority.

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
