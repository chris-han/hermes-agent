---
name: knowledge-graph-building
description: Use when a user asks to build or create a knowledge graph from a governed source, or starts the ContextGraph Studio knowledge-graph journey.
author: Semantier Runtime
version: 1.0.0
category: governance
triggers:
  - build knowledge graph from source
  - create knowledge graph from source
  - ContextGraph Studio knowledge graph journey
tags:
  - governance
  - knowledge-graph
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
        - all_phrases:
            - knowledge graph
          any_phrases:
            - build
            - create
          context_phrases:
            - source
            - upload
            - uploaded
            - document
            - file
        - all_phrases:
            - ContextGraph Studio
            - knowledge graph
      exclude_phrases:
        - corpus graph schema induction
        - induce corpus schema
        - induction profile
---

# Build Knowledge Graph

Use this journey for a user-owned source-to-knowledge-graph outcome. Semantier core remains the authority for identity, source admission, intent review, candidate persistence, grounding, release, replay, and audit.

## Source-first procedure

1. Run **Graph-build preflight** before document intake or compiler work. Resolve the authenticated workspace, active organization membership when the route is tenant-member scoped, knowledge write entitlement when upload may occur, governed source-representation access, and schema-profile availability. Shell labels such as `My workspace` are not proof that these prerequisites exist. If preflight is not `READY`, do not upload, ingest, enter semantic-intent proposal, or invoke the compiler; expose the actual prerequisite and its recovery owner. After recovery, re-run preflight rather than resuming from stale context.
2. Resolve only governed upload or artifact refs. Verify `SourceIdentity` and `CanonicalSourceIR`; reject host paths and prompt-memory substitutes.
3. Propose `GraphIntent` with intended use, competency questions, generic source-role evidence, endpoint types and endpoint roles, direction, representation, and EvidenceRefs. GraphIntent is not taxonomy authority; it cannot admit relation families or types.
4. Continue only after accepted or human-edited GraphIntent review; rejection returns to source or intent revision.
5. Before Gate 1, discover or reconstruct the exact source/GraphIntent 32+6 witness barrier: 32 positives and 6 material hard negatives. Changed GraphIntent needs matching adjudicated witness lineage; discovery supplies no gold expectations.
6. Invoke `corpus-graph-schema-induction` with exact source, intent, witness, taxonomy, profile, and catalog pins. `Candidate.v4` is the first source-specific family/type capture; `Decision.v3` is the sole admission gate. System and human definitions use the same resolver.
7. A missing predicate becomes a candidate-only profile-catalog proposal pinned to the active catalog. It remains non-instantiable until a separate versioned catalog successor admits it and a successor Decision.v3 accepts it; never mutate the active catalog in that transaction.
8. Run admitted-only extraction. Emit candidate edges with stable `relation_type_id` and exact admission pins, derive family internally, and record deterministic unresolved relation evidence instead of arbitrary labels. Do not put projected `relation_family_id` on candidate edges.
9. Route Grounding without mutation: occurrence supersession selects an admitted type; repeated type error requires a schema successor; missing predicate requires a profile successor; family-definition error requires the exact taxonomy successor or profile-definition successor from its admitted binding.
10. Publish material correction as `accepted | pending_adjudication | route_only`. Unchanged-GraphIntent schema corrections may produce an accepted comparable rerun after 32+6. Changed GraphIntent always stops at `pending_adjudication` and emits no Candidate or Decision; any later promotion is a separate governed future action, never automatic in this journey. Occurrence/systematic routes stay `route_only`. Chat and direct edits with identical normalized typed content use one backend and equivalent semantic outcomes.
11. Pin source, EvidenceRefs, Candidate/Decision/extract, and correction material. Never substitute latest state or rewrite history; persistence and projection gates remain independent.
12. Keep journey learning under `knowledge-graph-building`; compiler, profile, and taxonomy learning stays under those identities.
13. Hand only an accepted, grounded candidate to governed release. This skill does not approve, admit, activate, materialize, or mutate a graph, and it does not create UI or release authority.

Direct requests to induce a corpus schema, call a tool/API, or execute a profile belong to the internal compiler surface, not this selectable journey.

| Material correction | Current outcome | Successor Candidate/Decision |
| --- | --- | --- |
| Schema-only, GraphIntent unchanged, clean 32+6 | `accepted` | present |
| GraphIntent changed | `pending_adjudication` | none |
| Occurrence/systematic route | `route_only` | none |

## Boundary quick reference

| Request | Surface |
| --- | --- |
| Build or create a knowledge graph from a source | This canonical journey |
| Induce a corpus schema or operate a profile/tool/API directly | Internal compiler capability |
| Replay an old skill ref | Read-only historical registry |

## Common mistakes

- Opening document intake before proving the authenticated workspace/organization/profile prerequisites, then misreporting a tenant-context failure as a document error.
- Asking ontology questions before verifying governed source identity and canonical IR.
- Treating generated intent or compiler output as accepted semantic authority.
- Letting GraphIntent choose taxonomy, bypassing the 32+6 barrier, or treating Candidate.v4 as admitted before Decision.v3.
- Instantiating a missing predicate from a proposal before a separate versioned catalog successor and Decision.v3.
- Emitting arbitrary relation labels or `relation_family_id` on candidate extraction edges.
- Recompiling occurrence-only or routed systematic feedback instead of recording `route_only`.
- Recording journey learning under the compiler identity, or compiler/profile learning under the journey identity.
- Rewriting an old run, learning row, decision, or release instead of creating a successor and comparable re-test.
