---
name: corpus-graph-schema-induction
description: Use for direct authenticated corpus compiler tool, API, profile, candidate review, sample execution, or feedback/retest operations.
metadata:
  hermes:
    selectable: false
---

# Internal Corpus Graph Schema Induction Tool Surface

Use these tools to list governed induction profiles, induce a candidate from a
governed session upload, inspect candidate decisions, record candidate review,
execute or replay deterministic samples, and compare an accepted baseline with
a distinct edited correction through the pinned feedback/retest operation.

When Chat is given a `knowledge_workbench_context.v3` with complete GraphIntent
pins, use `corpus_graph_schema_intent_get` to load that exact immutable intent.
Use `corpus_graph_schema_intent_review` for explicit user accept, edit, or reject
requests. Always pass the exact ref/hash from the active context. Material
questions are cleared through an `edited` review with the exact governed
question IDs and a complete typed primary-relationship/source-role edit.
Identity and authority fields are derived from authenticated context; a Chat
paraphrase is not a new shadow intent state.

The selectable user journey is `knowledge-graph-building@1.0.0`; it invokes
this internal `corpus-graph-schema-induction` capability after governed
source verification and accepted or edited GraphIntent review.

## Graph-build preflight boundary

`knowledge-graph-building@1.0.0` owns the graph-build preflight before this
plugin is entered. The caller arrives with a resolved authenticated workspace,
active organization membership where tenant-member routes require it,
governed source access, and schema-profile availability. A missing prerequisite
returns to the journey recovery surface and re-runs preflight after recovery;
this compiler plugin is not a repair path for missing organization or membership context.

runtime-owner: TOOL | invariant-class: EXPLAINED_AND_ENFORCED | enforced-by: src/plugins/corpus_graph_schema_induction/plugin.yaml capability_contract
The tool surface is candidate-scoped: it never approves or admits a schema into
authority, changes authority bindings, submits governance decisions, or mutates
releases. An accepted review means accepted for candidate evaluation.

Sources use governed session upload references such as
`sessions/<session-id>/uploads/<document>.docx`. Tool schemas reject host paths,
arbitrary profile paths, workspace IDs, organization IDs, user IDs, and actor
IDs. Authority comes from the authenticated session.
