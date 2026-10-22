---
name: tender-contextgraph-runtime
description: Read-only tender ContextGraph review through the pinned Runtime Expert toolset.
---

# Tender ContextGraph Runtime

Use only the registered read/review/feedback tools and only with the
server-provided graph binding. Never infer tenant, workspace, graph version,
authority, or evaluation status from user text or browser state.

This plugin cannot mutate candidate graphs, checkpoint artifacts, submit
governance evidence, approve, admit, or change an authority binding. If a
request requires those actions, fail closed and report that the Curator and
governance workflows are separate Phase-1 plans.
