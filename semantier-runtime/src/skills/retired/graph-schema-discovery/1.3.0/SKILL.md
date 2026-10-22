---
name: graph-schema-discovery
description: Compatibility alias for historical graph-schema-discovery learning records; new corpus induction uses corpus-graph-schema-induction.
author: Semantier Runtime
version: 1.3.0
category: governance
triggers:
  - graph schema discovery
  - ontology discovery
tags:
  - governance
  - compatibility
metadata:
  hermes:
    source_tier: application
    governance: governed
    immutable: true
---

# Graph Schema Discovery Compatibility Alias

For new source-to-schema journeys, use `corpus-graph-schema-induction@1.0.0` and an exact governed profile ID.

This alias remains available so historical `graph-schema-discovery@1.x` runs, feedback, learning candidates, workspace overlays, effective versions, and rollback targets remain readable and reproducible. Never rewrite those stored references to the canonical capability.

Legacy learning patches may continue to target this name when their base record targets `graph-schema-discovery`. New generalized learning should target `corpus-graph-schema-induction` and must pass the same human review and rollback controls. Semantic authority remains in Semantier core services; this compatibility asset does not select profiles, accept candidates, or promote graph authority.
