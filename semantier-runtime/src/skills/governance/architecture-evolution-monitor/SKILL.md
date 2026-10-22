---
name: architecture-evolution-monitor
description: Governed architecture monitoring skill for deterministic metric assessment, institutional review, and bounded handoff requests.
author: Semantier Runtime
version: 1.0.0
category: governance
triggers:
  - architecture metrics
  - architecture review
  - AGE decision gate
tags:
  - governance
  - architecture
  - monitoring
metadata:
  hermes:
    source_tier: application
    governance: governed
    immutable: true
---

# Architecture Evolution Monitor

Use this skill when Semantier architecture or knowledge-runtime metrics need a deterministic advisory assessment.

The skill may:

- validate `architecture_metric_snapshot.v1` inputs;
- select active governed threshold profiles by scope and half-open effective intervals;
- produce deterministic indicator assessments and learning-loop classifications;
- create evidence-backed `architecture_review_package.v1` records;
- request P1 benchmarks, ADR preparation, or coding-agent analysis only after an accountable review decision.

The skill must not:

- modify canonical or derived specifications;
- migrate stores or provision infrastructure;
- activate architecture or refactor code automatically;
- treat LLM recommendations as deterministic assessment or promotion policy;
- copy restricted payloads into review packages when safe references and hashes are available.

Governed contracts are implemented by `semantier.architecture_evolution_monitor`.
