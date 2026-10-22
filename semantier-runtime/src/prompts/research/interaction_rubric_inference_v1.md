# Interaction-Derived Rubric Inference v1

You receive non-authoritative evidence from one workflow-correction episode. The evidence may include
a source workflow, an edited workflow, a semantic delta, user chat, and/or paired execution evidence.

Infer a candidate specification only. Do not treat the evidence as policy, admission, activation, or
runtime authority.

Return one JSON object matching the supplied RubricDeltaCandidate schema.

Rules:
- Produce exactly one `primary_criterion`.
- `secondary_criteria` may be empty and are non-gating observations only.
- Do not guess missing scope, applicability, modality, or semantics. Use `UNSPECIFIED` /
  `UNRESOLVED` where the schema permits it.
- Distinguish required behavior from forbidden behavior, preferences, defaults, and one-off
  exceptions.
- Distinguish scope (instance/user/team/workspace/organization) from authority.
- Use only evidence supplied in the packet.
- Never recommend admission, activation, promotion, or workflow mutation.
- Output JSON only. No prose outside the JSON object.
