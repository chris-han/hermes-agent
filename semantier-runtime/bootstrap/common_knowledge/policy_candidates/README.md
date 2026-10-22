# Governed Policy Candidate Seeds

This directory is the checked-in source of truth for repo-owned governed
`policy_candidates` seed files.

Rules:

- Keep active seed artifacts here as YAML files under tier/domain folders.
- Loader logic lives in `src/`; runtime state lives in `.semantier-home/eos.db`.
- Top-level `policies/` remains reserved for executable embedded `.rego`
  policies. Do not mix governed candidate YAML files into that directory.
- Bootstrap must deterministically load these files into `policy_candidates`.

Suggested layout:

- `t5-preference/<domain>/*.yaml`
- `t4-policy/<domain>/*.yaml`
- `t3-doctrine/<domain>/*.yaml` only when the artifact is intentionally seeded
  through the candidate store rather than KGL CSV/bootstrap flows

Each file must contain:

- `name`
- `category`
- `semantic_tier`
- `content`
- activation metadata appropriate to the tier

For `T5` files also include:

- `scoped_owner`
- `rollback_window_hours`
- `rollback_strategy`
