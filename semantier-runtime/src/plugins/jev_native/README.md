# Jev native execution-contract candidate

Default-disabled candidate. No runtime paid-dispatch tool, automatic installation, profile grants or production admission.

- register(ctx) records task ownership and a status-only CLI
- host.py reuses pinned real Hermes loader/profile/trust/task policy helpers
- native.py validates the immutable body through exact pinned upstream source and captures transport receipts
- durable.py owns file-backed one-shot operational evidence, not semantic authority
- service.py supervises worker.py with deadline/cancellation handling; execute_live always denies and execute_offline requires an explicit fake-HTTP fixture
- specimen.py provides one separately user-run qualification entry point, fixed to the previously exposed TTPC-V3-0034 request. It requires a terminal confirmation and user-selected existing credential profile, uses an ephemeral host profile, and stores its one-attempt fuse at one fixed experiment path outside that temporary home
- OPERATOR_FREEZE.json pins reviewed source content only. It is not an approval, paid permit, or runtime authority

Run scripts/qualify_jev_plugin_specimen.py without flags for a local, non-credential-reading dry-run. The --execute path is for the user to run after checking account tariff, sufficient existing credits, and auto-refill OFF. It cannot promise a provider-enforced billing cap and must not be run by an assistant handling the API key.

The six focused tests/test_jev_native_*.py files cover offline and mocked-external operator paths. See docs/research/logs/hermes-jev-live-readiness-v1/report.md for the exact handoff and current evidence. Prior offline qualification remains historical at docs/research/logs/hermes-jev-execution-v1/.

The upstream MIT copyright/license is retained in upstream/LICENSE. UPSTREAM.json binds its source revision. No new package was installed.
