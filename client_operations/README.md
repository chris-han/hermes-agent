# Client publication owner

Publisher, immutable artifact custody, release journal/targets and CH1 storage
were moved from the parent Runtime checkout at
`14d4230fbf771ea2fe9df26076bc7586a2839b09`. Edit this package, not the frozen
`semantier-runtime` export. Its composition manifest and bytes remain unchanged.

Runtime retains authenticated route integration, governed identity lookup,
feedback and service lifecycle in `agents.client_operations`. It consumes this
package directly, without old import shims. The existing contracts and SQL schema
move with their publisher; no schema or authorization semantics change.

Run local staging from the Hermes checkout:

```sh
python scripts/qualification/stage_local_native.py --help
python -m client_operations.ecs_endpoint --help
```

Staging verifies the frozen Runtime composition as payload evidence; it does not
import publisher code from that export. It emits artifact-only receipts and does
not assert installed update qualification. PostgreSQL uses the existing
`postgresql` extra; storage uses the existing HTTP server dependencies.

Architecture Fit: the requested source-owner move is the sole verified delta.
Reuse existing transport, fencing, custody, head-last readback and journal
contracts. Keep Runtime identity/authentication and the frozen export intact;
add no store, admission gate or new dependency. Qualification uses synthetic
local artifacts, isolated existing-PostgreSQL schemas and loopback storage.
It does not authorize or establish live ECS deployment/publication.

The source-pinned historical CI Runtime container probes in
`scripts/releases/semantier_native_ci.py` retain that immutable image's import
paths. They query the captured container rather than loading local publisher
source; replacing that image requires separate composition qualification.
