# Windows 0.0.64 manual installer

This separately versioned bootstrapper is pinned to the unchanged acceptance-family
0.0.64.0 ZIP, its six signed package objects, graph and public certificate. It is
not an updater and does not make a CI qualification claim. It refuses an installed
newer version, downloads from the fixed HTTPS origin, verifies the complete ZIP and
members, and registers the bundle as the initiating user. If trust is absent, the
same signed program is copied to a private, administrator-readable task directory;
the elevated helper verifies the inputs again, requests consent and imports only the
pinned public certificate to LocalMachine TrustedPeople. It does not trust a root CA.

The `TESTING` compilation exposes `--self-test` for disposable path-replacement and
partial-file cleanup checks; omit that symbol from the release build. These checks do
not perform a package installation, modify a certificate store, or establish alternate
administrator behavior. A separate real Windows environment is required for those
claims. The bootstrapper's own signing, hash, timestamp and source identity are
recorded separately from the immutable 0.0.64 package.

## Branded native installer (bootstrapper 1.1.0)

IA gate: GO_FOR_UI. The user opens the downloaded EXE to install Semantier for
their Windows account. Keep one primary action, visible release/track, truthful
download/verification/permission/installation/completion/error states and
recoverable technical details. The existing producer remains the installation
owner; the form never downloads, trusts or installs on construction or preview.

Adapt the existing desktop DESIGN.md and semantierTheme: embedded existing
assets/icon.png and icon.ico, flat neutral surface, yellow primary action, native
Segoe UI with Chinese font fallback, DPI scaling, keyboard primary action and
native progress. No web assets, online fonts or extra runtime dependency.

Read the initiating user's Windows display/UI culture (CurrentUICulture), not
region, keyboard or the elevated administrator's settings. Chinese cultures use
Simplified Chinese; English and unsupported languages use English. This release
does not claim Traditional Chinese or other translations. Pass the resolved
whitelisted language to the elevated helper so consent remains in the original
user's language. System-owned UAC/button labels remain controlled by Windows.

Build.ps1 compiles the existing installer plus InstallerUI.cs, embeds the existing
brand assets/public certificate and an asInvoker/DPI-aware manifest, exercises
the TESTING-only producer failure/completion message loops in both languages and
renders each UI phase. It does not install packages or mutate certificate stores.
It emits an unsigned semantier-setup.exe into a new task-owned output directory;
use the existing SignTool/certificate owner to sign and verify that separate EXE.
Keep TESTING out of the release. Existing .64 ZIP and graph remain pinned.


## Bootstrapper 1.1.1

Drain redirected PowerShell stdout and stderr concurrently. Progress records can fill
stderr even after deployment succeeds; sequential reads leave setup waiting forever.
The Windows self-test exercises a child emitting more than a pipe buffer on stderr,
checks stdout completion, and preserves nonzero-exit diagnostics. Application 0.0.64
and its immutable package graph remain unchanged.
