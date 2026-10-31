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
