# Windows reusable channel setup

`Build.ps1 -Channel channel1 -OutputDirectory <fresh-work-directory>` and
`Build.ps1 -Channel channel2 -OutputDirectory <fresh-work-directory>` build
unsigned `semantier-setup-ch1.exe` and `semantier-setup-ch2.exe` respectively.
The destination must be a fresh directory under `D:\repo\deliverables\work`;
the existing signing owner must separately sign and verify each release EXE.
The build embeds the public publisher certificate and branding, but no package
version, build ID, package URL, size or digest.

On an ordinary launch the selected channel's HTTPS origin and trusted publisher
identity are fixed. The installer reads `semantier-acceptance.json`, downloads
its hash-bound immutable build manifest, rechecks the channel record, and freezes
the manifest's installation object set. It downloads each object without
following redirects, checks length and SHA-256, requires Windows Authenticode to
report the pinned publisher as valid, and checks package identity, version and
bundle-member closure before calling Windows package deployment. It refuses a
newer installed version and does not modify certificate stores. A missing or
untrusted publisher certificate therefore fails clearly rather than changing OS
trust. Channel 1 attempts HTTP/3 then TCP; Channel 2 currently uses TCP while
its direct R2 HTTP/3 qualification is open. Transport success alone is not an
installed-update or HTTP/3 qualification.

`--inspect-metadata <before> <manifest> <after>` is a read-only test entry that
uses the same compiled resolver and channel origin with supplied fixture bytes;
it does not fetch, install or change trust. Normal interactive launches always
resolve the channel from the network. The Windows fixture test builds one EXE
per channel and exercises active-head changes, rollback and malformed metadata;
transport and package-closure controls are separate. These synthetic controls
and unsigned builds do not replace signing, public readback, Windows install,
upgrade/relaunch, profile-preservation or actual negotiated-protocol evidence.

The old version-pinned ZIP bootstrapper and trust-helper behavior are historical;
its immutable releases and receipts retain their original interpretation.
