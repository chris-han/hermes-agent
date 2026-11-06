# Semantier Windows native Setup

Authoritative source for both independently signed Setup executables. Uses the
Windows .NET Framework compiler, WinForms and native WinHTTP; no bundled .NET 8.

- CH0 `semantier-setup-ch0.exe`: `LOCAL_SETUP` build; installs `SemantierAgent.msix` next
  to the EXE, regardless of the current directory. If that filename is absent,
  accepts a single adjacent MSIX. No network requests. Missing or ambiguous
  fallback packages produce an error. Does not remove user profiles.
- CH1 `semantier-setup-ch1.exe`: online build; downloads the latest signed package
  from `https://download.semantier.cn/releases/SemantierAgent.msix` and checks its
  SHA-256 sidecar. Prefers HTTP/3 with HTTP/2 fallback and resumed transfers.

`Program.cs` owns both entry paths via compile-time selection; `WinHttp.cs` holds
transport and `InstallerWindow.cs` holds shared Asimov bilingual UI. The existing
PowerShell builder assembles these source fragments, embeds the canonical desktop
icon/logo, applies `setup.manifest`, compiles and signs a separate EXE.

From the repository root in native Windows PowerShell:

```powershell
# Build/sign offline CH0. Never publishes or installs the application.
.\scripts\deploy-ecs.ps1 -LocalSetup -BuildMode Prod

# Build/sign CH1 locally only.
.\scripts\deploy-ecs.ps1 -UpdateSetup -LocalOnly -BuildMode Prod

# Publish CH1 only when intended (not part of routine MSIX releases).
.\scripts\deploy-ecs.ps1 -SetupOnly -BuildMode Prod
```

Outputs and evidence stay under `D:\repo\deliverables`, outside this source
folder. No private keys or credentials belong here. The public-only `SemantierPublisher.cer` is embedded and pinned to thumbprint `57BF15739B4AA447842EAF388856A01B4E5029FF`. Windows Add-AppxPackage
performs package signature enforcement during installation for both channels.

This directory is excluded from sealed runtime snapshots and from Windows MSIX
payload resources, including when an older prepared payload contains its copy.
Do not copy Setup source or binaries into the app package.

Each installation attempt records timestamped progress and complete Windows errors in
`%LOCALAPPDATA%\Semantier\Setup\<run>\log.txt`. The failure screen’s View log button
opens that file in Notepad; it does not expand an embedded text frame.

On a new PC, Setup requests Windows administrator consent to import only the pinned
public publisher certificate into Local Computer / Trusted People. It never adds
a root CA. Package installation then runs as the original user, preserving their
profile. Declining UAC stops installation. This addresses MSIX error 0x800B0109.
