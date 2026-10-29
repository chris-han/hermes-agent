import { randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import path from 'node:path'

import type { App } from 'electron'

import { AppInstallerStrategy, type AppInstallerStrategyDeps } from './app-installer'
import { downloadPinnedArtifact } from './artifact'
import type { ChannelTarget } from './channel'
import { prepareChannelArtifactCache } from './channel-artifact-cache'
import { inspectRunningChannelApp } from './channel-native'
import { type NativeCommandResult, runChannelPowerShell } from './channel-native'
import { type ChannelBuild, decodeNativePackageGraph } from './channel-protocol'
import { registerUpdateRelaunch } from './relaunch'
import { startUpdateRelaunchWaiter } from './relaunch-waiter'

/** Verify downloaded bytes and native metadata before App Installer can stop the backend. */
export async function verifyPreparedChannelInstaller(
  file: string,
  target: ChannelTarget,
  command: (script: string, input: string) => Promise<NativeCommandResult> = runChannelPowerShell,
  fetchArtifact: typeof fetch = fetch
): Promise<string | void> {
  const pkg = target.package

  if (pkg.platform !== 'win32' || !pkg.publisher) {
    throw new Error('Expected a publisher-bound Windows package')
  }

  const directory: string = pkg.artifact.chunks
    ? await prepareChannelArtifactCache(path.dirname(file), {
        manifestSha256: target.manifestSha256,
        artifactSha256: pkg.artifact.sha256,
        buildId: target.manifest.request.buildId,
        version: pkg.version,
        identity: pkg.identity,
        publisher: pkg.publisher
      })
    : await mkdtemp(path.join(path.dirname(file), '.channel-artifact-'))

  try {
    const artifact: string = await downloadPinnedArtifact(
      directory,
      {
        url: target.artifactUrl,
        chunks: pkg.artifact.chunks,
        chunkBase: target.manifest.request.publicBase,
        sha256: pkg.artifact.sha256,
        size: pkg.artifact.size,
        format: pkg.artifact.key.endsWith('.msixbundle') ? 'msixbundle' : 'msix'
      },
      fetchArtifact
    )

    await command(
      String.raw`
$ErrorActionPreference='Stop'
$p=[Console]::In.ReadToEnd() | ConvertFrom-Json
function Read-SafeXml($stream) {
  $settings=[Xml.XmlReaderSettings]::new(); $settings.DtdProcessing=[Xml.DtdProcessing]::Prohibit; $settings.XmlResolver=$null
  $reader=[Xml.XmlReader]::Create($stream,$settings)
  try { $xml=[Xml.XmlDocument]::new(); $xml.XmlResolver=$null; $xml.Load($reader); return ,$xml } finally { $reader.Dispose() }
}
$stream=[IO.File]::OpenRead($p.file)
try { $xml=Read-SafeXml $stream } finally { $stream.Dispose() }
$root=$xml.DocumentElement
if ($root.LocalName -cne 'AppInstaller' -or $root.GetAttribute('Uri') -cne $p.feedUrl -or $root.GetAttribute('Version') -cne $p.version) { throw 'Channel descriptor binding mismatch' }
$main=@($root.ChildNodes | Where-Object { $_.LocalName -in @('MainPackage','MainBundle') })
if ($main.Count -ne 1 -or $main[0].GetAttribute('Name') -cne $p.identity -or $main[0].GetAttribute('Publisher') -cne $p.publisher -or $main[0].GetAttribute('Version') -cne $p.version -or $main[0].GetAttribute('Uri') -cne $p.artifactUrl) { throw 'Descriptor does not reference the pinned native package' }
if (@($root.ChildNodes | Where-Object { $_.LocalName -notin @('MainPackage','MainBundle') }).Count) { throw 'Pinned channel descriptor must not add dependencies, optional packages or background update policies' }
$signature=Get-AuthenticodeSignature -LiteralPath $p.artifact
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $p.publisher) { throw 'Native artifact signature/publisher mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead($p.artifact)
try {
  $name=if ($p.artifact.EndsWith('.msixbundle')) { 'AppxMetadata/AppxBundleManifest.xml' } else { 'AppxManifest.xml' }
  $entry=$zip.GetEntry($name)
  if (-not $entry) { throw 'Native manifest missing' }
  $stream=$entry.Open()
  try { $native=Read-SafeXml $stream } finally { $stream.Dispose() }
  if ($name -ceq 'AppxManifest.xml') { $identity=$native.Package.Identity; if ($identity.ProcessorArchitecture -cne $p.arch) { throw 'Package architecture mismatch' } }
  else { $identity=$native.Bundle.Identity; $slices=@($native.Bundle.Packages.Package | Where-Object { $_.Type -ceq 'application' -and $_.Architecture -ceq $p.arch -and $_.Version -ceq $p.version }); if ($slices.Count -ne 1) { throw 'Bundle architecture/version mismatch' } }
  if ($identity.Name -cne $p.identity -or $identity.Publisher -cne $p.publisher -or $identity.Version -cne $p.version) { throw 'Native manifest binding mismatch' }
} finally { $zip.Dispose() }
`,
      JSON.stringify({
        file,
        artifact,
        feedUrl: target.feedUrl,
        artifactUrl: target.artifactUrl,
        identity: pkg.identity,
        publisher: pkg.publisher,
        version: pkg.version,
        arch: pkg.arch
      })
    )

    // Keep exact verified bytes alive for Windows' local-package handoff.
    if (pkg.artifact.chunks) {
      await rm(path.join(directory, `chunks-${pkg.artifact.sha256}`), { recursive: true, force: true })

      return artifact
    }
  } finally {
    if (!pkg.artifact.chunks) {
      await rm(directory, { recursive: true, force: true })
    }
  }
}

/** Query the current user's OS package, never infer support from an OS version string. */
export async function inspectNativePackageUriEligibility(
  fullName: string,
  command: (script: string, input: string) => Promise<NativeCommandResult> = runChannelPowerShell
): Promise<{ supported: boolean; reason: string }> {
  const result = await command(
    String.raw`
$ErrorActionPreference='Stop'
$p=[Console]::In.ReadToEnd()|ConvertFrom-Json
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Management.Deployment.PackageManager,Windows.Management.Deployment,ContentType=WindowsRuntime]
$null=[Windows.Foundation.Metadata.ApiInformation,Windows.Foundation,ContentType=WindowsRuntime]
$api=[Windows.Foundation.Metadata.ApiInformation]::IsMethodPresent('Windows.Management.Deployment.PackageManager','AddPackageByUriAsync') -and [Windows.Foundation.Metadata.ApiInformation]::IsPropertyPresent('Windows.Management.Deployment.AddPackageOptions','ExpectedDigests')
if(!$api){@{supported=$false;reason='native-api-unavailable'}|ConvertTo-Json -Compress;exit 0}
$manager=[Windows.Management.Deployment.PackageManager]::new()
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$package=$manager.FindPackageForUser($sid,$p.fullName)
if(!$package -or $package.Id.FullName -cne $p.fullName){throw 'Current-user package unavailable'}
$info=$package.GetAppInstallerInfo()
if($info -and $info.Uri){@{supported=$false;reason='appinstaller-association-migration-unqualified'}|ConvertTo-Json -Compress;exit 0}
@{supported=$true;reason='native-uri-unassociated-package'}|ConvertTo-Json -Compress
`,
    JSON.stringify({ fullName })
  )

  const value: { supported?: boolean; reason?: string } = JSON.parse(result.stdout)

  if (typeof value.supported !== 'boolean' || !value.reason || !/^[a-z-]+$/.test(value.reason))
    {throw new Error('Invalid native capability observation')}

  return { supported: value.supported, reason: value.reason }
}

/** Reuse the packaged handoff while the external waiter owns native deployment. */
export function createNativeUriAppInstallerStrategy(
  deps: AppInstallerStrategyDeps,
  target: ChannelTarget,
  build: ChannelBuild,
  context: {
    app: Pick<App, 'getPath'>
    resourcesPath: string
    log: (message: string) => void
    inspect?: typeof inspectRunningChannelApp
    eligibility?: typeof inspectNativePackageUriEligibility
    startWaiter?: typeof startUpdateRelaunchWaiter
  }
): AppInstallerStrategy {
  return new AppInstallerStrategy({
    ...deps,
    nativeUri: { version: target.package.version },
    registerPendingRelaunch: async fromVersion => {
      // Recheck the current actor's installed package before handing off.

      const installed = await (context.inspect ?? inspectRunningChannelApp)(build)

      if (
        !installed.packageFullName ||
        !installed.packageFamilyName ||
        !installed.applicationId ||
        !target.package.publisher
      )
        {throw new Error('Native package identity unavailable')}

      const eligible = await (context.eligibility ?? inspectNativePackageUriEligibility)(installed.packageFullName)

      if (!eligible.supported) {throw new Error(`Native update unavailable: ${eligible.reason}`)}

      const graph = decodeNativePackageGraph(
        JSON.stringify(target.nativeGraph),
        target.manifest.request,
        target.package
      )

      const updates = path.join(context.app.getPath('userData'), 'updates')
      await mkdir(updates, { recursive: true })

      return registerUpdateRelaunch(context.app, fromVersion, {
        relaunch: () =>
          (context.startWaiter ?? startUpdateRelaunchWaiter)(
            installed.identity,
            context.resourcesPath,
            context.log,
            1860,
            {
              schema: 1,
              base: target.manifest.request.publicBase,
              graph,
              manifestSha256: target.manifestSha256,
              publisher: target.package.publisher!,
              identity: installed.identity,
              version: target.package.version,
              currentVersion: installed.nativeVersion,
              targetBuildId: target.manifest.request.buildId,
              targetCommit: target.manifest.request.commit,
              fullName: installed.packageFullName!,
              familyName: installed.packageFamilyName!,
              applicationId: installed.applicationId!,
              executable: installed.executable,
              journalDirectory: path.join(updates, `native-${target.manifestSha256}-${randomUUID().replaceAll('-', '')}`)
            }
          )
      })
    }
  })
}
