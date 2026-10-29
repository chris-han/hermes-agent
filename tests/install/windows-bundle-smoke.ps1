# Real deployment of receipt-verified bytes. Default requires disposable CI; the
# explicit Semantier switch permits only the separately owned acceptance family.
param(
    [Parameter(Mandatory=$true)][string]$Artifact,
    [Parameter(Mandatory=$true)][ValidateSet('x64','arm64')][string]$Arch,
    [Parameter(Mandatory=$true)][string]$Commit,
    [string]$Tag = '',
    [string]$ChannelRequest = '',
    [switch]$SemantierAcceptanceHost,
    [string]$NativeGraph = '',
    [string]$NativeLayoutPlan = '',
    [string]$Python = '',
    [string]$ExpectedSid = '',
    [Parameter(Mandatory=$true)][string]$Work,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ((!$SemantierAcceptanceHost -and ($env:GITHUB_ACTIONS -cne 'true' -or $env:RUNNER_ENVIRONMENT -cne 'github-hosted')) -or $env:OS -cne 'Windows_NT') {
    throw 'Disposable native GitHub-hosted Windows runner required'
}
if ($NativeGraph) {
    $actualSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    if (!$SemantierAcceptanceHost -or $env:GITHUB_ACTIONS -cne 'true' -or !$NativeLayoutPlan -or !$Python -or
        !$ExpectedSid -or $ExpectedSid -cne $actualSid -or $actualSid -in @('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011')) {
        throw 'Native graph smoke requires the separate explicitly bound CI account; normal/baseline contexts excluded'
    }
} elseif ($NativeLayoutPlan) { throw 'Native layout requires the complete graph' }
if ($Commit -cnotmatch '^[a-f0-9]{40}$') { throw 'Expected exact lowercase full commit SHA' }
if (-not [IO.Path]::IsPathRooted($Artifact) -or -not (Test-Path -LiteralPath $Artifact -PathType Leaf)) { throw 'Absolute artifact path required' }
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
$processors = @(Get-CimInstance Win32_Processor)
$nativeCode = @{ x64=9; arm64=12 }[$Arch]
if (-not $processors.Count -or @($processors | Where-Object { $_.Architecture -ne $nativeCode }).Count) {
    throw "Host is not native $Arch; emulated x64 on ARM is not coverage"
}
$Assets = Join-Path $PSScriptRoot 'e2e-assets'
$Node = (Get-Command node.exe -ErrorAction Stop).Source
$Metadata = Join-Path $Assets 'bundle-smoke-metadata.mjs'
function Run-Node([string[]]$Argv) {
    & $Node @Argv
    if ($LASTEXITCODE -ne 0) { throw "Node exited $LASTEXITCODE ($($Argv[0]))" }
}
$nodeArch = Run-Node @('-p', 'process.arch')
if ($nodeArch -cne $Arch) { throw 'Driver Node must also use the native architecture' }
$identityArgs = @('--commit', $Commit)
if ($Tag) { $identityArgs += @('--tag', $Tag) }
if ($ChannelRequest) { $identityArgs += @('--channel-request', $ChannelRequest) }
$Expected = (Run-Node (@($Metadata, 'identity') + $identityArgs)) | ConvertFrom-Json
if ($SemantierAcceptanceHost) {
    if (!$ChannelRequest -or $Expected.msixIdentity -cne 'Semantier.Agent.Acceptance' -or $Expected.publisher -cne 'CN=Semantier Technology, O=Semantier Technology') { throw 'Owned acceptance identity required' }
    $scope = Get-Content -Raw -LiteralPath $ChannelRequest | ConvertFrom-Json
    if ($scope.semantierScope -cne 'semantier-windows-x64/v1' -or $scope.channel -cne 'semantier-acceptance') { throw 'Acceptance scope required' }
}
Run-Node @($Metadata, 'prepare', '--work', $Work, '--out', $Out)
. (Join-Path $Assets 'windows-bundle-metadata.ps1')
. (Join-Path $Assets 'windows-bundle-plugin-smoke.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$HomeDir = Join-Path $Work 'home'
$UserData = Join-Path $Work 'user-data'
New-Item -ItemType Directory -Path $HomeDir, $UserData | Out-Null
Start-Transcript -Path (Join-Path $Out 'native-install.log') | Out-Null
$attempted = $false
$installedFullName = $null
$failed = $false
try {
    $beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant()
    @{ path=$Artifact; sha256=$beforeHash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact.json')
    $signature = Get-AuthenticodeSignature -LiteralPath $Artifact
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -cne $Expected.publisher) { throw 'Artifact signature/publisher invalid' }
    $externalPackage = $null
    $graph = $null
    if ($NativeGraph) {
        $graphHash = (Get-FileHash -LiteralPath $NativeGraph -Algorithm SHA256).Hash.ToLowerInvariant()
        $graph = Get-Content -LiteralPath $NativeGraph -Raw | ConvertFrom-Json
        $members = @($graph.objects)
        $main = @($members | Where-Object { $_.kind -ceq 'main-package' })
        $bundle = @($members | Where-Object { $_.key -ceq $graph.root })
        if ($main.Count -ne 1 -or $bundle.Count -ne 1 -or
            [IO.Path]::GetFileName($Artifact) -cne ($bundle[0].key -split '/')[-1] -or $beforeHash -cne $bundle[0].sha256) { throw 'Native graph root binding mismatch' }
        . (Join-Path $PSScriptRoot '..\..\apps\desktop\scripts\native-package-digest.ps1')
        $nativeDigests = [ordered]@{}
        foreach ($member in $members) {
            $filename = ($member.key -split '/')[-1]
            if ($filename -cnotmatch '^SemantierAgent(?:-main|-runtime[0-9]{4})?\.msix(?:bundle)?$') { throw 'Unexpected native member filename' }
            $file = Join-Path (Split-Path -Parent $Artifact) $filename
            $signature = Get-AuthenticodeSignature -LiteralPath $file
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $Expected.publisher) { throw 'Native member signature mismatch' }
            $nativeDigests[$filename] = Get-NativePackageDigest $file
        }
        $digestsPath = Join-Path $Out 'native-digests-readback.json'
        $nativeDigests | ConvertTo-Json | Set-Content -LiteralPath $digestsPath -Encoding utf8
        $readbackPath = Join-Path $Out 'native-graph-readback.json'
        & $Python (Join-Path $PSScriptRoot '..\..\scripts\releases\semantier_windows.py') native-graph --directory (Split-Path -Parent $Artifact) --plan $NativeLayoutPlan --digests $digestsPath --commit $Commit --output $readbackPath
        if ($LASTEXITCODE -ne 0) { throw 'Complete final native graph verification failed' }
        $readback = Get-Content -LiteralPath $readbackPath -Raw | ConvertFrom-Json
        if (($readback | ConvertTo-Json -Depth 16 -Compress) -cne ($graph | ConvertTo-Json -Depth 16 -Compress)) { throw 'Final native graph readback differs' }
        $externalPackage = $main[0]
        if (@(Get-AppxPackage -PackageTypeFilter Resource -Name $Expected.msixIdentity).Count) { throw 'Existing related resources conflict; preserve them' }
    }
    $inputMetadata = Read-BundleSmokeMetadata $Artifact $Arch $Expected $externalPackage
    $inputMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact-metadata.json')
    $existing = @(if ($SemantierAcceptanceHost) { Get-AppxPackage -Name $Expected.msixIdentity } else { Get-AppxPackage -AllUsers -Name $Expected.msixIdentity })
    if ($existing.Count) { throw 'Package identity already exists; refusing to replace it' }
    $attempted = $true
    Add-AppxPackage -Path $Artifact -ErrorAction Stop
    $packages = @(Get-AppxPackage -Name $Expected.msixIdentity)
    if ($packages.Count -ne 1) { throw 'Expected exactly one deployed package' }
    $pkg = $packages[0]
    $installedFullName = $pkg.PackageFullName
    if ($pkg.Publisher -cne $Expected.publisher -or $pkg.Architecture.ToString() -ine $Arch -or
        $pkg.Version.ToString() -cne $inputMetadata.Version -or $pkg.Status.ToString() -cne 'Ok') {
        throw 'Deployed package identity/version/native architecture/status mismatch'
    }
    $manifest = Get-AppxPackageManifest -Package $pkg.PackageFullName
    $installed = Read-SmokePackageManifest $manifest $Arch $Expected
    if ($installed.Version -cne $inputMetadata.Version) { throw 'Installed manifest version mismatch' }
    if ($inputMetadata.Executable -and $installed.Executable -cne $inputMetadata.Executable) { throw 'Installed executable differs from input manifest' }
    $exe = [IO.Path]::GetFullPath((Join-Path $pkg.InstallLocation $installed.Executable))
    if (-not $exe.StartsWith($pkg.InstallLocation.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Installed manifest executable is absent or outside package' }
    $resources = Join-Path (Split-Path -Parent $exe) 'resources'
    $root = Join-Path $resources 'agent-payload'
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'Installed agent-payload missing' }
    $stampPath = Join-Path $resources 'install-stamp.json'
    $stamp = Get-Content -Raw -LiteralPath $stampPath | ConvertFrom-Json
    $displayVersion = (Run-Node (@($Metadata, 'stamp', '--platform', 'win32', '--stamp', $stampPath) + $identityArgs)) | ConvertFrom-Json
    if ($ChannelRequest) {
        if ($pkg.Version.ToString() -cne $displayVersion) { throw 'Package version disagrees with channel request' }
    } elseif (-not $Tag) {
        if ($pkg.Version.ToString() -cne ($displayVersion + '.0')) { throw 'Commit package must use stamped semver with zero revision' }
    }
    # A tagged package carries the release-time native quad (msix-shared.mjs nativeQuad), not its semver.
    # Stable admission checks that quad against the claim epoch; here it must only match the input manifest (above).
    @{ packageFullName=$pkg.PackageFullName; publisher=$pkg.Publisher; architecture=$pkg.Architecture.ToString();
        version=$pkg.Version.ToString(); exe=$exe; root=$root; stamp=$stamp } |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Out 'installed-identity.json')
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant() -cne $beforeHash) { throw 'Input artifact changed during installation' }
    if ($graph) {
        Run-Node @($Metadata,'native-payload','--root',$pkg.InstallLocation,'--plan',$NativeLayoutPlan,'--graph',$NativeGraph,'--out',(Join-Path $Out 'native-payload-folding.json'))
        $resources = @(Get-AppxPackage -PackageTypeFilter Resource -Name $Expected.msixIdentity)
        $assets = @($graph.objects | Where-Object { $_.kind -ceq 'asset-package' })
        if ($resources.Count -ne $assets.Count) { throw 'Installed native resource set incomplete' }
        $installedMembers = @()
        foreach ($asset in $assets) {
            $matches = @($resources | Where-Object { $_.ResourceId -ceq $asset.resourceId -and $_.Version.ToString() -ceq $asset.version -and $_.PackageFamilyName -ceq $pkg.PackageFamilyName -and $_.Publisher -ceq $Expected.publisher -and $_.Status.ToString() -ceq 'Ok' -and $_.Architecture.ToString() -ieq $asset.arch -and $_.SignatureKind.ToString() -ceq 'Developer' -and !$_.IsDevelopmentMode })
            if ($matches.Count -ne 1) { throw 'Installed native resource identity mismatch' }
            $child = $matches[0]
            $installedMembers += @{full=$child.PackageFullName;family=$child.PackageFamilyName;name=$child.Name;publisher=$child.Publisher;version=$child.Version.ToString();arch=$child.Architecture.ToString();resource_id=$child.ResourceId}
        }
        @{producedAt=[DateTime]::UtcNow.ToString('o');graphSha256=$graphHash;related=$installedMembers;sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;nativeRemoteDigestEnforcement='NOT_RUN'} | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $Out 'native-installed-graph.json') -Encoding utf8
    }
    Push-Location $Work
    try {
        Run-Node @((Join-Path $Assets 'desktop-smoke.ts'), '--exe', $exe, '--root', $root, '--origin', 'bundled',
            '--home', $HomeDir, '--user-data', $UserData, '--out', $Out, '--phase', 'installed', '--expect-commit', $Commit)
    } finally { Pop-Location }
    if ($NativeGraph) { Test-BundleNativePayload $root $pkg $Commit $stamp.channelBuild.buildId $Out }
    # Plugins whose Python dependencies PM builds must work inside the package (#135236).
    Test-BundlePluginInstall $root $Out -IsolatedHome:$SemantierAcceptanceHost
    if ($NativeGraph -and (Get-FileHash -LiteralPath $NativeGraph -Algorithm SHA256).Hash.ToLowerInvariant() -cne $graphHash) { throw 'Native graph changed during qualification' }
} catch {
    $failed = $true
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-install-error.txt')
    Write-Error $_ -ErrorAction Continue
} finally {
    try {
        # A failed deployment can still have registered the package. Admission
        # ruled out prior identities, so only this invocation's exact package is eligible.
        if ($attempted) {
            $remaining = @(Get-AppxPackage -Name $Expected.msixIdentity)
            foreach ($pkg in $remaining) {
                if ($pkg.Publisher -cne $Expected.publisher -or $pkg.Version.ToString() -cne $inputMetadata.Version -or
                    ($installedFullName -and $pkg.PackageFullName -cne $installedFullName)) { throw 'Refusing cleanup of an unowned package' }
                # gateway ensure owns a detached process, so closing Electron is
                # insufficient. Only this fresh smoke home's live incarnation in
                # this exact installed payload is eligible for cleanup.
                foreach ($smokeHome in @($HomeDir, (Join-Path $UserData 'hermes-home'))) {
                    $lifecycle = Join-Path $smokeHome 'state\gateway.lifecycle.json'
                    if (-not (Test-Path -LiteralPath $lifecycle)) { continue }
                    $record = Get-Content -Raw -LiteralPath $lifecycle | ConvertFrom-Json
                    if ($record.phase -ne 'running') { continue }
                    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record.pid)"
                    if (-not $owner) { continue }
                    $created = ([DateTimeOffset]$owner.CreationDate).ToUnixTimeMilliseconds() / 1000.0
                    $payloadPrefix = (Join-Path $pkg.InstallLocation 'app\resources\agent-payload').TrimEnd('\') + '\'
                    if (-not $record.create_time -or [Math]::Abs($created - $record.create_time) -ge 2 -or
                        -not $owner.ExecutablePath.StartsWith($payloadPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                        throw 'Refusing cleanup of an unproven gateway incarnation'
                    }
                    Stop-Process -Id $owner.ProcessId -ErrorAction Stop
                    Wait-Process -Id $owner.ProcessId -Timeout 60 -ErrorAction SilentlyContinue
                    if (Get-CimInstance Win32_Process -Filter "ProcessId=$($owner.ProcessId)") { throw 'Owned gateway did not stop' }
                }
                Remove-AppxPackage -Package $pkg.PackageFullName -ErrorAction Stop
                if (Get-AppxPackage -Name $Expected.msixIdentity) { throw 'Package removal did not complete' }
                if ($NativeGraph -and @(Get-AppxPackage -PackageTypeFilter Resource -Name $Expected.msixIdentity).Count) { throw 'Related resources remain; preserve them and fail cleanup' }
            }
        }
    } catch {
        $failed = $true
        $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-cleanup-error.txt')
        Write-Error $_ -ErrorAction Continue
    } finally {
        @{ exitCode=[int]$failed } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'native-install-exit.json')
        Stop-Transcript | Out-Null
    }
}
if ($failed) { exit 1 }