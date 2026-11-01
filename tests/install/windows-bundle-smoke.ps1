# Real deployment of receipt-verified bytes. Default requires disposable CI; the
# explicit Semantier switch permits only the separately owned acceptance family.
param(
    [string]$Artifact,
    [ValidateSet('x64','arm64')][string]$Arch,
    [string]$Commit,
    [string]$Tag = '',
    [string]$ChannelRequest = '',
    [switch]$SemantierAcceptanceHost,
    [string]$NativeGraph = '',
    [string]$NativeLayoutPlan = '',
    [string]$Python = '',
    [string]$ExpectedSid = '',
    [string]$Work,
    [string]$Out
)
function Invoke-CIPrivateBundleSmoke {
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
    $receiptKey = [Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')
    try {
        [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$null,'Process')
        & $Node @Argv
        if ($LASTEXITCODE -ne 0) { throw "Node exited $LASTEXITCODE ($($Argv[0]))" }
    } finally { [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$receiptKey,'Process') }
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
. (Join-Path $Assets 'windows-bundle-installed.ps1')
. (Join-Path $Assets 'windows-physical-path.ps1')
. (Join-Path $Assets 'windows-bundle-plugin-smoke.ps1')
. (Join-Path $Assets 'native-ci-cleanup-policy.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$HomeDir = Join-Path $Work 'home'
$UserData = Join-Path $Work 'user-data'
New-Item -ItemType Directory -Path $HomeDir, $UserData | Out-Null
Start-Transcript -Path (Join-Path $Out 'native-install.log') | Out-Null
$attempted = $false
$installedFullName = $null
$nativeUriState=@{unknown=$false}
$physicalRoots=@()
$verified=$null
$failed = $false
$layoutHash = ''; $graphHash = ''
try {
    $beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant()
    @{ path=$Artifact; sha256=$beforeHash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact.json')
    $signature = Get-AuthenticodeSignature -LiteralPath $Artifact
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -cne $Expected.publisher) { throw 'Artifact signature/publisher invalid' }
    $externalPackage = $null
    $graph = $null
    if ($NativeGraph) {
        $layoutHash = (Get-FileHash -LiteralPath $NativeLayoutPlan -Algorithm SHA256).Hash.ToLowerInvariant()
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
        if (@(Get-AppxPackage -PackageTypeFilter Optional -Name ($Expected.msixIdentity+'.*')).Count) { throw 'Existing related optional package conflict; preserve it' }
    }
    $inputMetadata = Read-BundleSmokeMetadata $Artifact $Arch $Expected $externalPackage
    $inputMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact-metadata.json')
    $existing = @(if ($SemantierAcceptanceHost) { Get-AppxPackage -Name $Expected.msixIdentity } else { Get-AppxPackage -AllUsers -Name $Expected.msixIdentity })
    if ($existing.Count) { throw 'Package identity already exists; refusing to replace it' }
    $attempted = $true
    if($graph){
        . (Join-Path $Assets 'native-package-deployment.ps1')
        Invoke-CINativeGraphUri $graph $NativeGraph $Artifact $Node $Out $ExpectedSid $nativeUriState $Commit
    }else{Add-AppxPackage -Path $Artifact -ErrorAction Stop}
    $capturedPackages = @(Get-AppxPackage -Name $Expected.msixIdentity)
    if ($capturedPackages.Count -ne 1) { throw 'Expected exactly one deployed package' }
    $installedFullName = $capturedPackages[0].PackageFullName
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant() -cne $beforeHash) { throw 'Input artifact changed during installation' }
    if($graph){
        $bound=Get-CIBoundInstalledVerification -Arch $Arch -Expected $Expected -InputMetadata $inputMetadata -ChannelRequest $ChannelRequest -IdentityArgs $identityArgs -Metadata $Metadata -Tag $Tag -Graph $graph -NativeLayoutPlan $NativeLayoutPlan -NativeGraph $NativeGraph -GraphHash $graphHash -LayoutHash $layoutHash -ExpectedSid $ExpectedSid
        $physicalRoots=$bound.roots; $verified=$bound.verification
    }else{
        $verified=Get-BundleInstalledVerification -Arch $Arch -Expected $Expected -InputMetadata $inputMetadata -ChannelRequest $ChannelRequest -IdentityArgs $identityArgs -Metadata $Metadata -Tag $Tag
    }
    $pkg = $verified.package
    if ($pkg.PackageFullName -cne $installedFullName) { throw 'Installed package changed during verification' }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant() -cne $beforeHash) { throw 'Input artifact changed during installed verification' }
    $exe = $verified.identity.exe; $root = $verified.identity.root; $stamp = $verified.identity.stamp
    $verified.identity | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Out 'installed-identity.json')
    if ($graph) {
        $verified.folding | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Out 'native-payload-folding.json') -Encoding utf8
        $verified.registration.nativeRemoteDigestEnforcement = 'PASS'
        $verified.registration | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $Out 'native-installed-graph.json') -Encoding utf8
    }
    Invoke-BundleInstalledQualification -Assets $Assets -Work $Work -Exe $exe -Root $root -HomeDir $HomeDir -UserData $UserData -Out $Out -Commit $Commit -NativeGraph $NativeGraph -Package $pkg -Stamp $stamp -IsolatedHome:$SemantierAcceptanceHost
    if ($NativeGraph -and ((Get-FileHash -LiteralPath $NativeGraph -Algorithm SHA256).Hash.ToLowerInvariant() -cne $graphHash -or (Get-FileHash -LiteralPath $NativeLayoutPlan -Algorithm SHA256).Hash.ToLowerInvariant() -cne $layoutHash)) { throw 'Native graph or layout changed during qualification' }
} catch {
    $failed = $true
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-install-error.txt')
    Write-Error $_ -ErrorAction Continue
} finally {
    try {
        # A failed deployment can still have registered the package. Admission
        # ruled out prior identities, so only this invocation's exact package is eligible.
        if ($nativeUriState.unknown) { throw 'Native URI UNKNOWN hold; preserve registered resources and origin' }
        if ($attempted) {
            $remaining = @(Get-AppxPackage -Name $Expected.msixIdentity)
            foreach ($pkg in $remaining) {
                if($graph){
                    Assert-CINativeCleanupOwnership $graph $installedFullName $nativeUriState $verified $pkg $ExpectedSid $NativeGraph $Commit $physicalRoots $Out
                }else{Assert-CINativeCleanupOwnership $null $installedFullName}
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
                    $payloadPrefix = (Join-Path $pkg.InstallLocation 'app\resources\agent-payload').TrimEnd('\') + '\'
                    Stop-CIOwnedGateway $record $payloadPrefix
                }
                if($graph){
                    Invoke-CINativeOwnedRemoval $graph $installedFullName $nativeUriState $verified $pkg $ExpectedSid $NativeGraph $Commit ([ref]$physicalRoots) $Out
                } else {
                    Remove-AppxPackage -Package $installedFullName -ErrorAction Stop
                    if (Get-AppxPackage -Name $Expected.msixIdentity) { throw 'Package removal did not complete' }
                }
            }
        }
    } catch {
        $failed = $true
        $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-cleanup-error.txt')
        Write-Error $_ -ErrorAction Continue
    } finally {
        foreach($heldRoot in $physicalRoots){$heldRoot.root.handle.Dispose()}
        @{ exitCode=[int]$failed } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'native-install-exit.json')
        Stop-Transcript | Out-Null
    }
}
if ($failed) { throw 'Bundle smoke failed; see native-install-exit.json' }
}
if ($MyInvocation.InvocationName -ne '.') {
    if ($NativeGraph) { throw 'Native graph smoke requires coordinator-owned private dispatch' }
    Invoke-CIPrivateBundleSmoke @PSBoundParameters
}
