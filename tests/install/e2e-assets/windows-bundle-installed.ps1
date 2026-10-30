# Installed-package verification and remaining qualification shared by callers.
# Callers must independently validate and bind the signed input graph, layout,
# actor, selected tools and physical custody before treating this as evidence.
function Invoke-BundleVerificationNode([string[]]$Argv) {
 $receiptKey=[Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')
 try {
  [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$null,'Process')
  return (Run-Node $Argv)
 } finally { [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$receiptKey,'Process') }
}

# Execute only after the caller has freshly verified the installed candidate
# and established authority to run the remaining same-run qualification.
function Invoke-BundleInstalledQualification {
 param([string]$Assets,[string]$Work,[string]$Exe,[string]$Root,[string]$HomeDir,
       [string]$UserData,[string]$Out,[string]$Commit,[string]$NativeGraph,
       $Package,$Stamp,[switch]$IsolatedHome)
 $receiptKey=[Environment]::GetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY','Process')
 try {
  [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$null,'Process')
  Push-Location $Work
  try {
   Run-Node @((Join-Path $Assets 'desktop-smoke.ts'), '--exe', $Exe, '--root', $Root, '--origin', 'bundled',
    '--home', $HomeDir, '--user-data', $UserData, '--out', $Out, '--phase', 'installed', '--expect-commit', $Commit)
  } finally { Pop-Location }
  if($NativeGraph){Test-BundleNativePayload $Root $Package $Commit $Stamp.channelBuild.buildId $Out}
  Test-BundlePluginInstall $Root $Out -IsolatedHome:$IsolatedHome
 } finally { [Environment]::SetEnvironmentVariable('SEMANTIER_CLIENT_RECEIPT_KEY',$receiptKey,'Process') }
}

function Assert-BundleWinRTReady($Package,[string]$Sid) {
 $snapshot=Get-CIWinRTPackageStatus $Sid $Package.PackageFullName
 if($snapshot.presence -cne 'PRESENT' -or $snapshot.identity.full -cne $Package.PackageFullName -or
    $snapshot.identity.family -cne $Package.PackageFamilyName -or $snapshot.identity.name -cne $Package.Name -or
    $snapshot.identity.publisher -cne $Package.Publisher -or $snapshot.identity.version -cne $Package.Version.ToString() -or
    $snapshot.identity.architecture -ine $Package.Architecture.ToString() -or $snapshot.identity.resourceId -cne $Package.ResourceId){throw 'WinRT installed identity mismatch or unavailable'}
 if($snapshot.installedLocation.state -cne 'VALUE' -or !$snapshot.installedLocation.path -or !$Package.InstallLocation -or
    [IO.Path]::GetFullPath($snapshot.installedLocation.path).TrimEnd('\') -ine [IO.Path]::GetFullPath($Package.InstallLocation).TrimEnd('\')){throw 'WinRT installed location mismatch or unavailable'}
 foreach($name in @('DeploymentInProgress','Servicing','IsPartiallyStaged','VerifyIsOK')){
  if(!$snapshot.status.ContainsKey($name) -or $snapshot.status[$name].available -ne $true -or $snapshot.status[$name].value -isnot [bool]){throw 'WinRT installed status unavailable'}
 }
 if($snapshot.status.DeploymentInProgress.value -or $snapshot.status.Servicing.value -or $snapshot.status.IsPartiallyStaged.value -or -not $snapshot.status.VerifyIsOK.value){throw 'WinRT installed package not ready'}
 return $snapshot
}

function Get-BundleInstalledVerification {
 param(
  [string]$Arch, $Expected, $InputMetadata, [string]$ChannelRequest,
  [string[]]$IdentityArgs, [string]$Metadata, [string]$Tag,
  $Graph=$null, [string]$NativeLayoutPlan='', [string]$NativeGraph='',
  [string]$GraphHash='', [string]$LayoutHash='', [string]$ExpectedSid=''
 )
 $sid=''
 if($Graph){
  # Caller still owns signed-graph/applicability and WTS authorization.
  # Refuse unsupported scope before package reads or verifier execution.
  $sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
  if(!$ExpectedSid -or $ExpectedSid -cne $sid){throw 'Native verification actor mismatch'}
  if(@($Graph.objects|Where-Object{$_.kind -cnotin @('main-bundle','main-package','asset-package')}).Count){throw 'Native package applicability not established for this graph'}
  $main=@($Graph.objects|Where-Object{$_.kind -ceq 'main-package'})
  if($main.Count -ne 1 -or $main[0].identity -cne $Expected.msixIdentity -or $main[0].publisher -cne $Expected.publisher -or
     $main[0].version -cne $InputMetadata.Version -or $main[0].arch -ine $Arch){throw 'Native graph main identity mismatch'}
 }
 $packages=@(Get-AppxPackage -Name $Expected.msixIdentity)
 if($packages.Count -ne 1){throw 'Expected exactly one deployed package'}
 $pkg=$packages[0]
 if($pkg.Publisher -cne $Expected.publisher -or $pkg.Architecture.ToString() -ine $Arch -or
    $pkg.Version.ToString() -cne $InputMetadata.Version -or $pkg.Status.ToString() -cne 'Ok'){
  throw 'Deployed package identity/version/native architecture/status mismatch'
 }
 $manifest=Get-AppxPackageManifest -Package $pkg.PackageFullName
 $installed=Read-SmokePackageManifest $manifest $Arch $Expected
 if($installed.Version -cne $InputMetadata.Version){throw 'Installed manifest version mismatch'}
 if($InputMetadata.Executable -and $installed.Executable -cne $InputMetadata.Executable){throw 'Installed executable differs from input manifest'}
 $exe=[IO.Path]::GetFullPath((Join-Path $pkg.InstallLocation $installed.Executable))
 if(-not $exe.StartsWith($pkg.InstallLocation.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath $exe -PathType Leaf)){throw 'Installed manifest executable is absent or outside package'}
 $resources=Join-Path (Split-Path -Parent $exe) 'resources'
 $root=Join-Path $resources 'agent-payload'
 if(-not (Test-Path -LiteralPath $root -PathType Container)){throw 'Installed agent-payload missing'}
 $stampPath=Join-Path $resources 'install-stamp.json'
 $stamp=Get-Content -Raw -LiteralPath $stampPath|ConvertFrom-Json
 $displayVersion=(Invoke-BundleVerificationNode (@($Metadata,'stamp','--platform','win32','--stamp',$stampPath)+$IdentityArgs))|ConvertFrom-Json
 if($ChannelRequest){if($pkg.Version.ToString() -cne $displayVersion){throw 'Package version disagrees with channel request'}}
 elseif(-not $Tag){if($pkg.Version.ToString() -cne ($displayVersion+'.0')){throw 'Commit package must use stamped semver with zero revision'}}
 $identity=@{packageFullName=$pkg.PackageFullName;publisher=$pkg.Publisher;architecture=$pkg.Architecture.ToString();version=$pkg.Version.ToString();exe=$exe;root=$root;stamp=$stamp}
 $folding=$null;$registration=$null;$statusSnapshots=@()
 if($Graph){
  $statusSnapshots+=Assert-BundleWinRTReady $pkg $sid
  $beforeMainFull=[string]$pkg.PackageFullName
  $beforeMainPath=[IO.Path]::GetFullPath($pkg.InstallLocation).TrimEnd('\')
  $related=@(Get-AppxPackage -PackageTypeFilter Resource -Name $Expected.msixIdentity)
  $assets=@($Graph.objects|Where-Object{$_.kind -ceq 'asset-package'})
  if($related.Count -ne $assets.Count){throw 'Installed native resource set incomplete'}
  $members=@();$beforePaths=@{}
  foreach($asset in $assets){
   if($asset.identity -cne $Expected.msixIdentity -or $asset.publisher -cne $Expected.publisher -or $asset.version -cne $InputMetadata.Version -or !$asset.resourceId){throw 'Native graph asset identity mismatch'}
   $matches=@($related|Where-Object{$_.ResourceId -ceq $asset.resourceId -and $_.Version.ToString() -ceq $asset.version -and $_.PackageFamilyName -ceq $pkg.PackageFamilyName -and $_.Publisher -ceq $Expected.publisher -and $_.Status.ToString() -ceq 'Ok' -and $_.Architecture.ToString() -ieq $asset.arch -and $_.SignatureKind.ToString() -ceq 'Developer' -and !$_.IsDevelopmentMode})
   if($matches.Count -ne 1){throw 'Installed native resource identity mismatch'}
   $child=$matches[0]
   $statusSnapshots+=Assert-BundleWinRTReady $child $sid
   $beforePaths[$child.PackageFullName]=[IO.Path]::GetFullPath($child.InstallLocation).TrimEnd('\')
   $members+=@{full=$child.PackageFullName;family=$child.PackageFamilyName;name=$child.Name;publisher=$child.Publisher;version=$child.Version.ToString();arch=$child.Architecture.ToString();resource_id=$child.ResourceId}
  }
  $folding=(Invoke-BundleVerificationNode @($Metadata,'native-payload','--root',$pkg.InstallLocation,'--plan',$NativeLayoutPlan,'--graph',$NativeGraph))|ConvertFrom-Json
  $afterMain=@(Get-AppxPackage -Name $Expected.msixIdentity)
  $afterRelated=@(Get-AppxPackage -PackageTypeFilter Resource -Name $Expected.msixIdentity)
  if($afterMain.Count -ne 1 -or $afterMain[0].PackageFullName -cne $beforeMainFull -or
     $afterMain[0].Publisher -cne $Expected.publisher -or $afterMain[0].Version.ToString() -cne $InputMetadata.Version -or
     $afterMain[0].Architecture.ToString() -ine $Arch -or
     [IO.Path]::GetFullPath($afterMain[0].InstallLocation).TrimEnd('\') -ine $beforeMainPath -or
     $afterRelated.Count -ne $assets.Count){throw 'Installed package graph changed during content verification'}
  $afterSnapshots=@(Assert-BundleWinRTReady $afterMain[0] $sid)
  foreach($asset in $assets){
   $matches=@($afterRelated|Where-Object{$_.ResourceId -ceq $asset.resourceId -and $_.Version.ToString() -ceq $asset.version -and $_.PackageFamilyName -ceq $afterMain[0].PackageFamilyName -and $_.Publisher -ceq $Expected.publisher -and $_.Status.ToString() -ceq 'Ok' -and $_.Architecture.ToString() -ieq $asset.arch -and $_.SignatureKind.ToString() -ceq 'Developer' -and !$_.IsDevelopmentMode})
   $before=@($members|Where-Object{$_.resource_id -ceq $asset.resourceId})
   if($matches.Count -ne 1 -or $before.Count -ne 1 -or $matches[0].PackageFullName -cne $before[0].full -or
      [IO.Path]::GetFullPath($matches[0].InstallLocation).TrimEnd('\') -ine $beforePaths[$before[0].full]){throw 'Installed native resource changed during content verification'}
   $afterSnapshots+=Assert-BundleWinRTReady $matches[0] $sid
  }
  $registration=@{producedAt=[DateTime]::UtcNow.ToString('o');graphSha256=$GraphHash;layoutPlanSha256=$LayoutHash;related=$members;sid=$sid;statusBefore=$statusSnapshots;statusSnapshots=$afterSnapshots;scope='bounded snapshots around payload verification; not atomic'}
 }
 return @{package=$pkg;identity=$identity;folding=$folding;registration=$registration}
}
