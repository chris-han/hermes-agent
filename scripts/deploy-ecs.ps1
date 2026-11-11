#Requires -Version 7.6
<#
.SYNOPSIS
  Build an exact -Commit, sign, locally accept and publish one MSIX to ECS.
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -WhatIf
.EXAMPLE
  .\scripts\deploy-ecs.ps1
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -Output 'D:\repo\deliverables\work\ecs-deploy-20261105-221733-b7667f08'
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -NewRun
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -SetupOnly
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -UpdateSetup
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -VerifyPublicDownload
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -Commit eede5238816caa515089f99c558964fe6adec2e9
.EXAMPLE
  .\scripts\deploy-ecs.ps1 -Latest -BuildMode Prod -VerifyPublicDownload
.EXAMPLE
  pwsh.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\repo\semantier-runtime\hermes-agent\scripts\deploy-ecs.ps1' -WhatIf
  If Windows blocks this unsigned network-source script before it starts, use this process-scoped launch.
.DESCRIPTION
  BuildMode defaults to Beta: sign, install and publish without release verification.
  Use -BuildMode Prod for full verification and acceptance. Timings use checkpoints.jsonl.
  Run in an interactive Windows PowerShell session as the installing user.
  Native SSH/SFTP prefer environment variable ALI_ECS_SSH_PASSWORD, then an explicit -CredentialsEnv,
  otherwise use your SSH key/agent or prompt for the ECS password. Secrets
  are never embedded in this script. No npm/uv install, Python publisher, WSL,
  release lock/database, split packages, bundles, ZIPs, or CH2/R2 operations.
  -BuildOnly with -Commit stops after the unsigned package build: no signing, install or publication.
  -BuildSource must equal <selected Output>\source; detached worktrees share Hermes objects.
  -Latest commits Runtime changes, merges/pushes GitHub main and syncs pinned Hermes before building.
  Dirty Hermes changes are committed and pushed, then require separate qualification before the gitlink advances. Git conflicts stop without discarding work.
  -Latest -WhatIf is read-only. If syncing changes this script, re-run it to load the new operator.
  -LocalSetup builds a signed semantier-setup-ch0.exe for the single adjacent MSIX; never uploads.
  -SetupOnly builds/reuses and publishes Setup in its own ecs-setup-* run.
  -UpdateSetup forces a new Setup build; neither mode builds or installs MSIX.
  Routine MSIX releases require an already compatible Setup on ECS.
  A signed Setup EXE downloads a single fixed MSIX over HTTP/3, with no
  current-release manifest or catalog. Existing Windows profile is preserved.
  A local result.json is evidence of this invocation, not a publication journal.
  Default reruns resume the latest matching incomplete checkpointed run. Use
  -Output to select a run explicitly, or -NewRun to start fresh. Successful
  stages and progress are recorded in the single append-only checkpoints.jsonl.
#>
[CmdletBinding(SupportsShouldProcess=$true, ConfirmImpact='Low')]
param(
  [string]$Msix = 'D:\repo\deliverables\release\SemantierAgent-0.0.86-win-x64.msix',
  [string]$ExpectedInputSha256 = '416d715b00abc184a670e3c9e789a9baf18bd6996c47c20c67c05a7efb124ac7',
  [string]$SourceCommit = '5e9b6d5bd9025b9c64a28ae14243068e0125c3c0',
  [string]$Version = '0.0.86.0',
  [string]$BuildId = '7a76c0b5c57d4845a29ca56e429ed2c1',
  [string]$Thumbprint = '57BF15739B4AA447842EAF388856A01B4E5029FF',
  [string]$SshTarget = 'root@123.57.184.194',
  [string]$IdentityFile = '',
  [string]$RemoteRoot = '/opt/semantier-client-distribution/channel1-storage-root/releases',
  [string]$PublicBase = 'https://download.semantier.cn',
  [string]$UserProfile = "$env:LOCALAPPDATA\hermes-semantier-acceptance",
  [string]$Output = ('D:\repo\deliverables\work\ecs-deploy-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)),
  [string]$SetupExe = 'D:\repo\deliverables\cache\ecs-setup\semantier-setup-ch1.exe',
  [string]$SignTool = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe',
  [string]$Node = '',
  [string]$Repository = '',
  [string]$CredentialsEnv = '',
  [string]$TimestampUrl = 'http://timestamp.digicert.com',
  [Alias('PrepareOnly')][switch]$LocalOnly,
  [switch]$BuildOnly,
  [switch]$DesktopOnly,
  [string]$BuildSource = '',
  [switch]$SkipOldCleanup,
  [switch]$NewRun,
  [switch]$SetupOnly,
  [switch]$LocalSetup,
  [switch]$UpdateSetup,
  [switch]$VerifyPublicDownload,
  [switch]$Latest,
  [string]$Commit = '',
  [string]$BuildCache = 'D:\repo\deliverables\cache\desktop-inputs',
  [string]$BuildPython = '',
  [string]$AcceptanceNodeModules = '',
  [string]$VersionState = 'D:\repo\semantier-agent-build-version.json',
  [ValidateSet('Beta','Prod')][string]$BuildMode = 'Beta',
  [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$identity = 'Semantier.Agent.Acceptance'
$publisher = 'CN=Semantier Technology, O=Semantier Technology'
$utf8 = New-Object Text.UTF8Encoding($false)
$script:statusClock = [Diagnostics.Stopwatch]::StartNew()
$script:milestoneClock = [Diagnostics.Stopwatch]::StartNew()
$script:statusLog = $null
$script:askPassExe = $null
$script:releaseEnvironment = @{}
$script:sourceLease = $null
$script:sourceOwner = $null
$script:sourceRecreated = $false
function Get-ProfileFileHolders([string[]]$Files) {
  if(-not ('SemantierProfileLocks' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
public static class SemantierProfileLocks {
  [StructLayout(LayoutKind.Sequential)] public struct UniqueProcess { public int pid; public System.Runtime.InteropServices.ComTypes.FILETIME started; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] public struct ProcessInfo {
    public UniqueProcess process;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=256)] public string app;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=64)] public string service;
    public int type; public uint status; public uint session; [MarshalAs(UnmanagedType.Bool)] public bool restartable;
  }
  [DllImport("rstrtmgr.dll", CharSet=CharSet.Unicode)] static extern int RmStartSession(out uint handle, int flags, System.Text.StringBuilder key);
  [DllImport("rstrtmgr.dll")] static extern int RmEndSession(uint handle);
  [DllImport("rstrtmgr.dll", CharSet=CharSet.Unicode)] static extern int RmRegisterResources(uint handle, uint count, string[] files, uint apps, IntPtr processes, uint services, IntPtr names);
  [DllImport("rstrtmgr.dll")] static extern int RmGetList(uint handle, out uint needed, ref uint count, [In, Out] ProcessInfo[] info, ref uint reasons);
  public static ProcessInfo[] Find(string[] files) {
    uint h; int rc=RmStartSession(out h,0,new System.Text.StringBuilder(33));
    if(rc!=0) throw new System.ComponentModel.Win32Exception(rc);
    try {
      rc=RmRegisterResources(h,(uint)files.Length,files,0,IntPtr.Zero,0,IntPtr.Zero);
      if(rc!=0) throw new System.ComponentModel.Win32Exception(rc);
      uint needed, count=0, reasons=0;
      rc=RmGetList(h,out needed,ref count,null,ref reasons);
      if(rc==0) return new ProcessInfo[0];
      if(rc!=234) throw new System.ComponentModel.Win32Exception(rc);
      for(int retry=0;retry<3;retry++) {
        count=needed; var list=new ProcessInfo[count];
        rc=RmGetList(h,out needed,ref count,list,ref reasons);
        if(rc==0) { Array.Resize(ref list,(int)count); return list; }
        if(rc!=234) throw new System.ComponentModel.Win32Exception(rc);
      }
      throw new InvalidOperationException("Profile holder list changed repeatedly.");
    } finally { RmEndSession(h); }
  }
}
'@
  }
  foreach($file in $Files) {
    foreach($holder in [SemantierProfileLocks]::Find(@($file))) {
      $ticks=([long]$holder.process.started.dwHighDateTime -shl 32) -bor ([long]$holder.process.started.dwLowDateTime -band 0xffffffffL)
      [pscustomobject]@{Id=$holder.process.pid; Started=[DateTime]::FromFileTimeUtc($ticks)}
    }
  }
}
function Stop-ProfileBrowserHolders([string]$ProfilePath) {
  $locked=@(Get-ChildItem -LiteralPath $ProfilePath -Recurse -File | Where-Object {
    $_.FullName -match '[\\/]browser-profile[\\/]' -and $_.Length -ge 0
  } | Where-Object {
    try { $stream=[IO.File]::Open($_.FullName,'Open','Read','None'); $stream.Dispose(); $false } catch [IO.IOException] { $true }
  } | ForEach-Object { $_.FullName })
  $holders=@(Get-ProfileFileHolders $locked | Sort-Object Id -Unique)
  $browsers=@()
  foreach($holder in $holders) {
    $process=Get-Process -Id $holder.Id -ErrorAction SilentlyContinue
    if(!$process) { continue }
    if($process.ProcessName -notin @('msedge','chrome','chromium','brave') -or $process.StartTime.ToUniversalTime() -ne $holder.Started) {
      throw "Profile file held by process $($holder.Id); close it and retry."
    }
    $browsers+= $process
  }
  foreach($process in $browsers) {
    if($process.HasExited) { continue }
    Report-Status "Stopping browser process $($process.Id) holding a selected profile file."
    Stop-Process -Id $process.Id -Force -ErrorAction Stop
    $process.WaitForExit()
  }
  return $holders.Count
}
function Set-ReleaseEnvironment([string]$Name,[string]$Value) {
  if(!$script:releaseEnvironment.ContainsKey($Name)){$script:releaseEnvironment[$Name]=[Environment]::GetEnvironmentVariable($Name,'Process')}
  [Environment]::SetEnvironmentVariable($Name,$Value,'Process')
}
function Assert-LocalWindowsPath([string]$Path) {
  if(!$Path -or ![IO.Path]::IsPathRooted($Path)){throw 'Build paths must be absolute local Windows paths.'}
  $full=[IO.Path]::GetFullPath($Path)
  $drive=New-Object IO.DriveInfo([IO.Path]::GetPathRoot($full))
  if($full.StartsWith('\\') -or $drive.DriveType -ne [IO.DriveType]::Fixed){throw 'Build staging, tools and output must be on a local fixed Windows drive.'}
  return $full
}
function Get-EcsSshPassword([string]$Path) {
  if($env:ALI_ECS_SSH_PASSWORD){return $env:ALI_ECS_SSH_PASSWORD}
  if(!$Path){return $null}
  if(!(Test-Path -LiteralPath $Path -PathType Leaf)){throw 'The explicit -CredentialsEnv file is unavailable.'}
  $password=$null;$found=$false
  foreach($line in [IO.File]::ReadAllLines($Path)) {
    if($line -cnotmatch '^\s*(?:export\s+)?ALI_ECS_SSH_PASSWORD\s*=\s*(.*)$'){continue}
    if($found){throw 'Duplicate SSH password assignments in -CredentialsEnv.'}
    $found=$true;$value=$Matches[1].Trim()
    if($value.StartsWith('"') -or $value.StartsWith("'")) {
      if($value -notmatch '^(["''])(.*?)\1\s*(?:#.*)?$'){throw 'Malformed SSH password assignment in -CredentialsEnv.'}
      $password=$Matches[2]
    } else {$password=($value -replace '\s+#.*$','').Trim()}
  }
  if(!$password){throw 'Credentials file must contain nonempty ALI_ECS_SSH_PASSWORD; omit -CredentialsEnv to use SSH keys.'}
  return $password
}
function Assert-ExecutionPolicyPreflight {
  $effective=Get-ExecutionPolicy
  $policies=@(Get-ExecutionPolicy -List)
  $managed=@($policies | Where-Object {$_.Scope -in @('MachinePolicy','UserPolicy') -and $_.ExecutionPolicy -ne 'Undefined'})
  if($effective -in @('Restricted','AllSigned')){
    if($managed.Count){throw "Execution policy $effective is enforced by Group Policy. Use administrator-approved signed build scripts; this operator cannot override managed policy."}
    throw "Execution policy $effective can block unsigned build helpers. Relaunch with: pwsh.exe -NoProfile -ExecutionPolicy Bypass -File '$PSCommandPath' (append the same build arguments). Persistent CurrentUser Bypass is an explicit user choice; this script never sets it."
  }
  Report-Status ("Execution-policy preflight: {0}; persistent policy unchanged." -f $effective)
  if($effective -eq 'RemoteSigned'){
    Report-Status "UNC/R: unsigned scripts may be blocked before preflight. If so, use pwsh.exe -NoProfile -ExecutionPolicy Bypass -File '$PSCommandPath' with the same arguments."
  }
}
function Initialize-ReleasePaths {
  Assert-ExecutionPolicyPreflight
  if([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT){throw 'Run on native Windows.'}
  foreach($name in @('GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_INDEX_FILE')){
    if([Environment]::GetEnvironmentVariable($name,'Process')){throw ('Clear Git repository override before building: '+$name)}
  }
  if($env:SEMANTIER_WSL_GIT_DISTRO){throw 'Clear SEMANTIER_WSL_GIT_DISTRO; WSL Git is not supported by this workflow.'}
  if(!$Repository){$script:Repository=Split-Path -Parent $PSScriptRoot}
  $null=Assert-LocalWindowsPath $Repository
  Assert-NoReparseAncestor $Repository
  $script:Repository=(Resolve-Path -LiteralPath $Repository -ErrorAction Stop).ProviderPath.TrimEnd('\','/')
  if(!(Test-Path -LiteralPath (Join-Path $Repository 'scripts\bundles\desktop.py'))){throw 'Repository must name the Hermes source checkout.'}
  $script:BuildCache=Assert-LocalWindowsPath $BuildCache
  $script:Output=Assert-LocalWindowsPath $Output
  $script:VersionState=Assert-LocalWindowsPath $VersionState
  $script:SetupExe=Assert-LocalWindowsPath $SetupExe
  $lock=Get-Content -LiteralPath (Join-Path $Repository 'pm\lock.json') -Raw | ConvertFrom-Json
  if(!$BuildPython){$script:BuildPython=Join-Path $BuildCache ('tools\python-'+$lock.packages.python.version+'-win32-x64\python.exe')}
  if(!$Node){$script:Node=Join-Path $BuildCache ('tools\node-'+$lock.packages.node.version+'-win32-x64\node.exe')}
  foreach($tool in @($BuildPython,$Node)) {
    $null=Assert-LocalWindowsPath $tool
    if(!(Test-Path -LiteralPath $tool -PathType Leaf)){throw 'Existing cached Windows Python/Node unavailable; specify -BuildPython/-Node. No installation attempted.'}
  }
  if($AcceptanceNodeModules){$script:AcceptanceNodeModules=Assert-LocalWindowsPath $AcceptanceNodeModules}
  if($CredentialsEnv){$script:CredentialsEnv=(Resolve-Path -LiteralPath $CredentialsEnv -ErrorAction Stop).ProviderPath;$null=Get-EcsSshPassword $CredentialsEnv}
  $git=(Get-Command git.exe -CommandType Application -ErrorAction Stop).Source
  Set-ReleaseEnvironment 'PATH' ((Split-Path -Parent $git)+';'+$env:PATH)
  $trust=@($Repository.Replace('\','/'))
  $script:RuntimeRoot=$null
  $parent=Split-Path -Parent $Repository
  if((Split-Path -Leaf $Repository) -eq 'hermes-agent' -and (Test-Path -LiteralPath (Join-Path $parent '.gitmodules'))){$script:RuntimeRoot=$parent;$trust+=$parent.Replace('\','/')}
  $count=0
  if($env:GIT_CONFIG_COUNT -and (![int]::TryParse($env:GIT_CONFIG_COUNT,[ref]$count) -or $count -lt 0)){throw 'Invalid inherited GIT_CONFIG_COUNT.'}
  foreach($path in $trust){
    Set-ReleaseEnvironment "GIT_CONFIG_KEY_$count" 'safe.directory'
    Set-ReleaseEnvironment "GIT_CONFIG_VALUE_$count" $path
    $count++
  }
  Set-ReleaseEnvironment "GIT_CONFIG_KEY_$count" 'core.longpaths'
  Set-ReleaseEnvironment "GIT_CONFIG_VALUE_$count" 'true'
  Set-ReleaseEnvironment 'GIT_CONFIG_COUNT' ([string]($count+1))
  $head=& git.exe -C $Repository rev-parse HEAD
  if($LASTEXITCODE -ne 0){throw 'Native Windows Git cannot inspect the source repository.'}
  $script:SourceBinding=[ordered]@{repository=$Repository;toolingHermesHead=[string]$head;runtimeHead=$null;hermesGitlink=$null}
  if($RuntimeRoot){
    $runtimeGitArgs=@('-C',$RuntimeRoot)
    $runtimeHead=& git.exe @runtimeGitArgs rev-parse HEAD
    if($LASTEXITCODE -ne 0){throw 'Cannot inspect Runtime HEAD.'}
    $link=& git.exe @runtimeGitArgs ls-tree HEAD hermes-agent
    if($LASTEXITCODE -ne 0 -or $link -notmatch '^160000 commit ([a-f0-9]{40})\s'){throw 'Cannot inspect recorded Hermes gitlink.'}
    $script:SourceBinding.runtimeHead=[string]$runtimeHead;$script:SourceBinding.hermesGitlink=$Matches[1]
    if($Commit -and ($Commit -cne $script:SourceBinding.hermesGitlink -or $head -cne $Commit)){
      throw 'Requested commit and actual Hermes HEAD must equal the Runtime gitlink; synchronize the pinned source first.'
    }
  }
  if($Commit -and !$RuntimeRoot){throw 'Commit builds require the Runtime checkout and its pinned Hermes gitlink.'}
}
function Initialize-ReactCompilerCache([string]$Source) {
  # The admitted Vite config already keys compiled output by module text,
  # relative module path, lockfile, compiler helper, config and NODE_ENV.
  # Keep that cache outside disposable source without patching pinned code.
  $sourceRoot=Assert-LocalWindowsPath $Source
  $shared=Assert-LocalWindowsPath (Join-Path $BuildCache 'react-compiler\desktop')
  $local=Join-Path $sourceRoot 'apps\desktop\node_modules\.cache\hermes-react-compiler'
  if(!(Test-Path -LiteralPath (Join-Path $sourceRoot 'apps\desktop\scripts\compiler-cache.mjs'))){
    Report-Status 'React Compiler cache: selected source has no cache helper; unchanged.'
    return
  }
  New-Item -ItemType Directory -Path $shared -Force | Out-Null
  New-Item -ItemType Directory -Path (Split-Path -Parent $local) -Force | Out-Null
  # Resolve dependency junctions before writing: never mutate Linux dependencies.
  $realParent=& $Node -e 'const fs=require("node:fs");process.stdout.write(fs.realpathSync(process.argv[1]))' (Split-Path -Parent $local)
  if($LASTEXITCODE -ne 0){throw 'Cannot resolve React Compiler cache parent.'}
  $null=Assert-LocalWindowsPath ([string]$realParent)
  if(Test-Path -LiteralPath $local){
    $item=Get-Item -LiteralPath $local -Force
    if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){
      $realCache=& $Node -e 'const fs=require("node:fs");process.stdout.write(fs.realpathSync(process.argv[1]))' $local
      if($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath([string]$realCache).TrimEnd('\') -ine $shared.TrimEnd('\')){
        throw 'React Compiler cache points elsewhere; preserve and inspect existing link.'
      }
      Report-Status ('Reusing persistent React Compiler cache: '+$shared)
      return
    }
    if(!$item.PSIsContainer){throw 'React Compiler cache path is not a directory.'}
    # Seed warm entries once; retain the original rather than deleting work.
    & "$env:WINDIR\System32\robocopy.exe" $local $shared /E /XJ /R:1 /W:1 /NFL /NDL /NP | Out-Null
    if($LASTEXITCODE -gt 7){throw 'React Compiler cache seed failed.'}
    Move-Item -LiteralPath $local -Destination ($local+'.previous-'+[Guid]::NewGuid().ToString('N'))
  }
  New-Item -ItemType Junction -Path $local -Target $shared | Out-Null
  Report-Status ('Persistent React Compiler cache ready: '+$shared)
}
function Assert-NoReparseAncestor([string]$Path) {
  $cursor=[IO.Path]::GetFullPath($Path)
  while($cursor){
    if(Test-Path -LiteralPath $cursor){
      if((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse point in build path: $cursor"}
    }
    $cursor=[IO.Path]::GetDirectoryName($cursor)
  }
}
function Set-RunBuildSource {
  $run=Assert-LocalWindowsPath $Output
  $work=[IO.Path]::GetFullPath('D:\repo\deliverables\work')
  if([IO.Path]::GetDirectoryName($run) -ine $work -or (Split-Path -Leaf $run) -cnotmatch '^ecs-deploy-\d{8}-\d{6}-[a-f0-9]{8}$'){
    throw 'Commit builds require a unique ecs-deploy-<date>-<time>-<id> run directly under D:\repo\deliverables\work.'
  }
  $expected=Join-Path $run 'source'
  if($BuildSource -and [IO.Path]::GetFullPath($BuildSource) -ine $expected){throw '-BuildSource must equal the selected run Output\source.'}
  Assert-NoReparseAncestor $expected
  foreach($protected in @($Repository,$BuildCache,$VersionState,$SetupExe,$UserProfile,$CredentialsEnv,$AcceptanceNodeModules,$BuildPython,$Node,$SignTool)){
    if(!$protected){continue}
    $full=[IO.Path]::GetFullPath($protected)
    if($full -ieq $run -or $full.StartsWith($run+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
      throw 'Source retirement cannot contain a protected source/cache/ledger/release input.'
    }
  }
  $script:BuildSource=$expected
}
function Get-SourceGit([string]$Root,[string[]]$Arguments) {
  $value=& git.exe -C $Root @Arguments
  if($LASTEXITCODE -ne 0){throw ('Native Git inspection failed: '+($Arguments -join ' '))}
  return ($value -join "`n").Trim()
}
function Save-HermesChanges([string]$Hermes) {
  if(Get-SourceGit $Hermes @('ls-files','--unmerged')){throw 'Resolve existing Hermes merge conflicts before sync.'}
  Report-Status 'Saving all tracked and untracked Hermes changes before GitHub sync.'
  Run 'git.exe' @('-C',$Hermes,'add','--all')
  Run 'git.exe' @('-C',$Hermes,'commit','-m','chore: save local Hermes changes before latest deployment')
  if(Get-SourceGit $Hermes @('status','--porcelain','--untracked-files=all')){throw 'Hermes changed during auto-commit; retry after active writers finish.'}
  Run 'git.exe' @('-C',$Hermes,'fetch','--no-tags','origin','main')
  $fetched=Get-SourceGit $Hermes @('rev-parse','FETCH_HEAD^{commit}')
  Run 'git.exe' @('-C',$Hermes,'merge','--no-edit',$fetched)
  Run 'git.exe' @('-C',$Hermes,'push','origin','HEAD:refs/heads/main')
}
function Save-RuntimeChanges([string]$Runtime) {
  if(Get-SourceGit $Runtime @('ls-files','--unmerged')){throw 'Resolve existing Runtime merge conflicts before sync.'}
  if(Get-SourceGit $Runtime @('status','--porcelain','--','hermes-agent')){
    throw 'Hermes changes require separate qualification; commit and synchronize the qualified pin before -Latest.'
  }
  if(Get-SourceGit $Runtime @('status','--porcelain','--untracked-files=all')){
    Report-Status 'Saving Runtime local changes in Git before GitHub sync.'
    Run 'git.exe' @('-C',$Runtime,'add','--all')
    Run 'git.exe' @('-C',$Runtime,'commit','-m','chore: save local Runtime changes before latest deployment')
    if(Get-SourceGit $Runtime @('status','--porcelain','--untracked-files=all')){
      throw 'Runtime changed during auto-commit; retry after active writers finish.'
    }
  }
}
function Sync-GitHubLatestSource([string]$Runtime,[string]$Operator) {
  if($WhatIfPreference){
    Write-Output 'PLAN_ONLY: save and push dirty Hermes changes, stopping for qualification; otherwise commit Runtime changes, merge/push GitHub main, synchronize pinned Hermes, then build exact gitlink. No files changed.'
    return $null
  }
  if(!$IsWindows){throw '-Latest requires native Windows PowerShell and Git for Windows.'}
  $expected='D:\repo\semantier-runtime'
  if([IO.Path]::GetFullPath($Runtime).TrimEnd('\') -ine $expected -or
      [IO.Path]::GetFullPath($Operator) -ine (Join-Path $expected 'hermes-agent\scripts\deploy-ecs.ps1')){
    throw '-Latest requires the canonical D:\repo\semantier-runtime checkout and operator.'
  }
  $null=Assert-LocalWindowsPath $Runtime
  Assert-NoReparseAncestor $Runtime
  foreach($name in @('GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_INDEX_FILE','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES')){
    if([Environment]::GetEnvironmentVariable($name,'Process')){throw ('Unset Git override before sync: '+$name)}
  }
  if($env:SEMANTIER_WSL_GIT_DISTRO){throw 'WSL Git is not supported by the native release workflow.'}
  $null=Get-Command git.exe -CommandType Application -ErrorAction Stop
  $root=Get-SourceGit $Runtime @('rev-parse','--show-toplevel')
  if($root.Replace('\','/') -ine $expected.Replace('\','/')){throw 'Runtime Git top-level mismatch.'}
  if((Get-SourceGit $Runtime @('branch','--show-current')) -cne 'main'){
    throw 'GitHub sync requires checked-out Runtime main.'
  }
  $remote=Get-SourceGit $Runtime @('remote','get-url','origin')
  if($remote -cnotmatch '^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)chris-han/semantier-runtime(?:\.git)?$'){
    throw 'origin is not the expected chris-han/semantier-runtime GitHub repository.'
  }
  $hermes=Join-Path $Runtime 'hermes-agent'
  if((Test-Path -LiteralPath $hermes) -and (Get-SourceGit $hermes @('status','--porcelain','--untracked-files=all'))){
    if((Get-SourceGit $hermes @('branch','--show-current')) -cne 'main'){throw 'Dirty Hermes sync requires checked-out Hermes main.'}
    $hermesRemote=Get-SourceGit $hermes @('remote','get-url','origin')
    if($hermesRemote -cnotmatch '^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)chris-han/hermes-agent(?:\.git)?$'){throw 'Hermes origin is not the expected GitHub repository.'}
    Save-HermesChanges $hermes
    throw 'Hermes changes were committed and pushed. Qualify the resulting Hermes revision and synchronize the Runtime gitlink before retrying -Latest.'
  }
  Save-RuntimeChanges $Runtime
  $operatorHash=(Get-FileHash -LiteralPath $Operator -Algorithm SHA256).Hash
  Report-Status 'GitHub sync: fetch and merge Runtime main, then push preserved local commits.'
  Run 'git.exe' @('-C',$Runtime,'fetch','--no-tags','origin','main')
  $fetched=Get-SourceGit $Runtime @('rev-parse','FETCH_HEAD^{commit}')
  $local=Get-SourceGit $Runtime @('rev-parse','HEAD')
  Run 'git.exe' @('-C',$Runtime,'merge','--no-edit',$fetched)
  if((Get-SourceGit $Runtime @('rev-parse','HEAD')) -cne $fetched){
    Run 'git.exe' @('-C',$Runtime,'push','origin','HEAD:refs/heads/main')
  }
  Run 'git.exe' @('-C',$Runtime,'submodule','sync','--','hermes-agent')
  Run 'git.exe' @('-C',$Runtime,'submodule','update','--init','--checkout','--','hermes-agent')
  $link=Get-SourceGit $Runtime @('ls-tree','HEAD','hermes-agent')
  if($link -cnotmatch '^160000 commit ([a-f0-9]{40})\s+hermes-agent$'){
    throw 'Updated Runtime HEAD has no valid Hermes gitlink.'
  }
  $pinned=$Matches[1]
  if((Get-SourceGit $hermes @('rev-parse','HEAD')) -cne $pinned){
    throw 'Hermes HEAD is not the Runtime-pinned gitlink after GitHub sync.'
  }
  if((Get-FileHash -LiteralPath $Operator -Algorithm SHA256).Hash -cne $operatorHash){
    throw 'GitHub updated deploy-ecs.ps1. Re-run the same -Latest invocation to load the new operator before building.'
  }
  Report-Status ('GitHub sync PASS: Runtime '+(Get-SourceGit $Runtime @('rev-parse','HEAD'))+'; Hermes '+$pinned)
  return $pinned
}
function Assert-StageInactive($Owner) {
  # Never kill a process inferred from command text. Any possible stage use
  # defers retirement; the synchronous build/acceptance callers normally exit first.
  $processes=@(Get-CimInstance Win32_Process -ErrorAction Stop)
  foreach($process in $processes){
    if($process.ProcessId -eq $PID){continue}
    if(($process.ProcessId -eq $Owner.pid -and $process.CreationDate.ToUniversalTime().ToString('o') -ceq $Owner.started) -or
       ($process.ParentProcessId -eq $Owner.pid -and $process.CreationDate.ToUniversalTime() -ge [datetime]$Owner.started) -or
       ($process.ExecutablePath -and ($process.ExecutablePath.StartsWith($Owner.source+'\',[StringComparison]::OrdinalIgnoreCase) -or
         $process.ExecutablePath.StartsWith($Owner.cache+'\',[StringComparison]::OrdinalIgnoreCase))) -or
       ($process.CommandLine -and $process.CommandLine.IndexOf($Owner.source,[StringComparison]::OrdinalIgnoreCase) -ge 0)){
      throw 'Stage may have active children or an active previous operator; cleanup/recovery deferred.'
    }
  }
}
function Read-SourceOwner {
  $last=$null
  foreach($line in [IO.File]::ReadLines($script:statusLog)){
    if($line -notmatch '"type"\s*:\s*"source-owner"\s*[,}]'){continue}
    try{$event=$line | ConvertFrom-Json -ErrorAction Stop}catch{continue}
    if($event.type -ceq 'source-owner'){$last=$event.owner}
  }
  return $last
}
function Assert-SourceOwner($Owner,[string]$Source,[string]$Revision) {
  if(!$Owner -or $Owner.kind -cne 'native-git-worktree-v1' -or $Owner.run -ine $Output -or
    $Owner.source -ine $Source -or $Owner.repository -ine $Repository -or $Owner.cache -ine $BuildCache -or $Owner.commit -cne $Revision){
    throw 'Source ownership differs; preserve the foreign stage and inspect the run.'
  }
  Assert-NoReparseAncestor $Source
}
function Get-SourceRegistration([string]$Source,[string]$Revision) {
  $records=(Get-SourceGit $Repository @('worktree','list','--porcelain')) -split "`n`n"
  $registrations=@($records | Where-Object {($_ -split "`n")[0] -ceq ('worktree '+$Source.Replace('\','/'))})
  if(!$registrations.Count){return $null}
  if($registrations.Count -ne 1 -or $registrations[0] -notmatch '(?:\n)detached(?:\n|$)' -or
    $registrations[0] -notmatch ('(?:\n)HEAD '+$Revision+'(?:\n|$)') -or $registrations[0] -match '(?:\n)locked'){
    throw 'Exact detached stage registration differs or is locked.'
  }
  return $registrations[0]
}
function Assert-RegisteredSource([string]$Source,[string]$Revision) {
  if(!(Test-Path -LiteralPath (Join-Path $Source '.git') -PathType Leaf)){throw 'Expected a linked-worktree .git file.'}
  Assert-NoReparseAncestor (Join-Path $Source '.git')
  if((Get-SourceGit $Source @('rev-parse','--show-toplevel')).Replace('\','/') -ine $Source.Replace('\','/')){throw 'Stage Git root differs from the owned source.'}
  $common=Get-SourceGit $Repository @('rev-parse','--path-format=absolute','--git-common-dir')
  if($IsWindows){$null=Assert-LocalWindowsPath $common;Assert-NoReparseAncestor $common}
  $actual=Get-SourceGit $Source @('rev-parse','--path-format=absolute','--git-common-dir')
  if($actual -ine $common -or (Test-Path -LiteralPath (Join-Path $common 'objects/info/alternates'))){throw 'Stage must share the Hermes common directory without alternates.'}
  if((Get-SourceGit $Source @('rev-parse','HEAD')) -cne $Revision){throw 'Stage commit differs from the exact pinned SHA.'}
  if(!(Get-SourceRegistration $Source $Revision)){throw 'Exact detached stage registration is missing.'}
  return $common
}
function Initialize-LocalBuildSource([string]$Source,[string]$Revision) {
  if($WhatIfPreference){return}
  if($Revision -cnotmatch '^[a-f0-9]{40}$'){throw 'Exact source SHA required.'}
  if($Source -ine $BuildSource){throw 'Stage must be the selected run source.'}
  Assert-NoReparseAncestor $Source
  $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
  if(!$script:sourceLease){
    if(!(Test-Path -LiteralPath $Output)){New-Item -ItemType Directory -Path $Output | Out-Null}
    $lock=Join-Path $Output '.source.lock'
    Assert-NoReparseAncestor $lock
    $script:sourceLease=[IO.File]::Open($lock,'OpenOrCreate','ReadWrite','None')
    if(!(Test-Path -LiteralPath $script:statusLog)){[IO.File]::WriteAllText($script:statusLog,'')}
    Assert-NoReparseAncestor $script:statusLog
    $previous=Read-SourceOwner
    if($previous){Assert-SourceOwner $previous $Source $Revision;Assert-StageInactive $previous}
    elseif((Test-Path -LiteralPath $Source) -or (Get-Item -LiteralPath $script:statusLog).Length -gt 0 -or
      @(Get-ChildItem -LiteralPath $Output -Force | Where-Object {$_.Name -notin @('.source.lock','checkpoints.jsonl')}).Count){
      throw 'Occupied run/source has no workflow ownership; preserve it.'
    }
    if(Test-Path -LiteralPath $Source){
      $null=Assert-RegisteredSource $Source $Revision
      if(Get-SourceGit $Source @('status','--porcelain','--untracked-files=all')){throw 'Recovered stage has source changes; preserve and inspect.'}
    }
    $self=Get-Process -Id $PID
    $script:sourceOwner=[ordered]@{kind='native-git-worktree-v1';run=$Output;source=$Source;repository=$Repository;
      commit=$Revision;cache=$BuildCache;pid=$PID;started=$self.StartTime.ToUniversalTime().ToString('o')}
    Write-RunEvent @{type='source-owner';owner=$script:sourceOwner}
  }
  Assert-SourceOwner $script:sourceOwner $Source $Revision
  if(Test-Path -LiteralPath $Source){
    $null=Assert-RegisteredSource $Source $Revision
    $dirty=Get-SourceGit $Source @('status','--porcelain','--untracked-files=all')
    if($dirty){throw 'Recovered stage has source changes; preserve and inspect before rebuilding.'}
  } else {
    # Repair only this owned, exact-SHA registration after an interrupted
    # retirement. Never prune unrelated registrations.
    if(Get-SourceRegistration $Source $Revision){Run 'git.exe' @('-C',$Repository,'worktree','remove','--force',$Source)}
    Run 'git.exe' @('-C',$Repository,'-c','core.autocrlf=false','worktree','add','--detach',$Source,$Revision)
    $script:sourceRecreated=$true
  }
  $common=Assert-RegisteredSource $Source $Revision
  Write-RunEvent @{type='source-verified';source=$Source;commit=$Revision;commonDir=$common;binding=$script:SourceBinding}
}
function Remove-OwnedBuildSource {
  if(!$script:sourceOwner){return 'NOT_CREATED'}
  try {
    Assert-SourceOwner $script:sourceOwner $BuildSource $Commit
    Assert-StageInactive $script:sourceOwner
    if(Test-Path -LiteralPath $BuildSource){
      $null=Assert-RegisteredSource $BuildSource $Commit
      Run 'git.exe' @('-C',$Repository,'worktree','remove','--force',$BuildSource)
    }
    $listed=Get-SourceGit $Repository @('worktree','list','--porcelain')
    if((Test-Path -LiteralPath $BuildSource) -or ($listed -split "`n") -contains ('worktree '+$BuildSource.Replace('\','/'))){throw 'Stage or registration remains.'}
    return 'REMOVED'
  } catch {Write-Warning ('CLEANUP_DEFERRED: '+$_.Exception.Message);return ('CLEANUP_DEFERRED: '+$_.Exception.Message)}
}
function Reset-StalePreparation([string]$Prepared) {
  if(!$script:sourceRecreated -or !(Test-Path -LiteralPath $Prepared)){return}
  $expected=Join-Path $Output 'desktop-job\prepared.json'
  if([IO.Path]::GetFullPath($Prepared) -ine $expected){throw 'Prepared receipt escaped the selected run.'}
  Assert-NoReparseAncestor $Prepared
  # The receipt may reference retired payload paths. Regenerate through the
  # existing builder; persistent native/npm/uv/toolchain caches are untouched.
  Remove-Item -LiteralPath $Prepared -Force
  Report-Status 'Retired source preparation invalidated; reusing verified persistent caches through the existing builder.'
}
function Report-Status([string]$Message) {
  $line = '[{0}] [+{1:0}s] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $script:statusClock.Elapsed.TotalSeconds, $Message
  Write-Host $line
  if ($script:statusLog) { Write-RunEvent @{type='progress';at=[DateTime]::UtcNow.ToString('o');message=$Message} }
}
function Write-RunEvent($Event) {
  # Leading newline isolates a partial final record left by an interrupted write.
  $json=ConvertTo-Json -InputObject $Event -Depth 32 -Compress
  $deadline=[Diagnostics.Stopwatch]::StartNew()
  while($true) {
    try {
      [IO.File]::AppendAllText($script:statusLog,"`n"+$json+"`n",$utf8)
      return
    } catch [IO.IOException] {
      # Readers and other appenders can briefly deny write sharing on Windows.
      # Retry only contention, never permission, disk-space, or path errors.
      $code=$_.Exception.HResult -band 0xffff
      if($code -notin @(32,33) -or $deadline.Elapsed.TotalSeconds -ge 10){throw}
      Start-Sleep -Milliseconds 50
    }
  }
}
function Read-RunCheckpoint([string]$Path) {
  $last=$null
  foreach($line in [IO.File]::ReadLines($Path)) {
    if(!$line.Trim()){continue}
    if($line -notmatch '"type"\s*:\s*"checkpoint"\s*[,}]'){continue}
    try {$entry=$line | ConvertFrom-Json -ErrorAction Stop}
    catch {Write-Warning 'Ignoring an incomplete or malformed run-log record; saved artifacts will still be verified.';continue}
    if($entry.PSObject.Properties['type'] -and $entry.type -eq 'checkpoint') {$last=$entry}
  }
  return $last
}
function Save-Json($Value, [string]$Path) {
  $temporary=$Path+'.tmp'
  [IO.File]::WriteAllText($temporary, (ConvertTo-Json -InputObject $Value -Depth 32), $utf8)
  if ([IO.File]::Exists($Path)) {[IO.File]::Replace($temporary,$Path,$Path+'.previous')} else {[IO.File]::Move($temporary,$Path)}
}
function Complete-Checkpoint([string]$Stage) {
  $seconds=[math]::Round($script:milestoneClock.Elapsed.TotalSeconds,3)
  Write-RunEvent @{type='milestone-timing';stage=$Stage;seconds=$seconds;mode=$BuildMode;at=[DateTime]::UtcNow.ToString('o')}
  Report-Status ("Milestone {0}: {1:N3}s since previous checkpoint" -f $Stage,$seconds)
  $script:milestoneClock.Restart()
  Save-Json $receipt (Join-Path $Output 'result.json')
  $script:checkpoints[$Stage]=[DateTime]::UtcNow.ToString('o')
  Write-RunEvent @{type='checkpoint';stage=$Stage;status='SUCCESS';at=$script:checkpoints[$Stage];
    binding=$script:runBinding;completed=$script:checkpoints;receipt=$receipt}
  Report-Status "Checkpoint saved: $Stage"
}
function Test-UnsignedRepackageResume($Saved, [string]$Binding, $Build) {
  if(!$Build -or $Build.stage -cne 'built' -or @($Saved.completed.PSObject.Properties).Count){return $false}
  try{$old=$Saved.binding | ConvertFrom-Json;$current=$Binding | ConvertFrom-Json}catch{return $false}
  $oldNames=@($old.PSObject.Properties.Name | Sort-Object)
  $newNames=@($current.PSObject.Properties.Name | Sort-Object)
  if(($oldNames -join '|') -cne ($newNames -join '|')){return $false}
  foreach($name in $oldNames){
    if($name -cne 'inputSha256' -and $old.$name -cne $current.$name){return $false}
  }
  return ($old.inputSha256 -cne $current.inputSha256 -and $Build.commit -ceq $current.source -and
    ($Build.version+'.0') -ceq $current.version -and $Build.sha256 -ceq $current.inputSha256)
}
function Run([string]$Exe, [string[]]$Arguments) {
  $timer = [Diagnostics.Stopwatch]::StartNew()
  Report-Status (("Running {0} {1}" -f [IO.Path]::GetFileName($Exe), ($Arguments -join ' ')).TrimEnd())
  if([IO.Path]::GetFileName($Exe) -ieq 'sftp.exe') {
    # The progress meter needs the inherited console, not a PowerShell pipe.
    Invoke-SshNative $Exe $Arguments
  } elseif([IO.Path]::GetFileName($Exe) -ieq 'ssh.exe') {
    Invoke-SshNative $Exe $Arguments | Out-Host
  } elseif([IO.Path]::GetFileName($Exe) -in @('python.exe','python3.exe')) {
    # Windows PowerShell represents native stderr as ErrorRecord objects.
    # Keep draining both streams so the actual builder error reaches the run log.
    $previousErrorPreference=$ErrorActionPreference
    try {
      $ErrorActionPreference='Continue'
      & $Exe @Arguments 2>&1 | ForEach-Object {
        $line=[string]$_
        Write-Host $line
        if($script:statusLog){Write-RunEvent @{type='output';at=[DateTime]::UtcNow.ToString('o');tool=[IO.Path]::GetFileName($Exe);message=$line}}
      }
      $nativeExit=$LASTEXITCODE
    } finally {$ErrorActionPreference=$previousErrorPreference}
    if($script:statusLog){Write-RunEvent @{type='milestone-timing';stage=($Arguments -join ' ');tool=[IO.Path]::GetFileName($Exe);seconds=[math]::Round($timer.Elapsed.TotalSeconds,3);status=$(if($nativeExit -eq 0){'SUCCESS'}else{'FAIL'});mode=$BuildMode}}
    if($nativeExit -ne 0){throw "$Exe failed with exit code $nativeExit; see builder output in $script:statusLog"}
    Report-Status ("Completed {0} ({1:0}s)" -f [IO.Path]::GetFileName($Exe), $timer.Elapsed.TotalSeconds)
    return
  } else { & $Exe @Arguments | Out-Host }
  if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
  Report-Status ("Completed {0} ({1:0}s)" -f [IO.Path]::GetFileName($Exe), $timer.Elapsed.TotalSeconds)
}
function Run-Setup([string]$Exe, [string[]]$Arguments) {
  Report-Status ("Setup: {0}; waiting for completion" -f $Arguments[0])
  # GUI-subsystem EXEs can otherwise return control to Windows PowerShell before
  # the command finishes. Pass a quoted command line and explicitly wait.
  $argsLine=($Arguments | ForEach-Object { '"' + $_.Replace('"','') + '"' }) -join ' '
  $p=Start-Process -FilePath $Exe -ArgumentList $argsLine -Wait -PassThru
  if ($p.ExitCode -ne 0) {throw "Setup command failed with exit code $($p.ExitCode): $Exe $($Arguments[0])"}
  Report-Status ("Setup completed: {0}" -f $Arguments[0])
}
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant() }
function Preserve-Databases([string]$Mode) {
  $python=$BuildPython
  $code=@'
import os,sqlite3,sys,time,uuid
from contextlib import closing
from pathlib import Path
mode,home,backup=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
verify=sys.argv[4]=='Prod'
suffixes={'.db','.sqlite','.sqlite3'}
def connect(path,immutable=False):
    return sqlite3.connect(path.resolve().as_uri()+('?mode=ro&immutable=1' if immutable else '?mode=ro'),uri=True,timeout=5)
if mode=='backup':
    for source in home.rglob('*'):
        if not source.is_file() or source.suffix.lower() not in suffixes: continue
        target=backup/source.relative_to(home);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.snapshot-'+uuid.uuid4().hex)
        deadline=time.monotonic()+60
        def progress(status,remaining,total):
            if time.monotonic()>deadline: raise TimeoutError('SQLite snapshot exceeded 60 seconds: '+str(source))
        try:
            with closing(connect(source)) as src,closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst,pages=256,progress=progress,sleep=0.05)
                if verify and dst.execute('pragma integrity_check').fetchone()[0]!='ok': raise RuntimeError('Invalid SQLite snapshot: '+str(source))
            os.replace(temporary,target)
        finally:
            if temporary.exists(): temporary.unlink()
        print('SQLite snapshot verified: '+str(source.relative_to(home)))
else:
    for original in backup.rglob('*'):
        if not original.is_file() or original.suffix.lower() not in suffixes: continue
        current=home/original.relative_to(backup)
        with closing(connect(original,True)) as before,closing(connect(current)) as after:
            if after.execute('pragma integrity_check').fetchone()[0]!='ok': raise RuntimeError('Current database integrity failure: '+str(current))
            for (table,) in before.execute("select name from sqlite_master where type='table' and name in ('sessions','messages')"):
                columns={row[1] for row in before.execute('pragma table_info('+table+')')}
                if 'id' in columns:
                    old={row[0] for row in before.execute('select id from '+table)}
                    new={row[0] for row in after.execute('select id from '+table)}
                    if not old<=new: raise RuntimeError('Existing session/message IDs lost: '+str(current))
        print('SQLite data preserved: '+str(original.relative_to(backup)))
'@
  # Windows PowerShell 5.1 strips embedded quotes in native -c arguments.
  # Standard input preserves the source exactly, including SQL string literals.
  Report-Status ('SQLite '+$Mode+': running snapshot/preservation checks.')
  $code | & $python -B - $Mode $UserProfile $backup $BuildMode | Out-Host
  if($LASTEXITCODE -ne 0){throw "SQLite $Mode failed with exit code $LASTEXITCODE"}
}
function Test-PublicMsix([switch]$FullDownload) {
  $python=$BuildPython
  $code=@'
import hashlib,sys
from curl_cffi import requests,CurlHttpVersion
url,wanted,size,mode=sys.argv[1:];size=int(size)
with requests.Session() as client:
    options=dict(http_version=CurlHttpVersion.V3ONLY,allow_redirects=False,timeout=(15,60),headers={'Cache-Control':'no-cache','Accept-Encoding':'identity'})
    if mode=='head':
        response=client.head(url,**options)
        if response.status_code!=200 or int(response.headers.get('Content-Length','-1'))!=size:
            raise RuntimeError('Public H3 HEAD status/size mismatch')
        if response.headers.get('Accept-Ranges','').lower()!='bytes':
            raise RuntimeError('Public endpoint does not advertise byte-range support')
        print('PUBLIC_H3_HEAD_PASS: size='+str(size)+'; body download skipped')
    else:
        response=client.get(url,stream=True,**options)
        try:
            if response.status_code!=200: raise RuntimeError('Public H3 GET failed')
            digest=hashlib.sha256();total=0
            for chunk in response.iter_content():
                total+=len(chunk)
                if total>size: raise RuntimeError('Public package exceeds expected size')
                digest.update(chunk)
            if total!=size or digest.hexdigest()!=wanted: raise RuntimeError('Public package differs from local signed MSIX')
            print('PUBLIC_H3_FULL_SHA256_PASS')
        finally: response.close()
'@
  $mode=if($FullDownload){'full'}else{'head'}
  $code | & $python -B - "$PublicBase/releases/SemantierAgent.msix" $hash ([string](Get-Item -LiteralPath $signed).Length) $mode | Out-Host
  if($LASTEXITCODE -ne 0){throw "Public H3 $mode verification failed"}
}
function Require-Signature([string]$Path) {
  if($BuildMode -eq 'Beta'){return}
  Run $SignTool @('verify','/pa','/v',$Path)
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -cne $Thumbprint) {
    throw "Signature or signer mismatch: $Path"
  }
}
function Sign-Copy([string]$InputFile, [string]$Destination) {
  # Publisher equality is a signing prerequisite even when Beta tests are off.
  if([IO.Path]::GetExtension($InputFile) -ieq '.msix') {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $package=[IO.Compression.ZipFile]::OpenRead($InputFile)
    try{[xml]$manifest=Read-ZipText $package 'AppxManifest.xml'}finally{$package.Dispose()}
    $cert=Get-Item -LiteralPath ('Cert:\CurrentUser\My\'+$Thumbprint)
    if($manifest.Package.Identity.Publisher -cne $cert.Subject){throw 'MSIX Publisher does not match signing certificate; resume -Commit to repackage with the configured publisher.'}
  }
  Copy-Item -LiteralPath $InputFile -Destination $Destination
  $signature = Get-AuthenticodeSignature -LiteralPath $Destination
  if ($signature.Status -eq 'NotSigned') {
    Run $SignTool @('sign','/fd','SHA256','/sha1',$Thumbprint,'/tr',$TimestampUrl,'/td','SHA256',$Destination)
  }
  Require-Signature $Destination
}
function Read-ZipText($Archive, [string]$Name) {
  $entry = $Archive.GetEntry($Name)
  if (!$entry -or $entry.Length -gt 4194304) { throw "Missing/oversized package entry: $Name" }
  $reader = New-Object IO.StreamReader($entry.Open())
  try { $reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Invoke-SshNative([string]$Exe, [string[]]$Arguments) {
  $password=Get-EcsSshPassword $CredentialsEnv
  if(!$password) { & $Exe @Arguments; return }
  if(!$script:askPassExe) {
    $helperDirectory=Join-Path 'D:\repo\deliverables\work' ('ecs-ssh-askpass-'+[Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $helperDirectory | Out-Null
    $helper=Join-Path $helperDirectory 'askpass.exe'
    $source=Join-Path $helperDirectory 'askpass.cs'
    [IO.File]::WriteAllText($source,@'
using System;
using System.Text;
class AskPass {
  static int Main(string[] args) {
    if(args.Length!=1 || args[0].IndexOf("password",StringComparison.OrdinalIgnoreCase)<0)return 1;
    Console.OutputEncoding=new UTF8Encoding(false);
    Console.WriteLine(Environment.GetEnvironmentVariable("ALI_ECS_SSH_PASSWORD"));return 0;
  }
}
'@,$utf8)
    & "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:exe "/out:$helper" $source | Out-Host
    if($LASTEXITCODE -ne 0){throw 'Cannot compile native SSH password helper.'}
    $script:askPassExe=$helper
  }
  $names=@('SSH_ASKPASS','SSH_ASKPASS_REQUIRE','DISPLAY','ALI_ECS_SSH_PASSWORD')
  $savedEnvironment=@{}
  foreach($name in $names){$savedEnvironment[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
  try {
    [Environment]::SetEnvironmentVariable('SSH_ASKPASS',$script:askPassExe,'Process')
    [Environment]::SetEnvironmentVariable('SSH_ASKPASS_REQUIRE','force','Process')
    [Environment]::SetEnvironmentVariable('DISPLAY','semantier','Process')
    [Environment]::SetEnvironmentVariable('ALI_ECS_SSH_PASSWORD',$password,'Process')
    $Arguments=@('-o','PreferredAuthentications=password','-o','NumberOfPasswordPrompts=1')+$Arguments
    & $Exe @Arguments
  } finally {
    foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process')}
    $password=$null
  }
}
function Ssh([string]$Command) {
  # Base64 avoids Windows/POSIX quoting differences, including $ and newlines.
  $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Command.Replace("`r",'')))
  $previousErrorPreference=$ErrorActionPreference
  try {
    $ErrorActionPreference='Continue'
    $text = @(Invoke-SshNative 'ssh.exe' ($script:sshOptions + @($SshTarget,"printf '%s' '$encoded' | base64 -d | sh")) 2>&1)
    $remoteExit=$LASTEXITCODE
  } finally {$ErrorActionPreference=$previousErrorPreference}
  if ($remoteExit -ne 0) {
    $detail=($text | ForEach-Object {[string]$_}) -join "`n"
    if($script:statusLog){Write-RunEvent @{type='remote-error';exitCode=$remoteExit;message=$detail;at=[DateTime]::UtcNow.ToString('o')}}
    throw "ECS SSH command failed ($remoteExit): $detail"
  }
  return ($text -join "`n")
}
function Transfer-Sftp([string]$LocalFile, [string]$RemoteFile, [switch]$Download) {
  if($LocalFile -match '[\r\n"]' -or $RemoteFile -match '[\r\n"]'){throw 'Invalid SFTP filename.'}
  if(!$Download) {
    # Windows OpenSSH reput requires the remote file to exist. Create only a
    # missing staging file; never truncate an existing resumable upload.
    if($RemoteFile -match "'"){throw 'Invalid remote SFTP filename.'}
    $null=Ssh "set -eu; if test ! -e '$RemoteFile'; then umask 077; : > '$RemoteFile'; fi; test -f '$RemoteFile'; test ! -L '$RemoteFile'"
  }
  $local=$LocalFile.Replace('\','/').Replace('[','\[').Replace(']','\]').Replace('*','\*').Replace('?','\?')
  $command=if($Download){'get "'+$RemoteFile+'" "'+$local+'"'}else{'reput "'+$local+'" "'+$RemoteFile+'"'}
  $batch=Join-Path $Output ('sftp-'+[Guid]::NewGuid().ToString('N')+'.txt')
  [IO.File]::WriteAllText($batch,"progress`n"+$command+"`nbye`n",$utf8)
  try {
    # -N clears quiet mode; the batch's progress command enables the meter.
    # BatchMode=no permits ASKPASS/password authentication; errors still abort.
    Run 'sftp.exe' ($script:sshOptions + @('-o','BatchMode=no','-N','-b',$batch,$SshTarget))
  } finally {
    Remove-Item -LiteralPath $batch -Force
  }
}
function Upload([string]$Path, [string]$RemoteFile) {
  if($BuildMode -eq 'Beta'){Transfer-Sftp $Path $RemoteFile;Complete-Checkpoint ('upload-'+[IO.Path]::GetFileName($Path));return}
  if([IO.Path]::GetExtension($Path) -ieq '.exe') {
    Require-Signature $Path
    Report-Status ('Uploading signed Setup: '+[IO.Path]::GetFileName($Path))
    Transfer-Sftp $Path $RemoteFile
    Complete-Checkpoint ('upload-'+[IO.Path]::GetFileName($Path))
    return
  }
  $localHash=Hash $Path
  $remoteHash=Ssh "if test -f '$RemoteFile'; then sha256sum '$RemoteFile' | cut -d' ' -f1; fi"
  if($remoteHash -ceq $localHash) {Report-Status ('Reusing verified upload: '+[IO.Path]::GetFileName($Path));return}
  Report-Status ("Uploading {0} ({1:N1} MiB)" -f [IO.Path]::GetFileName($Path), ((Get-Item -LiteralPath $Path).Length / 1MB))
  Transfer-Sftp $Path $RemoteFile
  $remoteHash=Ssh "sha256sum '$RemoteFile' | cut -d' ' -f1"
  if($remoteHash -cne $localHash){throw 'Uploaded file hash mismatch.'}
  Complete-Checkpoint ('upload-'+[IO.Path]::GetFileName($Path))
}
function Prepare-IndependentSetup {
  $setupName=if($LocalSetup){'semantier-setup-ch0.exe'}else{'semantier-setup-ch1.exe'}
  $setup=Join-Path $Output $setupName
  $setupExpected=if($LocalSetup){'local-adjacent-single-msix'}else{"${PublicBase}/releases/SemantierAgent.msix"}
  $setupContractVersion=$setupExpected+'|verify-remote-expected-sha256-v2'
  if($UpdateSetup -and !$receipt.Contains('setupExplicitUpdate')){$script:checkpoints.Remove('setup')}
  if($script:checkpoints.ContainsKey('setup')) {
    Require-Signature $setup
    Report-Status 'Reusing checkpointed signed Setup.'
  } else {
  $setupExpected=if($LocalSetup){'local-adjacent-single-msix'}else{"${PublicBase}/releases/SemantierAgent.msix"}
  $setupIsReusable=$false
  # Do not execute an older cached Setup merely to discover its command-line API.
  # Only our own prior fixed-file builds carry this separate contract marker.
  $cachedContract=$SetupExe+'.fixed-msix-url.txt'
  $setupContractVersion=$setupExpected+'|verify-remote-expected-sha256-v2'
  if (!$UpdateSetup -and (Test-Path -LiteralPath $SetupExe -PathType Leaf) -and
      (Get-Item -LiteralPath $SetupExe).Length -le 2097152 -and
      (Test-Path -LiteralPath $cachedContract -PathType Leaf) -and
      (([IO.File]::ReadAllText($cachedContract)).Trim() -ceq ($setupContractVersion+'|asimov-v1'))) {
    try {
      Require-Signature $SetupExe
      $probe=Join-Path $Output 'setup-contract.txt'
      Run-Setup $SetupExe @('--describe-fixed-download',$probe)
      if ((Test-Path -LiteralPath $probe -PathType Leaf) -and
          (([IO.File]::ReadAllText($probe)).Trim() -ceq $setupExpected)) {
        $setupIsReusable=$true
        Copy-Item -LiteralPath $SetupExe -Destination $setup
      }
    } catch { Write-Host 'Cached Setup does not support fixed-file contract; will build the small Setup only.' }
  }
  if (!$setupIsReusable) {
    Report-Status ('Building small '+$(if($LocalSetup){'offline'}else{'WinHTTP'})+' Setup with Windows .NET Framework; no runtime bundling.')
    $compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    $setupWork=Join-Path $Output 'setup-build'
    New-Item -ItemType Directory -Path $setupWork -Force | Out-Null
    $installerSource=Join-Path $Repository 'win-native-installer'
    $source=[IO.File]::ReadAllText((Join-Path $installerSource 'Program.cs'))+"`n"+
      [IO.File]::ReadAllText((Join-Path $installerSource 'WinHttp.cs'))+"`n"+
      [IO.File]::ReadAllText((Join-Path $installerSource 'InstallerWindow.cs'))
    [IO.File]::WriteAllText((Join-Path $setupWork 'Program.cs'),$source,$utf8)
    $manifestPath=Join-Path $setupWork 'setup.manifest'
    Copy-Item -LiteralPath (Join-Path $installerSource 'setup.manifest') -Destination $manifestPath
    $built=Join-Path $setupWork 'SemantierSetup.exe'
    $setupDefine=if($LocalSetup){'/define:LOCAL_SETUP'}else{'/define:ONLINE_SETUP'}
    Run $compiler @($setupDefine,'/nologo','/target:winexe','/platform:x64','/optimize+',
      ('/out:'+$built),('/win32manifest:'+$manifestPath),('/win32icon:'+(Join-Path $Repository 'apps\desktop\assets\icon.ico')),
      ('/resource:'+(Join-Path $Repository 'apps\desktop\assets\icon.ico')+',Semantier.icon.ico'),
      ('/resource:'+(Join-Path $installerSource 'SemantierPublisher.cer')+',Semantier.publisher.cer'),
      ('/resource:'+(Join-Path $Repository 'apps\desktop\assets\icon.png')+',Semantier.logo.png'),
      '/reference:System.Windows.Forms.dll','/reference:System.Drawing.dll',
      '/reference:System.Xml.Linq.dll','/reference:System.IO.Compression.dll',
      '/reference:System.IO.Compression.FileSystem.dll',(Join-Path $setupWork 'Program.cs'))
    if((Get-Item -LiteralPath $built).Length -gt 2097152){throw 'Native Setup exceeds the 2 MiB size budget.'}
    Sign-Copy $built $setup
    New-Item -ItemType Directory -Path (Split-Path -Parent $SetupExe) -Force | Out-Null
    Copy-Item -LiteralPath $setup -Destination $SetupExe
    [IO.File]::WriteAllText($cachedContract,($setupContractVersion+'|asimov-v1'),$utf8)
  }
  Require-Signature $setup
  $probe=Join-Path $Output 'setup-contract-verified.txt'
  Run-Setup $setup @('--describe-fixed-download',$probe)
  if (!(Test-Path -LiteralPath $probe -PathType Leaf) -or
      ([IO.File]::ReadAllText($probe)).Trim() -cne $setupExpected) {
    throw 'SETUP_PROTOCOL_INCOMPATIBLE: signed Setup does not target the one fixed ECS MSIX.'
  }
  $receipt.setupExe=$setup
  $receipt.setupBytes=(Get-Item -LiteralPath $setup).Length
  $receipt.setupImplementation=if($LocalSetup){'local-framework-asimov-v1'}else{'winhttp-framework-asimov-v2'}
  $receipt.fixedMsixUrl=$setupExpected
  Save-Json $receipt (Join-Path $Output 'result.json')
  if($UpdateSetup){$receipt.setupExplicitUpdate=$true}
  Complete-Checkpoint 'setup'
  }
  return $setup
}
function Read-CommitBuild([string]$Path) {
  $last=$null
  if(Test-Path -LiteralPath $Path){
    foreach($line in [IO.File]::ReadLines($Path)){
      if(!$line.Trim()){continue}
        if($line -notmatch '"type"\s*:\s*"build"\s*[,}]'){continue}
      try{$entry=$line | ConvertFrom-Json -ErrorAction Stop}catch{continue}
      if($entry.PSObject.Properties['type'] -and $entry.type -eq 'build'){$last=$entry.build}
    }
  }
  return $last
}
function Remove-OlderDeploymentRuns([string]$KeepRun, [string]$CurrentVersion) {
  $root=[IO.Path]::GetFullPath('D:\repo\deliverables\work').TrimEnd('\')
  $keep=[IO.Path]::GetFullPath($KeepRun).TrimEnd('\')
  $current=[Version]$CurrentVersion
  $active=@(Get-CimInstance Win32_Process | Where-Object {$_.ExecutablePath} | ForEach-Object {$_.ExecutablePath})
  $removed=@();$deferred=@()
  foreach($directory in (Get-ChildItem -LiteralPath $root -Directory)){
    if($directory.Name -cnotmatch '^ecs-deploy-\d{8}-\d{6}-[a-f0-9]{8}$'){continue}
    $path=[IO.Path]::GetFullPath($directory.FullName).TrimEnd('\')
    if($path -ieq $keep){continue}
    if([IO.Path]::GetDirectoryName($path) -ine $root){throw 'Deployment cleanup path escaped work root.'}
    $build=Read-CommitBuild (Join-Path $path 'checkpoints.jsonl')
    $version=if($build){[Version]$build.version}else{$null}
    $cancelled=$build -and $build.stage -eq 'cancelled'
    if(!$version -and (Test-Path -LiteralPath (Join-Path $path 'result.json'))){
      $old=Get-Content -LiteralPath (Join-Path $path 'result.json') -Raw | ConvertFrom-Json
      if($old.PSObject.Properties['plan'] -and $old.plan.PSObject.Properties['version']){$version=[Version]$old.plan.version}
    }
    if(!$cancelled -and (!$version -or $version -ge $current)){continue}
    $log=Join-Path $path 'checkpoints.jsonl'
    $checkpoint=if(Test-Path -LiteralPath $log){Read-RunCheckpoint $log}else{$null}
    if(!$cancelled -and (!$checkpoint -or !$checkpoint.completed.PSObject.Properties['finished'])){continue}
    if(Test-Path -LiteralPath (Join-Path $path 'source')){$deferred+=@($path);continue}
    if(@(Get-ChildItem -LiteralPath $path -Recurse -File -Include '*.msix','*.exe','*.pfx').Count){$deferred+=@($path);continue}
    if(($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
      (Test-Path -LiteralPath (Join-Path $path '.git')) -or
      @($active | Where-Object {$_.StartsWith($path+'\',[StringComparison]::OrdinalIgnoreCase)}).Count){
      $deferred+=@($path);continue
    }
    try{
      if(@(Get-ChildItem -LiteralPath $path -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){
        $deferred+=@($path);continue
      }
      Report-Status ('Removing obsolete local deployment run, including its backups and artifacts: '+$path)
      Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
      $removed+=@($path)
    }catch{Report-Status ('Local cleanup deferred: '+$path+'; '+$_.Exception.Message);$deferred+=@($path)}
  }
  return @{removed=$removed;deferred=$deferred}
}
function Initialize-RustPreflight {
  $rustRoot='D:\repo\deliverables\cache\rust\toolchain'
  $rustBin=Join-Path $rustRoot 'bin'
  if(!(Test-Path -LiteralPath (Join-Path $rustBin 'rustc.exe')) -or
     !(Test-Path -LiteralPath (Join-Path $rustBin 'cargo.exe')) -or
     !(Test-Path -LiteralPath (Join-Path $rustRoot 'lib\rustlib\x86_64-pc-windows-msvc\lib'))){
    $pattern='D:\repo\deliverables\work\ecs-deploy-*\desktop-job\home\AppData\Local\puccinialin\puccinialin\Cache\rustup\toolchains\stable-x86_64-pc-windows-msvc'
    $candidates=@(Get-Item -Path $pattern -ErrorAction SilentlyContinue | Where-Object {
      (Test-Path -LiteralPath (Join-Path $_.FullName 'bin\rustc.exe')) -and
      (Test-Path -LiteralPath (Join-Path $_.FullName 'bin\cargo.exe')) -and
      (Test-Path -LiteralPath (Join-Path $_.FullName 'lib\rustlib\x86_64-pc-windows-msvc\lib'))
    } | Sort-Object LastWriteTime -Descending)
    if(!$candidates.Count){throw 'No downloaded Windows x64 Rust toolchain available. Preflight will not install Rust automatically.'}
    Report-Status 'Preflight: copying the existing Rust toolchain to a fixed short cache path; no download.'
    New-Item -ItemType Directory -Path $rustRoot -Force | Out-Null
    & "$env:WINDIR\System32\robocopy.exe" $candidates[0].FullName $rustRoot /E /XJ /R:1 /W:1 /NFL /NDL /NP | Out-Host
    if($LASTEXITCODE -gt 7){throw 'Rust toolchain cache copy failed.'}
  }
  $cargoHome='D:\repo\deliverables\cache\cargo'
  New-Item -ItemType Directory -Path $cargoHome -Force | Out-Null
  $env:CARGO_HOME=$cargoHome
  $env:RUSTC=Join-Path $rustBin 'rustc.exe'
  $env:RUSTDOC=Join-Path $rustBin 'rustdoc.exe'
  $env:PATH=$rustBin+';'+$env:PATH
  $rustVersion=& $env:RUSTC -vV
  if($LASTEXITCODE -ne 0 -or !($rustVersion -contains 'host: x86_64-pc-windows-msvc')){throw 'Cached Rust compiler is invalid or not Windows x64 MSVC.'}
  $cargoVersion=& (Join-Path $rustBin 'cargo.exe') --version
  if($LASTEXITCODE -ne 0){throw 'Cached Cargo cannot start.'}
  $rustVersion | Out-Host;$cargoVersion | Out-Host
  Report-Status "Preflight: existing Rust ready; short Cargo cache: $cargoHome"
}
function Initialize-ElectronPreflight {
  if($Commit){
    $json=& git.exe -C $Repository show ($Commit+':apps/desktop/package.json')
    if($LASTEXITCODE -ne 0){throw 'Cannot read Electron version at requested commit.'}
    $package=($json -join "`n") | ConvertFrom-Json
  } else {$package=Get-Content -LiteralPath (Join-Path $Repository 'apps\desktop\package.json') -Raw | ConvertFrom-Json}
  $version=[string]$package.devDependencies.electron
  if($version -cnotmatch '^\d+\.\d+\.\d+$'){throw 'Electron version must be exactly pinned.'}
  $name="electron-v$version-win32-x64.zip"
  $base="https://github.com/electron/electron/releases/download/v$version"
  $cache=Join-Path $BuildCache 'packager\electron'
  New-Item -ItemType Directory -Path $cache -Force | Out-Null
  $zip=Join-Path $cache $name
  $sums=Join-Path $cache "SHASUMS256-v$version.txt"
  $pattern='(?m)^([a-fA-F0-9]{64})[ \t]+\*?'+[Regex]::Escape($name)+'\r?$'
  $legacy=Join-Path $cache 'SHASUMS256.txt'
  if(!(Test-Path -LiteralPath $sums) -and (Test-Path -LiteralPath $legacy)){
    if([Regex]::IsMatch([IO.File]::ReadAllText($legacy),$pattern)){Copy-Item -LiteralPath $legacy -Destination $sums}
  }
  if(!(Test-Path -LiteralPath $sums)){
    Report-Status "Preflight: downloading official Electron $version checksum list."
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','180',
      '--output',($sums+'.download'),"$base/SHASUMS256.txt")
    if(![Regex]::IsMatch([IO.File]::ReadAllText($sums+'.download'),$pattern)){throw 'Official Electron checksum entry missing.'}
    Move-Item -LiteralPath ($sums+'.download') -Destination $sums -Force
  }
  $match=[Regex]::Match([IO.File]::ReadAllText($sums),$pattern)
  if(!$match.Success){throw 'Cached Electron checksum list does not match requested version.'}
  $expected=$match.Groups[1].Value.ToLowerInvariant()
  if(!(Test-Path -LiteralPath $zip) -or (Hash $zip) -cne $expected){
    Report-Status "Preflight: downloading Electron $version; absent or invalid ZIP cache."
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','900',
      '--output',($zip+'.download'),"$base/$name")
    if((Hash ($zip+'.download')) -cne $expected){throw 'Electron archive failed official SHA-256 verification.'}
    Move-Item -LiteralPath ($zip+'.download') -Destination $zip -Force
  }
  $preload=Join-Path $cache "verified-electron-cache-v$version.mjs"
  $source=@'
import fs from 'node:fs';
import {Readable} from 'node:stream';
const files=new Map([[__SUM_URL__,__SUM_FILE__],[__ZIP_URL__,__ZIP_FILE__]]);
const original=globalThis.fetch;
globalThis.fetch=async(input,options)=>{
 const url=input instanceof Request?input.url:String(input),file=files.get(url);
 if(!file)return original(input,options);
 const method=options?.method||(input instanceof Request?input.method:'GET');
 if(method!=='GET'&&method!=='HEAD')return original(input,options);
 console.error('[VERIFIED LOCAL CACHE]',file);
 return new Response(method==='HEAD'?null:Readable.toWeb(fs.createReadStream(file)),{
  status:200,headers:{'content-length':String(fs.statSync(file).size),'content-type':file.endsWith('.zip')?'application/zip':'text/plain'}
 });
};
'@
  foreach($entry in @(@('__SUM_URL__',"$base/SHASUMS256.txt"),@('__SUM_FILE__',$sums),@('__ZIP_URL__',"$base/$name"),@('__ZIP_FILE__',$zip))){
    $source=$source.Replace($entry[0],(ConvertTo-Json -InputObject $entry[1] -Compress))
  }
  [IO.File]::WriteAllText($preload,$source,$utf8)
  Run $Node @('--check',$preload)
  $option='--import='+([Uri]$preload).AbsoluteUri
  $existing=[Environment]::GetEnvironmentVariable('NODE_OPTIONS','Process')
  if(!$existing -or !$existing.Contains($option)){$env:NODE_OPTIONS=($existing+' '+$option).Trim()}
  Write-RunEvent @{type='preflight-electron';at=[DateTime]::UtcNow.ToString('o');version=$version;sha256=$expected;status='PASS'}
  Report-Status "Preflight: Electron $version official SHA-256 verified; Node will reuse the local archive and checksums."
}
function Initialize-BuildPreflight {
  param([switch]$NativeOnly)
  if([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT){throw 'Build preflight requires native Windows.'}
  Report-Status 'Preflight: initializing Windows PATH and VS 2022 x64 SDK environment.'
  $systemPaths=@($PSHOME,"$env:WINDIR\System32","$env:WINDIR","$env:WINDIR\System32\Wbem",
    "$env:WINDIR\System32\OpenSSH","$env:WINDIR\System32\WindowsPowerShell\v1.0",'D:\Program Files\Git\cmd')
  $vs='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
  $sdk='10.0.26100.0'
  if(!(Test-Path -LiteralPath $vs)){throw 'VS 2022 BuildTools is missing; preflight does not install tools.'}
  $command='set "PATH='+($systemPaths -join ';')+'" && call "'+$vs+'" -arch=x64 -host_arch=x64 -winsdk='+$sdk+' && set'
  $lines=& "$env:WINDIR\System32\cmd.exe" /d /s /c $command
  if($LASTEXITCODE -ne 0){
    $lines | Where-Object {$_ -match '^\[ERROR'} | Out-Host
    throw 'VS/SDK initialization failed. Build stopped before dependency preparation.'
  }
  foreach($line in $lines){
    $separator=$line.IndexOf('=');if($separator -le 0){continue}
    $name=$line.Substring(0,$separator)
    if($name -match '^(PATH|INCLUDE|LIB|LIBPATH|WindowsSDK.*|UniversalCRT.*|UCRTVersion|VCTools.*|VSINSTALLDIR|VCINSTALLDIR|VSCMD_ARG_.*)$'){
      [Environment]::SetEnvironmentVariable($name,$line.Substring($separator+1),'Process')
    }
  }
  if(!$env:WindowsSdkDir){throw 'VS did not supply WindowsSdkDir.'}
  foreach($relative in @("Include\$sdk\um\winsdkver.h","Include\$sdk\shared\sdkddkver.h",
    "Include\$sdk\ucrt\stdio.h","Lib\$sdk\um\x64\gdi32.lib","Lib\$sdk\ucrt\x64\ucrt.lib")){
    $path=Join-Path $env:WindowsSdkDir $relative
    $stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
    $stream.Dispose()
  }
  foreach($tool in @($BuildPython,$Node,$SignTool)){if(!(Test-Path -LiteralPath $tool -PathType Leaf)){throw "Required existing tool missing: $tool"}}
  $null=Get-Command git.exe,curl.exe,tar.exe -ErrorAction Stop
  # Fail before npm if MSBuild itself cannot recognize the selected SDK. This
  # target only checks SDK discovery; it does not compile or install anything.
  $probe=Join-Path $Output 'sdk-preflight.vcxproj'
  $project=@'
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup><Configuration>Release</Configuration><Platform>x64</Platform><WindowsTargetPlatformVersion>10.0.26100.0</WindowsTargetPlatformVersion></PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Label="Configuration"><ConfigurationType>Utility</ConfigurationType><PlatformToolset>v143</PlatformToolset></PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
'@
  [IO.File]::WriteAllText($probe,$project,$utf8)
  $msbuild=Join-Path $env:VSINSTALLDIR 'MSBuild\Current\Bin\MSBuild.exe'
  try {Run $msbuild @($probe,'/nologo','/v:minimal','/t:_CheckWindowsSDKInstalled')}
  catch {
    & $msbuild $probe /nologo /getProperty:WindowsSdkDir,WindowsSdkDir_10,TargetPlatformVersion,WindowsSDKInstalled | Out-Host
    throw 'Preflight failed: MSBuild cannot recognize the SDK despite VS initialization. No npm install or build was started.'
  }
  # Python wheels invoke link.exe directly, including its rc.exe/mt.exe children.
  # SDK file discovery alone does not prove that these tools are executable.
  $null=Get-Command cl.exe,link.exe,rc.exe,mt.exe -ErrorAction Stop
  $nativeProbe=Join-Path $Output 'native-toolchain-preflight'
  New-Item -ItemType Directory -Path $nativeProbe -Force | Out-Null
  $probeC=Join-Path $nativeProbe 'probe.c'
  $probeObj=Join-Path $nativeProbe 'probe.obj'
  $probeExe=Join-Path $nativeProbe 'probe.exe'
  [IO.File]::WriteAllText($probeC,'int main(void) { return 0; }',[Text.Encoding]::ASCII)
  Run 'cl.exe' @('/nologo','/c',$probeC,('/Fo'+$probeObj))
  Run 'link.exe' @('/nologo','/MANIFEST:EMBED',('/OUT:'+$probeExe),$probeObj)
  Run $probeExe @()
  Report-Status 'Preflight: native compile, SDK resource/manifest linking and execution passed.'
  Initialize-RustPreflight
  if($NativeOnly){return}

  $cacheRoot='D:\repo\deliverables\cache\node-pre-gyp\get-windows'
  $archive=Join-Path $cacheRoot 'v9.3.0\napi-9-win32-unknown-x64.tar.gz'
  New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force | Out-Null
  $valid=$false
  if((Test-Path -LiteralPath $archive) -and (Test-Path -LiteralPath ($archive+'.sha256'))){
    $valid=(Hash $archive) -ieq ([IO.File]::ReadAllText($archive+'.sha256')).Trim()
  }
  if(!$valid){
    Report-Status 'Preflight: downloading official get-windows 9.3.0 Windows x64 N-API 9 archive.'
    Run 'curl.exe' @('--fail','--location','--retry','3','--retry-delay','2','--connect-timeout','15','--max-time','180',
      '--output',($archive+'.download'),'https://github.com/sindresorhus/get-windows/releases/download/v9.3.0/napi-9-win32-unknown-x64.tar.gz')
    $entries=& tar.exe -tzf ($archive+'.download')
    if($LASTEXITCODE -ne 0 -or !($entries | Where-Object {$_ -match 'node-get-windows\.node$'})){throw 'Invalid official prebuilt archive.'}
    Move-Item -LiteralPath ($archive+'.download') -Destination $archive -Force
    [IO.File]::WriteAllText(($archive+'.sha256'),(Hash $archive),[Text.Encoding]::ASCII)
  }
  # node-pre-gyp 1.0.11 strips file:// instead of decoding a Windows file URL.
  # Its old url.resolve implementation therefore needs this readable local UNC.
  $mirror='file:////localhost/D$/repo/deliverables/cache/node-pre-gyp/get-windows/'
  $code=@'
const fs=require('node:fs'),url=require('node:url');
const path=url.resolve(url.resolve(process.argv[2],'v9.3.0/'),'napi-9-win32-unknown-x64.tar.gz').slice(7);
const fd=fs.openSync(path,'r');fs.closeSync(fd);console.log('Prebuilt cache readable through node-pre-gyp file path');
'@
  $code | & $Node - $mirror | Out-Host
  if($LASTEXITCODE -ne 0){throw 'Local prebuilt cache is not readable through localhost D$; no permissions were changed.'}
  [Environment]::SetEnvironmentVariable('npm_config_node_get-windows_binary_host_mirror',$mirror,'Process')
  Initialize-ElectronPreflight
  Write-RunEvent @{type='preflight';at=[DateTime]::UtcNow.ToString('o');status='PASS';sdk=$sdk;
    prebuiltArchive=$archive;prebuiltSha256=(Hash $archive)}
  Report-Status 'Build preflight PASS: Windows tools, VS/SDK, Rust, native prebuilt and verified Electron caches ready.'
}
function Remove-ExistingAcceptancePackage {
  # Removing the bundle unregisters its main/resource packages for this user.
  # Do not remove other products, other users' installations, or profile files.
  $bundles=@(Get-AppxPackage -Name $identity -PackageTypeFilter Bundle)
  $packages=if($bundles.Count){$bundles}else{@(Get-AppxPackage -Name $identity)}
  foreach($package in $packages){
    Report-Status ('Uninstalling existing package: '+$package.PackageFullName)
    Remove-AppxPackage -Package $package.PackageFullName -ErrorAction Stop
  }
  if(@(Get-AppxPackage -Name $identity).Count -or @(Get-AppxPackage -Name $identity -PackageTypeFilter Bundle).Count){
    throw 'Existing acceptance package remains registered; installation stopped.'
  }
}
function Build-DesktopCheck {
  if($Commit -cnotmatch '^[a-f0-9]{40}$'){throw '-DesktopOnly requires an exact -Commit.'}
  if($WhatIfPreference){
    [ordered]@{result='DESKTOP_CHECK_PLAN';commit=$Commit;source=$BuildSource;output=$Output;
      compiler='existing scripts/build/desktop.mjs';nativeRuntime='separate full-build qualification';
      signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN'} | ConvertTo-Json
    return
  }
  Initialize-LocalBuildSource $BuildSource $Commit
  Initialize-BuildPreflight -NativeOnly
  $desktopPrepared=Join-Path $Output 'desktop-check-job\prepared.json'
  # Rebuild job-local bindings on every Desktop check; PM owns cache validation.
  Run $BuildPython @('-B',(Join-Path $BuildSource 'scripts\bundles\desktop.py'),'--repo',$BuildSource,
    '--commit',$Commit,'--variant','light','--prepare-only','--work',(Split-Path -Parent $desktopPrepared),'--cache',$BuildCache)
  $receipt=Join-Path $BuildSource 'node_modules\.hermes-node-deps.native-toolchain'
  if(!(Test-Path -LiteralPath $receipt)){throw 'Desktop dependencies are not prepared; use the existing preparation workflow first.'}
  $nativeIdentity=([IO.File]::ReadAllLines($receipt) | Select-Object -Last 1) | ConvertFrom-Json
  if($nativeIdentity -isnot [string] -or $nativeIdentity -cnotmatch '^[a-f0-9]{32}$'){throw 'Unsupported native-input receipt; no fallback attempted.'}
  $nativeInputs=Join-Path $BuildCache ('native\node\'+$nativeIdentity)
  if(!(Test-Path -LiteralPath $nativeInputs -PathType Container)){throw 'Prepared native Desktop inputs are unavailable.'}
  if(!(Test-Path -LiteralPath $Output)){New-Item -ItemType Directory -Path $Output | Out-Null}
  $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
  Set-ReleaseEnvironment 'HERMES_BUILD_COMMIT' $Commit
  Set-ReleaseEnvironment 'HERMES_DESKTOP_VARIANT' ''
  Set-ReleaseEnvironment 'HERMES_PAYLOAD_TAG' ''
  Set-ReleaseEnvironment '_HERMES_CHANNEL_REQUEST_JSON' ''
  $code=@'
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const [source,output,nativeDeps,commit,tooling]=process.argv.slice(1);
const stamps=await import(pathToFileURL(path.join(source,'apps/desktop/scripts/write-build-stamp.mjs')));
const resolved=stamps.resolveStamp({repoRoot:source,execFn:(argv,opts)=>{
  const started=Date.now();
  try{return execFileSync(argv[0],argv.slice(1),{encoding:'utf8',stdio:['ignore','pipe','pipe'],timeout:5000,...opts}).trim();}
  catch(error){console.error(`Desktop identity ${argv.slice(1).join(' ')} failed after ${Date.now()-started}ms: ${String(error.stderr||error.message).trim()}`);throw error;}
}});
if(resolved.commit!==commit || resolved.dirty)throw Error('Desktop source differs from the requested clean commit');
const built=stamps.buildStampPayload(resolved);
const inputs=path.join(output,'inputs');stamps.writeDesktopStamp(inputs,built);
const app=path.join(output,'apps/desktop');
const {buildDesktop}=await import(pathToFileURL(path.join(source,'scripts/build/desktop.mjs')));
await buildDesktop({source,out:path.join(app,'dist'),icons:source,stamp:path.join(inputs,'install-stamp.json'),nativeDeps});
fs.copyFileSync(path.join(source,'apps/desktop/package.json'),path.join(app,'package.json'));
// Stage the development identity module and its transitive dependencies in the final app tree.
const {stageDesktopIdentityClosure}=await import(pathToFileURL(path.join(tooling,'apps/desktop/scripts/stage-desktop-identity.mjs')));
const identityClosure=stageDesktopIdentityClosure({source,app});
const {checkDesktopLaunch}=await import(pathToFileURL(path.join(tooling,'apps/desktop/scripts/assert-dist-built.mjs')));
const launch=checkDesktopLaunch(path.join(app,'dist'));if(!launch.ok)throw Error(launch.error);
fs.writeFileSync(path.join(output,'result.json'),JSON.stringify({result:'DESKTOP_CHECK_BUILT',commit,appRoot:app,stamp:built,identityClosure:identityClosure.copied.map(file=>path.relative(output,file)),scope:'Desktop UI check; full bundled-runtime qualification remains separate',signing:'NOT_RUN',installation:'NOT_RUN',publication:'NOT_RUN'},null,2));
'@
  Report-Status 'Compiling the same Desktop source with prepared native inputs; no runtime preparation, package signing or installation.'
  Push-Location -LiteralPath $BuildSource
  try { Run $Node @('--input-type=module','-e',$code,$BuildSource,$Output,$nativeInputs,$Commit,$Repository) }
  finally { Pop-Location }
  Report-Status 'Desktop check build complete; this is not full bundled application qualification.'
}

function Assert-MsixWithoutUvCache([string]$Path) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $archive=[IO.Compression.ZipFile]::OpenRead($Path)
  try {
    foreach($entry in $archive.Entries){
      if($entry.FullName -match '(?i)(^|[/\\])uv-cache([/\\]|$)'){
        throw "MSIX contains forbidden uv-cache content: $($entry.FullName). Repackage with current R: tooling; shared caches must not be deleted."
      }
    }
  } finally {$archive.Dispose()}
}
function Initialize-CommitInputs($record) {
  $source=[string]$record.source
  Initialize-BuildPreflight -NativeOnly:($BuildMode -ne 'Prod')
  if(!(Test-Path -LiteralPath $record.request)){
    Run $BuildPython @('-B',(Join-Path $source 'scripts\releases\semantier_windows.py'),
      '--source',$source,'--commit',$Commit,'--controller',$Commit,'--version',$record.version,
      '--base',$PublicBase,'--output',$record.request)
  }
  $request=Get-Content -LiteralPath $record.request -Raw | ConvertFrom-Json
  if($request.commit -cne $Commit -or $request.version -cne $record.version -or $request.publicBase -cne $PublicBase){throw 'Saved build request mismatch.'}
  $builder=Join-Path $source 'scripts\bundles\desktop.py'
  if(!(Test-Path -LiteralPath $record.prepared)){
    $payload=Join-Path $source 'apps\desktop\build\agent-payload'
    $retention=$payload+'.retention.json'
    if((Test-Path -LiteralPath $payload -PathType Container) -and
      @(Get-ChildItem -LiteralPath $payload -Force).Count -gt 0 -and
      !(Test-Path -LiteralPath $retention)) {
      # A failed cold preparation cannot publish its retention receipt. Preserve
      # that unadmitted output instead of forging ownership or deleting caches.
      $resolvedPayload=[IO.Path]::GetFullPath($payload)
      $expectedPayload=[IO.Path]::GetFullPath((Join-Path $source 'apps\desktop\build\agent-payload'))
      $savedPayload=[IO.Path]::GetFullPath((Join-Path $Output ('interrupted-payload-'+[Guid]::NewGuid().ToString('N'))))
      $runRoot=[IO.Path]::GetFullPath($Output).TrimEnd('\')+'\'
      if($resolvedPayload -cne $expectedPayload -or !$savedPayload.StartsWith($runRoot,[StringComparison]::OrdinalIgnoreCase) -or
        ((Get-Item -LiteralPath $payload -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Interrupted payload preservation path is unsafe; inspect manually.'
      }
      Report-Status ('Preserving unadmitted interrupted payload at '+$savedPayload+'; npm dependencies and shared caches remain in place.')
      Move-Item -LiteralPath $resolvedPayload -Destination $savedPayload
    }
    Report-Status 'Preparing dependencies through the existing native builder; its receipts determine cache HIT/MISS.'
    Run $BuildPython @('-B',$builder,'--repo',$source,'--channel-request',$record.request,'--variant','bundled',
      '--prepare-only','--work',(Split-Path -Parent $record.prepared),'--cache',$BuildCache)
  }
}
function Build-CommitPackage {
  if($Commit -cnotmatch '^[a-f0-9]{40}$'){throw '-Commit requires a complete 40-character Git commit.'}
  $resolved=& git.exe -C $Repository rev-parse --verify ($Commit+'^{commit}')
  if($LASTEXITCODE -ne 0 -or $resolved -cne $Commit){throw 'Requested commit is not available locally; fetch it first.'}
  if($WhatIfPreference){
    [ordered]@{commit=$Commit;builder='existing scripts/bundles/desktop.py --prepare-only / --prepared';
      version='reserve next version using existing next-windows-build-version.ps1; reuse on resume';
      source=$script:SourceBinding;localBuildSource=$BuildSource;credentialsEnv=$CredentialsEnv;python=$BuildPython;node=$Node;
      mode=$BuildMode;verification=($(if($BuildMode -eq 'Beta'){'SKIPPED_BETA'}else{'REQUIRED_PROD'}));cache=$BuildCache;output=$Output;publish=(!$LocalOnly -and !$BuildOnly);unsignedOnly=[bool]$BuildOnly} | ConvertTo-Json
    return
  }
  if(!(Test-Path -LiteralPath $BuildPython -PathType Leaf)){throw 'Existing build Python is unavailable; specify -BuildPython.'}
  if($NewRun -and $script:explicitOutput -and (Test-Path -LiteralPath $Output)){throw '-NewRun requires a fresh -Output directory.'}
  if(!(Test-Path -LiteralPath $Output)){New-Item -ItemType Directory -Path $Output | Out-Null}
  $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
  $record=Read-CommitBuild $script:statusLog
  if($record -and ($record.commit -cne $Commit -or $record.publicBase -cne $PublicBase -or $record.cache -cne $BuildCache -or $record.source -cne $BuildSource)){
    throw 'Saved build configuration differs. Use -NewRun or a fresh -Output.'
  }
  $inputs=([ordered]@{python=(Hash $BuildPython);node=(Hash $Node);
    driver=(Hash (Join-Path $Repository 'scripts\bundles\desktop.py'));
    runner=(Hash (Join-Path $Repository 'apps\desktop\scripts\run-electron-builder.mjs'));
    operator=(Hash $PSCommandPath);lock=(Hash (Join-Path $Repository 'pm\lock.json'))} | ConvertTo-Json -Compress)
  if($record){
    foreach($entry in @{request='channel-request.json';prepared='desktop-job\prepared.json';msix='unsigned.msix'}.GetEnumerator()){
      if($record.($entry.Key) -ine (Join-Path $Output $entry.Value)){throw 'Saved build output escaped the selected run.'}
      Assert-NoReparseAncestor $record.($entry.Key)
    }
    if(!$record.PSObject.Properties['inputs'] -or $record.inputs -cne $inputs){throw 'Saved source/toolchain configuration changed; preserve the reservation and use a fresh run.'}
  }
  if($record -and $record.stage -eq 'cancelled'){
    throw 'This tooling-only version reservation was cancelled. Resume the application source commit and its original run instead.'
  }
  $source=if($record){[string]$record.source}else{$BuildSource}
  $null=Assert-LocalWindowsPath $source
  Initialize-LocalBuildSource $source $Commit
  Reset-StalePreparation (Join-Path $Output 'desktop-job\prepared.json')
  if(!$record){
    $next=& (Join-Path $PSScriptRoot 'next-windows-build-version.ps1') -State $VersionState -SourceCommit $Commit
    $record=[pscustomobject]@{inputs=$inputs;commit=$Commit;version=[string]$next;source=$source;publicBase=$PublicBase;cache=$BuildCache;
      request=(Join-Path $Output 'channel-request.json');prepared=(Join-Path $Output 'desktop-job\prepared.json');
      msix=(Join-Path $Output 'unsigned.msix');sha256='';stage='reserved'}
    Write-RunEvent @{type='build';build=$record;at=[DateTime]::UtcNow.ToString('o')}
    Report-Status ('Reserved version '+$record.version+' for '+$Commit)
  }
  if($record.stage -eq 'built'){
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $package=[IO.Compression.ZipFile]::OpenRead($record.msix)
    try{[xml]$manifest=Read-ZipText $package 'AppxManifest.xml'}finally{$package.Dispose()}
    if($manifest.Package.Identity.Publisher -cne $publisher){
      Report-Status 'Existing unsigned MSIX has the wrong Publisher; repackaging the same version with cached build products.'
      $record.stage='publisher-repackage-required'
      Write-RunEvent @{type='build';build=$record;at=[DateTime]::UtcNow.ToString('o')}
    }
  }
  if($record.stage -eq 'built'){
    if($BuildMode -eq 'Prod' -and (Hash $record.msix) -cne $record.sha256){throw 'Checkpoint unsigned MSIX integrity mismatch.'}
    Assert-MsixWithoutUvCache $record.msix
    if(!$BuildOnly -and !$AcceptanceNodeModules){Initialize-CommitInputs $record}
    Report-Status 'Reusing checkpointed unsigned MSIX; no rebuild.'
    return $record
  }
  Initialize-CommitInputs $record
  # --prepared performs the builder's own source-bound integrity checks. Never
  # delete a rejected preparation or silently reinstall all dependencies.
  Initialize-ReactCompilerCache $source
  Report-Status 'Building one complete unsigned MSIX from validated prepared inputs.'
  # Sign only the finished distribution MSIX later. Do not let builder
  # certificate discovery or inherited Azure credentials sign every payload PE.
  $signingNames=@(Get-ChildItem Env: | Where-Object {$_.Name -match '^(AZURE_|CSC_|WIN_CSC_)'} | ForEach-Object {$_.Name})
  $signingNames=@($signingNames+'CSC_IDENTITY_AUTO_DISCOVERY' | Select-Object -Unique)
  $signingEnvironment=@{}
  foreach($name in $signingNames){
    $signingEnvironment[$name]=[Environment]::GetEnvironmentVariable($name,'Process')
    [Environment]::SetEnvironmentVariable($name,$null,'Process')
  }
  try {
    [Environment]::SetEnvironmentVariable('CSC_IDENTITY_AUTO_DISCOVERY','false','Process')
    # This single-MSIX lane deliberately skips afterPack's full payload walk,
    # PE repair/batch signing and digest refresh. No inner files are modified;
    # final MSIX signature validation and real installed acceptance follow.
    Report-Status 'Packaging only: afterPack payload scan/repair and inner binary signing disabled; signing the final MSIX only.'
    $packagingDriver=Join-Path $Repository 'scripts\bundles\desktop.py'
    $packagingRunner=Join-Path $Repository 'apps\desktop\scripts\run-electron-builder.mjs'
    $toolingHead=& git.exe -C $Repository rev-parse HEAD
    if($LASTEXITCODE -ne 0){throw 'Cannot record packaging tooling commit.'}
    $tooling=[ordered]@{applicationCommit=$Commit;toolingCommit=$toolingHead;
      driverSha256=(Hash $packagingDriver);runnerSha256=(Hash $packagingRunner);
      deployScriptSha256=(Hash $PSCommandPath);version=$record.version;sharedSource=$script:SourceBinding}
    $record | Add-Member -NotePropertyName tooling -NotePropertyValue $tooling -Force
    Write-RunEvent @{type='tooling';at=[DateTime]::UtcNow.ToString('o');tooling=$tooling}
    Report-Status ('Application source '+$Commit+'; packaging tooling '+$toolingHead+'; preserving version '+$record.version)
    Run $BuildPython @('-B',$packagingDriver,'--prepared',$record.prepared,'--variant','bundled',
      '--packaging-controller',$packagingRunner,'--build-mode',$BuildMode,'--',
      '-c.win.sign=false','-c.forceCodeSigning=false','-c.afterPack=null',
      ('-c.msix.publisher='+$publisher),'-c.msix.publisherDisplayName=Semantier Technology',
      ('-c.msix.displayName=Semantier Agent '+$record.version))
  } finally {
    foreach($name in $signingNames){[Environment]::SetEnvironmentVariable($name,$signingEnvironment[$name],'Process')}
  }
  $release=Join-Path $source 'apps\desktop\release'
  $packages=@(Get-ChildItem -LiteralPath $release -File -Filter '*.msix' | Where-Object {$_.Name -like ('*'+$record.version+'*')})
  if($packages.Count -ne 1){throw 'Expected exactly one newly built version-matching MSIX.'}
  if($BuildMode -eq 'Prod'){
  $request=Get-Content -LiteralPath $record.request -Raw | ConvertFrom-Json
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $zip=[IO.Compression.ZipFile]::OpenRead($packages[0].FullName)
  try{
    $stamp=Read-ZipText $zip 'app/resources/install-stamp.json' | ConvertFrom-Json
    [xml]$packageManifest=Read-ZipText $zip 'AppxManifest.xml'
    if($packageManifest.Package.Properties.DisplayName -cne ('Semantier Agent '+$record.version)){
      throw 'New MSIX display name does not match Semantier Agent and the application version.'
    }
  }
  finally{$zip.Dispose()}
  if($stamp.commit -cne $Commit -or $stamp.dirty -or $stamp.channelBuild.buildId -cne $request.buildId){throw 'New MSIX source/build stamp mismatch.'}
  }
  Assert-MsixWithoutUvCache $packages[0].FullName
  Copy-Item -LiteralPath $packages[0].FullName -Destination $record.msix
  $record.sha256=Hash $record.msix;$record.stage='built'
  Write-RunEvent @{type='build';build=$record;at=[DateTime]::UtcNow.ToString('o')}
  return $record
}
function Publish-IndependentSetup {
  if(!$script:explicitOutput){
    $Output=Join-Path 'D:\repo\deliverables\work' ('ecs-setup-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8))
  }
  $script:runBinding=([ordered]@{kind='setup';localSetup=[bool]$LocalSetup;signer=$Thumbprint;target=$SshTarget;root=$RemoteRoot;origin=$PublicBase;forceRebuild=[bool]$UpdateSetup;implementation='winhttp-framework-asimov-v2';contract='verify-remote-expected-sha256-v2'} | ConvertTo-Json -Compress)
  if($WhatIfPreference){[ordered]@{kind='Setup only';output=$Output;update=[bool]$UpdateSetup;stages=$(if($LocalSetup){@('build signed offline Setup','verify signature and adjacent-file contract')}else{@('reuse or build signed Setup','verify signature and download contract','SFTP upload Setup only','activate Setup only')});msix='UNCHANGED';CH2='UNCHANGED'} | ConvertTo-Json;return}
  if(!$NewRun -and !$script:explicitOutput -and (Test-Path -LiteralPath (Split-Path -Parent $Output))){
    foreach($candidate in (Get-ChildItem 'D:\repo\deliverables\work' -Directory -Filter 'ecs-setup-*' | Sort-Object LastWriteTime -Descending)){
      $log=Join-Path $candidate.FullName 'checkpoints.jsonl'
      if(!(Test-Path -LiteralPath $log)){continue}
      $last=Read-RunCheckpoint $log
      if($last -and $last.binding -ceq $script:runBinding){
        if(!$last.completed.PSObject.Properties['finished']){$Output=$candidate.FullName}
        break
      }
    }
  }
  if(!(Test-Path -LiteralPath $Output)){New-Item -ItemType Directory -Path $Output | Out-Null}
  $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
  $script:checkpoints=@{}
  $receipt=[ordered]@{kind='setup';result='IN_PROGRESS';setupExe=$null;publication='NOT_STARTED';MSIX='UNCHANGED';CH2='UNCHANGED';toolingCommit=(& git.exe -C $Repository rev-parse HEAD)}
  if(Test-Path -LiteralPath $script:statusLog){
    $saved=Read-RunCheckpoint $script:statusLog
    if($saved){
      if($saved.binding -cne $script:runBinding){throw 'Setup run configuration changed; use -NewRun.'}
      foreach($property in $saved.completed.PSObject.Properties){$script:checkpoints[$property.Name]=$property.Value}
      foreach($property in $saved.receipt.PSObject.Properties){$receipt[$property.Name]=$property.Value}
    }
  }
  try {
    Report-Status 'Independent Setup release; application version, MSIX and user profile remain unchanged.'
    $setup=Prepare-IndependentSetup
    if($LocalOnly){$receipt.result='SETUP_LOCAL_VERIFIED';Complete-Checkpoint 'finished';return}
    $remoteSetup="$RemoteRoot/releases/installers/semantier-setup-ch1.exe"
    $contract="${PublicBase}/releases/SemantierAgent.msix|verify-remote-expected-sha256-v2"
    if(!$script:checkpoints.ContainsKey('published')){
      if(!$receipt.Contains('remoteStage')){$receipt.remoteStage=$RemoteRoot.Substring(0,$RemoteRoot.LastIndexOf('/'))+'/.semantier-setup-stage-'+[Guid]::NewGuid().ToString('N')}
      $stage=[string]$receipt.remoteStage
      $parent=$RemoteRoot.Substring(0,$RemoteRoot.LastIndexOf('/'))
      if($stage -cnotmatch ('^'+[Regex]::Escape($parent)+'/\.semantier-setup-stage-[a-f0-9]{32}$')){throw 'Invalid Setup staging path.'}
      Complete-Checkpoint 'setup-stage'
      $null=Ssh "set -eu; test ! -L '$stage'; mkdir -p '$stage'; chmod 700 '$stage'"
      Upload $setup "$stage/semantier-setup-ch1.exe"
      $marker=Join-Path $Output 'setup.contract.txt'
      [IO.File]::WriteAllText($marker,$contract,$utf8)
      Upload $marker "$stage/setup.contract.txt"
      $receipt.publication=Ssh "set -eu; test -s '$stage/semantier-setup-ch1.exe'; test ! -L '$remoteSetup'; mkdir -p '$RemoteRoot/releases/installers'; mv -f '$stage/semantier-setup-ch1.exe' '$remoteSetup'; mv -f '$stage/setup.contract.txt' '$remoteSetup.contract.txt'; chmod 644 '$remoteSetup' '$remoteSetup.contract.txt'; rmdir '$stage'; printf SETUP_ONLY_PUBLISH_PASS"
      Complete-Checkpoint 'published'
    }
    $state=Ssh "set -eu; test -s '$remoteSetup'; cat '$remoteSetup.contract.txt'"
    if($state.Trim() -cne $contract){throw 'Published Setup contract verification failed.'}
    $receipt.result='SETUP_PUBLISHED_VERIFIED'
    Complete-Checkpoint 'finished'
  } catch {
    $receipt.result='FAIL';$receipt.error=$_.Exception.Message;Report-Status ('FAILED: '+$_.Exception.Message);throw
  } finally {
    $receipt.finishedAt=[DateTime]::UtcNow.ToString('o')
    Save-Json $receipt (Join-Path $Output 'result.json')
    [IO.File]::WriteAllText((Join-Path $Output 'release-note.md'),"Result: $($receipt.result)`nArtifact: Setup EXE`nSetup: $($receipt.setupExe)`nPublication: $($receipt.publication)`nMSIX: unchanged`nCH2: unchanged`n",$utf8)
    Report-Status ('Finished: '+$receipt.result)
    Write-Host "Result and evidence: $Output"
  }
}
try {
if($Latest){
  if($Commit -or $SetupOnly -or $LocalSetup -or $UpdateSetup -or $PreflightOnly -or $PSBoundParameters.ContainsKey('Msix')){
    throw '-Latest is for a Runtime-pinned commit build only; do not combine with -Commit, Setup, preflight-only or a fixed MSIX.'
  }
  if($WhatIfPreference){$null=Sync-GitHubLatestSource '' '';return}
  $runtime=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
  $Commit=Sync-GitHubLatestSource $runtime $PSCommandPath
}
if($BuildOnly -and (!$Commit -or $SetupOnly -or $LocalSetup -or $UpdateSetup -or $PreflightOnly)){throw '-BuildOnly requires -Commit and cannot be combined with Setup or preflight modes.'}
$script:explicitOutput=$PSBoundParameters.ContainsKey('Output')
Initialize-ReleasePaths
if($Commit){
  if(!$NewRun -and !$script:explicitOutput){
    foreach($candidate in (Get-ChildItem -LiteralPath (Split-Path -Parent $Output) -Directory -Filter 'ecs-deploy-*' | Sort-Object LastWriteTime -Descending)){
      $log=Join-Path $candidate.FullName 'checkpoints.jsonl';$record=Read-CommitBuild $log
      if($record -and $record.commit -ceq $Commit){
        $checkpoint=if(Test-Path -LiteralPath $log){Read-RunCheckpoint $log}else{$null}
        if(!$checkpoint -or !$checkpoint.completed.PSObject.Properties['finished']){$script:Output=$candidate.FullName}
        break
      }
    }
  }
  Set-RunBuildSource
}
if($DesktopOnly){
  if(!$Commit){throw '-DesktopOnly requires an exact -Commit.'}
  if($BuildOnly -or $SetupOnly -or $LocalSetup -or $UpdateSetup -or $PreflightOnly){throw '-DesktopOnly cannot be combined with package/setup/preflight modes.'}
  Build-DesktopCheck
  return
}
$script:explicitOutput=$PSBoundParameters.ContainsKey('Output')
if($LocalSetup){
  $SetupOnly=$true;$UpdateSetup=$true;$LocalOnly=$true
  $SetupExe=Join-Path 'D:\repo\deliverables\cache\ecs-setup' 'semantier-setup-ch0.exe'
}
if(($SetupOnly -or $UpdateSetup) -and ($Commit -or $PreflightOnly)){throw 'Setup release is independent: do not combine -SetupOnly/-UpdateSetup with -Commit or -PreflightOnly.'}
if($PreflightOnly){
  if($WhatIfPreference){Write-Output 'Preflight only: Windows tools, VS/SDK discovery and official prebuilt cache; no version reservation or build.';return}
  if(!(Test-Path -LiteralPath $Output)){New-Item -ItemType Directory -Path $Output | Out-Null}
  $script:statusLog=Join-Path $Output 'checkpoints.jsonl'
  Initialize-BuildPreflight -NativeOnly:($BuildMode -ne 'Prod')
  return
}
if($Commit){
  $built=Build-CommitPackage
  if($WhatIfPreference){$built;return}
  if($BuildOnly){
    $unsigned=[ordered]@{result='UNSIGNED_LOCAL_BUILD';commit=$Commit;version=$built.version;unsignedMsix=$built.msix;
      sha256=$built.sha256;source=$built.source;binding=$script:SourceBinding;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN'}
    [IO.File]::WriteAllText((Join-Path $Output 'result.json'),($unsigned | ConvertTo-Json -Depth 8),$utf8)
    Write-RunEvent @{type='build-only';at=[DateTime]::UtcNow.ToString('o');result=$unsigned}
    Report-Status 'Unsigned local build complete; signing, installation and publication were not run.'
    return
  }
  if(!$AcceptanceNodeModules){$AcceptanceNodeModules=Join-Path $built.source 'node_modules'}
  $Msix=$built.msix;$ExpectedInputSha256=$built.sha256;$SourceCommit=$Commit;$Version=$built.version+'.0'
  $buildRequest=Get-Content -LiteralPath $built.request -Raw | ConvertFrom-Json
  $BuildId=$buildRequest.buildId
}
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'Run on native Windows.' }
if ($SshTarget -cnotmatch '^[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+$' -or
    $RemoteRoot -cnotmatch '^/opt/[A-Za-z0-9_/-]+/releases$' -or $RemoteRoot.Contains('/../') -or
    $PublicBase -cne 'https://download.semantier.cn' -or $BuildId -cnotmatch '^[a-f0-9]{32}$' -or
    $SourceCommit -cnotmatch '^[a-f0-9]{40}$' -or $ExpectedInputSha256 -cnotmatch '^[a-f0-9]{64}$' -or
    $Version -cnotmatch '^\d+\.\d+\.\d+\.\d+$' -or $Thumbprint -cnotmatch '^[A-F0-9]{40}$') {
  throw 'Invalid release/SSH/path configuration. This script publishes CH1 only.'
}
$script:sshOptions = @('-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
if ($IdentityFile) { $script:sshOptions += @('-i',(Resolve-Path -LiteralPath $IdentityFile).Path) }
if($SetupOnly -or $UpdateSetup){Publish-IndependentSetup;return}
$plan = [ordered]@{input=$Msix;version=$Version;sourceCommit=$SourceCommit;ssh=$SshTarget;
  remoteRoot=$RemoteRoot;fixedMsixUrl="$PublicBase/releases/SemantierAgent.msix";
  profileBackup=(Join-Path $Output 'profile-backup');output=$Output;
  mode=$BuildMode;verification=($(if($BuildMode -eq 'Beta'){'SKIPPED_BETA'}else{'REQUIRED_PROD'}));stages=@('verify existing input','sign one MSIX','check independently published Setup',
    'back up profile','install and observe Electron/real PTY','SFTP reput upload with progress to unpublished directory',
    'verify remote hash','atomic replace one stable MSIX',
    $(if($VerifyPublicDownload){'explicit full public H3 download and hash'}else{'lightweight public H3 HEAD; no package body download'}),
    'verify live Landing before old-package cleanup')}
if($BuildMode -eq 'Beta'){$plan.stages=@('build/reuse without release verification','sign MSIX','back up profile','install MSIX','SFTP upload','atomic ECS activation; UNVERIFIED')}
if ($WhatIfPreference) { $plan | ConvertTo-Json -Depth 5; return }
Report-Status 'Validating existing MSIX hash, source, identity and node-pty inputs; no dependency rebuild.'
Assert-MsixWithoutUvCache $Msix
if ($BuildMode -eq 'Prod' -and (!(Test-Path -LiteralPath $Msix -PathType Leaf) -or (Hash $Msix) -cne $ExpectedInputSha256)) { throw 'Existing MSIX integrity check failed; no rebuild attempted.' }
$tools=@($SignTool,$Node)
foreach ($tool in $tools) {
  if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Required existing tool unavailable: $tool" }
}
if($BuildMode -eq 'Prod'){
  if(!$AcceptanceNodeModules){throw 'Prod acceptance requires -AcceptanceNodeModules pointing to existing Windows dependencies, or a -Commit build worktree.'}
  $null=Assert-LocalWindowsPath $AcceptanceNodeModules
  if(!(Test-Path -LiteralPath (Join-Path $AcceptanceNodeModules '@playwright\test'))){throw 'Existing Windows Playwright dependencies unavailable; no npm installation attempted.'}
  & $BuildPython -B -c 'import curl_cffi'
  if($LASTEXITCODE -ne 0){throw 'Prod public checks need an existing Windows Python with curl_cffi; select it with -BuildPython. No dependencies installed.'}
}
if($BuildMode -eq 'Prod'){
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($Msix)
try {
  [xml]$manifest = Read-ZipText $archive 'AppxManifest.xml'
  $stamp = Read-ZipText $archive 'app/resources/install-stamp.json' | ConvertFrom-Json
  if ($manifest.Package.Identity.Name -cne $identity -or $manifest.Package.Identity.Version -cne $Version -or
      $manifest.Package.Identity.Publisher -cne $publisher -or $manifest.Package.Identity.ProcessorArchitecture -cne 'x64' -or
      $stamp.commit -cne $SourceCommit -or $stamp.dirty -or $stamp.channelBuild.buildId -cne $BuildId -or
      $stamp.channelBuild.windowsVersion -cne $Version) { throw 'Package identity/version/source stamp mismatch.' }
  $asar = $archive.GetEntry('app/resources/app.asar')
  if (!$asar) { throw 'Electron ASAR missing.' }
  $reader = New-Object IO.BinaryReader($asar.Open())
  try {
    $null=$reader.ReadBytes(12); $length=$reader.ReadUInt32()
    if ($length -le 0 -or $length -gt 33554432) { throw 'Invalid ASAR index size.' }
    $index = [Text.Encoding]::UTF8.GetString($reader.ReadBytes($length)) | ConvertFrom-Json
    $pty = $index.files.dist.files.node_modules.files.'node-pty'.files
    if (!$pty.'package.json' -or !$pty.lib.files.'index.js') { throw 'node-pty package.json/index.js missing.' }
  } finally { $reader.Dispose() }
  foreach ($name in @('pty.node','conpty.node','conpty_console_list.node','winpty-agent.exe','winpty.dll','conpty/OpenConsole.exe','conpty/conpty.dll')) {
    $entry=$archive.GetEntry('app/resources/app.asar.unpacked/dist/node_modules/node-pty/prebuilds/win32-x64/'+$name)
    if (!$entry -or $entry.Length -le 0) { throw "node-pty native closure missing: $name" }
  }
} finally { $archive.Dispose() }
}
if (!$LocalOnly) {
  $null=Get-Command ssh.exe,sftp.exe -ErrorAction Stop
  Write-Host 'Checking native SSH access (enter the ECS password if prompted).'
  $null=Ssh "set -eu; test -d '$RemoteRoot/releases'; test ! -L '$RemoteRoot'; command -v sha256sum"
}
$script:runBinding=([ordered]@{mode=$BuildMode;inputSha256=$ExpectedInputSha256;source=$SourceCommit;version=$Version;build=$BuildId;
  signer=$Thumbprint;target=$SshTarget;root=$RemoteRoot;origin=$PublicBase;profile=$UserProfile;
  setupContract='verify-remote-expected-sha256-v2'} | ConvertTo-Json -Compress)
if (!$Commit -and !$NewRun -and !$PSBoundParameters.ContainsKey('Output')) {
  $candidates=Get-ChildItem -LiteralPath (Split-Path -Parent $Output) -Directory -Filter 'ecs-deploy-*' |
    Sort-Object LastWriteTime -Descending
  foreach($candidate in $candidates) {
    $savedPath=Join-Path $candidate.FullName 'checkpoints.jsonl'
    if (!(Test-Path -LiteralPath $savedPath)) {continue}
    $saved=Read-RunCheckpoint $savedPath
    if ($saved -and $saved.binding -ceq $script:runBinding) {
      if (!$saved.completed.PSObject.Properties['finished']) {$Output=$candidate.FullName}
      # A successful latest run starts fresh; do not revive older failed runs.
      break
    }
  }
}
if (!(Test-Path -LiteralPath $Output)) {New-Item -ItemType Directory -Path $Output | Out-Null}
$plan.output=$Output;$plan.profileBackup=Join-Path $Output 'profile-backup'
$script:statusLog = Join-Path $Output 'checkpoints.jsonl'
Report-Status "Input verification passed. Run log: $script:statusLog"
$receipt=[ordered]@{result='IN_PROGRESS';mode=$BuildMode;plan=$plan;unsignedSha256=$ExpectedInputSha256;CH2='UNCHANGED';
  reused=@('complete MSIX','npm dependencies','node-pty','Electron inputs','Python runtime/venv','uv cache','.NET/NuGet cache');
  signedMsix=$null;sha256=$null;localAcceptance=$null;publication='NOT_STARTED';
  h3='NOT_RUN';publicReadback='NOT_RUN';cleanup='NOT_STARTED';landingCutover='NOT_VERIFIED'}
if($Commit -and $built.PSObject.Properties['tooling']){$receipt.tooling=$built.tooling}
$script:checkpoints=@{}
$checkpointPath=Join-Path $Output 'checkpoints.jsonl'
$saved=Read-RunCheckpoint $checkpointPath
$rebound=$false
if($saved -and $saved.binding -cne $script:runBinding){
  $build=Read-CommitBuild $checkpointPath
  if(!(Test-UnsignedRepackageResume $saved $script:runBinding $build)){throw 'Run configuration changed. Use -NewRun or a new -Output.'}
  Write-RunEvent @{type='unsigned-repackage-binding';at=[DateTime]::UtcNow.ToString('o');previousBinding=$saved.binding;binding=$script:runBinding;reason='Same source/version; replacement unsigned build; no completed release stages'}
  Report-Status 'Adopting the replacement unsigned MSIX for the same source/version; no completed signing, installation or publication stages exist.'
  $saved=$null;$rebound=$true
}
if($saved) {
  foreach($property in $saved.completed.PSObject.Properties){$script:checkpoints[$property.Name]=$property.Value}
  $previous=$saved.receipt
  foreach($property in $previous.PSObject.Properties){$receipt[$property.Name]=$property.Value}
  Report-Status ('Resuming run; completed checkpoints: '+($script:checkpoints.Keys -join ', '))
} elseif(!$rebound -and (Test-Path -LiteralPath (Join-Path $Output 'result.json'))) {
  # Explicitly selected older runs may predate checkpoints. Adopt only their
  # verified signed artifact; never infer local acceptance or publication.
  $previous=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw | ConvertFrom-Json
  if($previous.unsignedSha256 -cne $ExpectedInputSha256 -or $previous.plan.sourceCommit -cne $SourceCommit){throw 'Older run source mismatch.'}
  if($previous.sha256 -and (Test-Path -LiteralPath (Join-Path $Output 'SemantierAgent.msix')) -and
    (Hash (Join-Path $Output 'SemantierAgent.msix')) -ceq $previous.sha256) {
    $receipt.sha256=$previous.sha256;$script:checkpoints['signed']='imported-verified-hash'
  }
}
$receipt.result='IN_PROGRESS'
Write-RunEvent @{type='checkpoint';stage='resume-ready';binding=$script:runBinding;completed=$script:checkpoints;receipt=$receipt}
try {
  Save-Json $receipt (Join-Path $Output 'result.json')
  $signed=Join-Path $Output 'SemantierAgent.msix'
  Report-Status 'Signing and verifying the distribution MSIX copy.'
  if($script:checkpoints.ContainsKey('signed')) {
    if($BuildMode -eq 'Prod' -and (Hash $signed) -cne $receipt.sha256){throw 'Checkpoint signed MSIX hash changed.'}
    Require-Signature $signed
    Report-Status 'Reusing checkpointed signed MSIX; no re-signing.'
  } else {Sign-Copy $Msix $signed}
  $hash=Hash $signed
  $receipt.signedMsix=$signed; $receipt.sha256=$hash
  $sidecar=$signed+'.sha256'
  [IO.File]::WriteAllText($sidecar, "$hash  $([IO.Path]::GetFileName($signed))`n",[Text.Encoding]::ASCII)
  Complete-Checkpoint 'signed'

  # Setup has its own release run; routine MSIX publication never builds it.
  $remoteSetup="$RemoteRoot/releases/installers/semantier-setup-ch1.exe"
  $setupContractVersion="${PublicBase}/releases/SemantierAgent.msix|verify-remote-expected-sha256-v2"
  if(!$LocalOnly -and $BuildMode -eq 'Prod') {
    $state=Ssh "if test -s '$remoteSetup' && test -f '$remoteSetup.contract.txt'; then cat '$remoteSetup.contract.txt'; else printf MISSING_OR_UNVERIFIED; fi"
    if($state.Trim() -cne $setupContractVersion){throw 'Compatible Setup is missing or unverified. Run -SetupOnly separately, then resume this MSIX run.'}
  }
  $receipt.setupExe=$remoteSetup
  $receipt.setupAction='REUSE_INDEPENDENT_ECS_SETUP'
  # Stop only executables inside this registered package before copying its
  # existing profile. The acceptance app uses isolated state and cannot alter it.
  if(!$script:checkpoints.ContainsKey('local')) {
  $registered=@(Get-AppxPackage -Name $identity)
  Report-Status 'Closing the installed application and backing up the user profile.'
  foreach ($package in $registered) {
    $prefix=$package.InstallLocation.TrimEnd('\')+'\'
    Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) } | Stop-Process -Force
  }
  $backup=Join-Path $Output 'profile-backup'
  $profileFiles=@()
  if($script:checkpoints.ContainsKey('backup')) {
    # Direct assignment avoids nesting the JSON array in Windows PowerShell 5.1.
    $profileFiles=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $Output 'profile-files.json')))
    foreach($file in $profileFiles){
      if($file.relative -isnot [string]){throw 'Invalid profile inventory entry: expected a single relative path.'}
      if(!(Test-Path -LiteralPath (Join-Path $backup $file.relative) -PathType Leaf)){throw 'Checkpoint profile backup file missing.'}
    }
    Report-Status 'Reusing original verified profile backup.'
  } else {
  if (Test-Path -LiteralPath $UserProfile) {
    $profilePath=(Resolve-Path -LiteralPath $UserProfile).Path.TrimEnd('\')
    # Runtime coordination files are regenerated, not persistent user state.
    $runtimeFiles=@('gateway.lock','gateway.pid','gateway_state.json','cron.pid','processes.json')
    Report-Status 'Backing up profile files, excluding transient gateway/PID state; no per-file hashing.'
    $databasePatterns=@('*.db','*.db-wal','*.db-shm','*.db-journal','*.sqlite','*.sqlite-*','*.sqlite3','*.sqlite3-*')
    & robocopy.exe $profilePath $backup /E /XJ /R:1 /W:1 /NFL /NDL /NP /XF @runtimeFiles @databasePatterns | Out-Host
    if ($LASTEXITCODE -gt 7) {
      Report-Status 'Checking for browser processes holding files in the selected profile.'
      if((Stop-ProfileBrowserHolders $profilePath) -gt 0) {
        & robocopy.exe $profilePath $backup /E /XJ /R:1 /W:1 /NFL /NDL /NP /XF @runtimeFiles @databasePatterns | Out-Host
      }
      if($LASTEXITCODE -gt 7) { throw "Profile backup failed (robocopy exit $LASTEXITCODE). Close any remaining file holders and retry." }
    }
    Report-Status 'Taking consistent SQLite snapshots, including committed WAL data; live databases are not byte-hashed.'
    Preserve-Databases 'backup'
    $profileFiles=@(Get-ChildItem -LiteralPath $profilePath -Recurse -File | Where-Object { $_.Name -notin $runtimeFiles -and $_.Name -notmatch '\.(db|sqlite|sqlite3)(-wal|-shm|-journal)?$' } | ForEach-Object {
      $relative=$_.FullName.Substring($profilePath.Length+1)
      if (!(Test-Path -LiteralPath (Join-Path $backup $relative) -PathType Leaf)) {throw "Profile backup file missing: $relative"}
      @{relative=$relative}
    })
    $profileFiles+=@(Get-ChildItem -LiteralPath $backup -Recurse -File | Where-Object { $_.Extension -in @('.db','.sqlite','.sqlite3') } | ForEach-Object {
      @{relative=$_.FullName.Substring($backup.Length+1)}
    })
  }
  Save-Json @($profileFiles) (Join-Path $Output 'profile-files.json')
  Complete-Checkpoint 'backup'
  }
  Report-Status 'Preparing clean local installation; preserving the backed-up user profile.'
  if(!$script:checkpoints.ContainsKey('installed')) {
    $existing=@(Get-AppxPackage -Name $identity)
    $volumes=@(Get-AppxVolume)
    if($existing.Count -eq 1) {
      $installVolume=@($volumes | Where-Object {
        $existing[0].InstallLocation.StartsWith($_.PackageStorePath.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)
      })
    } else {$installVolume=@($volumes | Where-Object {$_.IsSystemVolume})}
    if($installVolume.Count -ne 1 -or $installVolume[0].IsOffline){throw 'Existing/system AppX installation volume is unavailable.'}
    if($BuildMode -eq 'Prod' -and (Test-Path -LiteralPath $backup)){Preserve-Databases 'verify'}
    Remove-ExistingAcceptancePackage
    Complete-Checkpoint 'uninstalled'
    Report-Status ('Installing on explicit AppX volume: '+$installVolume[0].PackageStorePath)
    Add-AppxPackage -Path $signed -Volume $installVolume[0] -ForceApplicationShutdown -ForceUpdateFromAnyVersion
  }
  if($BuildMode -eq 'Prod'){
  $registered=@(Get-AppxPackage -Name $identity)
  if ($registered.Count -ne 1 -or $registered[0].Version.ToString() -cne $Version) {throw 'Installed registration mismatch.'}
  $archive=[IO.Compression.ZipFile]::OpenRead($signed)
  try {
    $stream=$archive.GetEntry('AppxSignature.p7x').Open();$sha=[Security.Cryptography.SHA256]::Create()
    try {$expectedSignature=[BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()}
    finally {$sha.Dispose();$stream.Dispose()}
  } finally {$archive.Dispose()}
  if ((Hash (Join-Path $registered[0].InstallLocation 'AppxSignature.p7x')) -cne $expectedSignature) {throw 'Windows retained different same-version package bytes.'}
  $receipt.installedSignature=$expectedSignature
  Complete-Checkpoint 'installed'
  $binding=Join-Path $Output 'installed-binding.json';$uiOutput=Join-Path $Output ('acceptance-'+[Guid]::NewGuid().ToString('N'))
  Save-Json @{version=([Version]$Version).ToString(3);sourceCommit=$SourceCommit;artifactSha256=$hash;
    executable=(Join-Path $registered[0].InstallLocation 'app\Semantier Agent.exe');output=$uiOutput;
    playwright=(Join-Path $AcceptanceNodeModules '@playwright\test')} $binding
  $harness=@'
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const b=JSON.parse(fs.readFileSync(process.argv[2],'utf8').replace(/^\uFEFF/,''));
const {_electron}=require(b.playwright);fs.mkdirSync(b.output);let app;const errors=[];
(async()=>{try{
 const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>/^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|USERPROFILE|SYSTEMDRIVE|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS|APPDATA|LOCALAPPDATA|TEMP|TMP|PROGRAMFILES|PROGRAMFILES\(X86\))$/i.test(k)));
 app=await _electron.launch({executablePath:b.executable,cwd:path.dirname(b.executable),timeout:120000,
 env:{...env,HERMES_HOME:path.join(b.output,'hermes-home'),HERMES_DESKTOP_USER_DATA_DIR:path.join(b.output,'userdata'),
 HERMES_DESKTOP_IGNORE_EXISTING:'1',HERMES_DESKTOP_SKIP_QUIT_CONFIRM:'1',HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION:'1',HERMES_DESKTOP_CDP_PORT:'off'}});
 app.process().stderr.on('data',x=>{if(String(x).includes('ERR_MODULE_NOT_FOUND'))errors.push(String(x));});
 const facts=await app.evaluate(({app})=>({packaged:app.isPackaged,version:app.getVersion(),executable:process.execPath,platform:process.platform}));
 assert.equal(facts.packaged,true);assert.equal(facts.version,b.version);assert.equal(facts.platform,'win32');assert.equal(path.resolve(facts.executable),path.resolve(b.executable));
 const page=await app.firstWindow({timeout:120000});await page.waitForLoadState('domcontentloaded');assert(await page.isVisible('body'));
 assert(await app.evaluate(({BrowserWindow})=>BrowserWindow.getAllWindows().some(w=>w.isVisible())));
 await page.waitForFunction(async()=>{try{const p=await window.hermesDesktop.api({path:'/api/profiles/active'});return typeof p.current==='string'&&p.current.length>0;}catch{return false;}},null,{timeout:180000,polling:2000});
 const pty=await page.evaluate(async()=>{const api=window.hermesDesktop.terminal,s=await api.start({cols:100,rows:30});let off,timer;
 try{return await new Promise((resolve,reject)=>{let output='';timer=setTimeout(()=>reject(Error('PTY deadline exceeded')),30000);
 off=api.onData(s.id,data=>{output+=typeof data==='string'?data:JSON.stringify(data);if(output.includes('SEMANTIER_REAL_PTY_AMD64_OK')){clearTimeout(timer);resolve({shell:s.shell,output});}});
 (async()=>{try{await api.attach(s.id);await api.write(s.id,'cmd.exe /d /c "echo SEMANTIER_REAL_PTY_%PROCESSOR_ARCHITECTURE%_OK"\r');}catch(e){clearTimeout(timer);reject(e);}})();});}
 finally{clearTimeout(timer);if(off)off();await api.dispose(s.id);}});
 assert.deepEqual(errors,[]);await page.screenshot({path:path.join(b.output,'window.png')});
 fs.writeFileSync(path.join(b.output,'acceptance.json'),JSON.stringify({result:'WINDOW_PTY_PASS',binding:b,facts,pty,errors},null,2));
 }finally{if(app)await app.close();}})().catch(e=>{fs.writeFileSync(path.join(b.output,'failure.txt'),e.stack);process.exitCode=1;});
'@
  $harnessPath=Join-Path $Output 'accept-installed.cjs';[IO.File]::WriteAllText($harnessPath,$harness,$utf8)
  Report-Status 'Opening Electron and testing a real PTY command; backend startup can take several minutes.'
  Run $Node @($harnessPath,$binding)
  $ui=Get-Content -LiteralPath (Join-Path $uiOutput 'acceptance.json') -Raw | ConvertFrom-Json
  if ($ui.result -cne 'WINDOW_PTY_PASS' -or $ui.binding.artifactSha256 -cne $hash) {throw 'Installed UI/PTY evidence mismatch.'}
  foreach ($file in $profileFiles) {
    if($file.relative -match '\.(db|sqlite|sqlite3)$'){continue}
    if (!(Test-Path -LiteralPath (Join-Path $UserProfile $file.relative) -PathType Leaf)) {throw "User profile file missing: $($file.relative)"}
  }
  if($BuildMode -eq 'Prod' -and (Test-Path -LiteralPath $backup)){Preserve-Databases 'verify'}
  $receipt.localAcceptance='LOCAL_SIGNED_MSIX_ELECTRON_PTY_PASS';$receipt.profileBackup=$backup
  Report-Status 'Local Electron, PTY and profile-preservation acceptance passed.'
  Save-Json $receipt (Join-Path $Output 'result.json')
  $receipt.uiReceipt=Join-Path $uiOutput 'acceptance.json'
  Complete-Checkpoint 'local'
  } else {
    $receipt.localAcceptance='SKIPPED_BETA';Complete-Checkpoint 'installed';Complete-Checkpoint 'local'
  }
  } elseif($BuildMode -eq 'Prod') {
    $registered=@(Get-AppxPackage -Name $identity)
    if($registered.Count -ne 1 -or $registered[0].Version.ToString() -cne $Version -or
      (Hash (Join-Path $registered[0].InstallLocation 'AppxSignature.p7x')) -cne $receipt.installedSignature){throw 'Installed package changed since local acceptance; start a new run.'}
    $ui=Get-Content -LiteralPath $receipt.uiReceipt -Raw | ConvertFrom-Json
    if($ui.result -cne 'WINDOW_PTY_PASS' -or $ui.binding.artifactSha256 -cne $hash){throw 'Checkpoint local acceptance mismatch.'}
    Report-Status 'Reusing verified installation and Electron/PTY acceptance.'
  }
  if ($LocalOnly) {
    $receipt.h3='NOT_RUN_LOCAL_ONLY'
    $receipt.publicReadback='NOT_RUN_LOCAL_ONLY'
    $receipt.result=if($BuildMode -eq 'Beta'){'BETA_INSTALLED_UNVERIFIED_NOT_PUBLISHED'}else{'LOCAL_PASS_NOT_PUBLISHED'}
    return
  }

  # One public MSIX and one published signed Setup; neither is version-pinned.
  # Upload outside Caddy's document root; mv on the SAME filesystem is atomic.
  $remoteMsix="$RemoteRoot/releases/SemantierAgent.msix"
  $remoteProof="set -eu; test `$(sha256sum '$remoteMsix' | cut -d' ' -f1) = '$hash'; test -s '$RemoteRoot/releases/installers/semantier-setup-ch1.exe'"
  if($script:checkpoints.ContainsKey('published')) {
    if($BuildMode -eq 'Prod'){$null=Ssh $remoteProof}
    Report-Status 'Remote published hashes still match; skipping upload and activation.'
  } else {
  $remoteChecksum="$remoteMsix.sha256"
  $remoteSetup="$RemoteRoot/releases/installers/semantier-setup-ch1.exe"
  $msixSize=(Get-Item -LiteralPath $signed).Length
  # Compute the POSIX parent in PowerShell without Windows Split-Path (which uses backslashes).
  $posixParent=$RemoteRoot.Substring(0,$RemoteRoot.LastIndexOf('/'))
  $stage="$posixParent/.semantier-stage-$([Guid]::NewGuid().ToString('N'))"
  if($receipt.Contains('remoteStage')) {$stage=[string]$receipt.remoteStage}
  if($stage -cnotmatch ('^'+[Regex]::Escape($posixParent)+'/\.semantier-stage-[a-f0-9]{32}$')) {throw 'Invalid saved staging path.'}
  $minimumFree=$msixSize+67108864
  $prepare=@'
set -eu
root='__ROOT__'
stage='__STAGE__'
minimum='__MIN_FREE__'
test -d "$root/releases" && test ! -L "$root"
if test -d "$stage" && test ! -L "$stage"; then exit 0; fi
test ! -e "$stage"
free=$(df -PB1 "$root" | awk 'NR==2 {print $4}')
test "$free" -ge "$minimum" || { printf 'Insufficient ECS staging space: available=%s required=%s deficit=%s bytes\n' "$free" "$minimum" "$((minimum-free))" >&2; exit 9; }
mkdir -m 700 "$stage"
'@
  $prepare=$prepare.Replace('__ROOT__',$RemoteRoot).Replace('__STAGE__',$stage).Replace('__MIN_FREE__',[string]$minimumFree)
  Report-Status 'Checking ECS staging capacity and preparing the upload directory.'
  $null=Ssh $prepare
  $receipt.remoteStage=$stage
  Save-Json $receipt (Join-Path $Output 'result.json')
  Upload $signed "$stage/SemantierAgent.msix"
  Upload $sidecar "$stage/SemantierAgent.msix.sha256"
  $publish=@'
set -eu
root='__ROOT__'
stage='__STAGE__'
msix="$root/releases/SemantierAgent.msix"
setup="$root/releases/installers/semantier-setup-ch1.exe"
checksum="$root/releases/SemantierAgent.msix.sha256"
test -d "$stage" && test ! -L "$stage"
cd "$stage"
sha256sum -c SemantierAgent.msix.sha256
mkdir -p "$root/releases/installers"
# A reader opening the stable MSIX gets either the old complete inode or the
# new complete inode. There is NO manifest/catalog/version swap.
mv -f SemantierAgent.msix "$msix"
mv -f SemantierAgent.msix.sha256 "$checksum"
chmod 644 "$msix" "$checksum"
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
rmdir "$stage"
printf 'SINGLE_MSIX_ATOMIC_REPLACE_PASS\n'
'@
  $publish=$publish.Replace('__ROOT__',$RemoteRoot).Replace('__STAGE__',$stage)
  if($BuildMode -eq 'Beta'){$publish=($publish -split "`n" | Where-Object {$_ -notmatch 'sha256sum -c'}) -join "`n"}
  Report-Status 'Verifying uploaded hashes on ECS and replacing the public files.'
  $receipt.publication=Ssh $publish
  Save-Json $receipt (Join-Path $Output 'result.json')
  Complete-Checkpoint 'published'
  }
  if($BuildMode -eq 'Beta'){
    $receipt.result='BETA_PUBLISHED_UNVERIFIED';$receipt.h3='SKIPPED_BETA';$receipt.publicReadback='SKIPPED_BETA';$receipt.cleanup='SKIPPED_BETA';Complete-Checkpoint 'finished';return
  }
  $null=Ssh $remoteProof
  Report-Status 'Verifying public H3 status and package size with HEAD; no package body download.'
  Test-PublicMsix
  $receipt.h3='PUBLIC_H3_HEAD_STATUS_SIZE_PASS'
  $receipt.publicReadback='LIGHTWEIGHT_HEAD_ONLY; full public content hash NOT verified; ECS disk SHA256 matches local signed MSIX'
  if($VerifyPublicDownload) {
    if(!$script:checkpoints.ContainsKey('public-readback')) {
      Report-Status 'Explicit full public download requested; this incurs ECS outbound traffic.'
      Test-PublicMsix -FullDownload
      Complete-Checkpoint 'public-readback'
    }
    $receipt.publicReadback='FULL_MSIX_H3_DOWNLOAD_MATCHES_LOCAL_SIGNED_SHA256'
  }
  Save-Json $receipt (Join-Path $Output 'result.json')

  # Cleanup is permitted only after the deployed CN Landing stops depending on
  # legacy catalog.json. Check the ACTUAL production page with a browser.
  # Failed/missing Playwright or an old Landing => retain old catalog/packages.
  $landingReady=$false
  if (!$SkipOldCleanup) {
    $landingProbe=Join-Path $Output 'verify-live-landing.cjs'
    $landingProbeSource=@'
const {chromium}=require(process.argv[2]+'/@playwright/test');
const fs=require('node:fs');
(async()=>{
  const expected='https://download.semantier.cn/releases/installers/semantier-setup-ch1.exe';
  const browser=await chromium.launch({headless:true});
  const observations=[];
  try {
    for(const host of ['https://semantier.cn/','https://www.semantier.cn/']) {
      const page=await browser.newPage();
      const catalogRequests=[];
      page.on('request',r=>{if(/\/(?:manual\/)?catalog\.json(?:\?|$)/i.test(r.url())) catalogRequests.push(r.url());});
      const response=await page.goto(host,{waitUntil:'domcontentloaded',timeout:30000});
      if(!response || !response.ok()) throw Error('Landing HTTP failure: '+host);
      const link=page.locator('a[href="'+expected+'"]');
      await link.first().waitFor({state:'visible',timeout:15000});
      const href=await link.first().getAttribute('href');
      if(new URL(href,page.url()).href!==expected) throw Error('Landing installer link mismatch: '+host);
      if(catalogRequests.length) throw Error('Landing still calls catalog: '+host);
      observations.push({host,url:page.url(),setupHref:expected,visibleFixedLinks:await link.count(),catalogRequests:0});
      await page.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(process.argv[3],JSON.stringify({result:'LIVE_LANDING_FIXED_SETUP_PASS',observations},null,2));
})().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
'@
    [IO.File]::WriteAllText($landingProbe,$landingProbeSource,$utf8)
    $landingEvidence=Join-Path $Output 'landing-cutover.json'
    try {
      Report-Status 'Checking both live Landing hosts before cleanup.'
      Run $Node @($landingProbe,$AcceptanceNodeModules,$landingEvidence)
      $landingProof=Get-Content -LiteralPath $landingEvidence -Raw | ConvertFrom-Json
      if ($landingProof.result -cne 'LIVE_LANDING_FIXED_SETUP_PASS' -or $landingProof.observations.Count -ne 2) {
        throw 'Landing browser evidence incomplete.'
      }
      $landingReady=$true
      $receipt.landingCutover='LIVE_CN_LANDING_CUTOVER_VERIFIED'
    } catch {
      $receipt.landingCutover='NOT_VERIFIED: '+$_.Exception.Message
      $receipt.cleanup='DEFERRED_LANDING_CUTOVER_REQUIRED'
      Write-Warning 'MSIX is published and hash verified; legacy cleanup deferred because live Landing is not yet verified on the fixed Setup link.'
    }
    Save-Json $receipt (Join-Path $Output 'result.json')
  }
  if (!$SkipOldCleanup -and $landingReady -and !$script:checkpoints.ContainsKey('cleanup')) {
    # No more version-specific installer/MSIX publication. Keep ALL old Setup
    # EXEs because existing Landing copies may still link to them. Remove ONLY
    # old Semantier release-package formats, never publisher state or R2.
    $cleanup=@'
set -eu
root='__ROOT__'
( cd "$root/releases" && sha256sum -c SemantierAgent.msix.sha256 )
old="$root/releases/channel-builds"
if test -d "$old"; then
  find "$old" -mindepth 2 -maxdepth 5 -type f \( -name '*.msix' -o -name '*.msixbundle' -o -name '*.zip' -o -name '*.msix.sha256' -o -name '*.msixbundle.sha256' -o -name '*.zip.sha256' \) -print -delete
fi
# The running Landing has been browser-verified to use the fixed Setup link.
# Only now may these legacy discovery pointers be removed.
rm -f "$root/releases/current-release.json" "$root/releases/build.json" \
      "$root/releases/channels/semantier-acceptance.json" \
      "$root/releases/manual/catalog.json"
printf 'OLD_PACKAGE_CLEANUP_PASS\n'
'@
    Report-Status 'Removing obsolete ECS packages and legacy discovery files.'
    $receipt.cleanup=Ssh ($cleanup.Replace('__ROOT__',$RemoteRoot))
    Complete-Checkpoint 'cleanup'
  } elseif ($SkipOldCleanup) {
    $receipt.cleanup='SKIPPED_BY_REQUEST'
  }
  if ((Hash $Msix) -cne $ExpectedInputSha256 -or (Hash $signed) -cne $hash) {
    throw 'Local package changed during publication.'
  }
  $receipt.result=if ($landingReady) { 'MSIX_PUBLISHED_VERIFIED_LANDING_CUTOVER_PASS' } else { 'MSIX_PUBLISHED_VERIFIED_CLEANUP_DEFERRED' }
  if($landingReady -and !$SkipOldCleanup){Complete-Checkpoint 'finished'}
} catch {
  $receipt.result='FAIL';$receipt.error=$_.Exception.Message
  Report-Status ('FAILED: '+$_.Exception.Message)
  throw
} finally {
  $receipt.finishedAt=[DateTime]::UtcNow.ToString('o')
  if($receipt.result -like 'MSIX_PUBLISHED_VERIFIED_*'){
    try{$receipt.tempCleanup=Remove-OlderDeploymentRuns $Output $Version}
    catch{$receipt.tempCleanup=@{error=$_.Exception.Message};Report-Status ('Local temporary-run cleanup deferred: '+$_.Exception.Message)}
  }
  Save-Json $receipt (Join-Path $Output 'result.json')
  Report-Status ('Finished: '+$receipt.result)
  [IO.File]::WriteAllText((Join-Path $Output 'release-note.md'),
    "Result: $($receipt.result)`nVersion: $Version`nSource: $SourceCommit`nSigned MSIX: $($receipt.signedMsix)`nSHA-256: $($receipt.sha256)`nFixed public MSIX: $PublicBase/releases/SemantierAgent.msix`nFixed Setup: $PublicBase/releases/installers/semantier-setup-ch1.exe`nLocal acceptance: $($receipt.localAcceptance)`nPublication: $($receipt.publication)`nH3: $($receipt.h3)`nPublic readback: $($receipt.publicReadback)`nLanding cutover: $($receipt.landingCutover)`nCleanup: $($receipt.cleanup)`nCH2: unchanged`n",$utf8)
  Write-Host "Result and evidence: $Output"
}

} catch {
  if($script:sourceOwner){
    $failurePath=Join-Path $Output 'result.json'
    if(Test-Path -LiteralPath $failurePath){
      $failure=Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
    } else {
      $failure=[pscustomobject]@{commit=$Commit;binding=$script:SourceBinding;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN'}
    }
    $failure | Add-Member -NotePropertyName result -NotePropertyValue 'FAIL' -Force
    $failure | Add-Member -NotePropertyName error -NotePropertyValue $_.Exception.Message -Force
    Save-Json $failure $failurePath
  }
  throw
} finally {
  try {
    if($script:sourceLease -and $script:sourceOwner){
      $cleanup=Remove-OwnedBuildSource
      Write-RunEvent @{type='source-cleanup';source=$BuildSource;result=$cleanup;at=[DateTime]::UtcNow.ToString('o')}
      $resultPath=Join-Path $Output 'result.json'
      if(Test-Path -LiteralPath $resultPath){
        $result=Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        $result | Add-Member -NotePropertyName sourceCleanup -NotePropertyValue $cleanup -Force
        Save-Json $result $resultPath
        [IO.File]::AppendAllText((Join-Path $Output 'release-note.md'),"Result: $($result.result)`nApplication commit: $Commit`nSource cleanup: $cleanup`n",$utf8)
      }
    }
  } finally {
    if($script:sourceLease){$script:sourceLease.Dispose();$script:sourceLease=$null}
    foreach($name in $script:releaseEnvironment.Keys){[Environment]::SetEnvironmentVariable($name,$script:releaseEnvironment[$name],'Process')}
  }
}
