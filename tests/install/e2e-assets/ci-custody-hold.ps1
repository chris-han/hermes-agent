# Holds the existing CI run-custody path and its parents against replacement.
# Files inside remain governed by the CI lock and durable operation records.
param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
if ($Root -ine 'D:\repo\qualification-private') { throw 'Unexpected CI custody root' }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class CICustodyHandle {
 [StructLayout(LayoutKind.Sequential)] public struct Info {
  public uint Attributes, CreationLow, CreationHigh, AccessLow, AccessHigh, WriteLow, WriteHigh;
  public uint VolumeSerial, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
 }
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 public static extern SafeFileHandle CreateFile(string path, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GetFileInformationByHandle(SafeFileHandle h, out Info info);
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 public static extern uint GetFinalPathNameByHandle(SafeFileHandle h, System.Text.StringBuilder path, uint size, uint flags);
}
'@
function Hold-CITool([string]$Path) {
 $full=[IO.Path]::GetFullPath($Path)
 if(!$full.StartsWith('D:\repo\hermes-build\',[StringComparison]::OrdinalIgnoreCase)){throw 'CI tool outside owned build checkout'}
 $parents=@();$cursor=[IO.Path]::GetDirectoryName($full)
 while($cursor){$parents=@($cursor)+$parents;$next=[IO.Path]::GetDirectoryName($cursor);if(!$next -or $next -eq $cursor){break};$cursor=$next}
 $held=@()
 try {
  foreach($path in @($parents)+@($full)){
   $isFile=($path -ieq $full)
   $share=if($isFile){1}else{3}
   $handle=[CICustodyHandle]::CreateFile($path,[uint32]2147483648,[uint32]$share,[IntPtr]::Zero,3,0x02200000,[IntPtr]::Zero)
   if($handle.IsInvalid){throw "CI tool path unavailable: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"}
   $held+=@($handle)
   $info=New-Object CICustodyHandle+Info
   if(-not [CICustodyHandle]::GetFileInformationByHandle($handle,[ref]$info) -or ($info.Attributes -band 0x400) -ne 0 -or
      (($info.Attributes -band 0x10) -ne 0) -eq $isFile){throw 'CI tool path is reparse or wrong type'}
   $final=New-Object System.Text.StringBuilder 32768
   $count=[CICustodyHandle]::GetFinalPathNameByHandle($handle,$final,[uint32]$final.Capacity,0)
   if(!$count -or $count -ge $final.Capacity -or $final.ToString().TrimEnd('\') -ine ('\\?\'+$path).TrimEnd('\')){throw 'CI tool physical path mismatch'}
  }
  return @{path=$full;sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant();handles=$held}
 } catch {foreach($handle in $held){$handle.Dispose()};throw}
}
$handles=@()
try {
 foreach ($path in @('D:\','D:\repo',$Root)) {
  $handle=[CICustodyHandle]::CreateFile($path,[uint32]2147483648,3,[IntPtr]::Zero,3,0x02200000,[IntPtr]::Zero)
  if ($handle.IsInvalid) { throw "CI custody open failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }
  $handles+=@($handle)
  $info=New-Object CICustodyHandle+Info
  if (-not [CICustodyHandle]::GetFileInformationByHandle($handle,[ref]$info)) { throw 'CI custody identity unavailable' }
  if (($info.Attributes -band 0x10) -eq 0 -or ($info.Attributes -band 0x400) -ne 0) { throw 'CI custody path is not a physical directory' }
  $final=New-Object System.Text.StringBuilder 32768
  $count=[CICustodyHandle]::GetFinalPathNameByHandle($handle,$final,[uint32]$final.Capacity,0)
  if (!$count -or $count -ge $final.Capacity -or $final.ToString().TrimEnd('\') -ine ('\\?\'+$path).TrimEnd('\')) { throw 'CI custody final path mismatch' }
 }
 [Console]::Out.WriteLine('CI_CUSTODY_READY')
 [Console]::Out.Flush()
 $line=[Console]::In.ReadLine()
 if($line){
  $parts=$line -split ':',2
  if($parts.Count -ne 2 -or $parts[0] -cnotmatch '^[a-f0-9]{32}$'){throw 'Invalid CI custody command frame'}
  $status='OK'
  try {
   $request=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($parts[1]))|ConvertFrom-Json
   $names=@($request.PSObject.Properties.Name|Sort-Object)
   if(($names -join ',') -cne 'artifact,channelRequest,commit,expectedSid,graph,operation,plan,python,work'){throw 'Invalid CI native dispatch fields'}
   if($request.operation -cne 'native-smoke' -or $request.commit -cnotmatch '^[a-f0-9]{40}$' -or
      !(Get-Command Assert-CISession -ErrorAction SilentlyContinue) -or !$ciContextBefore -or
      $request.expectedSid -cne $ciContextBefore.sid){throw 'Fixed CI session-bound native dispatch required'}
   $dispatchContext=Assert-CISession
   if($dispatchContext.sid -cne $ciContextBefore.sid -or $dispatchContext.session -ne $ciContextBefore.session){throw 'CI session changed before native dispatch'}
   $ciContextBefore=$dispatchContext
   if($request.work -cnotmatch '^D:\\repo\\qualification-private\\client-feedback-ci-([0-9]+-[0-9]+)$'){throw 'Invalid CI work path'}
   $run=$Matches[1];$release='D:\repo\deliverables\SemantierAgent-client-feedback-ci-'+$run
   $artifact=[IO.Path]::GetFullPath([string]$request.artifact)
   $python=[IO.Path]::GetFullPath([string]$request.python)
   $artifactName=[IO.Path]::GetFileName($artifact)
   $validArtifact=if($request.graph){$artifactName -ceq 'SemantierAgent.msixbundle'}else{$artifactName -cmatch '^SemantierAgent-0\.0\.[1-9][0-9]*-win-x64\.msix$'}
   if($request.channelRequest -ine ($release+'\channel-request.json') -or
      (Split-Path -Parent $artifact) -ine ($release+'\signed\distribution') -or
      !$validArtifact -or
      !$python.StartsWith('D:\repo\hermes-build\apps\desktop\build\agent-payload\',[StringComparison]::OrdinalIgnoreCase)) {throw 'CI native input path mismatch'}
   if(([bool]$request.graph -ne [bool]$request.plan) -or ($request.graph -and
      ($request.graph -ine ($release+'\signed\distribution\native-graph.json') -or $request.plan -ine ($release+'\signed\native-layout\layout-plan.json')))){throw 'CI native graph path mismatch'}
   $nodeHold=$null;$pythonHold=$null
   try {
    $nodeHold=Hold-CITool 'D:\repo\hermes-build\.cache\desktop-inputs\tools\node-26.7.0-win32-x64\node.exe'
    $pythonHold=Hold-CITool $python
    $env:HERMES_PYTHON=$pythonHold.path;$env:PATH=(Split-Path -Parent $nodeHold.path)+';'+$env:PATH
    if((Get-Command node.exe -ErrorAction Stop).Source -ine $nodeHold.path){throw 'Selected CI Node changed'}
    $env:RUNNER_TEMP=$request.work
    if(Test-Path -LiteralPath 'D:\repo\hermes-build.distribution.lock'){throw 'Owned Windows builder remains active'}
    New-Item -ItemType Directory -Path $env:RUNNER_TEMP -ErrorAction Stop|Out-Null
    $ciContextBefore.tools=@{node=@{path=$nodeHold.path;sha256=$nodeHold.sha256};python=@{path=$pythonHold.path;sha256=$pythonHold.sha256}}
    $ciContextBefore|ConvertTo-Json -Depth 5 -Compress|Set-Content -LiteralPath (Join-Path $env:RUNNER_TEMP 'ci-context-before.json') -Encoding utf8
    $arguments=@{SemantierAcceptanceHost=$true;Arch='x64';Commit=$request.commit;Artifact=$artifact;ChannelRequest=$request.channelRequest;Work=($request.work+'\work');Out=($request.work+'\out');ExpectedSid=$request.expectedSid}
    if($request.graph){$arguments.NativeGraph=$request.graph;$arguments.NativeLayoutPlan=$request.plan;$arguments.Python=$pythonHold.path}
    . 'D:\repo\hermes-build\tests\install\windows-bundle-smoke.ps1'
    Invoke-CIPrivateBundleSmoke @arguments
    if($LASTEXITCODE -ne 0){throw 'Native smoke failed'}
    Assert-CISession|ConvertTo-Json -Compress|Set-Content -LiteralPath (Join-Path $env:RUNNER_TEMP 'ci-context-after.json') -Encoding utf8
   } finally {if($pythonHold){foreach($h in $pythonHold.handles){$h.Dispose()}};if($nodeHold){foreach($h in $nodeHold.handles){$h.Dispose()}}}
  } catch {
   $status='FAILED'
   [Console]::Error.WriteLine($_.Exception.Message)
  }
  [Console]::Out.WriteLine('CI_CUSTODY_DONE:'+$parts[0]+':'+$status)
  [Console]::Out.Flush()
  $null=[Console]::In.ReadLine()
 }
} finally {
 foreach ($handle in $handles) { $handle.Dispose() }
}
