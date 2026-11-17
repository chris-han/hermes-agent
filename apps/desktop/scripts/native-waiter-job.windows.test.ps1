param([string]$ResultRoot=(Join-Path ([IO.Path]::GetTempPath()) ('native-waiter-job-'+[Guid]::NewGuid().ToString('N'))))
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $ResultRoot){throw 'Use a fresh owned fixture root; preserve previous results'}
$null=New-Item -ItemType Directory -Path $ResultRoot
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'native-package-update.ps1'),[ref]$tokens,[ref]$errors)
$function=@($ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -ceq 'Invoke-NativeWaiterBreakaway'},$false))
if($errors.Count -or $function.Count -ne 1){throw 'Production breakaway function unavailable'}
$definition=$function[0].Extent.Text
Add-Type @'
using System;using System.ComponentModel;using System.Runtime.InteropServices;
public static class WaiterJobFixture {
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr CreateJobObjectW(IntPtr attributes,string name);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job,int kind,IntPtr data,uint size);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool IsProcessInJob(IntPtr process,IntPtr job,out bool member);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool TerminateJobObject(IntPtr job,uint code);
 [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
 public static IntPtr Create(bool allowed) {
  IntPtr job=CreateJobObjectW(IntPtr.Zero,null);if(job==IntPtr.Zero)throw new Win32Exception(Marshal.GetLastWin32Error());
  IntPtr data=Marshal.AllocHGlobal(144);
  try {
   Marshal.Copy(new byte[144],0,data,144);Marshal.WriteInt32(data,16,allowed?0x800:0);
   if(!SetInformationJobObject(job,9,data,144))throw new Win32Exception(Marshal.GetLastWin32Error());
   return job;
  } catch {CloseHandle(job);throw;}finally{Marshal.FreeHGlobal(data);}
 }
}
'@
function Literal([string]$value){return "'"+$value.Replace("'","''")+"'"}
function Await-File([string]$file){$until=[DateTime]::UtcNow.AddSeconds(30);while(!(Test-Path -LiteralPath $file)){if([DateTime]::UtcNow -gt $until){throw ('Fixture handshake timeout: '+[IO.Path]::GetFileName($file))};Start-Sleep -Milliseconds 50}}
$target=Join-Path $ResultRoot 'target.ps1'
@'
param([string]$Result,[string]$Release,[string]$Echo)
Add-Type @"
using System;using System.Runtime.InteropServices;
public static class WaiterConsoleFixture {
 [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr window);
}
"@
@{pid=$PID;birth=(Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o');sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;session=(Get-Process -Id $PID).SessionId;echo=$Echo;consoleVisible=[WaiterConsoleFixture]::IsWindowVisible([WaiterConsoleFixture]::GetConsoleWindow())}|ConvertTo-Json|Set-Content -LiteralPath $Result
$until=[DateTime]::UtcNow.AddSeconds(45)
while(!(Test-Path -LiteralPath $Release)){if([DateTime]::UtcNow -gt $until){exit 3};Start-Sleep -Milliseconds 50}
exit 0
'@|Set-Content -LiteralPath $target
$results=@()
foreach($mode in @('negative-control','survive-job','cancel-tree','refuse-forbidden-job')){
 $root=Join-Path $ResultRoot $mode;$null=New-Item -ItemType Directory -Path $root
 $ready=Join-Path $root 'ready.json';$started=Join-Path $root 'started';$gate=Join-Path $root 'gate';$release=Join-Path $root 'release';$errorFile=Join-Path $root 'error.json';$bootstrap=Join-Path $root 'bootstrap.ps1'
 $body=$definition
 if($mode -ceq 'negative-control'){$body=$body.Replace('0x01000010','0x00000010')}
 $body+="`nSet-Content -LiteralPath $(Literal $started) -Value 'started'`nwhile(!(Test-Path -LiteralPath $(Literal $gate))){Start-Sleep -Milliseconds 50}`n"
 $echo='literal "quoted"; trailing\'
 $body+="try{`$code=Invoke-NativeWaiterBreakaway $(Literal $target) @('-Result',$(Literal $ready),'-Release',$(Literal $release),'-Echo',$(Literal $echo));exit `$code}catch{@{code=`$_.Exception.GetBaseException().NativeErrorCode}|ConvertTo-Json|Set-Content -LiteralPath $(Literal $errorFile);exit 1}`n"
 $body|Set-Content -LiteralPath $bootstrap
 $job=[WaiterJobFixture]::Create($mode -cne 'refuse-forbidden-job');$parent=$null;$child=$null
 try {
  $parent=Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',('"'+$bootstrap+'"')) -WindowStyle Hidden -PassThru
  Await-File $started
  if(![WaiterJobFixture]::AssignProcessToJobObject($job,$parent.Handle)){throw ('Cannot assign exclusively owned fixture job: '+[Runtime.InteropServices.Marshal]::GetLastWin32Error())}
  Set-Content -LiteralPath $gate -Value 'go'
  if($mode -ceq 'refuse-forbidden-job'){
   if(!$parent.WaitForExit(30000)){throw 'Forbidden-job refusal did not complete'}
   Await-File $errorFile
   $failure=Get-Content $errorFile -Raw|ConvertFrom-Json
   if($failure.code -ne 5 -or (Test-Path $ready)){throw 'Forbidden job did not fail closed before child startup'}
   $results+=@{case=$mode;result='PASS';nativeError=$failure.code};continue
  }
  Await-File $ready
  $binding=Get-Content $ready -Raw|ConvertFrom-Json
  if($binding.echo -cne $echo -or $binding.consoleVisible){throw 'Argument quoting or hidden console contract failed'}
  $child=Get-Process -Id $binding.pid
  if([Math]::Abs(($child.StartTime.ToUniversalTime()-[DateTime]::Parse($binding.birth).ToUniversalTime()).TotalSeconds) -gt 1 -or $binding.sid -cne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value -or $binding.session -ne (Get-Process -Id $PID).SessionId){throw 'Fixture worker identity/context changed'}
  $member=$false
  if(![WaiterJobFixture]::IsProcessInJob($child.Handle,$job,[ref]$member)){throw 'Cannot verify fixture job membership'}
  if($member -ne ($mode -ceq 'negative-control')){throw 'Unexpected native breakaway membership'}
  if($mode -ceq 'cancel-tree'){
   & (Join-Path ([Environment]::GetFolderPath('System')) 'taskkill.exe') /PID $parent.Id /T /F | Out-Null
   if($LASTEXITCODE -ne 0 -or !$child.WaitForExit(10000)){throw 'Existing PID-tree cancellation lost breakaway child'}
  }else{
   if(![WaiterJobFixture]::TerminateJobObject($job,1)){throw 'Cannot terminate exclusively owned fixture job'}
   if($mode -ceq 'negative-control'){
    if(!$child.WaitForExit(10000) -or $child.ExitCode -ne 1){throw 'Negative control failed to reproduce job termination'}
   }else{
    Start-Sleep -Milliseconds 300
    if($child.HasExited){throw 'Independent waiter died with original job'}
    Set-Content -LiteralPath $release -Value 'done'
    if(!$child.WaitForExit(10000) -or $child.ExitCode -ne 0){throw 'Independent waiter did not complete normally'}
   }
  }
  $results+=@{case=$mode;result='PASS';wasInOriginalJob=$member;childExit=$child.ExitCode}
 }finally{
  # Only this test's unnamed job and PID/birth-bound child are eligible.
  $null=[WaiterJobFixture]::TerminateJobObject($job,1);$null=[WaiterJobFixture]::CloseHandle($job)
  if($child -and !$child.HasExited){$child.Kill();$null=$child.WaitForExit(5000)}
  if($child){$child.Dispose()};if($parent){$parent.Dispose()}
 }
}
$results|ConvertTo-Json -Depth 4|Set-Content (Join-Path $ResultRoot 'results.json')
$results|ForEach-Object {Write-Output ($_.case+': PASS')}
