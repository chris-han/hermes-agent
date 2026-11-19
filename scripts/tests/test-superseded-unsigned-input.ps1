param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
if(!$IsWindows){throw 'This path/lease fixture requires native Windows'}
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($f.Extent.Text))}
function Require([bool]$Value,[string]$Message){if(!$Value){throw $Message}}
function Reject([scriptblock]$Action){$failed=$false;try{& $Action|Out-Null}catch{$failed=$true};Require $failed 'Unsafe retirement accepted'}
function Require-Signature([string]$Path){if(!(Test-Path $Path -PathType Leaf)){throw 'Fixture missing'}}
$root=Join-Path 'D:\repo\deliverables\work' ('ecs-deploy-20991231-235959-'+[guid]::NewGuid().ToString('N').Substring(0,8))
$utf8=[Text.UTF8Encoding]::new($false)
$null=New-Item (Join-Path $root 'setup-download') -ItemType Directory
try {
 $SupersededBuildOutput=$root;$Output=$root+'-replacement';$ExpectedVersion='0.0.132.0';$Repository='D:\repo\semantier-runtime\hermes-agent';$BuildCache=Join-Path $root 'fixture-cache'
 $null=New-Item $BuildCache -ItemType Directory
 $unsigned=Join-Path $root 'unsigned.msix';$signed=Join-Path $root 'SemantierAgent.msix';$cache=Join-Path $root 'setup-download\SemantierAgent.msix'
 [IO.File]::WriteAllText($unsigned,'unsigned fixture');[IO.File]::WriteAllText($signed,'signed fixture');Copy-Item $signed $cache
 $prior=@{result='STOP';publication='NOT_STARTED';ossFullTransferCount=1;sha256=(Hash $signed);unsignedSha256=(Hash $unsigned);plan=@{version='0.0.131.0';sourceCommit=('a'*40)}}
 $resultPath=Join-Path $root 'result.json';$prior|ConvertTo-Json|Set-Content $resultPath
 $owner=@{kind='native-git-worktree-v1';run=$root;source=(Join-Path $root 'source');repository=$Repository;cache=$BuildCache;commit=('a'*40);pid=99999999;started='2000-01-01T00:00:00.0000000Z'}
 foreach($name in @('run','source','repository','cache')){$owner[$name]=$owner[$name].Replace('\','\\')}
 @{type='source-owner';owner=$owner}|ConvertTo-Json -Compress|Set-Content (Join-Path $root 'checkpoints.jsonl')
 $original=[IO.File]::ReadAllText($resultPath)
 $prior.publication='PUBLISHED';$prior|ConvertTo-Json|Set-Content $resultPath;Reject {Remove-SupersededUnsignedInput $root};[IO.File]::WriteAllText($resultPath,$original)
 $ExpectedVersion='0.0.130.0';Reject {Remove-SupersededUnsignedInput $root};$ExpectedVersion='0.0.132.0'
 [IO.File]::WriteAllText($cache,'bad');Reject {Remove-SupersededUnsignedInput $root};Copy-Item $signed $cache -Force
 [IO.File]::WriteAllText($unsigned,'bad');Reject {Remove-SupersededUnsignedInput $root};[IO.File]::WriteAllText($unsigned,'unsigned fixture')
 $lock=[IO.File]::Open((Join-Path $root '.source.lock'),'OpenOrCreate','ReadWrite','None')
 try {Reject {Remove-SupersededUnsignedInput $root}}finally{$lock.Dispose()}
 Require (Test-Path $unsigned) 'Rejected cleanup deleted unsigned input'
 $event=Remove-SupersededUnsignedInput $root
 Require ($event.state -ceq 'REMOVED' -and !(Test-Path $unsigned)) 'Eligible unsigned input not removed'
 Require ((Hash $signed) -ceq $prior.sha256 -and (Hash $cache) -ceq $prior.sha256) 'Signed/cache bytes lost'
 Require ([IO.File]::ReadAllText($resultPath) -ceq $original) 'Historical failure overwritten'
 Require ((Get-Content (Join-Path $root 'checkpoints.jsonl') -Raw) -match 'INTENT_SIGNED_AND_CACHE_PRESERVED') 'Retirement intent missing'
 $null=Remove-SupersededUnsignedInput $root
 Write-Output 'PASS: guarded unsigned-only retirement, hash/status/version/lease denials and replay; signed/cache/history preserved'
}finally{Remove-Item -LiteralPath $root -Recurse -Force}
