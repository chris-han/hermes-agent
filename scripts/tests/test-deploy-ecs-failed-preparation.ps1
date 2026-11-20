param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$r=[ordered]@{}
$fixture='D:\repo\deliverables\work\ecs-deploy-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8)
$Repository='D:\repo\semantier-runtime\hermes-agent'
$outside=$fixture+'-outside'
try {
 $ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
 foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($fn.Extent.Text))}
 $rev='e2ee4d42c46b126e486473aac84cb6240641eccc'
 New-Item -ItemType Directory -Path $fixture,$outside,(Join-Path $fixture 'source/node_modules/pkg'),(Join-Path $fixture 'source/apps/desktop/build/agent-payload')|Out-Null
 'protect-cache'|Set-Content (Join-Path $outside 'keep.txt')
 'protect-source'|Set-Content (Join-Path $fixture 'source/keep.txt')
 'generated'|Set-Content (Join-Path $fixture 'source/node_modules/pkg/generated.bin')
 'generated'|Set-Content (Join-Path $fixture 'source/apps/desktop/build/agent-payload/generated.bin')
 New-Item -ItemType Junction -Path (Join-Path $fixture 'source/node_modules/cache-link') -Target $outside|Out-Null
 [IO.File]::WriteAllText((Join-Path $fixture '.source.lock'),'')
 $owner=@{kind='native-git-worktree-v1';repository=$Repository;run=$fixture;source=(Join-Path $fixture 'source');commit=$rev;pid=2147483000;started='2026-11-20T00:00:00Z'}
 @{type='source-owner';owner=$owner}|ConvertTo-Json -Compress|Set-Content (Join-Path $fixture 'checkpoints.jsonl')
 $result=@{result='STOP';commit=$rev;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN'}
 $result|ConvertTo-Json|Set-Content (Join-Path $fixture 'result.json')
 function Get-SourceGit([string]$Root,[string[]]$Arguments){if($Arguments[-1] -eq '--show-toplevel'){return $Root};if($Arguments[0] -eq 'check-ignore'){return $Arguments[1]};if($Arguments[0] -eq 'rev-parse'){return 'e2ee4d42c46b126e486473aac84cb6240641eccc'};return ''}
 function Get-CimInstance {param($ClassName) return @([pscustomobject]@{ProcessId=12345;ParentProcessId=0;CreationDate=[datetime]'2026-11-20T00:00:00Z';ExecutablePath='D:\repo\deliverables\cache\desktop-inputs\tools\python-fixture\python.exe';CommandLine='python -m hermes_cli.main'})}
 $value=Remove-FailedPreparationFiles $fixture $rev
 if($value.files -ne 2 -or $value.preservedLinks -ne 1){throw 'Generated-file/link accounting failed'}
 if(!(Test-Path (Join-Path $outside 'keep.txt')) -or !(Test-Path (Join-Path $fixture 'source/keep.txt'))){throw 'Protected files lost'}
 $result.signing='PASS';$result|ConvertTo-Json|Set-Content (Join-Path $fixture 'result.json')
 $refused=$false;try{Remove-FailedPreparationFiles $fixture $rev}catch{$refused=$true}
 if(!$refused){throw 'Signed run accepted'}
 $result.signing='NOT_RUN';$result|ConvertTo-Json|Set-Content (Join-Path $fixture 'result.json')
 $hold=[IO.File]::Open((Join-Path $fixture '.source.lock'),'Open','ReadWrite','None')
 try {$refused=$false;try{Remove-FailedPreparationFiles $fixture $rev}catch{$refused=$true};if(!$refused){throw 'Active lease accepted'}}finally{$hold.Dispose()}
 $release=Join-Path $fixture 'source/apps/desktop/release';New-Item -ItemType Directory -Path (Join-Path $release 'win-unpacked')|Out-Null
 'discard-generated-copy'|Set-Content (Join-Path $release 'win-unpacked/generated.exe')
 $package=Join-Path $release 'SemantierAgent-0.0.999-win-x64.msix'
 Add-Type -AssemblyName System.IO.Compression.FileSystem
 $zip=[IO.Compression.ZipFile]::Open($package,'Create')
 try {
  foreach($pair in @{'AppxManifest.xml'='<Package><Identity Name="Semantier.Agent.Acceptance" Version="0.0.999.0" Publisher="CN=Semantier Technology, O=Semantier Technology" /></Package>';'app/resources/install-stamp.json'='{"commit":"e2ee4d42c46b126e486473aac84cb6240641eccc","dirty":false}'}.GetEnumerator()){
   $writer=[IO.StreamWriter]::new($zip.CreateEntry($pair.Key).Open());try{$writer.Write($pair.Value)}finally{$writer.Dispose()}
  }
 }finally{$zip.Dispose()}
 @{type='build';build=@{commit=$rev;version='0.0.999';stage='reserved'}}|ConvertTo-Json -Compress|Add-Content (Join-Path $fixture 'checkpoints.jsonl')
 [IO.File]::WriteAllBytes((Join-Path $fixture 'unsigned.msix'),[byte[]]@(0x50))
 $hash=Hash $package;$unpacked=Remove-FailedPreparationFiles $fixture $rev -Unpacked
 if((Hash $package) -cne $hash -or $unpacked.retainedUnsignedSha256 -cne $hash){throw 'Complete package changed'}
 if(Test-Path (Join-Path $fixture 'unsigned.msix')){throw 'Incomplete duplicate retained'}
 if(Test-Path (Join-Path $release 'win-unpacked/generated.exe')){throw 'Disposable unpacked copy retained'}
 $r.lockedRunRefused=$true;$r.completePackagePreserved=$true
 $r.result='PASS';$r.generatedFiles=2;$r.cacheJunctionPreserved=$true;$r.signedRunRefused=$true;$r.sharedInterpreterUntouched=$true
}catch{$r.result='FAIL';$r.error=$_.Exception.Message}
finally{
 if(Test-Path (Join-Path $fixture 'source/node_modules/cache-link')){[IO.Directory]::Delete((Join-Path $fixture 'source/node_modules/cache-link'))}
 if(Test-Path $fixture){Remove-Item -LiteralPath $fixture -Recurse -Force}
 if(Test-Path $outside){Remove-Item -LiteralPath $outside -Recurse -Force}
 $r|ConvertTo-Json
 if($r.result -ne 'PASS'){exit 1}
 exit 0
}
