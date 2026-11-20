param([string]$Channel=(Join-Path $PSScriptRoot '../deploy-ecs-channel1.ps1'))
$ErrorActionPreference='Stop';$r=[ordered]@{}
$p=Join-Path ([IO.Path]::GetTempPath()) ('native-release-space-'+[guid]::NewGuid().ToString('N')+'.msix')
try {
 $ast=[Management.Automation.Language.Parser]::ParseFile($Channel,[ref]$null,[ref]$null)
 $fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-NativeReleaseRequiredSpace'},$true)
 . ([scriptblock]::Create($fn.Extent.Text))
 $file=[IO.File]::Open($p,'CreateNew');try{$file.SetLength(1MB)}finally{$file.Dispose()}
 if((Get-NativeReleaseRequiredSpace $p) -ne 514MB){throw 'Fixed package space budget incorrect'}
 if((Get-NativeReleaseRequiredSpace $p -Build) -ne 12GB){throw 'Full build guard weakened'}
 $refused=$false;try{Get-NativeReleaseRequiredSpace ($p+'.missing')}catch{$refused=$true};if(!$refused){throw 'Missing input accepted'}
 $r.result='PASS';$r.fixedPackageBytes=514MB;$r.fullBuildBytes=12GB;$r.missingInputRefused=$true;$r.exitCode=0
}catch{$r.error=$_.Exception.Message;$r.exitCode=1}
finally{if(Test-Path $p){Remove-Item -LiteralPath $p};$r|ConvertTo-Json;if($r.exitCode -ne 0){exit 1};exit 0}
