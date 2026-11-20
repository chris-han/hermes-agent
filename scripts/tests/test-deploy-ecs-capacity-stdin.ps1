param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop'
$Operator=[IO.Path]::GetFullPath($Operator)
$Repository=Split-Path (Split-Path $Operator)
$PSScriptRoot=Split-Path $Operator
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$errors)
if($errors){throw ($errors|Out-String)}
$definitions=@()
foreach($name in @('ConvertTo-ReleasePayload','Get-EcsCapacityRecovery')){
 $fn=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
 $definitions+=$fn.Extent.Text
}
$functions=Join-Path (Split-Path $Operator) ('.capacity-stdin-fixture-'+[guid]::NewGuid().ToString('N')+'.ps1')
[IO.File]::WriteAllText($functions,($definitions -join "`n"))
try {
. $functions
$RemoteRoot='/opt/semantier-client-distribution/channel1-storage-root/releases'
$ExpectedChannelSha256='a'*64
$script:received=$null
function Ssh([string]$Command,[string]$InputText){
 if($Command.Length -gt 300 -or !$PSBoundParameters.ContainsKey('InputText')){throw 'Capacity source still travels in native command arguments'}
 $script:received=$InputText
 return '{"result":"CAPACITY_READY"}'
}
$result=Get-EcsCapacityRecovery -Stage '/opt/semantier-client-distribution/channel1-storage-root/.semantier-stage-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' -Size 4MB
if($result.result -cne 'CAPACITY_READY' -or $script:received.Length -le 24000){throw 'Large actual owner payload was not transported through stdin'}
$bytes=[Convert]::FromBase64String($script:received)
$stream=[IO.MemoryStream]::new($bytes)
$gzip=[IO.Compression.GZipStream]::new($stream,[IO.Compression.CompressionMode]::Decompress)
$reader=[IO.StreamReader]::new($gzip)
try{$body=$reader.ReadToEnd()}finally{$reader.Dispose();$gzip.Dispose();$stream.Dispose()}
if(!$body.Contains('expected_previous_sha256') -or !$body.Contains('recover_space') -or !$body.Contains('protected_stage')){throw 'Capacity controls missing from transported owner'}
function ConvertTo-ReleasePayload([byte[]]$Bytes){return ('a'*2097153)}
$rejected=$false;try{Get-EcsCapacityRecovery -Stage '/opt/fixture/stage' -Size 4MB|Out-Null}catch{$rejected=$true}
if(!$rejected){throw 'Oversized stdin payload accepted'}
'CAPACITY_STDIN_PAYLOAD_PASS'

} finally {Remove-Item -LiteralPath $functions -Force}
