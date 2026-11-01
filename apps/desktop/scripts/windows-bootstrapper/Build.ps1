param([Parameter(Mandatory=$true)][string]$OutputDirectory,[string]$Channel1Descriptor)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a fresh task-owned output directory.'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$assets=Join-Path $PSScriptRoot '..\..\assets'
foreach($name in @('Installer.cs','InstallerUI.cs','SemantierAgent-signing.cer','bootstrapper.manifest')){
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $OutputDirectory
}
Copy-Item -LiteralPath (Join-Path $assets 'icon.png') -Destination (Join-Path $OutputDirectory 'icon.png')
Copy-Item -LiteralPath (Join-Path $assets 'icon.ico') -Destination (Join-Path $OutputDirectory 'icon.ico')
$channel1Source=$null
if($Channel1Descriptor){
    $d=Get-Content -LiteralPath $Channel1Descriptor -Raw | ConvertFrom-Json
    foreach($field in @('origin','zip_path','zip_sha256','cert_sha256','graph_sha256','thumbprint','identity','app_version')){if($d.$field -isnot [string]){throw 'Invalid Channel1 descriptor type.'}}
    if($d.origin -cne 'https://channel1.semantier.com' -or $d.zip_path -cnotmatch '^/releases/channel-builds/[a-f0-9]{32}/[A-Za-z0-9_.-]+\.zip$' -or $d.zip_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.cert_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.graph_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.thumbprint -cnotmatch '^[A-F0-9]{40}$' -or $d.identity -cne 'Semantier.Agent.Acceptance' -or $d.app_version -cnotmatch '^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){3}$' -or ($d.zip_size -isnot [long] -and $d.zip_size -isnot [int]) -or $d.zip_size -le 0){throw 'Invalid Channel1 descriptor.'}
    foreach($part in $d.app_version.Split('.')){if([long]$part -gt 65535){throw 'Invalid application version.'}}
    $members=@($d.members)
    if($members.Count -lt 4 -or $members.Count -gt 32){throw 'Invalid member count.'}
    $names=@{};$lines=@()
    foreach($member in $members){
        if($member.name -isnot [string] -or $member.sha256 -isnot [string]){throw 'Invalid Channel1 member type.'}
        $n=$member.name;$h=$member.sha256
        if($n -cnotmatch '^(native-graph\.json|SemantierAgent-signing\.cer|SemantierAgent\.msixbundle|SemantierAgent-main\.msix|SemantierAgent-runtime[0-9]{4}\.msix)$' -or $h -cnotmatch '^[a-f0-9]{64}$' -or $names.ContainsKey($n)){throw 'Invalid Channel1 member.'}
        $names[$n]=$h;$lines+='        {"'+$n+'","'+$h+'"},'
    }
    foreach($required in @('native-graph.json','SemantierAgent-signing.cer','SemantierAgent.msixbundle','SemantierAgent-main.msix')){if(!$names.ContainsKey($required)){throw 'Required Channel1 member missing.'}}
    if($names['native-graph.json'] -cne $d.graph_sha256 -or $names['SemantierAgent-signing.cer'] -cne $d.cert_sha256){throw 'Descriptor member binding mismatch.'}
    $actualCert=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDirectory 'SemantierAgent-signing.cer')).Hash.ToLowerInvariant()
    if($actualCert -cne $d.cert_sha256){throw 'Embedded certificate mismatch.'}
    $channel1Source=Join-Path $OutputDirectory 'Channel1Pins.cs'
    $content=@(
      'using System;','using System.Collections.Generic;','static class Channel1Pins {',
      ('    public const string Origin = "'+$d.origin+'";'),
      ('    public const string ZipUrl = "'+$d.origin+$d.zip_path+'";'),
      ('    public const string ZipHash = "'+$d.zip_sha256+'";'),
      ('    public const long ZipSize = '+$d.zip_size+'L;'),
      ('    public const string CertHash = "'+$d.cert_sha256+'";'),
      ('    public const string Thumbprint = "'+$d.thumbprint+'";'),
      ('    public const string GraphHash = "'+$d.graph_sha256+'";'),
      ('    public const string Identity = "'+$d.identity+'";'),
      ('    public const string AppVersion = "'+$d.app_version+'";'),
      '    public static readonly Dictionary<string,string> Expected = new Dictionary<string,string>(StringComparer.Ordinal) {'
    )+$lines+@('    };','}')
    [IO.File]::WriteAllLines($channel1Source,$content,[Text.UTF8Encoding]::new($false))
}
$csc=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
Push-Location $OutputDirectory
try {
    $common=@('/nologo','/codepage:65001','/optimize+','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.IO.Compression.dll','/r:System.IO.Compression.FileSystem.dll','/win32icon:icon.ico','/win32manifest:bootstrapper.manifest','/resource:SemantierAgent-signing.cer,SemantierAgent-signing.cer','/resource:icon.png,Semantier.logo.png','/resource:icon.ico,Semantier.icon.ico','Installer.cs','InstallerUI.cs')
    $defines=if($channel1Source){'/define:TESTING,CHANNEL1'}else{'/define:TESTING'}
    if($channel1Source){$common+= 'Channel1Pins.cs'}
    & $csc /target:exe $defines /out:SemantierSetup.Tests.exe @common
    if($LASTEXITCODE -ne 0){throw 'Test compilation failed.'}
    & '.\SemantierSetup.Tests.exe' --self-test
    if($LASTEXITCODE -ne 0){throw 'Bootstrapper self-test failed.'}
    foreach($language in @('en','zh-CN')){
        foreach($state in @('ready','download','verify','trust','install','error','success')){
            & '.\SemantierSetup.Tests.exe' --preview $language $state (Join-Path $OutputDirectory "$language-$state.png")
            if($LASTEXITCODE -ne 0){throw "Preview failed: $language/$state"}
        }
    }
    $releaseName=if($channel1Source){'semantier-setup-channel1.exe'}else{'semantier-setup.exe'}
    $releaseDefine=@()
    if($channel1Source){$releaseDefine+='/define:CHANNEL1'}
    & $csc /target:winexe @releaseDefine ('/out:'+$releaseName) @common
    if($LASTEXITCODE -ne 0){throw 'Release compilation failed.'}
    Write-Output "BUILD_AND_UI_TESTS_PASS $OutputDirectory"
    Write-Output 'No installation or certificate-store mutation was performed. Release EXE is unsigned until the existing signing owner signs it.'
} finally {Pop-Location}
