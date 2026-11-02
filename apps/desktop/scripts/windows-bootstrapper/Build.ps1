param([Parameter(Mandatory=$true)][string]$OutputDirectory,[string]$Channel1Descriptor,[string]$Channel1Zip,[string]$Channel2Zip,[string]$Channel2Bundle)
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
$channel2=!!$Channel2Zip
if($channel2){
    if($Channel1Descriptor){throw 'Channel 1 and Channel 2 cannot be combined.'}
    $item=Get-Item -LiteralPath $Channel2Zip
    if($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.Length -ne 2264240015){throw 'Channel2 ZIP mismatch.'}
    if((Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash -cne '59AD5F557BABC3905835F239C502EA44AD5AA43207603FFFF8CBCB757998C229'){throw 'Channel2 ZIP mismatch.'}
    if(!$Channel2Bundle){throw 'Channel2 sparse bundle required.'}
    $bundle=Get-Item -LiteralPath $Channel2Bundle
    if($bundle.PSIsContainer -or ($bundle.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $bundle.Length -ne 2984 -or (Get-FileHash -LiteralPath $bundle.FullName -Algorithm SHA256).Hash -cne '2AD23B092EF7E17FC7517922AB22198902D73FA7A2C52978EF910CAF07B33DB0'){throw 'Channel2 bundle mismatch.'}
    $sig=Get-AuthenticodeSignature -LiteralPath $bundle.FullName
    if($sig.Status -ne 'Valid' -or !$sig.SignerCertificate -or $sig.SignerCertificate.Thumbprint -cne '57BF15739B4AA447842EAF388856A01B4E5029FF'){throw 'Channel2 bundle signature mismatch.'}
    Copy-Item -LiteralPath $bundle.FullName -Destination (Join-Path $OutputDirectory 'channel2.msixbundle')
}
if($Channel1Descriptor){
    $d=Get-Content -LiteralPath $Channel1Descriptor -Raw | ConvertFrom-Json
    foreach($field in @('origin','zip_path','zip_sha256','cert_sha256','graph_sha256','thumbprint','identity','app_version')){if($d.$field -isnot [string]){throw 'Invalid Channel1 descriptor type.'}}
    if($d.origin -cne 'https://update.semantier.cn' -or $d.zip_path -cnotmatch '^/releases/channel-builds/[a-f0-9]{32}/[A-Za-z0-9_.-]+\.zip$' -or $d.zip_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.cert_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.graph_sha256 -cnotmatch '^[a-f0-9]{64}$' -or $d.thumbprint -cnotmatch '^[A-F0-9]{40}$' -or $d.identity -cne 'Semantier.Agent.Acceptance' -or $d.app_version -cnotmatch '^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){3}$' -or ($d.zip_size -isnot [long] -and $d.zip_size -isnot [int]) -or $d.zip_size -le 0){throw 'Invalid Channel1 descriptor.'}
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
    $cert=New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 (Join-Path $OutputDirectory 'SemantierAgent-signing.cer')
    try {if($cert.Thumbprint -cne $d.thumbprint){throw 'Descriptor thumbprint mismatch.'}} finally {$cert.Dispose()}
    if(!$Channel1Zip){throw 'Channel1 ZIP required.'}
    $zipItem=Get-Item -LiteralPath $Channel1Zip
    if($zipItem.PSIsContainer -or ($zipItem.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Channel1 ZIP must be a regular file.'}
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipStream=[IO.File]::Open($zipItem.FullName,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        if($zipStream.Length -ne $d.zip_size){throw 'Channel1 ZIP size mismatch.'}
        $sha=[Security.Cryptography.SHA256]::Create()
        try {$actualZip=[BitConverter]::ToString($sha.ComputeHash($zipStream)).Replace('-','').ToLowerInvariant()} finally {$sha.Dispose()}
        if($actualZip -cne $d.zip_sha256){throw 'Channel1 ZIP hash mismatch.'}
        $zipStream.Position=0
        $zip=New-Object IO.Compression.ZipArchive($zipStream,[IO.Compression.ZipArchiveMode]::Read,$true)
        try {
            foreach($name in $names.Keys){
                $found=@($zip.Entries | Where-Object {$_.FullName -ceq $name})
                if($found.Count -ne 1){throw 'Channel1 ZIP member missing or duplicated.'}
                $inputStream=$found[0].Open();$sha=[Security.Cryptography.SHA256]::Create()
                try {$actual=[BitConverter]::ToString($sha.ComputeHash($inputStream)).Replace('-','').ToLowerInvariant()} finally {$sha.Dispose();$inputStream.Dispose()}
                if($actual -cne $names[$name]){throw 'Channel1 ZIP member hash mismatch.'}
            }
            $entry=@($zip.Entries | Where-Object {$_.FullName -ceq 'native-graph.json'})[0]
            if($entry.Length -gt 2097152){throw 'Channel1 graph too large.'}
            $reader=New-Object IO.StreamReader($entry.Open(),[Text.Encoding]::UTF8)
            try {$graph=$reader.ReadToEnd() | ConvertFrom-Json} finally {$reader.Dispose()}
            $prefix=$d.zip_path.Substring(1,$d.zip_path.LastIndexOf('/'))
            if($graph.schema -ne 1 -or $graph.transport -cne 'windows-package-uri' -or $graph.root -cne ($prefix+'SemantierAgent.msixbundle') -or @($graph.objects).Count -ne ($names.Count-2)){throw 'Channel1 graph inventory mismatch.'}
            $seen=@{}
            foreach($object in @($graph.objects)){
                if($object.key -isnot [string] -or $object.sha256 -isnot [string] -or $object.version -cne $d.app_version){throw 'Channel1 graph object invalid.'}
                if(!$object.key.StartsWith($prefix,[StringComparison]::Ordinal)){throw 'Channel1 graph release mismatch.'}
                $name=$object.key.Substring($prefix.Length)
                if(!$names.ContainsKey($name) -or $seen.ContainsKey($name) -or $names[$name] -cne $object.sha256){throw 'Channel1 graph member mismatch.'}
                if(($name -ceq 'SemantierAgent-main.msix' -and $object.kind -cne 'main-package') -or ($name -ceq 'SemantierAgent.msixbundle' -and $object.kind -cne 'main-bundle') -or ($name -cmatch '^SemantierAgent-runtime[0-9]{4}\.msix$' -and $object.kind -cne 'asset-package')){throw 'Channel1 graph package kind mismatch.'}
                if($object.identity -cne $d.identity -or $object.publisher -cne 'CN=Semantier Technology, O=Semantier Technology'){throw 'Channel1 graph identity mismatch.'}
                $seen[$name]=$true
            }
        } finally {$zip.Dispose()}
    } finally {$zipStream.Dispose()}
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
    $defines=if($channel1Source){'/define:TESTING,CHANNEL1'}elseif($channel2){'/define:TESTING,CHANNEL2'}else{'/define:TESTING'}
    if($channel2){$common+= '/resource:channel2.msixbundle,Semantier.channel2.msixbundle'}
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
    $releaseName=if($channel1Source){'semantier-setup-channel1.exe'}elseif($channel2){'semantier-setup-channel2.exe'}else{'semantier-setup.exe'}
    $releaseDefine=@()
    if($channel1Source){$releaseDefine+='/define:CHANNEL1'}elseif($channel2){$releaseDefine+='/define:CHANNEL2'}
    & $csc /target:winexe @releaseDefine ('/out:'+$releaseName) @common
    if($LASTEXITCODE -ne 0){throw 'Release compilation failed.'}
    Write-Output "BUILD_AND_UI_TESTS_PASS $OutputDirectory"
    Write-Output 'No installation or certificate-store mutation was performed. Release EXE is unsigned until the existing signing owner signs it.'
} finally {Pop-Location}
