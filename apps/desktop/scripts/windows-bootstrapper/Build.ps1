param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $OutputDirectory){throw 'Use a fresh task-owned output directory.'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$assets=Join-Path $PSScriptRoot '..\..\assets'
foreach($name in @('Installer.cs','InstallerUI.cs','SemantierAgent-signing.cer','bootstrapper.manifest')){
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $OutputDirectory
}
Copy-Item -LiteralPath (Join-Path $assets 'icon.png') -Destination (Join-Path $OutputDirectory 'icon.png')
Copy-Item -LiteralPath (Join-Path $assets 'icon.ico') -Destination (Join-Path $OutputDirectory 'icon.ico')
$csc=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
Push-Location $OutputDirectory
try {
    $common=@('/nologo','/codepage:65001','/optimize+','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.IO.Compression.dll','/r:System.IO.Compression.FileSystem.dll','/win32icon:icon.ico','/win32manifest:bootstrapper.manifest','/resource:SemantierAgent-signing.cer,SemantierAgent-signing.cer','/resource:icon.png,Semantier.logo.png','/resource:icon.ico,Semantier.icon.ico','Installer.cs','InstallerUI.cs')
    & $csc /target:exe /define:TESTING /out:SemantierSetup.Tests.exe @common
    if($LASTEXITCODE -ne 0){throw 'Test compilation failed.'}
    & '.\SemantierSetup.Tests.exe' --self-test
    if($LASTEXITCODE -ne 0){throw 'Bootstrapper self-test failed.'}
    foreach($language in @('en','zh-CN')){
        foreach($state in @('ready','download','verify','trust','install','error','success')){
            & '.\SemantierSetup.Tests.exe' --preview $language $state (Join-Path $OutputDirectory "$language-$state.png")
            if($LASTEXITCODE -ne 0){throw "Preview failed: $language/$state"}
        }
    }
    & $csc /target:winexe /out:semantier-setup.exe @common
    if($LASTEXITCODE -ne 0){throw 'Release compilation failed.'}
    Write-Output "BUILD_AND_UI_TESTS_PASS $OutputDirectory"
    Write-Output 'No installation or certificate-store mutation was performed. Release EXE is unsigned until the existing signing owner signs it.'
} finally {Pop-Location}
