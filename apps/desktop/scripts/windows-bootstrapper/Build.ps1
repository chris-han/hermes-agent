param([Parameter(Mandatory=$true)][ValidateSet('channel1','channel2')][string]$Channel,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$workRoot=[IO.Path]::GetFullPath('D:\repo\deliverables\work').TrimEnd('\')+'\'
$output=[IO.Path]::GetFullPath($OutputDirectory)
if(!$output.StartsWith($workRoot,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $output)) {throw 'Use a fresh task-owned directory under D:\repo\deliverables\work.'}
$check=[IO.DirectoryInfo]::new($output).Parent
while($check) {if($check.Exists -and ($check.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Redirected output path refused.'};$check=$check.Parent}
foreach($cachePath in @('D:\repo\deliverables\cache\dotnet','D:\repo\deliverables\cache\nuget')) {
    $check=[IO.DirectoryInfo]::new($cachePath)
    while($check) {if($check.Exists -and ($check.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Redirected cache path refused.'};$check=$check.Parent}
}
New-Item -ItemType Directory -Path $output | Out-Null
foreach($name in @('Installer.cs','InstallerUI.cs','ActiveRelease.cs','PackageIdentity.cs','PackageSignature.cs','BootstrapperTransport.cs','SemantierAgent-signing.cer','bootstrapper.manifest')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $output
}
$assets=Join-Path $PSScriptRoot '..\..\assets'
Copy-Item -LiteralPath (Join-Path $assets 'icon.png') -Destination (Join-Path $output 'icon.png')
Copy-Item -LiteralPath (Join-Path $assets 'icon.ico') -Destination (Join-Path $output 'icon.ico')
$cert=Join-Path $output 'SemantierAgent-signing.cer'
if((Get-FileHash -Algorithm SHA256 -LiteralPath $cert).Hash -cne '10A6B55E9A03BBC0DCEB8D899C1776BF1FBAD963C82050C16534DA4983C9661D'){throw 'Pinned certificate mismatch.'}
$project=@'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0-windows</TargetFramework><UseWindowsForms>true</UseWindowsForms>
    <RuntimeIdentifier>win-x64</RuntimeIdentifier><GenerateAssemblyInfo>false</GenerateAssemblyInfo>
    <ApplicationIcon>icon.ico</ApplicationIcon><ApplicationManifest>bootstrapper.manifest</ApplicationManifest>
    <EnableNETAnalyzers>false</EnableNETAnalyzers><OutputType Condition="'$(Testing)'=='true'">Exe</OutputType>
    <OutputType Condition="'$(Testing)'!='true'">WinExe</OutputType>
    <DefineConstants Condition="'$(Testing)'=='true'">__CHANNEL__;TESTING</DefineConstants>
    <DefineConstants Condition="'$(Testing)'!='true'">__CHANNEL__</DefineConstants>
  </PropertyGroup>
  <ItemGroup>
    <EmbeddedResource Include="SemantierAgent-signing.cer" LogicalName="SemantierAgent-signing.cer" />
    <EmbeddedResource Include="icon.png" LogicalName="Semantier.logo.png" />
    <EmbeddedResource Include="icon.ico" LogicalName="Semantier.icon.ico" />
  </ItemGroup>
</Project>
'@
$project=$project.Replace('__CHANNEL__',$Channel.ToUpperInvariant())
[IO.File]::WriteAllText((Join-Path $output 'Bootstrapper.csproj'),$project,[Text.UTF8Encoding]::new($false))
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1'
$env:DOTNET_GENERATE_ASPNET_CERTIFICATE='false'
$env:DOTNET_CLI_TELEMETRY_OPTOUT='1'
$env:DOTNET_CLI_HOME='D:\repo\deliverables\cache\dotnet'
$env:NUGET_PACKAGES='D:\repo\deliverables\cache\nuget'
$dotnet=Join-Path $env:ProgramFiles 'dotnet\dotnet.exe'
if(!(Test-Path -LiteralPath $dotnet)){throw '.NET 8 SDK is required.'}
& $dotnet publish (Join-Path $output 'Bootstrapper.csproj') -c Release -o (Join-Path $output 'test') -p:Testing=true --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true
if($LASTEXITCODE -ne 0){throw 'Bootstrapper test compilation failed.'}
$testExe=Join-Path $output 'test\Bootstrapper.exe'
& $testExe --self-test
if($LASTEXITCODE -ne 0){throw 'Bootstrapper self-test failed.'}
& $dotnet publish (Join-Path $output 'Bootstrapper.csproj') -c Release -o (Join-Path $output 'release') -p:Testing=false --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true
if($LASTEXITCODE -ne 0){throw 'Bootstrapper release compilation failed.'}
$name=if($Channel -eq 'channel1'){'semantier-setup-ch1.exe'}else{'semantier-setup-ch2.exe'}
Copy-Item -LiteralPath (Join-Path $output 'release\Bootstrapper.exe') -Destination (Join-Path $output $name)
Write-Output "BUILD_AND_UI_TESTS_PASS $output"
Write-Output 'Release EXE is unsigned until the existing signing owner signs it.'
