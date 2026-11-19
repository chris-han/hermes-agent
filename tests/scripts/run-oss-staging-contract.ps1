param([string]$Root)
$ErrorActionPreference='Stop'
$root=$Root
$source='using System; using System.IO; using System.ComponentModel; using System.Globalization; using System.Runtime.InteropServices;'+[IO.File]::ReadAllText((Join-Path $root 'hermes-agent/win-native-installer/WinHttp.cs'))+[IO.File]::ReadAllText((Join-Path $root 'hermes-agent/tests/scripts/oss-staging-download-contract.cs'))
Add-Type -TypeDefinition $source
$t=[AppDomain]::CurrentDomain.GetAssemblies().GetTypes() | Where-Object Name -eq OssStagingContractTests
$t.GetMethod('Run',[Reflection.BindingFlags]'Static,NonPublic').Invoke($null,@())
