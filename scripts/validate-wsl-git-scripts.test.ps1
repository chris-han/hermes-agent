$ErrorActionPreference='Stop'
foreach($name in @('build-windows-distribution.ps1','build-bundle.ps1','wsl-linux-git.ps1')) {
  $tokens=$null; $errors=$null
  [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot $name),[ref]$tokens,[ref]$errors) | Out-Null
  if($errors){throw ($name+': '+$errors[0].Message)}
  Write-Output ('PS_PARSE_PASS '+$name)
}
