import shutil, subprocess
from pathlib import Path
import pytest

@pytest.mark.parametrize("case", ["upgrade", "failure", "downgrade", "foreign"])
def test_local_pilot_preserves_registration(tmp_path, case):
    script=tmp_path/"check.ps1"
    script.write_text(r"""
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$null,[ref]$null)
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Install-LocalPilotPackage'},$true)
if(!$f){throw 'Missing preserving pilot installer'}
Invoke-Expression $f.Extent.Text
$identity='Semantier.Agent.Acceptance';$publisher='CN=Semantier Technology, O=Semantier Technology';$Version='0.0.124.0'
$script:case=$args[1];$script:added=0
function Get-AppxPackage {
 [pscustomobject]@{Publisher=$(if($script:case -eq 'foreign'){'CN=Foreign'}else{$publisher});
 Version=$(if($script:case -eq 'downgrade'){[version]'0.0.125.0'}else{[version]'0.0.123.0'})}
}
function Remove-AppxPackage {throw 'UNINSTALL_FORBIDDEN'}
function Add-AppxPackage {
 param($Path,$Volume,[switch]$ForceTargetApplicationShutdown,[switch]$ForceUpdateFromAnyVersion)
 if($ForceUpdateFromAnyVersion){throw 'DOWNGRADE_FORBIDDEN'}
 if(!$ForceTargetApplicationShutdown){throw 'Target shutdown missing'}
 $script:added++
 if($script:case -eq 'failure'){throw 'SIMULATED_INSTALL_FAILURE'}
}
try {Install-LocalPilotPackage 'signed.msix' 'volume';$ok=$true}
catch {
 if($script:case -eq 'upgrade'){throw}
 if($script:case -eq 'failure' -and $_ -notmatch 'SIMULATED_INSTALL_FAILURE'){throw}
 $ok=$false
}
if($script:case -eq 'upgrade' -and (!$ok -or $script:added -ne 1)){throw 'Upgrade absent'}
if($script:case -in @('downgrade','foreign') -and $script:added){throw 'Unsafe upgrade attempted'}
'PRESERVING_PILOT_PASS'
""")
    operator=Path(__file__).resolve().parents[2]/"scripts/deploy-ecs.ps1"
    r=subprocess.run([shutil.which("pwsh"),"-NoProfile","-File",str(script),str(operator),case],capture_output=True,text=True,timeout=30,check=False)
    assert r.returncode==0,r.stdout+r.stderr
    assert "PRESERVING_PILOT_PASS" in r.stdout
