"""Native Windows source migration: local clone reuse and non-mutating planning."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")
SOURCE = Path(__file__).resolve().parents[2] / "scripts/deploy-ecs.ps1"


def invoke_ps(script, *args):
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(script), *map(str, args)], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=60, check=False)


def git(repo, *args):
    result = subprocess.run([shutil.which("git.exe"), "-C", str(repo), *args],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=30, check=True)
    return result.stdout.strip()


def fixture_repo(tmp_path):
    repo = tmp_path / "shared-source"
    (repo / "scripts/bundles").mkdir(parents=True)
    (repo / "pm").mkdir()
    (repo / "scripts/bundles/desktop.py").write_text("# source fixture\n", encoding="utf-8")
    (repo / "pm/lock.json").write_text('{"packages":{"python":{"version":"fixture"},"node":{"version":"fixture"}}}', encoding="utf-8")
    shutil.copyfile(SOURCE, repo / "scripts/deploy-ecs.ps1")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    return repo


EXTRACT = r"""
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$errors=$null;$tokens=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$tokens,[ref]$errors)
if($errors){throw ($errors | Out-String)}
foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -in @('Assert-LocalWindowsPath','Initialize-LocalBuildSource','Get-EcsSshPassword','Run')},$true)){
 . ([scriptblock]::Create($f.Extent.Text))
}
function Report-Status([string]$Message){}
$script:statusLog=$null
"""


def test_local_clone_reuses_dependencies_fetches_commit_and_preserves_dirty_work(tmp_path):
    repo = fixture_repo(tmp_path)
    first = git(repo, "rev-parse", "HEAD")
    script = tmp_path / "clone.ps1"
    script.write_text(EXTRACT + r"""
$Repository=$args[1];$local=$args[2]
Initialize-LocalBuildSource $local $args[3]
if((Get-Item -Force -LiteralPath (Join-Path $local '.git')).PSIsContainer -ne $true){throw 'Not independent'}
if(Test-Path -LiteralPath (Join-Path $local '.git/objects/info/alternates')){throw 'Cross-filesystem alternates'}
New-Item -ItemType Directory -Path (Join-Path $local 'node_modules') | Out-Null
[IO.File]::WriteAllText((Join-Path $local '.git/info/exclude'),"node_modules/`n")
[IO.File]::WriteAllText((Join-Path $local 'node_modules/retained.txt'),'existing dependencies')
'CLONE_PASS'
""", encoding="utf-8")
    local = tmp_path / "native-build"
    result = invoke_ps(script, SOURCE, repo, local, first)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLONE_PASS" in result.stdout
    assert not (repo / ".git/worktrees").exists()
    (repo / "application.txt").write_text("next committed source", encoding="utf-8")
    git(repo, "add", "application.txt")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "next")
    second = git(repo, "rev-parse", "HEAD")
    script.write_text(EXTRACT + r"""
$Repository=$args[1]
Initialize-LocalBuildSource $args[2] $args[3]
'REUSE_PASS'
""", encoding="utf-8")
    result = invoke_ps(script, SOURCE, repo, local, second)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(local, "rev-parse", "HEAD") == second
    assert (local / "node_modules/retained.txt").read_text(encoding="utf-8") == "existing dependencies"
    (local / "application.txt").write_text("uncommitted user changes", encoding="utf-8")
    result = invoke_ps(script, SOURCE, repo, local, first)
    assert result.returncode != 0
    assert (local / "application.txt").read_text(encoding="utf-8") == "uncommitted user changes"
    assert git(local, "rev-parse", "HEAD") == second


def test_credentials_are_explicit_validated_and_never_echoed(tmp_path):
    script = tmp_path / "credentials.ps1"
    script.write_text(EXTRACT + r"""
if($null -ne (Get-EcsSshPassword '')){throw 'Unexpected implicit credential source'}
if($args[2] -eq 'invalid'){
 try{$null=Get-EcsSshPassword $args[1]}catch{'REJECTED';exit 0}
 throw 'Invalid credentials accepted'
}
$value=Get-EcsSshPassword $args[1]
if($value -cne 'fixture secret # retained'){throw 'Password parsing mismatch'}
'PASSWORD_PARSE_PASS'
""", encoding="utf-8")
    env = tmp_path / "explicit.env"
    env.write_text('export ALI_ECS_SSH_PASSWORDE="fixture secret # retained" # comment\n', encoding="utf-8")
    result = invoke_ps(script, SOURCE, env, "valid")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSWORD_PARSE_PASS" in result.stdout
    assert "fixture secret" not in result.stdout + result.stderr
    for text in ["UNRELATED=value\n", "ALI_ECS_SSH_PASSWORDE=\n",
                 'ALI_ECS_SSH_PASSWORDE="secret\n',
                 "ALI_ECS_SSH_PASSWORDE=one\nALI_ECS_SSH_PASSWORDE=two\n"]:
        env.write_text(text, encoding="utf-8")
        result = invoke_ps(script, SOURCE, env, "invalid")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "REJECTED" in result.stdout
    result = invoke_ps(script, SOURCE, tmp_path / "missing.env", "invalid")
    assert result.returncode == 0, result.stdout + result.stderr


def test_whatif_from_other_directory_preserves_ledger_environment_and_output(tmp_path):
    repo = fixture_repo(tmp_path)
    commit = git(repo, "rev-parse", "HEAD")
    output = tmp_path / "never-created"
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(b'{"preserve":"existing reservation"}')
    cache = tmp_path / "existing-cache"
    cache.mkdir()
    script = tmp_path / "whatif.ps1"
    script.write_text(r"""
$ErrorActionPreference='Stop'
$env:GIT_CONFIG_COUNT='1';$env:GIT_CONFIG_KEY_0='core.autocrlf';$env:GIT_CONFIG_VALUE_0='false'
& $args[0] -Commit $args[1] -Output $args[2] -VersionState $args[3] -BuildCache $args[4] -BuildPython $args[5] -Node $args[5] -WhatIf
if($env:GIT_CONFIG_COUNT -cne '1' -or $env:GIT_CONFIG_VALUE_0 -cne 'false'){throw 'Environment changed'}
if(Test-Path Env:GIT_CONFIG_KEY_1){throw 'Temporary trust leaked'}
'WHATIF_PASS'
""", encoding="utf-8")
    before = dict(os.environ)
    result = invoke_ps(script, repo / "scripts/deploy-ecs.ps1", commit, output, ledger, cache, sys.executable)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WHATIF_PASS" in result.stdout
    assert not output.exists()
    assert ledger.read_bytes() == b'{"preserve":"existing reservation"}'
    assert not (tmp_path / "ledger.json.lock").exists()
    assert not (repo / ".git/worktrees").exists()
    assert dict(os.environ) == before
    env = dict(os.environ, SEMANTIER_WSL_GIT_DISTRO="must-not-run")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(repo / "scripts/deploy-ecs.ps1"), "-WhatIf"], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=30, check=False)
    assert result.returncode != 0
    assert "Clear SEMANTIER_WSL_GIT_DISTRO" in result.stderr
    assert not output.exists()


_BUILD_ONLY_BOUNDARY_PS = r'''
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$e=$null;$t=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$t,[ref]$e)
if($e){throw ($e | Out-String)}
$main=@($ast.EndBlock.Statements | Where-Object {$_ -is [Management.Automation.Language.TryStatementAst]})[-1]
$BuildOnly=$true;$Commit='a'*40;$SetupOnly=$false;$LocalSetup=$false;$UpdateSetup=$false;$PreflightOnly=$false
$WhatIfPreference=$false;$Output=$args[1];$utf8=New-Object Text.UTF8Encoding($false)
$script:releaseEnvironment=@{}
function Initialize-ReleasePaths {}
function Build-CommitPackage {return [pscustomobject]@{version='0.0.90';msix='owned-unsigned.msix';sha256=('b'*64);source='owned-source'}}
function Assert-ReleaseToolchain {throw 'Signing/release preflight must not run'}
function Sign-Copy {throw 'Signing must not run'}
function Report-Status([string]$Message) {}
function Write-RunEvent($Event) {if($Event.type -cne 'build-only'){throw 'Unexpected release event'}}
& ([scriptblock]::Create($main.Extent.Text))
$result=Get-Content -LiteralPath (Join-Path $Output 'result.json') -Raw | ConvertFrom-Json
if($result.result -cne 'UNSIGNED_LOCAL_BUILD' -or $result.commit -cne $Commit){throw 'Unsigned result missing'}
foreach($field in @('signing','installation','publication')){if($result.$field -cne 'NOT_RUN'){throw "Unexpected $field"}}
'BUILD_ONLY_BOUNDARY_PASS'
'''


def test_build_only_returns_before_signing_installation_and_publication(tmp_path):
    script = tmp_path / 'build-only-boundary.ps1'
    script.write_text(_BUILD_ONLY_BOUNDARY_PS, encoding='utf-8')
    result = invoke_ps(script, SOURCE, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'BUILD_ONLY_BOUNDARY_PASS' in result.stdout
