"""Exercise production PowerShell lifecycle helpers with real disposable Git repos.

The portable lane proves the Git algorithm; native Windows path/process and
BuildOnly qualification remain separate from this result.
"""
import shutil
import subprocess
import json
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize('stamp_build_id', ['expected-build', 'wrong-build'])
def test_prod_package_validation_reads_its_saved_request(tmp_path, stamp_build_id):
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    package = tmp_path / 'package.msix'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('app/resources/install-stamp.json', json.dumps({
            'commit': 'a' * 40, 'dirty': False,
            'channelBuild': {'buildId': stamp_build_id},
        }))
        archive.writestr('AppxManifest.xml',
                         '<Package><Properties><DisplayName>Semantier Agent 0.0.100'
                         '</DisplayName></Properties></Package>')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'buildId': 'expected-build'}))
    script = tmp_path / 'validate.ps1'
    script.write_text(r'''
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$null,[ref]$errors)
if($errors){throw ($errors | Out-String)}
$helper=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Read-ZipText'},$true)
Invoke-Expression $helper.Extent.Text
$build=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Build-CommitPackage'},$true)
$validation=@($build.Body.EndBlock.Statements | Where-Object {
  $_ -is [Management.Automation.Language.IfStatementAst] -and
  $_.Clauses[0].Item1.Extent.Text -eq ('$' + "BuildMode -eq 'Prod'")
})
if($validation.Count -ne 1){throw 'Expected one production package validation block.'}
$BuildMode='Prod'
$Commit='a'*40
$record=[pscustomobject]@{version='0.0.100';request=$args[2]}
$packages=@(Get-Item -LiteralPath $args[1])
Invoke-Expression $validation[0].Extent.Text
''')
    operator = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(script), str(operator), str(package), str(request)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if stamp_build_id == 'expected-build':
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert 'New MSIX source/build stamp mismatch.' in result.stderr


def test_real_git_worktree_lifecycle():
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(root / 'scripts/tests/test-deploy-ecs-worktree.ps1')],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMMON_DIR_EQUAL:' in result.stdout
    assert 'PASS: exact SHA' in result.stdout


@pytest.mark.parametrize("options", [[], ["-BuildOnly", "-BuildMode", "Prod"]])
def test_latest_whatif_is_read_only(tmp_path, options):
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    operator = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    ledger = tmp_path / 'reserved.json'
    ledger.write_bytes(b'{"reserved":"preserve"}')
    output = tmp_path / 'must-not-create'
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(operator), '-Latest', '-WhatIf',
         '-Output', str(output), '-VersionState', str(ledger), *options],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not output.exists()
    assert ledger.read_bytes() == b'{"reserved":"preserve"}'
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("options", [
    ['-Commit', 'a' * 40], ['-SetupOnly'], ['-LocalSetup'],
    ['-UpdateSetup'], ['-PreflightOnly'], ['-Msix', 'fixed.msix'],
])
def test_latest_rejects_incompatible_modes_before_sync(tmp_path, options):
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    operator = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(operator), '-Latest', '-WhatIf', *options],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert '-Latest is for a Runtime-pinned commit build only' in result.stderr
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize('state', ['equal', 'behind', 'ahead', 'diverged'])
def test_latest_requires_local_main_to_be_ancestor_of_fetched_main(tmp_path, state):
    pwsh = shutil.which('pwsh') or shutil.which('pwsh.exe')
    if not pwsh:
        pytest.skip('PowerShell 7 unavailable')
    git = shutil.which('git')
    repo = tmp_path / 'runtime'
    repo.mkdir()

    def run_git(*args):
        return subprocess.check_output(
            [git, '-C', str(repo), *args], text=True, stderr=subprocess.STDOUT,
        ).strip()

    def commit(value):
        (repo / 'value').write_text(value)
        run_git('add', '.')
        run_git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-qm', value)
        return run_git('rev-parse', 'HEAD')

    run_git('init', '-q')
    base = commit('base')
    fetched = commit('remote') if state in ('behind', 'diverged') else base
    run_git('checkout', '--detach', base)
    before = commit('local') if state in ('ahead', 'diverged') else base
    # Bind FETCH_HEAD without network or touching the maintained checkouts.
    (repo / '.git/FETCH_HEAD').write_text(fetched + '\n')
    script = tmp_path / 'sync-guard.ps1'
    script.write_text(r'''
$ErrorActionPreference='Stop'
$e=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$null,[ref]$e)
if($e){throw ($e | Out-String)}
foreach($name in @('Get-SourceGit','Run')){
  $f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
  . ([scriptblock]::Create($f.Extent.Text))
}
function Report-Status([string]$Message){}
# Execute the production ancestry/merge/equality segment. Windows path and
# process qualification are separate; this fixture proves only the Git logic.
$git=$args[2]
function git.exe { & $git @args }
$f=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Sync-GitHubLatestSource'},$true)
$collect=$false;$segment=@()
foreach($statement in $f.Body.EndBlock.Statements){
  if($statement -is [Management.Automation.Language.AssignmentStatementAst] -and $statement.Left.Extent.Text -eq '$fetched'){$collect=$true}
  if($collect -and $statement.Extent.Text -like "Run *'submodule'*"){break}
  if($collect){$segment+=$statement.Extent.Text}
}
if(!$segment.Count){throw 'Missing production sync guard'}
$Runtime=$args[1]
& ([scriptblock]::Create($segment -join "`n"))
''', encoding='utf-8')
    operator = Path(__file__).resolve().parents[2] / 'scripts/deploy-ecs.ps1'
    result = subprocess.run(
        [pwsh, '-NoProfile', '-File', str(script), str(operator), str(repo), git],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if state in ('equal', 'behind'):
        assert result.returncode == 0, result.stdout + result.stderr
        assert run_git('rev-parse', 'HEAD') == fetched
    else:
        assert result.returncode != 0
        assert 'ahead of or diverged from GitHub main' in result.stderr
        assert run_git('rev-parse', 'HEAD') == before
    assert not run_git('status', '--porcelain')
