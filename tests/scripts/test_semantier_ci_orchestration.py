"""Controlled subprocess coverage of the CI orchestration seam, not Windows acceptance."""
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.releases.semantier_native_ci import orchestrate_ci as production_orchestrate_ci


def orchestrate_ci(root, phases, reconcile=None):
    if phases[0][0] != 'preflight':
        phases = [('preflight', command('pass'))] + phases
    return production_orchestrate_ci(root, phases, reconcile or (lambda: True))


def command(code, *args):
    return [sys.executable, '-c', code, *map(str, args)]


def test_windows_child_environment_excludes_receipt_key_and_wslenv_forwarding(monkeypatch):
    from scripts.releases.semantier_native_ci import native_environment
    monkeypatch.setenv('SEMANTIER_CLIENT_RECEIPT_KEY', 'fixture-secret')
    monkeypatch.setenv('WSLENV', 'SEMANTIER_CLIENT_RECEIPT_KEY/u:OTHER_VALUE:semantier_client_receipt_key/w')
    env = native_environment()
    assert 'SEMANTIER_CLIENT_RECEIPT_KEY' not in env
    assert all(item.split('/')[0].upper() != 'SEMANTIER_CLIENT_RECEIPT_KEY' for item in env['WSLENV'].split(':'))
    assert 'OTHER_VALUE' in env['WSLENV']


def test_workflow_build_dispatch_scopes_windows_git_to_owned_checkout(tmp_path, monkeypatch):
    import scripts.releases.semantier_native_ci as ci
    source = 'a' * 40
    controller = 'b' * 40
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    monkeypatch.setattr(ci, 'CI_INCOMING_ROOT', incoming)
    for key, value in {'GITHUB_ACTIONS':'true', 'GITHUB_REPOSITORY':'chris-han/hermes-agent',
                       'GITHUB_SHA':source, 'GITHUB_RUN_ID':'123', 'GITHUB_RUN_ATTEMPT':'1',
                       'SEMANTIER_CLIENT_RECEIPT_KEY':'x'*32, 'ACCEPTANCE_PACKAGE_LAYOUT':'native-assets',
                       'ACCEPTANCE_WINDOWS_PYTHON':r'D:\repo\python.exe'}.items():
        monkeypatch.setenv(key, value)
    def output(argv, **_kwargs):
        if argv[0] == 'wslpath': return '/tmp/fixture-path\n'
        if 'status' in argv: return ''
        if 'rev-parse' in argv: return (controller if str(tmp_path) in argv else source) + '\n'
        raise AssertionError(argv)
    monkeypatch.setattr(ci.subprocess, 'check_output', output)
    monkeypatch.setattr(ci.subprocess, 'run', lambda *args, **kwargs: None)
    captured = []; preflight_calls=[]
    monkeypatch.setattr(ci,'preflight',lambda *args:preflight_calls.append(args))
    def capture(root,phases,*args,**kwargs):
        captured.extend(phases)
        kwargs['stage_preflight']()
    monkeypatch.setattr(ci, 'orchestrate_ci', capture)
    ci.orchestrate_workflow('S-1-5-21-1-2-3-4', str(incoming), tmp_path, controller, 'https://updates.semantier.com')
    build = dict(captured)['build']
    assert '--native-local-git' in build and '--native-assets' in build
    assert build[build.index('--source') + 1] == str(ci.ROOT)
    assert preflight_calls==[('S-1-5-21-1-2-3-4',str(incoming),tmp_path,controller,'https://updates.semantier.com')]


@pytest.mark.parametrize('scenario', ['success', 'unresolved', 'preflight-fails'])
def test_actual_workflow_orders_reconciliation_and_protected_phases(tmp_path, monkeypatch, scenario):
    """Production orchestration; Windows processes and deployment are simulated."""
    from contextlib import contextmanager
    import scripts.releases.semantier_native_ci as ci
    source, controller = 'a'*40, 'b'*40
    incoming=tmp_path/'incoming';incoming.mkdir()
    monkeypatch.setattr(ci,'CI_INCOMING_ROOT',incoming)
    for key,value in {'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'chris-han/hermes-agent',
        'GITHUB_SHA':source,'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1',
        'SEMANTIER_CLIENT_RECEIPT_KEY':'x'*32,'ACCEPTANCE_PACKAGE_LAYOUT':'native-assets',
        'ACCEPTANCE_WINDOWS_PYTHON':r'D:\repo\python.exe'}.items():monkeypatch.setenv(key,value)
    def output(argv,**kwargs):
        if argv[0]=='wslpath':return '/tmp/fixture-path\n'
        if 'status' in argv:return ''
        if 'rev-parse' in argv:return (controller if str(tmp_path) in argv else source)+'\n'
        raise AssertionError(argv)
    monkeypatch.setattr(ci.subprocess,'check_output',output)
    events=[]
    def run(argv,**kwargs):
        if '--preflight' in argv:
            events.append('preflight')
            if scenario=='preflight-fails':raise subprocess.CalledProcessError(1,argv)
    monkeypatch.setattr(ci.subprocess,'run',run)
    @contextmanager
    def custody(_fd,_sid):
        events.append('lock-and-custody')
        yield type('FixtureProcess',(),{'poll':lambda self:None})()
    monkeypatch.setattr(ci,'hold_ci_run_custody',custody)
    def reconcile(**kwargs):
        events.append('reconcile')
        if scenario=='unresolved':raise ValueError('prior operation unresolved')
        return True
    monkeypatch.setattr(ci,'hold_unfinished_ci_runs',reconcile)
    monkeypatch.setattr(ci,'run_owned_ci_child',lambda argv,env,fd,process:events.append('stage' if '--notes' in argv else 'build'))
    monkeypatch.setattr(ci,'_native_smoke',lambda a,process,fd:events.append('native'))
    monkeypatch.setattr(ci,'preflight',lambda *args:events.append('stage-preflight'))
    actual=ci.orchestrate_ci
    def invoke(root,phases,*args,**kwargs):
        kwargs['run_intent']=tmp_path/'run.intent'
        return actual(root,phases,*args,**kwargs)
    monkeypatch.setattr(ci,'orchestrate_ci',invoke)
    if scenario=='success':
        ci.orchestrate_workflow('S-1-5-21-1-2-3-4',str(incoming),tmp_path,controller,'https://updates.semantier.com')
        assert events==['lock-and-custody','reconcile','preflight','build','native','stage-preflight','stage']
    else:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):
            ci.orchestrate_workflow('S-1-5-21-1-2-3-4',str(incoming),tmp_path,controller,'https://updates.semantier.com')
        assert not {'build','native','stage'}.intersection(events)


def test_build_child_cannot_forward_receipt_key_but_staging_can_use_it(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('SEMANTIER_CLIENT_RECEIPT_KEY', 'fixture-secret')
    monkeypatch.setenv('WSLENV', 'SEMANTIER_CLIENT_RECEIPT_KEY/u:OTHER_VALUE')
    output = tmp_path / 'env.jsonl'
    code = ('import json,os,sys; open(sys.argv[1],"a").write(json.dumps({'
            '"key":bool(os.environ.get("SEMANTIER_CLIENT_RECEIPT_KEY")),'
            '"forwarded":"SEMANTIER_CLIENT_RECEIPT_KEY" in os.environ.get("WSLENV","")})+"\\n")')
    production_orchestrate_ci(tmp_path, [('preflight', command('pass')), ('build', command(code, output)),
                                         ('stage', command(code, output))], lambda: True)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {'key': False, 'forwarded': False}, {'key': True, 'forwarded': True}]


def test_phases_run_in_order_and_failure_stops_successors(tmp_path):
    log = tmp_path / 'phases'
    write = 'import sys; open(sys.argv[1],"a").write(sys.argv[2]+"\\n")'
    phases = [(name, command(write, log, name)) for name in ('preflight', 'build', 'native', 'stage')]
    orchestrate_ci(tmp_path, phases)
    assert log.read_text().splitlines() == ['preflight', 'build', 'native', 'stage']
    next_root = tmp_path / 'next'
    next_root.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        orchestrate_ci(next_root, [('preflight', command('raise SystemExit(7)')), ('build', command(write, log, 'unexpected'))])
    assert 'unexpected' not in log.read_text()


def test_competing_owner_is_refused(tmp_path):
    from tests.install.semantier_registered_update import exclusive
    with exclusive(tmp_path / 'ci-execution.lock'):
        with pytest.raises((BlockingIOError, OSError)):
            orchestrate_ci(tmp_path, [('preflight', command('pass'))])


def test_custody_is_held_before_discovery_and_until_phases_finish(tmp_path):
    from contextlib import contextmanager
    calls = []
    @contextmanager
    def custody(_lock_fd):
        calls.append('hold')
        try:
            yield type('Custody', (), {'poll': lambda self: None})()
        finally:
            calls.append('release')
    def reconcile():
        assert calls == ['hold']
        calls.append('reconcile')
        return True
    def native(_custody, _lock_fd):
        assert calls == ['hold', 'reconcile']
        calls.append('native')
    production_orchestrate_ci(tmp_path, [('preflight', command('pass')), ('native', native)], reconcile, custody=custody)
    assert calls == ['hold', 'reconcile', 'native', 'release']


@pytest.mark.parametrize('phase', ['build', 'native'])
def test_custody_exit_interrupts_protected_child_before_later_phases(tmp_path, phase):
    from contextlib import contextmanager
    from scripts.releases.semantier_native_ci import run_owned_ci_child
    started = tmp_path / 'started'
    completed = tmp_path / 'completed'
    staged = tmp_path / 'staged'
    class Custody:
        def poll(self):
            return 1 if started.exists() else None
    @contextmanager
    def custody(_lock_fd):
        yield Custody()
    code = ('from pathlib import Path; import sys,time; '
            'Path(sys.argv[1]).touch(); time.sleep(2); Path(sys.argv[2]).touch()')
    child = command(code, started, completed)
    def native(custody_process, lock_fd):
        import os
        run_owned_ci_child(child, os.environ.copy(), lock_fd, custody_process)
    with pytest.raises(ValueError, match='custody handle terminated'):
        production_orchestrate_ci(tmp_path, [('preflight', command('pass')),
            (phase, child if phase == 'build' else native),
            ('stage', command('from pathlib import Path; import sys; Path(sys.argv[1]).touch()', staged))],
            lambda: True, custody=custody)
    assert started.exists() and not completed.exists() and not staged.exists()
    assert (tmp_path / '.ci-execution-pending.json').exists()


def test_custody_process_executes_one_scoped_command_and_reports_failure():
    import os
    from scripts.releases.semantier_native_ci import run_ci_custody_command
    server = ('import sys,json,base64; line=sys.stdin.readline().strip(); nonce,body=line.split(":",1); '
              'assert json.loads(base64.b64decode(body))=={"operation":"native-smoke"}; '
              'print("fixture output",flush=True); print("CI_CUSTODY_DONE:"+nonce+":OK",flush=True); '
              'sys.stdin.readline()')
    process = subprocess.Popen([sys.executable, '-u', '-c', server], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        with pytest.raises(ValueError,match='Fixed CI native dispatch'):
            run_ci_custody_command(process, 'simulated arbitrary command', timeout=5)
        run_ci_custody_command(process, {'operation':'native-smoke'}, timeout=5)
        assert process.poll() is None
    finally:
        process.stdin.close()
        process.wait(timeout=5)


def test_interrupted_mutating_phase_holds_next_invocation(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        orchestrate_ci(tmp_path, [('preflight', command('pass')), ('build', command('raise SystemExit(1)'))])
    with pytest.raises(ValueError, match='reconciliation'):
        orchestrate_ci(tmp_path, [('preflight', command('pass'))])


def test_coordinator_checks_pending_marker_after_reconciliation(tmp_path):
    marker = tmp_path / '.ci-execution-pending.json'
    phase = tmp_path / 'phase'
    def discover_new_hold():
        marker.write_text('unresolved operation')
        return True
    with pytest.raises(ValueError, match='requires reconciliation'):
        production_orchestrate_ci(tmp_path, [('preflight', command('import sys; open(sys.argv[1],"w").close()', phase))], discover_new_hold)
    assert not phase.exists()


def test_parent_death_preserves_hold_while_child_survives(tmp_path):
    import os
    import time

    pidfile = tmp_path / 'child.pid'
    child = """import os,sys,time
from pathlib import Path
open(sys.argv[1], 'w').write(str(os.getpid()))
stop=Path(sys.argv[1]+'.stop')
deadline=time.monotonic()+30
while not stop.exists() and time.monotonic()<deadline:
    time.sleep(.02)
"""
    parent = ('from scripts.releases.semantier_native_ci import orchestrate_ci; '
              'import sys; orchestrate_ci(sys.argv[1], [("preflight", [sys.executable, "-c", "pass"]), ("native", [sys.executable, "-c", sys.argv[2], sys.argv[3]])], lambda: True)')
    process = subprocess.Popen([sys.executable, '-c', parent, str(tmp_path), child, str(pidfile)])
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('orchestrator exited before child started')
            time.sleep(.02)
        child_pid = int(pidfile.read_text())
        process.kill()
        process.wait(timeout=5)
        os.kill(child_pid, 0)
        with pytest.raises(BlockingIOError):
            orchestrate_ci(tmp_path, [('preflight', command('pass'))])
        Path(str(pidfile) + '.stop').touch()
        deadline = time.monotonic() + 5
        while True:
            try:
                orchestrate_ci(tmp_path, [('preflight', command('pass'))])
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
            except ValueError as error:
                assert 'reconciliation' in str(error)
                break
            else:
                pytest.fail('interrupted operation was treated as complete')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            Path(str(pidfile) + '.stop').touch()


def test_surviving_build_retains_lock_after_coordinator_death(tmp_path):
    import time

    started = tmp_path / 'build-started'
    stop = tmp_path / 'stop-build'
    child = ('from pathlib import Path; import sys,time; '
             'Path(sys.argv[1]).touch(); '
             'exec("while not Path(sys.argv[2]).exists():\\n time.sleep(.02)")')
    parent = ('from scripts.releases.semantier_native_ci import orchestrate_ci; import sys; '
              'orchestrate_ci(sys.argv[1], [("preflight", [sys.executable,"-c","pass"]), '
              '("build", [sys.executable,"-c",sys.argv[2],sys.argv[3],sys.argv[4]])], lambda: True)')
    process = subprocess.Popen([sys.executable, '-c', parent, str(tmp_path), child, str(started), str(stop)])
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('coordinator exited before build started')
            time.sleep(.02)
        assert started.exists()
        process.kill()
        process.wait(timeout=5)
        with pytest.raises(BlockingIOError):
            production_orchestrate_ci(tmp_path, [('preflight', command('pass'))], lambda: True)
    finally:
        stop.touch()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_failed_native_does_not_stage(tmp_path):
    staged = tmp_path / 'staged'
    with pytest.raises(subprocess.CalledProcessError):
        orchestrate_ci(tmp_path, [('preflight', command('pass')), ('native', command('raise SystemExit(2)')),
                                  ('stage', command('import sys; open(sys.argv[1],"w").close()', staged))])
    assert not staged.exists()


def test_unfinished_previous_run_is_held_before_preflight(tmp_path):
    from scripts.releases.semantier_native_ci import hold_unfinished_ci_runs
    run = tmp_path / 'client-feedback-ci-123-1' / 'out'
    run.mkdir(parents=True)
    intent = run / ('ci-native-intent-' + 'a' * 32 + '.json')
    intent.write_text('{}')
    phase = tmp_path / 'phase-ran'
    with pytest.raises(ValueError, match='Prior native run requires reconciliation'):
        orchestrate_ci(tmp_path, [('preflight', command('import sys; open(sys.argv[1],"w").close()', phase))],
                       lambda: hold_unfinished_ci_runs(tmp_path))
    assert intent.exists()
    assert not phase.exists()


def test_direct_native_entrypoint_requires_owned_lock(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/releases/semantier_native_ci.py'
    result = subprocess.run([sys.executable, str(script), '--output', str(tmp_path), '--work', 'D:\\unused',
                             '--commit', 'a' * 40, '--expected-sid', 'S-1-5-21-1-2-3-4'],
                            text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'Owned CI execution lock required' in result.stderr


def test_replaced_marker_is_preserved_during_completion(tmp_path, monkeypatch):
    import os
    import scripts.releases.semantier_native_ci as ci
    marker = tmp_path / '.ci-execution-pending.json'
    original_write = os.write
    replaced = []
    def replace_before_append(fd, data):
        if data.startswith(bytes((10, 123))):
            marker.rename(tmp_path / 'original-journal')
            marker.write_text('replacement owned by another writer')
            replaced.append(True)
        return original_write(fd, data)
    monkeypatch.setattr(ci.os, 'write', replace_before_append)
    with pytest.raises(ValueError, match='marker identity changed'):
        orchestrate_ci(tmp_path, [('build', command('pass'))])
    assert replaced
    assert marker.read_text() == 'replacement owned by another writer'
    assert (tmp_path / 'original-journal').exists()


def test_missing_or_unresolved_prerequisites_never_enter_build(tmp_path):
    target = tmp_path / 'build'
    phase = ('build', command('import sys; open(sys.argv[1], "w").close()', target))
    preflight = ('preflight', command('pass'))
    with pytest.raises(ValueError, match='preflight and reconciliation'):
        production_orchestrate_ci(tmp_path, [phase], lambda: True)
    with pytest.raises(ValueError, match='preflight and reconciliation'):
        production_orchestrate_ci(tmp_path, [preflight, phase])
    with pytest.raises(ValueError, match='reconciliation unresolved'):
        production_orchestrate_ci(tmp_path, [preflight, phase], lambda: None)
    assert not target.exists()


def test_public_powershell_native_entry_refuses_before_os_work():
    import os
    from pathlib import Path
    powershell='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    if not Path(powershell).is_file():pytest.skip('Windows PowerShell interop unavailable')
    script=Path(__file__).resolve().parents[1]/'install'/'windows-bundle-smoke.ps1'
    try:
        windows_script=subprocess.check_output(['wslpath','-w',str(script)],text=True,timeout=10).strip()
        result=subprocess.run([powershell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',windows_script,
            '-NativeGraph',r'C:\does-not-exist.json','-Artifact',r'C:\does-not-exist.msixbundle',
            '-Arch','x64','-Commit','a'*40,'-Work',r'C:\no-work','-Out',r'C:\no-out'],
            capture_output=True,text=True,timeout=20,env=os.environ.copy(),check=False)
    except (OSError,subprocess.TimeoutExpired):pytest.skip('Windows PowerShell interop unavailable')
    assert result.returncode!=0
    assert 'coordinator-owned private dispatch' in result.stdout+result.stderr
    legacy=subprocess.run([powershell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',windows_script,
        '-SemantierAcceptanceHost','-Artifact',r'C:\does-not-exist.msix',
        '-Arch','x64','-Commit','a'*40,'-Work',r'C:\no-work','-Out',r'C:\no-out'],
        capture_output=True,text=True,timeout=20,env=os.environ.copy(),check=False)
    assert legacy.returncode!=0
    assert 'Absolute artifact path required' in legacy.stdout+legacy.stderr


def test_internal_native_phase_requires_successful_prerequisites(tmp_path):
    calls = []
    def native():
        calls.append('native')
    preflight = ('preflight', command('pass'))
    with pytest.raises(ValueError, match='reconciliation unresolved'):
        production_orchestrate_ci(tmp_path, [preflight, ('native', native)], lambda: False)
    assert calls == []
    with pytest.raises(subprocess.CalledProcessError):
        production_orchestrate_ci(tmp_path, [('preflight', command('raise SystemExit(2)')), ('native', native)], lambda: True)
    assert calls == []
    production_orchestrate_ci(tmp_path, [preflight, ('native', native)], lambda: True)
    assert calls == ['native']


def test_unexpected_marker_created_during_preflight_blocks_build(tmp_path):
    marker = tmp_path / '.ci-execution-pending.json'
    built = tmp_path / 'built'
    create = 'import sys; open(sys.argv[1],"w").write("other owner")'
    with pytest.raises(FileExistsError):
        production_orchestrate_ci(tmp_path, [('preflight', command(create, marker)),
                                              ('build', command(create, built))], lambda: True)
    assert marker.read_text() == 'other owner'
    assert not built.exists()


def test_marker_replaced_between_phases_blocks_next_phase(tmp_path):
    marker = tmp_path / '.ci-execution-pending.json'
    built = tmp_path / 'staged'
    replace = 'import os,sys; os.rename(sys.argv[1],sys.argv[1]+".old"); open(sys.argv[1],"w").write("other")'
    with pytest.raises(ValueError, match='marker identity changed'):
        production_orchestrate_ci(tmp_path, [('preflight', command('pass')), ('build', command(replace, marker)),
                                              ('stage', command('import sys; open(sys.argv[1],"w").close()', built))], lambda: True)
    assert marker.read_text() == 'other'
    assert not built.exists()


def test_removed_marker_after_validation_does_not_hide_surviving_child(tmp_path):
    import os
    import time
    state = tmp_path / 'incoming'
    runs = tmp_path / 'runs'
    state.mkdir(); runs.mkdir()
    pidfile = tmp_path / 'child.pid'
    child = """import os,sys,time
from pathlib import Path
open(sys.argv[1], 'w').write(str(os.getpid()))
stop=Path(sys.argv[1]+'.stop')
deadline=time.monotonic()+30
while not stop.exists() and time.monotonic()<deadline:
    time.sleep(.02)
"""
    parent = ('from scripts.releases.semantier_native_ci import orchestrate_ci; import sys; '
              'orchestrate_ci(sys.argv[1], [("preflight", [sys.executable,"-c","pass"]), '
              '("build", [sys.executable,"-c",sys.argv[2],sys.argv[3]])], lambda: True, run_intent=sys.argv[4])')
    run_intent = runs / 'client-feedback-ci-123-1-coordinator.intent'
    process = subprocess.Popen([sys.executable, '-c', parent, str(state), child, str(pidfile), str(run_intent)])
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('orchestrator exited before protected child started')
            time.sleep(.02)
        assert pidfile.exists()
        (state / '.ci-execution-pending.json').unlink()
        process.kill(); process.wait(timeout=5)
        os.kill(int(pidfile.read_text()), 0)
        from scripts.releases.semantier_native_ci import hold_unfinished_ci_runs
        with pytest.raises(BlockingIOError):
            production_orchestrate_ci(state, [('preflight', command('pass')), ('build', command('pass'))],
                                      lambda: hold_unfinished_ci_runs(runs), run_intent=runs / 'client-feedback-ci-124-1-coordinator.intent')
        Path(str(pidfile) + '.stop').touch()
        deadline = time.monotonic() + 5
        while True:
            try:
                production_orchestrate_ci(state, [('preflight', command('pass')), ('build', command('pass'))],
                                          lambda: hold_unfinished_ci_runs(runs), run_intent=runs / 'client-feedback-ci-124-1-coordinator.intent')
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
            except ValueError as error:
                assert 'prior CI run requires reconciliation' in str(error)
                break
            else:
                pytest.fail('missing marker hid the unresolved run intent')
        assert run_intent.exists()
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
        Path(str(pidfile) + '.stop').touch()


def test_unreadable_run_custody_is_not_treated_as_empty(tmp_path, monkeypatch):
    import scripts.releases.semantier_native_ci as ci
    def denied(_fd):
        raise PermissionError('custody enumeration denied')
    monkeypatch.setattr(ci.os, 'scandir', denied)
    with pytest.raises(PermissionError, match='enumeration denied'):
        ci.hold_unfinished_ci_runs(tmp_path)


def test_prior_source_drift_is_observed_but_cannot_enter_protected_phase(tmp_path, monkeypatch):
    import io
    import json
    import scripts.releases.semantier_native_ci as ci
    import tests.install.semantier_registered_update as journal
    runs = tmp_path / 'runs'
    out = runs / 'client-feedback-ci-123-1' / 'out'
    out.mkdir(parents=True)
    operation_id = 'a' * 32
    (out / f'ci-native-intent-{operation_id}.json').write_text(json.dumps({'operationId': operation_id}))
    graph = {'objects': []}
    graph_path = tmp_path / 'native-graph.json'
    graph_path.write_text(json.dumps(graph))
    state = tmp_path / 'state'
    state.mkdir()
    calls = []
    real_safe_file = journal.safe_file
    def safe_file(path, mode):
        if Path(path).name == 'channel-request.json':
            return io.StringIO('{"commit":"' + 'a' * 40 + '"}')
        if Path(path).name == 'receipt.json':
            return io.StringIO(json.dumps({'sourceCommit': 'a' * 40, 'nativeGraph': graph}))
        return real_safe_file(path, mode)
    monkeypatch.setattr(journal, 'safe_file', safe_file)
    monkeypatch.setattr(ci, 'select_native_candidate', lambda *args: {'graph': graph_path})
    def observe(intent, sid, source, graph, names):
        calls.append(source)
        return {}
    monkeypatch.setattr(ci, 'collect_ci_native_observation', observe)
    effect = tmp_path / 'protected-effect'
    with pytest.raises(ValueError, match='reconciliation'):
        production_orchestrate_ci(state, [('preflight', command('pass')), ('build', command('import sys; open(sys.argv[1],"w").close()', effect))],
                                  lambda: ci.hold_unfinished_ci_runs(runs, 'S-1-5-21-1-2-3-4', 'b' * 40))
    assert calls == ['a' * 40]  # Simulated Windows observation, bound to the historical source.
    assert not effect.exists()
    assert not (state / '.ci-execution-pending.json').exists()


def test_collector_rejects_mismatched_operation_before_windows_boundary(monkeypatch):
    import base64
    import scripts.releases.semantier_native_ci as ci
    digest = base64.b64encode(b'\x01' * 32).decode()
    graph = {'sha256': 'c' * 64, 'root': 'build/root.msixbundle',
             'objects': [{'key': 'build/root.msixbundle', 'nativeDigest': digest}]}
    intent = {'schema': 1, 'operationId': 'd' * 32, 'sourceCommit': 'a' * 40,
              'graphSha256': 'c' * 64, 'operation': 'positive', 'actorPid': 123,
              'actorBirth': '2026-10-30T01:00:00+00:00', 'createdAt': '2026-10-30T01:01:00+00:00',
              'actor': {'sid': 'S-1-5-21-1-2-3-4', 'session_sid': 'S-1-5-21-1-2-3-4', 'session': 3},
              'root': 'http://127.0.0.1:1234/build/root.msixbundle',
              'uris': ['http://127.0.0.1:1234/build/root.msixbundle'], 'digests': [digest]}
    calls = []
    def boundary(args, **kwargs):
        calls.append(args)
        return 'C:\\source.ps1\n' if args[0] == 'wslpath' else '{"process":"same-incarnation","packages":[],"events":[]}'
    monkeypatch.setattr(ci.subprocess, 'check_output', boundary)
    result = ci.collect_ci_native_observation(intent, 'S-1-5-21-1-2-3-4', 'a' * 40, graph)
    assert result['process'] == 'same-incarnation'
    assert len(calls) == 2  # Simulated Windows boundary, not native evidence.
    for altered in ({**intent, 'sourceCommit': 'b' * 40}, {**intent, 'graphSha256': 'e' * 64},
                    {**intent, 'actor': {**intent['actor'], 'sid': 'S-1-5-21-9-9-9-9'}},
                    {**intent, 'uris': ['http://127.0.0.1:1234/other']}):
        with pytest.raises(ValueError, match='mismatch'):
            ci.collect_ci_native_observation(altered, 'S-1-5-21-1-2-3-4', 'a' * 40, graph)
    assert len(calls) == 2


def test_native_outcome_decision_never_uses_pid_or_package_for_lost_outcome():
    from scripts.releases.semantier_native_ci import ci_native_decision
    intent = {'operationId': 'a' * 32, 'operation': 'positive', 'root': 'http://127.0.0.1:1234/package', 'actor': {'sid': 'S-1-5-21-1-2-3-4'}}
    observed = {'process': 'absent', 'processIdentity': 'unavailable', 'packages': [{'full': 'matching'}], 'events': []}
    result = ci_native_decision(intent, None, observed, {})
    assert result['deployment'] == 'UNKNOWN'
    assert result['nextAction'] == 'HOLD'
    assert result['currentGraph'] == 'UNKNOWN' and result['operationEvidence'] == 'UNKNOWN'
    assert result['qualificationAction'] == 'HOLD' and result['cleanupAction'] == 'HOLD'
    assert result['cleanupAuthorized'] is False


def test_native_completed_outcome_allows_only_read_only_review():
    from scripts.releases.semantier_native_ci import ci_native_decision
    activity = '37e94520-1eb4-4f8c-ad6e-4e2fe892c036'
    intent = {'operationId': 'a' * 32, 'operation': 'positive', 'root': 'http://127.0.0.1:1234/package', 'actor': {'sid': 'S-1-5-21-1-2-3-4'}}
    outcome = {'schema': 1, 'operationId': intent['operationId'], 'result': {'status': 'SUCCESS', 'operationCompleted': True, 'hresult': '00000000', 'activityId': activity}}
    event = {'id': 478, 'activity': activity, 'sid': intent['actor']['sid'], 'uri': intent['root'], 'operation': '1', 'callingProcess': 'powershell.exe', 'full': 'Semantier.Agent.Acceptance_0.0.59.0_x64__p2m4tf4r6sydm', 'recordId': 4}
    start = {'id': 603, 'activity': activity, 'sid': intent['actor']['sid'], 'operation': '1', 'callingProcess': 'powershell.exe', 'recordId': 3}
    graph = {'objects': [{'kind': 'main-package', 'identity': 'Semantier.Agent.Acceptance', 'version': '0.0.59.0', 'arch': 'x64', 'publisher': 'CN=Example'}]}
    package = {'full': event['full'], 'publisher': 'CN=Example', 'version': '0.0.59.0', 'resourceId': ''}
    observed = {'events': [start, event], 'eventQuery': {'state': 'AVAILABLE'}, 'process': 'absent', 'packages': [package]}
    result = ci_native_decision(intent, outcome, observed, graph)
    assert result['deployment'] == 'SUCCESS_REPORTED'
    assert result['nextAction'] == 'REVIEW_INSTALLED_GRAPH'
    assert result['cleanupAuthorized'] is False
    changed = {**event, 'activity': 'b3eef0f9-26b8-4b9d-b88b-64b458018429'}
    assert ci_native_decision(intent, outcome, {**observed, 'events': [start, changed]}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'events': [event]}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'eventQuery': {'state': 'TRUNCATED'}}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'eventQuery': {'state': 'QUERY_ERROR'}}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'events': [start, start, event]}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'events': [{**start, 'recordId': 5}, event]}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, outcome, {**observed, 'processIdentity': 'mismatch'}, graph)['nextAction'] == 'REFUSE_IDENTITY_MISMATCH'
    assert ci_native_decision(intent, outcome, {**observed, 'processIdentity': 'QUERY_ERROR'}, graph)['nextAction'] == 'HOLD_PROCESS_QUERY_ERROR'
    assert ci_native_decision(intent, outcome, {**observed, 'packages': [{**package, 'full': 'unrelated'}]}, graph)['deployment'] == 'UNKNOWN'
    assert ci_native_decision(intent, None, observed, graph)['deployment'] == 'UNKNOWN'
    failed = {**outcome, 'result': {**outcome['result'], 'status': 'FAILURE', 'hresult': '80070005'}}
    failure = ci_native_decision(intent, failed, observed, graph)
    assert failure['deployment'] == 'UNKNOWN' and failure['reportedOutcome'] == 'FAILURE'
    assert failure['nextAction'] == 'HOLD' and failure['cleanupAuthorized'] is False


def test_existing_marker_does_not_hide_read_only_discovery(tmp_path):
    marker = tmp_path / '.ci-execution-pending.json'
    marker.write_text('unresolved')
    calls = []
    with pytest.raises(ValueError, match='requires reconciliation'):
        production_orchestrate_ci(tmp_path, [('preflight', command('pass'))], lambda: calls.append('discovered') or True)
    assert calls == ['discovered']
    assert marker.read_text() == 'unresolved'


def test_mutation_guard_runs_before_first_protected_phase(tmp_path):
    built = tmp_path / 'built'
    def refuse():
        raise ValueError('native dispatch remains closed')
    with pytest.raises(ValueError, match='native dispatch remains closed'):
        production_orchestrate_ci(tmp_path, [('preflight', command('pass')), ('build', command('import sys; open(sys.argv[1],"w").close()', built))], lambda: True, mutation_guard=refuse)
    assert not built.exists()
    assert not (tmp_path / '.ci-execution-pending.json').exists()


def test_stage_revalidation_failure_prevents_stage_effect(tmp_path):
    stage_effect=tmp_path/'stage-effect'
    def revalidate():
        raise ValueError('Runtime changed before staging')
    with pytest.raises(ValueError,match='Runtime changed before staging'):
        production_orchestrate_ci(tmp_path,[('preflight',command('pass')),
            ('build',command('pass')),
            ('stage',command('import sys; open(sys.argv[1],"w").close()',stage_effect))],
            lambda:True,stage_preflight=revalidate)
    assert not stage_effect.exists()
    success=tmp_path/'success';success.mkdir();checked=success/'checked';effect=success/'effect'
    production_orchestrate_ci(success,[('preflight',command('pass')),
        ('stage',command('import pathlib,sys; assert pathlib.Path(sys.argv[1]).exists(); pathlib.Path(sys.argv[2]).touch()',checked,effect))],
        lambda:True,stage_preflight=lambda:checked.touch())
    assert effect.exists()


def test_marker_replaced_during_stage_revalidation_prevents_stage(tmp_path):
    marker=tmp_path/'.ci-execution-pending.json';effect=tmp_path/'stage-effect'
    def replace():
        marker.rename(tmp_path/'original-marker')
        marker.write_text('replacement must survive')
    with pytest.raises(ValueError,match='marker identity changed'):
        production_orchestrate_ci(tmp_path,[('preflight',command('pass')),
            ('build',command('pass')),
            ('stage',command('import pathlib,sys; pathlib.Path(sys.argv[1]).touch()',effect))],
            lambda:True,stage_preflight=replace)
    assert marker.read_text()=='replacement must survive'
    assert not effect.exists()


def test_discovery_reports_all_retained_run_holds(tmp_path):
    from scripts.releases.semantier_native_ci import hold_unfinished_ci_runs
    for name in ('client-feedback-ci-123-1', 'client-feedback-ci-456-1'):
        (tmp_path / name / 'out').mkdir(parents=True)
    with pytest.raises(ValueError) as error:
        hold_unfinished_ci_runs(tmp_path)
    assert 'client-feedback-ci-123-1' in str(error.value)
    assert 'client-feedback-ci-456-1' in str(error.value)


@pytest.mark.parametrize(('values', 'expected'), [
    ({'DeploymentInProgress': True, 'Servicing': False, 'IsPartiallyStaged': False, 'VerifyIsOK': True}, 'SERVICING'),
    ({'DeploymentInProgress': False, 'Servicing': False, 'IsPartiallyStaged': True, 'VerifyIsOK': True}, 'PARTIAL_OR_UNHEALTHY'),
    ({'DeploymentInProgress': False, 'Servicing': False, 'IsPartiallyStaged': False, 'VerifyIsOK': False}, 'PARTIAL_OR_UNHEALTHY'),
    ({'DeploymentInProgress': False, 'Servicing': False, 'IsPartiallyStaged': False, 'VerifyIsOK': True}, 'ALL_OBSERVED_HEALTHY_UNBOUND'),
])
def test_winrt_package_status_is_not_historical_operation_evidence(values, expected):
    from scripts.releases.semantier_native_ci import ci_native_decision
    status = {key: {'available': True, 'value': value} for key, value in values.items()}
    intent = {'operationId': 'a' * 32, 'operation': 'positive', 'actor': {'sid': 'S-1-5-21-1-2-3-4'}, 'root': 'http://127.0.0.1:1234/package'}
    result = ci_native_decision(intent, None, {'packages': [{'winrt': {'presence': 'PRESENT', 'status': status}}]}, {})
    assert result['currentGraph'] == 'UNKNOWN'
    assert result['observedPackageCondition'] == expected
    assert result['deployment'] == 'UNKNOWN'
    assert result['cleanupAuthorized'] is False and result['continueQualification'] is False


def test_winrt_unavailable_property_is_unknown():
    from scripts.releases.semantier_native_ci import ci_native_decision
    result = ci_native_decision({}, None, {'packages': [{'winrt': {'presence': 'PRESENT', 'status': {'VerifyIsOK': {'available': False, 'value': None}}}}]}, {})
    assert result['currentGraph'] == 'UNKNOWN' and result['deployment'] == 'UNKNOWN'


def test_cloudflare_runtime_custody_readback_and_origin(tmp_path, monkeypatch):
    import hashlib, json
    import scripts.releases.semantier_native_ci as ci
    primary=tmp_path/'incoming';primary.mkdir()
    target=tmp_path/'incoming_r2';target.mkdir()
    config={'base':'https://update2.semantier.com','incoming':'/fixture/qualified-v3/incoming_r2'}
    def runtime_call(argv, **kwargs):
        if 'ClientOperations' in argv[-1]: return json.dumps(config)
        return hashlib.sha256((target/argv[-1]).read_bytes()).hexdigest()
    monkeypatch.setattr(ci.subprocess,'check_output',runtime_call)
    ci.verify_cloudflare_custody('fixture-container',primary)
    assert list(target.iterdir())==[]
    config['base']='https://example.invalid'
    with pytest.raises(ValueError,match='target configuration mismatch'):
        ci.verify_cloudflare_custody('fixture-container',primary)
    assert list(target.iterdir())==[]
    config['base']='https://update2.semantier.com'
    def wrong_readback(argv, **kwargs):
        return json.dumps(config) if 'ClientOperations' in argv[-1] else '0'*64
    monkeypatch.setattr(ci.subprocess,'check_output',wrong_readback)
    with pytest.raises(ValueError,match='Runtime readback mismatch'):
        ci.verify_cloudflare_custody('fixture-container',primary)
    assert list(target.iterdir())==[]


def test_runtime_source_identity_requires_exact_labels_and_inventory(monkeypatch):
    import json
    import scripts.releases.semantier_native_ci as ci
    labels={'io.semantier.source.parent':ci.CI_RUNTIME_PARENT,'io.semantier.source.native':ci.CI_RUNTIME_NATIVE}
    monkeypatch.setattr(ci.subprocess,'check_output',lambda *args,**kwargs:json.dumps(labels))
    calls=[]
    monkeypatch.setattr(ci.subprocess,'run',lambda argv,**kwargs:calls.append((argv,kwargs)))
    ci.verify_runtime_identity('fixture-container')
    assert len(calls)==1 and ci.CI_RUNTIME_INVENTORY in calls[0][0][-1] and calls[0][1]['check'] is True
    labels['io.semantier.source.native']='0'*40
    with pytest.raises(ValueError,match='source labels mismatch'):
        ci.verify_runtime_identity('fixture-container')
    assert len(calls)==1


def test_installed_content_uses_existing_inventory_verifier(tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil
    from scripts.releases.semantier_native_ci import verify_ci_installed_content
    import scripts.releases.semantier_native_ci as ci
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable')
    root = tmp_path / 'installed'; root.mkdir()
    (root / 'payload.txt').write_bytes(b'expected')
    files = [{'path': 'payload.txt', 'sha256': hashlib.sha256(b'expected').hexdigest(), 'size': 8}]
    digest = hashlib.sha256(json.dumps(files, separators=(',', ':')).encode()).hexdigest()
    plan = tmp_path / 'plan.json'; graph = tmp_path / 'graph.json'
    plan.write_text(json.dumps({'files': files, 'assignment': {'payload.txt': 'main'}, 'payloadInventorySha256': digest}))
    graph.write_text(json.dumps({'payloadInventorySha256': digest}))
    monkeypatch.setenv('SEMANTIER_CLIENT_RECEIPT_KEY', 'fixture-secret')
    original = ci.subprocess.check_output
    def check_child(args, **kwargs):
        assert 'SEMANTIER_CLIENT_RECEIPT_KEY' not in kwargs['env']
        return original(args, **kwargs)
    monkeypatch.setattr(ci.subprocess, 'check_output', check_child)
    result = verify_ci_installed_content(node, str(root), str(plan), str(graph))
    assert result['filesChecked'] == 1 and result['payloadInventorySha256'] == digest
    (root / 'payload.txt').write_bytes(b'tampered')
    with pytest.raises(subprocess.CalledProcessError):
        verify_ci_installed_content(node, str(root), str(plan), str(graph))
