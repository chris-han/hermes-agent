"""Owned registered Windows update transaction. No package mutation in preflight.

Run with Windows Python in the explicitly authorized disposable interactive user.
The SID argument constrains that external authorization; it cannot grant it.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import stat
import threading
from datetime import UTC, datetime


class Interrupted(RuntimeError):
    pass


@contextlib.contextmanager
def safe_file(path, mode):
    path = Path(path)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400 or before.st_nlink != 1:
            raise ValueError('Journal/lock link refused')
    except FileNotFoundError:
        before = None
    flags = os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    if mode == 'a': flags |= os.O_APPEND | os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    stream = os.fdopen(descriptor, 'a+' if mode == 'a' else 'r+', encoding='utf-8')
    try:
        opened = os.fstat(stream.fileno()); current = path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino) or opened.st_nlink != 1 or getattr(current, 'st_file_attributes', 0) & 0x400 or (before and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)):
            raise ValueError('Journal/lock physical identity changed')
        yield stream
    finally:
        stream.close()


@contextlib.contextmanager
def exclusive(path):
    # OS lock, released on process death; no age-based lock stealing or deletion.
    with safe_file(path, 'a') as handle:
        with _locked(handle):
            yield


@contextlib.contextmanager
def _locked(handle):
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write('0'); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


@contextlib.contextmanager
def held_windows_root(root, expected_id):
    import ctypes
    from ctypes import wintypes
    class Info(ctypes.Structure):
        _fields_ = [('attr', wintypes.DWORD), ('created', wintypes.FILETIME), ('accessed', wintypes.FILETIME),
                    ('written', wintypes.FILETIME), ('volume', wintypes.DWORD), ('high', wintypes.DWORD),
                    ('low', wintypes.DWORD), ('links', wintypes.DWORD), ('indexHigh', wintypes.DWORD), ('indexLow', wintypes.DWORD)]
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(Info)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.CreateFileW(str(root), 0, 1, None, 3, 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = Info()
        if not api.GetFileInformationByHandle(handle, ctypes.byref(info)): raise ctypes.WinError(ctypes.get_last_error())
        if f'{info.volume}:{info.indexHigh}:{info.indexLow}' != expected_id or info.attr & 0x400:
            raise ValueError('Root physical identity changed before journal access')
        yield
    finally: api.CloseHandle(handle)


class Transaction:
    def __init__(self, file, platform):
        self.file = Path(file); self.platform = platform; self.fault = None

    def events(self):
        if not self.file.exists(): return []
        # A torn final write is an explicit hold, never silently discarded.
        with safe_file(self.file, 'r') as stream:
            rows = [json.loads(line) for line in stream.read().splitlines()]
        pending = None
        for index, row in enumerate(rows):
            if index == 0:
                if row.get('kind') != 'owner' or not isinstance(row.get('owner'), dict): raise ValueError('Invalid owner record')
                continue
            if row.get('kind') == 'intent':
                if pending or row.get('action') not in ('register','launch','update','remove') or not isinstance(row.get('args'), dict) or not isinstance(row.get('before'), dict): raise ValueError('Invalid intent record')
                if row['action'] in ('register','update') and not row['args']: raise ValueError('Empty package intent')
                pending = row
            elif row.get('kind') == 'outcome':
                if not pending or row.get('intent_at') != pending['at'] or row.get('action') != pending['action'] or row.get('status') not in ('performed','performed_partial','not_performed') or not isinstance(row.get('observed'), dict): raise ValueError('Unlinked outcome')
                pending = None
            elif row.get('kind') == 'uncertain':
                if not pending or row.get('intent_at') != pending['at']: raise ValueError('Unlinked uncertainty')
            elif row.get('kind') not in ('baseline_ready','successor_ready','conservation_before','conservation_failed','acceptance','cleanup'): raise ValueError('Unknown journal record')
        return rows

    def record(self, kind, **facts):
        row = {'kind': kind, 'at': datetime.now(UTC).isoformat(), **facts}
        with safe_file(self.file, 'a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())
        return row

    def initialize(self, owner):
        if self.events(): raise ValueError('Existing transaction: reconcile it')
        self.record('owner', owner=owner)

    def pending(self):
        rows = self.events(); pending = None
        for row in rows:
            if row['kind'] == 'intent':
                if pending: raise ValueError('Overlapping intent')
                pending = row
            elif row['kind'] == 'outcome': pending = None
        return pending

    def validate(self):
        rows = self.events()
        if not rows or rows[0]['kind'] != 'owner': raise ValueError('Missing owner')
        self.platform.validate(rows[0]['owner'])

    @staticmethod
    def matches(expected, actual):
        return bool(expected) and isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items())

    def expected(self, action, args, before, after):
        if action == 'register': return self.matches(args, after['package'])
        if action == 'remove': return after['package'] is None and after['process'] is None
        if action == 'launch': return after['package'] == before['package'] and after['process'] is not None
        if action == 'update': return self.platform.successor(args, before, after)
        raise ValueError('Unknown operation')

    def assert_owned(self, state, require_process=False):
        rows = self.events()
        packages = [row['observed']['package'] for row in rows if row['kind']=='outcome' and row['action'] in ('register','update') and row['status'] in ('performed','performed_partial')]
        if state['package'] not in packages: raise ValueError('Registration replaced outside transaction')
        processes = [row['observed']['process'] for row in rows if row['kind']=='outcome' and row['action'] in ('launch','update') and row['status']=='performed']
        if state['process'] is not None and state['process'] not in processes: raise ValueError('Process incarnation replaced outside transaction')
        if require_process and state['process'] is None: raise ValueError('Owned running process required')

    def operation(self, action, args):
        self.validate()
        if self.pending(): raise ValueError('Pending intent: reconcile first')
        before = self.platform.observe()
        if action != 'register': self.assert_owned(before, require_process=action=='update')
        intent = self.record('intent', action=action, args=args, before=before)
        self.validate(); self.platform.check(action, args, before) if hasattr(self.platform, 'check') else None
        if self.fault == 'before_action': raise Interrupted('before action')
        try:
            self.platform.perform(action, args)
            if self.fault == 'after_action': raise Interrupted('after action')
            after = self.platform.observe()
            if not self.expected(action, args, before, after): raise ValueError('Observed outcome mismatch')
            self.record('outcome', intent_at=intent['at'], action=action, status='performed', observed=after)
            return after
        except Exception as error:
            self.record('uncertain', intent_at=intent['at'], error=type(error).__name__)
            raise

    def reconcile(self):
        self.validate(); intent = self.pending()
        if not intent: return 'settled'
        now = self.platform.observe(); before = intent['before']; action = intent['action']
        if hasattr(self.platform, 'recovery_check'): self.platform.recovery_check(intent)
        if action == 'update' and self.matches(intent['args'], now['package']) and now['package']['family'] == before['package']['family'] and now['process'] is None:
            self.record('outcome', intent_at=intent['at'], action=action, status='performed_partial', observed=now)
            return 'performed_partial'
        if action == 'remove' and now['package'] == before['package'] and now['process'] is None:
            self.record('outcome', intent_at=intent['at'], action=action, status='not_performed', observed=now)
            return 'not_performed'
        if action == 'launch' and now != before: raise ValueError('Unrecorded launch incarnation: preserve for inspection')
        if self.expected(action, intent['args'], before, now): status = 'performed'
        elif now == before and action in ('register', 'remove', 'launch'): status = 'not_performed'
        else: raise ValueError('Uncertain OS state: preserve package/process and evidence')
        self.record('outcome', intent_at=intent['at'], action=action, status=status, observed=now)
        return status


class Windows:
    def __init__(self, request):
        self.request = request
        import ctypes
        directory = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(directory, len(directory))
        if not length or length >= len(directory): raise ValueError('Actual Windows directory unavailable')
        self.powershell = str(Path(directory.value)/'System32/WindowsPowerShell/v1.0/powershell.exe')
        if not Path(self.powershell).is_file(): raise ValueError('Host PowerShell unavailable')
    def call(self, action, **facts):
        payload = {**self.request, **facts, 'action': action}
        script = Path(__file__).parent/'e2e-assets/semantier-registered-platform.ps1'
        result = subprocess.run([self.powershell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(script)],
            input=json.dumps(payload), text=True, capture_output=True, check=False,
            timeout=1800 if action == 'update' else 60 if action == 'download' else 300)
        if result.returncode: raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout.lstrip('\ufeff'))
    def observe(self):
        now = self.call('observe')
        if not now.get('session_sid') or now['session_sid'] != now.get('sid'):
            raise ValueError('WTS signed-in account differs from process account')
        return now
    def validate(self, owner):
        now = self.observe()
        if not owner.get('session_sid') or owner['session_sid'] != owner.get('sid'):
            raise ValueError('Recorded WTS account binding unavailable; preserve transaction')
        if any(now[k] != owner[k] for k in ('sid','session','session_sid','root_id')): raise ValueError('OS owner or physical root changed')
    def recovery_check(self, intent):
        if intent['action'] in ('register','update'):
            side = 'baseline' if intent['action']=='register' else 'target'
            if intent['args'] != self.call('artifact',side=side): raise ValueError('Intent differs from verified artifact')

    def check(self, action, args, before):
        self.precondition = before
        self.recovery_check({'action':action,'args':args})
        if action == 'register' and before['package'] is not None: raise ValueError('Conflicting package')
        if action in ('launch','remove','update') and before['package'] is None: raise ValueError('Owned package absent')
        if action == 'remove':
            owned = self.request['owned_packages']
            if before['package'] not in owned: raise ValueError('Unowned package: preserve it')
    def perform(self, action, args):
        if action != 'update': return self.call(action, operation=args, expected_before=self.precondition)
        # Capture while the production handoff cache exists; successor startup
        # may retire it. A missed receipt is a failure, never inferred success.
        stop = threading.Event(); captured = []
        def watch():
            while not stop.is_set():
                try:
                    proof = self.call('download')
                    file = Path(self.request['root'])/'download-readback.json'
                    if file.exists(): raise ValueError('Existing download evidence requires inspection')
                    with safe_file(file, 'a') as stream:
                        stream.write(json.dumps(proof)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    captured.append(proof); return
                except (RuntimeError, ValueError, subprocess.TimeoutExpired): stop.wait(0.25)
        thread = threading.Thread(target=watch); thread.start()
        try:
            result = self.call(action, operation=args, expected_before=self.precondition)
        finally:
            stop.set(); thread.join()
        if not captured: raise ValueError('Production handoff readback missing; preserve actual update state')
        return result

    def successor(self, args, before, after):
        return (Transaction.matches(args, after['package']) and after['process'] is not None and before['process'] is not None
            and after['process']['birth'] > before['process']['birth'] and after['process']['pid'] != before['process']['pid']
            and after['package']['family'] == before['package']['family'])


def run_upgrade(tx, platform):
    tx.reconcile()
    if not any(row['kind']=='baseline_ready' for row in tx.events()): raise ValueError('Baseline readiness evidence required before publication/update')
    before = platform.call('conservation')
    tx.record('conservation_before', facts=before)
    target = platform.call('artifact', side='target')
    current = platform.observe()
    if current['package'] is None or current['process'] is None:
        raise ValueError('Observed registered running baseline required')
    if tuple(map(int, target['version'].split('.'))) <= tuple(map(int, current['package']['version'].split('.'))):
        raise ValueError('Target must be newer')
    tx.operation('update', target)
    ready = platform.call('ready')
    tx.record('successor_ready', facts=ready, observed=platform.observe())
    after = platform.call('conservation')
    if before != after:
        tx.record('conservation_failed', facts=after)
        raise ValueError('Conservation mismatch; preserve state')
    tx.record('acceptance', status='registered_readback_and_selected_sentinels_passed',
              native_acceptance='requires complete original matrices', facts=after)


def run_baseline(tx, platform, owner):
    if not tx.events():
        if owner['package'] is not None or owner['process'] is not None: raise ValueError('Existing package/process: refuse new transaction')
        tx.initialize(owner)
    tx.reconcile()
    if platform.observe()['package'] is None: tx.operation('register',platform.call('artifact',side='baseline'))
    tx.assert_owned(platform.observe())
    if platform.observe()['process'] is None: tx.operation('launch',{})
    ready=platform.call('ready')
    tx.record('baseline_ready', facts=ready, observed=platform.observe())
    print('Baseline retained; observe/seed conservation files before publication and upgrade')


def run_cleanup(tx, platform, request):
    tx.reconcile()
    rows=tx.events(); owned=[r['observed']['package'] for r in rows if r['kind']=='outcome' and r['action'] in ('register','update') and r['status'] in ('performed','performed_partial') and r['observed']['package'] is not None]
    request['owned_packages']=owned
    request['owned_processes']=[r['observed']['process'] for r in rows if r['kind']=='outcome' and r['status']=='performed' and r['observed']['process'] is not None]
    request['owned_gateways']=[r['facts']['gateway'] for r in rows if r['kind'] in ('baseline_ready','successor_ready') and r['facts'].get('gateway') is not None]
    current=platform.observe()
    if current['package'] is not None: tx.operation('remove',{})
    tx.record('cleanup',status='owned_package_removed_data_and_evidence_retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','baseline','upgrade','reconcile','cleanup'])
    p.add_argument('--request',type=Path,required=True)
    a=p.parse_args()
    if sys.platform != 'win32': raise SystemExit('Use native Windows Python in the authorized disposable session')
    request=json.loads(a.request.read_text(encoding='utf-8-sig')); platform=Windows(request)
    owner=platform.observe()  # OS refusal precedes journal/root writes.
    if a.mode=='preflight':
        baseline=platform.call('artifact',side='baseline')
        if owner['package'] is not None: raise ValueError('Conflicting baseline registration')
        print(json.dumps({'scope':'READ_ONLY','observed':owner,'baseline':baseline})); return
    root=Path(request['root']); journal=root/'registered-update.jsonl'
    with held_windows_root(root, owner['root_id']), exclusive(root/'registered-update.lock'):
        tx=Transaction(journal,platform)
        handlers = {
            'baseline': lambda: run_baseline(tx, platform, owner),
            'reconcile': lambda: print(tx.reconcile()),
            'upgrade': lambda: run_upgrade(tx, platform),
            'cleanup': lambda: run_cleanup(tx, platform, request),
        }
        handlers[a.mode]()

if __name__=='__main__': main()
