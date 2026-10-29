"""Owned registered Windows update transaction. No package mutation in preflight.

Run with Windows Python in the explicitly authorized disposable interactive user.
The SID argument constrains that external authorization; it cannot grant it.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import stat
import threading
from datetime import UTC, datetime


NATIVE_QUALIFICATION='wd1-native-layout-qualification'


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
    flags = (os.O_RDONLY if mode == 'r' else os.O_RDWR) | getattr(os, 'O_NOFOLLOW', 0)
    if mode == 'a': flags |= os.O_APPEND | os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    stream = os.fdopen(descriptor, 'a+' if mode == 'a' else 'r', encoding='utf-8', newline='')
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
            elif row.get('kind') not in ('baseline_ready','successor_ready','conservation_before','conservation_failed','acceptance','cleanup','native_qualification','native_qualification_ready','semantic_before','semantic_preservation'): raise ValueError('Unknown journal record')
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

    @staticmethod
    def related(state):
        rows = state.get('related', [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not row.get('full') for row in rows):
            raise ValueError('Related package observation unavailable')
        if len({row['full'] for row in rows}) != len(rows):
            raise ValueError('Related package observation ambiguous')
        return rows

    @classmethod
    def package_matches(cls, args, state):
        expected = {row['full']: row for row in cls.related(args)}
        actual = {row['full']: row for row in cls.related(state)}
        return (cls.matches({k:v for k,v in args.items() if k != 'related'}, state['package'])
            and expected.keys() == actual.keys()
            and all(cls.matches(row, actual[full]) for full, row in expected.items()))

    def expected(self, action, args, before, after):
        if action == 'register': return self.package_matches(args, after)
        if action == 'remove': return after['package'] is None and after['process'] is None and self.related(after) == []
        if action == 'launch': return after['package'] == before['package'] and after['process'] is not None and self.related(after) == self.related(before)
        if action == 'update': return self.package_matches(args, after) and self.platform.successor(args, before, after)
        raise ValueError('Unknown operation')

    def assert_owned(self, state, require_process=False):
        rows = self.events()
        packages = [row['observed']['package'] for row in rows if row['kind']=='outcome' and row['action'] in ('register','update') and row['status'] in ('performed','performed_partial')]
        if state['package'] not in packages: raise ValueError('Registration replaced outside transaction')
        if not any(row['observed']['package'] == state['package'] and self.related(row['observed']) == self.related(state)
                   for row in rows if row['kind']=='outcome' and row['action'] in ('register','update') and row['status'] in ('performed','performed_partial')):
            raise ValueError('Related package set replaced outside transaction')
        processes = [row['observed']['process'] for row in rows if row['kind']=='outcome' and row['action'] in ('launch','update') and row['status']=='performed']
        if state['process'] is not None and state['process'] not in processes: raise ValueError('Process incarnation replaced outside transaction')
        if require_process and state['process'] is None: raise ValueError('Owned running process required')

    def operation(self, action, args):
        self.validate()
        if self.pending(): raise ValueError('Pending intent: reconcile first')
        before = self.platform.observe()
        if action != 'register': self.assert_owned(before, require_process=action=='update')
        self.platform.check(action, args, before) if hasattr(self.platform, 'check') else None
        intent = self.record('intent', action=action, args=args, before=before)
        self.validate()
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
        if action=='register' and hasattr(self.platform,'reconcile_registration'):
            registration=self.platform.reconcile_registration(intent,now)
            if registration is not None:
                status=registration['status']
                if status=='performed' and not self.expected(action,intent['args'],before,now):raise ValueError('Native registered graph differs')
                if status=='not_performed' and now!=before:raise ValueError('Native failed registration changed state')
                if status not in ('performed','not_performed'):raise ValueError('Native registration remains UNKNOWN')
                self.record('outcome',intent_at=intent['at'],action=action,status=status,observed=now,native_registration_proof=registration['proof'])
                return status
        if action=='update' and now['process'] is None and hasattr(self.platform,'reconcile_native'):
            native=self.platform.reconcile_native(intent,now)
            if native in ('BASELINE_INTACT_FAILED','NOT_STARTED'):
                self.record('outcome',intent_at=intent['at'],action=action,status='not_performed',observed=now)
                return 'not_performed'
        if action == 'update' and self.package_matches(intent['args'], now) and now['package']['family'] == before['package']['family'] and now['process'] is None:
            self.record('outcome', intent_at=intent['at'], action=action, status='performed_partial', observed=now)
            return 'performed_partial'
        if action == 'remove' and now['package'] == before['package'] and now['process'] is None and self.related(now) == self.related(before):
            self.record('outcome', intent_at=intent['at'], action=action, status='not_performed', observed=now)
            return 'not_performed'
        if action == 'launch' and now != before: raise ValueError('Unrecorded launch incarnation: preserve for inspection')
        if self.expected(action, intent['args'], before, now): status = 'performed'
        elif now == before and action in ('register', 'remove', 'launch'): status = 'not_performed'
        else: raise ValueError('Uncertain OS state: preserve package/process and evidence')
        self.record('outcome', intent_at=intent['at'], action=action, status=status, observed=now)
        return status


class Windows:
    def __init__(self, request, proof_source=None):
        self.request = request
        self.proof_source = proof_source
        import ctypes
        directory = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(directory, len(directory))
        if not length or length >= len(directory): raise ValueError('Actual Windows directory unavailable')
        self.powershell = str(Path(directory.value)/'System32/WindowsPowerShell/v1.0/powershell.exe')
        if not Path(self.powershell).is_file(): raise ValueError('Host PowerShell unavailable')
    def proof_sources(self, root):
        paths=[root/'tests/install/semantier_registered_update.py', *(root/'tests/install/e2e-assets'/name for name in (
            'semantier-registered-platform.ps1','native-package-deployment.ps1','native-payload-locations.mjs',
            'native-payload-probe.py','windows-bundle-plugin-smoke.ps1','windows-bundle-metadata.ps1','desktop-smoke-windows.ps1')),
            root/'apps/desktop/scripts/native-package-digest.ps1',root/'scripts/releases/semantier_windows.py',
            root/'hermes_cli/release_channels.py']
        return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def proof_binding(self, source_root=None):
        root=Path(source_root) if source_root else Path(__file__).resolve().parents[2]
        sources=self.proof_sources(root)
        host={'windows':tuple(sys.getwindowsversion()),'powershell':hashlib.sha256(Path(self.powershell).read_bytes()).hexdigest()}
        return hashlib.sha256(json.dumps({'request':self.request,'sources':sources,'host':host},sort_keys=True).encode()).hexdigest()

    def proof_verifiers(self, root):
        # Observation/relaunch fixes do not change completed byte/import/model claims.
        sources=self.proof_sources(root)
        sources.pop(str(Path('tests/install/semantier_registered_update.py')))
        name=Path('tests/install/e2e-assets/semantier-registered-platform.ps1')
        text=(root/name).read_text(encoding='utf-8-sig')
        prefix=text[:text.index(' function Artifact(')]
        start=text.index('  native-proof {')
        native=text[start:text.index('\n  ready {',start)]
        sources[str(name)]=hashlib.sha256((prefix+native).encode()).hexdigest()
        return sources

    def proof_bindings(self, current):
        if not self.proof_source:return {current}
        path,expected=self.proof_source
        with safe_file(path,'r') as stream:raw=stream.read()
        if path.name!='source-binding.json' or len(raw)>4*1024*1024 or hashlib.sha256(raw.encode()).hexdigest()!=expected:
            raise ValueError('Pinned prior proof source binding mismatch')
        manifest=json.loads(raw);root=path.parent
        for row in manifest['files']:
            name=row['path'];relative=PurePosixPath(name)
            if relative.is_absolute() or relative.as_posix()!=name or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('Prior proof source path refused')
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Prior proof source bytes changed')
        if self.proof_verifiers(root)!=self.proof_verifiers(Path(__file__).resolve().parents[2]):return {current}
        # Recompute the original opaque binding: host and complete request must still match.
        return {current,self.proof_binding(root)}

    def call(self, action, **facts):
        payload = {**self.request, **facts, 'action': action}
        script = Path(__file__).parent/'e2e-assets/semantier-registered-platform.ps1'
        result = subprocess.run([self.powershell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(script)],
            input=json.dumps(payload), text=True, capture_output=True, check=False,
            timeout=2400 if (action=='update' and self.request.get('target',{}).get('native_graph')) or (action=='register' and self.request.get('baseline',{}).get('native_graph')) else 1800 if action in ('update','native-proof') or (action=='register' and self.request.get('native_origin')) else 60 if action == 'download' else 300)
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
        if owner.get('purpose')==NATIVE_QUALIFICATION and owner.get('native_origin')!=self.request.get('native_origin'):raise ValueError('Native origin changed across transaction')
        if any(now[k] != owner[k] for k in ('sid','session','session_sid','root_id')): raise ValueError('OS owner or physical root changed')
    def recovery_check(self, intent):
        if intent['action'] in ('register','update'):
            side = 'baseline' if intent['action']=='register' and self.request.get('purpose')!=NATIVE_QUALIFICATION else 'target'
            if intent['args'] != self.call('artifact',side=side): raise ValueError('Intent differs from verified artifact')

    def reconcile_registration(self,intent,now):
        side='target' if self.request.get('purpose')==NATIVE_QUALIFICATION else 'baseline'
        if self.request.get(side,{}).get('native_graph') is None:return None
        proof=self.call('native-registration-proof',registration_intent_at=intent['at'],operation=intent['args'])
        if any(proof['observed'].get(key)!=now.get(key) for key in ('sid','session_sid','session','root_id','package','process','related')):
            raise ValueError('Native registration observation changed')
        if proof.get('operationCompleted') is not True:raise ValueError('Native registration remains UNKNOWN')
        if proof['status'] in ('SUCCESS','RECONCILED_OS_SUCCESS'):return {'status':'performed','proof':proof}
        if proof['status']=='FAILURE':return {'status':'not_performed','proof':proof}
        raise ValueError('Native registration remains UNKNOWN')

    def semantic_owner(self):
        from types import ModuleType
        binding=self.request.get('semantic_observer',{})
        file=Path(binding.get('path',''))
        with safe_file(file,'r') as stream:source=stream.read()
        if hashlib.sha256(source.encode()).hexdigest()!=binding.get('sha256'):
            raise ValueError('Pinned semantic observer changed')
        module=ModuleType('registered_semantic_observer');module.__file__=str(file)
        exec(compile(source,str(file),'exec'),module.__dict__)
        return module

    def capture_semantics(self,prior=None):
        from uuid import uuid4
        specs=self.request.get('semantic_databases')
        if specs is None:
            if self.request.get('target',{}).get('native_graph') is not None:
                raise ValueError('Native upgrade requires explicit nonempty semantic SQLite fixtures')
            return None
        if not isinstance(specs,list) or not 0<len(specs)<=8:raise ValueError('Bounded semantic database set required')
        owner=self.semantic_owner();root=owner.unlinked(Path(self.request['root']))
        selected={}
        for spec in specs:
            relative=PurePosixPath(spec['path'])
            if (not relative.parts or relative.is_absolute() or relative.as_posix()!=spec['path'] or '..' in relative.parts
                or ':' in spec['path'] or '\\' in spec['path'] or relative.parts[0]!='userdata' or spec['path'] in selected):
                raise ValueError('Unique owned relative semantic database path required')
            source=owner.unlinked(root/relative)
            if source.suffix not in ('.db','.sqlite','.sqlite3') or source.stat().st_size>2*1024**3:
                raise ValueError('Bounded SQLite source required')
            tables=spec.get('tables')
            if not isinstance(tables,dict) or not tables or any(not isinstance(columns,list) or not columns or any(not isinstance(column,str) for column in columns) for columns in tables.values()):
                raise ValueError('Explicit nonempty semantic table/column contract required')
            selected[spec['path']]=(source,tables)
        contracts={name:tables for name,(_,tables) in selected.items()}
        if prior is not None and (prior['observer_sha256']!=self.request['semantic_observer']['sha256'] or prior['table_contracts']!=contracts or set(prior['databases'])!=set(selected)):
            raise ValueError('Semantic observer/database scope changed')
        directory=root/('conservation-'+uuid4().hex);directory.mkdir()
        snapshots={}
        for index,(name,(source,tables)) in enumerate(selected.items()):
            snapshots[name]=owner.backup_database_snapshot(source,directory/(str(index)+'.db'),prior['databases'][name] if prior else None,tables)
        result={'observer_sha256':self.request['semantic_observer']['sha256'],'table_contracts':contracts,'databases':snapshots,'backup_directory':str(directory)}
        if len(json.dumps(result).encode())>4*1024*1024:raise ValueError('Semantic evidence exceeds bound; preserve backups')
        return result

    def verify_semantics(self,before):
        if before is None:return None
        after=self.capture_semantics(before);owner=self.semantic_owner()
        return {'scope':'Explicit SQLite fixture rows/columns only; not full native acceptance',
                'observer_sha256':before['observer_sha256'],'backup_directory':after['backup_directory'],
                'databases':{name:owner.verify_database(prior,after['databases'][name]) for name,prior in before['databases'].items()}}

    def check(self, action, args, before):
        self.precondition = before
        if action=='update' and self.request.get('target',{}).get('native_graph'):validate_native_update_inputs(self.request)
        self.recovery_check({'action':action,'args':args})
        if action == 'register' and before['package'] is not None: raise ValueError('Conflicting package')
        if action in ('launch','remove','update') and before['package'] is None: raise ValueError('Owned package absent')
        if action == 'remove':
            owned = self.request['owned_packages']
            if before['package'] not in owned: raise ValueError('Unowned package: preserve it')
    def perform(self, action, args):
        if action != 'update': return self.call(action, operation=args, expected_before=self.precondition)
        if self.request.get('target',{}).get('native_graph'):
            result=self.call(action,operation=args,expected_before=self.precondition)
            proof=self.call('download',expected_before=self.precondition)
            verify_native_handoff(proof,self.request,self.precondition)
            file=Path(self.request['root'])/'download-readback.json'
            if file.exists():raise ValueError('Existing native handoff evidence requires inspection')
            with safe_file(file,'a') as stream:
                stream.write(json.dumps(proof)+'\n');stream.flush();os.fsync(stream.fileno())
            return result
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

    def reconcile_native(self, intent, now):
        if not self.request.get('target',{}).get('native_graph'):return None
        proof=self.call('native-reconcile',native_intent_at=intent['at'])
        status=verify_native_reconciliation(proof,self.request,intent['before'],now,intent['args'])
        self.call('native-reconciled',expected_before=now,native_intent_at=intent['at'],
                  native_directory=proof['directory'],native_proof_hashes=proof['fileSha256'],native_status=status)
        return status

    def successor(self, args, before, after):
        if self.request.get('target',{}).get('native_graph'):
            proof=self.call('download',expected_before=before)
            verify_native_handoff(proof,self.request,before)
            if any(proof['observed'].get(key)!=after.get(key) for key in ('package','process','related')):
                raise ValueError('Native successor changed across handoff readback')
        return (Transaction.package_matches(args, after) and after['process'] is not None and before['process'] is not None
            and after['process']['birth'] > before['process']['birth'] and after['process']['pid'] != before['process']['pid']
            and after['package']['family'] == before['package']['family'])


def validate_native_origin(request):
    value=request.get('native_origin')
    if value is None:return None
    if request.get('purpose')!=NATIVE_QUALIFICATION:raise ValueError('Native origin requires WD1 purpose')
    from urllib.parse import urlsplit
    if not isinstance(value,str):raise ValueError('Invalid native engineering origin')
    try:
        parsed=urlsplit(value)
        port=parsed.port
        if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or not port or value!=f'http://127.0.0.1:{port}':
            raise ValueError('Invalid native engineering origin')
    except (TypeError,ValueError) as error:raise ValueError('Invalid native engineering origin') from error
    return value


def load_native_side(request, side):
    if side not in ('baseline','target'):raise ValueError('Unknown native artifact side')
    # The staging tree preserves these existing owners' paths; no alternate validator.
    root=Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from scripts.releases.semantier_windows import validate_native_graph
    target=request[side];binding=target['native_graph'];file=Path(binding['path'])
    if file.name!='native-graph.json' or file.parent!=Path(target['path']).parent:
        raise ValueError('Native graph must remain beside the pinned root')
    with safe_file(file,'r') as stream:
        raw=stream.read()
    if len(raw)>4*1024*1024 or hashlib.sha256(raw.encode('utf-8')).hexdigest()!=binding['sha256']:
        raise ValueError('Native graph source digest mismatch')
    graph=json.loads(raw.lstrip('\ufeff'))
    validate_native_graph(graph,build_id=target['build_id'],identity='Semantier.Agent.Acceptance',
        publisher='CN=Semantier Technology, O=Semantier Technology',version=target['version'],arch='x64')
    row=next(row for row in graph['objects'] if row['key']==graph['root'])
    if row['sha256']!=target['sha256'] or row['key'].rsplit('/',1)[1]!=Path(target['path']).name:
        raise ValueError('Native graph root differs from pinned artifact')
    return graph


def load_native_target(request):
    if request.get('purpose')==NATIVE_QUALIFICATION:validate_native_origin(request)
    return load_native_side(request,'target')


def validate_native_update_inputs(request):
    if request.get('purpose')==NATIVE_QUALIFICATION:raise ValueError('WD1 local qualification does not authorize production update')
    import re
    from hermes_cli.release_channels import public_base
    value=request.get('selected_delivery_base')
    if not isinstance(value,str) or public_base(value)!=value:
        raise ValueError('Pinned selected delivery base required')
    if any(re.fullmatch('[a-f0-9]{64}',request.get(key,'')) is None for key in ('native_handoff_source_sha256','native_digest_source_sha256')):
        raise ValueError('Qualified baseline native helper hashes required')


def native_handoff_inputs(request,before):
    validate_native_update_inputs(request)
    target=request['target']
    return {'authorizedSid':request['authorized_sid'],'session':before['session'],
                  'manifestSha256':target['manifest_sha256'],'graphSha256':target['native_graph']['sha256'],
                  'sourceSha256':request['native_handoff_source_sha256'],'digestSourceSha256':request['native_digest_source_sha256'],'base':request['selected_delivery_base'],
                  'targetBuildId':target['build_id'],'targetCommit':target['commit'],'version':target['version'],
                  'currentVersion':before['package']['version'],'fullName':before['package']['full'],
                  'familyName':before['package']['family'],'graph':request['native_graph_value']}


def verify_native_handoff(proof, request, before):
    """Receipt/OS-incarnation binding only; never differential/cache acceptance."""
    validate_native_update_inputs(request)
    try:
        intent=proof['intent'];completion=proof['completion']
        expected=native_handoff_inputs(request,before)
        if proof.get('failure') or proof['transport']!='windows-package-uri' or not Transaction.matches(expected,intent):
            raise ValueError('Native handoff input binding mismatch')
        verify_native_successor(proof,request,before)
        times=[datetime.fromisoformat(value) for value in
               (proof['deployment']['producedAt'],proof['relaunch_intent']['producedAt'],completion['birth'],completion['producedAt'])]
        if any(value.tzinfo is None or value.utcoffset().total_seconds()!=0 for value in times) or times!=sorted(times):
            raise ValueError('Native handoff UTC chronology mismatch')
        return proof
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('Native handoff evidence incomplete') from error


def verify_native_reconciliation(proof,request,before,now,target):
    from uuid import UUID
    import re
    try:
        if (proof['transport']!='windows-package-uri' or proof['waiterAlive'] is not False
            or not Transaction.matches(native_handoff_inputs(request,before),proof['intent'])
            or proof['observed']!=now or now['process'] is not None
            or now['sid']!=request['authorized_sid'] or now['session']!=before['session']):
            raise ValueError('Native reconciliation binding/live waiter mismatch')
        deployment=proof.get('deployment',{});failure=proof.get('failure',{})
        if (deployment and failure) or ((deployment or failure) and proof.get('deploymentIntentPresent') is not True):
            raise ValueError('Contradictory native deployment history')
        if (deployment.get('status')=='OS_SUCCESS' and deployment.get('manifestSha256')==request['target']['manifest_sha256']
            and UUID(deployment['activityId']).int and Transaction.package_matches(target,now)):
            return 'TARGET_GRAPH_REGISTERED'
        intact=now['package']==before['package'] and Transaction.related(now)==Transaction.related(before)
        if intact and proof.get('deploymentIntentPresent') is False and not deployment and not proof.get('failure'):
            return 'NOT_STARTED'
        failure=proof.get('failure',{})
        if (intact and failure.get('status')=='FAILURE' and failure.get('operationCompleted') is True
            and failure.get('manifestSha256')==request['target']['manifest_sha256']
            and re.fullmatch('[A-F0-9]{8}',failure.get('hresult','')) and failure['hresult']!='00000000'):
            return 'BASELINE_INTACT_FAILED'
        raise ValueError('Uncertain native OS state; preserve resources and evidence')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('Native reconciliation evidence incomplete') from error


def verify_native_successor(proof, request, before):
    from uuid import UUID
    target=request['target'];completion=proof['completion'];observed=proof['observed']
    if (proof['deployment']['status']!='OS_SUCCESS' or completion['status']!='TARGET_GRAPH_REGISTERED_PROCESS_LAUNCHED'
            or any(proof[part]['manifestSha256']!=target['manifest_sha256'] for part in ('deployment','relaunch_intent','completion'))
            or UUID(proof['deployment']['activityId']).int==0):
        raise ValueError('Native deployment/relaunch result unproved')
    package={'full':completion['fullName'],'family':before['package']['family'],'commit':target['commit'],
             'build_id':target['build_id'],'version':target['version']}
    process={'pid':completion['pid'],'birth':completion['birth'],'exe':completion['exe'],'exe_id':completion['exeId']}
    if (not Transaction.matches(package,observed['package']) or not Transaction.matches(process,observed['process'])
            or completion['sid']!=request['authorized_sid'] or completion['session']!=before['session']
            or observed['sid']!=request['authorized_sid'] or observed['session']!=before['session']
            or completion['pid']==before['process']['pid'] or completion['birth']<=before['process']['birth']):
        raise ValueError('Native automatically launched incarnation mismatch')


def run_upgrade(tx, platform):
    tx.reconcile()
    if not any(row['kind']=='baseline_ready' for row in tx.events()): raise ValueError('Baseline readiness evidence required before publication/update')
    before = platform.call('conservation')
    tx.record('conservation_before', facts=before)
    semantic_before=platform.capture_semantics() if hasattr(platform,'capture_semantics') else None
    if semantic_before is not None:tx.record('semantic_before',facts=semantic_before)
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
    if semantic_before is not None:
        try:semantic_after=platform.verify_semantics(semantic_before)
        except Exception:
            tx.record('conservation_failed',scope='SQLite semantic content; preserve before/after backups')
            raise
        tx.record('semantic_preservation',facts=semantic_after)
    tx.record('acceptance', status='registered_readback_and_selected_sentinels_passed',
              native_acceptance='requires complete original matrices', facts=after)


def run_native_qualification(tx, platform, owner, *, require_existing=False):
    if platform.request.get('purpose')!=NATIVE_QUALIFICATION:raise ValueError('Native qualification purpose required')
    if require_existing and (owner['package'] is None or not tx.events()):raise ValueError('Existing native registration and transaction required for resume')
    if not tx.events():
        if owner['package'] is not None or owner['process'] is not None or Transaction.related(owner):
            raise ValueError('Existing package/process/related set: refuse native qualification')
        tx.initialize({**owner,'purpose':NATIVE_QUALIFICATION,'native_origin':platform.request.get('native_origin')})
    if tx.events()[0]['owner'].get('purpose')!=NATIVE_QUALIFICATION:raise ValueError('Existing transaction has another purpose')
    tx.reconcile()
    if platform.observe()['package'] is None:
        if require_existing:raise ValueError('Existing native registration disappeared; no deployment during resume')
        tx.operation('register',platform.call('artifact',side='target'))
    tx.assert_owned(platform.observe())
    state=platform.observe();binding=platform.proof_binding();bindings=platform.proof_bindings(binding)
    # Reuse only a completed proof with unchanged source/input and OS incarnation.
    prior=[row for row in tx.events() if row['kind']=='native_qualification' and row.get('binding') in bindings
           and all(row['observed'].get(key)==state.get(key) for key in ('sid','session_sid','session','root_id','package','related'))]
    if prior:
        proof=prior[-1]['facts']
    else:
        proof=platform.call('native-proof')
        tx.assert_owned(platform.observe())
        tx.record('native_qualification',binding=binding,facts=proof,observed=platform.observe(),scope='local layout/runtime only; not CI or migration')
    if platform.observe()['process'] is None:tx.operation('launch',{})
    ready=platform.call('ready')
    tx.record('native_qualification_ready',facts=ready,observed=platform.observe(),proof_binding=prior[-1]['binding'] if prior else binding,execution_binding=binding,proof_reuse='PARTIAL_REUSE' if prior and prior[-1]['binding']!=binding else 'EXACT_REUSE' if prior else 'NEW',scope='local layout/runtime only; not CI or migration')
    print(json.dumps({'scope':'LOCAL_NATIVE_LAYOUT_QUALIFICATION_NOT_CI_OR_MIGRATION','proof':proof,'ready':ready}))


def run_baseline(tx, platform, owner):
    if not tx.events():
        if owner['package'] is not None or owner['process'] is not None or Transaction.related(owner): raise ValueError('Existing package/process/related set: refuse new transaction')
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
    request['owned_gateways']=[r['facts']['gateway'] for r in rows if r['kind'] in ('baseline_ready','successor_ready','native_qualification_ready') and r['facts'].get('gateway') is not None]
    current=platform.observe()
    if current['package'] is None and Transaction.related(current): raise ValueError('Related packages remain without owned main; preserve for reconciliation')
    if current['package'] is not None: tx.operation('remove',{})
    tx.record('cleanup',status='owned_package_removed_data_and_evidence_retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','baseline','native-qualify','native-resume','upgrade','reconcile','cleanup'])
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--proof-source',type=Path)
    p.add_argument('--proof-source-sha256')
    a=p.parse_args()
    if sys.platform != 'win32': raise SystemExit('Use native Windows Python in the authorized disposable session')
    if bool(a.proof_source)!=bool(a.proof_source_sha256) or (a.proof_source and a.mode!='native-resume'):
        raise ValueError('Pinned prior proof source is only valid for native-resume')
    request=json.loads(a.request.read_text(encoding='utf-8-sig')); platform=Windows(request,(a.proof_source,a.proof_source_sha256) if a.proof_source else None)
    owner=platform.observe()  # OS refusal precedes journal/root writes.
    if request.get('target',{}).get('native_graph') is not None:request['native_graph_value']=load_native_target(request)
    if request.get('baseline',{}).get('native_graph') is not None:request['native_baseline_graph_value']=load_native_side(request,'baseline')
    if a.mode=='baseline' and request.get('baseline',{}).get('native_graph') is not None:
        from hermes_cli.release_channels import public_base
        base=request.get('baseline_delivery_base')
        if not isinstance(base,str) or public_base(base)!=base:raise ValueError('Pinned native baseline delivery base required')
    if a.mode=='preflight':
        if owner['package'] is not None or Transaction.related(owner): raise ValueError('Conflicting baseline registration or related package set')
        native=request.get('purpose')==NATIVE_QUALIFICATION
        side='target' if native else 'baseline'
        artifact=platform.call('artifact',side=side)
        print(json.dumps({'scope':'READ_ONLY_NATIVE_LAYOUT' if native else 'READ_ONLY','observed':owner,side:artifact})); return
    if a.mode in ('native-qualify','native-resume') and 'native_graph_value' not in request:raise ValueError('Native graph qualification inputs required')
    root=Path(request['root']); journal=root/'registered-update.jsonl'
    with held_windows_root(root, owner['root_id']), exclusive(root/'registered-update.lock'):
        tx=Transaction(journal,platform)
        handlers = {
            'baseline': lambda: run_baseline(tx, platform, owner),
            'native-qualify': lambda: run_native_qualification(tx,platform,owner),
            'native-resume': lambda: run_native_qualification(tx,platform,owner,require_existing=True),
            'reconcile': lambda: print(tx.reconcile()),
            'upgrade': lambda: run_upgrade(tx, platform),
            'cleanup': lambda: run_cleanup(tx, platform, request),
        }
        handlers[a.mode]()

if __name__=='__main__': main()
