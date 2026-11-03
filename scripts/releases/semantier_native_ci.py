"""WSL bridge to the existing real native installation smoke, owned family only."""
import argparse,base64,hashlib,hmac,json,os,re,signal,subprocess,sys,secrets,stat
from contextlib import contextmanager, nullcontext
from pathlib import Path
import select
import time
from hermes_cli.product_composition import verify_product_modules

ROOT=Path(__file__).resolve().parents[2]
CI_INCOMING_ROOT=Path('/home/chris/repo/semantier-runtime/.worktrees/cloudflare-update2-three-sources-v2/.artifacts/update2-publication-v2/qualified-v3/incoming')
CI_RUNTIME_CONTAINER='semantier-update2-publication-native-d866-backend'
CI_RUNTIME_IMAGE='sha256:bcbe111c2a37cb6de130aee388d1f18dfd3b644bfd349e0774591c3354838ee0'
CI_RUNTIME_PARENT='d866bfe2d8ad2470e237eb116e42d3dee6515eb1'
CI_RUNTIME_NATIVE='21ab5de240ee9ed8f5b3fe47506b2ba72502c842'
CI_RUNTIME_INVENTORY='b3f240e68410e9129d5b022d7fdb74e01fef98a05eca106849adb549c7d43af0'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def remove_receipt_key(env):
    for name in tuple(env):
        if name.upper() == 'SEMANTIER_CLIENT_RECEIPT_KEY':
            env.pop(name)
    if 'WSLENV' in env:
        env['WSLENV'] = ':'.join(item for item in env['WSLENV'].split(':')
                                 if item and item.split('/')[0].upper() != 'SEMANTIER_CLIENT_RECEIPT_KEY')
    return env


@contextmanager
def hold_ci_run_custody(lock_fd=None, expected_sid=None):
    """Hold the Windows CI custody directory and its parents while inspecting it."""
    root = subprocess.check_output(['wslpath', '-w', '/mnt/d/repo/qualification-private'], text=True).strip()
    if root.lower() != r'd:\repo\qualification-private':
        raise ValueError('CI custody Windows mapping mismatch')
    script = (ROOT / 'tests/install/e2e-assets/ci-custody-hold.ps1').read_bytes()
    payload = base64.b64encode(script).decode('ascii')
    guard = ci_session_guard(expected_sid) if expected_sid else ''
    command = "$ProgressPreference='SilentlyContinue';" + guard + "& ([scriptblock]::Create([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + payload + "')))) -Root 'D:\\repo\\qualification-private'"
    proc = subprocess.Popen(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
        '-NoProfile', '-NonInteractive', '-EncodedCommand', base64.b64encode(command.encode('utf-16le')).decode('ascii')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=native_environment(),
        pass_fds=() if lock_fd is None else (lock_fd,))
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 20)
        if not ready or proc.stdout.readline().strip() != 'CI_CUSTODY_READY' or proc.poll() is not None:
            raise ValueError('Windows CI custody handle unavailable')
        yield proc
        if proc.poll() is not None:
            raise ValueError('Windows CI custody handle terminated')
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)


def run_ci_custody_command(process, command, timeout=2700):
    """Send fixed native-dispatch data to the Windows custody process."""
    if not isinstance(command,dict) or command.get('operation')!='native-smoke':
        raise ValueError('Fixed CI native dispatch required')
    nonce=secrets.token_hex(16)
    payload=base64.b64encode(json.dumps(command,separators=(',',':')).encode('utf-8')).decode('ascii')
    process.stdin.write(nonce+':'+payload+'\n')
    process.stdin.flush()
    deadline=time.monotonic()+timeout
    pending=b''
    marker=('CI_CUSTODY_DONE:'+nonce+':').encode('ascii')
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise ValueError('Windows CI custody process exited during native phase')
        ready, _, _=select.select([process.stdout],[],[],min(1,max(0,deadline-time.monotonic())))
        if not ready:continue
        data=os.read(process.stdout.fileno(),65536)
        if not data:raise ValueError('Windows CI custody output closed during native phase')
        pending+=data
        if len(pending)>1048576:raise ValueError('Windows CI custody output exceeded bound')
        while b'\n' in pending:
            line,pending=pending.split(b'\n',1)
            line=line.rstrip(b'\r')
            if line.startswith(marker):
                if line[len(marker):]!=b'OK':raise ValueError('Windows CI native phase failed; preserve operation')
                return
            print(line.decode('utf-8',errors='replace'),flush=True)
    raise TimeoutError('Windows CI native phase unresolved; preserve operation')


def run_owned_ci_child(argv, env, lock_fd, custody_process):
    """Stop the direct child on lost custody; its native descendants remain unresolved."""
    proc = subprocess.Popen(argv, env=env, pass_fds=(lock_fd,))
    try:
        while True:
            if custody_process is not None and custody_process.poll() is not None:
                raise ValueError('Windows CI custody handle terminated')
            try:
                code = proc.wait(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if custody_process is not None and custody_process.poll() is not None:
            raise ValueError('Windows CI custody handle terminated')
        if code:
            raise subprocess.CalledProcessError(code, argv)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


def orchestrate_ci(state_root, phases, reconcile=None, run_intent=None, mutation_guard=None, custody=None, stage_preflight=None):
    """Hold the existing OS lock across every child; retain uncertain work on death."""
    from tests.install.semantier_registered_update import safe_file, _locked
    if not phases or phases[0][0] != 'preflight' or reconcile is None:
        raise ValueError('CI preflight and reconciliation required')
    state_root = Path(state_root)
    marker = state_root / '.ci-execution-pending.json'
    marker_identity = None
    with safe_file(state_root / 'ci-execution.lock', 'a') as lock, _locked(lock), (custody(lock.fileno()) if custody else nullcontext()) as custody_process:
        pending = marker.exists() or marker.is_symlink()
        if reconcile() is not True:
            raise ValueError('CI reconciliation unresolved')
        # Reconciliation can itself discover or create a hold. Never enter a
        # phase based only on the state captured before that discovery, and do
        # not treat disappearance of a previously observed marker as proof.
        if pending or marker.exists() or marker.is_symlink():
            raise ValueError(f'CI execution requires reconciliation: {marker}')
        for name, argv in phases:
            if custody_process is not None and custody_process.poll() is not None:
                raise ValueError('Windows CI custody handle terminated')
            # A create-new durable intent survives parent death and never claims
            # that Windows children stopped merely because this lock was released.
            if name != 'preflight' and marker_identity is None:
                if mutation_guard is not None:
                    mutation_guard()
                if run_intent is not None:
                    intent = Path(run_intent)
                    if intent.parent.is_symlink() or not intent.parent.is_dir():
                        raise ValueError('CI run intent custody unavailable')
                    directory = os.open(intent.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        opened, named = os.fstat(directory), intent.parent.lstat()
                        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                            raise ValueError('CI run intent custody identity changed')
                        descriptor = os.open(intent.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                            json.dump({'pid': os.getpid(), 'sourceCommit': os.environ.get('GITHUB_SHA'),
                                       'runId': os.environ.get('GITHUB_RUN_ID'), 'attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
                                       'firstPhase': name}, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.fsync(directory)
                        named = intent.parent.lstat()
                        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                            raise ValueError('CI run intent custody identity changed')
                    finally:
                        os.close(directory)
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump({'phase': name, 'pid': os.getpid()}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                    info = os.fstat(stream.fileno())
                    marker_identity = (info.st_dev, info.st_ino)
                directory = os.open(state_root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            elif name != 'preflight':
                current = marker.lstat()
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or (current.st_dev, current.st_ino) != marker_identity:
                    raise ValueError('CI execution marker identity changed; preserve for reconciliation')
            if name == 'stage' and stage_preflight is not None:
                stage_preflight()
                if custody_process is not None and custody_process.poll() is not None:
                    raise ValueError('Windows CI custody handle terminated')
                current=marker.lstat()
                if not stat.S_ISREG(current.st_mode) or current.st_nlink!=1 or (current.st_dev,current.st_ino)!=marker_identity:
                    raise ValueError('CI execution marker identity changed; preserve for reconciliation')
            child_env = os.environ.copy()
            if name in ('build', 'native'):
                remove_receipt_key(child_env)
            if callable(argv):
                if name != 'native':
                    raise ValueError('Unexpected internal CI phase')
                key = os.environ.pop('SEMANTIER_CLIENT_RECEIPT_KEY', None)
                try:
                    if custody_process is None:
                        argv()
                    else:
                        argv(custody_process, lock.fileno())
                finally:
                    if key is not None:
                        os.environ['SEMANTIER_CLIENT_RECEIPT_KEY'] = key
            elif name in ('build', 'native', 'stage'):
                # Keep the same ownership interval if the coordinator dies
                # while a protected child is still executing.
                run_owned_ci_child(argv, child_env, lock.fileno(), custody_process)
            else:
                subprocess.run(argv, check=True, env=child_env)
        if marker_identity is not None:
            # Never unlink a pathname after checking it: another writer could
            # replace that entry between lstat and unlink. Append a terminal
            # observation through the original inode and retain the journal.
            # Until reconciliation can validate retained records, it remains a
            # conservative hold on the next invocation.
            fd = os.open(marker, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                current = os.fstat(fd)
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or (current.st_dev, current.st_ino) != marker_identity:
                    raise ValueError('CI execution marker identity changed; preserve for reconciliation')
                os.write(fd, b'\n{"completed":true}\n')
                os.fsync(fd)
                named = marker.lstat()
                if (named.st_dev, named.st_ino) != marker_identity or not stat.S_ISREG(named.st_mode):
                    raise ValueError('CI execution marker identity changed; preserve for reconciliation')
            finally:
                os.close(fd)

def native_environment():
    """Forward existing proxies and genuine CI flag across WSL without host changes."""
    env=remove_receipt_key(os.environ.copy())
    names=('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','GITHUB_ACTIONS')
    forwarded=[item for item in env.get('WSLENV','').split(':') if item and item.split('/')[0].upper() not in (*names, 'SEMANTIER_CLIENT_RECEIPT_KEY')]
    for name in names:
        value=env.get(name) or env.get(name.lower())
        env.pop(name.lower(),None)
        if value:
            env[name]=value
            forwarded.append(name)
    env['WSLENV']=':'.join(forwarded)
    return env


def select_native_candidate(output,commit,controller=None):
    """Select exact built inputs; signature/installed qualification is native-side."""
    from scripts.releases.semantier_windows import validate_native_graph
    output=Path(output)
    read=lambda name:json.loads((output/name).read_text(encoding='utf-8-sig'))
    signed=read('signed/receipt.json');request=read('channel-request.json');result=read('result.json')
    if signed.get('sourceCommit')!=commit or request.get('commit')!=commit or result.get('sourceCommit')!=commit or result.get('result')!='PASS':
        raise ValueError('Completed exact source required for native smoke')
    if controller is not None and (not re.fullmatch('[0-9a-f]{40}',controller) or request.get('controllerCommit')!=controller or result.get('controllerCommit')!=controller):
        raise ValueError('Completed exact controller required for native smoke')
    directory=output/'signed/distribution'
    files=[p for p in directory.iterdir() if p.suffix.lower() in ('.msix','.msixbundle')]
    graph=signed.get('nativeGraph')
    if graph is None:
        if len(files)!=1 or files[0].suffix!='.msix' or files[0].is_symlink():raise ValueError('One signed legacy candidate required')
        return dict(artifact=files[0],graph=None,plan=None)
    validate_native_graph(graph,build_id=request['buildId'],identity=request['identity']['msixAppIdWithOrg'],publisher=signed['publisher'],version=request['windowsVersion'],arch='x64')
    expected={row['key'].rsplit('/',1)[1] for row in graph['objects']}
    if {p.name for p in files}!=expected or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('Native graph candidate must be complete')
    # Selection must bind every transferred package, not merely filenames and
    # the bundle's receipt. Native smoke still owns signature and OS checks.
    for row in graph['objects']:
        path=directory/row['key'].rsplit('/',1)[1]
        descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        with os.fdopen(descriptor,'rb') as stream:
            before=os.fstat(stream.fileno())
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
            after=os.fstat(stream.fileno())
        named=path.lstat()
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if (not stat.S_ISREG(named.st_mode) or identity(before)!=identity(after) or identity(after)!=identity(named)
                or after.st_size!=row['size'] or digest!=row['sha256']):
            raise ValueError('Native graph object bytes changed or mismatched')
    root=next(row for row in graph['objects'] if row['key']==graph['root'])
    if root['sha256']!=signed['signedSha256']:raise ValueError('Native root digest differs from signed receipt')
    plan=output/'signed/native-layout/layout-plan.json';graph_file=directory/'native-graph.json'
    if not plan.is_file() or plan.is_symlink() or not graph_file.is_file() or graph_file.is_symlink():
        raise ValueError('Native graph and complete layout inputs required')
    if json.loads(graph_file.read_text(encoding='utf-8-sig'))!=graph:raise ValueError('Signed graph file differs from receipt')
    return dict(artifact=directory/root['key'].rsplit('/',1)[1],graph=graph_file,plan=plan)


def ci_session_guard(expected_sid):
    return r"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SemantierCIWts {
 [DllImport("wtsapi32.dll",SetLastError=true,CharSet=CharSet.Unicode)]
 static extern bool WTSQuerySessionInformation(IntPtr server,int session,int kind,out IntPtr value,out int bytes);
 [DllImport("wtsapi32.dll")] static extern void WTSFreeMemory(IntPtr value);
 public static string Text(int session,int kind) {
  IntPtr value;int bytes;
  if(!WTSQuerySessionInformation(IntPtr.Zero,session,kind,out value,out bytes))throw new InvalidOperationException("WTS identity unavailable");
  try {if(bytes<2)throw new InvalidOperationException("WTS identity empty");return Marshal.PtrToStringUni(value);}finally{WTSFreeMemory(value);}
 }
 public static int State(int session) {
  IntPtr value;int bytes;
  if(!WTSQuerySessionInformation(IntPtr.Zero,session,8,out value,out bytes))throw new InvalidOperationException("WTS state unavailable");
  try {if(bytes<4)throw new InvalidOperationException("WTS state empty");return Marshal.ReadInt32(value);}finally{WTSFreeMemory(value);}
 }
}
'@
function Assert-CISession {
 $actualSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
 $session=(Get-Process -Id $PID).SessionId
 $user=[SemantierCIWts]::Text($session,5);$domain=[SemantierCIWts]::Text($session,7)
 $sessionSid=([Security.Principal.NTAccount]::new($domain,$user)).Translate([Security.Principal.SecurityIdentifier]).Value
 if($actualSid -cne __EXPECTED_SID__ -or $sessionSid -cne $actualSid -or -not [Environment]::UserInteractive -or $session -le 0 -or [SemantierCIWts]::State($session) -ne 0){throw 'Separate real active CI account required'}
 return @{producedAt=[DateTime]::UtcNow.ToString('o');sid=$actualSid;session_sid=$sessionSid;session=$session;wts_state=[SemantierCIWts]::State($session);wts_domain=$domain;wts_user=$user;interactive=[Environment]::UserInteractive;os=[Environment]::OSVersion.Version.ToString()}
}
$ciContextBefore=Assert-CISession;
""".replace('__EXPECTED_SID__',"'"+str(expected_sid).replace("'","''")+"'")


def validate_ci_preflight(source, controller, expected_controller, expected_sid, receipt_key):
    if os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('GITHUB_REPOSITORY') != 'chris-han/hermes-agent':
        raise ValueError('Real CI execution context required')
    if source != os.environ.get('GITHUB_SHA') or not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Exact source checkout required')
    if controller != expected_controller or not re.fullmatch(r'[0-9a-f]{40}', controller):
        raise ValueError('Exact controller checkout required')
    if not expected_sid.startswith('S-1-5-21-') or expected_sid in ('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011'):
        raise ValueError('Separate bound CI account required')
    if len(receipt_key.encode('utf-8')) < 32:
        raise ValueError('CI receipt key unavailable')


def custody_probe(incoming, runtime_read):
    """Create, sync, compare from Runtime, then remove only the exact owned probe."""
    root = Path(incoming)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Incoming root unavailable')
    name = '.ci-custody-' + secrets.token_hex(16)
    content = secrets.token_bytes(64)
    dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    filefd = None
    inode = None
    try:
        filefd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dirfd)
        inode = os.fstat(filefd)
        if os.write(filefd, content) != len(content):
            raise OSError('Incomplete custody probe write')
        os.fsync(filefd)
        os.fsync(dirfd)
        os.lseek(filefd, 0, os.SEEK_SET)
        if hashlib.sha256(os.read(filefd, 65)).digest() != hashlib.sha256(content).digest():
            raise ValueError('Runner readback mismatch')
        if runtime_read(name) != hashlib.sha256(content).hexdigest():
            raise ValueError('Runtime readback mismatch')
    finally:
        if filefd is not None:
            os.close(filefd)
            try:
                current = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                if inode is not None and stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (inode.st_dev, inode.st_ino):
                    os.unlink(name, dir_fd=dirfd)
                    os.fsync(dirfd)
            except FileNotFoundError:
                pass
        os.close(dirfd)


def verify_receipt_key(key, challenge, runtime_mac):
    expected = hmac.new(key.encode('utf-8'), challenge, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, runtime_mac(challenge)):
        raise ValueError('CI and Runtime receipt key mismatch')


def validate_primary_base(candidate_base, runtime_base):
    if candidate_base != 'https://updates.semantier.com' or runtime_base != candidate_base:
        raise ValueError('CI and Runtime publication base mismatch')


def verify_cloudflare_custody(cid, primary_root):
    """Check the actual Runtime target and read back a CI-owned probe there."""
    configuration = subprocess.check_output(['docker', 'exec', cid, 'python', '-c',
        'import json,os; from agents.client_operations.runtime import ClientOperations; '
        'c=ClientOperations(os.environ); p=c.targets.owner("channel2"); '
        'print(json.dumps({"base":p.base,"incoming":str(p.incoming)}))'], text=True)
    target = json.loads(configuration)
    if target != {'base': 'https://update2.semantier.com', 'incoming': '/fixture/qualified-v3/incoming_r2'}:
        raise ValueError('Cloudflare Runtime target configuration mismatch')
    target_root = Path(primary_root).parent / 'incoming_r2'
    if target_root.is_symlink() or not target_root.is_dir() or os.path.samefile(primary_root, target_root):
        raise ValueError('Cloudflare incoming custody unavailable')
    def runtime_read(name):
        return subprocess.check_output(['docker', 'exec', cid, 'python', '-c',
            'import sys; from pathlib import Path; from agents.client_operations.artifact_io import safe_file,file_facts; '
            'p=safe_file(Path("/fixture/qualified-v3/incoming_r2"),sys.argv[1]); '
            'size,digest=file_facts(p); assert size==64; print(digest)', name], text=True).strip()
    custody_probe(target_root, runtime_read)


def verify_runtime_identity(cid):
    """Bind the live container to the reviewed image and its source inventory."""
    labels=json.loads(subprocess.check_output(['docker','inspect',cid,'--format','{{json .Config.Labels}}'],text=True))
    if (not isinstance(labels,dict) or labels.get('io.semantier.source.parent')!=CI_RUNTIME_PARENT or
            labels.get('io.semantier.source.native')!=CI_RUNTIME_NATIVE):
        raise ValueError('Runtime source labels mismatch')
    subprocess.run(['docker','exec',cid,'python','-c',
        'import pathlib,json,hashlib\nraw=pathlib.Path("/app/update2-source-inventory.json").read_bytes()\n'
        'if hashlib.sha256(raw).hexdigest()!='+repr(CI_RUNTIME_INVENTORY)+': raise SystemExit("Runtime inventory digest mismatch")\n'
        'm=json.loads(raw)\n'
        'if m.get("parent_sha")!='+repr(CI_RUNTIME_PARENT)+' or m.get("native_sha")!='+repr(CI_RUNTIME_NATIVE)+': raise SystemExit("Runtime inventory source mismatch")\n'
        'if len(m.get("inputs",[]))!=22721: raise SystemExit("Runtime inventory size mismatch")\n'
        'if not all(hashlib.sha256(pathlib.Path(i["image_path"]).read_bytes()).hexdigest()==i["sha256"] for i in m["inputs"]): raise SystemExit("Runtime source file mismatch")'],check=True)


def preflight(expected_sid, incoming, controller, expected_controller, public_base):
    # The later staging child uses this same interpreter.  Refuse before any
    # Windows/build work if its installed product helpers are absent or stale.
    verify_product_modules(ROOT / 'semantier-product',
                           ('__init__', 'artifact_io', 'chunks', 'client_operations', 'package_graph'))
    checkout = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    actual_controller = subprocess.check_output(['/usr/bin/git', '-C', str(controller), 'rev-parse', 'HEAD'], text=True).strip()
    for tree in (ROOT, controller):
        if subprocess.check_output(['/usr/bin/git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
            raise ValueError('CI checkout is not clean')
    validate_ci_preflight(checkout, actual_controller, expected_controller, expected_sid, os.environ.get('SEMANTIER_CLIENT_RECEIPT_KEY', ''))
    if not os.environ.get('ACCEPTANCE_WINDOWS_PYTHON'):
        raise ValueError('Windows Python unavailable')
    # Verify an ephemeral challenge in memory; never export the key or sign a package.
    cmd = "$ErrorActionPreference='Stop';" + ci_session_guard(expected_sid) + r"""
$c=Get-Item -LiteralPath 'Cert:\CurrentUser\My\57BF15739B4AA447842EAF388856A01B4E5029FF'
$now=Get-Date
$basic=@($c.Extensions|Where-Object {$_.Oid.Value -eq '2.5.29.19'})
if(!$c.HasPrivateKey -or $c.NotBefore -gt $now -or $c.NotAfter -le $now -or $c.EnhancedKeyUsageList.ObjectId -notcontains '1.3.6.1.5.5.7.3.3' -or $basic.Count -ne 1 -or $basic[0].CertificateAuthority){throw 'CI signer unavailable'}
$key=[System.Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($c)
if(!$key){throw 'CI signer private-key handle unavailable'}
try {
 $challenge=[byte[]]::new(32);$rng=[System.Security.Cryptography.RandomNumberGenerator]::Create()
 try {$rng.GetBytes($challenge)} finally {$rng.Dispose()}
 $signature=$key.SignData($challenge,[System.Security.Cryptography.HashAlgorithmName]::SHA256,[System.Security.Cryptography.RSASignaturePadding]::Pkcs1)
 if(!$key.VerifyData($challenge,$signature,[System.Security.Cryptography.HashAlgorithmName]::SHA256,[System.Security.Cryptography.RSASignaturePadding]::Pkcs1)){throw 'CI signer challenge verification failed'}
} finally {$key.Dispose()}
# Match the native smoke's refusal before the allocator or build runs.
if(@(Get-AppxPackage -Name 'Semantier.Agent.Acceptance').Count -or
   @(Get-AppxPackage -PackageTypeFilter Resource -Name 'Semantier.Agent.Acceptance').Count -or
   @(Get-AppxPackage -PackageTypeFilter Optional -Name 'Semantier.Agent.Acceptance.*').Count){throw 'CI Acceptance package or related resources already registered'}
if(!(Test-Path -LiteralPath $env:ACCEPTANCE_WINDOWS_PYTHON -PathType Leaf)){throw 'Windows Python unavailable'}
if(!(Test-Path -LiteralPath 'D:\repo\hermes-build\.cache\desktop-inputs\tools\node-26.7.0-win32-x64\node.exe' -PathType Leaf)){throw 'Windows Node unavailable'}
$sdk=Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
$selectedSignTool=@(Get-ChildItem -Path (Join-Path $sdk '*\x64\signtool.exe') -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object FullName | Select-Object -First 1)
if($selectedSignTool.Count -ne 1){throw 'SignTool unavailable'}
if(!(Test-Path -LiteralPath (Join-Path (Split-Path -Parent $selectedSignTool[0]) 'makeappx.exe') -PathType Leaf)){throw 'MakeAppx unavailable beside selected SignTool'}
Assert-CISession|Out-Null
"""
    env = native_environment()
    env['WSLENV'] += ':ACCEPTANCE_WINDOWS_PYTHON'
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(cmd.encode('utf-16le')).decode()], check=True, env=env)
    expected_root = CI_INCOMING_ROOT
    if Path(incoming).resolve(strict=True) != expected_root.resolve(strict=True):
        raise ValueError('Unexpected incoming custody root')
    container = CI_RUNTIME_CONTAINER
    inspect = json.loads(subprocess.check_output(['docker','inspect',container,'--format','{{json .Mounts}}'], text=True))
    expected_mount = str(expected_root.parents[1])
    if not any(m.get('Source') == expected_mount and m.get('Destination') == '/fixture' for m in inspect):
        raise ValueError('Runtime custody mount mismatch')
    if any(m.get('Destination') == '/app' or str(m.get('Destination','')).startswith('/app/') for m in inspect):
        raise ValueError('Runtime application source mount is not permitted')
    details = subprocess.check_output(['docker','inspect',container,'--format','{{.Id}} {{.Image}} {{.State.Running}}'], text=True).strip().split()
    if len(details) != 3 or details[1] != CI_RUNTIME_IMAGE or details[2] != 'true':
        raise ValueError('Runtime identity or state mismatch')
    cid = details[0]
    if not cid or any(c not in '0123456789abcdef' for c in cid):
        raise ValueError('Runtime container identity unavailable')
    verify_runtime_identity(cid)
    def runtime_read(name):
        result = subprocess.check_output(['docker','exec',cid,'python','-c',
            'import sys; from pathlib import Path; from agents.client_operations.artifact_io import safe_file,file_facts; p=safe_file(Path("/fixture/qualified-v3/incoming"),sys.argv[1]); size,digest=file_facts(p); assert size==64; print(digest)', name], text=True)
        return result.strip()
    runtime_base = subprocess.check_output(['docker','exec',cid,'python','-c',
        'import os; print(os.environ["SEMANTIER_CLIENT_PUBLIC_BASE"])'], text=True).strip()
    validate_primary_base(public_base, runtime_base)
    def runtime_mac(challenge):
        return subprocess.check_output(['docker','exec',cid,'python','-c',
            'import os,sys,hmac,hashlib; key=os.environ["SEMANTIER_CLIENT_RECEIPT_KEY"].encode(); print(hmac.new(key,bytes.fromhex(sys.argv[1]),hashlib.sha256).hexdigest())',
            challenge.hex()], text=True).strip()
    verify_receipt_key(os.environ['SEMANTIER_CLIENT_RECEIPT_KEY'], secrets.token_bytes(32), runtime_mac)
    custody_probe(expected_root, runtime_read)
    verify_cloudflare_custody(cid, expected_root)
    print('CI session, signing access, tooling and Runtime custody preflight PASS')


def collect_ci_native_observation(intent, expected_sid, source_commit, graph, expected_full_names=()):
    """Collect provisional facts; file and OS observations do not grant authority."""
    from datetime import datetime
    actor = intent.get('actor') or {}
    objects = graph['objects']
    root = intent.get('root', '')
    suffix = '/' + graph['root']
    if (intent.get('schema') != 1 or intent.get('sourceCommit') != source_commit or
        intent.get('graphSha256') != graph['sha256'] or actor.get('sid') != expected_sid or
        actor.get('session_sid') != expected_sid or not isinstance(actor.get('session'), int) or actor['session'] <= 0 or
        not isinstance(intent.get('actorPid'), int) or intent['actorPid'] <= 0 or
        not isinstance(root, str) or not root.endswith(suffix) or
        not re.fullmatch(r'http://127\.0\.0\.1:[1-9][0-9]{0,4}', root[:-len(suffix)]) or
        not re.fullmatch(r'[0-9a-f]{32}', str(intent.get('operationId', ''))) or
        intent.get('operation') not in ('positive', 'wrong-digest')):
        raise ValueError('Prior CI operation identity mismatch; preserve hold')
    try:
        for field in ('createdAt', 'actorBirth'):
            if datetime.fromisoformat(intent[field]).utcoffset() is None:
                raise ValueError('naive timestamp')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Prior CI operation timestamp invalid; preserve hold') from error
    base = root[:-len(suffix)]
    if intent.get('uris') != [base + '/' + row['key'] for row in objects]:
        raise ValueError('Prior CI operation URI graph mismatch; preserve hold')
    digests = intent.get('digests')
    expected = [str(row['nativeDigest']) for row in objects]
    if not isinstance(digests, list) or len(digests) != len(expected):
        raise ValueError('Prior CI operation digest graph mismatch; preserve hold')
    different = sum(a != b for a, b in zip(digests, expected))
    if (intent['operation'] == 'positive' and different != 0) or (intent['operation'] == 'wrong-digest' and different != 1):
        raise ValueError('Prior CI operation digest graph mismatch; preserve hold')
    if different:
        try:
            index = next(i for i, (a, b) in enumerate(zip(digests, expected)) if a != b)
            actual_bytes = base64.b64decode(digests[index], validate=True)
            expected_bytes = base64.b64decode(expected[index], validate=True)
            if not expected_bytes or actual_bytes != bytes([expected_bytes[0] ^ 1]) + expected_bytes[1:]:
                raise ValueError('wrong mutation')
        except (TypeError, ValueError) as error:
            raise ValueError('Prior CI operation digest graph mismatch; preserve hold') from error
    if any(not isinstance(name, str) or not name or len(name) > 512 for name in expected_full_names) or len(set(expected_full_names)) != len(expected_full_names):
        raise ValueError('Prior CI expected package query names invalid')
    query = {**intent, 'observationExpectedFullNames': list(expected_full_names)}
    payload = base64.b64encode(json.dumps(query, separators=(',', ':')).encode()).decode()
    source = subprocess.check_output(['wslpath', '-w', str(ROOT / 'tests/install/e2e-assets/native-package-deployment.ps1')], text=True).strip()
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = ("$ErrorActionPreference='Stop';" + ci_session_guard(expected_sid) +
               '. ' + quote(source) + ';$intent=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(' + quote(payload) + '))|ConvertFrom-Json;' +
               'Get-CINativeObservation $intent|ConvertTo-Json -Depth 16 -Compress')
    output = subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', base64.b64encode(command.encode('utf-16le')).decode()], text=True, env=native_environment(), timeout=60)
    return json.loads(output.lstrip('\ufeff'))


def verify_ci_installed_content(node, installed_root, plan, graph):
    """Invoke the existing installed-payload verifier without writing a receipt."""
    script = ROOT / 'tests/install/e2e-assets/native-payload-locations.mjs'
    arguments = [str(script), str(installed_root), str(plan), str(graph)]
    if str(node).lower().endswith('.exe'):
        for index in (0, 2, 3):
            arguments[index] = subprocess.check_output(['wslpath', '-w', arguments[index]], text=True, timeout=30).strip()
    program = ("import fs from 'node:fs'; import {pathToFileURL} from 'node:url'; "
               "const args=process.argv.slice(1); process.argv[1]='[ci-recovery-import]'; "
               "const m=await import(pathToFileURL(args[0]).href); "
               "const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\\uFEFF/,'')); "
               "console.log(JSON.stringify(m.verifyNativePayload(args[1],read(args[2]),read(args[3]))));")
    env = os.environ.copy()
    env.pop('SEMANTIER_CLIENT_RECEIPT_KEY', None)
    output = subprocess.check_output([str(node), '--input-type=module', '-e', program, *arguments], text=True, timeout=900, env=env)
    return json.loads(output)


def ci_native_decision(intent, outcome, observation, graph):
    """Describe bounded evidence; neither report permits mutation or new build."""
    import uuid
    decision = {'deployment': 'UNKNOWN', 'nextAction': 'HOLD',
                'currentGraph': 'UNKNOWN', 'observedPackageCondition': 'UNAVAILABLE',
                'operationEvidence': 'UNKNOWN', 'qualificationAction': 'HOLD', 'cleanupAction': 'HOLD',
                'continueQualification': False, 'cleanupAuthorized': False}
    packages = observation.get('packages')
    if isinstance(packages, list) and packages:
        states = [package.get('winrt') for package in packages if isinstance(package, dict)]
        fields = ('DeploymentInProgress', 'Servicing', 'IsPartiallyStaged', 'VerifyIsOK')
        if len(states) == len(packages) and all(isinstance(state, dict) and state.get('presence') == 'PRESENT' and
            isinstance(state.get('status'), dict) and all(isinstance(state['status'].get(field), dict) and
            state['status'][field].get('available') is True and isinstance(state['status'][field].get('value'), bool)
            for field in fields) for state in states):
            if any(state['status']['DeploymentInProgress']['value'] or state['status']['Servicing']['value'] for state in states):
                decision['observedPackageCondition'] = 'SERVICING'
            elif any(state['status']['IsPartiallyStaged']['value'] or not state['status']['VerifyIsOK']['value'] for state in states):
                decision['observedPackageCondition'] = 'PARTIAL_OR_UNHEALTHY'
            else:
                decision['observedPackageCondition'] = 'ALL_OBSERVED_HEALTHY_UNBOUND'
            # Healthy status alone is not proof of the complete applicable graph
            # or installed content; leave currentGraph UNKNOWN until verified.
    if observation.get('processIdentity') in ('mismatch', 'QUERY_ERROR'):
        decision['nextAction'] = 'REFUSE_IDENTITY_MISMATCH' if observation['processIdentity'] == 'mismatch' else 'HOLD_PROCESS_QUERY_ERROR'
        decision['qualificationAction'] = 'REFUSE'
        return decision
    if not isinstance(outcome, dict) or outcome.get('schema') != 1 or outcome.get('operationId') != intent.get('operationId'):
        return decision
    result = outcome.get('result')
    if not isinstance(result, dict) or result.get('operationCompleted') is not True:
        return decision
    try:
        activity = uuid.UUID(str(result.get('activityId')))
        if activity.int == 0:
            return decision
    except (ValueError, TypeError, AttributeError):
        return decision
    if result.get('status') == 'FAILURE':
        decision['reportedOutcome'] = 'FAILURE'
        return decision
    if intent.get('operation') != 'positive' or result.get('status') != 'SUCCESS' or result.get('hresult') != '00000000':
        return decision
    if not isinstance(observation.get('eventQuery'), dict) or observation['eventQuery'].get('state') != 'AVAILABLE':
        return decision
    events = observation.get('events')
    if not isinstance(events, list):
        return decision
    matches = [event for event in events if isinstance(event, dict) and event.get('id') == 478 and
               str(event.get('activity', '')).lower() == str(activity) and event.get('sid') == intent['actor']['sid'] and
               event.get('uri') == intent['root'] and event.get('operation') == '1' and
               str(event.get('callingProcess', '')).lower() == 'powershell.exe' and event.get('full') and
               isinstance(event.get('recordId'), int) and event['recordId'] > 0]
    starts = [event for event in events if isinstance(event, dict) and event.get('id') == 603 and
              str(event.get('activity', '')).lower() == str(activity) and event.get('sid') == intent['actor']['sid'] and
              event.get('operation') == '1' and str(event.get('callingProcess', '')).lower() == 'powershell.exe' and
              isinstance(event.get('recordId'), int) and event['recordId'] > 0]
    mains = [row for row in graph.get('objects', []) if row.get('kind') == 'main-package']
    assets = [row for row in graph.get('objects', []) if row.get('kind') == 'asset-package']
    packages = observation.get('packages')
    if len(mains) != 1 or not isinstance(packages, list) or len(packages) != len(assets) + 1:
        return decision
    main = mains[0]
    if len(matches) != 1 or len(starts) != 1 or starts[0]['recordId'] >= matches[0]['recordId']:
        return decision
    main_prefix = f"{main.get('identity')}_{main.get('version')}_{main.get('arch')}_"
    main_packages = [package for package in packages if package.get('full') == matches[0]['full'] and
                     package.get('full', '').startswith(main_prefix) and package.get('publisher') == main.get('publisher') and
                     package.get('version') == main.get('version') and not package.get('resourceId')]
    resources = [package for package in packages if package.get('resourceId')]
    if len(main_packages) != 1 or len(resources) != len(assets):
        return decision
    if any(len([package for package in resources if package.get('resourceId') == asset.get('resourceId') and
                package.get('publisher') == asset.get('publisher') and package.get('version') == asset.get('version')]) != 1
           for asset in assets):
        return decision
    if len(matches) == 1 and len(starts) == 1:
        decision.update(deployment='SUCCESS_REPORTED', nextAction='REVIEW_INSTALLED_GRAPH')
    return decision


def hold_unfinished_ci_runs(root=Path('/mnt/d/repo/qualification-private'), expected_sid=None, source_commit=None):
    """Preserve prior CI runs until their OS-backed reconciliation is available."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('CI native run custody unavailable')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened, named = os.fstat(fd), root.lstat()
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError('CI native run custody identity changed')
        # scandir errors must propagate: unreadable custody is never an empty set.
        with os.scandir(fd) as entries:
            # Examine native run directories before the coordinator intent
            # files so an intent file cannot mask available observations.
            ordered = sorted(entries, key=lambda item: (not item.is_dir(follow_symlinks=False), item.name))
            holds = []
            for entry in ordered:
                if not entry.name.startswith('client-feedback-ci-'):
                    continue
                try:
                    run = root / entry.name
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        raise ValueError(f'Unsafe prior CI run requires reconciliation: {run}')
                    out = run / 'out'
                    if out.is_symlink():
                        raise ValueError(f'Unsafe prior CI output requires reconciliation: {out}')
                    if not out.is_dir():
                        raise ValueError(f'Incomplete prior CI run requires reconciliation: {run}')
                    if expected_sid and source_commit:
                        # Only a complete, source-bound native candidate may be
                        # observed. Observations remain non-authoritative here.
                        match = re.fullmatch(r'client-feedback-ci-([0-9]+)-([0-9]+)', entry.name)
                        if match:
                            output = Path('/mnt/d/repo/deliverables') / ('SemantierAgent-client-feedback-ci-' + '-'.join(match.groups()))
                            from tests.install.semantier_registered_update import safe_file
                            with safe_file(output / 'channel-request.json', 'r') as stream:
                                request_text = stream.read(4194305)
                            if len(request_text.encode('utf-8')) > 4194304:
                                raise ValueError('Oversized prior CI channel request')
                            prior_source = json.loads(request_text.lstrip('\ufeff')).get('commit')
                            if not isinstance(prior_source, str) or not re.fullmatch(r'[0-9a-f]{40}', prior_source):
                                raise ValueError('Prior CI source identity unavailable')
                            candidate = select_native_candidate(output, prior_source)
                            if candidate['graph']:
                                graph_path = candidate['graph']
                                with safe_file(graph_path, 'r') as stream:
                                    graph_bytes = stream.read(4194305).encode('utf-8')
                                if len(graph_bytes) > 4194304:
                                    raise ValueError('Oversized prior CI graph')
                                graph = json.loads(graph_bytes.decode('utf-8-sig'))
                                with safe_file(output / 'signed/receipt.json', 'r') as stream:
                                    receipt_text = stream.read(4194305)
                                if len(receipt_text.encode('utf-8')) > 4194304:
                                    raise ValueError('Oversized prior CI signed receipt')
                                signed = json.loads(receipt_text.lstrip('\ufeff'))
                                if signed.get('sourceCommit') != prior_source or graph != signed.get('nativeGraph'):
                                    raise ValueError('Prior CI graph changed after candidate selection')
                                graph['sha256'] = hashlib.sha256(graph_bytes).hexdigest()
                                expected_names = []
                                installed_path, members_path = out / 'installed-identity.json', out / 'native-installed-graph.json'
                                if installed_path.exists() and members_path.exists():
                                    with safe_file(installed_path, 'r') as stream:
                                        installed_text = stream.read(4194305)
                                    with safe_file(members_path, 'r') as stream:
                                        members_text = stream.read(4194305)
                                    if max(len(installed_text.encode('utf-8')), len(members_text.encode('utf-8'))) > 4194304:
                                        raise ValueError('Oversized prior CI installed graph record')
                                    installed = json.loads(installed_text.lstrip('\ufeff'))
                                    members = json.loads(members_text.lstrip('\ufeff'))
                                    if members.get('graphSha256') != graph['sha256'] or members.get('sid') != expected_sid:
                                        raise ValueError('Prior CI installed graph record mismatch')
                                    expected_names = [installed['packageFullName']] + [row['full'] for row in members['related']]
                                with os.scandir(out) as records:
                                    intent_paths = sorted((out / entry.name for entry in records if entry.name.startswith('ci-native-intent-') and entry.name.endswith('.json')), key=lambda path: path.name)
                                for intent_path in intent_paths:
                                    try:
                                        with safe_file(intent_path, 'r') as stream:
                                            content = stream.read(4194305)
                                        if len(content.encode('utf-8')) > 4194304:
                                            raise ValueError('Oversized prior CI native intent')
                                        intent = json.loads(content.lstrip('\ufeff'))
                                        if intent_path.name != 'ci-native-intent-' + str(intent.get('operationId')) + '.json':
                                            raise ValueError('Prior CI operation filename mismatch')
                                        outcome_path = out / ('ci-native-outcome-' + str(intent.get('operationId')) + '.json')
                                        outcome = None
                                        if outcome_path.exists() or outcome_path.is_symlink():
                                            with safe_file(outcome_path, 'r') as stream:
                                                outcome_text = stream.read(4194305)
                                            if len(outcome_text.encode('utf-8')) > 4194304:
                                                raise ValueError('Oversized prior CI native outcome')
                                            outcome = json.loads(outcome_text.lstrip('\ufeff'))
                                            if outcome.get('schema') != 1 or outcome.get('operationId') != intent.get('operationId'):
                                                raise ValueError('Prior CI outcome operation mismatch')
                                        observation = collect_ci_native_observation(intent, expected_sid, prior_source, graph, expected_names)
                                        snapshots = observation.get('expectedPackages') or []
                                        if expected_names and len(snapshots) == len(expected_names):
                                            main_snapshot = snapshots[0]
                                            location = main_snapshot.get('installedLocation') or {}
                                            if (main_snapshot.get('presence') == 'PRESENT' and
                                                (main_snapshot.get('identity') or {}).get('full') == expected_names[0] and
                                                location.get('state') == 'VALUE' and location.get('path')):
                                                # This is prior to full tool preflight. A pathname
                                                # alone does not authorize executing Node here.
                                                # Keep the installed-content proof unresolved until
                                                # selected-tool identity is bound and revalidated.
                                                holds.append(f"{intent_path.name}: installed-content verification requires selected Node identity")
                                        decision = ci_native_decision(intent, outcome, observation, graph)
                                        # The physical custody and event coverage needed to
                                        # persist or act on this decision are not proved.
                                        # Surface the bounded result while preserving the hold.
                                        holds.append(f"Prior native operation {intent['operationId']}: {decision['deployment']}; {decision['nextAction']}; reconciliation required")
                                    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                                        holds.append(f'{intent_path.name}: {error}')
                    # Legacy runs also require independently observed quiescence.
                    raise ValueError(f'Prior native run requires reconciliation: {run}')
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                    holds.append(f'{entry.name}: {error}')
            if holds:
                raise ValueError('Prior CI reconciliation holds: ' + '; '.join(holds))
    finally:
        os.close(fd)
    return True


def assert_ci_native_owner():
    # A caller can open and lock the same file itself; that is not proof that
    # the orchestrator performed preflight and reconciliation. Keep the native
    # path closed until it is bound to the coordinator without that bypass.
    raise ValueError('Owned CI execution lock required; native dispatch binding incomplete')


def orchestrate_workflow(expected_sid, incoming, controller, controller_commit, public_base):
    root = Path(incoming)
    expected = CI_INCOMING_ROOT
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError('Unexpected incoming custody root')
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    actual_controller = subprocess.check_output(['/usr/bin/git', '-C', str(controller), 'rev-parse', 'HEAD'], text=True).strip()
    validate_ci_preflight(commit, actual_controller, controller_commit, expected_sid, os.environ.get('SEMANTIER_CLIENT_RECEIPT_KEY', ''))
    for tree in (ROOT, controller):
        if subprocess.check_output(['/usr/bin/git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
            raise ValueError('CI checkout is not clean')
    guard = "$ErrorActionPreference='Stop';" + ci_session_guard(expected_sid)
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', base64.b64encode(guard.encode('utf-16le')).decode()], check=True, env=native_environment())
    run_id, attempt = os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_RUN_ATTEMPT']
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError('Numeric CI run identity required')
    output_win = f'D:\\repo\\deliverables\\SemantierAgent-client-feedback-ci-{run_id}-{attempt}'
    work = f'D:\\repo\\qualification-private\\client-feedback-ci-{run_id}-{attempt}'
    output = subprocess.check_output(['wslpath', '-u', output_win], text=True).strip()
    native = subprocess.check_output(['wslpath', '-u', work], text=True).strip() + '/out'
    script = str(Path(__file__).resolve())
    build = [sys.executable, str(controller / 'scripts/build_hermes_windows.py'), '--source', str(ROOT), '--output', output_win, '--thumbprint', '57BF15739B4AA447842EAF388856A01B4E5029FF', '--acceptance-public-base', public_base, '--native-local-git']
    layout = os.environ.get('ACCEPTANCE_PACKAGE_LAYOUT', 'legacy')
    previous = os.environ.get('ACCEPTANCE_PREVIOUS_NATIVE_LAYOUT', '')
    if layout == 'native-assets':
        build += ['--native-assets', '--python', os.environ['ACCEPTANCE_WINDOWS_PYTHON']]
        if previous:
            build += ['--previous-native-layout', previous]
    elif layout != 'legacy' or previous:
        raise ValueError('Invalid native layout selection')
    phases = [
        ('preflight', [sys.executable, script, '--preflight', '--expected-sid', expected_sid, '--incoming', str(root), '--controller', str(controller), '--controller-commit', controller_commit, '--public-base', public_base]),
        ('build', build),
        ('native', lambda custody_process, lock_fd: _native_smoke(argparse.Namespace(output=Path(output), work=work, commit=commit, expected_sid=expected_sid), custody_process, lock_fd)),
        ('stage', [sys.executable, str(ROOT / 'scripts/releases/semantier_candidate.py'), '--output', output, '--native', native, '--incoming', str(root), '--target-incoming', str(root.parent/'incoming_r2'), '--notes', str(ROOT / 'scripts/releases/semantier-acceptance-notes.json')]),
    ]
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        intent = Path('/mnt/d/repo/qualification-private') / f'client-feedback-ci-{run_id}-{attempt}-coordinator.intent'
        # The supported trusted-operator entry runs reconciliation and real
        # preflight inside this ownership interval; neither lock possession nor
        # a caller-provided receipt substitutes for those checks. Historical
        # unknowns still hold, and the standalone native CLI remains refused.
        orchestrate_ci(root, phases, lambda: hold_unfinished_ci_runs(expected_sid=expected_sid, source_commit=commit), run_intent=intent, custody=lambda fd:hold_ci_run_custody(fd,expected_sid),
                       stage_preflight=lambda: preflight(expected_sid, str(root), controller, controller_commit, public_base))
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _native_smoke(a, custody_process, lock_fd):
    windows=lambda path:subprocess.check_output(['wslpath','-w',str(path)],text=True,encoding='utf-8',errors='replace',timeout=30).strip()
    candidate=select_native_candidate(a.output,a.commit,os.environ.get('ACCEPTANCE_CONTROLLER_COMMIT',''))
    if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Real CI execution context required')
    if not a.expected_sid.startswith('S-1-5-21-') or a.expected_sid in ('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011'):raise ValueError('Separate bound CI account required')
    if len(a.commit)!=40 or any(c not in 'abcdef0123456789' for c in a.commit):raise ValueError('Exact source required')
    payload=Path('/mnt/d/repo/hermes-build/apps/desktop/build/agent-payload')
    manifest=json.loads((payload/'manifest.json').read_text(encoding='utf-8-sig'))
    python=(payload/manifest['runtime']['storePython']).resolve()
    if not python.is_relative_to(payload.resolve()) or not python.is_file():
        raise ValueError('Prepared native Python must remain inside the owned payload')
    command={'operation':'native-smoke','commit':a.commit,'expectedSid':a.expected_sid,'work':a.work,
             'artifact':windows(candidate['artifact']),'channelRequest':windows(a.output/'channel-request.json'),
             'python':windows(python),'graph':windows(candidate['graph']) if candidate['graph'] else '',
             'plan':windows(candidate['plan']) if candidate['plan'] else ''}
    run_ci_custody_command(custody_process, command)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--work');p.add_argument('--commit');p.add_argument('--expected-sid',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--orchestrate',action='store_true');p.add_argument('--incoming');p.add_argument('--controller',type=Path);p.add_argument('--controller-commit');p.add_argument('--public-base');a=p.parse_args()
    if a.orchestrate:
        if not a.incoming or not a.controller or not a.controller_commit or not a.public_base: p.error('orchestration requires incoming/controller/controller-commit/public-base')
        orchestrate_workflow(a.expected_sid, a.incoming, a.controller, a.controller_commit, a.public_base);sys.exit(0)
    if a.preflight:
        if not a.incoming or not a.controller or not a.controller_commit or not a.public_base: p.error('preflight requires incoming/controller/controller-commit/public-base')
        preflight(a.expected_sid,a.incoming,a.controller,a.controller_commit,a.public_base);sys.exit(0)
    if not a.output or not a.work or not a.commit: p.error('native smoke requires output/work/commit')
    assert_ci_native_owner()
