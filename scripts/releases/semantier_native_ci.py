"""WSL bridge to the existing real native installation smoke, owned family only."""
import argparse,base64,hashlib,hmac,json,os,re,subprocess,sys,secrets,stat
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def native_environment():
    """Forward existing proxies and genuine CI flag across WSL without host changes."""
    env=os.environ.copy()
    names=('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','GITHUB_ACTIONS')
    forwarded=[item for item in env.get('WSLENV','').split(':') if item and item.split('/')[0].upper() not in names]
    for name in names:
        value=env.get(name) or env.get(name.lower())
        env.pop(name.lower(),None)
        if value:
            env[name]=value
            forwarded.append(name)
    env['WSLENV']=':'.join(forwarded)
    return env


def select_native_candidate(output,commit):
    """Select exact built inputs; signature/installed qualification is native-side."""
    from scripts.releases.semantier_windows import validate_native_graph
    output=Path(output)
    read=lambda name:json.loads((output/name).read_text(encoding='utf-8-sig'))
    signed=read('signed/receipt.json');request=read('channel-request.json');result=read('result.json')
    if signed.get('sourceCommit')!=commit or request.get('commit')!=commit or result.get('sourceCommit')!=commit or result.get('result')!='PASS':
        raise ValueError('Completed exact source required for native smoke')
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


def preflight(expected_sid, incoming, controller, expected_controller, public_base):
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
if(!(Get-ChildItem -Path (Join-Path $sdk '*\x64\signtool.exe') -ErrorAction SilentlyContinue)){throw 'SignTool unavailable'}
if(!(Get-ChildItem -Path (Join-Path $sdk '*\x64\makeappx.exe') -ErrorAction SilentlyContinue)){throw 'MakeAppx unavailable'}
Assert-CISession|Out-Null
"""
    env = native_environment()
    env['WSLENV'] += ':ACCEPTANCE_WINDOWS_PYTHON'
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(cmd.encode('utf-16le')).decode()], check=True, env=env)
    expected_root = Path('/home/chris/repo/semantier-runtime/.worktrees/cloudflare-update2-three-sources-v2/.artifacts/update2-publication-v2/qualified-v3/incoming')
    if Path(incoming).resolve(strict=True) != expected_root.resolve(strict=True):
        raise ValueError('Unexpected incoming custody root')
    container = 'semantier-update2-publication-v14-backend'
    inspect = json.loads(subprocess.check_output(['docker','inspect',container,'--format','{{json .Mounts}}'], text=True))
    expected_mount = str(expected_root.parents[1])
    if not any(m.get('Source') == expected_mount and m.get('Destination') == '/fixture' for m in inspect):
        raise ValueError('Runtime custody mount mismatch')
    details = subprocess.check_output(['docker','inspect',container,'--format','{{.Id}} {{.Image}} {{.State.Running}}'], text=True).strip().split()
    if len(details) != 3 or details[1] != 'sha256:3e72e6308b5b67e81cb715c950340447edff623f4b8b035cf8ac0ace903309cb' or details[2] != 'true':
        raise ValueError('Runtime identity or state mismatch')
    cid = details[0]
    if not cid or any(c not in '0123456789abcdef' for c in cid):
        raise ValueError('Runtime container identity unavailable')
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
    print('CI session, signing access, tooling and Runtime custody preflight PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--work');p.add_argument('--commit');p.add_argument('--expected-sid',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--incoming');p.add_argument('--controller',type=Path);p.add_argument('--controller-commit');p.add_argument('--public-base');a=p.parse_args()
    if a.preflight:
        if not a.incoming or not a.controller or not a.controller_commit or not a.public_base: p.error('preflight requires incoming/controller/controller-commit/public-base')
        preflight(a.expected_sid,a.incoming,a.controller,a.controller_commit,a.public_base);sys.exit(0)
    if not a.output or not a.work or not a.commit: p.error('native smoke requires output/work/commit')
    # All interpolated paths go through PowerShell literal quoting, never shell evaluation.
    quote=lambda value:"'"+str(value).replace("'","''")+"'"
    windows=lambda path:subprocess.check_output(['wslpath','-w',str(path)],text=True,encoding='utf-8',errors='replace',timeout=30).strip()
    candidate=select_native_candidate(a.output,a.commit)
    if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Real CI execution context required')
    if not a.expected_sid.startswith('S-1-5-21-') or a.expected_sid in ('S-1-5-21-1928370875-4287869425-2092521300-1003','S-1-5-21-1928370875-4287869425-2092521300-1011'):raise ValueError('Separate bound CI account required')
    if len(a.commit)!=40 or any(c not in 'abcdef0123456789' for c in a.commit):raise ValueError('Exact source required')
    payload=Path('/mnt/d/repo/hermes-build/apps/desktop/build/agent-payload')
    manifest=json.loads((payload/'manifest.json').read_text(encoding='utf-8-sig'))
    python=(payload/manifest['runtime']['storePython']).resolve()
    if not python.is_relative_to(payload.resolve()) or not python.is_file():
        raise ValueError('Prepared native Python must remain inside the owned payload')
    guard=ci_session_guard(a.expected_sid)
    command="$ErrorActionPreference='Stop';"+guard+"$env:HERMES_PYTHON="+quote(windows(python))+";$env:PATH='D:\\repo\\hermes-build\\.cache\\desktop-inputs\\tools\\node-26.7.0-win32-x64;'+$env:PATH;$env:RUNNER_TEMP="+quote(a.work)+";if(Test-Path -LiteralPath 'D:\\repo\\hermes-build.distribution.lock'){throw 'Owned Windows builder remains active'};New-Item -ItemType Directory -Path $env:RUNNER_TEMP -ErrorAction Stop | Out-Null;$ciContextBefore|ConvertTo-Json -Compress|Set-Content -LiteralPath (Join-Path $env:RUNNER_TEMP 'ci-context-before.json') -Encoding utf8; & 'D:\\repo\\hermes-build\\tests\\install\\windows-bundle-smoke.ps1' -SemantierAcceptanceHost -Arch x64 -Commit "+quote(a.commit)+" -Artifact "+quote(windows(candidate['artifact']))+" -ChannelRequest "+quote(windows(a.output/'channel-request.json'))+" -Work "+quote(a.work+'\\work')+" -Out "+quote(a.work+'\\out')+" -ExpectedSid "+quote(a.expected_sid)+(" -NativeGraph "+quote(windows(candidate['graph']))+" -NativeLayoutPlan "+quote(windows(candidate['plan']))+" -Python "+quote(windows(python)) if candidate['graph'] else '')+"; if($LASTEXITCODE -ne 0){exit 1};Assert-CISession|ConvertTo-Json -Compress|Set-Content -LiteralPath (Join-Path $env:RUNNER_TEMP 'ci-context-after.json') -Encoding utf8"
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],check=True,env=native_environment())
