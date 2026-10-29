"""WSL bridge to the existing real native installation smoke, owned family only."""
import argparse,base64,json,os,subprocess,sys
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--work',required=True);p.add_argument('--commit',required=True);p.add_argument('--expected-sid',required=True);a=p.parse_args()
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
