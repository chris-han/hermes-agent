"""WSL bridge to the existing real native installation smoke, owned family only."""
import argparse,base64,json,os,subprocess
from pathlib import Path

def native_environment():
    """Forward existing process-local proxies across WSL without host changes."""
    env=os.environ.copy()
    names=('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY')
    forwarded=[item for item in env.get('WSLENV','').split(':') if item and item.split('/')[0].upper() not in names]
    for name in names:
        value=env.get(name) or env.get(name.lower())
        env.pop(name.lower(),None)
        if value:
            env[name]=value
            forwarded.append(name)
    env['WSLENV']=':'.join(forwarded)
    return env


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--work',required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    # All interpolated paths go through PowerShell literal quoting, never shell evaluation.
    quote=lambda value:"'"+str(value).replace("'","''")+"'"
    windows=lambda path:subprocess.check_output(['wslpath','-w',str(path)],text=True,encoding='utf-8',errors='replace',timeout=30).strip()
    files=list((a.output/'signed/distribution').glob('*.msix'))
    if len(files)!=1:raise ValueError('One signed candidate required')
    if len(a.commit)!=40 or any(c not in 'abcdef0123456789' for c in a.commit):raise ValueError('Exact source required')
    payload=Path('/mnt/d/repo/hermes-build/apps/desktop/build/agent-payload')
    manifest=json.loads((payload/'manifest.json').read_text(encoding='utf-8-sig'))
    python=(payload/manifest['runtime']['storePython']).resolve()
    if not python.is_relative_to(payload.resolve()) or not python.is_file():
        raise ValueError('Prepared native Python must remain inside the owned payload')
    command="$ErrorActionPreference='Stop';$env:HERMES_PYTHON="+quote(windows(python))+";$env:PATH='D:\\repo\\hermes-build\\.cache\\desktop-inputs\\tools\\node-26.7.0-win32-x64;'+$env:PATH;$env:RUNNER_TEMP="+quote(a.work)+";New-Item -ItemType Directory -Path $env:RUNNER_TEMP -ErrorAction Stop | Out-Null; & 'D:\\repo\\hermes-build\\tests\\install\\windows-bundle-smoke.ps1' -SemantierAcceptanceHost -Arch x64 -Commit "+quote(a.commit)+" -Artifact "+quote(windows(files[0]))+" -ChannelRequest "+quote(windows(a.output/'channel-request.json'))+" -Work "+quote(a.work+'\\work')+" -Out "+quote(a.work+'\\out')+"; if($LASTEXITCODE -ne 0){exit 1}"
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],check=True,env=native_environment())
