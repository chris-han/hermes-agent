"""Run native qualification using only an extracted package's own Python code.

Invoke with its shipped base Python and -I. Never activate an extracted venv or
execute editable .pth files, which can redirect imports to the build checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--payload', required=True, type=Path)
parser.add_argument('--script', required=True, type=Path)
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
assert os.name == 'nt' and sys.flags.isolated, 'shipped Windows base Python -I required'
payload = args.payload.resolve()
manifest = json.loads((payload / 'manifest.json').read_text(encoding='utf-8-sig'))
runtime = manifest['runtime']
executable = (payload / runtime['storePython']).resolve()
assert Path(sys.executable).resolve() == executable
assert Path(sys._base_executable).resolve() == executable
assert Path(sys.base_prefix).resolve().is_relative_to(payload / 'tools')
source = (payload / runtime['repoDir']).resolve()
packages = (payload / runtime['sitePackages']).resolve()
assert source.is_relative_to(payload) and packages.is_relative_to(payload)
script = args.script.resolve()
assert script.parent == Path(__file__).resolve().parent
assert script.name in {'windows_cloud_device.py', 'windows_cloud_lifecycle.py', 'windows_cloud_profile_setup.py'}
sys.path[:0] = [str(source), str(packages)]
os.environ['HERMES_QUALIFICATION_PAYLOAD'] = str(payload)
os.environ['HERMES_QUALIFICATION_RUNNER'] = str(Path(__file__).resolve())
arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
assert '--source' in arguments and Path(arguments[arguments.index('--source') + 1]).resolve() == source
assert '--root' in arguments
home = Path(arguments[arguments.index('--root') + 1]).resolve()
assert home.parent.parent.name == 'windows-cloud-client-r1'


def retain_origins():
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if not origin or origin in {'frozen', 'built-in'}:
            continue
        path = Path(origin).resolve()
        assert path.is_relative_to(payload) or path.parent == script.parent, (name, str(path))
        if path.is_file():
            origins[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    home.parent.mkdir(parents=True, exist_ok=True)
    (home.parent / ('payload-origins-' + str(os.getpid()) + '.json')).write_text(json.dumps({
        'scope': 'EXTRACTED_RUNTIME_ONLY', 'source_commit': manifest['ref'],
        'executable': str(executable), 'base_prefix': sys.base_prefix,
        'modules': origins}, indent=2), encoding='utf-8')


sys.argv = [str(script), *arguments]
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    retain_origins()
