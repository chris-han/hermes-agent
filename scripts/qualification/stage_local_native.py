"""Stage locally built Windows artifacts for the existing Channel 2 owner.

This emits only an artifact receipt. It does not claim an installed update.
The private receipt key is read from the environment and never printed.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, UTC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from client_operations.artifact_io import immutable_copy, immutable_write, safe_file
from client_operations.package_graph import decode_package_graph
from client_operations.contracts import canonical, digest
from scripts.bundles.parent_composition import verify as verify_composition
from scripts.releases.semantier_native_ci import select_native_candidate


def read(path):
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('invalid evidence file')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('evidence must be an object')
    return value


def verify_windows_signatures(files, thumbprint):
    paths = [subprocess.check_output(['wslpath', '-w', str(path.resolve(strict=True))], text=True, encoding='utf-8', timeout=30).strip() for path in files]
    encoded = base64.b64encode(json.dumps(paths).encode('utf-8')).decode('ascii')
    if len(thumbprint) != 40 or any(c not in '0123456789ABCDEF' for c in thumbprint):
        raise ValueError('invalid expected signer thumbprint')
    script = r"""
$ErrorActionPreference='Stop'
$files=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PATHS')) | ConvertFrom-Json)
$tool=@(Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName -Descending | Select-Object -First 1)
if($tool.Count -ne 1){throw 'SignTool unavailable'}
foreach($file in $files){
  & $tool[0].FullName verify /pa $file *> $null
  if($LASTEXITCODE -ne 0){throw 'SignTool verification failed'}
  $signature=Get-AuthenticodeSignature -LiteralPath $file
  if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -cne 'THUMB'){throw 'Signature or signer mismatch'}
}
""".replace('PATHS', encoded).replace('THUMB', thumbprint)
    command = base64.b64encode(script.encode('utf-16le')).decode('ascii')
    env = os.environ.copy()
    for name in tuple(env):
        if name.upper() == 'SEMANTIER_CLIENT_RECEIPT_KEY':
            env.pop(name)
    env['WSLENV'] = ':'.join(item for item in env.get('WSLENV', '').split(':')
                             if item and item.split('/')[0].upper() != 'SEMANTIER_CLIENT_RECEIPT_KEY')
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
                    '-NoProfile', '-NonInteractive', '-EncodedCommand', command], check=True, env=env, timeout=900)


def stage(output, source, incoming, notes, receipt_key, *, source_commit, controller_commit,
          windows_version, thumbprint, package_family):
    output, source, incoming = map(Path, (output, source, incoming))
    if len(receipt_key) < 32:
        raise ValueError('local receipt key unavailable')
    if subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, encoding='utf-8', timeout=30).strip() != source_commit:
        raise ValueError('source checkout differs from built commit')
    selected = select_native_candidate(output, source_commit, controller_commit)
    if selected['graph'] is None:
        raise ValueError('signed native graph required')
    request = read(output / 'channel-request.json')
    signed = read(output / 'signed/receipt.json')
    result = read(output / 'result.json')
    if (request.get('windowsVersion') != windows_version or request.get('publicBase') != 'https://s3.semantier.com'
            or signed.get('signing') != 'PASS' or not signed.get('verification', '').startswith('PASS:')
            or signed.get('certificateThumbprint') != thumbprint or signed.get('packageVersion') != windows_version
            or result.get('certificateThumbprint') != thumbprint or signed.get('architecture') != 'x64'):
        raise ValueError('signed candidate identity mismatch')
    graph = signed['nativeGraph']
    directory = output / 'signed/distribution'
    graph_file = selected['graph']
    graph_body = graph_file.read_bytes()
    graph_hash = hashlib.sha256(graph_body).hexdigest()
    root = next(row for row in graph['objects'] if row['key'] == graph['root'])
    descriptor = {'key':graph['root'].rsplit('/', 1)[0]+'/native-graph.json',
                  'size':len(graph_body), 'sha256':graph_hash}
    package = {'platform':'win32', 'arch':'x64', 'variant':'native-msix-assets',
               'identity':request['identity']['msixAppIdWithOrg'], 'publisher':signed['publisher'],
               'version':windows_version, 'artifact':{key:root[key] for key in ('key', 'size', 'sha256')},
               'feed':{'key':descriptor['key'], 'channel':'stable'},
               'native':{'schema':1, 'api':'PackageManager.AddPackageByUriAsync',
                         'digestMethod':'IAppxDigestProvider.GetDigest', 'digestEncoding':'base64', 'graph':descriptor}}
    if (decode_package_graph(graph_body, package, request['buildId']) != graph
            or not package_family.startswith(package['identity']+'_')
            or signed.get('packageName') != package['identity']):
        raise ValueError('native graph or package family mismatch')
    composition = source / 'semantier-runtime'
    manifest_bytes = (composition/'composition.json').read_bytes()
    committed_manifest = subprocess.check_output(['/usr/bin/git', '-C', str(source), 'show',
                                                  source_commit+':semantier-runtime/composition.json'], timeout=30)
    if manifest_bytes != committed_manifest:
        raise ValueError('source composition differs from built commit')
    composition_info = verify_composition(composition)
    files = [directory / row['key'].rsplit('/', 1)[1] for row in graph['objects']]
    verify_windows_signatures(files, thumbprint)
    for row in graph['objects']:
        immutable_copy(directory / row['key'].rsplit('/', 1)[1], safe_file(incoming, row['key']), row)
    immutable_copy(graph_file, safe_file(incoming, descriptor['key']), descriptor)
    receipt = {'produced_at':datetime.now(UTC).isoformat(), 'provenance':'local-ci',
               'qualification_contract':'windows-native-uri-graph/local-v1', 'source_commit':source_commit,
               'controller_commit':controller_commit, 'build_id':request['buildId'], 'repository':request['repository'],
               'platform':'win32', 'arch':'x64', 'identity':package['identity'], 'publisher':package['publisher'],
               'application_id':request['identity']['appNamePascal'], 'windows_version':windows_version,
               'sha256':root['sha256'], 'size':root['size'], 'signature_result':'PASS',
               'qualification':'windows-x64-native-uri-graph', 'package_family':package_family,
               'notes_sha256':digest(notes), 'runtime_parent_commit':composition_info['parent_commit'],
               'runtime_composition_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
               'native_graph_sha256':graph_hash, 'native_objects_sha256':digest(graph['objects']), 'artifact_result':'PASS'}
    return {'release_id':request['buildId'], 'manifest':{'schema':2, 'request':request, 'packages':[package]},
            'notes':notes, 'native_receipt':receipt,
            'receipt_hmac':hmac.new(receipt_key, canonical(receipt), hashlib.sha256).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    for name in ('output', 'source', 'incoming', 'notes', 'candidate', 'source-commit', 'controller-commit',
                 'windows-version', 'thumbprint', 'package-family'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    notes = read(Path(args.notes))
    result = stage(args.output, args.source, args.incoming, notes,
                   os.environ.get('SEMANTIER_CLIENT_RECEIPT_KEY', '').encode(), source_commit=args.source_commit,
                   controller_commit=args.controller_commit, windows_version=args.windows_version,
                   thumbprint=args.thumbprint, package_family=args.package_family)
    destination = Path(args.candidate)
    if destination.parent.stat().st_mode & 0o077:
        raise ValueError('candidate directory must be private')
    immutable_write(destination, canonical(result))
    print('Local native artifact staged:', result['release_id'])


if __name__ == '__main__':
    main()
