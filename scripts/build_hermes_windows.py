#!/usr/bin/env python3
"""Build, locally sign and validate a Windows distribution from WSL. No remote push."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
ROOT = Path(__file__).resolve().parents[1]
# Runtime is optional: standalone Hermes builds use their own controller/helpers.
RUNTIME_ROOT = (ROOT.parent if (ROOT.parent / 'hermes-agent').resolve() == ROOT
                and (ROOT.parent / 'docs/canonical/architecture.md').is_file() else None)


def windows_process_environment():
    """Forward existing proxy configuration across WSL, once per Windows name."""
    environment = os.environ.copy()
    names = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')
    forwarded = [item for item in environment.get('WSLENV', '').split(':')
                 if item and item.split('/')[0].upper() not in names]
    for name in names:
        value = environment.get(name) or environment.get(name.lower())
        environment.pop(name.lower(), None)
        if value:
            environment[name] = value
            forwarded.append(name)
    environment['WSLENV'] = ':'.join(forwarded)
    return environment


def run(*args):
    return subprocess.check_output(args, text=True).strip()


@contextmanager
def source_bundle(source, commit, *, base=None):
    """A unique temporary ref lets Git bundle any exact commit without moving HEAD."""
    ref = f'refs/local-windows-build/{uuid.uuid4().hex}'
    subprocess.run(['/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'update-ref', ref, commit], check=True)
    try:
        with tempfile.TemporaryDirectory(prefix='hermes-windows-') as folder:
            bundle = Path(folder) / 'source.bundle'
            subprocess.run(['/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'bundle', 'create', str(bundle), ref,
                            *([f'^{base}'] if base else [])], check=True)
            subprocess.run(['/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'bundle', 'verify', str(bundle)], check=True)
            yield bundle, ref
    finally:
        subprocess.run(['/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'update-ref', '-d', ref, commit], check=True)


def transfer_base(source, repository, commit):
    """Read native checkout HEAD with Linux Git before preparing a bundle.

    An unavailable/unrelated checkout gets the existing full-history transfer.
    Include the final commit even on identical retries (a tiny checked bundle).
    """
    try:
        native_checkout = run('wslpath', '-u', repository)
        candidate = run('/usr/bin/git', '-C', native_checkout, 'rev-parse', '--verify', 'HEAD')
        if not re.fullmatch(r'[0-9a-f]{40}', candidate):
            return None
        base = run('/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'merge-base', commit, candidate)
        if base == commit:
            base = run('/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'rev-parse', f'{commit}^')
        print(f'Incremental source bundle, prerequisite {base}', flush=True)
        return base
    except subprocess.CalledProcessError:
        print('Native base unavailable; using complete source bundle', flush=True)
        return None


def preseed_owned_worktree(repository, worktree, bundle, ref, commit):
    """Transfer an exact bundle into the existing owned worktree under both locks."""
    repository, worktree = Path(repository), Path(worktree)
    marker = worktree.with_name(worktree.name + '.distribution-owner.json')
    owner = json.loads(marker.read_text(encoding='utf-8-sig'))
    if owner.get('kind') != 'hermes-windows-distribution-v1' or owner.get('repository', '').lower() != r'd:\repo\hermes-build' or owner.get('commonDirectory', '').lower() != r'd:\repo\hermes-build\.git':
        raise ValueError('owned Windows worktree marker mismatch')
    for path in (repository, repository / '.git', worktree, marker):
        if path.is_symlink() or not path.exists():
            raise ValueError('owned Windows build path is unavailable or redirected')
    config = (repository / '.git/config').read_text(encoding='utf-8')
    if re.search(r'(?im)^\s*\[include(?:if)?\b|^\s*worktree\s*=', config) or (repository / '.git/objects/info/alternates').exists():
        raise ValueError('native Git repository redirects are not allowed')
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    locks = []
    try:
        for path in (repository.with_name(repository.name + '.distribution.lock'), worktree.with_name(worktree.name + '.distribution.lock')):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            locks.append((path, descriptor, os.fstat(descriptor)))
        def git(path, *args):
            return subprocess.check_output(['/usr/bin/git', '-C', str(path), *args], env=environment, text=True).strip()
        common = Path(git(worktree, 'rev-parse', '--path-format=absolute', '--git-common-dir')).resolve()
        if common != (repository / '.git').resolve() or git(worktree, 'branch', '--show-current') or git(worktree, 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('owned build worktree is dirty or not detached')
        original_head = git(repository, 'rev-parse', 'HEAD')
        original_status = git(repository, 'status', '--porcelain', '--untracked-files=all')
        subprocess.run(['/usr/bin/git', '-C', str(repository), 'bundle', 'verify', str(bundle)], env=environment, check=True)
        subprocess.run(['/usr/bin/git', '-C', str(repository), 'fetch', '--no-write-fetch-head', str(bundle), ref], env=environment, check=True)
        if git(repository, 'rev-parse', commit + '^{commit}') != commit:
            raise ValueError('transferred source commit mismatch')
        if git(worktree, 'rev-parse', 'HEAD') != commit:
            subprocess.run(['/usr/bin/git', '-C', str(worktree), 'checkout', '--detach', commit], env=environment, check=True)
        if git(worktree, 'rev-parse', 'HEAD') != commit or git(worktree, 'status', '--porcelain', '--untracked-files=all') or git(repository, 'rev-parse', 'HEAD') != original_head or git(repository, 'status', '--porcelain', '--untracked-files=all') != original_status:
            raise ValueError('source or original checkout changed during transfer')
    finally:
        for path, descriptor, identity in reversed(locks):
            try:
                current = path.stat()
                if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                    raise RuntimeError('build lock identity changed')
                path.unlink()
            finally:
                os.close(descriptor)


def windows_path(path):
    return run('wslpath', '-w', str(Path(path).resolve()))


def stage_parent_composition(parent: Path, hermes: Path) -> Path:
    """Freeze committed parent resources inside the clean-build source repository.

    This is a source-composition step, not a parallel build/signing pipeline. The
    caller reviews/commits the generated snapshot before the existing controller.
    """
    import ast
    import hashlib
    import io
    import tarfile

    paths = ['src', 'bootstrap/common_knowledge', 'docs/workshop/materials', 'pyproject.toml']
    licenses = 'docs/operational/evidence/windows-cpu-sensevoice-package-v1'
    paths += [f'{licenses}/{name}' for name in
              ('FunASR-MODEL-LICENSE.txt', 'export-LICENSE.txt', 'SenseVoice-model-card.md', 'license-provenance.json')]
    if run('/usr/bin/git', '-C', str(parent), '--work-tree=' + str(parent), 'status', '--porcelain', '--untracked-files=all', '--', *paths):
        raise ValueError('parent composition inputs must be committed and clean')
    commit = run('/usr/bin/git', '-C', str(parent), '--work-tree=' + str(parent), 'rev-parse', 'HEAD')
    destination = hermes / 'semantier-runtime'
    if destination.exists():
        raise ValueError('composition destination already exists; preserve/reconcile it before restaging')
    archive = subprocess.check_output(['/usr/bin/git', '-C', str(parent), '--work-tree=' + str(parent), 'archive', commit, '--', *paths])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not (member.isfile() or member.isdir()):
                raise ValueError('parent composition contains an unsafe archive entry')
        stream.extractall(destination)
    provider = ast.parse((destination / 'src/plugins/sensevoice_stt/provider.py').read_text())
    assets = next(ast.literal_eval(node.value) for node in provider.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ASSETS' for t in node.targets))
    revision = '2365baeacb507f821a0c8120fcee3d484dba7a07'
    model_repo = 'csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17'
    notice = destination / 'MODEL-NOTICE.txt'
    notice.write_text('SenseVoice Small — FunAudioLLM / Alibaba Group.\n'
        'INT8 ONNX export: ' + model_repo + ' at ' + revision + '.\n'
        'Model weights use the retained FunASR Model Open Source License Agreement1.1, not Apache2.0.\n'
        'Full model terms, attribution and upstream model card: ' + licenses + '/\n', encoding='utf-8')
    files = [{'path': str(p.relative_to(destination)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(destination.rglob('*')) if p.is_file()]
    manifest = dict(schema=1, parent_commit=commit, files=files,
                    models=[dict(name=name, bytes=size, sha256=digest,
                        url=f'https://huggingface.co/{model_repo}/resolve/{revision}/{name}')
                        for name, (size,digest) in assets.items()])
    (destination / 'composition.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return destination


def validate_source(source, *, expected_ref, expected_tree, receipt_path=None, legacy=False):
    """Validate exact build inputs only; no transfer, dependency install or packaging."""
    import hashlib
    source = Path(source).resolve(strict=True)
    if legacy:
        if expected_ref or expected_tree or receipt_path:
            raise ValueError('source pins/receipt require detached mode, not legacy mode')
    elif not all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value)
                 for value in (expected_ref, expected_tree)):
        raise ValueError('detached --source-ref and --source-tree require full exact Git identities')
    try:
        common = Path(run('/usr/bin/git', '-C', str(RUNTIME_ROOT or ROOT), 'rev-parse', '--git-common-dir'))
        common = ((RUNTIME_ROOT or ROOT) / common).resolve() if not common.is_absolute() else common.resolve()
        if not legacy and RUNTIME_ROOT is not None and source.is_relative_to(common.parent):
            raise ValueError('detached source must be outside Runtime, including symlink aliases')
        top = Path(run('/usr/bin/git', '-C', str(source), 'rev-parse', '--show-toplevel')).resolve()
        if top != source:
            raise ValueError('source must be the repository root')
        source_common = Path(run('/usr/bin/git', '-C', str(source), 'rev-parse', '--git-common-dir'))
        source_common = (source / source_common).resolve() if not source_common.is_absolute() else source_common.resolve()
        if not legacy and RUNTIME_ROOT is not None and source_common.is_relative_to(common):
            raise ValueError('detached source Git ownership must be outside Runtime')
        commit = run('/usr/bin/git', '-C', str(source), 'rev-parse', '--verify', 'HEAD')
        tree = run('/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD^{tree}')
        if not legacy and (commit != expected_ref or tree != expected_tree):
            raise ValueError('source commit/tree mismatch')
        if run('/usr/bin/git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'):
            raise ValueError('Hermes source must be committed and clean; no implicit stash/reset')
        controller = source / 'scripts/build-windows-distribution.ps1'
        if not controller.is_file() or not controller.resolve().is_relative_to(source):
            raise ValueError('source does not contain an owned scripts/build-windows-distribution.ps1')
        result = dict(sourceCommit=commit, sourceTree=tree, controllerSourceCommit=commit,
            parentControllerCommit=run('/usr/bin/git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
            sourceMode='legacy-submodule' if legacy else 'detached', sourceRoot=str(source),
            mode='source-validation', result='PASS')
        if not legacy:
            tracked = set(run('/usr/bin/git', '-C', str(source), 'ls-files', '-z').split('\0'))
            if 'scripts/build-windows-distribution.ps1' not in tracked:
                raise ValueError('source controller must be tracked')
            composition = source / 'semantier-runtime/composition.json'
            if ('semantier-runtime/composition.json' not in tracked
                    or not composition.is_file() or composition.is_symlink()):
                raise ValueError('detached product source requires its existing parent composition receipt')
            manifest = json.loads(composition.read_text())
            if (manifest.get('schema') != 1 or not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('parent_commit', '')))
                    or not isinstance(manifest.get('files'), list) or not manifest['files']):
                raise ValueError('invalid parent composition receipt')
            for entry in manifest['files']:
                relative = Path(entry['path'])
                path = composition.parent / relative
                if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                        or path.relative_to(source).as_posix() not in tracked
                        or not path.resolve().is_relative_to(composition.parent.resolve())
                        or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
                    raise ValueError('parent composition file identity mismatch')
            result.update(parentCompositionCommit=manifest['parent_commit'],
                          parentCompositionSha256=hashlib.sha256(composition.read_bytes()).hexdigest())
        if receipt_path is not None:
            previous = json.loads(Path(receipt_path).read_text(encoding='utf-8-sig'))
            keys = ('sourceCommit', 'sourceTree', 'controllerSourceCommit',
                    'parentCompositionCommit', 'parentCompositionSha256')
            if previous.get('result') != 'PASS' or any(previous.get(key) != result.get(key) for key in keys):
                raise ValueError('source receipt identity mismatch')
        return result
    except subprocess.CalledProcessError as exc:
        raise ValueError('source Git identity unavailable') from exc


def main():
    started = time.monotonic()
    produced_at = datetime.now(datetime.UTC).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Explicit detached Desktop product source repository')
    parser.add_argument('--source-ref', help='Exact expected Desktop source commit (40 hex)')
    parser.add_argument('--source-tree', help='Exact expected Desktop source tree (40 hex)')
    parser.add_argument('--source-receipt', type=Path, help='Read and verify an existing source-bound controller receipt')
    parser.add_argument('--validate-source', action='store_true', help='Emit source-validation fields only; no Windows invocation')
    parser.add_argument('--legacy-source', action='store_true', help='Explicit incumbent mode; defaults --source to this Hermes checkout, without detached qualification')
    parser.add_argument('--thumbprint', default='', help='Explicit existing CurrentUser/My signing certificate')
    parser.add_argument('--windows-repo', help='Optional separate Git repository; defaults to --worktree')
    parser.add_argument('--worktree', default=r'D:\repo\hermes-build')
    parser.add_argument('--output', help='Fresh Windows output directory')
    parser.add_argument('--python', default='', help='Windows Python 3.11+ bootstrap executable')
    parser.add_argument('--signtool', default='')
    parser.add_argument('--sevenzip', default='')
    parser.add_argument('--timestamp-url', default='')
    parser.add_argument('--native-assets', action='store_true', help='Build signed same-family asset packages and external flat bundle; native qualification remains separate')
    parser.add_argument('--native-local-git', action='store_true', help='Restrict native Git to the owned D:\\repo\\hermes-build checkout')
    parser.add_argument('--wsl-linux-git', action='store_true', help='Use this WSL distribution and /usr/bin/git for native build Git operations')
    parser.add_argument('--reuse-owned-windows-cache', action='store_true', help='Use the owned native checkout cache under its distribution lock from a separate build worktree')
    parser.add_argument('--preseed-owned-worktree', action='store_true', help='Transfer the exact bundle with Linux Git into the already owned isolated Windows worktree')
    parser.add_argument('--previous-native-layout', default='', help='Windows path to retained layout-plan.json for stable native A to B file assignment')
    parser.add_argument('--unsigned-receipt', default='', help='Windows path to prior unsigned-build-receipt.json; skip build')
    parser.add_argument('--retry-version', default='', help='Explicitly retry the latest recorded Semantier version without advancing/resetting its counter')
    parser.add_argument('--acceptance-public-base', default='', help='Build the fixed Semantier Windows x64 acceptance family against this validated update origin')
    parser.add_argument('--preflight', action='store_true', help='Check source transfer, certificate and tools without building')
    parser.add_argument('--runtime-source', type=Path, help='Committed Runtime repository for parent composition staging')
    parser.add_argument('--stage-parent-composition', action='store_true', help='Freeze committed parent source into --source for review/commit; no build or signing')
    args = parser.parse_args()
    if args.stage_parent_composition:
        if args.source is None:
            parser.error('source composition requires explicit --source')
        if args.preflight or args.unsigned_receipt or args.retry_version or args.validate_source:
            parser.error('source composition cannot combine with build/re-sign modes')
        parent = args.runtime_source or RUNTIME_ROOT
        if parent is None:
            parser.error('--stage-parent-composition requires --runtime-source outside a Runtime checkout')
        print(stage_parent_composition(parent.resolve(), args.source.resolve()))
        return
    if args.previous_native_layout and not args.native_assets:
        parser.error('--previous-native-layout requires --native-assets')
    if args.native_assets and not args.python:
        parser.error('--native-assets requires an explicit Windows --python for the byte-bound producer')
    if args.preflight and args.unsigned_receipt:
        parser.error('cannot combine --preflight and --unsigned-receipt')
    if args.retry_version and (args.preflight or args.unsigned_receipt):
        parser.error('--retry-version requires a full build')
    if args.retry_version and not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', args.retry_version):
        parser.error('--retry-version must be a numeric x.y.z version')
    if args.validate_source and (args.preflight or args.unsigned_receipt or args.retry_version):
        parser.error('source validation cannot combine with packaging/preflight modes')
    if args.source is None and not args.legacy_source:
        parser.error('explicit --source required; incumbent callers must select --legacy-source')
    try:
        identity = validate_source(args.source or ROOT, expected_ref=args.source_ref,
            expected_tree=args.source_tree, receipt_path=args.source_receipt, legacy=args.legacy_source)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.validate_source:
        print(json.dumps(identity, sort_keys=True))
        return
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.thumbprint):
        parser.error('--thumbprint must be the full 40 hexadecimal characters')
    distro = os.environ.get('WSL_DISTRO_NAME', '') if args.wsl_linux_git else ''
    if args.wsl_linux_git and not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', distro):
        parser.error('--wsl-linux-git requires an explicit valid WSL_DISTRO_NAME')
    if args.reuse_owned_windows_cache and (not args.wsl_linux_git or args.native_local_git or
            (args.windows_repo or '').lower() != r'd:\repo\hermes-build' or
            args.worktree.lower() == r'd:\repo\hermes-build'):
        parser.error('--reuse-owned-windows-cache requires Linux Git, the owned Windows repository and a separate worktree')
    if args.preseed_owned_worktree and (not args.reuse_owned_windows_cache or args.unsigned_receipt or args.worktree.lower() != r'd:\repo\hermes-channel1-build'):
        parser.error('--preseed-owned-worktree requires the owned Channel1 worktree and shared-cache mode')
    source = args.source.resolve()
    commit = run('/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'rev-parse', 'HEAD')
    if run('/usr/bin/git', '-C', str(source), '--work-tree=' + str(source), 'status', '--porcelain', '--untracked-files=all'):
        parser.error('Hermes source must be committed and clean; no implicit stash/reset')
    controller = source / 'scripts/build-windows-distribution.ps1'
    parent_commit = run('/usr/bin/git', '-C', str(ROOT), '--work-tree=' + str(ROOT), 'rev-parse', 'HEAD')
    stamp = datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or fr'D:\repo\deliverables\SemantierAgent-{stamp}'
    config = dict(repository=args.windows_repo or args.worktree, worktree=args.worktree, output=output,
                  commit=commit, controllerCommit=parent_commit, thumbprint=args.thumbprint.upper(), python=args.python,
                  signTool=args.signtool, sevenZip=args.sevenzip, timestampUrl=args.timestamp_url,
                  nativeAssets=args.native_assets, localGitOnly=args.native_local_git, wslLinuxGitDistro=distro, reuseOwnedWindowsCache=args.reuse_owned_windows_cache, preseedOwnedWorktree=args.preseed_owned_worktree, previousNativeLayout=args.previous_native_layout, unsignedReceipt=args.unsigned_receipt, preflight=args.preflight, retryVersion=args.retry_version, acceptancePublicBase=args.acceptance_public_base)
    base = None if args.unsigned_receipt else transfer_base(source, args.windows_repo or args.worktree, commit)
    if args.unsigned_receipt:
        transfer = nullcontext((None, ''))
    else:
        transfer = source_bundle(source, commit, base=base) if base else source_bundle(source, commit)
    with tempfile.TemporaryDirectory(prefix='hermes-request-') as folder, transfer as (bundle, ref):
        if args.preseed_owned_worktree:
            preseed_owned_worktree(Path(run('wslpath', '-u', args.windows_repo)), Path(run('wslpath', '-u', args.worktree)), bundle, ref, commit)
        config.update(bundle=windows_path(bundle) if bundle else '', bundleRef=ref)
        config_path = Path(folder) / 'request.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        print(f'Hermes {commit}\nWindows output: {output}', flush=True)
        subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                        '-File', windows_path(ROOT / 'scripts/windows_build_environment.ps1'),
                        '-Controller', windows_path(controller), '-Config', windows_path(config_path)], check=True,
                       env=windows_process_environment())
    elapsed = time.monotonic() - started
    native_output = Path(run('wslpath', '-u', output))
    if native_output.is_absolute() and native_output.is_dir():
        result_file = native_output / 'result.json'
        artifact_source = json.loads(result_file.read_text(encoding='utf-8-sig'))['sourceCommit'] if result_file.exists() else commit
        (native_output / 'wsl-controller-total.json').write_text(json.dumps({
            'startedAt': produced_at, 'finishedAt': datetime.now(datetime.UTC).isoformat(),
            'elapsedSeconds': elapsed, 'sourceCommit': artifact_source, 'controllerSourceCommit': commit, 'parentControllerCommit': parent_commit, 'result': 'PASS',
            'mode': 'preflight' if args.preflight else ('packaging-only' if args.unsigned_receipt else 'full-build'),
            'retryVersion': args.retry_version,
            'sourceMode': identity['sourceMode'], 'sourceRoot': str(source),
            'controllerSourceTree': identity['sourceTree'],
            'sourceTree': identity['sourceTree'] if artifact_source == commit else None,
        }, indent=2), encoding='utf-8')
    print(f'Completed in {elapsed:.3f}s. Receipts and logs: {output}')


if __name__ == '__main__':
    main()
