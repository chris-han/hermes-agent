"""Admit an interrupted native Windows build using its existing reservation."""

import re
from pathlib import Path

from hermes_cli.release_channels import decode_json


def admit_windows_allocation(output, allocation_path, ledger_path, acceptance_path, ui_path, manifest):
    """Require matching allocator, signed-package and actual Windows acceptance evidence."""
    def read(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Unsafe Windows release evidence')
        return decode_json(path.read_bytes())

    allocation = read(allocation_path)
    ledger = read(ledger_path)
    acceptance = read(acceptance_path)
    ui = read(ui_path)
    original = read(Path(allocation_path).with_name('channel-request.json'))
    request = manifest['request']
    version = allocation.get('version')
    if (allocation != ledger or allocation.get('kind') != 'semantier-windows-build-version-v1'
            or not re.fullmatch(r'0\.0\.[1-9][0-9]*', str(version))
            or not allocation.get('reservedAt')
            or not re.fullmatch('[a-f0-9]{40}', str(allocation.get('sourceCommit')))
            or request.get('commit') != allocation['sourceCommit']
            or request.get('version') != version or request.get('windowsVersion') != version + '.0'
            or request.get('sequence') != int(version.split('.')[2])):
        raise ValueError('Windows allocator source or version mismatch')
    # Channel aliases change delivery metadata only; compiled source remains bound
    # to the original request and the exact same signed installation closure.
    if (request.get('publicBase') not in ('https://download.semantier.com', 'https://download.semantier.cn')
            or {k: v for k, v in request.items() if k != 'publicBase'}
            != {k: v for k, v in original.items() if k != 'publicBase'}):
        raise ValueError('Windows delivery projection differs from original source request')
    package = manifest['packages'][0]
    root_hash = package['artifact']['sha256']
    if (acceptance.get('result') != 'LOCAL_PACKAGE_REPAIR_PASS'
            or acceptance.get('version') != request['windowsVersion']
            or acceptance.get('sourceCommit') != request['commit']
            or acceptance.get('artifactSha256') != root_hash
            or any(acceptance.get(k) is not True for k in ('profilePreserved', 'sessionsPreserved'))
            or any(acceptance.get(k) != 'PASS' for k in ('window', 'pty'))
            or acceptance.get('moduleErrors') != []):
        raise ValueError('Windows installed package acceptance mismatch')
    expected = sorted((row['key'].rsplit('/', 1)[1], row['size'], row['sha256'])
                      for row in manifest['installation']['objects'])
    observed = sorted((row['name'], row['size'], row['sha256'])
                      for row in acceptance.get('verifiedPackages', []))
    if expected != observed:
        raise ValueError('Accepted signed package closure differs')
    binding = ui.get('binding', {})
    if (ui.get('result') != 'WINDOW_PTY_PASS' or ui.get('backendReady') is not True
            or ui.get('windowOpened') is not True or ui.get('errors') != []
            or binding.get('artifactSha256') != root_hash
            or binding.get('sourceCommit') != request['commit'] or binding.get('version') != version
            or ui.get('identity', {}).get('packaged') is not True
            or ui.get('identity', {}).get('platform') != 'win32'):
        raise ValueError('Windows application/backend acceptance mismatch')
    signed = read(Path(output) / 'signed/receipt.json')
    if signed.get('signedSha256') != root_hash or signed.get('sourceCommit') != request['commit']:
        raise ValueError('Windows signed receipt mismatch')
    return True
