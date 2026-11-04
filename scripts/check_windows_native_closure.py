"""Inspect the actual MSIX Electron ASAR and node-pty Windows-x64 closure."""
import json
from email.parser import BytesParser
from contextlib import ExitStack
import struct
import zipfile
from pathlib import Path, PurePosixPath

BASE = 'dist/node_modules/node-pty/'
REQUIRED = ('package.json', 'lib/index.js', 'prebuilds/win32-x64/pty.node',
            'prebuilds/win32-x64/conpty.node', 'prebuilds/win32-x64/conpty_console_list.node',
            'prebuilds/win32-x64/winpty-agent.exe', 'prebuilds/win32-x64/winpty.dll',
            'prebuilds/win32-x64/conpty/OpenConsole.exe', 'prebuilds/win32-x64/conpty/conpty.dll')


def inspect(path, prepared_pty=None, packages=()):
    with ExitStack() as stack:
        archives = [stack.enter_context(zipfile.ZipFile(p)) for p in (path, *packages)]
        entries = {}
        for archive in archives:
            for name in archive.namelist():
                if not name.startswith('app/'):
                    continue
                if name in entries: raise ValueError('duplicate MSIX app entry: ' + name)
                entries[name] = archive
        if 'app/resources/app.asar' not in entries: raise ValueError('Electron ASAR missing')
        asar_archive = entries['app/resources/app.asar']
        with asar_archive.open('app/resources/app.asar') as stream:
            prefix = stream.read(16)
            if len(prefix) != 16:
                raise ValueError('truncated ASAR header')
            marker, header_size, pickle_size, length = struct.unpack('<4I', prefix)
            if marker != 4 or length <= 0 or length > 32 * 1024 * 1024 or pickle_size != header_size-4 or length > pickle_size-4:
                raise ValueError('invalid ASAR header')
            data = stream.read(length)
            if len(data) != length:
                raise ValueError('truncated ASAR index')
        index = json.loads(data)
        def read(name):
            item = index
            for part in PurePosixPath(name).parts:
                item = item.get('files', {}).get(part)
                if not isinstance(item, dict):
                    raise ValueError('missing ASAR path: ' + name)
            size = item.get('size')
            if item.get('link') or type(size) is not int or size <= 0 or size > 64 * 1024 * 1024:
                raise ValueError('invalid ASAR file: ' + name)
            if item.get('unpacked'):
                try:
                    key = 'app/resources/app.asar.unpacked/' + name
                    body = entries[key].read(key)
                except KeyError as error: raise ValueError('missing unpacked ASAR file: ' + name) from error
            else:
                offset = item.get('offset')
                if not isinstance(offset, str) or not offset.isdigit():
                    raise ValueError('invalid ASAR offset: ' + name)
                with asar_archive.open('app/resources/app.asar') as stream:
                    stream.seek(8 + header_size + int(offset))
                    body = stream.read(size)
            if len(body) != size:
                raise ValueError('ASAR size mismatch: ' + name)
            return body, bool(item.get('unpacked'))
        main, _ = read('dist/electron-main.mjs')
        if b'node-pty' not in main:
            raise ValueError('Electron main does not reference node-pty')
        objects = {}
        for suffix in REQUIRED:
            body, unpacked = read(BASE + suffix)
            objects[suffix] = body
            if suffix.endswith(('.node', '.dll', '.exe')):
                if not unpacked or len(body) < 64 or body[:2] != b'MZ':
                    raise ValueError('native Windows binary is invalid or packed: ' + suffix)
                offset = struct.unpack_from('<I', body, 60)[0]
                if offset + 6 > len(body) or body[offset:offset+4] != b'PE\0\0' or struct.unpack_from('<H', body, offset+4)[0] != 0x8664:
                    raise ValueError('native binary is not Windows x64: ' + suffix)
        package = json.loads(objects['package.json'])
        if package.get('name') != 'node-pty' or package.get('version') != '1.1.0' or package.get('main') not in ('lib/index.js', './lib/index.js'):
            raise ValueError('node-pty package identity or entrypoint mismatch')
        count = len(REQUIRED)
        if prepared_pty is not None:
            root = Path(prepared_pty)
            if root.is_symlink() or not root.is_dir():
                raise ValueError('prepared node-pty tree missing')
            count = 0
            for source in sorted(root.rglob('*')):
                if source.is_symlink(): raise ValueError('redirected prepared node-pty file')
                if not source.is_file(): continue
                relative = source.relative_to(root).as_posix()
                body, _ = read(BASE + relative)
                # Native PEs may be Authenticode-signed during electron-builder's
                # afterPack step. JavaScript and package metadata must match exactly.
                if source.suffix.lower() not in ('.node', '.dll', '.exe') and body != source.read_bytes():
                    raise ValueError('prepared node-pty content mismatch: ' + relative)
                count += 1
            if count < len(REQUIRED): raise ValueError('prepared node-pty tree incomplete')
        # This product config defaults Windows Cloud connections to HTTP/3.
        # Reject packages which would otherwise install successfully but fail at sign-in.
        site = 'app/resources/agent-payload/venv/Lib/site-packages/'
        required = ('curl_cffi/__init__.py', 'curl_cffi/requests/__init__.py',
                    'curl_cffi/_wrapper.pyd', '_cffi_backend.cp314-win_amd64.pyd',
                    'certifi/cacert.pem', 'curl_cffi-0.16.0.dist-info/METADATA')
        for relative in required:
            if site + relative not in entries:
                raise ValueError('HTTP/3 runtime file missing: ' + relative)
        dll_prefix = site + 'curl_cffi.libs/libcurl-impersonate-'
        dlls = [name for name in entries if name.startswith(dll_prefix) and name.endswith('.dll')]
        if len(dlls) != 1:
            raise ValueError('HTTP/3 native DLL missing or ambiguous')
        metadata = BytesParser().parsebytes(entries[site + required[-1]].read(site + required[-1]))
        if metadata.get('Name', '').lower().replace('_', '-') != 'curl-cffi' or metadata.get('Version') != '0.16.0':
            raise ValueError('HTTP/3 runtime version mismatch')
        for key in (site + 'curl_cffi/_wrapper.pyd', site + '_cffi_backend.cp314-win_amd64.pyd', dlls[0]):
            archive = entries[key]
            if archive.getinfo(key).file_size > 64 * 1024 * 1024:
                raise ValueError('HTTP/3 native binary exceeds size limit')
            body = archive.read(key)
            if len(body) < 64 or body[:2] != b'MZ':
                raise ValueError('HTTP/3 native binary is invalid')
            offset = struct.unpack_from('<I', body, 60)[0]
            if offset + 6 > len(body) or body[offset:offset+4] != b'PE\0\0' or struct.unpack_from('<H', body, offset+4)[0] != 0x8664:
                raise ValueError('HTTP/3 native binary is not Windows x64')
        return {'nodePtyFiles': count, 'layout': 'asar-and-unpacked', 'architecture': 'win32-x64', 'http3': 'present'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('msix')
    parser.add_argument('--prepared-pty', required=True)
    parser.add_argument('--package', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(inspect(args.msix, args.prepared_pty, args.package), sort_keys=True))
