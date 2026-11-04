"""Inspect the actual MSIX Electron ASAR and node-pty Windows-x64 closure."""
import json
import struct
import zipfile
from pathlib import PurePosixPath

BASE = 'dist/node_modules/node-pty/'
REQUIRED = ('package.json', 'lib/index.js', 'prebuilds/win32-x64/pty.node',
            'prebuilds/win32-x64/conpty.node', 'prebuilds/win32-x64/conpty_console_list.node',
            'prebuilds/win32-x64/winpty-agent.exe', 'prebuilds/win32-x64/winpty.dll',
            'prebuilds/win32-x64/conpty/OpenConsole.exe', 'prebuilds/win32-x64/conpty/conpty.dll')


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate MSIX entries')
        with archive.open('app/resources/app.asar') as stream:
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
                try: body = archive.read('app/resources/app.asar.unpacked/' + name)
                except KeyError as error: raise ValueError('missing unpacked ASAR file: ' + name) from error
            else:
                offset = item.get('offset')
                if not isinstance(offset, str) or not offset.isdigit():
                    raise ValueError('invalid ASAR offset: ' + name)
                with archive.open('app/resources/app.asar') as stream:
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
        if package.get('name') != 'node-pty' or package.get('main') not in ('lib/index.js', './lib/index.js'):
            raise ValueError('node-pty package identity or entrypoint mismatch')
        return {'nodePtyFiles': len(REQUIRED), 'layout': 'asar-and-unpacked', 'architecture': 'win32-x64'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('msix')
    args = parser.parse_args()
    print(json.dumps(inspect(args.msix), sort_keys=True))
