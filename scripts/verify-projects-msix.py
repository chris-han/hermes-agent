"""Inspect the compiled local-mode Projects entry in a Windows release package."""
import argparse
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile

p = argparse.ArgumentParser()
p.add_argument('package')
p.add_argument('--require-version')
p.add_argument('--require-mount', action='store_true')
args = p.parse_args()
with ZipFile(args.package) as msix:
    identity = ET.fromstring(msix.read('AppxManifest.xml')).find('{*}Identity').attrib
    assert identity['Name'] == 'Semantier.Agent.Acceptance'
    assert identity['ProcessorArchitecture'] == 'x64'
    if args.require_version:
        assert identity['Version'] == args.require_version
    payload = msix.read('app/resources/app.asar')
    size = struct.unpack_from('<I', payload, 4)[0]
    header_length = struct.unpack_from('<I', payload, 12)[0]
    header = json.loads(payload[16:16 + header_length])
    files = []
    def walk(dir, entry):
        for name, meta in entry.get('files', {}).items():
            path = f'{dir}/{name}' if dir else name
            if 'files' in meta:
                walk(path, meta)
            elif re.search(r'/gateway-settings-[^/]+\.js$', path):
                if meta.get('unpacked'):
                    asset = msix.read('app/resources/app.asar.unpacked/' + path)
                else:
                    offset = 8 + size + int(meta['offset'])
                    asset = payload[offset:offset + meta['size']]
                assert len(asset) == meta['size']
                if b'Use your Workspace account to continue a project.' not in asset:
                    continue
                mount = re.search(rb'connections:\s*[^,{}]{1,60},\s*projectsOnly:\s*(?:!0|true)\s*}', asset)
                files.append({'asset': path, 'asset_sha256': hashlib.sha256(asset).hexdigest(),
                              'project_only_mount': mount is not None})
    walk('', header)
    assert len(files) == 1, files
    proof = dict(version=identity['Version'], asar_sha256=hashlib.sha256(payload).hexdigest(), **files[0])
    print(json.dumps(proof))
    if args.require_mount and not proof['project_only_mount']:
        raise SystemExit('Project-only Settings mount absent')
