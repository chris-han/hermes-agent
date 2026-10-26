"""Verify the generated parent-owned snapshot without rewriting its source."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(parent: Path):
    parent = parent.resolve()
    manifest = json.loads((parent / 'composition.json').read_text(encoding='utf-8'))
    seen = set()
    for entry in manifest['files']:
        key = entry['path']
        relative = PurePosixPath(key)
        path = parent.joinpath(*relative.parts)
        if (key in seen or relative.is_absolute() or '\\' in key
                or any(p in {'.', '..'} for p in relative.parts)
                or path.is_symlink() or not path.resolve().is_relative_to(parent)
                or any(p.is_symlink() for p in path.parents if p != parent.parent)
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
            raise ValueError('parent composition integrity mismatch: ' + key)
        seen.add(key)
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parent', type=Path)
    args = parser.parse_args()
    result = verify(args.parent)
    print(json.dumps({'parent_commit': result['parent_commit'], 'verified_files': len(result['files'])}))
