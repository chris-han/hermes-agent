"""Verify bounded product source and imported helper identities."""
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def verify_product(product: Path):
    """Verify the bounded parent-owned product inputs before a consumer uses them."""
    if product.is_symlink():
        raise ValueError('product composition integrity mismatch: linked root')
    product = product.resolve(strict=True)
    receipt = product / 'source-receipt.json'
    if receipt.is_symlink():
        raise ValueError('product composition integrity mismatch: linked receipt')
    manifest = json.loads(receipt.read_text(encoding='utf-8-sig'))
    if (manifest.get('schema') != 1 or not re.fullmatch('[0-9a-f]{40}', str(manifest.get('source_commit', '')))
            or not isinstance(manifest.get('files'), list) or not manifest['files']):
        raise ValueError('product composition integrity mismatch: invalid receipt')
    seen = set()
    for entry in manifest['files']:
        key = entry['path']
        relative = PurePosixPath(key)
        path = product.joinpath(*relative.parts)
        if (key in seen or relative.is_absolute() or '\\' in key
                or any(part in {'.', '..'} for part in relative.parts)
                or any(part.is_symlink() for part in (path, *path.parents) if part != product.parent)
                or not path.resolve().is_relative_to(product) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']):
            raise ValueError('product composition integrity mismatch: ' + key)
        seen.add(key)
    return manifest


def verify_product_modules(product: Path, names: tuple[str, ...]):
    """Bind the actual importable product helpers to the verified source receipt."""
    import importlib.util
    import sys

    manifest = verify_product(product)
    expected = {entry['path']: entry['sha256'] for entry in manifest['files']}
    spec = importlib.util.find_spec('semantier_product')
    if spec is None or spec.origin is None:
        raise ValueError('installed product package unavailable')
    root = Path(spec.origin).parent
    for name in names:
        if not re.fullmatch('[a-z_]+', name):
            raise ValueError('invalid installed product module name')
        relative = f'semantier_product/{name}.py'
        path = root / f'{name}.py'
        loaded = sys.modules.get(f'semantier_product.{name}')
        if (relative not in expected or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected[relative]
                or (loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path.resolve())):
            raise ValueError('installed product module does not match receipt: ' + name)
    return manifest
