"""Compose owned supplier overlays on a complete native catalogue."""
from copy import deepcopy


def compose_supplier_maps(base, suppliers, supplier, updates, *, excluded=()):
    if not isinstance(base, dict) or not base:
        raise ValueError('FULL_BASE_MAP_REQUIRED')
    owned = deepcopy(suppliers)
    for other, prices in owned.items():
        if other != supplier and (set(updates) | set(excluded)) & prices.keys():
            raise ValueError('SUPPLIER_OWNERSHIP_COLLISION')
    selected = owned.setdefault(supplier, {})
    for key in excluded:
        selected.pop(key, None)
    selected.update(deepcopy(updates))
    result = deepcopy(base)
    seen = set()
    for prices in owned.values():
        if seen & prices.keys():
            raise ValueError('SUPPLIER_OWNERSHIP_COLLISION')
        seen.update(prices)
        result.update(deepcopy(prices))
    return {'map': result, 'suppliers': owned}


def compose_projection(base, prior, supplier, projection, *, excluded_keys=()):
    import hashlib
    from billing.supplier_pricing_projection import canonical, digest

    if digest(base) != projection['base_sha256']:
        raise ValueError('BASE_HASH_MISMATCH')
    previous = deepcopy(prior.get('suppliers', {}))
    versions = deepcopy(prior.get('supplier_versions', {}))
    if not previous and prior.get('overlays'):
        legacy = 'alibaba:alibaba-model-studio-cn-beijing'
        previous[legacy] = {key: prior['map'][key] for key in prior['overlays']}
        if prior.get('snapshot_version'):
            versions[legacy] = prior['snapshot_version']
    updates = {key: projection['map'][key] for key in projection['overlays']}
    combined = compose_supplier_maps(base, previous, supplier, updates, excluded=excluded_keys)
    body = canonical(combined['map'])
    result = deepcopy(projection)
    result.update(combined)
    result['supplier_versions'] = {**versions, supplier: projection['snapshot_version']}
    result['sha256'] = hashlib.sha256(body).hexdigest()
    result['native_revision'] = hashlib.sha1(b'blob %d\0' % len(body) + body, usedforsecurity=False).hexdigest()
    return result
