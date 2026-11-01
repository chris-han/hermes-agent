"""Independent price and ownership controls for native replay expectations."""
from billing.supplier_pricing_projection import digest


def registration_overlays(base, snapshot, expected, capabilities=None):
    """Return owned expected deltas, or None when their inputs cannot justify them."""
    if not all(isinstance(x, dict) for x in (base, snapshot, expected)) or snapshot.get('replay_order') != ['router', 'durable']:
        return None
    rows, durable = snapshot.get('registrations'), snapshot.get('durable_registrations')
    if not isinstance(rows, list) or not isinstance(durable, list): return None
    by_id, durable_by_key, keys = {}, {}, set(base)
    try:
        for row in rows:
            key, model = row['id'], row['model']
            if not all(isinstance(v, str) and v for v in (key, model, row['alias'])) or key in by_id or key in base:
                return None
            if not isinstance(row['metadata'], dict) or not isinstance(row['explicit_parameters'], dict): return None
            by_id[key] = row
            keys.update((key, model))
            if row.get('provider'): keys.add(row['provider'] + '/' + model)
        for row in durable:
            key = row['key']
            if not isinstance(key, str) or not key or key in durable_by_key or not isinstance(row['metadata'], dict): return None
            durable_by_key[key] = row['metadata']
            keys.add(key)
        if set(expected) - keys or set(base) - set(expected): return None
        for key, row in by_id.items():
            actual = expected.get(key)
            if not isinstance(actual, dict): return None
            for field, value in {**row['metadata'], **row['explicit_parameters'], **durable_by_key.get(key, {})}.items():
                if actual.get(field, object()) != value: return None
        for key, fields in durable_by_key.items():
            if not isinstance(expected.get(key), dict) or any(expected[key].get(f, object()) != v for f, v in fields.items()): return None
        for key, fields in base.items():
            if not isinstance(fields, dict) or not isinstance(expected.get(key), dict): return None
            for field, value in fields.items():
                actual = expected[key].get(field, object())
                if actual != value and durable_by_key.get(key, {}).get(field, object()) != actual: return None
        changes, claims = {}, {}
        for key, fields in expected.items():
            if not isinstance(fields, dict): return None
            previous = base.get(key, {})
            delta = {f: v for f, v in fields.items() if f not in previous or previous[f] != v}
            if not delta: continue
            changes[key] = delta
            for field, value in delta.items():
                price = 'cost' in field.lower() or 'pric' in field.lower()
                changed = field in previous and previous[field] != value
                if price and changed and value is None:
                    return None
                if price and value is not None:
                    sources = []
                    if durable_by_key.get(key, {}).get(field, object()) == value: sources.append('durable')
                    if key in by_id:
                        row = by_id[key]
                        if row['explicit_parameters'].get(field, object()) == value: sources.append('router')
                        for backend in (row['model'], (row.get('provider') or '') + '/' + row['model']):
                            if isinstance(base.get(backend), dict) and base[backend].get(field, object()) == value: sources.append('base')
                    if not sources: return None
                elif changed and durable_by_key.get(key, {}).get(field, object()) != value:
                    return None
                elif not price and value is not None:
                    relevant = [row for row in rows if key in (row['id'], row['model'], (row.get('provider') or '') + '/' + row['model'])]
                    justified = durable_by_key.get(key, {}).get(field, object()) == value
                    for row in relevant:
                        physical_key = key != row['id']
                        justified = justified or row['metadata'].get(field, object()) == value or row['explicit_parameters'].get(field, object()) == value
                        if key == row['id'] and isinstance(base.get(row['model']), dict):
                            justified = justified or base[row['model']].get(field, object()) == value
                        if physical_key and field == 'key' and value == key:
                            justified = True
                        if physical_key and field == 'supported_openai_params' and isinstance(capabilities, dict) and capabilities.get(row['model'], object()) == value:
                            justified = True
                        if physical_key and field == 'litellm_provider' and value in (row.get('provider'), row['model'].split('/', 1)[0]):
                            justified = True
                    if not justified: return None
                if changed or (price and value is not None):
                    claims.setdefault(key, {})[field] = {'value': value, 'owner': 'native-registration', 'revision': digest(snapshot)}
        return changes, claims
    except (KeyError, TypeError, ValueError):
        return None
