"""Native channel active/archive transition, independent of Runtime and databases."""
from copy import deepcopy


def advance(record, new_head):
    """Return a new record; callers must commit it with the observed ETag."""
    old = record.get('nativeHead')
    if old == new_head:
        return deepcopy(record)
    if (type(new_head.get('sequence')) is not int
            or new_head['sequence'] >= record['nextSequence']
            or (old is not None and new_head['sequence'] <= old['sequence'])):
        raise ValueError('native release sequence is stale or unallocated')
    result = deepcopy(record)
    result['nativeArchive'] = deepcopy(old)
    result['nativeHead'] = deepcopy(new_head)
    result['revision'] += 1
    return result
