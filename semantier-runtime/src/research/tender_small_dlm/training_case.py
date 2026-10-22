"""Frozen group split; every descendant inherits the parent's assignment."""
from hashlib import sha256


def group_split(parents):
    groups=[p['source_group_id'] for p in parents]
    if len(groups)!=19 or len(set(groups))!=19:
        raise ValueError('EXACT_UNIQUE_B1_SOURCE_GROUPS_REQUIRED')
    result={group:'TRAIN' for group in groups}
    for label,count in [('A',2),('B',3)]:
        eligible=[p['source_group_id'] for p in parents if p['gold']==label]
        if len(eligible)<=count:raise ValueError('CLASS_COVERAGE_REQUIRED')
        ordered=sorted(eligible,key=lambda group:sha256(('17:'+group).encode()).hexdigest())
        for group in ordered[:count]:result[group]='DEV'
    return result
