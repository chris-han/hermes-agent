"""Conservative finalization of the complete retained physical-attempt evidence."""


def final_native_observation(receipts,expected_provider=None):
    if not receipts:
        return None
    if any(r['physical_send_count']!=1 for r in receipts):
        return None
    if receipts[-1]['completion_state']!='FINAL':
        return None
    finals=[r for r in receipts if r['completion_state']=='FINAL']
    if not finals:
        return None
    last=finals[-1]
    if expected_provider and last['provider']!=expected_provider:
        return None
    if any(r['provider']!=last['provider'] or r['raw_usage']!=last['raw_usage'] for r in finals):
        return None
    return last
