"""Exact-version authored teaching annotations, never policy authority or detection."""
from copy import deepcopy
import json
from pathlib import Path
from semantier.sensitive_governance_contracts.hashing import canonical_hash


def load_fixture() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / 'catalogs/workshop/finance-sensitive-demo.v1.json').read_text())


def bind_finance_demo(*, source_hash: str, session_id: str, filename: str) -> dict:
    fixture = load_fixture()
    if source_hash != fixture['source_hash']:
        raise ValueError('FINANCE_DEMO_SOURCE_VERSION_MISMATCH: 此文件不是预置标注对应的财务简报版本')
    rule_ids = {r['id'] for r in fixture['rules']}
    for finding in fixture['findings']:
        loc = finding.get('presentation_location') or {}
        if (finding.get('rule_id') not in rule_ids or loc.get('kind') != 'pdf_page_rect'
            or loc.get('page') not in (1, 2) or not finding.get('quote')
            or any(not isinstance(loc.get(k), (int, float)) for k in ('x','y','width','height'))
            or min(loc['x'],loc['y']) < 0 or min(loc['width'],loc['height']) <= 0):
            raise ValueError('FINANCE_DEMO_ANCHOR_OR_RULE_UNAVAILABLE')
    result = deepcopy(fixture)
    result['fixture_hash'] = 'sha256:' + canonical_hash(fixture)
    result['source_identity_ref'] = f'session-upload:{session_id}:{source_hash}'
    result['document_name'] = filename
    result['session_id'] = session_id
    nodes, edges = [], []
    for finding in result['findings']:
        fid, rid, eid = finding['id'], finding['rule_id'], finding['evidence_ref']
        rule = next(r for r in result['rules'] if r['id'] == rid)
        nodes.extend([{'id': fid, 'kind': 'finding', 'label': finding['label']},
                      {'id': rid, 'kind': 'rule', 'label': '教学规则：'+rule['label']},
                      {'id': eid, 'kind': 'evidence', 'label': f"原文第{finding['presentation_location']['page']}页"},
                      {'id': rid+':action', 'kind': 'action', 'label': rule['recommendation']}])
        edges.extend([{'source':fid,'target':rid,'kind':'explained_by'},
                      {'source':fid,'target':eid,'kind':'supported_by'},
                      {'source':rid,'target':rid+':action','kind':'recommends'}])
    result['graph'] = {'nodes':nodes,'edges':edges}
    result['result_hash'] = 'sha256:' + canonical_hash(result)
    return result
