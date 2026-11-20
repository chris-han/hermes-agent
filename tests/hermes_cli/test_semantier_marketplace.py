"""Native HTTP discovery uses exact source and preserves partial failures."""
import asyncio
from tools.skills_hub_models import SkillMeta
from hermes_cli.web_routers import skills


class Source:
    def __init__(self, sid, count=1, error=None):
        self.sid, self.count, self.error = sid, count, error
    def source_id(self):
        return self.sid
    def search(self, query, limit=10):
        if self.error:
            raise self.error
        return [SkillMeta('Same name', query, self.sid, f'{self.sid}/{i:03}', 'community') for i in range(min(limit, self.count))]


def test_exact_empty_listing_and_fair_cap(monkeypatch):
    sources = [Source('semantier'), Source('official', 80), Source('browse-sh', 80)]
    monkeypatch.setattr(skills, '_hub_sources', lambda _: sources)
    monkeypatch.setattr(skills, '_installed_hub_identifiers', lambda _: {})
    selected = asyncio.run(skills.search_skills_hub(q='', source='semantier'))
    assert {r['source'] for r in selected['results']} == {'semantier'}
    all_rows = asyncio.run(skills.search_skills_hub(q='demo', source='all', limit=100))
    assert len(all_rows['results']) == 50
    assert {r['source'] for r in all_rows['results']} == {'semantier', 'official', 'browse-sh'}


def test_provider_failure_and_spoof_are_scoped(monkeypatch):
    monkeypatch.setattr(skills, '_hub_sources', lambda _: [Source('semantier', error=ValueError('bad registry')), Source('official')])
    monkeypatch.setattr(skills, '_installed_hub_identifiers', lambda _: {})
    out = asyncio.run(skills.search_skills_hub(q='demo'))
    assert out['errors'][0]['source'] == 'semantier'
    assert out['results'][0]['source'] == 'official'
    forged = SkillMeta('Semantier', '', 'url', 'https://example.org/SKILL.md', 'community', extra={'publisher_verified': True, 'publisher': 'Semantier'})
    assert 'publisher_verified' not in skills._skill_meta_to_payload(forged)
    forged.source = 'semantier'
    assert 'publisher_verified' not in skills._skill_meta_to_payload(forged)


def test_fair_merge_does_not_depend_on_completion_order():
    sources = [Source('semantier'), Source('official', 80)]
    semantier = sources[0].search('same', 50)
    nous = sources[1].search('same', 50)
    a = skills._fair_results(nous + semantier, sources, 50)
    b = skills._fair_results(semantier + nous, sources, 50)
    assert [r.identifier for r in a] == [r.identifier for r in b]
    assert a[0].source == 'semantier'
