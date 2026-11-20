"""Skill selection must retain the declared task triggers through index snapshots."""
import json
from agent import prompt_builder as pb


def test_task_triggers_survive_cold_and_legacy_snapshot_indexes(tmp_path):
    root = tmp_path / 'skills'
    skill = root / 'reasoning-guide'
    skill.mkdir(parents=True)
    description = ('Use when a user asks to critically examine a belief, evidence, interpretation, '
                   'consequential choice, proposal, argument, or draft. Route decision analysis and '
                   '企业采购决策 by the requested deliverable; do not activate for routine factual answers.')
    (skill / 'SKILL.md').write_text('---\nname: reasoning-guide\ndescription: ' + description + '\n---\nFollow the selected reasoning workflow.\n')
    first = pb.build_skills_system_prompt(available_tools={'skill_view'}, skills_dir_override=root)
    assert description in first
    snapshot = root.parent / '.skills_prompt_snapshot.json'
    data = json.loads(snapshot.read_text())
    data['version'] = 4
    for row in data['skills']:
        row['description'] = description[:57] + '...'
    snapshot.write_text(json.dumps(data))
    pb.clear_skills_system_prompt_cache()
    restored = pb.build_skills_system_prompt(available_tools={'skill_view'}, skills_dir_override=root)
    assert description in restored
