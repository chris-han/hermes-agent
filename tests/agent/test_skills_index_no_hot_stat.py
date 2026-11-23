"""Warm Skills Index cache hits must not validate file signatures on every model turn."""

from agent import prompt_builder as pb


def test_warm_skills_index_avoids_manifest_walk(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill_file = root / "example" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: example\ndescription: Example.\n---\n", encoding="utf-8")
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        first = pb.build_skills_system_prompt(available_tools={"skill_view"}, skills_dir_override=root)
        assert "example" in first

        def reject_signature_scan(*args, **kwargs):
            raise AssertionError("warm cache hit must not stat every skill")

        monkeypatch.setattr(pb, "_build_skills_manifest", reject_signature_scan)
        assert pb.build_skills_system_prompt(
            available_tools={"skill_view"}, skills_dir_override=root,
        ) == first
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)
