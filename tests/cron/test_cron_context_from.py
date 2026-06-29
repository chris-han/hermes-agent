"""Tests for cron job context_from feature (issue #5439 Option C)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestJobContextFromField:
    """Test that context_from is stored and retrieved correctly."""

    def test_create_job_with_context_from_string(self, cron_env):
        from cron.jobs import create_job, get_job

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize findings",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        assert job_b["context_from"] == [job_a["id"]]
        loaded = get_job(job_b["id"])
        assert loaded["context_from"] == [job_a["id"]]

    def test_create_job_with_context_from_list(self, cron_env):
        from cron.jobs import create_job, get_job

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Find weather", schedule="every 1h")
        job_c = create_job(
            prompt="Summarize everything",
            schedule="every 2h",
            context_from=[job_a["id"], job_b["id"]],
        )

        assert job_c["context_from"] == [job_a["id"], job_b["id"]]

    def test_create_job_without_context_from(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("context_from") is None

    def test_context_from_empty_string_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", context_from="")
        assert job.get("context_from") is None

    def test_context_from_empty_list_normalized_to_none(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="Hello", schedule="every 1h", context_from=[])
        assert job.get("context_from") is None


class TestBuildJobPromptContextFrom:
    """Test that _build_job_prompt() injects context from referenced jobs."""

    def test_injects_latest_output(self, cron_env):
        from cron.jobs import create_job, save_job_output_record
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        save_job_output_record(job_a["id"], "Today's top story: AI is everywhere.")

        job_b = create_job(
            prompt="Summarize the news",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "Today's top story: AI is everywhere." in prompt
        assert f"Output from job '{job_a['id']}'" in prompt

    def test_injects_latest_sqlite_output_when_no_markdown_artifact(self, cron_env):
        from cron.jobs import create_job, output_dir, save_job_output_record
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        save_job_output_record(job_a["id"], "SQLite-only continuity context")

        assert not (output_dir() / job_a["id"]).exists()

        job_b = create_job(
            prompt="Summarize the news",
            schedule="every 2h",
            context_from=job_a["id"],
        )

        prompt = _build_job_prompt(job_b)
        assert "SQLite-only continuity context" in prompt
        assert f"Output from job '{job_a['id']}'" in prompt

    def test_uses_most_recent_output(self, cron_env):
        from cron.jobs import create_job, save_job_output_record
        from cron.scheduler import _build_job_prompt
        import time

        job_a = create_job(prompt="Find news", schedule="every 1h")
        save_job_output_record(job_a["id"], "Old output")
        time.sleep(0.01)
        save_job_output_record(job_a["id"], "New output")

        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)
        assert "New output" in prompt
        assert "Old output" not in prompt

    def test_raises_when_sqlite_output_missing(self, cron_env):
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"]
        )

        with pytest.raises(RuntimeError, match="has no SQLite output"):
            _build_job_prompt(job_b)

    def test_injects_multiple_context_jobs(self, cron_env):
        from cron.jobs import create_job, save_job_output_record
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Find weather", schedule="every 1h")

        for job, content in [(job_a, "News: AI boom"), (job_b, "Weather: Sunny")]:
            save_job_output_record(job["id"], content)

        job_c = create_job(
            prompt="Daily briefing",
            schedule="every 2h",
            context_from=[job_a["id"], job_b["id"]],
        )
        prompt = _build_job_prompt(job_c)
        assert "News: AI boom" in prompt
        assert "Weather: Sunny" in prompt

    def test_context_injected_before_prompt(self, cron_env):
        """Context should appear before the job's own prompt."""
        from cron.jobs import create_job, save_job_output_record
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        save_job_output_record(job_a["id"], "Context data")

        job_b = create_job(
            prompt="Process the data above",
            schedule="every 2h",
            context_from=job_a["id"],
        )
        prompt = _build_job_prompt(job_b)
        context_pos = prompt.find("Context data")
        prompt_pos = prompt.find("Process the data above")
        assert context_pos < prompt_pos

    def test_output_truncated_at_8k_chars(self, cron_env):
        """Output longer than 8000 chars should be truncated."""
        from cron.jobs import create_job, save_job_output_record
        from cron.scheduler import _build_job_prompt

        job_a = create_job(prompt="Find data", schedule="every 1h")
        big_output = "x" * 10000
        save_job_output_record(job_a["id"], big_output)

        job_b = create_job(
            prompt="Process", schedule="every 2h", context_from=job_a["id"]
        )
        prompt = _build_job_prompt(job_b)
        assert "truncated" in prompt
        assert "x" * 10000 not in prompt

    def test_invalid_job_id_skipped(self, cron_env):
        """context_from with path traversal job_id should be skipped."""
        from cron.jobs import create_job
        from cron.scheduler import _build_job_prompt

        job = create_job(prompt="Process", schedule="every 2h")
        # Manually inject invalid context_from (simulating tampered jobs.json)
        job["context_from"] = ["../../../etc/passwd"]
        prompt = _build_job_prompt(job)
        # Should not crash and should not inject anything malicious
        assert "Process" in prompt
        assert "etc/passwd" not in prompt



class TestUpdateContextFrom:
    """Verify the cronjob tool's `update` action wires context_from through.

    Without this, the create-path stores the field but users can never modify
    or clear it via the tool (schema promises "pass an empty array to clear").
    """

    def test_update_adds_context_from_to_existing_job(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(prompt="Summarize", schedule="every 2h")
        assert job_b.get("context_from") is None

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=job_a["id"],
        ))
        assert result["success"] is True

        reloaded = get_job(job_b["id"])
        assert reloaded["context_from"] == [job_a["id"]]

    def test_update_changes_context_from_reference(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_a2 = create_job(prompt="Find weather", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )
        assert job_b["context_from"] == [job_a["id"]]

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=[job_a2["id"]],
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] == [job_a2["id"]]

    def test_update_clears_context_from_with_empty_list(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )
        assert get_job(job_b["id"])["context_from"] == [job_a["id"]]

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=[],
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] is None

    def test_update_clears_context_from_with_empty_string(self, cron_env):
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from="",
        ))
        assert result["success"] is True
        assert get_job(job_b["id"])["context_from"] is None

    def test_update_rejects_unknown_job_reference(self, cron_env):
        from cron.jobs import create_job
        from tools.cronjob_tools import cronjob
        import json

        job_b = create_job(prompt="Summarize", schedule="every 2h")

        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            context_from=["deadbeef0000"],
        ))
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_update_preserves_context_from_when_not_passed(self, cron_env):
        """Updating other fields must not clobber context_from."""
        from cron.jobs import create_job, get_job
        from tools.cronjob_tools import cronjob
        import json

        job_a = create_job(prompt="Find news", schedule="every 1h")
        job_b = create_job(
            prompt="Summarize", schedule="every 2h", context_from=job_a["id"],
        )

        # Update an unrelated field
        result = json.loads(cronjob(
            action="update",
            job_id=job_b["id"],
            prompt="Summarize v2",
        ))
        assert result["success"] is True
        reloaded = get_job(job_b["id"])
        assert reloaded["prompt"] == "Summarize v2"
        assert reloaded["context_from"] == [job_a["id"]]
