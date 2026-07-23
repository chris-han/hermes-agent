from __future__ import annotations

import pytest

from hermes_cli import web_server


def test_curated_memory_short_paths_resolve_to_memories_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)

    relative_path, full_path = web_server._resolve_memory_file_path("USER.md")

    assert relative_path == "USER.md"
    assert full_path == tmp_path / "memories" / "USER.md"


@pytest.mark.asyncio
async def test_read_curated_memory_short_path_reads_memories_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "USER.md").write_text("User's name is Chris.", encoding="utf-8")

    payload = await web_server.read_memory_file("USER.md")

    assert payload == {"path": "USER.md", "content": "User's name is Chris."}


@pytest.mark.asyncio
async def test_missing_curated_memory_short_path_reads_as_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)

    payload = await web_server.read_memory_file("MEMORY.md")

    assert payload == {"path": "MEMORY.md", "content": ""}

