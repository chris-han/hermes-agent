"""Workspace-bound retirement file stager.

Plan A M3 (D5, Law 6): the stager is the only filesystem sink for staged
retirement files. It lives under
`workspaces/<workspace_id>/.semantier-retirement/trash/<source_id>/<utc-timestamp>/<basename>`
and is the inverse of the restore path. No global scratch paths are
permitted.

The stager is intentionally tiny: it owns three operations
(stage, restore, unlink) and is dependency-injected into the orchestrator.
This lets tests substitute a recording stager and lets the orchestrator
remain storage-agnostic about filesystem layout.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RepresentationMissing(FileNotFoundError):
    """A declared member of the governed representation set is unavailable."""


class WorkspaceRetirementStager:
    """Workspace-bound staging, restore, and unlink for retirement files."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        workspace_id: str,
        allow_missing: bool = False,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        self._workspace_id = workspace_id
        self._allow_missing = allow_missing

    @property
    def trash_root(self) -> Path:
        """The trash root, lazily created on first stage call."""
        return self._workspace_root / ".semantier-retirement" / "trash"

    def _stage_path(self, source_id: str, original_path: str) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        basename = Path(original_path).name or source_id
        return self.trash_root / source_id / ts / basename

    def stage(self, *, source_id: str, original_path: str) -> str:
        """Move the source file into the workspace trash.

        `original_path` is interpreted relative to the workspace root. Returns
        the absolute staged path as a string.
        """
        if Path(original_path).is_absolute():
            raise ValueError("original_path must be workspace-relative")
        src = self._workspace_root / original_path
        if not src.exists():
            raise FileNotFoundError(f"source file not found: {src}")
        dst = self._stage_path(source_id, original_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return str(dst)

    def _resolve_original(self, original_path: str) -> Path:
        relative = Path(original_path)
        if relative.is_absolute():
            raise ValueError("original_path must be workspace-relative")
        resolved = (self._workspace_root / relative).resolve()
        if not resolved.is_relative_to(self._workspace_root):
            raise ValueError("original_path must remain inside workspace")
        return resolved

    @staticmethod
    def _write_manifest(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def stage_many(
        self,
        *,
        source_id: str,
        original_paths: list[str],
        operation_id: str | None = None,
        lineage_id: str | None = None,
        inventory_hash: str | None = None,
    ) -> str:
        """Stage a validated representation set and persist its compensation manifest."""
        paths = list(dict.fromkeys(original_paths))
        if not paths:
            raise ValueError("at least one original_path is required")
        resolved = [(relative, self._resolve_original(relative)) for relative in paths]
        missing = [relative for relative, path in resolved if not path.is_file()]
        if missing and not self._allow_missing:
            missing_path = self._resolve_original(missing[0])
            raise RepresentationMissing(f"representation file not found: {missing_path}")
        present = [item for item in resolved if item[1].is_file()]

        operation_root = self._stage_path(source_id, "manifest.json").parent
        operation_root.mkdir(parents=True, exist_ok=False)
        manifest_path = operation_root / "manifest.json"
        manifest = {
            "schema_version": "source_retirement_file_manifest.v1",
            "workspace_id": self._workspace_id,
            "source_id": source_id,
            "lineage_id": lineage_id or source_id,
            "operation_id": operation_id,
            "inventory_hash": inventory_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "missing_paths": missing,
            "entries": [
                {
                    "original_path": relative,
                    "staged_path": str(operation_root / "files" / f"{index:04d}" / path.name),
                    "moved": False,
                }
                for index, (relative, path) in enumerate(present)
            ],
        }
        self._write_manifest(manifest_path, manifest)
        try:
            for entry, (_, source) in zip(manifest["entries"], present, strict=True):
                destination = Path(entry["staged_path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                entry["moved"] = True
                self._write_manifest(manifest_path, manifest)
        except Exception:
            for entry in reversed(manifest["entries"]):
                staged = Path(entry["staged_path"])
                original = self._resolve_original(entry["original_path"])
                if entry["moved"] and staged.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged), str(original))
                    self._cleanup_empty_parents(staged)
            manifest_path.unlink(missing_ok=True)
            self._cleanup_empty_parents(manifest_path)
            raise
        return str(manifest_path)

    def stage_inventory(self, inventory: Any) -> str:
        if str(inventory.workspace_id) != self._workspace_id:
            raise ValueError("inventory is bound to a different workspace")
        paths = [
            str(entry.relative_path)
            for entry in inventory.entries
            if entry.physically_removable and entry.relative_path is not None
        ]
        return self.stage_many(
            source_id=str(inventory.lineage_id),
            original_paths=paths,
            operation_id=str(inventory.operation_id),
            lineage_id=str(inventory.lineage_id),
            inventory_hash=str(inventory.inventory_hash),
        )

    def restore(self, *, source_id: str, staged_path: str) -> str:
        """Restore the staged file to its original workspace-relative path.

        Plan A M3 invariant: a successful restore leaves the trash directory
        empty (no leftover file, no leftover empty parent directories). The
        trash is a transient scratch surface inside the workspace bound; a
        restore must not preserve any visible artefact.
        """
        src = Path(staged_path)
        if src.name == "manifest.json":
            if not src.exists():
                return ""
            manifest = json.loads(src.read_text(encoding="utf-8"))
            restored: list[str] = []
            for entry in reversed(manifest["entries"]):
                staged = Path(entry["staged_path"])
                original = self._resolve_original(entry["original_path"])
                if staged.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staged), str(original))
                    self._cleanup_empty_parents(staged)
                if original.exists():
                    restored.append(entry["original_path"])
            src.unlink(missing_ok=True)
            self._cleanup_empty_parents(src)
            return restored[-1] if restored else ""
        if not src.exists():
            raise FileNotFoundError(f"staged file missing: {src}")
        # Recover the original relative path by stripping the trash prefix.
        relative = src.relative_to(self.trash_root / source_id)
        # `relative` is `<timestamp>/<basename>`; the basename is the original
        # relative path captured at stage time. The orchestrator records the
        # original path in the operation row; we use the basename here.
        basename = relative.name
        dst = self._workspace_root / basename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        # Remove the now-empty per-stage directories so the trash directory
        # does not retain visible artefacts after a restore.
        self._cleanup_empty_parents(src)
        return str(dst.relative_to(self._workspace_root))

    def _cleanup_empty_parents(self, path: Path) -> None:
        """Remove empty parent directories up to (but not including) trash_root."""
        parent = path.parent
        while parent != self.trash_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                # Directory not empty (e.g. a sibling staged file shares an
                # ancestor): stop climbing. This is the normal termination.
                break
            parent = parent.parent

    def unlink(self, *, staged_path: str) -> None:
        """Atomically remove the staged file.

        Plan A M3 invariant: a successful unlink leaves the trash directory
        empty (no leftover file, no leftover empty parent directories).
        """
        path = Path(staged_path)
        if path.name == "manifest.json":
            if not path.exists():
                return
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for entry in manifest["entries"]:
                staged = Path(entry["staged_path"])
                staged.unlink(missing_ok=True)
                self._cleanup_empty_parents(staged)
            path.unlink(missing_ok=True)
            self._cleanup_empty_parents(path)
            return
        if path.exists():
            path.unlink()
        self._cleanup_empty_parents(path)
