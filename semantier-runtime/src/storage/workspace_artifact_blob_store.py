"""Workspace-stable original bytes; callers supply governed context and limits."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import BinaryIO
from uuid import uuid4
import zipfile
from xml.etree import ElementTree

from agents.auth_session import RequestContext
from contracts.ontology_workbench_context import StagedWorkspaceArtifactV1


class WorkspaceArtifactBlobStore:
    def __init__(self, *, ctx: RequestContext, max_bytes: int):
        if not ctx.authenticated or not ctx.organization_id or not ctx.workspace_id:
            raise ValueError("AUTHORITY_CONTEXT_REQUIRED")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("INVALID_BYTE_LENGTH")
        if not ctx.workspace_root.is_absolute():
            raise ValueError("WORKSPACE_REQUIRED")
        self._root = ctx.workspace_root
        self._max_bytes = max_bytes

    @staticmethod
    def _stage_name(stage_ref: str) -> str:
        if (
            re.fullmatch(
                r"artifacts/workspace-originals/\.staging/[0-9a-f]{32}", stage_ref
            )
            is None
        ):
            raise ValueError("invalid staging reference")
        return stage_ref.rsplit("/", 1)[1]

    def discard_stage(self, *, stage_ref: str) -> None:
        self._stage_name(stage_ref)
        try:
            with self._parent(stage_ref) as (parent, name):
                metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("staging reference is not a regular file")
                os.unlink(name, dir_fd=parent)
                os.fsync(parent)
        except FileNotFoundError:
            return

    def cleanup_unreferenced_stages(
        self, *, older_than: datetime, referenced_stage_refs: tuple[str, ...]
    ) -> int:
        if not isinstance(older_than, datetime) or older_than.utcoffset() != timedelta(
            0
        ):
            raise ValueError("cleanup cutoff must be timezone-aware UTC")
        referenced = {self._stage_name(ref) for ref in referenced_stage_refs}
        removed = 0
        try:
            with self._parent("artifacts/workspace-originals/.staging/scan") as (
                parent,
                _,
            ):
                for name in os.listdir(parent):
                    if (
                        name in referenced
                        or re.fullmatch(r"[0-9a-f]{32}", name) is None
                    ):
                        continue
                    try:
                        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_nlink != 1
                            or metadata.st_mtime >= older_than.timestamp()
                        ):
                            continue
                        # Recheck the same inode immediately before deleting its staging link.
                        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        if (
                            current.st_dev,
                            current.st_ino,
                            current.st_mtime_ns,
                            current.st_nlink,
                        ) != (
                            metadata.st_dev,
                            metadata.st_ino,
                            metadata.st_mtime_ns,
                            1,
                        ):
                            continue
                        os.unlink(name, dir_fd=parent)
                        removed += 1
                    except FileNotFoundError:
                        continue
                if removed:
                    os.fsync(parent)
        except FileNotFoundError:
            return 0
        return removed

    @contextmanager
    def _parent(self, ref: str, *, create: bool = False):
        parts = ref.split("/")
        if (
            "\\" in ref
            or ":" in ref
            or "\x00" in ref
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise ValueError("invalid governed byte reference")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(self._root, flags)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                successor = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = successor
            yield fd, parts[-1]
        finally:
            os.close(fd)

    @staticmethod
    def _digest(expected_hash: str) -> str:
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError("ARTIFACT_HASH_MISMATCH")
        return expected_hash[7:]

    @staticmethod
    def _verify(stream: BinaryIO, expected_hash: str, expected_size: int):
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if "sha256:" + digest != expected_hash:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        stream.seek(0)

    def _detect(self, stream: BinaryIO, name: str, declared: str) -> str:
        if not name or any(char in name for char in ("/", "\\", "\x00", ":")):
            raise ValueError("INVALID_MEDIA_TYPE")
        suffix = Path(name).suffix.lower()
        allowed = {
            ".html": ("text/html",),
            ".htm": ("text/html",),
            ".pdf": ("application/pdf",),
            ".docx": (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ".md": ("text/markdown", "text/plain"),
            ".txt": ("text/plain",),
            ".csv": ("text/csv", "application/csv", "application/vnd.ms-excel"),
            ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
            ".ipynb": ("application/x-ipynb+json",),
        }
        if declared not in allowed.get(suffix, ()):
            raise ValueError("INVALID_MEDIA_TYPE")
        magic = stream.read(8)
        stream.seek(0)
        if suffix in {".html", ".htm"}:
            from services.original_html_source import decode
            try:
                decode(stream.read(self._max_bytes + 1))
                valid = True
            except (ValueError, UnicodeError, LookupError):
                valid = False
            stream.seek(0)
        elif suffix == ".pdf":
            valid = magic.startswith(b"%PDF-")
        elif suffix == ".xlsx":
            try:
                with zipfile.ZipFile(stream) as archive:
                    names = set(archive.namelist())
                    valid = {"[Content_Types].xml", "xl/workbook.xml"} <= names and not any("vbaproject" in name.lower() for name in names)
                    valid = valid and sum(item.file_size for item in archive.infolist()) <= self._max_bytes * 4
                    valid = valid and all(item.file_size <= self._max_bytes for item in archive.infolist())
            except zipfile.BadZipFile:
                valid = False
        elif suffix == ".docx":
            try:
                with zipfile.ZipFile(stream) as archive:
                    names = set(archive.namelist())
                    valid = {
                        "[Content_Types].xml",
                        "word/document.xml",
                    } <= names and not any(
                        "vbaproject" in name.lower() for name in names
                    )
                    if valid:

                        class NoDoctype(ElementTree.TreeBuilder):
                            def doctype(self, name, pubid, system):
                                raise ValueError("INVALID_MEDIA_TYPE")

                        roots = []
                        for member in ("[Content_Types].xml", "word/document.xml"):
                            if archive.getinfo(member).file_size > self._max_bytes:
                                raise ValueError("INVALID_BYTE_LENGTH")
                            parser = ElementTree.XMLParser(target=NoDoctype())
                            roots.append(
                                ElementTree.fromstring(
                                    archive.read(member), parser=parser
                                )
                            )
                        types, document = roots
                        package_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
                        valid = (
                            types.tag == f"{{{package_ns}}}Types"
                            and document.tag
                            == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document"
                            and any(
                                item.tag == f"{{{package_ns}}}Override"
                                and item.get("PartName") == "/word/document.xml"
                                and item.get("ContentType")
                                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
                                for item in types
                            )
                        )
            except (zipfile.BadZipFile, ElementTree.ParseError, RuntimeError):
                valid = False
        else:
            valid = not magic.startswith((b"MZ", b"PK", b"%PDF-", b"\x7fELF"))
            try:
                # Text validation is bounded by the policy-enforced staged size.
                raw_text = stream.read()
                if suffix == ".csv":
                    from charset_normalizer import from_bytes
                    match = from_bytes(raw_text).best()
                    if match is None:
                        raise UnicodeDecodeError("utf-8", raw_text, 0, 1, "unknown CSV encoding")
                    text = str(match)
                else:
                    text = raw_text.decode("utf-8-sig")
                valid = valid and "\x00" not in text
            except UnicodeDecodeError:
                valid = False
        stream.seek(0)
        if not valid:
            raise ValueError("INVALID_MEDIA_TYPE")
        return "text/markdown" if suffix == ".md" else allowed[suffix][0]

    def stage(
        self,
        *,
        byte_stream: BinaryIO,
        expected_hash: str,
        expected_size: int,
        original_name: str,
        declared_media_type: str,
    ) -> StagedWorkspaceArtifactV1:
        self._digest(expected_hash)
        if type(expected_size) is not int or not 0 <= expected_size <= self._max_bytes:
            raise ValueError("INVALID_BYTE_LENGTH")
        ref = "artifacts/workspace-originals/.staging/" + uuid4().hex
        with self._parent(ref, create=True) as (parent, name):
            fd = os.open(
                name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            try:
                with os.fdopen(fd, "w+b") as stream:
                    size = 0
                    while chunk := byte_stream.read(65536):
                        size += len(chunk)
                        if size > expected_size:
                            raise ValueError("INVALID_BYTE_LENGTH")
                        stream.write(chunk)
                    stream.flush()
                    stream.seek(0)
                    self._verify(stream, expected_hash, expected_size)
                    media = self._detect(stream, original_name, declared_media_type)
                    os.fsync(stream.fileno())
                os.fsync(parent)
            except BaseException:
                os.unlink(name, dir_fd=parent)
                raise
        return StagedWorkspaceArtifactV1(
            schema_version="staged_workspace_artifact.v1",
            stage_ref=ref,
            content_hash=expected_hash,
            byte_size=expected_size,
            detected_media_type=media,
        )

    def exact_get(
        self, *, byte_owner_ref: str, expected_hash: str, expected_size: int
    ) -> BinaryIO:
        digest = self._digest(expected_hash)
        if (
            byte_owner_ref
            != f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}"
            or type(expected_size) is not int
            or expected_size < 0
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        with self._parent(byte_owner_ref) as (parent, name):
            stream = os.fdopen(
                os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                ),
                "rb",
            )
        try:
            self._verify(stream, expected_hash, expected_size)
            return stream
        except BaseException:
            stream.close()
            raise

    def publish(self, *, stage: StagedWorkspaceArtifactV1, artifact_ref: str) -> str:
        stage = StagedWorkspaceArtifactV1.model_validate(stage)
        if not artifact_ref:
            raise ValueError("artifact_ref required")
        digest = self._digest(stage.content_hash)
        ref = f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}"
        with self._parent(stage.stage_ref) as (source_parent, source_name):
            with os.fdopen(
                os.open(
                    source_name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=source_parent,
                ),
                "rb",
            ) as source:
                self._verify(source, stage.content_hash, stage.byte_size)
            with self._parent(ref, create=True) as (target_parent, target_name):
                try:
                    os.link(
                        source_name,
                        target_name,
                        src_dir_fd=source_parent,
                        dst_dir_fd=target_parent,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
                with self.exact_get(
                    byte_owner_ref=ref,
                    expected_hash=stage.content_hash,
                    expected_size=stage.byte_size,
                ):
                    pass
                os.fsync(target_parent)
            os.unlink(source_name, dir_fd=source_parent)
            os.fsync(source_parent)
        return ref
