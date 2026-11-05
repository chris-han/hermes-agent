"""Held Linux dirfds for the fixed ECS public/private storage roots."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import stat

from .ecs_protocol import fail

DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def identity(info):
    return {"device": info.st_dev, "inode": info.st_ino}


def trusted(info):
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid not in (0, os.geteuid())
        or info.st_mode & 0o022
    ):
        fail(503, "ecs_untrusted_storage_parent")


def absolute_directory(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        fail(503, "ecs_storage_root_invalid")
    descriptor = os.open("/", DIR)
    try:
        trusted(os.fstat(descriptor))
        for part in path.parts[1:]:
            child = os.open(part, DIR, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            trusted(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class Root:
    def __init__(self, path, expected, *, private=False):
        self.path = Path(path)
        self.private = private
        self.fd = None
        try:
            self.fd = absolute_directory(path)
            self.identity = identity(os.fstat(self.fd))
            self.device = self.identity["device"]
            if self.identity != expected:
                fail(503, "ecs_storage_identity_changed")
        except OSError:
            self.close()
            fail(503, "ecs_storage_root_unavailable")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def check(self):
        try:
            current = absolute_directory(self.path)
            try:
                if identity(os.fstat(current)) != self.identity:
                    fail(503, "ecs_storage_identity_changed")
            finally:
                os.close(current)
        except OSError:
            fail(503, "ecs_storage_root_unavailable")

    @staticmethod
    def parts(key):
        if (
            not isinstance(key, str)
            or len(key) > 512
            or re.fullmatch("[A-Za-z0-9._/-]+", key) is None
            or any(part in ("", ".", "..") for part in key.split("/"))
        ):
            fail(400, "ecs_storage_key_invalid")
        return key.split("/")

    @contextmanager
    def parent(self, key, *, create=False):
        parts = self.parts(key)
        self.check()
        descriptor = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(
                            part, 0o700 if self.private else 0o755, dir_fd=descriptor
                        )
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                child = os.open(part, DIR, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
                info = os.fstat(descriptor)
                trusted(info)
                if info.st_dev != self.device:
                    fail(503, "ecs_cross_device_storage")
            yield descriptor, parts[-1]
        except OSError as error:
            if isinstance(error, FileNotFoundError):
                raise
            fail(503, "ecs_storage_path_unavailable")
        finally:
            os.close(descriptor)

    def regular(self, info):
        if not stat.S_ISREG(info.st_mode) or info.st_dev != self.device:
            fail(503, "ecs_storage_file_invalid")

    @contextmanager
    def reader(self, key):
        with self.parent(key) as (parent, name):
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                self.regular(os.fstat(stream.fileno()))
                yield stream

    @contextmanager
    def creator(self, key):
        with self.parent(key, create=True) as (parent, name):
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            os.fsync(parent)
            with os.fdopen(descriptor, "wb") as stream:
                yield stream
            os.fsync(parent)

    def read(self, key, maximum):
        try:
            with self.reader(key) as stream:
                if os.fstat(stream.fileno()).st_size > maximum:
                    fail(413, "ecs_file_limit")
                return stream.read(maximum + 1)
        except FileNotFoundError:
            return None

    def write_new(self, key, body):
        with self.creator(key) as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())

    def facts(self, key):
        try:
            with self.reader(key) as stream:
                before = os.fstat(stream.fileno())
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    fail(409, "ecs_file_changed")
                return {
                    **identity(after),
                    "type": "regular",
                    "size": after.st_size,
                    "sha256": digest,
                }
        except FileNotFoundError:
            return None

    def check_parent(self, key, descriptor):
        with self.parent(key) as (current, _):
            if identity(os.fstat(current)) != identity(os.fstat(descriptor)):
                fail(409, "ecs_parent_changed")

    def publish(self, source, source_key, key, digest, size):
        facts = source.facts(source_key)
        if source.device != self.device:
            fail(503, "ecs_cross_device_staging")
        if not facts or facts["sha256"] != digest or facts["size"] != size:
            fail(422, "ecs_object_integrity_mismatch")
        with (
            source.parent(source_key) as (origin, name),
            self.parent(key, create=True) as (parent, destination),
        ):
            self.check_parent(key, parent)
            source.check_parent(source_key, origin)
            if identity(os.stat(name, dir_fd=origin, follow_symlinks=False)) != {
                k: facts[k] for k in ("device", "inode")
            }:
                fail(409, "ecs_staging_changed")
            os.chmod(name, 0o444, dir_fd=origin, follow_symlinks=False)
            created = True
            try:
                os.link(
                    name,
                    destination,
                    src_dir_fd=origin,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                created = False
                existing = self.facts(key)
                if not existing or (existing["size"], existing["sha256"]) != (
                    size,
                    digest,
                ):
                    fail(409, "ecs_immutable_conflict")
            os.fsync(parent)
            return created

    def replace_from(self, source, source_key, key):
        if source.device != self.device:
            fail(503, "ecs_cross_device_staging")
        with (
            source.parent(source_key) as (origin, name),
            self.parent(key, create=True) as (parent, destination),
        ):
            self.check_parent(key, parent)
            source.check_parent(source_key, origin)
            self.regular(os.stat(name, dir_fd=origin, follow_symlinks=False))
            try:
                self.regular(os.stat(destination, dir_fd=parent, follow_symlinks=False))
            except FileNotFoundError:
                pass
            os.chmod(name, 0o444, dir_fd=origin, follow_symlinks=False)
            os.replace(name, destination, src_dir_fd=origin, dst_dir_fd=parent)
            os.fsync(parent)
            os.fsync(origin)

    def unlink(self, key, expected):
        current = self.facts(key)
        if current is None:
            return False
        if current != expected:
            fail(409, "ecs_cleanup_identity_changed")
        with self.parent(key) as (parent, name):
            self.check_parent(key, parent)
            now = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if identity(now) != {
                k: expected[k] for k in ("device", "inode")
            } or not stat.S_ISREG(now.st_mode):
                fail(409, "ecs_cleanup_identity_changed")
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)
        return True

    def entries(self):
        self.check()
        result = []

        def walk(descriptor, prefix):
            for name in os.listdir(descriptor):
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                key = prefix + name
                if info.st_dev != self.device:
                    fail(503, "ecs_cross_device_storage")
                if stat.S_ISDIR(info.st_mode):
                    trusted(info)
                    child = os.open(name, DIR, dir_fd=descriptor)
                    try:
                        walk(child, key + "/")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    result.append((key, info))
                else:
                    fail(503, "ecs_storage_inventory_unknown")

        walk(self.fd, "")
        return result
