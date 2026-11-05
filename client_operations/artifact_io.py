"""Standard-library publication custody primitives shared with the CI producer."""
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath


class OperationError(ValueError):
    def __init__(self,status,code,*,mutation_outcome='unknown'):
        self.status,self.code,self.mutation_outcome=status,code,mutation_outcome
        super().__init__(code)


def safe_file(root,key):
    relative=PurePosixPath(key)
    if relative.is_absolute() or any(p in {'','..','.'} for p in relative.parts) or '\\' in key:
        raise OperationError(422,'invalid_artifact_key')
    target=root.joinpath(*relative.parts)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p!=root.parent):
        raise OperationError(422,'artifact_symlink_refused')
    if not target.resolve().is_relative_to(root.resolve()):raise OperationError(422,'artifact_outside_root')
    return target


def atomic_write(target,body):
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.publishing-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(name,target)
        directory=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def immutable_write(target,body):
    """Publish once with an atomic no-replace link; concurrent writers cannot win silently."""
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.admitting-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        try:os.link(name,target)
        except FileExistsError:
            if target.read_bytes()!=body:raise OperationError(409,'immutable_artifact_conflict')
        directory=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def file_facts(source):
    checksum = hashlib.sha256()
    size = 0
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
            size += len(chunk)
    return size, checksum.hexdigest()


def immutable_copy(source, destination, expected):
    """Stage signed multi-GB bytes without materializing the package in memory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.admitting-', dir=destination.parent)
    checksum = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as incoming:
            for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                output.write(chunk)
                checksum.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != expected['size'] or checksum.hexdigest() != expected['sha256']:
            raise OperationError(422, 'artifact_integrity_mismatch')
        try:
            os.link(name, destination)
        except FileExistsError:
            if file_facts(destination) != (size, checksum.hexdigest()):
                raise OperationError(409, 'immutable_artifact_conflict')
        if file_facts(destination) != (size, checksum.hexdigest()):
            raise OperationError(503, 'artifact_readback_failed')
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


