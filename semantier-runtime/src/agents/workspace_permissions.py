"""POSIX ACLs for settings shared by backend and workspace service identities.

Deployment UID is an OS service identity. Application authorization still owns
tenant access; this never changes file ownership or grants world write access.
"""
import errno
import os
from pathlib import Path
import stat
import struct

_VERSION = 2  # linux/posix_acl_xattr.h
_OWNER, _USER, _GROUP, _MASK, _OTHER = 1, 2, 4, 16, 32
_UNDEFINED = 0xffffffff


def _grant(path: Path, uid: int, *, default: bool = False) -> None:
    name = 'system.posix_acl_default' if default else 'system.posix_acl_access'
    mode = path.stat(follow_symlinks=False).st_mode
    try:
        raw = os.getxattr(path, name, follow_symlinks=False)
        if struct.unpack('<I', raw[:4])[0] != _VERSION:
            raise RuntimeError(f'UNSUPPORTED_WORKSPACE_ACL:{path}')
        entries = list(struct.iter_unpack('<HHI', raw[4:]))
    except OSError as exc:
        if exc.errno != errno.ENODATA:
            raise RuntimeError(f'WORKSPACE_ACL_UNAVAILABLE:{path}:{exc}') from exc
        entries = [(_OWNER, (mode >> 6) & 7, _UNDEFINED),
                   (_GROUP, (mode >> 3) & 7, _UNDEFINED),
                   (_OTHER, mode & 7, _UNDEFINED)]
    old_mask = next((p for tag, p, _ in entries if tag == _MASK),
                    next(p for tag, p, _ in entries if tag == _GROUP))
    executable = any((perm if tag in (_OWNER, _OTHER) else perm & old_mask) & 1
                     for tag, perm, _ in entries if tag != _MASK)
    required = 7 if default or stat.S_ISDIR(mode) or executable else 6
    updated = []
    for tag, perm, identity in entries:
        if tag == _MASK or (tag == _USER and identity == uid):
            continue
        # Expanding the mask for this service must not unmask other ACL grants.
        if tag in (_USER, _GROUP, 8):
            perm &= old_mask
        if tag == _OWNER and path.stat(follow_symlinks=False).st_uid == uid:
            perm |= required
        updated.append((tag, perm, identity))
    updated += [(_USER, required, uid), (_MASK, old_mask | required, _UNDEFINED)]
    updated.sort(key=lambda row: (row[0], row[2]))
    raw = struct.pack('<I', _VERSION) + b''.join(struct.pack('<HHI', *row) for row in updated)
    os.setxattr(path, name, raw, follow_symlinks=False)


def grant_runtime_directory(path: Path, uid: int) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError('WORKSPACE_PERMISSION_ROOT_INVALID')
    _grant(path, uid)
    _grant(path, uid, default=True)



def grant_runtime_settings_access(root: Path, uid: int) -> None:
    """Grant the configured workspace service access to its settings files only."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or uid < 0:
        raise ValueError("WORKSPACE_PERMISSION_ROOT_OR_UID_INVALID")
    paths = [root / name for name in ("config.yaml", ".env")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("WORKSPACE_SETTINGS_SYMLINK_REJECTED")
    grant_runtime_directory(root, uid)
    for path in paths:
        if path.is_file():
            _grant(path, uid)
