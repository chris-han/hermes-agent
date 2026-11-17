"""Read MSIX source identity through Windows AppModel; never installs or starts."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys


def _package_string(function: str, full_name: str) -> str | None:
    api = getattr(ctypes.WinDLL('kernel32', use_last_error=True), function)
    api.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.UINT), wintypes.LPWSTR]
    api.restype = wintypes.LONG
    size = wintypes.UINT()
    result = api(full_name, ctypes.byref(size), None)
    if result in {87, 1168, 15700}:  # Invalid name / absent registration, not an API failure.
        return None
    if result != 122:
        raise OSError(result, 'Native package identity unavailable')
    if not 1 < size.value <= 32768:
        raise OSError('Invalid native package identity length')
    output = ctypes.create_unicode_buffer(size.value)
    result = api(full_name, ctypes.byref(size), output)
    if result:
        raise OSError(result, 'Native package identity unavailable')
    return output.value


def package_source(executable: str | Path, *, registered: bool):
    """(package root, family, version, architecture/resource) for an image path.

    The current client's root must match Windows registration. A retired owner's
    full name may only be parsed; its trust/store relation is checked by the caller.
    """
    if sys.platform != 'win32':
        return None
    for root in Path(executable).resolve().parents:
        parts = root.name.rsplit('_', 4)
        if len(parts) != 5:
            continue
        family = _package_string('PackageFamilyNameFromFullName', root.name)
        if not family:
            continue
        version = tuple(int(part) for part in parts[1].split('.'))
        if len(version) != 4 or any(not 0 <= part <= 65535 for part in version):
            raise ValueError('Invalid native package version')
        if registered:
            actual = _package_string('GetPackagePathByFullName', root.name)
            if not actual or Path(actual).resolve() != root:
                raise ValueError('Native client package path is not registered')
        return root, family, version, '_'.join(parts[2:4])
    return None
