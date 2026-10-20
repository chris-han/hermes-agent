"""Observe the pinned interpreter's loaded SSL binaries, not a PATH openssl CLI."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import ssl
import sys


def observe():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel.GetCurrentProcess()
    handles = (wintypes.HMODULE * 2048)()
    needed = wintypes.DWORD()
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    if not psapi.EnumProcessModules(process, handles, ctypes.sizeof(handles), ctypes.byref(needed)) or needed.value > ctypes.sizeof(handles):
        raise OSError('cannot enumerate loaded SSL modules')
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    modules = {}
    for handle in handles[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        length = kernel.GetModuleFileNameW(handle, name, len(name))
        if not length or length == len(name):
            raise OSError('cannot identify loaded module')
        path = Path(name.value)
        if any(word in path.name.lower() for word in ('ssl', 'crypto')):
            with path.open('rb') as stream:
                modules[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not any('libssl' in Path(name).name.lower() for name in modules):
        raise OSError('actual loaded libssl identity unavailable')
    return {'version': ssl.OPENSSL_VERSION, 'python': sys.version, 'abi': sys.implementation.cache_tag, 'modules': modules}


if __name__ == '__main__':
    print(json.dumps(observe(), sort_keys=True))
