"""Bind opted-in Windows bundle Git calls to the controller's WSL Git."""
import os
import re
import subprocess


def linux_path(distro, path):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", distro):
        raise ValueError("invalid WSL Git distribution")
    for attempt in range(3):
        try:
            value = subprocess.check_output(
                ["wsl.exe", "-d", distro, "--exec", "/usr/bin/wslpath", "-u", str(path)],
                text=True, encoding="utf-8", timeout=30,
            ).strip()
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
    if not value.startswith("/") or "\n" in value or "\r" in value:
        raise ValueError("invalid WSL Git path")
    return value


def git_command(source, *args):
    distro = os.environ.get("SEMANTIER_WSL_GIT_DISTRO", "")
    if os.name != "nt" or not distro:
        return ["git", *map(str, args)]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", distro):
        raise ValueError("invalid WSL Git distribution")

    converted = [linux_path(distro, arg) if os.path.isabs(str(arg)) else str(arg) for arg in args]
    cleared = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
               "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG",
               "GIT_CONFIG_SYSTEM", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_NAMESPACE")
    return ["wsl.exe", "-d", distro, "--exec", "/usr/bin/env",
            *(part for name in cleared for part in ("-u", name)),
            "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "/usr/bin/git",
            "-C", linux_path(distro, source), *converted]
