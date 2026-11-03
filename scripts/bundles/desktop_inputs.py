"""Build identity and process environment from an admitted preparation."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.bundles.desktop_prepare import BuildRequest, PreparedDesktop


def validate_windows_external_layout(source: str, work: str, cache: str) -> None:
    """Validate the controller's exact source, cache and job-local work paths."""
    import ntpath

    def canonical(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z]:\\[^\\/]+(?:\\[^\\/]+)*', value):
            raise ValueError('invalid Windows external build layout')
        if any(part in ('.', '..') or ':' in part for part in value[3:].split('\\')):
            raise ValueError('invalid Windows external build layout')
        return ntpath.normcase(ntpath.normpath(value))

    source, work, cache = map(canonical, (source, work, cache))
    prefix = ntpath.normcase(r'D:\repo\deliverables\work') + '\\'
    if (source != ntpath.normcase(r'D:\repo\hermes-build')
            or cache != ntpath.normcase(r'D:\repo\deliverables\cache\desktop-inputs')
            or not work.startswith(prefix)):
        raise ValueError('invalid Windows external build layout')
    parts = work[len(prefix):].split('\\')
    if (len(parts) != 2 or parts[1] != 'desktop-job'
            or not re.fullmatch(r'[a-z0-9_-][a-z0-9._-]*', parts[0])
            or parts[0].endswith('.')
            or re.match(r'^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)', parts[0])):
        raise ValueError('invalid Windows external build layout')


def require_external_output_readiness(selected: bool, source=None, work=None, cache=None) -> None:
    """Admit Windows external jobs only at the controller's guarded paths."""
    if not selected or sys.platform != "win32":
        return
    from pm.filesystem import is_junction, is_reparse_point
    from scripts.bundles.desktop_prepare import require_external_paths_unredirected

    if source is None or work is None or cache is None:
        raise ValueError("invalid Windows external build layout")
    validate_windows_external_layout(str(source), str(work), str(cache))
    source, work, cache = map(Path, (source, work, cache))
    require_external_paths_unredirected(source, work, cache)
    checkout_lock = source / '.build' / '.desktop-build.lock'
    distribution_lock = Path(str(source) + '.distribution.lock')
    for lock in (checkout_lock, distribution_lock):
        if lock.is_symlink() or is_junction(lock) or is_reparse_point(lock) or not lock.is_file():
            raise ValueError("Windows external build requires existing checkout and distribution locks")


@contextmanager
def build_lock(source: Path, *, external_cache: Path | None = None):
    from hermes_cli.runtime_state import _lock
    from pm.filesystem import is_junction, is_reparse_point

    if external_cache is not None:
        if (not external_cache.is_absolute() or external_cache.resolve().is_relative_to(source.resolve())
                or source.resolve().is_relative_to(external_cache.resolve())):
            raise ValueError("external build lock cache must be outside source")
        for path in (external_cache, *external_cache.parents):
            if path.is_symlink() or is_junction(path) or is_reparse_point(path):
                raise ValueError("external build lock cache is redirected")
    directory = source / ".build"
    if directory.is_symlink() or is_junction(directory) or is_reparse_point(directory):
        raise ValueError("desktop build scratch is redirected")
    existing_only = external_cache is not None and sys.platform == 'win32'
    if existing_only:
        if not directory.is_dir() or not (directory / '.desktop-build.lock').is_file():
            raise ValueError("external Windows build requires an existing checkout mutex")
    else:
        directory.mkdir(exist_ok=True)
    with ExitStack() as locks:
        lock_path = directory / ".desktop-build.lock"
        if lock_path.is_symlink() or is_junction(lock_path) or is_reparse_point(lock_path):
            raise ValueError("desktop checkout lock is redirected")
        fd = os.open(lock_path, os.O_RDWR | (0 if existing_only else os.O_CREAT) | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        checkout_lock = locks.enter_context(os.fdopen(fd, 'a+b'))
        if not _lock(checkout_lock.fileno(), wait=False):
            raise ValueError("another desktop build or preparation is using this source checkout")
        if external_cache is not None:
            external_cache.mkdir(parents=True, exist_ok=True)
            lock_path = external_cache / ".desktop-build.lock"
            if lock_path.is_symlink() or is_junction(lock_path) or is_reparse_point(lock_path):
                raise ValueError("external build lock is redirected")
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            cache_lock = locks.enter_context(os.fdopen(fd, 'a+b'))
            if not _lock(cache_lock.fileno(), wait=False):
                raise ValueError("another desktop build or preparation is using this shared cache")
        yield


def build_environment(prepared: PreparedDesktop, variant: str, inherited: Mapping[str, str]) -> dict[str, str]:
    from pm.build_operations import verified_tools
    from pm.lock import Lockfile
    from scripts.bundles.desktop_toolchain import bootstrap_environment

    request = prepared.request
    env = bootstrap_environment(request.source, request.work, request.cache, inherited)
    selection = verified_tools(["python", "node", "npm"], source_store=request.cache / "tools",
                               target=request.target, lock=Lockfile(request.source / "pm/lock.json"))
    if prepared.python != selection.entries["python"].binary or prepared.node != selection.entries["node"].binary:
        raise ValueError("prepared Python/Node paths do not match selected tools; prepare again")
    env = selection.environment(env)
    env["HERMES_PYTHON"] = str(prepared.python)
    env = identity_environment(request, variant, env)
    if request.external_outputs:
        env = external_node_environment(request, prepared.node_root, env)
    return env


def external_node_environment(request: BuildRequest, root: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    """Bind the job's inspected dependency tree to every Node child process."""
    from scripts.bundles.desktop_prepare import require_node_root

    require_node_root(request, root)
    hook = request.source / "scripts/build/external-node-resolver.mjs"
    if not hook.is_file() or hook.is_symlink():
        raise ValueError("external Node resolver is missing or redirected")
    # Do not let an inherited loader inject code into a source-bound build.
    env = {key: value for key, value in inherited.items() if key.upper() != "NODE_OPTIONS"}
    env["NODE_OPTIONS"] = "--import=" + hook.as_uri()
    env["HERMES_DESKTOP_NODE_ROOT"] = str(root)
    env["HERMES_NODE_SOURCE_ROOT"] = str(request.source)
    env["HERMES_NODE_DEPENDENCY_ROOT"] = str(root)
    env["HERMES_DESKTOP_SERIAL_CACHE"] = str(request.cache)
    return env


def identity_environment(request: BuildRequest, variant: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """One build-only bridge shared by dependency workers, stamp and packagers."""
    request.validate_channel()
    if request.channel_request is not None and variant != "bundled":
        raise ValueError("channel builds currently support only the bundled variant")
    env = dict(inherited)
    env.pop("HERMES_DESKTOP_BUILD_DIR", None)
    # The external Node tree is not yet bound to the preparation identity.
    # Never let an inherited value choose the build's dependencies.
    env.pop("HERMES_DESKTOP_NODE_ROOT", None)
    env.pop("HERMES_NODE_SOURCE_ROOT", None)
    env.pop("HERMES_NODE_DEPENDENCY_ROOT", None)
    if request.external_outputs:
        env["HERMES_DESKTOP_BUILD_DIR"] = str(request.desktop_build_dir)
    env.update(CI="true", PYTHONUTF8="1", GITHUB_SHA=request.commit,
               HERMES_DESKTOP_VARIANT=variant,
               HERMES_PAYLOAD_VERSION=request.version,
               HERMES_BUNDLE_ENV_JSON=json.dumps(request.bundle_env, sort_keys=True))
    if request.release_epoch is None:
        env.pop("HERMES_RELEASE_EPOCH", None)
    else:
        env["HERMES_RELEASE_EPOCH"] = str(request.release_epoch)
    env.pop("BUILD_NUMBER", None)
    env.pop("GITHUB_HEAD_REF", None)
    env.pop("_HERMES_CHANNEL_REQUEST_JSON", None)
    if request.channel_request is not None:
        env["_HERMES_CHANNEL_REQUEST_JSON"] = json.dumps(request.channel_request, sort_keys=True)
        env["GITHUB_REPOSITORY"] = request.channel_request["repository"]
        for key in ("HERMES_BUILD_COMMIT", "HERMES_PAYLOAD_TAG", "GITHUB_REF_NAME"):
            env.pop(key, None)
    elif request.tag is None:
        env["HERMES_BUILD_COMMIT"] = request.commit
        env.pop("HERMES_PAYLOAD_TAG", None)
        env.pop("GITHUB_REF_NAME", None)
    else:
        env.pop("HERMES_BUILD_COMMIT", None)
        env["HERMES_PAYLOAD_TAG"] = request.tag
        env["GITHUB_REF_NAME"] = request.tag
    return env


def validate_builder_identity(request: BuildRequest, args: list[str]) -> None:
    if request.channel_request is not None and any(
        re.match(r"(?:-c|--config)\.extraMetadata\.(?:version|shortVersion|shortVersionWindows)=", arg)
        for arg in args
    ):
        raise ValueError("channel package versions come only from the admitted request")


def validate_external_package_output(request: BuildRequest, args: list[str], inherited: Mapping[str, str]) -> None:
    if not request.external_outputs or not request.target.startswith("win32-"):
        return
    from pm.filesystem import is_junction, is_reparse_point

    outputs = [arg.split("=", 1)[1] for arg in args
               if re.match(r"(?:-c|--config)\.directories\.output=", arg)]
    expected = request.work.parent / "electron-output"
    if len(outputs) != 1 or Path(outputs[0]) != expected or (
        inherited.get("SEMANTIER_DESKTOP_BUILD_OUTPUT")
        and Path(inherited["SEMANTIER_DESKTOP_BUILD_OUTPUT"]) != expected
    ):
        raise ValueError("external Windows package output must be the job-local electron-output")
    if any(path.is_symlink() or is_junction(path) or is_reparse_point(path)
           for path in (expected, *expected.parents)):
        raise ValueError("external Windows package output is redirected")


def packaging_environment(build: Mapping[str, str], inherited: Mapping[str, str],
                          target: str) -> dict[str, str]:
    env = dict(build)
    if target.startswith("darwin-"):
        # Security.framework needs the login HOME for both key import and signing,
        # even with an explicit keychain. Keep dependency preparation isolated.
        env["HOME"] = inherited.get("HERMES_REAL_HOME") or inherited.get("HOME") or str(Path.home())
    return env


def select_variant(prepared: PreparedDesktop, variant: str | None) -> str:
    from scripts.termux.deb_version import channel_for_tag

    request = prepared.request
    variant = variant or request.variant
    if variant != request.variant and not (
        {variant, request.variant} == {"bundled", "store"} and request.tag
        and channel_for_tag(request.tag) == "stable"
    ):
        raise ValueError("requested variant was not prepared")
    return variant
