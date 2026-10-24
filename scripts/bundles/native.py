"""Native payload staging through PM's existing package authority."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from pm.install import _lockfile
from pm.lock import Facts
from pm.registry import get_package, walk
from pm.store import current_target

def _bundle_package_names() -> list[str]:
    names = [
        n
        for n in _lockfile().names()
        if not get_package(n).internal or n == "uv"
    ]
    from pm.features import distribution_extras
    from pm.paths import repo_root
    if distribution_extras(repo_root(), target=current_target()) is not None:
        names = [name for name in names if not name.startswith("llamacpp-")]
    if "python" not in names:
        names.append("python")
    return names


def _arch_guard(store_dir: Path) -> list[str]:
    """Every staged binary must be built for this machine's target — a
    payload staged with a mismatched interpreter or PATH tool ships an
    artifact that cannot run. Reads facts, probes each entry binary."""
    from pm.package import machine_matches_binary

    facts = Facts(store_dir / "facts.json")
    problems = []
    target = current_target()
    for name in _lockfile().names():
        package = get_package(name)
        fact = facts.get(name)
        if fact is None or "entry" not in fact:
            continue
        binary = package.binary(store_dir / fact["entry"], target)
        if binary is None or not binary.is_file():
            continue
        verdict = machine_matches_binary(binary, target)
        # A package that declares this target as emulated (x64 binary run
        # under Windows ARM64 built-in emulation) is fine with the x64 PE.
        if verdict is False and target not in package.emulated_arch_targets:
            problems.append(f"{name}: {binary.name} is not a {target} binary")
    return problems



from pm.uv_cache_prune import lock_package_names, prune_uv_cache_to_lock

__all__ = ["lock_package_names", "prune_uv_cache_to_lock", "stage_uv_cache"]


def stage_uv_cache(source: Path, destination: Path) -> None:
    """Ship everything a per-install venv rebuild resolves from offline.

    sdist source trees (including Rust target/ outputs) are build-only bulk —
    offline rebuilds install the built wheel entries, never recompile from
    source — so they stay out. Built wheel ZIPs (including the copies uv keeps
    beside sdist metadata.msgpack shards) ship: without them an offline
    rebuild falls back to wanting the sdist download and fails closed.
    """
    def ignore(directory: str, names: list[str]) -> set[str]:
        path = Path(directory)
        parts = path.relative_to(source).parts
        if not parts or not parts[0].startswith("sdists-v"):
            return set()
        # Source trees live under a revision selected by a sibling pointer.
        if "src" in names and (path / "src").is_dir() and any(
            (path.parent / pointer).is_file() for pointer in ("revision.http", "revision.rev")
        ):
            return {"src"}
        return set()

    shutil.copytree(source, destination, ignore=ignore)



def compact_uv_cache(cache: Path) -> None:
    """Collapse a staged cache's Windows filesystem entries, keeping every wheel.

    Only the owned staging copy is replaced. Verify the completed archive before
    removing loose entries; build-host caches and non-Windows seeds stay intact.
    """
    import zipfile
    from pm.filesystem import is_junction, remove_tree

    archive = cache / "seed.zip"
    temporary = cache / ".seed.zip.tmp"
    if archive.exists() or temporary.exists():
        raise ValueError("cache archive already exists")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, strict_timestamps=False) as output:
            for directory, dirs, files in os.walk(cache):
                for name in sorted(dirs + files):
                    entry = Path(directory) / name
                    if entry == temporary:
                        continue
                    if entry.is_symlink() or is_junction(entry):
                        raise ValueError(f"linked staged cache entry: {entry}")
                    output.write(entry, entry.relative_to(cache).as_posix())
        with zipfile.ZipFile(temporary) as output:
            bad = output.testzip()
            if bad:
                raise ValueError(f"cache archive integrity failed: {bad}")
        temporary.replace(archive)
        for entry in cache.iterdir():
            if entry != archive:
                if entry.is_dir():
                    remove_tree(entry)
                else:
                    entry.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def verify_offline_cache(root: Path, python: Path, env: dict) -> None:
    """Exercise the real writable repair workspace before sealing Windows data."""
    from pm import build_environment
    from pm.workspace import _generate_pyproject

    with tempfile.TemporaryDirectory(prefix=".offline-cache-check-", dir=root.parent) as scratch:
        workspace = Path(scratch) / "project"
        _generate_pyproject([], workspace, source=root / "hermes-agent")
        shutil.copy2(root / "hermes-agent/uv.lock", workspace / "uv.lock")
        build_environment(source=workspace, python=python, out=Path(scratch) / "venv",
                          cache=root / "uv-cache", env=env, all_extras=True,
                          sealed=True, offline=True, explicit=True, timeout=20 * 60)


def stage_distribution_assets(repo: Path, cache: Path) -> None:
    """Verify the parent composition and stage its product backend and pinned CPU model."""
    import hashlib
    import json
    from pm.features import distribution_extras
    from pm.downloader import Download, Source
    if distribution_extras(repo, target=current_target()) is None:
        return
    parent = repo / "semantier-runtime"
    manifest = json.loads((parent / "composition.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = parent / entry["path"]
        if not path.resolve().is_relative_to(parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("parent composition integrity mismatch: " + entry["path"])
    # Product-owned implementation stays in the parent composition. Only the
    # staged distribution exposes it through Hermes' bundled backend owner;
    # copying into user plugins would require a separate user activation.
    plugin = parent / "src/plugins/sensevoice_stt"
    destination_plugin = repo / "plugins/sensevoice_stt"
    destination_plugin.mkdir(parents=True)  # refuse an unexpected collision
    for name in ("__init__.py", "provider.py", "plugin.yaml"):
        relative = str((plugin / name).relative_to(parent)).replace("\\", "/")
        if not any(entry["path"] == relative for entry in manifest["files"]):
            raise ValueError("unbound product plugin input: " + relative)
        shutil.copyfile(plugin / name, destination_plugin / name)
    model_cache = cache.parent / "semantier-sensevoice-assets"
    model_cache.mkdir(parents=True, exist_ok=True)
    sources = []
    for entry in manifest["models"]:
        if entry["name"] not in {"model.int8.onnx", "tokens.txt"}:
            raise ValueError("unselected model asset in Windows CPU composition")
        sources.append(Source(entry["url"], model_cache / entry["name"], entry["sha256"]))
    Download(sources, partials_dir=model_cache / ".partials").run()
    destination = parent / "assets/sensevoice"
    destination.mkdir(parents=True)
    for entry in manifest["models"]:
        path = model_cache / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("pinned model size mismatch: " + entry["name"])
        shutil.copyfile(path, destination / entry["name"])


def stage_pm_runtime(root: Path, python: Path, repo: Path, *, offline: bool = False,
                     cache: Path | None = None) -> None:
    """Publish the same PM dependency graph as source installs, ready offline."""
    from pm import stage_manager_runtime
    from scripts.bundles.payload import seal_pm_runtime

    destination = root / "pm-runtime"
    if destination.exists():
        shutil.rmtree(destination)
    stage_manager_runtime(python=python, destination=destination, project=repo / "pm", offline=offline, cache=cache)
    seal_pm_runtime(root, python)


def stage_native(args) -> int:
    """Isolate HOME and PM state, but retain the provider's reusable build cache."""
    from pm.packages import uv_cache_dir

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[2]
    # Resolve before HOME isolation so the build warms the cache CI saves.
    cache = Path(getattr(args, "cache", None) or os.environ.get("UV_CACHE_DIR") or uv_cache_dir()).resolve()
    base_env = dict(os.environ)
    if current_target() == "win32-arm64":
        from pm.native_build import prepare_windows_environment

        base_env = prepare_windows_environment(source=root, state=out.parent / ".build-deps", env=base_env)
    # Rustup resolves its installed toolchain under HOME unless these are explicit.
    # Preserve the build host's compiler and Cargo cache before isolating app state.
    for key, directory in (("CARGO_HOME", ".cargo"), ("RUSTUP_HOME", ".rustup")):
        base_env.setdefault(key, str(Path.home() / directory))
    with tempfile.TemporaryDirectory(prefix=".build-", dir=out) as work:
        env = {**base_env, "HOME": work, "USERPROFILE": work,
               "HERMES_HOME": str(Path(work) / ".hermes"),
               "HERMES_RUNTIME_DIR": str(out / "tools"),
               "HERMES_PYTHON_SRC_ROOT": str(root),
               "XDG_CACHE_HOME": str(Path(work) / "cache"),
               "XDG_CONFIG_HOME": str(Path(work) / "config"),
               "UV_CACHE_DIR": str(cache),
               "PYTHONPATH": os.pathsep.join([str(root), *filter(None, sys.path)])}
        command = [sys.executable, "-B", "-m", "scripts.bundles.native", "--out", str(out),
                   "--ref", args.ref or "HEAD", "--source", str(root)]
        for name, product in getattr(args, "frontends", {}).items():
            command += [f"--{name}", str(product)]
        return subprocess.run(command, cwd=root, env=env, check=False).returncode


def prune_staged_store(store_dir: Path, names: list[str]) -> None:
    """Prune this build's store, never the ambient user's tool store."""
    facts = Facts(store_dir / "facts.json", strict=True)
    facts.retain({package.name for package in walk(names)})
    keep = facts.entries_in_use()
    for entry in store_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in keep:
            shutil.rmtree(entry)


def prepare_native(*, out: Path, ref: str, source: Path, cache: Path,
                   tools: Path | None = None, env: dict | None = None,
                   runtime_cache: tuple[Path, str] | None = None,
                   dependency_identity: str | None = None) -> Path:
    """Prepare final payload dependencies inside the caller's isolated PM process.

    The caller supplies its compiler environment; only the full standalone
    wrapper provisions machine prerequisites and isolates HOME.
    """
    from scripts.bundles.native_prepared import preparation_lock, prepared_path

    out = Path(out).absolute()
    if out != out.resolve():
        raise ValueError("symlinked native output")
    out.mkdir(parents=True, exist_ok=True)
    with preparation_lock(out):
        from scripts.bundles.runtime_cache import retained_runtime
        owned_candidate = retained_runtime(out, Path(source).resolve())
        if runtime_cache is not None and any(out.iterdir()) and not owned_candidate:
            raise ValueError('nonempty native output has no admitted retention ownership')
        prepared_path(out).unlink(missing_ok=True)
        (out / "manifest.json").unlink(missing_ok=True)
        from scripts.bundles.runtime_cache import measure
        log_root = Path(env["HERMES_DESKTOP_SERIAL_CACHE"]) if env and env.get("HERMES_DESKTOP_SERIAL_CACHE") else None
        def cold(reuse_dependencies=False):
            with measure(log_root, "runtime.prepare"):
                extra = {"reuse_dependencies": True} if reuse_dependencies else {}
                return _prepare_native(out=out, ref=ref, source=Path(source).resolve(),
                                       cache=Path(cache).resolve(), tools=tools, env=env, **extra)
        if runtime_cache is None:
            return cold()
        from scripts.bundles.runtime_cache import cache_lock, restore_cache, publish_cache
        entry, key = runtime_cache
        with cache_lock(entry):
            restored = restore_cache(entry, key, out, Path(source).resolve(), ref)
            from scripts.bundles.runtime_cache import admit_dependencies, publish_dependencies, retain_runtime
            reuse = (restored is None and owned_candidate and dependency_identity is not None
                     and admit_dependencies(out, Path(source).resolve(), dependency_identity))
            if restored is None and owned_candidate and not reuse:
                from pm.filesystem import remove_tree
                from scripts.bundles.runtime_cache import revoke_retention
                same_key = retained_runtime(out, Path(source).resolve(), key)
                revoke_retention(out)
                remove_tree(out)
                out.mkdir()
                if not same_key:
                    restored = restore_cache(entry, key, out, Path(source).resolve(), ref)
            if restored is not None:
                if dependency_identity is not None:
                    from scripts.bundles.native_prepared import load_prepared
                    # This interpreter may also render build icons before assembly.
                    # Admit retained bytes before any Python execution.
                    load_prepared(restored)
                    publish_dependencies(restored, dependency_identity)
                print(f"runtime: reusing sealed preparation candidate {key}; byte admission follows", flush=True)
                return restored
            result = cold(reuse_dependencies=reuse)
            if dependency_identity is not None:
                publish_dependencies(result, dependency_identity)
                # The fixed-layout dependency environment is already sealed and retained.
                # Keep it through interrupted packaging without writing another multi-GB
                # backend-bound archive; legacy whole-archive callers remain supported.
                retain_runtime(out, Path(source).resolve(), key)
                from scripts.bundles.runtime_cache import diagnostic
                diagnostic(entry.parents[2], result='dependency-hit' if reuse else 'dependency-published', key=dependency_identity)
            else:
                publish_cache(entry, key, out, result)
            return result


def _prepare_native(*, out: Path, ref: str, source: Path, cache: Path,
                    tools: Path | None, env: dict | None, reuse_dependencies: bool = False) -> Path:
    from pm import paths
    from pm.package import InstallError

    from scripts.bundles.runtime_cache import measure
    log_root = Path(env["HERMES_DESKTOP_SERIAL_CACHE"]) if env and env.get("HERMES_DESKTOP_SERIAL_CACHE") else None
    store_dir = out / "tools"
    store_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = out / "hermes-agent"
    from scripts.bundles.payload import INERT_SNAPSHOT_DIRS, snapshot
    print(f"staging repo snapshot ({ref})…", flush=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=source,
        text=True, encoding="utf-8").strip()
    with measure(log_root, "runtime.snapshot"):
        snapshot(source, revision, repo_dir, exclude=INERT_SNAPSHOT_DIRS)
        stage_distribution_assets(repo_dir, cache)
    # PM's provider code reads its adjacent lock. Never combine that tool graph
    # with a revision selecting different pins.
    if (repo_dir / "pm/lock.json").read_bytes() != paths.lockfile_path().read_bytes():
        raise ValueError("selected revision's PM lock differs from the builder; use a checkout at that revision")
    build_env = os.environ if env is None else env

    names = [
        n for n in _bundle_package_names()
        if get_package(n).missing_reason(current_target()) is None
    ]
    from pm import prepare_tools, stage_tools

    if not reuse_dependencies:
        with measure(log_root, "runtime.tools.prepare"):
            prepare_tools(names, out=Path(tools) if tools is not None else store_dir,
                          target=current_target(), cache=cache)
        if tools is not None:
            with measure(log_root, "runtime.tools.copy"):
                stage_tools(names, source_store=Path(tools), out=store_dir, target=current_target())

        # Prune the staged store BEFORE the venv sync and packaging: drop the
        # fetch-<sha> download-cache archives (needed only at install time — dead
        # weight in the shipped payload AND in the CI cache that restores this
        # dir) and any orphaned package versions left over from an older lock
        # the cache carried in. A lean staged store = a lean CI cache.

        # Only this build's store is ours to prune; machine-wide partials are not.
        # Cached facts may still name packages removed from the current selection.
        # Retain the dependency closure before using facts as the deletion roots.
        prune_staged_store(store_dir, names)


    facts = Facts(store_dir / "facts.json")
    python_fact = facts.get("python")
    if python_fact is None:
        raise InstallError("venv", "no staged interpreter to build on")
    python_bin = get_package("python").binary(
        store_dir / python_fact["entry"], current_target()
    )

    if python_bin is None:
        raise FileNotFoundError("staged Python executable is missing")
    venv_dir = out / "venv"
    if not reuse_dependencies:
        with measure(log_root, "runtime.pm-environment"):
            stage_pm_runtime(out, python_bin, repo_dir, cache=cache)
        print("✓ pm-runtime (independent locked dependencies)", flush=True)

        # Build + sync INSIDE the staged repo: the editable project install
        # must point at the payload's own tree, not this checkout.
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        env = dict(build_env)
        from pm import build_environment

        # Cold native wheels need a larger budget than interactive installs.
        with measure(log_root, "runtime.environment"):
            build_environment(source=repo_dir, python=python_bin, out=venv_dir,
                              env=env, cache=cache, all_extras=True, sealed=True, explicit=True,
                              timeout=2 * 60 * 60)
    print("✓ venv (declared distribution selection, on the staged interpreter)")

    # Inventory the staged interpreter before publishing the bundle contract.
    from pm.features import installed_extras, write_features, distribution_extras, installed_package_names

    features = installed_extras(repo_dir, venv_dir, python_exe=python_bin)
    selected = distribution_extras(repo_dir, target=current_target())
    if selected is not None:
        features = sorted(set(features) & set(selected))
    write_features(features, out)

    # The payload venv was built from this exact lock and feature set. Publish
    # that state so the first runtime sync can reuse the shipped environment
    # instead of rebuilding it before any plugin dependencies are requested.
    from pm.packages import Venv

    facts.record_state("venv", Venv(repo_dir).expected_stamp(features, plugin_dirs=[]), features)
    print(f"✓ enabled-features.json ({len(features)} extras recorded)")

    # Ship the complete lock-scoped offline cache (only sdist source
    # trees are excluded; built wheel ZIPs remain). Any per-install venv rebuild —
    # plugin extras, feature changes — resolves entirely from the shipped
    # wheels with zero network access, on every profile of this install.
    payload_cache = out / "uv-cache"
    if not reuse_dependencies:
        if payload_cache.exists():
            shutil.rmtree(payload_cache, ignore_errors=True)
        src_cache = cache
        if src_cache.is_dir():
            print(f"  uv-cache: copying {src_cache} → payload...", flush=True)
            with measure(log_root, "runtime.uv-cache"):
                stage_uv_cache(src_cache, payload_cache)
                resolved = installed_package_names(venv_dir, python_exe=python_bin) if selected is not None else None
                pruned = prune_uv_cache_to_lock(payload_cache, repo_dir, selected_packages=resolved)
            print(f"✓ uv-cache (lock-scoped: pruned {pruned} stale entries; offline rebuilds resolve from shipped wheels)", flush=True)
        else:
            raise InstallError("uv-cache", "runtime dependency cache is missing")

        if current_target().startswith("win32-"):
            with measure(log_root, "runtime.offline-cache-validation"):
                verify_offline_cache(out, python_bin, env)
            print("✓ offline cache (full frozen distribution-selection writable rebuild)", flush=True)

        bad = _arch_guard(store_dir)
        for line in bad:
            print(f"✗ arch: {line}")
        if bad:
            raise InstallError("tools", "native architecture verification failed")
        from scripts.bundles.payload import record_tools
        recorded = {name: fact["entry"] for name in names if (fact := facts.get(name)) and "entry" in fact}
        with measure(log_root, "runtime.tool-digests"):
            record_tools(out, paths.lockfile_path(), current_target(), recorded)
    from scripts.build.inputs import AgentInputs, RESOURCE_ENV, dependency_site
    from scripts.bundles.native_prepared import publish_prepared

    site = dependency_site(venv_dir, python_fact["version"], current_target())
    (site / "hermes-agent.pth").write_text(
        Path(os.path.relpath(repo_dir, site)).as_posix() + "\n", encoding="utf-8")
    from scripts.bundles.payload import relativize_links
    with measure(log_root, "runtime.links"):
        relativize_links(out)
    from scripts.bundles.bytecode import bake_bytecode
    with measure(log_root, "runtime.bytecode"):
        baked = bake_bytecode(out, python_bin, **({"source_only": True} if reuse_dependencies else {}))
    print(f"✓ baked bytecode ({baked['modules']} modules, unchecked-hash, read-only caches)")
    inputs = AgentInputs(
        project=repo_dir / "pyproject.toml", code=repo_dir, repo="hermes-agent",
        placement="contained", target=current_target(), python=python_bin,
        site_packages=site, environment=venv_dir,
        tools=store_dir, pm_runtime=out / "pm-runtime", ref=ref,
        resources={name: repo_dir / name for name in RESOURCE_ENV},
        features=out / "enabled-features.json",
    )
    with measure(log_root, "runtime.seal"):
        return publish_prepared(out, source, revision, inputs)


def finish_native(prepared: Path, frontends: dict[str, Path]) -> int:
    """Consume verified job-local paths. No dependency acquisition or repair."""
    from scripts.build.agent import assemble
    from scripts.build.inputs import AgentInputs
    from scripts.bundles.native_prepared import load_prepared, preparation_lock

    prepared = Path(prepared).absolute()
    if not prepared.name.endswith(".prepared.json") or not prepared.is_file():
        raise ValueError("native preparation is missing or invalid; run preparation again")
    out = prepared.with_name(prepared.name.removesuffix(".prepared.json"))
    with preparation_lock(out):
        (out / "manifest.json").unlink(missing_ok=True)
        from scripts.bundles.runtime_cache import measure
        serial = os.environ.get("HERMES_DESKTOP_SERIAL_CACHE")
        log_root = Path(serial) if serial else None
        with measure(log_root, "runtime.admission"):
            inputs = load_prepared(prepared)
        values = asdict(inputs)
        values["frontends"] = {name: Path(path).absolute() for name, path in frontends.items()}
        print("assembling verified payload", flush=True)
        with measure(log_root, "runtime.assembly"):
            assemble(AgentInputs.from_dict(values), out)
    print(f"✓ manifest ({out / 'manifest.json'})")
    return 0


def _stage_native(args) -> int:
    from pm import paths
    from pm.features import FeatureProbeError
    from pm.package import InstallError

    try:
        prepared = prepare_native(
            out=Path(args.out), ref=args.ref or "HEAD",
            source=getattr(args, "source", None) or paths.repo_root(),
            cache=Path(getattr(args, "cache", None) or os.environ["UV_CACHE_DIR"]),
            tools=getattr(args, "tools", None),
        )
        return finish_native(prepared, getattr(args, "frontends", {}))
    except (InstallError, FeatureProbeError) as exc:
        print(f"✗ {exc}")
        return 1




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tui", type=Path)
    parser.add_argument("--web", type=Path)
    args = parser.parse_args()
    args.frontends = {name: path for name in ("tui", "web") if (path := getattr(args, name)) is not None}
    return _stage_native(args)


if __name__ == "__main__":
    raise SystemExit(main())
