"""Build a complete desktop bundle with the shared Python payload tools."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("bundle: " + subprocess.list2cmdline(argv), flush=True)
    from scripts.bundles.runtime_cache import measure
    cache = env.get("HERMES_DESKTOP_SERIAL_CACHE")
    stage = next((Path(arg).name for arg in argv[1:] if arg.endswith((".py", ".mjs"))), Path(argv[0]).stem)
    with measure(Path(cache) if cache else None, stage):
        subprocess.run(argv, cwd=cwd, env=env, check=True)


def capture(argv: list[str], repo: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(argv, cwd=repo, env=env, text=True, encoding="utf-8").strip()


@contextmanager
def flavored_assets(icons: Path, assets: Path):
    """Hold the flavored icon set at its workspace path for the packaging window.

    electron-builder resolves every piece of packaging artwork from the
    workspace — the app icon, the icon.ico extraResource, the MSIX logo set
    before-build stages into build/appx, and the exe identity stamp — so a
    flavored (canary/commit) build cannot package from the rendered product
    directory alone. Restore the admitted files once packaging ends: the source
    custody check refuses a dirty checkout, and a build that leaves its own
    render behind breaks every later build in that tree and `hermes update`.
    """
    admitted = {path.relative_to(assets): path.read_bytes()
                for path in assets.rglob("*") if path.is_file()}
    try:
        shutil.copytree(icons, assets, dirs_exist_ok=True)
        yield
    finally:
        for path in assets.rglob("*"):
            if path.is_file() and path.relative_to(assets) not in admitted:
                path.unlink()
        for relative, data in admitted.items():
            path = assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def stage_external_assets(source: Path, rendered: Path, destination: Path) -> None:
    """Copy source assets and overlay generated icons into a fresh job directory."""
    from pm.filesystem import is_junction
    if destination.exists() or destination.is_symlink() or is_junction(destination):
        raise ValueError("external asset output already exists")
    for root in (source, rendered):
        if root.is_symlink() or is_junction(root) or not root.is_dir():
            raise ValueError("asset input is missing or redirected")
        if any(path.is_symlink() or is_junction(path) for path in root.rglob("*")):
            raise ValueError("asset input contains a redirected path")
    shutil.copytree(source, destination)
    shutil.copytree(rendered, destination, dirs_exist_ok=True)


def release_version(_repo: Path, tag: str) -> str:
    from scripts.termux.deb_version import channel_for_tag

    channel_for_tag(tag)  # shared release tag grammar, not a second version parser
    return tag[1:]


def npm_command(node: str) -> list[str]:
    # Dependency preparation and packaging must resolve the same npm identity.
    return [node, str(ROOT / "scripts/build/node-deps.mjs"), "--npm"]


def build(repo: Path, tag: str | None, variant: str, builder_args: list[str],
          commit_build: str | None = None) -> None:
    from scripts.bundles.desktop_prepare import BuildRequest, prepare
    from scripts.releases.bundle_env import decode
    request = BuildRequest.create(repo, tag=tag, commit=commit_build, variant=variant,
                                  work=repo / ".build/desktop-job", cache=repo / ".cache/desktop-inputs",
                                  bundle_env=decode(os.environ.get("HERMES_BUNDLE_ENV_JSON", "")))
    build_prepared(prepare(request), builder_args)


def build_prepared(path: Path, builder_args: list[str], variant: str | None = None) -> None:
    from scripts.bundles.desktop_prepare import PreparedDesktop
    from scripts.bundles.desktop_inputs import build_lock, require_external_output_readiness
    prepared = PreparedDesktop.load(path)
    require_external_output_readiness(prepared.request.external_outputs)
    with build_lock(prepared.request.source, external_cache=prepared.request.cache if prepared.request.external_outputs else None):
        _build_prepared(prepared, builder_args, variant)


def _build_prepared(prepared, builder_args: list[str], variant: str | None) -> None:
    prepared.validate()
    from scripts.bundles.desktop_inputs import (build_environment, packaging_environment, select_variant,
                                                validate_builder_identity, validate_external_package_output)
    from scripts.bundles.native import finish_native

    request = prepared.request
    validate_builder_identity(request, builder_args)
    validate_external_package_output(request, builder_args, os.environ)
    if request.channel_request is not None and not request.target.startswith(("darwin-", "win32-")):
        raise ValueError("channel builds require a supported native macOS or Windows target")
    variant = select_variant(prepared, variant)
    repo, node = request.source, str(prepared.node)
    env = build_environment(prepared, variant, os.environ)
    if request.release_epoch is not None:
        env["HERMES_RELEASE_EPOCH"] = str(request.release_epoch)
    desktop = repo / "apps/desktop"
    targets = {"win32": ["--win", "msix"], "darwin": ["--mac", "dmg", "zip"], "linux": ["--linux", "AppImage"]}[sys.platform]
    app_version = env.get("SEMANTIER_DESKTOP_VERSION") or request.version
    if app_version != request.version and (sys.platform != "win32" or request.tag is not None or request.channel_request is not None):
        raise ValueError("Semantier app version is only valid for local Windows commit builds")
    package_args = ["--prepared", str(prepared.packager), "--native-deps", str(prepared.native),
                    *targets, f"-c.extraMetadata.version={app_version}"]
    run([node, "scripts/run-electron-builder.mjs", "--validate-only", *package_args, *builder_args],
        cwd=desktop, env=env)
    dependency_root = prepared.node_root if request.external_outputs else repo
    run([node, "scripts/build/node-deps.mjs", "--source", str(dependency_root), "--reuse", "--no-install",
         "--native-toolchain", prepared.native_toolchain,
         *[arg for name in request.workspaces() for arg in ("--workspace", name)]], cwd=repo, env=env)
    build_dir = request.desktop_build_dir
    products = build_dir / "products"
    dist_dir = build_dir / "dist" if request.external_outputs else desktop / "dist"
    icons = products / "icons"
    run([str(prepared.icon_python), "-I", str(repo / "scripts/generate_icons.py"),
         "--source", str(repo), "--out", str(icons)], cwd=repo, env=env)
    if variant != "light":
        run([node, "scripts/build/tui.mjs", "--source", str(repo), "--out", str(products / "tui")], cwd=repo, env=env)
        run([node, "scripts/build/web.mjs", "--source", str(repo), "--icons", str(products / "icons"),
             "--out", str(products / "web")], cwd=repo, env=env)
        if prepared.payload is None:
            raise ValueError("payload dependencies were not prepared")
        if finish_native(prepared.payload, {"tui": products / "tui", "web": products / "web"}):
            raise RuntimeError("prepared payload assembly failed")
    from scripts.bundles.desktop_prepare import require_source
    require_source(repo, request.commit)
    # The admitted checkout stays clean across the whole build: packaging reads
    # its artwork from the workspace, so the render lives there only while
    # electron-builder holds it, and the final check proves it was handed back.
    # Stamp before the render lands: the stamp's git check would call the
    # checkout dirty while the flavored icons sit over committed files.
    stamp_args = ["--output-dir", str(build_dir)] if request.external_outputs else []
    run([node, "scripts/write-build-stamp.mjs", *stamp_args], cwd=desktop, env=env)
    if request.external_outputs:
        stage_external_assets(desktop / "assets", icons / "apps/desktop/assets", build_dir / "assets")
    assets_context = nullcontext() if request.external_outputs else flavored_assets(icons / "apps/desktop/assets", desktop / "assets")
    with assets_context:
        run([node, "scripts/build/desktop.mjs", "--source", str(repo), "--icons", str(icons),
             "--stamp", str(build_dir / "install-stamp.json"), "--native-deps", str(prepared.native),
             "--out", str(dist_dir)], cwd=repo, env=env)
        version_args = []
        if sys.platform == "win32" and request.channel_request is None and request.tag is not None:
            script = "const m=require('./scripts/msix-shared.mjs');console.log(m.nativeQuad(process.argv[1], Number(process.env.HERMES_RELEASE_EPOCH)))"
            quad = capture([node, "-e", script, request.tag], repo, env)
            version_args = [f'-c.extraMetadata.shortVersion={quad}', f'-c.extraMetadata.shortVersionWindows={quad}']
        run([node, "scripts/run-electron-builder.mjs", *package_args, *version_args, *builder_args], cwd=desktop,
            env=packaging_environment(env, os.environ, request.target))
    require_source(repo, request.commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=False,
                        help="Release tag (vX.Y.Z / canary), exclusive with --commit/--channel-request")
    parser.add_argument("--commit", dest="commit_build", default=None,
                        help="Commit-only build: exact full 40-char SHA the checkout is at; "
                             "version comes from the target pyproject, no tag is referenced")
    parser.add_argument("--release-commit", help="Admitted commit for a stable tag not created until green")
    parser.add_argument("--channel-request", type=Path, help="Immutable admitted channel request JSON")
    parser.add_argument("--variant", choices=["bundled", "store", "light"])
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--external-outputs", action="store_true", help="Place Desktop build products inside the preparation job")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="Remove this checkout's previous build outputs, under the checkout lock, before preparing")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("builder_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    builder_args = [v for v in args.builder_args if v != "--"]
    try:
        from scripts.bundles.desktop_inputs import require_external_output_readiness
        require_external_output_readiness(args.external_outputs)
        if args.prepared:
            if (args.tag or args.commit_build or args.release_commit or args.channel_request
                    or args.prepare_only or args.work or args.cache or args.clean or args.external_outputs):
                parser.error("--prepared supplies the complete build request")
            build_prepared(args.prepared, builder_args, args.variant)
        else:
            from scripts.bundles.desktop_prepare import BuildRequest, prepare
            from scripts.releases.bundle_env import decode
            request = BuildRequest.create(args.repo, tag=args.tag, commit=args.commit_build,
                                          variant=args.variant or "bundled",
                                          work=args.work or args.repo / ".build/desktop-job",
                                          cache=args.cache or args.repo / ".cache/desktop-inputs",
                                          bundle_env=decode(os.environ.get("HERMES_BUNDLE_ENV_JSON", "")),
                                          channel_request=json.loads(args.channel_request.read_text(encoding="utf-8-sig"))
                                          if args.channel_request else None,
                                          release_commit=args.release_commit, external_outputs=args.external_outputs)
            if args.prepare_only and builder_args:
                parser.error("builder arguments belong to the build phase")
            result = prepare(request, clean=args.clean)
            if args.prepare_only:
                print(result)
            else:
                build_prepared(result, builder_args)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"desktop build: {exc}\n")


if __name__ == "__main__":
    main()
