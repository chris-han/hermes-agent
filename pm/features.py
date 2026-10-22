"""The enabled-features file: the exact extras the install carries.

The bundle is built with `uv sync --all-extras` minus the declared
opt-in extras (`opt_in_extras`), but WHICH extras
actually resolve differs per platform (markers gate some off). The
shipped default is the EXACT list that installed — written at bundle
time beside the payload, read at sync time:

- lazy installs OFF (security.allow_lazy_installs: false): the feature
  list is FROZEN to this file; pm sync never deviates from it and never
  installs a plugin dep.
- lazy installs ON: the file is the baseline; extras union on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FEATURES_FILENAME = "enabled-features.json"


class FeatureProbeError(RuntimeError):
    """A failed inventory must not become an empty feature list."""


def features_path(base_dir: Optional[Path] = None) -> Path:
    """Where the enabled-features file lives. In a bundle, the payload
    root (beside manifest.json — bundle-written, sealed-shipped). At
    runtime, the runtime dir (beside the byte store — per-install,
    writable on every install kind). Same relative location on both:
    a sealed install's store_root() is <payload>/tools, and the file
    sits at the payload root = store_root().parent."""
    if base_dir is not None:
        return base_dir / FEATURES_FILENAME
    from pm.paths import store_root

    return store_root().parent / FEATURES_FILENAME


def write_features(extras: list[str], base_dir: Optional[Path] = None) -> Path:
    """Record the exact extras an install carries. Sorted, deduped."""
    path = features_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": 1, "extras": sorted(set(extras))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_features() -> Optional[list[str]]:
    """The frozen/baseline feature list; None when no file exists (source
    installs without a bundle — the baseline is the recorded venv
    state)."""
    try:
        data = json.loads(features_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    extras = data.get("extras")
    if not isinstance(extras, list):
        return None
    return sorted({str(e) for e in extras})


def declared_extras(repo_dir: Path) -> list[str]:
    """Every extra the repo's pyproject declares."""
    import tomllib

    with (repo_dir / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def opt_in_extras(repo_dir: Path) -> list[str]:
    """Extras that only an explicit selection installs, never an all-extras build.

    ``[tool.hermes] opt-in-extras`` in the repo's pyproject. They stay
    installable through ``sync_venv([extra])``; bundles and other
    ``--all-extras`` builds leave them out, so their closures are absent
    from shipped payloads and from the recorded feature list.
    """
    import tomllib

    with (repo_dir / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    names = data.get("tool", {}).get("hermes", {}).get("opt-in-extras", [])
    declared = set(data.get("project", {}).get("optional-dependencies", {}))
    unknown = sorted(set(names) - declared)
    if unknown:
        raise ValueError(f"[tool.hermes] opt-in-extras names undeclared extras: {unknown}")
    return sorted(set(names))


def installed_extras(repo_dir: Path, venv_dir: Path, *, python_exe: Path) -> list[str]:
    """Inventory every required anchor in one isolated target process.

    The staged interpreter owns the site layout. Only the selected tree's
    .pth files are processed, so editable packages keep their launch behavior.
    """
    import subprocess

    from pm.extras import _anchors

    required = {extra: _anchors(extra) for extra in declared_extras(repo_dir)}
    anchors = sorted({anchor for group in required.values() for anchor in group})
    probe = """
import contextlib, importlib.util, json, os, site, sys, sysconfig
base, anchors = sys.argv[1], json.loads(sys.argv[2])
paths = sysconfig.get_paths(vars={"base": base, "platbase": base})
sites = dict.fromkeys(paths[key] for key in ("purelib", "platlib"))
if not any(os.path.isdir(path) for path in sites):
    raise RuntimeError("target dependency tree has no site-packages")
result = {}
with contextlib.redirect_stdout(sys.stderr):
    for path in sites:
        site.addsitedir(path)
    for anchor in anchors:
        try:
            result[anchor] = importlib.util.find_spec(anchor) is not None
        except (ImportError, ValueError):
            result[anchor] = False
print(json.dumps(result))
"""
    try:
        child = subprocess.run(
            [str(python_exe), "-B", "-I", "-S", "-c", probe,
             str(venv_dir.resolve()), json.dumps(anchors)],
            cwd=repo_dir, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        raise FeatureProbeError(f"feature inventory failed on {python_exe}: {detail}") from exc
    try:
        resolved = json.loads(child.stdout)
    except ValueError as exc:
        raise FeatureProbeError("feature inventory returned invalid JSON") from exc
    if (not isinstance(resolved, dict) or set(resolved) != set(anchors)
            or any(type(value) is not bool for value in resolved.values())):
        raise FeatureProbeError("feature inventory returned incomplete anchor results")
    return sorted(extra for extra, group in required.items() if all(resolved[anchor] for anchor in group))


def distribution_extras(repo_dir: Path, *, target: str) -> list[str] | None:
    """The declared Windows product selection; other source/target builds are unchanged."""
    if target != "win32-x64":
        return None
    import tomllib
    with (repo_dir / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)
    profile = data.get("tool", {}).get("hermes", {}).get("windows-cpu-distribution")
    if profile is None:
        return None
    extras = profile.get("extras")
    declared = data.get("project", {}).get("optional-dependencies", {})
    if (not isinstance(extras, list) or not extras or
            any(not isinstance(name, str) or name not in declared for name in extras)):
        raise ValueError("invalid Windows CPU distribution extras")
    return sorted(set(extras))


def runtime_distribution_extras() -> list[str] | None:
    """Only a sealed product constrains source capability and lazy acquisition."""
    from pm.paths import repo_root
    from pm.store import current_target
    from hermes_cli.steward import is_bundled_payload
    root = repo_root()
    if not is_bundled_payload(root):
        return None
    return distribution_extras(root, target=current_target())


def installed_package_names(venv_dir: Path, *, python_exe: Path) -> set[str]:
    """Inventory resolved distributions in the actual target environment for its seed."""
    import subprocess
    probe = """
import importlib.metadata, json, sys, sysconfig
paths = sysconfig.get_paths(vars={"base": sys.argv[1], "platbase": sys.argv[1]})
print(json.dumps(sorted({d.metadata['Name'] for d in importlib.metadata.distributions(
    path=list(dict.fromkeys(paths[k] for k in ('purelib', 'platlib'))))})))
"""
    result = subprocess.run([str(python_exe), "-I", "-S", "-c", probe, str(venv_dir)],
                            capture_output=True, text=True, check=True, timeout=60)
    names = json.loads(result.stdout)
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise FeatureProbeError("invalid installed distribution inventory")
    return set(names)
