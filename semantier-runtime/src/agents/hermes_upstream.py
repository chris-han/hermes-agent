"""Pinned external Hermes discovery. Never imports Hermes or provisions dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM_URL = 'https://github.com/NousResearch/hermes-agent.git'
UPSTREAM_COMMIT = '26ba38a098d8a68c5a260133bcf858a444d0b1b5'
UPSTREAM_TREE = '5f62d8d02aaef557ce5637631023e3196a715f6d'


class UpstreamConfigurationError(ValueError):
    pass


def runtime_roots() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['/usr/bin/git', '-C', str(root), 'rev-parse', '--git-common-dir'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        common = Path(result.stdout.strip())
        common = common if common.is_absolute() else root / common
        return root, common.resolve().parent
    return (root,)


@dataclass(frozen=True)
class UpstreamInstallation:
    root: Path
    python: Path
    commit: str = UPSTREAM_COMMIT
    tree: str = UPSTREAM_TREE

    @classmethod
    def from_environment(cls) -> UpstreamInstallation:
        root = os.environ.get('SEMANTIER_HERMES_UPSTREAM_ROOT', '').strip()
        if not root:
            raise UpstreamConfigurationError(
                'SEMANTIER_HERMES_UPSTREAM_ROOT required; run scripts/bootstrap_hermes_upstream.py explicitly')
        path = Path(root).expanduser().absolute()
        python = Path(os.environ.get('SEMANTIER_HERMES_UPSTREAM_PYTHON') or
                      path / '.venv/bin/python').expanduser().absolute()
        return cls(path, python)

    def verify_location(self) -> None:
        for runtime in runtime_roots():
            if self.root.resolve().is_relative_to(runtime):
                raise UpstreamConfigurationError('UPSTREAM_MUST_BE_OUTSIDE_RUNTIME')
            if self.python.absolute().is_relative_to(runtime):
                raise UpstreamConfigurationError('UPSTREAM_PYTHON_MUST_BE_OUTSIDE_RUNTIME')

    def verify(self) -> None:
        self.verify_location()
        def git(*args):
            result = subprocess.run(['/usr/bin/git', '-C', str(self.root), *args],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise UpstreamConfigurationError('UPSTREAM_SOURCE_NOT_READY')
            return result.stdout.strip()
        if git('rev-parse', 'HEAD') != self.commit or git('rev-parse', 'HEAD^{tree}') != self.tree:
            raise UpstreamConfigurationError('UPSTREAM_PIN_MISMATCH')
        if git('status', '--porcelain', '--untracked-files=all'):
            raise UpstreamConfigurationError('UPSTREAM_SOURCE_DIRTY')
        if not (self.root / 'hermes_cli/main.py').is_file():
            raise UpstreamConfigurationError('UPSTREAM_CLI_NOT_READY')
        if not self.python.is_file() or not os.access(self.python, os.X_OK):
            raise UpstreamConfigurationError('UPSTREAM_PYTHON_NOT_READY')

    def command(self, *args: str) -> list[str]:
        return [str(self.python), '-m', 'hermes_cli.main', *args]

    def environment(self) -> dict[str, str]:
        # Source and provider secrets from Semantier's process never cross this
        # boundary. The explicitly selected external profile owns host config.
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'SYSTEMROOT') if key in os.environ}
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate the configured installation without starting Hermes')
    parser.add_argument('--home', type=Path, help='explicit local DEVELOPMENT Hermes profile; required to start the CLI')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='public Hermes CLI arguments after --')
    args = parser.parse_args(argv)
    try:
        installation = UpstreamInstallation.from_environment()
        installation.verify()
        if args.check:
            print(json.dumps({'root': str(installation.root), 'python': str(installation.python),
                              'commit': installation.commit, 'tree': installation.tree}))
            return 0
        if args.home is None:
            parser.error('--home is required for external CLI execution')
        env = installation.environment()
        env.update(HOME=str(args.home.expanduser().resolve()), HERMES_HOME=str(args.home.expanduser().resolve()))
        forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
        return subprocess.call(installation.command(*forwarded), cwd=installation.root, env=env)
    except (UpstreamConfigurationError, OSError, subprocess.SubprocessError) as exc:
        print(f'External Hermes configuration error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
