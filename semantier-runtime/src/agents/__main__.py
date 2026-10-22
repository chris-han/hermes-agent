from __future__ import annotations

import argparse
import sys

from agents.launcher import run_agent_cli, run_api, run_gateway_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m semantier.agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api")
    api_parser.add_argument("args", nargs=argparse.REMAINDER)

    gateway_parser = subparsers.add_parser("gateway")
    gateway_parser.add_argument("args", nargs=argparse.REMAINDER)

    agent_parser = subparsers.add_parser("agent")
    agent_parser.add_argument("args", nargs=argparse.REMAINDER)

    parsed = parser.parse_args(argv)
    extra_args = parsed.args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if parsed.command == "api":
        return run_api(extra_args)
    if parsed.command == "gateway":
        return run_gateway_cli(extra_args)
    return run_agent_cli(extra_args)


if __name__ == "__main__":
    sys.exit(main())