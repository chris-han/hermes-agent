"""Profile-bound access to real pinned upstream package lifecycle APIs."""

from __future__ import annotations
import json
import fcntl
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time
import uuid
from agents.hermes_upstream import UpstreamInstallation


class SdkBootstrapRequired(RuntimeError):
    pass


def reserve_host():
    path = os.environ.get("SEMANTIER_SDK_QUALIFICATION_LEDGER")
    if not path:
        return
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        data = json.load(f)
        if data["host_starts"] >= data["limits"]["host_starts"]:
            raise RuntimeError("SDK_HOST_BUDGET_EXHAUSTED")
        data["host_starts"] += 1
        data["sdk_worker_starts"] = data.get("sdk_worker_starts", 0) + 1
        f.seek(0)
        json.dump(data, f)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())


class ProfileSdk:
    """One worker/home for its whole lifetime; parent environment never changes."""

    def __init__(self, *, profile: Path, installation=None):
        self.installation = installation or UpstreamInstallation.from_environment()
        self.installation.verify()
        self.profile = Path(profile).absolute()
        if self.profile.is_symlink() or self.profile.resolve() != self.profile:
            raise ValueError("SDK_PROFILE_SYMLINK")
        self.profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.profile.chmod(0o700)
        self._lease = os.fdopen(
            os.open(
                self.profile / ".sdk-owner.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            ),
            "a",
        )
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lease.close()
            raise RuntimeError("SDK_PROFILE_ALREADY_OWNED") from exc
        self._lock = threading.Lock()
        self._closed = False
        env = {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}
        env.update(
            HOME=str(self.profile),
            HERMES_HOME=str(self.profile),
            PYTHONPATH=str(self.installation.root),
            PYTHONDONTWRITEBYTECODE="1",
        )
        tools = os.environ.get("SEMANTIER_HERMES_SDK_TOOLS_ROOT")
        if tools:
            tools_path = Path(tools).absolute()
            from agents.hermes_upstream import runtime_roots

            if any(
                tools_path.resolve().is_relative_to(root.resolve())
                for root in runtime_roots()
            ):
                self._lease.close()
                raise ValueError("SDK_TOOLS_MUST_BE_EXTERNAL")
            env["HERMES_RUNTIME_DIR"] = str(tools_path)
        env["PATH"] = (
            str(self.installation.python.parent)
            + ":/usr/bin:/bin:"
            + os.environ.get("PATH", "")
        )
        try:
            reserve_host()
            self.process = subprocess.Popen(
                self._worker_command(),
                cwd=self.installation.root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                bufsize=1,
            )
        except BaseException:
            self._lease.close()
            raise

    def _worker_command(self):
        return [
            str(self.installation.python),
            str(Path(__file__).with_name("hermes_sdk_worker.py")),
        ]

    def call(self, operation, **arguments):
        with self._lock:
            if self._closed:
                raise RuntimeError("SDK_WORKER_CLOSED")
            for name in ("config.yaml", ".env", "plugins", "skills"):
                path = self.profile / name
                if path.is_symlink():
                    raise ValueError("SDK_PROFILE_ASSET_SYMLINK")
            identity = uuid.uuid4().hex
            self.process.stdin.write(
                json.dumps(
                    {"id": identity, "operation": operation, "arguments": arguments}
                )
                + "\n"
            )
            self.process.stdin.flush()
            deadline = time.monotonic() + 30
            line = bytearray()
            while not line.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if (
                    remaining <= 0
                    or not select.select([self.process.stdout], [], [], remaining)[0]
                ):
                    self.close()
                    raise TimeoutError("SDK_OPERATION_TIMEOUT")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk or len(line) + len(chunk) > 1024 * 1024:
                    self.close()
                    raise RuntimeError("SDK_PROTOCOL_FAILURE")
                line.extend(chunk)
            result = json.loads(line)
            if result.get("id") != identity:
                raise RuntimeError("SDK_RESPONSE_IDENTITY_MISMATCH")
            if not result.get("ok"):
                if result.get("error") == "SdkBootstrapRequired":
                    raise SdkBootstrapRequired(
                        "Explicit scripts/bootstrap_hermes_sdk.py required"
                    )
                raise RuntimeError("SDK_OPERATION_FAILED:" + str(result.get("error")))
            return result["result"]

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=3)
        for pipe in (self.process.stdin, self.process.stdout):
            pipe.close()
        self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
