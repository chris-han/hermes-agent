"""Bounded, disposable Chat processes; Core remains the identity/transcript owner."""
from __future__ import annotations
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import os
import signal
import sys
import time

_LOG = logging.getLogger("uvicorn.error")
class ChatQueueFull(RuntimeError):
    pass


_READY = b'SEMANTIER_CHAT_READY='


@dataclass(eq=False)
class ChatLease:
    pool: 'ChatExecutionPool'
    process: asyncio.subprocess.Process
    session_key: tuple[str, str]
    session_lock: asyncio.Lock
    base_url: str
    api_key: str
    deadline: float
    started: float
    closed: bool = False
    cleanup_task: asyncio.Task | None = None

    async def ensure_started(self):
        if self.closed or self.process.returncode is not None:
            raise RuntimeError('CHAT_WORKER_EXITED')

    @property
    def remaining(self):
        return max(0.001, self.deadline-time.monotonic())

    async def close(self):
        if self.cleanup_task is None:
            self.closed = True
            self.cleanup_task = asyncio.create_task(self._reap())
        await asyncio.shield(self.cleanup_task)

    async def _reap(self):
        try:
            # This process group belongs to this turn, including its tool children.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            # A leader can exit before a tool child that ignores SIGTERM.
            # Its process group still belongs to this lease after leader exit.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        finally:
            self.pool._active.discard(self)
            self.pool._slots.release()
            self.session_lock.release()
            self.pool._unreference(self.session_key)
            _LOG.info('chat_execution stage=reaped pid=%s elapsed_seconds=%.3f',
                      self.process.pid, time.monotonic()-self.started)


class ChatExecutionPool:
    def __init__(self, *, max_workers=4, queue_timeout=60., turn_timeout=120.,
                 startup_timeout=45., max_pending=32, worker_command=None):
        if max_workers < 1 or min(queue_timeout,turn_timeout,startup_timeout) <= 0:
            raise ValueError('CHAT_EXECUTION_LIMIT_INVALID')
        self.max_pending = max_pending
        self._pending = 0
        self._slots = asyncio.Semaphore(max_workers)
        self._session_locks = {}
        self._active = set()
        self._closing = False
        self.queue_timeout, self.turn_timeout = queue_timeout, turn_timeout
        self.startup_timeout = startup_timeout
        self.worker_command = worker_command or [sys.executable,'-m','agents.hermes_chat_worker']

    def _unreference(self, key):
        lock, count = self._session_locks[key]
        if count == 1:
            del self._session_locks[key]
        else:
            self._session_locks[key] = (lock,count-1)

    async def acquire(self, workspace_id, session_id):
        if not workspace_id or not session_id or self._closing:
            raise RuntimeError('CHAT_EXECUTION_SCOPE_OR_STATE_INVALID')
        if self._pending >= self.max_pending:
            raise ChatQueueFull('CHAT_EXECUTION_QUEUE_FULL')
        self._pending += 1
        key = (workspace_id,session_id)
        lock, count = self._session_locks.get(key,(asyncio.Lock(),0))
        self._session_locks[key] = (lock,count+1)
        locked = slotted = False
        process = lease = None
        queued = time.monotonic()
        try:
            async with asyncio.timeout(self.queue_timeout):
                await lock.acquire()
                locked = True
                await self._slots.acquire()
                slotted = True
            if self._closing:
                raise RuntimeError('CHAT_EXECUTION_CLOSED')
            started = time.monotonic()
            process = await asyncio.create_subprocess_exec(*self.worker_command,
                stdout=asyncio.subprocess.PIPE, stderr=None, start_new_session=True,
                env={**os.environ,'SEMANTIER_CHAT_WORKER_PROVISIONED':'1'})
            lease = ChatLease(self,process,key,lock,'','',started+self.turn_timeout,started)
            self._active.add(lease)
            async with asyncio.timeout(min(self.startup_timeout,lease.remaining)):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        raise RuntimeError('CHAT_WORKER_STARTUP_FAILED')
                    if line.startswith(_READY):
                        ready = json.loads(line[len(_READY):])
                        lease.base_url, lease.api_key = ready['base_url'],ready['api_key']
                        # No caller-supplied destination or externally reachable worker.
                        from urllib.parse import urlparse
                        address = urlparse(lease.base_url)
                        if address.hostname != '127.0.0.1' or address.scheme != 'http' or not lease.api_key:
                            raise RuntimeError('CHAT_WORKER_ADDRESS_INVALID')
                        break
            _LOG.info('chat_execution stage=ready pid=%s queue_seconds=%.3f startup_seconds=%.3f',
                      process.pid,started-queued,time.monotonic()-started)
            return lease
        except BaseException:
            if lease:
                await asyncio.shield(lease.close())
            else:
                if slotted:self._slots.release()
                if locked:lock.release()
                self._unreference(key)
            raise

        finally:
            self._pending -= 1

    async def shutdown(self):
        self._closing = True
        await asyncio.gather(*(x.close() for x in list(self._active)),return_exceptions=False)
