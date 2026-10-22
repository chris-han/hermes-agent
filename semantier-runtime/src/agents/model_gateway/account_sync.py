"""Bounded account synchronization inside the existing backend lifecycle.

PostgreSQL owns claims, current policy, retries and fenced acknowledgement. HTTP
occurs only after those transactions have closed. No customer request starts a
worker or waits for a native management operation.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from uuid import uuid4

log=logging.getLogger(__name__)


class AccountSyncWorker:
    def __init__(self,store,proxy=None,keyring=None,*,max_in_flight=2):
        if isinstance(max_in_flight,bool) or max_in_flight not in {1,2}:
            raise ValueError('Account synchronization requires bounded request concurrency')
        self.store,self.proxy=store,proxy
        self.keyring=keyring if keyring is not None else getattr(store,'keyring',None)
        self.max_in_flight=max_in_flight
        self.owner='account-sync-'+uuid4().hex
        self._task=None
        self._stopping=asyncio.Event()
        self._drain_lock=asyncio.Lock()

    def start(self):
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task=asyncio.create_task(self._run(),name=self.owner)
        return self._task

    async def stop(self):
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task,return_exceptions=True)

    async def _apply(self,claim):
        if self.proxy is None or self.keyring is None:
            await self.store.fail(claim,'BLOCKED_DEPENDENCY',retryable=False)
            return
        try:
            # The entire aggregate ends before its 60-second database lease.
            # A late remote completion cannot pass the database activation fence.
            async with asyncio.timeout(45):
                projection=await self.store.refresh(claim)
                if projection is None:
                    return
                from agents.model_gateway.native_account_adapter import NativeAccountAdapter
                adapter=NativeAccountAdapter(self.proxy)
                credentials=await self.store.credential_rows(claim)
                async def prepare(user_id):
                    return await self.store.prepare_credential(claim,user_id)
                observed=await adapter.reconcile(projection,prepare_credential=prepare,credentials=credentials)
                await self.store.acknowledge(claim,observed)
        except asyncio.CancelledError:
            # Retain the lease/progress for crash-safe recovery. No speculative
            # compensation can delete a native object or authoritative account.
            raise
        except TimeoutError:
            await self.store.fail(claim,'NATIVE_TIMEOUT',retryable=True)
        except Exception as exc:
            from agents.model_gateway.native_account_adapter import NativeAccountError
            if isinstance(exc,NativeAccountError):
                code=exc.code if isinstance(exc.code,str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}',exc.code) else 'SYNC_FAILED'
                await self.store.fail(claim,code,retryable=exc.retryable,retry_after=exc.retry_after)
            else:
                await self.store.fail(claim,'SYNC_FAILED',retryable=True)

    async def run_once(self,limit=2):
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:
            raise ValueError('Positive bounded account claim limit required')
        async with self._drain_lock:
            claims=await self.store.claim(self.owner,limit=min(limit,self.max_in_flight),lease_seconds=60)
            results=await asyncio.gather(*(self._apply(claim) for claim in claims),return_exceptions=True)
            if any(isinstance(result,Exception) for result in results):
                # Error strings/tracebacks can contain credentials or response
                # bodies. Durable sanitized outcomes are recorded by the store.
                log.warning('account_sync STORE_UNAVAILABLE')
            return len(claims)

    async def reconcile_due(self,limit=20):
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=20:
            raise ValueError('Account reconciliation batch must be 1..20')
        async with self._drain_lock:
            return await self.store.reconcile_due(limit=limit)

    async def _run(self):
        next_reconcile=0.0
        while not self._stopping.is_set():
            try:
                if time.monotonic()>=next_reconcile:
                    await self.reconcile_due()
                    next_reconcile=time.monotonic()+300
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning('account_sync STORE_UNAVAILABLE')
            try:
                await asyncio.wait_for(self._stopping.wait(),timeout=1)
            except TimeoutError:
                pass
