"""One-shot signal transport over the existing timer/wake owners. No business IO."""

import asyncio
import logging
from contracts.meeting_delivery import utc
from agents.timer_occurrence import TimerOccurrenceStore

log = logging.getLogger(__name__)
PREFIX = "core-one-shot:v1:"


class CoreOneShotTimerProvider:
    def __init__(self, timer_store, validate):
        self.timers = timer_store
        self.validate = validate

    def arm(self, *, timer_id, scheduled_at):
        self.validate(timer_id, scheduled_at)
        return PREFIX + timer_id


class CoreOneShotDueDriver:
    def __init__(self, timers: TimerOccurrenceStore, ingress, *, clock=utc, batch=32, workflow_type="meeting_negotiation"):
        self.timers, self.ingress = timers, ingress
        self.clock, self.batch = clock, batch
        self.workflow_type = workflow_type

    def scan(self):
        now = self.clock()
        due = self.timers.core_due(now=now, limit=self.batch, workflow_type=self.workflow_type)
        for row in due:
            self.ingress(row["timer_id"], now=now)
        return len(due)


async def bounded_poll(action, *, interval=0.25, stop=None):
    """Cancellation drains an in-progress owner call before lifecycle closes."""
    delay = interval
    while stop is None or not stop.is_set():
        call = asyncio.create_task(asyncio.to_thread(action))
        try:
            await asyncio.shield(call)
            delay = interval
        except asyncio.CancelledError:
            await call
            raise
        except Exception:
            log.exception("Core meeting timer owner scan failed; retrying with backoff")
            delay = min(max(interval, delay * 2), 30)
        await asyncio.sleep(delay)
