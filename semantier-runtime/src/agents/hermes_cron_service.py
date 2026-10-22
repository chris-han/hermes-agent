from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HermesCronStatus:
    running: bool
    last_tick_at: str | None = None
    last_error: str | None = None


class HermesCronTickerService:
    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._last_tick_at: str | None = None
        self._last_error: str | None = None

    def start(self) -> HermesCronStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="semantier-cron-ticker",
            )
            self._thread.start()
            return self.status()

    def shutdown(self) -> None:
        thread = self._thread
        self._stop_event.set()
        if thread is not None:
            thread.join(timeout=self.shutdown_timeout_seconds)
            if thread.is_alive():
                message = "Semantier cron ticker thread did not stop"
                logger.error(message)
                raise RuntimeError(message)
        self._thread = None

    def _set_status(
        self,
        *,
        last_tick_at: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._status_lock:
            if last_tick_at is not None:
                self._last_tick_at = last_tick_at
            self._last_error = last_error

    def status(self) -> HermesCronStatus:
        thread = self._thread
        running = bool(thread is not None and thread.is_alive())
        with self._status_lock:
            return HermesCronStatus(
                running=running,
                last_tick_at=self._last_tick_at,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        from datetime import datetime, timezone

        while not self._stop_event.is_set():
            try:
                from cron.scheduler import tick_known_homes

                tick_known_homes(verbose=False, adapters=None, loop=None)
                self._set_status(
                    last_tick_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    last_error=None,
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                self._set_status(last_error=error_text)
                logger.exception("Semantier cron ticker failed: %s", error_text)
            self._stop_event.wait(self.interval_seconds)


_SERVICE = HermesCronTickerService()


def get_hermes_cron_ticker_service() -> HermesCronTickerService:
    return _SERVICE
