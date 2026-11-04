"""The ledger's events applied on a background thread, to take the checks off the engine thread.

Every ledger call becomes a queue entry applied in order by one worker thread; the engine thread
pays only the extraction and the enqueue. Reports are identical and in order, but exist after the
step that produced them rather than within it, so a fail-fast adapter must stay inline. Snapshots
and descriptors must be immutable after enqueue; ledger shadow reads are not forwarded.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

_CLOSE = object()


class LedgerWorker:
    """Forwards the ledger's event methods to a background thread, in order."""

    EVENTS = (
        "register_pool",
        "allocate",
        "free",
        "store",
        "reuse",
        "external_copy",
        "external_fill",
        "external_read",
        "external_write",
        "begin_resource_use",
        "end_resource_use",
        "validate_padding_destinations",
        "observe",
        "rollback",
        "dispatch",
        "complete",
    )

    def __init__(self, ledger: Any, *, name: str = "kvguard-checker"):
        self.ledger = ledger
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.error: BaseException | None = None
        self.errors = 0
        self.applied = 0
        self.busy_s = 0.0  # time the thread spent inside ledger calls
        self.max_depth = 0
        self._depth = 0
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()
        for event in self.EVENTS:
            setattr(self, event, self._forwarder(event))

    def _forwarder(self, name: str):
        def call(*args: Any, **kwargs: Any) -> None:
            with self._lock:
                self._depth += 1
                if self._depth > self.max_depth:
                    self.max_depth = self._depth
            self.queue.put((name, args, kwargs))

        call.__name__ = name
        return call

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until every event enqueued so far has been applied; False on timeout."""
        done = threading.Event()
        self.queue.put(done)
        return done.wait(timeout)

    def close(self, timeout: float | None = None) -> bool:
        ok = self.flush(timeout)
        self.queue.put(_CLOSE)
        self.thread.join(timeout)
        return ok and not self.thread.is_alive()

    @property
    def depth(self) -> int:
        with self._lock:
            return self._depth

    def stats(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "busy_s": round(self.busy_s, 4),
            "max_depth": self.max_depth,
            "errors": self.errors,
            "first_error": repr(self.error)[:300] if self.error else None,
        }

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is _CLOSE:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            name, args, kwargs = item
            t0 = time.perf_counter()
            try:
                getattr(self.ledger, name)(*args, **kwargs)
            except BaseException as exc:  # the engine never sees a checker exception
                self.errors += 1
                if self.error is None:
                    self.error = exc
            self.busy_s += time.perf_counter() - t0
            self.applied += 1
            with self._lock:
                self._depth -= 1


__all__ = ["LedgerWorker"]
