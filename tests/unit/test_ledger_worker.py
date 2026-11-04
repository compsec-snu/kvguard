"""The checker thread (`kvguard/core/worker.py`): events applied in order on another thread,
flush and close semantics, exceptions kept away from the caller, and the same reports as the
inline ledger on a real scenario."""

from __future__ import annotations

import threading

from kvguard.core import Ledger, PoolGeometry, Reporter, Snapshot
from kvguard.core.worker import LedgerWorker


class Recording:
    def __init__(self):
        self.calls = []
        self.thread_ids = set()

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.thread_ids.add(threading.get_ident())
            if name == "free" and args and args[0] == "boom":
                raise RuntimeError("boom")
            self.calls.append((name, args, kwargs))

        return call


def test_events_are_applied_in_order_on_the_worker_thread():
    rec = Recording()
    w = LedgerWorker(rec)
    w.register_pool("g")
    w.allocate("p", [1, 2], site="a")
    w.validate_padding_destinations([3, 0], 1, {0}, site="graph")
    w.observe("snap1")
    w.dispatch("d1")
    w.complete("d1")
    assert w.flush(timeout=5)
    assert rec.calls == [
        ("register_pool", ("g",), {}),
        ("allocate", ("p", [1, 2]), {"site": "a"}),
        (
            "validate_padding_destinations",
            ([3, 0], 1, {0}),
            {"site": "graph"},
        ),
        ("observe", ("snap1",), {}),
        ("dispatch", ("d1",), {}),
        ("complete", ("d1",), {}),
    ]
    assert rec.thread_ids == {w.thread.ident} and w.thread.ident != threading.get_ident()
    assert w.applied == 6 and w.depth == 0 and w.max_depth >= 1
    assert w.close(timeout=5) and not w.thread.is_alive()


def test_an_exception_on_the_worker_is_recorded_not_raised():
    rec = Recording()
    w = LedgerWorker(rec)
    w.free("boom")
    w.observe("after")
    assert w.flush(timeout=5)
    assert w.errors == 1 and isinstance(w.error, RuntimeError)
    assert rec.calls == [("observe", ("after",), {})], "the queue keeps going"
    assert w.stats()["first_error"].startswith("RuntimeError")
    w.close(timeout=5)


def test_same_reports_as_the_inline_ledger():
    geometry = PoolGeometry("kv", num_blocks=8, block_size=4)

    def scenario(ledger):
        ledger.register_pool(geometry)
        ledger.allocate("kv", [1, 2], site="alloc")
        # A snapshot that maps a block the free set also lists: use after free (P2).
        snap = Snapshot(1, (), (), {"kv": frozenset(range(8))})
        ledger.observe(snap)
        ledger.free("kv", [1], site="free")
        ledger.free("kv", [1], site="free-again")  # double free (P1)

    inline = Reporter(fail_fast=False)
    scenario(Ledger(inline))
    threaded = Reporter(fail_fast=False)
    w = LedgerWorker(Ledger(threaded))
    scenario(w)
    assert w.flush(timeout=5) and w.errors == 0
    assert [str(r) for r in threaded.reports] == [str(r) for r in inline.reports]
    assert inline.reports, "the scenario produces reports"
    w.close(timeout=5)
