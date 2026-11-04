"""The vLLM adapter's model of a sliding-window KV cache group: the positions below the window are
dead and their blocks holes; the attention reads the window and consumes the table from its first
block, so a retired block's stale id below it is not compared while a wrong consumed entry is."""

from __future__ import annotations

import numpy as np
from kvguard.adapters.vllm.translate import build_dispatches, mapping_tables, window_start
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    Ledger,
    PoolGeometry,
    Reporter,
    RequestSnapshot,
    Snapshot,
)

from tests.unit.conftest import kinds

POOL = "swa0"
BS = 4
W = 8


def ledger() -> Ledger:
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(
        PoolGeometry(POOL, num_blocks=32, block_size=BS, reserved_blocks=frozenset({0}))
    )
    return lg


def step(lg, seq, rid, blocks, pre, sched, *, table=None):
    """One request's snapshot (its block list with holes, the window's start) and the two
    dispatches of a step, from the mapping or from ``table`` (the gathered device row)."""
    ids = tuple(blocks)
    lg.observe(
        Snapshot(
            seq,
            (RequestSnapshot(rid, POOL, ids, pre + sched, pre, window_start(pre, W)),),
            tuple(BlockSnapshot(POOL, b, BlockState.PRIVATE) for b in ids if b is not None),
        )
    )
    tbl, slots = mapping_tables([list(ids)], [pre], [sched], BS, 1)
    if table is not None:
        tbl = np.array([table])
    write, attn = build_dispatches(seq, POOL, BS, [rid], [pre], [sched], tbl, 1, slots, window=W)
    for d in (write, attn):
        lg.dispatch(d)
        lg.complete(d.dispatch_id)


def test_a_sliding_window_request_life_is_silent():
    lg = ledger()
    lg.allocate(POOL, list(range(1, 10)))
    step(lg, 1, "A", list(range(1, 10)), 0, 36)  # prefill: positions 0..35 in blocks 1..9
    # Decode: the window is [29, 37); the manager freed blocks 1..7 and nulled their columns.
    lg.free(POOL, list(range(1, 8)))
    lg.allocate(POOL, [10])
    step(lg, 2, "A", [None] * 7 + [8, 9, 10], 36, 1)
    assert kinds(lg) == []


def test_a_stale_id_below_the_window_is_not_consumed_but_a_wrong_entry_in_it_is():
    lg = ledger()
    lg.allocate(POOL, list(range(1, 10)))
    step(lg, 1, "A", list(range(1, 10)), 0, 36)
    lg.free(POOL, list(range(1, 8)))
    lg.allocate(POOL, [10])
    # The worker's table still names the retired blocks below the window: silent.
    step(lg, 2, "A", [None] * 7 + [8, 9, 10], 36, 1, table=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert kinds(lg) == []
    # An entry inside the window names another block: reported.
    step(lg, 3, "A", [None] * 7 + [8, 9, 10], 37, 1, table=[0, 0, 0, 0, 0, 0, 0, 8, 2, 10])
    assert kinds(lg) == ["mapping_mismatch"]


def test_a_hole_inside_the_window_is_a_missing_block():
    lg = ledger()
    lg.allocate(POOL, list(range(1, 10)))
    step(lg, 1, "A", list(range(1, 10)), 0, 36)
    lg.free(POOL, list(range(1, 9)))  # block 8 freed too early: position 29..31 lose their block
    lg.allocate(POOL, [10])
    step(lg, 2, "A", [None] * 8 + [9, 10], 36, 1)
    assert "live_position_without_block" in kinds(lg) and "access_unmapped_position" in kinds(lg)
