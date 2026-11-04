"""The recurrent-state pool's snapshot (pure) and its geometry."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from kvguard.adapters.sglang.state import STATE_POOL, state_geometry, state_snapshot
from kvguard.core import BlockState, Ledger, Reporter


class FakeMambaPool:
    size = 8


class FakeReqPool:
    mamba_pool = FakeMambaPool()
    mamba_map = (0, 1, 2)


def test_geometry_counts_the_reserved_padding_slot():
    g = state_geometry(FakeReqPool(), layers=3)
    assert (g.pool, g.num_blocks, g.block_size, g.layers, g.rewritable) == (
        STATE_POOL,
        9,
        1,
        3,
        True,
    )
    assert g.reserved_blocks == frozenset({0})


def test_unified_geometry_uses_the_routed_state_view():
    g = state_geometry(
        SimpleNamespace(),
        layers=18,
        state_storage=SimpleNamespace(pool=SimpleNamespace(size=12)),
    )
    assert (g.num_blocks, g.layers, g.reserved_blocks) == (13, 18, frozenset({0}))


def test_unified_state_locations_inverse_map_physical_to_virtual_slots():
    from kvguard.adapters.sglang.state import StateMediation

    state = object.__new__(StateMediation)
    state.unified = True
    state.band = SimpleNamespace(ids=SimpleNamespace(physical_to_virtual=[-1, 9, 4]))

    assert state._logical_slots([1, 2, 0, -1, 7]) == [9, 4, -1, -1, -1]


def test_snapshot_lists_requests_tree_slots_and_leaks():
    # Rows 0 and 2 hold requests: A with active slot 3 and tracking slots 5 and 6 (one a hole),
    # B with slot 4 and no tracking slots yet; the tree holds slots 1 and 2; slot 7 is allocated
    # to nobody; slot 8 is free.
    requests, blocks, free = state_snapshot(
        rid_of_row={0: "A", 2: "B"},
        active_of_row=[3, 0, 4, 0],
        tracks_of_row=[[5, -1], [-1, -1], [-1, -1], [-1, -1]],
        free_slots=[8],
        tree_slots=[1, 2],
        size=8,
    )
    assert [(r.request, r.blocks, r.num_tokens, r.num_computed) for r in requests] == [
        ("A", (3, 5), 2, 0),
        ("B", (4,), 1, 0),
    ]
    by = {b.block: b for b in blocks}
    assert set(by) == {1, 2, 3, 4, 5, 6, 7}
    assert by[1].state is BlockState.PUBLISHED and by[1].reserved
    assert by[3].state is BlockState.PRIVATE and by[7].state is BlockState.PRIVATE
    assert free == frozenset({8})


def test_a_tree_slot_still_mapped_by_a_finishing_request_is_published_and_read():
    _, blocks, _ = state_snapshot(
        rid_of_row={0: "A"},
        active_of_row=[3],
        tracks_of_row=None,
        free_slots=[],
        tree_slots=[3],
        size=3,
    )
    by = {b.block: b for b in blocks}
    assert by[3].state is BlockState.PUBLISHED and not by[3].reserved


def test_the_core_reports_a_leaked_state_slot_and_accepts_the_rest():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(state_geometry(FakeReqPool(), layers=1))
    ledger.allocate(STATE_POOL, [3, 4, 7])
    from kvguard.core import Snapshot

    requests, blocks, free = state_snapshot(
        rid_of_row={0: "A", 1: "B"},
        active_of_row=[3, 4],
        tracks_of_row=None,
        free_slots=[1, 2, 5, 6, 8],
        tree_slots=[],
        size=8,
    )
    ledger.observe(Snapshot(1, tuple(requests), tuple(blocks), {STATE_POOL: free}))
    kinds = [r.kind for r in ledger.reporter.reports]
    assert kinds == ["private_unmapped"]
    assert ledger.reporter.reports[0].block == 7


def test_the_state_dispatch_reads_and_writes_each_rows_slot_and_the_tracked_write():
    from kvguard.adapters.sglang.state import build_state_dispatch
    from kvguard.core.model import PADDING_SLOT

    d = build_state_dispatch(
        7,
        ["A", "B", None],
        [3, 4, -1],
        track_rows=[1],
        track_slots=[0, 9, 0],
        tracks_of_row=[[5, -1], [8, 9], []],
    )
    assert d.dispatch_id == "state@7" and d.pool == STATE_POOL
    assert [(r.request, tuple(r.reads), tuple(r.writes), r.padding) for r in d.rows] == [
        ("A", (0,), (0,), False),
        ("B", (0,), (0,), False),
        (None, (), (), True),
        ("B", (), (2,), False),
    ]
    m = d.mapping
    assert list(zip(m.slots, m.rows, m.positions, strict=True)) == [
        (3, 0, 0),
        (4, 1, 0),
        (PADDING_SLOT, 2, 0),
        (9, 3, 2),
    ]


def test_graph_state_descriptor_uses_the_live_track_mask():
    from kvguard.adapters.sglang.state import StateMediation

    state = object.__new__(StateMediation)
    state.att = SimpleNamespace(seq=9)
    state.stats = Counter()
    state.tracks_by_row = [[5, -1], [8, 9]]
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=[3, 4],
            mamba_track_indices=[5, 9],
            # Replay metadata keeps the capture-time default even though the captured kernel
            # reads the live mask.
            has_mamba_track_mask=False,
            mamba_track_mask_indices=None,
        )
    )
    fb = SimpleNamespace(req_pool_indices=[0, 1], mamba_track_mask=[False, True])

    d = state.descriptor(fb, ["A", "B"], backend)

    assert d is not None
    assert [(row.request, tuple(row.writes)) for row in d.rows] == [
        ("A", (0,)),
        ("B", (0,)),
        ("B", (2,)),
    ]
    assert d.mapping.slots == [3, 4, 9]
    assert state.stats["state_graph_tracked_rows"] == 1


def test_prefill_graph_state_descriptor_uses_the_tables_consumed_by_the_graph():
    from kvguard.adapters.sglang.state import StateMediation

    state = object.__new__(StateMediation)
    state.att = SimpleNamespace(seq=10)
    state.stats = Counter()
    state.tracks_by_row = [[5, -1], [8, 9]]
    backend = SimpleNamespace(
        # Deliberately stale eager metadata: a prefill graph does not consume this object.
        forward_metadata=SimpleNamespace(mamba_cache_indices=[30, 40]),
        prefill_graph_metadata=SimpleNamespace(
            cache_indices=[3, 4, -1, -1],
            track_mode=[0, 2, 0, 0],
            track_dst=[0, 9, 0, 0],
        ),
    )
    mode = SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False)
    fb = SimpleNamespace(req_pool_indices=[0, 1], forward_mode=mode)

    d = state.descriptor(fb, ["A", "B"], backend)

    assert d is not None
    assert [(row.request, tuple(row.writes)) for row in d.rows] == [
        ("A", (0,)),
        ("B", (0,)),
        ("B", (2,)),
    ]
    assert d.mapping.slots == [3, 4, 9]
    assert state.stats["state_prefill_graph_dispatches"] == 1
    assert state.stats["state_prefill_graph_tracked_rows"] == 1


def test_waiting_request_owns_state_allocated_before_admission():
    from kvguard.adapters.sglang.state import StateMediation

    waiting = SimpleNamespace(
        rid="waiting",
        kv=SimpleNamespace(mamba_pool_idx=7, mamba_ping_pong_track_buffer=[8, -1]),
    )
    state = object.__new__(StateMediation)
    state.att = SimpleNamespace(
        scheduler=SimpleNamespace(waiting_queue=[waiting]),
        _request_owner=lambda req: f"{req.rid}#session-owner",
    )
    state.stats = Counter()

    requests = state._waiting_requests(set())

    assert [(request.request, request.blocks) for request in requests] == [
        ("waiting#session-owner", (7, 8))
    ]
    assert state.stats["state_snapshot_waiting_requests"] == 1
    assert state._waiting_requests({"waiting#session-owner"}) == []


def test_pending_buffer_load_owns_the_published_state_checkpoint():
    from kvguard.adapters.sglang.state import StateMediation

    state = object.__new__(StateMediation)
    state.att = SimpleNamespace(_hicache_loads={7: (object(), (("kv", (1, 2)), ("state", (8, 9))))})
    state.stats = Counter()

    requests = state._pending_load_requests([8])

    assert [(request.request, request.blocks) for request in requests] == [
        ("hicache-load:7:state", (8,))
    ]
    assert state.stats["state_snapshot_pending_loads"] == 1


def test_the_state_dispatch_against_the_ledger():
    from kvguard.adapters.sglang.state import build_state_dispatch
    from kvguard.core import ExternalFill, Snapshot

    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(state_geometry(FakeReqPool(), layers=1))
    ledger.allocate(STATE_POOL, [3, 5])
    requests, blocks, free = state_snapshot(
        rid_of_row={0: "A"},
        active_of_row=[3],
        tracks_of_row=[[5, -1]],
        free_slots=[1, 2, 4, 6, 7, 8],
        tree_slots=[],
        size=8,
    )
    ledger.observe(Snapshot(1, tuple(requests), tuple(blocks), {STATE_POOL: free}))
    ledger.external_fill(ExternalFill(STATE_POOL, 3, range(0, 1), source="clear"))
    # Step 1: the prefill reads the cleared slot and writes it, and tracks into slot 5.
    ledger.dispatch(build_state_dispatch(1, ["A"], [3], [0], [5], [[5, -1]]))
    ledger.complete("state@1")
    # Step 2: a decode rewrites the slot (an update in a rewritable pool).
    ledger.dispatch(build_state_dispatch(2, ["A"], [3]))
    ledger.complete("state@2")
    assert [r.kind for r in ledger.reporter.reports] == []
    assert ledger.block(STATE_POOL, 5).is_filled(0)
    # Step 3: a row consuming another request's slot is an access without permission.
    ledger.allocate(STATE_POOL, [4])
    requests, blocks, free = state_snapshot(
        rid_of_row={0: "A", 1: "B"},
        active_of_row=[3, 4],
        tracks_of_row=[[5, -1], [-1, -1]],
        free_slots=[1, 2, 6, 7, 8],
        tree_slots=[],
        size=8,
    )
    ledger.observe(Snapshot(2, tuple(requests), tuple(blocks), {STATE_POOL: free}))
    ledger.dispatch(build_state_dispatch(3, ["A", "B"], [3, 3]))
    kinds = [r.kind for r in ledger.reporter.reports]
    assert kinds and all(k in ("slot_map_mismatch", "access_foreign_private") for k in kinds)


def test_the_checkpoint_pool_lists_the_trees_slots_and_leaks():
    from kvguard.adapters.sglang.state import CKPT_POOL, ckpt_snapshot

    blocks, free = ckpt_snapshot(free_slots=[0, 3, 4], tree_slots=[1], size=4)
    assert {(b.pool, b.block, b.state, b.reserved) for b in blocks} == {
        (CKPT_POOL, 1, BlockState.PUBLISHED, True),
        (CKPT_POOL, 2, BlockState.PRIVATE, False),
    }
    assert free == frozenset({3, 4})
