"""The vLLM adapter's model of a hybrid model's KV cache groups over one block pool: the
full-attention pool and a recurrent-state pool share the block ids; the state dispatch reads the
previous column and writes the running one; the partition audit counts the other group's blocks as
free."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from kvguard.adapters.vllm.translate import build_state_dispatch, cow_copy, dense_table
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

KV, ST = "gpu", "state0"
BS = 544
N = 16


def test_mla_specialization_is_the_full_attention_anchor():
    """Kimi Linear's MLA group has the ordinary full-attention scheduler lifecycle."""

    from kvguard.adapters.vllm.attach import VllmAttachment

    mamba = type("MambaSpec", (), {})()
    mamba.mamba_cache_mode = "align"
    mamba.num_speculative_blocks = 0
    mla = type("MLAAttentionSpec", (), {})()
    attachment = VllmAttachment.__new__(VllmAttachment)
    attachment.managers = (
        SimpleNamespace(block_size=128),
        SimpleNamespace(block_size=128),
    )
    attachment.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(kv_cache_spec=mamba, layer_names=("kda",)),
                SimpleNamespace(kv_cache_spec=mla, layer_names=("mla",)),
            )
        )
    )

    groups = attachment._classify_groups()

    assert groups[0]["kind"] == "state"
    assert groups[1] == {
        "group": 1,
        "spec": "MLAAttentionSpec",
        "manager": "SimpleNamespace",
        "block_size": 128,
        "layers": 1,
        "kind": "attention",
        "pool": "gpu",
    }


def test_step3p5_mtp_groups_are_distinct_from_the_target_anchor():
    from kvguard.adapters.vllm.attach import VllmAttachment, window_pool

    full = type("FullAttentionSpec", (), {"is_draft": False})()
    target_window = type(
        "SlidingWindowSpec",
        (),
        {"is_draft": False, "sliding_window": 128, "extra_retained_tokens": 2},
    )()
    draft_window = type(
        "SlidingWindowSpec",
        (),
        {"is_draft": True, "sliding_window": 128, "extra_retained_tokens": 2},
    )()
    attachment = VllmAttachment.__new__(VllmAttachment)
    attachment.managers = tuple(SimpleNamespace(block_size=16) for _ in range(3))
    attachment.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(kv_cache_spec=full, layer_names=("target.full",)),
                SimpleNamespace(kv_cache_spec=target_window, layer_names=("target.swa",)),
                SimpleNamespace(kv_cache_spec=draft_window, layer_names=("draft.mtp",)),
            )
        )
    )

    groups = attachment._classify_groups()

    assert groups[0]["pool"] == "gpu" and "draft" not in groups[0]
    assert groups[1]["pool"] == window_pool(1) and "draft" not in groups[1]
    assert groups[2]["pool"] == window_pool(2) and groups[2]["draft"] is True


def ledger(shared: bool = True) -> Ledger:
    """The two pools over one block id space (``allocator``), as the adapter registers them."""
    alloc = "block_pool" if shared else None
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(
        PoolGeometry(
            KV, num_blocks=N, block_size=BS, reserved_blocks=frozenset({0}), allocator=alloc
        )
    )
    lg.register_pool(
        PoolGeometry(
            ST,
            num_blocks=N,
            block_size=1,
            reserved_blocks=frozenset({0}),
            rewritable=True,
            allocator=alloc,
        )
    )
    return lg


def state_step(lg, seq, rid, columns, pre, sched, *, table=None, n_pad=1):
    """One request's state snapshot and dispatch: ``columns`` the block list (None for a
    retired column), ``table`` the gathered device row when it differs from the list."""
    ids = tuple(columns)
    lg.observe(
        Snapshot(
            seq,
            (RequestSnapshot(rid, ST, ids, len(ids), 0),),
            tuple(BlockSnapshot(ST, b, BlockState.PRIVATE) for b in ids if b is not None),
        )
    )
    row = table if table is not None else [0 if b is None else b for b in ids]
    tbl = np.array([row] + [[0] * len(row)] * (n_pad - 1))
    d = build_state_dispatch(seq, ST, BS, "align", [rid], [pre], [sched], tbl, n_pad)
    lg.dispatch(d)
    lg.complete(d.dispatch_id)
    return d


def test_a_request_life_in_the_align_mode_is_silent():
    lg = ledger()
    lg.allocate(ST, [5])
    state_step(lg, 1, "A", [None, 5], 0, 600)  # prefill of 600 tokens: column 1 written
    state_step(lg, 2, "A", [None, 5], 600, 1)  # decode in column 1: read and rewrite
    lg.allocate(ST, [6])
    state_step(lg, 3, "A", [None, 5, 6], 1088, 1)  # the boundary: read column 1, write 2
    lg.free(ST, [5])  # the previous state block retired (remove_skipped_blocks)
    state_step(lg, 4, "A", [None, None, 6], 1089, 1)
    assert kinds(lg) == []


def test_a_fresh_block_read_as_the_previous_state_is_uninitialized():
    """The read row stands alone: the step's own write of the same column cannot satisfy it."""
    lg = ledger()
    lg.allocate(ST, [5])
    state_step(lg, 1, "A", [None, 5], 600, 1)  # claims prior state in column 1, never written
    assert kinds(lg) == ["read_uninitialized"]


def test_a_retired_columns_old_entry_is_not_consumed_but_a_wrong_consumed_column_is_reported():
    """The worker keeps a retired column's old id in its table and never reads it (the kernels
    gather one column per row), so it is not compared; the consumed column is."""
    lg = ledger()
    lg.allocate(ST, [5])
    state_step(lg, 1, "A", [None, 5], 0, 600)
    lg.allocate(ST, [6])
    state_step(lg, 2, "A", [None, 5, 6], 1088, 1)
    lg.free(ST, [5])
    # The device table still names block 5 in column 1 after the manager retired it: silent.
    state_step(lg, 3, "A", [None, None, 6], 1089, 1, table=[0, 5, 6])
    assert kinds(lg) == []
    # The consumed column names another block: reported on both rows (one report).
    state_step(lg, 4, "A", [None, None, 6], 1090, 1, table=[0, 5, 7])
    assert kinds(lg) == ["mapping_mismatch"]


def test_a_freed_state_block_read_by_the_step_is_an_access_to_free_memory():
    lg = ledger()
    lg.allocate(ST, [5])
    state_step(lg, 1, "A", [None, 5], 0, 600)
    lg.free(ST, [5])  # freed while the request still maps it (an early free)
    lg.observe(Snapshot(2, (RequestSnapshot("A", ST, (None, 5), 2, 0),), ()))
    d = build_state_dispatch(2, ST, BS, "align", ["A"], [600], [1], np.array([[0, 5]]), 1)
    lg.dispatch(d)
    assert "mapping_to_free_block" in kinds(lg) and "access_free_block" in kinds(lg)


def test_the_partition_audit_runs_once_over_the_pools_of_a_shared_allocator():
    """The allocator's free set (or count) is given under one pool; the pools' listed blocks
    together with it must partition the id space. Without the shared allocator each pool is
    audited alone and the other pool's blocks look unaccounted."""
    lg = ledger()
    lg.allocate(KV, [1, 2])
    lg.allocate(ST, [5])
    allocator_free = frozenset(range(3, N)) - {5}
    kv_req = RequestSnapshot("A", KV, (1, 2), 600, 0)
    st_req = RequestSnapshot("A", ST, (None, 5), 2, 0)
    blocks = (
        BlockSnapshot(KV, 1, BlockState.PRIVATE),
        BlockSnapshot(KV, 2, BlockState.PRIVATE),
        BlockSnapshot(ST, 5, BlockState.PRIVATE),
    )
    lg.observe(Snapshot(1, (kv_req, st_req), blocks, {KV: allocator_free}))
    assert kinds(lg) == []
    lg.observe(Snapshot(2, (kv_req, st_req), blocks, None, free_count={KV: 12}))
    assert kinds(lg) == []
    # A block the allocator still counts free while a pool holds it; a wrong count.
    lg.observe(Snapshot(3, (kv_req, st_req), blocks, {KV: allocator_free | {5}}))
    assert kinds(lg) == ["block_free_and_allocated"]
    lg.observe(Snapshot(4, (kv_req, st_req), blocks, None, free_count={KV: 13}))
    assert kinds(lg)[-1] == "block_count_mismatch"
    # A block the allocator does not count free and no pool lists: unaccounted, once.
    lg.observe(Snapshot(5, (kv_req, st_req), blocks, {KV: allocator_free - {9}}))
    assert kinds(lg)[-1] == "block_unaccounted" and kinds(lg).count("block_unaccounted") == 1
    # The same pools without a shared allocator: each audited alone.
    alone = ledger(shared=False)
    alone.allocate(KV, [1, 2])
    alone.allocate(ST, [5])
    alone.observe(Snapshot(1, (kv_req, st_req), blocks, {KV: allocator_free, ST: allocator_free}))
    assert kinds(alone) == ["block_unaccounted", "block_unaccounted"]


def test_a_block_listed_by_two_pools_of_one_allocator_is_an_integration_error():
    lg = ledger()
    lg.allocate(KV, [1])
    lg.allocate(ST, [1])  # the adapter credited one physical block to two groups
    blocks = (BlockSnapshot(KV, 1, BlockState.PRIVATE), BlockSnapshot(ST, 1, BlockState.PRIVATE))
    reqs = (RequestSnapshot("A", KV, (1,), 1, 0), RequestSnapshot("A", ST, (1,), 1, 0))
    lg.observe(Snapshot(1, reqs, blocks, {KV: frozenset(range(2, N))}))
    assert "block_in_two_pools" in kinds(lg)


def test_a_cow_copy_initializes_the_private_copy_from_the_shared_tail():
    """A partial prefix hit in the attention group: the shared block's computed positions are
    copied into the request's own block before the step, so its computed claim holds."""
    lg = ledger()
    lg.allocate(KV, [1])
    lg.observe(
        Snapshot(
            1, (RequestSnapshot("P", KV, (1,), 40, 0),), (BlockSnapshot(KV, 1, BlockState.PRIVATE),)
        )
    )
    from kvguard.core import DispatchDescriptor, DispatchRow

    lg.dispatch(
        DispatchDescriptor(
            "w@1",
            KV,
            "write",
            (DispatchRow("P", writes=range(0, 40)),),
            dense_table_mapping((1,), 40),
        )
    )
    lg.complete("w@1")
    lg.allocate(KV, [2])
    lg.external_copy(cow_copy(KV, 1, 2, BS, num_computed=32), site="cow@2")
    lg.observe(
        Snapshot(
            2,
            (RequestSnapshot("P", KV, (1,), 41, 40), RequestSnapshot("Q", KV, (2,), 40, 32)),
            (BlockSnapshot(KV, 1, BlockState.PUBLISHED), BlockSnapshot(KV, 2, BlockState.PRIVATE)),
        )
    )
    assert kinds(lg) == []
    assert dense_table([[1], [2]], 2).tolist() == [[1], [2]]


def dense_table_mapping(blocks, num_tokens):
    from kvguard.core import DenseBlockTable

    return DenseBlockTable([list(blocks)], [-(-num_tokens // BS)])
