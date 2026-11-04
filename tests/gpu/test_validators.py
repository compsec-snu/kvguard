"""Differential tests of the GPU dispatch validators against the host reference (opt-in: needs
the A6000 and ``build/validators/validators.ptx`` from ``make code``; not part of ``make check``).

Run: ``envs/tools/bin/python -m pytest tests/gpu``. Each scenario is played through the host
ledger (``tests.unit.conftest.World``) and the same tables are put on the device; the validator
must report exactly when the host reports a violation of the same class, and nothing on the
clean cases.
"""

import ctypes as C
from array import array

import pytest
from kvguard.core import CSRPageList, DenseBlockTable, Ledger, Reporter, SlotMap, TokenTable
from kvguard.core.model import PoolGeometry
from kvguard.gpu import validate as V

from tests.unit.conftest import BS, POOL, World, kinds, row

pytestmark = pytest.mark.skipif(
    not V.default_ptx_path().exists(), reason="run `make code` first (validators.ptx)"
)


@pytest.fixture(scope="module")
def validators():
    try:
        return V.Validators(timing=True)
    except Exception as exc:  # no driver, no GPU
        pytest.skip(f"no CUDA device: {exc}")


@pytest.fixture
def ledger():
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(
        PoolGeometry(pool=POOL, num_blocks=16, block_size=BS, reserved_blocks=frozenset({0}))
    )
    return lg


@pytest.fixture
def world(ledger):
    return World(ledger)


def dev_i32(drv, values):
    a = array("i", values)
    p = drv.mem_alloc(max(len(a) * 4, 4))
    if len(a):
        drv.memcpy_htod(p, a.buffer_info()[0], len(a) * 4)
    return p


def dev_i64(drv, values):
    a = array("q", values)
    p = drv.mem_alloc(max(len(a) * 8, 8))
    if len(a):
        drv.memcpy_htod(p, a.buffer_info()[0], len(a) * 8)
    return p


def run_dense(v, mirror, table, valid_rows, owners, ends, starts=None):
    width = max((len(r) for r in table), default=1) or 1
    flat = [x for r in table for x in r + [0] * (width - len(r))]
    p = dev_i32(v.drv, flat)
    mirror.sync(0)
    v.validate_dense(mirror, p, width, len(table), width, owners, ends, 0, "dense", starts)
    (res,) = v.drain()
    v.drv.mem_free(p)
    return res


def owner_ids(mirror, rids):
    return [V.PADDING_ROW if r is None else mirror.owners.id_of(r) for r in rids]


def test_clean_dense_table_reports_nothing(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    m = validators.mirror(ledger, POOL)
    res = run_dense(validators, m, [[3, 4]], [2], owner_ids(m, ["A"]), [6])
    assert res.count == 0 and res.gpu_ms is not None


def test_dense_foreign_private_block(validators, world, ledger):
    world.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    world.write("A", 0, 2).write("B", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    # The host: row A's table names B's block.
    world.dispatch("d", [row("A", reads=range(0, 2))], DenseBlockTable([[4]], [1]))
    assert "mapping_mismatch" in kinds(ledger)
    res = run_dense(validators, m, [[4]], [1], owner_ids(m, ["A"]), [2])
    assert (
        res.count == 1
        and res.name == "access_foreign_private"
        and (res.row, res.index, res.entry) == (0, 0, 4)
    )


def test_dense_free_block_and_out_of_pool(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.drop_request("A").release(3).observe()
    m = validators.mirror(ledger, POOL)
    res = run_dense(validators, m, [[3]], [1], owner_ids(m, ["A"]), [2])
    assert res.count == 1 and res.name == "access_free_block"
    res = run_dense(validators, m, [[99]], [1], owner_ids(m, ["A"]), [2])
    assert res.count == 1 and res.name == "mapping_out_of_pool"


def test_dense_under_filled_read(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 4).observe()
    m = validators.mirror(ledger, POOL)
    world.dispatch("d", [row("A", reads=range(0, 6))], DenseBlockTable([[3, 4]], [2]))
    assert "read_uninitialized" in kinds(ledger)
    res = run_dense(validators, m, [[3, 4]], [2], owner_ids(m, ["A"]), [6])
    assert res.count == 1 and res.name == "read_uninitialized" and res.index == 1
    # Reading only the written prefix is fine.
    res = run_dense(validators, m, [[3, 4]], [1], owner_ids(m, ["A"]), [4])
    assert res.count == 0


def test_dense_published_block_readable_by_another_request(validators, world, ledger):
    world.alloc(3).request("A", [3], 4).observe().write("A", 0, 4)
    world.published(3).request("B", [3], 4).observe()
    m = validators.mirror(ledger, POOL)
    res = run_dense(validators, m, [[3], [3]], [1, 1], owner_ids(m, ["A", "B"]), [4, 4])
    assert res.count == 0


def test_dense_padding_rows(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    res = run_dense(validators, m, [[3], [0]], [1, 1], owner_ids(m, ["A", None]), [2, 0])
    assert res.count == 0  # the null block is reserved
    res = run_dense(validators, m, [[3], [3]], [1, 1], owner_ids(m, ["A", None]), [2, 0])
    assert res.count == 1 and res.name == "padding_row_names_managed_block" and res.row == 1
    # A reserved block at a live position of a real row.
    res = run_dense(validators, m, [[0]], [1], owner_ids(m, ["A"]), [2])
    assert res.count == 1 and res.name == "access_unmapped_position"


def test_dense_windowed_row_reads_from_its_start(validators, world, ledger):
    """A sliding-window row (`row_start`): the entries below the window's block are not
    consumed, so stale ids there are silent; inside the window the usual checks apply, and the
    first block needs only its slots from the window's start."""
    world.alloc(3, 4, 5).request("A", [3, 4, 5], 12).observe().write("A", 0, 10).observe()
    m = validators.mirror(ledger, POOL)
    # The host: positions [5, 10) read; block 3 (positions 0..3) is below the window.
    world.dispatch(
        "d",
        [row("A", reads=range(5, 10))],
        DenseBlockTable([[3, 4, 5]], [3], [1]),
    )
    assert kinds(ledger) == []
    # Entry 0 is outside the pool: not consumed, silent; the window's blocks are fine.
    res = run_dense(validators, m, [[99, 4, 5]], [3], owner_ids(m, ["A"]), [10], [5])
    assert res.count == 0
    # B wrote positions [5, 10) only: block 7 holds positions 4..7 with slot 4 unfilled, below
    # the window, so a window from 5 passes and a window from 4 reads an unfilled slot.
    world.alloc(6, 7, 8).request("B", [6, 7, 8], 12).observe().write("B", 5, 10).observe()
    mb = validators.mirror(ledger, POOL)
    res = run_dense(validators, mb, [[6, 7, 8]], [3], owner_ids(mb, ["B"]), [10], [5])
    assert res.count == 0
    res = run_dense(validators, mb, [[6, 7, 8]], [3], owner_ids(mb, ["B"]), [10], [4])
    assert res.count == 1 and res.name == "read_uninitialized" and res.index == 1


def test_wide_blocks_span_several_mask_words(validators):
    """A pool whose block size is neither a power of two nor within one mask word (48 slots,
    two words per block): the dense and slot-map validators follow the host on reads across
    the word boundary and on a write into an initialized slot."""
    from kvguard.core import DispatchDescriptor, DispatchRow

    lg = Ledger(Reporter(fail_fast=False))
    pool = "wide"
    lg.register_pool(
        PoolGeometry(pool=pool, num_blocks=8, block_size=48, reserved_blocks=frozenset({0}))
    )
    w = World(lg, pool=pool)
    w.alloc(3, 4).request("A", [3, 4], 70).observe()
    # Positions [0, 40) written: block 3's low word full, 8 bits of its high word.
    lg.dispatch(
        DispatchDescriptor(
            "w1",
            pool,
            "write",
            (DispatchRow("A", writes=range(0, 40)),),
            DenseBlockTable([[3, 4]], [2]),
        )
    )
    lg.complete("w1")
    m = validators.mirror(lg, pool)
    res = run_dense(validators, m, [[3, 4]], [2], owner_ids(m, ["A"]), [40])
    assert res.count == 0
    res = run_dense(validators, m, [[3, 4]], [2], owner_ids(m, ["A"]), [41])
    assert res.count == 1 and res.name == "read_uninitialized" and res.index == 0
    # A write into slot 39 (initialized) and into slot 40 (free of writes) of block 3.
    p = dev_i64(validators.drv, [3 * 48 + 40, 3 * 48 + 39])
    m.sync(0)
    validators.validate_slot_map(m, p, 8, 2, [m.owners.id_of("A")] * 2, 0, "slots")
    (res,) = validators.drain()
    validators.drv.mem_free(p)
    assert res.count == 1 and res.name == "double_write" and (res.row, res.index) == (1, 39)


def test_mirror_follows_a_free_between_syncs(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    assert run_dense(validators, m, [[3]], [1], owner_ids(m, ["A"]), [2]).count == 0
    world.drop_request("A").release(3).observe()
    assert (
        run_dense(validators, m, [[3]], [1], owner_ids(m, ["A"]), [2]).name == "access_free_block"
    )


def test_csr_page_list(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    drv = validators.drv
    indptr, indices = dev_i32(drv, [0, 2]), dev_i32(drv, [3, 4])
    good, bad = dev_i32(drv, [6 - BS]), dev_i32(drv, [1])
    validators.validate_csr(m, indptr, indices, good, 1, owner_ids(m, ["A"]), [6], 0, "csr-ok")
    validators.validate_csr(m, indptr, indices, bad, 1, owner_ids(m, ["A"]), [6], 0, "csr-lpl")
    ok, lpl = validators.drain()
    assert ok.count == 0
    assert lpl.count == 1 and lpl.name == "page_length_mismatch"
    world.dispatch("d", [row("A", reads=range(0, 6))], CSRPageList([0, 2], [3, 4], [1]))
    assert "page_length_mismatch" in kinds(ledger)
    # A page of another request.
    world.alloc(5).request("B", [5], 2).observe().write("B", 0, 2).observe()
    m.sync(0)
    foreign = dev_i32(drv, [3, 5])
    validators.validate_csr(m, indptr, foreign, good, 1, owner_ids(m, ["A"]), [6], 0, "csr-foreign")
    (res,) = validators.drain()
    assert res.count == 1 and res.name == "access_foreign_private" and res.entry == 5


def test_token_csr_page_list_against_the_mapping(validators, world, ledger):
    """The per-token page list at a page size above one: every entry must be the slot the
    mapping gives its position (``mapping_mismatch`` otherwise, also for a legal block of the
    same request), then the permission checks; the entry count must be the position count; a
    padding row names reserved blocks only."""
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    drv = validators.drv
    owners = owner_ids(m, ["A"])
    slots = [3 * BS + i for i in range(BS)] + [4 * BS + i for i in range(6 - BS)]
    indptr = dev_i32(drv, [0, len(slots)])
    good = dev_i32(drv, slots)
    validators.validate_token_csr(m, indptr, good, 1, owners, [0], [6], [[3, 4]], 0, "tc-ok")
    # The same request's own pages in the wrong order: legal blocks, a mapping mismatch.
    swapped = dev_i32(drv, [4 * BS + i for i in range(BS)] + [3 * BS + i for i in range(6 - BS)])
    validators.validate_token_csr(m, indptr, swapped, 1, owners, [0], [6], [[3, 4]], 0, "tc-swap")
    # A suffix table: the row reads positions [BS, 6) only, from its second page.
    suffix = dev_i32(drv, [4 * BS + i for i in range(6 - BS)])
    sptr = dev_i32(drv, [0, 6 - BS])
    validators.validate_token_csr(m, sptr, suffix, 1, owners, [BS], [6], [[3, 4]], 0, "tc-suffix")
    # The suffix table consumed from position 0: the entries are not the mapping's slots.
    validators.validate_token_csr(m, sptr, suffix, 1, owners, [0], [6 - BS], [[3, 4]], 0, "tc-old")
    # An entry count that is not the position count.
    validators.validate_token_csr(m, sptr, suffix, 1, owners, [0], [6], [[3, 4]], 0, "tc-len")
    ok, swap, suf, old, length = validators.drain()
    assert ok.count == 0 and suf.count == 0
    assert swap.count == 6 and swap.name == "mapping_mismatch" and swap.row == 0 and swap.index == 0
    assert old.count == 6 - BS and old.name == "mapping_mismatch"
    assert length.count >= 1 and "page_length_mismatch" in {length.name}
    # A hole in the mapping and a padding row.
    validators.validate_token_csr(m, indptr, good, 1, owners, [0], [6], [[None, 4]], 0, "tc-hole")
    pad_indptr = dev_i32(drv, [0, 1])
    pad_zero, pad_live = dev_i32(drv, [0]), dev_i32(drv, [3 * BS])
    validators.validate_token_csr(
        m, pad_indptr, pad_zero, 1, [V.PADDING_ROW], [0], [0], [], 0, "tc-pad-ok"
    )
    validators.validate_token_csr(
        m, pad_indptr, pad_live, 1, [V.PADDING_ROW], [0], [0], [], 0, "tc-pad-live"
    )
    hole, pad_ok, pad_bad = validators.drain()
    assert hole.count == BS and hole.name == "mapping_mismatch"
    assert (
        pad_ok.count == 0
        and pad_bad.count == 1
        and pad_bad.name == "padding_row_names_managed_block"
    )
    # The permission checks still apply once the entries match: a free block.
    world.alloc(5).request("B", [5], 2).observe().write("B", 0, 2).observe()
    world.release(5).drop_request("B").observe()
    m.sync(0)
    freed = dev_i32(drv, [5 * BS, 5 * BS + 1])
    validators.validate_token_csr(
        m, dev_i32(drv, [0, 2]), freed, 1, owners, [0], [2], [[5]], 0, "tc-free"
    )
    (res,) = validators.drain()
    assert res.count == 2 and res.name == "access_free_block"


def test_slot_map_expected_slots(validators, world, ledger):
    """With expected slots, a token writing another slot than the mapping's is a mismatch even
    when that slot is the request's own; ``None`` leaves a token unchecked."""
    world.alloc(3).request("A", [3], 2).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    drv = validators.drv
    owners = owner_ids(m, ["A", "A"])
    slots = dev_i32(drv, [3 * BS, 3 * BS + 1])
    validators.validate_slot_map(m, slots, 4, 2, owners, 0, "sm-ok", expected=[3 * BS, 3 * BS + 1])
    validators.validate_slot_map(
        m, slots, 4, 2, owners, 0, "sm-swap", expected=[3 * BS + 1, 3 * BS]
    )
    validators.validate_slot_map(m, slots, 4, 2, owners, 0, "sm-hole", expected=[None, 3 * BS])
    ok, swap, hole = validators.drain()
    assert ok.count == 0
    assert swap.count == 2 and swap.name == "mapping_mismatch"
    assert hole.count == 1 and hole.name == "mapping_mismatch" and hole.row == 1


def test_token_table(validators, world, ledger):
    world.alloc(3).request("A", [3], 3).observe().write("A", 0, 3).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    drv = validators.drv
    slots = [3 * BS + i for i in range(3)]
    t = dev_i32(drv, slots + [0] * 5)
    validators.validate_token_table(m, t, 8, 1, owner_ids(m, ["A"]), [3], 0, "tt-ok")
    validators.validate_token_table(m, t, 8, 1, owner_ids(m, ["A"]), [4], 0, "tt-beyond")
    ok, beyond = validators.drain()
    assert ok.count == 0
    # Position 3 maps slot 0: block 0 is reserved, a live position may not use it.
    assert beyond.count == 1 and beyond.name == "access_unmapped_position"
    world.dispatch("d", [row("A", reads=range(0, 4))], TokenTable([[*slots, 0]], [4]))
    assert kinds(ledger)  # the host rejects it too (beyond the mapping)
    assert BS >= 4
    stale = dev_i32(drv, [3 * BS, 3 * BS + 1, 3 * BS + 3])  # offset 3 of block 3 never written
    validators.validate_token_table(m, stale, 3, 1, owner_ids(m, ["A"]), [3], 0, "tt-unfilled")
    (res,) = validators.drain()
    assert res.count == 1 and res.name == "read_uninitialized" and res.index == 2


def test_slot_map_writes(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    drv = validators.drv
    a = m.owners.id_of("A")
    slots = (
        [3 * BS + i for i in range(6)]
        if BS >= 6
        else [3 * BS + i for i in range(BS)] + [4 * BS + i for i in range(6 - BS)]
    )
    p64 = dev_i64(drv, slots)
    validators.validate_slot_map(m, p64, 8, 6, [a] * 6, 0, "sm-ok")
    (res,) = validators.drain()
    assert res.count == 0
    # The host marks the written slots filled; after the sync a second write is a double write.
    world.dispatch("w", [row("A", writes=range(0, 6))], SlotMap(slots, [0] * 6, list(range(6))))
    assert kinds(ledger) == []
    m.sync(0)
    validators.validate_slot_map(m, p64, 8, 6, [a] * 6, 0, "sm-double")
    (res,) = validators.drain()
    assert res.count == 6 and res.name == "double_write"
    # -1 writes nothing; a padding token may name a reserved slot only.
    p32 = dev_i32(drv, [-1, 0, 3 * BS])
    validators.validate_slot_map(m, p32, 4, 3, [a, V.PADDING_ROW, V.PADDING_ROW], 0, "sm-pad")
    (res,) = validators.drain()
    assert res.count == 1 and res.name == "padding_row_names_managed_block" and res.row == 2
    # Published block: writable only at its pending slots.
    world.alloc(5).request("B", [5], 2).observe()
    world.published(5).observe()  # declared published before its slots were written
    m.sync(0)
    b = m.owners.id_of("B")
    pend = dev_i64(drv, [5 * BS, 5 * BS + 1, 5 * BS + 2])
    validators.validate_slot_map(m, pend, 8, 3, [b] * 3, 0, "sm-pending")
    (res,) = validators.drain()
    assert res.count == 1 and res.name == "write_to_published" and res.index == 2


def test_results_become_core_reports(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.drop_request("A").release(3).observe()
    m = validators.mirror(ledger, POOL)
    res = run_dense(validators, m, [[3]], [1], owner_ids(m, ["A"]), [2])
    rep = Reporter(fail_fast=False)
    assert V.report_results([res], rep, POOL) == 1
    r = rep.reports[0]
    assert r.kind == "access_free_block" and r.block == 3 and r.site == "gpu-validate:dense"


def test_many_launches_wrap_the_result_ring(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    m.sync(0)
    p = dev_i32(validators.drv, [3])
    n = validators.slots + 5
    for k in range(n):
        validators.validate_dense(m, p, 1, 1, 1, owner_ids(m, ["A"]), [2], 0, f"d{k}")
    out = validators.completed() + validators.drain()
    assert len(out) == n and all(r.count == 0 for r in out)
    assert C.sizeof(C.c_uint32) == 4


# TensorRT-LLM offsets

DIVISOR = 32  # layers_per_pool 16 x kv_factor 2, the Llama-3.2-1B pool of the TRT-LLM probe


def offsets_pairs(blocks):
    """The (K, V) pairs ``setOffsets`` writes for a row of block ids."""
    return [(b * DIVISOR, b * DIVISOR + 1) for b in blocks]


def run_offsets(v, mirror, pairs_rows, owners, ends, width=None):
    """``pairs_rows[r]`` is the row's list of (K, V) pairs; the table is laid out as TensorRT-LLM's
    ``[rows, 2, width]`` int32 batch table (K half, then V half)."""
    width = width or (max((len(r) for r in pairs_rows), default=1) or 1)
    flat = []
    for pairs in pairs_rows:
        ks = [k for k, _ in pairs] + [0] * (width - len(pairs))
        vs = [vv for _, vv in pairs] + [0] * (width - len(pairs))
        flat.extend(ks + vs)
    p = dev_i32(v.drv, flat)
    mirror.sync(0)
    v.validate_dense_offsets(
        mirror, p, 2 * width, width, len(pairs_rows), width, DIVISOR, owners, ends, 0, "offsets"
    )
    (res,) = v.drain()
    v.drv.mem_free(p)
    return res


def test_offsets_clean_table_reports_nothing(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    m = validators.mirror(ledger, POOL)
    res = run_offsets(validators, m, [offsets_pairs([3, 4])], owner_ids(m, ["A"]), [6])
    assert res.count == 0 and res.gpu_ms is not None
    # Stale entries beyond the consumed prefix are not decoded (the batch table is torch.empty).
    res = run_offsets(
        validators, m, [[*offsets_pairs([3, 4]), (7, -5)]], owner_ids(m, ["A"]), [6], width=3
    )
    assert res.count == 0


def test_offsets_positive_length_padding_checks_only_its_consumed_prefix(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    # The retained TensorRT-LLM dummy consumes one token through reserved block 0. The physical
    # row is wider and its unused cells are not valid K/V pairs.
    row = [*offsets_pairs([0]), (0, 0), (7, -5)]
    res = run_offsets(validators, m, [row], [V.PADDING_ROW], [1], width=3)
    assert res.count == 0
    row[0] = offsets_pairs([3])[0]
    res = run_offsets(validators, m, [row], [V.PADDING_ROW], [1], width=3)
    assert res.count == 1 and res.name == "padding_row_names_managed_block"


def test_offsets_foreign_private_block(validators, world, ledger):
    from kvguard.adapters.trtllm.translate import decode_offset

    world.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    world.write("A", 0, 2).write("B", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    pairs = offsets_pairs([4])
    assert decode_offset(*pairs[0], 16, 2) == 4
    # The host: row A's decoded table names B's block.
    world.dispatch("d", [row("A", reads=range(0, 2))], DenseBlockTable([[4]], [1]))
    assert "mapping_mismatch" in kinds(ledger)
    res = run_offsets(validators, m, [pairs], owner_ids(m, ["A"]), [2])
    assert (
        res.count == 1
        and res.name == "access_foreign_private"
        and (res.row, res.index, res.entry) == (0, 0, 4)
    )


def test_offsets_free_block(validators, world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.drop_request("A").release(3).observe()
    m = validators.mirror(ledger, POOL)
    res = run_offsets(validators, m, [offsets_pairs([3])], owner_ids(m, ["A"]), [2])
    assert res.count == 1 and res.name == "access_free_block" and res.entry == 3


def test_offsets_undecodable_pair_is_out_of_pool(validators, world, ledger):
    from kvguard.adapters.trtllm.translate import OUT_OF_POOL, decode_offset

    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    m = validators.mirror(ledger, POOL)
    k = 3 * DIVISOR
    cases = {
        "v_not_k_plus_1": (k, k + 2),
        "k_misaligned": (k + 1, k + 2),
        "secondary_pool_flag": (k | (1 << 31) - (1 << 32), k + 1),  # bit 31 set: negative int32
        "negative_v": (k, -1),
    }
    for name, (kk, vv) in cases.items():
        assert decode_offset(kk, vv, 16, 2) == OUT_OF_POOL, name
        res = run_offsets(validators, m, [[(kk, vv)]], owner_ids(m, ["A"]), [2])
        assert res.count == 1 and res.name == "mapping_out_of_pool", name
        assert (res.row, res.index, res.entry) == (0, 0, kk & 0xFFFFFFFF), name
    # Decodable but beyond the pool (16 blocks): the decoded block is the entry, as for the dense table.
    res = run_offsets(validators, m, [offsets_pairs([99])], owner_ids(m, ["A"]), [2])
    assert res.count == 1 and res.name == "mapping_out_of_pool" and res.entry == 99
    # The host, which knows the request's mapping, reports the decoded -2 at a live position as
    # a disagreement with the mapping (its stronger check); the device reports the pair itself.
    world.dispatch("d", [row("A", reads=range(0, 2))], DenseBlockTable([[OUT_OF_POOL]], [1]))
    assert "mapping_mismatch" in kinds(ledger)


def test_offsets_under_filled_and_published_rows(validators, world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 4).observe()
    m = validators.mirror(ledger, POOL)
    res = run_offsets(validators, m, [offsets_pairs([3, 4])], owner_ids(m, ["A"]), [6])
    assert res.count == 1 and res.name == "read_uninitialized" and res.index == 1
    world.write("A", 4, 6).published(3, 4).request("B", [3, 4], 6).observe()
    res = run_offsets(validators, m, [offsets_pairs([3, 4])] * 2, owner_ids(m, ["A", "B"]), [6, 6])
    assert res.count == 0, "published blocks are readable by any request"


def test_dense_write_rows_check_ownership_not_fills(validators):
    """A rewritable pool's state columns written through a dense table (`row_write`): the row's
    own block passes whether or not it was written before; another request's private block,
    a published block without pending slots and a free block are rejected."""
    from kvguard.core import DispatchDescriptor, DispatchRow

    lg = Ledger(Reporter(fail_fast=False))
    pool = "state"
    lg.register_pool(
        PoolGeometry(
            pool=pool, num_blocks=16, block_size=1, reserved_blocks=frozenset({0}), rewritable=True
        )
    )
    w = World(lg, pool=pool)
    w.alloc(3, 4, 5).request("A", [None, 3], 2).request("B", [None, 4], 2).observe()
    lg.dispatch(
        DispatchDescriptor(
            "w1",
            pool,
            "state",
            (DispatchRow("A", writes=range(1, 2)),),
            DenseBlockTable([[0, 3]], [2], [1]),
        )
    )
    lg.complete("w1")
    w.published(5).observe()
    m = validators.mirror(lg, pool)

    def write_row(entry, owner="A"):
        p = dev_i32(validators.drv, [0, entry])
        m.sync(0)
        validators.validate_dense(m, p, 2, 1, 2, owner_ids(m, [owner]), [2], 0, "w", [1], [1])
        (res,) = validators.drain()
        validators.drv.mem_free(p)
        return res

    assert write_row(3).count == 0  # A's own column, already initialized: an update
    res = write_row(4)
    assert res.count == 1 and res.name == "access_foreign_private"
    res = write_row(5)
    assert res.count == 1 and res.name == "write_to_published"
    res = write_row(9)
    assert res.count == 1 and res.name == "access_free_block"
    # The read of A's column needs the fill; B's column was never written.
    p = dev_i32(validators.drv, [0, 4])
    m.sync(0)
    validators.validate_dense(m, p, 2, 1, 2, owner_ids(m, ["B"]), [2], 0, "r", [1])
    (res,) = validators.drain()
    validators.drv.mem_free(p)
    assert res.count == 1 and res.name == "read_uninitialized"
