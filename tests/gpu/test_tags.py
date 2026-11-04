"""Differential tests of the tag tier's kernels against the host reference ``TagReference``
(opt-in: needs the A6000 and ``build/validators/validators.ptx`` from ``make code``; not part of
``make check``).

Run: ``envs/tools/bin/python -m pytest tests/gpu``. Every scenario is played through the
reference and the device with the same stamps, maintenance records and tables; the device must
count exactly the reference's violations and record one of them first.
"""

import ctypes as C
from array import array

import pytest
from kvguard.core import Reporter
from kvguard.core.model import ExternalCopy, ExternalFill, PoolGeometry
from kvguard.gpu import tags as T
from kvguard.gpu import validate as V

from tests.gpu.test_validators import dev_i32, dev_i64
from tests.unit.conftest import BS, POOL

pytestmark = pytest.mark.skipif(
    not V.default_ptx_path().exists(), reason="run `make code` first (validators.ptx)"
)

A, B = 1, 2


@pytest.fixture(scope="module")
def validators():
    try:
        return V.Validators(timing=True)
    except Exception as exc:  # no driver, no GPU
        pytest.skip(f"no CUDA device: {exc}")


@pytest.fixture
def geometry():
    return PoolGeometry(pool=POOL, num_blocks=16, block_size=BS, reserved_blocks=frozenset({0}))


@pytest.fixture
def pair(validators, geometry):
    """The reference and the device mirror of one fresh pool, every managed block allocated."""
    ref, dev = T.TagReference(geometry), T.TagMirror(validators, geometry)
    blocks = list(range(1, geometry.num_blocks))
    ref.allocate(blocks)
    dev.allocate(blocks, 0)
    same(drain(dev), [])
    return ref, dev


def drain(mirror):
    return [r for r in mirror.v.drain() if T.is_tag_result(r)]


def same(results, expected):
    """The device's results of one launch agree with the reference's violation list."""
    (res,) = results
    assert res.count == len(expected), (res, expected)
    if expected:
        firsts = {(v.kind, v.row, v.position, v.slot, v.w0, v.w1) for v in expected}
        assert (res.kind, res.row, res.index, res.entry, *res.stored) in firsts, (res, expected)
    return res


def stamp_both(ref, dev, slots, owners, positions, rewrite=False):
    exp = ref.stamp(slots, owners, positions, rewrite)
    p = dev_i64(dev.drv, slots)
    dev.stamp(p, 8, len(slots), owners, positions, 0, "write", rewrite)
    res = same(drain(dev), exp)
    dev.drv.mem_free(p)
    return exp, res


def row_slots(blocks, start, stop):
    return [blocks[p // BS] * BS + p % BS for p in range(start, stop)]


def dense_both(ref, dev, table, owners, ends, starts=None):
    width = max((len(r) for r in table), default=1) or 1
    exp = ref.check_dense(table, owners, ends, starts, width)
    flat = [x for r in table for x in r + [0] * (width - len(r))]
    p = dev_i32(dev.drv, flat)
    dev.check_dense(p, width, len(table), width, owners, ends, 0, "attn", starts)
    res = same(drain(dev), exp)
    dev.drv.mem_free(p)
    return exp, res


def test_stamp_and_dense_read_agree(pair):
    ref, dev = pair
    n = 6
    stamp_both(ref, dev, row_slots([3, 4], 0, n), [A] * n, list(range(n)))
    assert dev.read_back([3 * BS, 4 * BS + 1]) == [
        ((A << 8) | T.WRITTEN | T.ALLOC, 0),
        ((A << 8) | T.WRITTEN | T.ALLOC, 5),
    ]
    assert dense_both(ref, dev, [[3, 4]], [A], [6])[0] == []
    assert dense_both(ref, dev, [[3, 4]], [A], [7])[1].name == "read_uninitialized"
    assert dense_both(ref, dev, [[3, 4]], [B], [6])[1].name == "access_foreign_private"
    assert dense_both(ref, dev, [[4, 3]], [A], [2])[1].name == "tag_position_mismatch"
    assert dense_both(ref, dev, [[3, 4]], [A], [6], [4])[0] == []  # a window's suffix
    # Padding rows and the reserved namespace.
    assert dense_both(ref, dev, [[0, 0]], [V.PADDING_ROW], [0])[0] == []
    assert dense_both(ref, dev, [[0, 3]], [V.PADDING_ROW], [0])[1].name == (
        "padding_row_names_managed_block"
    )
    assert dense_both(ref, dev, [[0]], [A], [1])[1].name == "access_unmapped_position"
    assert dense_both(ref, dev, [[99]], [A], [1])[1].name == "mapping_out_of_pool"


def test_stamp_faults_agree(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3], 0, 2), [A, A], [0, 1])
    assert stamp_both(ref, dev, [3 * BS], [A], [0])[1].name == "double_write"
    assert stamp_both(ref, dev, [3 * BS], [A], [0], rewrite=True)[0] == []
    assert stamp_both(ref, dev, [3 * BS], [B], [0], rewrite=True)[1].name == (
        "access_foreign_private"
    )
    assert stamp_both(ref, dev, [0, 1], [A, A], [8, 9])[1].name == "access_unmapped_position"
    assert stamp_both(ref, dev, [3 * BS + 3], [V.PADDING_ROW], [0])[1].name == (
        "padding_row_names_managed_block"
    )
    assert stamp_both(ref, dev, [-1, 0], [A, V.PADDING_ROW], [0, 0])[0] == []
    assert stamp_both(ref, dev, [99 * BS], [A], [0])[1].name == "mapping_out_of_pool"
    # A 32-bit slot map (SGLang's out_cache_loc).
    exp = ref.stamp([5 * BS], [A], [7])
    p = dev_i32(dev.drv, [5 * BS])
    dev.stamp(p, 4, 1, [A], [7], 0, "write32")
    same(drain(dev), exp)
    dev.drv.mem_free(p)
    assert dev.read_back([5 * BS]) == [((A << 8) | T.WRITTEN | T.ALLOC, 7)]


def test_free_publish_and_fill_agree(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3], 0, BS), [A] * BS, list(range(BS)))
    ref.publish([3])
    dev.publish([3], 0)
    same(drain(dev), [])
    assert dense_both(ref, dev, [[3]], [B], [BS])[0] == []
    ref.free([3])
    dev.free([3], 0)
    same(drain(dev), [])
    res = dense_both(ref, dev, [[3]], [A], [BS])[1]
    assert res.name == "access_free_block" and res.stored[0] >> 8 == A
    assert dev.read_back([3 * BS]) == [((A << 8) | T.FREED, 0)]
    # A write into the freed block is a write into free memory until it is allocated again;
    # then the next write clears the free and the old publication.
    assert stamp_both(ref, dev, [3 * BS], [B], [0])[1].name == "access_free_block"
    ref.allocate([3])
    dev.allocate([3], 0)
    same(drain(dev), [])
    stamp_both(ref, dev, row_slots([3], 1, BS), [B] * (BS - 1), list(range(1, BS)))
    assert dense_both(ref, dev, [[3]], [A], [BS])[1].name == "access_foreign_private"
    # A fill carries positions; a fill over written slots is a double write.
    exp = ref.fill(5, 0, BS, A, pos0=0)
    dev.fill(5, 0, BS, A, 0, pos0=0)
    same(drain(dev), exp)
    assert dense_both(ref, dev, [[5]], [A], [BS])[0] == []
    exp = ref.fill(5, 0, 1, A, pos0=0)
    dev.fill(5, 0, 1, A, 0, pos0=0)
    assert same(drain(dev), exp).name == "double_write"
    exp = ref.update([(99, T.OP_FREE, 0, 0, 0, BS)])
    dev.update([(99, T.OP_FREE, 0, 0, 0, BS)], 0)
    assert same(drain(dev), exp).name == "mapping_out_of_pool"


def test_csr_and_token_table_agree(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3, 4], 0, 6), [A] * 6, list(range(6)))

    def csr(indptr, indices, lpl, owners, ends):
        exp = ref.check_csr(indptr, indices, lpl, owners, ends)
        ps = [dev_i32(dev.drv, x) for x in (indptr, indices, lpl)]
        dev.check_csr(*ps, len(indptr) - 1, owners, ends, 0, "csr")
        res = same(drain(dev), exp)
        for p in ps:
            dev.drv.mem_free(p)
        return exp, res

    assert csr([0, 2], [3, 4], [2], [A], [6])[0] == []
    assert csr([0, 2], [3, 4], [3], [A], [6])[1].name == "page_length_mismatch"
    assert csr([0, 1], [4], [2], [A], [2])[1].name == "tag_position_mismatch"
    assert csr([0, 2, 3], [3, 4, 0], [2, 1], [A, V.PADDING_ROW], [6, 0])[0] == []

    def tt(table, owners, ends, firsts=None):
        width = max(len(r) for r in table)
        exp = ref.check_token_table(table, owners, ends, firsts)
        p = dev_i32(dev.drv, [x for r in table for x in r + [0] * (width - len(r))])
        dev.check_token_table(p, width, len(table), owners, ends, 0, "tt", firsts)
        res = same(drain(dev), exp)
        dev.drv.mem_free(p)
        return exp, res

    slots = row_slots([3, 4], 0, 6)
    assert tt([slots], [A], [6])[0] == []
    assert tt([slots[1:] + slots[:1]], [A], [6])[1].name == "tag_position_mismatch"
    assert tt([slots[2:]], [A], [6], [2])[0] == []
    assert tt([[-1]], [A], [1])[1].name == "mapping_out_of_pool"
    assert tt([slots, slots], [A, B], [6, 6])[1].name == "access_foreign_private"


def test_rollback_and_csr_starts_agree(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3, 4], 0, 8), [A] * 8, list(range(8)))
    exp = ref.invalidate([(3, 1), (3, 2), (3, 3)])
    dev.invalidate([(3, 1), (3, 2), (3, 3)], 0)
    same(drain(dev), exp)
    assert dev.read_back([3 * BS + 1]) == [((A << 8) | T.ALLOC, 1)]
    stamp_both(ref, dev, row_slots([3], 1, 4), [A] * 3, [1, 2, 3])
    assert dense_both(ref, dev, [[3, 4]], [A], [8])[0] == []

    def csr(indptr, indices, lpl, owners, ends, starts=None):
        exp = ref.check_csr(indptr, indices, lpl, owners, ends, entry_slots=1, row_start=starts)
        ps = [dev_i32(dev.drv, x) for x in (indptr, indices, lpl)]
        dev.check_csr(*ps, len(indptr) - 1, owners, ends, 0, "csr", entry_slots=1, row_start=starts)
        res = same(drain(dev), exp)
        for p in ps:
            dev.drv.mem_free(p)
        return exp, res

    slots = row_slots([3, 4], 0, 8)
    assert csr([0, 3], slots[5:], [1], [A], [8], [5])[0] == []
    assert csr([0, 3], slots[5:], [1], [A], [8])[1].name in (
        "page_length_mismatch",
        "tag_position_mismatch",
    )
    assert csr([0, 3], slots[4:7], [1], [A], [8], [5])[1].name == "tag_position_mismatch"


def test_dense_offsets_agree(validators):
    g = PoolGeometry(pool="trt", num_blocks=16, block_size=BS)
    ref, dev = T.TagReference(g), T.TagMirror(validators, g)
    ref.allocate([3, 4])
    dev.allocate([3, 4], 0)
    same(drain(dev), [])
    divisor = 4  # layers * kv factor
    stamp_both(ref, dev, row_slots([3, 4], 0, 6), [A] * 6, list(range(6)))

    def offsets(pairs_by_row, owners, ends):
        width = max(len(r) for r in pairs_by_row)
        table = []
        for r in pairs_by_row:
            ks = [k for k, _ in r] + [0] * (width - len(r))
            vs = [v for _, v in r] + [0] * (width - len(r))
            table += ks + vs
        p = dev_i32(dev.drv, table)
        dev.check_dense_offsets(p, 2 * width, width, len(pairs_by_row), width, divisor, owners,
                                ends, 0, "offsets")  # fmt: skip
        res = [r for r in dev.v.drain() if T.is_tag_result(r)]
        dev.drv.mem_free(p)
        return res

    enc = lambda b: (b * divisor, b * divisor + 1)  # noqa: E731
    (res,) = offsets([[enc(3), enc(4)]], [A], [6])
    assert res.count == 0
    (res,) = offsets([[enc(4), enc(3)]], [A], [2])
    assert res.count == 2 and res.name == "tag_position_mismatch"
    (res,) = offsets([[(5, 7)]], [A], [1])
    assert res.count == 1 and res.name == "mapping_out_of_pool"


def test_ranges_and_external_descriptors_agree(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3], 0, BS), [A] * BS, list(range(BS)))

    def ranges(recs):
        exp = ref.check_ranges(recs)
        dev.check_ranges(recs, 0, "copy")
        return exp, same(drain(dev), exp)

    assert ranges([(3, A, 0, 0, BS)])[0] == []
    assert ranges([(3, B, 0, 0, BS)])[1].name == "access_foreign_private"
    assert ranges([(3, 0, T.ANY_POS, 0, BS)])[0] == []  # owner 0: anyone, no position
    assert ranges([(3, A, 1, 0, BS)])[1].name == "tag_position_mismatch"
    assert ranges([(6, A, T.ANY_POS, 0, 1)])[1].name == "read_uninitialized"
    # The core's descriptors: a copy reads the source and stamps the destination, a fill stamps.
    dev.owners.id_of("A")  # owner id 1 == A
    exp_src = ref.check_ranges([(3, A, T.ANY_POS, 0, 2)])
    exp_dst = ref.fill(7, 0, 2, A)
    dev.external_copy(ExternalCopy(POOL, 3, POOL, 7, range(0, 2), request="A"), dev, 0)
    rs = drain(dev)
    assert len(rs) == 2 and rs[0].count == len(exp_src) == 0 and rs[1].count == len(exp_dst) == 0
    assert dev.read_back([7 * BS, 7 * BS + 1]) == [((A << 8) | T.WRITTEN | T.ALLOC, T.ANY_POS)] * 2
    exp = ref.fill(8, 0, 1, A)
    dev.external_fill(ExternalFill(POOL, 8, range(0, 1), request="A"), 0)
    same(drain(dev), exp)
    assert ranges([(8, B, T.ANY_POS, 0, 1)])[1].name == "access_foreign_private"


def test_reports_and_stats(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3], 0, 2), [A, A], [0, 1])
    dev.free([3], 0)
    p = dev_i32(dev.drv, [3])
    dev.check_dense(p, 1, 1, 1, [A], [2], 0, "attn-kv@7")
    rep = Reporter(fail_fast=False)
    rs = drain(dev)
    assert T.report_tag_results(rs, rep, POOL) == 1
    (r,) = rep.reports
    assert r.kind == "access_free_block" and r.site == "gpu-tags:attn-kv@7"
    assert "tag held owner 1 position 0 flags F" in r.detail
    dev.drv.mem_free(p)
    st = dev.stats.as_dict()
    assert st["stamps"] == 1 and st["tokens_stamped"] == 2 and st["updates"] == 2
    assert st["checks"] == 1 and st["positions_checked"] == 2 and st["launches"] == 4
    assert dev.nbytes == 8 * 16 * BS


def test_set_tag_injects_a_device_only_fault(pair):
    ref, dev = pair
    stamp_both(ref, dev, row_slots([3], 0, BS), [A] * BS, list(range(BS)))
    dev.set_tag(3 * BS + 1, (B << 8) | T.WRITTEN | T.ALLOC, 1, 0)
    p = dev_i32(dev.drv, [3])
    dev.check_dense(p, 1, 1, 1, [A], [BS], 0, "attn")
    (res,) = drain(dev)
    assert res.count == 1 and res.name == "access_foreign_private" and res.entry == 3 * BS + 1
    assert ref.check_dense([[3]], [A], [BS]) == []  # the reference did not see the fault
    dev.drv.mem_free(p)


def test_cost_of_a_decode_batch(validators, capsys):
    """Not a measurement for the record: prints the device time of one dense check over 64
    rows of 512 positions and of the matching stamp, for a first look."""
    g = PoolGeometry(pool="big", num_blocks=4096, block_size=16, reserved_blocks=frozenset({0}))
    dev = T.TagMirror(validators, g)
    rows, ctx = 64, 512
    bs = 16
    table = [[1 + r * (ctx // bs) + j for j in range(ctx // bs)] for r in range(rows)]
    dev.allocate([b for r in table for b in r], 0)
    owners = [1 + r for r in range(rows)]
    # Stamp every position of every row.
    slots, tok_owner, tok_pos = [], [], []
    for r in range(rows):
        for p in range(ctx):
            slots.append(table[r][p // bs] * bs + p % bs)
            tok_owner.append(owners[r])
            tok_pos.append(p)
    ps = dev_i64(dev.drv, slots)
    dev.stamp(ps, 8, len(slots), tok_owner, tok_pos, 0, "stamp")
    pt = dev_i32(dev.drv, [x for r in table for x in r])
    for _ in range(3):
        dev.check_dense(pt, ctx // bs, rows, ctx // bs, owners, [ctx] * rows, 0, "attn")
    rs = drain(dev)
    assert all(r.count == 0 for r in rs)
    ms = [r.gpu_ms for r in rs]
    with capsys.disabled():
        print(
            f"\ntag tier: stamp of {len(slots)} tokens {ms[0] * 1e3:.1f} us; dense check of "
            f"{rows} rows x {ctx} positions {', '.join(f'{m * 1e3:.1f}' for m in ms[1:])} us"
        )
    for p in (ps, pt):
        dev.drv.mem_free(p)


def test_slot_struct_size():
    assert C.sizeof(C.c_uint32) * 2 == 8 and array("I").itemsize == 4


def test_update_slots_and_host_stamp_agree(pair):
    ref, dev = pair
    # A host-derived slot map (TensorRT-LLM's KV append has no device slot map).
    slots = row_slots([3, 4], 0, 6)
    exp = ref.stamp(slots, [A] * 6, list(range(6)))
    dev.stamp_host(slots, [A] * 6, list(range(6)), 0, "write-host")
    same(drain(dev), exp)
    assert dense_both(ref, dev, [[3, 4]], [A], [6])[0] == []
    # Frees and publishes by slot list, from a host list and from a device tensor as it is.
    exp = ref.update_slots([3 * BS + 1, -1, 3 * BS + 2], T.OP_FREE)
    dev.update_slots(T.OP_FREE, 0, "free-slots", slots=[3 * BS + 1, -1, 3 * BS + 2])
    same(drain(dev), exp)
    assert dense_both(ref, dev, [[3, 4]], [A], [6])[1].name == "access_free_block"
    p = dev_i64(dev.drv, [4 * BS, 4 * BS + 1])
    exp = ref.update_slots([4 * BS, 4 * BS + 1], T.OP_PUBLISH)
    dev.update_slots(T.OP_PUBLISH, 0, "publish-slots", slots_ptr=p, elem_bytes=8, n=2)
    same(drain(dev), exp)
    dev.drv.mem_free(p)
    assert dense_both(ref, dev, [[3, 4]], [B], [6], [4])[0] == []  # shared now
    exp = ref.update_slots([99 * BS], T.OP_FREE)
    dev.update_slots(T.OP_FREE, 0, "bad", slots=[99 * BS])
    assert same(drain(dev), exp).name == "mapping_out_of_pool"


def test_tag_tier_follows_the_ledger_events(validators, geometry):
    """The tier as the ledger's listener: a mediated free, store, reuse, external fill and copy
    each reach the tags without the adapter naming the tier."""
    from kvguard.core import Ledger, Reporter
    from kvguard.core.model import BlockSnapshot, BlockState, RequestSnapshot, Snapshot

    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(geometry)
    tier = T.TagTier(ledger, [POOL], validators)
    tier.stream = lambda: 0
    dev = tier.mirrors[POOL]
    a = tier.owners.id_of("A")
    ledger.allocate(POOL, [3, 4, 5, 6])
    ledger.observe(
        Snapshot(
            1,
            (RequestSnapshot("A", POOL, (3, 4), 2 * BS, 0, 0),),
            tuple(BlockSnapshot(POOL, b, BlockState.PRIVATE) for b in (3, 4, 5, 6)),
        )
    )
    dev.stamp_host(row_slots([3, 4], 0, 2 * BS), [a] * (2 * BS), list(range(2 * BS)), 0, "w")
    # An external fill with a known position, and a copy that keeps it.
    ledger.external_fill(ExternalFill(POOL, 5, range(0, BS), request="A", position=2 * BS))
    ledger.external_copy(ExternalCopy(POOL, 5, POOL, 6, range(0, BS), request="A", position=2 * BS))
    rs = drain(dev)
    assert all(r.count == 0 for r in rs) and [r.tag for r in rs][-3:] == [
        f"tag:{POOL}|fill-external",
        f"tag:{POOL}|copy-copy",
        f"tag:{POOL}|copy-copy",
    ]  # fmt: skip
    assert dev.read_back([6 * BS + 1]) == [((a << 8) | T.WRITTEN | T.ALLOC, 2 * BS + 1)]
    # The ledger's free reaches the tags; a store publishes; a reuse frees the tail.
    ledger.free(POOL, [3])
    ledger.store(POOL, [4])
    rs = drain(dev)
    assert [r.tag for r in rs] == [f"tag:{POOL}|free@1", f"tag:{POOL}|store@1"]
    assert all(r.count == 0 for r in rs)
    assert dev.read_back([3 * BS, 4 * BS]) == [
        ((a << 8) | T.FREED, 0),
        ((a << 8) | T.WRITTEN | T.SHARED | T.ALLOC, BS),
    ]
    ledger.reuse(POOL, 4, 2)
    drain(dev)
    assert dev.read_back([4 * BS + 1, 4 * BS + 2]) == [
        ((a << 8) | T.WRITTEN | T.SHARED | T.ALLOC, BS + 1),
        ((a << 8) | T.ALLOC, BS + 2),
    ]
    # The collector routes the tier's results and leaves the validators' alone.
    p = dev_i32(dev.drv, [3])
    dev.check_dense(p, 1, 1, 1, [a], [BS], 0, "attn@2")
    rep = Reporter(fail_fast=False)
    rest = tier.collect(tier.drain(), rep)
    assert rest == [] and [r.kind for r in rep.reports] == ["access_free_block"]
    assert tier.stats()["tag_reports"] == 1 and tier.stats()[f"tag_{POOL}_checks"] == 2
    dev.drv.mem_free(p)
    ledger.listeners.remove(tier)
