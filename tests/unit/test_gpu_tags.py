"""The tag tier's semantics, hermetic: the host reference ``TagReference`` slot for slot, and the
differential against the ledger on the fault scenarios the two tiers share. The device kernels are
covered by ``tests/gpu/test_tags.py`` (opt-in, needs the GPU)."""

import pytest
from kvguard.core import DenseBlockTable
from kvguard.core.model import PoolGeometry
from kvguard.gpu import tags as T
from kvguard.gpu.validate import KINDS, PADDING_ROW

from tests.unit.conftest import BS, POOL, kinds, row

A, B = 1, 2  # owner ids


@pytest.fixture
def ref(geometry):
    r = T.TagReference(geometry)
    r.allocate(range(1, geometry.num_blocks))  # every managed block allocated, as a test world
    return r


def names(violations):
    return sorted({v.name for v in violations})


def stamp_row(ref, owner, blocks, start, stop, rewrite=False):
    """Write positions [start, stop) of a request mapped to ``blocks``."""
    slots = [blocks[p // BS] * BS + p % BS for p in range(start, stop)]
    return ref.stamp(slots, [owner] * len(slots), list(range(start, stop)), rewrite)


def test_reserved_blocks_are_any_in_pool_set():
    assert T.reserved_set(PoolGeometry(POOL, 8, BS)) == frozenset()
    assert T.reserved_set(PoolGeometry(POOL, 8, BS, reserved_blocks=frozenset({0, 3}))) == {0, 3}
    ref = T.TagReference(PoolGeometry(POOL, 8, BS, reserved_blocks=frozenset({3})))
    assert ref.check_dense([[3]], [PADDING_ROW], [0]) == []
    assert names(ref.check_dense([[3]], [A], [1])) == ["access_unmapped_position"]
    assert names(ref.check_dense([[0]], [PADDING_ROW], [0])) == ["padding_row_names_managed_block"]


def test_stamp_then_read_by_the_owner_at_the_position(ref):
    assert stamp_row(ref, A, [3, 4], 0, 6) == []
    assert ref.tag(3 * BS) == ((A << 8) | T.WRITTEN | T.ALLOC, 0)
    assert ref.tag(4 * BS + 1) == ((A << 8) | T.WRITTEN | T.ALLOC, 5)
    assert ref.check_dense([[3, 4]], [A], [6]) == []
    # The row asks for one more position than was written.
    assert names(ref.check_dense([[3, 4]], [A], [7])) == ["read_uninitialized"]
    # Another request reads the private block.
    assert names(ref.check_dense([[3, 4]], [B], [6])) == ["access_foreign_private"]
    # The same slots read as other positions (a table that feeds the wrong prefix).
    v = ref.check_dense([[4, 3]], [A], [2])
    assert names(v) == ["tag_position_mismatch"] and v[0].position == 0 and v[0].w1 == 4
    assert len(v) == 2 and v[1].position == 1 and v[1].w1 == 5


def test_published_blocks_are_readable_by_anyone_at_their_positions(ref):
    stamp_row(ref, A, [3], 0, BS)
    ref.publish([3])
    assert ref.check_dense([[3]], [B], [BS]) == []
    assert names(ref.check_dense([[3]], [B], [BS], [BS // 2])) == []  # a window's suffix
    # A second owner writing a published slot is a double write; a pending slot is fine.
    stamp_row(ref, A, [5], 0, 2)
    ref.publish([5])
    assert names(stamp_row(ref, B, [5], 0, 1)) == ["double_write"]
    assert stamp_row(ref, B, [5], 2, 3) == []


def test_free_keeps_the_diagnosis_until_the_next_write(ref):
    stamp_row(ref, A, [3], 0, BS)
    ref.free([3])
    v = ref.check_dense([[3]], [A], [BS])
    assert names(v) == ["access_free_block"] and v[0].w0 >> 8 == A and v[0].w1 == 0
    # A never-written block freed is free to a read (no allocation, no write).
    ref.free([6])
    assert names(ref.check_dense([[6]], [A], [1])) == ["access_free_block"]
    # A write into a freed, not reallocated block is a write into free memory; after the
    # allocation event the next write clears the freed flag and the old publication.
    ref.publish([3])
    ref.free([3])
    assert names(stamp_row(ref, B, [3], 0, 1)) == ["access_free_block"]
    ref.allocate([3])
    assert stamp_row(ref, B, [3], 1, BS) == []
    assert names(ref.check_dense([[3]], [A], [BS])) == ["access_foreign_private"]
    # A never-allocated block is free to a read and to a write.
    fresh = T.TagReference(ref.geometry)
    assert names(fresh.check_dense([[3]], [A], [1])) == ["access_free_block"]
    assert names(stamp_row(fresh, A, [3], 0, 1)) == ["access_free_block"]
    fresh.allocate([3])
    assert stamp_row(fresh, A, [3], 1, 2) == []


def test_double_write_and_rewritable_pools(ref):
    stamp_row(ref, A, [3], 0, 2)
    assert names(stamp_row(ref, A, [3], 0, 1)) == ["double_write"]
    assert stamp_row(ref, A, [3], 0, 2, rewrite=True) == []
    assert names(stamp_row(ref, B, [3], 0, 1, rewrite=True)) == ["access_foreign_private"]
    ref.publish([3])
    assert names(stamp_row(ref, A, [3], 0, 1, rewrite=True)) == ["write_to_published"]


def test_reserved_namespace_and_padding_rows(ref):
    stamp_row(ref, A, [3], 0, BS)
    # A padding row may name the null block only, a live row never.
    assert ref.check_dense([[0, 0]], [PADDING_ROW], [0]) == []
    assert names(ref.check_dense([[0, 3]], [PADDING_ROW], [0])) == [
        "padding_row_names_managed_block"
    ]
    assert names(ref.check_dense([[0]], [A], [1])) == ["access_unmapped_position"]
    assert names(ref.check_dense([[99]], [A], [1])) == ["mapping_out_of_pool"]
    # The same for the slot map: a live token into the null block.
    assert names(ref.stamp([0, 1], [A, A], [8, 9])) == ["access_unmapped_position"]
    assert names(ref.stamp([3 * BS + 3], [PADDING_ROW], [0])) == ["padding_row_names_managed_block"]
    assert ref.stamp([-1, 0], [A, PADDING_ROW], [0, 0]) == []
    assert names(ref.stamp([99 * BS], [A], [0])) == ["mapping_out_of_pool"]


def test_csr_and_token_table_representations(ref):
    stamp_row(ref, A, [3, 4], 0, 6)
    assert ref.check_csr([0, 2], [3, 4], [2], [A], [6]) == []
    assert names(ref.check_csr([0, 2], [3, 4], [3], [A], [6])) == ["page_length_mismatch"]
    assert names(ref.check_csr([0, 1], [4], [2], [A], [2])) == ["tag_position_mismatch"]
    slots = [3 * BS + i for i in range(BS)] + [4 * BS, 4 * BS + 1]
    assert ref.check_token_table([slots], [A], [6]) == []
    rotated = slots[1:] + slots[:1]
    assert names(ref.check_token_table([rotated], [A], [6])) == ["tag_position_mismatch"]
    # A suffix row starting at its window: positions [2, 6) through columns 0..3.
    assert ref.check_token_table([slots[2:]], [A], [6], [2]) == []
    assert names(ref.check_token_table([[-1]], [A], [1])) == ["mapping_out_of_pool"]


def test_rollback_unwrites_rejected_slots(ref):
    # A speculative verify writes candidates; the rejected ones are rewritten next step.
    stamp_row(ref, A, [3], 0, 4)
    assert names(stamp_row(ref, A, [3], 1, 4)) == ["double_write"]
    assert ref.invalidate([(3, 1), (3, 2), (3, 3)]) == []
    assert ref.tag(3 * BS + 1) == ((A << 8) | T.ALLOC, 1)  # allocated, no longer written
    assert stamp_row(ref, A, [3], 1, 4) == []
    assert ref.check_dense([[3]], [A], [4]) == []
    # The invalidation keeps a publication; a rewrite of a published slot is still reported.
    ref.publish([3])
    assert ref.invalidate([(3, 2)]) == []
    assert ref.tag(3 * BS + 2)[0] & T.SHARED
    assert stamp_row(ref, A, [3], 2, 3) == []  # the pending slot of a published block


def test_csr_rows_may_start_at_their_window(ref):
    stamp_row(ref, A, [3, 4], 0, 8)
    # A sliding-window wrapper's row reads positions [5, 8) through three slot entries.
    slots = [3 * BS + 1, 3 * BS + 2, 3 * BS + 3, 4 * BS, 4 * BS + 1, 4 * BS + 2, 4 * BS + 3]
    tail = slots[4:]  # the slots of positions 5, 6, 7
    assert ref.check_csr([0, 3], tail, [1], [A], [8], entry_slots=1, row_start=[5]) == []
    assert names(ref.check_csr([0, 3], tail, [1], [A], [8], entry_slots=1)) == [
        "page_length_mismatch", "tag_position_mismatch",
    ]  # fmt: skip
    assert names(
        ref.check_csr([0, 3], slots[3:6], [1], [A], [8], entry_slots=1, row_start=[5])
    ) == ["tag_position_mismatch"]


def test_fills_copies_and_ranges(ref):
    # A restore from another tier stamps the destination with the positions it carries.
    assert ref.fill(3, 0, BS, A, pos0=0) == []
    assert ref.check_dense([[3]], [A], [BS]) == []
    # A copy reads the source range and fills the destination; an unknown position compares
    # with nothing (a recurrent-state slot).
    assert ref.check_ranges([(3, A, 0, 0, BS)]) == []
    assert ref.fill(5, 0, 1, A) == [] and ref.tag(5 * BS) == (
        (A << 8) | T.WRITTEN | T.ALLOC,
        T.ANY_POS,
    )
    assert ref.check_ranges([(5, A, 7, 0, 1)]) == []  # stored position unknown
    assert ref.check_ranges([(5, 0, T.ANY_POS, 0, 1)]) == []  # owner 0: anyone
    assert names(ref.check_ranges([(5, B, T.ANY_POS, 0, 1)])) == ["access_foreign_private"]
    assert names(ref.check_ranges([(6, A, T.ANY_POS, 0, 1)])) == ["read_uninitialized"]
    # A fill over written slots is a double write unless rewriting.
    assert names(ref.fill(3, 0, 1, A, pos0=0)) == ["double_write"]
    assert ref.fill(3, 0, 1, A, pos0=0, rewrite=True) == []
    assert names(ref.update([(99, T.OP_FREE, 0, 0, 0, BS)])) == ["mapping_out_of_pool"]


def test_kinds_are_named_as_the_host_names_them():
    assert KINDS[T.K_TAG_POSITION][0] == "tag_position_mismatch"
    for k in (T.K_FREE, T.K_FOREIGN, T.K_UNDER_FILLED, T.K_DOUBLE_WRITE, T.K_WRITE_PUBLISHED):
        assert k in KINDS


# -- the two tiers on the same scenarios --------------------------------------------------------


def both(world, ledger, ref, scenario):
    """Play ``scenario`` through the ledger (World) and the tag reference; return the kinds each
    reports."""
    ledger.reporter.reports.clear()
    tag_kinds = scenario(world, ref)
    return kinds(ledger), sorted(set(tag_kinds))


def test_early_free_reported_by_both_tiers(world, ledger, ref):
    def scenario(w, ref):
        w.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
        stamp_row(ref, A, [3, 4], 0, 6)
        w.release(3)  # freed while A still maps it
        ref.free([3])
        w.dispatch("attn", [row("A", reads=range(0, 6))], DenseBlockTable([[3, 4]], [2]))
        return names(ref.check_dense([[3, 4]], [A], [6]))

    host, tags = both(world, ledger, ref, scenario)
    assert "access_free_block" in host and tags == ["access_free_block"]


def test_uninitialized_read_reported_by_both_tiers(world, ledger, ref):
    def scenario(w, ref):
        w.alloc(3, 4).request("A", [3, 4], 8).observe().write("A", 0, 6).observe()
        stamp_row(ref, A, [3, 4], 0, 6)
        w.dispatch("attn", [row("A", reads=range(0, 8))], DenseBlockTable([[3, 4]], [2]))
        return names(ref.check_dense([[3, 4]], [A], [8]))

    host, tags = both(world, ledger, ref, scenario)
    assert "read_uninitialized" in host and tags == ["read_uninitialized"]


def test_swapped_table_entry_is_a_mapping_mismatch_to_the_host_and_a_tag_to_the_device(
    world, ledger, ref
):
    def scenario(w, ref):
        w.alloc(3, 4, 5).request("A", [3, 4], 6).request("B", [5], 3).observe()
        w.write("A", 0, 6).write("B", 0, 3).observe()
        stamp_row(ref, A, [3, 4], 0, 6)
        stamp_row(ref, B, [5], 0, 3)
        w.dispatch("attn", [row("A", reads=range(0, 6))], DenseBlockTable([[3, 5]], [2]))
        return names(ref.check_dense([[3, 5]], [A], [6]))

    host, tags = both(world, ledger, ref, scenario)
    # The host compares the table with the manager's mapping; the tag sees B's private block.
    assert "mapping_mismatch" in host and tags == ["access_foreign_private"]


def test_leaked_block_is_the_shadow_tier_only(world, ledger, ref):
    def scenario(w, ref):
        w.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
        stamp_row(ref, A, [3, 4], 0, 6)
        # A finishes; the framework frees block 3 and forgets block 4 (allocated, mapped by no
        # request): the allocator counts 14 free blocks where the mapping accounts for 15.
        w.drop_request("A").release(3)
        ref.free([3])
        w.observe(free_count=14)
        # No access to block 4 follows in the engine; the tag tier sees only what is read.
        return names(ref.check_dense([[3]], [A], [BS])) + names(
            ref.check_dense([[3, 4]], [A], [6], [BS])
        )

    host, tags = both(world, ledger, ref, scenario)
    assert "private_unmapped" in host
    assert tags == ["access_free_block"]  # block 3's free; block 4's leak is invisible


def test_ledger_records_snapshot_publications_for_hookless_frameworks(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    assert ledger.drain_published(POOL) == []
    world.published(3).observe()  # the snapshot shows block 3 cached
    assert ledger.drain_published(POOL) == [3]
    assert ledger.drain_published(POOL) == []
    world.observe()
    assert ledger.drain_published(POOL) == []  # published already: no new transition
