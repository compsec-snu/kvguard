"""Regression tests from a review of the core: slot-map epochs hold their blocks, an unchanged
window hole is not a mutation, writes_first is symmetric, out-of-pool entries are reported wherever
they sit, token tables are checked like dense tables, CSR page lengths are bound, descriptors can
be bound to a snapshot, a request may own several rows, and external fills honour ownership and
pending publication."""

from kvguard.core import (
    CSRPageList,
    DenseBlockTable,
    DispatchDescriptor,
    ExternalFill,
    Ledger,
    PoolGeometry,
    Reporter,
    SlotMap,
    TokenTable,
)

from tests.unit.conftest import BS, POOL, World, kinds, row


def token_world(num_blocks=32, block_size=1):
    g = PoolGeometry("tok", num_blocks=num_blocks, block_size=block_size)
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(g)
    return World(lg, pool="tok"), lg


# 1. slot-map epochs
def test_slot_map_dispatch_holds_its_blocks_in_flight(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    sm = SlotMap(slots=[3 * BS + 0, 3 * BS + 1], rows=[0, 0], positions=[0, 1])
    world.dispatch("w1", [row("A", writes=range(0, 2))], sm, family="reshape_and_cache")
    assert ledger.epochs["w1"].blocks == frozenset({(POOL, 3)})
    assert ledger.inflight(POOL, 3) == 1
    ledger.free(POOL, [3])
    assert "free_while_in_flight" in kinds(ledger)


def test_slot_map_padding_token_naming_a_managed_slot_is_reported(world, ledger):
    world.alloc(3).request("A", [3], 1).observe()
    sm = SlotMap(slots=[3 * BS, 3 * BS + 1, -1], rows=[0, 1, 1], positions=[0, 0, 1])
    world.dispatch(
        "w1",
        [row("A", writes=range(0, 1)), row(None, padding=True)],
        sm,
        family="reshape_and_cache",
    )
    assert kinds(ledger) == ["padding_row_names_managed_block"]


# 2. hole vs null block
def test_unchanged_window_hole_is_not_a_mutation_while_in_flight(world, ledger):
    world.alloc(4).request("A", [None, 4], 8, first_token=4).observe()
    world.dispatch(
        "d1", [row("A", reads=range(4, 8), writes=range(4, 8))], DenseBlockTable([[0, 4]], [2])
    )
    assert kinds(ledger) == []
    world.request("A", [None, 4], 8, num_computed=8, first_token=4).observe()
    assert kinds(ledger) == []
    world.complete("d1")


# 3. writes_first
def _shared_block_rows(world):
    world.alloc(3).request("A", [3], BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()


def test_without_writes_first_a_row_reads_only_its_own_writes(world, ledger):
    _shared_block_rows(world)
    rows = (row("A", writes=range(0, BS)), row("B", reads=range(0, BS)))
    ledger.dispatch(
        DispatchDescriptor("f", POOL, "fused", rows, DenseBlockTable([[3], [3]], [1, 1]))
    )
    assert kinds(ledger) == ["read_uninitialized"] * BS


def test_writes_first_satisfies_reads_whatever_the_row_order(world, ledger):
    _shared_block_rows(world)
    rows = (row("B", reads=range(0, BS)), row("A", writes=range(0, BS)))
    ledger.dispatch(
        DispatchDescriptor(
            "f", POOL, "fused", rows, DenseBlockTable([[3], [3]], [1, 1]), writes_first=True
        )
    )
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).filled == (1 << BS) - 1


def test_two_rows_writing_one_slot_is_a_double_write(world, ledger):
    world.alloc(3).request("A", [3], 1).observe()
    rows = (row("A", writes=range(0, 1)), row("A", writes=range(0, 1)))  # two beam rows
    ledger.dispatch(
        DispatchDescriptor("f", POOL, "fused", rows, DenseBlockTable([[3], [3]], [1, 1]))
    )
    assert "double_write" in kinds(ledger)


# 4. out-of-pool entries
def test_out_of_pool_entry_at_a_dead_position_is_reported(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], DenseBlockTable([[3, 999]], [2]))
    assert kinds(ledger) == ["mapping_out_of_pool"]
    world.dispatch("d2", [row("A", reads=range(0, 2))], DenseBlockTable([[3, -7]], [2]))
    assert kinds(ledger) == ["mapping_out_of_pool"] * 2  # the reporter keeps every report


def test_negative_slot_other_than_the_padding_sentinel_is_reported(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    sm = SlotMap(slots=[-5, 3 * BS + 1], rows=[0, 0], positions=[0, 1])
    world.dispatch("w1", [row("A", writes=range(0, 2))], sm, family="reshape_and_cache")
    assert kinds(ledger) == ["mapping_out_of_pool"]


# 5. token tables
def test_token_table_dead_position_naming_a_managed_slot_is_stale():
    w, lg = token_world()
    w.alloc(6, 7, 9).request("A", [None, 6, 7], 3, first_token=1).request("B", [9], 1).observe()
    lg.dispatch(
        DispatchDescriptor(
            "d1", "tok", "attn", (row("A", writes=range(1, 3)),), TokenTable([[9, 6, 7]], [3])
        )
    )
    assert kinds(lg) == ["stale_table_entry"]


def test_token_table_checks_the_valid_length():
    w, lg = token_world()
    w.alloc(5, 6, 7).request("A", [5, 6, 7], 3).observe()
    lg.dispatch(
        DispatchDescriptor(
            "w", "tok", "w", (row("A", writes=range(0, 3)),), TokenTable([[5, 6, 7]], [3])
        )
    )
    lg.complete("w")
    w.request("A", [5, 6, 7], 3, num_computed=3).observe()
    lg.dispatch(
        DispatchDescriptor(
            "r", "tok", "attn", (row("A", reads=range(0, 3)),), TokenTable([[5, 6, 7]], [2])
        )
    )
    assert kinds(lg) == ["position_beyond_valid_length"]


def test_token_table_with_a_larger_block_size_is_supported():
    w, lg = token_world(num_blocks=8, block_size=4)
    w.alloc(2).request("A", [2], 3).observe()
    slots = [8, 9, 10]  # block 2 holds slots 8..11
    lg.dispatch(
        DispatchDescriptor(
            "w", "tok", "w", (row("A", writes=range(0, 3)),), TokenTable([slots], [3])
        )
    )
    assert kinds(lg) == []
    assert lg.epochs["w"].blocks == frozenset({("tok", 2)})


# 6. CSR page length
def test_csr_last_page_len_must_match_the_row(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe().write("A", 0, 6).observe()
    ok = CSRPageList(indptr=[0, 2], indices=[3, 4], last_page_len=[2])
    world.dispatch("d0", [row("A", reads=range(0, 6))], ok)
    assert kinds(ledger) == []
    world.dispatch("d1", [row("A", reads=range(0, 6))], CSRPageList([0, 2], [3, 4], [4]))
    assert kinds(ledger) == ["page_length_mismatch"]
    world.dispatch("d2", [row("A", reads=range(0, 6))], CSRPageList([0, 2], [3, 4], [0]))
    assert kinds(ledger) == ["page_length_mismatch"]


# 9, 10. descriptor binding, rows
def test_descriptor_bound_to_another_snapshot_is_an_integration_error(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    ledger.dispatch(
        DispatchDescriptor(
            "d",
            POOL,
            "attn",
            (row("A", writes=range(0, 2)),),
            DenseBlockTable([[3]], [1]),
            snapshot_seq=ledger.seq + 5,
        )
    )
    assert kinds(ledger) == ["dispatch_snapshot_mismatch"]


def test_a_request_with_two_rows_keeps_both_in_flight(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 2 * BS).observe()
    rows = (row("A", writes=range(0, BS)), row("A", writes=range(BS, 2 * BS)))
    ledger.dispatch(
        DispatchDescriptor("d", POOL, "beams", rows, DenseBlockTable([[3, 4], [3, 4]], [2, 2]))
    )
    assert kinds(ledger) == []
    assert {b for _, b in ledger.epochs["d"].blocks} == {3, 4}
    assert [(rid, tuple(ix), e) for rid, ix, e in ledger.epochs["d"].consumed] == [
        ("A", (0, 1), (3, 4)),
        ("A", (0, 1), (3, 4)),
    ]


# 8. external fills
def test_external_fill_into_another_requests_private_block(world, ledger):
    world.alloc(3).request("A", [3], 1).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 1), source="onload", request="B"))
    assert kinds(ledger) == ["access_foreign_private"]


def test_external_fill_may_complete_a_pending_publication(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.request("A", [3], 2).published(3).request("B", [3], 2, num_computed=2).observe()
    assert ledger.block(POOL, 3).pending_publish == 0b11
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 2), source="transfer"))
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).filled == 0b11 and ledger.block(POOL, 3).pending_publish == 0
    ledger.external_fill(ExternalFill(POOL, 3, range(2, 3), source="transfer"))
    assert kinds(ledger) == ["write_to_published"]
