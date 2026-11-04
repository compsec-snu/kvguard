"""P3 mapping validation at dispatch (the host reference of the dispatch-time validator), P2/P4
access checks, epochs and the mapping representations."""

from kvguard.core import (
    CSRPageList,
    DenseBlockTable,
    DispatchDescriptor,
    ExternalFill,
    KVGuardViolation,
    Ledger,
    PoolGeometry,
    Reporter,
    SlotMap,
    TokenTable,
)
from kvguard.core.violations import OutcomeKind, Policy

from tests.unit.conftest import BS, POOL, World, kinds, row


def test_mapping_mismatch_in_dense_table(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe()
    world.dispatch("d1", [row("A", writes=range(0, 6))], DenseBlockTable([[3, 5]], [2]))
    assert kinds(ledger) == ["mapping_mismatch"]
    assert ledger.reporter.reports[0].policy is Policy.P3


def test_stale_table_entry_beyond_request_mapping(world, ledger):
    """A row whose table names another request's block for positions the row does not hold."""
    world.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], DenseBlockTable([[3, 4]], [2]))
    assert kinds(ledger) == ["stale_table_entry"]


def test_position_beyond_valid_length(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe()
    world.dispatch("d1", [row("A", writes=range(0, 6))], DenseBlockTable([[3, 4]], [1]))
    assert "position_beyond_valid_length" in kinds(ledger)


def test_padding_row_touching_a_live_block(world, ledger):
    """A CUDA-graph padding row retains stale entries of a live block."""
    world.alloc(3).request("A", [3], 2).observe()
    world.dispatch(
        "d1",
        [row("A", writes=range(0, 2)), row(None, padding=True)],
        DenseBlockTable([[3], [3]], [1, 1]),
    )
    assert kinds(ledger) == ["padding_row_names_managed_block"]


def test_padding_row_with_null_block_is_fine(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.dispatch(
        "d1",
        [row("A", writes=range(0, 2)), row(None, padding=True)],
        DenseBlockTable([[3], [0]], [1, 1]),
    )
    assert kinds(ledger) == []


def test_narrow_padding_descriptor_requires_reserved_destinations():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.validate_padding_destinations([21, 22, 13, 14], 2, {0}, site="graph-replay")
    assert kinds(ledger) == [
        "padding_row_names_managed_block",
        "padding_row_names_managed_block",
    ]
    assert [report.slot for report in ledger.reporter.reports] == [13, 14]


def test_narrow_padding_descriptor_accepts_sentinels_and_rejects_bad_extent():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.validate_padding_destinations([21, 22, 0, 0], 2, {0})
    ledger.validate_padding_destinations([21], 2, {0})
    assert kinds(ledger) == ["malformed_padding_mapping"]


def test_row_for_a_finished_request(world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.drop_request("A").release(3).observe()
    world.dispatch("d1", [row("A", reads=range(0, 2))], DenseBlockTable([[3]], [1]))
    assert kinds(ledger) == ["row_for_unknown_request"]


def test_read_before_write_is_uninitialized(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.dispatch("d1", [row("A", reads=range(0, 2))])
    assert kinds(ledger) == ["read_uninitialized", "read_uninitialized"]
    assert ledger.reporter.reports[0].policy is Policy.P4


def test_same_dispatch_write_then_read_is_initialized(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.dispatch("d1", [row("A", reads=range(0, 2), writes=range(0, 2))])
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).filled == 0b11


def test_double_write_without_invalidation(world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.dispatch("d2", [row("A", writes=range(1, 2))])
    assert kinds(ledger) == ["double_write"]


def test_speculative_rollback_invalidates_then_rewrites(world, ledger):
    """Rejected draft tokens: num_computed shrinks, the slots are invalidated, rewriting is legal."""
    world.alloc(3).request("A", [3], 4).observe().write("A", 0, 4).observe()
    assert ledger.block(POOL, 3).filled == 0b1111
    world.request("A", [3], 4, num_computed=2).observe()  # positions 2,3 rejected
    assert ledger.block(POOL, 3).filled == 0b0011
    world.dispatch("d2", [row("A", reads=range(0, 4), writes=range(2, 4))]).complete("d2")
    assert kinds(ledger) == []


def test_speculative_rollback_uses_the_dispatch_written_frontier(world, ledger):
    """Adapters snapshot the pre-dispatch frontier, then the dispatch writes draft positions.
    A later rejection retreats from the written frontier even though the preceding snapshot's
    ``num_computed`` never included those writes."""
    world.alloc(3).request("A", [3], 4, num_computed=0).observe()
    world.dispatch("draft", [row("A", writes=range(0, 4))]).complete("draft")
    assert ledger.block(POOL, 3).filled == 0b1111
    world.request("A", [3], 4, num_computed=2).observe()
    assert ledger.block(POOL, 3).filled == 0b0011
    world.dispatch("retry", [row("A", writes=range(2, 4))]).complete("retry")
    assert kinds(ledger) == []


def test_execution_time_speculative_rollback_invalidates_before_rewrite(world, ledger):
    """An async engine may expose its corrected frontier only in the execution descriptor."""
    world.alloc(3, 4).request("A", [3, 4], 6, num_computed=0).observe()
    world.dispatch("draft", [row("A", writes=range(0, 6))]).complete("draft")
    ledger.rollback(POOL, "A", 2, site="execution")
    assert ledger.block(POOL, 3).filled == 0b0011
    world.dispatch("retry", [row("A", writes=range(2, 6))]).complete("retry")
    assert kinds(ledger) == []
    assert ledger.counters["speculative_rollbacks"] == 1


def test_execution_time_rollback_does_not_silence_published_invalidation(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).published(3).observe()
    ledger.rollback(POOL, "A", 2, site="execution")
    assert kinds(ledger) == ["invalidate_published", "invalidate_published"]


def test_invalidating_a_published_block_is_reported(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).published(3).observe()
    world.request("A", [3], BS, num_computed=2).observe()
    assert kinds(ledger) == ["invalidate_published", "invalidate_published"]


def test_private_rollback_precedes_same_snapshot_publication(world, ledger):
    """A private speculative tail can be rejected just before a second request shares the
    stable prefix. The recomputing dispatch completes the new publication's pending tail."""
    world.alloc(3).request("A", [3], BS).observe()
    world.dispatch("draft", [row("A", writes=range(0, BS))]).complete("draft")
    assert ledger.block(POOL, 3).filled == 0b1111

    world.request("A", [3], BS, num_computed=2).request("B", [3], BS, num_computed=2).published(
        3
    ).observe()
    shadow = ledger.block(POOL, 3)
    assert kinds(ledger) == []
    assert shadow.filled == 0b0011
    assert shadow.pending_publish == 0b1100

    world.dispatch("retry", [row("A", writes=range(2, BS))]).complete("retry")
    world.request("A", [3], BS, num_computed=BS).request("B", [3], BS, num_computed=BS).observe()
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).filled == 0b1111


def test_computed_claim_beyond_written_slots(world, ledger):
    """The framework counts positions as computed that nothing ever wrote."""
    world.alloc(3).request("A", [3], 3, num_computed=3).observe()
    assert kinds(ledger) == ["computed_beyond_filled"]


def test_same_step_publication_is_settled_by_the_writing_dispatch(world, ledger):
    """vLLM registers a full block's hash in the step that computes it, so a same-step prefix hit
    maps the block before it is written. Legal when the step's write dispatch precedes the reads."""
    world.alloc(3).request("A", [3], BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()
    assert kinds(ledger) == []
    world.dispatch("w1", [row("A", writes=range(0, BS))], family="reshape_and_cache").complete("w1")
    world.dispatch("a1", [row("A", reads=range(0, BS)), row("B", reads=range(0, BS))]).complete(
        "a1"
    )
    world.request("A", [3], BS, num_computed=BS).observe()
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).pending_publish == 0


def test_publication_never_written_is_reported_at_the_next_snapshot(world, ledger):
    world.alloc(3).request("A", [3], BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()
    world.observe()  # no dispatch wrote the block
    # The publication is reported, and B's claim of computed positions is now false too.
    assert kinds(ledger) == ["publish_uninitialized", "computed_beyond_filled"]


def test_same_step_reader_before_the_writer_is_uninitialized(world, ledger):
    """The reader's dispatch runs before the dispatch that writes the shared block (two
    micro-batches), so it reads slots that are not yet initialized."""
    world.alloc(3).request("A", [3], BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()
    world.dispatch("a0", [row("B", reads=range(0, BS))])
    assert kinds(ledger) == ["read_uninitialized"] * BS


def test_writes_first_dispatch_shares_writes_across_rows(world, ledger):
    world.alloc(3).request("A", [3], BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()
    rows = (row("A", writes=range(0, BS)), row("B", reads=range(0, BS)))
    table = DenseBlockTable([[3], [3]], [1, 1])
    ledger.dispatch(DispatchDescriptor("fused", POOL, "fused", rows, table, writes_first=True))
    assert kinds(ledger) == []
    second = DispatchDescriptor(
        "fused2", POOL, "fused", (row("B", writes=range(0, 1)),), DenseBlockTable([[3]], [1])
    )
    ledger.dispatch(second)
    assert kinds(ledger) == ["write_to_published"]


def test_write_to_published_block(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    world.published(3).request("B", [3], BS, num_computed=BS).observe()
    world.dispatch("d2", [row("A", writes=range(1, 2))])
    assert kinds(ledger) == ["write_to_published"]


def test_sole_reader_may_rewrite_one_authorized_volatile_published_slot():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry("draft", 16, BS, rewritable=True))
    world = World(ledger, pool="draft")
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).published(3).observe()

    assert ledger.authorize_volatile_rewrite("draft", "A", BS - 1, site="mtp-tail")
    world.dispatch("rewrite", [row("A", writes=range(BS - 1, BS))]).complete("rewrite")
    assert kinds(ledger) == []
    assert ledger.block("draft", 3).pending_publish == 0

    # The permission is one-shot; a second write remains a publication violation.
    world.dispatch("rewrite-again", [row("A", writes=range(BS - 1, BS))])
    assert kinds(ledger) == ["write_to_published"]


def test_volatile_rewrite_permission_rejects_a_shared_published_block():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry("draft", 16, BS, rewritable=True))
    world = World(ledger, pool="draft")
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).published(3).request(
        "B", [3], BS, num_computed=BS
    ).observe()

    assert not ledger.authorize_volatile_rewrite("draft", "A", BS - 1, site="mtp-tail")
    world.dispatch("rewrite", [row("A", writes=range(BS - 1, BS))])
    assert kinds(ledger) == ["write_to_published"]


def test_scratch_rewrite_permission_reopens_the_writers_own_slots_in_a_shared_published_block():
    """A standalone drafter's provisional writes (its correction slot and draft positions) land
    in a private rewritable block; the scheduler publishes the block and a second request maps
    it before the drafter's next forward rewrites them. The adapter re-opens exactly those
    slots, for that request; other slots and other requests stay protected."""
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry("draft", 16, BS, rewritable=True))
    world = World(ledger, pool="draft")
    # A computed 2 positions, wrote 4 (two scratch), then the block was published and shared.
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    world.published(3).request("B", [3], BS, num_computed=BS).observe()

    assert ledger.authorize_scratch_rewrite("draft", "A", range(2, BS), site="scratch") == 2
    assert ledger.block("draft", 3).pending_publish == 0b1100
    world.dispatch("rewrite", [row("A", writes=range(2, BS))]).complete("rewrite")
    assert kinds(ledger) == []
    assert ledger.block("draft", 3).pending_publish == 0
    assert ledger.counters["scratch_rewrites_authorized"] == 2

    # Committed content and other requests' writes remain publication violations, and the
    # permission names only slots the request maps and has written.
    world.dispatch("committed", [row("A", writes=range(0, 1))])
    assert kinds(ledger) == ["write_to_published"]
    assert ledger.authorize_scratch_rewrite("draft", "A", range(BS, 2 * BS), site="s") == 0
    assert ledger.authorize_scratch_rewrite("draft", "C", range(2, BS), site="s") == 0
    assert ledger.authorize_scratch_rewrite(POOL, "A", range(2, BS), site="s") == 0


def test_shared_published_block_may_be_read_by_both(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    world.published(3).request("B", [3], BS, num_computed=BS).observe()
    world.dispatch("d2", [row("A", reads=range(0, BS)), row("B", reads=range(0, BS))])
    assert kinds(ledger) == []


def test_row_naming_another_requests_private_block(world, ledger):
    world.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    world.dispatch("d1", [row("B", reads=range(0, 2))], DenseBlockTable([[3]], [1]))
    assert kinds(ledger)[0] == "mapping_mismatch"


def test_block_freed_between_snapshot_and_dispatch(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    ledger.free(POOL, [3])
    world.dispatch("d1", [row("A", writes=range(0, 2))])
    assert kinds(ledger) == ["access_free_block", "access_free_block"]


def test_mapping_mutated_while_dispatch_open(world, ledger):
    world.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))])
    world.request("A", [4], 2).request("B", [3], 2).observe()  # tables swapped under the kernel
    assert "mapping_mutated_in_flight" in kinds(ledger)
    world.complete("d1")


def test_duplicate_slot_within_a_request(world, ledger):
    world.alloc(3).request("A", [3, 3], 6).observe()
    assert "duplicate_slot_in_request" in kinds(ledger)


def test_slot_map_representation(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 4)))
    world.request("A", [3, 4], 6, num_computed=4).observe()
    good = SlotMap(slots=[4 * BS + 0, 4 * BS + 1], rows=[0, 0], positions=[4, 5])
    world.dispatch("w1", [row("A", writes=range(4, 6))], good, family="reshape_and_cache")
    assert kinds(ledger) == []
    world.request("A", [3, 4], 8, num_computed=6).observe()
    bad = SlotMap(slots=[3 * BS + 0, -1], rows=[0, 0], positions=[6, 7])
    world.dispatch("w2", [row("A", writes=range(6, 8))], bad, family="reshape_and_cache")
    assert kinds(ledger) == ["slot_map_mismatch"]


def test_token_table_representation():
    g = PoolGeometry("tok", num_blocks=32, block_size=1)
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(g)
    w = World(lg, pool="tok")
    w.alloc(5, 6, 7).request("A", [5, 6, 7], 3).observe()
    w.dispatch("d1", [row("A", writes=range(0, 3))], TokenTable([[5, 6, 7]], [3]))
    assert kinds(lg) == []
    w.dispatch("d2", [row("A", reads=range(0, 3))], TokenTable([[5, 9, 7]], [3]))
    assert kinds(lg) == ["mapping_mismatch"]


def test_token_table_at_a_page_size_takes_the_expanded_fast_path(world, ledger):
    # SGLang's FlashInfer page list: one entry per token over a paged mapping (BS slots per
    # block). A row equal to the mapping's expansion is one tuple comparison; a wrong entry
    # still falls through to the loop that names it.
    world.alloc(3, 4).request("A", [3, 4], 2 * BS).observe().write("A", 0, 2 * BS)
    slots = [3 * BS + o for o in range(BS)] + [4 * BS + o for o in range(BS)]
    world.dispatch("r1", [row("A", reads=range(0, 2 * BS))], TokenTable([slots], [2 * BS]))
    assert kinds(ledger) == []
    assert ledger.counters["token_table_fast_path"] == 1
    assert ledger.requests[("A", POOL)].slot_cache[1] == tuple(slots)
    bad = list(slots)
    bad[BS + 1] = 9 * BS
    world.dispatch("r2", [row("A", reads=range(0, 2 * BS))], TokenTable([bad], [2 * BS]))
    assert kinds(ledger) == ["mapping_mismatch"]
    assert ledger.counters["token_table_fast_path"] == 1
    # The table may cover fewer positions than the mapping, from the row's first position.
    world.dispatch("r3", [row("A", reads=range(0, BS))], TokenTable([slots[:BS]], [BS]))
    assert kinds(ledger) == ["mapping_mismatch"]
    assert ledger.counters["token_table_fast_path"] == 2


def test_token_table_suffix_starts_at_its_declared_logical_position():
    g = PoolGeometry("tok", num_blocks=32, block_size=1)
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(g)
    w = World(lg, pool="tok")
    w.alloc(5, 6, 7).request("A", [5, 6, 7], 3).observe()
    w.dispatch(
        "fill",
        [row("A", writes=range(0, 3))],
        TokenTable([[5, 6, 7]], [3]),
    ).complete("fill")
    w.dispatch(
        "suffix",
        [row("A", reads=range(1, 3))],
        TokenTable([[6, 7]], [2], first=[1]),
    )
    assert kinds(lg) == []
    w.dispatch(
        "bad-suffix",
        [row("A", reads=range(1, 3))],
        TokenTable([[5, 7]], [2], first=[1]),
    )
    assert kinds(lg) == ["mapping_mismatch"]


def test_csr_page_list_representation(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 6).observe()
    good = CSRPageList(indptr=[0, 2], indices=[3, 4], last_page_len=[2])
    world.dispatch("d1", [row("A", writes=range(0, 6))], good)
    assert kinds(ledger) == []
    bad = CSRPageList(indptr=[0, 2], indices=[4, 3], last_page_len=[2])
    world.dispatch("d2", [row("A", reads=range(0, 6))], bad)
    assert kinds(ledger)[0] == "mapping_mismatch"


def test_sliding_window_hole_is_not_live(world, ledger):
    """Positions below first_token map to no block; the table carries the null block there."""
    world.alloc(4).request("A", [None, 4], 8, first_token=4).observe()
    world.dispatch(
        "d1", [row("A", reads=range(4, 8), writes=range(4, 8))], DenseBlockTable([[0, 4]], [2])
    )
    assert kinds(ledger) == []


def test_dispatch_without_snapshot_and_unknown_completion(ledger):
    ledger.dispatch(DispatchDescriptor("d0", POOL, "attn", (row("A"),), DenseBlockTable([[]], [0])))
    ledger.complete("nope")
    assert kinds(ledger) == ["dispatch_before_snapshot", "unknown_dispatch"]
    assert all(r.outcome is OutcomeKind.INTEGRATION_ERROR for r in ledger.reporter.reports)


def test_fail_fast_reporter_raises(geometry):
    lg = Ledger(Reporter(fail_fast=True))
    lg.register_pool(geometry)
    lg.allocate(POOL, [3])
    try:
        lg.allocate(POOL, [3])
    except KVGuardViolation as exc:
        assert exc.report.kind == "allocate_non_free"
    else:
        raise AssertionError("expected a violation")


def test_drain_reporter_deduplicates(geometry):
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(geometry)
    lg.free(POOL, [3])
    lg.free(POOL, [3])
    assert kinds(lg) == ["double_free"]
    assert lg.reporter.duplicates == 1


def test_dispatch_through_a_mapping_that_names_an_out_of_pool_block(world, ledger):
    """The mapping itself is wrong (reported at the snapshot) and the table matches it: the
    access check reports the out-of-pool block instead of failing on the shadow lookup."""
    world.alloc(3).request("A", [3, 99], 2 * BS).observe()
    assert "mapping_out_of_pool" in kinds(ledger)
    world.dispatch("d", [row("A", writes=range(0, 2 * BS))], DenseBlockTable([[3, 99]], [2]))
    assert kinds(ledger).count("mapping_out_of_pool") >= 2
