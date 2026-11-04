"""P1 lifecycle and P2 permission configurations: legal sequences report nothing; each illegal
transition is reported once at its first observation."""

from kvguard.core import BlockState, ExternalFill
from kvguard.core.violations import OutcomeKind, Policy

from tests.unit.conftest import BS, POOL, kinds, row


def test_legal_prefill_decode_finish_cycle(world, ledger):
    # Step 1: request A allocates two blocks and prefills 6 tokens.
    world.alloc(1, 2).request("A", [1, 2], 6, num_computed=0).observe()
    world.dispatch("d1", [row("A", reads=range(0, 6), writes=range(0, 6))]).complete("d1")
    # Step 2: decode writes position 6, reads 0..6.
    world.request("A", [1, 2], 7, num_computed=6).observe()
    world.dispatch("d2", [row("A", reads=range(0, 7), writes=range(6, 7))]).complete("d2")
    # Block 1 became full (positions 0..3) and is published; A still reads it.
    world.request("A", [1, 2], 8, num_computed=7).published(1).observe()
    world.dispatch("d3", [row("A", reads=range(0, 8), writes=range(7, 8))]).complete("d3")
    # Finish: A releases; block 1 stays reserved in the prefix cache, block 2 is freed.
    world.drop_request("A").published(1, reserved=True).release(2).observe()
    # A new request B hits the cached block 1 and gets block 2 again (new generation).
    world.alloc(2).request("B", [1, 2], 5, num_computed=4).observe()
    world.dispatch("d4", [row("B", reads=range(0, 5), writes=range(4, 5))]).complete("d4")
    assert kinds(ledger) == []
    assert ledger.block(POOL, 2).generation == 2
    assert ledger.block(POOL, 1).state is BlockState.PUBLISHED
    assert ledger.block(POOL, 1).readers == {"B"}


def test_double_free_is_reported_at_the_event(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.drop_request("A").release(3)
    ledger.free(POOL, [3])
    assert kinds(ledger) == ["double_free"]
    assert ledger.reporter.reports[0].policy is Policy.P1


def test_allocating_a_private_block_is_reported(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    ledger.allocate(POOL, [3])
    assert kinds(ledger) == ["allocate_non_free"]


def test_evicting_a_reserved_block_by_allocation_is_legal(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.drop_request("A").published(3, reserved=True).observe()
    assert ledger.block(POOL, 3).reserved
    world.alloc(3).request("B", [3], 1, num_computed=0).observe()
    assert kinds(ledger) == []
    assert ledger.block(POOL, 3).owner == "B"
    assert ledger.block(POOL, 3).generation == 2


def test_allocating_a_shared_published_block_is_reported(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.published(3).observe()
    ledger.allocate(POOL, [3])
    assert kinds(ledger) == ["allocate_non_free"]


def test_free_while_dispatch_in_flight(world, ledger):
    """A block returned to the pool while a dispatch still uses it."""
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))])
    ledger.free(POOL, [3])
    assert kinds(ledger) == ["free_while_in_flight"]
    world.complete("d1")
    assert ledger.inflight(POOL, 3) == 0


def test_unmediated_allocation_is_an_integration_error(world, ledger):
    world.private(5).request("A", [5], 1).observe()
    rep = ledger.reporter.reports
    assert [r.kind for r in rep] == ["unmediated_allocation"]
    assert rep[0].outcome is OutcomeKind.INTEGRATION_ERROR


def test_unmediated_ledger_accepts_snapshot_only_allocation(geometry):
    from kvguard.core import Ledger, Reporter

    from tests.unit.conftest import World

    lg = Ledger(Reporter(fail_fast=False), mediated=False)
    lg.register_pool(geometry)
    w = World(lg)
    w.private(5).request("A", [5], 1, num_computed=0).observe()
    w.drop_request("A").freed(5).observe()
    assert kinds(lg) == []
    assert lg.block(POOL, 5).generation == 1


def test_unmediated_free_is_an_integration_error(world, ledger):
    world.alloc(3).request("A", [3], 1).observe()
    world.drop_request("A").freed(3).observe()
    assert kinds(ledger) == ["unmediated_free"]


def test_mapping_to_a_free_block_is_use_after_free(world, ledger):
    world.alloc(3).request("A", [3], 1).observe()
    ledger.free(POOL, [3])  # framework frees it but A keeps mapping it
    world.freed(3).observe()
    assert kinds(ledger) == ["mapping_to_free_block"]
    assert ledger.reporter.reports[0].policy is Policy.P2


def test_private_block_mapped_by_two_requests(world, ledger):
    world.alloc(3).request("A", [3], 2).request("B", [3], 2).observe()
    assert "private_multiple_owners" in kinds(ledger)


def test_private_block_without_any_mapping(world, ledger):
    world.alloc(3).observe()
    assert kinds(ledger) == ["private_unmapped"]


def test_published_block_neither_read_nor_reserved(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.drop_request("A").published(3, reserved=False).observe()
    assert kinds(ledger) == ["published_unreferenced"]


def test_ownership_transfer_without_free(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.drop_request("A").request("B", [3], 2).observe()
    assert kinds(ledger) == ["ownership_transfer_without_free"]


def test_published_to_private_without_free_is_aba(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.drop_request("A").published(3, reserved=True).observe()
    # The framework hands the reserved block to B without going through the allocator.
    world.private(3).request("B", [3], 1, num_computed=0).observe()
    assert kinds(ledger) == ["published_to_private"]


def test_partition_audit_free_and_allocated(world, ledger):
    """A block simultaneously in the allocator's free set and held by a request."""
    world.alloc(3).request("A", [3], 2).observe()
    world.free = set(range(1, 16)) - {3}
    world.observe()
    assert kinds(ledger) == []
    world.free.add(3)
    world.observe()
    assert kinds(ledger) == ["block_free_and_allocated"]


def test_partition_audit_unaccounted_block_is_integration_error(world, ledger):
    world.alloc(3).request("A", [3], 2).observe()
    world.free = set(range(1, 16)) - {3, 7}
    world.observe()
    assert kinds(ledger) == ["block_unaccounted"]


def test_state_for_reserved_block_is_rejected(world, ledger):
    world.private(0).observe()  # block 0 is the null block
    assert kinds(ledger) == ["state_for_unmanaged_block"]


def test_external_fill_respects_permissions(world, ledger):
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 2), source="onload"))
    assert kinds(ledger) == []
    ledger.external_fill(ExternalFill(POOL, 3, range(1, 2), source="onload"))
    assert kinds(ledger) == ["double_write"]
    world.published(3).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(2, 3), source="onload"))
    assert kinds(ledger)[-1] == "write_to_published"


def test_reference_count_disagreeing_with_mappings(world, ledger):
    """The framework's sharing count must equal the number of requests mapping the block: a
    sharer's release that leaves its mapping in place (or a stray decrement) is reported."""
    from kvguard.core import BlockSnapshot, BlockState

    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    world.blocks[3] = BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=2)
    world.request("B", [3], BS, num_computed=BS).observe()
    assert kinds(ledger) == []
    world.blocks[3] = BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=1)
    world.observe()
    assert kinds(ledger) == ["reference_count_mismatch"]


def test_partial_reuse_of_a_reserved_block_keeps_its_content(world, ledger):
    """TensorRT-LLM's take-over of a partially matched leaf: the stored block becomes private to
    the new request with its matched prefix still initialized (one event, no copy)."""
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.drop_request("A").published(3, reserved=True).observe()
    ledger.reuse(POOL, 3, 2)
    assert kinds(ledger) == []
    sh = ledger.block(POOL, 3)
    assert sh.state is BlockState.PRIVATE and sh.generation == 2 and sh.filled == 0b11
    world.private(3).request("B", [3], 3, num_computed=2).observe()
    assert kinds(ledger) == [] and sh.owner == "B"
    world.dispatch("d", [row("B", reads=range(0, 2), writes=range(2, 3))])
    assert kinds(ledger) == []


def test_reuse_of_a_free_or_mapped_block_is_reported(world, ledger):
    ledger.reuse(POOL, 5, 1)
    assert kinds(ledger) == ["reuse_free_block"]
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.published(3).observe()  # shared by a reader, not reserved
    ledger.reuse(POOL, 3, 1)
    assert kinds(ledger)[-1] == "reuse_non_reserved"
    assert ledger.block(POOL, 3).state is BlockState.PUBLISHED


def test_reuse_retaining_unwritten_slots_is_reported(world, ledger):
    world.alloc(3).request("A", [3], 1).observe().write("A", 0, 1).observe()
    world.drop_request("A").published(3, reserved=True).observe()
    ledger.reuse(POOL, 3, 2)
    assert kinds(ledger) == ["reuse_uninitialized_content"]
    assert ledger.block(POOL, 3).filled == 0b01  # only the written slot is retained


def test_reuse_while_a_dispatch_is_in_flight(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.dispatch("d", [row("A", reads=range(0, BS))])
    world.drop_request("A").published(3, reserved=True).observe()
    ledger.reuse(POOL, 3, BS)
    assert "reuse_in_flight" in kinds(ledger)


def test_store_then_take_over_within_one_snapshot_interval(world, ledger):
    """TensorRT-LLM releases a finished request's blocks into its reuse tree and may hand the
    partial leaf to the next request before any snapshot: store, then reuse."""
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    ledger.store(POOL, [3])
    sh = ledger.block(POOL, 3)
    assert sh.state is BlockState.PUBLISHED and sh.reserved and sh.owner is None
    ledger.reuse(POOL, 3, 2)
    assert kinds(ledger) == [] and sh.state is BlockState.PRIVATE and sh.filled == 0b11
    world.drop_request("A").private(3).request("B", [3], 3, num_computed=2).observe()
    assert kinds(ledger) == [] and sh.owner == "B"


def test_store_of_a_free_block_is_reported(world, ledger):
    ledger.store(POOL, [5])
    assert kinds(ledger) == ["store_free_block"]


def test_early_release_into_the_reuse_tree_is_judged_at_the_reuse(world, ledger):
    """The B13 shape under a reuse cache: the request is released while its dispatch reads the
    block; the store reports it, and so do the take-over and the next snapshot."""
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.dispatch("d", [row("A", reads=range(0, BS))])
    ledger.store(POOL, [3])
    assert kinds(ledger) == ["store_while_in_flight"]
    assert ledger.block(POOL, 3).reserved
    ledger.reuse(POOL, 3, BS)
    assert kinds(ledger) == ["store_while_in_flight", "reuse_in_flight"]
    world.drop_request("A").private(3).request("B", [3], BS, num_computed=BS).observe()
    assert "mapping_mutated_in_flight" in kinds(ledger)
