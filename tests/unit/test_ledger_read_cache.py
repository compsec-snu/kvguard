"""A read-only row skips the positions its request's previous read-only row was found allowed
to read, unless the mapping entries changed or one of their blocks was touched since: the same
reports as the full check, at a fraction of the work for a decoding request's context."""

from __future__ import annotations

from tests.unit.conftest import kinds, row


def test_a_decoding_requests_context_is_rechecked_only_where_something_changed(world, ledger):
    world.alloc(1, 2, 3).request("A", [1, 2, 3], 12).observe().write("A", 0, 9).observe()
    world.dispatch("a1", [row("A", reads=range(0, 9))]).complete("a1")
    assert ledger.counters["read_positions_skipped"] == 0
    # The next step reads one more position; its write touched block 3 (positions 8..11), so
    # the two untouched blocks' eight positions are skipped and block 3 is rechecked.
    world.write("A", 9, 10).observe()
    world.dispatch("a2", [row("A", reads=range(0, 10))]).complete("a2")
    assert ledger.counters["read_positions_skipped"] == 8 and kinds(ledger) == []
    world.write("A", 10, 11).observe()
    world.dispatch("a3", [row("A", reads=range(0, 11))]).complete("a3")
    assert ledger.counters["read_positions_skipped"] == 16 and kinds(ledger) == []


def test_the_skip_does_not_hide_a_freed_block(world, ledger):
    world.alloc(1, 2, 3).request("A", [1, 2, 3], 12).observe().write("A", 0, 10).observe()
    world.dispatch("a1", [row("A", reads=range(0, 10))]).complete("a1")
    # Block 2 is freed while the request still maps it: the snapshot reports the mapping, and the
    # next read of positions 4..7 is an access to a free block, cache or not.
    world.release(2).observe()
    world.dispatch("a2", [row("A", reads=range(0, 10))])
    assert "mapping_to_free_block" in kinds(ledger) and "access_free_block" in kinds(ledger)


def test_the_skip_does_not_hide_a_changed_mapping(world, ledger):
    world.alloc(1, 2, 3).request("A", [1, 2, 3], 12).observe().write("A", 0, 10).observe()
    world.dispatch("a1", [row("A", reads=range(0, 10))]).complete("a1")
    # The request's second block is replaced by a fresh, unwritten one.
    world.alloc(5).request("A", [1, 5, 3], 12, num_computed=10).observe()
    world.dispatch("a2", [row("A", reads=range(0, 10))])
    assert "read_uninitialized" in kinds(ledger)
    assert ledger.counters["read_positions_skipped"] == 0


def test_the_skip_does_not_hide_a_block_handed_to_another_request(world, ledger):
    world.alloc(1, 2).request("A", [1, 2], 8).observe().write("A", 0, 8).observe()
    world.dispatch("a1", [row("A", reads=range(0, 8))]).complete("a1")
    # Block 2 leaves A for B without a free (the snapshot reports the transfer); A's read of it
    # is a foreign access, which the touch at the transfer exposes to the next check.
    world.request("A", [1, 2], 8, num_computed=8).request("B", [2], 4, num_computed=4).observe()
    world.dispatch("a2", [row("A", reads=range(0, 8))])
    assert "private_multiple_owners" in kinds(ledger) or "ownership_transfer_without_free" in kinds(
        ledger
    )
    assert ledger.counters["read_positions_skipped"] <= 4
