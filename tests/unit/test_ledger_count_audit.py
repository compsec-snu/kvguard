"""The partition audit by count: the allocator's free count against the listed blocks that hold a
reference, for the steps whose free set is not read (the vLLM adapter's ordinary steps)."""

from __future__ import annotations

from tests.unit.conftest import kinds

POOL_BLOCKS = 16  # the test geometry; block 0 is reserved


def test_a_consistent_count_is_silent(world, ledger):
    world.alloc(3, 4, 5).request("A", [3, 4, 5], 2 * 4).observe(free_count=POOL_BLOCKS - 1 - 3)
    assert kinds(ledger) == []


def test_a_leaked_block_shows_in_the_count(world, ledger):
    world.alloc(3, 4).request("A", [3, 4], 8).observe(free_count=POOL_BLOCKS - 1 - 3)
    assert kinds(ledger) == ["block_count_mismatch"]
    assert (
        "3 block(s)" in ledger.reporter.reports[0].detail
        and "2 listed" in ledger.reporter.reports[0].detail
    )


def test_a_reserved_published_block_counts_as_free(world, ledger):
    world.alloc(3).request("A", [3], 4).observe().write("A", 0, 4).observe()
    world.drop_request("A").published(3, reserved=True).observe(free_count=POOL_BLOCKS - 1)
    assert kinds(ledger) == []
    world.observe(free_count=POOL_BLOCKS - 2)
    assert kinds(ledger) == ["block_count_mismatch"]


def test_a_reserved_block_outside_the_free_set_counts_as_held():
    # SGLang's unlocked radix node: listed reserved, its page not in the allocator's free set.
    from kvguard.core import Ledger, PoolGeometry, Reporter

    from tests.unit.conftest import BS, World

    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(
        PoolGeometry("kv", POOL_BLOCKS, BS, reserved_blocks=frozenset({0}), reserved_in_free=False)
    )
    world = World(ledger, pool="kv")
    world.alloc(3).request("A", [3], 4).observe().write("A", 0, 4).observe()
    world.drop_request("A").published(3, reserved=True).observe(free_count=POOL_BLOCKS - 2)
    assert kinds(ledger) == []
    world.observe(free_count=POOL_BLOCKS - 1)
    assert kinds(ledger) == ["block_count_mismatch"]


def test_the_free_set_takes_precedence(world, ledger):
    world.alloc(3).request("A", [3], 4)
    world.free = set(range(1, POOL_BLOCKS)) - {3}
    world.observe(free_count=0)  # a wrong count is ignored while the set is given
    assert kinds(ledger) == []
