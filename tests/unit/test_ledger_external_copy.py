"""The external copy: a non-kernel engine reads one block and fills another."""

from __future__ import annotations

from kvguard.core import ExternalCopy, ExternalFill, Ledger, PoolGeometry, Reporter

from tests.unit.conftest import World, kinds

POOL = "state"


def make():
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(
        PoolGeometry(
            POOL, num_blocks=8, block_size=1, reserved_blocks=frozenset({0}), rewritable=True
        )
    )
    return lg, World(lg, pool=POOL)


def test_a_copy_from_a_written_published_block_fills_the_destination():
    ledger, world = make()
    world.alloc(3).request("A", [3], 1).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 1), source="clear", request="A"))
    world.drop_request("A").published(3, reserved=True).observe()
    world.alloc(5).request("B", [5], 1, num_computed=0).observe()
    ledger.external_copy(ExternalCopy(POOL, 3, POOL, 5, range(0, 1), request="B"))
    assert kinds(ledger) == []
    assert ledger.block(POOL, 5).is_filled(0)


def test_a_copy_from_a_free_or_foreign_or_unwritten_source_is_reported():
    ledger, world = make()
    world.alloc(3).request("A", [3], 1).observe()
    world.alloc(5).request("B", [5], 1).observe()
    ledger.external_copy(ExternalCopy(POOL, 6, POOL, 5, range(0, 1), request="B"))
    assert kinds(ledger)[-1] == "access_free_block"
    ledger.reporter.clear()
    ledger.external_copy(ExternalCopy(POOL, 3, POOL, 5, range(0, 1), request="B"))
    assert kinds(ledger) == ["access_foreign_private"]
    ledger.reporter.clear()
    ledger.external_copy(ExternalCopy(POOL, 3, POOL, 5, range(0, 1), request="A"))
    assert kinds(ledger)[0] == "read_uninitialized"  # A's own slot, never written


def test_a_copy_across_pools_reads_one_and_fills_the_other():
    ledger, world = make()
    ledger.register_pool(PoolGeometry("ckpt", num_blocks=4, block_size=1))
    world.alloc(3).request("A", [3], 1).observe()
    ledger.external_fill(ExternalFill(POOL, 3, range(0, 1), source="clear", request="A"))
    ledger.allocate("ckpt", [1])
    # The checkpoint slot is the tree's, not the request's: the copy carries no request, so the
    # destination's ownership is not checked against one.
    ledger.external_copy(ExternalCopy(POOL, 3, "ckpt", 1, range(0, 1), source="quantize"))
    assert kinds(ledger) == []
    assert ledger.block("ckpt", 1).is_filled(0)
