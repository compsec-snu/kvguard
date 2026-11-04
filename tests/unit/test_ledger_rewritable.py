"""A rewritable pool (state slots rewritten in place by their owner): repeated writes are
updates; initialization before a read and the single-writer rule still hold."""

from __future__ import annotations

import pytest
from kvguard.core import ExternalFill, Ledger, PoolGeometry, Reporter

from tests.unit.conftest import World, kinds, row

POOL = "state"


@pytest.fixture(params=[False, True], ids=["kv", "rewritable"])
def rewritable(request):
    return request.param


@pytest.fixture
def ledger(rewritable):
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(
        PoolGeometry(
            POOL, num_blocks=8, block_size=1, reserved_blocks=frozenset({0}), rewritable=rewritable
        )
    )
    return lg


@pytest.fixture
def world(ledger):
    return World(ledger, pool=POOL)


def test_a_repeated_write_is_an_update_only_in_a_rewritable_pool(world, ledger, rewritable):
    world.alloc(3).request("A", [3], 1, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 1))]).complete("d1")
    world.dispatch("d2", [row("A", reads=range(0, 1), writes=range(0, 1))]).complete("d2")
    assert kinds(ledger) == ([] if rewritable else ["double_write"])


def test_a_read_before_any_write_is_reported_in_both(world, ledger):
    world.alloc(3).request("A", [3], 1, num_computed=0).observe()
    world.dispatch("d1", [row("A", reads=range(0, 1))])
    assert kinds(ledger) == ["read_uninitialized"]


def test_a_clear_initializes_and_a_later_clear_is_an_update_in_a_rewritable_pool(
    world, ledger, rewritable
):
    world.alloc(3).request("A", [3], 1, num_computed=0).observe()
    ledger.external_fill(ExternalFill(POOL, 3, (0,), source="clear", request="A"))
    world.dispatch("d1", [row("A", reads=range(0, 1), writes=range(0, 1))]).complete("d1")
    ledger.external_fill(ExternalFill(POOL, 3, (0,), source="copy", request="A"))
    # The KV pool reports the dispatch's write and the second fill (the latter deduplicated
    # by the drain reporter: same kind, block and slot); the rewritable pool neither.
    assert kinds(ledger) == ([] if rewritable else ["double_write"])
    assert ledger.reporter.duplicates == (0 if rewritable else 1)


def test_two_rows_writing_one_slot_in_one_dispatch_is_reported_in_both(world, ledger):
    world.alloc(3).request("A", [3], 1, num_computed=0).observe()
    world.alloc(4).request("B", [4], 1, num_computed=0).observe()
    # B's row names A's slot (a swapped state index): the foreign access is the first report.
    from kvguard.core import SlotMap

    world.dispatch(
        "d1",
        [row("A", writes=range(0, 1)), row("B", writes=range(0, 1))],
        mapping=SlotMap([3, 3], [0, 1], [0, 0]),
        family="state",
    )
    assert "slot_map_mismatch" in kinds(ledger) or "access_foreign_private" in kinds(ledger)
