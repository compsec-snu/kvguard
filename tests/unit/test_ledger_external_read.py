"""Asynchronous non-kernel reads use the same lifecycle epochs as GPU dispatches."""

from __future__ import annotations

from kvguard.core import ExternalFill, ExternalRead, ExternalWrite

from tests.unit.conftest import BS, POOL, kinds, row

S1, S2 = 41, 42


def start_written_read(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    ledger.external_read(
        ExternalRead("store:1", POOL, (3,), range(BS), "cpu_offload_store", "A", S1),
        site="offload-store:1",
    )
    assert kinds(ledger) == []
    assert ledger.inflight(POOL, 3) == 1


def test_external_read_reuse_before_completion_is_reported(world, ledger):
    start_written_read(world, ledger)
    world.drop_request("A").release(3)
    assert kinds(ledger) == []
    world.alloc(3).request("B", [3], 2).observe()
    world.dispatch("new-owner", [row("B", writes=range(2))], stream=S2)
    assert kinds(ledger) == ["reuse_before_epoch_retired"]


def test_external_read_completion_fences_the_new_owner(world, ledger):
    start_written_read(world, ledger)
    world.drop_request("A").release(3)
    world.alloc(3).request("B", [3], 2).observe()
    ledger.complete("store:1")
    world.dispatch("new-owner", [row("B", writes=range(2))], stream=S2)
    assert kinds(ledger) == []


def test_external_read_checks_liveness_ownership_and_initialization(world, ledger):
    world.alloc(3).request("A", [3], BS).observe()
    ledger.external_read(ExternalRead("free", POOL, (6,), range(BS), request="A"))
    ledger.external_read(ExternalRead("foreign", POOL, (3,), range(BS), request="B"))
    ledger.external_read(ExternalRead("unwritten", POOL, (3,), range(BS), request="A"))
    assert kinds(ledger) == [
        "access_free_block",
        "access_foreign_private",
        "read_uninitialized",
    ]


def test_external_read_accepts_physically_cleared_tail_without_filling_logical_kv(world, ledger):
    world.alloc(3).request("A", [3], BS).observe()
    ledger.external_read(
        ExternalRead(
            "zeroed-store",
            POOL,
            (3,),
            range(BS),
            request="A",
            preinitialized_blocks=frozenset({3}),
        )
    )
    ledger.complete("zeroed-store")
    world.dispatch("first-write", [row("A", writes=range(BS))])

    assert kinds(ledger) == []


def test_external_read_accepts_allocator_free_source_retained_by_transfer_fence(world, ledger):
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    world.drop_request("A").release(3).observe()

    ledger.external_read(
        ExternalRead(
            "retained-store",
            POOL,
            (3,),
            range(BS),
            request="A",
            retained_free_blocks=frozenset({3}),
        )
    )
    assert kinds(ledger) == []
    assert ledger.inflight(POOL, 3) == 1

    world.alloc(3)
    assert kinds(ledger) == ["allocate_in_flight"]


def test_external_write_protects_destination_without_initializing_it(world, ledger):
    world.alloc(3).request("A", [3], BS).observe()
    ledger.external_write(
        ExternalWrite("receive", POOL, (3,), range(BS), "pd_receive", "A"),
        site="receive",
    )

    world.dispatch("premature-read", [row("A", reads=range(BS))])
    assert kinds(ledger) == ["read_uninitialized"] * BS


def test_external_write_release_is_in_flight_until_completion(world, ledger):
    world.alloc(3).request("A", [3], BS).observe()
    ledger.external_write(ExternalWrite("receive", POOL, (3,), range(BS)))
    world.drop_request("A").release(3)

    assert kinds(ledger) == ["free_while_in_flight"]


def test_external_write_success_is_filled_only_at_observed_completion(world, ledger):
    world.alloc(3).request("A", [3], BS).observe()
    ledger.external_write(ExternalWrite("receive", POOL, (3,), range(BS), request="A"))
    ledger.external_fill(ExternalFill(POOL, 3, range(BS), "pd_receive", "A"))
    ledger.complete("receive")
    world.dispatch("read", [row("A", reads=range(BS))])

    assert kinds(ledger) == []


def test_external_write_can_precede_first_snapshot(world, ledger):
    ledger.allocate(POOL, (3,))
    ledger.external_write(ExternalWrite("receive", POOL, (3,), range(BS)))

    assert kinds(ledger) == []
    assert ledger.inflight(POOL, 3) == 1
