"""The stream-order epoch rule: a release while an epoch that carries a stream is open retires
instead of being reported, and the first dispatch that touches the block again is judged by stream
order. Descriptors without a stream keep the host-order judgment of test_ledger_lifecycle.py."""

from __future__ import annotations

from kvguard.core import BlockState, Ledger, Policy, Reporter

from tests.unit.conftest import BS, POOL, kinds, row

S1, S2 = 11, 22  # two stream handles


def released_row_reused(world, ledger, *, stream2, after=()):
    """Request A's block is freed while A's epoch on stream S1 is open; the block goes to B,
    whose dispatch runs on ``stream2``."""
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], stream=S1)
    world.drop_request("A").release(3)
    assert kinds(ledger) == []
    assert ledger.counters["released_in_flight"] == 1
    assert (POOL, 3) in ledger.retiring
    world.alloc(3).request("B", [3], 2, num_computed=0).observe(stream=S1)
    assert kinds(ledger) == []
    assert ledger.counters["row_released_in_flight"] == 1
    assert (POOL, 3) in ledger.retiring  # the allocation keeps the retiring record
    world.dispatch("d2", [row("B", writes=range(0, 2))], stream=stream2, after=after)


def test_same_stream_release_and_reuse_is_silent(world, ledger):
    released_row_reused(world, ledger, stream2=S1)
    assert kinds(ledger) == []
    assert ledger.counters["retired_in_order"] == 1
    assert ledger.retiring == {}
    world.complete("d1").complete("d2")
    assert ledger.epochs == {}


def test_cross_stream_reuse_without_after_is_reported(world, ledger):
    released_row_reused(world, ledger, stream2=S2)
    assert kinds(ledger) == ["reuse_before_epoch_retired"]
    r = ledger.reporter.reports[0]
    assert r.policy is Policy.P1 and r.block == 3 and r.site == "dispatch:d2"
    assert "'d1'" in r.detail and f"stream {S1}" in r.detail and f"stream {S2}" in r.detail
    assert ledger.retiring == {}  # judged once, at the first access


def test_cross_stream_reuse_with_a_declared_after_is_silent(world, ledger):
    released_row_reused(world, ledger, stream2=S2, after=("d1",))
    assert kinds(ledger) == []
    assert ledger.counters["retired_by_after"] == 1


def test_completion_before_the_reuse_makes_any_reuse_silent(world, ledger):
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], stream=S1)
    world.drop_request("A").release(3)
    assert (POOL, 3) in ledger.retiring
    world.complete("d1")
    assert ledger.retiring == {} and ledger._retiring_epochs == {}
    world.alloc(3).request("B", [3], 2, num_computed=0).observe()
    world.dispatch("d2", [row("B", writes=range(0, 2))], stream=S2)
    assert kinds(ledger) == []


def test_host_order_mode_keeps_the_release_reports(world, ledger):
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))])  # no stream
    ledger.free(POOL, [3])
    assert kinds(ledger) == ["free_while_in_flight"]
    assert ledger.retiring == {} and ledger.counters["released_in_flight"] == 0
    world.freed(3).drop_request("A").observe()
    assert "mapping_mutated_in_flight" in kinds(ledger)  # the released row, in host order


def test_a_released_row_is_silent_under_a_stream(world, ledger):
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], stream=S1)
    world.drop_request("A").release(3).observe(stream=S2)  # another stream: still a release
    assert kinds(ledger) == []
    assert ledger.counters["row_released_in_flight"] == 1


def test_mixed_epochs_report_host_order_and_retire_the_streamed_one(world, ledger):
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], stream=S1)
    world.dispatch("d0", [row("A", reads=range(0, 2))])  # host order
    ledger.free(POOL, [3])
    assert kinds(ledger) == ["free_while_in_flight"]
    assert "by 1 open dispatch" in ledger.reporter.reports[0].detail
    assert ledger.retiring == {(POOL, 3): {"d1": S1}}


def test_store_and_take_over_under_a_stream_are_judged_at_the_next_access(world, ledger):
    """The TensorRT-LLM shape: a finished request's block goes to the reuse tree and is taken
    over while the epoch that reads it is open; legal when the taker's kernels follow on the
    same stream, the use-after-free when they run on another stream with no declared order."""
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    world.dispatch("d", [row("A", reads=range(0, BS))], stream=S1)
    ledger.store(POOL, [3])
    ledger.reuse(POOL, 3, BS)
    assert kinds(ledger) == []
    assert ledger.counters["released_in_flight"] == 2  # the store and the take-over
    assert ledger.block(POOL, 3).state is BlockState.PRIVATE
    world.drop_request("A").private(3).request("B", [3], BS, num_computed=BS).observe(stream=S1)
    assert kinds(ledger) == []
    world.dispatch("d2", [row("B", reads=range(0, BS))], stream=S2)
    assert kinds(ledger) == ["reuse_before_epoch_retired"]


def repointed_row(world):
    """SGLang's checkpoint shape: a running request's row is repointed (entry 1 from block 4 to
    block 5, block 4 freed) while the step that consumed it is open on stream S1."""
    world.alloc(3, 4).request("A", [3, 4], 2 * BS, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2 * BS))], stream=S1)
    world.alloc(5).request("A", [3, 5], 2 * BS, num_computed=0)
    world.release(4)


def test_a_table_rewrite_enqueued_on_the_epochs_stream_is_ordered(world, ledger):
    repointed_row(world)
    world.observe(stream=S1)
    assert kinds(ledger) == []
    assert ledger.counters["mapping_rewritten_in_order"] == 1


def test_a_table_rewrite_on_another_stream_is_a_mutation_in_flight(world, ledger):
    repointed_row(world)
    world.observe(stream=S2)
    assert kinds(ledger) == ["mapping_mutated_in_flight"]


def test_a_table_rewrite_behind_a_declared_fence_is_ordered(world, ledger):
    repointed_row(world)
    world.observe(stream=S2, after=("d1",))
    assert kinds(ledger) == []
    assert ledger.counters["mapping_rewritten_in_order"] == 1


def test_a_snapshot_side_free_retires_under_a_stream(geometry):
    """A block that leaves the non-free set between snapshots (no mediated free: an unmediated
    pool) retires like a mediated free does."""
    from tests.unit.conftest import World

    ledger = Ledger(Reporter(fail_fast=False), mediated=False)
    ledger.register_pool(geometry)
    world = World(ledger)
    world.alloc(3).request("A", [3], 2, num_computed=0).observe()
    world.dispatch("d1", [row("A", writes=range(0, 2))], stream=S1)
    world.drop_request("A").freed(3).observe(stream=S1)
    assert kinds(ledger) == []
    assert ledger.retiring == {(POOL, 3): {"d1": S1}}
    assert ledger.block(POOL, 3).state is BlockState.FREE
    world.alloc(3).request("B", [3], 2, num_computed=0).observe(stream=S1)
    world.dispatch("d2", [row("B", writes=range(0, 2))], stream=S1)
    assert kinds(ledger) == []
    assert ledger.retiring == {}


def test_the_retiring_record_survives_several_releases_and_epochs(world, ledger):
    world.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    world.dispatch("d1", [row("A", reads=range(0, 2))], stream=S1)
    world.dispatch("d2", [row("A", reads=range(0, 2))], stream=S2)
    world.drop_request("A").release(3)
    assert ledger.retiring == {(POOL, 3): {"d1": S1, "d2": S2}}
    world.complete("d2")
    assert ledger.retiring == {(POOL, 3): {"d1": S1}}
    world.alloc(3).request("B", [3], 2, num_computed=0).observe(stream=S1)
    world.dispatch("d3", [row("B", writes=range(0, 2))], stream=S1)
    assert kinds(ledger) == []
    assert ledger.retiring == {} and ledger._retiring_epochs == {}
