"""The incremental snapshot: `Ledger.observe` keeps per-request and per-block state between
snapshots and re-derives only what changed; it must report exactly what a full derivation reports.
Every scenario runs on three ledgers in lockstep: the incremental one fed complete listings, a
reference that forgets the incremental state before each snapshot (`_forget_incremental_state`),
and the incremental one fed only the change (`Snapshot.delta_pools`). After every call the three
report streams must agree, and after every snapshot the shadows and kept listings must agree; a
randomized scenario sweeps allocator events, mapping changes and injected faults."""

from __future__ import annotations

import random

from kvguard.core import (
    BlockSnapshot,
    BlockState,
    ExternalFill,
    Ledger,
    PoolGeometry,
    Reporter,
    Snapshot,
)

from tests.unit.conftest import BS, POOL, World


class FullLedger(Ledger):
    """The reference: every snapshot derived from scratch (the shadow itself is kept)."""

    def observe(self, snapshot):
        self._forget_incremental_state()
        super().observe(snapshot)


class DeltaWorld(World):
    """Feeds its ledger the change since the previous snapshot: the block objects that are not
    the ones listed before (new, or replaced by the world) and the blocks that left."""

    def __init__(self, ledger, pool=POOL):
        super().__init__(ledger, pool)
        self._prev: dict[int, BlockSnapshot] = {}

    def observe(self, stream=None, after=(), free_count=None):
        self.seq += 1
        prev, cur = self._prev, dict(self.blocks)
        changed = tuple(bs for b, bs in cur.items() if prev.get(b) is not bs)
        gone = frozenset(prev.keys() - cur.keys())
        snap = Snapshot(
            self.seq,
            tuple(self.requests.values()),
            changed,
            {self.pool: frozenset(self.free)} if self.free is not None else None,
            free_count={self.pool: free_count} if free_count is not None else None,
            stream=stream,
            after=tuple(after),
            delta_pools=frozenset({self.pool}),
            gone={self.pool: gone} if gone else None,
        )
        self.ledger.observe(snap)
        self._prev = cur
        return self


def _emitted(ledger):
    """What the ledger reported since the last call, then clear (so a persistent condition that
    fires at every snapshot is visible at every snapshot)."""
    rep = ledger.reporter
    out = (
        [
            (r.kind, r.detail, r.pool, r.block, r.slot, r.generation, r.request, r.site)
            for r in rep.reports
        ],
        rep.duplicates,
    )
    rep.clear()
    return out


def _shadow(ledger):
    blocks = {
        pool: {
            b: (
                sh.state,
                sh.owner,
                sh.readers,
                sh.reserved,
                sh.filled,
                sh.pending_publish,
                sh.generation,
            )
            for b, sh in enumerate(shadows)
            if sh.state is not BlockState.FREE or sh.generation
        }
        for pool, shadows in ledger.blocks.items()
    }
    requests = {k: v.snapshot for k, v in ledger.requests.items()}
    dirty = {p: set(s) for p, s in ledger.dirty.items()}
    for pool, mask in ledger.nonfree_mask.items():  # the mask is the set, at every point
        assert {b for b, v in enumerate(mask) if v} == ledger.nonfree[pool], pool
    return blocks, requests, dict(ledger.nonfree), dirty, set(ledger.epochs)


def _listing(ledger):
    """The listing `observe` keeps per pool, by value (the delta path updates it in place)."""
    return {pool: dict(lp) for pool, lp in ledger._listed.items()}


class Pair:
    """Three worlds driven in lockstep. Every `World` call is forwarded to both; the ledgers must
    then have reported the same, and after a snapshot hold the same shadow."""

    def __init__(self, geometry=None):
        geometry = geometry or PoolGeometry(POOL, 16, BS, reserved_blocks=frozenset({0}))
        self.inc = Ledger(Reporter(fail_fast=False))
        self.ref = FullLedger(Reporter(fail_fast=False))
        self.dlt = Ledger(Reporter(fail_fast=False))
        for lg in (self.inc, self.ref, self.dlt):
            lg.register_pool(geometry)
        self.worlds = (World(self.inc), World(self.ref), DeltaWorld(self.dlt))
        self.kinds: list[str] = []
        """The kinds the last call reported."""
        self.seen: set[str] = set()
        """Every kind reported so far."""

    @property
    def world(self) -> World:
        return self.worlds[0]

    def set_free(self, free):
        for w in self.worlds:
            w.free = None if free is None else set(free)
        return self

    def set_block(self, b, state, reserved=False, refs=None):
        for w in self.worlds:
            w.blocks[b] = BlockSnapshot(POOL, b, state, reserved, refs)
        return self

    def fill(self, b, offsets):
        """An external fill of a block (no request named: legal into any non-free block)."""
        for w in self.worlds:
            w.ledger.external_fill(ExternalFill(POOL, b, offsets))
        return self._compare("fill")

    def _compare(self, name):
        a, b, c = _emitted(self.inc), _emitted(self.ref), _emitted(self.dlt)
        assert a == b, f"{name}: incremental {a} != full {b}"
        assert c == a, f"{name}: delta-fed {c} != incremental {a}"
        self.kinds = [k for k, *_ in a[0]]
        self.seen.update(self.kinds)
        if name == "observe":
            assert _shadow(self.inc) == _shadow(self.ref)
            assert _shadow(self.dlt) == _shadow(self.inc)
            assert _listing(self.dlt) == _listing(self.inc), "the kept listing drifted"
            assert {p: v for p, v in self.dlt._held.items()} == {
                p: v for p, v in self.inc._held.items()
            }
        return self

    def __getattr__(self, name):
        if name.startswith("_") or not callable(getattr(World, name, None)):
            raise AttributeError(name)

        def call(*args, **kwargs):
            for w in self.worlds:
                getattr(w, name)(*args, **kwargs)
            return self._compare(name)

        return call


def _judged(ledger):
    """Spy on the blocks the ledger judges (step 3) at each snapshot."""
    seen: list[int] = []
    original = ledger._judge_block

    def spy(pool, b, bs, req_index, site):
        seen.append(b)
        return original(pool, b, bs, req_index, site)

    ledger._judge_block = spy
    return seen


# the scenarios of the brief
def test_decode_step_rederives_only_the_growing_entries():
    pair = Pair()
    judged = _judged(pair.inc)
    pair.alloc(1, 2, 3, 4).request("A", [1, 2], 5).request("B", [3, 4], 6).observe()
    assert pair.kinds == [] and sorted(judged) == [1, 2, 3, 4]
    pair.write("A", 0, 5).write("B", 0, 6)
    # A grows by one token inside its last block; B only moved its computed count.
    judged.clear()
    pair.request("A", [1, 2], 6, num_computed=5).observe()
    assert pair.kinds == [] and judged == [2]
    # A fills block 2 and takes block 5; B is unchanged.
    pair.write("A", 5, 6)
    judged.clear()
    pair.alloc(5).request("A", [1, 2, 5], 9, num_computed=6).observe()
    assert pair.kinds == [] and sorted(judged) == [2, 5]
    assert pair.inc._mappers[POOL] == {
        1: {"A": 1},
        2: {"A": 1},
        5: {"A": 1},
        3: {"B": 1},
        4: {"B": 1},
    }
    assert [len(s.entries) for s in pair.inc.requests.values()] == [3, 2]
    # An identical snapshot judges nothing and dirties nothing.
    for lg in (pair.inc, pair.ref, pair.dlt):
        lg.drain_dirty(POOL)
    judged.clear()
    pair.observe()
    assert pair.kinds == [] and judged == [] and pair.inc.drain_dirty(POOL) == set()


def test_block_changes_state_while_no_request_changes():
    pair = Pair()
    pair.alloc(3).request("A", [3], BS).observe().write("A", 0, BS).observe()
    pair.published(3).observe()  # the full block is published, A reads it
    assert pair.kinds == [] and pair.inc.block(POOL, 3).readers == {"A"}
    pair.set_block(3, BlockState.PUBLISHED, refs=2).observe()
    assert pair.kinds == ["reference_count_mismatch"]
    pair.observe()  # the condition persists: reported again, as the full derivation does
    assert pair.kinds == ["reference_count_mismatch"]
    pair.drop_request("A").published(3).observe()
    assert pair.kinds == ["published_unreferenced"]
    pair.published(3, reserved=True).observe()
    assert pair.kinds == []


def test_request_disappears_and_leaves_its_block_listed():
    pair = Pair()
    pair.alloc(3, 4).request("A", [3], 2).request("B", [4], 2).observe()
    pair.drop_request("A").observe()
    assert pair.kinds == ["private_unmapped"]
    pair.observe()
    assert pair.kinds == ["private_unmapped"]
    pair.release(3).observe()
    assert pair.kinds == [] and pair.inc._mappers[POOL] == {4: {"B": 1}}


def test_block_reallocated_to_another_request_between_snapshots():
    """The first request still maps the block (stale): two owners, and its computed claim is now
    false because the allocation reset the initialization record."""
    pair = Pair()
    pair.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    pair.release(3).alloc(3).request("B", [3], 1).observe()
    assert pair.kinds == ["private_multiple_owners", "computed_beyond_filled"]
    assert pair.inc.reporter.reports == []  # cleared by the comparison; the kinds are above
    pair.observe()
    assert pair.kinds == ["private_multiple_owners", "computed_beyond_filled"]


def test_block_reallocated_after_its_request_left():
    pair = Pair()
    pair.alloc(3).request("A", [3], 2).observe().write("A", 0, 2).observe()
    pair.drop_request("A").release(3).alloc(3).request("B", [3], 1).observe()
    assert pair.kinds == [] and pair.inc.block(POOL, 3).owner == "B"
    assert pair.inc.block(POOL, 3).generation == 2


def test_identical_snapshots_report_nothing_and_dirty_nothing():
    pair = Pair()
    pair.alloc(3, 4).request("A", [3, 4], 6, num_computed=0).observe()
    for lg in (pair.inc, pair.ref, pair.dlt):
        lg.drain_dirty(POOL)
    pair.observe().observe()
    assert pair.kinds == [] and pair.inc.drain_dirty(POOL) == set()


# persistence and kept prefixes
def test_persistent_mapping_condition_fires_at_every_snapshot():
    pair = Pair()
    pair.alloc(3).request("A", [3, 3], 6).observe()
    assert pair.kinds == ["duplicate_slot_in_request"] * 2
    pair.observe()  # A is kept; its reports are re-emitted
    assert pair.kinds == ["duplicate_slot_in_request"] * 2
    pair.private(0).observe()  # a structural report, every snapshot
    assert pair.kinds == ["state_for_unmanaged_block"] + ["duplicate_slot_in_request"] * 2
    pair.observe()
    assert pair.kinds == ["state_for_unmanaged_block"] + ["duplicate_slot_in_request"] * 2
    pair.freed(0).observe()  # dropped (a delta names it gone): the structural report stops
    assert pair.kinds == ["duplicate_slot_in_request"] * 2


def test_kept_prefix_still_detects_a_duplicate_in_the_new_entry():
    pair = Pair()
    pair.alloc(3).request("A", [3], 4).observe()
    pair.request("A", [3, 3], 5).observe()  # the new entry repeats the kept block
    assert pair.kinds == ["duplicate_slot_in_request"]


def test_mapping_to_a_free_block_until_it_is_allocated_again():
    pair = Pair()
    pair.alloc(3).request("A", [3], 2).observe().write("A", 0, 2)
    pair.release(3).observe()  # freed under the unchanged request
    assert pair.kinds == ["mapping_to_free_block"]
    pair.observe()
    assert pair.kinds == ["mapping_to_free_block"]
    pair.alloc(3).observe()  # listed again: the request is re-derived, its claim re-checked
    assert pair.kinds == ["computed_beyond_filled"]


def test_sliding_window_advance_rederives_the_request():
    pair = Pair()
    pair.alloc(4).request("A", [None, 4], 8, first_token=3).observe()
    assert pair.kinds == ["live_position_without_block"]
    pair.observe()
    assert pair.kinds == ["live_position_without_block"]
    pair.request("A", [None, 4], 8, first_token=4).observe()
    assert pair.kinds == []
    pair.alloc(5).request("A", [None, 4, 5], 9, first_token=5).observe()
    assert pair.kinds == []


def test_shrinking_tokens_rederives_the_request():
    pair = Pair()
    pair.alloc(3, 4).request("A", [3, 4], 6).observe()
    pair.request("A", [3, 4], 3).observe()
    assert pair.kinds == [] and len(pair.inc.requests[("A", POOL)].entries) == 2
    pair.request("A", [3, 3], 6).observe()
    assert pair.kinds == ["duplicate_slot_in_request"] * 2 + ["private_unmapped"]


# the initialization record
def test_publication_never_written_hits_the_unchanged_reader():
    pair = Pair()
    pair.alloc(3).request("A", [3], BS).published(3).request("B", [3], BS, num_computed=BS)
    pair.observe()
    assert pair.kinds == []
    pair.observe()  # nothing wrote the block
    assert pair.kinds == ["publish_uninitialized", "computed_beyond_filled"]
    pair.observe()  # B's claim stays false
    assert pair.kinds == ["computed_beyond_filled"]


def test_rollback_shrink_is_seen_by_a_sharing_request_before_claims():
    """A fault with two owners: A's rollback clears slots B counts as computed. Rollbacks are
    applied before every request's claim, so declaration order cannot delay B's report."""
    for b_first in (False, True):
        pair = Pair()
        pair.alloc(3)
        for rid in ("B", "A") if b_first else ("A", "B"):
            pair.request(rid, [3], 4)
        pair.observe()
        assert pair.kinds == ["private_multiple_owners"]
        # With two owners nobody may write the block through a dispatch; an external fill with
        # no request named initializes it, then both count the four positions as computed.
        pair.fill(3, range(0, 4))
        pair.request("A", [3], 4, num_computed=4).request("B", [3], 4, num_computed=4).observe()
        assert pair.kinds == ["private_multiple_owners"]
        pair.request("A", [3], 4, num_computed=2).observe()  # A rolls back: slots 2, 3 cleared
        assert pair.kinds == ["private_multiple_owners", "computed_beyond_filled"], b_first
        pair.observe()
        assert pair.kinds == ["private_multiple_owners", "computed_beyond_filled"], b_first


def test_claim_watermark_falls_back_with_a_rollback_and_grows_again():
    pair = Pair()
    pair.alloc(3).request("A", [3], 4).observe().write("A", 0, 4).observe()
    pair.request("A", [3], 4, num_computed=2).observe()  # rollback: slots 2, 3 invalidated
    assert pair.kinds == [] and pair.inc.block(POOL, 3).filled == 0b0011
    pair.request("A", [3], 4, num_computed=4).observe()  # claimed again without a write
    assert pair.kinds == ["computed_beyond_filled"]
    pair.write("A", 2, 4).observe()
    assert pair.kinds == []


def test_malformed_snapshot_leaves_the_state_untouched():
    pair = Pair()
    pair.alloc(3).request("A", [3], 2).observe()
    for w in pair.worlds:
        w.seq += 1
        from kvguard.core import Snapshot

        snap = Snapshot(w.seq, tuple(w.requests.values()), tuple(w.blocks.values()) * 2)
        w.ledger.observe(snap)
    a, b, c = _emitted(pair.inc), _emitted(pair.ref), _emitted(pair.dlt)
    assert a == b == c and [k for k, *_ in a[0]] == ["malformed_snapshot"]
    pair.request("A", [3], 3).observe()
    assert pair.kinds == []


def _delta(seq, blocks=(), gone=None, requests=(), pool=POOL):
    return Snapshot(
        seq,
        tuple(requests),
        tuple(blocks),
        delta_pools=frozenset({pool}),
        gone={pool: frozenset(gone)} if gone else None,
    )


def test_delta_listing_finds_the_allocated_block_the_change_omits():
    """A block allocated since the previous snapshot that the change does not list left the
    non-free set for the ledger, as it does for a listing in full (the step is over the blocks
    the allocator events touched and the blocks the change says left, not over the pool);
    a block that left the listing is retired the same way; the non-free mask follows."""
    pair = Pair()
    pair.alloc(1, 2).request("A", [1, 2], 6).observe()
    assert pair.kinds == []
    for lg in (pair.inc, pair.ref, pair.dlt):  # 3 allocated, not listed: freed without a free
        lg.allocate(POOL, (3,))
    pair.observe()
    assert "unmediated_free" in pair.kinds
    assert all(3 not in lg.nonfree[POOL] for lg in (pair.inc, pair.ref, pair.dlt))
    pair.alloc(3).request("A", [1, 2, 3], 10).observe()
    assert "unmediated_free" not in pair.kinds
    pair.drop_request("A")
    for w in pair.worlds:  # 2 leaves the listing while still non-free: no mediated free
        w.blocks.pop(2)
    pair.observe()
    assert "unmediated_free" in pair.kinds and pair.inc.nonfree[POOL] == {1, 3}
    pair.release(1).freed(1).observe()  # a mediated free, then gone from the listing: silent
    assert "unmediated_free" not in pair.kinds and pair.inc.nonfree[POOL] == {3}


def test_malformed_delta_leaves_the_listing_untouched():
    """A block both listed and gone, a block gone that was not listed, or blocks gone from a
    pool listed in full: one integration error each, nothing adopted, and the next well-formed
    snapshot is judged against the listing as it was."""
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry(POOL, 16, BS, reserved_blocks=frozenset({0})))
    ledger.allocate(POOL, (3, 4))
    a = BlockSnapshot(POOL, 3, BlockState.PRIVATE)
    b = BlockSnapshot(POOL, 4, BlockState.PRIVATE)
    ledger.observe(_delta(1, (a, b)))
    assert [r.kind for r in ledger.reporter.reports] == ["private_unmapped"] * 2
    ledger.reporter.clear()
    before = _listing(ledger), dict(ledger._held)
    c = BlockSnapshot(POOL, 4, BlockState.PUBLISHED, reserved=True)
    for snap in (
        _delta(2, (c,), gone=(4,)),  # listed and gone
        _delta(2, (), gone=(5,)),  # gone but never listed
        Snapshot(2, (), (a, b), gone={POOL: frozenset({4})}),  # gone from a pool listed in full
    ):
        ledger.observe(snap)
        assert [r.kind for r in ledger.reporter.reports] == ["malformed_snapshot"], snap
        ledger.reporter.clear()
        assert (_listing(ledger), dict(ledger._held)) == before
    ledger.observe(_delta(2, (c,)))  # 3 kept (its object listed before), 4 published
    assert [r.kind for r in ledger.reporter.reports] == ["private_unmapped"]
    # 4 is reserved and the geometry puts reserved blocks in the free set: 3 alone is held.
    assert _listing(ledger) == {POOL: {3: a, 4: c}} and ledger._held == {POOL: 1}
    ledger.reporter.clear()
    ledger.free(POOL, (3,))
    ledger.observe(_delta(3, (), gone=(3,)))
    assert ledger.reporter.reports == [] and _listing(ledger) == {POOL: {4: c}}
    assert ledger._held == {POOL: 0}


def test_delta_listing_keeps_the_count_for_the_audit():
    """The partition audit by count uses the held count the listing keeps: a delta that frees,
    publishes or reserves a block moves it exactly as a full listing would (`reserved_in_free`
    on and off)."""
    for rif in (True, False):
        ledger = Ledger(Reporter(fail_fast=False))
        ledger.register_pool(
            PoolGeometry(POOL, 16, BS, reserved_blocks=frozenset({0}), reserved_in_free=rif)
        )
        ledger.allocate(POOL, (3, 4, 5))
        blocks = [BlockSnapshot(POOL, b, BlockState.PRIVATE) for b in (3, 4, 5)]
        ledger.observe(Snapshot(1, (), tuple(blocks), free_count={POOL: 15 - 3}))
        kinds = [r.kind for r in ledger.reporter.reports]
        assert "block_count_mismatch" not in kinds and ledger._held[POOL] == 3
        ledger.reporter.clear()
        reserved = BlockSnapshot(POOL, 5, BlockState.PUBLISHED, reserved=True)
        # The free count the allocator reports: a reserved block is in its free set with rif.
        ledger.observe(
            _delta(2, (reserved,)).__class__(
                2,
                (),
                (reserved,),
                free_count={POOL: 15 - (2 if rif else 3)},
                delta_pools=frozenset({POOL}),
            )
        )
        kinds = [r.kind for r in ledger.reporter.reports]
        assert "block_count_mismatch" not in kinds, (rif, kinds)
        assert ledger._held[POOL] == (2 if rif else 3)
        ledger.reporter.clear()
        ledger.free(POOL, (3,))
        ledger.observe(
            Snapshot(
                3,
                (),
                (),
                free_count={POOL: 15 - (1 if rif else 2)},
                delta_pools=frozenset({POOL}),
                gone={POOL: frozenset({3})},
            )
        )
        kinds = [r.kind for r in ledger.reporter.reports]
        assert "block_count_mismatch" not in kinds, (rif, kinds)
        assert ledger._held[POOL] == (1 if rif else 2)
        ledger.reporter.clear()
        ledger.observe(
            Snapshot(
                4, (), (), free_count={POOL: 15 - (2 if rif else 3)}, delta_pools=frozenset({POOL})
            )
        )
        assert [r.kind for r in ledger.reporter.reports].count("block_count_mismatch") == 1


# randomized differential sweep
def _random_scenario(seed, steps):
    rng = random.Random(seed)
    pair = Pair()
    w = pair.world
    managed = list(range(1, 16))
    counter = iter(range(10_000))

    def new_id():
        return f"r{next(counter)}"

    def free_blocks():
        return [b for b in managed if b not in w.blocks]

    for _ in range(steps):
        for _ in range(rng.randint(1, 3)):
            op = rng.random()
            reqs = list(w.requests)
            fb = free_blocks()
            if op < 0.18 and fb:  # admit a request
                bl = rng.sample(fb, rng.randint(1, min(2, len(fb))))
                pair.alloc(*bl).request(new_id(), bl, rng.randint(1, len(bl) * BS))
            elif op < 0.40 and reqs:  # a decode step: one more token, a block when needed
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                bl = list(rs.blocks)
                nt = rs.num_tokens + 1
                if nt > len(bl) * BS:
                    if not fb:
                        continue
                    b = rng.choice(fb)
                    pair.alloc(b)
                    bl.append(b)
                pair.request(rid, bl, nt, rs.num_computed, rs.first_token)
            elif op < 0.55 and reqs:  # write what is not yet computed
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                if rs.num_computed < rs.num_tokens:
                    pair.write(rid, rs.num_computed, rs.num_tokens)
            elif op < 0.65 and reqs:  # publish a full written block, maybe with a prefix hit
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                full = [
                    b
                    for i, b in enumerate(rs.blocks)
                    if b is not None and 0 < b < 16 and (i + 1) * BS <= rs.num_computed
                ]
                if full:
                    b = rng.choice(full)
                    pair.published(b)
                    if rng.random() < 0.5 and fb:
                        own = rng.choice(fb)
                        pair.alloc(own).request(new_id(), [b, own], BS + 1, num_computed=BS)
            elif op < 0.75 and reqs:  # finish a request; sometimes leave a block behind
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                pair.drop_request(rid)
                others = {b for s in w.requests.values() for b in s.blocks}
                for b in rs.blocks:
                    bs = w.blocks.get(b) if b is not None else None
                    if bs is None or b in others:
                        continue
                    if bs.state is BlockState.PUBLISHED:
                        if rng.random() < 0.7:
                            pair.published(b, reserved=True)
                    elif rng.random() < 0.85:
                        pair.release(b)
            elif op < 0.80 and reqs:  # a rollback of the computed count
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                if rs.num_computed:
                    nc = rng.randint(0, rs.num_computed - 1)
                    pair.request(rid, rs.blocks, rs.num_tokens, nc, rs.first_token)
            elif op < 0.86 and reqs:  # fault: a mapped block is freed
                rid = rng.choice(reqs)
                bl = [b for b in w.requests[rid].blocks if b is not None and b in w.blocks]
                if bl:
                    pair.release(rng.choice(bl))
            elif op < 0.90 and reqs:  # fault: a mapped block is reallocated to a new request
                rid = rng.choice(reqs)
                bl = [b for b in w.requests[rid].blocks if b is not None and 0 < b < 16]
                if bl:
                    b = rng.choice(bl)
                    if b in w.blocks:
                        pair.release(b)
                    pair.alloc(b).request(new_id(), [b], 1)
            elif op < 0.93 and reqs:  # fault: the framework's holder count disagrees
                rid = rng.choice(reqs)
                bl = [b for b in w.requests[rid].blocks if b is not None and b in w.blocks]
                if bl:
                    b = rng.choice(bl)
                    bs = w.blocks[b]
                    pair.set_block(b, bs.state, bs.reserved, refs=rng.randint(0, 3))
            elif op < 0.96 and reqs:  # fault: a duplicate, reserved or missing entry
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                bl = [*rs.blocks, rng.choice([rs.blocks[0], 0, None])]
                pair.request(rid, bl, rs.num_tokens + 1, rs.num_computed, rs.first_token)
            elif reqs:  # the window advances
                rid = rng.choice(reqs)
                rs = w.requests[rid]
                if rs.first_token + 1 <= rs.num_tokens:
                    pair.request(rid, rs.blocks, rs.num_tokens, rs.num_computed, rs.first_token + 1)
        if rng.random() < 0.3:  # the partition audit, sometimes with a fault
            free = set(free_blocks())
            if rng.random() < 0.3 and w.blocks:
                free.add(rng.choice(list(w.blocks)))
            elif rng.random() < 0.3 and free:
                free.discard(rng.choice(sorted(free)))
            pair.set_free(free)
        else:
            pair.set_free(None)
        pair.observe()
    return pair


def test_randomized_scenarios_match_the_full_derivation():
    seen: set[str] = set()
    for seed in range(150):
        seen |= _random_scenario(seed, steps=20).seen
    # The sweep is only as good as what it reaches: these snapshot kinds must all have fired.
    expected = {
        "mapping_to_free_block",
        "private_multiple_owners",
        "private_unmapped",
        "published_unreferenced",
        "reference_count_mismatch",
        "duplicate_slot_in_request",
        "live_position_without_block",
        "mapping_reserved_block",
        "computed_beyond_filled",
        "block_free_and_allocated",
        "block_unaccounted",
    }
    assert not expected - seen, sorted(expected - seen)
