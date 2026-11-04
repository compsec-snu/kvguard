"""The vLLM adapter's incremental snapshot cache against the full-read builder (the oracle)."""

from __future__ import annotations

from kvguard.adapters.vllm.snapshot_cache import SnapshotCache
from kvguard.adapters.vllm.translate import BlockFacts, RequestFacts, build_snapshot

POOL = "gpu"
BS = 4


class Block:
    def __init__(self, block_id: int, is_null: bool = False):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.is_null = is_null


class Pool:
    """A pool with the mutations the attachment wraps; every mutation marks the cache."""

    def __init__(self, cache: SnapshotCache, num_blocks: int = 16):
        self.blocks = [Block(0, is_null=True)] + [Block(i) for i in range(1, num_blocks)]
        self.cache = cache

    def allocate(self, ids):
        out = []
        for i in ids:
            b = self.blocks[i]
            assert b.ref_cnt == 0
            b.block_hash = None
            b.ref_cnt = 1
            out.append(b)
        self.cache.mark(ids)
        return out

    def touch(self, ids):
        for i in ids:
            self.blocks[i].ref_cnt += 1
        self.cache.mark(ids)

    def hash_blocks(self, ids):
        for i in ids:
            self.blocks[i].block_hash = f"h{i}"
        self.cache.mark(ids)

    def free(self, ids):
        for i in ids:
            self.blocks[i].ref_cnt -= 1
        self.cache.mark(ids)

    def evict(self, ids):
        for i in ids:
            self.blocks[i].block_hash = None
        self.cache.mark(ids)

    def block_of(self, i):
        return self.blocks[i]


def oracle(seq, entries, pool):
    """``build_snapshot`` over a full read of the pool: the reference."""
    reqs = [
        RequestFacts(rid, tuple(None if b.is_null else b.block_id for b in blocks), comp, sched)
        for rid, blocks, comp, sched in entries
    ]
    facts = [
        BlockFacts(b.block_id, b.ref_cnt, b.block_hash is not None, b.is_null) for b in pool.blocks
    ]
    return build_snapshot(seq, POOL, BS, reqs, facts, None)


def step(cache, pool, seq, entries, *, audit=False):
    requests = cache.requests_step(entries)
    if audit or not cache.valid:
        drift, example = cache.resync(pool.block_of, pool.blocks)
    else:
        cache.blocks_step(pool.block_of)
        drift, example = 0, None
    ref = oracle(seq, entries, pool)
    assert set(cache.blocks()) == set(ref.blocks), (seq, cache.blocks(), ref.blocks)
    assert requests == list(ref.requests)
    return drift, example


def test_cache_follows_the_full_read_through_a_request_life():
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    a = pool.allocate([1, 2, 3])
    # Step 1: A's 10-token prompt scheduled in full (first fill of the cache).
    step(cache, pool, 1, [("A", a, 10, 10)])
    assert {b.block for b in cache.blocks()} == {1, 2, 3}
    assert cache.stats["snapshot_requests_rebuilt"] == 1
    # Step 2: decode; the pool hashes A's two full blocks; A covers them now.
    pool.hash_blocks([1, 2])
    step(cache, pool, 2, [("A", a, 11, 1)])
    listed = {b.block: b for b in cache.blocks()}
    assert listed[1].state.value == "published" and listed[3].state.value == "private"
    assert cache.stats["snapshot_requests_reused"] == 1
    # Step 3: B shares block 1 (a prefix hit) and gets block 4; A grows into block 5.
    pool.touch([1])
    b_blocks = [pool.blocks[1], *pool.allocate([4])]
    a.extend(pool.allocate([5]))
    step(cache, pool, 3, [("A", a, 17, 6), ("B", b_blocks, 6, 2)])
    assert cache.stats["snapshot_requests_extended"] == 1
    assert {b.block: b.refs for b in cache.blocks()}[1] == 2
    # Step 4: A finishes: block 5 and 3 go free, 2 stays cached (reserved), 1 is B's.
    pool.free([5, 3, 2, 1])
    step(cache, pool, 4, [("B", b_blocks, 7, 1)])
    listed = {b.block: b for b in cache.blocks()}
    assert set(listed) == {1, 2, 4}
    assert listed[2].reserved and not listed[1].reserved
    # Step 5: an audited step finds nothing to correct; the objects are reused.
    before = {b.block: b for b in cache.blocks()}
    drift, _ = step(cache, pool, 5, [("B", b_blocks, 8, 1)], audit=True)
    assert drift == 0
    after = {b.block: b for b in cache.blocks()}
    assert all(after[b] is before[b] for b in after if b in before and b != 4)


def test_unchanged_blocks_keep_their_snapshot_objects():
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    a = pool.allocate([1, 2])
    step(cache, pool, 1, [("A", a, 8, 8)])
    first = {b.block: b for b in cache.blocks()}
    step(cache, pool, 2, [("A", a, 9, 1)])  # the coverage moves to blocks 1 and 2
    second = {b.block: b for b in cache.blocks()}
    assert second[1] is not first[1]  # covered now
    step(cache, pool, 3, [("A", a, 10, 1)])
    third = {b.block: b for b in cache.blocks()}
    assert third[1] is second[1] and third[2] is second[2]


def test_the_cache_lists_its_change_between_listings():
    """``delta()`` gives the objects created and the blocks dropped since the last listing;
    applied to the kept objects they reproduce the oracle's complete listing, and a complete
    listing (``blocks()``) resets the change."""
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    kept: dict[int, object] = {}

    def advance(seq, entries):
        cache.requests_step(entries)
        if not cache.valid:
            cache.resync(pool.block_of, pool.blocks)
        else:
            cache.blocks_step(pool.block_of)
        changed, gone = cache.delta()
        for b in gone:
            del kept[b]
        for x in changed:
            kept[x.block] = x
        assert set(kept.values()) == set(oracle(seq, entries, pool).blocks), seq
        return {x.block for x in changed}, gone

    a = pool.allocate([1, 2])
    assert advance(1, [("A", a, 8, 8)]) == ({1, 2}, frozenset())
    b = pool.allocate([3])
    assert advance(2, [("A", a, 9, 1), ("B", b, 4, 4)]) == ({1, 2, 3}, frozenset())  # 1, 2 covered
    pool.free([1, 2])
    pool.evict([1, 2])
    assert advance(3, [("B", b, 5, 1)]) == ({3}, frozenset({1, 2}))  # 3 covered now
    three = kept[3]
    assert advance(4, [("B", b, 6, 1)]) == (set(), frozenset())
    assert cache.listed[3] is three, "an unchanged block keeps its object"
    # A block that leaves and comes back within one step is a change, not gone.
    pool.free([3])
    pool.allocate([3])
    assert advance(5, [("C", pool.blocks[3:4], 4, 4)]) == ({3}, frozenset())
    assert {x.block for x in cache.blocks()} == {3} and cache.delta() == ((), frozenset())


def test_a_missed_mutation_shows_as_drift_at_the_audit_and_is_corrected():
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    a = pool.allocate([1, 2])
    step(cache, pool, 1, [("A", a, 8, 8)])
    pool.blocks[2].ref_cnt += 1  # a path the wraps do not see: no mark
    cache.blocks_step(pool.block_of)
    assert {b.block: b.refs for b in cache.blocks()}[2] == 1  # stale, by construction
    drift, example = cache.resync(pool.block_of, pool.blocks)
    assert drift == 1 and "block 2" in example
    assert {b.block: b.refs for b in cache.blocks()}[2] == 2
    assert cache.stats["snapshot_drift"] == 1


def test_a_replaced_block_list_rebuilds_the_request_and_its_coverage():
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    a = pool.allocate([1, 2, 3])
    pool.hash_blocks([1, 2])
    step(cache, pool, 1, [("A", a, 9, 1)])  # A covers 1 and 2
    assert cache.cover == {1: 1, 2: 1}
    # Preempted and resumed with a new list: blocks 1 and 2 kept (cached), 3 replaced by 6.
    pool.free([3])
    new_list = [pool.blocks[1], pool.blocks[2], *pool.allocate([6])]
    step(cache, pool, 2, [("A", new_list, 9, 1)])
    assert cache.stats["snapshot_requests_rebuilt"] == 2
    assert cache.cover == {1: 1, 2: 1}
    # The request leaves: its coverage is withdrawn and the hashed blocks become reserved.
    pool.free([6, 2, 1])
    step(cache, pool, 3, [])
    assert cache.cover == {} and cache._req == {}
    assert {b.block: b.reserved for b in cache.blocks()} == {1: True, 2: True}


def test_null_blocks_in_a_list_are_holes():
    cache = SnapshotCache(POOL, BS)
    pool = Pool(cache)
    a = [pool.blocks[0], *pool.allocate([7])]
    requests = cache.requests_step([("A", a, 8, 4)])
    assert requests[0].blocks == (None, 7)
    cache.resync(pool.block_of, pool.blocks)
    assert {b.block for b in cache.blocks()} == {7}


def test_member_filter_lists_only_the_groups_blocks_and_follows_a_block_across_groups():
    """Two groups share one block pool: each cache lists the blocks its group holds, and a
    block freed from one group and allocated to the other leaves the first cache."""
    group_of = {}
    kv = SnapshotCache("gpu", BS, member=lambda b: group_of.get(b) == 0)
    st = SnapshotCache(
        "state1", BS, member=lambda b: group_of.get(b) == 1, columns=True, rebuild=True
    )
    pool = Pool(kv, num_blocks=16)
    pool.cache = st  # marks reach both caches below
    a = pool.allocate([1, 2])
    kv.mark([1, 2])
    group_of.update({1: 0, 2: 0})
    s = pool.allocate([3])
    kv.mark([3])
    group_of[3] = 1
    kv.requests_step([("A", a, 8, 8)])
    st.requests_step([("A", s, 8, 8)])
    kv.resync(pool.block_of, pool.blocks)
    st.resync(pool.block_of, pool.blocks)
    assert {b.block for b in kv.blocks()} == {1, 2} and {b.block for b in st.blocks()} == {3}
    # Block 3 is freed by group 1 and reallocated by group 0.
    pool.free([3])
    kv.mark([3])
    group_of.pop(3)
    pool.allocate([3])
    kv.mark([3])
    group_of[3] = 0
    a.append(pool.blocks[3])
    kv.requests_step([("A", a, 9, 1)])
    st.requests_step([])
    kv.blocks_step(pool.block_of)
    st.blocks_step(pool.block_of)
    assert {b.block for b in kv.blocks()} == {1, 2, 3} and st.blocks() == ()
    assert kv.held() == 3 and st.held() == 0


def test_columns_mode_emits_the_list_as_positions_and_rebuilds_mutated_lists():
    """A recurrent-state group's request maps its state columns (block size one for the core,
    the group's block size for the coverage); a column retired in place is seen next step."""
    cache = SnapshotCache("state0", 544, columns=True, rebuild=True)
    pool = Pool(cache, num_blocks=16)
    lst = [pool.blocks[0], *pool.allocate([5])]  # [null, 5]: column 0 skipped, column 1 live
    (r,) = cache.requests_step([("A", lst, 600, 600)])
    assert (r.blocks, r.num_tokens, r.num_computed) == ((None, 5), 2, 0)
    cache.resync(pool.block_of, pool.blocks)
    # The state in column 1 is final once 1088 tokens are computed: coverage, so a hash
    # publishes it; a hash before that leaves it private (the owner still rewrites it).
    pool.hash_blocks([5])
    cache.requests_step([("A", lst, 1087, 1)])
    cache.blocks_step(pool.block_of)
    assert {b.block: b.state.value for b in cache.blocks()} == {5: "private"}
    lst.append(pool.allocate([6])[0])  # the boundary step: column 2 allocated
    cache.requests_step([("A", lst, 1089, 1)])
    cache.blocks_step(pool.block_of)
    assert {b.block: b.state.value for b in cache.blocks()} == {5: "published", 6: "private"}
    # Column 1 retired in place (the align mode frees the previous state block).
    pool.free([5])
    pool.evict([5])
    lst[1] = pool.blocks[0]
    (r,) = cache.requests_step([("A", lst, 1090, 1)])
    cache.blocks_step(pool.block_of)
    assert r.blocks == (None, None, 6) and {b.block for b in cache.blocks()} == {6}
    assert cache.stats["snapshot_requests_rebuilt"] == 3  # every change of the list rebuilds


def test_window_mode_emits_the_window_start_as_the_first_live_position():
    cache = SnapshotCache("swa0", BS, rebuild=True, window=(8, 0))
    pool = Pool(cache, num_blocks=16)
    lst = pool.allocate([1, 2, 3, 4, 5, 6, 7, 8, 9])
    (r,) = cache.requests_step([("A", lst, 36, 36)])
    assert (r.first_token, r.num_computed, r.num_tokens) == (0, 0, 36)
    cache.resync(pool.block_of, pool.blocks)
    # The manager retires the blocks below the window in place: positions below 29 are dead.
    pool.free([1, 2, 3, 4, 5, 6, 7])
    for j in range(7):
        lst[j] = pool.blocks[0]
    lst.append(pool.allocate([10])[0])
    (r,) = cache.requests_step([("A", lst, 37, 1)])
    cache.blocks_step(pool.block_of)
    assert r.blocks == (None,) * 7 + (8, 9, 10) and r.first_token == 29
    assert {b.block for b in cache.blocks()} == {8, 9, 10}


def test_rebuild_mode_rebuilds_only_the_mutated_requests_when_told():
    """With the mutated set given, a request whose list only grew extends; one the framework
    changed in place is rebuilt; None rebuilds every request."""
    cache = SnapshotCache("swa0", BS, rebuild=True, window=(8, 0))
    pool = Pool(cache, num_blocks=16)
    a = pool.allocate([1, 2, 3])
    b = pool.allocate([4, 5, 6])
    cache.requests_step([("A", a, 12, 12), ("B", b, 12, 12)], set())
    cache.resync(pool.block_of, pool.blocks)
    a.append(pool.allocate([7])[0])  # A grows: extension
    pool.free([4])
    b[0] = pool.blocks[0]  # B retires block 4 in place
    b.append(pool.allocate([8])[0])
    (ra, rb) = cache.requests_step([("A", a, 13, 1), ("B", b, 13, 1)], {"B"})
    assert ra.blocks == (1, 2, 3, 7) and rb.blocks == (None, 5, 6, 8)
    assert (
        cache.stats["snapshot_requests_extended"] == 1
        and cache.stats["snapshot_requests_rebuilt"] == 3
    )
    # Without the set every request is rebuilt (the reference behaviour).
    cache.requests_step([("A", a, 14, 1), ("B", b, 14, 1)])
    assert (
        cache.stats["snapshot_requests_rebuilt"] == 3
        and cache.stats["snapshot_requests_reused"] == 2
    )
