"""The vLLM adapter's pure translation rules (no vLLM needed)."""

from collections import Counter, deque
from types import SimpleNamespace

import numpy as np
from kvguard.adapters.vllm.translate import (
    BlockFacts,
    RequestFacts,
    block_state,
    build_dispatches,
    build_positioned_dispatches,
    build_snapshot,
    mediated_frees,
    pre_step_computed,
)
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    DenseBlockTable,
    RequestSnapshot,
    SlotMap,
    Snapshot,
)


def test_historical_free_blocks_queue_controls_are_forwarded():
    """The adapter keeps older vLLM's ``prepend`` argument and materializes generators once."""
    from kvguard.adapters.vllm.attach import _call_free_blocks

    calls = []

    def free(blocks, *, prepend=False):
        calls.append((blocks, prepend))

    blocks = _call_free_blocks(free, (x for x in (1, 2)), prepend=True)
    assert blocks == [1, 2]
    assert calls == [([1, 2], True)]


def test_current_main_cpu_offload_splits_grouped_gpu_shape():
    from kvguard.adapters.vllm.attach import _grouped_gpu_blocks

    spec = SimpleNamespace(
        block_ids=np.asarray([3, 7, 11], dtype=np.int32),
        group_sizes=[2, 1],
        block_indices=[4, 9],
    )
    assert _grouped_gpu_blocks(spec, (0, 6)) == {0: (3, 7), 6: (11,)}


def test_current_main_cpu_offload_rejects_malformed_grouped_gpu_shape():
    import pytest
    from kvguard.adapters.vllm.attach import _grouped_gpu_blocks

    spec = SimpleNamespace(
        block_ids=np.asarray([3, 7], dtype=np.int32),
        group_sizes=[1, 0],
        block_indices=[4, 9],
    )
    with pytest.raises(ValueError, match="invalid grouped"):
        _grouped_gpu_blocks(spec, (0, 6))


def test_current_main_tiered_cpu_filesystem_offload_is_classified():
    from kvguard.adapters.vllm.attach import VllmAttachment

    primary = type("CPUPrimaryTierOffloadingManager", (), {})()
    filesystem = type("FileSystemTierManager", (), {})()
    manager = type(
        "TieringOffloadingManager",
        (),
        {"primary_tier": primary, "secondary_tiers": (filesystem,)},
    )()
    offload = SimpleNamespace(
        manager=manager,
        config=SimpleNamespace(
            kv_group_configs=(SimpleNamespace(group_idx=0, tokens_per_block=16),),
            blocks_per_chunk=2,
            num_workers=1,
        ),
    )
    connector = type("OffloadingConnector", (), {"connector_scheduler": offload})()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0},)
    att.managers = (SimpleNamespace(block_size=16),)
    att.offload_kind = None
    att.offload_tier_types = ()

    assert att._classify_offload() is offload
    assert att.offload_kind == "managed"
    assert att.offload_tier_types == ("FileSystemTierManager",)


def test_current_main_tiered_cpu_p2p_offload_is_classified():
    from kvguard.adapters.vllm.attach import VllmAttachment

    primary = type("CPUPrimaryTierOffloadingManager", (), {})()
    p2p = type("P2PSecondaryTierManager", (), {})()
    manager = type(
        "TieringOffloadingManager",
        (),
        {"primary_tier": primary, "secondary_tiers": (p2p,)},
    )()
    offload = SimpleNamespace(
        manager=manager,
        config=SimpleNamespace(
            kv_group_configs=(SimpleNamespace(group_idx=0, tokens_per_block=16),),
            blocks_per_chunk=2,
            num_workers=1,
        ),
    )
    connector = type("OffloadingConnector", (), {"connector_scheduler": offload})()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0},)
    att.managers = (SimpleNamespace(block_size=16),)
    att.offload_kind = None
    att.offload_tier_types = ()

    assert att._classify_offload() is offload
    assert att.offload_kind == "managed"
    assert att.offload_tier_types == ("P2PSecondaryTierManager",)


def test_current_main_tiered_cpu_kvcr_offload_is_classified():
    from kvguard.adapters.vllm.attach import VllmAttachment

    primary = type("CPUPrimaryTierOffloadingManager", (), {})()
    kvcr = type("KVCRSecondaryTierManager", (), {})()
    manager = type(
        "TieringOffloadingManager",
        (),
        {"primary_tier": primary, "secondary_tiers": (kvcr,)},
    )()
    offload = SimpleNamespace(
        manager=manager,
        config=SimpleNamespace(
            kv_group_configs=(SimpleNamespace(group_idx=0, tokens_per_block=16),),
            blocks_per_chunk=2,
            num_workers=1,
        ),
    )
    connector = type("OffloadingConnector", (), {"connector_scheduler": offload})()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0},)
    att.managers = (SimpleNamespace(block_size=16),)
    att.offload_kind = None
    att.offload_tier_types = ()

    assert att._classify_offload() is offload
    assert att.offload_kind == "managed"
    assert att.offload_tier_types == ("KVCRSecondaryTierManager",)


def test_current_main_nixl_pull_pd_is_classified_at_narrow_boundary():
    from kvguard.adapters.vllm.attach import VllmAttachment

    pd = type("NixlPullConnectorScheduler", (), {})()
    config = SimpleNamespace(
        kv_role="kv_consumer",
        kv_buffer_device="cuda",
        kv_connector_extra_config={},
    )
    connector = type(
        "NixlPullConnector",
        (),
        {"connector_scheduler": pd, "kv_transfer_config": config},
    )()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0, "kind": "attention"},)
    att.hybrid = False
    att.eagle3 = False
    att.dflash = False
    att.standalone_draft = False
    att.multi_module_mtp = False
    att.pd_kind = None
    att.pd = None

    assert att._classify_offload() is None
    assert att.pd_kind == "nixl_pull"
    assert att.pd is pd


def test_current_main_mooncake_store_is_classified_at_rank_local_hybrid_boundary():
    from kvguard.adapters.vllm.attach import VllmAttachment

    store = type("MooncakeStoreScheduler", (), {"_store_group_ids": (0, 1), "kv_role": "kv_both"})()
    config = SimpleNamespace(kv_role="kv_both")
    connector = type(
        "MooncakeStoreConnector",
        (),
        {"connector_scheduler": store, "kv_transfer_config": config},
    )()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0}, {"group": 1})
    att.speculative = False
    att.eagle3 = False
    att.dflash = False
    att.standalone_draft = False
    att.multi_module_mtp = False
    att.mooncake_kind = None
    att.mooncake = None

    assert att._classify_offload() is None
    assert att.mooncake_kind == "store"
    assert att.mooncake is store


def test_current_main_mooncake_store_admits_bounded_eagle3():
    from kvguard.adapters.vllm.attach import VllmAttachment

    store = type("MooncakeStoreScheduler", (), {"_store_group_ids": (0,), "kv_role": "kv_both"})()
    connector = type(
        "MooncakeStoreConnector",
        (),
        {"connector_scheduler": store},
    )()
    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(connector=connector)
    att.groups = ({"group": 0},)
    att.speculative = True
    att.eagle3 = True
    att.dflash = False
    att.standalone_draft = False
    att.multi_module_mtp = False
    att.mooncake_kind = None
    att.mooncake = None

    assert att._classify_offload() is None
    assert att.mooncake_kind == "store"
    assert att.mooncake is store


def test_mooncake_native_buffers_decode_stride_separately_from_valid_bytes():
    from kvguard.adapters.vllm.attach import VllmAttachment

    group0 = SimpleNamespace(
        store_layout=SimpleNamespace(
            kv_caches_base_addr=(1000, 2000),
            block_stride=(100, 100),
            block_len=(80, 80),
        )
    )
    group1 = SimpleNamespace(
        store_layout=SimpleNamespace(
            kv_caches_base_addr=(3000,),
            block_stride=(64,),
            block_len=(20,),
        )
    )
    att = object.__new__(VllmAttachment)
    att.pool = SimpleNamespace(blocks=range(8))
    worker = SimpleNamespace(token_dbs=(group0, group1))

    assert att._mooncake_decode_buffers(
        worker,
        ("model@group:0@hash", "model@group:1@hash"),
        ((1200, 2200), (3192,)),
        ((180, 180), (20,)),
    ) == ((0, (2, 3)), (1, (3,)))


def test_mooncake_completed_native_reads_and_writes_reach_the_core():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment, state_pool

    calls = []
    ledger = SimpleNamespace(
        external_read=lambda value, site: calls.append(("read", value, site)),
        external_write=lambda value, site: calls.append(("write", value, site)),
        external_fill=lambda value, site: calls.append(("fill", value, site)),
        complete=lambda did: calls.append(("complete", did)),
        free=lambda *args, **kwargs: calls.append(("free", args, kwargs)),
    )
    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_store_group_ids=(0, 1))
    att.pool_of_group = {0: POOL, 1: state_pool(1)}
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(block_size=16), state_pool(1): SimpleNamespace(block_size=1)}
    )
    att.ledger = ledger
    att._mooncake_events = deque(
        [
            {
                "kind": "read_start",
                "token": "put:1",
                "request": "a",
                "job": 3,
                "items": ((0, (4,)), (1, (5,))),
            },
            {
                "kind": "read_complete",
                "token": "put:1",
                "request": "a",
                "items": ((0, (4,)), (1, (5,))),
                "successful": ((0, (4,)), (1, (5,))),
            },
            {
                "kind": "write_start",
                "token": "get:2",
                "request": "b",
                "job": None,
                "items": ((0, (7,)),),
            },
            {
                "kind": "write_complete",
                "token": "get:2",
                "request": "b",
                "items": ((0, (7,)),),
                "successful": ((0, (7,)),),
            },
        ]
    )
    att.mooncake_fault = ""
    att.mooncake_fault_done = False
    att.stats = Counter()
    att.seq = 9

    att._drain_mooncake_events()
    transfers = [
        (call[0], call[1].pool, call[1].blocks, tuple(call[1].offsets))
        for call in calls
        if call[0] in ("read", "write")
    ]
    assert transfers == [
        ("read", POOL, (4,), tuple(range(16))),
        ("read", state_pool(1), (5,), (0,)),
        ("write", POOL, (7,), tuple(range(16))),
    ]
    assert [(call[1].pool, call[1].block) for call in calls if call[0] == "fill"] == [(POOL, 7)]
    assert not att._mooncake_events


def test_mooncake_eagle_transfers_route_to_target_and_draft_pools():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    calls = []
    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_store_group_ids=(0,))
    att.pool_of_group = {0: POOL}
    att.eagle3 = True
    att.standalone_draft = False
    att.dflash = False
    att.mtp_shared = False
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            DRAFT_POOL: SimpleNamespace(block_size=16),
        }
    )
    att.ledger = SimpleNamespace(
        external_read=lambda value, site: calls.append(("read", value, site)),
        external_write=lambda value, site: calls.append(("write", value, site)),
        external_fill=lambda value, site: calls.append(("fill", value, site)),
        complete=lambda did: calls.append(("complete", did)),
        free=lambda *args, **kwargs: None,
    )
    att._mooncake_events = deque(
        [
            {
                "kind": "write_start",
                "token": "get:1",
                "request": "r",
                "items": ((0, (7,)),),
            },
            {
                "kind": "write_complete",
                "token": "get:1",
                "request": "r",
                "items": ((0, (7,)),),
                "successful": ((0, (7,)),),
            },
        ]
    )
    att.mooncake_fault = ""
    att.mooncake_fault_done = False
    att.stats = Counter()
    att.seq = 4

    att._drain_mooncake_events()
    assert [(call[0], call[1].pool) for call in calls if call[0] == "write"] == [
        ("write", POOL),
        ("write", DRAFT_POOL),
    ]
    assert [(call[0], call[1].pool) for call in calls if call[0] == "fill"] == [
        ("fill", POOL),
        ("fill", DRAFT_POOL),
    ]
    assert [call[1] for call in calls if call[0] == "complete"] == [
        "mooncake-get:1:gpu",
        "mooncake-get:1:draft_kv",
    ]


def test_mooncake_pending_load_hides_optimistic_external_frontier():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment, state_pool

    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_store_group_ids=(0, 1))
    att.managers = (SimpleNamespace(block_size=544), SimpleNamespace(block_size=544))
    att.pool_of_group = {0: POOL, 1: state_pool(1)}
    att._mooncake_pending = {}
    att._offload_pending_by_req = {}
    att.stats = Counter()
    load = SimpleNamespace(can_load=True, vllm_cached_tokens=0, kvpool_cached_tokens=544)
    req = SimpleNamespace(req_id="r", load_spec=load, block_ids=([3], [9]))
    metadata = type("MooncakeStoreConnectorMetadata", (), {"requests": [req]})()

    att._mooncake_loads(SimpleNamespace(kv_connector_metadata=metadata))
    assert att._offload_computed("r", 544) == 0
    assert att._mooncake_pending == {"r": (0, 544, {POOL: 3, state_pool(1): 9})}


def test_mooncake_transfer_pins_are_not_counted_as_request_mappings():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment, state_pool

    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_pinned_saves={4: ([3, 9], 1)})
    att._mooncake_store_owners = {(POOL, 3): "done", (state_pool(1), 9): "live"}
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16, rewritable=False),
            state_pool(1): SimpleNamespace(block_size=1, rewritable=True),
        }
    )
    att.stats = Counter()
    snap = Snapshot(
        2,
        (RequestSnapshot("live", state_pool(1), (9,), 1, 1),),
        (
            BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(state_pool(1), 9, BlockState.PUBLISHED, refs=2),
        ),
    )

    normalized = att._normalize_mooncake_reservations(snap)
    blocks = {(block.pool, block.block): block for block in normalized.blocks}
    assert blocks[(POOL, 3)].refs == 1 and blocks[(POOL, 3)].state is BlockState.PUBLISHED
    assert blocks[(state_pool(1), 9)].refs == 1
    assert blocks[(state_pool(1), 9)].state is BlockState.PRIVATE
    assert ("done", POOL, (3,)) in [
        (row.request, row.pool, row.blocks) for row in normalized.requests
    ]


def test_mooncake_pin_cannot_republish_partial_eagle_tail():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_pinned_saves={4: ([3], 1)})
    att._mooncake_store_owners = {(POOL, 3): "r", (DRAFT_POOL, 3): "r"}
    att._published_blocks = {(POOL, 3), (DRAFT_POOL, 3)}
    att._offload_filled = set()
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16, rewritable=False),
            DRAFT_POOL: SimpleNamespace(block_size=16, rewritable=False),
        }
    )
    att.group_of = {}
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(3)]
        + [SimpleNamespace(block_hash=b"hash")]
    )
    att.stats = Counter()
    snap = Snapshot(
        2,
        (
            RequestSnapshot("r", POOL, (3,), 8, 8),
            RequestSnapshot("r", DRAFT_POOL, (3,), 11, 8),
        ),
        (
            BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=2),
            BlockSnapshot(DRAFT_POOL, 3, BlockState.PUBLISHED, refs=2),
        ),
    )

    normalized = att._normalize_mooncake_reservations(snap)
    normalized = att._normalize_publications(normalized)

    assert all(block.refs == 1 for block in normalized.blocks)
    assert all(block.state is BlockState.PRIVATE for block in normalized.blocks)
    assert not att._published_blocks

    # The physical target hash remains after Store drops its transient pin. It still cannot
    # publish the draft tail whose independent logical frontier has not filled this block.
    att.mooncake._pinned_saves = {}
    att._published_blocks = {(POOL, 3), (DRAFT_POOL, 3)}
    unpinned = Snapshot(
        snap.seq,
        snap.requests,
        blocks=(
            BlockSnapshot(POOL, 3, BlockState.PRIVATE, refs=1),
            BlockSnapshot(DRAFT_POOL, 3, BlockState.PUBLISHED, refs=1),
        ),
    )
    unpinned = att._normalize_mooncake_reservations(unpinned)
    unpinned = att._normalize_publications(unpinned)
    by_pool = {block.pool: block for block in unpinned.blocks}
    assert by_pool[POOL].state is BlockState.PUBLISHED
    assert by_pool[DRAFT_POOL].state is BlockState.PRIVATE
    assert att._published_blocks == {(POOL, 3)}

    # Once the last request releases a complete cached generation, its zero-mapper entry stays
    # published; only an exactly-one-mapper draft tail is writable/private.
    att._published_blocks = {(DRAFT_POOL, 3)}
    cached = Snapshot(
        snap.seq + 1,
        (),
        (BlockSnapshot(DRAFT_POOL, 3, BlockState.PUBLISHED, refs=0, reserved=True),),
    )
    cached = att._normalize_publications(cached)
    assert cached.blocks[0].state is BlockState.PUBLISHED
    assert att._published_blocks == {(DRAFT_POOL, 3)}


def test_publication_normalization_over_a_change_listing():
    """A snapshot that lists the pool as its change is normalized over the change: the listed
    (new or changed) objects, plus the kept blocks whose external facts changed, which join the
    change re-normalized from their cached source; the blocks gone leave the cache and the
    published set, and the rest of the cache is kept."""
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att._published_blocks = set()
    att._offload_filled = set()
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(block_size=16, rewritable=False)}
    )
    att.group_of = {}
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=b"h", block_hash_num_tokens=16) for _ in range(6)]
    )
    att.stats = Counter()
    full = Snapshot(
        1,
        (RequestSnapshot("r", POOL, (3, 4), 32, 32),),
        (
            BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=2),
            BlockSnapshot(POOL, 4, BlockState.PRIVATE, refs=1),
            BlockSnapshot(POOL, 5, BlockState.PRIVATE, refs=1),
        ),
    )
    first = att._normalize_publications(full)
    assert {b.block: b.state for b in first.blocks} == {
        3: BlockState.PUBLISHED,
        4: BlockState.PRIVATE,
        5: BlockState.PRIVATE,
    }
    assert att._published_blocks == {(POOL, 3)} and set(att._normalized_cache) == {
        (POOL, 3),
        (POOL, 4),
        (POOL, 5),
    }
    # The change: 4's refs dropped to 0 (now cached: published, reserved), 5 is gone.
    delta = Snapshot(
        2,
        full.requests,
        (BlockSnapshot(POOL, 4, BlockState.PUBLISHED, refs=0, reserved=True),),
        delta_pools=frozenset({POOL}),
        gone={POOL: frozenset({5})},
    )
    second = att._normalize_publications(delta)
    assert [(b.block, b.state) for b in second.blocks] == [(4, BlockState.PUBLISHED)]
    assert second.delta_pools == frozenset({POOL}) and second.gone == {POOL: frozenset({5})}
    assert att._published_blocks == {(POOL, 3), (POOL, 4)}
    assert set(att._normalized_cache) == {(POOL, 3), (POOL, 4)}, "5 left the cache, 3 is kept"
    # An external fact changes for the kept block 3 (an offload fill of it): it is
    # re-normalized from its cached source and joins the change although the cache did not
    # list it.
    att._offload_filled = {3}
    third = att._normalize_publications(
        Snapshot(3, full.requests, (), delta_pools=frozenset({POOL}))
    )
    assert [b.block for b in third.blocks] == [3] and third.blocks[0].state is BlockState.PUBLISHED
    assert (
        att._normalize_publications(
            Snapshot(4, full.requests, (), delta_pools=frozenset({POOL}))
        ).blocks
        == ()
    )


def test_mooncake_beam_store_source_keeps_its_sparse_logical_positions():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    source = RequestSnapshot("beam", POOL, (4, None, 5), 48, 48)
    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_pinned_saves={7: ([4, 5], 1)})
    att._mooncake_store_owners = {(POOL, 4): "beam", (POOL, 5): "beam"}
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(block_size=16, rewritable=False)},
        requests={("beam", POOL): SimpleNamespace(snapshot=source)},
    )
    att.stats = Counter()
    snap = Snapshot(
        3,
        (RequestSnapshot("beam", POOL, (8, 9, 10), 48, 32),),
        (
            BlockSnapshot(POOL, 4, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(POOL, 5, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(POOL, 8, BlockState.PRIVATE, refs=1),
            BlockSnapshot(POOL, 9, BlockState.PRIVATE, refs=1),
            BlockSnapshot(POOL, 10, BlockState.PRIVATE, refs=1),
        ),
    )

    normalized = att._normalize_mooncake_reservations(snap)
    store_rows = [
        row
        for row in normalized.requests
        if row.request.startswith("__kvguard_mooncake_store__:beam:gpu:")
    ]
    assert len(store_rows) == 1
    assert store_rows[0].blocks == (4, None, 5)
    assert store_rows[0].num_tokens == 48
    assert store_rows[0].num_computed == 48


def test_mooncake_sliding_store_source_keeps_retired_pinned_blocks():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    source = RequestSnapshot("window", POOL, (4, 5, 6, 7), 64, 64, first_token=48)
    att = object.__new__(VllmAttachment)
    att.mooncake_kind = "store"
    att.mooncake = SimpleNamespace(_pinned_saves={8: ([4, 5], 1)})
    att._mooncake_store_owners = {(POOL, 4): "window", (POOL, 5): "window"}
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(block_size=16, rewritable=False)},
        requests={("window", POOL): SimpleNamespace(snapshot=source)},
    )
    att.stats = Counter()
    snap = Snapshot(
        4,
        (),
        (
            BlockSnapshot(POOL, 4, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(POOL, 5, BlockState.PUBLISHED, refs=1),
        ),
    )

    normalized = att._normalize_mooncake_reservations(snap)

    assert len(normalized.requests) == 1
    assert normalized.requests[0].request.startswith("__kvguard_mooncake_store__:window:gpu:")
    assert normalized.requests[0].blocks == (4, 5, None, None)
    assert normalized.requests[0].first_token == 0
    assert normalized.requests[0].num_tokens == 32
    assert normalized.requests[0].num_computed == 32

    first_reader = normalized.requests[0].request
    att.mooncake._pinned_saves = {9: ([4], 1)}
    shrunk = att._normalize_mooncake_reservations(
        Snapshot(5, (), (BlockSnapshot(POOL, 4, BlockState.PUBLISHED, refs=1),))
    )
    assert len(shrunk.requests) == 1
    assert shrunk.requests[0].request != first_reader
    assert shrunk.requests[0].blocks == (4, None, None, None)
    assert shrunk.requests[0].num_tokens == 16
    assert shrunk.requests[0].num_computed == 16


def test_current_main_nixl_pull_credits_only_exact_completed_prefix():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    fills = []
    request = SimpleNamespace(num_computed_tokens=34, num_tokens=34)
    req_meta = SimpleNamespace(
        awaiting_kvs=True,
        remote=SimpleNamespace(num_tokens=34),
        local_block_ids=([3, 5, 7],),
        local_num_computed_blocks=(0,),
    )
    att = object.__new__(VllmAttachment)
    att.pd_kind = "nixl_pull"
    att.block_size = 16
    att.scheduler = SimpleNamespace(requests={"r": request})
    att._pd_pending = {}
    att._offload_pending_by_req = {}
    att._offload_recompute_tails = {}
    att.stats = Counter()
    att.seq = 4
    att.ledger = SimpleNamespace(
        external_fill=lambda fill, site: fills.append((fill, site)),
        complete=lambda _did: None,
    )
    att._pd_send_dispatches = {}
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=type("NixlConnectorMetadata", (), {"reqs_to_recv": {"r": req_meta}})()
    )

    att._pd_loads(scheduler_output)
    assert att._offload_computed("r", 34) == 0
    assert fills == []

    model_output = SimpleNamespace(
        kv_connector_output=SimpleNamespace(
            finished_recving={"r"}, failed_recving=set(), finished_sending=set()
        )
    )
    att._pd_completions(model_output)
    assert [(fill.pool, fill.block, tuple(fill.offsets)) for fill, _ in fills] == [
        (POOL, 3, tuple(range(16))),
        (POOL, 5, tuple(range(16))),
        (POOL, 7, (0, 1)),
    ]
    assert not att._pd_pending
    assert att._offload_recompute_tails == {"r": (33, {POOL: 7}, False)}


def test_current_main_nixl_pull_source_read_lives_until_finished_sending():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    reads = []
    completions = []
    att = object.__new__(VllmAttachment)
    att.pd_kind = "nixl_pull"
    att.block_size = 16
    att._pd_send_dispatches = {}
    att._pd_send_seq = 0
    att._pd_pending = {}
    att._zeroed_blocks = {(POOL, 9)}
    att.stats = Counter()
    att.seq = 6
    att.ledger = SimpleNamespace(
        external_read=lambda read, site: reads.append((read, site)),
        complete=completions.append,
    )

    att._open_pd_send("producer", ([9, 11],), 24, ([9, 11],))
    assert len(reads) == 2
    full, full_site = reads[0]
    tail, tail_site = reads[1]
    assert full.blocks == (9,)
    assert tuple(full.offsets) == tuple(range(16))
    assert full_site == "nixl-send:1:full@6"
    assert tail.blocks == (11,)
    assert tuple(tail.offsets) == tuple(range(8))
    assert tail_site == "nixl-send:1:tail@6"

    att._pd_completions(
        SimpleNamespace(
            kv_connector_output=SimpleNamespace(
                finished_recving=set(),
                failed_recving=set(),
                finished_sending={"producer"},
            )
        )
    )
    assert completions == ["nixl-send:1:full", "nixl-send:1:tail"]
    assert not att._pd_send_dispatches


def test_current_main_nixl_pull_pairs_source_suffix_digests(tmp_path):
    import json

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    att = object.__new__(VllmAttachment)
    att._pd_content_dir = tmp_path
    att._pd_content_expected = {}
    att.block_size = 16
    att.reporter = Reporter(fail_fast=False)
    att.stats = Counter()
    att.seq = 8
    remote = SimpleNamespace(request_id="source", block_ids=([4, 5, 6],))
    path = att._pd_manifest_path("source")
    assert path is not None
    path.write_text(
        json.dumps(
            {
                "request": "source",
                "num_tokens": 40,
                "block_size": 16,
                "blocks": [4, 5, 6],
                "digests": ["a", "b", "c"],
            }
        )
    )

    att._pd_expect_content("decoder", remote, (20, 21), 40)

    assert att._pd_content_expected == {"decoder": ((20, "b"), (21, "c"))}
    assert not att.reporter.reports


def test_current_main_nixl_pull_reports_a_source_plan_mismatch(tmp_path):
    import json

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    att = object.__new__(VllmAttachment)
    att._pd_content_dir = tmp_path
    att._pd_content_expected = {}
    att.block_size = 16
    att.reporter = Reporter(fail_fast=False)
    att.stats = Counter()
    att.seq = 9
    remote = SimpleNamespace(request_id="source", block_ids=([4, 7],))
    path = att._pd_manifest_path("source")
    assert path is not None
    path.write_text(
        json.dumps(
            {
                "request": "source",
                "num_tokens": 32,
                "block_size": 16,
                "blocks": [4, 6],
                "digests": ["a", "b"],
            }
        )
    )

    att._pd_expect_content("decoder", remote, (20, 21), 32)

    assert [report.kind for report in att.reporter.reports] == ["mapping_mismatch"]
    assert not att._pd_content_expected


def test_current_main_async_external_load_reports_permanent_capacity_deadlock():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    manager = SimpleNamespace(allocate_slots=lambda *_args, **_kwargs: None)
    pool = SimpleNamespace(num_gpu_blocks=16, get_num_free_blocks=lambda: 15)
    att = object.__new__(VllmAttachment)
    att.kvm = manager
    att.pool = pool
    att.stats = Counter()
    att.reporter = Reporter(fail_fast=False)
    att.seq = 9

    att._install_external_load_capacity_guard()
    request = SimpleNamespace(request_id="remote")
    for _ in range(2):
        assert (
            manager.allocate_slots(
                request,
                0,
                num_external_computed_tokens=224,
                delay_cache_blocks=True,
                reserved_blocks=1,
                has_scheduled_reqs=False,
            )
            is None
        )

    assert [report.kind for report in att.reporter.reports] == ["external_load_capacity_deadlock"]
    assert att.reporter.reports[0].policy is None
    assert att.reporter.reports[0].request == "remote"
    assert att.stats["external_load_capacity_deadlocks"] == 1


def test_current_main_external_load_capacity_guard_allows_transient_pressure():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    manager = SimpleNamespace(allocate_slots=lambda *_args, **_kwargs: None)
    pool = SimpleNamespace(num_gpu_blocks=16, get_num_free_blocks=lambda: 14)
    att = object.__new__(VllmAttachment)
    att.kvm = manager
    att.pool = pool
    att.stats = Counter()
    att.reporter = Reporter(fail_fast=False)
    att.seq = 9

    att._install_external_load_capacity_guard()
    assert (
        manager.allocate_slots(
            SimpleNamespace(request_id="remote"),
            0,
            num_external_computed_tokens=224,
            delay_cache_blocks=True,
            reserved_blocks=1,
            has_scheduled_reqs=False,
        )
        is None
    )

    assert att.reporter.reports == []
    assert att.stats["external_load_capacity_deadlocks"] == 0


def test_current_main_tiering_observer_preserves_results_and_counts_work():
    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    finished = [
        SimpleNamespace(job_id=11, success=True, deferred_release=object()),
        SimpleNamespace(job_id=12, success=False, deferred_release=None),
    ]

    class FileSystemTierManager:
        def submit_store(self, job):
            return ("store", job)

        def submit_load(self, job):
            return ("load", job)

        def get_finished_jobs(self):
            return list(finished)

    tier = FileSystemTierManager()
    att = object.__new__(VllmAttachment)
    manager = SimpleNamespace(
        secondary_tiers=(tier,),
        primary_tier=SimpleNamespace(_num_chunks=4),
        prepare_load=lambda keys, ctx: (keys, ctx),
        on_request_finished=lambda ctx: ctx,
    )
    att.offload = SimpleNamespace(manager=manager, _req_status={})
    att.groups = ({"kind": "attention"},)
    att.stats = Counter()
    att.reporter = Reporter(fail_fast=False)
    att.seq = 0
    att._install_tiering_observer()

    store = SimpleNamespace(keys=(1, 2, 3))
    load = SimpleNamespace(keys=(4, 5))
    assert tier.submit_store(store) == ("store", store)
    assert tier.submit_load(load) == ("load", load)
    assert tier.get_finished_jobs() == finished
    assert att.stats == Counter(
        {
            "tiering_secondary_0_FileSystemTierManager": 1,
            "tiering_FileSystemTierManager_store_jobs": 1,
            "tiering_FileSystemTierManager_store_chunks": 3,
            "tiering_FileSystemTierManager_load_jobs": 1,
            "tiering_FileSystemTierManager_load_chunks": 2,
            "tiering_FileSystemTierManager_success_jobs": 1,
            "tiering_FileSystemTierManager_failure_jobs": 1,
            "tiering_FileSystemTierManager_deferred_release_jobs": 1,
        }
    )


def test_current_main_tiering_observer_reports_closed_primary_capacity_cycle():
    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    finished = []

    class KVCRSecondaryTierManager:
        def submit_store(self, job):
            return job

        def submit_load(self, job):
            return job

        def get_finished_jobs(self):
            out = list(finished)
            finished.clear()
            return out

    tier = KVCRSecondaryTierManager()
    policy = SimpleNamespace(chunks={})
    manager = SimpleNamespace(
        secondary_tiers=(tier,),
        primary_tier=SimpleNamespace(_num_chunks=2, _policy=policy),
        prepare_load=lambda keys, ctx: (keys, ctx),
        on_request_finished=lambda ctx: ctx,
    )
    att = object.__new__(VllmAttachment)
    att.offload = SimpleNamespace(manager=manager, _req_status={"remote": object()})
    att.groups = ({"kind": "attention"},)
    att.stats = Counter()
    att.reporter = Reporter(fail_fast=False)
    att.seq = 17
    att._install_tiering_observer()

    ctx = SimpleNamespace(req_id="remote")
    jobs = (
        ((b"a", b"b"), {b"a": -1, b"b": -1}),
        ((b"c",), {b"b": 0, b"c": -1}),
        ((b"c",), {b"b": 0, b"c": -1}),
    )
    for job_id, (keys, primary_state) in enumerate(jobs, start=1):
        policy.chunks = {
            key: SimpleNamespace(ref_cnt=ref_cnt) for key, ref_cnt in primary_state.items()
        }
        job = SimpleNamespace(
            job_id=job_id,
            keys=keys,
            is_promotion=True,
            req_context=ctx,
        )
        tier.submit_load(job)
        finished.append(
            SimpleNamespace(
                job_id=job_id,
                success=True,
                successful_keys=None,
                deferred_release=None,
            )
        )
        tier.get_finished_jobs()

    assert [report.kind for report in att.reporter.reports] == ["tiering_promotion_capacity_cycle"]
    assert att.reporter.reports[0].request == "remote"
    assert att.reporter.reports[0].policy is None
    assert att.stats["tiering_promotion_capacity_cycles"] == 1


def test_current_main_tiering_cycle_guard_resets_after_gpu_load():
    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import Reporter

    finished = []

    class FileSystemTierManager:
        def submit_store(self, job):
            return job

        def submit_load(self, job):
            return job

        def get_finished_jobs(self):
            out = list(finished)
            finished.clear()
            return out

    tier = FileSystemTierManager()
    policy = SimpleNamespace(chunks={})
    manager = SimpleNamespace(
        secondary_tiers=(tier,),
        primary_tier=SimpleNamespace(_num_chunks=2, _policy=policy),
        prepare_load=lambda keys, ctx: (keys, ctx),
        on_request_finished=lambda ctx: ctx,
    )
    att = object.__new__(VllmAttachment)
    att.offload = SimpleNamespace(manager=manager, _req_status={"progressing": object()})
    att.groups = ({"kind": "attention"},)
    att.stats = Counter()
    att.reporter = Reporter(fail_fast=False)
    att.seq = 0
    att._install_tiering_observer()

    ctx = SimpleNamespace(req_id="progressing")
    for job_id, keys in enumerate(((b"a", b"b"), (b"c",)), start=1):
        job = SimpleNamespace(
            job_id=job_id,
            keys=keys,
            is_promotion=True,
            req_context=ctx,
        )
        tier.submit_load(job)
        finished.append(
            SimpleNamespace(
                job_id=job_id,
                success=True,
                successful_keys=None,
                deferred_release=None,
            )
        )
        tier.get_finished_jobs()

    manager.prepare_load((b"b", b"c"), ctx)
    job = SimpleNamespace(job_id=3, keys=(b"a",), is_promotion=True, req_context=ctx)
    tier.submit_load(job)
    finished.append(
        SimpleNamespace(
            job_id=3,
            success=True,
            successful_keys=None,
            deferred_release=None,
        )
    )
    tier.get_finished_jobs()

    assert att.reporter.reports == []
    assert att.stats["tiering_promotion_capacity_cycles"] == 0


def test_current_main_uniform_full_attention_wrapper_is_classified():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    full_spec = type("FullAttentionSpec", (), {})()
    wrapper = type(
        "UniformTypeKVCacheSpecs",
        (),
        {"kv_cache_specs": {"layer.0": full_spec, "layer.1": full_spec}},
    )()
    att = object.__new__(VllmAttachment)
    att.managers = (SimpleNamespace(block_size=16),)
    att.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(
                    kv_cache_spec=wrapper,
                    layer_names=("layer.0", "layer.1"),
                    is_eagle_group=False,
                ),
            )
        )
    )

    assert att._classify_groups() == [
        {
            "group": 0,
            "spec": "UniformTypeKVCacheSpecs[FullAttentionSpec]",
            "manager": "SimpleNamespace",
            "block_size": 16,
            "layers": 2,
            "kind": "attention",
            "pool": POOL,
        }
    ]


def test_current_main_eagle_mixed_group_remains_the_target_group():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    target_spec = type("FullAttentionSpec", (), {"is_draft": True})()
    draft_spec = type("FullAttentionSpec", (), {"is_draft": True})()
    att = object.__new__(VllmAttachment)
    att.runner = SimpleNamespace(speculator=SimpleNamespace(draft_attn_layer_names={"draft.layer"}))
    att.managers = (
        SimpleNamespace(block_size=16),
        SimpleNamespace(block_size=16),
    )
    att.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(
                    kv_cache_spec=target_spec,
                    layer_names=("target.layer", "draft.layer"),
                    is_eagle_group=True,
                ),
                SimpleNamespace(
                    kv_cache_spec=draft_spec,
                    layer_names=("draft.layer",),
                    is_eagle_group=True,
                ),
            )
        )
    )

    groups = att._classify_groups()

    assert groups[0]["kind"] == "attention"
    assert groups[0]["pool"] == POOL
    assert groups[0]["draft_layers"] == 1
    assert "draft" not in groups[0]
    assert groups[1]["draft"] is True


def test_current_main_dflash_layers_inside_target_group_are_classified():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    draft_model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: ("draft.layer",))
    att = object.__new__(VllmAttachment)
    att.runner = SimpleNamespace(speculator=SimpleNamespace(model=draft_model))
    att.managers = (SimpleNamespace(block_size=16),)
    att.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(
                    kv_cache_spec=type("FullAttentionSpec", (), {})(),
                    layer_names=("target.layer", "draft.layer"),
                    is_eagle_group=False,
                ),
            )
        )
    )

    groups = att._classify_groups()

    assert groups == [
        {
            "group": 0,
            "spec": "FullAttentionSpec",
            "manager": "SimpleNamespace",
            "block_size": 16,
            "layers": 2,
            "draft_layers": 1,
            "kind": "attention",
            "pool": POOL,
        }
    ]


def test_current_main_dflash2_split_target_and_draft_groups_are_classified():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment, attention_pool, window_pool

    target = type("FullAttentionSpec", (), {"is_draft": False})()
    draft = type(
        "SlidingWindowSpec",
        (),
        {
            "is_draft": True,
            "sliding_window": 128,
            "extra_retained_tokens": 0,
        },
    )()
    att = object.__new__(VllmAttachment)
    att.runner = SimpleNamespace(
        speculator=SimpleNamespace(draft_attn_layer_names=("draft.layer",))
    )
    att.managers = tuple(SimpleNamespace(block_size=16) for _ in range(4))
    att.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(kv_cache_spec=target, layer_names=("target.0",)),
                SimpleNamespace(kv_cache_spec=target, layer_names=("target.1",)),
                SimpleNamespace(kv_cache_spec=target, layer_names=("target.2",)),
                SimpleNamespace(kv_cache_spec=draft, layer_names=("draft.layer",)),
            )
        )
    )

    groups = att._classify_groups()

    assert [group["pool"] for group in groups] == [
        POOL,
        attention_pool(1),
        attention_pool(2),
        window_pool(3),
    ]
    assert groups[3]["draft"] is True


def test_current_main_dflash2_snapshot_uses_the_draft_groups_write_frontier():
    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att.dflash_pool = "swa8"
    att.dflash_gid = 8
    att.eagle_lookahead = 8
    att.window_of = {8: 128}
    att.extra_of = {8: 1}
    att.core_ledger = SimpleNamespace(pools={"swa8": SimpleNamespace(block_size=16)})
    att._dflash_frontiers = {"r": 180, "finished": 7}
    att._dflash_pending_writes = {"finished": {7}}
    att.stats = Counter()
    row = RequestSnapshot("r", "swa8", tuple(range(1, 15)), 200, 200, first_token=73)

    got = att._with_dflash_separate_snapshot(Snapshot(1, (row,), ()))

    assert got.requests == (
        RequestSnapshot("r", "swa8", tuple(range(1, 15)), 208, 180, first_token=52),
    )
    assert att.stats["dflash_snapshot_caps"] == 1
    assert att.stats["dflash_snapshot_capped_tokens"] == 20
    assert att._dflash_frontiers == {"r": 180}
    assert not att._dflash_pending_writes


def test_current_main_deferred_free_keeps_request_row_until_pool_release():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att.block_size = 16
    att.hybrid = False
    att.eagle3 = False
    att.core_ledger = SimpleNamespace(
        requests={}, pools={"gpu": SimpleNamespace(block_size=16, rewritable=False)}
    )
    att.reporter = SimpleNamespace(emit=lambda report: None)
    att.seq = 7
    att.stats = Counter()
    att._deferred_rows = {}
    att._deferred_counts = {}
    att._deferred_by_block = {}
    att._offload_pending_by_req = {}
    att._speculative_caps = {}
    blocks = [
        SimpleNamespace(block_id=3, is_null=False),
        SimpleNamespace(block_id=5, is_null=False),
    ]

    att._defer_request_row(SimpleNamespace(request_id="r", num_computed_tokens=17), blocks)
    row = att._deferred_rows[("r", "gpu")]
    assert row.blocks == (3, 5)
    assert (row.num_tokens, row.num_computed) == (32, 17)
    att._release_deferred_rows(reversed(blocks))
    assert not att._deferred_rows and not att._deferred_counts and not att._deferred_by_block
    assert att.stats["deferred_rows_released"] == 1


def test_current_main_deferred_free_preserves_each_hybrid_pool_row():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {3: 0, 4: 1, 5: 2}
    att.pool_of_group = {0: POOL, 1: "state1", 2: "state2"}
    att.core_ledger = SimpleNamespace(
        requests={},
        pools={
            POOL: SimpleNamespace(block_size=16, rewritable=False),
            "state1": SimpleNamespace(block_size=1, rewritable=True),
            "state2": SimpleNamespace(block_size=1, rewritable=True),
        },
    )
    att.reporter = SimpleNamespace(emit=lambda report: None)
    att.seq = 7
    att.stats = Counter()
    att._deferred_rows = {}
    att._deferred_counts = {}
    att._deferred_by_block = {}
    att._offload_pending_by_req = {}
    att._speculative_caps = {}
    blocks = [
        SimpleNamespace(block_id=3, is_null=False),
        SimpleNamespace(block_id=4, is_null=False),
        SimpleNamespace(block_id=5, is_null=False),
    ]

    att._defer_request_row(SimpleNamespace(request_id="r", num_computed_tokens=12), blocks)

    assert set(att._deferred_rows) == {("r", POOL), ("r", "state1"), ("r", "state2")}
    assert att._deferred_rows[("r", POOL)].blocks == (3,)
    assert att._deferred_rows[("r", "state1")].blocks == (4,)
    assert att._deferred_rows[("r", "state2")].blocks == (5,)
    assert att.stats["deferred_rows"] == 3
    att._release_deferred_rows(blocks)
    assert not att._deferred_rows and not att._deferred_counts and not att._deferred_by_block
    assert att.stats["deferred_rows_released"] == 3


def test_current_main_eagle_deferred_row_is_cloned_once_by_the_snapshot():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.block_size = 16
    att.eagle_lookahead = 3
    att.hybrid = False
    att.eagle3 = True
    att.core_ledger = SimpleNamespace(
        requests={}, pools={POOL: SimpleNamespace(block_size=16, rewritable=False)}
    )
    att.reporter = SimpleNamespace(emit=lambda report: None)
    att.seq = 7
    att.stats = Counter()
    att._deferred_rows = {}
    att._deferred_counts = {}
    att._deferred_by_block = {}
    att._offload_pending_by_req = {}
    att._speculative_caps = {}
    blocks = [SimpleNamespace(block_id=3, is_null=False)]

    att._defer_request_row(SimpleNamespace(request_id="r", num_computed_tokens=12), blocks)

    assert set(att._deferred_rows) == {("r", POOL)}
    snap = Snapshot(
        8,
        tuple(att._deferred_rows.values()),
        (BlockSnapshot(POOL, 3, BlockState.PRIVATE, refs=1),),
    )
    got = att._with_eagle_snapshot(snap)
    assert set(got.request_index()) == {("r", POOL), ("r", DRAFT_POOL)}


def test_current_main_async_offload_waits_for_worker_completion_before_fill():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    fills = []
    att = object.__new__(VllmAttachment)
    att.offload = SimpleNamespace(
        _req_status={"r": SimpleNamespace(num_locally_computed_tokens=16)}
    )
    att.scheduler = SimpleNamespace(requests={"r": SimpleNamespace(is_finished=lambda: True)})
    att._offload_pending = {7: ("r", (3, 5))}
    att._offload_pending_by_req = {"r": 7}
    att._offload_flushed = set()
    att._offload_store_jobs = {}
    att._offload_store_epochs = set()
    att._offload_filled = set()
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: POOL}
    att.block_size = 16
    att.seq = 9
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(pools={POOL: SimpleNamespace(block_size=16)})
    att.ledger = SimpleNamespace(external_fill=lambda fill, site: fills.append((fill, site)))

    assert att._offload_computed("r", 48) == 16
    output = SimpleNamespace(
        kv_connector_output=SimpleNamespace(
            kv_connector_worker_meta=SimpleNamespace(completed_jobs={7: 1})
        )
    )
    att._offload_completions(output)
    assert [fill.block for fill, _ in fills] == [3, 5]
    assert att._offload_filled == {3, 5}
    assert not att._offload_pending and not att._offload_pending_by_req
    assert att._offload_computed("r", 48) == 48
    assert att.stats["offload_async_loads_completed_after_abort"] == 1


def test_current_main_async_offload_full_hit_defers_the_recomputed_tail():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    fills = []
    blocks = [
        SimpleNamespace(block_id=3, is_null=False),
        SimpleNamespace(block_id=5, is_null=False),
    ]
    request = SimpleNamespace(
        num_computed_tokens=32,
        num_tokens=32,
        is_finished=lambda: False,
    )
    att = object.__new__(VllmAttachment)
    att.offload_kind = "managed"
    att.offload = SimpleNamespace()
    att.scheduler = SimpleNamespace(requests={"r": request})
    att.manager = SimpleNamespace(block_size=16, req_to_blocks={"r": blocks})
    att.managers = {0: att.manager}
    att.kv_gid = 0
    att.hybrid = False
    att.eagle3 = False
    att.dflash = True
    att.group_of = {3: 0, 5: 0}
    att.pool_of_group = {0: POOL}
    att._offload_pending = {7: ("r", (3, 5))}
    att._offload_pending_by_req = {"r": 7}
    att._offload_flushed = set()
    att._offload_store_jobs = {}
    att._managed_offload_routes = {7: {POOL: (3, 5), DRAFT_POOL: (3, 5)}}
    att._offload_filled = set()
    att._offload_filled_pools = set()
    att._offload_recompute_tails = {}
    att.seq = 9
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            DRAFT_POOL: SimpleNamespace(block_size=16),
        }
    )
    att.ledger = SimpleNamespace(external_fill=lambda fill, site: fills.append((fill, site)))
    output = SimpleNamespace(
        kv_connector_output=SimpleNamespace(
            kv_connector_worker_meta=SimpleNamespace(completed_jobs={7: 1})
        )
    )

    att._offload_completions(output)

    assert [(fill.pool, fill.block, tuple(fill.offsets)) for fill, _ in fills] == [
        (POOL, 3, tuple(range(16))),
        (POOL, 5, tuple(range(16))),
        (DRAFT_POOL, 3, tuple(range(16))),
        (DRAFT_POOL, 5, tuple(range(16))),
    ]
    assert att.stats["offload_full_hit_tail_recomputes"] == 1
    assert att._offload_recompute_tails == {"r": (31, {POOL: 5, DRAFT_POOL: 5}, False)}
    assert not att._offload_pending and not att._offload_pending_by_req


def test_current_main_async_offload_recompute_tail_requires_one_mapper():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.scheduler = SimpleNamespace(requests={"r": SimpleNamespace(is_finished=lambda: False)})
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            DRAFT_POOL: SimpleNamespace(block_size=16),
        }
    )
    att._offload_recompute_tails = {"r": (31, {POOL: 5, DRAFT_POOL: 5}, False)}
    att.stats = Counter()
    private = Snapshot(
        1,
        (
            RequestSnapshot("r", POOL, (3, 5), 31, 31),
            RequestSnapshot("r", DRAFT_POOL, (3, 5), 31, 31),
        ),
        (
            BlockSnapshot(POOL, 5, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(DRAFT_POOL, 5, BlockState.PUBLISHED, refs=1),
        ),
    )

    normalized, invalidations = att._normalize_offload_recompute_tails(private)

    assert all(block.state is BlockState.PRIVATE for block in normalized.blocks)
    assert [row.num_tokens for row in normalized.requests] == [32, 32]
    assert invalidations == ((POOL, "r", 31), (DRAFT_POOL, "r", 31))
    assert att._offload_recompute_tails["r"][2]

    shared = Snapshot(
        2,
        (
            *private.requests,
            RequestSnapshot("peer", POOL, (3, 5), 32, 32),
            RequestSnapshot("peer", DRAFT_POOL, (3, 5), 32, 32),
        ),
        (
            BlockSnapshot(POOL, 5, BlockState.PUBLISHED, refs=2),
            BlockSnapshot(DRAFT_POOL, 5, BlockState.PUBLISHED, refs=2),
        ),
    )
    normalized, invalidations = att._normalize_offload_recompute_tails(shared)

    assert all(block.state is BlockState.PUBLISHED for block in normalized.blocks)
    assert invalidations == ()
    assert not att._offload_recompute_tails
    assert att.stats["offload_full_hit_tail_recompute_rejected"] == 1


def test_current_main_simple_cpu_offload_credits_exact_completed_event():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    fills = []
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: POOL}
    blocks = [
        SimpleNamespace(block_id=3, is_null=False),
        SimpleNamespace(block_id=5, is_null=False),
    ]
    att.offload = SimpleNamespace(
        _reqs_to_load={
            "r": SimpleNamespace(
                request=SimpleNamespace(num_computed_tokens=24),
                transfer_meta=SimpleNamespace(gpu_block_ids=[3, 5]),
            )
        }
    )
    att.scheduler = SimpleNamespace(requests={"r": SimpleNamespace(is_finished=lambda: False)})
    att._simple_pending_loads = {}
    att._offload_pending_by_req = {}
    att._offload_store_jobs = {}
    att._offload_filled = set()
    att.kv_gid = 0
    att.manager = SimpleNamespace(block_size=16, req_to_blocks={"r": blocks})
    att.block_size = 16
    att.seq = 9
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(pools={POOL: SimpleNamespace(block_size=16)})
    att.ledger = SimpleNamespace(external_fill=lambda fill, site: fills.append((fill, site)))
    metadata = type(
        "SimpleCPUOffloadMetadata",
        (),
        {
            "load_event": 7,
            "load_gpu_blocks": [3, 5],
            "load_event_to_reqs": {7: ["r"]},
            "store_event": 8,
            "store_gpu_blocks": [9],
        },
    )()

    att._simple_offload_loads(metadata)
    assert att._offload_computed("r", 48) == 0
    assert not fills
    assert att._offload_store_jobs == {8: (None, (9,))}

    att._complete_simple_offload_load(7)
    assert [fill.block for fill, _ in fills] == [3, 5]
    assert att._offload_filled == {3, 5}
    assert not att._simple_pending_loads and not att._offload_pending_by_req
    assert att._offload_computed("r", 48) == 48


def test_current_main_simple_cpu_offload_credits_completion_after_abort():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    fills = []
    att = object.__new__(VllmAttachment)
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: POOL}
    att.scheduler = SimpleNamespace(requests={})
    att._simple_pending_loads = {7: (("aborted", (3, 5), 24),)}
    att._offload_pending_by_req = {"aborted": 7}
    att._offload_filled = set()
    att.kv_gid = 0
    att.manager = SimpleNamespace(
        block_size=16,
        req_to_blocks={
            "aborted": [
                SimpleNamespace(block_id=3, is_null=False),
                SimpleNamespace(block_id=5, is_null=False),
            ]
        },
    )
    att.seq = 9
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(pools={POOL: SimpleNamespace(block_size=16)})
    att.ledger = SimpleNamespace(external_fill=lambda fill, site: fills.append((fill, site)))

    att._complete_simple_offload_load(7)

    assert [fill.block for fill, _ in fills] == [3, 5]
    assert att._offload_filled == {3, 5}
    assert not att._simple_pending_loads and not att._offload_pending_by_req
    assert att.stats["offload_async_loads_completed_after_abort"] == 1
    assert att.stats["offload_async_load_blocks_completed_after_abort"] == 2
    assert att.stats["offload_async_loads_completed"] == 1


def test_current_main_simple_cpu_transfer_pins_are_not_request_references():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: POOL}
    att.offload = SimpleNamespace(
        _reqs_to_load={
            "loading": SimpleNamespace(transfer_meta=SimpleNamespace(gpu_block_ids=[5]))
        },
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks=set(),
        _store_event_to_blocks={11: SimpleNamespace(gpu_block_ids=[3, 4])},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[SimpleNamespace(gpu_block_ids=[6])],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(6)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(rewritable=False, block_size=16)}
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (RequestSnapshot("active", POOL, (3, 5), 16, 0),),
        (
            BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=2),
            BlockSnapshot(POOL, 4, BlockState.PRIVATE, refs=1),
            BlockSnapshot(POOL, 5, BlockState.PRIVATE, refs=2),
        ),
    )

    normalized = att._normalize_offload_reservations(snap)
    got = normalized.block_index()
    assert got[(POOL, 3)].refs == 1 and not got[(POOL, 3)].reserved
    assert got[(POOL, 4)].refs == 1 and not got[(POOL, 4)].reserved
    assert got[(POOL, 4)].state is BlockState.PUBLISHED
    assert got[(POOL, 5)].refs == 1 and got[(POOL, 5)].state is BlockState.PRIVATE
    assert got[(POOL, 6)].refs == 1 and not got[(POOL, 6)].reserved
    reservations = {request.request: request for request in normalized.requests}
    assert reservations[f"__kvguard_simple_store__:{POOL}:4"].blocks == (4,)
    assert reservations[f"__kvguard_simple_store__:{POOL}:6"].blocks == (6,)


def test_current_main_simple_cpu_store_keeps_recurrent_state_private_and_rewritable():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {3: 0, 4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={3, 4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(3)]
        + [SimpleNamespace(ref_cnt=1), SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(rewritable=False, block_size=16),
            state_pool: SimpleNamespace(rewritable=True, block_size=1),
        }
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (),
        (
            BlockSnapshot(POOL, 3, BlockState.PRIVATE, refs=1),
            # The unnormalized cache sees ref_cnt=2 (owner + connector pin) and mistakes the
            # hashed state block for a shared/published block. Normalization must undo it.
            BlockSnapshot(state_pool, 4, BlockState.PUBLISHED, refs=1),
        ),
    )

    normalized = att._normalize_offload_reservations(snap)
    got = normalized.block_index()
    assert got[(POOL, 3)].state is BlockState.PUBLISHED
    assert got[(state_pool, 4)].state is BlockState.PRIVATE
    assert got[(POOL, 3)].refs == got[(state_pool, 4)].refs == 1
    reservations = {request.request: request for request in normalized.requests}
    assert reservations[f"__kvguard_simple_store__:{POOL}:3"].blocks == (3,)
    assert reservations[f"__kvguard_simple_store__:{state_pool}:4"].blocks == (4,)


def test_current_main_simple_cpu_store_does_not_duplicate_a_deferred_state_owner():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(4)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={state_pool: SimpleNamespace(rewritable=True, block_size=1)}
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (RequestSnapshot("deferred", state_pool, (4,), 1, 0),),
        (BlockSnapshot(state_pool, 4, BlockState.PRIVATE, refs=1),),
    )

    normalized = att._normalize_offload_reservations(snap)

    assert normalized.requests == snap.requests
    assert normalized.block_index()[(state_pool, 4)].refs == 1
    assert normalized.block_index()[(state_pool, 4)].state is BlockState.PRIVATE


def test_current_main_simple_cpu_store_retains_prior_shadow_owner():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(4)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={state_pool: SimpleNamespace(rewritable=True)},
        block=lambda _pool, _block: SimpleNamespace(owner="producer"),
        requests={
            ("producer", state_pool): SimpleNamespace(
                snapshot=RequestSnapshot("producer", state_pool, (2, 3, 4), 3, 3)
            )
        },
    )
    att._simple_store_owners = {}
    att.stats = Counter()
    snap = Snapshot(
        1,
        (),
        (BlockSnapshot(state_pool, 4, BlockState.PRIVATE, refs=1),),
    )

    normalized = att._normalize_offload_reservations(snap)

    assert normalized.requests == (RequestSnapshot("producer", state_pool, (None, None, 4), 3, 3),)
    assert att.stats["simple_offload_shadow_owner_blocks"] == 1
    assert att.stats["simple_offload_source_rows_retained"] == 1


def test_current_main_simple_cpu_sparse_source_row_preserves_initialization():
    from dataclasses import replace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment
    from kvguard.core import ExternalFill, Ledger, PoolGeometry, Reporter

    state_pool = "state0"
    reporter = Reporter()
    ledger = Ledger(reporter)
    ledger.register_pool(PoolGeometry(state_pool, 5, 1, rewritable=True))
    ledger.allocate(state_pool, (2, 3, 4))
    source = RequestSnapshot("producer", state_pool, (2, 3, 4), 3, 0)
    block_snapshots = tuple(
        BlockSnapshot(state_pool, block, BlockState.PRIVATE, refs=1) for block in source.blocks
    )
    ledger.observe(
        Snapshot(
            1,
            (source,),
            block_snapshots,
        )
    )
    for block in source.blocks:
        ledger.external_fill(ExternalFill(state_pool, block, range(1), request="producer"))
    source = replace(source, num_computed=3)
    ledger.observe(Snapshot(2, (source,), block_snapshots))
    ledger.free(state_pool, (2, 3))

    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(4)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = ledger
    att._simple_store_owners = {}
    att.stats = Counter()

    normalized = att._normalize_offload_reservations(
        Snapshot(3, (), (BlockSnapshot(state_pool, 4, BlockState.PRIVATE, refs=1),))
    )
    ledger.observe(normalized)

    assert normalized.requests == (RequestSnapshot("producer", state_pool, (None, None, 4), 3, 3),)
    assert ledger.block(state_pool, 4).is_filled(0)
    assert reporter.reports == []


def test_current_main_simple_cpu_sparse_attention_source_marks_released_prefix_dead():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment
    from kvguard.core import ExternalFill, Ledger, PoolGeometry, Reporter

    source = RequestSnapshot("producer", POOL, (1, 2), 256, 256)
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: POOL}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={2},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(2)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(rewritable=False, block_size=128)},
        requests={("producer", POOL): SimpleNamespace(snapshot=source)},
    )
    att._simple_store_owners = {(POOL, 2): "producer"}
    att.stats = Counter()

    normalized = att._normalize_offload_reservations(
        Snapshot(2, (), (BlockSnapshot(POOL, 2, BlockState.PUBLISHED, refs=1),))
    )

    assert normalized.requests == (
        RequestSnapshot("producer", POOL, (None, 2), 256, 256, first_token=128),
    )
    reporter = Reporter()
    ledger = Ledger(reporter)
    ledger.register_pool(PoolGeometry(POOL, 3, 128))
    ledger.allocate(POOL, (2,))
    ledger.observe(
        Snapshot(
            1,
            (RequestSnapshot("producer", POOL, (None, 2), 256, 0, first_token=128),),
            (BlockSnapshot(POOL, 2, BlockState.PRIVATE, refs=1),),
        )
    )
    ledger.external_fill(ExternalFill(POOL, 2, range(128), request="producer"))
    ledger.observe(normalized)
    assert reporter.reports == []


def test_current_main_simple_cpu_store_merges_migrated_state_columns():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {3: 1, 5: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={3},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(3)]
        + [SimpleNamespace(ref_cnt=1)]
        + [SimpleNamespace(ref_cnt=0)]
        + [SimpleNamespace(ref_cnt=1)]
    )
    prior = RequestSnapshot("producer", state_pool, (3, None), 2, 0)
    att.core_ledger = SimpleNamespace(
        pools={state_pool: SimpleNamespace(rewritable=True)},
        block=lambda _pool, _block: SimpleNamespace(owner="producer"),
        requests={("producer", state_pool): SimpleNamespace(snapshot=prior)},
    )
    att._simple_store_owners = {(state_pool, 3): "producer"}
    att.stats = Counter()
    snap = Snapshot(
        1,
        (RequestSnapshot("producer", state_pool, (None, 5), 2, 0),),
        (
            BlockSnapshot(state_pool, 3, BlockState.PRIVATE, refs=1),
            BlockSnapshot(state_pool, 5, BlockState.PRIVATE, refs=1),
        ),
    )

    normalized = att._normalize_offload_reservations(snap)

    assert normalized.requests == (RequestSnapshot("producer", state_pool, (3, 5), 2, 0),)
    assert att.stats["simple_offload_source_rows_retained"] == 1
    assert att.stats["simple_offload_source_rows_merged"] == 1


def test_current_main_simple_cpu_store_merges_retired_draft_window():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    draft_pool = "swa2"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {3: 2, 5: 2, 6: 2}
    att.pool_of_group = {0: POOL, 2: draft_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={3},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(3)]
        + [SimpleNamespace(ref_cnt=1)]
        + [SimpleNamespace(ref_cnt=0)]
        + [SimpleNamespace(ref_cnt=1), SimpleNamespace(ref_cnt=1)]
    )
    prior = RequestSnapshot(
        "producer", draft_pool, (None, None, None, None, None, 3, None), 96, 96, 80
    )
    att.core_ledger = SimpleNamespace(
        pools={draft_pool: SimpleNamespace(rewritable=True)},
        requests={("producer", draft_pool): SimpleNamespace(snapshot=prior)},
    )
    att._simple_store_owners = {(draft_pool, 3): "producer"}
    att.stats = Counter()
    snap = Snapshot(
        1,
        (
            RequestSnapshot(
                "producer", draft_pool, (None, None, None, None, None, None, 5, 6), 128, 96, 96
            ),
        ),
        (
            BlockSnapshot(draft_pool, 3, BlockState.PRIVATE, refs=1),
            BlockSnapshot(draft_pool, 5, BlockState.PRIVATE, refs=1),
            BlockSnapshot(draft_pool, 6, BlockState.PRIVATE, refs=1),
        ),
    )

    normalized = att._normalize_offload_reservations(snap)

    assert normalized.requests == (
        RequestSnapshot(
            "producer", draft_pool, (None, None, None, None, None, 3, 5, 6), 128, 96, 80
        ),
    )
    assert att.stats["simple_offload_source_rows_retained"] == 1
    assert att.stats["simple_offload_source_rows_merged"] == 1


def test_current_main_simple_cpu_store_keeps_owner_after_deferred_row_drains():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(4)] + [SimpleNamespace(ref_cnt=1)]
    )
    att.core_ledger = SimpleNamespace(
        pools={state_pool: SimpleNamespace(rewritable=True)},
        requests={
            ("deferred", state_pool): SimpleNamespace(
                snapshot=RequestSnapshot("deferred", state_pool, (4,), 1, 0)
            )
        },
    )
    att.stats = Counter()
    row = RequestSnapshot("deferred", state_pool, (4,), 1, 0)
    key = (row.request, row.pool)
    att._deferred_rows = {key: row}
    att._deferred_counts = {key: Counter({4: 1})}
    att._deferred_by_block = {4: {key}}
    att._simple_store_owners = {}

    att._release_deferred_rows([SimpleNamespace(block_id=4)])
    normalized = att._normalize_offload_reservations(
        Snapshot(1, (), (BlockSnapshot(state_pool, 4, BlockState.PRIVATE, refs=1),))
    )

    assert normalized.requests == (row,)
    assert not att._deferred_rows
    assert att._simple_store_owners == {(state_pool, 4): "deferred"}


def test_current_main_simple_cpu_finish_store_captures_recurrent_state_owner():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    pending = []

    def queue_finished(_request, _block_ids):
        pending.append(SimpleNamespace(gpu_block_ids=[4]))

    att = object.__new__(VllmAttachment)
    att.offload = SimpleNamespace(
        _queue_finished_eager_store=queue_finished,
        _pending_finished_stores=pending,
    )
    att.hybrid = True
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att._simple_store_owners = {}
    att.stats = Counter()

    att._install_simple_offload_scheduler()
    att.offload._queue_finished_eager_store(SimpleNamespace(request_id="producer"), ([], [4]))

    assert att._simple_store_owners == {(state_pool, 4): "producer"}
    assert att.stats["simple_offload_scheduler_wrapped"] == 1


def test_current_main_simple_cpu_store_keeps_shared_recurrent_checkpoint_published():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0) for _ in range(4)] + [SimpleNamespace(ref_cnt=3)]
    )
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(rewritable=False),
            state_pool: SimpleNamespace(rewritable=True),
        }
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (
            RequestSnapshot("producer", state_pool, (4,), 1, 0),
            RequestSnapshot("consumer", state_pool, (4,), 1, 0),
        ),
        (BlockSnapshot(state_pool, 4, BlockState.PUBLISHED, refs=3),),
    )

    normalized = att._normalize_offload_reservations(snap)
    block = normalized.block_index()[(state_pool, 4)]
    assert block.state is BlockState.PUBLISHED
    assert block.refs == 2


def test_current_main_async_store_opens_a_source_read_until_completion():
    from collections import Counter, deque
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment

    reads = []
    completions = []
    att = object.__new__(VllmAttachment)
    att._offload_store_jobs = {9: ("r", (3, 5)), 10: ("r", (7,))}
    att._offload_store_epochs = set()
    att._offload_store_dispatches = {}
    att._offload_store_events = {}
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: "gpu"}
    att.block_size = 16
    att.seq = 4
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(pools={"gpu": SimpleNamespace(block_size=16)})
    att.ledger = SimpleNamespace(
        external_read=lambda read, site: reads.append((read, site)),
        complete=completions.append,
    )
    event = SimpleNamespace(query=lambda: True)
    worker = SimpleNamespace(
        _store_handler=SimpleNamespace(
            _transfers=deque([SimpleNamespace(job_id=9, stream=SimpleNamespace(cuda_stream=77))]),
            _transfer_events={9: event},
        )
    )

    att._offload_store_started(9, worker, flushed=False)
    assert reads[0][0].blocks == (3, 5)
    assert reads[0][0].request == "r" and reads[0][0].stream == 77
    assert reads[0][1] == "offload-store:9@4"
    assert att._offload_store_epochs == {9}
    att._reap_offload_store_epochs()
    assert completions == ["offload-store:9"]
    assert not att._offload_store_epochs
    assert att.stats["offload_store_epochs_queried_complete"] == 1

    att._offload_store_started(10, worker, flushed=True)
    assert len(reads) == 1
    assert att.stats["offload_store_flushes"] == 1


def test_deferred_row_claim_follows_the_high_water_in_each_pool():
    """A request finished after its last step was scheduled keeps a deferred row whose
    computed count was the pre-step one; the claim is raised to the shadow's high water once
    the step's writes land, in each pool from its own writes, so neither pool reads the row
    as a rollback."""
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment
    from kvguard.core import Ledger, PoolGeometry, Reporter

    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry(POOL, 8, 4, reserved_blocks=frozenset({0})))
    ledger.register_pool(PoolGeometry(DRAFT_POOL, 8, 4, reserved_blocks=frozenset({0})))
    att = object.__new__(VllmAttachment)
    att.core_ledger = ledger
    att.block_size = 4
    att.eagle_lookahead = 2
    att.stats = Counter()
    att.dflash = False
    att.standalone_draft = False
    att._dflash_frontiers = {}
    att._dflash_pending_writes = {}

    row = RequestSnapshot("r", POOL, (1, 2), 8, 4)  # the step's live range, pre-step count
    att._deferred_rows = {("r", POOL): row}
    ledger.allocate(POOL, (1, 2))
    ledger.allocate(DRAFT_POOL, (1, 2))
    base = Snapshot(
        1,
        (row,),
        tuple(BlockSnapshot(POOL, b, BlockState.PRIVATE) for b in (1, 2)),
        None,
    )
    ledger.observe(att._with_eagle_snapshot(base))
    target = ledger.requests[("r", POOL)]
    draft = ledger.requests[("r", DRAFT_POOL)]
    assert (target.computed_high_water, draft.computed_high_water) == (4, 4)

    # The target's writes of the in-flight step land; the draft's have not yet.
    target.computed_high_water = 8
    rows = att._deferred_rows_for_snapshot()
    assert rows[0].num_computed == 8 and att._deferred_rows[("r", POOL)].num_computed == 8
    assert att.stats["deferred_rows_claim_raised"] == 1
    mirrored = att._with_eagle_snapshot(Snapshot(2, rows, base.blocks, None))
    draft_row = next(r for r in mirrored.requests if r.pool == DRAFT_POOL)
    assert draft_row.num_computed == 4  # the draft pool's own high water
    draft.computed_high_water = 8
    mirrored = att._with_eagle_snapshot(Snapshot(3, rows, base.blocks, None))
    draft_row = next(r for r in mirrored.requests if r.pool == DRAFT_POOL)
    assert draft_row.num_computed == 8


def test_current_main_flushed_store_epochs_close_before_the_step_is_validated():
    """A job in the step's ``jobs_to_flush`` is waited by the worker before any of the step's
    kernels run (``handle_preemptions`` from ``pre_forward``, after ``prepare_attn``), so its
    read epoch is closed before the step's dispatch; a job the handler no longer tracks has
    been observed complete by ``get_finished`` even when its pooled event was re-recorded."""
    from collections import Counter, deque
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment

    reads = []
    completions = []
    att = object.__new__(VllmAttachment)
    att.offload_kind = "managed"
    att._offload_store_jobs = {9: ("r", (3, 5)), 10: ("s", (7,)), 11: ("t", (8,))}
    att._offload_store_epochs = set()
    att._offload_store_dispatches = {}
    att._offload_store_events = {}
    att.hybrid = False
    att.eagle3 = False
    att.pool_of_group = {0: "gpu"}
    att.block_size = 16
    att.seq = 4
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(pools={"gpu": SimpleNamespace(block_size=16)})
    att.ledger = SimpleNamespace(
        external_read=lambda read, site: reads.append((read, site)),
        complete=completions.append,
    )
    stale = SimpleNamespace(query=lambda: False)  # re-recorded for a later transfer
    handler = SimpleNamespace(_transfers=deque(), _transfer_events={})
    worker = SimpleNamespace(_store_handler=handler)
    att._offload_transfer_worker = worker
    for job in (9, 10, 11):
        handler._transfers.append(
            SimpleNamespace(job_id=job, stream=SimpleNamespace(cuda_stream=77))
        )
        handler._transfer_events[job] = stale
        att._offload_store_started(job, worker, flushed=False)
    assert att._offload_store_epochs == {9, 10, 11}

    # Nothing flushed, nothing finished: every epoch stays open across the reap.
    att._flush_offload_store_epochs(SimpleNamespace(kv_connector_metadata=None))
    att._reap_offload_store_epochs()
    assert att._offload_store_epochs == {9, 10, 11} and completions == []

    # The step flushes job 9 before its forward; the handler has finished job 10.
    metadata = SimpleNamespace(jobs_to_flush={9, 42})
    att._flush_offload_store_epochs(SimpleNamespace(kv_connector_metadata=metadata))
    assert completions == ["offload-store:9"]
    assert att.stats["offload_store_epochs_flushed_before_forward"] == 1
    assert att.stats["offload_store_epochs_waited"] == 1
    del handler._transfer_events[10]
    att._reap_offload_store_epochs()
    assert completions == ["offload-store:9", "offload-store:10"]
    assert att.stats["offload_store_epochs_handler_finished"] == 1
    assert att._offload_store_epochs == {11}
    assert att.stats["offload_store_epochs_queried_complete"] == 0

    # The worker's later wait of the flushed job finds its epoch already closed.
    att._offload_store_complete(9, waited=True)
    assert completions == ["offload-store:9", "offload-store:10"]


def test_current_main_eagle_offload_mirrors_loads_and_store_reads_to_draft_pool():
    from collections import Counter, deque
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    fills = []
    reads = []
    completions = []
    att = object.__new__(VllmAttachment)
    att.eagle3 = True
    att.hybrid = False
    att.pool_of_group = {0: POOL}
    att.block_size = 16
    att.seq = 4
    att.stats = Counter()
    att._offload_filled = set()
    att._offload_store_jobs = {9: ("r", (3, 5))}
    att._offload_store_epochs = set()
    att._offload_store_dispatches = {}
    att._offload_store_events = {}
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            DRAFT_POOL: SimpleNamespace(block_size=16),
        }
    )
    att.ledger = SimpleNamespace(
        external_fill=lambda fill, site: fills.append((fill, site)),
        external_read=lambda read, site: reads.append((read, site)),
        complete=completions.append,
    )

    att._credit_offload_load(7, (3, 5), 4)
    assert [(fill.pool, fill.block) for fill, _ in fills] == [
        (POOL, 3),
        (POOL, 5),
        (DRAFT_POOL, 3),
        (DRAFT_POOL, 5),
    ]
    assert att._offload_filled == {3, 5}

    event = SimpleNamespace(query=lambda: True)
    worker = SimpleNamespace(
        _store_handler=SimpleNamespace(
            _transfers=deque([SimpleNamespace(job_id=9, stream=SimpleNamespace(cuda_stream=77))]),
            _transfer_events={9: event},
        )
    )
    att._offload_store_started(9, worker, flushed=False)
    assert [(read.dispatch_id, read.pool) for read, _ in reads] == [
        ("offload-store:9", POOL),
        ("offload-store:9:draft", DRAFT_POOL),
    ]
    assert all(read.blocks == (3, 5) and read.stream == 77 for read, _ in reads)
    att._offload_store_complete(9, waited=False)
    assert completions == ["offload-store:9", "offload-store:9:draft"]
    assert not att._offload_store_epochs


def test_current_main_managed_offload_routes_sliding_window_groups():
    from collections import Counter

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment, window_pool

    swa_pool = window_pool(1)
    fills = []
    reads = []
    att = object.__new__(VllmAttachment)
    att.eagle3 = False
    att.hybrid = True
    att.pool_of_group = {0: POOL, 1: swa_pool}
    att.group_of = {3: 0, 5: 0, 7: 1}
    att.offload = SimpleNamespace(
        config=SimpleNamespace(
            kv_group_configs=(
                SimpleNamespace(group_idx=0),
                SimpleNamespace(group_idx=1),
            )
        )
    )
    att._offload_filled = set()
    att._offload_filled_pools = set()
    att._offload_store_epochs = set()
    att._offload_store_dispatches = {}
    att._offload_store_events = {}
    att._zeroed_blocks = set()
    att.seq = 4
    att.stats = Counter()
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            swa_pool: SimpleNamespace(block_size=16),
        }
    )
    att.ledger = SimpleNamespace(
        external_fill=lambda fill, site: fills.append((fill, site)),
        external_read=lambda read, site: reads.append((read, site)),
    )
    spec = SimpleNamespace(
        block_ids=np.asarray([3, 5, 7], dtype=np.int32),
        group_sizes=[2, 1],
        block_indices=[0, 8],
    )

    block_ids, routed = att._managed_offload_blocks(spec)
    assert block_ids == (3, 5, 7)
    assert routed == {POOL: (3, 5), swa_pool: (7,)}
    att._credit_offload_load(8, block_ids, 4, routed=routed)
    assert [(fill.pool, fill.block) for fill, _ in fills] == [
        (POOL, 3),
        (POOL, 5),
        (swa_pool, 7),
    ]
    assert att._offload_filled_pools == {(POOL, 3), (POOL, 5), (swa_pool, 7)}

    att._open_offload_store(
        9,
        block_ids,
        request="r",
        stream=77,
        event=SimpleNamespace(query=lambda: False),
        routed=routed,
    )
    assert [(read.dispatch_id, read.pool, read.blocks) for read, _ in reads] == [
        ("offload-store:9", POOL, (3, 5)),
        ("offload-store:9:swa1", swa_pool, (7,)),
    ]


def test_transformers_mtp_offload_routes_shared_group_to_each_logical_pool():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.hybrid = False
    att.mtp_shared = True
    att.mtp_pools = ("mtp_draft_0", "mtp_draft_1")
    att.pool_of_group = {0: POOL}
    att.group_of = {3: 0, 5: 0}
    att.offload = SimpleNamespace(
        config=SimpleNamespace(kv_group_configs=(SimpleNamespace(group_idx=0),))
    )
    spec = SimpleNamespace(
        block_ids=np.asarray([3, 5], dtype=np.int32),
        group_sizes=[2],
        block_indices=[0],
    )

    assert att._offload_pool_blocks((3, 5)) == {
        POOL: (3, 5),
        "mtp_draft_0": (3, 5),
        "mtp_draft_1": (3, 5),
    }
    block_ids, routed = att._managed_offload_blocks(spec)
    assert block_ids == (3, 5)
    assert routed == {
        POOL: (3, 5),
        "mtp_draft_0": (3, 5),
        "mtp_draft_1": (3, 5),
    }


def test_current_main_managed_offload_swa_fault_targets_loaded_table_block():
    from kvguard.adapters.vllm.attach import VllmAttachment, window_pool

    freed = []
    pool = window_pool(1)
    att = object.__new__(VllmAttachment)
    att.offload_fault = "swa_load_early_free"
    att.offload_fault_done = False
    att._offload_filled_pools = {(pool, 7), (pool, 11)}
    att.ledger = SimpleNamespace(
        free=lambda selected_pool, blocks, site: freed.append((selected_pool, blocks, site))
    )
    att.seq = 12
    att.stats = Counter()

    att._inject_offload_window_fault(pool, np.asarray([[0, 5, 7, 11]]))
    att._inject_offload_window_fault(pool, np.asarray([[11]]))

    assert freed == [(pool, (7,), "offload-fault@12")]
    assert att.offload_fault_done
    assert att.stats["offload_fault_swa_load_early_free"] == 1


def test_current_main_managed_offload_state_fault_targets_a_consumed_loaded_block():
    from kvguard.adapters.vllm.attach import VllmAttachment

    freed = []
    pool = "state1"
    att = object.__new__(VllmAttachment)
    att.offload_fault = "state_load_early_free"
    att.offload_fault_done = False
    att._offload_filled_pools = {(pool, 7), (pool, 11)}
    att._columns = [(1, 2), (-1, 1)]
    att.ledger = SimpleNamespace(
        free=lambda selected_pool, blocks, site: freed.append((selected_pool, blocks, site))
    )
    att.seq = 13
    att.stats = Counter()

    # Loaded block 7 is in a retired column and must not be selected. Loaded block 11 is the
    # second row's running column and is consumed by the recurrent-state write.
    table = np.asarray([[0, 5, 6, 7], [0, 11, 12, 0]])
    att._inject_offload_state_fault(pool, table)
    att._inject_offload_state_fault(pool, np.asarray([[0, 7], [0, 11]]))

    assert freed == [(pool, (11,), "offload-fault@13")]
    assert att.offload_fault_done
    assert att.stats["offload_fault_state_load_early_free"] == 1


def test_current_main_offload_trace_is_bounded_metadata_only():
    from collections import deque
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att.seq = 8
    att.stats = Counter()
    key_chunks = {
        "key-z": SimpleNamespace(chunk_id=7),
        "key-b": SimpleNamespace(chunk_id=3),
        "key-a": SimpleNamespace(chunk_id=5),
        "key-c": SimpleNamespace(chunk_id=11),
    }
    att.offload = SimpleNamespace(
        manager=SimpleNamespace(_policy=SimpleNamespace(get=lambda key: key_chunks.get(key))),
        _jobs={
            1: SimpleNamespace(keys={"key-z"}),
            2: SimpleNamespace(keys={"key-b", "key-a"}),
            3: SimpleNamespace(keys={"key-c"}),
        },
    )
    att.offload_events = deque(maxlen=2)

    att._trace_offload(
        "load_async", 1, "r1", (3, 5), SimpleNamespace(src_spec=SimpleNamespace(block_ids=[7]))
    )
    att._trace_offload(
        "store", 2, "r2", (7,), SimpleNamespace(dst_spec=SimpleNamespace(block_ids=[5, 3]))
    )
    att._trace_offload(
        "load_sync", 3, "r3", (9,), SimpleNamespace(src_spec=SimpleNamespace(block_ids=[11]))
    )

    assert list(att.offload_events) == [
        {
            "seq": 9,
            "kind": "store",
            "job_id": 2,
            "request": "r2",
            "blocks": [7],
            "keys": ["'key-a'", "'key-b'"],
            "cpu_chunks": [5, 3],
            "key_chunks": [
                {"cpu_chunk": 3, "key": "'key-b'"},
                {"cpu_chunk": 5, "key": "'key-a'"},
            ],
        },
        {
            "seq": 9,
            "kind": "load_sync",
            "job_id": 3,
            "request": "r3",
            "blocks": [9],
            "keys": ["'key-c'"],
            "cpu_chunks": [11],
            "key_chunks": [{"cpu_chunk": 11, "key": "'key-c'"}],
        },
    ]
    assert att.stats["offload_trace_errors"] == 0


def test_current_main_offload_trace_cannot_break_serving():
    from collections import deque
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment

    class BrokenKey:
        def __repr__(self):
            raise RuntimeError("diagnostic repr failed")

    att = object.__new__(VllmAttachment)
    att.seq = 8
    att.stats = Counter()
    att.offload = SimpleNamespace(
        manager=SimpleNamespace(_policy=SimpleNamespace(get=lambda key: None)),
        _jobs={1: SimpleNamespace(keys={BrokenKey()})},
    )
    att.offload_events = deque(maxlen=2)

    att._trace_offload("store", 1, "r1", (7,))

    assert list(att.offload_events) == []
    assert att.stats["offload_trace_errors"] == 1


def test_current_main_simple_cpu_hybrid_offload_routes_each_physical_block_to_its_pool():
    from collections import Counter
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    fills = []
    reads = []
    completions = []
    att = object.__new__(VllmAttachment)
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {3: 0, 5: 1}
    att.pool_of_group = {0: POOL, 1: "state0"}
    att.core_ledger = SimpleNamespace(
        pools={
            POOL: SimpleNamespace(block_size=16),
            "state0": SimpleNamespace(block_size=1),
        }
    )
    att.ledger = SimpleNamespace(
        external_fill=lambda fill, site: fills.append((fill, site)),
        external_read=lambda read, site: reads.append((read, site)),
        complete=completions.append,
    )
    att._offload_filled = set()
    att._offload_store_epochs = set()
    att._offload_store_dispatches = {}
    att._offload_store_events = {}
    att.seq = 4
    att.stats = Counter()

    att._credit_offload_load(7, (3, 5), 4)
    assert [(fill.pool, fill.block, tuple(fill.offsets)) for fill, _ in fills] == [
        (POOL, 3, tuple(range(16))),
        ("state0", 5, (0,)),
    ]

    att._open_offload_store(9, (3, 5), request="r", stream=77, event=None)
    assert [(read.dispatch_id, read.pool, read.blocks) for read, _ in reads] == [
        ("offload-store:9", POOL, (3,)),
        ("offload-store:9:state0", "state0", (5,)),
    ]
    assert [tuple(read.offsets) for read, _ in reads] == [tuple(range(16)), (0,)]
    att._offload_store_complete(9, waited=False)
    assert completions == ["offload-store:9", "offload-store:9:state0"]


def test_current_main_worker_zeroing_fills_only_the_named_attention_blocks():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.hybrid = True
    att.eagle3 = False
    att.kv_gid = 0
    att.group_of = {3: 0, 5: 1}
    att.pool_of_group = {0: POOL, 1: "state0"}
    att.core_ledger = SimpleNamespace(
        pools={POOL: SimpleNamespace(block_size=16), "state0": SimpleNamespace(block_size=1)}
    )
    att._zeroed_blocks = set()
    att.seq = 4
    att.stats = Counter()

    att._zero_fills(SimpleNamespace(new_block_ids_to_zero=[3]))

    assert att._zeroed_blocks == {(POOL, 3)}
    assert att.stats["zero_filled_blocks"] == 1


def test_current_main_cow_execution_pins_are_not_request_references():
    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att._cow_pins = Counter({3: 1, 4: 1})
    att._cow_cache_blocks = {("state0", 4)}
    att._cow_request_blocks = set()
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(4)]
        + [SimpleNamespace(block_hash=b"cached")]
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (RequestSnapshot("producer", "state0", (3,), 1, 0),),
        (
            BlockSnapshot("state0", 3, BlockState.PRIVATE, refs=2),
            BlockSnapshot("state0", 4, BlockState.PRIVATE, refs=1),
        ),
    )

    normalized = att._normalize_cow_reservations(snap)

    got = normalized.block_index()
    assert got[("state0", 3)].refs == 1
    assert got[("state0", 4)] == BlockSnapshot(
        "state0", 4, BlockState.PUBLISHED, reserved=False, refs=1
    )
    assert normalized.requests[-1] == RequestSnapshot(
        "__kvguard_cow_pin__:state0:4", "state0", (4,), 1, 1
    )
    att._release_cow_pins([SimpleNamespace(block_id=3), SimpleNamespace(block_id=4)])
    assert not att._cow_pins


def test_current_main_overlapping_cow_pins_are_one_cache_reader():
    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att._cow_pins = Counter({4: 2})
    att._cow_cache_blocks = {("state0", 4)}
    att._cow_request_blocks = set()
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(4)]
        + [SimpleNamespace(block_hash=b"cached")]
    )
    att.stats = Counter()

    normalized = att._normalize_cow_reservations(
        Snapshot(
            1,
            (),
            (BlockSnapshot("state0", 4, BlockState.PUBLISHED, refs=2),),
        )
    )

    assert normalized.blocks == (
        BlockSnapshot("state0", 4, BlockState.PUBLISHED, reserved=False, refs=1),
    )
    assert normalized.requests == (
        RequestSnapshot("__kvguard_cow_pin__:state0:4", "state0", (4,), 1, 1),
    )


def test_current_main_cow_source_pin_does_not_publish_partial_attention_tail():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att._cow_pins = Counter({3: 1})
    att._cow_cache_blocks = set()
    att._cow_request_blocks = set()
    att.group_of = {3: 0}
    att.managers = [SimpleNamespace(block_size=544)]
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(3)]
        + [SimpleNamespace(block_hash=b"partial", block_hash_num_tokens=288)]
    )
    att.stats = Counter()

    normalized = att._normalize_cow_reservations(
        Snapshot(
            1,
            (RequestSnapshot("consumer", POOL, (3,), 305, 304),),
            (BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=2),),
        )
    )

    assert normalized.blocks == (
        BlockSnapshot(POOL, 3, BlockState.PRIVATE, reserved=False, refs=1),
    )
    assert normalized.requests == (RequestSnapshot("consumer", POOL, (3,), 305, 304),)


def test_current_main_simple_cpu_store_keeps_cow_cache_destination_published():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    state_pool = "state0"
    att = object.__new__(VllmAttachment)
    att.offload_kind = "simple"
    att.hybrid = True
    att.eagle3 = False
    att.group_of = {4: 1}
    att.pool_of_group = {0: POOL, 1: state_pool}
    att._cow_cache_blocks = {(state_pool, 4)}
    att._simple_store_owners = {}
    att.offload = SimpleNamespace(
        _reqs_to_load={},
        _abandoned_reqs_to_load={},
        _in_flight_store_gpu_blocks={4},
        _store_event_to_blocks={},
        _abandoned_store_event_to_blocks={},
        _pending_finished_stores=[],
    )
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(ref_cnt=0, block_hash=None) for _ in range(4)]
        + [SimpleNamespace(ref_cnt=1, block_hash=b"cached")]
    )
    att.core_ledger = SimpleNamespace(
        pools={state_pool: SimpleNamespace(rewritable=True, block_size=1)}
    )
    att.stats = Counter()

    normalized = att._normalize_offload_reservations(
        Snapshot(1, (), (BlockSnapshot(state_pool, 4, BlockState.PRIVATE, refs=1),))
    )

    assert normalized.requests == (
        RequestSnapshot(f"__kvguard_simple_store__:{state_pool}:4", state_pool, (4,), 1, 0),
    )
    assert normalized.blocks == (
        BlockSnapshot(state_pool, 4, BlockState.PUBLISHED, reserved=False, refs=1),
    )


def test_current_main_request_cow_destination_remains_private_after_pin_normalization():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att._cow_pins = Counter({4: 1})
    att._cow_cache_blocks = set()
    att._cow_request_blocks = {(POOL, 4)}
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(4)]
        + [SimpleNamespace(block_hash=b"partial")]
    )
    att.stats = Counter()
    snap = Snapshot(
        1,
        (RequestSnapshot("consumer", POOL, (4,), 8, 7),),
        (BlockSnapshot(POOL, 4, BlockState.PUBLISHED, refs=2),),
    )

    normalized = att._normalize_cow_reservations(snap)

    assert normalized.blocks == (
        BlockSnapshot(POOL, 4, BlockState.PRIVATE, reserved=False, refs=1),
    )


def test_current_main_unmapped_cow_source_pin_counts_as_a_published_reader():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att._cow_pins = Counter({3: 1})
    att._cow_cache_blocks = set()
    att._cow_request_blocks = set()
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(3)]
        + [SimpleNamespace(block_hash=b"partial")]
    )
    att.core_ledger = SimpleNamespace(pools={POOL: SimpleNamespace(block_size=16)})
    att.stats = Counter()

    normalized = att._normalize_cow_reservations(
        Snapshot(1, (), (BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=1),))
    )

    assert normalized.blocks == (
        BlockSnapshot(POOL, 3, BlockState.PUBLISHED, reserved=False, refs=1),
    )
    assert normalized.requests == (RequestSnapshot("__kvguard_cow_pin__:gpu:3", POOL, (3,), 16, 0),)


def test_async_speculative_snapshot_caps_only_the_newest_inflight_step():
    from collections import Counter

    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att._offload_pending_by_req = {}
    att._speculative_steps = {}
    att._speculative_caps = {}
    att.stats = Counter()

    att._record_speculative_step(11, ["r"], np.asarray([232]), np.asarray([4]))
    assert att._snapshot_computed("r", 240, 0) == 236
    assert att._snapshot_computed("r", 240, 4) == 240
    assert att._snapshot_computed("r", 64, 0) == 64

    # A newer step supersedes the same request's cap. Output completion does not remove the
    # physical-write frontier: the scheduler can expose a lagging connector-local count for
    # one more snapshot, and the next dispatch replaces the frontier before accessing KV.
    att._record_speculative_step(12, ["r"], np.asarray([233]), np.asarray([4]))
    att._complete_speculative_step(11)
    assert att._snapshot_computed("r", 240, 0) == 237
    att._complete_speculative_step(12)
    assert att._snapshot_computed("r", 240, 0) == 237
    assert att._snapshot_computed("r", 240, 4) == 240
    assert "r" not in att._speculative_caps
    assert att.stats["speculative_completed_caps_retired"] == 1
    assert att.stats["speculative_snapshot_caps"] == 3
    assert att.stats["speculative_snapshot_capped_tokens"] == 10


def test_speculative_snapshot_retains_concrete_target_writes_beyond_a_lagging_cpu_count():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment
    from kvguard.core import ExternalFill, Ledger, PoolGeometry, Reporter

    reporter = Reporter()
    ledger = Ledger(reporter)
    ledger.register_pool(PoolGeometry(POOL, 5, 16))
    ledger.allocate(POOL, range(5))
    blocks = tuple(BlockSnapshot(POOL, block, BlockState.PRIVATE, refs=1) for block in range(5))
    empty = RequestSnapshot("r", POOL, tuple(range(5)), 80, 0)
    ledger.observe(Snapshot(1, (empty,), blocks))
    for block in range(4):
        ledger.external_fill(ExternalFill(POOL, block, range(16), request="r"))
    lagging = RequestSnapshot("r", POOL, tuple(range(5)), 80, 64)
    ledger.observe(Snapshot(2, (lagging,), blocks))

    att = object.__new__(VllmAttachment)
    att.groups = ({"kind": "attention", "pool": POOL, "draft": False},)
    att._speculative_caps = {"r": (7, 87)}
    att._speculative_retained = {}
    att.core_ledger = ledger
    att.ledger = ledger
    att.stats = Counter()
    att.seq = 2

    att._restore_speculative_target_writes(Snapshot(2, (lagging,), blocks))

    assert ledger.block(POOL, 4).filled == (1 << 16) - 1
    assert att._speculative_retained == {(POOL, "r"): 80}
    assert att.stats["speculative_target_slots_retained"] == 16
    assert reporter.reports == []


def test_async_mtp_snapshot_uses_each_draft_modules_concrete_frontier():
    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att.mtp_gids = (2, 3, 4)
    att.mtp_pools = ("swa2", "swa3", "swa4")
    att.mtp_shared = False
    att.pool_of_group = {2: "swa2", 3: "swa3", 4: "swa4"}
    att.window_of = {2: 64, 3: 64, 4: 64}
    att.extra_of = {2: 2, 3: 2, 4: 2}
    att._mtp_frontiers = {
        ("swa2", "r"): 83,
        ("swa3", "r"): 84,
        ("swa4", "r"): 85,
    }
    att.stats = Counter()
    snap = Snapshot(
        3,
        (
            RequestSnapshot("r", "swa1", (1,) * 8, 128, 90, 0),
            RequestSnapshot("r", "swa2", (2,) * 8, 128, 90, 0),
            RequestSnapshot("r", "swa3", (3,) * 8, 128, 90, 0),
            RequestSnapshot("r", "swa4", (4,) * 8, 128, 90, 0),
        ),
        (),
    )

    normalized = att._with_mtp_frontiers(snap)

    assert [row.num_computed for row in normalized.requests] == [90, 83, 84, 85]
    assert [row.first_token for row in normalized.requests] == [0, 18, 19, 20]
    assert att.stats["mtp_snapshot_caps"] == 3
    assert att.stats["mtp_snapshot_capped_tokens"] == 18


def test_transformers_mtp_snapshot_clones_shared_physical_group_per_module():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment
    from kvguard.core import Ledger, PoolGeometry, Reporter

    reporter = Reporter()
    ledger = Ledger(reporter)
    for pool in (POOL, "mtp_draft_0", "mtp_draft_1"):
        ledger.register_pool(PoolGeometry(pool, 6, 16))

    att = object.__new__(VllmAttachment)
    att.mtp_gids = (0, 0)
    att.mtp_pools = ("mtp_draft_0", "mtp_draft_1")
    att.mtp_shared = True
    att.eagle_lookahead = 2
    att.core_ledger = ledger
    att.window_of = {0: None}
    att.extra_of = {0: 0}
    att._mtp_frontiers = {
        ("mtp_draft_0", "r"): 18,
        ("mtp_draft_1", "r"): 17,
    }
    att.stats = Counter()
    snap = Snapshot(
        3,
        (RequestSnapshot("r", POOL, (1, 2), 30, 20),),
        (
            BlockSnapshot(POOL, 1, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(POOL, 2, BlockState.PRIVATE, refs=1),
        ),
        free={POOL: frozenset({3, 4, 5})},
        free_count={POOL: 3},
    )

    normalized = att._with_mtp_frontiers(snap)

    assert [(row.pool, row.num_tokens, row.num_computed) for row in normalized.requests] == [
        (POOL, 30, 20),
        ("mtp_draft_0", 32, 18),
        ("mtp_draft_1", 32, 17),
    ]
    assert [(block.pool, block.block, block.state) for block in normalized.blocks] == [
        (POOL, 1, BlockState.PUBLISHED),
        (POOL, 2, BlockState.PRIVATE),
        ("mtp_draft_0", 1, BlockState.PUBLISHED),
        ("mtp_draft_0", 2, BlockState.PRIVATE),
        ("mtp_draft_1", 1, BlockState.PUBLISHED),
        ("mtp_draft_1", 2, BlockState.PRIVATE),
    ]
    assert normalized.free == {
        POOL: frozenset({3, 4, 5}),
        "mtp_draft_0": frozenset({3, 4, 5}),
        "mtp_draft_1": frozenset({3, 4, 5}),
    }
    assert normalized.free_count == {POOL: 3, "mtp_draft_0": 3, "mtp_draft_1": 3}
    assert att.stats["mtp_snapshot_caps"] == 2
    assert att.stats["mtp_snapshot_capped_tokens"] == 5


def test_mtp_full_graph_dispatches_each_module_from_persistent_slots():
    from kvguard.adapters.vllm.attach import VllmAttachment

    att = object.__new__(VllmAttachment)
    att.mtp_gids = (2, 3, 4)
    att.mtp_layers = ("mtp.0", "mtp.1", "mtp.2")
    att.kvm = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(layer_names=("target.full",)),
                SimpleNamespace(layer_names=("target.swa",)),
                SimpleNamespace(layer_names=("mtp.0",)),
                SimpleNamespace(layer_names=("mtp.1",)),
                SimpleNamespace(layer_names=("mtp.2",)),
            )
        )
    )
    att.stats = Counter()
    calls = []
    att._dispatch_mtp = lambda speculator, tokens, slots, module, full_graph=False: calls.append(
        (tokens, slots, module, full_graph)
    )
    speculator = SimpleNamespace(
        block_tables=SimpleNamespace(slot_mappings=("full", "swa", "swa2", "swa3", "swa4"))
    )

    att._dispatch_mtp_full_graph(speculator, SimpleNamespace(num_tokens=32))

    assert [call[2] for call in calls] == [0, 1, 2]
    assert all(call[0] == 32 for call in calls)
    assert calls[0][1] == {"mtp.0": "swa2", "mtp.1": "swa3", "mtp.2": "swa4"}
    assert all(call[3] for call in calls)
    assert att.stats["mtp_full_graph_replays"] == 1


def test_current_main_publication_is_monotonic_until_hash_or_generation_changes():
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import BlockSnapshot, Snapshot

    att = object.__new__(VllmAttachment)
    att._published_blocks = {("gpu", 3)}
    att._offload_filled = set()
    att.core_ledger = SimpleNamespace(pools={"gpu": SimpleNamespace(rewritable=False)})
    blocks = [SimpleNamespace(block_hash=None) for _ in range(4)]
    blocks[3].block_hash = b"hash"
    att.pool = SimpleNamespace(blocks=blocks)
    snap = Snapshot(2, (), (BlockSnapshot("gpu", 3, BlockState.PRIVATE, refs=1),))

    normalized = att._normalize_publications(snap)
    assert normalized.blocks[0].state is BlockState.PUBLISHED
    blocks[3].block_hash = None
    normalized = att._normalize_publications(snap)
    assert normalized.blocks[0].state is BlockState.PRIVATE
    assert not att._published_blocks


def test_current_main_publication_is_monotonic_for_rewritable_state():
    from types import SimpleNamespace

    from kvguard.adapters.vllm.attach import VllmAttachment
    from kvguard.core import BlockSnapshot, Snapshot

    att = object.__new__(VllmAttachment)
    att._published_blocks = {("gpu", 3), ("state0", 4)}
    att._offload_filled = set()
    att.core_ledger = SimpleNamespace(
        pools={
            "gpu": SimpleNamespace(rewritable=False),
            "state0": SimpleNamespace(rewritable=True),
        }
    )
    blocks = [SimpleNamespace(block_hash=None) for _ in range(5)]
    blocks[3].block_hash = blocks[4].block_hash = b"hash"
    att.pool = SimpleNamespace(blocks=blocks)
    snap = Snapshot(
        2,
        (),
        (
            BlockSnapshot("gpu", 3, BlockState.PRIVATE, refs=1),
            BlockSnapshot("state0", 4, BlockState.PRIVATE, refs=1),
        ),
    )

    normalized = att._normalize_publications(snap)
    got = normalized.block_index()
    assert got[("gpu", 3)].state is BlockState.PUBLISHED
    assert got[("state0", 4)].state is BlockState.PUBLISHED
    assert att._published_blocks == {("gpu", 3), ("state0", 4)}


def test_current_main_partial_attention_hash_can_return_to_private_for_tail_fill():
    from kvguard.adapters.vllm.attach import POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att._published_blocks = {(POOL, 3)}
    att._offload_filled = {3}
    att.group_of = {3: 0}
    att.managers = [SimpleNamespace(block_size=16)]
    att.pool = SimpleNamespace(
        blocks=[SimpleNamespace(block_hash=None) for _ in range(3)]
        + [SimpleNamespace(block_hash=b"partial", block_hash_num_tokens=20)]
    )
    snap = Snapshot(2, (), (BlockSnapshot(POOL, 3, BlockState.PRIVATE, refs=1),))

    normalized = att._normalize_publications(snap)

    assert normalized.blocks[0].state is BlockState.PRIVATE
    assert not att._published_blocks


def test_current_main_eagle_allocation_forgets_target_and_draft_publications():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.eagle3 = True
    att._published_blocks = {
        (POOL, 3),
        (DRAFT_POOL, 3),
        (POOL, 4),
        (DRAFT_POOL, 4),
    }

    att._forget_allocated_publications(POOL, (3,))

    assert att._published_blocks == {(POOL, 4), (DRAFT_POOL, 4)}


def test_current_main_eagle_snapshot_keeps_target_and_draft_initialization_separate():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.block_size = 16
    att.eagle_lookahead = 3
    target = RequestSnapshot("r", POOL, (3, 5), 30, 27)
    other = RequestSnapshot("r", "state0", (7,), 1, 0)
    blocks = (
        BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=1),
        BlockSnapshot(POOL, 5, BlockState.PRIVATE, refs=1),
        BlockSnapshot("state0", 7, BlockState.PRIVATE, refs=1),
    )
    snap = Snapshot(
        4,
        (target, other),
        blocks,
        {POOL: frozenset({8, 9}), "state0": frozenset({10})},
        {POOL: 2, "state0": 1},
        stream=77,
        after=("older",),
    )

    got = att._with_eagle_snapshot(snap)

    draft = got.request_index()[("r", DRAFT_POOL)]
    assert draft.blocks == target.blocks
    assert (draft.num_tokens, draft.num_computed) == (32, 27)
    assert got.request_index()[("r", POOL)] == target
    assert (DRAFT_POOL, 3) in got.block_index() and (DRAFT_POOL, 5) in got.block_index()
    assert got.block_index()[(DRAFT_POOL, 3)].state is BlockState.PUBLISHED
    assert got.free[DRAFT_POOL] == got.free[POOL]
    assert got.free_count[DRAFT_POOL] == got.free_count[POOL]
    assert (got.stream, got.after) == (77, ("older",))


def test_current_main_dflash_snapshot_stops_at_concrete_draft_frontier():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, POOL, VllmAttachment

    att = object.__new__(VllmAttachment)
    att.block_size = 16
    att.eagle_lookahead = 9
    att.dflash = True
    att._dflash_frontiers = {"r": 26, "gone": 9}
    att._dflash_pending_writes = {"gone": {10}}
    att.stats = Counter()
    target = RequestSnapshot("r", POOL, (3, 5), 30, 27)
    snap = Snapshot(
        4,
        (target,),
        (
            BlockSnapshot(POOL, 3, BlockState.PUBLISHED, refs=1),
            BlockSnapshot(POOL, 5, BlockState.PRIVATE, refs=1),
        ),
        {POOL: frozenset({8, 9})},
        {POOL: 2},
    )

    got = att._with_eagle_snapshot(snap)

    draft = got.request_index()[("r", DRAFT_POOL)]
    assert (draft.num_tokens, draft.num_computed) == (32, 26)
    assert att._dflash_frontiers == {"r": 26}
    assert att._dflash_pending_writes == {}
    assert att.stats["dflash_snapshot_caps"] == 1
    assert att.stats["dflash_snapshot_capped_tokens"] == 1


def test_current_main_eagle_dispatch_covers_prefill_and_each_autoregressive_draft():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, VllmAttachment

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    dispatched = []
    opened = []
    att = object.__new__(VllmAttachment)
    att.cache = SimpleNamespace(ids=lambda rid: [1, 2] if rid == "a" else [3, 4])
    att.block_size = 16
    att.pool = SimpleNamespace(null_block=SimpleNamespace(block_id=0))
    att.ledger = SimpleNamespace(dispatch=dispatched.append)
    att._opened = opened.append
    att._eagle_dispatch_seq = 0
    att.eagle_fault = ""
    att.eagle_fault_done = False
    att.stream_epochs = False
    att.seq = 9
    att.stats = Counter()
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["a", "b"],
        num_scheduled_tokens=np.asarray([2, 1]),
        seq_lens=DeviceArray([12, 21]),
    )

    att._dispatch_eagle(batch, 3)

    writes = dispatched[::2]
    assert len(dispatched) == 6 and len(opened) == 6
    assert all(descriptor.pool == DRAFT_POOL for descriptor in dispatched)
    assert [list(write.mapping.positions) for write in writes] == [[10, 11, 20], [12, 21], [13, 22]]
    assert att.stats["eagle_draft_forwards"] == 3
    assert att.stats["eagle_draft_tokens"] == 7


def test_current_main_eagle_publishes_draft_descriptors_with_the_target_job():
    from kvguard.adapters.vllm.attach import VllmAttachment

    events = []
    batch = SimpleNamespace()
    att = object.__new__(VllmAttachment)
    att._eagle_prepare_lookahead = None
    att._eagle_prepared_batches = {}
    att._dispatch_eagle = lambda got, k: events.append(("dispatch", got, k))

    att._begin_eagle_target(SimpleNamespace(skip_drafts=False, num_spec_tokens_to_schedule=7))
    att._prepare_eagle_descriptors(batch)
    events.append(("release",))
    att._match_eagle_proposal(batch, 7)

    assert events == [("dispatch", batch, 7), ("release",)]
    assert att._eagle_prepared_batches == {}


def test_current_main_eagle_zero_width_job_still_publishes_draft_prefill():
    from kvguard.adapters.vllm.attach import VllmAttachment

    dispatched = []
    batch = SimpleNamespace()
    att = object.__new__(VllmAttachment)
    att._eagle_prepare_lookahead = None
    att._eagle_prepared_batches = {}
    att._dispatch_eagle = lambda got, k: dispatched.append((got, k))

    att._begin_eagle_target(SimpleNamespace(skip_drafts=True, num_spec_tokens_to_schedule=7))
    att._prepare_eagle_descriptors(batch)
    att._match_eagle_proposal(batch, 0)

    assert dispatched == [(batch, 0)]
    assert att._eagle_prepared_batches == {}


def test_current_main_eagle_fault_frees_a_block_written_by_the_current_batch():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, VllmAttachment

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    freed = []
    att = object.__new__(VllmAttachment)
    att.cache = SimpleNamespace(ids=lambda _rid: [1, 2, 3])
    att.block_size = 16
    att.pool = SimpleNamespace(null_block=SimpleNamespace(block_id=0))
    att.ledger = SimpleNamespace(
        free=lambda pool, blocks, site: freed.append((pool, blocks, site)),
        dispatch=lambda _descriptor: None,
    )
    att._opened = lambda _dispatch_id: None
    att._eagle_dispatch_seq = 0
    att.eagle_fault = "draft_early_free"
    att.eagle_fault_done = False
    att.stream_epochs = False
    att.seq = 9
    att.stats = Counter()
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["a"],
        num_scheduled_tokens=np.asarray([2]),
        seq_lens=DeviceArray([18]),
    )

    att._dispatch_eagle(batch, 1)

    assert freed == [(DRAFT_POOL, (2,), "eagle-fault@9")]
    assert att.eagle_fault_done
    assert att.stats["eagle_fault_block"] == 2


def test_current_main_dflash_context_dispatch_uses_concrete_projection_slots():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, VllmAttachment
    from kvguard.core import DispatchRow

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    dispatched = []
    opened = []
    att = object.__new__(VllmAttachment)
    att._dflash_input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["a", "b"],
        query_start_loc=DeviceArray([0, 2, 4]),
    )
    att.dflash_fault = ""
    att.dflash_fault_done = False
    att._dflash_frontiers = {}
    att._dflash_pending_writes = {}
    att._dflash_dispatch_seq = 0
    att.stream_epochs = False
    att.seq = 11
    att.stats = Counter()
    att.ledger = SimpleNamespace(dispatch=dispatched.append)
    att._opened = opened.append
    speculator = SimpleNamespace(
        context_positions=DeviceArray([0, 1, 0, 1]),
        _context_slot_mappings=[DeviceArray([16, 17, -1, 33])],
    )

    att._dispatch_dflash_context(speculator, 0, 4)

    assert len(dispatched) == 1 and len(opened) == 1
    descriptor = dispatched[0]
    assert descriptor.pool == DRAFT_POOL
    assert tuple(descriptor.rows[0].writes) == (0, 1)
    assert tuple(descriptor.rows[1].writes) == (1,)
    assert list(descriptor.mapping.slots) == [16, 17, 33]
    assert list(descriptor.mapping.rows) == [0, 0, 1]
    assert list(descriptor.mapping.positions) == [0, 1, 1]
    assert att.stats["dflash_context_tokens"] == 3
    assert att._dflash_frontiers == {"a": 2, "b": 0}
    assert att._dflash_pending_writes == {"b": {1}}

    att._advance_dflash_frontiers((DispatchRow("b", writes=range(1)),))

    assert att._dflash_frontiers == {"a": 2, "b": 2}
    assert att._dflash_pending_writes == {}


def test_current_main_dflash_query_dispatch_uses_concrete_gpu_layout():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, VllmAttachment

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    dispatched = []
    opened = []
    draft_model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: ("draft.layer",))
    speculator = SimpleNamespace(
        model=draft_model,
        draft_kv_cache_group_id=0,
        input_buffers=SimpleNamespace(
            query_start_loc=DeviceArray([0, 3, 6]),
            positions=DeviceArray([0, 1, 2, 0, 1, 2]),
            seq_lens=DeviceArray([3, 3]),
        ),
        block_tables=SimpleNamespace(
            input_block_tables={0: DeviceArray([[2], [3]])},
            slot_mappings={0: DeviceArray([32, 33, 34, 48, 49, -1])},
        ),
    )
    att = object.__new__(VllmAttachment)
    att._dflash_input_batch = SimpleNamespace(num_reqs=2, req_ids=["a", "b"])
    att.kv_gid = 0
    att.block_size = 16
    att.dflash_fault = ""
    att.dflash_fault_done = False
    att._dflash_frontiers = {}
    att._dflash_pending_writes = {}
    att._dflash_dispatch_seq = 0
    att.stream_epochs = False
    att.seq = 12
    att.stats = Counter()
    att.ledger = SimpleNamespace(dispatch=dispatched.append)
    att._opened = opened.append
    slots = {"draft.layer": DeviceArray([32, 33, 34, 48, 49, -1])}

    att._dispatch_dflash_query(speculator, 6, slots)

    assert len(dispatched) == 2 and len(opened) == 2
    write, attention = dispatched
    assert write.pool == attention.pool == DRAFT_POOL
    assert [tuple(row.writes) for row in write.rows] == [(0, 1, 2), (0, 1)]
    assert [tuple(row.reads) for row in attention.rows] == [(0, 1, 2), (0, 1)]
    assert list(write.mapping.slots) == [32, 33, 34, 48, 49, -1]
    assert list(write.mapping.positions) == [0, 1, 2, 0, 1, 2]
    assert att.stats["dflash_query_tokens"] == 6
    assert att._dflash_frontiers == {"a": 3, "b": 2}
    assert att._dflash_pending_writes == {}

    dispatched.clear()
    opened.clear()
    att._dispatch_dflash_graph_query(speculator, 8)

    assert len(dispatched) == 2 and len(opened) == 2
    assert list(dispatched[0].mapping.slots) == [32, 33, 34, 48, 49, -1]
    assert att.stats["dflash_graph_replays"] == 1


def test_current_main_standalone_draft_dispatch_uses_concrete_rewrite_layout():
    from kvguard.adapters.vllm.attach import DRAFT_POOL, VllmAttachment

    class DeviceArray:
        _next_ptr = 100

        def __init__(self, values, ptr=None):
            self.values = np.asarray(values)
            if ptr is None:
                ptr = DeviceArray._next_ptr
                DeviceArray._next_ptr += 1
            self.ptr = ptr

        @property
        def shape(self):
            return self.values.shape

        def __getitem__(self, item):
            return DeviceArray(self.values[item], self.ptr)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def data_ptr(self):
            return self.ptr

    slots = DeviceArray([22, 23, 24, 45, 46])
    metadata = type("FlashAttentionMetadata", (), {})()
    metadata.num_actual_tokens = 5
    metadata.query_start_loc = DeviceArray([0, 3, 5])
    metadata.seq_lens = DeviceArray([5, 7])
    metadata.block_table = DeviceArray([[10, 11, 12, 0], [20, 21, 22, 23]])
    metadata.slot_mapping = slots
    # Row a: two accepted tokens then its correction slot at position 4; row b: one accepted
    # token and its correction slot at position 6 (``last_token_indices`` names the expanded
    # row's correction token).
    speculator = SimpleNamespace(
        draft_attn_layer_names={"draft.layer"}, last_token_indices=DeviceArray([2, 4])
    )
    dispatched = []
    opened = []
    reopened = []
    att = object.__new__(VllmAttachment)
    att._standalone_input_batch = SimpleNamespace(num_reqs=2, req_ids=["a", "b"])
    att.standalone_fault = ""
    att.standalone_fault_done = False
    att._standalone_dispatch_seq = 0
    att._standalone_calls = 0
    att._standalone_scratch = {"a": (2, 4)}  # the previous proposal's scratch of request a
    att._dflash_frontiers = {"a": 2, "b": 5}
    att._dflash_pending_writes = {}
    att.block_size = 2
    att.stream_epochs = False
    att.seq = 13
    att.stats = Counter()
    att.ledger = SimpleNamespace(dispatch=dispatched.append)
    att.core_ledger = SimpleNamespace(
        authorize_scratch_rewrite=lambda pool, rid, positions, site: (
            reopened.append((pool, rid, tuple(positions), site)) or 1
        )
    )
    att._opened = opened.append

    att._dispatch_standalone_draft(
        speculator,
        DeviceArray([1, 2, 3, 4, 5]),
        DeviceArray([2, 3, 4, 5, 6]),
        {"draft.layer": metadata},
        {"draft.layer": slots},
    )

    assert len(dispatched) == 2 and len(opened) == 2
    write, attention = dispatched
    assert write.pool == attention.pool == DRAFT_POOL
    assert [tuple(row.writes) for row in write.rows] == [(2, 3, 4), (5, 6)]
    assert [tuple(row.reads) for row in attention.rows] == [tuple(range(5)), tuple(range(7))]
    assert list(write.mapping.slots) == [22, 23, 24, 45, 46]
    assert list(write.mapping.positions) == [2, 3, 4, 5, 6]
    assert att._dflash_frontiers == {"a": 5, "b": 7}
    assert att.stats["standalone_draft_forwards"] == 1
    assert att.stats["standalone_draft_tokens"] == 5
    # The previous scratch of a (positions 2 and 3) is re-opened before this write; the new
    # scratch starts at each row's correction slot.
    assert reopened == [(DRAFT_POOL, "a", (2, 3), "standalone-scratch@13")]
    assert att._standalone_scratch == {"a": (4, 5), "b": (6, 7)}
    assert att.stats["standalone_scratch_rewrites"] == 2
    assert att.stats["standalone_scratch_reopened_slots"] == 1
    assert att._standalone_calls == 1

    # A later one-token forward extends the scratch of each row from its correction slot.
    metadata.num_actual_tokens = 2
    metadata.query_start_loc = DeviceArray([0, 1, 2])
    metadata.seq_lens = DeviceArray([6, 8])
    later = DeviceArray([25, 47])
    metadata.slot_mapping = later
    att._dispatch_standalone_draft(
        speculator,
        DeviceArray([7, 8]),
        DeviceArray([5, 7]),
        {"draft.layer": metadata},
        {"draft.layer": later},
    )
    assert att._standalone_scratch == {"a": (4, 6), "b": (6, 8)}
    assert len(reopened) == 1
    assert att._standalone_calls == 2


def test_historical_zeroer_resources_name_single_staging_pair():
    from kvguard.adapters.vllm.attach import _zeroer_resources

    class Tensor:
        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

    class Zeroer:
        _ids_pinned = Tensor(10)
        _ids_gpu = Tensor(20)

    assert _zeroer_resources(Zeroer()) == (
        "vllm-zero-pinned:10",
        "vllm-zero-device:20",
    )


def test_fixed_zeroer_resources_follow_rotating_pre_call_index():
    from kvguard.adapters.vllm.attach import _zeroer_resources

    class Tensor:
        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

    class Zeroer:
        def __init__(self):
            self._ids_pinned = [Tensor(10), Tensor(11)]
            self._ids_gpu = [Tensor(20), Tensor(21)]
            self._id_buffer_index = 1

    assert _zeroer_resources(Zeroer()) == (
        "vllm-zero-pinned:11",
        "vllm-zero-device:21",
    )


def test_historical_v2_pre_step_positions_are_recovered_from_sequence_lengths():
    """Older Model Runner V2 had no CPU ``num_computed_tokens_np`` field."""
    from kvguard.adapters.vllm.attach import _input_pre_step_computed

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Batch:
        seq_lens = DeviceArray([31, 81, 0])

    computed, fallback = _input_pre_step_computed(Batch(), 2, np.asarray([31, 17]))
    assert computed.tolist() == [0, 64]
    assert fallback


def test_current_v2_pre_step_positions_prefer_the_cpu_mirror():
    from kvguard.adapters.vllm.attach import _input_pre_step_computed

    class Batch:
        num_computed_tokens_np = np.asarray([0, 64, 999])

    computed, fallback = _input_pre_step_computed(Batch(), 2, np.asarray([31, 17]))
    assert computed.tolist() == [0, 64]
    assert not fallback


def test_speculative_v2_pre_step_positions_use_the_corrected_device_frontier():
    from kvguard.adapters.vllm.attach import _input_pre_step_computed

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Batch:
        num_computed_tokens_np = np.asarray([15, 80, 999])  # optimistic host mirror
        seq_lens = DeviceArray([16, 81, 0])

    computed, fallback = _input_pre_step_computed(Batch(), 2, np.asarray([6, 1]), force_device=True)
    assert computed.tolist() == [10, 80]
    assert fallback


def test_adaptive_verification_uses_the_device_query_lengths():
    from kvguard.adapters.vllm.attach import _input_execution_counts

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    batch = SimpleNamespace(
        num_scheduled_tokens=np.asarray([4, 4, 4]),
        max_query_len=4,
        query_start_loc=DeviceArray([0, 4, 6, 7]),
        num_tokens=7,
    )

    counts, adaptive = _input_execution_counts(batch, 3)

    assert adaptive
    assert counts.tolist() == [4, 2, 1]


def test_adaptive_verification_rejects_query_lengths_above_the_host_bounds():
    import pytest
    from kvguard.adapters.vllm.attach import _input_execution_counts

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    batch = SimpleNamespace(
        num_scheduled_tokens=np.asarray([4, 2]),
        max_query_len=4,
        query_start_loc=DeviceArray([0, 4, 7]),
        num_tokens=7,
    )

    with pytest.raises(ValueError, match="outside their CPU upper bounds"):
        _input_execution_counts(batch, 2)


def test_adaptive_verification_rejects_an_inconsistent_executed_total():
    import pytest
    from kvguard.adapters.vllm.attach import _input_execution_counts

    class DeviceArray:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, item):
            return DeviceArray(self.values[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    batch = SimpleNamespace(
        num_scheduled_tokens=np.asarray([4, 4]),
        max_query_len=4,
        query_start_loc=DeviceArray([0, 4, 6]),
        num_tokens=7,
    )

    with pytest.raises(ValueError, match="do not sum to the executed token count"):
        _input_execution_counts(batch, 2)


def test_pre_step_computed_undoes_the_schedule_time_advance():
    assert pre_step_computed(num_computed_after_schedule=160, num_scheduled=160) == 0
    assert pre_step_computed(161, 1) == 160


def test_block_state_rules():
    assert block_state(0, False, False) == (None, False)  # free
    assert block_state(0, True, False) == (
        BlockState.PUBLISHED,
        True,
    )  # cached, unreferenced: reserved
    assert block_state(1, False, False) == (BlockState.PRIVATE, False)
    assert block_state(1, True, False) == (
        BlockState.PRIVATE,
        False,
    )  # hashed but still being written
    assert block_state(1, True, True) == (BlockState.PUBLISHED, False)  # hashed and complete
    assert block_state(2, True, False) == (BlockState.PUBLISHED, False)  # same-step prefix hit


def test_build_snapshot_derives_states_and_truly_free_set():
    bs = 16
    reqs = [
        RequestFacts(
            "A", (1, 2, 3), num_computed_after_schedule=40, num_scheduled=8
        ),  # pre 32: blocks 1,2 full
        RequestFacts(
            "B", (1, 4), num_computed_after_schedule=17, num_scheduled=1
        ),  # hit on block 1
    ]
    blocks = [
        BlockFacts(0, 0, False, is_null=True),
        BlockFacts(1, 2, True),  # shared published prefix block
        BlockFacts(2, 1, True),  # A's second block, full and hashed
        BlockFacts(3, 1, False),  # A's third block, private, in progress
        BlockFacts(4, 1, False),  # B's own block
        BlockFacts(5, 0, True),  # cached, unreferenced: reserved
        BlockFacts(6, 0, False),  # free
    ]
    snap = build_snapshot(7, "gpu", bs, reqs, blocks, free_queue_ids=[5, 6], free_queue_hashed=[5])
    by_block = {b.block: b for b in snap.blocks}
    assert by_block[1].state is BlockState.PUBLISHED and by_block[1].refs == 2
    assert by_block[2].state is BlockState.PUBLISHED and by_block[2].refs == 1
    assert by_block[3].state is BlockState.PRIVATE
    assert by_block[4].state is BlockState.PRIVATE
    assert (
        by_block[5].state is BlockState.PUBLISHED and by_block[5].reserved and by_block[5].refs == 0
    )
    assert 6 not in by_block and 0 not in by_block
    assert snap.free == {"gpu": frozenset({6})}
    a = next(r for r in snap.requests if r.request == "A")
    assert (a.num_tokens, a.num_computed) == (40, 32)


def test_build_dispatches_orders_write_before_attention_and_keeps_padding_rows():
    bs = 4
    table = np.array([[3, 4, 0], [5, 0, 0], [0, 0, 0], [7, 0, 0]])
    write, attn = build_dispatches(
        3, "gpu", bs, ["A", "B"], [4, 0], [2, 3], table, num_reqs_after_padding=4,
        slot_mapping=[16, 17, 20, 21, 22, -1, -1],
    )  # fmt: skip
    assert isinstance(write.mapping, SlotMap)
    assert write.mapping.slots == [16, 17, 20, 21, 22, -1, -1]
    assert write.mapping.rows == [0, 0, 1, 1, 1, 2, 2]
    assert write.mapping.positions == [4, 5, 0, 1, 2, 0, 0]
    assert [r.writes for r in write.rows] == [range(4, 6), range(0, 3), range(0)]
    assert write.rows[-1].padding
    assert isinstance(attn.mapping, DenseBlockTable)
    assert attn.mapping.table == [[3, 4], [5], [], [7]] and attn.mapping.valid == [2, 1, 0, 1]
    assert [r.padding for r in attn.rows] == [False, False, True, True]
    assert [r.reads for r in attn.rows[:2]] == [range(0, 6), range(0, 3)]


def test_build_dispatches_does_not_describe_accesses_for_a_zero_query_row():
    table = np.array([[3, 4, 5, 6, 7]])
    write, attn = build_dispatches(
        4,
        "gpu",
        16,
        ["blocked"],
        [80],
        [0],
        table,
        num_reqs_after_padding=1,
        slot_mapping=[],
    )

    assert write.rows[0].writes == range(0)
    assert attn.rows[0].reads == range(0)
    assert attn.mapping == DenseBlockTable([[]], [0])


def test_positioned_mtp_dispatch_keeps_reprefill_positions_and_trailing_padding():
    table = np.array(
        [
            [3, 4, 5, 6, 7, 8],
            [9, 0, 0, 0, 0, 0],
        ]
    )
    write, attn = build_positioned_dispatches(
        9,
        "mtp2",
        4,
        ["A", "B"],
        [0, 4, 7],
        [20, 21, 22, 99, 0, 1, 2],
        # A's rejected trailing token remains in the attention seq_len, but its
        # slot is padded and it is causally beyond the live query at position 22.
        [24, 3],
        table,
        [32, 33, 34, -1, 36, 37, 38],
        window=8,
        site="mtp-0",
    )

    assert write.mapping == SlotMap(
        [32, 33, 34, -1, 36, 37, 38],
        [0, 0, 0, 0, 1, 1, 1],
        [20, 21, 22, 99, 0, 1, 2],
    )
    assert [row.writes for row in write.rows] == [range(20, 23), range(0, 3)]
    assert [row.reads for row in attn.rows] == [range(13, 23), range(0, 3)]
    assert attn.mapping == DenseBlockTable([[3, 4, 5, 6, 7, 8], [9]], [6, 1], [3, 0])


def test_positioned_mtp_dispatch_rejects_a_noncontiguous_live_query():
    import pytest

    with pytest.raises(ValueError, match="non-contiguous"):
        build_positioned_dispatches(
            1,
            "mtp",
            4,
            ["A"],
            [0, 2],
            [4, 6],
            [7],
            np.array([[1, 2]]),
            [4, 6],
        )


def test_mediated_frees_keeps_hashed_blocks_reserved():
    after = [
        {"block_id": 1, "ref_cnt": 0, "has_hash": False, "is_null": False},
        {"block_id": 2, "ref_cnt": 0, "has_hash": True, "is_null": False},
        {"block_id": 3, "ref_cnt": 1, "has_hash": False, "is_null": False},
    ]
    assert mediated_frees(after) == [1]


def test_snapshot_without_a_free_queue_carries_no_free_set():
    from kvguard.adapters.vllm.translate import BlockFacts, RequestFacts, build_snapshot

    snap = build_snapshot(
        1,
        "kv",
        16,
        [RequestFacts("A", (3,), 4, 4)],
        [BlockFacts(3, 1, False, False)],
        free_queue_ids=None,
    )
    assert snap.free is None and len(snap.blocks) == 1
    audited = build_snapshot(
        1,
        "kv",
        16,
        [RequestFacts("A", (3,), 4, 4)],
        [BlockFacts(3, 1, False, False)],
        free_queue_ids=[5, 6],
    )
    assert audited.free == {"kv": frozenset({5, 6})}


def test_mapping_tables_derive_the_step_tables_from_the_block_lists():
    """The host dispatch's tables from the KV manager's mapping: padded rows and holes are the
    null block, and each scheduled token's slot follows its position."""
    from kvguard.adapters.vllm.translate import mapping_tables

    table, slots = mapping_tables([[5, 6], [7]], [3, 0], [2, 1], 4, num_reqs_after_padding=3)
    assert table.tolist() == [[5, 6], [7, 0], [0, 0]]
    assert slots.tolist() == [5 * 4 + 3, 6 * 4 + 0, 7 * 4 + 0]
    table, slots = mapping_tables([[None, 9]], [4], [1], 4, 1, null_block=0)
    assert table.tolist() == [[0, 9]] and slots.tolist() == [9 * 4]


def test_state_columns_follow_the_align_mode_rule():
    """Column (pre - 1) // bs is read, ceil((pre + sched) / bs) - 1 written; a fresh request
    reads nothing; the 'none' mode keeps one column."""
    from kvguard.adapters.vllm.translate import state_columns

    bs = 544
    assert state_columns(0, 600, bs, "align") == (-1, 1)  # prefill into column 1, column 0 skipped
    assert state_columns(600, 1, bs, "align") == (1, 1)  # decode within the block
    assert state_columns(1088, 1, bs, "align") == (1, 2)  # the boundary: migrate 1 -> 2
    assert state_columns(1087, 1, bs, "align") == (1, 1)  # the last token of block 1
    assert state_columns(544, 10, bs, "align") == (0, 1)  # a hit of exactly one block
    assert state_columns(0, 5, bs, "none") == (-1, 0)
    assert state_columns(7, 1, bs, "none") == (0, 0)


def test_build_state_dispatch_reads_first_and_names_every_column_up_to_the_touched_one():
    from kvguard.adapters.vllm.translate import build_state_dispatch

    table = np.array([[0, 11, 12], [21, 0, 0], [0, 0, 0], [0, 5, 0]])
    d = build_state_dispatch(
        4, "state0", 544, "align", ["A", "B"], [1088, 0], [1, 300], table, num_reqs_after_padding=4
    )
    assert d.pool == "state0" and d.family == "recurrent_state"
    # A: reads column 1, writes column 2 (two rows, the read first); B: fresh, writes column 0.
    assert [(r.request, r.reads, r.writes) for r in d.rows[:3]] == [
        ("A", range(1, 2), range(0)),
        ("A", range(0), range(2, 3)),
        ("B", range(0), range(0, 1)),
    ]
    assert isinstance(d.mapping, DenseBlockTable)
    assert d.mapping.table[:3] == [[0, 11], [0, 11, 12], [21]] and d.mapping.valid[:3] == [2, 3, 1]
    # Each row consumes its own column only (the kernels gather one column per row).
    assert list(d.mapping.first) == [1, 2, 0, 0, 0]
    # Padding rows carry the gathered row's entries, so a non-null entry there is visible.
    assert [r.padding for r in d.rows[3:]] == [True, True]
    assert d.mapping.table[3:] == [[], [0, 5]] and d.mapping.valid[3:] == [0, 2]


def test_dense_table_and_cow_copy():
    from kvguard.adapters.vllm.translate import cow_copy, dense_table

    table = dense_table([[None, 7], [8]], 3, null_block=0)
    assert table.tolist() == [[0, 7], [8, 0], [0, 0]]
    # The attention group's partial hit covers the computed positions of the shared block.
    c = cow_copy("gpu", 7, 9, 544, num_computed=1088 + 32)
    assert (c.pool, c.block, c.dst_pool, c.dst_block) == ("gpu", 7, "gpu", 9)
    assert c.offsets == range(0, 32) and c.source == "cow" and c.request is None
    # A state group's block is one slot; an unknown count copies the block.
    assert cow_copy("state1", 7, 9, 544, None).offsets == range(0, 544)
    assert cow_copy("state1", 7, 9, 1, 600).offsets == range(0, 1)


def test_window_start_and_windowed_dispatches():
    """A sliding-window group's row attends the keys from start - window + 1 and consumes the
    table from that key's block; the dispatch ids carry the group's site."""
    from kvguard.adapters.vllm.translate import window_start

    assert window_start(0, 8) == 0 and window_start(36, 8) == 29 and window_start(36, 8, 2) == 27
    bs = 4
    table = np.array([[0, 0, 0, 0, 0, 0, 0, 7, 8, 9], [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]])
    write, attn = build_dispatches(
        5, "swa0", bs, ["A", "B"], [36, 0], [1, 6], table, 2, [9 * 4, 20, 21, 22, 23, 24, 25],
        window=8, site="swa0",
    )  # fmt: skip
    assert (write.dispatch_id, attn.dispatch_id) == ("write-swa0@5", "attn-swa0@5")
    assert [r.reads for r in attn.rows] == [range(29, 37), range(0, 6)]
    assert attn.mapping.valid == [10, 2] and list(attn.mapping.first) == [7, 0]
    assert attn.mapping.table[0][7:] == [7, 8, 9]
    full_write, full_attn = build_dispatches(5, "gpu", bs, ["A"], [36], [1], table[:1], 1, [36])
    assert full_attn.rows[0].reads == range(0, 37) and full_attn.mapping.first is None
    assert full_write.dispatch_id == "write@5"
