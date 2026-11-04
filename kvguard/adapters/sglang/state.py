"""The recurrent-state pool of a hybrid model under the SGLang attachment: mediation of the
Mamba slot allocator and pool, and the state pool's part of the snapshot.

A hybrid GDN model keeps, per request, one active state slot and (with
``enable_mamba_extra_buffer``) a ping-pong pair of tracking slots; the radix cache holds donated
slots as node values. The core sees them as one ``rewritable`` single-slot pool: slot 0 reserved,
a request mapping its active slot at position 0 and its tracking slots at positions 1 and 2, the
tree's slots published and reserved. The allocator's alloc/free, the pool's clear (an
initialization) and copy-on-write (a read plus an initialization) are mediated by instance-level
wraps. The int8 checkpoint store is opt-in.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kvguard.adapters.sglang.translate import wrap_allocator_clear
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    DispatchDescriptor,
    DispatchRow,
    ExternalCopy,
    ExternalFill,
    PoolGeometry,
    RequestSnapshot,
    SlotMap,
)
from kvguard.core.model import PADDING_SLOT

STATE_POOL = "state"
CKPT_POOL = "mamba_ckpt"
HOLE = -1


def _ints(x: Any) -> list[int]:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, int):
        return [x]
    return [int(v) for v in x]


def state_geometry(req_pool: Any, *, layers: int, state_storage: Any | None = None) -> PoolGeometry:
    """The state pool's geometry from a ``HybridReqToTokenPool``: ``mamba_pool.size`` slots
    numbered from 1, plus the reserved slot 0."""
    pool = state_storage.pool if state_storage is not None else req_pool.mamba_pool
    size = int(pool.size)
    return PoolGeometry(
        STATE_POOL,
        num_blocks=size + 1,
        block_size=1,
        layers=max(int(layers), 1),
        kv_heads=1,
        components=("conv", "ssm"),
        reserved_blocks=frozenset({0}),
        rewritable=True,
    )


def ckpt_geometry(ckpt: Any, *, layers: int) -> PoolGeometry:
    """The int8 checkpoint store's geometry: ``num_slots`` slots from 1, slot 0 reserved."""
    size = int(ckpt.num_slots)
    return PoolGeometry(
        CKPT_POOL,
        num_blocks=size + 1,
        block_size=1,
        layers=max(int(layers), 1),
        kv_heads=1,
        components=("conv", "ssm_int8"),
        reserved_blocks=frozenset({0}),
        rewritable=True,
    )


def ckpt_snapshot(
    *, free_slots: Iterable[int], tree_slots: Iterable[int], size: int
) -> tuple[list[BlockSnapshot], frozenset[int]]:
    """The checkpoint pool's blocks and free set: no request maps a checkpoint slot; a slot
    the tree holds is published and reserved, any other allocated slot private and unmapped."""
    free = frozenset(int(s) for s in free_slots if s != 0)
    tree = frozenset(int(s) for s in tree_slots if s > 0)
    blocks = [
        BlockSnapshot(CKPT_POOL, slot, BlockState.PUBLISHED, reserved=True)
        if slot in tree
        else BlockSnapshot(CKPT_POOL, slot, BlockState.PRIVATE)
        for slot in range(1, size + 1)
        if slot not in free
    ]
    return blocks, free


def state_snapshot(
    *,
    rid_of_row: Mapping[int, str],
    active_of_row: Sequence[int],
    tracks_of_row: Sequence[Sequence[int]] | None,
    free_slots: Iterable[int],
    tree_slots: Iterable[int],
    size: int,
) -> tuple[list[RequestSnapshot], list[BlockSnapshot], frozenset[int]]:
    """The state pool's requests, blocks and free set for one step.

    A request maps its active slot (position 0) and its tracking slots (positions 1 and 2; holes
    skipped); ``num_computed`` is 0 since initialization is recorded by the clear, copy and
    dispatches. Every slot outside the free list is listed: private when a request maps it,
    published when the tree holds it (reserved when unmapped), and private with no mapping
    otherwise (reported as a leaked slot)."""
    free = frozenset(int(s) for s in free_slots if s != 0)
    tree = frozenset(int(s) for s in tree_slots if s > 0)
    requests: list[RequestSnapshot] = []
    mapped: set[int] = set()
    for row, rid in rid_of_row.items():
        if row >= len(active_of_row):
            continue
        active = int(active_of_row[row])
        if active <= 0:
            continue  # the row holds no state (slot 0 is the padding target)
        blocks = [active]
        if tracks_of_row is not None and row < len(tracks_of_row):
            blocks.extend(int(t) for t in tracks_of_row[row] if int(t) > 0)
        requests.append(RequestSnapshot(rid, STATE_POOL, tuple(blocks), len(blocks), 0, 0))
        mapped.update(blocks)
    blocks_out: list[BlockSnapshot] = []
    for slot in range(1, size + 1):
        if slot in free:
            continue
        if slot in tree:
            blocks_out.append(
                BlockSnapshot(STATE_POOL, slot, BlockState.PUBLISHED, reserved=slot not in mapped)
            )
        else:
            blocks_out.append(BlockSnapshot(STATE_POOL, slot, BlockState.PRIVATE))
    return requests, blocks_out, free


def build_state_dispatch(
    seq: int,
    rids: Sequence[str | None],
    slots: Sequence[int],
    track_rows: Sequence[int] = (),
    track_slots: Sequence[int] = (),
    tracks_of_row: Sequence[Sequence[int]] | None = None,
    *,
    site: str = "state",
) -> DispatchDescriptor:
    """The state pool's dispatch of one step: row ``i`` (request ``rids[i]``, ``None`` for a
    padded row) reads and writes position 0 through ``slots[i]``, the active slot the GDN kernels
    consume. A tracked row (``track_rows``) also writes its tracking slot ``track_slots[i]`` at
    position 1 + k, where k is that slot's index in the row's tracking buffer (``tracks_of_row``;
    0 when unknown), as one more row of the same request."""
    rows: list[DispatchRow] = []
    map_slots: list[int] = []
    map_rows: list[int] = []
    map_positions: list[int] = []
    for i, (rid, s) in enumerate(zip(rids, slots, strict=False)):
        s = int(s)
        if rid is None:
            rows.append(DispatchRow(None, padding=True))
            map_slots.append(PADDING_SLOT if s < 0 else s)
        else:
            rows.append(DispatchRow(rid, reads=range(0, 1), writes=range(0, 1)))
            map_slots.append(s)
        map_rows.append(i)
        map_positions.append(0)
    for i in track_rows:
        if not (0 <= i < len(track_slots) and 0 <= i < len(rids)) or rids[i] is None:
            continue
        t = int(track_slots[i])
        k = 0
        if tracks_of_row is not None and i < len(tracks_of_row):
            buf = [int(x) for x in tracks_of_row[i]]
            if t in buf:
                k = buf.index(t)
        pos = 1 + k
        rows.append(DispatchRow(rids[i], writes=range(pos, pos + 1)))
        map_slots.append(t)
        map_rows.append(len(rows) - 1)
        map_positions.append(pos)
    return DispatchDescriptor(
        f"{site}@{seq}", STATE_POOL, "gdn", tuple(rows), SlotMap(map_slots, map_rows, map_positions)
    )


class StateMediation:
    """The wraps on one live ``HybridReqToTokenPool`` and the per-step reads."""

    def __init__(
        self,
        att: Any,
        req_pool: Any,
        tree_core: Any,
        *,
        state_storage: Any | None = None,
        state_band: Any | None = None,
    ):
        self.att = att
        self.req_pool = req_pool
        self.unified = state_storage is not None
        self.storage = state_storage
        self.band = state_band
        self.allocator = state_storage.slots if self.unified else req_pool.mamba_allocator
        self.pool = state_storage.pool if self.unified else req_pool.mamba_pool
        self.tree_core = tree_core
        self.size = int(self.pool.size)
        self.ckpt = (
            getattr(state_storage, "ckpt_pool", None)
            if self.unified
            else getattr(req_pool, "mamba_ckpt_pool", None)
        )
        """The int8 checkpoint store (``--enable-int8-mamba-checkpoint``) or None: with it the
        tree's values are checkpoint slots, a donation quantizes an active slot into one and a
        hit dequantizes one into the request's slot."""
        self.ckpt_size = int(self.ckpt.num_slots) if self.ckpt is not None else 0
        self.extra_buffer = bool(getattr(req_pool, "enable_mamba_extra_buffer", False))
        self.stats: Counter = att.stats
        self.trace_enabled = os.environ.get("KVG_STATE_TRACE", "0") == "1"
        self.slot_history: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=24))
        self.trace_anomalies: list[dict[str, Any]] = []
        self.trace_anomaly_slots: set[int] = set()
        self.tracks_by_row: list[list[int]] | None = None
        self.ckpt_free: frozenset[int] = frozenset()
        self._install()

    def _logical_slots(self, slots: Any) -> list[int]:
        """Return stable virtual state ids for physical GDN/store indices."""
        values = _ints(slots)
        if not getattr(self, "unified", False):
            return values
        p2v = _ints(self.band.ids.physical_to_virtual)
        return [p2v[value] if 0 <= value < len(p2v) else -1 for value in values]

    def _trace_slots(self, kind: str, slots: Iterable[int]) -> None:
        if not getattr(self, "trace_enabled", False):
            return
        event = {"seq": self.att.seq, "kind": kind}
        for slot in slots:
            slot = int(slot)
            if slot > 0:
                self.slot_history[slot].append(event)

    # ------------------------------------------------------------------ mediation
    def _install(self) -> None:
        me, alloc, pool = self, self.allocator, self.pool
        if self.unified:
            band = self.band
            orig_alloc = band.alloc

            def band_alloc(need_size):
                out = orig_alloc(need_size)
                if out is not None:
                    ids = _ints(out)
                    if ids:
                        me.att.ledger.allocate(STATE_POOL, ids, site=f"state_alloc@{me.att.seq}")
                        me._trace_slots("allocate", ids)
                        me.stats["state_alloc_slots"] += len(ids)
                        me.stats["state_alloc_calls"] += 1
                return out

            band.alloc = band_alloc
            orig_free = band.free

            def band_free(free_index, *args, **kwargs):
                before = set(_ints(band.free_virtual_ids))
                out = orig_free(free_index, *args, **kwargs)
                ids = sorted(set(_ints(band.free_virtual_ids)) - before)
                if ids:
                    me.att.ledger.free(STATE_POOL, ids, site=f"state_free@{me.att.seq}")
                    me._trace_slots("free", ids)
                    me.stats["state_free_slots"] += len(ids)
                    me.stats["state_free_calls"] += 1
                return out

            band.free = band_free
            orig_clear_band = band.clear

            def clear_band(*args, **kwargs):
                before = set(_ints(band.free_virtual_ids))
                out = orig_clear_band(*args, **kwargs)
                released = sorted(set(_ints(band.free_virtual_ids)) - before)
                if released:
                    me.att.ledger.free(
                        STATE_POOL,
                        released,
                        site=f"state_allocator_clear@{me.att.seq}",
                    )
                    me._trace_slots("allocator_clear", released)
                    me.stats["state_allocator_clear_calls"] += 1
                    me.stats["state_allocator_clear_slots"] += len(released)
                return out

            band.clear = clear_band
        else:
            orig_do_alloc = alloc._do_alloc

            def _do_alloc(need_size):
                out = orig_do_alloc(need_size)
                if out is not None:
                    ids = _ints(out)
                    if ids:
                        me.att.ledger.allocate(STATE_POOL, ids, site=f"state_alloc@{me.att.seq}")
                        me._trace_slots("allocate", ids)
                        me.stats["state_alloc_slots"] += len(ids)
                        me.stats["state_alloc_calls"] += 1
                return out

            alloc._do_alloc = _do_alloc
            orig_free = alloc.free

            def free(free_index):
                ids = _ints(free_index)
                orig_free(free_index)
                if ids:
                    me.att.ledger.free(STATE_POOL, ids, site=f"state_free@{me.att.seq}")
                    me._trace_slots("free", ids)
                    me.stats["state_free_slots"] += len(ids)
                    me.stats["state_free_calls"] += 1

            alloc.free = free

            def state_allocator_cleared(released: list[int]) -> None:
                me.att.ledger.free(STATE_POOL, released, site=f"state_allocator_clear@{me.att.seq}")
                me._trace_slots("allocator_clear", released)
                me.stats["state_allocator_clear_calls"] += 1
                me.stats["state_allocator_clear_slots"] += len(released)

            wrap_allocator_clear(
                alloc,
                lambda: _ints(alloc.free_slots),
                state_allocator_cleared,
            )
        orig_clear = pool.clear_slots

        def clear_slots(indices):
            out = orig_clear(indices)
            site = f"state_clear@{me.att.seq}"
            logical = me._logical_slots(indices)
            for b in logical:
                if b > 0:
                    me.att.ledger.external_fill(
                        ExternalFill(STATE_POOL, b, range(0, 1), source="clear"), site=site
                    )
            me._trace_slots("clear", logical)
            me.stats["state_clears"] += 1
            return out

        pool.clear_slots = clear_slots
        orig_copy = pool.copy_from

        def copy_from(src_indices, dst_indices):
            out = orig_copy(src_indices, dst_indices)
            site = f"state_copy@{me.att.seq}"
            logical_src = me._logical_slots(src_indices)
            logical_dst = me._logical_slots(dst_indices)
            for s, d in zip(logical_src, logical_dst, strict=False):
                me.att.ledger.external_copy(
                    ExternalCopy(STATE_POOL, s, STATE_POOL, d, range(0, 1), source="copy"),
                    site=site,
                )
                me.stats["state_copies"] += 1
            me._trace_slots("copy_source", logical_src)
            me._trace_slots("copy_destination", logical_dst)
            return out

        pool.copy_from = copy_from
        ckpt = self.ckpt
        if ckpt is not None:
            calloc = ckpt.allocator
            orig_cdo = calloc._do_alloc

            def ckpt_do_alloc(need_size):
                out = orig_cdo(need_size)
                if out is not None:
                    ids = _ints(out)
                    if ids:
                        me.att.ledger.allocate(CKPT_POOL, ids, site=f"ckpt_alloc@{me.att.seq}")
                        me.stats["ckpt_alloc_slots"] += len(ids)
                return out

            calloc._do_alloc = ckpt_do_alloc
            orig_cfree = calloc.free

            def ckpt_free(free_index):
                ids = _ints(free_index)
                orig_cfree(free_index)
                if ids:
                    me.att.ledger.free(CKPT_POOL, ids, site=f"ckpt_free@{me.att.seq}")
                    me.stats["ckpt_free_slots"] += len(ids)

            calloc.free = ckpt_free

            def checkpoint_allocator_cleared(released: list[int]) -> None:
                me.att.ledger.free(CKPT_POOL, released, site=f"ckpt_allocator_clear@{me.att.seq}")
                me.stats["ckpt_allocator_clear_slots"] += len(released)

            wrap_allocator_clear(
                calloc,
                lambda: _ints(calloc.free_slots),
                checkpoint_allocator_cleared,
            )
            orig_store = ckpt.store_from_active

            def store_from_active(active_pool, active_slots, ckpt_slots):
                out = orig_store(active_pool, active_slots, ckpt_slots)
                site = f"ckpt_store@{me.att.seq}"
                for s, d in zip(_ints(active_slots), _ints(ckpt_slots), strict=False):
                    me.att.ledger.external_copy(
                        ExternalCopy(STATE_POOL, s, CKPT_POOL, d, range(0, 1), source="quantize"),
                        site=site,
                    )
                    me.stats["ckpt_stores"] += 1
                return out

            ckpt.store_from_active = store_from_active
            orig_load = ckpt.load_to_active

            def load_to_active(active_pool, ckpt_slots, active_slots):
                out = orig_load(active_pool, ckpt_slots, active_slots)
                site = f"ckpt_load@{me.att.seq}"
                for s, d in zip(_ints(ckpt_slots), _ints(active_slots), strict=False):
                    me.att.ledger.external_copy(
                        ExternalCopy(CKPT_POOL, s, STATE_POOL, d, range(0, 1), source="dequantize"),
                        site=site,
                    )
                    me.stats["ckpt_loads"] += 1
                return out

            ckpt.load_to_active = load_to_active

    # ------------------------------------------------------------------ the snapshot
    def snapshot(self, rid_of_row: Mapping[int, str]):
        rp = self.req_pool
        active = _ints(rp.req_index_to_mamba_index_mapping)
        tracks = None
        if self.extra_buffer:
            pp = rp.req_index_to_mamba_ping_pong_track_buffer_mapping
            tracks = pp.tolist() if hasattr(pp, "tolist") else [list(r) for r in pp]
        self.tracks_by_row = tracks
        free = (
            _ints(self.band.free_virtual_ids) if self.unified else _ints(self.allocator.free_slots)
        )
        tree = _ints(self.tree_core.all_mamba_values_flatten())
        if self.trace_enabled:
            mapped = {slot for slot in active if slot > 0}
            if tracks is not None:
                mapped.update(int(slot) for row in tracks for slot in row if int(slot) > 0)
            owned = mapped | {int(slot) for slot in tree if int(slot) > 0} | set(free)
            unowned = sorted(set(range(1, self.size + 1)).difference(owned))
            if unowned and len(self.trace_anomalies) < 32:
                waiting_owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for req in getattr(self.att.scheduler, "waiting_queue", ()):
                    req_slots = []
                    active_slot = getattr(req.kv, "mamba_pool_idx", None)
                    if active_slot is not None:
                        req_slots.extend((int(slot), "active") for slot in _ints(active_slot))
                    track_slots = getattr(req.kv, "mamba_ping_pong_track_buffer", None)
                    if track_slots is not None:
                        req_slots.extend((int(slot), "track") for slot in _ints(track_slots))
                    for slot, role in req_slots:
                        if slot in unowned:
                            waiting_owners[str(slot)].append(
                                {"request": str(req.rid), "role": role}
                            )
                self.trace_anomaly_slots.update(unowned)
                self.trace_anomalies.append(
                    {
                        "seq": self.att.seq,
                        "slots": unowned,
                        "waiting_owners": dict(waiting_owners),
                        "history": {
                            str(slot): list(self.slot_history.get(slot, ())) for slot in unowned
                        },
                    }
                )
        if self.ckpt is None:
            requests, blocks, s_free = state_snapshot(
                rid_of_row=rid_of_row,
                active_of_row=active,
                tracks_of_row=tracks,
                free_slots=free,
                tree_slots=tree,
                size=self.size,
            )
            requests.extend(self._waiting_requests(set(rid_of_row.values())))
            requests.extend(self._pending_load_requests(tree))
            return requests, blocks, s_free
        # With the int8 store the tree's values are checkpoint slots: the state pool lists no
        # tree slot, the checkpoint pool lists them.
        requests, blocks, s_free = state_snapshot(
            rid_of_row=rid_of_row,
            active_of_row=active,
            tracks_of_row=tracks,
            free_slots=free,
            tree_slots=(),
            size=self.size,
        )
        requests.extend(self._waiting_requests(set(rid_of_row.values())))
        requests.extend(self._pending_load_requests(tree))
        c_blocks, c_free = ckpt_snapshot(
            free_slots=_ints(self.ckpt.allocator.free_slots), tree_slots=tree, size=self.ckpt_size
        )
        self.ckpt_free = c_free
        return requests, blocks + c_blocks, s_free

    def _pending_load_requests(self, tree_slots: Iterable[int]) -> list[RequestSnapshot]:
        """Temporary owners of buffer-mode state checkpoints published before H2D completes.

        Buffer load-back inserts its node checkpoint before the layer-gated copy; the checkpoint
        has no request row, so naming the transfer as its temporary reader carries the pending
        initialization until the completion event applies the fill.
        """
        tree = {int(slot) for slot in tree_slots if int(slot) > 0}
        out: list[RequestSnapshot] = []
        for key, (_event, pools) in getattr(self.att, "_hicache_loads", {}).items():
            for pool, slots in pools:
                if pool != STATE_POOL:
                    continue
                pending = tuple(int(slot) for slot in slots if int(slot) in tree)
                if pending:
                    out.append(
                        RequestSnapshot(
                            f"hicache-load:{key}:state",
                            STATE_POOL,
                            pending,
                            len(pending),
                            0,
                            0,
                        )
                    )
        if out:
            self.stats["state_snapshot_pending_loads"] += len(out)
        return out

    def _waiting_requests(self, already_seen: set[str]) -> list[RequestSnapshot]:
        """State reservations held by requests still in the admission queue.

        Buffer-mode admission may allocate a request's active slot one scheduler turn before it
        joins a forward batch, so it has no KV row yet but still owns the state slot. Use the same
        incarnation label the row will receive on admission.
        """
        out: list[RequestSnapshot] = []
        for req in getattr(self.att.scheduler, "waiting_queue", ()):
            kv = req.kv
            slots = _ints(getattr(kv, "mamba_pool_idx", None))
            tracks = getattr(kv, "mamba_ping_pong_track_buffer", None)
            if tracks is not None:
                slots.extend(_ints(tracks))
            slots = list(dict.fromkeys(slot for slot in slots if slot > 0))
            if not slots:
                continue
            request = self.att._request_owner(req)
            if request in already_seen:
                continue
            already_seen.add(request)
            out.append(RequestSnapshot(request, STATE_POOL, tuple(slots), len(slots), 0, 0))
        if out:
            self.stats["state_snapshot_waiting_requests"] += len(out)
        return out

    # ------------------------------------------------------------------ the dispatch
    def pending_fills(self, fb: Any) -> None:
        """The clears and copies the runner will execute on the forward stream before this
        step's linear layers, recorded at the pre-kernel point so the dispatch sees the fresh
        slots initialized; the wraps record them again when they run (an update in a rewritable
        pool)."""
        site = f"state_pending@{self.att.seq}"
        for b in _ints(getattr(fb, "mamba_clear_indices", None)):
            if b > 0:
                self.att.ledger.external_fill(
                    ExternalFill(STATE_POOL, b, range(0, 1), source="clear"), site=site
                )
        src = _ints(getattr(fb, "mamba_cow_src_indices", None))
        dst = _ints(getattr(fb, "mamba_cow_dst_indices", None))
        src_pool = CKPT_POOL if self.ckpt is not None else STATE_POOL  # int8: a dequantization
        for s, d in zip(src, dst, strict=False):
            self.att.ledger.external_copy(
                ExternalCopy(src_pool, s, STATE_POOL, d, range(0, 1), source="copy"), site=site
            )
        if src or dst:
            self.stats["state_pending_copies"] += len(src)

    def descriptor(
        self, fb: Any, rids: Sequence[str | None], linear_backend: Any
    ) -> DispatchDescriptor | None:
        """The step's state dispatch from the exact GDN metadata consumed by this forward.

        Eager execution and decode-graph replay consume ``forward_metadata``; a captured GDN
        prefill extend instead consumes ``prefill_graph_metadata`` directly. Select the graph
        tables only for an extend replay (target verification takes the ordinary path).
        """
        mode = getattr(fb, "forward_mode", None)
        graph_md = getattr(linear_backend, "prefill_graph_metadata", None)
        use_prefill_graph = (
            graph_md is not None
            and mode is not None
            and mode.is_extend()
            and not mode.is_target_verify()
        )
        md = graph_md if use_prefill_graph else getattr(linear_backend, "forward_metadata", None)
        indices = (
            getattr(md, "cache_indices", None)
            if use_prefill_graph
            else getattr(md, "mamba_cache_indices", None)
        )
        if md is None or indices is None:
            self.stats["state_dispatch_without_metadata"] += 1
            return None
        n = len(rids)
        slots = self._logical_slots(indices)
        if use_prefill_graph:
            # Padding rows in the captured tables carry PAD_SLOT_ID and every graph kernel
            # returns for them. The dispatch needs only the real request rows.
            slots = slots[:n]
        if len(slots) < n:
            self.stats["state_dispatch_short_indices"] += 1
            n = len(slots)
        rids = list(rids[:n]) + [None] * (len(slots) - n)
        track_rows: list[int] = []
        track_slots: list[int] = []
        if use_prefill_graph:
            track_modes = _ints(getattr(md, "track_mode", None))[:n]
            track_rows = [i for i, selected in enumerate(track_modes) if selected]
            track_slots = self._logical_slots(getattr(md, "track_dst", None))[:n]
            self.stats["state_prefill_graph_dispatches"] += 1
            if track_rows:
                self.stats["state_tracked_rows"] += len(track_rows)
                self.stats["state_prefill_graph_tracked_rows"] += len(track_rows)
        elif getattr(md, "has_mamba_track_mask", False):
            track_rows = _ints(getattr(md, "mamba_track_mask_indices", None))
            track_slots = self._logical_slots(getattr(md, "mamba_track_indices", None))
            self.stats["state_tracked_rows"] += len(track_rows)
        elif getattr(fb, "mamba_track_mask", None) is not None:
            # Decode graphs capture the track-save kernel against a mutable mask; replay metadata
            # leaves ``has_mamba_track_mask`` false, so the live mask on the replay batch is
            # authoritative for the slots the captured kernel writes.
            mask = _ints(fb.mamba_track_mask)
            track_rows = [i for i, selected in enumerate(mask[:n]) if selected]
            if track_rows:
                track_slots = self._logical_slots(getattr(md, "mamba_track_indices", None))
                self.stats["state_tracked_rows"] += len(track_rows)
                self.stats["state_graph_tracked_rows"] += len(track_rows)
        tracks = None
        if self.tracks_by_row is not None:
            row_ids = _ints(fb.req_pool_indices)
            tracks = [
                self.tracks_by_row[r] if 0 <= r < len(self.tracks_by_row) else []
                for r in row_ids[: len(slots)]
            ]
        descriptor = build_state_dispatch(
            self.att.seq, rids, slots, track_rows, track_slots, tracks, site="state"
        )
        self._trace_slots("dispatch_active", slots[:n])
        self._trace_slots(
            "dispatch_track",
            (track_slots[row] for row in track_rows if 0 <= row < len(track_slots)),
        )
        return descriptor

    def record(self) -> dict[str, Any]:
        out = {
            "size": self.size,
            "ckpt_size": self.ckpt_size,
            "extra_buffer": self.extra_buffer,
            "allocator": type(self.allocator).__name__,
            "pool": type(self.pool).__name__,
            "unified": self.unified,
        }
        if self.trace_enabled:
            out["trace_anomalies"] = self.trace_anomalies
            out["trace_slot_history_final"] = {
                str(slot): list(self.slot_history.get(slot, ()))
                for slot in sorted(self.trace_anomaly_slots)
            }
        return out


__all__ = [
    "CKPT_POOL",
    "HOLE",
    "STATE_POOL",
    "StateMediation",
    "build_state_dispatch",
    "ckpt_geometry",
    "ckpt_snapshot",
    "state_geometry",
    "state_snapshot",
]
