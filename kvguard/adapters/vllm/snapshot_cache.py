"""The snapshot's parts kept between steps, so an ordinary step costs what changed.

Per request: the tuple of block ids built from its block list (extended by the list's new tail
while the list object and its prefix are unchanged) and its *coverage*, how many leading blocks
it has fully computed. Per block: the facts the pool exposes (``ref_cnt``, whether it carries a
hash) plus whether any live request covers it, and the ``BlockSnapshot`` derived from them, kept
as the same object while the facts hold so the ledger's ``observe`` can skip it.

The pool's facts change only through its methods, which the attachment wraps to ``mark`` blocks
dirty; an ordinary step re-reads only the dirty blocks. Every audited step re-reads every block
(``resync``) and counts the blocks whose cached snapshot the full read contradicts (*drift*),
correcting the cache from the full read.

One cache serves one core pool. On a hybrid model several KV cache groups share the block pool:
each group's cache lists only its own blocks (``member``); a recurrent-state group emits its
requests in ``columns`` and rebuilds each request's tuple per step (``rebuild``) since its lists
are mutated in place; a sliding-window group (``window``) emits each request with the window's
start as its first live position.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from kvguard.core import BlockId, BlockSnapshot, RequestId, RequestSnapshot

from .translate import block_state, pre_step_computed, window_start


class SnapshotCache:
    def __init__(
        self,
        pool: str,
        block_size: int,
        stats: Counter | None = None,
        *,
        member: Callable[[BlockId], bool] | None = None,
        columns: bool = False,
        rebuild: bool = False,
        window: tuple[int, int] | None = None,
    ):
        self.pool = pool
        self.block_size = block_size
        self.stats: Counter = stats if stats is not None else Counter()
        self.member = member
        """Which blocks belong to this cache's pool (a group of a shared block pool); None when
        the pool is the whole block pool."""
        self.columns = columns
        """Emit request snapshots in columns: ``num_tokens`` the list's length, ``num_computed`` 0."""
        self.rebuild = rebuild
        """Rebuild every request's tuple from its list each step (lists mutated in place)."""
        self.window = window
        """``(sliding_window, extra_retained_tokens)`` of a sliding-window group: a request's
        ``first_token`` is ``window_start`` of its pre-step computed count."""
        self.valid = False
        """False until the first full read (``resync``) and after ``invalidate``."""
        self.listed: dict[BlockId, BlockSnapshot] = {}
        self.changed: dict[BlockId, BlockSnapshot] = {}
        """The listed objects created since the last ``blocks()`` or ``delta()``."""
        self.gone: set[BlockId] = set()
        """The blocks dropped from the listing since the last ``blocks()`` or ``delta()``."""
        self.facts: dict[BlockId, tuple[int, bool, bool]] = {}
        """Per listed block: (ref_cnt, has_hash, covered) its snapshot was derived from."""
        self.cover: dict[BlockId, int] = {}
        """Per block: the live requests that have fully computed it within their mapping."""
        self.dirty: set[BlockId] = set()
        self._req: dict[RequestId, tuple[Any, int, Any, tuple[BlockId | None, ...], int]] = {}
        """Per request: (list object, its length, its last element, the ids, the coverage)."""

    # ------------------------------------------------------------------ marks
    def mark(self, block_ids: Iterable[BlockId]) -> None:
        self.dirty.update(block_ids)

    def invalidate(self) -> None:
        self.valid = False

    # ------------------------------------------------------------------ requests
    def requests_step(
        self,
        entries: Iterable[tuple[RequestId, Any, int, int]],
        mutated: set[RequestId] | None = None,
    ) -> list[RequestSnapshot]:
        """The request snapshots of this step from ``(request id, block list, num_computed_tokens
        after schedule, num_scheduled)`` per live request with blocks; coverage moves with the
        computed count and requests that left withdraw theirs. With ``rebuild``, ``mutated`` names
        the requests whose lists changed in place (``None`` rebuilds every request)."""
        out: list[RequestSnapshot] = []
        live: set[RequestId] = set()
        bs = self.block_size
        columns = self.columns
        window = self.window
        for rid, blocks, computed, scheduled in entries:
            live.add(rid)
            ids, k_old = self._ids(rid, blocks, mutated is None or rid in mutated)
            pre = pre_step_computed(computed, scheduled)
            k = min(pre // bs, len(ids))
            if k != k_old:
                self._cover(ids, k_old, k)
                e = self._req[rid]
                self._req[rid] = (e[0], e[1], e[2], e[3], k)
            if columns:
                out.append(RequestSnapshot(rid, self.pool, ids, len(ids), 0))
            elif window is not None:
                first = window_start(pre, window[0], window[1])
                out.append(RequestSnapshot(rid, self.pool, ids, pre + scheduled, pre, first))
            else:
                out.append(RequestSnapshot(rid, self.pool, ids, pre + scheduled, pre))
        if len(live) != len(self._req):
            for rid in list(self._req.keys() - live):
                e = self._req.pop(rid)
                self._cover(e[3], e[4], 0)
        return out

    def _ids(
        self, rid: RequestId, blocks: Any, mutated: bool = True
    ) -> tuple[tuple[BlockId | None, ...], int]:
        n = len(blocks)
        e = self._req.get(rid)
        if e is not None:
            lst, m, last, ids, k = e
            if self.rebuild and mutated:
                new_ids = tuple(None if b.is_null else b.block_id for b in blocks)
                if new_ids == ids:
                    self.stats["snapshot_requests_reused"] += 1
                    return ids, k
                self._cover(ids, k, 0)
                self.dirty.update(b for b in new_ids if b is not None)
                self._req[rid] = (blocks, n, blocks[n - 1] if n else None, new_ids, 0)
                self.stats["snapshot_requests_rebuilt"] += 1
                return new_ids, 0
            if lst is blocks and n >= m and (m == 0 or blocks[m - 1] is last):
                if n > m:
                    tail = tuple(None if b.is_null else b.block_id for b in blocks[m:])
                    self.dirty.update(b for b in tail if b is not None)
                    ids = ids + tail
                    self._req[rid] = (blocks, n, blocks[n - 1], ids, k)
                    self.stats["snapshot_requests_extended"] += 1
                else:
                    self.stats["snapshot_requests_reused"] += 1
                return ids, k
            self._cover(ids, k, 0)  # the old mapping's coverage is withdrawn
        ids = tuple(None if b.is_null else b.block_id for b in blocks)
        self.dirty.update(b for b in ids if b is not None)  # every block a mapping names is read
        self._req[rid] = (blocks, n, blocks[n - 1] if n else None, ids, 0)
        self.stats["snapshot_requests_rebuilt"] += 1
        return ids, 0

    def _cover(self, ids: tuple[BlockId | None, ...], k_from: int, k_to: int) -> None:
        cover, dirty = self.cover, self.dirty
        if k_to > k_from:
            for b in ids[k_from:k_to]:
                if b is not None:
                    cover[b] = cover.get(b, 0) + 1
                    dirty.add(b)
        else:
            for b in ids[k_to:k_from]:
                if b is not None:
                    c = cover.get(b, 0) - 1
                    if c > 0:
                        cover[b] = c
                    else:
                        cover.pop(b, None)
                    dirty.add(b)

    # ------------------------------------------------------------------ blocks
    def blocks_step(self, block_of: Callable[[BlockId], Any]) -> None:
        """Refresh the dirty blocks from the pool (``block_of``: the pool's block object by id)."""
        dirty = self.dirty
        if not dirty:
            return
        self.stats["snapshot_dirty_blocks"] += len(dirty)
        refresh = self._refresh
        for b in dirty:
            refresh(b, block_of(b))
        dirty.clear()

    def _refresh(self, b: BlockId, obj: Any) -> bool:
        """Re-derive one block's snapshot from the pool object; True when it changed."""
        if obj.is_null or (self.member is not None and not self.member(b)):
            self.facts.pop(b, None)
            return self._drop(b)
        rc = obj.ref_cnt
        f = (rc, obj.block_hash is not None, b in self.cover)
        if self.facts.get(b) == f:
            return False
        state, reserved = block_state(*f)
        if state is None:
            self.facts.pop(b, None)
            return self._drop(b)
        self.facts[b] = f
        self.listed[b] = self.changed[b] = BlockSnapshot(
            self.pool, b, state, reserved=reserved, refs=max(rc, 0)
        )
        self.gone.discard(b)
        return True

    def _drop(self, b: BlockId) -> bool:
        if self.listed.pop(b, None) is None:
            return False
        self.changed.pop(b, None)
        self.gone.add(b)
        return True

    def resync(
        self, block_of: Callable[[BlockId], Any], objects: Iterable[Any]
    ) -> tuple[int, str | None]:
        """Apply the dirty set, then re-read every block of ``objects`` (the whole pool) and
        correct the cache. Returns the number of blocks the full read contradicted (zero while
        the cache was not valid: the first fill) and an example."""
        self.blocks_step(block_of)
        before = dict(self.listed) if self.valid else None
        refresh = self._refresh
        changed: list[BlockId] = []
        for obj in objects:
            if refresh(obj.block_id, obj):
                changed.append(obj.block_id)
        drift = 0
        example = None
        if before is not None:
            for b in changed:
                drift += 1
                if example is None:
                    example = f"block {b}: {before.get(b)} read as {self.listed.get(b)}"
        self.valid = True
        self.stats["snapshot_resyncs"] += 1
        self.stats["snapshot_drift"] += drift
        return drift, example

    def blocks(self) -> tuple[BlockSnapshot, ...]:
        """The complete listing; it resets the change since the last listing."""
        self.changed.clear()
        self.gone.clear()
        return tuple(self.listed.values())

    def delta(self) -> tuple[tuple[BlockSnapshot, ...], frozenset[BlockId]]:
        """The change since the last ``blocks()`` or ``delta()``: the objects created (new
        blocks and blocks whose facts changed) and the blocks dropped, for a snapshot that
        lists the pool as its change (``Snapshot.delta_pools``); then cleared."""
        changed = tuple(self.changed.values())
        gone = frozenset(self.gone)
        self.changed.clear()
        self.gone.clear()
        return changed, gone

    def ids(self, rid: RequestId) -> tuple[BlockId | None, ...]:
        """The request's block ids as the last ``requests_step`` read them (``None`` for a
        hole), or an empty tuple: the step's dispatch reads the mapping from here instead of
        the block objects again."""
        e = self._req.get(rid)
        return e[3] if e is not None else ()

    def held(self) -> int:
        """The listed blocks that hold a reference (not reserved): the pool's share of the
        allocator's non-free blocks, for the partition audit by count over a shared block pool."""
        return sum(1 for bs in self.listed.values() if not bs.reserved)


__all__ = ["SnapshotCache"]
