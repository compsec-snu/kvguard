"""Translation of SGLang state into the core's schema. Everything here is a pure function of
plain Python data or duck-typed objects, so the rules are unit-testable without SGLang or torch;
a run-time attachment extracts the data from the live objects.

Facts used: the KV pool (one slot per token at page size 1; slot 0 is the never-handed-out
padding slot), the per-request ``req_to_token`` slot rows, the radix tree's published/reserved
nodes (``value`` and ``lock_ref``), the allocator's free set, and the consumed mapping
(FlashInfer's ``kv_indptr``/``kv_indices`` plus ``out_cache_loc`` for the KV write).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kvguard.core import (
    BlockId,
    BlockSnapshot,
    BlockState,
    CSRPageList,
    DenseBlockTable,
    DispatchDescriptor,
    DispatchRow,
    PoolGeometry,
    PositionedTokenTable,
    RequestId,
    RequestSnapshot,
    SlotId,
    SlotMap,
    Snapshot,
    TokenTable,
)
from kvguard.core.violations import Unsupported, unsupported

POOL = "kv"
DRAFT_POOL = "draft_kv"
PADDING_SLOT = 0
"""The allocator hands out slots ``1..size``; slot 0 takes the dummy writes of CUDA-graph padding."""

SUPPORT_BOUNDARY = {
    "framework": "sglang",
    "version": "0.5.22; newer revisions through explicit opt-in ports (NGRAM, EAGLE3, Qwen3.5 MTP/NEXTN, DFlash, Step-3.5 multi-layer EAGLE, HiCache, external linker, beam search, PDMUX, independent full/SWA rails)",
    "page_size": "1 (TokenToKVPoolAllocator), or a power of two (PagedTokenToKVPoolAllocator, and SWATokenToKVPoolAllocator with independent full/SWA rails); pages are the blocks; the attention descriptor is the per-token table FlashInfer gathers; the device validators run only at page size one",
    "allocator": "TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator, or SWATokenToKVPoolAllocator with independent full/SWA rails",
    "pool": "one MHATokenToKVPool over every layer (slot 0 reserved for padding); HybridLinearKVPool (full attention plus a recurrent-state pool); or an independent full/SWA target pool, optionally with three SWA-only multi-layer EAGLE draft pools",
    "prefix_cache": "UnifiedRadixCache (FULL component, Rust or Python tree core) or RadixCache, enabled or disabled; device-resident nodes only; the opt-in HiCache and external-linker ports add bounded host, file and direct-linker tiers",
    "attention": "FlashInfer decode and extend consume kv_indptr/kv_indices gathered from req_to_token rows; the KV write is set_kv_buffer over out_cache_loc; sliding-window paths carry a logical start on each suffix token table",
    "excluded": (
        "HiRadixCache; HiCache storage other than the local file backend; external-linker layouts outside the opt-in port",
        "speculative decoding outside the opt-in ports",
        "other KVCache subclasses, and separate sliding-window or hybrid pools outside the opt-in ports",
        "P/D disaggregation, P/D multiplexing and beam search outside the opt-in ports",
        "dLLM", "multi-item scoring", "encoder-decoder",
        "tensor, data and context parallelism, DCP and KV sharding", "quantized KV caches (dequant workspace)",
        "other attention backends (Triton, FlashAttention, TRT-LLM MHA)",
    ),
    "notes": (
        "overlap scheduling does not change the translation; the attachment must complete epochs in stream order",
        "chunked prefill checkpoints move row slots into the tree between steps and are translated as publication",
    ),
}  # fmt: skip


@dataclass(frozen=True)
class RowFacts:
    """What the adapter reads from one running ``Req`` and its ``req_to_token`` row."""

    request_id: RequestId
    slots: Sequence[SlotId]
    """``req_to_token[req.kv.req_pool_idx, :kv_allocated_len]`` (read it, never keep it)."""
    kv_allocated_len: int
    kv_committed_len: int
    cache_protected_len: int
    """Positions below this are prefix-cache slots, not the request's own."""
    new_tokens: int = 0
    """Positions this step's dispatch writes for the request (0 when not in the batch)."""
    swa_first_token: int = 0
    """First request position whose auxiliary SWA mapping remains live; only the independent SWA
    view uses it, to discard a retained tree-prefix island before a window-evicted gap."""


@dataclass(frozen=True)
class NodeFacts:
    """What the adapter reads from one radix-tree node (not the root, not evicted)."""

    node_id: int
    slots: tuple[SlotId, ...]
    """``TreeNode.value``: the slots holding the node's tokens, in key order."""
    lock_ref: int
    """Running requests whose matched prefix includes the node."""


def _as_list(x: Any) -> list:
    """A tensor, array or sequence as a plain list. A torch tensor goes through NumPy, which is
    much faster per element than ``Tensor.tolist``."""
    if x is None:
        return []
    numpy = getattr(x, "numpy", None)
    if callable(numpy) and callable(getattr(x, "detach", None)):
        try:
            v = x.detach().cpu().numpy().tolist()
        except (TypeError, RuntimeError):  # a dtype NumPy does not take (bfloat16)
            v = x.tolist()
        return v if isinstance(v, list) else [v]
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        v = tolist()
        return v if isinstance(v, list) else [v]
    return list(x)


def _ints(x: Any) -> list[int]:
    v = _as_list(x)
    if v and type(v[0]) is int:  # a tensor's or array's own ints: nothing to convert
        return v
    return list(map(int, v))


def external_read_groups(
    page_indices: Sequence[int], num_tokens: int, page_size: int
) -> tuple[tuple[tuple[int, ...], range], ...]:
    """Group one non-staging P/D source transfer into ``ExternalRead`` shapes.

    The range begins on a page boundary. Every page except possibly the last is read in full;
    ``num_tokens`` trims the last page. ``ExternalRead`` carries one offset range for all of its
    blocks, so a partial tail is a separate group.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    pages = tuple(map(int, page_indices))
    needed = (num_tokens + page_size - 1) // page_size if num_tokens else 0
    if len(pages) != needed:
        raise ValueError(
            f"P/D source has {len(pages)} page indices for {num_tokens} tokens "
            f"at page size {page_size}; expected {needed}"
        )
    if not pages:
        return ()
    tail = num_tokens % page_size
    if not tail:
        return ((pages, range(page_size)),)
    if len(pages) == 1:
        return ((pages, range(tail)),)
    return ((pages[:-1], range(page_size)), ((pages[-1],), range(tail)))


def external_slot_groups(
    slots: Sequence[int], page_size: int
) -> tuple[tuple[tuple[int, ...], range], ...]:
    """Group a request row's token slots into external-transfer block shapes.

    Paged allocations place logical position ``p`` at offset ``p % page_size`` of its page.
    Retraction backup and restore copy a row prefix rather than a page-id list, so derive the
    concrete page ids and split a partial final page just as :func:`external_read_groups` does.
    Reject a row that does not have that allocator shape instead of silently widening the copy.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    values = tuple(map(int, slots))
    if not values:
        return ()
    pages: list[int] = []
    for position, slot in enumerate(values):
        if slot < 0:
            raise ValueError(f"external transfer names negative slot {slot} at position {position}")
        if slot % page_size != position % page_size:
            raise ValueError(
                f"external transfer slot {slot} at position {position} has offset "
                f"{slot % page_size}; expected {position % page_size}"
            )
        if position % page_size == 0:
            pages.append(slot // page_size)
        elif slot // page_size != pages[-1]:
            raise ValueError(
                f"external transfer changes page within logical page at position {position}"
            )
    return external_read_groups(pages, len(values), page_size)


def _rows_of(table: Any, indices: list[int], lengths: list[int]) -> list[list[int]]:
    """The first ``lengths[i]`` entries of row ``indices[i]``, as lists of ints. A 2-D tensor is
    read with one gather and host copy (one device sync per snapshot); any other table row by row."""
    if not indices:
        return []
    dim = getattr(table, "dim", None)
    if callable(dim) and dim() == 2 and callable(getattr(table, "tolist", None)):
        rows = _as_list(table[indices, : max(lengths)])
        return [r[:n] for r, n in zip(rows, lengths, strict=True)]
    return [_ints(table[i][:n]) for i, n in zip(indices, lengths, strict=True)]


# ------------------------------------------------------------------ geometry and mediation


def geometry(
    size: int, page_size: int, pool: str = POOL, *, layers: int = 1, kv_heads: int = 1
) -> PoolGeometry:
    """Pool registration from ``allocator.size`` and ``allocator.page_size``.

    With ``page_size`` 1 every KV slot is a one-token block: ``size + 1`` blocks (slot 0 reserved).
    With a larger page size a page is a ``page_size``-token block: page 0 is reserved, and token
    slot ``s`` sits at offset ``s % page_size`` of page ``s // page_size``.
    """
    page_size = int(page_size)
    if page_size < 1 or (page_size > 1 and page_size & (page_size - 1)):
        raise Unsupported(
            unsupported(
                "page_size",
                f"page_size {page_size}; the SGLang adapter supports 1 or a power of two",
                pool=pool,
            )
        )
    return PoolGeometry(
        pool,
        num_blocks=int(size) // page_size + 1,
        block_size=page_size,
        layers=layers,
        kv_heads=kv_heads,
        reserved_blocks=frozenset({PADDING_SLOT}),
        # An unlocked radix node (listed reserved) keeps its pages until evicted: they are
        # not in the allocator's free set, so the count audit counts them as held.
        reserved_in_free=False,
    )


def pages_of(slots: Sequence[SlotId], page_size: int) -> tuple[BlockId, ...]:
    """The pages a row's token slots occupy, in position order: every ``page_size``-th slot's
    page (a request's slots sit at offset ``position % page_size`` of their page). At page size
    1 the slots themselves."""
    if page_size == 1:
        return tuple(slots)
    return tuple(s // page_size for s in slots[::page_size])


def pages_of_prefix(slots: Sequence[SlotId], n: int, page_size: int) -> tuple[BlockId, ...]:
    """``pages_of(slots[:n], page_size)`` without copying the prefix: one slot per page is read."""
    if page_size == 1:
        return tuple(slots[:n])
    return tuple(s // page_size for s in slots[:n:page_size])


def page_set(slots: Iterable[SlotId], page_size: int) -> set[BlockId]:
    """The set of pages a set of token slots touches."""
    if page_size == 1:
        return set(slots)
    return {s // page_size for s in slots}


def _np_ints(x: Any) -> np.ndarray:
    """A list of slot indices as an int64 array: an array or a tensor without a per-element
    conversion, anything else through ``_ints``."""
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False)
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy().astype(np.int64, copy=False)
        except (TypeError, RuntimeError):
            pass
    return np.asarray(_ints(x), dtype=np.int64)


def page_set_np(
    slots: np.ndarray, page_size: int, *, duplicates: bool = False
) -> set[BlockId] | tuple[set[BlockId], list[int]]:
    """``page_set`` of a slot array, in array operations (a walk of the whole tree is hundreds of
    thousands of slots); with ``duplicates``, also the slots listed more than once, sorted."""
    if slots.size == 0:
        return (set(), []) if duplicates else set()
    if duplicates:
        unique, counts = np.unique(slots, return_counts=True)
        dups = unique[counts > 1].tolist()
        pages = (
            set(unique.tolist()) if page_size == 1 else set(np.unique(unique // page_size).tolist())
        )
        return pages, dups
    if page_size == 1:
        return set(np.unique(slots).tolist())
    return set(np.unique(slots // page_size).tolist())


def row_pages(slots: Sequence[SlotId], start: int, stop: int, page_size: int) -> set[BlockId]:
    """The pages of a row's positions ``[start, stop)``: one slot per page is read (the slots
    of a row sit at offset ``position % page_size`` of their page, as ``pages_of`` reads
    them), so a row of a thousand slots costs its pages, not its slots."""
    if stop <= start:
        return set()
    if page_size == 1:
        return set(slots[start:stop])
    pages = {slots[i] // page_size for i in range(start, stop, page_size)}
    pages.add(slots[stop - 1] // page_size)
    return pages


def eagle_tree_branch_starts(
    prefix: int, topk: int, num_steps: int, page_size: int
) -> tuple[int, ...]:
    """Request-table position of each EAGLE tree branch's first draft slot.

    At page size one the branches are consecutive ``num_steps`` regions. With a paged allocator
    SGLang gives every branch enough whole pages for the prefix's partial tail plus
    ``num_steps`` draft slots. Branch zero shares the canonical tail page; every later branch
    starts in its own page-aligned region after a copied tail.
    """
    prefix, topk, num_steps, page_size = map(int, (prefix, topk, num_steps, page_size))
    if prefix < 0 or topk < 1 or num_steps < 1 or page_size < 1:
        raise ValueError(
            f"invalid EAGLE tree layout prefix={prefix}, topk={topk}, "
            f"steps={num_steps}, page_size={page_size}"
        )
    if page_size == 1:
        return tuple(prefix + branch * num_steps for branch in range(topk))
    last = prefix % page_size
    base = prefix - last
    branch_span = ((last + num_steps + page_size - 1) // page_size) * page_size
    return tuple(base + branch * branch_span + last for branch in range(topk))


def eagle_tree_workspace_positions(
    prefix: int, topk: int, num_steps: int, page_size: int
) -> tuple[int, ...]:
    """Private positions initialized by one EAGLE tree iteration.

    This is the sparse set invalidated before draft-extend rewrites the compact candidate
    window. At a page size above one it includes each later branch's copied prefix-tail holes as
    well as every branch's draft slots; the canonical prefix tail of branch zero is retained.
    """
    starts = eagle_tree_branch_starts(prefix, topk, num_steps, page_size)
    if page_size == 1:
        return tuple(range(prefix, prefix + topk * num_steps))
    last = int(prefix) % int(page_size)
    out: list[int] = []
    for branch, start in enumerate(starts):
        if branch:
            out.extend(range(start - last, start))
        out.extend(range(start, start + int(num_steps)))
    return tuple(out)


def geometry_from_allocator(allocator: Any, pool: str = POOL, **kw) -> PoolGeometry:
    """``geometry`` from a ``BaseTokenToKVPoolAllocator``-like object (``size``, ``page_size``)."""
    return geometry(int(allocator.size), int(allocator.page_size), pool, **kw)


def allocated_slots(out_cache_loc: Any) -> list[SlotId]:
    """The slots an ``alloc(need_size)`` call handed out (``None`` when the allocator refused)."""
    return _ints(out_cache_loc)


def mediated_frees(free_index: Any, free_group: Any) -> list[SlotId]:
    """The slots a ``free(free_index)`` call actually returns to the allocator.

    While a free group is open (``free_group`` is a list) the call only appends to it;
    ``free_group_end`` calls ``free`` again with the concatenation, so only calls made with no
    open group are mediated frees.
    """
    if free_group is not None:
        return []
    return _ints(free_index)


def wrap_allocator_clear(
    allocator: Any,
    free_ids: Callable[[], Iterable[int]],
    on_released: Callable[[list[int]], None],
) -> None:
    """Mediate an allocator-wide reset by diffing its free set.

    A successful ``flush_cache`` resets the radix tree and calls ``allocator.clear()`` instead of
    returning pages through ``free``; report exactly the ids it added to the free set as releases.
    """

    original = allocator.clear

    def clear(*args: Any, **kwargs: Any) -> Any:
        before = {int(value) for value in free_ids()}
        out = original(*args, **kwargs)
        released = sorted({int(value) for value in free_ids()} - before)
        if released:
            on_released(released)
        return out

    allocator.clear = clear


# ------------------------------------------------------------------ snapshots


def pre_step_computed(kv_committed_len: int, new_tokens: int) -> int:
    """Positions written before this step: ``kv_committed_len`` minus the coming step's writes."""
    return int(kv_committed_len) - int(new_tokens)


def slot_states(
    pool: str,
    rows: Iterable[RowFacts],
    nodes: Iterable[NodeFacts],
    page_size: int = 1,
    shared_slots: Iterable[SlotId] = (),
    reserved_slots: Iterable[SlotId] = (),
) -> dict[BlockId, BlockSnapshot]:
    """Per-slot states from the tree and the rows.

    - A slot held by a tree node is PUBLISHED; ``refs`` is the node's ``lock_ref``; it is reserved
      (kept only by the cache, evictable) when ``lock_ref`` is 0.
    - A slot in a row at a position at or beyond ``cache_protected_len`` is the request's own:
      PRIVATE, unless the tree holds it (the tree wins; the core then judges the row's access).
    - A slot in a row below ``cache_protected_len`` is the request's claim that the tree holds it:
      it contributes the mapping only. If the tree does not hold it, the core sees a request
      mapping a block the snapshot lists as free (``mapping_to_free_block``) or another request's
      private block.
    - Slots in no node and no row are free (absent).

    A slot listed by two nodes breaks the tree's invariant; it is raised as ``ValueError`` because
    the schema has one state per block and the adapter must not pick one.
    """
    states: dict[BlockId, BlockSnapshot] = {}
    holder: dict[BlockId, int] = {}
    for node in nodes:
        for b in pages_of(node.slots, page_size):
            if b in holder and holder[b] != node.node_id:
                raise ValueError(f"block {b} held by radix nodes {holder[b]} and {node.node_id}")
            holder[b] = node.node_id
            states[b] = BlockSnapshot(
                pool,
                b,
                BlockState.PUBLISHED,
                reserved=node.lock_ref <= 0,
                refs=max(int(node.lock_ref), 0),
            )
    # Beam search aliases immutable history between physical member rows without inserting it into
    # the radix tree; it is still shared read-only state. ``refs=None`` disables the radix
    # lock-count comparison, since member rows have no tree locks of their own.
    for b in page_set(reserved_slots, page_size):
        states[b] = BlockSnapshot(pool, b, BlockState.PUBLISHED, reserved=True, refs=None)
    for b in page_set(shared_slots, page_size):
        states[b] = BlockSnapshot(pool, b, BlockState.PUBLISHED, reserved=False, refs=None)
    for row in rows:
        for b in page_set(row.slots[row.cache_protected_len : row.kv_allocated_len], page_size):
            if b not in states:
                states[b] = BlockSnapshot(pool, b, BlockState.PRIVATE)
    return states


def build_snapshot(
    seq: int,
    pool: str,
    rows: Iterable[RowFacts],
    nodes: Iterable[NodeFacts],
    free_slots: Iterable[SlotId],
    page_size: int = 1,
    shared_slots: Iterable[SlotId] = (),
    reserved_slots: Iterable[SlotId] = (),
) -> Snapshot:
    """Normalized snapshot of one scheduler state, taken after the batch's rows and
    ``out_cache_loc`` are written and before its forward runs.

    Each row becomes a request mapping of ``kv_allocated_len`` positions with pre-step computed
    count ``kv_committed_len - new_tokens``; its blocks are its slots (page size 1) or pages.
    ``free_slots`` is the allocator's free set for the partition audit.
    """
    rows = list(rows)
    requests = tuple(
        RequestSnapshot(
            r.request_id,
            pool,
            pages_of_prefix(r.slots, r.kv_allocated_len, page_size),
            int(r.kv_allocated_len),
            pre_step_computed(r.kv_committed_len, r.new_tokens),
        )
        for r in rows
    )
    states = slot_states(pool, rows, nodes, page_size, shared_slots, reserved_slots)
    blocks = tuple(states[s] for s in sorted(states))
    return Snapshot(seq, requests, blocks, {pool: frozenset(_ints(free_slots))})


# ------------------------------------------------------------------ extraction from live objects


def _gather_cells(table: Any, rows: list[int], cols: list[int]) -> list[int]:
    """``table[rows[i], cols[i]]`` for every i: one gather and one copy for a tensor."""
    dim = getattr(table, "dim", None)
    if callable(dim) and dim() == 2 and callable(getattr(table, "tolist", None)):
        return _ints(table[rows, cols])
    return [int(table[r][c]) for r, c in zip(rows, cols, strict=True)]


def _band_of(table: Any, rows: list[int], lo: int, hi: int) -> list[list[int]]:
    """Columns ``[lo, hi)`` of the given rows, as lists of ints."""
    dim = getattr(table, "dim", None)
    if callable(dim) and dim() == 2 and callable(getattr(table, "tolist", None)):
        return _as_list(table[rows, lo:hi])
    return [_ints(table[r][lo:hi]) for r in rows]


class RowSlotCache:
    """The rows' slot lists kept between snapshots, keyed by ``req_to_token`` row, to avoid
    re-reading every row each step.

    A row's slots below its allocated length are written once and not rewritten on ordinary
    paths, so a row whose owner and length are unchanged is not re-read; a decode row is read at
    its new position, a grown row over its new columns, and a new, re-owned or shrunk row in full.
    The owner is the adapter's per-``Req`` checker identity. The one ordinary rewrite below a
    row's length is the tree's checkpoint of an unfinished request (positions from the old to the
    new ``cache_protected_len`` repointed to canonical slots), so a row whose protected length
    grew is re-read from its old protected length. An audited snapshot re-reads every row and
    counts a kept prefix that differs (``drift``). Paths that otherwise rewrite a row's slots
    below its length (speculative rollbacks, P/D, beams) do not use the cache."""

    def __init__(self) -> None:
        self.rows: dict[int, tuple[str, list[int], int, int]] = {}
        """Row index -> (owner, slots read so far, committed count, protected length) at last read."""
        self.stats: Counter = Counter()
        self.drift = 0
        """Kept prefixes an audit found changed on the device."""
        self.drift_examples: list[tuple[int, str, int, int, int, int]] = []
        """The first few: (row, rid, kept length, first position that differs, kept, device)."""

    def read(
        self,
        table: Any,
        held: list[Any],
        audit: bool,
        owners: Sequence[str] | None = None,
    ) -> list[Sequence[int]]:
        """The slots of ``held``'s rows (``kv.req_pool_idx``, ``kv.kv_allocated_len``,
        ``kv.kv_committed_len``, ``kv.cache_protected_len``), in order, owned by ``owners``
        (the requests' ``rid`` when not given); the lists are the cache's own (read, do not
        keep)."""
        idx = [int(req.kv.req_pool_idx) for req in held]
        lens = [int(req.kv.kv_allocated_len) for req in held]
        committed = [int(req.kv.kv_committed_len) for req in held]
        protected = [int(req.kv.cache_protected_len) for req in held]
        who = [str(o) for o in owners] if owners is not None else [str(req.rid) for req in held]
        if audit or not self.rows:
            fresh = _rows_of(table, idx, lens)
            if self.rows:
                for ri, n, c, pr, owner, row in zip(
                    idx, lens, committed, protected, who, fresh, strict=True
                ):
                    ent = self.rows.get(ri)
                    kept = (
                        ent is not None
                        and ent[0] == owner
                        and len(ent[1]) <= n
                        and ent[2] <= c
                        and ent[3] == pr
                    )
                    if kept and row[: len(ent[1])] != ent[1]:
                        self.drift += 1
                        if len(self.drift_examples) < 8:
                            pairs = enumerate(zip(row, ent[1], strict=False))
                            at = next(i for i, (a, b) in pairs if a != b)
                            self.drift_examples.append(
                                (ri, owner, len(ent[1]), at, int(ent[1][at]), int(row[at]))
                            )
            self.rows = {
                ri: (owner, list(row), c, pr)
                for ri, owner, c, pr, row in zip(idx, who, committed, protected, fresh, strict=True)
            }
            if audit:
                self.stats["audited"] += 1
            self.stats["rows_full"] += len(idx)
            return [self.rows[ri][1] for ri in idx]
        out: list[Sequence[int] | None] = [None] * len(idx)
        full: list[int] = []
        one: list[int] = []
        more: list[int] = []
        starts: dict[int, int] = {}
        repointed = 0
        for i, (ri, n, c, pr, owner) in enumerate(
            zip(idx, lens, committed, protected, who, strict=True)
        ):
            ent = self.rows.get(ri)
            if ent is None or ent[0] != owner or len(ent[1]) > n or ent[2] > c or ent[3] > pr:
                full.append(i)
            elif ent[3] < pr:  # checkpointed: repointed from the old protected length
                more.append(i)
                starts[i] = min(ent[3], len(ent[1]))
                repointed += 1
            elif len(ent[1]) == n:
                out[i] = ent[1]
                self.rows[ri] = (owner, ent[1], c, pr)
            elif n - len(ent[1]) == 1:
                one.append(i)
            else:
                more.append(i)
                starts[i] = len(ent[1])
        if full:
            rows = _rows_of(table, [idx[i] for i in full], [lens[i] for i in full])
            for i, row in zip(full, rows, strict=True):
                slots = list(row)
                self.rows[idx[i]] = (who[i], slots, committed[i], protected[i])
                out[i] = slots
        if one:
            cells = _gather_cells(table, [idx[i] for i in one], [lens[i] - 1 for i in one])
            for i, v in zip(one, cells, strict=True):
                owner, slots, _, _ = self.rows[idx[i]]
                slots.append(v)
                self.rows[idx[i]] = (owner, slots, committed[i], protected[i])
                out[i] = slots
        if more:
            lo = min(starts[i] for i in more)
            hi = max(lens[i] for i in more)
            band = _band_of(table, [idx[i] for i in more], lo, hi)
            for i, row in zip(more, band, strict=True):
                owner, slots, _, _ = self.rows[idx[i]]
                start = starts[i]
                del slots[start:]
                slots.extend(row[start - lo : lens[i] - lo])
                self.rows[idx[i]] = (owner, slots, committed[i], protected[i])
                out[i] = slots
        if len(self.rows) > len(idx):
            seen = set(idx)
            self.rows = {ri: ent for ri, ent in self.rows.items() if ri in seen}
        st = self.stats
        st["rows_kept"] += len(idx) - len(full) - len(one) - len(more)
        st["rows_full"] += len(full)
        st["rows_one"] += len(one)
        st["rows_more"] += len(more)
        st["rows_repointed"] += repointed
        return out  # type: ignore[return-value]


def rows_from_requests(
    reqs: Iterable[Any],
    req_to_token: Any,
    new_tokens: Mapping[Any, int] | None = None,
    *,
    cache: RowSlotCache | None = None,
    audit: bool = False,
    owners: Sequence[str] | None = None,
) -> list[RowFacts]:
    """``RowFacts`` for every ``Req``-like object holding a row, reading its ``req_to_token`` row
    or, with ``cache``, only what grew since the previous call (every row when ``audit``; rows
    owned by ``owners``, the checker identities). ``new_tokens`` maps the coming batch's rids to
    the positions it writes."""
    if owners is not None:
        pairs = [(r, o) for r, o in zip(reqs, owners, strict=True) if r.kv.req_pool_idx is not None]
        held = [r for r, _ in pairs]
        held_owners: Sequence[str] | None = [o for _, o in pairs]
    else:
        held = [req for req in reqs if req.kv.req_pool_idx is not None]
        held_owners = None
    if cache is not None:
        slot_rows: list[Sequence[int]] = cache.read(req_to_token, held, audit, held_owners)
    else:
        slot_rows = [
            tuple(r)
            for r in _rows_of(
                req_to_token,
                [int(req.kv.req_pool_idx) for req in held],
                [int(req.kv.kv_allocated_len) for req in held],
            )
        ]
    out = []
    for req, slots in zip(held, slot_rows, strict=True):
        kv = req.kv
        out.append(
            RowFacts(
                str(req.rid),
                slots,
                int(kv.kv_allocated_len),
                int(kv.kv_committed_len),
                int(kv.cache_protected_len),
                int(new_tokens.get(req.rid, 0)) if new_tokens else 0,
            )
        )
    return out


def beam_member_rows(
    leader: Any,
    owner: RequestId,
    req_to_token: Any,
    *,
    new_tokens: int = 0,
) -> tuple[list[RowFacts], dict[int, RequestId], set[SlotId]]:
    """Translate SGLang's columnar beam member rows.

    The leader keeps the ordinary request identity; each physical row gets a stable virtual
    identity from its column number. Positions computed before this step are returned as
    ``shared_slots`` (selection may alias them between columns), while the current step allocates
    a distinct private write slot per physical row.
    """

    group = getattr(leader, "beam_group", None)
    member_rows_obj = getattr(group, "member_rows_cpu", None)
    if group is None or member_rows_obj is None or getattr(group, "retired", False):
        return [], {}, set()

    member_indices = _ints(member_rows_obj)
    if not member_indices:
        return [], {}, set()
    kv = leader.kv
    length = int(kv.kv_allocated_len)
    member_mappings = _rows_of(req_to_token, member_indices, [length] * len(member_indices))
    out: list[RowFacts] = []
    row_ids: dict[int, RequestId] = {}
    for column, (row_index, slots) in enumerate(
        zip(member_indices, member_mappings, strict=True), start=1
    ):
        rid = f"{owner}#beam{column}"
        row_ids[row_index] = rid
        out.append(
            RowFacts(
                rid,
                tuple(slots),
                length,
                int(kv.kv_committed_len),
                int(kv.cache_protected_len),
                int(new_tokens),
            )
        )

    all_indices = [int(kv.req_pool_idx), *member_indices]
    all_mappings = _rows_of(req_to_token, all_indices, [length] * len(all_indices))
    computed = max(pre_step_computed(kv.kv_committed_len, new_tokens), 0)
    shared = {int(slot) for mapping in all_mappings for slot in mapping[:computed]}
    return out, row_ids, shared


def beam_pending_orphan_slots(leader: Any) -> set[SlotId]:
    """Slots staged by beam reparenting for deferred reclamation: allocated but unreachable from
    every current row until the overlap result commits and ``_reclaim_orphans`` frees them."""

    group = getattr(leader, "beam_group", None)
    out: set[int] = set()
    for entry in getattr(group, "pending_orphans", ()) or ():
        old = entry.old_mapping
        new = entry.new_mapping
        if hasattr(old, "reshape"):
            old = old.reshape(-1)
        if hasattr(new, "reshape"):
            new = new.reshape(-1)
        out.update(set(_ints(old)) - set(_ints(new)))
    return out


def nodes_from_tree(root: Any) -> list[NodeFacts]:
    """``NodeFacts`` for every node below ``root``. The root holds no slots and is skipped; a node
    whose ``value`` is ``None`` (its device copy evicted by a hierarchical cache) is skipped."""
    out = []
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        if node.value is None:
            continue
        out.append(NodeFacts(int(node.id), tuple(_ints(node.value)), int(node.lock_ref)))
    return sorted(out, key=lambda n: n.node_id)


def free_slots_from_allocator(allocator: Any) -> list[SlotId]:
    """The allocator's free set: ``get_all_free_pages()`` (``free_pages`` plus ``release_pages``)."""
    virtual = getattr(allocator, "free_virtual_ids", None)
    if virtual is not None:
        return _ints(virtual)
    pages = getattr(allocator, "get_all_free_pages", None)
    if callable(pages):
        return _ints(pages())
    return _ints(allocator.free_pages) + _ints(getattr(allocator, "release_pages", None))


def free_count_from_allocator(allocator: Any) -> int:
    """The size of the allocator's free set without reading it back (the count the core's
    partition audit by count takes on the steps that do not read the set)."""

    def size(x: Any) -> int:
        if x is None:
            return 0
        numel = getattr(x, "numel", None)
        return int(numel()) if callable(numel) else len(x)

    virtual = getattr(allocator, "free_virtual_ids", None)
    if virtual is not None:
        return size(virtual)
    pages = getattr(allocator, "get_all_free_pages", None)
    if callable(pages):
        return size(pages())
    return size(allocator.free_pages) + size(getattr(allocator, "release_pages", None))


# ------------------------------------------------------------------ dispatch descriptors


def _csr(
    kv_indptr: Any,
    kv_indices: Any,
    n_rows: int,
    extra: Mapping[int, Sequence[int]] | None = None,
    last_page_len: Sequence[int] | None = None,
    page_size: int = 1,
) -> CSRPageList:
    """The first ``n_rows`` CSR rows as the kernel consumes them (slots at page size 1, pages
    otherwise, as FlashInfer's ``kv_indices`` are); ``extra`` appends the slots a padded lane
    writes to its row (as pages at a page size above 1); ``last_page_len`` is the wrapper's per
    row (1 at page size 1). Rows beyond the indptr are empty before ``extra``."""
    indptr = _ints(kv_indptr)
    indices = _ints(kv_indices)
    per_row = []
    for i in range(n_rows):
        entries = indices[indptr[i] : indptr[i + 1]] if i + 1 < len(indptr) else []
        if extra and i in extra:
            entries = entries + [int(x) // page_size for x in extra[i]]
        per_row.append(entries)
    out_indptr = [0]
    flat: list[int] = []
    for entries in per_row:
        flat.extend(entries)
        out_indptr.append(len(flat))
    lpl = [1] * n_rows
    if last_page_len is not None:
        given = [int(x) for x in last_page_len][:n_rows]
        lpl = given + [1] * (n_rows - len(given))
    return CSRPageList(out_indptr, flat, lpl)


def _token_table(
    kv_indptr: Any,
    kv_indices: Any,
    n_rows: int,
    extra: Mapping[int, Sequence[int]] | None = None,
    first: Sequence[int] | None = None,
) -> TokenTable:
    """The first ``n_rows`` CSR rows as a per-token table: even at a page size above one the
    FlashInfer backend gathers one entry per position, so the entries are token slots, which the
    core checks position by position against the request's pages."""
    indptr = _ints(kv_indptr)
    indices = _ints(kv_indices)
    table: list[list[int]] = []
    for i in range(n_rows):
        entries = indices[indptr[i] : indptr[i + 1]] if i + 1 < len(indptr) else []
        if extra and i in extra:
            entries = entries + [int(x) for x in extra[i]]
        table.append(entries)
    starts = None
    if first is not None:
        starts = [int(x) for x in first[:n_rows]] + [0] * max(n_rows - len(first), 0)
    return TokenTable(table, [len(t) for t in table], starts)


def _attention_mapping(kv_indptr, kv_indices, n_rows, extra, last_page_len, page_size, first=None):
    # A suffix gather needs its logical start even when one token is one block.  Current SGLang
    # can expose such a row after HiCache pressure/retraction; CSRPageList has no start field,
    # while TokenTable represents the exact same physical entries plus that offset.
    if page_size > 1 or (first is not None and any(int(x) for x in first)):
        return _token_table(kv_indptr, kv_indices, n_rows, extra, first)
    return _csr(kv_indptr, kv_indices, n_rows, extra, last_page_len, page_size)


def build_decode_dispatches(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    seq_lens: Any,
    kv_indptr: Any,
    kv_indices: Any,
    out_cache_loc: Any,
    *,
    num_rows_after_padding: int | None = None,
    last_page_len: Sequence[int] | None = None,
    page_size: int = 1,
    read_starts: Sequence[int] | None = None,
    tag: str = "",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """The decode step's two dispatches, in kernel order: the KV write (``set_kv_buffer`` over
    ``out_cache_loc``, one token per row at position ``seq_len - 1``) and the read-only decode
    attention over ``[0, seq_len)`` (``BatchDecodeWithPagedKVCacheWrapper`` over
    ``kv_indptr``/``kv_indices``).

    ``num_rows_after_padding`` is the CUDA graph's batch size: lanes beyond the real rows are
    padding rows; they read what the graph buffers hold for them (``req_to_token`` row 0, i.e. slot
    0) and write ``out_cache_loc[lane]`` (0), and the descriptor lists both so the core checks that
    no padded lane touches a request-managed slot.
    """
    n = len(req_ids)
    n_pad = n if num_rows_after_padding is None else int(num_rows_after_padding)
    lens = _ints(seq_lens)[:n]
    starts = [0] * n if read_starts is None else [int(x) for x in read_starts[:n]]
    indptr = _ints(kv_indptr)
    table_lens = [indptr[i + 1] - indptr[i] for i in range(n)]
    loc = _ints(out_cache_loc)
    wrows = tuple(DispatchRow(req_ids[i], writes=range(lens[i] - 1, lens[i])) for i in range(n))
    rrows = [
        DispatchRow(req_ids[i], reads=range(starts[i], starts[i] + table_lens[i])) for i in range(n)
    ]
    extra: dict[int, list[int]] = {}
    for i in range(n, n_pad):
        rrows.append(DispatchRow(None, padding=True))
        if i < len(loc):
            extra[i] = [loc[i]]
    write = DispatchDescriptor(
        f"write{tag}@{seq}", pool, "set_kv_buffer", wrows,
        SlotMap(loc[:n], list(range(n)), [L - 1 for L in lens]),
    )  # fmt: skip
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}", pool, "attention", tuple(rrows),
        _attention_mapping(
            kv_indptr, kv_indices, n_pad, extra, last_page_len, page_size, starts
        ),
    )  # fmt: skip
    return write, attn


def build_extend_dispatches(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    seq_lens: Any,
    prefix_lens: Any,
    kv_indptr: Any,
    kv_indices: Any,
    out_cache_loc: Any,
    *,
    use_ragged: bool,
    num_rows_after_padding: int | None = None,
    last_page_len: Sequence[int] | None = None,
    page_size: int = 1,
    read_starts: Sequence[int] | None = None,
    tag: str = "",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """The extend (prefill) step's two dispatches, in kernel order.

    Row ``i`` writes positions ``[prefix_len, seq_len)`` through ``out_cache_loc`` (flat, rows in
    batch order). With ``use_ragged`` (the default when the FlashInfer backend is not deterministic
    and not paged-only) the paged wrapper reads only the cached prefix ``[0, prefix_len)`` from the
    pool, the new tokens attend to each other from the batch's own K/V, and the KV write runs after
    the attention; otherwise the write runs first and the paged wrapper reads ``[0, seq_len)``.

    Tokens of ``out_cache_loc`` beyond the batch's real tokens (a padded token axis) are added
    to padding rows of the attention descriptor so their write targets are checked too.
    ``num_rows_after_padding`` names a fixed-width CUDA graph's captured request count; without
    it the historical variable-width prefill representation keeps one aggregate padding row.
    """
    n = len(req_ids)
    lens, pre = _ints(seq_lens)[:n], _ints(prefix_lens)[:n]
    starts = [0] * n if read_starts is None else [int(x) for x in read_starts[:n]]
    indptr = _ints(kv_indptr)
    table_lens = [indptr[i + 1] - indptr[i] for i in range(n)]
    loc = _ints(out_cache_loc)
    wrows, rrows, srows, positions = [], [], [], []
    for i in range(n):
        wrows.append(DispatchRow(req_ids[i], writes=range(pre[i], lens[i])))
        rrows.append(DispatchRow(req_ids[i], reads=range(starts[i], starts[i] + table_lens[i])))
        srows.extend([i] * (lens[i] - pre[i]))
        positions.extend(range(pre[i], lens[i]))
    total = len(positions)
    extra: dict[int, list[int]] = {}
    n_rows = n
    if len(loc) > total:
        n_pad = n + 1 if num_rows_after_padding is None else int(num_rows_after_padding)
        if n_pad <= n:
            raise ValueError(
                f"{len(loc) - total} padded locations but padded row count is {n_pad} <= {n}"
            )
        tail = loc[total:]
        pad_rows = n_pad - n
        if num_rows_after_padding is not None and len(tail) % pad_rows:
            raise ValueError(
                f"{len(tail)} padded locations do not divide across {pad_rows} padding rows"
            )
        width = len(tail) if num_rows_after_padding is None else len(tail) // pad_rows
        for row in range(n, n_pad):
            rrows.append(DispatchRow(None, padding=True))
            offset = (row - n) * width
            extra[row] = tail[offset : offset + width]
        n_rows = n_pad
    write = DispatchDescriptor(
        f"write{tag}@{seq}", pool, "set_kv_buffer", tuple(wrows),
        SlotMap(loc[:total], srows, positions),
    )  # fmt: skip
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}", pool, "attention", tuple(rrows),
        _attention_mapping(
            kv_indptr, kv_indices, n_rows, extra, last_page_len, page_size, starts
        ),
    )  # fmt: skip
    return (attn, write) if use_ragged else (write, attn)


def build_mapping_dispatches(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    seq_lens: Sequence[int],
    prefix_lens: Sequence[int] | None,
    pages_of: Callable[[RequestId], Sequence[BlockId | None]],
    page_size: int,
    *,
    mode: str,
    use_ragged: bool = False,
    num_rows_after_padding: int | None = None,
    tag: str = "",
) -> tuple[tuple[DispatchDescriptor, DispatchDescriptor], list[int]]:
    """The step's two dispatches from the framework's bookkeeping alone (the host tier's
    ``mapping`` tables): the attention over a dense block table of each request's pages (the
    ledger's mapping, one entry per page) and the KV write over the slots the mapping gives the
    written positions. No device table is read; the device copies the kernels consume are the
    device validators' business. ``mode`` is ``decode`` (rows read ``[0, seq_len)`` and write
    ``seq_len - 1``; lanes beyond the real rows are padding rows with no entries) or ``extend``
    (rows write ``[prefix, seq_len)`` and read the cached prefix with ``use_ragged``, else
    ``[0, seq_len)``). Returns the dispatches in kernel order and the slots written, in token
    order (the slot map the device validator compares its own against)."""
    n = len(req_ids)
    ps = int(page_size)
    lens = [int(x) for x in seq_lens[:n]]
    pre = [0] * n if mode == "decode" else [int(x) for x in prefix_lens[:n]]
    pages = [tuple(pages_of(r)) for r in req_ids[:n]]

    def slot(i: int, p: int) -> int:
        k = p // ps
        page = pages[i][k] if k < len(pages[i]) else None
        return -1 if page is None else int(page) * ps + p % ps

    wrows, rrows, srows, positions, loc, table, valid = [], [], [], [], [], [], []
    for i in range(n):
        if mode == "decode":
            w = range(lens[i] - 1, lens[i])
            end = lens[i]
        else:
            w = range(pre[i], lens[i])
            end = pre[i] if use_ragged else lens[i]
        wrows.append(DispatchRow(req_ids[i], writes=w))
        rrows.append(DispatchRow(req_ids[i], reads=range(0, end)))
        for p in w:
            srows.append(i)
            positions.append(p)
            loc.append(slot(i, p))
        blocks = -(-end // ps)
        table.append(list(pages[i][:blocks]))
        valid.append(min(blocks, len(pages[i])))
    n_pad = n if num_rows_after_padding is None else max(int(num_rows_after_padding), n)
    for _ in range(n, n_pad):
        rrows.append(DispatchRow(None, padding=True))
        table.append([])
        valid.append(0)
    write = DispatchDescriptor(
        f"write{tag}@{seq}", pool, "set_kv_buffer", tuple(wrows),
        SlotMap(list(loc), srows, positions),
    )  # fmt: skip
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}", pool, "attention", tuple(rrows), DenseBlockTable(table, valid)
    )
    order = (attn, write) if mode == "extend" and use_ragged else (write, attn)
    return order, loc


def build_verify_dispatches(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    prefix_lens: Any,
    kv_indptr: Any,
    kv_indices: Any,
    out_cache_loc: Any,
    draft_token_num: int,
    *,
    num_rows_after_padding: int | None = None,
    last_page_len: Sequence[int] | None = None,
    page_size: int = 1,
    read_starts: Sequence[int] | None = None,
    tag: str = "-verify",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """A linear-layout speculative target-verify step (NGRAM or top-k-1 EAGLE3).

    SGLang reserves ``draft_token_num`` consecutive slots per request for the candidate tree
    (depth is carried in the attention mask), so the row's virtual layout is the committed prefix
    ``[0, prefix_len)`` followed by all candidate slots ``[prefix_len, prefix_len + draft)``.
    CUDA graphs can pad both axes; padding rows are included and trailing write destinations
    attach to one padding write row, so both checks require them to name the reserved slot.
    """
    n = len(req_ids)
    draft = int(draft_token_num)
    if draft <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft}")
    prefixes = _ints(prefix_lens)[:n]
    starts = [0] * n if read_starts is None else [int(x) for x in read_starts[:n]]
    indptr = _ints(kv_indptr)
    n_pad = n if num_rows_after_padding is None else int(num_rows_after_padding)
    if len(indptr) < n_pad + 1:
        raise ValueError(f"verify indptr has {len(indptr)} entries for {n_pad} rows")
    loc = _ints(out_cache_loc)
    real_tokens = n * draft

    wrows = [
        DispatchRow(req_ids[i], writes=range(prefixes[i], prefixes[i] + draft)) for i in range(n)
    ]
    srows = [i for i in range(n) for _ in range(draft)]
    positions = [p for base in prefixes for p in range(base, base + draft)]
    if len(loc) > real_tokens:
        wrows.append(DispatchRow(None, padding=True))
        srows.extend([n] * (len(loc) - real_tokens))
        positions.extend(range(len(loc) - real_tokens))

    rrows = []
    for i in range(n):
        table_len = int(indptr[i + 1]) - int(indptr[i])
        rrows.append(DispatchRow(req_ids[i], reads=range(starts[i], starts[i] + table_len)))
    rrows.extend(DispatchRow(None, padding=True) for _ in range(n, n_pad))

    write = DispatchDescriptor(
        f"write{tag}@{seq}", pool, "set_kv_buffer", tuple(wrows),
        SlotMap(loc, srows, positions),
    )  # fmt: skip
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}", pool, "attention", tuple(rrows),
        _attention_mapping(
            kv_indptr,
            kv_indices,
            n_pad,
            None,
            last_page_len,
            page_size,
            starts,
        ),
    )  # fmt: skip
    return write, attn


def build_dflash_materialize_dispatch(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    positions: Any,
    out_cache_loc: Any,
    valid_lens: Any,
    *,
    row_width: int | None = None,
    tag: str = "-dflash-materialize",
) -> DispatchDescriptor:
    """A DFlash target-hidden write into the draft KV view.

    Prefill materialization is packed (no ``row_width``). Decode materialization receives a dense
    ``[batch, block_size]`` window but commits only each row's accepted prefix, so ``row_width``
    preserves the stride across rejected suffix entries.
    """
    n = len(req_ids)
    lens = _ints(valid_lens)[:n]
    if len(lens) != n or any(length < 0 for length in lens):
        raise ValueError(f"DFlash materialize has invalid lengths {lens} for {n} rows")
    width = None if row_width is None else int(row_width)
    if width is not None and (width <= 0 or any(length > width for length in lens)):
        raise ValueError(f"DFlash materialize lengths {lens} exceed row width {width}")
    pos = _ints(positions)
    loc = _ints(out_cache_loc)
    required = sum(lens) if width is None else n * width
    if len(pos) < required or len(loc) < required:
        raise ValueError(
            f"DFlash materialize has positions={len(pos)}, locations={len(loc)}, "
            f"requires={required}"
        )

    rows: list[DispatchRow] = []
    selected_positions: list[int] = []
    selected_locations: list[int] = []
    slot_rows: list[int] = []
    packed_offset = 0
    for row, (rid, length) in enumerate(zip(req_ids, lens, strict=True)):
        offset = row * width if width is not None else packed_offset
        row_positions = pos[offset : offset + length]
        row_locations = loc[offset : offset + length]
        if row_positions and row_positions != list(
            range(row_positions[0], row_positions[0] + length)
        ):
            raise ValueError(
                f"DFlash materialize row {row} positions {row_positions} are not contiguous"
            )
        start = row_positions[0] if row_positions else 0
        rows.append(DispatchRow(rid, writes=range(start, start + length)))
        selected_positions.extend(row_positions)
        selected_locations.extend(row_locations)
        slot_rows.extend([row] * length)
        if width is None:
            packed_offset += length

    return DispatchDescriptor(
        f"write{tag}@{seq}",
        pool,
        "dflash_target_hidden",
        tuple(rows),
        SlotMap(selected_locations, slot_rows, selected_positions),
    )


def build_eagle_tree_draft_decode_dispatches(
    seq: int,
    pool: str,
    req_ids: Sequence[RequestId],
    prefix_lens: Any,
    topk: int,
    num_steps: int,
    step: int,
    kv_indptr: Any,
    kv_indices: Any,
    out_cache_loc: Any,
    *,
    num_requests_after_padding: int | None = None,
    page_size: int = 1,
    tag: str = "-draft-tree",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """One EAGLE3 tree-decode inner step.

    At page size one SGLang reserves ``topk * num_steps`` consecutive slots after the prefix;
    with a paged allocator each branch gets a page-aligned region with a copied prefix tail and
    its ``num_steps`` slots. Attention has one row per branch that gathers the committed prefix
    followed by that branch's draft slots through ``step``. CUDA graphs pad the request axis;
    their padding rows read and write slot zero and are checked to stay in the reserved block.
    """
    n = len(req_ids)
    k, steps, inner = int(topk), int(num_steps), int(step)
    if k <= 1:
        raise ValueError(f"tree draft topk must exceed one, got {k}")
    if steps <= 1 or not 0 <= inner < steps - 1:
        raise ValueError(f"tree draft step {inner} falls outside [0, {steps - 1})")
    prefixes = _ints(prefix_lens)[:n]
    if len(prefixes) != n:
        raise ValueError(f"tree draft has {len(prefixes)} prefixes for {n} requests")
    n_pad_req = n if num_requests_after_padding is None else int(num_requests_after_padding)
    if n_pad_req < n:
        raise ValueError(f"tree draft padding batch {n_pad_req} is smaller than {n}")
    real_rows, total_rows = n * k, n_pad_req * k
    indptr = _ints(kv_indptr)
    if len(indptr) < total_rows + 1:
        raise ValueError(f"tree draft indptr has {len(indptr)} entries for {total_rows} rows")
    indices = _ints(kv_indices)
    loc = _ints(out_cache_loc)
    if len(loc) == real_rows and total_rows > real_rows:
        loc.extend([PADDING_SLOT] * (total_rows - real_rows))
    if len(loc) != total_rows:
        raise ValueError(f"tree draft has {len(loc)} write slots for {total_rows} rows")

    wrows: list[DispatchRow] = []
    rrows: list[DispatchRow] = []
    table: list[list[int]] = []
    logical_rows: list[tuple[int, ...]] = []
    slot_rows: list[int] = []
    slot_positions: list[int] = []
    for row in range(total_rows):
        entries = indices[indptr[row] : indptr[row + 1]]
        table.append(entries)
        if row >= real_rows:
            wrows.append(DispatchRow(None, padding=True))
            rrows.append(DispatchRow(None, padding=True))
            logical_rows.append(())
            slot_rows.append(row)
            slot_positions.append(0)
            continue
        request_index, branch = divmod(row, k)
        prefix = prefixes[request_index]
        branch_start = eagle_tree_branch_starts(prefix, k, steps, page_size)[branch]
        reads = tuple(range(prefix)) + tuple(range(branch_start, branch_start + inner + 1))
        if len(entries) != len(reads):
            raise ValueError(
                f"tree draft row {row} has {len(entries)} entries, expected "
                f"prefix {prefix} + depth {inner + 1}"
            )
        write_position = branch_start + inner
        wrows.append(
            DispatchRow(req_ids[request_index], writes=range(write_position, write_position + 1))
        )
        rrows.append(DispatchRow(req_ids[request_index], reads=reads))
        logical_rows.append(reads)
        slot_rows.append(row)
        slot_positions.append(write_position)

    write = DispatchDescriptor(
        f"write{tag}@{seq}",
        pool,
        "set_kv_buffer",
        tuple(wrows),
        SlotMap(loc, slot_rows, slot_positions),
        advances_computed=False,
    )
    attention = DispatchDescriptor(
        f"attn{tag}@{seq}",
        pool,
        "attention",
        tuple(rrows),
        PositionedTokenTable(table, logical_rows),
    )
    return write, attention


def build_ngram_copy_dispatch(
    seq: int,
    pool: str,
    requests: Sequence[tuple[RequestId, int]],
    target_locs: Any,
    source_locs: Any,
    accepted_lens: Any,
    accept_index: Any,
    draft_token_num: int,
    tag: str = "",
) -> DispatchDescriptor:
    """The accepted-candidate compaction after an NGRAM target-verify step.

    Both arrays are packed by request and accepted token: row ``i`` contributes
    ``accepted_lens[i]`` candidate destinations, its nonnegative ``accept_index`` entries the
    sources. Describing both operands as reads validates ownership and mapping without pretending
    the destination is newly initialized. The mover's fixed-size zero-filled suffix is an actual
    padding-slot copy and is represented by padding rows too.
    """
    n = len(requests)
    draft = int(draft_token_num)
    if draft <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft}")
    lens = _ints(accepted_lens)
    if len(lens) != n:
        raise ValueError(f"accepted_lens has {len(lens)} entries for {n} verify rows")
    selected = _ints(accept_index)
    if len(selected) != n * draft:
        raise ValueError(f"accept_index has {len(selected)} entries for {n} rows of width {draft}")
    targets, sources = _ints(target_locs), _ints(source_locs)
    if len(targets) != len(sources):
        raise ValueError(
            f"accepted-token copy has {len(targets)} targets and {len(sources)} sources"
        )
    expected = sum(lens)
    if any(length < 0 or length > draft for length in lens):
        raise ValueError(f"accepted lengths {lens} fall outside [0, {draft}]")
    if expected > len(targets):
        raise ValueError(
            f"accepted-token copy has {len(targets)} entries for {expected} accepted tokens"
        )

    rows: list[DispatchRow] = []
    table: list[list[int]] = []
    first: list[int] = []
    cursor = 0
    for i, ((rid, base), length) in enumerate(zip(requests, lens, strict=True)):
        row_indices = [value for value in selected[i * draft : (i + 1) * draft] if value >= 0]
        if len(row_indices) != length:
            raise ValueError(
                f"verify row {i} accepts {length} tokens but has "
                f"{len(row_indices)} nonnegative indices"
            )
        for offset, source_index in enumerate(row_indices):
            target_pos = int(base) + offset
            source_pos = int(base) + int(source_index) % draft
            rows.append(DispatchRow(rid, reads=range(target_pos, target_pos + 1)))
            table.append([targets[cursor]])
            first.append(target_pos)
            rows.append(DispatchRow(rid, reads=range(source_pos, source_pos + 1)))
            table.append([sources[cursor]])
            first.append(source_pos)
            cursor += 1

    for target, source in zip(targets[cursor:], sources[cursor:], strict=True):
        rows.extend((DispatchRow(None, padding=True), DispatchRow(None, padding=True)))
        table.extend(([target], [source]))
        first.extend((0, 0))

    return DispatchDescriptor(
        f"spec-copy{tag}@{seq}",
        pool,
        "spec_accept_copy",
        tuple(rows),
        TokenTable(table, [1] * len(table), first),
    )


def build_token_table(rows: Iterable[Any], seq_lens: Any) -> TokenTable:
    """The per-token representation for a backend that reads ``req_to_token`` rows directly (the
    Triton backend). Outside the FlashInfer boundary; provided for the representation only."""
    lens = _ints(seq_lens)
    table = [_ints(r)[: lens[i]] for i, r in enumerate(rows)]
    return TokenTable(table, lens)


__all__ = [
    "DRAFT_POOL",
    "PADDING_SLOT",
    "POOL",
    "SUPPORT_BOUNDARY",
    "NodeFacts",
    "RowFacts",
    "allocated_slots",
    "beam_member_rows",
    "beam_pending_orphan_slots",
    "build_decode_dispatches",
    "build_dflash_materialize_dispatch",
    "build_eagle_tree_draft_decode_dispatches",
    "build_extend_dispatches",
    "build_ngram_copy_dispatch",
    "build_snapshot",
    "build_token_table",
    "build_verify_dispatches",
    "free_count_from_allocator",
    "free_slots_from_allocator",
    "geometry",
    "geometry_from_allocator",
    "mediated_frees",
    "nodes_from_tree",
    "page_set",
    "pages_of",
    "pre_step_computed",
    "rows_from_requests",
    "slot_states",
    "wrap_allocator_clear",
]
