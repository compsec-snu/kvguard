"""Translation of vLLM state into the core's schema. Everything here is a pure function of plain
Python data, so the rules are unit-testable without vLLM; `attach.py` extracts the data from the
live objects.

Facts used: the per-request block lists of each KV cache group over the shared ``BlockPool``;
block sharing and publication (``KVCacheBlock.ref_cnt`` and ``block_hash``); the allocator's free
queue; computed positions (``num_computed_tokens`` after ``schedule()`` already includes the
step's scheduled tokens, so the pre-step count subtracts them); and the consumed mapping from
``prepare_attn``'s block tables and slot mappings plus the ``InputBatch``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from kvguard.core import (
    BlockId,
    BlockSnapshot,
    BlockState,
    DenseBlockTable,
    DispatchDescriptor,
    DispatchRow,
    ExternalCopy,
    RequestId,
    RequestSnapshot,
    SlotMap,
    Snapshot,
)

SUPPORT_BOUNDARY = {
    "framework": "vllm",
    "version": "0.32.0",
    "kv_cache_groups": "one full-attention group (including MLAAttentionSpec), alone or with recurrent-state (Mamba) groups over the same block pool in the 'align' or 'none' Mamba cache modes, and with sliding-window groups",
    "attention": "one full-attention group (FullAttentionSpec or MLAAttentionSpec) consumed by the FLASH_ATTN, FLASHINFER, TRITON_ATTN or MLA backends; a kernel block size below the group's is the worker's view of the same table",
    "recurrent_state": "one rewritable core pool per Mamba group: a step reads a request's previous state column and writes the running one; copy-on-write copies of partial prefix hits are external copies; the align mode's state migration is derived from the scheduler's counts",
    "sliding_window": "one core pool per sliding-window group: positions below the window start are dead and the attention reads the window only; the device validators cover the full-attention group only",
    "scheduling": "synchronous and asynchronous (a step's dispatches complete when its output is processed)",
    "speculative": "on newer vLLM revisions (Model Runner V2), host-side: n-gram, EAGLE3 (including adaptive verification), MultiModuleMTPSpeculator, DFlashSpeculator and StandaloneARSpeculator, with target and draft layers as distinct logical pools",
    "kv_offload": "on newer vLLM revisions: OffloadingConnector, TieringOffloadingManager (CPU primary with filesystem or P2P secondary tiers), SimpleCPUOffloadConnector (CPU or direct-I/O disk) and the rank-local single-rank MooncakeStoreConnector; transfers are external reads and fills on their streams",
    "sleep": "on newer vLLM revisions: CuMem sleep levels 0, 1 and 2 over one full-attention group",
    "checkpoint": "on newer vLLM revisions: process-checkpoint prepare/restore over ordinary, EAGLE3, hybrid recurrent-state and CPU-offloaded caches",
    "pd_transfer": "on newer vLLM revisions: NIXL pull P/D over one full-attention group with direct CUDA buffers and equal pages; source reads stay open until sending finishes and destinations are credited when receiving finishes",
    "excluded": (
        "other drafters and speculative compositions; CUDA graphs on the standalone draft; device validators on the standalone, MTP and EAGLE3 drafts",
        "chunked-local attention groups",
        "other KV connectors, Mooncake Store topologies and NIXL P/D modes",
        "pipeline, tensor and data parallelism",
        "encoder-decoder",
    ),
}  # fmt: skip


@dataclass(frozen=True)
class BlockFacts:
    """What the adapter reads from one ``KVCacheBlock``."""

    block_id: BlockId
    ref_cnt: int
    has_hash: bool
    is_null: bool = False


@dataclass(frozen=True)
class RequestFacts:
    """What the adapter reads from one scheduler ``Request`` and its block list."""

    request_id: RequestId
    block_ids: tuple[BlockId | None, ...]
    """``None`` for a null block in the list (sliding-window hole)."""
    num_computed_after_schedule: int
    num_scheduled: int


def pre_step_computed(num_computed_after_schedule: int, num_scheduled: int) -> int:
    """Positions written before this step: ``num_computed_after_schedule`` minus the scheduled
    tokens (which ``schedule()`` already added)."""
    return num_computed_after_schedule - num_scheduled


def block_state(
    ref_cnt: int, has_hash: bool, fully_computed: bool
) -> tuple[BlockState | None, bool]:
    """``(state, reserved)`` for one block, or ``(None, False)`` when it is free.

    A block with a registered hash is immutable only once its owner has computed every position
    in it, or once a second request maps it (a same-step prefix hit); until then it is PRIVATE and
    its owner is still writing it. A hashed block with no references is kept by the prefix cache:
    PUBLISHED and reserved.
    """
    if ref_cnt <= 0:
        return (BlockState.PUBLISHED, True) if has_hash else (None, False)
    if has_hash and (ref_cnt > 1 or fully_computed):
        return BlockState.PUBLISHED, False
    return BlockState.PRIVATE, False


def build_snapshot(
    seq: int,
    pool: str,
    block_size: int,
    requests: Iterable[RequestFacts],
    blocks: Iterable[BlockFacts],
    free_queue_ids: Iterable[BlockId] | None,
    free_queue_hashed: Iterable[BlockId] = (),
) -> Snapshot:
    """Normalized snapshot from the facts of one post-``schedule()`` state.

    ``free_queue_ids`` are the blocks in the allocator's free queue; those that still carry a hash
    (``free_queue_hashed``) are reserved published blocks, not free, and are excluded from the
    partition audit's free set. ``None`` means the free queue was not read this step: the
    snapshot carries no free set and the core skips the partition audit (the adapter then lists
    every block that is referenced by a request or held by the prefix cache, which is every
    non-free block of a healthy pool; a leaked block surfaces on the next audited step).
    """
    request_snapshots = []
    covering: dict[BlockId, bool] = {}
    for rf in requests:
        if not rf.block_ids:
            continue
        pre = pre_step_computed(rf.num_computed_after_schedule, rf.num_scheduled)
        live = pre + rf.num_scheduled
        request_snapshots.append(
            RequestSnapshot(rf.request_id, pool, tuple(rf.block_ids), live, pre)
        )
        for j, b in enumerate(rf.block_ids):
            if b is None:
                continue
            covering[b] = covering.get(b, False) or pre >= (j + 1) * block_size
    states = []
    for bf in blocks:
        if bf.is_null:
            continue
        state, reserved = block_state(bf.ref_cnt, bf.has_hash, covering.get(bf.block_id, False))
        if state is None:
            continue
        states.append(
            BlockSnapshot(pool, bf.block_id, state, reserved=reserved, refs=max(bf.ref_cnt, 0))
        )
    if free_queue_ids is None:
        return Snapshot(seq, tuple(request_snapshots), tuple(states), None)
    hashed = set(free_queue_hashed)
    truly_free = frozenset(b for b in free_queue_ids if b not in hashed)
    return Snapshot(seq, tuple(request_snapshots), tuple(states), {pool: truly_free})


def build_dispatches(
    seq: int,
    pool: str,
    block_size: int,
    req_ids: Sequence[RequestId],
    pre_step_computed_per_row: Sequence[int],
    num_scheduled_per_row: Sequence[int],
    block_table: np.ndarray,
    num_reqs_after_padding: int,
    slot_mapping: Sequence[int],
    window: int | None = None,
    site: str = "",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """The step's two dispatches from ``prepare_attn``'s outputs: the cache write (slot map,
    first) and the read-only attention (dense block table with padding rows). With ``window``
    (a sliding-window group) a row's queries ``[start, end)`` attend the keys
    ``[start - window + 1, end)`` and the kernel consumes the table from that first key's
    block (``DenseBlockTable.first``); the entries below belong to retired blocks the worker's
    table keeps and the kernel does not read. ``site`` names the group when several attention
    groups dispatch in one step (``write-<site>@seq``)."""
    n = len(req_ids)
    rows, wrows, table, valid, first = [], [], [], [], []
    for i in range(n):
        start, cnt = int(pre_step_computed_per_row[i]), int(num_scheduled_per_row[i])
        end = start + cnt
        lo = max(0, start - window + 1) if window is not None else 0
        # A request may remain in the input batch while it is blocked on an
        # external-cache load.  Such a row has no queries, so the attention
        # kernel consumes neither its table nor any KV positions even though
        # ``start`` still records the request's computed frontier.
        rows.append(DispatchRow(req_ids[i], reads=range(lo, end) if cnt else range(0)))
        wrows.append(DispatchRow(req_ids[i], writes=range(start, end)))
        v = -(-end // block_size) if cnt else 0
        table.append(block_table[i, :v].tolist())
        valid.append(v)
        first.append(lo // block_size if cnt else 0)
    for i in range(n, num_reqs_after_padding):
        rows.append(DispatchRow(None, padding=True))
        nz = np.flatnonzero(block_table[i])
        v = int(nz[-1]) + 1 if len(nz) else 0
        table.append(block_table[i, :v].tolist())
        valid.append(v)
        first.append(0)
    sched = np.asarray(num_scheduled_per_row[:n], dtype=np.int64)
    srows = np.repeat(np.arange(n), sched)
    positions = (
        np.concatenate(
            [
                np.arange(
                    int(pre_step_computed_per_row[i]),
                    int(pre_step_computed_per_row[i]) + int(sched[i]),
                )
                for i in range(n)
            ]
        )
        if n
        else np.zeros(0, dtype=np.int64)
    )
    total = int(sched.sum())
    slots = [int(x) for x in slot_mapping]
    if len(slots) < total:
        raise ValueError(f"slot mapping has {len(slots)} entries for {total} scheduled tokens")
    # CUDA graphs consume the mapping through the padded token count.  Keep that tail and
    # describe it as one padding row: every entry must be the padding sentinel (or a permanently
    # reserved discard slot), never a stale request-managed slot from an earlier replay.
    padding = len(slots) - total
    if padding:
        wrows.append(DispatchRow(None, padding=True))
    tag = f"-{site}" if site else ""
    write = DispatchDescriptor(
        f"write{tag}@{seq}", pool, "reshape_and_cache", tuple(wrows),
        SlotMap(
            slots,
            srows.tolist() + [n] * padding,
            positions.tolist() + [0] * padding,
        ),
    )  # fmt: skip
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}",
        pool,
        "attention",
        tuple(rows),
        DenseBlockTable(table, valid, first if window is not None else None),
    )
    return write, attn


def build_positioned_dispatches(
    seq: int,
    pool: str,
    block_size: int,
    req_ids: Sequence[RequestId],
    query_start_loc: Sequence[int],
    positions: Sequence[int],
    seq_lens: Sequence[int],
    block_table: np.ndarray,
    slot_mapping: Sequence[int],
    window: int | None = None,
    site: str = "",
) -> tuple[DispatchDescriptor, DispatchDescriptor]:
    """Describe a draft forward whose query positions come from device buffers.

    Speculative drafters may prepend correction or rejected-token re-prefills, and multi-module
    MTP pads one trailing token in a request row. Their slot maps therefore cannot be reconstructed
    from the scheduler's simple ``computed + scheduled`` interval. The live
    (non-negative-slot) prefix of each row is still one contiguous logical interval; retain the
    engine's concrete positions and slots while deriving the attention union from that interval.
    """
    n = len(req_ids)
    qsl = [int(value) for value in query_start_loc]
    if len(qsl) != n + 1 or qsl[0] != 0:
        raise ValueError(f"query_start_loc has {len(qsl)} entries for {n} requests")
    if any(left > right for left, right in pairwise(qsl)):
        raise ValueError("query_start_loc is not monotonic")
    if len(seq_lens) < n or block_table.ndim != 2 or block_table.shape[0] < n:
        raise ValueError("positioned dispatch has incomplete sequence lengths or block-table rows")
    total = qsl[-1]
    slots = [int(value) for value in slot_mapping]
    pos = [int(value) for value in positions]
    if len(slots) < total or len(pos) < total:
        raise ValueError(
            f"positioned dispatch has {len(slots)} slots and {len(pos)} positions for {total} tokens"
        )

    write_rows: list[DispatchRow] = []
    attn_rows: list[DispatchRow] = []
    table: list[list[int]] = []
    valid: list[int] = []
    first: list[int] = []
    token_rows: list[int] = []
    for row, request in enumerate(req_ids):
        begin, limit = qsl[row], qsl[row + 1]
        row_slots = slots[begin:limit]
        padding_at = next((i for i, value in enumerate(row_slots) if value < 0), len(row_slots))
        if any(value >= 0 for value in row_slots[padding_at:]):
            raise ValueError(f"request {request!r} has non-trailing draft slot padding")
        live_positions = pos[begin : begin + padding_at]
        if not live_positions:
            raise ValueError(f"request {request!r} has no live draft query position")
        start = live_positions[0]
        end = live_positions[-1] + 1
        if live_positions != list(range(start, end)):
            raise ValueError(f"request {request!r} has non-contiguous draft query positions")
        # Multi-module MTP keeps the target's rejected trailing token in the
        # attention sequence length while padding its draft-cache slot.  It is
        # causally beyond the last live draft query, so no KV access consumes
        # it; the live position range remains the exact read/write frontier.
        seq_len = int(seq_lens[row])
        if seq_len not in (end, end + 1):
            raise ValueError(
                f"request {request!r} draft query ends at {end}, sequence length is {seq_len}"
            )
        lo = max(0, start - window + 1) if window is not None else 0
        write_rows.append(DispatchRow(request, writes=range(start, end)))
        attn_rows.append(DispatchRow(request, reads=range(lo, end)))
        entries = -(-end // block_size)
        table.append(block_table[row, :entries].tolist())
        valid.append(entries)
        first.append(lo // block_size)
        token_rows.extend([row] * (limit - begin))

    tag = f"-{site}" if site else ""
    write = DispatchDescriptor(
        f"write{tag}@{seq}",
        pool,
        "reshape_and_cache",
        tuple(write_rows),
        SlotMap(slots[:total], token_rows, pos[:total]),
    )
    attn = DispatchDescriptor(
        f"attn{tag}@{seq}",
        pool,
        "attention",
        tuple(attn_rows),
        DenseBlockTable(table, valid, first if window is not None else None),
    )
    return write, attn


def window_start(pre: int, window: int, extra_retained: int = 0) -> int:
    """The first live position of a sliding-window group's request before a step: vLLM frees the
    blocks below it (``computed - window + 1`` less the tokens retained for speculative
    re-prefill), and the step's queries attend from it."""
    return max(0, int(pre) - int(window) + 1 - int(extra_retained))


def mapping_tables(
    block_lists: Sequence[Sequence[BlockId | None]],
    pre_step_computed_per_row: Sequence[int],
    num_scheduled_per_row: Sequence[int],
    block_size: int,
    num_reqs_after_padding: int,
    null_block: BlockId = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The block table and slot map of a step derived from the authoritative mapping (the KV
    manager's block lists) and the scheduled positions, in the shape ``build_dispatches`` takes:
    rows beyond the real requests are the null block (as the gathered device table's padded
    rows), a hole in the mapping is the null block, and token ``t`` of request ``i`` at position
    ``p`` writes slot ``table[i][p // bs] * bs + p % bs``. The device copies are not read, so
    the step costs no device-to-host copy and no synchronization; the device copies' own
    content is then checked by the device validators only."""
    n = len(block_lists)
    table = dense_table(block_lists, num_reqs_after_padding, null_block)
    width = table.shape[1]
    slots: list[int] = []
    for i in range(n):
        start, cnt = int(pre_step_computed_per_row[i]), int(num_scheduled_per_row[i])
        for p in range(start, start + cnt):
            j = p // block_size
            b = int(table[i, j]) if j < width else null_block
            slots.append(b * block_size + p % block_size)
    return table, np.asarray(slots, dtype=np.int64)


def dense_table(
    block_lists: Sequence[Sequence[BlockId | None]],
    num_reqs_after_padding: int,
    null_block: BlockId = 0,
) -> np.ndarray:
    """The block lists as the gathered device table: one row per request (holes and the padded
    rows beyond the real requests are the null block), as wide as the longest list."""
    n = len(block_lists)
    width = max((len(bl) for bl in block_lists), default=0)
    table = np.full((max(num_reqs_after_padding, n), max(width, 1)), null_block, dtype=np.int32)
    for i, bl in enumerate(block_lists):
        if len(bl):
            table[i, : len(bl)] = [null_block if b is None else b for b in bl]
    return table


def state_columns(pre: int, scheduled: int, block_size: int, mode: str) -> tuple[int, int]:
    """The columns of a recurrent-state group's block list that one step of a request reads and
    writes: ``(src, dst)``, ``src`` -1 when the request has no prior state.

    In the ``align`` Mamba cache mode column ``j`` holds the state after block ``j``'s tokens;
    the read is the column after the last computed token, ``(pre - 1) // block_size``, and the
    write is the running block's, ``ceil((pre + scheduled) / block_size) - 1``. In the ``none``
    mode a request has one state block, column 0."""
    if mode == "none":
        return (0 if pre > 0 else -1), 0
    dst = -(-(pre + scheduled) // block_size) - 1
    src = (pre - 1) // block_size if pre > 0 else -1
    return src, dst


def build_state_dispatch(
    seq: int,
    pool: str,
    block_size: int,
    mode: str,
    req_ids: Sequence[RequestId],
    pre_step_computed_per_row: Sequence[int],
    num_scheduled_per_row: Sequence[int],
    block_table: np.ndarray,
    num_reqs_after_padding: int,
    *,
    site: str = "state",
) -> DispatchDescriptor:
    """One recurrent-state group's dispatch of a step over its gathered block table (the table's
    column is the core pool's position). Request ``i`` reads column ``src`` and writes column
    ``dst`` (``state_columns``) as two rows, the read first and alone so that it is satisfied only
    by an earlier step's write or an external copy, never by this step's own write of the same
    column. Each row's table consumes exactly its column (``DenseBlockTable.first``): the worker
    keeps a retired column's old id and never reads it. Padding rows carry the gathered row's
    entries as the attention dispatch does."""
    n = len(req_ids)
    rows: list[DispatchRow] = []
    table: list[list[int]] = []
    valid: list[int] = []
    first: list[int] = []
    for i in range(n):
        src, dst = state_columns(
            int(pre_step_computed_per_row[i]), int(num_scheduled_per_row[i]), block_size, mode
        )
        entries = block_table[i, : dst + 1].tolist()
        if src >= 0:
            rows.append(DispatchRow(req_ids[i], reads=range(src, src + 1)))
            table.append(entries[: src + 1])
            valid.append(src + 1)
            first.append(src)
        rows.append(DispatchRow(req_ids[i], writes=range(dst, dst + 1)))
        table.append(entries)
        valid.append(dst + 1)
        first.append(dst)
    for i in range(n, num_reqs_after_padding):
        rows.append(DispatchRow(None, padding=True))
        nz = np.flatnonzero(block_table[i])
        v = int(nz[-1]) + 1 if len(nz) else 0
        table.append(block_table[i, :v].tolist())
        valid.append(v)
        first.append(0)
    return DispatchDescriptor(
        f"{site}@{seq}",
        pool,
        "recurrent_state",
        tuple(rows),
        DenseBlockTable(table, valid, first),
    )


def cow_copy(
    pool: str, src: BlockId, dst: BlockId, block_size: int, num_computed: int | None
) -> ExternalCopy:
    """The worker's copy of a partial prefix hit's shared tail block into the request's private
    block: the ``num_computed % block_size`` positions the hit covers, or the whole block of a
    state group (block size one) and when the count is unknown. Recorded before the snapshot names
    the request as the block's owner, so it carries no request: the source's initialization and
    the destination's fill are judged, ownership by the snapshot's own rules."""
    k = block_size
    if block_size > 1 and num_computed is not None and num_computed % block_size:
        k = num_computed % block_size
    return ExternalCopy(pool, int(src), pool, int(dst), range(0, k), "cow")


def mediated_frees(blocks: Iterable[Mapping[str, int | bool]]) -> list[BlockId]:
    """Given ``free_blocks``' arguments after the call, the blocks that actually left the non-free
    set: unreferenced and unhashed. A hashed block with zero references stays reserved."""
    return [
        int(b["block_id"])
        for b in blocks
        if b["ref_cnt"] == 0 and not b["is_null"] and not b["has_hash"]
    ]


__all__ = [
    "SUPPORT_BOUNDARY",
    "BlockFacts",
    "RequestFacts",
    "block_state",
    "build_dispatches",
    "build_positioned_dispatches",
    "build_snapshot",
    "build_state_dispatch",
    "cow_copy",
    "dense_table",
    "mapping_tables",
    "mediated_frees",
    "pre_step_computed",
    "state_columns",
    "window_start",
]
