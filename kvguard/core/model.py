"""Object model and the normalized state the core consumes.

Plain data built by adapters from framework state; the core never sees framework objects. A pool
allocates physical blocks; a block holds slots (one per logical token position); a request's
mapping binds logical positions to slots. Blocks are FREE, PRIVATE (one writer) or PUBLISHED
(immutable, shareable); a reuse mechanism may reserve a published block that no request maps.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

PoolId = str
RequestId = str
BlockId = int
SlotId = int  # pool-wide slot index: block * block_size + offset


class BlockState(enum.Enum):
    FREE = "free"
    PRIVATE = "private"
    PUBLISHED = "published"


@dataclass(frozen=True)
class PoolGeometry:
    """Registration of one KV pool (adapter contract item 1)."""

    pool: PoolId
    num_blocks: int
    block_size: int
    """Slots (token positions) per block."""
    layers: int = 1
    kv_heads: int = 1
    components: tuple[str, ...] = ("K", "V")
    reserved_blocks: frozenset[BlockId] = frozenset()
    """Blocks permanently outside the request-managed namespace (null or scratch blocks)."""
    allocator: str | None = None
    """The allocator this pool shares with other pools, when one block id space backs several
    pools (vLLM KV cache groups): their non-free blocks and the allocator's free set form one
    partition, audited once over the union."""
    reserved_in_free: bool = True
    """Whether a listed block marked ``reserved`` (held only by the cache, evictable) also sits in
    the allocator's free set. The partition audit by count (``Snapshot.free_count``) counts a
    reserved listed block as free or held accordingly; the audit by set needs no flag."""
    rewritable: bool = False
    """A pool of owner-private slots intentionally rewritten in place (a recurrent layer's
    per-request state, a speculative draft layer's stale positions during re-prefill). A write to
    an initialized slot is an update, not ``double_write``; a slot must still be initialized before
    it is read, and two rows of one dispatch may not write one slot. A ``None`` mapping entry is a
    retired state column (no live position, no block), not a mapping error."""

    def __post_init__(self) -> None:
        if self.num_blocks <= 0 or self.block_size <= 0:
            raise ValueError("pool geometry needs positive num_blocks and block_size")
        bad = [b for b in self.reserved_blocks if not 0 <= b < self.num_blocks]
        if bad:
            raise ValueError(f"reserved blocks outside the pool: {sorted(bad)}")

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def slot(self, block: BlockId, offset: int) -> SlotId:
        return block * self.block_size + offset

    def block_of(self, slot: SlotId) -> BlockId:
        return slot // self.block_size

    def in_pool(self, block: BlockId) -> bool:
        return 0 <= block < self.num_blocks


@dataclass(frozen=True)
class RequestSnapshot:
    """One request's mapping in one pool, as the framework's authoritative structures describe it.

    ``blocks[i]`` holds logical positions ``[i * block_size, (i + 1) * block_size)``. ``num_tokens``
    is the number of live logical positions (the mapping length); positions at or beyond it are not
    mapped even if ``blocks`` has room for them. ``num_computed`` is the framework's count of
    positions whose KV it considers written; the core cross-checks it against its own record.
    """

    request: RequestId
    pool: PoolId
    blocks: tuple[BlockId | None, ...]
    """``None`` marks a block index whose positions are not live (a sliding-window hole)."""
    num_tokens: int
    num_computed: int
    first_token: int = 0
    """Positions below this are no longer live (sliding window); they may map to no block."""

    def __post_init__(self) -> None:
        if self.num_tokens < 0 or self.num_computed < 0 or self.first_token < 0:
            raise ValueError("token counts must be non-negative")
        if self.first_token > self.num_tokens:
            raise ValueError("first_token must not exceed num_tokens")

    def is_live(self, position: int) -> bool:
        return self.first_token <= position < self.num_tokens

    def live_positions(self) -> range:
        return range(self.first_token, self.num_tokens)

    def block_at(self, geometry: PoolGeometry, position: int) -> BlockId | None:
        index = position // geometry.block_size
        if not self.is_live(position) or index >= len(self.blocks):
            return None
        return self.blocks[index]

    def slot_at(self, geometry: PoolGeometry, position: int) -> SlotId | None:
        block = self.block_at(geometry, position)
        if block is None:
            return None
        return geometry.slot(block, position % geometry.block_size)

    def positions_in_block(self, geometry: PoolGeometry, index: int) -> range:
        """Live logical positions held by ``blocks[index]`` (empty for a ``None`` hole)."""
        if index < len(self.blocks) and self.blocks[index] is None:
            return range(0)
        start = index * geometry.block_size
        lo = max(start, self.first_token)
        hi = min(start + geometry.block_size, self.num_tokens)
        return range(lo, max(lo, hi))


@dataclass(frozen=True)
class BlockSnapshot:
    """The framework's view of one non-free block.

    ``reserved`` marks a published block kept by a reuse mechanism (prefix cache) while no request
    maps it. Owners and readers are *not* part of the snapshot: the core derives them from the
    request mappings and checks the result against the state (P2 permission configuration).
    """

    pool: PoolId
    block: BlockId
    state: BlockState
    reserved: bool = False
    refs: int | None = None
    """The framework's own count of requests holding the block (vLLM ``ref_cnt``), when it keeps
    one. The core compares it with the number of requests whose mappings name the block."""

    def __post_init__(self) -> None:
        if self.state is BlockState.FREE:
            raise ValueError("free blocks are implied by absence; do not list them")
        if self.reserved and self.state is not BlockState.PUBLISHED:
            raise ValueError("only published blocks can be reserved")


@dataclass(frozen=True)
class Snapshot:
    """A complete normalized snapshot at an access-issue point.

    ``blocks`` lists every non-free block of every registered pool; a block absent from it is free.
    ``free`` optionally carries the allocator's own free set per pool, for the partition audit. A
    pool named in ``delta_pools`` is listed as the change since the previous snapshot instead.
    """

    seq: int
    requests: tuple[RequestSnapshot, ...]
    blocks: tuple[BlockSnapshot, ...]
    free: Mapping[PoolId, frozenset[BlockId]] | None = None
    free_count: Mapping[PoolId, int] | None = None
    """The allocator's count of free blocks per pool, for the partition audit by count when the
    free set itself is not read this step: the blocks neither free nor reserved by the geometry
    must be exactly the listed blocks that hold a reference (a reserved published block sits in
    the allocator's free set). Ignored when ``free`` is given."""
    stream: int | None = None
    """The stream the framework enqueued this step's table writes on, when known: a consumed entry
    that changed under an open epoch on the same stream is an ordered rewrite, not
    ``mapping_mutated_in_flight``."""
    after: tuple[str, ...] = ()
    """Dispatch ids of the open epochs whose table reads the framework fenced before this step's
    table writes (the adapter asserts it observed the fence), for the same check."""
    delta_pools: frozenset[PoolId] = frozenset()
    """Pools listed as the change since the previous snapshot: ``blocks`` carries only the blocks
    of such a pool whose ``BlockSnapshot`` is new or differs, ``gone`` names the blocks that left
    its listing, and every other block stays listed with its previous object (recognized by
    identity). Other pools are listed in full."""
    gone: Mapping[PoolId, Collection[BlockId]] | None = None
    """Per pool of ``delta_pools``, the blocks that left its listing since the previous snapshot;
    none for a pool not named."""

    def request_index(self) -> dict[tuple[RequestId, PoolId], RequestSnapshot]:
        index: dict[tuple[RequestId, PoolId], RequestSnapshot] = {}
        for r in self.requests:
            key = (r.request, r.pool)
            if key in index:
                raise ValueError(f"request {r.request!r} listed twice for pool {r.pool!r}")
            index[key] = r
        return index

    def block_index(self) -> dict[tuple[PoolId, BlockId], BlockSnapshot]:
        index: dict[tuple[PoolId, BlockId], BlockSnapshot] = {}
        for b in self.blocks:
            key = (b.pool, b.block)
            if key in index:
                raise ValueError(f"block {b.block} of pool {b.pool!r} listed twice")
            index[key] = b
        return index


@dataclass(frozen=True)
class DispatchRow:
    """One batch entry of a dispatch: which request, and which logical positions it may touch.

    Positions are normally ranges because kernels address whole prefixes or whole suffixes.
    ``reads`` may instead be an immutable sequence for a gather such as one EAGLE tree branch,
    which reads the committed prefix followed by that branch's private speculative slots. A
    position listed in ``writes`` is written by this dispatch before any read of it in the same
    dispatch (stream order inside the dispatch), so the same row may read it.
    """

    request: RequestId | None
    reads: range | tuple[int, ...] = range(0)
    writes: range = range(0)
    padding: bool = False
    """A padding row (CUDA-graph batch padding): it belongs to no request and must touch no
    request-managed slot."""


@dataclass(frozen=True)
class DenseBlockTable:
    """``table[row][j]`` is the block holding logical positions ``[j*bs, (j+1)*bs)`` of the row's
    request; entries at ``j >= valid[row]`` are not consumed (vLLM FA2/Triton, TensorRT-LLM block
    offsets after decoding offsets to block ids)."""

    table: Sequence[Sequence[BlockId]]
    valid: Sequence[int]
    first: Sequence[int] | None = None
    """Per row, the first entry the kernel consumes, when it gathers a window ``[first, valid)`` of
    the row instead of its prefix (a recurrent-state kernel gathers one column); entries below it
    are not consumed and not compared with the mapping. ``None`` means every row starts at 0."""

    def entries(self, row: int) -> Sequence[BlockId]:
        return self.table[row][: self.valid[row]]

    def first_entry(self, row: int) -> int:
        return int(self.first[row]) if self.first is not None and row < len(self.first) else 0


@dataclass(frozen=True)
class SlotMap:
    """Flat per-token slot map (vLLM ``slot_mapping``, SGLang ``out_cache_loc``): token ``t`` of the
    dispatch is written to ``slots[t]``; ``rows[t]`` says which row the token belongs to and
    ``positions[t]`` its logical position. ``-1`` marks padding tokens that write nothing."""

    slots: Sequence[SlotId]
    rows: Sequence[int]
    positions: Sequence[int]


PADDING_SLOT: SlotId = -1
"""The slot-map entry of a padding token; any other slot outside the pool is a violation."""


@dataclass(frozen=True)
class CSRPageList:
    """FlashInfer-style page list: pages of row ``r`` are ``indices[indptr[r]:indptr[r+1]]`` and the
    last page holds ``last_page_len[r]`` positions."""

    indptr: Sequence[int]
    indices: Sequence[BlockId]
    last_page_len: Sequence[int]

    def entries(self, row: int) -> Sequence[BlockId]:
        return self.indices[self.indptr[row] : self.indptr[row + 1]]


@dataclass(frozen=True)
class TokenTable:
    """SGLang's per-token table.

    ``table[row][p - first[row]]`` is the slot of logical position ``p`` for
    ``first[row] <= p < first[row] + seq_len[row]``. ``first=None`` is the ordinary prefix
    table starting at position zero. The explicit start is needed by sliding-window FlashInfer
    wrappers, which gather only a suffix of the authoritative request row.
    """

    table: Sequence[Sequence[SlotId]]
    seq_len: Sequence[int]
    first: Sequence[int] | None = None

    def first_position(self, row: int) -> int:
        return int(self.first[row]) if self.first is not None and row < len(self.first) else 0


@dataclass(frozen=True)
class PositionedTokenTable:
    """A per-token gather whose entries name explicit logical positions.

    ``table[row][j]`` is the physical slot for ``positions[row][j]``.  This is distinct from a
    :class:`TokenTable`, whose positions are one contiguous interval.  SGLang EAGLE tree decode
    uses this form: every attention row reads the request prefix and one speculative branch,
    skipping the reserved slots of the other branches.
    """

    table: Sequence[Sequence[SlotId]]
    positions: Sequence[Sequence[int]]


MappingRepresentation = DenseBlockTable | SlotMap | CSRPageList | TokenTable | PositionedTokenTable


@dataclass(frozen=True)
class DispatchDescriptor:
    """What one GPU dispatch will access, bound to the snapshot it was prepared from."""

    dispatch_id: str
    pool: PoolId
    family: str
    rows: tuple[DispatchRow, ...]
    mapping: MappingRepresentation
    layers: tuple[int, ...] | None = None
    """Layers touched; None means all layers of the pool."""
    writes_first: bool = False
    """True when every write of this dispatch is ordered before every read (for example a
    cache-write kernel followed by attention over the whole batch), so a row may read what any
    row wrote in the same dispatch, whatever the row order. False keeps the conservative rule: a
    read is satisfied only by an earlier write of the same row."""
    snapshot_seq: int | None = None
    """The ``Snapshot.seq`` the descriptor was prepared from; when given, a dispatch validated
    against another snapshot is an integration error."""
    stream: int | None = None
    """The stream the dispatch's kernels run on, when known. With it the epoch is judged by stream
    order: a block freed or stored while the epoch is open retires instead of being reported, and
    the first dispatch that touches it again is legal on the same stream, legal on another stream
    only when its ``after`` names the epoch, and ``reuse_before_epoch_retired`` otherwise. ``None``
    keeps the host-order judgment, where the release itself is reported."""
    after: tuple[str, ...] = ()
    """Dispatch ids of the epochs this dispatch's stream has waited on, for the reuse of retiring
    blocks across streams."""
    advances_computed: bool = True
    """Whether writes advance the request's dense computed frontier. Sparse speculative-tree writes
    initialize their concrete slots but leave this false; a later explicit rollback reconciles the
    accepted frontier without treating holes between branches as computed."""


@dataclass
class ExternalFill:
    """Slots initialized by a non-kernel engine (onload, P/D receive)."""

    pool: PoolId
    block: BlockId
    offsets: range
    source: str = "external"
    request: RequestId | None = None
    """The request on whose behalf the fill happens, when known: a fill into another request's
    private block is then an access without permission."""
    position: int | None = None
    """The logical position of ``offsets.start`` in the request, when known; the tag tier stamps
    the filled slots with it and compares positions at later reads. ``None`` compares nothing."""


@dataclass(frozen=True)
class ExternalCopy:
    """Slots copied by a non-kernel engine from one block to another (a prefix hit's copy of a
    recurrent state into a request's slot, a checkpoint's store or load): the source is read and
    the destination is filled, both under the policies (``Ledger.external_copy``)."""

    pool: PoolId
    block: BlockId
    """The source."""
    dst_pool: PoolId
    dst_block: BlockId
    offsets: range
    source: str = "copy"
    request: RequestId | None = None
    """The request on whose behalf the copy happens, when known."""
    position: int | None = None
    """The logical position of ``offsets.start`` in the request at the source and the
    destination (a copy keeps positions), when known; see :attr:`ExternalFill.position`."""


@dataclass(frozen=True)
class ExternalRead:
    """An asynchronous non-kernel read (e.g. an offload store) that remains live until
    ``Ledger.complete``. Naming its source blocks lets the lifecycle epoch cover a later free or
    reuse; ``stream`` and ``after`` order it as on :class:`DispatchDescriptor`.
    """

    dispatch_id: str
    pool: PoolId
    blocks: tuple[BlockId, ...]
    offsets: range
    source: str = "external"
    request: RequestId | None = None
    stream: int | None = None
    preinitialized_blocks: frozenset[BlockId] = frozenset()
    """Blocks whose physical bytes were initialized by a framework clear. A clear creates no
    logical KV (so it must not make a later model write a ``double_write``) but makes a whole-page
    DMA read safe beyond a partial logical prefix. Only external byte-copy reads use this."""
    retained_free_blocks: frozenset[BlockId] = frozenset()
    """Allocator-free sources whose bytes a framework transfer fence still retains. The adapter
    binds this to the generation observed when the store job was published; reallocation advances
    that generation and cannot use the exception."""
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalWrite:
    """An asynchronous non-kernel write that remains live until ``Ledger.complete``.

    Opening this epoch protects the physical destinations from release or reuse but does *not*
    initialize their logical slots. The adapter issues :class:`ExternalFill` only after successful
    completion; a failed operation just completes the epoch once no writer can still touch them.
    """

    dispatch_id: str
    pool: PoolId
    blocks: tuple[BlockId, ...]
    offsets: range
    source: str = "external"
    request: RequestId | None = None
    stream: int | None = None
    after: tuple[str, ...] = ()


def positions_of(rows: Iterable[DispatchRow]) -> int:
    """Total number of logical positions a set of rows may touch (reads and writes)."""
    return sum(len(r.reads) + len(r.writes) for r in rows)


__all__ = [
    "PADDING_SLOT",
    "BlockId",
    "BlockSnapshot",
    "BlockState",
    "CSRPageList",
    "DenseBlockTable",
    "DispatchDescriptor",
    "DispatchRow",
    "ExternalCopy",
    "ExternalFill",
    "ExternalRead",
    "ExternalWrite",
    "MappingRepresentation",
    "PoolGeometry",
    "PoolId",
    "PositionedTokenTable",
    "RequestId",
    "RequestSnapshot",
    "SlotId",
    "SlotMap",
    "Snapshot",
    "TokenTable",
    "field",
    "positions_of",
]
