"""The shadow ledger: shadow state, allocator mediation, snapshot differencing, dispatch
validation and dispatch epochs.

The ledger is the only place P1-P4 decisions are made. Adapters feed it, in order of trust:
mediated allocator events (``allocate``/``free``, which advance per-block generations); snapshots
(``observe``, the authoritative management state, differenced against the shadow and validated
against the P1 automaton, P2 permissions, P3 mapping consistency and P4 publication/initialization);
and dispatch descriptors (``dispatch``/``complete``, the exact mapping a kernel will consume, which
also open an epoch so a free of a referenced block before ``complete`` is reported). It is the
semantic oracle; the GPU validator and the adapters' fast paths must agree with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import repeat
from typing import Any

from .model import (
    PADDING_SLOT,
    BlockId,
    BlockSnapshot,
    BlockState,
    CSRPageList,
    DenseBlockTable,
    DispatchDescriptor,
    DispatchRow,
    ExternalCopy,
    ExternalFill,
    ExternalRead,
    ExternalWrite,
    PoolGeometry,
    PoolId,
    PositionedTokenTable,
    RequestId,
    RequestSnapshot,
    SlotMap,
    Snapshot,
    TokenTable,
)
from .violations import Policy, Report, Reporter, integration_error, unsupported, violation


@dataclass
class BlockShadow:
    state: BlockState = BlockState.FREE
    generation: int = 0
    owner: RequestId | None = None
    readers: frozenset[RequestId] = frozenset()
    reserved: bool = False
    filled: int = 0
    """Bitmask over slot offsets: bit i set means slot i of the block holds initialized entries."""
    pending_publish: int = 0
    """Bitmask of slot offsets still unwritten when the framework declared the block published.
    Publication takes effect with the dispatch that writes these slots; they must be written
    before the next snapshot, by a request that maps the block."""
    alloc_seen: bool = False
    free_seen: bool = False
    """Mediated events observed since the last snapshot; cleared by `observe`."""

    def is_filled(self, offset: int) -> bool:
        return bool(self.filled >> offset & 1)


@dataclass
class RequestShadow:
    snapshot: RequestSnapshot
    seq: int
    entries: list[tuple[BlockId | None, int]] = field(default_factory=list)
    """Per block index of the mapping: the managed block the entry counts toward (None for a hole,
    out-of-pool or reserved id) and the live offsets it holds there. The snapshot's incremental
    state, with the fields below."""
    full_count: int = 0
    full_seen: dict[BlockId, int] = field(default_factory=dict)
    """Duplicate-slot masks over the first ``full_count`` entries (the mapping's full blocks)."""
    reports: list[tuple[int, Report]] = field(default_factory=list)
    """Mapping reports per entry index, re-emitted while the entry is kept."""
    verified: int = 0
    """Positions below this were found initialized by the last computed-claim check."""
    computed_high_water: int = 0
    """Highest logical position observed computed or written. Dispatch writes advance this
    watermark so a later speculative rollback can invalidate rejected slots before rewriting
    them."""
    read_lo: int = 0
    read_hi: int = 0
    read_gen: int = -1
    read_base: int = 0
    read_blocks: tuple[BlockId | None, ...] = ()
    slot_cache: tuple[tuple[BlockId | None, ...], tuple[int, ...]] | None = None
    """The mapping's blocks expanded to one slot per position (``-1`` for a hole), kept with the
    block tuple it was built from, so a per-token table compares a row in one tuple comparison."""
    """The positions ``[read_lo, read_hi)`` a read-only row was allowed to read at touch
    generation ``read_gen``, through entries ``read_blocks`` from block index ``read_base``: a
    later read-only row skips those positions whose entries are unchanged and whose blocks were
    not touched since."""


@dataclass
class Epoch:
    descriptor: DispatchDescriptor | ExternalRead | ExternalWrite
    seq: int
    blocks: frozenset[tuple[PoolId, BlockId]]
    consumed: tuple[tuple[RequestId, Sequence[int], tuple[BlockId | None, ...]], ...]
    """Per row (a request may own several): the request, the block indices the row consumed and
    the managed block (or None) at each; a later snapshot that changes one while the epoch is open
    is a mutation of a table in use. Reserved and out-of-pool entries are stored as None, so an
    unchanged hole does not read as a change."""
    stream: int | None = None
    """The descriptor's stream: with it, releases of the epoch's blocks retire and are judged at
    their reuse by stream order; without it they are judged in host order at the release."""


def _full_mask(n: int) -> int:
    return (1 << n) - 1


def _mask(offsets: Iterable[int]) -> int:
    m = 0
    for o in offsets:
        m |= 1 << o
    return m


class Ledger:
    """Shadow state and policy decisions for all registered pools."""

    def __init__(self, reporter: Reporter | None = None, *, mediated: bool = True):
        self.reporter = reporter or Reporter(fail_fast=True)
        self._gen = 0
        self.touched_at: dict[PoolId, list[int]] = {}
        self.nreports = 0
        self.mediated = mediated
        self.pools: dict[PoolId, PoolGeometry] = {}
        self.blocks: dict[PoolId, list[BlockShadow]] = {}
        self.requests: dict[tuple[RequestId, PoolId], RequestShadow] = {}
        self.epochs: dict[str, Epoch] = {}
        self.resource_uses: dict[str, frozenset[str]] = {}
        """Asynchronous operations and the mutable descriptor buffers they still consume. Adapters
        end each use only after the framework's completion primitive says the consumer is done; a
        second use of the same buffer before then can change which KV entries an already-issued
        operation accesses."""
        self.resource_users: dict[str, set[str]] = {}
        """Inverse of ``resource_uses``: active operation ids per resource identity."""
        self.seq: int | None = None
        self.listeners: list[Any] = []
        """Objects told of the mediated events after the ledger processed them: ``on_allocate``,
        ``on_free``, ``on_store``, ``on_reuse(pool, block, retained)``, ``on_invalidate(pool,
        pairs)`` for cleared (block, offset) pairs, ``on_external_fill(fill)`` and
        ``on_external_copy(copy)``. The tag tier keeps its tags from these and the publish stamps;
        it reads no shadow state."""
        self.dirty: dict[PoolId, set[BlockId]] = {}
        self.published_since: dict[PoolId, list[BlockId]] = {}
        """Blocks a snapshot's transitions published since ``drain_published`` (for a tag tier
        whose framework offers no publish hook; the publication is then state-derived)."""
        self.nonfree: dict[PoolId, set[BlockId]] = {}
        self.nonfree_mask: dict[PoolId, bytearray] = {}
        """``nonfree`` as one byte per block id (1 when non-free), for an adapter that derives its
        view of a pool with array operations."""
        """Blocks whose shadow is not free, so whole-pool passes walk the live state instead of the
        pool. ``dirty``: blocks whose shadow changed since ``drain_dirty`` (for a device mirror of
        the shadow, the dispatch validators' input)."""
        # The incremental state of `observe`, per pool.
        self._touched: dict[PoolId, set[BlockId]] = {}
        """Blocks an allocator event, or a shrink of the initialization record inside a snapshot,
        touched since the last snapshot's claim checks."""
        self._pending: dict[PoolId, set[BlockId]] = {}
        """Blocks given a pending publication by the last snapshot."""
        self._listed: dict[PoolId, dict[BlockId, BlockSnapshot]] = {}
        self._held: dict[PoolId, int] = {}
        self._rejected: dict[PoolId, dict[BlockId, BlockSnapshot]] = {}
        """Per pool listed as a change, the entries the structural check rejected and the adapter
        has not dropped or replaced since; their reports repeat until the delta names them gone."""
        """``_held``: per pool, the listed blocks that hold a reference, kept for the partition
        audit by count. ``_listed``: the accepted block entries of the last snapshot."""
        self._mappers: dict[PoolId, dict[BlockId, dict[RequestId, int]]] = {}
        """Per block, the requests whose mapping names it and how many of their entries do."""
        self._block_reports: dict[PoolId, dict[BlockId, list[Report]]] = {}
        """Per block, the configuration reports of its last judgement, re-emitted while unchanged."""
        self.retiring: dict[tuple[PoolId, BlockId], dict[str, int]] = {}
        """Blocks released while stream-carrying epochs were open: per block, those epochs and
        their streams. The release is judged at the first dispatch that touches the block again
        (``_judge_retiring``), by stream order; a completion drops the epoch from every entry."""
        self._retiring_epochs: dict[str, set[tuple[PoolId, BlockId]]] = {}
        self.counters: dict[str, int] = {
            "released_in_flight": 0,
            "retired_in_order": 0,
            "retired_by_after": 0,
            "row_released_in_flight": 0,
            "mapping_rewritten_in_order": 0,
            "read_positions_skipped": 0,
            "token_table_fast_path": 0,
            "speculative_rollbacks": 0,
            "speculative_sparse_invalidations": 0,
            "scratch_rewrites_authorized": 0,
        }
        """Counted legal events visible in the adapters' records: the stream-order rule's
        (releases deferred, reuses found ordered, rows released and tables rewritten under an open
        epoch) and the read positions a read-only row skipped as already verified."""

    SHADOW_ATTRIBUTES = (
        "pools", "blocks", "requests", "epochs", "resource_uses", "resource_users", "dirty",
        "published_since", "nonfree", "nonfree_mask", "touched_at", "retiring",
        "_touched", "_pending", "_listed", "_held", "_rejected", "_mappers", "_block_reports",
        "_retiring_epochs",
    )  # fmt: skip
    """The structures that are the host shadow: :meth:`memory_bytes` sums them."""

    def memory_bytes(self, seen: set[int] | None = None) -> dict[str, int]:
        """The host shadow's deep size in bytes, per structure and in total, each object counted
        once across the structures (and across the caller's sums when it passes ``seen``)."""
        from .memsize import deep_size

        seen = set() if seen is None else seen
        out = {name: deep_size(getattr(self, name), seen) for name in self.SHADOW_ATTRIBUTES}
        out["total"] = sum(out.values())
        return out

    def register_pool(self, geometry: PoolGeometry) -> None:
        if geometry.pool in self.pools:
            raise ValueError(f"pool {geometry.pool!r} already registered")
        self.pools[geometry.pool] = geometry
        self.blocks[geometry.pool] = [BlockShadow() for _ in range(geometry.num_blocks)]
        self.dirty[geometry.pool] = set()
        self.published_since[geometry.pool] = []
        self.touched_at[geometry.pool] = [0] * geometry.num_blocks
        self.nonfree[geometry.pool] = set()
        self.nonfree_mask[geometry.pool] = bytearray(geometry.num_blocks)
        self._touched[geometry.pool] = set()
        self._pending[geometry.pool] = set()
        self._listed[geometry.pool] = {}
        self._held[geometry.pool] = 0
        self._rejected[geometry.pool] = {}
        self._mappers[geometry.pool] = {}
        self._block_reports[geometry.pool] = {}

    def _touch(self, pool: PoolId, block: BlockId) -> None:
        self.dirty[pool].add(block)
        self._gen += 1
        self.touched_at[pool][block] = self._gen

    def drain_dirty(self, pool: PoolId) -> set[BlockId]:
        """The blocks whose shadow changed since the previous call; clears the set."""
        out = self.dirty[pool]
        self.dirty[pool] = set()
        return out

    def drain_published(self, pool: PoolId) -> list[BlockId]:
        """The blocks a snapshot published since the last call (see ``published_since``)."""
        out = self.published_since[pool]
        self.published_since[pool] = []
        return out

    def block(self, pool: PoolId, block: BlockId) -> BlockShadow:
        return self.blocks[pool][block]

    def inflight(self, pool: PoolId, block: BlockId) -> int:
        """The open epochs that may access the block: the open epochs whose touched set names
        it."""
        key = (pool, block)
        return sum(1 for e in self.epochs.values() if key in e.blocks)

    def _referencing(self, pool: PoolId, block: BlockId) -> tuple[int, dict[str, int]]:
        """The open epochs that may access the block, by how they judge a release of it: the
        count of host-order epochs (no stream) and, per epoch that carries a stream, the stream."""
        key = (pool, block)
        host = 0
        streamed: dict[str, int] = {}
        for did, e in self.epochs.items():
            if key in e.blocks:
                if e.stream is None:
                    host += 1
                else:
                    streamed[did] = e.stream
        return host, streamed

    def _retire(self, pool: PoolId, block: BlockId, streamed: dict[str, int]) -> None:
        """Record a release of the block while the given stream-carrying epochs are open; the
        first dispatch that touches the block again judges it (``_judge_retiring``)."""
        if not streamed:
            return
        key = (pool, block)
        entry = self.retiring.setdefault(key, {})
        for did, stream in streamed.items():
            if did not in entry:
                entry[did] = stream
                self._retiring_epochs.setdefault(did, set()).add(key)
        self.counters["released_in_flight"] += 1

    def _geometry(self, pool: PoolId, site: str) -> PoolGeometry | None:
        geometry = self.pools.get(pool)
        if geometry is None:
            self._report(
                unsupported(
                    "unknown_pool", f"pool {pool!r} is not registered", pool=pool, site=site
                )
            )
        return geometry

    def _report(self, report: Report) -> None:
        self.nreports += 1
        self.reporter.emit(report)

    # ------------------------------------------------ asynchronous descriptor resources

    def begin_resource_use(
        self, operation: str, resources: Iterable[str], *, site: str = "resource"
    ) -> None:
        """Register mutable descriptor buffers consumed by an asynchronous operation; the
        framework must not mutate one while its consumer is still in flight."""
        if operation in self.resource_uses:
            self._report(
                integration_error(
                    "duplicate_resource_operation",
                    f"asynchronous resource operation {operation!r} is already active",
                    site=site,
                )
            )
            return
        selected = frozenset(str(resource) for resource in resources)
        conflicts = {
            resource: tuple(sorted(self.resource_users.get(resource, ())))
            for resource in selected
            if self.resource_users.get(resource)
        }
        if conflicts:
            detail = ", ".join(
                f"{resource} still used by {list(users)!r}"
                for resource, users in sorted(conflicts.items())
            )
            self._report(
                violation(
                    Policy.P3,
                    "resource_reused_in_flight",
                    f"operation {operation!r} reuses mutable descriptor storage: {detail}",
                    site=site,
                )
            )
        self.resource_uses[operation] = selected
        for resource in selected:
            self.resource_users.setdefault(resource, set()).add(operation)

    def end_resource_use(self, operation: str, *, site: str = "resource") -> None:
        """Release an asynchronous operation's descriptor buffers after observed completion."""
        selected = self.resource_uses.pop(operation, None)
        if selected is None:
            self._report(
                integration_error(
                    "unknown_resource_operation",
                    f"asynchronous resource operation {operation!r} is not active",
                    site=site,
                )
            )
            return
        for resource in selected:
            users = self.resource_users[resource]
            users.discard(operation)
            if not users:
                del self.resource_users[resource]

    def validate_padding_destinations(
        self,
        destinations: Iterable[int],
        active_length: int,
        reserved: Iterable[int],
        *,
        site: str = "padding",
    ) -> None:
        """Check executed descriptor entries beyond the real batch against discard sentinels.

        For a narrow adapter that observes an exact persistent graph descriptor but cannot build
        the full pool snapshot; an executed padding lane may touch only a declared reserved
        destination.
        """
        values = tuple(int(destination) for destination in destinations)
        allowed = frozenset(int(destination) for destination in reserved)
        if not 0 <= active_length <= len(values):
            self._report(
                integration_error(
                    "malformed_padding_mapping",
                    f"active length {active_length} is outside descriptor length {len(values)}",
                    site=site,
                )
            )
            return
        for index, destination in enumerate(values[active_length:], start=active_length):
            if destination in allowed:
                continue
            self._report(
                violation(
                    Policy.P2,
                    "padding_row_names_managed_block",
                    f"padding entry {index} names destination {destination}; "
                    f"expected a reserved destination in {sorted(allowed)}",
                    slot=destination,
                    site=site,
                )
            )

    # ------------------------------------------------------------------ allocator mediation

    def allocate(self, pool: PoolId, blocks: Iterable[BlockId], *, site: str = "allocator") -> None:
        """A mediated allocation. Legal from FREE, or from a reserved published block that no request
        maps (eviction and reallocation in one allocator step)."""
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        accepted: list[BlockId] = []
        for b in blocks:
            if not geometry.in_pool(b):
                self._report(
                    violation(
                        Policy.P1,
                        "allocate_out_of_pool",
                        f"allocated block {b} outside pool of {geometry.num_blocks} blocks",
                        pool=pool,
                        block=b,
                        site=site,
                    )
                )
                continue
            if b in geometry.reserved_blocks:
                self._report(
                    violation(
                        Policy.P1,
                        "allocate_reserved_block",
                        "allocated a block reserved outside the request-managed namespace",
                        pool=pool,
                        block=b,
                        site=site,
                    )
                )
                continue
            sh = self.blocks[pool][b]
            evictable = sh.state is BlockState.PUBLISHED and sh.reserved and not sh.readers
            if sh.state is not BlockState.FREE and not evictable:
                self._report(
                    violation(
                        Policy.P1,
                        "allocate_non_free",
                        f"allocated a {sh.state.value} block (owner={sh.owner}, readers={sorted(sh.readers)}, reserved={sh.reserved})",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=sh.owner,
                        site=site,
                    )
                )
                continue
            inflight, streamed = self._referencing(pool, b)
            if inflight:
                self._report(
                    violation(
                        Policy.P1,
                        "allocate_in_flight",
                        f"allocated a block still referenced by {inflight} open dispatch(es)",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
            self._retire(pool, b, streamed)
            sh.generation += 1
            sh.state = BlockState.PRIVATE
            sh.owner = None
            sh.readers = frozenset()
            sh.reserved = False
            sh.filled = 0
            sh.pending_publish = 0
            sh.alloc_seen = True
            self.nonfree[pool].add(b)
            self.nonfree_mask[pool][b] = 1
            self._touch(pool, b)
            accepted.append(b)
            self._touched[pool].add(b)
        if accepted:
            for listener in self.listeners:
                listener.on_allocate(pool, accepted)

    def free(self, pool: PoolId, blocks: Iterable[BlockId], *, site: str = "allocator") -> None:
        """A mediated free. Mappings may still name the block until the next snapshot (the request
        table is updated in the same framework step); a block that stays mapped is reported there."""
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        accepted: list[BlockId] = []
        for b in blocks:
            if not geometry.in_pool(b):
                self._report(
                    violation(
                        Policy.P1,
                        "free_out_of_pool",
                        f"freed block {b} outside the pool",
                        pool=pool,
                        block=b,
                        site=site,
                    )
                )
                continue
            sh = self.blocks[pool][b]
            if sh.state is BlockState.FREE:
                self._report(
                    violation(
                        Policy.P1,
                        "double_free",
                        "freed a block that is already free",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
                continue
            inflight, streamed = self._referencing(pool, b)
            if inflight:
                self._report(
                    violation(
                        Policy.P1,
                        "free_while_in_flight",
                        f"freed a block referenced by {inflight} open dispatch(es)",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=sh.owner,
                        site=site,
                    )
                )
            self._retire(pool, b, streamed)
            sh.state = BlockState.FREE
            sh.owner = None
            sh.readers = frozenset()
            sh.reserved = False
            sh.filled = 0
            sh.pending_publish = 0
            sh.free_seen = True
            self.nonfree[pool].discard(b)
            self.nonfree_mask[pool][b] = 0
            self._touch(pool, b)
            self._touched[pool].add(b)
            accepted.append(b)
        if accepted:
            for listener in self.listeners:
                listener.on_free(pool, accepted)

    def store(self, pool: PoolId, blocks: Iterable[BlockId], *, site: str = "allocator") -> None:
        """A mediated release into a reuse cache: the owner gives the blocks up and they stay
        published with their content, reserved for a later hit or take-over. Legal from a private
        block; a block already published keeps its readers. Storing while a dispatch still
        references the block makes it eligible for reuse before the epoch retired: reported here,
        and again at the later ``reuse`` or allocation and at the next snapshot."""
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        stored: list[BlockId] = []
        for b in blocks:
            if not geometry.in_pool(b) or b in geometry.reserved_blocks:
                self._report(
                    violation(
                        Policy.P1,
                        "store_out_of_pool",
                        f"stored block {b} is outside the request-managed namespace",
                        pool=pool,
                        block=b,
                        site=site,
                    )
                )
                continue
            sh = self.blocks[pool][b]
            if sh.state is BlockState.FREE:
                self._report(
                    violation(
                        Policy.P1,
                        "store_free_block",
                        "stored a free block for reuse",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
                continue
            inflight, streamed = self._referencing(pool, b)
            if inflight:
                self._report(
                    violation(
                        Policy.P1,
                        "store_while_in_flight",
                        f"released for reuse a block referenced by {inflight} open dispatch(es)",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=sh.owner,
                        site=site,
                    )
                )
            self._retire(pool, b, streamed)
            if sh.state is BlockState.PRIVATE:
                sh.state = BlockState.PUBLISHED
                stored.append(b)
                sh.owner = None
                sh.readers = frozenset()
                sh.reserved = True
                sh.pending_publish = 0  # a stored partial leaf keeps its unwritten tail unread
                self._touch(pool, b)
                self._touched[pool].add(b)
        if stored:
            for listener in self.listeners:
                listener.on_store(pool, stored)

    def reuse(
        self, pool: PoolId, block: BlockId, retained: int, *, site: str = "allocator"
    ) -> None:
        """A mediated reuse with content: a published block kept by a reuse mechanism (reserved,
        no reader) is handed to a new request as its private block with its first ``retained``
        slots still initialized. Legal from a reserved published block only; the retained slots
        must have been written."""
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        if not geometry.in_pool(block) or block in geometry.reserved_blocks:
            self._report(
                violation(
                    Policy.P1,
                    "reuse_out_of_pool",
                    f"reused block {block} is outside the request-managed namespace",
                    pool=pool,
                    block=block,
                    site=site,
                )
            )
            return
        sh = self.blocks[pool][block]
        if sh.state is BlockState.FREE:
            self._report(
                violation(
                    Policy.P1,
                    "reuse_free_block",
                    "reused the content of a free block",
                    pool=pool,
                    block=block,
                    generation=sh.generation,
                    site=site,
                )
            )
            return
        if sh.state is not BlockState.PUBLISHED or not sh.reserved or sh.readers:
            self._report(
                violation(
                    Policy.P1,
                    "reuse_non_reserved",
                    f"reused a {sh.state.value} block that is mapped (owner={sh.owner}, readers={sorted(sh.readers)}, reserved={sh.reserved})",
                    pool=pool,
                    block=block,
                    generation=sh.generation,
                    request=sh.owner,
                    site=site,
                )
            )
            return
        inflight, streamed = self._referencing(pool, block)
        if inflight:
            self._report(
                violation(
                    Policy.P1,
                    "reuse_in_flight",
                    f"reused a block still referenced by {inflight} open dispatch(es)",
                    pool=pool,
                    block=block,
                    generation=sh.generation,
                    site=site,
                )
            )
        self._retire(pool, block, streamed)
        retained = max(0, min(int(retained), geometry.block_size))
        keep = _full_mask(retained)
        unwritten = keep & ~sh.filled
        if unwritten:
            o = (unwritten & -unwritten).bit_length() - 1
            self._report(
                violation(
                    Policy.P4,
                    "reuse_uninitialized_content",
                    f"reuse keeps {retained} slots but slot offset {o} was never written",
                    pool=pool,
                    block=block,
                    slot=geometry.slot(block, o),
                    generation=sh.generation,
                    site=site,
                )
            )
        sh.generation += 1
        sh.state = BlockState.PRIVATE
        sh.owner = None
        sh.readers = frozenset()
        sh.reserved = False
        sh.filled &= keep
        sh.pending_publish = 0
        sh.alloc_seen = True
        self._touch(pool, block)
        self._touched[pool].add(block)
        for listener in self.listeners:
            listener.on_reuse(pool, block, retained)

    def external_copy(self, copy: ExternalCopy, *, site: str = "external") -> None:
        """A non-kernel engine copies slots from one block to another. The source is read: a
        free block, or one private to another request, is an access without permission (P2),
        and a slot never written is read uninitialized (P4); the destination is then filled as
        ``external_fill`` does."""
        geometry = self._geometry(copy.pool, site)
        if geometry is None:
            return
        if not geometry.in_pool(copy.block):
            self._report(
                violation(
                    Policy.P2,
                    "access_out_of_pool",
                    f"{copy.source} reads block {copy.block} outside the pool",
                    pool=copy.pool,
                    block=copy.block,
                    site=site,
                )
            )
        else:
            sh = self.blocks[copy.pool][copy.block]
            if sh.state is BlockState.FREE:
                self._report(
                    violation(
                        Policy.P2,
                        "access_free_block",
                        f"{copy.source} reads free block {copy.block}",
                        pool=copy.pool,
                        block=copy.block,
                        generation=sh.generation,
                        request=copy.request,
                        site=site,
                    )
                )
            elif (
                sh.state is BlockState.PRIVATE
                and copy.request is not None
                and sh.owner is not None
                and sh.owner != copy.request
            ):
                self._report(
                    violation(
                        Policy.P2,
                        "access_foreign_private",
                        f"{copy.source} for request {copy.request!r} reads block {copy.block}, private to {sh.owner!r}",
                        pool=copy.pool,
                        block=copy.block,
                        generation=sh.generation,
                        request=copy.request,
                        site=site,
                    )
                )
            else:
                for o in copy.offsets:
                    if 0 <= o < geometry.block_size and not sh.is_filled(o):
                        self._report(
                            violation(
                                Policy.P4,
                                "read_uninitialized",
                                f"{copy.source} reads slot offset {o} of block {copy.block}, never written",
                                pool=copy.pool,
                                block=copy.block,
                                slot=geometry.slot(copy.block, o),
                                generation=sh.generation,
                                request=copy.request,
                                site=site,
                            )
                        )
                        break
        self._fill(
            ExternalFill(copy.dst_pool, copy.dst_block, copy.offsets, copy.source, copy.request),
            site=site,
        )
        if geometry.in_pool(copy.block):
            for listener in self.listeners:
                listener.on_external_copy(copy)

    def external_read(self, read: ExternalRead, *, site: str = "external") -> None:
        """Validate a non-kernel source read and open its lifecycle epoch.

        Unlike :meth:`external_copy` it has no managed destination and is not complete when the
        host call returns; the adapter closes ``read.dispatch_id`` only when the device is done.
        """
        if self.seq is None:
            self._report(
                integration_error(
                    "external_read_before_snapshot",
                    "no snapshot has been observed",
                    pool=read.pool,
                    site=site,
                )
            )
            return
        if read.dispatch_id in self.epochs:
            self._report(
                integration_error(
                    "duplicate_dispatch_id",
                    f"dispatch {read.dispatch_id!r} is already open",
                    pool=read.pool,
                    site=site,
                )
            )
            return
        geometry = self._geometry(read.pool, site)
        if geometry is None:
            return
        mask = _mask(o for o in read.offsets if 0 <= o < geometry.block_size)
        touched: set[BlockId] = set()
        for block in read.blocks:
            if not geometry.in_pool(block) or block in geometry.reserved_blocks:
                self._report(
                    violation(
                        Policy.P2,
                        "access_out_of_pool",
                        f"{read.source} reads block {block} outside the request-managed pool",
                        pool=read.pool,
                        block=block,
                        request=read.request,
                        site=site,
                    )
                )
                continue
            touched.add(block)
            shadow = self.blocks[read.pool][block]
            retained_free = shadow.state is BlockState.FREE and block in read.retained_free_blocks
            if shadow.state is BlockState.FREE and not retained_free:
                self._report(
                    violation(
                        Policy.P2,
                        "access_free_block",
                        f"{read.source} reads free block {block}",
                        pool=read.pool,
                        block=block,
                        generation=shadow.generation,
                        request=read.request,
                        site=site,
                    )
                )
                continue
            if (
                not retained_free
                and shadow.state is BlockState.PRIVATE
                and read.request is not None
                and shadow.owner is not None
                and shadow.owner != read.request
            ):
                self._report(
                    violation(
                        Policy.P2,
                        "access_foreign_private",
                        f"{read.source} for request {read.request!r} reads block {block}, "
                        f"private to {shadow.owner!r}",
                        pool=read.pool,
                        block=block,
                        generation=shadow.generation,
                        request=read.request,
                        site=site,
                    )
                )
                continue
            missing = mask & ~shadow.filled
            if retained_free or block in read.preinitialized_blocks:
                missing = 0
            if missing:
                offset = (missing & -missing).bit_length() - 1
                self._report(
                    violation(
                        Policy.P4,
                        "read_uninitialized",
                        f"{read.source} reads slot offset {offset} of block {block}, never written",
                        pool=read.pool,
                        block=block,
                        slot=geometry.slot(block, offset),
                        generation=shadow.generation,
                        request=read.request,
                        site=site,
                    )
                )
        if self.retiring:
            self._judge_retiring(read, touched, site)
        self.epochs[read.dispatch_id] = Epoch(
            read,
            self.seq,
            frozenset(zip(repeat(read.pool), touched)),
            (),
            read.stream,
        )

    def external_write(self, write: ExternalWrite, *, site: str = "external") -> None:
        """Validate asynchronous destination blocks and open their lifecycle epoch.

        Unlike :meth:`external_fill`, submission does not initialize logical KV (the writer may
        still be in flight); a successful completion is one or more ``external_fill`` calls then
        ``complete``, a failed one a bare ``complete`` once the writer is proven stopped. May
        precede the first snapshot: mediated allocation is enough to establish destination liveness.
        """
        if write.dispatch_id in self.epochs:
            self._report(
                integration_error(
                    "duplicate_dispatch_id",
                    f"dispatch {write.dispatch_id!r} is already open",
                    pool=write.pool,
                    site=site,
                )
            )
            return
        geometry = self._geometry(write.pool, site)
        if geometry is None:
            return
        invalid_offsets = [o for o in write.offsets if not 0 <= o < geometry.block_size]
        if invalid_offsets:
            self._report(
                violation(
                    Policy.P3,
                    "fill_offset_out_of_block",
                    f"offset {invalid_offsets[0]} outside the block",
                    pool=write.pool,
                    site=site,
                )
            )
        touched: set[BlockId] = set()
        for block in write.blocks:
            if not geometry.in_pool(block) or block in geometry.reserved_blocks:
                self._report(
                    violation(
                        Policy.P2,
                        "access_out_of_pool",
                        f"{write.source} writes block {block} outside the request-managed pool",
                        pool=write.pool,
                        block=block,
                        request=write.request,
                        site=site,
                    )
                )
                continue
            touched.add(block)
            shadow = self.blocks[write.pool][block]
            if shadow.state is BlockState.FREE:
                self._report(
                    violation(
                        Policy.P2,
                        "write_to_free_block",
                        f"{write.source} writes free block {block}",
                        pool=write.pool,
                        block=block,
                        generation=shadow.generation,
                        request=write.request,
                        site=site,
                    )
                )
            elif (
                shadow.state is BlockState.PRIVATE
                and write.request is not None
                and shadow.owner is not None
                and shadow.owner != write.request
            ):
                self._report(
                    violation(
                        Policy.P2,
                        "access_foreign_private",
                        f"{write.source} for request {write.request!r} writes block {block}, "
                        f"private to {shadow.owner!r}",
                        pool=write.pool,
                        block=block,
                        generation=shadow.generation,
                        request=write.request,
                        site=site,
                    )
                )
        if self.retiring:
            self._judge_retiring(write, touched, site)
        self.epochs[write.dispatch_id] = Epoch(
            write,
            self.seq if self.seq is not None else 0,
            frozenset(zip(repeat(write.pool), touched)),
            (),
            write.stream,
        )

    def external_fill(self, fill: ExternalFill, *, site: str = "external") -> None:
        """Slots initialized by a non-kernel engine (onload, transfer receive). Writes must still
        obey P2 (private block) and P4 (write once)."""
        if self._fill(fill, site=site):
            for listener in self.listeners:
                listener.on_external_fill(fill)

    def _fill(self, fill: ExternalFill, *, site: str) -> bool:
        """``external_fill`` without the listeners; True when the fill names an in-pool block
        (whatever the policies said of it)."""
        geometry = self._geometry(fill.pool, site)
        if geometry is None:
            return False
        if not geometry.in_pool(fill.block):
            self._report(
                violation(
                    Policy.P1,
                    "fill_out_of_pool",
                    "external fill names a block outside the pool",
                    pool=fill.pool,
                    block=fill.block,
                    site=site,
                )
            )
            return False
        sh = self.blocks[fill.pool][fill.block]
        if sh.state is BlockState.FREE:
            self._report(
                violation(
                    Policy.P2,
                    "write_to_free_block",
                    f"{fill.source} wrote a free block",
                    pool=fill.pool,
                    block=fill.block,
                    generation=sh.generation,
                    site=site,
                )
            )
            return True
        if sh.state is BlockState.PRIVATE and fill.request is not None and sh.owner != fill.request:
            self._report(
                violation(
                    Policy.P2,
                    "access_foreign_private",
                    f"{fill.source} for request {fill.request!r} wrote block {fill.block}, private to {sh.owner!r}",
                    pool=fill.pool,
                    block=fill.block,
                    generation=sh.generation,
                    request=fill.request,
                    site=site,
                )
            )
            return True
        if sh.state is BlockState.PUBLISHED:
            # A pending publication (the sharer's unfilled offsets) may be completed by any writer,
            # as by a dispatch; anything else is a write to published memory.
            pending = all(
                sh.pending_publish >> o & 1 for o in fill.offsets if 0 <= o < geometry.block_size
            )
            if not pending:
                self._report(
                    violation(
                        Policy.P2,
                        "write_to_published",
                        f"{fill.source} wrote a published block",
                        pool=fill.pool,
                        block=fill.block,
                        generation=sh.generation,
                        site=site,
                    )
                )
                return True
        for o in fill.offsets:
            if not 0 <= o < geometry.block_size:
                self._report(
                    violation(
                        Policy.P3,
                        "fill_offset_out_of_block",
                        f"offset {o} outside the block",
                        pool=fill.pool,
                        block=fill.block,
                        site=site,
                    )
                )
                continue
            if sh.is_filled(o) and not geometry.rewritable:
                self._report(
                    violation(
                        Policy.P4,
                        "double_write",
                        f"{fill.source} wrote an initialized slot",
                        pool=fill.pool,
                        block=fill.block,
                        slot=geometry.slot(fill.block, o),
                        generation=sh.generation,
                        request=sh.owner,
                        site=site,
                    )
                )
            sh.filled |= 1 << o
            sh.pending_publish &= ~(1 << o)
            self._touch(fill.pool, fill.block)
        return True

    def observe(self, snapshot: Snapshot) -> None:
        """Validate a snapshot against the shadow and adopt it.

        The work is incremental over what changed since the previous snapshot, but the reports are
        those of a full re-derivation (the reference the tests replay). A request re-derives its
        entries from the first index whose block or live range can differ; a listed block is
        re-judged when it is new, its ``BlockSnapshot`` changed, an allocator event touched it or
        its mappers changed; every other block re-emits its persistent reports. The computed claim
        is checked beyond the request's verified watermark, and the partition audit and in-flight
        mapping check run per snapshot.

        A pool named in ``delta_pools`` is listed as its change (new and changed blocks in
        ``blocks``, removed ones in ``gone``); its previous listing is kept and updated in place
        once the whole snapshot is validated. Reports come in the order of the full derivation,
        except that per-block reports are grouped by pool in registration order and by block id.
        """
        site = f"snapshot:{snapshot.seq}"
        if self.seq is not None and snapshot.seq <= self.seq:
            self._report(
                integration_error(
                    "snapshot_seq_not_increasing",
                    f"snapshot seq {snapshot.seq} after {self.seq}",
                    site=site,
                )
            )
        try:
            req_index = snapshot.request_index()
        except ValueError as exc:
            self._report(integration_error("malformed_snapshot", str(exc), site=site))
            return

        # A. Index listed blocks and find the new, changed and reclassified ones; no state changes
        #    yet, so a malformed snapshot leaves the ledger as it was.
        delta_pools = snapshot.delta_pools
        changes: dict[PoolId, dict[BlockId, BlockSnapshot]] = {pool: {} for pool in self.pools}
        judge: dict[PoolId, set[BlockId]] = {pool: set() for pool in self.pools}
        redo_blocks: list[tuple[PoolId, BlockId]] = []  # their mappers are re-derived
        invalid: list[Report] = []
        rejected: set[tuple[PoolId, BlockId]] = set()
        for bs in snapshot.blocks:
            pool, b = bs.pool, bs.block
            lp = changes.get(pool)
            if lp is None and (pool, b) not in rejected:
                rejected.add((pool, b))
                invalid.append(
                    unsupported(
                        "unknown_pool", f"snapshot names pool {pool!r}", pool=pool, site=site
                    )
                )
                continue
            if lp is None or b in lp or (rejected and (pool, b) in rejected):
                self._report(
                    integration_error(
                        "malformed_snapshot",
                        f"block {b} of pool {pool!r} listed twice",
                        site=site,
                    )
                )
                return
            prev_bs = self._listed[pool].get(b)
            if prev_bs is bs:  # the adapter kept the object: unchanged by construction
                lp[b] = bs
                continue
            if prev_bs is None:
                geometry = self.pools[pool]
                if not geometry.in_pool(b) or b in geometry.reserved_blocks:
                    rejected.add((pool, b))
                    invalid.append(
                        violation(
                            Policy.P1,
                            "state_for_unmanaged_block",
                            f"snapshot gives block {b} a {bs.state.value} state but it is outside the request-managed namespace",
                            pool=pool,
                            block=b,
                            site=site,
                        )
                    )
                    continue
                judge[pool].add(b)
                redo_blocks.append((pool, b))
            elif (
                prev_bs.state is not bs.state
                or prev_bs.reserved != bs.reserved
                or prev_bs.refs != bs.refs
            ):  # the fields of BlockSnapshot besides the key
                judge[pool].add(b)
            lp[b] = bs
        gone: dict[PoolId, set[BlockId]] = {}
        for pool, ids in (snapshot.gone or {}).items():
            prev_pool = self._listed.get(pool)
            if prev_pool is None:
                invalid.append(
                    unsupported(
                        "unknown_pool", f"snapshot names pool {pool!r}", pool=pool, site=site
                    )
                )
                continue
            if pool not in delta_pools:
                self._report(
                    integration_error(
                        "malformed_snapshot",
                        f"blocks gone from pool {pool!r}, which the snapshot lists in full",
                        pool=pool,
                        site=site,
                    )
                )
                return
            lp = changes[pool]
            ids = set(ids)
            rejected_before = self._rejected[pool]
            for b in ids:
                if b in lp or (b not in prev_pool and b not in rejected_before):
                    why = "listed" if b in lp else "not listed before"
                    self._report(
                        integration_error(
                            "malformed_snapshot",
                            f"block {b} of pool {pool!r} is gone but {why}",
                            pool=pool,
                            block=b,
                            site=site,
                        )
                    )
                    return
            gone[pool] = ids
        # The listing of this snapshot per pool, with the listed blocks that hold a reference
        # counted for the partition audit.
        listed: dict[PoolId, dict[BlockId, BlockSnapshot]] = {}
        for pool, cp in changes.items():
            rif = self.pools[pool].reserved_in_free
            if pool in delta_pools:
                lp = self._listed[pool]
                held = self._held[pool]
                rej = self._rejected[pool]
                for b in gone.get(pool, ()):
                    prev_bs = lp.pop(b, None)
                    if prev_bs is None:
                        rej.pop(b)  # a rejected entry the adapter dropped
                    elif not (rif and prev_bs.reserved):
                        held -= 1
                for rpool, b in rejected:
                    if rpool == pool:
                        rej[b] = next(
                            bs for bs in snapshot.blocks if bs.pool == pool and bs.block == b
                        )
                for b, bs in rej.items():
                    if (pool, b) not in rejected:  # rejected at an earlier delta: still listed
                        invalid.append(
                            violation(
                                Policy.P1,
                                "state_for_unmanaged_block",
                                f"snapshot gives block {b} a {bs.state.value} state but it is outside the request-managed namespace",
                                pool=pool,
                                block=b,
                                site=site,
                            )
                        )
                for b, bs in cp.items():
                    prev_bs = lp.get(b)
                    if prev_bs is bs:
                        continue
                    if prev_bs is not None and not (rif and prev_bs.reserved):
                        held -= 1
                    if not (rif and bs.reserved):
                        held += 1
                    lp[b] = bs
                self._held[pool] = held
            else:
                lp = cp
                self._held[pool] = sum(1 for bs in cp.values() if not (rif and bs.reserved))
            listed[pool] = lp

        # 0. Publications declared at the previous snapshot must have been written since.
        self._settle_pending_publications(site)

        # 1. Structural validity of the entries not accepted before.
        for r in invalid:
            self._report(r)

        # A request is re-derived when a block it maps entered or left the listing, or stayed
        # unlisted through an allocator event (its generation changed); listed blocks the
        # allocator events touched are judged.
        for pool, lp in listed.items():
            cached = self._block_reports[pool]
            if pool in delta_pools:
                left = gone.get(pool, ())
            else:
                left = self._listed[pool].keys() - lp.keys()
            for b in left:
                redo_blocks.append((pool, b))
                if cached:
                    cached.pop(b, None)
            jp = judge[pool]
            mappers = self._mappers[pool]
            for b in self._touched[pool]:
                if b in lp:
                    jp.add(b)
                elif b in mappers:
                    redo_blocks.append((pool, b))
        redo: set[tuple[RequestId, PoolId]] = set()
        for pool, b in redo_blocks:
            for rid in self._mappers[pool].get(b, ()):
                redo.add((rid, pool))
        for key in self.requests.keys() - req_index.keys():
            shadow = self.requests.pop(key)
            self._remove_entries(key[1], key[0], shadow.entries, 0, judge[key[1]])

        # 2. Derive owners and readers from the mappings; check mapping validity.
        for (rid, pool), rs in req_index.items():
            geometry = self.pools.get(pool)
            if geometry is None:
                self._report(
                    unsupported(
                        "unknown_pool",
                        f"request {rid!r} maps pool {pool!r}",
                        pool=pool,
                        request=rid,
                        site=site,
                    )
                )
                continue
            block_size = geometry.block_size
            needed = -(-rs.num_tokens // block_size) if rs.num_tokens else 0
            if len(rs.blocks) < needed:
                self._report(
                    integration_error(
                        "mapping_shorter_than_tokens",
                        f"request {rid!r} has {rs.num_tokens} tokens but {len(rs.blocks)} block entries",
                        pool=pool,
                        request=rid,
                        site=site,
                    )
                )
            if rs.num_computed > rs.num_tokens:
                self._report(
                    integration_error(
                        "computed_beyond_tokens",
                        f"request {rid!r} claims {rs.num_computed} computed of {rs.num_tokens} tokens",
                        pool=pool,
                        request=rid,
                        site=site,
                    )
                )
            key = (rid, pool)
            jp = judge[pool]
            shadow = self.requests.get(key)
            if shadow is None:
                shadow = self.requests[key] = RequestShadow(rs, snapshot.seq)
                k = 0
            else:
                k = 0 if key in redo else self._stable_prefix(shadow, rs)
                self._remove_entries(pool, rid, shadow.entries, k, jp)
                del shadow.entries[k:]
                if shadow.verified > k * block_size:
                    shadow.verified = k * block_size
            kept: list[tuple[int, Report]] = []
            for i, r in shadow.reports:
                if i >= k:
                    break
                kept.append((i, r))
                self._report(replace(r, site=site))
            shadow.reports = kept
            self._derive_entries(rid, pool, rs, geometry, listed[pool], k, shadow, jp, site)

        # 2b. Apply speculative retreats before the transitions: a framework may invalidate a
        #     private speculative tail and publish the block in one snapshot; clearing the old
        #     private writes lets the transition mark those offsets pending for the recompute.
        rolled_back: set[tuple[RequestId, PoolId]] = set()
        for key, rs in req_index.items():
            pool = key[1]
            geometry = self.pools.get(pool)
            shadow = self.requests.get(key)
            if geometry is None or shadow is None:
                continue
            prev = shadow.snapshot
            if prev is not rs and rs.num_computed < shadow.computed_high_water:
                self._invalidate(
                    pool,
                    geometry,
                    rs,
                    prev,
                    site,
                    self._touched[pool],
                    shadow.computed_high_water,
                )
                rolled_back.add(key)

        # 3. Permission configurations (P2) and transitions (P1) for the judged blocks; the
        #    others re-emit their persistent reports.
        for pool, lp in listed.items():
            jp = judge[pool]
            cached = self._block_reports[pool]
            if not jp and not cached:
                continue
            todo = jp & lp.keys()  # a judged block left unlisted has nothing to judge
            for b in sorted(todo | cached.keys()) if cached else sorted(todo):
                if b in todo:
                    reports = self._judge_block(pool, b, lp[b], req_index, site)
                    if reports:
                        cached[b] = reports
                    elif cached:
                        cached.pop(b, None)
                else:
                    for r in cached.get(b, ()):
                        self._report(replace(r, site=site))

        # 4. Blocks that left the non-free set: for a full listing, every omitted non-free block;
        #    for a delta listing, the touched blocks and those the change says left.
        for pool, shadows in self.blocks.items():
            lp = listed[pool]
            if pool in delta_pools:
                nf = self.nonfree[pool]
                left_nonfree = {
                    b
                    for b in self._touched[pool].union(gone.get(pool, ()))
                    if b in nf and b not in lp
                }
            else:
                left_nonfree = self.nonfree[pool] - lp.keys()
            if not left_nonfree:
                continue
            touched = self._touched[pool]
            for b in sorted(left_nonfree):
                sh = shadows[b]
                if sh.state is not BlockState.FREE:
                    if self.mediated and not sh.free_seen:
                        self._report(
                            integration_error(
                                "unmediated_free",
                                f"block became free without a mediated free (was {sh.state.value}, owner={sh.owner})",
                                pool=pool,
                                block=b,
                                generation=sh.generation,
                                request=sh.owner,
                                site=site,
                            )
                        )
                    inflight, streamed = self._referencing(pool, b)
                    if inflight:
                        self._report(
                            violation(
                                Policy.P1,
                                "free_while_in_flight",
                                f"block left the non-free set while {inflight} dispatch(es) reference it",
                                pool=pool,
                                block=b,
                                generation=sh.generation,
                                request=sh.owner,
                                site=site,
                            )
                        )
                    self._retire(pool, b, streamed)
                    sh.state = BlockState.FREE
                    sh.owner = None
                    sh.readers = frozenset()
                    sh.reserved = False
                    sh.filled = 0
                    sh.pending_publish = 0
                    self.nonfree[pool].discard(b)
                    self.nonfree_mask[pool][b] = 0
                    self._touch(pool, b)
                    touched.add(b)

        # 5. Partition audit against the allocator's free set, when given: per pool, or once
        #    over the pools that share an allocator (their listed blocks form one partition
        #    with the free set, given under any one of them).
        if snapshot.free is not None:
            for pools, free_set in self._audit_groups(snapshot.free):
                geometry = self.pools[pools[0]]
                # Set operations over the free set and the listed keys, once per pool.
                held: set[BlockId] = set()
                n_held = 0
                for pool in pools:
                    lp = listed[pool]
                    for b in sorted(free_set & lp.keys()):
                        self._report(
                            violation(
                                Policy.P1,
                                "block_free_and_allocated",
                                f"block {b} is in the allocator's free set and {lp[b].state.value}",
                                pool=pool,
                                block=b,
                                generation=self.blocks[pool][b].generation,
                                site=site,
                            )
                        )
                    if len(pools) > 1:
                        for b in sorted(held & lp.keys()):
                            self._report(
                                integration_error(
                                    "block_in_two_pools",
                                    f"block {b} is listed by two pools of one allocator, {pool!r} among them",
                                    pool=pool,
                                    block=b,
                                    site=site,
                                )
                            )
                        held |= lp.keys()
                    n_held += len(lp)
                reserved = geometry.reserved_blocks
                # Complete by count when the free set (less the reserved ids it may carry), the
                # listed blocks and the reserved blocks fill the pool; the ids are enumerated
                # only to name the unaccounted ones.
                n_free = len(free_set) - len(free_set & reserved)
                if n_free + n_held + len(reserved) == geometry.num_blocks:
                    continue
                held_keys = listed[pools[0]].keys() if len(pools) == 1 else held
                missing = set(range(geometry.num_blocks)) - reserved - free_set - held_keys
                if missing:
                    self._report(
                        integration_error(
                            "block_unaccounted",
                            f"{len(missing)} block(s) neither free nor in any state, e.g. {sorted(missing)[:5]}",
                            pool=pools[0],
                            site=site,
                        )
                    )

        elif snapshot.free_count is not None:
            # 5b. The partition audit by count (the free set not read this step): the blocks
            #     neither free nor reserved must be exactly the listed blocks holding a reference
            #     (summed over the pools of one allocator), per ``reserved_in_free``.
            for pools, free_count in self._audit_groups(snapshot.free_count):
                geometry = self.pools[pools[0]]
                held = sum(self._held[pool] for pool in pools)
                expected = geometry.num_blocks - len(geometry.reserved_blocks) - int(free_count)
                if held != expected:
                    self._report(
                        integration_error(
                            "block_count_mismatch",
                            f"the allocator's free count leaves {expected} block(s) neither free nor reserved but {held} listed block(s) hold a reference",
                            pool=pools[0],
                            site=site,
                        )
                    )

        # 6. Tables in use by open dispatches must not change, unless the framework ordered the
        #    rewrite after the epoch on its stream (the same stream, or a declared fence); a row
        #    released under an epoch on a stream is a release in flight (its blocks retire),
        #    not a mutation.
        for epoch in self.epochs.values():
            epool = epoch.descriptor.pool
            ordered = epoch.stream is not None and (
                epoch.stream == snapshot.stream or epoch.descriptor.dispatch_id in snapshot.after
            )
            for rid, indices, entries in epoch.consumed:
                rs = req_index.get((rid, epool))
                if rs is None and epoch.stream is not None:
                    self.counters["row_released_in_flight"] += 1
                    continue
                blocks = rs.blocks if rs is not None else ()
                n = len(blocks)
                if type(indices) is range and indices.start == 0 and indices.step == 1:
                    k = len(indices)
                    now = tuple(blocks[:k])
                    if k > n:
                        now += (None,) * (k - n)
                else:
                    now = tuple(blocks[i] if i < n else None for i in indices)
                if now == entries:
                    continue
                if ordered:
                    self.counters["mapping_rewritten_in_order"] += 1
                    continue
                for index, b, cur in zip(indices, entries, now, strict=True):
                    if cur == b:
                        continue
                    self._report(
                        violation(
                            Policy.P3,
                            "mapping_mutated_in_flight",
                            f"request {rid!r} entry {index} changed from {b} to {cur} while dispatch {epoch.descriptor.dispatch_id!r} is open",
                            pool=epool,
                            block=b,
                            request=rid,
                            site=site,
                        )
                    )

        # 7. Initialization claims (P4) against the core's record, then adopt the snapshots; the
        #    touched set is drained here, so blocks whose record shrank are re-checked.
        shrunk = self._touched
        self._touched = {pool: set() for pool in self.pools}
        for pool, blocks in shrunk.items():
            shadows = self.blocks[pool]
            for b in blocks:
                sh = shadows[b]
                sh.alloc_seen = False
                sh.free_seen = False
        for (rid, pool), rs in req_index.items():
            geometry = self.pools.get(pool)
            if geometry is None:
                continue
            key = (rid, pool)
            shadow = self.requests[(rid, pool)]
            prev = shadow.snapshot
            if (
                key not in rolled_back
                and prev is not rs
                and rs.num_computed < shadow.computed_high_water
            ):
                self._invalidate(
                    pool,
                    geometry,
                    rs,
                    prev,
                    site,
                    shrunk[pool],
                    shadow.computed_high_water,
                )
            self._check_computed_claim(pool, geometry, rs, site, shadow, shrunk[pool])
            shadow.snapshot = rs
            shadow.seq = snapshot.seq
            shadow.computed_high_water = rs.num_computed
        self._listed = listed
        self.seq = snapshot.seq

    def _audit_groups(self, given: Mapping[PoolId, Any]) -> list[tuple[list[PoolId], Any]]:
        """The partition audit's units: a pool of its own, or every registered pool of a shared
        allocator together (the value given under any one of them)."""
        out: list[tuple[list[PoolId], Any]] = []
        seen: set[str] = set()
        for pool, value in given.items():
            geometry = self.pools.get(pool)
            if geometry is None:
                continue
            alloc = geometry.allocator
            if alloc is None:
                out.append(([pool], value))
                continue
            if alloc in seen:
                continue
            seen.add(alloc)
            pools = [p for p, g in self.pools.items() if g.allocator == alloc]
            out.append((pools, value))
        return out

    def _forget_incremental_state(self) -> None:
        """Drop what `observe` keeps between snapshots, so the next snapshot is derived in full;
        the shadow itself is kept. A reference path, not needed in operation."""
        for pool in self.pools:
            self._listed[pool] = {}
            self._rejected[pool] = {}
            self._mappers[pool] = {}
            self._block_reports[pool] = {}
        for shadow in self.requests.values():
            shadow.entries = []
            shadow.full_count = 0
            shadow.full_seen = {}
            shadow.reports = []
            shadow.verified = 0

    @staticmethod
    def _stable_prefix(shadow: RequestShadow, rs: RequestSnapshot) -> int:
        """The number of leading entries whose block and live range match the previous derivation
        (the previous full blocks when the mapping grew without moving its window), else 0."""
        prev = shadow.snapshot
        if prev.first_token != rs.first_token or rs.num_tokens < prev.num_tokens:
            return 0
        if (
            rs.num_tokens == prev.num_tokens
            and rs.blocks == prev.blocks
            and len(shadow.entries) == len(prev.blocks)
        ):
            return len(rs.blocks)  # the same mapping, derived before: every entry is kept
        k = shadow.full_count
        if k > len(rs.blocks) or rs.blocks[:k] != prev.blocks[:k]:
            return 0
        return k

    def _remove_entries(self, pool, rid, entries, k, jp: set[BlockId]) -> None:
        """Withdraw the contribution of entries ``k`` onwards from the mappers; the blocks they
        named are judged again."""
        mappers = self._mappers[pool]
        for i in range(k, len(entries)):
            b = entries[i][0]
            if b is None:
                continue
            holders = mappers[b]
            n = holders[rid] - 1
            if n:
                holders[rid] = n
            else:
                del holders[rid]
                if not holders:
                    del mappers[b]
            jp.add(b)

    def _derive_entries(self, rid, pool, rs, geometry, lp, k, shadow, jp, site) -> None:
        """Entries ``k`` onwards of a request's mapping: the mapping checks (P3, P2), the
        contribution to the mappers and the duplicate-slot masks. Entries below ``k`` are kept
        as derived, with the masks of the full blocks in ``full_seen``."""
        bs = geometry.block_size
        reserved = geometry.reserved_blocks
        num_blocks = geometry.num_blocks
        ft, nt = rs.first_token, rs.num_tokens
        blocks = rs.blocks
        entries = shadow.entries
        reports = shadow.reports
        mappers = self._mappers[pool]
        seen = dict(shadow.full_seen) if k else {}
        full_count = min(nt // bs, len(blocks))
        for index in range(k, len(blocks)):
            if index == full_count:
                shadow.full_seen = dict(seen)
            b = blocks[index]
            start = index * bs
            lo = start if start > ft else ft
            hi = start + bs
            if hi > nt:
                hi = nt
            n = hi - lo  # the live positions of this entry are [lo, hi)
            if b is None:
                # A hole: a sliding window's retired prefix (dead by first_token), or in a
                # rewritable pool a retired state column, which holds no live position.
                if n > 0 and not geometry.rewritable:
                    r = violation(
                        Policy.P3,
                        "live_position_without_block",
                        f"request {rid!r} positions {lo}..{hi - 1} have no block",
                        pool=pool,
                        request=rid,
                        site=site,
                    )
                    self._report(r)
                    reports.append((index, r))
                entries.append((None, 0))
                continue
            if not 0 <= b < num_blocks:
                r = violation(
                    Policy.P3,
                    "mapping_out_of_pool",
                    f"request {rid!r} maps block {b} outside the pool",
                    pool=pool,
                    block=b,
                    request=rid,
                    site=site,
                )
                self._report(r)
                reports.append((index, r))
                entries.append((None, 0))
                continue
            if b in reserved:
                if n > 0:
                    r = violation(
                        Policy.P3,
                        "mapping_reserved_block",
                        f"request {rid!r} maps live positions to reserved block {b}",
                        pool=pool,
                        block=b,
                        request=rid,
                        site=site,
                    )
                    self._report(r)
                    reports.append((index, r))
                entries.append((None, 0))
                continue
            holders = mappers.get(b)
            if holders is None:
                mappers[b] = {rid: 1}
            else:
                holders[rid] = holders.get(rid, 0) + 1
            jp.add(b)
            if b not in lp:
                r = violation(
                    Policy.P2,
                    "mapping_to_free_block",
                    f"request {rid!r} maps block {b}, which the snapshot lists as free",
                    pool=pool,
                    block=b,
                    generation=self.blocks[pool][b].generation,
                    request=rid,
                    site=site,
                )
                self._report(r)
                reports.append((index, r))
                entries.append((b, 0))
                continue
            mask = 0
            if n > 0:
                # The live positions of one block index are contiguous offsets of one block, so
                # two positions share a slot only when the block id repeats.
                mask = ((1 << n) - 1) << (lo % bs)
                before = seen.get(b, 0)
                dup = before & mask
                if dup:
                    for p in range(lo, hi):
                        o = p % bs
                        if dup >> o & 1:
                            r = violation(
                                Policy.P3,
                                "duplicate_slot_in_request",
                                f"request {rid!r} maps two live positions to slot {geometry.slot(b, o)}",
                                pool=pool,
                                block=b,
                                slot=geometry.slot(b, o),
                                request=rid,
                                site=site,
                            )
                            self._report(r)
                            reports.append((index, r))
                seen[b] = before | mask
            entries.append((b, mask))
        if full_count >= len(blocks):
            shadow.full_seen = seen
        shadow.full_count = full_count

    def _judge_block(self, pool, b, bs: BlockSnapshot, req_index, site) -> list[Report]:
        """The permission configuration (P2) and the transition (P1) of one listed block.
        Returns the configuration reports, which persist while the block is unchanged."""
        geometry = self.pools[pool]
        sh = self.blocks[pool][b]
        holders = self._mappers[pool].get(b) or ()
        reports: list[Report] = []
        if bs.state is BlockState.PRIVATE:
            own, rdr = holders, ()
            if len(own) > 1:
                reports.append(
                    violation(
                        Policy.P2,
                        "private_multiple_owners",
                        f"private block mapped by {sorted(own)}",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
            elif not own:
                reports.append(
                    violation(
                        Policy.P1,
                        "private_unmapped",
                        "private block mapped by no request",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
        else:
            own, rdr = (), holders
            if not rdr and not bs.reserved:
                reports.append(
                    violation(
                        Policy.P1,
                        "published_unreferenced",
                        "published block with no reader and no reservation",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        site=site,
                    )
                )
        if bs.refs is not None and bs.refs != len(own) + len(rdr):
            reports.append(
                violation(
                    Policy.P2,
                    "reference_count_mismatch",
                    f"framework counts {bs.refs} holder(s) but {len(own) + len(rdr)} request(s) map the block",
                    pool=pool,
                    block=b,
                    generation=sh.generation,
                    site=site,
                )
            )
        for r in reports:
            self._report(r)
        new_owner = next(iter(own)) if len(own) == 1 else None
        self._transition(pool, b, sh, bs, new_owner, frozenset(rdr), req_index, geometry, site)
        return reports

    def _transition(
        self,
        pool,
        b,
        sh: BlockShadow,
        bs: BlockSnapshot,
        new_owner,
        new_readers,
        req_index,
        geometry,
        site,
    ) -> None:
        old = sh.state
        new = bs.state
        pending_before = sh.pending_publish
        if new is BlockState.PUBLISHED and old is not BlockState.PUBLISHED:
            self.published_since[pool].append(b)
        if old is BlockState.FREE:
            if new is BlockState.PUBLISHED:
                sh.pending_publish = self._unfilled_live_offsets(
                    pool, b, sh, new_owner, new_readers, req_index, geometry
                )
            if self.mediated and not sh.alloc_seen:
                self._report(
                    integration_error(
                        "unmediated_allocation",
                        f"block became {new.value} without a mediated allocation",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=new_owner,
                        site=site,
                    )
                )
            elif not self.mediated:
                sh.generation += 1
                sh.filled = 0
        elif old is BlockState.PRIVATE and new is BlockState.PRIVATE:
            if (
                sh.owner is not None
                and new_owner is not None
                and new_owner != sh.owner
                and not sh.alloc_seen
            ):
                self._report(
                    violation(
                        Policy.P1,
                        "ownership_transfer_without_free",
                        f"private block passed from {sh.owner!r} to {new_owner!r} without being freed",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=new_owner,
                        site=site,
                    )
                )
        elif old is BlockState.PRIVATE and new is BlockState.PUBLISHED:
            sh.pending_publish = self._unfilled_live_offsets(
                pool, b, sh, new_owner, new_readers, req_index, geometry
            )
        elif old is BlockState.PUBLISHED and new is BlockState.PRIVATE:
            if not sh.alloc_seen:
                self._report(
                    violation(
                        Policy.P1,
                        "published_to_private",
                        f"published block became private to {new_owner!r} without being freed",
                        pool=pool,
                        block=b,
                        generation=sh.generation,
                        request=new_owner,
                        site=site,
                    )
                )
        elif old is BlockState.PUBLISHED and new is BlockState.PUBLISHED and sh.alloc_seen:
            # Evicted and reallocated, then published again within one interval: a new life.
            pass
        changed = (
            sh.state is not new or sh.owner != new_owner or pending_before != sh.pending_publish
        )
        sh.state = new
        sh.owner = new_owner
        sh.readers = new_readers
        sh.reserved = bs.reserved
        self.nonfree[pool].add(b)
        self.nonfree_mask[pool][b] = 1
        if sh.pending_publish:
            self._pending[pool].add(b)
        if changed:  # the mirror words depend on the state, the owner and the pending mask
            self._touch(pool, b)

    def _unfilled_live_offsets(
        self, pool, b, sh: BlockShadow, owner, readers, req_index, geometry
    ) -> int:
        """Offsets of live positions, over every request that maps the block in the new snapshot,
        that the core has not seen written. These are the publication's pending slots."""
        mask = 0
        for rid in set(readers) | ({owner} if owner else set()):
            rs = req_index.get((rid, pool))
            if rs is None:
                continue
            for index, blk in enumerate(rs.blocks):
                if blk != b:
                    continue
                for p in rs.positions_in_block(geometry, index):
                    o = p % geometry.block_size
                    if not sh.is_filled(o):
                        mask |= 1 << o
        return mask

    def _settle_pending_publications(self, site) -> None:
        """A publication declared at the previous snapshot must have been written by now."""
        for pool, pend in self._pending.items():
            if not pend:
                continue
            shadows = self.blocks[pool]
            geometry = self.pools[pool]
            touched = self._touched[pool]
            for b in sorted(pend):
                sh = shadows[b]
                if not sh.pending_publish:
                    continue
                still = sh.pending_publish & ~sh.filled
                sh.pending_publish = 0
                self._touch(pool, b)
                if still:
                    touched.add(b)  # the initialization record shrank: claims are re-checked
                    if sh.state is BlockState.PUBLISHED:
                        o = (still & -still).bit_length() - 1
                        self._report(
                            violation(
                                Policy.P4,
                                "publish_uninitialized",
                                f"block was published with slot offset {o} never written by the dispatches that followed",
                                pool=pool,
                                block=b,
                                slot=geometry.slot(b, o),
                                generation=sh.generation,
                                site=site,
                            )
                        )
            pend.clear()

    def _invalidate(
        self,
        pool,
        geometry,
        rs: RequestSnapshot,
        prev: RequestSnapshot,
        site,
        shrunk,
        computed_high_water: int,
    ) -> None:
        """num_computed shrank (rollback): the request's slots beyond the new count are
        invalidated, which is legal only in its own private blocks. The blocks join ``shrunk``
        (this snapshot's claim checks) and the touched set (the next snapshot's); the listeners
        are told the invalidated (block, offset) pairs (``on_invalidate``)."""
        invalidated: list[tuple[BlockId, int]] = []
        for p in range(rs.num_computed, min(int(computed_high_water), prev.num_tokens)):
            index = p // geometry.block_size
            b = prev.blocks[index] if index < len(prev.blocks) else None
            if b is None or not geometry.in_pool(b):
                continue
            sh = self.blocks[pool][b]
            o = p % geometry.block_size
            if sh.state is BlockState.PUBLISHED:
                self._report(
                    violation(
                        Policy.P4,
                        "invalidate_published",
                        f"request {rs.request!r} invalidated position {p} in a published block",
                        pool=pool,
                        block=b,
                        slot=geometry.slot(b, o),
                        generation=sh.generation,
                        request=rs.request,
                        site=site,
                    )
                )
                continue
            sh.filled &= ~(1 << o)
            self._touch(pool, b)
            shrunk.add(b)
            self._touched[pool].add(b)
            invalidated.append((b, o))
        if invalidated:
            for listener in self.listeners:
                listener.on_invalidate(pool, invalidated)

    def rollback(
        self,
        pool: PoolId,
        request: RequestId,
        num_computed: int,
        *,
        site: str = "rollback",
    ) -> None:
        """Reconcile an execution-time speculative rollback before the next dispatch.

        An engine may prepare the next snapshot optimistically while its execution descriptor
        carries the corrected GPU-side computed frontier; reporting that retreat here invalidates
        rejected private slots before the next cache write. Invalidating published content is
        still a P4 violation.
        """
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        shadow = self.requests.get((request, pool))
        if shadow is None:
            self._report(
                integration_error(
                    "rollback_unknown_request",
                    f"speculative rollback names unknown request {request!r}",
                    pool=pool,
                    request=request,
                    site=site,
                )
            )
            return
        frontier = int(num_computed)
        if frontier < 0 or frontier > shadow.snapshot.num_tokens:
            self._report(
                integration_error(
                    "malformed_rollback",
                    f"computed frontier {frontier} is outside request {request!r}'s "
                    f"{shadow.snapshot.num_tokens} live tokens",
                    pool=pool,
                    request=request,
                    site=site,
                )
            )
            return
        if frontier >= shadow.computed_high_water:
            return
        corrected = replace(shadow.snapshot, num_computed=frontier)
        self._invalidate(
            pool,
            geometry,
            corrected,
            shadow.snapshot,
            site,
            self._touched[pool],
            shadow.computed_high_water,
        )
        shadow.snapshot = corrected
        shadow.computed_high_water = frontier
        shadow.verified = min(shadow.verified, frontier)
        self.counters["speculative_rollbacks"] += 1

    def invalidate_positions(
        self,
        pool: PoolId,
        request: RequestId,
        positions: Iterable[int],
        *,
        site: str = "invalidate",
    ) -> None:
        """Invalidate an explicit private speculative workspace before the framework rewrites it.

        Unlike :meth:`rollback` it does not move the dense computed frontier; it is for layouts
        such as an EAGLE tree whose branch slots have holes in request position order. Published
        slots remain immutable and produce the same P4 report as a dense rollback.
        """
        geometry = self._geometry(pool, site)
        if geometry is None:
            return
        shadow = self.requests.get((request, pool))
        if shadow is None:
            self._report(
                integration_error(
                    "invalidation_unknown_request",
                    f"speculative invalidation names unknown request {request!r}",
                    pool=pool,
                    request=request,
                    site=site,
                )
            )
            return
        rs = shadow.snapshot
        invalidated = 0
        pairs: list[tuple[BlockId, int]] = []
        first_position: int | None = None
        for value in positions:
            p = int(value)
            if p < 0 or p >= rs.num_tokens:
                self._report(
                    integration_error(
                        "malformed_invalidation",
                        f"position {p} is outside request {request!r}'s {rs.num_tokens} live tokens",
                        pool=pool,
                        request=request,
                        site=site,
                    )
                )
                continue
            s = rs.slot_at(geometry, p)
            if s is None:
                self._report(
                    integration_error(
                        "invalidation_unmapped_position",
                        f"position {p} of request {request!r} has no mapped slot",
                        pool=pool,
                        request=request,
                        site=site,
                    )
                )
                continue
            b = geometry.block_of(s)
            if b in geometry.reserved_blocks or not geometry.in_pool(b):
                continue
            block = self.blocks[pool][b]
            offset = s - b * geometry.block_size
            mask = 1 << offset
            if block.state is BlockState.PUBLISHED:
                self._report(
                    violation(
                        Policy.P4,
                        "invalidate_published",
                        f"request {request!r} invalidated published position {p}",
                        pool=pool,
                        block=b,
                        slot=s,
                        generation=block.generation,
                        request=request,
                        site=site,
                    )
                )
                continue
            if block.filled & mask:
                block.filled &= ~mask
                self._touch(pool, b)
                self._touched[pool].add(b)
                invalidated += 1
                pairs.append((b, mask.bit_length() - 1))
                first_position = p if first_position is None else min(first_position, p)
        if first_position is not None:
            shadow.verified = min(shadow.verified, first_position)
        self.counters["speculative_sparse_invalidations"] += invalidated
        if pairs:
            for listener in self.listeners:
                listener.on_invalidate(pool, pairs)

    def authorize_volatile_rewrite(
        self,
        pool: PoolId,
        request: RequestId,
        position: int,
        *,
        site: str = "volatile-rewrite",
    ) -> bool:
        """Authorize one rewrite of a framework-declared volatile published slot.

        Some speculative caches publish a stable prefix while declaring one terminal slot volatile
        and re-prefilling it for the sole adopting request. Narrow, not a general relaxation: the
        pool must be rewritable, the slot already initialized, and exactly one request must map the
        block exactly once. The one-shot permission reuses ``pending_publish`` (which dispatch
        consumes and ``_pending`` clears at the next snapshot if unused).
        """
        geometry = self._geometry(pool, site)
        if geometry is None or not geometry.rewritable:
            return False
        request_shadow = self.requests.get((request, pool))
        if request_shadow is None:
            return False
        rs = request_shadow.snapshot
        p = int(position)
        if p < 0 or p >= rs.num_tokens:
            return False
        slot = rs.slot_at(geometry, p)
        if slot is None:
            return False
        block_id = geometry.block_of(slot)
        if block_id in geometry.reserved_blocks or not geometry.in_pool(block_id):
            return False
        block = self.blocks[pool][block_id]
        offset = slot - block_id * geometry.block_size
        bit = 1 << offset
        if (
            block.state is not BlockState.PUBLISHED
            or block.readers != frozenset({request})
            or self._mappers[pool].get(block_id) != {request: 1}
            or not block.filled & bit
        ):
            return False
        if not block.pending_publish & bit:
            block.pending_publish |= bit
            self._pending[pool].add(block_id)
            self._touch(pool, block_id)
        return True

    def authorize_scratch_rewrite(
        self,
        pool: PoolId,
        request: RequestId,
        positions: Iterable[int],
        *,
        site: str = "scratch-rewrite",
    ) -> int:
        """Re-open, for the next write, slots that hold a request's own provisional content.

        A rewritable draft pool lets a drafter rewrite private slots it wrote beyond the committed
        frontier; when the framework publishes the block in between, those slots are not committed
        content, so the adapter declares them here before the write that replaces them. Each gets
        the one-shot ``pending_publish`` permission. Only a rewritable pool, the request's current
        mapping, published blocks and already-initialized slots. Returns the number re-opened.
        """
        geometry = self._geometry(pool, site)
        if geometry is None or not geometry.rewritable:
            return 0
        request_shadow = self.requests.get((request, pool))
        if request_shadow is None:
            return 0
        rs = request_shadow.snapshot
        bs = geometry.block_size
        shadows = self.blocks[pool]
        reopened = 0
        for position in positions:
            p = int(position)
            if p < 0 or p >= rs.num_tokens:
                continue
            slot = rs.slot_at(geometry, p)
            if slot is None:
                continue
            block_id = geometry.block_of(slot)
            if block_id in geometry.reserved_blocks or not geometry.in_pool(block_id):
                continue
            block = shadows[block_id]
            bit = 1 << (slot - block_id * bs)
            if block.state is not BlockState.PUBLISHED or not block.filled & bit:
                continue
            if not block.pending_publish & bit:
                block.pending_publish |= bit
                self._pending[pool].add(block_id)
                self._touch(pool, block_id)
            reopened += 1
        if reopened:
            self.counters["scratch_rewrites_authorized"] += reopened
        return reopened

    def _check_computed_claim(
        self, pool, geometry, rs: RequestSnapshot, site, shadow: RequestShadow, shrunk
    ) -> None:
        """Positions the framework calls computed must be initialized in the core's record.
        Positions below ``shadow.verified`` passed at an earlier snapshot and are checked again
        only in blocks of ``shrunk``; the first missing position is reported, as a check of the
        whole claim would report it, and becomes the watermark."""
        bs = geometry.block_size
        stop = min(rs.num_computed, rs.num_tokens)
        v = shadow.verified
        if v > stop:
            v = stop
        first = rs.first_token
        blocks = rs.blocks
        shadows = self.blocks[pool]
        reserved = geometry.reserved_blocks

        def missing_in(index: int, p: int, p_hi: int) -> int:
            """The first position of [p, p_hi) in block ``index`` whose slot was never written,
            or -1 (the computed positions of a block are one mask)."""
            b = blocks[index] if index < len(blocks) else None
            if b is None or not geometry.in_pool(b) or b in reserved:
                return -1
            sh = shadows[b]
            if sh.state is BlockState.FREE:
                return -1  # already reported as mapping_to_free_block
            mask = ((1 << (p_hi - p)) - 1) << (p % bs)
            missing = mask & ~(sh.filled | sh.pending_publish)
            if not missing:
                return -1
            return index * bs + (missing & -missing).bit_length() - 1

        def report(p: int) -> None:
            b = blocks[p // bs]
            self._report(
                violation(
                    Policy.P4,
                    "computed_beyond_filled",
                    f"request {rs.request!r} counts position {p} as computed but its slot was never written",
                    pool=pool,
                    block=b,
                    slot=geometry.slot(b, p % bs),
                    generation=shadows[b].generation,
                    request=rs.request,
                    site=site,
                )
            )
            shadow.verified = p

        # Below the watermark only the blocks whose initialization record shrank need a look,
        # found by one set intersection with the mapping's prefix.
        if shrunk and v > first:
            k = -(-v // bs)
            hits = shrunk.intersection(blocks[:k])
            if hits:
                for index, b in enumerate(blocks[:k]):
                    if b not in hits:
                        continue
                    lo_p = max(index * bs, first)
                    hi_p = min((index + 1) * bs, stop)
                    if lo_p < hi_p:
                        p = missing_in(index, lo_p, hi_p)
                        if p >= 0:
                            report(p)
                            return
        p = max(first, v)
        while p < stop:
            p_hi = min(stop, (p // bs + 1) * bs)
            miss = missing_in(p // bs, p, p_hi)
            if miss >= 0:
                report(miss)
                return
            p = p_hi
        shadow.verified = stop

    def dispatch(self, d: DispatchDescriptor) -> None:
        """Validate the exact mapping a dispatch consumes and open its epoch."""
        site = f"dispatch:{d.dispatch_id}"
        if self.seq is None:
            self._report(
                integration_error(
                    "dispatch_before_snapshot", "no snapshot has been observed", site=site
                )
            )
            return
        if d.dispatch_id in self.epochs:
            self._report(
                integration_error(
                    "duplicate_dispatch_id",
                    f"dispatch {d.dispatch_id!r} is already open",
                    site=site,
                )
            )
            return
        geometry = self._geometry(d.pool, site)
        if geometry is None:
            return
        if d.snapshot_seq is not None and d.snapshot_seq != self.seq:
            self._report(
                integration_error(
                    "dispatch_snapshot_mismatch",
                    f"descriptor prepared from snapshot {d.snapshot_seq}, the ledger holds snapshot {self.seq}",
                    pool=d.pool,
                    site=site,
                )
            )
        bs = geometry.block_size

        def managed(b):
            """The managed block an entry can reach: None for holes, reserved and out-of-pool ids."""
            if b is None or not geometry.in_pool(b) or b in geometry.reserved_blocks:
                return None
            return b

        # writes_first: every write precedes every read, so a read is satisfied by any row's
        # write; collect them before the rows are checked.
        pre_written: dict[int, int] | None = None  # block -> offsets every row writes
        if d.writes_first:
            pre_written = {}
            for row in d.rows:
                if row.padding or row.request is None:
                    continue
                shadow = self.requests.get((row.request, d.pool))
                if shadow is None:
                    continue
                for p in row.writes:
                    s = shadow.snapshot.slot_at(geometry, p)
                    if s is not None:
                        b = geometry.block_of(s)
                        if b not in geometry.reserved_blocks:
                            pre_written[b] = pre_written.get(b, 0) | (1 << (s - b * bs))
        touched: set[BlockId] = set()
        consumed: list[tuple[RequestId, Sequence[int], tuple[BlockId | None, ...]]] = []
        written_all: dict[int, int] = {}  # block -> offsets written by the rows so far
        slot_pairs: dict[int, list[tuple[int, BlockId]]] | None = None
        if isinstance(d.mapping, SlotMap):
            # The (block index, block) pairs of a slot map grouped by row once, not per row.
            slot_pairs = {}
            m = d.mapping
            for s, r, p in zip(m.slots, m.rows, m.positions, strict=True):
                if 0 <= s < geometry.num_slots:
                    slot_pairs.setdefault(r, []).append((p // bs, geometry.block_of(s)))
        for row_index, row in enumerate(d.rows):
            entries = self._row_entries(d, row_index, row, geometry, site)
            if entries is None:
                continue
            if row.padding:
                self._check_padding_row(d, row_index, entries, geometry, site)
                continue
            shadow = self.requests.get((row.request, d.pool)) if row.request is not None else None
            if shadow is None:
                self._report(
                    violation(
                        Policy.P2,
                        "row_for_unknown_request",
                        f"row {row_index} belongs to request {row.request!r}, which the current snapshot does not know",
                        pool=d.pool,
                        request=row.request,
                        site=site,
                    )
                )
                continue
            rs = shadow.snapshot
            m = d.mapping
            start = m.first_entry(row_index) if isinstance(m, DenseBlockTable) else 0
            token_start = m.first_position(row_index) if isinstance(m, TokenTable) else 0
            matched = self._check_row_mapping(d, row_index, row, rs, entries, geometry, site, start)
            # (block index, block) pairs the kernel reaches through this row, per representation.
            if slot_pairs is not None:
                pairs = slot_pairs.get(row_index, [])
            elif isinstance(m, PositionedTokenTable):
                logical = list(m.positions[row_index])
                pairs = [
                    (p // bs, geometry.block_of(s) if 0 <= s < geometry.num_slots else None)
                    for p, s in zip(logical, entries, strict=False)
                ]
            elif isinstance(m, TokenTable) and not matched:
                pairs = [
                    (p // bs, geometry.block_of(s) if 0 <= s < geometry.num_slots else None)
                    for p, s in enumerate(entries, token_start)
                ]
            else:
                # A dense table or page list, or a per-token table that equals the mapping's
                # expansion (the fast path above): the consumed blocks by index, no
                # per-position pairs.
                pairs = None
            if pairs is None:
                if isinstance(m, TokenTable) and bs > 1:
                    # The table equals the mapping's slots for its positions: the blocks the
                    # positions fall in, from the mapping (per block, not per position).
                    lo = token_start // bs
                    hi = -(-(token_start + len(entries)) // bs)
                    indices = range(lo, hi)
                    blocks_of_row = rs.blocks[lo:hi]
                elif isinstance(m, TokenTable):
                    indices = range(token_start, token_start + len(entries))
                    blocks_of_row = entries
                else:
                    indices = range(start, len(entries))
                    blocks_of_row = entries[start:] if start else entries
            else:
                indices = [i for i, _ in pairs]
                blocks_of_row = [b for _, b in pairs]
            if (
                blocks_of_row
                and None not in blocks_of_row
                and min(blocks_of_row) >= 0
                and max(blocks_of_row) < geometry.num_blocks
                and not geometry.reserved_blocks.intersection(blocks_of_row)
            ):
                # Every entry is a managed block: no per-entry classification needed.
                consumed.append((row.request, indices, tuple(blocks_of_row)))
                touched.update(blocks_of_row)
            else:
                managed_blocks = tuple(map(managed, blocks_of_row))
                consumed.append((row.request, indices, managed_blocks))
                touched.update(b for b in managed_blocks if b is not None)
            for b, m in self._check_row_access(
                d, row_index, row, rs, geometry, site, pre_written, written_all, shadow
            ).items():
                written_all[b] = written_all.get(b, 0) | m
            if not geometry.rewritable and d.advances_computed and row.writes:
                shadow.computed_high_water = max(shadow.computed_high_water, row.writes.stop)
        # The dispatch's writes initialize their slots once every row is checked (a write
        # satisfies no read of another row unless writes_first says so).
        for b, m in written_all.items():
            sh = self.blocks[d.pool][b]
            sh.filled |= m
            sh.pending_publish &= ~m
            self._touch(d.pool, b)
        if self.retiring:
            self._judge_retiring(d, touched, site)
        self.epochs[d.dispatch_id] = Epoch(
            d, self.seq, frozenset(zip(repeat(d.pool), touched)), tuple(consumed), d.stream
        )

    def _judge_retiring(
        self,
        d: DispatchDescriptor | ExternalRead | ExternalWrite,
        touched: set[BlockId],
        site: str,
    ) -> None:
        """The stream-order rule's judgment: this dispatch touches a block released while epochs
        on a stream were open. The access is ordered after an epoch on the same stream, and after
        one this dispatch declares in ``after``; otherwise it is the P1 violation
        ``reuse_before_epoch_retired``, reported at the first access instead of at the release."""
        keys = [k for k in self.retiring if k[0] == d.pool and k[1] in touched]
        for key in keys:
            entry = self.retiring.pop(key)
            for did, stream in entry.items():
                held = self._retiring_epochs.get(did)
                if held is not None:
                    held.discard(key)
                    if not held:
                        del self._retiring_epochs[did]
                if d.stream is not None and stream == d.stream:
                    self.counters["retired_in_order"] += 1
                elif did in d.after:
                    self.counters["retired_by_after"] += 1
                else:
                    self._report(
                        violation(
                            Policy.P1,
                            "reuse_before_epoch_retired",
                            f"dispatch {d.dispatch_id!r} on stream {d.stream} touches block {key[1]}, released while dispatch {did!r} on stream {stream} is open, with no order declared between them",
                            pool=d.pool,
                            block=key[1],
                            generation=self.blocks[d.pool][key[1]].generation,
                            site=site,
                        )
                    )

    def complete(self, dispatch_id: str) -> None:
        epoch = self.epochs.pop(dispatch_id, None)
        if epoch is None:
            self._report(
                integration_error(
                    "unknown_dispatch",
                    f"completion of unknown dispatch {dispatch_id!r}",
                    site=f"dispatch:{dispatch_id}",
                )
            )
            return
        held = self._retiring_epochs.pop(dispatch_id, None)
        if held:
            for key in held:
                entry = self.retiring.get(key)
                if entry is not None:
                    entry.pop(dispatch_id, None)
                    if not entry:
                        del self.retiring[key]

    def _row_entries(self, d, row_index, row, geometry, site) -> list[BlockId | None] | None:
        """The block entries (by block index) the kernel consumes for this row, from the mapping
        representation. Slot maps are per token and handled in _check_row_mapping."""
        m = d.mapping
        if isinstance(m, DenseBlockTable):
            if row_index >= len(m.table) or row_index >= len(m.valid):
                self._report(
                    integration_error(
                        "row_without_table_entry",
                        f"row {row_index} has no table row",
                        pool=d.pool,
                        site=site,
                    )
                )
                return None
            return list(m.entries(row_index))
        if isinstance(m, CSRPageList):
            if row_index + 1 >= len(m.indptr):
                self._report(
                    integration_error(
                        "row_without_table_entry",
                        f"row {row_index} has no indptr entry",
                        pool=d.pool,
                        site=site,
                    )
                )
                return None
            return list(m.entries(row_index))
        if isinstance(m, TokenTable):
            if row_index >= len(m.table) or row_index >= len(m.seq_len):
                self._report(
                    integration_error(
                        "row_without_table_entry",
                        f"row {row_index} has no table row",
                        pool=d.pool,
                        site=site,
                    )
                )
                return None
            return list(m.table[row_index][: m.seq_len[row_index]])
        if isinstance(m, PositionedTokenTable):
            if row_index >= len(m.table) or row_index >= len(m.positions):
                self._report(
                    integration_error(
                        "row_without_table_entry",
                        f"row {row_index} has no positioned table row",
                        pool=d.pool,
                        site=site,
                    )
                )
                return None
            return list(m.table[row_index])
        if isinstance(m, SlotMap):
            return []
        self._report(
            unsupported(
                "mapping_representation",
                f"unsupported mapping representation {type(m).__name__}",
                pool=d.pool,
                site=site,
            )
        )
        return None

    def _check_padding_row(self, d, row_index, entries, geometry, site) -> None:
        """A padding row may reach reserved blocks only (vLLM's null block, SGLang's slot 0)."""
        m = d.mapping
        if isinstance(m, SlotMap):
            named = [
                (geometry.block_of(s) if 0 <= s < geometry.num_slots else s, s)
                for s, r in zip(m.slots, m.rows, strict=True)
                if r == row_index and s != PADDING_SLOT
            ]
        elif isinstance(m, (TokenTable, PositionedTokenTable)):
            named = [
                (geometry.block_of(s) if 0 <= s < geometry.num_slots else s, s) for s in entries
            ]
        else:
            named = [(b, None) for b in entries]
        for b, slot in named:
            if b is None or b in geometry.reserved_blocks:
                continue
            sh = self.blocks[d.pool][b] if geometry.in_pool(b) else None
            state = sh.state.value if sh else "out of pool"
            self._report(
                violation(
                    Policy.P2,
                    "padding_row_names_managed_block",
                    f"padding row {row_index} names block {b} ({state}); padding must touch no request-managed slot",
                    pool=d.pool,
                    block=b,
                    slot=slot,
                    generation=sh.generation if sh else None,
                    request=sh.owner if sh else None,
                    site=site,
                )
            )

    def _check_dead_entry(self, d, row_index, j, b, rs, geometry, site, what="entry") -> None:
        """A table entry for positions the request does not hold (a window hole, or beyond its
        tokens): legal for a reserved block, out of range for an id outside the pool, stale for a
        managed block the kernel may still touch."""
        if b is None or b in geometry.reserved_blocks:
            return
        if not geometry.in_pool(b):
            self._report(
                violation(
                    Policy.P3,
                    "mapping_out_of_pool",
                    f"row {row_index} {what} {j} names {b}, outside the pool of {geometry.num_blocks} blocks",
                    pool=d.pool,
                    block=b,
                    request=rs.request,
                    site=site,
                )
            )
            return
        self._report(
            violation(
                Policy.P3,
                "stale_table_entry",
                f"row {row_index} {what} {j} names block {b} for positions request {rs.request!r} does not hold",
                pool=d.pool,
                block=b,
                generation=self.blocks[d.pool][b].generation,
                request=rs.request,
                site=site,
            )
        )

    def _check_valid_length(
        self,
        d,
        row_index,
        row,
        rs,
        n_entries,
        geometry,
        site,
        *,
        token_entries: bool = False,
    ) -> int | None:
        """P3: the row's declared positions lie within the entries the kernel consumes. Returns
        the last position the row may touch, or None when it touches nothing."""
        touched = [r[-1] for r in (row.reads, row.writes) if len(r)]
        if not touched:
            return None
        last = max(touched)
        beyond = last >= n_entries if token_entries else last // geometry.block_size >= n_entries
        if beyond:
            self._report(
                violation(
                    Policy.P3,
                    "position_beyond_valid_length",
                    f"row {row_index} may touch position {last} but only {n_entries} entries are valid",
                    pool=d.pool,
                    request=rs.request,
                    site=site,
                )
            )
        return last

    def _check_row_mapping(
        self,
        d,
        row_index,
        row: DispatchRow,
        rs: RequestSnapshot,
        entries,
        geometry,
        site,
        start: int = 0,
    ) -> None:
        """P3: the consumed representation equals the request's current mapping (for a dense
        table, its entries from ``start``, the row's first consumed one)."""
        m = d.mapping
        if isinstance(m, SlotMap):
            for t, (s, r, p) in enumerate(zip(m.slots, m.rows, m.positions, strict=True)):
                if r != row_index:
                    continue
                if s == PADDING_SLOT:
                    continue
                if not 0 <= s < geometry.num_slots:
                    self._report(
                        violation(
                            Policy.P3,
                            "mapping_out_of_pool",
                            f"token {t} of request {rs.request!r} writes slot {s}, outside the pool of {geometry.num_slots} slots",
                            pool=d.pool,
                            slot=s,
                            request=rs.request,
                            site=site,
                        )
                    )
                    continue
                expected = rs.slot_at(geometry, p)
                if expected is None:
                    self._report(
                        violation(
                            Policy.P3,
                            "slot_map_position_not_live",
                            f"token {t} writes position {p} of request {rs.request!r}, which is not live",
                            pool=d.pool,
                            slot=s,
                            request=rs.request,
                            site=site,
                        )
                    )
                elif s != expected:
                    self._report(
                        violation(
                            Policy.P3,
                            "slot_map_mismatch",
                            f"token {t} of request {rs.request!r} position {p} writes slot {s}, mapping says {expected}",
                            pool=d.pool,
                            block=geometry.block_of(s) if 0 <= s < geometry.num_slots else None,
                            slot=s,
                            request=rs.request,
                            site=site,
                        )
                    )
                if p not in row.writes:
                    self._report(
                        integration_error(
                            "slot_map_position_undeclared",
                            f"token {t} writes position {p}, outside the row's declared writes",
                            pool=d.pool,
                            request=rs.request,
                            site=site,
                        )
                    )
            return
        if isinstance(m, TokenTable):
            first = m.first_position(row_index)
            self._check_valid_length(
                d,
                row_index,
                row,
                rs,
                first + len(entries),
                geometry,
                site,
                token_entries=True,
            )
            if (
                geometry.block_size == 1
                and tuple(entries) == rs.blocks[first : first + len(entries)]
            ):
                # Block size one: a slot is a block id, so the table's prefix equals the mapping's
                # in one tuple comparison (a hole is None and never equals an entry).
                return True
            if geometry.block_size > 1 and entries:
                # Pages: the mapping expanded to one slot per position, cached on the request's
                # shadow with the block tuple it came from and compared as one tuple; a hole
                # expands to -1 and never equals an entry, so a mismatch falls through to the loop.
                shadow = self.requests.get((rs.request, d.pool))
                if shadow is not None:
                    cache = shadow.slot_cache
                    if cache is None or cache[0] != rs.blocks:
                        bs = geometry.block_size
                        slots = tuple(
                            (b * bs + o) if b is not None else -1
                            for b in rs.blocks
                            for o in range(bs)
                        )
                        shadow.slot_cache = cache = (rs.blocks, slots)
                    stop = first + len(entries)
                    if (
                        rs.first_token <= first
                        and stop <= rs.num_tokens
                        and tuple(entries) == cache[1][first:stop]
                    ):
                        self.counters["token_table_fast_path"] += 1
                        return True
            for p, s in enumerate(entries, first):
                expected = rs.slot_at(geometry, p)
                if expected is None:
                    if rs.is_live(p) or p >= rs.num_tokens:
                        self._report(
                            violation(
                                Policy.P3,
                                "table_beyond_mapping",
                                f"row {row_index} covers position {p}, beyond request {rs.request!r}'s {rs.num_tokens} tokens",
                                pool=d.pool,
                                request=rs.request,
                                site=site,
                            )
                        )
                    else:
                        b = geometry.block_of(s) if 0 <= s < geometry.num_slots else s
                        self._check_dead_entry(
                            d, row_index, p, b, rs, geometry, site, what="position"
                        )
                    continue
                if s != expected:
                    self._report(
                        violation(
                            Policy.P3,
                            "mapping_mismatch",
                            f"row {row_index} position {p}: table says slot {s}, request {rs.request!r} maps {expected}",
                            pool=d.pool,
                            block=geometry.block_of(s) if 0 <= s < geometry.num_slots else None,
                            slot=s,
                            request=rs.request,
                            site=site,
                        )
                    )
            return False
        if isinstance(m, PositionedTokenTable):
            logical = tuple(int(p) for p in m.positions[row_index])
            if len(logical) != len(entries):
                self._report(
                    integration_error(
                        "positioned_table_shape",
                        f"row {row_index} has {len(entries)} table entries but "
                        f"{len(logical)} logical positions",
                        pool=d.pool,
                        request=rs.request,
                        site=site,
                    )
                )
            if tuple(row.reads) != logical:
                self._report(
                    integration_error(
                        "positioned_table_positions_undeclared",
                        f"row {row_index} consumes positions {logical[:8]}, outside its "
                        f"declared reads {tuple(row.reads)[:8]}",
                        pool=d.pool,
                        request=rs.request,
                        site=site,
                    )
                )
            for j, (p, s) in enumerate(zip(logical, entries, strict=False)):
                expected = rs.slot_at(geometry, p)
                if expected is None:
                    self._report(
                        violation(
                            Policy.P3,
                            "table_beyond_mapping",
                            f"row {row_index} entry {j} covers position {p}, which request "
                            f"{rs.request!r} does not map",
                            pool=d.pool,
                            request=rs.request,
                            site=site,
                        )
                    )
                elif s != expected:
                    self._report(
                        violation(
                            Policy.P3,
                            "mapping_mismatch",
                            f"row {row_index} position {p}: table says slot {s}, mapping says "
                            f"{expected}",
                            pool=d.pool,
                            block=geometry.block_of(s) if 0 <= s < geometry.num_slots else None,
                            slot=s,
                            request=rs.request,
                            site=site,
                        )
                    )
            return
        # Dense block table or CSR page list: entry j covers positions [j*bs, (j+1)*bs).
        last = self._check_valid_length(d, row_index, row, rs, len(entries), geometry, site)
        if isinstance(m, CSRPageList) and entries:
            # FlashInfer reads (pages - 1) * page_size + last_page_len positions of the row; that
            # must be exactly what the row declares.
            lpl = m.last_page_len[row_index] if row_index < len(m.last_page_len) else None
            covered = (
                (len(entries) - 1) * geometry.block_size + lpl
                if lpl is not None and 1 <= lpl <= geometry.block_size
                else None
            )
            if covered is None or (last is not None and covered != last + 1):
                self._report(
                    violation(
                        Policy.P3,
                        "page_length_mismatch",
                        f"row {row_index} last_page_len {lpl} covers {covered} positions, the row touches up to position {last}",
                        pool=d.pool,
                        request=rs.request,
                        site=site,
                    )
                )
        # Fast path: every entry names the mapped block of its index; the per-entry loop below
        # then has nothing to report.
        exp = rs.blocks[start : len(entries)]
        if len(exp) == len(entries) - start and None not in exp and tuple(entries[start:]) == exp:
            return
        for j, b in enumerate(entries):
            if j < start:
                continue
            expected = rs.blocks[j] if j < len(rs.blocks) else None
            live = rs.positions_in_block(geometry, j)
            if expected is None:
                if live:
                    self._report(
                        violation(
                            Policy.P3,
                            "table_beyond_mapping",
                            f"row {row_index} entry {j} ({b}) covers positions request {rs.request!r} does not map",
                            pool=d.pool,
                            block=b,
                            request=rs.request,
                            site=site,
                        )
                    )
                else:
                    self._check_dead_entry(d, row_index, j, b, rs, geometry, site)
                continue
            if b != expected:
                self._report(
                    violation(
                        Policy.P3,
                        "mapping_mismatch",
                        f"row {row_index} entry {j}: table says block {b}, request {rs.request!r} maps block {expected}",
                        pool=d.pool,
                        block=b,
                        request=rs.request,
                        site=site,
                    )
                )

    def _check_row_access(
        self,
        d,
        row_index,
        row: DispatchRow,
        rs: RequestSnapshot,
        geometry,
        site,
        pre_written: dict[int, int] | None = None,
        written_all: dict[int, int] | None = None,
        shadow: RequestShadow | None = None,
    ) -> dict[int, int]:
        """P1, P2 and P4 for every position the row may read or write, using the request's mapping
        (already shown to equal the consumed table) and the shadow. ``pre_written`` are the slots
        every row writes (``writes_first``); ``written_all`` the slots earlier rows wrote; both as
        ``{block: offset mask}``. Returns the slots this row writes, in the same form.

        A block whose positions all pass is judged with one mask operation; a block where anything
        fails is re-judged position by position. A read-only row of a request whose shadow is given
        skips the positions its previous read-only row was allowed to read, where the mapping
        entries are unchanged and none of their blocks was touched since (``RequestShadow.read_*``)."""
        bs = geometry.block_size
        row_written: dict[int, int] = {}
        satisfied = pre_written if pre_written is not None else row_written
        earlier = written_all if written_all is not None else {}
        skip_lo = skip_hi = 0
        cacheable = (
            shadow is not None
            and pre_written is None
            and len(row.writes) == 0
            and isinstance(row.reads, range)
            and row.reads.step == 1
        )
        if cacheable and shadow.read_gen >= 0 and len(row.reads):
            a, b = max(row.reads.start, shadow.read_lo), min(row.reads.stop, shadow.read_hi)
            if a < b:
                ia, ib = a // bs, -(-b // bs)
                base = shadow.read_base
                same = rs.blocks[ia:ib] == shadow.read_blocks[ia - base : ib - base]
                if same and ib - ia == len(rs.blocks[ia:ib]):
                    # The verified prefix up to the first block touched since.
                    touched_at = self.touched_at[d.pool]
                    g = shadow.read_gen
                    k = 0
                    for blk in rs.blocks[ia:ib]:
                        if blk is None or touched_at[blk] > g:
                            break
                        k += 1
                    hi = min(b, (ia + k) * bs)
                    if k and hi > a:
                        skip_lo, skip_hi = a, hi
                        self.counters["read_positions_skipped"] += hi - a
        reports_before = self.nreports
        # The loop below runs once per block a row touches, once per position at page size 1; the
        # predicates are inlined and the lookups hoisted. `is_live(p) and is_live(p_hi - 1)` is
        # `first <= p` and `p_hi <= num_tokens`, as p < p_hi.
        shadows = self.blocks[d.pool]
        reserved = geometry.reserved_blocks
        num_blocks = geometry.num_blocks
        rblocks = rs.blocks
        nblocks = len(rblocks)
        first, ntok = rs.first_token, rs.num_tokens
        request = rs.request
        private, published = BlockState.PRIVATE, BlockState.PUBLISHED
        for kind, positions in (("write", row.writes), ("read", row.reads)):
            if not isinstance(positions, range) or positions.step != 1:
                for q in positions:
                    self._check_position(
                        d,
                        row_index,
                        kind,
                        q,
                        rs,
                        geometry,
                        site,
                        satisfied,
                        earlier,
                        row_written,
                    )
                continue
            p, stop = positions.start, positions.stop
            is_write = kind == "write"
            while p < stop:
                if not is_write and skip_lo <= p < skip_hi:
                    p = skip_hi
                    continue
                index = p // bs
                p_hi = (index + 1) * bs
                if p_hi > stop:
                    p_hi = stop
                fast = False
                if first <= p and p_hi <= ntok and index < nblocks:
                    b = rblocks[index]
                    if b is not None and 0 <= b < num_blocks and b not in reserved:
                        sh = shadows[b]
                        mask = ((1 << (p_hi - p)) - 1) << (p - index * bs)
                        st = sh.state
                        if st is private and sh.owner == request:
                            allowed = True
                        elif st is published:
                            allowed = not is_write or (sh.pending_publish & mask) == mask
                        else:
                            allowed = False
                        if allowed:
                            if is_write:
                                taken = sh.filled | earlier.get(b, 0) | row_written.get(b, 0)
                                fast = not (taken & mask)
                                if fast:
                                    row_written[b] = row_written.get(b, 0) | mask
                            else:
                                f = sh.filled
                                fast = (f & mask) == mask or (
                                    (f | satisfied.get(b, 0)) & mask
                                ) == mask
                if not fast:
                    for q in range(p, p_hi):
                        self._check_position(
                            d,
                            row_index,
                            kind,
                            q,
                            rs,
                            geometry,
                            site,
                            satisfied,
                            earlier,
                            row_written,
                        )
                p = p_hi
        if cacheable and len(row.reads) and self.nreports == reports_before:
            lo, hi = row.reads.start, row.reads.stop
            shadow.read_lo, shadow.read_hi, shadow.read_gen = lo, hi, self._gen
            shadow.read_base = lo // bs
            shadow.read_blocks = rs.blocks[lo // bs : -(-hi // bs)]
        return row_written

    def _check_position(
        self, d, row_index, kind, p, rs, geometry, site, satisfied, earlier, row_written
    ) -> None:
        """One position of ``_check_row_access``, reported as the per-position check does."""
        s = rs.slot_at(geometry, p)
        if s is None:
            self._report(
                violation(
                    Policy.P3,
                    "access_unmapped_position",
                    f"row {row_index} {kind}s position {p}, which request {rs.request!r} does not map",
                    pool=d.pool,
                    request=rs.request,
                    site=site,
                )
            )
            return
        b = geometry.block_of(s)
        if b in geometry.reserved_blocks:
            return
        if not geometry.in_pool(b):
            # The request's own mapping names a block outside the pool; the access reaches no
            # shadow.
            self._report(
                violation(
                    Policy.P3,
                    "mapping_out_of_pool",
                    f"row {row_index} {kind}s position {p} through block {b}, outside the pool",
                    pool=d.pool,
                    block=b,
                    slot=s,
                    request=rs.request,
                    site=site,
                )
            )
            return
        sh = self.blocks[d.pool][b]
        o = s - b * geometry.block_size
        bit = 1 << o
        if sh.state is BlockState.FREE:
            self._report(
                violation(
                    Policy.P2,
                    "access_free_block",
                    f"row {row_index} {kind}s position {p} in free block {b}",
                    pool=d.pool,
                    block=b,
                    slot=s,
                    generation=sh.generation,
                    request=rs.request,
                    site=site,
                )
            )
            return
        if sh.state is BlockState.PRIVATE and sh.owner != rs.request:
            self._report(
                violation(
                    Policy.P2,
                    "access_foreign_private",
                    f"row {row_index} {kind}s position {p} in block {b} private to {sh.owner!r}",
                    pool=d.pool,
                    block=b,
                    slot=s,
                    generation=sh.generation,
                    request=rs.request,
                    site=site,
                )
            )
            return
        if kind == "write":
            if sh.state is BlockState.PUBLISHED and not (sh.pending_publish & bit):
                self._report(
                    violation(
                        Policy.P2,
                        "write_to_published",
                        f"row {row_index} writes position {p} in published block {b}",
                        pool=d.pool,
                        block=b,
                        slot=s,
                        generation=sh.generation,
                        request=rs.request,
                        site=site,
                    )
                )
                return
            if (sh.is_filled(o) and not geometry.rewritable) or (
                earlier.get(b, 0) & bit or row_written.get(b, 0) & bit
            ):
                self._report(
                    violation(
                        Policy.P4,
                        "double_write",
                        f"row {row_index} writes position {p} whose slot is already initialized",
                        pool=d.pool,
                        block=b,
                        slot=s,
                        generation=sh.generation,
                        request=rs.request,
                        site=site,
                    )
                )
                return
            row_written[b] = row_written.get(b, 0) | bit
        elif not sh.is_filled(o) and not (satisfied.get(b, 0) & bit):
            self._report(
                violation(
                    Policy.P4,
                    "read_uninitialized",
                    f"row {row_index} reads position {p} whose slot was never written",
                    pool=d.pool,
                    block=b,
                    slot=s,
                    generation=sh.generation,
                    request=rs.request,
                    site=site,
                )
            )


__all__ = ["BlockShadow", "Epoch", "Ledger", "RequestShadow"]
