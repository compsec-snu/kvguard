"""Dispatch-time validators on the GPU.

The core's :class:`~kvguard.core.ledger.Ledger` holds the shadow on the host. A
:class:`DeviceMirror` keeps a device copy of its block shadows (one 32-bit state word, a filled
mask and a pending-publication mask per block), refreshed from the ledger's dirty set on the
framework's stream. Before the kernels that consume a mapping table run, the adapter launches the
representation's validator on that stream: every entry a row consumes must name an in-pool block
that is allocated, owned by the row's request or shared, and filled for the positions the row
reads; a slot map's targets must be writable once. Violations are counted on the device and the
first one recorded; :meth:`Validators.completed` reads them back without a host synchronization and
:func:`report_results` turns them into the core's reports.

The mirror's masks are ``ceil(block_size / 32)`` words per block, so any block size is served; the
kernels divide by the block size, which need not be a power of two.
"""

from __future__ import annotations

import ctypes as C
import os
from array import array
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.ledger import BlockShadow, Ledger
from ..core.model import BlockState, PoolId, RequestId
from ..core.violations import Policy, Reporter, violation
from .driver import Driver

ALLOC, PUBLISHED, SHARED, RESERVED = 1, 2, 4, 8
OWNER_SHIFT = 8
MAX_OWNER = (1 << 24) - 1
PADDING_ROW = 0xFFFFFFFF
"""Row owner id of a CUDA-graph padding row: it may touch reserved blocks only."""

KINDS: dict[int, tuple[str, Policy]] = {
    1: ("mapping_out_of_pool", Policy.P3),
    2: ("access_unmapped_position", Policy.P3),  # a reserved block at a live position
    3: ("access_free_block", Policy.P2),
    4: ("access_foreign_private", Policy.P2),
    5: ("read_uninitialized", Policy.P4),
    6: ("padding_row_names_managed_block", Policy.P2),
    7: ("write_to_published", Policy.P2),
    8: ("double_write", Policy.P4),
    9: ("page_length_mismatch", Policy.P3),
    10: ("tag_position_mismatch", Policy.P3),  # the slot holds another logical position
    11: ("mapping_mismatch", Policy.P3),  # the device entry is not the mapping's slot
}
"""Device violation kinds, named as the host reference names the same check."""

KERNELS = (
    "kvg_validate_dense",
    "kvg_validate_dense_offsets",
    "kvg_validate_csr",
    "kvg_validate_token_csr",
    "kvg_validate_token_table",
    "kvg_validate_slot_map",
    "kvg_mirror_update",
    # The tag tier (``kvguard.gpu.tags``), in the same module.
    "kvg_tag_stamp",
    "kvg_tag_check_dense",
    "kvg_tag_check_dense_offsets",
    "kvg_tag_check_csr",
    "kvg_tag_check_token_table",
    "kvg_tag_check_ranges",
    "kvg_tag_update",
    "kvg_tag_update_slots",
)
_RESULT_WORDS = 8  # counter, record[6], padding
HOLE = 0xFFFFFFFF
"""A position of a mapping without a page (``kvg_validate_token_csr``) or a token without an
expected slot (``kvg_validate_slot_map``), as uploaded."""


def default_ptx_path() -> Path:
    env = os.environ.get("KVGUARD_VALIDATORS_PTX")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "build" / "validators" / "validators.ptx"


def pack_block(
    sh: BlockShadow, owner_id: int, reserved: bool = False, words: int = 1
) -> tuple[int, ...]:
    """The mirror words of one block shadow: the state word, then ``words`` filled-mask words,
    then ``words`` pending-mask words (``words`` = ceil(block size / 32); the masks' low words
    first). ``reserved`` is the pool geometry's permanent reservation (null or scratch block),
    not the shadow's cached-and-unmapped flag, which is a published block like any other to a
    reader."""
    st = 0
    if sh.state is not BlockState.FREE:
        st |= ALLOC
    if sh.state is BlockState.PUBLISHED:
        st |= PUBLISHED | SHARED
    if reserved:
        st |= RESERVED
    st |= (owner_id & MAX_OWNER) << OWNER_SHIFT
    out = [st]
    for mask in (sh.filled, sh.pending_publish):
        out.extend((mask >> (32 * w)) & 0xFFFFFFFF for w in range(words))
    return tuple(out)


class OwnerIds:
    """Dense 24-bit ids for request ids (never reused within a run)."""

    def __init__(self) -> None:
        self._ids: dict[RequestId, int] = {}

    def id_of(self, rid: RequestId | None) -> int:
        if rid is None:
            return 0
        i = self._ids.get(rid)
        if i is None:
            i = len(self._ids) + 1
            if i > MAX_OWNER:
                raise RuntimeError("more than 2^24 - 1 requests seen: owner ids exhausted")
            self._ids[rid] = i
        return i

    def __len__(self) -> int:
        return len(self._ids)


class _Ring:
    """A ring of equally sized device and pinned host buffers; an entry is reused only after the
    stream work that read it (an event recorded by ``release``) has completed."""

    def __init__(self, drv: Driver, n: int, nbytes: int):
        self.drv = drv
        self.n = n
        self.cap = [nbytes] * n
        self.host = [drv.mem_alloc_host(nbytes) for _ in range(n)]
        self.dev = [drv.mem_alloc(nbytes) for _ in range(n)]
        self.events = [drv.event_create() for _ in range(n)]
        self.armed = [False] * n
        self.next = 0

    @property
    def nbytes(self) -> int:
        """Device bytes of the ring (the pinned host side holds as many)."""
        return sum(self.cap)

    def take(self, nbytes: int) -> int:
        i = self.next
        self.next = (i + 1) % self.n
        if self.armed[i]:
            self.drv.event_synchronize(self.events[i])
            self.armed[i] = False
        if nbytes > self.cap[i]:
            self.drv.mem_free_host(self.host[i])
            self.drv.mem_free(self.dev[i])
            self.cap[i] = max(nbytes, 2 * self.cap[i])
            self.host[i] = self.drv.mem_alloc_host(self.cap[i])
            self.dev[i] = self.drv.mem_alloc(self.cap[i])
        return i

    def upload(self, i: int, data: array, stream: int) -> int:
        """Copy ``data`` (an ``array('I')``) through entry ``i`` to the device; returns the device
        address. ``release`` must follow the launches that read it."""
        nbytes = len(data) * data.itemsize
        if nbytes:
            C.memmove(self.host[i], data.buffer_info()[0], nbytes)
            self.drv.memcpy_htod_async(self.dev[i], self.host[i], nbytes, stream)
        return self.dev[i]

    def release(self, i: int, stream: int) -> None:
        self.drv.event_record(self.events[i], stream)
        self.armed[i] = True


@dataclass
class Result:
    """The outcome of one validator launch, read back after it completed."""

    tag: str
    count: int
    kind: int
    row: int
    index: int
    entry: int
    gpu_ms: float | None = None
    stored: tuple[int, int] = (0, 0)
    """The tag tier's two tag words at the first violation (zero for the validators)."""

    @property
    def name(self) -> str:
        return KINDS.get(self.kind, (f"kind_{self.kind}", None))[0]


class Validators:
    """The validator kernels of one process: loads the PTX once, owns the result slots and the
    upload rings. One instance serves every pool and stream of the process."""

    def __init__(
        self,
        drv: Driver | None = None,
        ptx: str | None = None,
        slots: int = 64,
        timing: bool = False,
    ):
        self.drv = drv or Driver()
        self.drv.ensure_context()
        self.module = self.drv.load_ptx(ptx if ptx is not None else default_ptx_path().read_text())
        self.fn = {name: self.drv.function(self.module, name) for name in KERNELS}
        self.timing = timing
        self.slots = slots
        self.dev_results = self.drv.mem_alloc(slots * _RESULT_WORDS * 4)
        self.host_results = self.drv.mem_alloc_host(slots * _RESULT_WORDS * 4)
        self.drv.memset_d32_async(self.dev_results, 0, slots * _RESULT_WORDS, 0)
        self.drv.stream_synchronize(0)
        self.events = [self.drv.event_create() for _ in range(slots)]
        self.t_events = (
            [(self.drv.event_create(), self.drv.event_create()) for _ in range(slots)]
            if timing
            else None
        )
        self.in_use = [False] * slots
        self.pending: deque[tuple[int, str]] = deque()
        self.ready: list[Result] = []  # read while a slot was reclaimed, not yet returned
        self.next_slot = 0
        self.rows = _Ring(self.drv, slots, 64 << 10)
        self.mirror_ring = _Ring(self.drv, 4, 1 << 20)
        self.launches = 0
        self.gpu_ms_total = 0.0

    def device_bytes(self) -> int:
        """Device memory of the validators themselves (the result slots and the two rings);
        the mirrors are counted per pool (``DeviceMirror.nbytes``)."""
        return self.slots * _RESULT_WORDS * 4 + self.rows.nbytes + self.mirror_ring.nbytes

    def host_pinned_bytes(self) -> int:
        """Pinned host memory of the validators (the result readback and the rings' host side)."""
        return self.slots * _RESULT_WORDS * 4 + self.rows.nbytes + self.mirror_ring.nbytes

    def mirror(self, ledger: Ledger, pool: PoolId, owners: OwnerIds | None = None) -> DeviceMirror:
        return DeviceMirror(self, ledger, pool, owners)

    def _slot(self, stream: int) -> tuple[int, int, int]:
        # A launch may come from any thread (the tag tier's listener events, a probe's worker
        # thread): the driver API needs the device's context current on it.
        self.drv.ensure_context()
        i = self.next_slot
        self.next_slot = (i + 1) % self.slots
        if self.in_use[i]:
            # The ring wrapped before the adapter read the results: wait for that launch.
            self.drv.event_synchronize(self.events[i])
            res = self._collect(i)
            if res is not None:
                self.ready.append(res)
        self.in_use[i] = True
        base = self.dev_results + i * _RESULT_WORDS * 4
        self.drv.memset_d32_async(base, 0, _RESULT_WORDS, stream)
        return i, base, base + 4

    def _finish(self, i: int, stream: int, tag: str) -> None:
        base = self.dev_results + i * _RESULT_WORDS * 4
        self.drv.memcpy_dtoh_async(
            self.host_results + i * _RESULT_WORDS * 4, base, _RESULT_WORDS * 4, stream
        )
        self.drv.event_record(self.events[i], stream)
        self.pending.append((i, tag))
        self.launches += 1

    def _read(self, i: int, tag: str) -> Result:
        words = (C.c_uint32 * _RESULT_WORDS).from_address(self.host_results + i * _RESULT_WORDS * 4)
        ms = None
        if self.t_events is not None:
            ms = self.drv.event_elapsed_ms(*self.t_events[i])
            self.gpu_ms_total += ms
        self.in_use[i] = False
        return Result(
            tag,
            int(words[0]),
            int(words[1]),
            int(words[2]),
            int(words[3]),
            int(words[4]),
            ms,
            (int(words[5]), int(words[6])),
        )

    def _collect(self, i: int) -> Result | None:
        for k, (slot, tag) in enumerate(self.pending):
            if slot == i:
                del self.pending[k]
                return self._read(i, tag)
        return None

    def completed(self) -> list[Result]:
        """Results of the launches that have completed, oldest first; no host synchronization."""
        out, self.ready = self.ready, []
        while self.pending and self.drv.event_done(self.events[self.pending[0][0]]):
            i, tag = self.pending.popleft()
            out.append(self._read(i, tag))
        return out

    def drain(self) -> list[Result]:
        """Wait for every launch and return all results."""
        out, self.ready = self.ready, []
        while self.pending:
            i, tag = self.pending.popleft()
            self.drv.event_synchronize(self.events[i])
            out.append(self._read(i, tag))
        return out

    def _launch_plain(self, name: str, args: list, grid: tuple, block: tuple, stream: int) -> None:
        self.drv.launch(self.fn[name], grid, block, args, stream)

    def _launch(
        self, name: str, args: list, grid: tuple, block: tuple, stream: int, i: int
    ) -> None:
        if self.t_events is not None:
            self.drv.event_record(self.t_events[i][0], stream)
        self.drv.launch(self.fn[name], grid, block, args, stream)
        if self.t_events is not None:
            self.drv.event_record(self.t_events[i][1], stream)

    def validate_dense(
        self,
        mirror: DeviceMirror,
        table_ptr: int,
        row_stride: int,
        rows: int,
        width: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
        row_start: Sequence[int] | None = None,
        row_write: Sequence[int] | None = None,
    ) -> None:
        """Dense int32 block table at ``table_ptr`` (``row_stride`` elements per row, ``width``
        entries consumed at most); row ``r`` reads positions ``[row_start[r], row_end[r])``
        (``row_start`` all zero when omitted; a sliding-window row starts at its window) on
        behalf of owner ``row_owner[r]`` (``PADDING_ROW`` for a padding row), consuming the
        entries from its first read position's block. A padding row with positive ``row_end``
        consumes that prefix; zero retains the full-width padding-descriptor contract. A row with
        ``row_write[r]`` set writes
        those positions instead, as a rewritable pool's state columns are written: the block
        must be the row's own (or published with the slots pending); fills are not required."""
        i, counter, record = self._slot(stream)
        starts = array("I", row_start) if row_start is not None else array("I", [0] * rows)
        writes = array("I", row_write) if row_write is not None else array("I", [0] * rows)
        if len(starts) != rows or len(writes) != rows:
            raise ValueError("row_start and row_write must have one entry per row")
        data = array("I", row_owner) + array("I", row_end) + starts + writes
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(rows), C.c_uint32(width),
            C.c_void_p(p), C.c_void_p(p + 4 * rows), C.c_void_p(p + 8 * rows),
            C.c_void_p(p + 12 * rows), C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled), C.c_void_p(mirror.pending),
            C.c_uint32(mirror.num_blocks), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._launch("kvg_validate_dense", args, (max(rows, 1),), (128,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)

    def validate_dense_offsets(
        self,
        mirror: DeviceMirror,
        table_ptr: int,
        row_stride: int,
        kv_stride: int,
        rows: int,
        width: int,
        divisor: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
    ) -> None:
        """Block-offsets table (int32) at ``table_ptr``: row ``r``'s K offsets are
        ``table[r * row_stride + j]`` and its V offsets ``kv_stride`` elements further, ``width``
        pairs per row (a batch table ``[max_num_sequences, 2, max_blocks_per_seq]`` of one pool,
        as emitted by engines that express the KV mapping as paired K/V block offsets). A pair
        decodes to block ``K // divisor`` when ``K`` is a non-negative multiple of ``divisor``
        (``layers_per_pool * kv_factor``) and ``V == K + 1``; an undecodable pair is kind 1
        (``mapping_out_of_pool``) with the raw ``K`` offset as its entry. Rows, owners and read
        ends, including the positive-length padding rule, are as in :meth:`validate_dense`."""
        i, counter, record = self._slot(stream)
        data = array("I", row_owner) + array("I", row_end)
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(kv_stride), C.c_uint32(rows),
            C.c_uint32(width), C.c_uint32(divisor), C.c_void_p(p), C.c_void_p(p + 4 * rows),
            C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled),
            C.c_uint32(mirror.num_blocks), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._launch("kvg_validate_dense_offsets", args, (max(rows, 1),), (128,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)

    def validate_csr(
        self,
        mirror: DeviceMirror,
        indptr_ptr: int,
        indices_ptr: int,
        last_page_len_ptr: int,
        rows: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
    ) -> None:
        """FlashInfer-style int32 page list: row ``r``'s pages are
        ``indices[indptr[r]:indptr[r+1]]`` and its last page holds ``last_page_len[r]``
        positions."""
        i, counter, record = self._slot(stream)
        data = array("I", row_owner) + array("I", row_end)
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(indptr_ptr), C.c_void_p(indices_ptr), C.c_void_p(last_page_len_ptr),
            C.c_uint32(rows), C.c_void_p(p), C.c_void_p(p + 4 * rows),
            C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled), C.c_uint32(mirror.num_blocks),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._launch("kvg_validate_csr", args, (max(rows, 1),), (128,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)

    def validate_token_csr(
        self,
        mirror: DeviceMirror,
        indptr_ptr: int,
        indices_ptr: int,
        rows: int,
        row_owner: Sequence[int],
        row_start: Sequence[int],
        row_end: Sequence[int],
        pages: Sequence[Sequence[int | None]],
        stream: int,
        tag: str,
    ) -> None:
        """Per-token int32 page list at any page size (SGLang's FlashInfer ``kv_indices``): row
        ``r``'s entries ``indices[indptr[r]:indptr[r+1]]`` are the slots of its positions
        ``[row_start[r], row_end[r])`` in order, and ``pages[r]`` is the row's mapping (the
        page of each logical block, ``None`` for a hole), against which every entry is
        checked (``mapping_mismatch``) before the permission checks. ``pages`` may be shorter
        than the rows (a padding row has none)."""
        i, counter, record = self._slot(stream)
        if not (len(row_owner) == len(row_start) == len(row_end) == rows):
            raise ValueError("row_owner, row_start and row_end must have one entry per row")
        page_indptr = array("I", [0])
        flat = array("I")
        for r in range(rows):
            if r < len(pages):
                flat.extend(HOLE if b is None else int(b) for b in pages[r])
            page_indptr.append(len(flat))
        data = (
            array("I", row_owner) + array("I", row_end) + array("I", row_start) + page_indptr + flat
        )
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(indptr_ptr), C.c_void_p(indices_ptr), C.c_uint32(rows),
            C.c_void_p(p), C.c_void_p(p + 8 * rows), C.c_void_p(p + 4 * rows),
            C.c_void_p(p + 12 * rows), C.c_void_p(p + 16 * rows + 4),
            C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled), C.c_uint32(mirror.num_blocks),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._launch("kvg_validate_token_csr", args, (max(rows, 1),), (128,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)

    def validate_token_table(
        self,
        mirror: DeviceMirror,
        table_ptr: int,
        row_stride: int,
        rows: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
    ) -> None:
        """Per-token int32 table (SGLang ``req_to_token``): ``table[r][p]`` is the slot of
        position ``p`` for ``p < row_end[r]``."""
        i, counter, record = self._slot(stream)
        data = array("I", row_owner) + array("I", row_end)
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(rows),
            C.c_void_p(p), C.c_void_p(p + 4 * rows), C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled), C.c_uint32(mirror.num_blocks),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._launch("kvg_validate_token_table", args, (max(rows, 1),), (128,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)

    def validate_slot_map(
        self,
        mirror: DeviceMirror,
        slots_ptr: int,
        elem_bytes: int,
        n_tokens: int,
        tok_owner: Sequence[int],
        stream: int,
        tag: str,
        expected: Sequence[int | None] | None = None,
    ) -> None:
        """Flat slot map (int64 or int32 at ``slots_ptr``): token ``t`` writes ``slots[t]`` on
        behalf of owner ``tok_owner[t]``; ``-1`` writes nothing. With ``expected``, token
        ``t``'s slot must be ``expected[t]``, the slot the mapping gives its position
        (``mapping_mismatch``); ``None`` leaves a token unchecked against the mapping."""
        if elem_bytes not in (4, 8):
            raise ValueError("slot map elements must be 4 or 8 bytes")
        if expected is not None and len(expected) != n_tokens:
            raise ValueError("expected must have one entry per token")
        i, counter, record = self._slot(stream)
        data = array("I", tok_owner)
        if expected is not None:
            data += array("I", (HOLE if e is None else int(e) for e in expected))
        r = self.rows.take(len(data) * 4)
        p = self.rows.upload(r, data, stream)
        args = [
            C.c_void_p(slots_ptr), C.c_uint32(elem_bytes), C.c_uint32(n_tokens), C.c_void_p(p),
            C.c_void_p(p + 4 * n_tokens if expected is not None else 0),
            C.c_uint32(mirror.bs), C.c_uint32(mirror.words),
            C.c_void_p(mirror.state), C.c_void_p(mirror.filled),
            C.c_void_p(mirror.pending), C.c_uint32(mirror.num_blocks),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        grid = (max((n_tokens + 255) // 256, 1),)
        self._launch("kvg_validate_slot_map", args, grid, (256,), stream, i)
        self.rows.release(r, stream)
        self._finish(i, stream, tag)


class DeviceMirror:
    """The device copy of one pool's block shadows, refreshed from the ledger's dirty set."""

    def __init__(
        self, validators: Validators, ledger: Ledger, pool: PoolId, owners: OwnerIds | None = None
    ):
        self.v = validators
        self.drv = validators.drv
        self.ledger = ledger
        self.pool = pool
        self.owners = owners or OwnerIds()
        g = ledger.pools[pool]
        self.geometry = g
        self.num_blocks = g.num_blocks
        self.bs = g.block_size
        self.words = (g.block_size + 31) // 32
        """Mask words per block."""
        self.state = self.drv.mem_alloc(4 * g.num_blocks)
        self.filled = self.drv.mem_alloc(4 * g.num_blocks * self.words)
        self.pending = self.drv.mem_alloc(4 * g.num_blocks * self.words)
        self.drv.memset_d32_async(self.state, 0, g.num_blocks, 0)
        for p in (self.filled, self.pending):
            self.drv.memset_d32_async(p, 0, g.num_blocks * self.words, 0)
        self.drv.stream_synchronize(0)
        self.updates = 0
        self.blocks_updated = 0
        # Every block once: reserved bits and whatever the ledger already holds.
        ledger.drain_dirty(pool)
        self._upload(range(g.num_blocks), 0)
        self.drv.stream_synchronize(0)

    @property
    def nbytes(self) -> int:
        """Device bytes of the mirror: one state word and two mask words per block and mask word."""
        return 4 * self.num_blocks * (1 + 2 * self.words)

    def _upload(self, ids, stream: int) -> int:
        shadows = self.ledger.blocks[self.pool]
        reserved = self.geometry.reserved_blocks
        id_arr = array("I")
        vals = array("I")
        words = self.words
        for b in ids:
            sh = shadows[b]
            id_arr.append(b)
            vals.extend(pack_block(sh, self.owners.id_of(sh.owner), b in reserved, words))
        n = len(id_arr)
        if n == 0:
            return 0
        r = self.v.mirror_ring.take(4 * (n + len(vals)))
        p = self.v.mirror_ring.upload(r, id_arr + vals, stream)
        args = [
            C.c_void_p(self.state), C.c_void_p(self.filled), C.c_void_p(self.pending),
            C.c_void_p(p), C.c_void_p(p + 4 * n), C.c_uint32(n), C.c_uint32(words),
        ]  # fmt: skip
        self.v._launch_plain("kvg_mirror_update", args, ((n + 255) // 256,), (256,), stream)
        self.v.mirror_ring.release(r, stream)
        self.updates += 1
        self.blocks_updated += n
        return n

    def sync(self, stream: int) -> int:
        """Push the blocks whose shadow changed since the last sync; returns their number."""
        return self._upload(sorted(self.ledger.drain_dirty(self.pool)), stream)

    def set_words(self, block: int, state: int, filled: int, pending: int, stream: int) -> None:
        """Overwrite one block's mirror words (fault injection and tests); ``filled`` and
        ``pending`` are the whole masks, split into the mirror's words here."""
        n, words = 1, self.words
        vals = [state]
        for mask in (filled, pending):
            vals.extend((mask >> (32 * w)) & 0xFFFFFFFF for w in range(words))
        r = self.v.mirror_ring.take(4 * (1 + len(vals)))
        p = self.v.mirror_ring.upload(r, array("I", [block, *vals]), stream)
        args = [
            C.c_void_p(self.state), C.c_void_p(self.filled), C.c_void_p(self.pending),
            C.c_void_p(p), C.c_void_p(p + 4), C.c_uint32(n), C.c_uint32(words),
        ]  # fmt: skip
        self.v._launch_plain("kvg_mirror_update", args, (1,), (256,), stream)
        self.v.mirror_ring.release(r, stream)


def report_results(results: Sequence[Result], reporter: Reporter, pool: PoolId) -> int:
    """Emit one report per launch that found violations; returns the number emitted."""
    n = 0
    for res in results:
        if res.count == 0:
            continue
        name, policy = KINDS.get(res.kind, (f"kind_{res.kind}", Policy.P3))
        detail = (
            f"{res.count} entr{'y' if res.count == 1 else 'ies'} rejected by the device validator; "
            f"first: {name} at row/token {res.row}, entry {res.index}, block/slot {res.entry}"
        )
        reporter.emit(
            violation(
                policy,
                name,
                detail,
                pool=pool,
                block=res.entry if res.kind in (1, 2, 3, 4, 5, 6, 7, 8) else None,
                slot=res.entry if res.kind == 11 else None,
                site=f"gpu-validate:{res.tag}",
            )
        )
        n += 1
    return n


__all__ = [
    "ALLOC",
    "KINDS",
    "PADDING_ROW",
    "PUBLISHED",
    "RESERVED",
    "SHARED",
    "DeviceMirror",
    "OwnerIds",
    "Result",
    "Validators",
    "default_ptx_path",
    "pack_block",
    "report_results",
]
