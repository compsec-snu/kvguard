"""The tag tier: one 8-byte tag per slot, kept on the device from the write, the free, the publish
and the external fill, and compared at every read with the owner and logical position the row asks.

The tag of slot ``s`` is two 32-bit words: ``flags | owner << 8`` and the logical position
(:data:`ANY_POS` when none was declared). ``WRITTEN`` says the slot holds a value written since its
block was last freed, ``SHARED`` that the block is published (any owner may read), ``FREED`` that
the block was freed after the write, ``ALLOC`` that the block is allocated (a read or write of a
slot without it is an access to free memory). Owner id 0 is "unknown" (an external engine acting
for no request): readable and rewritable by anyone and checked against no ownership. The tier needs
no shadow: adapters feed it the slot map of every write (:meth:`TagMirror.stamp`), the freed blocks
(:meth:`TagMirror.free`), the published blocks (:meth:`TagMirror.publish`) and the filled or copied
slots (:meth:`TagMirror.fill`, :meth:`TagMirror.external_copy`); the read validators compare each
consumed position's tag with the row. Reserved blocks are the lowest ids of the pool; a padding row
may name those only, a live row never.

:class:`TagReference` is the same semantics on the host, in Python. The kernels (``kvg_tag_*``)
load with the validators' module; results come back through
:meth:`~kvguard.gpu.validate.Validators.completed` with tags prefixed ``tag:``, and
:func:`report_tag_results` turns them into the core's reports.
"""

from __future__ import annotations

import ctypes as C
import re
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.ledger import Ledger
from ..core.model import ExternalCopy, ExternalFill, PoolGeometry, PoolId
from ..core.violations import Policy, Reporter, violation
from .validate import KINDS, OWNER_SHIFT, PADDING_ROW, OwnerIds, Result, Validators

WRITTEN, SHARED, FREED, ALLOC = 1, 2, 4, 8
ANY_POS = 0xFFFFFFFF
"""A tag position or an expected position that is not known: no position is compared."""
OP_FREE, OP_PUBLISH, OP_FILL, OP_FILL_REWRITE, OP_ALLOC, OP_INVALIDATE = 1, 2, 3, 4, 5, 6
K_OUT_OF_POOL, K_RESERVED_LIVE, K_FREE, K_FOREIGN, K_UNDER_FILLED = 1, 2, 3, 4, 5
K_PADDING_MANAGED, K_WRITE_PUBLISHED, K_DOUBLE_WRITE, K_PAGE_LENGTH, K_TAG_POSITION = 6, 7, 8, 9, 10
TAG_PREFIX = "tag:"
"""Result tags of the tier's launches start with this, so a collector can route them."""


def reserved_set(geometry: PoolGeometry) -> frozenset[int]:
    """The geometry's reserved blocks (any set inside the pool)."""
    return frozenset(b for b in geometry.reserved_blocks if 0 <= b < geometry.num_blocks)


@dataclass(frozen=True)
class TagViolation:
    kind: int
    row: int
    position: int
    slot: int
    w0: int = 0
    w1: int = 0

    @property
    def name(self) -> str:
        return KINDS[self.kind][0]


def _tag_read(a: int, b: int, owner: int, pos: int) -> int:
    if a & FREED or not a & ALLOC:
        return K_FREE
    if not a & WRITTEN:
        return K_UNDER_FILLED
    held = a >> OWNER_SHIFT
    if not a & SHARED and owner != 0 and held != 0 and held != owner:
        return K_FOREIGN
    if pos != ANY_POS and b != ANY_POS and b != pos:
        return K_TAG_POSITION
    return 0


def _tag_write(a: int, owner: int, rewrite: bool) -> tuple[int, int]:
    """The violation kind of a write over tag word ``a`` and the new word."""
    shared = 0 if a & FREED else a & SHARED
    k = 0
    if not a & ALLOC:
        k = K_FREE
    elif a & WRITTEN and not a & FREED:
        if not rewrite:
            k = K_DOUBLE_WRITE
        elif a & SHARED:
            k = K_WRITE_PUBLISHED
        elif (a >> OWNER_SHIFT) not in (0, owner) and owner != 0:
            k = K_FOREIGN
    return k, (owner << OWNER_SHIFT) | WRITTEN | (a & ALLOC) | shared


class TagReference:
    """The tag tier's semantics on the host, slot for slot as the kernels compute them. Every
    violation is returned (the device counts them and records the first)."""

    def __init__(self, geometry: PoolGeometry):
        self.geometry = geometry
        self.bs = geometry.block_size
        self.num_blocks = geometry.num_blocks
        self.reserved = reserved_set(geometry)
        self.tags: dict[int, tuple[int, int]] = {}

    def tag(self, slot: int) -> tuple[int, int]:
        return self.tags.get(slot, (0, 0))

    def stamp(
        self,
        slots: Sequence[int],
        owners: Sequence[int],
        positions: Sequence[int],
        rewrite: bool = False,
    ) -> list[TagViolation]:
        out = []
        num_slots = self.num_blocks * self.bs
        for t, s in enumerate(slots):
            if s == -1:
                continue
            owner, pos = owners[t], positions[t]
            if s < 0 or s >= num_slots:
                out.append(TagViolation(K_OUT_OF_POOL, t, pos, s & 0xFFFFFFFF))
                continue
            reserved = s // self.bs in self.reserved
            if owner == PADDING_ROW:
                if not reserved:
                    out.append(TagViolation(K_PADDING_MANAGED, t, pos, s))
                continue
            if reserved:
                out.append(TagViolation(K_RESERVED_LIVE, t, pos, s))
                continue
            a, b = self.tag(s)
            k, new = _tag_write(a, owner, rewrite)
            self.tags[s] = (new, pos)
            if k:
                out.append(TagViolation(k, t, pos, s, a, b))
        return out

    def update(self, recs: Sequence[tuple[int, int, int, int, int, int]]) -> list[TagViolation]:
        out = []
        for i, (blk, op, owner, pos0, lo, hi) in enumerate(recs):
            hi = min(hi, self.bs)
            if blk < 0 or blk >= self.num_blocks:
                out.append(TagViolation(K_OUT_OF_POOL, i, pos0, blk & 0xFFFFFFFF))
                continue
            for o in range(lo, hi):
                s = blk * self.bs + o
                a, b = self.tag(s)
                if op == OP_FREE:
                    self.tags[s] = (
                        (a & ~(WRITTEN | SHARED | ALLOC)) | (FREED if a & WRITTEN else 0),
                        b,
                    )
                elif op == OP_PUBLISH:
                    self.tags[s] = (a | SHARED, b)
                elif op == OP_ALLOC:
                    self.tags[s] = ((a & ~FREED) | ALLOC, b)
                elif op == OP_INVALIDATE:
                    self.tags[s] = (a & ~WRITTEN, b)
                else:
                    pos = ANY_POS if pos0 == ANY_POS else pos0 + (o - lo)
                    k, new = _tag_write(a, owner, op == OP_FILL_REWRITE)
                    self.tags[s] = (new, pos)
                    if k:
                        out.append(TagViolation(k, i, pos, s, a, b))
        return out

    def update_slots(self, slots: Sequence[int], op: int) -> list[TagViolation]:
        """``OP_FREE``, ``OP_PUBLISH``, ``OP_ALLOC`` or ``OP_INVALIDATE`` over single slots
        (``-1`` skipped)."""
        out = []
        num_slots = self.num_blocks * self.bs
        for i, s in enumerate(slots):
            if s == -1:
                continue
            if s < 0 or s >= num_slots:
                out.append(TagViolation(K_OUT_OF_POOL, i, ANY_POS, s & 0xFFFFFFFF))
                continue
            a, b = self.tag(s)
            if op == OP_FREE:
                self.tags[s] = (
                    (a & ~(WRITTEN | SHARED | ALLOC)) | (FREED if a & WRITTEN else 0),
                    b,
                )
            elif op == OP_PUBLISH:
                self.tags[s] = (a | SHARED, b)
            elif op == OP_ALLOC:
                self.tags[s] = ((a & ~FREED) | ALLOC, b)
            elif op == OP_INVALIDATE:
                self.tags[s] = (a & ~WRITTEN, b)
        return out

    def invalidate(self, pairs: Sequence[tuple[int, int]]) -> list[TagViolation]:
        """A speculative rollback: the (block, offset) slots are unwritten again."""
        return self.update([(b, OP_INVALIDATE, 0, ANY_POS, o, o + 1) for b, o in pairs])

    def allocate(self, blocks: Sequence[int]) -> list[TagViolation]:
        return self.update([(b, OP_ALLOC, 0, ANY_POS, 0, self.bs) for b in blocks])

    def free(self, blocks: Sequence[int]) -> list[TagViolation]:
        return self.update([(b, OP_FREE, 0, ANY_POS, 0, self.bs) for b in blocks])

    def publish(self, blocks: Sequence[int]) -> list[TagViolation]:
        return self.update([(b, OP_PUBLISH, 0, ANY_POS, 0, self.bs) for b in blocks])

    def fill(
        self, block: int, lo: int, hi: int, owner: int, pos0: int = ANY_POS, rewrite: bool = False
    ) -> list[TagViolation]:
        return self.update([(block, OP_FILL_REWRITE if rewrite else OP_FILL, owner, pos0, lo, hi)])

    def _position(self, out, blk, j, o, r, owner, pos) -> None:
        if blk < 0 or blk >= self.num_blocks:
            out.append(TagViolation(K_OUT_OF_POOL, r, pos, j))
            return
        reserved = blk in self.reserved
        if owner == PADDING_ROW:
            if not reserved:
                out.append(TagViolation(K_PADDING_MANAGED, r, pos, blk))
            return
        if reserved:
            out.append(TagViolation(K_RESERVED_LIVE, r, pos, blk))
            return
        s = blk * self.bs + o
        a, b = self.tag(s)
        k = _tag_read(a, b, owner, pos)
        if k:
            out.append(TagViolation(k, r, pos, s, a, b))

    def check_dense(
        self,
        table: Sequence[Sequence[int]],
        row_owner: Sequence[int],
        row_end: Sequence[int],
        row_start: Sequence[int] | None = None,
        width: int | None = None,
    ) -> list[TagViolation]:
        out: list[TagViolation] = []
        bs = self.bs
        for r, row in enumerate(table):
            owner, end = row_owner[r], row_end[r]
            w = len(row) if width is None else width
            if owner == PADDING_ROW and end == 0:
                for j in range(w):
                    self._position(out, row[j], j, 0, r, owner, j * bs)
                continue
            start = 0 if owner == PADDING_ROW else (row_start[r] if row_start else 0)
            for i in range(start, min(end, w * bs)):
                j, o = divmod(i, bs)
                self._position(out, row[j], j, o, r, owner, i)
        return out

    def check_csr(
        self,
        indptr: Sequence[int],
        indices: Sequence[int],
        last_page_len: Sequence[int],
        row_owner: Sequence[int],
        row_end: Sequence[int],
        entry_slots: int | None = None,
        row_start: Sequence[int] | None = None,
    ) -> list[TagViolation]:
        """``entry_slots`` is the positions one list entry covers: the block size (an entry names a
        block, the default) or 1 (an entry names a slot); ``row_start`` the first position of each
        row (a sliding-window suffix row)."""
        out: list[TagViolation] = []
        es = self.bs if entry_slots is None else entry_slots
        for r in range(len(indptr) - 1):
            owner, end = row_owner[r], row_end[r]
            start = 0 if owner == PADDING_ROW or row_start is None else row_start[r]
            span = max(end - start, 0)
            lo, hi = indptr[r], indptr[r + 1]
            npages = max(hi - lo, 0)
            if owner != PADDING_ROW and span > 0:
                lpl = last_page_len[r]
                if not (npages > 0 and 1 <= lpl <= es and (npages - 1) * es + lpl == span):
                    out.append(TagViolation(K_PAGE_LENGTH, r, end, npages, lpl & 0xFFFFFFFF, 0))
            limit = npages * es if owner == PADDING_ROW and end == 0 else min(span, npages * es)
            for k in range(limit):
                i = start + k
                j, o = divmod(k, es)
                entry = indices[lo + j]
                if es == self.bs:
                    self._position(out, entry, j, o, r, owner, i)
                else:
                    s = entry * es + o
                    if s < 0 or s >= self.num_blocks * self.bs:
                        out.append(TagViolation(K_OUT_OF_POOL, r, i, s & 0xFFFFFFFF))
                        continue
                    b, o2 = divmod(s, self.bs)
                    self._position(out, b, j, o2, r, owner, i)
        return out

    def check_token_table(
        self,
        table: Sequence[Sequence[int]],
        row_owner: Sequence[int],
        row_end: Sequence[int],
        row_first: Sequence[int] | None = None,
    ) -> list[TagViolation]:
        out: list[TagViolation] = []
        bs, num_slots = self.bs, self.num_blocks * self.bs
        for r, row in enumerate(table):
            owner, end = row_owner[r], row_end[r]
            first = row_first[r] if row_first else 0
            for p in range(first, end):
                s = row[p - first]
                if s < 0 or s >= num_slots:
                    out.append(TagViolation(K_OUT_OF_POOL, r, p, s & 0xFFFFFFFF))
                    continue
                b, o = divmod(s, bs)
                self._position(out, b, b, o, r, owner, p)
        return out

    def check_ranges(self, recs: Sequence[tuple[int, int, int, int, int]]) -> list[TagViolation]:
        out: list[TagViolation] = []
        for i, (blk, owner, pos0, lo, hi) in enumerate(recs):
            for o in range(lo, min(hi, self.bs)):
                pos = ANY_POS if pos0 == ANY_POS else pos0 + (o - lo)
                self._position(out, blk, blk, o, i, owner, pos)
        return out


@dataclass
class TagStats:
    stamps: int = 0
    tokens_stamped: int = 0
    checks: int = 0
    positions_checked: int = 0
    updates: int = 0
    records: int = 0
    launches: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class TagMirror:
    """One pool's tags on the device, with the tier's launches. Shares the validators' module,
    result slots and upload rings (one :class:`~kvguard.gpu.validate.Validators` per process)."""

    v: Validators
    geometry: PoolGeometry
    owners: OwnerIds = field(default_factory=OwnerIds)
    stats: TagStats = field(default_factory=TagStats)

    def __post_init__(self) -> None:
        self.drv = self.v.drv
        self.pool: PoolId = self.geometry.pool
        self.bs = self.geometry.block_size
        self.num_blocks = self.geometry.num_blocks
        self.reserved_blocks = reserved_set(self.geometry)
        self.num_slots = self.num_blocks * self.bs
        self.tags = self.drv.mem_alloc(8 * max(self.num_slots, 1))
        self.drv.memset_d32_async(self.tags, 0, 2 * max(self.num_slots, 1), 0)
        # One byte per block: the geometry's reserved blocks (the null block).
        nres = (max(self.num_blocks, 1) + 3) // 4 * 4
        self.reserved = self.drv.mem_alloc(nres)
        self.drv.memset_d32_async(self.reserved, 0, nres // 4, 0)
        if self.reserved_blocks:
            flags = array("B", [0] * nres)
            for b in self.reserved_blocks:
                flags[b] = 1
            self.drv.memcpy_htod(self.reserved, flags.buffer_info()[0], nres)
        self.drv.stream_synchronize(0)

    @property
    def nbytes(self) -> int:
        return 8 * self.num_slots

    def _rows(self, data: array, stream: int) -> tuple[int, int]:
        self.drv.ensure_context()
        r = self.v.rows.take(max(len(data) * 4, 4))
        return r, self.v.rows.upload(r, data, stream)

    def _run(self, name: str, args: list, grid: tuple, block: tuple, stream: int, tag: str, r: int):
        i = self._i
        self.v._launch(name, args, grid, block, stream, i)
        self.v.rows.release(r, stream)
        # The result's tag names the pool before the launch's own tag, so a collector routes it
        # without parsing the dispatch id.
        self.v._finish(i, stream, f"{TAG_PREFIX}{self.pool}|{tag}")
        self.stats.launches += 1

    def stamp(
        self,
        slots_ptr: int,
        elem_bytes: int,
        n_tokens: int,
        tok_owner: Sequence[int],
        tok_pos: Sequence[int],
        stream: int,
        tag: str,
        rewrite: bool = False,
    ) -> None:
        """Token ``t`` writes ``slots[t]`` (int64 or int32 at ``slots_ptr``; ``-1`` writes
        nothing) on behalf of owner ``tok_owner[t]`` (``PADDING_ROW`` for a padding token) at
        logical position ``tok_pos[t]``. ``rewrite`` admits writes into written slots of the
        owner's own (a rewritable pool)."""
        if elem_bytes not in (4, 8):
            raise ValueError("slot map elements must be 4 or 8 bytes")
        if len(tok_owner) != n_tokens or len(tok_pos) != n_tokens:
            raise ValueError("tok_owner and tok_pos must have one entry per token")
        self._i, counter, record = self.v._slot(stream)
        r, p = self._rows(array("I", tok_owner) + array("I", tok_pos), stream)
        args = [
            C.c_void_p(slots_ptr), C.c_uint32(elem_bytes), C.c_uint32(n_tokens), C.c_void_p(p),
            C.c_void_p(p + 4 * n_tokens), C.c_uint32(self.bs), C.c_uint32(self.num_blocks),
            C.c_void_p(self.reserved), C.c_uint32(1 if rewrite else 0), C.c_void_p(self.tags),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_stamp", args, (max((n_tokens + 255) // 256, 1),), (256,), stream, tag, r)
        self.stats.stamps += 1
        self.stats.tokens_stamped += n_tokens

    def update(
        self, recs: Sequence[tuple[int, int, int, int, int, int]], stream: int, tag: str = "update"
    ) -> None:
        """Apply maintenance records ``(block, op, owner, pos0, lo, hi)`` (``OP_FREE``,
        ``OP_PUBLISH``, ``OP_FILL``, ``OP_FILL_REWRITE``, ``OP_ALLOC``) over slots ``[lo, hi)``
        of each block; a fill reports as a stamp does. The records of one call run concurrently
        (one thread block each) and must not overlap; order dependent updates go in separate
        calls, which the stream serializes."""
        if not recs:
            return
        self._i, counter, record = self.v._slot(stream)
        data = array("I", [x & 0xFFFFFFFF for rec in recs for x in rec])
        r, p = self._rows(data, stream)
        args = [
            C.c_void_p(p), C.c_uint32(len(recs)), C.c_uint32(self.bs), C.c_uint32(self.num_blocks),
            C.c_void_p(self.tags), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_update", args, (len(recs),), (128,), stream, tag, r)
        self.stats.updates += 1
        self.stats.records += len(recs)

    def update_slots(
        self,
        op: int,
        stream: int,
        tag: str = "update",
        slots: Sequence[int] | None = None,
        slots_ptr: int | None = None,
        elem_bytes: int = 4,
        n: int | None = None,
    ) -> None:
        """``OP_FREE`` or ``OP_PUBLISH`` over single slots: a host list ``slots`` (uploaded as
        int32) or a device list at ``slots_ptr`` of ``n`` elements of ``elem_bytes`` (a
        framework tensor as it is); ``-1`` entries are skipped."""
        if op not in (OP_FREE, OP_PUBLISH, OP_ALLOC, OP_INVALIDATE):
            raise ValueError("update_slots takes OP_FREE, OP_PUBLISH, OP_ALLOC or OP_INVALIDATE")
        if slots is not None:
            n = len(slots)
            if n == 0:
                return
            self._i, counter, record = self.v._slot(stream)
            r, slots_ptr = self._rows(array("i", [int(x) for x in slots]), stream)
            elem_bytes = 4
        else:
            if slots_ptr is None or n is None:
                raise ValueError("update_slots needs slots or slots_ptr and n")
            if n == 0:
                return
            if elem_bytes not in (4, 8):
                raise ValueError("slot list elements must be 4 or 8 bytes")
            self._i, counter, record = self.v._slot(stream)
            r, _ = self._rows(array("I", [0]), stream)
        args = [
            C.c_void_p(slots_ptr), C.c_uint32(elem_bytes), C.c_uint32(n), C.c_uint32(op),
            C.c_uint32(self.bs), C.c_uint32(self.num_blocks), C.c_void_p(self.tags),
            C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_update_slots", args, (max((n + 255) // 256, 1),), (256,), stream, tag, r)
        self.stats.updates += 1
        self.stats.records += n

    def stamp_host(
        self,
        slots: Sequence[int],
        tok_owner: Sequence[int],
        tok_pos: Sequence[int],
        stream: int,
        tag: str,
        rewrite: bool = False,
    ) -> None:
        """:meth:`stamp` of a slot map the adapter derived on the host (TensorRT-LLM's KV append
        writes through the offsets table and has no device slot map): the list is uploaded as
        int64 through the rows ring."""
        n = len(slots)
        if n == 0:
            return
        self.drv.ensure_context()
        r = self.v.rows.take(8 * n)
        data = array("q", [int(x) for x in slots])
        C.memmove(self.v.rows.host[r], data.buffer_info()[0], 8 * n)
        self.drv.memcpy_htod_async(self.v.rows.dev[r], self.v.rows.host[r], 8 * n, stream)
        self.stamp(self.v.rows.dev[r], 8, n, tok_owner, tok_pos, stream, tag, rewrite)
        self.v.rows.release(r, stream)

    def allocate(self, blocks: Sequence[int], stream: int, tag: str = "alloc") -> None:
        self.update([(b, OP_ALLOC, 0, ANY_POS, 0, self.bs) for b in blocks], stream, tag)

    def invalidate(
        self, pairs: Sequence[tuple[int, int]], stream: int, tag: str = "invalidate"
    ) -> None:
        """A speculative rollback: the (block, offset) slots are unwritten again (allocated and
        shared as they were), so the next write of them is not a double write."""
        if pairs:
            self.update_slots(OP_INVALIDATE, stream, tag, slots=[b * self.bs + o for b, o in pairs])

    def free(self, blocks: Sequence[int], stream: int, tag: str = "free") -> None:
        self.update([(b, OP_FREE, 0, ANY_POS, 0, self.bs) for b in blocks], stream, tag)

    def publish(self, blocks: Sequence[int], stream: int, tag: str = "publish") -> None:
        self.update([(b, OP_PUBLISH, 0, ANY_POS, 0, self.bs) for b in blocks], stream, tag)

    def fill(
        self,
        block: int,
        lo: int,
        hi: int,
        owner: int,
        stream: int,
        tag: str = "fill",
        pos0: int = ANY_POS,
        rewrite: bool = False,
    ) -> None:
        op = OP_FILL_REWRITE if rewrite else OP_FILL
        self.update([(block, op, owner, pos0, lo, hi)], stream, tag)

    def external_fill(
        self, desc: ExternalFill, stream: int, pos0: int = ANY_POS, rewrite: bool = False
    ) -> None:
        """Stamp the slots an :class:`~kvguard.core.model.ExternalFill` initializes."""
        owner = self.owners.id_of(desc.request)
        self.fill(
            desc.block, desc.offsets.start, desc.offsets.stop, owner, stream,
            f"fill-{desc.source}", pos0, rewrite,
        )  # fmt: skip

    def external_copy(
        self,
        desc: ExternalCopy,
        dst: TagMirror,
        stream: int,
        src_pos0: int = ANY_POS,
        dst_pos0: int = ANY_POS,
        rewrite: bool = False,
    ) -> None:
        """Check the source slots of an :class:`~kvguard.core.model.ExternalCopy` as a read by
        its request, then stamp the destination (on ``dst``, the destination pool's mirror)."""
        owner = self.owners.id_of(desc.request)
        lo, hi = desc.offsets.start, desc.offsets.stop
        self.check_ranges([(desc.block, owner, src_pos0, lo, hi)], stream, f"copy-{desc.source}")
        dst.fill(desc.dst_block, lo, hi, owner, stream, f"copy-{desc.source}", dst_pos0, rewrite)

    def check_dense(
        self,
        table_ptr: int,
        row_stride: int,
        rows: int,
        width: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
        row_start: Sequence[int] | None = None,
    ) -> None:
        """Dense int32 block table: row ``r`` reads positions ``[row_start[r], row_end[r])``
        through entry ``pos // bs``; a padding row with ``row_end == 0`` may name reserved blocks
        only over its whole width."""
        self._i, counter, record = self.v._slot(stream)
        starts = array("I", row_start) if row_start is not None else array("I", [0] * rows)
        if len(starts) != rows or len(row_owner) != rows or len(row_end) != rows:
            raise ValueError("row arrays must have one entry per row")
        r, p = self._rows(array("I", row_owner) + array("I", row_end) + starts, stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(rows), C.c_uint32(width),
            C.c_void_p(p), C.c_void_p(p + 4 * rows), C.c_void_p(p + 8 * rows),
            C.c_uint32(self.bs), C.c_uint32(self.num_blocks), C.c_void_p(self.reserved),
            C.c_void_p(self.tags), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_check_dense", args, (max(rows, 1),), (128,), stream, tag, r)
        self._count(sum(max(e - s, 0) for e, s in zip(row_end, starts, strict=True)))

    def check_dense_offsets(
        self,
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
        """TensorRT-LLM block-offsets table, decoded as the validator decodes it."""
        self._i, counter, record = self.v._slot(stream)
        r, p = self._rows(array("I", row_owner) + array("I", row_end), stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(kv_stride), C.c_uint32(rows),
            C.c_uint32(width), C.c_uint32(divisor), C.c_void_p(p), C.c_void_p(p + 4 * rows),
            C.c_uint32(self.bs), C.c_uint32(self.num_blocks), C.c_void_p(self.reserved),
            C.c_void_p(self.tags), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_check_dense_offsets", args, (max(rows, 1),), (128,), stream, tag, r)
        self._count(sum(row_end))

    def check_csr(
        self,
        indptr_ptr: int,
        indices_ptr: int,
        last_page_len_ptr: int,
        rows: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
        entry_slots: int | None = None,
        row_start: Sequence[int] | None = None,
    ) -> None:
        """FlashInfer-style int32 page list: positions ``[row_start[r], row_end[r])`` (starts
        all zero when omitted; a sliding-window wrapper's row starts at its window) through the
        row's pages; the pages and ``last_page_len`` must cover exactly those positions. An entry
        covers ``entry_slots`` positions: the block size (the default) or 1 when the list names
        slots (SGLang's page sizes above one)."""
        es = self.bs if entry_slots is None else int(entry_slots)
        if es not in (1, self.bs):
            raise ValueError(f"entry_slots must be 1 or the block size, got {es}")
        starts = array("I", row_start) if row_start is not None else array("I", [0] * rows)
        if len(starts) != rows:
            raise ValueError("row_start must have one entry per row")
        self._i, counter, record = self.v._slot(stream)
        r, p = self._rows(array("I", row_owner) + array("I", row_end) + starts, stream)
        args = [
            C.c_void_p(indptr_ptr), C.c_void_p(indices_ptr), C.c_void_p(last_page_len_ptr),
            C.c_uint32(rows), C.c_void_p(p), C.c_void_p(p + 4 * rows), C.c_void_p(p + 8 * rows),
            C.c_uint32(es), C.c_uint32(self.bs), C.c_uint32(self.num_blocks),
            C.c_void_p(self.reserved), C.c_void_p(self.tags), C.c_void_p(counter),
            C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_check_csr", args, (max(rows, 1),), (128,), stream, tag, r)
        self._count(sum(max(e - s, 0) for e, s in zip(row_end, starts, strict=True)))

    def check_token_table(
        self,
        table_ptr: int,
        row_stride: int,
        rows: int,
        row_owner: Sequence[int],
        row_end: Sequence[int],
        stream: int,
        tag: str,
        row_first: Sequence[int] | None = None,
    ) -> None:
        """Per-token int32 table: ``table[r][c]`` is the slot of position ``row_first[r] + c``
        below ``row_end[r]`` (``row_first`` all zero when omitted)."""
        self._i, counter, record = self.v._slot(stream)
        firsts = array("I", row_first) if row_first is not None else array("I", [0] * rows)
        r, p = self._rows(array("I", row_owner) + array("I", row_end) + firsts, stream)
        args = [
            C.c_void_p(table_ptr), C.c_uint32(row_stride), C.c_uint32(rows),
            C.c_void_p(p), C.c_void_p(p + 4 * rows), C.c_void_p(p + 8 * rows),
            C.c_uint32(self.bs), C.c_uint32(self.num_blocks), C.c_void_p(self.reserved),
            C.c_void_p(self.tags), C.c_void_p(counter), C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_check_token_table", args, (max(rows, 1),), (128,), stream, tag, r)
        self._count(sum(max(e - f, 0) for e, f in zip(row_end, firsts, strict=True)))

    def check_ranges(
        self, recs: Sequence[tuple[int, int, int, int, int]], stream: int, tag: str
    ) -> None:
        """Ranges a non-kernel engine reads: ``(block, owner, pos0, lo, hi)`` per record, slots
        ``[lo, hi)`` at positions ``pos0 + (o - lo)`` (``ANY_POS``: none; owner 0: any)."""
        if not recs:
            return
        self._i, counter, record = self.v._slot(stream)
        data = array("I", [x & 0xFFFFFFFF for rec in recs for x in rec])
        r, p = self._rows(data, stream)
        args = [
            C.c_void_p(p), C.c_uint32(len(recs)), C.c_uint32(self.bs), C.c_uint32(self.num_blocks),
            C.c_void_p(self.reserved), C.c_void_p(self.tags), C.c_void_p(counter),
            C.c_void_p(record),
        ]  # fmt: skip
        self._run("kvg_tag_check_ranges", args, (len(recs),), (128,), stream, tag, r)
        self._count(sum(max(min(hi, self.bs) - lo, 0) for _, _, _, lo, hi in recs))

    def _count(self, positions: int) -> None:
        self.stats.checks += 1
        self.stats.positions_checked += positions

    def read_back(self, slots: Sequence[int]) -> list[tuple[int, int]]:
        """The tag words of ``slots`` after every launch so far (synchronizes)."""
        self.drv.ensure_context()
        self.drv.stream_synchronize(0)
        buf = (C.c_uint32 * 2)()
        out = []
        for s in slots:
            self.drv.memcpy_dtoh(C.addressof(buf), self.tags + 8 * s, 8)
            out.append((int(buf[0]), int(buf[1])))
        return out

    def set_tag(self, slot: int, w0: int, w1: int, stream: int) -> None:
        """Overwrite one slot's tag (fault injection and tests)."""
        self.drv.ensure_context()
        data = array("I", [w0 & 0xFFFFFFFF, w1 & 0xFFFFFFFF])
        r = self.v.rows.take(8)
        C.memmove(self.v.rows.host[r], data.buffer_info()[0], 8)
        self.drv.memcpy_htod_async(self.tags + 8 * slot, self.v.rows.host[r], 8, stream)
        self.v.rows.release(r, stream)


def is_tag_result(res: Result) -> bool:
    return res.tag.startswith(TAG_PREFIX)


_POOL_OF_TAG = re.compile(r"(?:write|attn|read)-([^@]+)@")


class TagTier:
    """The tag tier of one process: a :class:`TagMirror` per covered pool, kept from the
    ledger's listener events (``Ledger.listeners``: the mediated frees, stores, reuses,
    external fills and copies). Adapters subclass it for the per-step stamp and check launches
    and for the publish hook their framework offers; ``stream()`` is the stream the maintenance
    launches go to (the current torch stream by default)."""

    def __init__(
        self,
        ledger: Ledger,
        pools: Iterable[PoolId],
        validators: Validators | None = None,
        owners: OwnerIds | None = None,
    ):
        self.ledger = ledger
        self.v = validators if validators is not None else Validators(timing=True)
        self.owners = owners if owners is not None else OwnerIds()
        self.mirrors: dict[PoolId, TagMirror] = {}
        for pool in pools:
            if pool not in self.mirrors:
                self.mirrors[pool] = TagMirror(self.v, ledger.pools[pool], self.owners)
        ledger.listeners.append(self)
        self.reports = 0
        self.violations = 0
        self.gpu_ms = 0.0
        self.host_s = 0.0
        self.publish_from_snapshot = False
        """Take the publish event from the ledger's snapshot transitions
        (:meth:`flush_published`) when the framework offers no publish hook; the publication is
        then state-derived, which the evaluation states."""

    def flush_published(self, stream: int | None = None) -> int:
        """Publish the blocks the ledger's snapshots published since the last flush; returns
        their number (0 unless ``publish_from_snapshot``)."""
        if not self.publish_from_snapshot:
            return 0
        n = 0
        for pool, m in self.mirrors.items():
            blocks = self.ledger.drain_published(pool)
            if blocks:
                m.publish(
                    blocks,
                    self.stream() if stream is None else stream,
                    self._site("snapshot-publish"),
                )
                n += len(blocks)
        return n

    def stream(self) -> int:
        import torch

        return torch.cuda.current_stream().cuda_stream

    def _site(self, what: str) -> str:
        return f"{what}@{self.ledger.seq}"

    def on_allocate(self, pool: PoolId, blocks: list) -> None:
        m = self.mirrors.get(pool)
        if m is not None:
            m.allocate(blocks, self.stream(), self._site("alloc"))

    def on_free(self, pool: PoolId, blocks: list) -> None:
        m = self.mirrors.get(pool)
        if m is not None:
            m.free(blocks, self.stream(), self._site("free"))

    def on_store(self, pool: PoolId, blocks: list) -> None:
        m = self.mirrors.get(pool)
        if m is not None:
            m.publish(blocks, self.stream(), self._site("store"))

    def on_invalidate(self, pool: PoolId, pairs: list) -> None:
        m = self.mirrors.get(pool)
        if m is not None:
            m.invalidate(pairs, self.stream(), self._site("invalidate"))

    def on_reuse(self, pool: PoolId, block: int, retained: int) -> None:
        """A take-over keeps the first ``retained`` slots; the tail is the new owner's to
        write, so its tags are freed (the kept prefix stays shared and readable)."""
        m = self.mirrors.get(pool)
        if m is None:
            return
        stream = self.stream()
        if retained < m.bs:
            # Two launches: the records of one update run concurrently and may not overlap.
            m.update([(block, OP_FREE, 0, ANY_POS, retained, m.bs)], stream, self._site("reuse"))
        m.allocate([block], stream, self._site("reuse"))

    def on_external_fill(self, fill: ExternalFill) -> None:
        m = self.mirrors.get(fill.pool)
        if m is not None:
            pos0 = ANY_POS if fill.position is None else fill.position
            m.external_fill(fill, self.stream(), pos0, self.ledger.pools[fill.pool].rewritable)

    def on_external_copy(self, copy: ExternalCopy) -> None:
        src, dst = self.mirrors.get(copy.pool), self.mirrors.get(copy.dst_pool)
        pos0 = ANY_POS if copy.position is None else copy.position
        if src is None:
            if dst is not None:
                dst.fill(
                    copy.dst_block, copy.offsets.start, copy.offsets.stop,
                    self.owners.id_of(copy.request), self.stream(), f"copy-{copy.source}", pos0,
                    self.ledger.pools[copy.dst_pool].rewritable,
                )  # fmt: skip
            return
        if dst is None:
            src.check_ranges(
                [(copy.block, self.owners.id_of(copy.request), pos0, copy.offsets.start, copy.offsets.stop)],
                self.stream(),
                f"copy-{copy.source}",
            )  # fmt: skip
            return
        src.external_copy(
            copy, dst, self.stream(), pos0, pos0, self.ledger.pools[copy.dst_pool].rewritable
        )

    def pool_of_tag(self, tag: str) -> PoolId:
        """The pool a launch's tag names (``write-swa3@12``: ``swa3``); the only pool, or the
        first registered, otherwise."""
        if "|" in tag:
            pool = tag.split("|", 1)[0]
            if pool in self.mirrors:
                return pool
        m = _POOL_OF_TAG.match(tag)
        if m and m.group(1) in self.mirrors:
            return m.group(1)
        return next(iter(self.mirrors))

    def collect(self, results: Sequence[Result], reporter: Reporter) -> list[Result]:
        """Report the tag tier's results among ``results`` through ``reporter``, routed to the
        pool their tag names; returns the other results."""
        rest: list[Result] = []
        by_pool: dict[PoolId, list[Result]] = {}
        for r in results:
            if not is_tag_result(r):
                rest.append(r)
                continue
            by_pool.setdefault(self.pool_of_tag(r.tag[len(TAG_PREFIX) :]), []).append(r)
            if r.count:
                self.violations += r.count
            if r.gpu_ms is not None:
                self.gpu_ms += r.gpu_ms
        for pool, rs in by_pool.items():
            self.reports += report_tag_results(rs, reporter, pool)
        return rest

    def completed(self) -> list[Result]:
        return self.v.completed()

    def drain(self) -> list[Result]:
        return self.v.drain()

    def readback(self, reporter: Reporter) -> list[Result]:
        """Report the tier's completed launches (no host synchronization); returns the other
        completed results (the validators', when the loader is shared and the attachment has no
        validator collector of its own)."""
        return self.collect(self.v.completed(), reporter)

    def drain_into(self, reporter: Reporter) -> list[Result]:
        """Wait for every launch and report the tier's; returns the others."""
        return self.collect(self.v.drain(), reporter)

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tag_reports": self.reports,
            "tag_violations": self.violations,
            "tag_gpu_ms": round(self.gpu_ms, 3),
            "tag_host_s": round(self.host_s, 4),
            "tag_bytes": sum(m.nbytes for m in self.mirrors.values()),
        }
        for pool, m in self.mirrors.items():
            for k, v in m.stats.as_dict().items():
                out[f"tag_{pool}_{k}"] = v
        return out


def report_tag_results(results: Sequence[Result], reporter: Reporter, pool: PoolId) -> int:
    """Emit one report per tag launch that found violations; returns the number emitted."""
    n = 0
    for res in results:
        if res.count == 0:
            continue
        name, policy = KINDS.get(res.kind, (f"kind_{res.kind}", Policy.P3))
        w0, w1 = res.stored
        held = ""
        if w0 or w1:
            flags = "".join(
                f
                for f, bit in (("W", WRITTEN), ("S", SHARED), ("F", FREED), ("A", ALLOC))
                if w0 & bit
            )
            pos = "any" if w1 == ANY_POS else str(w1)
            held = f", tag held owner {w0 >> OWNER_SHIFT} position {pos} flags {flags or '-'}"
        detail = (
            f"{res.count} position{'' if res.count == 1 else 's'} rejected by the tag tier; "
            f"first: {name} at row/token {res.row}, position {res.index}, slot/block {res.entry}"
            f"{held}"
        )
        launch = res.tag[len(TAG_PREFIX) :]
        launch = launch.split("|", 1)[1] if "|" in launch else launch
        reporter.emit(
            violation(
                policy,
                name,
                detail,
                pool=pool,
                block=res.entry if res.kind in (2, 6) else None,
                site=f"gpu-tags:{launch}",
            )
        )
        n += 1
    return n


__all__ = [
    "ALLOC",
    "ANY_POS",
    "FREED",
    "OP_ALLOC",
    "OP_FILL",
    "OP_FILL_REWRITE",
    "OP_FREE",
    "OP_INVALIDATE",
    "OP_PUBLISH",
    "SHARED",
    "TAG_PREFIX",
    "WRITTEN",
    "TagMirror",
    "TagReference",
    "TagStats",
    "TagTier",
    "TagViolation",
    "is_tag_result",
    "report_tag_results",
    "reserved_set",
]
