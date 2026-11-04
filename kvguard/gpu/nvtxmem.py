"""NVIDIA Compute Sanitizer told about the KV cache's sub-allocations through the NVTX memory
API (``docs.nvidia.com/compute-sanitizer/SanitizerNvtxGuide``): an evaluation baseline, not a
KVGuard tier.

The ledger's events drive the registrations, so the frameworks stay unmodified: the pool tensors
become sub-allocator heaps, every block becomes one region per lane (a layer's K or V slab) at its
allocation and is unregistered at its free, a published block becomes read-only once its content is
complete, and a reuse resets it. ``memcheck`` then reports kernel accesses to unregistered parts of
a heap and permission violations; ``initcheck`` treats a fresh region as uninitialized. The API
cannot express per-request permissions (they are per stream or process-wide, so another request's
read of a live private block is invisible) and checks neither the mapping (P3) nor the asynchronous
transfer rules. Under no sanitizer the calls are no-ops.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable, Sequence

from kvguard.core import ExternalCopy, ExternalFill, Ledger
from kvguard.core.model import BlockState, PoolId

READ, WRITE, ATOMIC, RESET = 0x1, 0x2, 0x4, 0x8
LEVEL = int(os.environ.get("KVGUARD_NVTX_LEVEL", "3"))
TRACE = os.environ.get("KVGUARD_NVTX_TRACE", "0") == "1"
"""Print every shim call to stderr (diagnosis of the tool's behaviour)."""


def _trace(msg: str) -> None:
    if TRACE:
        import sys

        print(f"[kvg-nvtx] {msg}", file=sys.stderr, flush=True)


"""Diagnostic depth: 0 the domain only, 1 the heaps, 2 the regions, 3 the permissions too."""
LIB = os.environ.get(
    "KVGUARD_NVTX_LIB",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "build",
        "nvtx",
        "libkvg_nvtx.so",
    ),
)


class Lane:
    """One contiguous slab per block: ``base + block * stride`` bytes of ``size``; ``heap`` is
    the handle of the registered heap (the CUDA allocation) that holds the lane."""

    __slots__ = ("base", "heap", "size", "stride")

    def __init__(self, base: int, stride: int, size: int):
        self.base, self.stride, self.size = int(base), int(stride), int(size)
        self.heap = 0

    def ptr(self, block: int) -> int:
        return self.base + block * self.stride


def block_lanes(tensor, num_blocks: int) -> list[Lane]:
    """The lanes of a KV cache tensor whose blocks are one axis (any position): one lane per
    index combination of the axes outside the block axis (K and V halves, for instance), with
    the block's slab the contiguous inner extent. Rejects a tensor whose inner extent is not
    contiguous (the block's bytes would not be one region)."""
    import itertools

    shape, strides = list(tensor.shape), list(tensor.stride())
    elem = tensor.element_size()
    axes = [i for i, n in enumerate(shape) if n == num_blocks]
    if not axes:
        raise ValueError(f"no axis of size {num_blocks} in shape {tuple(shape)}")
    k = axes[0]
    # The block's slab is one region when the inner axes tile the stride exactly, in any order
    # (vLLM's runner views a block as [heads, block_size, head_dim] over a [block_size, heads,
    # head_dim] layout): sorted by stride, each inner axis starts where the smaller ones end.
    inner_axes = sorted(
        (i for i in range(k + 1, len(shape)) if shape[i] > 1), key=lambda i: strides[i]
    )
    inner = 1
    for i in inner_axes:
        if strides[i] != inner:
            raise ValueError(
                f"block slab not contiguous in shape {tuple(shape)} strides {tuple(strides)}"
            )
        inner *= shape[i]
    if strides[k] != inner:
        raise ValueError(f"blocks not adjacent in shape {tuple(shape)} strides {tuple(strides)}")
    size = inner * elem
    outer = [range(shape[i]) for i in range(k)]
    lanes = []
    for idx in itertools.product(*outer) if outer else [()]:
        base = tensor.data_ptr() + sum(j * strides[i] for i, j in enumerate(idx)) * elem
        lanes.append(Lane(base, strides[k] * elem, size))
    return lanes


def allocation_of(tensor) -> tuple[int, int, list[tuple[int, int]]]:
    """The CUDA allocation holding the tensor: the caching allocator's segment (address, size
    and its blocks as address and size, from ``torch.cuda.memory._snapshot``) when one contains
    the tensor's storage, else the tensor's own range as one block."""
    base = int(tensor.data_ptr())
    nbytes = int(tensor.numel()) * int(tensor.element_size())
    try:
        import torch

        for seg in torch.cuda.memory._snapshot().get("segments", []):
            addr, size = int(seg["address"]), int(seg["total_size"])
            if addr <= base < addr + size:
                blocks, at = [], addr
                for b in seg.get("blocks", []):
                    n = int(b.get("size", 0))
                    blocks.append((at, n))
                    at += n
                _trace(
                    f"segment base=0x{addr:x} size={size} expandable={seg.get('is_expandable')} "
                    f"type={seg.get('segment_type')} blocks={[n for _, n in blocks][:8]}"
                )
                return addr, size, blocks
    except Exception:
        pass
    return base, nbytes, [(base, nbytes)]


def foreign_blocks(
    blocks: Iterable[tuple[int, int]], spans: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The allocation's blocks that overlap none of the pool tensors' byte spans: the caching
    allocator rounds a large allocation up to 2 MiB and keeps the remainder as a free block of
    the same segment, which it later hands to other tensors."""
    spans = list(spans)
    return [(a, n) for a, n in blocks if not any(a < e and s < a + n for s, e in spans)]


class NvtxMemory:
    """The shim's calls over the ledger's pools."""

    def __init__(self, ledger: Ledger, domain: str = "kvguard", lib=None):
        self.lib = ctypes.CDLL(LIB) if lib is None else lib
        self.lib.kvg_nvtx_init.restype = ctypes.c_int
        self.lib.kvg_nvtx_init.argtypes = [ctypes.c_char_p]
        self.lib.kvg_nvtx_heap_register.restype = ctypes.c_void_p
        self.lib.kvg_nvtx_heap_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.kvg_nvtx_heap_unregister.argtypes = [ctypes.c_void_p]
        U64 = ctypes.POINTER(ctypes.c_uint64)
        self.lib.kvg_nvtx_regions_register.argtypes = [ctypes.c_void_p, U64, U64, ctypes.c_size_t]
        self.lib.kvg_nvtx_regions_unregister.argtypes = [U64, ctypes.c_size_t]
        self.lib.kvg_nvtx_permissions_assign.argtypes = [U64, ctypes.c_size_t, ctypes.c_uint32]
        self.lib.kvg_nvtx_mark_initialized.argtypes = [U64, U64, ctypes.c_size_t, ctypes.c_void_p]
        self.lib.kvg_nvtx_set_default_access.argtypes = [ctypes.c_int, ctypes.c_uint32]
        self.ok = bool(self.lib.kvg_nvtx_init(domain.encode()))
        self.ledger = ledger
        self.lanes: dict[PoolId, list[Lane]] = {}
        self.heaps: dict[PoolId, list[int]] = {}
        self.registered: dict[PoolId, set[int]] = {}
        self.readonly: dict[PoolId, set[int]] = {}
        self.pending_readonly: dict[PoolId, set[int]] = {}
        self._readonly_due: dict[PoolId, set[int]] = {}
        self.stats = {
            "heaps": 0,
            "regions_registered": 0,
            "regions_unregistered": 0,
            "readonly_assigned": 0,
            "readonly_reset": 0,
            "batches": 0,
        }
        ledger.listeners.append(self)

    def add_pool(
        self, pool: PoolId, tensors: Iterable, num_blocks: int, reserved: Iterable[int] = ()
    ) -> int:
        """Register each tensor as a heap and derive its lanes; the reserved blocks (the null
        block) are registered as permanent regions since padding lanes touch them."""
        lanes: list[Lane] = []
        heaps: list[int] = []
        ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        spans: list[tuple[int, int]] = []
        for t in tensors:
            lanes.extend(block_lanes(t, num_blocks))
            base, size, blocks = allocation_of(t)
            ranges.setdefault((base, size), blocks)
            start = int(t.data_ptr())
            spans.append((start, start + int(t.numel()) * int(t.element_size())))
        device = 0
        for t in tensors:
            dev = getattr(t, "device", None)
            if getattr(dev, "index", None) is not None:
                device = int(dev.index)
                break
        # ``nvtxMemCudaSetPeerAccess`` on the process- or device-wide permissions kills the
        # sanitized process (CUDA 13.2's tool), so the default access is left as the tool defines
        # it and every region is assigned its permission explicitly.
        self.device = device
        handle_of: dict[tuple[int, int], int] = {}
        foreign_total = 0
        for (base, size), blocks in ranges.items():
            if LEVEL < 1:
                heaps.append(0)
                continue
            # The tool accepts a heap only over a CUDA allocation it saw being made: the caching
            # allocator's segment that holds the tensor, not the tensor's own sub-range.
            _trace(f"heap_register base=0x{base:x} size={size}")
            h = self.lib.kvg_nvtx_heap_register(ctypes.c_void_p(base), size)
            _trace(f"  -> handle {h}")
            heaps.append(h or 0)
            handle_of[(base, size)] = h or 0
            foreign = foreign_blocks(blocks, spans)
            if foreign and LEVEL >= 2:
                # The segment's remainder outside the pool is not the ledger's to track:
                # registered once, under this segment's heap, as regions with every permission,
                # the tool checks the tensors the allocator places there as it checks an
                # ordinary allocation (by the region's bounds).
                n = len(foreign)
                ptrs = (ctypes.c_uint64 * n)(*[a for a, _ in foreign])
                sizes = (ctypes.c_uint64 * n)(*[m for _, m in foreign])
                _trace(f"regions_register(foreign) heap={h} n={n} sizes={[m for _, m in foreign]}")
                self.lib.kvg_nvtx_regions_register(ctypes.c_void_p(h or 0), ptrs, sizes, n)
                if LEVEL >= 3:
                    self.lib.kvg_nvtx_permissions_assign(ptrs, n, READ | WRITE | ATOMIC)
                foreign_total += n
        if foreign_total:
            self.stats["foreign_regions"] = self.stats.get("foreign_regions", 0) + foreign_total
        # Each lane's regions are registered under the heap that holds the lane: one segment
        # on vLLM (every lane in it), one segment per layer tensor on SGLang.
        for lane in lanes:
            lane.heap = next(
                (h for (base, size), h in handle_of.items() if base <= lane.base < base + size), 0
            )
        self.lanes[pool] = lanes
        self.heaps[pool] = heaps
        self.stats["heaps_failed"] = self.stats.get("heaps_failed", 0) + sum(
            1 for h in heaps if not h
        )
        self.registered.setdefault(pool, set())
        self.readonly.setdefault(pool, set())
        self.pending_readonly.setdefault(pool, set())
        self._readonly_due.setdefault(pool, set())
        self.stats["heaps"] += len(heaps)
        self._register(pool, list(reserved))
        return len(lanes)

    def _arrays(self, pool: PoolId, blocks: Sequence[int]):
        lanes = self.lanes[pool]
        n = len(blocks) * len(lanes)
        ptrs = (ctypes.c_uint64 * n)()
        sizes = (ctypes.c_uint64 * n)()
        i = 0
        for b in blocks:
            for lane in lanes:
                ptrs[i] = lane.ptr(b)
                sizes[i] = lane.size
                i += 1
        return ptrs, sizes, n

    def _register(self, pool: PoolId, blocks: Sequence[int]) -> None:
        new = [b for b in blocks if b not in self.registered[pool]]
        if not new or not self.lanes.get(pool) or LEVEL < 2:
            return
        # A batch registers its regions under one heap, so the lanes are grouped by the heap
        # that holds them (one group on vLLM's single segment, one per layer tensor on SGLang).
        by_heap: dict[int, list[Lane]] = {}
        for lane in self.lanes[pool]:
            by_heap.setdefault(lane.heap, []).append(lane)
        for heap, lanes in by_heap.items():
            n = len(new) * len(lanes)
            ptrs = (ctypes.c_uint64 * n)()
            sizes = (ctypes.c_uint64 * n)()
            i = 0
            for b in new:
                for lane in lanes:
                    ptrs[i] = lane.ptr(b)
                    sizes[i] = lane.size
                    i += 1
            _trace(
                f"regions_register heap={heap} n={n} first=0x{int(ptrs[0]):x} size={int(sizes[0])}"
            )
            self.lib.kvg_nvtx_regions_register(ctypes.c_void_p(heap), ptrs, sizes, n)
            self.stats["regions_registered"] += n
            self.stats["batches"] += 1
        self.registered[pool].update(new)
        ptrs, sizes, n = self._arrays(pool, new)
        if LEVEL >= 3:
            # A fresh region carries no permission under the tool (its writes are reported
            # until one is assigned): the block's owner may read, write and update it.
            _trace(f"permissions_assign(new) n={n}")
            self.lib.kvg_nvtx_permissions_assign(ptrs, n, READ | WRITE | ATOMIC)
            self.stats["batches"] += 1

    def _unregister(self, pool: PoolId, blocks: Sequence[int]) -> None:
        old = [b for b in blocks if b in self.registered[pool]]
        if not old:
            return
        ptrs, _, n = self._arrays(pool, old)
        _trace(f"regions_unregister n={n}")
        self.lib.kvg_nvtx_regions_unregister(ptrs, n)
        self.registered[pool].difference_update(old)
        self.readonly[pool].difference_update(old)
        self.pending_readonly[pool].difference_update(old)
        self._readonly_due[pool].difference_update(old)
        self.stats["regions_unregistered"] += n
        self.stats["batches"] += 1

    def _permissions(self, pool: PoolId, blocks: Sequence[int], flags: int) -> None:
        if not blocks or LEVEL < 3:
            return
        ptrs, _, n = self._arrays(pool, blocks)
        _trace(f"permissions_assign n={n} flags={flags}")
        self.lib.kvg_nvtx_permissions_assign(ptrs, n, flags)
        self.stats["batches"] += 1

    def on_allocate(self, pool: PoolId, blocks: list) -> None:
        if pool in self.lanes:
            self._register(pool, blocks)

    def on_free(self, pool: PoolId, blocks: list) -> None:
        if pool in self.lanes:
            self._unregister(pool, [b for b in blocks if b not in self._reserved(pool)])

    def on_store(self, pool: PoolId, blocks: list) -> None:
        if pool in self.lanes:
            self.pending_readonly[pool].update(blocks)

    def on_reuse(self, pool: PoolId, block: int, retained: int) -> None:
        if pool in self.lanes:
            self._register(pool, [block])
            if block in self.readonly[pool]:
                self._permissions(pool, [block], READ | WRITE | ATOMIC)
                self.readonly[pool].discard(block)
                self.stats["readonly_reset"] += 1

    def on_invalidate(self, pool: PoolId, pairs: list) -> None:
        pass

    def on_external_fill(self, fill: ExternalFill) -> None:
        if fill.pool in self.lanes:
            self._register(fill.pool, [fill.block])

    def on_external_copy(self, copy: ExternalCopy) -> None:
        if copy.dst_pool in self.lanes:
            self._register(copy.dst_pool, [copy.dst_block])

    def _reserved(self, pool: PoolId) -> frozenset:
        return self.ledger.pools[pool].reserved_blocks

    def step(self) -> None:
        """Make the published blocks whose content is complete read-only, one step late.

        A block is published at the schedule that completes it, before the step that writes
        its tail, so the permission waits for the ledger's filled mask; and the ledger credits
        a step's writes when the dispatch is validated, before the kernels run, while the tool
        checks permissions at the launch. The blocks found complete at this call are therefore
        assigned at the next call, after this step's kernels were launched (the same thread
        launches them before it prepares the next step).
        """
        for pool, due in self._readonly_due.items():
            live = [b for b in due if b in self.registered[pool] and b not in self.readonly[pool]]
            if live:
                self._permissions(pool, live, READ)
                self.readonly[pool].update(live)
                self.stats["readonly_assigned"] += len(live)
            due.clear()
        drain = getattr(self.ledger, "drain_published", None)
        for pool, pending in self.pending_readonly.items():
            if drain is not None:
                # Publications the snapshot transitions found without a store hook.
                pending.update(drain(pool))
            if not pending:
                continue
            geometry = self.ledger.pools[pool]
            full = (1 << geometry.block_size) - 1
            ready = []
            for b in list(pending):
                sh = self.ledger.block(pool, b)
                if sh.state is not BlockState.PUBLISHED:
                    pending.discard(b)
                    continue
                if sh.filled & full == full and not sh.pending_publish:
                    ready.append(b)
            if ready:
                self._readonly_due.setdefault(pool, set()).update(ready)
                pending.difference_update(ready)

    def record(self) -> dict:
        return {
            "ok": self.ok,
            "lanes": {p: len(v) for p, v in self.lanes.items()},
            "registered": {p: len(v) for p, v in self.registered.items()},
            "readonly": {p: len(v) for p, v in self.readonly.items()},
            **self.stats,
        }


__all__ = ["ATOMIC", "READ", "RESET", "WRITE", "Lane", "NvtxMemory", "block_lanes"]
