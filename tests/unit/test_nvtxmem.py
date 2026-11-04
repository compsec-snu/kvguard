"""The Compute Sanitizer baseline's registrations follow the ledger's events (no GPU, no tool:
the shim is replaced by a recorder)."""

from __future__ import annotations

import pytest
from kvguard.core import Ledger, PoolGeometry, Reporter
from kvguard.gpu.nvtxmem import ATOMIC, READ, WRITE, NvtxMemory, block_lanes, foreign_blocks

from tests.unit.conftest import BS, World


class FakeTensor:
    def __init__(self, shape, strides, elem=2, ptr=0x10000):
        self.shape, self._strides, self._elem, self._ptr = shape, strides, elem, ptr

    def stride(self):
        return self._strides

    def element_size(self):
        return self._elem

    def data_ptr(self):
        return self._ptr

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def ndim(self):
        return len(self.shape)


def test_block_lanes_split_the_axes_outside_the_block_axis():
    # vLLM's FlashAttention layout (2, num_blocks, block_size, heads, head_dim), contiguous.
    t = FakeTensor((2, 8, 4, 2, 8), (8 * 4 * 2 * 8, 4 * 2 * 8, 2 * 8, 8, 1))
    lanes = block_lanes(t, 8)
    assert [(lane.base - t.data_ptr(), lane.stride, lane.size) for lane in lanes] == [
        (0, 128, 128),
        (8 * 128, 128, 128),
    ]
    # Blocks on axis 0: one lane.
    t = FakeTensor((8, 2, 4, 2, 8), (2 * 4 * 2 * 8, 4 * 2 * 8, 2 * 8, 8, 1))
    assert [(lane.stride, lane.size) for lane in block_lanes(t, 8)] == [(256, 256)]
    # Model Runner V2's view: [blocks, heads, block_size, head_dim] over a [blocks, block_size,
    # heads, head_dim] layout.
    t = FakeTensor((39485, 8, 16, 128), (16384, 128, 1024, 1))
    assert [(lane.stride, lane.size) for lane in block_lanes(t, 39485)] == [(32768, 32768)]
    with pytest.raises(ValueError):
        block_lanes(FakeTensor((8, 4), (1, 8)), 8)  # transposed: a block is not one slab
    with pytest.raises(ValueError):
        block_lanes(FakeTensor((8, 4), (8, 2)), 8)  # a gap inside the slab


class Recorder:
    """The shim's calls, recorded."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("kvg_nvtx_"):

            def call(*args):
                self.calls.append((name[len("kvg_nvtx_") :], args))
                return 1

            return call
        raise AttributeError(name)


def registered(rec, what):
    return [
        (a[1][i], a[2][i]) if what == "regions_register" else a[0][i]
        for name, a in rec.calls
        if name == what
        for i in range(a[-1] if what != "permissions_assign" else a[1])
    ]


def test_registrations_follow_allocate_free_publish_and_reuse():
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(PoolGeometry("kv", 16, BS, reserved_blocks=frozenset({0})))
    rec = Recorder()
    nv = NvtxMemory(ledger, lib=rec)
    t = FakeTensor((2, 16, BS, 1, 8), (16 * BS * 8, BS * 8, 8, 8, 1))
    assert nv.add_pool("kv", [t], 16, reserved=[0]) == 2
    assert [n for n, _ in rec.calls][:2] == ["init", "heap_register"]
    assert nv.registered["kv"] == {0}  # the null block, permanently

    world = World(ledger, pool="kv")
    world.alloc(3).request("A", [3], BS).observe().write("A", 0, BS)
    assert nv.registered["kv"] == {0, 3}
    regs = [a for n, a in rec.calls if n == "regions_register"]
    assert regs[-1][3] == 2  # two lanes per block
    assert list(regs[-1][1]) == [lane.ptr(3) for lane in nv.lanes["kv"]]

    # Publication: read-only once the content is complete (the step after the store).
    world.published(3).observe()
    nv.step()  # found complete: assigned at the next step, after this step's launches
    assert nv.readonly["kv"] == set()
    nv.step()
    assert nv.readonly["kv"] == {3}
    perms = [a for n, a in rec.calls if n == "permissions_assign"]
    assert perms[-1][2] == READ and perms[-1][1] == 2

    # A take-over resets the permission; a free unregisters the regions.
    nv.on_reuse("kv", 3, 2)
    assert 3 not in nv.readonly["kv"]
    assert [a for n, a in rec.calls if n == "permissions_assign"][-1][2] == READ | WRITE | ATOMIC
    world.drop_request("A")
    ledger.free("kv", (3,))
    assert nv.registered["kv"] == {0}
    assert [n for n, _ in rec.calls][-1] == "regions_unregister"
    assert nv.record()["regions_unregistered"] == 2


def test_foreign_blocks_are_the_segment_blocks_outside_the_pool_tensors():
    # vLLM's cache on 0.32.0: one active block holding the sixteen layer tensors and the
    # caching allocator's 1.5 MiB rounding remainder behind it.
    blocks = [(0x1000, 20_701_511_680), (0x1000 + 20_701_511_680, 1_572_864)]
    spans = [(0x1000 + i * 1_293_844_480, 0x1000 + (i + 1) * 1_293_844_480) for i in range(16)]
    assert foreign_blocks(blocks, spans) == [(0x1000 + 20_701_511_680, 1_572_864)]
    assert foreign_blocks(blocks, [(0, 0x1000 + 20_703_084_544)]) == []
    # A block partly covered by a tensor belongs to the pool.
    assert foreign_blocks([(0, 100), (100, 100)], [(50, 150)]) == []
