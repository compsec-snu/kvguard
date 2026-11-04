"""deep_size counts each object once and the ledger's footprint grows with its pool."""

from __future__ import annotations

import sys

import numpy as np
from kvguard.core.ledger import Ledger
from kvguard.core.memsize import deep_size
from kvguard.core.model import PoolGeometry


def test_deep_size_counts_shared_objects_once_and_enters_containers():
    shared = [1, 2, 3]
    obj = {"a": shared, "b": shared, "c": (shared, "x")}
    once = deep_size(obj)
    assert once >= sys.getsizeof(obj) + sys.getsizeof(shared) + sys.getsizeof("x")
    seen: set[int] = set()
    deep_size(shared, seen)
    assert deep_size(obj, seen) < once  # the shared list is already counted


def test_deep_size_counts_numpy_buffers_and_skips_types():
    arr = np.zeros(1024, dtype=np.int64)
    assert deep_size(arr) >= arr.nbytes
    view = arr[10:]
    assert deep_size(view) >= view.nbytes
    assert deep_size(int) == 0 and deep_size(deep_size) == 0


def test_ledger_memory_bytes_grows_with_the_pool():
    small, large = Ledger(), Ledger()
    small.register_pool(PoolGeometry("kv", 64, 16))
    large.register_pool(PoolGeometry("kv", 4096, 16))
    a, b = small.memory_bytes(), large.memory_bytes()
    assert set(Ledger.SHADOW_ATTRIBUTES) <= set(a) and a["total"] == sum(
        v for k, v in a.items() if k != "total"
    )
    assert b["blocks"] > a["blocks"] and b["total"] > a["total"]
    assert b["nonfree_mask"] >= 4096
