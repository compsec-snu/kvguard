"""Python side of the driver interposer: the ctypes control API of the ``LD_PRELOAD`` library and
the descriptor the instrumented kernels read.

The descriptor layout is the rewriter's (``kvguard.gpu.ptx.rewrite``): u64 slots, copied into each
substituted module's ``__kvg_desc`` on the launch stream whenever it changed.
"""

from __future__ import annotations

import ctypes
import json
import math
from collections.abc import Sequence

from .ptx import rewrite as R

MODE_OFF, MODE_SUBSTITUTE = 0, 1


def _lib():
    lib = ctypes.CDLL(None)
    if getattr(lib, "kvg_set_mode", None) is None:
        return None
    lib.kvg_set_mode.argtypes = [ctypes.c_int]
    lib.kvg_set_mode.restype = ctypes.c_int
    lib.kvg_get_mode.restype = ctypes.c_int
    lib.kvg_set_descriptor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.kvg_set_descriptor.restype = ctypes.c_int
    lib.kvg_substitutions.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.kvg_substitutions.restype = ctypes.c_size_t
    return lib


def available() -> bool:
    return _lib() is not None


class Interposer:
    """Control handle on the preloaded library."""

    def __init__(self):
        self.lib = _lib()
        if self.lib is None:
            raise RuntimeError(
                "the KVGuard interposer is not preloaded (LD_PRELOAD=build/interposer/libkvg_interposer.so)"
            )

    def set_mode(self, mode: int) -> int:
        """Set the substitution mode (MODE_OFF or MODE_SUBSTITUTE); returns the previous one."""
        return int(self.lib.kvg_set_mode(int(mode)))

    def mode(self) -> int:
        return int(self.lib.kvg_get_mode())

    def set_descriptor(self, slots: Sequence[int]) -> None:
        """Publish a new descriptor; substituted kernels launched afterwards read it."""
        arr = (ctypes.c_uint64 * len(slots))(*[int(v) & (2**64 - 1) for v in slots])
        rc = self.lib.kvg_set_descriptor(arr, len(slots))
        if rc != 0:
            raise RuntimeError(f"kvg_set_descriptor failed ({rc})")

    def substitutions(self) -> dict:
        """The twins loaded so far, their launch counts, the mode and the descriptor version."""
        n = int(self.lib.kvg_substitutions(None, 0)) + 1
        buf = ctypes.create_string_buffer(n)
        self.lib.kvg_substitutions(buf, n)
        return json.loads(buf.value.decode())


def tier1b_descriptor(
    *,
    counter_ptr: int,
    record_ptr: int,
    table_row_bytes: int,
    table_rows: int,
    row_len_ptr: int,
    block_size: int,
    num_blocks: int,
    entry_bytes: int = 4,
    flat_slot_map: bool = False,
    enabled: bool = True,
) -> list[int]:
    """Descriptor for the table-site bounds checks (Tier 1b): a launch may read
    row ``r`` of the table only below ``ceil(row_len[r] / block_size)`` entries, and only rows below
    ``table_rows``. ``row_len_ptr`` is a device ``int32[table_rows]`` of token counts per row, as the
    shadow knows them; ``counter_ptr`` a device ``int32`` and ``record_ptr`` a device ``int64[4]``
    (site id, row, column, kind) that receive the violations."""
    if block_size & (block_size - 1) or block_size <= 0:
        raise ValueError("block_size must be a power of two")
    d = [0] * R.N_DESC
    d[R.D_ENABLE] = R.EN_TABLE if enabled else 0
    d[R.D_COUNTER] = counter_ptr
    d[R.D_RECORD] = record_ptr
    d[R.D_TBL_ROW_BYTES] = table_row_bytes
    d[R.D_TBL_ENTRY_LOG2] = int(math.log2(entry_bytes))
    d[R.D_TBL_NROWS] = table_rows
    d[R.D_ROW_LEN] = row_len_ptr
    d[R.D_BS_LOG2] = int(math.log2(block_size))
    d[R.D_BS_MASK] = block_size - 1
    d[R.D_NBLOCKS] = num_blocks
    d[R.D_TBL_MODE] = 1 if flat_slot_map else 0
    return d


__all__ = ["MODE_OFF", "MODE_SUBSTITUTE", "Interposer", "available", "tier1b_descriptor"]
