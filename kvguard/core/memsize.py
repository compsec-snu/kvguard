"""The host shadow's memory footprint.

``deep_size`` walks an object graph once (each object counted once by identity) and sums
``sys.getsizeof``, adding a non-owning NumPy view's ``nbytes``. Types, modules, functions and
torch tensors are counted by their header only and never entered, so a framework object reached
from a descriptor stays outside the sum.
"""

from __future__ import annotations

import sys
import types
from array import array
from collections import deque
from collections.abc import Mapping

_ATOMS = (str, bytes, bytearray, int, float, complex, bool, type(None), range, memoryview, array)
_OPAQUE = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
)


def _is_numpy(o: object) -> bool:
    t = type(o)
    return t.__name__ == "ndarray" and t.__module__.startswith("numpy")


def _is_torch(o: object) -> bool:
    return type(o).__module__.startswith("torch")


def deep_size(obj: object, seen: set[int] | None = None) -> int:
    """Bytes held by ``obj`` and everything reachable the walker enters; ``seen`` carries
    identities across calls so shared objects count once."""
    seen = set() if seen is None else seen
    total = 0
    stack: list[object] = [obj]
    while stack:
        o = stack.pop()
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        if isinstance(o, _OPAQUE):
            continue
        try:
            total += sys.getsizeof(o)
        except TypeError:
            continue
        if isinstance(o, _ATOMS) or _is_torch(o):
            continue
        if _is_numpy(o):
            if getattr(o, "base", None) is not None:
                total += int(o.nbytes)
            continue
        if isinstance(o, Mapping):
            stack.extend(o.keys())
            stack.extend(o.values())
            continue
        if isinstance(o, (list, tuple, set, frozenset, deque)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if d is not None:
            stack.append(d)
        for cls in type(o).__mro__:
            for name in getattr(cls, "__slots__", ()):
                if name not in ("__dict__", "__weakref__") and hasattr(o, name):
                    stack.append(getattr(o, name))
    return total
