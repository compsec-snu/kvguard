"""SGLang adapter. ``translate`` holds the pure translation functions; ``attach`` installs the
run-time wrappers inside the scheduler subprocess, reached through ``hook`` from the injected
``inject/sitecustomize.py`` (``KVGUARD_SGLANG_ATTACH=1`` with ``kvguard/adapters/sglang/inject``
first on ``PYTHONPATH``)."""

from .translate import (
    SUPPORT_BOUNDARY,
    NodeFacts,
    RowFacts,
    build_decode_dispatches,
    build_extend_dispatches,
    build_snapshot,
    geometry,
    pre_step_computed,
    slot_states,
)

__all__ = [
    "SUPPORT_BOUNDARY",
    "NodeFacts",
    "RowFacts",
    "build_decode_dispatches",
    "build_extend_dispatches",
    "build_snapshot",
    "geometry",
    "pre_step_computed",
    "slot_states",
]
