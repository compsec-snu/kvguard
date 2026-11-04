"""vLLM adapter. ``translate`` holds the pure translation functions (unit-testable without vLLM);
``attach`` installs run-time wrappers on a live in-process engine; ``plugin`` attaches inside the
engine-core subprocess through vLLM's ``vllm.general_plugins`` entry point."""

from .translate import (
    SUPPORT_BOUNDARY,
    block_state,
    build_dispatches,
    build_snapshot,
    build_state_dispatch,
    cow_copy,
    dense_table,
    pre_step_computed,
    state_columns,
)

__all__ = [
    "SUPPORT_BOUNDARY",
    "block_state",
    "build_dispatches",
    "build_snapshot",
    "build_state_dispatch",
    "cow_copy",
    "dense_table",
    "pre_step_computed",
    "state_columns",
]
