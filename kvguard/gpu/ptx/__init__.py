"""PTX static provenance analysis: parse a module, compute for every global memory instruction
which kernel parameters and loaded values its address derives from, specialize on launch-time
pointer nullness, and decide per access whether it is table-derived on every path."""

from .analyze import KernelSummary, analyze_kernel, analyze_module, summarize
from .flow import CFG, must_table, reaching, specialize
from .parse import (
    Inst,
    Kernel,
    module_prelude,
    parse_kernel,
    reduce_module,
    split_entries,
    split_funcs,
    top_level_chunks,
)
from .provenance import Access, Provenance, Roles, call_bindings, classify
from .roles import KV, PTR, SCALAR, TABLE, parse_offsets, roles_by_name, roles_by_struct

__all__ = [
    "CFG",
    "KV",
    "PTR",
    "SCALAR",
    "TABLE",
    "Access",
    "Inst",
    "Kernel",
    "KernelSummary",
    "Provenance",
    "Roles",
    "analyze_kernel",
    "analyze_module",
    "call_bindings",
    "classify",
    "module_prelude",
    "must_table",
    "parse_kernel",
    "parse_offsets",
    "reaching",
    "reduce_module",
    "roles_by_name",
    "roles_by_struct",
    "specialize",
    "split_entries",
    "split_funcs",
    "summarize",
    "top_level_chunks",
]
