"""Driver: analyze every matching entry of a PTX module, interprocedurally, and summarize.

For each entry: build the provenance, specialize on the nullness of the KV and table pointer
parameters (they are non-null in any launch that touches the KV cache), classify every global
access, compute the every-path ("must") table derivation, and follow ``call`` instructions into
``.func`` definitions with the callee's parameters seeded from the caller's stored arguments.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from .flow import must_table, specialize
from .parse import Kernel, parse_kernel, split_entries, top_level_chunks
from .provenance import Provenance, Roles, call_bindings, classify

RolesFor = Callable[[Kernel], Roles]


@dataclass
class KernelSummary:
    kernel: str
    n_insts: int
    global_accesses: int
    categories: dict[str, int]
    kv_accesses: int
    kv_table_derived: int
    """KV accesses whose address may depend on a table load (flow-insensitive)."""
    kv_must_table: int
    """KV accesses whose address depends on a table load on every path."""
    kv_single_table_site: int
    unknown_base: int
    distinct_table_load_sites: int
    kv_kinds: dict[str, int]
    live_insts: int
    folded_preds: int
    callees: int
    accesses: list[dict] = field(default_factory=list)
    kv_must_table_or_zero: int = 0
    """KV accesses table-derived or a zero offset (the kernel's own fallback to page 0) on every
    path: weaker than ``kv_must_table``, named separately wherever it is used as a gate."""

    @property
    def complete(self) -> bool:
        """Every KV access is table-derived on every path and no access has an unknown base."""
        return self.kv_accesses == self.kv_must_table and self.unknown_base == 0

    def to_dict(self, detail: bool = False) -> dict:
        d = asdict(self)
        if not detail:
            d.pop("accesses")
        return d


def analyze_kernel(
    k: Kernel, roles: Roles, funcs: dict[str, str], seed: dict | None = None, depth: int = 0
):
    """Analyze one kernel and, recursively, the functions it calls. Returns ``(accesses, extra)``
    where each access dict carries ``must_table`` and the function it belongs to."""
    nonnull = {t for t, r in roles.items() if r in ("table", "kv")}
    an0 = Provenance(k, param_seed=seed)
    cfg, rd, pconst = specialize(k, an0, nonnull)
    an, res = classify(k, roles, cfg.live, pconst, param_seed=seed)
    must = must_table(k, an, roles, cfg, rd, pconst)
    must_or_zero = must_table(k, an, roles, cfg, rd, pconst, zero_fallback=True)
    for a in res["accesses"]:
        a["must_table"] = bool(must.get(a["idx"]))
        a["must_table_or_zero"] = bool(must_or_zero.get(a["idx"]))
        a["fn"] = k.name
    extra = {"live_insts": len(cfg.live), "folded_preds": len(pconst), "callees": 0}
    if depth < 4:
        for callee, argtoks in call_bindings(an, k):
            if callee not in funcs:
                continue
            ck = parse_kernel(funcs[callee])
            cseed = dict(zip(ck.params, argtoks, strict=False))
            cres, cextra = analyze_kernel(ck, roles, funcs, cseed, depth + 1)
            res["accesses"] += cres["accesses"]
            extra["callees"] += 1 + cextra["callees"]
            extra["live_insts"] += cextra["live_insts"]
            extra["folded_preds"] += cextra["folded_preds"]
    return res, extra


def summarize(res: dict, extra: dict) -> KernelSummary:
    cats = Counter(a["category"] for a in res["accesses"])
    kv = [a for a in res["accesses"] if a["category"].startswith("kv_")]
    tsites = sorted({s for a in kv for s in a["table_sites"]})
    return KernelSummary(
        kernel=res["kernel"],
        n_insts=res["n_insts"],
        global_accesses=len(res["accesses"]),
        categories=dict(cats),
        kv_accesses=len(kv),
        kv_table_derived=sum(a["category"] in ("kv_table", "kv_table_via_smem") for a in kv),
        kv_must_table=sum(a["must_table"] for a in kv),
        kv_must_table_or_zero=sum(a.get("must_table_or_zero", a["must_table"]) for a in kv),
        kv_single_table_site=sum(len(a["table_sites"]) == 1 for a in kv),
        unknown_base=cats.get("unknown_base", 0),
        distinct_table_load_sites=len(tsites),
        kv_kinds=dict(Counter(a["kind"] for a in kv)),
        live_insts=extra["live_insts"],
        folded_preds=extra["folded_preds"],
        callees=extra["callees"],
        accesses=res["accesses"],
    )


def analyze_module(ptx: str, roles_for: RolesFor, entry: str = ".") -> list[KernelSummary]:
    """Analyze every ``.entry`` whose name matches ``entry`` (a regex), with ``roles_for`` giving
    the parameter roles per kernel."""
    rx = re.compile(entry)
    funcs = {n: t for kd, n, t in top_level_chunks(ptx) if kd == "func"}
    out: list[KernelSummary] = []
    for name, text in split_entries(ptx).items():
        if not rx.search(name):
            continue
        k = parse_kernel(text)
        res, extra = analyze_kernel(k, roles_for(k), funcs)
        out.append(summarize(res, extra))
    return out


__all__ = ["KernelSummary", "RolesFor", "analyze_kernel", "analyze_module", "summarize"]
