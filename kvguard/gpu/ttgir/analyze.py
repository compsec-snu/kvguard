"""Must-provenance over Triton TTGIR (MLIR SSA).

For every KV-cache access (``tt.load``, ``tt.store`` or ``ttg.async_copy_global_to_local`` whose
pointer base is a KV argument) the analysis answers whether its pointer derives from a
block-table load on every path, including when the table entries are staged through a shared
buffer (``ttg.local_alloc`` / ``async_copy_global_to_local`` / ``local_load``).

The parser is a line-oriented reader of the textual IR as Triton prints it: one operation per
line, ``scf.for`` / ``scf.if`` / ``scf.while`` regions with ``iter_args`` and ``yield`` turned
into phi edges, ``:`` type annotations stripped. It covers the subset the unified-attention
kernels use; an unknown construct makes values conservatively not-derived.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_VAL = re.compile(r"%[A-Za-z0-9_.$#-]+")
_LOC = re.compile(r"\sloc\(.*$")
_RESULT_OP = re.compile(r"((?:%[^\s=]+(?:,\s*)?)+)\s*=\s*([a-z_]+\.[a-z_.]+)\s*(.*)$")
_PLAIN_OP = re.compile(r"([a-z_]+\.[a-z_.]+)\s*(.*)$")
_RESULT_NAME = re.compile(r"%[A-Za-z0-9_]+(?::\d+)?")
_ARG_PTR = re.compile(r"(%[A-Za-z0-9_]+):\s*!tt\.ptr")
_ARG_SCALAR = re.compile(r"(%[A-Za-z0-9_]+):\s*(?:i32|i64|f32)")
_ITER = re.compile(r"(%[A-Za-z0-9_]+)\s*=\s*(%[A-Za-z0-9_#]+)")
_BLOCK_ARGS = re.compile(r"(%[A-Za-z0-9_]+):")

PASS_THROUGH = (
    "tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout", "tt.reshape", "tt.trans", "arith.select",
)  # fmt: skip
MEMDESC_VIEWS = (
    "ttg.memdesc_index",
    "ttg.memdesc_subslice",
    "ttg.memdesc_trans",
    "ttg.memdesc_reshape",
)
CHOICE = ("arith.select", "arith.minsi", "arith.maxsi", "arith.minui", "arith.maxui")
UNDERIVED_SOURCES = (
    "arith.constant",
    "tt.make_range",
    "tt.get_program_id",
    "tt.get_num_programs",
    "scf.for.iv",
)


@dataclass
class Op:
    idx: int
    results: list[str]
    name: str
    operands: list[str]
    line: str


@dataclass
class TTGIRModule:
    args: list[str]
    ops: list[Op]
    phi: dict[str, list[str]] = field(default_factory=dict)


def _strip_types(rhs: str) -> str:
    """Operands precede the first top-level `` : `` type annotation."""
    rhs = _LOC.sub("", rhs)
    depth = 0
    for i, c in enumerate(rhs):
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == ":" and depth == 0 and i > 0 and rhs[i - 1] == " ":
            return rhs[:i]
    return rhs


def parse_ttgir(text: str) -> TTGIRModule:
    args: list[str] = []
    ops: list[Op] = []
    phi: dict[str, list[str]] = defaultdict(list)
    stack: list[dict] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("tt.func"):
            args = _ARG_PTR.findall(s) + _ARG_SCALAR.findall(s)
            continue
        if s.startswith(("} else", "} do")):
            if s.startswith("} do") and stack:
                stack[-1]["region"] = "after"
            continue
        if not s or s.startswith(("//", "#", "module", "}")):
            if s.startswith("}") and stack:
                stack.pop()
            continue
        if s.startswith("^bb"):
            names = _BLOCK_ARGS.findall(s)
            if stack and stack[-1]["kind"] == "scf.while":
                stack[-1]["after_args"] = names
            continue
        m = _RESULT_OP.match(s)
        if m:
            res_txt, name, rhs = m.groups()
            results: list[str] = []
            for r in _RESULT_NAME.findall(res_txt):
                if ":" in r:
                    base, n = r.split(":")
                    results += [f"{base}#{i}" for i in range(int(n))]
                else:
                    results.append(r)
            if name == "scf.for":
                iv_m = re.match(r"(%[A-Za-z0-9_]+)\s*=", rhs)
                iv = iv_m.group(1) if iv_m else f"%iv{len(ops)}"
                iters = _ITER.findall(rhs.split("iter_args", 1)[1]) if "iter_args" in rhs else []
                for ia, init in iters:
                    phi[ia].append(init)
                for r, (ia, _) in zip(results, iters, strict=False):
                    phi[r].append(ia)  # a loop result is the last iteration's value
                ops.append(Op(len(ops), [iv], "scf.for.iv", [], s))
                stack.append(
                    {"kind": "scf.for", "iters": [ia for ia, _ in iters], "results": results}
                )
                continue
            if name == "scf.if":
                stack.append({"kind": "scf.if", "results": results})
                continue
            if name == "scf.while":
                inits = _ITER.findall(rhs.split(")")[0])
                for ba, init in inits:
                    phi[ba].append(init)
                stack.append(
                    {
                        "kind": "scf.while",
                        "before_args": [b for b, _ in inits],
                        "results": results,
                        "region": "before",
                    }
                )
                continue
            ops.append(Op(len(ops), results, name, _VAL.findall(_strip_types(rhs)), s))
            continue
        if s.startswith("scf.yield"):
            vals = _VAL.findall(_strip_types(s[len("scf.yield") :]))
            if stack:
                fr = stack[-1]
                if fr["kind"] == "scf.for":
                    for ia, v in zip(fr["iters"], vals, strict=False):
                        phi[ia].append(v)
                elif fr["kind"] == "scf.if":
                    for r, v in zip(fr["results"], vals, strict=False):
                        phi[r].append(v)
                elif fr["kind"] == "scf.while":
                    for ba, v in zip(fr["before_args"], vals, strict=False):
                        phi[ba].append(v)
            continue
        if s.startswith("scf.condition"):
            vals = _VAL.findall(_strip_types(s.split(")", 1)[1]))
            if stack:
                fr = stack[-1]
                fr["cond_vals"] = vals
                for r, v in zip(fr["results"], vals, strict=False):
                    phi[r].append(v)
            continue
        mm = _PLAIN_OP.match(s)
        if mm:
            ops.append(Op(len(ops), [], mm.group(1), _VAL.findall(_strip_types(mm.group(2))), s))
    return TTGIRModule(args, ops, dict(phi))


@dataclass
class TTGIRResult:
    accesses: list[dict]

    @property
    def kv_accesses(self) -> int:
        return sum(a["kind"] == "kv" for a in self.accesses)

    @property
    def kv_must_table(self) -> int:
        return sum(a["kind"] == "kv" and a["must_table"] for a in self.accesses)

    @property
    def table_loads(self) -> int:
        return sum(a["kind"] == "table_load" for a in self.accesses)

    @property
    def complete(self) -> bool:
        return self.kv_accesses == self.kv_must_table

    def summary(self) -> dict:
        return {
            "kv_accesses": self.kv_accesses,
            "kv_must_table": self.kv_must_table,
            "table_loads": self.table_loads,
        }


def analyze_ttgir(mod: TTGIRModule, kv_args: list[str], table_args: list[str]) -> TTGIRResult:
    args, ops, phi = mod.args, mod.ops, mod.phi
    defn: dict[str, Op] = {}
    for op in ops:
        for r in op.results:
            defn[r] = op

    # Pointer base: which function-argument pointers a pointer value may derive from.
    base: dict[str, set[str]] = defaultdict(set)
    for a in args:
        base[a].add(a)
    changed = True
    while changed:
        changed = False
        for op in ops:
            if op.name == "tt.addptr" or op.name in PASS_THROUGH:
                src = op.operands[:1] if op.name == "tt.addptr" else op.operands
                new = set().union(*(base[o] for o in src)) if src else set()
                for r in op.results:
                    if not new <= base[r]:
                        base[r] |= new
                        changed = True
        for v, ins in phi.items():
            new = set().union(*(base[i] for i in ins))
            if not new <= base[v]:
                base[v] |= new
                changed = True

    # Shared buffer roots and their writers.
    root: dict[str, str] = {}
    for op in ops:
        if op.name == "ttg.local_alloc":
            root[op.results[0]] = op.results[0]
    changed = True
    while changed:
        changed = False
        for op in ops:
            if (
                op.name in MEMDESC_VIEWS
                and op.operands
                and op.operands[0] in root
                and op.results[0] not in root
            ):
                root[op.results[0]] = root[op.operands[0]]
                changed = True
        for v, ins in phi.items():
            rs = {root.get(i) for i in ins}
            if len(rs) == 1 and None not in rs and v not in root:
                root[v] = rs.pop()
                changed = True
    writers: dict[str | None, list[tuple[str, str]]] = defaultdict(list)
    for op in ops:
        if op.name == "ttg.async_copy_global_to_local":
            writers[root.get(op.operands[1])].append(("ptr", op.operands[0]))
        elif op.name == "ttg.local_store":
            writers[root.get(op.operands[1])].append(("val", op.operands[0]))
        elif op.name == "ttg.local_alloc" and op.operands:
            writers[op.results[0]].append(("val", op.operands[0]))

    table_set, kv_set = set(table_args), set(kv_args)

    def is_table_ptr(p: str) -> bool:
        return bool(base[p]) and base[p] <= table_set

    # Greatest fixed point of "table-derived on every path".
    vals = set(defn) | set(phi)
    T = dict.fromkeys(vals, True)

    def t(v: str) -> bool:
        if v in args:
            return False
        return T.get(v, False)

    def rule(v: str) -> bool:
        if v in phi and v not in defn:
            ins = phi[v]
            return bool(ins) and all(t(i) for i in ins)
        op = defn[v]
        if op.name in UNDERIVED_SOURCES:
            return False
        if op.name == "tt.load":
            return is_table_ptr(op.operands[0])
        if op.name == "ttg.local_load":
            ws = writers.get(root.get(op.operands[0]), [])
            return bool(ws) and all(is_table_ptr(x) if k == "ptr" else t(x) for k, x in ws)
        if op.name == "ttg.local_alloc":
            return False
        if op.name == "arith.select":
            return len(op.operands) >= 3 and t(op.operands[1]) and t(op.operands[2])
        if op.name in CHOICE:
            return all(t(o) for o in op.operands)
        return any(t(o) for o in op.operands)

    changed = True
    while changed:
        changed = False
        for v in vals:
            if T[v] and not rule(v):
                T[v] = False
                changed = True

    out: list[dict] = []
    for op in ops:
        if op.name in ("tt.load", "ttg.async_copy_global_to_local", "tt.store") and op.operands:
            p = op.operands[0]
            b = base[p]
            if b & kv_set:
                out.append({"op": op.name, "line": op.line[:120], "must_table": t(p), "kind": "kv"})
            elif b and b <= table_set:
                out.append({"op": op.name, "line": op.line[:120], "kind": "table_load"})
    return TTGIRResult(out)


def analyze_ttgir_text(text: str, kv_args: list[str], table_args: list[str]) -> TTGIRResult:
    return analyze_ttgir(parse_ttgir(text), kv_args, table_args)


__all__ = ["Op", "TTGIRModule", "TTGIRResult", "analyze_ttgir", "analyze_ttgir_text", "parse_ttgir"]
