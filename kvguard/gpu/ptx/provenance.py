"""Address provenance over a parsed PTX kernel.

For every memory instruction the analysis computes the set of *tokens* its address depends on:

- ``P:<param>+<off>``: a value loaded from a kernel parameter (field at byte offset ``off``);
- ``PA:<param>@<off>``: the *address* of a parameter field;
- ``L:<idx>``: the value loaded by instruction ``idx``;
- ``SPACE:shared`` / ``SPACE:local``: a pointer into shared or local memory;
- ``S:tid``: thread-index dependence.

The propagation is flow-insensitive over virtual registers (sound for "may depend"): a register's
tokens are the union over all its definitions. A KV access derives from a KV-pool parameter; it is
table-derived when its address also depends, transitively through loads, on an index-table
parameter. The flow-sensitive refinement lives in :mod:`kvguard.gpu.ptx.flow`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .parse import Inst, Kernel, address_space, regs

_IMM = re.compile(r"-?\d+")
_OFF = re.compile(r"\+\s*(-?\d+)")
_SYMBOL = re.compile(r"(?<![%\w])([A-Za-z_$][A-Za-z0-9_$]*)")
_PARAM_SYM = re.compile(r"\s*([A-Za-z_$][A-Za-z0-9_$]*)")

Roles = dict[str, str]
"""``{"P:<param>+<off>": role}`` with role in ``kv`` (a KV pool pointer), ``table`` (an index table
pointer), ``ptr`` (another pointer) or ``scalar``."""


@dataclass
class Access:
    inst: Inst
    space: str
    """global | generic | shared | local | param | const"""
    kind: str
    """load | store | async_copy | atomic"""
    addr_tokens: frozenset[str] = frozenset()


def selp_choice(inst: Inst, pconst: dict) -> str | None:
    """For a ``selp`` whose predicate is constant at this instruction, the operand it selects.
    ``pconst`` maps ``(inst_idx, pred_reg)`` to the folded value."""
    if not pconst or not inst.op.startswith("selp.") or len(inst.operands) != 4:
        return None
    p = inst.operands[3].strip()
    v = pconst.get((inst.idx, p))
    if v is None:
        return None
    return inst.operands[1].strip() if v else inst.operands[2].strip()


class Provenance:
    """Token propagation to a fixed point over one kernel.

    ``live`` restricts the analysis to the instruction indices a specialization found reachable;
    ``pconst`` carries folded predicate values; ``param_seed`` maps a callee's parameter names to
    the tokens the caller stored into them (interprocedural binding).
    """

    def __init__(
        self,
        kernel: Kernel,
        live: set[int] | None = None,
        pconst: dict | None = None,
        param_seed: dict[str, set[str]] | None = None,
    ):
        self.k = kernel
        self.live = live
        self.pconst = pconst or {}
        self.param_seed = param_seed or {}
        self.callparam_store: dict[str, set[str]] = defaultdict(set)
        self.tok: dict[str, set[str]] = defaultdict(set)
        self.pool: dict[str, set[str]] = defaultdict(set)  # shared / local / callparam pools
        self.site_addr: dict[int, frozenset[str]] = {}
        self._run()

    def _mem_tokens(self, mem: str) -> set[str]:
        toks: set[str] = set()
        for r in regs(mem):
            toks |= self.tok[r]
        for sym in _SYMBOL.findall(mem):
            if sym in self.k.params:
                off = _OFF.search(mem)
                toks.add(f"P:{sym}+{off.group(1) if off else 0}")
            elif sym in self.k.shared_syms:
                toks.add("SPACE:shared")
            elif sym in self.k.local_syms:
                toks.add("SPACE:local")
        return toks

    def _src_tokens(self, inst: Inst) -> set[str]:
        toks: set[str] = set()
        chosen = selp_choice(inst, self.pconst)
        if chosen is not None:
            return set(self.tok[chosen]) if chosen.startswith("%") else set()
        # Parameter-address arithmetic with an immediate keeps a constant offset.
        if inst.op.startswith(("add.", "sub.")) and len(inst.operands) == 3:
            a, b = inst.operands[1].strip(), inst.operands[2].strip()
            imm, reg = None, None
            if _IMM.fullmatch(b):
                imm, reg = int(b), a
            elif _IMM.fullmatch(a) and inst.op.startswith("add."):
                imm, reg = int(a), b
            if imm is not None and reg is not None and reg.startswith("%"):
                if inst.op.startswith("sub."):
                    imm = -imm
                for t in self.tok[reg]:
                    if t.startswith("PA:"):
                        name, off = t[3:].rsplit("@", 1)
                        toks.add(f"PA:{name}@{int(off) + imm}" if off != "?" else t)
                    else:
                        toks.add(t)
                return toks
        for s in inst.srcs:
            if s == "SPECIAL:tid":
                toks.add("S:tid")
            elif s.startswith("SYM:"):
                sym = s[4:]
                if sym in self.k.params:
                    toks.add(f"PA:{sym}@0")
                elif sym in self.k.shared_syms:
                    toks.add("SPACE:shared")
                elif sym in self.k.local_syms:
                    toks.add("SPACE:local")
            else:
                toks |= self.tok[s]
        return toks

    def _run(self) -> None:
        changed = True
        while changed:
            changed = False
            for inst in self.k.insts:
                if self.live is not None and inst.idx not in self.live:
                    continue
                new_dst: set[str] | None = None
                op = inst.op
                if inst.mem is not None:
                    addr = frozenset(self._mem_tokens(inst.mem))
                    if self.site_addr.get(inst.idx) != addr:
                        self.site_addr[inst.idx] = addr
                        changed = True
                    sp = address_space(op)
                    if sp == "generic" and "SPACE:shared" in addr:
                        sp = "shared"
                    if sp == "generic" and "SPACE:local" in addr:
                        sp = "local"
                    if inst.mem_role == "src" and not op.startswith("prefetch"):
                        if sp == "generic" and any(t.startswith("PA:") for t in addr):
                            sp = "param"
                        if sp == "param":
                            new_dst = self._param_load_tokens(inst, addr)
                        elif sp in ("shared", "local"):
                            new_dst = {f"L:{inst.idx}"} | self.pool[sp]
                        else:
                            new_dst = {f"L:{inst.idx}"}
                        if op.startswith("cp.async"):
                            # global -> shared copy: the loaded value lands in the shared pool
                            before = len(self.pool["shared"])
                            self.pool["shared"].add(f"L:{inst.idx}")
                            changed |= len(self.pool["shared"]) != before
                            new_dst = None
                    elif inst.mem_role == "dst":
                        vals: set[str] = set()
                        for s in inst.srcs:
                            vals |= self.tok[s]
                        key = {"shared": "shared", "local": "local", "param": "callparam"}.get(sp)
                        if sp == "param":
                            sym = _PARAM_SYM.match(inst.mem)
                            if sym and sym.group(1) not in self.k.params:
                                store = self.callparam_store[sym.group(1)]
                                before = len(store)
                                store |= vals
                                changed |= len(store) != before
                        if key:
                            before = len(self.pool[key])
                            self.pool[key] |= vals
                            changed |= len(self.pool[key]) != before
                        if op.startswith(("atom.", "red.")) and inst.dsts:
                            new_dst = {f"L:{inst.idx}"}
                elif inst.dsts:
                    new_dst = self._src_tokens(inst)
                    if op.startswith(("cvta.shared", "cvta.local")):
                        new_dst.add("SPACE:" + ("shared" if "shared" in op else "local"))
                if new_dst is not None:
                    for d in inst.dsts:
                        if not new_dst <= self.tok[d]:
                            self.tok[d] |= new_dst
                            changed = True

    def _param_load_tokens(self, inst: Inst, addr: frozenset[str]) -> set[str]:
        """A kernel-parameter load yields the parameter field, or the caller's tokens for a
        callee parameter, or the call-parameter pool."""
        ptoks = {t for t in addr if t.startswith("P:")}
        seeded: set[str] = set()
        for t in list(ptoks):
            sym = t[2:].rsplit("+", 1)[0]
            if sym in self.param_seed:
                ptoks.discard(t)
                seeded |= self.param_seed[sym]
        ptoks |= seeded
        for t in addr:  # a field read through a register that holds a parameter address
            if t.startswith("PA:"):
                m = _OFF.search(inst.mem or "")
                name, base = t[3:].rsplit("@", 1)
                k = int(m.group(1)) if m else 0
                ptoks.add(f"P:{name}+{int(base) + k}" if base != "?" else f"P:{name}+?")
        return ptoks if ptoks else set(self.pool["callparam"])

    def accesses(self) -> list[Access]:
        out: list[Access] = []
        for inst in self.k.insts:
            if inst.mem is None or inst.op.startswith("prefetch"):
                continue
            if self.live is not None and inst.idx not in self.live:
                continue
            addr = self.site_addr.get(inst.idx, frozenset())
            sp = address_space(inst.op)
            if sp == "generic":
                if "SPACE:shared" in addr:
                    sp = "shared"
                elif "SPACE:local" in addr:
                    sp = "local"
                elif any(t.startswith("PA:") for t in addr):
                    sp = "param"
            if inst.op.startswith("cp.async"):
                kind = "async_copy"
            elif inst.op.startswith(("atom.", "red.")):
                kind = "atomic"
            elif inst.mem_role == "dst":
                kind = "store"
            else:
                kind = "load"
            out.append(Access(inst, sp, kind, addr))
        return out

    def load_closure(self, tokens) -> set[int]:
        """All load sites whose values, transitively through addresses, feed ``tokens``."""
        seen: set[int] = set()
        stack = [t for t in tokens if t.startswith("L:")]
        while stack:
            t = stack.pop()
            site = int(t[2:])
            if site in seen:
                continue
            seen.add(site)
            stack.extend(x for x in self.site_addr.get(site, ()) if x.startswith("L:"))
        return seen

    def is_table_load(self, inst: Inst, roles: Roles) -> bool:
        return inst.is_load and any(
            roles.get(t) == "table" for t in self.site_addr.get(inst.idx, ())
        )


def classify(kernel: Kernel, roles: Roles, live=None, pconst=None, param_seed=None):
    """Run the provenance analysis and categorize every global or generic access.

    Categories: ``kv_table`` (KV access whose address depends on a table load), ``kv_table_via_smem``
    (the table entry was staged through shared memory, so the dependence is only "may"),
    ``kv_other_index`` (KV access depending on loads that are not table loads), ``kv_param_only``
    (KV access with no loaded value in its address), ``table_load``, ``non_kv`` and ``unknown_base``
    (an access whose base derives from no pointer parameter at all).
    """
    an = Provenance(kernel, live, pconst, param_seed)
    res = {"kernel": kernel.name, "n_insts": len(kernel.insts), "accesses": []}
    for acc in an.accesses():
        if acc.space in ("shared", "local", "param", "const"):
            continue
        direct = {t for t in acc.addr_tokens if t.startswith("P:")}
        r_direct = {roles.get(t, "unknown_param") for t in direct}
        loads = an.load_closure(acc.addr_tokens)
        table_sites = sorted(
            s for s in loads if any(roles.get(t) == "table" for t in an.site_addr.get(s, ()))
        )
        pooled = any(address_space(an.k.insts[s].op) in ("shared", "local") for s in loads)
        if "kv" in r_direct:
            if table_sites:
                via_smem = pooled and not any(
                    address_space(an.k.insts[s].op) == "global" for s in table_sites
                )
                cat = "kv_table_via_smem" if via_smem else "kv_table"
            elif loads:
                cat = "kv_other_index"
            else:
                cat = "kv_param_only"
        elif "table" in r_direct and not (r_direct - {"table", "scalar"}):
            cat = "table_load"
        elif not any(r in ("ptr", "kv", "table") for r in r_direct):
            cat = "unknown_base"
        else:
            cat = "non_kv"
        res["accesses"].append(
            {
                "idx": acc.inst.idx,
                "op": acc.inst.op,
                "kind": acc.kind,
                "space": acc.space,
                "category": cat,
                "params": sorted(direct),
                "table_sites": table_sites,
                "n_loads": len(loads),
                "pred": acc.inst.pred,
            }
        )
    return an, res


_CALL = re.compile(r"\s*(?:\(([^)]*)\)\s*,\s*)?([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:,\s*\(([^)]*)\))?")


def call_bindings(an: Provenance, kernel: Kernel) -> list[tuple[str, list[set[str]]]]:
    """For each ``call`` in the kernel: ``(callee, [tokens stored into each argument])``."""
    out = []
    for inst in kernel.insts:
        if not inst.op.startswith("call"):
            continue
        parts = inst.text.split(None, 1)
        if len(parts) < 2:
            continue
        m = _CALL.match(parts[1])
        if not m:
            continue
        args = [a.strip() for a in (m.group(3) or "").split(",") if a.strip()]
        out.append((m.group(2), [set(an.callparam_store.get(a, set())) for a in args]))
    return out


__all__ = ["Access", "Provenance", "Roles", "call_bindings", "classify", "selp_choice"]
