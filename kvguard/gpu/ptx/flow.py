"""Flow-sensitive refinement: control-flow graph, reaching definitions, launch-time nullness
specialization and the "must" (every path) table-derivation analysis.

``specialize`` folds predicates that test pointer parameters known to be non-null at launch (for
example ``block_table != nullptr`` in a paged launch), prunes the branches they disable and
recomputes reaching definitions until both are stable. ``must_table`` then asks, per memory
instruction, whether every reaching definition chain of its address includes a table load; it is
a greatest fixed point so that loop-carried pointers (``p = p + stride``) stay derived.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .parse import Inst, Kernel, regs
from .provenance import Provenance, Roles, selp_choice

_PRED = re.compile(r"%p\d+$")
_PRED_IN_TEXT = re.compile(r"%p\d+")
_IMM = re.compile(r"-?\d+")
_SETP_NULL = re.compile(r"setp\.(eq|ne)\.(s64|u64|b64)$")
_CHOICE_OPS = ("selp.", "min.", "max.", "slct.")


def is_pred(r: str) -> bool:
    return bool(_PRED.match(r))


def defs_of(inst: Inst) -> list[str]:
    ds = list(inst.dsts)
    if (
        inst.operands
        and is_pred(inst.operands[0].strip())
        and not inst.op.startswith(("bra", "st.", "red."))
    ):
        ds.append(inst.operands[0].strip())
    return ds


def uses_of(inst: Inst) -> list[str]:
    us = [s for s in inst.srcs if not s.startswith("SYM:")]
    if inst.mem is not None:
        us += regs(inst.mem)
    for o in inst.operands[1:]:
        us += _PRED_IN_TEXT.findall(o)
    if inst.pred:
        us.append(inst.pred.lstrip("@!"))
    return us


class CFG:
    """Basic blocks of a kernel under a set of folded predicate values (``pconst``): a branch
    whose predicate is constant has only one successor, so the other branch becomes unreachable.
    ``live`` is the set of reachable instruction indices."""

    def __init__(self, k: Kernel, pconst: dict):
        n = len(k.insts)
        leaders = {0}
        for i, inst in enumerate(k.insts):
            if inst.op.startswith("bra"):
                leaders.add(k.labels[inst.operands[0].strip()])
                leaders.add(i + 1)
            elif inst.op.startswith(("ret", "exit")):
                leaders.add(i + 1)
        leaders |= set(k.labels.values())
        starts = sorted(x for x in leaders if x < n)
        self.block_of: dict[int, int] = {}
        self.blocks: list[tuple[int, int]] = []
        for bi, s in enumerate(starts):
            e = starts[bi + 1] if bi + 1 < len(starts) else n
            self.blocks.append((s, e))
            for i in range(s, e):
                self.block_of[i] = bi
        self.succ: dict[int, list[int]] = defaultdict(list)
        for bi, (_s, e) in enumerate(self.blocks):
            last = k.insts[e - 1]
            taken, fall = True, True
            if last.pred:
                pr = last.pred.lstrip("@!")
                pv = pconst.get((e - 1, pr))
                if pv is not None:
                    v = pv != last.pred.startswith("@!")
                    taken, fall = v, not v
            elif last.op.startswith(("bra", "ret", "exit")):
                fall = False
            if last.op.startswith("bra") and taken:
                tgt = k.labels[last.operands[0].strip()]
                if tgt < n:  # a label at the very end behaves like ret
                    self.succ[bi].append(self.block_of[tgt])
            if fall and e < n:
                self.succ[bi].append(self.block_of[e])
        self.live_blocks: set[int] = set()
        stack = [0]
        while stack:
            b = stack.pop()
            if b in self.live_blocks:
                continue
            self.live_blocks.add(b)
            stack.extend(self.succ[b])
        self.pred: dict[int, list[int]] = defaultdict(list)
        for b in self.live_blocks:
            for s in self.succ[b]:
                self.pred[s].append(b)
        self.live = {i for b in self.live_blocks for i in range(*self.blocks[b])}


def reaching(k: Kernel, cfg: CFG) -> dict[int, dict[str, frozenset[int]]]:
    """``{inst_idx: {reg: definitions reaching the use}}`` over the live blocks."""
    gen: dict[int, dict[str, frozenset[int]]] = {}
    for b in cfg.live_blocks:
        s, e = cfg.blocks[b]
        g: dict[str, frozenset[int]] = {}
        for i in range(s, e):
            for d in defs_of(k.insts[i]):
                g[d] = frozenset([i])
        gen[b] = g
    IN: dict[int, dict] = {b: {} for b in cfg.live_blocks}
    OUT: dict[int, dict] = {b: dict(gen[b]) for b in cfg.live_blocks}
    work = sorted(cfg.live_blocks)
    inwork = set(work)
    while work:
        b = work.pop(0)
        inwork.discard(b)
        new_in: dict[str, frozenset[int]] = defaultdict(frozenset)
        for p in cfg.pred[b]:
            for r, ds in OUT[p].items():
                new_in[r] = new_in[r] | ds
        IN[b] = dict(new_in)
        out = dict(IN[b])
        out.update(gen[b])
        if out != OUT[b]:
            OUT[b] = out
            for s in cfg.succ[b]:
                if s not in inwork:
                    work.append(s)
                    inwork.add(s)
    at: dict[int, dict[str, frozenset[int]]] = {}
    for b in cfg.live_blocks:
        s, e = cfg.blocks[b]
        cur = dict(IN[b])
        for i in range(s, e):
            inst = k.insts[i]
            at[i] = {u: cur.get(u, frozenset()) for u in uses_of(inst)}
            for d in defs_of(inst):
                cur[d] = frozenset([i])
    return at


def specialize(k: Kernel, an: Provenance, nonnull_params: set[str], max_iter: int = 40):
    """Fold predicate uses that are constant once the pointer parameters in ``nonnull_params``
    (tokens ``P:<param>+<off>``) are known non-null at launch. Returns ``(cfg, reaching, pconst)``
    where ``pconst`` maps ``(inst_idx, pred_reg)`` to the folded value."""
    uc: dict = {}
    for _ in range(max_iter):
        cfg = CFG(k, uc)
        rd = reaching(k, cfg)
        live = sorted(cfg.live)
        nn_def: dict[int, bool] = {}

        def reg_nn(i, r, rd=rd, nn_def=nn_def):
            ds = rd.get(i, {}).get(r)
            return bool(ds) and all(nn_def.get(d, False) for d in ds)

        changed = True
        while changed:
            changed = False
            for i in live:
                inst = k.insts[i]
                if not inst.dsts or not inst.dsts[0].startswith("%rd") or nn_def.get(i):
                    continue
                ops = [o.strip() for o in inst.operands]
                v = False
                if inst.mem is not None:
                    toks = an.tok[inst.dsts[0]]
                    v = inst.mem_role == "src" and bool(toks) and toks <= nonnull_params
                elif (
                    inst.op.startswith(("mov.", "cvta.to.global", "cvta.global")) and len(ops) == 2
                ):
                    v = reg_nn(i, ops[1])
                elif inst.op.startswith(("add.s64", "add.u64")) and len(ops) == 3:
                    v = reg_nn(i, ops[1]) or reg_nn(i, ops[2])
                elif inst.op.startswith("selp.") and len(ops) == 4:
                    ch = selp_choice(inst, uc)
                    v = (
                        reg_nn(i, ch)
                        if ch is not None
                        else (reg_nn(i, ops[1]) and reg_nn(i, ops[2]))
                    )
                if v:
                    nn_def[i] = True
                    changed = True

        pval: dict[int, bool] = {}  # predicate definition -> constant value

        def pred_at(i, p, rd=rd, pval=pval):
            ds = rd.get(i, {}).get(p)
            if not ds:
                return None
            vals = {pval.get(d) for d in ds}
            return vals.pop() if len(vals) == 1 and None not in vals else None

        def imm_pred(o):
            return int(o) != 0 if _IMM.fullmatch(o) else None

        changed = True
        while changed:
            changed = False
            for i in live:
                inst = k.insts[i]
                ops = [o.strip() for o in inst.operands]
                if not ops or not is_pred(ops[0]) or i in pval:
                    continue
                val = None
                m = _SETP_NULL.match(inst.op)
                if m and len(ops) == 3:
                    a, b = ops[1], ops[2]
                    reg = a if b == "0" else (b if a == "0" else None)
                    if reg and reg_nn(i, reg):
                        val = m.group(1) == "ne"
                elif inst.op.startswith("mov.pred") and len(ops) == 2:
                    val = imm_pred(ops[1])
                    if val is None and is_pred(ops[1]):
                        val = pred_at(i, ops[1])
                elif inst.op.startswith("not.pred") and len(ops) == 2:
                    x = pred_at(i, ops[1])
                    val = None if x is None else not x
                elif inst.op.startswith(("or.pred", "and.pred", "xor.pred")) and len(ops) == 3:
                    x = pred_at(i, ops[1]) if is_pred(ops[1]) else imm_pred(ops[1])
                    y = pred_at(i, ops[2]) if is_pred(ops[2]) else imm_pred(ops[2])
                    if inst.op.startswith("or"):
                        val = True if (x or y) else (False if (x is False and y is False) else None)
                    elif inst.op.startswith("and"):
                        val = False if (x is False or y is False) else (True if (x and y) else None)
                    elif x is not None and y is not None:
                        val = x != y
                if val is not None:
                    pval[i] = val
                    changed = True

        new_uc = {}
        for i in live:
            inst = k.insts[i]
            preds = []
            if inst.pred:
                preds.append(inst.pred.lstrip("@!"))
            if inst.op.startswith("selp.") and len(inst.operands) == 4:
                preds.append(inst.operands[3].strip())
            for p in preds:
                v = pred_at(i, p)
                if v is not None:
                    new_uc[(i, p)] = v
        if new_uc == uc:
            return cfg, rd, uc
        uc = new_uc
    return cfg, rd, uc


def _is_zero_constant(inst: Inst) -> bool:
    """``mov.u64 %rd, 0`` (or ``.b64``/``.s64``): a zero offset, the start of the region the
    address is built on."""
    return (
        inst.op.startswith("mov.")
        and len(inst.operands) == 2
        and inst.operands[1].strip() in ("0", "0x0", "0x0000000000000000")
    )


def must_table(
    k: Kernel, an: Provenance, roles: Roles, cfg: CFG, rd, pconst, zero_fallback: bool = False
) -> dict[int, bool]:
    """Per live memory instruction: is its address table-derived on every path?

    Greatest fixed point: every register definition starts as table-derived and is refuted. A
    load is derived iff it reads a table; a definition with no register sources (constants,
    parameter loads, special registers) is not derived; a choice (``selp``, ``min``, ``max``,
    ``slct``) is derived only if every choice is; any other arithmetic is derived if any operand
    is. A register is derived at a use only if all of its reaching definitions are.

    With ``zero_fallback``, a zero constant counts as derived: the kernel's fallback for a
    position beyond the row (FlashInfer sets the page offset to 0 and still copies) then reads
    the start of the KV region, a fixed in-pool location, so "table-derived or page 0 on every
    path" is the result, a weaker statement the caller must name as such.
    """
    defs = [i for i in sorted(cfg.live) if k.insts[i].dsts]
    T = dict.fromkeys(defs, True)

    def reg_t(i, r):
        ds = rd.get(i, {}).get(r)
        return bool(ds) and all(T.get(d, False) for d in ds)

    def rule(i):
        inst = k.insts[i]
        ch = selp_choice(inst, pconst)
        if ch is not None:
            return ch.startswith("%") and reg_t(i, ch)
        if inst.mem is not None:
            return an.is_table_load(inst, roles)
        rs = [s for s in inst.srcs if not s.startswith("SYM:")]
        if not rs:
            return zero_fallback and _is_zero_constant(inst)
        if inst.op.startswith(_CHOICE_OPS):
            return all(reg_t(i, r) for r in rs)
        return any(reg_t(i, r) for r in rs)

    changed = True
    while changed:
        changed = False
        for i in defs:
            if T[i] and not rule(i):
                T[i] = False
                changed = True
    return {
        i: any(reg_t(i, r) for r in regs(k.insts[i].mem))
        for i in cfg.live
        if k.insts[i].mem is not None
    }


__all__ = ["CFG", "defs_of", "is_pred", "must_table", "reaching", "specialize", "uses_of"]
