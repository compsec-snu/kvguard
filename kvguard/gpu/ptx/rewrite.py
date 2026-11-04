"""Automatic PTX instrumentation driven by the provenance analysis.

Two kinds of checks are inserted into an unmodified kernel:

* **table-site checks**, right after every load from an index-table parameter: the entry's row
  and column must lie inside the row's valid length ("Tier 1b" when ``bounds_only``), and with
  full contents (Tier 1) the loaded block must be owned by the row's request (or shared for a
  read), allocated, and filled far enough for the row's tokens;
* **KV-site checks** (Tier 2), right before every KV access: the address is decoded into
  (block, slot); the block must be inside the pool, a read must target an initialized slot, a
  write must not target a published block, and a write marks its slot initialized;
* **use-guarded table checks** (``use_guarded``): the table-site bounds predicate is remembered
  in a per-site flag rather than reported where it is computed, and every KV access whose address
  derives from that table load reports it when the access executes, so a kernel that loads table
  entries for a whole tile and skips the copies beyond the sequence stays silent. Exact only when
  no intermediate value of the address chain can outlive a re-execution of its table load; a site
  that fails that static test keeps the immediate check.

Checks read a descriptor in a module ``.const`` array (no kernel ABI change) and are gated by an
enable word (``D_ENABLE``: ``EN_TABLE``, ``EN_KV``), so one binary runs with checks on or off and a
twin launched before any descriptor was published touches no memory. Violations increment a device
counter and record the first one. One thread per CTA checks a CTA-uniform table load; the row is
recovered from the address term that depends on the table parameter; ``.maxnreg`` is pinned to the
original register budget.
"""

from __future__ import annotations

import re

from .flow import specialize, uses_of
from .parse import (
    Kernel,
    module_prelude,
    parse_kernel,
    regs,
    split_entries,
    split_funcs,
    strip_comments,
)
from .provenance import Provenance, Roles, call_bindings, classify

# Descriptor layout (u64 slots in __kvg_desc).
D_ENABLE, D_COUNTER, D_STATE, D_INIT = 0, 1, 2, 3
D_BLOCK_SHIFT, D_SLOT_SHIFT, D_BS_MASK, D_NBLOCKS, D_BS_LOG2 = 4, 5, 6, 7, 8
D_TBL_ROW_BYTES, D_TBL_ENTRY_LOG2, D_TBL_NROWS, D_ROW_LEN, D_ROW_OWNER, D_TBL_MODE = (
    9,
    10,
    11,
    12,
    13,
    14,
)
D_RECORD = 15  # pointer to u64[4]: first-violation record (site id, block, slot or row, kind)
D_FILLED = 16  # u32 per block: number of initialized slots (prefix)
N_DESC = 17

EN_TABLE, EN_KV = 1, 2
ST_ALLOC, ST_PUBLISHED, ST_SHARED = 1, 2, 4

# Violation kinds written to the record.
KIND_KV_READ, KIND_KV_WRITE, KIND_TABLE_READ, KIND_TABLE_WRITE, KIND_BOUNDS = 1, 2, 3, 4, 5

REGS = ".reg .b64 %kgd<24>;\n.reg .b32 %kgr<16>;\n.reg .pred %kgp<13>;\n.reg .b16 %kgh<4>;\n.reg .b32 %kgv<2>;\n"

# Trace mode (dynamic cross-check of the analysis): every global or generic access tests its
# address against two KV ranges and one table range in a second constant array ``__kvg_trace``
# and sets its site's bit in a device bitmap when it hits, so a run with real arguments says
# which sites touch the KV pool or the table.
T_KV0_LO, T_KV0_HI, T_KV1_LO, T_KV1_HI, T_TBL_LO, T_TBL_HI, T_BITMAP_KV, T_BITMAP_TBL = range(8)
N_TRACE = 8
FLAG_REGS = ".reg .b32 %kgm<1>;\n"  # use-guarded mode: one "beyond" bit per flagged table site
MAX_FLAG_SITES = 32

_ADDR = re.compile(r"\s*(%[A-Za-z0-9_]+)\s*(?:\+\s*(-?\d+))?\s*$")
_LABEL = re.compile(r"^([$A-Za-z0-9_]+):")


def _addr(mem: str):
    m = _ADDR.match(mem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2) or 0)


def _param_ref(tok: str) -> str:
    name, off = tok[2:].rsplit("+", 1)
    return f"[{name}+{off}]" if off != "0" else f"[{name}]"


def _desc(i: int) -> str:
    return f"[__kvg_desc+{8 * i}]"


# Set by ``instrument`` for the duration of one call: when true, descriptor slots are read once
# per thread at kernel entry into ``%kgc<slot>`` registers and the checks read those registers,
# so no check reads the constant bank.
_DESC_IN_REGS = False
_DESC_SLOTS_USED: set[int] = set()


def _ld_desc(dst: str, slot: int) -> str:
    """Read descriptor slot ``slot`` into the 64-bit register ``dst``."""
    if _DESC_IN_REGS:
        _DESC_SLOTS_USED.add(slot)
        return f"mov.b64 {dst}, %kgc{slot};"
    return f"ld.const.u64 {dst}, {_desc(slot)};"


class UnsupportedCallee(ValueError):
    """A non-inlined callee holds a check site whose pointer parameter (the KV pool or the table)
    no call argument binds unambiguously, so the check cannot read the pointer there."""


# Set per function during one ``instrument`` call: ``None`` while the entry is emitted; for a
# callee, how its parameters reach the entry's parameter fields (see ``_param_bindings``).
_PARAM_PTR: dict | None = None


def _param_bindings(fn: str, seed: dict[str, set[str]]) -> dict:
    """How a callee reaches the entry's pointer parameters: a callee parameter that received the
    pointer value itself (``P:<param>+<off>``, the caller loaded the field and passed it) or the
    address of the entry's parameter block (``PA:<param>@<base>``, the caller passed
    ``cvta.param`` of its parameter struct, as nvcc does for a ``const Params&``). A parameter
    bound to several values binds nothing."""
    direct: dict[str, str] = {}
    block: dict[str, tuple[str, int]] = {}
    for cp, toks in seed.items():
        if len(toks) != 1:
            continue
        (t,) = toks
        if t.startswith("P:") and not t.endswith("+?"):
            direct.setdefault(t, cp)
        elif t.startswith("PA:"):
            name, base = t[3:].rsplit("@", 1)
            if base != "?":
                block.setdefault(name, (cp, int(base)))
    return {"fn": fn, "P": direct, "PA": block}


def _ld_param(dst: str, tok: str) -> list[str]:
    """Load the pointer in the entry's parameter field ``tok`` into ``dst``: from the parameter
    space in the entry; in a callee, from the callee parameter that carries the pointer value, or
    through the one that carries the address of the entry's parameter block (a generic load, as
    the callee's own field reads)."""
    if _PARAM_PTR is None:
        return [f"ld.param.u64 {dst}, {_param_ref(tok)};"]
    cp = _PARAM_PTR["P"].get(tok)
    if cp is not None:
        return [f"ld.param.u64 {dst}, [{cp}];"]
    name, off = tok[2:].rsplit("+", 1)
    if name in _PARAM_PTR["PA"] and off != "?":
        cp, base = _PARAM_PTR["PA"][name]
        return [f"ld.param.u64 {dst}, [{cp}];", f"ld.u64 {dst}, [{dst}+{int(off) - base}];"]
    raise UnsupportedCallee(f"{_PARAM_PTR['fn']}: no call argument binds {tok}")


# When set, a Tier 1b table site whose row-base register has a single, unpredicated definition
# computes its row index and row length once, right after that definition, instead of dividing
# and loading at every table load.
_HOIST_ROWS = True


def row_hoist_block(group: int, xb: str, table_tok: str) -> list[str]:
    """After the definition of row-base register ``xb``: the table row it addresses and that
    row's length in tokens (0 when the row lies outside the table, so every column of it is
    beyond bounds). Reads nothing when the descriptor is zero (row 0 is not below 0 rows)."""
    return [
        *_ld_param("%kgd2", table_tok),
        f"sub.s64 %kgd3, {xb}, %kgd2;",
        "cvt.u32.u64 %kgr9, %kgd3;",
        _ld_desc("%kgd4", D_TBL_ROW_BYTES),
        "cvt.u32.u64 %kgr10, %kgd4;",
        f"div.u32 %kgrow{group}, %kgr9, %kgr10;",
        f"cvt.u64.u32 %kgd5, %kgrow{group};",
        _ld_desc("%kgd8", D_TBL_NROWS),
        "setp.lt.u64 %kgp4, %kgd5, %kgd8;",
        _ld_desc("%kgd9", D_ROW_LEN),
        "shl.b64 %kgd10, %kgd5, 2;",
        "add.s64 %kgd11, %kgd9, %kgd10;",
        f"mov.u32 %kgrlen{group}, 0;",
        f"@%kgp4 ld.global.u32 %kgrlen{group}, [%kgd11];",
    ]


# When set, every check starts by testing its enable bit in the descriptor and folds it into the
# lane predicate, so a twin whose descriptor is zero performs no memory operation.
_ENABLE_GATE = True


def _guard(inst, extra_src_size: str | None, enable_bit: int | None = None) -> list[str]:
    """Leave the lane predicate (the instruction's own predicate, a non-zero cp.async source
    size, and the descriptor's enable bit) in %kgp1."""
    out = []
    if inst.pred:
        p = inst.pred.lstrip("@!")
        out.append(
            f"mov.pred %kgp1, {p};" if not inst.pred.startswith("@!") else f"not.pred %kgp1, {p};"
        )
    else:
        out.append("mov.pred %kgp1, -1;")
    if extra_src_size is not None:
        out.append(f"setp.ne.u32 %kgp2, {extra_src_size}, 0;")
        out.append("and.pred %kgp1, %kgp1, %kgp2;")
    if enable_bit is not None and _ENABLE_GATE:
        out += [
            _ld_desc("%kgd0", D_ENABLE),
            f"and.b64 %kgd0, %kgd0, {enable_bit};",
            "setp.ne.u64 %kgp2, %kgd0, 0;",
            "and.pred %kgp1, %kgp1, %kgp2;",
        ]
    return out


def _violation(site: int, kind: int, a: str, b: str) -> list[str]:
    return [
        f"mov.u64 %kgd22, {site};",
        f"mov.u64 %kgd23, {kind};",
        f"@%kgp9 ld.const.u64 %kgd20, {_desc(D_COUNTER)};",
        "@%kgp9 atom.global.add.u32 %kgr15, [%kgd20], 1;",
        "@%kgp9 setp.eq.u32 %kgp10, %kgr15, 0;",
        "and.pred %kgp10, %kgp10, %kgp9;",
        f"@%kgp10 ld.const.u64 %kgd21, {_desc(D_RECORD)};",
        f"@%kgp10 st.global.v2.u64 [%kgd21], {{%kgd22, {a}}};",
        f"@%kgp10 st.global.v2.u64 [%kgd21+16], {{{b}, %kgd23}};",
    ]


def kv_site_check(site_id, inst, kv_tok, is_write, src_size, read_alloc=False) -> list[str]:
    """Per-access check (Tier 2). Lifetime and ownership are established once per table entry at
    the table site, so a read needs only pool bounds and initialization."""
    reg, imm = _addr(inst.mem)
    c = _guard(inst, src_size, EN_KV)
    c += [
        f"add.s64 %kgd1, {reg}, {imm};",
        *_ld_param("%kgd2", kv_tok),
        "sub.s64 %kgd3, %kgd1, %kgd2;",
        _ld_desc("%kgd4", D_BLOCK_SHIFT),
        "cvt.u32.u64 %kgr1, %kgd4;",
        "shr.u64 %kgd5, %kgd3, %kgr1;",  # block
        _ld_desc("%kgd4", D_SLOT_SHIFT),
        "cvt.u32.u64 %kgr2, %kgd4;",
        "shr.u64 %kgd6, %kgd3, %kgr2;",
        _ld_desc("%kgd7", D_BS_MASK),
        "and.b64 %kgd6, %kgd6, %kgd7;",  # slot in block
        _ld_desc("%kgd8", D_NBLOCKS),
        "setp.ge.u64 %kgp3, %kgd5, %kgd8;",  # outside the KV pool
        "not.pred %kgp4, %kgp3;",
        "and.pred %kgp4, %kgp4, %kgp1;",
        _ld_desc("%kgd11", D_BS_LOG2),
        "cvt.u32.u64 %kgr5, %kgd11;",
        "shl.b64 %kgd12, %kgd5, %kgr5;",
        "add.s64 %kgd12, %kgd12, %kgd6;",
        _ld_desc("%kgd13", D_INIT),
        "add.s64 %kgd12, %kgd13, %kgd12;",
    ]
    if is_write:
        c += [
            _ld_desc("%kgd9", D_STATE),
            "shl.b64 %kgd10, %kgd5, 2;",
            "add.s64 %kgd10, %kgd9, %kgd10;",
            "mov.u32 %kgr3, 0;",
            "@%kgp4 ld.global.u32 %kgr3, [%kgd10];",
            "and.b32 %kgr4, %kgr3, 1;",
            "setp.eq.u32 %kgp5, %kgr4, 0;",  # not allocated
            "and.b32 %kgr6, %kgr3, 2;",
            "setp.ne.u32 %kgp6, %kgr6, 0;",  # write to published
            "not.pred %kgp7, %kgp5;",
            "and.pred %kgp7, %kgp7, %kgp4;",
            "mov.u16 %kgh1, 1;",
            "@%kgp7 st.global.u8 [%kgd12], %kgh1;",
            "or.pred %kgp9, %kgp3, %kgp5;",
            "or.pred %kgp9, %kgp9, %kgp6;",
        ]
    else:
        c += [
            "mov.u16 %kgh0, 1;",
            "@%kgp4 ld.global.u8 %kgh0, [%kgd12];",
            "setp.eq.u16 %kgp6, %kgh0, 0;",  # uninitialized read
            "or.pred %kgp9, %kgp3, %kgp6;",
        ]
        if read_alloc:
            # Fallback when no table site can be instrumented: the block must be allocated to
            # some request of this dispatch (batch-level check).
            c += [
                _ld_desc("%kgd9", D_STATE),
                "shl.b64 %kgd10, %kgd5, 2;",
                "add.s64 %kgd10, %kgd9, %kgd10;",
                "mov.u32 %kgr3, 1;",
                "@%kgp4 ld.global.u32 %kgr3, [%kgd10];",
                "and.b32 %kgr4, %kgr3, 1;",
                "setp.eq.u32 %kgp5, %kgr4, 0;",
                "or.pred %kgp9, %kgp9, %kgp5;",
            ]
    c += ["and.pred %kgp9, %kgp9, %kgp1;"]
    c += _violation(site_id, KIND_KV_WRITE if is_write else KIND_KV_READ, "%kgd5", "%kgd6")
    return c


def table_site_check(
    site_id,
    inst,
    table_tok,
    is_write,
    split=None,
    uniform=False,
    val=None,
    src_size=None,
    reload_bytes=0,
    bounds_only=False,
    flag_bit: int | None = None,
    hoisted: int | None = None,
) -> list[str]:
    """After a table load: row and column bounds (C4), and with full contents the block's
    lifetime and ownership. ``split=(row_base_reg, col_off_reg)`` recovers the row from the
    address term that depends on the table parameter and the column from the rest. With
    ``flag_bit`` (use-guarded mode) the bounds predicate is stored instead of reported. With
    ``hoisted`` (a row group, bounds only) the row and its length come from the registers
    ``row_hoist_block`` filled after the row base was defined."""
    reg, imm = _addr(inst.mem)
    val = val or inst.dsts[0]
    c = _guard(inst, src_size, EN_TABLE)
    if hoisted is not None:
        _xb, ycol = split
        if uniform:
            c += [
                "mov.u32 %kgr0, %tid.x;",
                "mov.u32 %kgr1, %tid.y;",
                "or.b32 %kgr0, %kgr0, %kgr1;",
                "mov.u32 %kgr1, %tid.z;",
                "or.b32 %kgr0, %kgr0, %kgr1;",
                "setp.eq.u32 %kgp2, %kgr0, 0;",
                "and.pred %kgp1, %kgp1, %kgp2;",
            ]
        c += [
            f"add.s64 %kgd3, {ycol}, {imm};",
            "cvt.u32.u64 %kgr12, %kgd3;",
            _ld_desc("%kgd7", D_TBL_ENTRY_LOG2),
            "cvt.u32.u64 %kgr1, %kgd7;",
            "shr.u32 %kgr12, %kgr12, %kgr1;",  # column
            f"cvt.u64.u32 %kgd5, %kgrow{hoisted};",
            "cvt.u64.u32 %kgd6, %kgr12;",
            f"cvt.u64.u32 %kgd12, %kgrlen{hoisted};",  # row length, 0 outside the table
            _ld_desc("%kgd16", D_BS_LOG2),
            "cvt.u32.u64 %kgr4, %kgd16;",
            _ld_desc("%kgd15", D_TBL_MODE),
            "setp.ne.u64 %kgp6, %kgd15, 0;",
            "selp.b32 %kgr13, 0, %kgr4, %kgp6;",
            "shl.b64 %kgd7, %kgd6, %kgr13;",  # first token covered by this entry
            "setp.ge.u64 %kgp9, %kgd7, %kgd12;",  # beyond the row (or the row beyond the table)
            "and.pred %kgp9, %kgp9, %kgp1;",
        ]
        if flag_bit is not None:
            return c + flag_set(flag_bit)
        return c + _violation(site_id, KIND_BOUNDS, "%kgd5", "%kgd6")
    if reload_bytes and not bounds_only:
        # An asynchronous copy of a table entry: re-read it synchronously (the table is immutable
        # while the kernel runs) so the entry value is available to check.
        c += [
            "mov.u32 %kgv0, 0;",
            f"@%kgp1 ld.global.{'s32' if reload_bytes == 4 else 's64'} {'%kgv0' if reload_bytes == 4 else '%kgd0'}, [{reg}+{imm}];",
        ]
        if reload_bytes == 8:
            val = "%kgd0"
    if uniform:
        # The table entry is the same for every thread of the CTA: one thread checks it.
        c += [
            "mov.u32 %kgr0, %tid.x;",
            "mov.u32 %kgr1, %tid.y;",
            "or.b32 %kgr0, %kgr0, %kgr1;",
            "mov.u32 %kgr1, %tid.z;",
            "or.b32 %kgr0, %kgr0, %kgr1;",
            "setp.eq.u32 %kgp2, %kgr0, 0;",
            "and.pred %kgp1, %kgp1, %kgp2;",
        ]
    c += [
        *_ld_param("%kgd2", table_tok),
        _ld_desc("%kgd4", D_TBL_ROW_BYTES),
        "cvt.u32.u64 %kgr10, %kgd4;",
    ]
    if split:
        xb, ycol = split
        c += [
            f"sub.s64 %kgd3, {xb}, %kgd2;",
            "cvt.u32.u64 %kgr9, %kgd3;",
            "div.u32 %kgr11, %kgr9, %kgr10;",  # row from the row-base term only
            f"add.s64 %kgd3, {ycol}, {imm};",
            "cvt.u32.u64 %kgr12, %kgd3;",
        ]
    else:
        c += [
            f"add.s64 %kgd1, {reg}, {imm};",
            "sub.s64 %kgd3, %kgd1, %kgd2;",  # byte offset in table
            "cvt.u32.u64 %kgr9, %kgd3;",
            "div.u32 %kgr11, %kgr9, %kgr10;",  # row
            "mul.lo.u32 %kgr12, %kgr11, %kgr10;",
            "sub.u32 %kgr12, %kgr9, %kgr12;",
        ]
    c += [
        _ld_desc("%kgd7", D_TBL_ENTRY_LOG2),
        "cvt.u32.u64 %kgr1, %kgd7;",
        "shr.u32 %kgr12, %kgr12, %kgr1;",  # column
        "cvt.u64.u32 %kgd5, %kgr11;",
        "cvt.u64.u32 %kgd6, %kgr12;",
        _ld_desc("%kgd8", D_TBL_NROWS),
        "setp.ge.u64 %kgp3, %kgd5, %kgd8;",  # row outside table
        "not.pred %kgp4, %kgp3;",
        "and.pred %kgp4, %kgp4, %kgp1;",
        _ld_desc("%kgd9", D_ROW_LEN),
        "shl.b64 %kgd10, %kgd5, 2;",
        "add.s64 %kgd11, %kgd9, %kgd10;",
        "mov.u32 %kgr2, 0;",
        "@%kgp4 ld.global.u32 %kgr2, [%kgd11];",
        "cvt.u64.u32 %kgd12, %kgr2;",  # row length in tokens
        _ld_desc("%kgd16", D_BS_LOG2),
        "cvt.u32.u64 %kgr4, %kgd16;",
        _ld_desc("%kgd15", D_TBL_MODE),
        "setp.ne.u64 %kgp6, %kgd15, 0;",
        "selp.b32 %kgr13, 0, %kgr4, %kgp6;",  # flat slot map: one entry per token
        "shl.b64 %kgd7, %kgd6, %kgr13;",  # first token covered by this entry
        "setp.ge.u64 %kgp5, %kgd7, %kgd12;",  # entry beyond row length (C4)
    ]
    if bounds_only:
        c += ["or.pred %kgp9, %kgp3, %kgp5;", "and.pred %kgp9, %kgp9, %kgp1;"]
        if flag_bit is not None:
            return c + flag_set(flag_bit)
        c += _violation(site_id, KIND_BOUNDS, "%kgd5", "%kgd6")
        return c
    c += [
        _ld_desc("%kgd13", D_ROW_OWNER),
        "add.s64 %kgd13, %kgd13, %kgd10;",
        "mov.u32 %kgr3, 0;",
        "@%kgp4 ld.global.u32 %kgr3, [%kgd13];",  # expected owner
        f"mov.b64 %kgd14, {val};" if val.startswith("%rd") else f"cvt.s64.s32 %kgd14, {val};",
        "setp.lt.s64 %kgp7, %kgd14, 0;",
        "and.pred %kgp7, %kgp7, %kgp6;",  # padding slot
        "shr.s64 %kgd17, %kgd14, %kgr4;",
        "selp.b64 %kgd14, %kgd17, %kgd14, %kgp6;",  # block id
        _ld_desc("%kgd18", D_NBLOCKS),
        "setp.ge.u64 %kgp8, %kgd14, %kgd18;",  # block outside the pool
        "not.pred %kgp2, %kgp8;",
        "and.pred %kgp2, %kgp2, %kgp4;",
        "not.pred %kgp11, %kgp7;",
        "and.pred %kgp2, %kgp2, %kgp11;",
        _ld_desc("%kgd19", D_STATE),
        "shl.b64 %kgd17, %kgd14, 2;",
        "add.s64 %kgd19, %kgd19, %kgd17;",
        "mov.u32 %kgr5, 0;",
        "@%kgp2 ld.global.u32 %kgr5, [%kgd19];",
        "shr.u32 %kgr6, %kgr5, 8;",  # owner
        "setp.ne.u32 %kgp10, %kgr6, %kgr3;",
    ]
    if is_write:
        c += [
            "and.b32 %kgr7, %kgr5, 2;",
            "setp.ne.u32 %kgp11, %kgr7, 0;",
            "or.pred %kgp10, %kgp10, %kgp11;",
        ]
    else:
        c += [
            "and.b32 %kgr7, %kgr5, 4;",
            "setp.ne.u32 %kgp11, %kgr7, 0;",  # shared: any reader allowed
            "not.pred %kgp11, %kgp11;",
            "and.pred %kgp10, %kgp10, %kgp11;",
            # Lifted initialization: the slots this row may read from the block must be filled.
            "sub.s64 %kgd20, %kgd12, %kgd7;",  # tokens of the row at/after this block
            "mov.u32 %kgr14, 1;",
            "shl.b32 %kgr14, %kgr14, %kgr4;",
            "cvt.u64.u32 %kgd21, %kgr14;",
            "min.s64 %kgd20, %kgd20, %kgd21;",  # need = min(bs, len - c*bs)
            _ld_desc("%kgd22", D_FILLED),
            "add.s64 %kgd22, %kgd22, %kgd17;",
            "mov.u32 %kgr15, 0;",
            "@%kgp2 ld.global.u32 %kgr15, [%kgd22];",
            "cvt.u64.u32 %kgd23, %kgr15;",
            "setp.lt.s64 %kgp11, %kgd23, %kgd20;",
            "and.pred %kgp11, %kgp11, %kgp2;",
            "not.pred %kgp12, %kgp6;",
            "and.pred %kgp11, %kgp11, %kgp12;",
            "or.pred %kgp10, %kgp10, %kgp11;",
        ]
    c += [
        "and.b32 %kgr8, %kgr5, 1;",
        "setp.eq.u32 %kgp11, %kgr8, 0;",
        "or.pred %kgp10, %kgp10, %kgp11;",
        "and.pred %kgp10, %kgp10, %kgp2;",
        "or.pred %kgp9, %kgp3, %kgp5;",
        "or.pred %kgp9, %kgp9, %kgp8;",
        "or.pred %kgp9, %kgp9, %kgp10;",
        "not.pred %kgp11, %kgp7;",
        "and.pred %kgp9, %kgp9, %kgp11;",
        "and.pred %kgp9, %kgp9, %kgp1;",
    ]
    c += _violation(site_id, KIND_TABLE_WRITE if is_write else KIND_TABLE_READ, "%kgd5", "%kgd6")
    return c


def trace_check(site_id: int, inst, src_size: str | None) -> list[str]:
    """Before any global or generic access: set bit ``site_id`` of the KV bitmap when the address
    lies in one of the two KV ranges, and of the table bitmap when it lies in the table range."""
    reg, imm = _addr(inst.mem)
    word, bit = site_id // 32, 1 << (site_id % 32)
    c = _guard(inst, src_size)  # the access's own predicate and a non-zero cp.async size
    c.append(f"add.s64 %kgd1, {reg}, {imm};")

    def in_range(lo: int, hi: int, dst: str) -> list[str]:
        return [
            f"ld.const.u64 %kgd2, [__kvg_trace+{8 * lo}];",
            f"ld.const.u64 %kgd3, [__kvg_trace+{8 * hi}];",
            "setp.ge.u64 %kgp2, %kgd1, %kgd2;",
            "setp.lt.u64 %kgp3, %kgd1, %kgd3;",
            f"and.pred {dst}, %kgp2, %kgp3;",
        ]

    c += in_range(T_KV0_LO, T_KV0_HI, "%kgp4")
    c += in_range(T_KV1_LO, T_KV1_HI, "%kgp5")
    c += ["or.pred %kgp4, %kgp4, %kgp5;", "and.pred %kgp4, %kgp4, %kgp1;"]
    c += in_range(T_TBL_LO, T_TBL_HI, "%kgp5")
    c += ["and.pred %kgp5, %kgp5, %kgp1;"]
    c += [
        f"ld.const.u64 %kgd4, [__kvg_trace+{8 * T_BITMAP_KV}];",
        f"add.s64 %kgd4, %kgd4, {4 * word};",
        f"mov.b32 %kgr1, {bit};",
        "@%kgp4 atom.global.or.b32 %kgr2, [%kgd4], %kgr1;",
        f"ld.const.u64 %kgd5, [__kvg_trace+{8 * T_BITMAP_TBL}];",
        f"add.s64 %kgd5, %kgd5, {4 * word};",
        "@%kgp5 atom.global.or.b32 %kgr2, [%kgd5], %kgr1;",
    ]
    return c


def _closure_tokens(an: Provenance, toks) -> set[str]:
    """Address tokens plus the address tokens of every load feeding them."""
    out, stack, seen = set(toks), [t for t in toks if t.startswith("L:")], set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        sub = an.site_addr.get(int(t[2:]), ())
        out |= set(sub)
        stack.extend(x for x in sub if x.startswith("L:"))
    return out


def flag_set(bit: int) -> list[str]:
    """Record the table-site predicate %kgp9 as bit ``bit`` of the flag register."""
    return [
        f"selp.b32 %kgr14, {1 << bit}, 0, %kgp9;",
        f"and.b32 %kgm0, %kgm0, {(~(1 << bit)) & 0xFFFFFFFF:#x};",
        "or.b32 %kgm0, %kgm0, %kgr14;",
    ]


def use_guard_check(site_id: int, inst, mask: int, src_size) -> list[str]:
    """Before a KV access: report when a table load its address derives from was beyond bounds
    (flag bits ``mask``) and the access executes (its predicate, a non-zero cp.async size). No
    enable gate here: the flag bits are set under the table site's gate, so a disabled
    descriptor leaves them clear and the guard falls through. The record's site is the first
    access of a guarded group (see ``instrument``)."""
    c = _guard(inst, src_size)
    c += [
        f"and.b32 %kgr15, %kgm0, {mask:#x};",
        "setp.ne.u32 %kgp9, %kgr15, 0;",
        "and.pred %kgp9, %kgp9, %kgp1;",
        f"@!%kgp9 bra $KVG_SKIP_{site_id};",
        "cvt.u64.u32 %kgd5, %kgr15;",
        "mov.u64 %kgd6, 0;",
    ]
    c += _violation(site_id, KIND_BOUNDS, "%kgd5", "%kgd6")
    c += [f"$KVG_SKIP_{site_id}:"]
    return c


def _address_chains(k, cfg, rd, table_loads: set[int], kv_idxs: list[int]):
    """Per KV access, the register-chain instructions between the table loads and its address
    (``chains``), and per chain instruction the table loads upstream of it (``upstream``)."""
    derived = set(table_loads)
    changed = True
    while changed:
        changed = False
        for i in sorted(cfg.live):
            inst = k.insts[i]
            if i in derived or not inst.dsts or inst.mem is not None:
                continue
            for r in uses_of(inst):
                if any(d in derived for d in rd.get(i, {}).get(r, ())):
                    derived.add(i)
                    changed = True
                    break
    upstream: dict[int, set[int]] = {}

    def up(i):
        if i in upstream:
            return upstream[i]
        upstream[i] = set()
        if i in table_loads:
            upstream[i] = {i}
            return upstream[i]
        acc: set[int] = set()
        for r in uses_of(k.insts[i]):
            for d in rd.get(i, {}).get(r, ()):
                if d in derived:
                    acc |= up(d)
        upstream[i] = acc
        return acc

    chains: dict[int, set[int]] = {}
    for kv in kv_idxs:
        seen: set[int] = set()
        stack = [d for r in regs(k.insts[kv].mem) for d in rd.get(kv, {}).get(r, ())]
        while stack:
            d = stack.pop()
            if d in seen or d not in derived:
                continue
            seen.add(d)
            if d in table_loads:
                continue
            for r in uses_of(k.insts[d]):
                stack.extend(rd.get(d, {}).get(r, ()))
        chains[kv] = seen
        for i in seen:
            up(i)
        # The access itself is not a chain register definition; its sources are its address's.
        acc: set[int] = set()
        for r in regs(k.insts[kv].mem):
            for d in rd.get(kv, {}).get(r, ()):
                if d in derived:
                    acc |= up(d)
        upstream[kv] = acc
    return derived, chains, upstream


def _reaches_before(k, cfg, start: int, target: int, avoid: set[int]) -> bool:
    """Can execution continue from just after instruction ``start`` to one of ``avoid`` without
    first reaching ``target``?"""
    seen_blocks: set[int] = set()
    first = [(cfg.block_of[start], start + 1)]
    stack = first
    while stack:
        b, pos = stack.pop()
        _s, e = cfg.blocks[b]
        hit_target = False
        for i in range(pos, e):
            if i == target:
                hit_target = True
                break
            if i in avoid:
                return True
        if hit_target:
            continue
        for nb in cfg.succ.get(b, ()):
            if nb not in seen_blocks:
                seen_blocks.add(nb)
                stack.append((nb, cfg.blocks[nb][0]))
    return False


def _flag_exact_sites(k, cfg, rd, table_loads, kv_idxs, chains, upstream) -> set[int]:
    """Table sites whose beyond flag is exact for every consumer: no chain value computed from
    the site's load can be consumed after the load ran again (then the flag would describe the
    newer load while the value came from the older one)."""
    inexact: set[int] = set()
    for kv in kv_idxs:
        members = chains[kv] | {kv}
        for c in members:
            inst = k.insts[c]
            for r in uses_of(inst):
                for d in rd.get(c, {}).get(r, ()):
                    if d in table_loads or d not in chains[kv]:
                        continue  # a load's own value is always the latest execution's
                    loads = upstream.get(d, set())
                    if loads and _reaches_before(k, cfg, d, c, loads):
                        inexact |= loads
    return set(table_loads) - inexact


def instrument(
    module_ptx: str,
    entry: str,
    roles: Roles,
    table_is_write: bool = False,
    modes=("table", "kv"),
    kv_read_alloc: bool = False,
    bounds_only: bool = False,
    maxnreg: int | None = None,
    use_guarded: bool = False,
    desc_in_regs: bool = False,
    guard_after: bool = False,
) -> tuple[str, dict]:
    """Return the instrumented module text and insertion statistics.

    ``modes``: ``("table",)`` with ``bounds_only=True`` is Tier 1b, ``("table",)`` Tier 1,
    ``("table", "kv")`` Tier 2. ``table_is_write`` marks the table as a write-side slot map
    (``reshape_and_cache``). ``maxnreg`` pins the register budget. ``use_guarded`` moves
    the report of a Tier 1b table check to the KV accesses that consume the entry.
    ``desc_in_regs`` reads the descriptor once per thread at kernel entry into registers.
    ``guard_after`` places a use guard after the access it guards instead of before it (the
    access's own predicate decides either way; the copy's issue is not delayed by the check).
    ``modes=("trace",)`` builds the dynamic cross-check twin instead: every global or generic
    access records in a device bitmap whether its address fell in the KV or table ranges of
    ``__kvg_trace`` (``trace_check``); no safety check is inserted.

    Non-inlined callees reached from the entry are instrumented in place with their parameters
    bound to the caller's stored arguments (the analysis's interprocedural binding); a
    callee whose check cannot reach a pointer parameter raises ``UnsupportedCallee``. Site ids
    of a callee are offset by the instruction counts of the functions emitted before it
    (``stats["site_base"]``).
    """
    global _DESC_IN_REGS
    _DESC_IN_REGS = desc_in_regs
    _DESC_SLOTS_USED.clear()
    opts = dict(
        table_is_write=table_is_write,
        modes=modes,
        kv_read_alloc=kv_read_alloc,
        bounds_only=bounds_only,
        use_guarded=use_guarded,
        desc_in_regs=desc_in_regs,
        guard_after=guard_after,
    )
    try:
        return _instrument(module_ptx, entry, roles, maxnreg, opts)
    finally:
        _DESC_IN_REGS = False
        _set_param_ptr(None)


def _set_param_ptr(value: dict | None) -> None:
    global _PARAM_PTR
    _PARAM_PTR = value


_MAX_CALL_DEPTH = 4  # as analyze_kernel


def _instrument(module_ptx, entry, roles, maxnreg, opts) -> tuple[str, dict]:
    text = split_entries(module_ptx)[entry]
    new_entry, stats, an, k = _instrument_function(text, roles, None, 0, maxnreg, opts)
    prelude = module_prelude(module_ptx)
    prelude += f"\n.const .align 8 .b64 __kvg_desc[{N_DESC}];\n"
    if "trace" in opts["modes"]:
        prelude += f".const .align 8 .b64 __kvg_trace[{N_TRACE}];\n"
    # Everything after the prelude (function prototypes, non-inlined callees) is kept in place;
    # only the entry is replaced. A module without callees reduces to prelude + entry.
    before, after = module_ptx[len(module_prelude(module_ptx)) :].split(text, 1)

    # Callees reachable from the entry, with each parameter seeded by the union of the tokens the
    # call sites store into it (a callee reached with a grown seed is followed again).
    funcs = split_funcs(module_ptx)
    parsed: dict[str, Kernel] = {}
    seeds: dict[str, dict[str, set[str]]] = {}
    work: list[tuple[Kernel, Provenance, int]] = [(k, an, 0)]
    while work:
        ck, can, depth = work.pop()
        if depth >= _MAX_CALL_DEPTH:
            continue
        for callee, argtoks in call_bindings(can, ck):
            if callee not in funcs:
                continue
            if callee not in parsed:
                parsed[callee] = parse_kernel(funcs[callee])
            fk = parsed[callee]
            grown = callee not in seeds
            seed = seeds.setdefault(callee, {p: set() for p in fk.params})
            for p, toks in zip(fk.params, argtoks, strict=False):
                if not toks <= seed[p]:
                    seed[p] |= toks
                    grown = True
            if grown:
                work.append((fk, Provenance(fk, param_seed=seed), depth + 1))

    site_base = len(k.insts)
    for callee in sorted(seeds, key=lambda n: module_ptx.find(funcs[n])):
        fk = parsed[callee]
        orig = funcs[callee]
        new_text, cstats, _, _ = _instrument_function(
            orig, roles, seeds[callee], site_base, None, opts
        )
        base, site_base = site_base, site_base + len(fk.insts)
        if not (cstats["kv_sites"] or cstats["table_sites"]):
            continue  # nothing to check there: the callee's text is kept as it is
        if before.count(orig) == 1:
            before = before.replace(orig, new_text, 1)
        elif after.count(orig) == 1:
            after = after.replace(orig, new_text, 1)
        else:
            raise RuntimeError(f"callee {callee} not found once in the module")
        for key, v in cstats.items():
            if isinstance(v, int) and key != "desc_slots_in_regs":
                stats[key] = stats.get(key, 0) + v
        stats.setdefault("callees", {})[callee] = cstats
        stats.setdefault("site_base", {})[callee] = base
    return prelude + before.lstrip("\n") + new_entry + after.rstrip("\n"), stats


def _instrument_function(
    text: str,
    roles: Roles,
    seed: dict[str, set[str]] | None,
    site_base: int,
    maxnreg: int | None,
    opts: dict,
) -> tuple[str, dict, Provenance, Kernel]:
    """Instrument one ``.entry`` (``seed`` None) or ``.func`` (``seed``: its parameters' tokens
    from the call sites) and return its new text, its statistics, and its provenance and parsed
    form (for the calls it makes). Site ids start at ``site_base``."""
    table_is_write = opts["table_is_write"]
    modes = opts["modes"]
    kv_read_alloc = opts["kv_read_alloc"]
    bounds_only = opts["bounds_only"]
    use_guarded = opts["use_guarded"]
    desc_in_regs = opts["desc_in_regs"]
    guard_after = opts["guard_after"]
    k = parse_kernel(text)
    _set_param_ptr(None if seed is None else _param_bindings(k.name, seed))
    _DESC_SLOTS_USED.clear()
    an0 = Provenance(k, param_seed=seed)
    nonnull = {t for t, r in roles.items() if r in ("table", "kv")}
    cfg, rd, pconst = specialize(k, an0, nonnull)
    an, res = classify(k, roles, cfg.live, pconst, param_seed=seed)
    sites: dict[int, tuple] = {}
    stats = {"kv_sites": 0, "table_sites": 0, "ambiguous_kv": 0}
    for a in res["accesses"]:
        inst = k.insts[a["idx"]]
        if a["category"].startswith("kv_"):
            kv = [t for t in a["params"] if roles.get(t) == "kv"]
            if len(kv) != 1:
                stats["ambiguous_kv"] += 1
                continue
            src_size = None
            if inst.op.startswith("cp.async") and len(inst.operands) >= 4:
                src_size = inst.operands[3].strip()
            sites[a["idx"]] = ("kv", kv[0], a["kind"] in ("store", "atomic"), src_size)
            stats["kv_sites"] += 1
        elif a["category"] == "table_load" and (inst.dsts or inst.op.startswith("cp.async")):
            tt = [t for t in a["params"] if roles.get(t) == "table"]
            split = None
            reg, _ = _addr(inst.mem)
            ds = rd.get(a["idx"], {}).get(reg, ())
            if reg and len(ds) == 1:
                d = k.insts[next(iter(ds))]
                if d.op.startswith("add.s64") and len(d.operands) == 3:
                    x, y = d.operands[1].strip(), d.operands[2].strip()
                    table_tok = tt[0]

                    def has(r, table_tok=table_tok):
                        return r.startswith("%") and table_tok in an.tok[r]

                    def rowful(r, table_tok=table_tok):
                        # split only when the row-base term carries a row offset beyond the table
                        # pointer itself (a 2-D table); a flat map keeps the full offset
                        return bool(an.tok[r] - {table_tok})

                    if has(x) and not has(y) and rowful(x):
                        split = (x, y)
                    elif has(y) and not has(x) and rowful(y):
                        split = (y, x)
            uniform = not any(
                t == "S:tid" for t in _closure_tokens(an, an.site_addr.get(a["idx"], ()))
            )
            reload = 0
            if not inst.dsts:  # cp.async [smem], [table+..], size
                reload = int(inst.operands[2].strip(), 0)
            ssz = inst.operands[3].strip() if (not inst.dsts and len(inst.operands) >= 4) else None
            addr_def = next(iter(ds)) if reg and len(ds) == 1 else None
            sites[a["idx"]] = ("table", tt[0], split, uniform, reload, ssz, addr_def)
            stats["table_uniform"] = stats.get("table_uniform", 0) + uniform
            stats["table_sites"] += 1
            stats["table_split"] = stats.get("table_split", 0) + (split is not None)

    if "trace" in modes:
        # Every global or generic access becomes a trace site, whatever its category.
        sites = {}
        for a in res["accesses"]:
            inst = k.insts[a["idx"]]
            reg, _ = _addr(inst.mem)
            if a["space"] not in ("global", "generic") or not reg:
                continue
            src_size = None
            if inst.op.startswith("cp.async") and len(inst.operands) >= 4:
                src_size = inst.operands[3].strip()
            sites[a["idx"]] = ("trace", src_size, a["category"])
        stats["trace_sites"] = len(sites)
        stats["trace_categories"] = {}
        for st in sites.values():
            stats["trace_categories"][st[2]] = stats["trace_categories"].get(st[2], 0) + 1

    # Use-guarded mode: which table sites get a flag, and which KV sites check which flags.
    flag_bits: dict[int, int] = {}
    guard_masks: dict[int, int] = {}
    if use_guarded and bounds_only and "table" in modes:
        table_idxs = {i for i, st in sites.items() if st[0] == "table"}
        kv_idxs = [i for i, st in sites.items() if st[0] == "kv"]
        _derived, chains, upstream = _address_chains(k, cfg, rd, table_idxs, kv_idxs)
        exact = _flag_exact_sites(k, cfg, rd, table_idxs, kv_idxs, chains, upstream)
        # A table site is flagged only if some KV access consumes its flag; a site whose loaded
        # value reaches no (guardable) KV access keeps the immediate check, else its bounds would
        # never be reported. Consumers that also depend on an unflagged site stay unguarded.
        candidates = set(sorted(exact)[:MAX_FLAG_SITES])
        while True:
            consumed = set()
            for kv in kv_idxs:
                src = upstream.get(kv, set())
                if src and src <= candidates:
                    consumed |= src
            if consumed == candidates:
                break
            candidates = consumed
        for bit, i in enumerate(sorted(candidates)):
            flag_bits[i] = bit
        for kv in kv_idxs:
            src = upstream.get(kv, set())
            if src and src <= set(flag_bits):
                guard_masks[kv] = sum(1 << flag_bits[i] for i in src)
        stats["guarded_table_sites"] = len(flag_bits)
        stats["inexact_table_sites"] = len(table_idxs) - len(exact)
        stats["unconsumed_table_sites"] = len(exact) - len(flag_bits)
        stats["guarded_kv_sites"] = len(guard_masks)
        stats["unguarded_kv_sites"] = len(kv_idxs) - len(guard_masks)

    # Row hoisting: table sites (bounds only) whose row-base register has one unpredicated
    # definition share a row group per definition; the row and its length are computed there.
    hoist_groups: dict[int, int] = {}  # defining instruction -> group
    hoisted_sites: dict[int, int] = {}  # table site -> group
    hoist_blocks: dict[int, list[str]] = {}
    hoist_tok: dict[int, str] = {}  # group -> table parameter token
    if _HOIST_ROWS and bounds_only and "table" in modes:
        for idx, st in sites.items():
            if st[0] != "table" or not st[2] or st[6] is None:
                continue
            xb = st[2][0]
            # The row base is used by the address definition, not by the load itself.
            ds = rd.get(st[6], {}).get(xb, ())
            if len(ds) != 1:
                continue
            d = next(iter(ds))
            if k.insts[d].pred or k.insts[d].mem is not None:
                continue
            if d in hoist_groups:
                g = hoist_groups[d]
                if hoist_tok[g] != st[1]:
                    continue  # another table parameter: keep the per-site code
            else:
                g = len(hoist_groups)
                hoist_groups[d] = g
                hoist_tok[g] = st[1]
                hoist_blocks[g] = row_hoist_block(g, xb, st[1])
            hoisted_sites[idx] = g
        stats["hoisted_table_sites"] = len(hoisted_sites)
        stats["row_groups"] = len(hoist_groups)

    # Re-emit the entry body with checks inserted (same statement splitting as parse_kernel).
    stripped = strip_comments(text)
    head, body = stripped.split("{", 1)
    body = body.rsplit("}", 1)[0]
    out: list[str] = []
    idx = 0
    flag_init_done = False
    last_guard = None
    desc_prelude_at = None
    for stmt in body.split(";"):
        rest, prefix = stmt, ""
        while True:
            s2 = rest.lstrip()
            prefix += rest[: len(rest) - len(s2)]
            rest = s2
            if rest[:1] in ("{", "}"):
                prefix += rest[0]
                rest = rest[1:]
                continue
            m = _LABEL.match(rest)
            if m:
                prefix += rest[: m.end()]
                rest = rest[m.end() :]
                continue
            break
        if not rest.strip():
            if prefix.strip():
                out.append(prefix)
            continue
        if rest.startswith("."):
            out.append(prefix + rest + ";")
            continue
        site = sites.get(idx)
        pre: list[str] = []
        post: list[str] = []
        if (flag_bits or desc_in_regs) and not flag_init_done:
            # First instruction of the body: no table site has set a flag yet; the descriptor
            # registers are filled here (the loads are inserted after the body is known).
            pre = ["mov.b32 %kgm0, 0;"] if flag_bits else []
            desc_prelude_at = len(out)
            flag_init_done = True
        if site and site[0] == "trace":
            pre += trace_check(site_base + idx, k.insts[idx], site[1])
        elif site and site[0] == "kv" and "kv" in modes:
            pre += kv_site_check(
                site_base + idx, k.insts[idx], site[1], site[2], site[3], kv_read_alloc
            )
        elif site and site[0] == "kv" and idx in guard_masks:
            # One guard per group: consecutive accesses of one basic block with the same source
            # mask, predicate and cp.async size execute together, and no table site between them
            # can change the flags, so the first access's guard covers the group.
            key = (cfg.block_of.get(idx), guard_masks[idx], k.insts[idx].pred, site[3])
            if key != last_guard:
                guard = use_guard_check(site_base + idx, k.insts[idx], guard_masks[idx], site[3])
                if guard_after:
                    post = guard
                else:
                    pre += guard
                stats["guard_groups"] = stats.get("guard_groups", 0) + 1
                last_guard = key
        elif site and site[0] == "table" and "table" in modes and idx in flag_bits:
            last_guard = None  # the flags may change here
            post = table_site_check(
                site_base + idx,
                k.insts[idx],
                site[1],
                table_is_write,
                site[2],
                site[3],
                val="%kgv0" if site[4] == 4 else None,
                src_size=site[5],
                reload_bytes=site[4],
                bounds_only=True,
                flag_bit=flag_bits[idx],
                hoisted=hoisted_sites.get(idx),
            )
        elif site and site[0] == "table" and "table" in modes:
            post = table_site_check(
                site_base + idx,
                k.insts[idx],
                site[1],
                table_is_write,
                site[2],
                site[3],
                val="%kgv0" if site[4] == 4 else None,
                src_size=site[5],
                reload_bytes=site[4],
                bounds_only=bounds_only,
                hoisted=hoisted_sites.get(idx) if bounds_only else None,
            )
        if idx in hoist_groups:
            post = post + hoist_blocks[hoist_groups[idx]]  # after the row base is defined
        out.append(
            prefix
            + "\n"
            + "\n".join(pre)
            + ("\n" if pre else "")
            + rest
            + ";"
            + ("\n" + "\n".join(post) if post else "")
        )
        idx += 1
    if idx != len(k.insts):
        raise RuntimeError(
            f"re-emission desynchronized: {idx} statements vs {len(k.insts)} instructions"
        )
    if maxnreg:
        head = re.sub(r"\.maxnreg\s+\d+", "", head).rstrip() + f"\n.maxnreg {maxnreg}\n"
    hoist_regs = (
        f".reg .b32 %kgrow<{len(hoist_groups)}>;\n.reg .b32 %kgrlen<{len(hoist_groups)}>;\n"
        if hoist_groups
        else ""
    )
    if desc_in_regs and desc_prelude_at is not None and _DESC_SLOTS_USED:
        loads = "\n".join(
            f"ld.const.u64 %kgc{slot}, {_desc(slot)};" for slot in sorted(_DESC_SLOTS_USED)
        )
        out[desc_prelude_at] = loads + "\n" + out[desc_prelude_at]
        hoist_regs += f".reg .b64 %kgc<{N_DESC}>;\n"
        stats["desc_slots_in_regs"] = len(_DESC_SLOTS_USED)
    new_text = (
        head
        + "{\n"
        + REGS
        + (FLAG_REGS if flag_bits else "")
        + hoist_regs
        + "\n".join(out)
        + "\n}\n"
    )
    return new_text, stats, an, k


__all__ = [
    "D_BLOCK_SHIFT",
    "D_BS_LOG2",
    "D_BS_MASK",
    "D_COUNTER",
    "D_ENABLE",
    "D_FILLED",
    "D_INIT",
    "D_NBLOCKS",
    "D_RECORD",
    "D_ROW_LEN",
    "D_ROW_OWNER",
    "D_SLOT_SHIFT",
    "D_STATE",
    "D_TBL_ENTRY_LOG2",
    "D_TBL_MODE",
    "D_TBL_NROWS",
    "D_TBL_ROW_BYTES",
    "EN_KV",
    "EN_TABLE",
    "KIND_BOUNDS",
    "KIND_KV_READ",
    "KIND_KV_WRITE",
    "KIND_TABLE_READ",
    "KIND_TABLE_WRITE",
    "N_DESC",
    "N_TRACE",
    "REGS",
    "ST_ALLOC",
    "ST_PUBLISHED",
    "ST_SHARED",
    "T_BITMAP_KV",
    "T_BITMAP_TBL",
    "T_KV0_HI",
    "T_KV0_LO",
    "T_KV1_HI",
    "T_KV1_LO",
    "T_TBL_HI",
    "T_TBL_LO",
    "UnsupportedCallee",
    "flag_set",
    "instrument",
    "kv_site_check",
    "row_hoist_block",
    "table_site_check",
    "trace_check",
    "use_guard_check",
]  # fmt: skip
