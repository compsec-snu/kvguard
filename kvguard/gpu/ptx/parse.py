"""PTX module and kernel parsing.

PTX is statement-oriented: directives and instructions end with ``;``, function bodies are brace
blocks, labels end with ``:``. The parser splits a module into top-level chunks and an entry or
function into statements, and classifies each instruction's registers and memory operand. It is a
statement tokenizer with explicit rules, not a full grammar: the analysis needs operands,
predicates and memory operands, not full typing. Special registers (``%tid``, ``%ctaid``, …) are
excluded from data flow except through the ``SPECIAL:tid`` thread-index marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_COMMENT_LINE = re.compile(r"//[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_ENTRY = re.compile(r"\.entry\s+([A-Za-z0-9_$]+)\s*\(")
_FUNC = re.compile(r"\.func\s+(?:\(([^)]*)\)\s*)?([A-Za-z0-9_$]+)\s*\(")
_REG = re.compile(r"%[A-Za-z_][A-Za-z0-9_.]*")
_MEM = re.compile(r"\[\s*([^\]]*?)\s*\]")
_NOSEMI_DIRECTIVE = re.compile(r"^\s*\.(loc|file)\b[^\n]*$", re.M)
_HEADER = re.compile(r"^\s*\.(version|target|address_size)\b[^\n]*$", re.M)
_SYMBOL = re.compile(r"(?<![%\w])([A-Za-z_$][A-Za-z0-9_$]*)")
_PARAM = re.compile(r"\.param\s+[^,)]*?([A-Za-z0-9_$]+)\s*(?:\[\d+\])?\s*(?:,|\))")
_LABEL = re.compile(r"^([$A-Za-z0-9_]+):")
_SHARED_LOCAL = re.compile(r"\.(shared|local)\b.*?([A-Za-z_$][A-Za-z0-9_$]*)\s*(\[|$)")
_PRED_REG = re.compile(r"%p\d")

SPECIAL_PREFIXES = (
    "%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%nwarpid", "%smid", "%nsmid",
    "%gridid", "%clock", "%globaltimer", "%lanemask", "%dynamic_smem_size", "%total_smem_size",
    "%envreg", "%pm", "%cluster",
)  # fmt: skip

_THREAD_SPECIAL = re.compile(r"%(tid|laneid|warpid|lanemask)")
_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|\d+")

CONTROL_OPS = (
    "bra", "ret", "exit", "bar.", "barrier", "membar", "fence", "cp.async.wait", "cp.async.commit",
    "trap", "brkpt",
)  # fmt: skip
CP_ASYNC_NON_COPY = (
    "cp.async.wait", "cp.async.commit", "cp.async.bulk.commit", "cp.async.bulk.wait", "cp.async.mbarrier",
)  # fmt: skip


@dataclass
class Inst:
    """One PTX instruction. ``mem`` is the text inside the memory operand's brackets, and
    ``mem_role`` is ``"src"`` for loads and ``"dst"`` for stores."""

    idx: int
    text: str
    pred: str | None
    op: str
    dsts: list[str]
    srcs: list[str]
    mem: str | None
    mem_role: str | None
    operands: list[str] = field(default_factory=list)

    @property
    def is_load(self) -> bool:
        return (
            self.mem is not None and self.mem_role == "src" and not self.op.startswith("prefetch")
        )

    @property
    def is_store(self) -> bool:
        return self.mem is not None and self.mem_role == "dst"


@dataclass
class Kernel:
    """A parsed ``.entry`` or ``.func``: parameters in order, instructions, declared shared and
    local symbols, and label positions (instruction index each label precedes)."""

    name: str
    params: list[str]
    insts: list[Inst]
    shared_syms: set[str] = field(default_factory=set)
    local_syms: set[str] = field(default_factory=set)
    labels: dict[str, int] = field(default_factory=dict)


def strip_comments(text: str) -> str:
    """Remove comments and the debug-line directives that are not ``;``-terminated."""
    return _NOSEMI_DIRECTIVE.sub("", _COMMENT_LINE.sub("", _COMMENT_BLOCK.sub("", text)))


def split_entries(ptx: str) -> dict[str, str]:
    """``{entry_name: entry_text}`` for every ``.entry`` of a module (text from its line start to
    the closing brace)."""
    out: dict[str, str] = {}
    for m in _ENTRY.finditer(ptx):
        start = ptx.rfind("\n", 0, m.start()) + 1
        brace = ptx.index("{", m.end())
        depth, i = 0, brace
        while True:
            c = ptx[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out[m.group(1)] = ptx[start : i + 1]
    return out


_FUNC_BODY = re.compile(r"\s*(?:\.\w+\s*)*\{")  # attributes such as .noreturn, then the body


def split_funcs(ptx: str) -> dict[str, str]:
    """``{name: text}`` for every ``.func`` *definition* of a module, text from its line start to
    the closing brace (prototypes, which end in ``;`` instead of a body, are skipped)."""
    out: dict[str, str] = {}
    for m in _FUNC.finditer(ptx):
        i, depth = m.end() - 1, 0  # the parameter list's "("
        while True:
            c = ptx[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        b = _FUNC_BODY.match(ptx, i + 1)
        if not b:
            continue
        start = ptx.rfind("\n", 0, m.start()) + 1
        depth, i = 0, b.end() - 1
        while True:
            c = ptx[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out[m.group(2)] = ptx[start : i + 1]
    return out


def module_prelude(ptx: str) -> str:
    """Header and module-scope declarations that precede the first function."""
    first = len(ptx)
    for kw in (
        ".visible .entry",
        ".entry",
        ".visible .func",
        ".func",
        ".weak .func",
        ".extern .func",
    ):
        j = ptx.find(kw)
        if j != -1:
            first = min(first, j)
    return ptx[:first]


def split_operands(s: str) -> list[str]:
    """Split an operand list on top-level commas (braces, brackets and parentheses nest)."""
    ops, depth, cur = [], 0, []
    for c in s:
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        if c == "," and depth == 0:
            ops.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur and "".join(cur).strip():
        ops.append("".join(cur).strip())
    return ops


def regs(s: str, keep_pred: bool = False) -> list[str]:
    """Virtual registers in a text, excluding special registers and (by default) predicates."""
    return [
        r
        for r in _REG.findall(s)
        if not r.startswith(SPECIAL_PREFIXES) and (keep_pred or not _PRED_REG.match(r))
    ]


def parse_kernel(entry_text: str) -> Kernel:
    """Parse one ``.entry`` or ``.func`` definition."""
    text = strip_comments(entry_text)
    brace = text.index("{")
    m = _ENTRY.search(text)
    if m and m.start() < brace:
        name = m.group(1)
    else:
        f = _FUNC.search(text)
        if f is None:
            raise ValueError("not a PTX entry or function definition")
        name = f.group(2)
    head, body = text.split("{", 1)
    if not m or m.start() > brace:
        head = head[_FUNC.search(head).end() - 1 :]  # drop a .func's return-value list
    params = _PARAM.findall(head)
    kernel = Kernel(name=name, params=params, insts=[])
    body = body.rsplit("}", 1)[0]
    for stmt in body.split(";"):
        # Scope braces (inline-asm blocks) and labels only ever lead a statement; the braces of
        # vector operands come after the opcode and are kept.
        while True:
            stmt = stmt.strip()
            if stmt[:1] in ("{", "}"):
                stmt = stmt[1:]
                continue
            lm = _LABEL.match(stmt)
            if lm:
                kernel.labels[lm.group(1)] = len(kernel.insts)
                stmt = stmt[lm.end() :]
                continue
            break
        if not stmt:
            continue
        if stmt.startswith("."):
            dm = _SHARED_LOCAL.match(stmt)
            if dm:
                (kernel.shared_syms if dm.group(1) == "shared" else kernel.local_syms).add(
                    dm.group(2)
                )
            continue
        pred = None
        if stmt.startswith("@"):
            pred, stmt = stmt.split(None, 1)
        parts = stmt.split(None, 1)
        op = parts[0]
        operands = split_operands(parts[1]) if len(parts) > 1 else []
        kernel.insts.append(make_inst(len(kernel.insts), stmt, pred, op, operands))
    return kernel


def make_inst(idx: int, text: str, pred: str | None, op: str, operands: list[str]) -> Inst:
    """Classify an instruction's registers and memory operand by opcode family."""
    mem, mem_role = None, None
    dsts: list[str] = []
    srcs: list[str] = []
    if op.startswith(("st.", "red.")):
        mem, mem_role = _MEM.search(operands[0]).group(1), "dst"
        srcs = [r for o in operands[1:] for r in regs(o)]
    elif op.startswith("cp.async") and not op.startswith(CP_ASYNC_NON_COPY):
        # cp.async[...] [dst_shared], [src_global], size{, src_size}: the destination is shared
        # memory and is modelled through the shared pool by the analysis.
        mem, mem_role = _MEM.search(operands[1]).group(1), "src"
        srcs = [r for o in operands[2:] for r in regs(o)]
        srcs += regs(_MEM.search(operands[0]).group(1))
    elif op.startswith(("ld.", "ldu.", "atom.", "prefetch", "ldmatrix")):
        dsts = regs(operands[0]) if not op.startswith("prefetch") else []
        mi = 0 if op.startswith("prefetch") else 1
        mem, mem_role = _MEM.search(operands[mi]).group(1), "src"
        srcs = [r for o in operands[mi + 1 :] for r in regs(o)]
    elif op.startswith(CONTROL_OPS):
        srcs = [r for o in operands for r in regs(o)]
    elif op.startswith("call"):
        # call (rets), func, (args): results come back through return-value params.
        srcs = [r for o in operands for r in regs(o)]
    elif operands:
        dsts = regs(operands[0])
        srcs = [r for o in operands[1:] for r in regs(o)]
        if any(_THREAD_SPECIAL.search(o) for o in operands[1:]):
            srcs.append("SPECIAL:tid")
        # mov/cvta of a symbol: record the symbol as a source token
        for o in operands[1:]:
            for sym in _SYMBOL.findall(o):
                if not _NUMBER.fullmatch(sym):
                    srcs.append("SYM:" + sym)
    return Inst(idx, text, pred, op, dsts, srcs, mem, mem_role, operands)


def address_space(op: str) -> str:
    """The state space named by a memory opcode, or ``generic``."""
    for sp in ("global", "shared", "local", "param", "const"):
        if f".{sp}" in op:
            return sp
    return "generic"


def top_level_chunks(ptx: str) -> list[tuple[str, str | None, str]]:
    """Split a module into ``(kind, name, text)`` chunks: ``entry``, ``func`` (definition), or
    ``decl`` (any ``;``-terminated top-level statement)."""
    text = _COMMENT_BLOCK.sub("", ptx)
    text = "\n".join(
        line if not line.lstrip().startswith("//") else "" for line in text.splitlines()
    )
    out: list[tuple[str, str | None, str]] = []
    i, n, start, depth = 0, len(text), 0, 0
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                m = _ENTRY.search(chunk)
                f = _FUNC.search(chunk)
                if m and (not f or m.start() < f.start()):
                    out.append(("entry", m.group(1), chunk))
                elif f:
                    out.append(("func", f.group(2), chunk))
                else:
                    out.append(("decl", None, chunk))
                start = i + 1
        elif c == ";" and depth == 0:
            out.append(("decl", None, text[start : i + 1]))
            start = i + 1
        i += 1
    if text[start:].strip():
        out.append(("decl", None, text[start:]))
    return out


_CALL = re.compile(r"call(?:\.uni)?\s*(?:\([^)]*\)\s*,\s*)?([A-Za-z_$][A-Za-z0-9_$]*)\s*,?")


def callees(fn_text: str) -> set[str]:
    return set(_CALL.findall(strip_comments(fn_text)))


def reduce_module(ptx: str, entry: str) -> str:
    """A module containing every declaration plus the entry and the functions it transitively
    calls (used to JIT one kernel out of a large module)."""
    header = "\n".join(m.group(0).strip() for m in _HEADER.finditer(ptx)) + "\n"
    ptx = _HEADER.sub("", ptx)
    chunks = top_level_chunks(ptx)
    funcs = {name: t for k, name, t in chunks if k == "func"}
    keep: set[str] = set()
    stack: list[str] = []
    for k, name, t in chunks:
        if k == "entry" and name == entry:
            stack.extend(callees(t))
    while stack:
        f = stack.pop()
        if f in keep or f not in funcs:
            continue
        keep.add(f)
        stack.extend(callees(funcs[f]))

    def keep_decl(t: str) -> bool:
        f = _FUNC.search(t)
        return not f or f.group(2) in keep  # drop prototypes of functions not kept

    return (
        header
        + "".join(
            t
            for k, name, t in chunks
            if (k == "decl" and keep_decl(t))
            or (k == "entry" and name == entry)
            or (k == "func" and name in keep)
        )
        + "\n"
    )


__all__ = [
    "CONTROL_OPS",
    "Inst",
    "Kernel",
    "address_space",
    "callees",
    "make_inst",
    "module_prelude",
    "parse_kernel",
    "reduce_module",
    "regs",
    "split_entries",
    "split_funcs",
    "split_operands",
    "strip_comments",
    "top_level_chunks",
]
