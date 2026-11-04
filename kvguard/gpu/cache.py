"""Content-addressed cache of instrumented kernels.

The interposer saves every loaded image that carries a KV-family kernel under its 64-bit FNV-1a
content hash (``<dump>/<image hash>.fatbin|ptx``). This module turns those images into the
instrumented binaries the interposer substitutes at launch:

    <cache>/<image hash>/<FNV-1a of the kernel name>.cubin   (or .ptx when ptxas is not run)
    <cache>/<image hash>/manifest.json                        (what was built, from what, and why not)

A kernel is instrumented only when the provenance analysis proves every KV access table-derived on
every path (the gate); the manifest records the gate result for every candidate. Triton kernels
reach the driver as cubins; their PTX is the sibling of the identical cubin in Triton's cache, and
their parameter names come positionally from the Python definition (constexpr arguments excluded,
plus Triton's two scratch pointers), so the same rewriter serves them. For Triton the PTX analysis
alone is may-only; the TTGIR analysis gives the must result the gate needs (``gate_override`` names
that record). The register budget is pinned to the shipped SASS's count (rounded up to 8) when the
image carries a cubin, else to ptxas's count for the original PTX. Everything here is host-only:
cuobjdump and ptxas from the CUDA toolkit, no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from .ptx import (
    Kernel,
    Roles,
    analyze_kernel,
    parse_kernel,
    parse_offsets,
    reduce_module,
    roles_by_name,
    roles_by_struct,
    split_entries,
    summarize,
    top_level_chunks,
)
from .ptx.rewrite import UnsupportedCallee, instrument
from .ptx.roles import (
    FA2_FLASH_FWD_PARAMS,
    FLASHINFER_BATCH_PREFILL_PAGED,
    TRITON_UNIFIED_ATTENTION,
)

RolesFor = Callable[[Kernel], Roles]

TIERS = {
    # The control: the original PTX recompiled by this toolkit's ptxas with no check inserted,
    # which separates the cost of recompilation from the cost of the checks.
    "recompiled": None,
    "tier1b": dict(modes=("table",), bounds_only=True),
    "tier1b_guarded": dict(modes=("table",), bounds_only=True, use_guarded=True),
    "tier1b_guarded_regs": dict(
        modes=("table",), bounds_only=True, use_guarded=True, desc_in_regs=True
    ),
    "tier1b_guarded_post": dict(
        modes=("table",), bounds_only=True, use_guarded=True, guard_after=True
    ),
    "tier1": dict(modes=("table",)),
    # The dynamic cross-check twin: no check, every global access reports which range it hit.
    "trace": dict(modes=("trace",)),
    "tier2": dict(modes=("table", "kv")),
}

FNV_OFFSET, FNV_PRIME, FNV_MASK = 14695981039346656037, 1099511628211, (1 << 64) - 1


def fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & FNV_MASK
    return h


def name_hash(name: str) -> str:
    """The file stem the interposer looks up for a kernel name."""
    return f"{fnv1a(name.encode()):016x}"


def image_hash(path: Path) -> str:
    return f"{fnv1a(path.read_bytes()):016x}"


def _find_tool(name: str) -> str | None:
    return shutil.which(name) or shutil.which(
        name, path=os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin")
    )


def _has_tool(name: str) -> bool:
    return _find_tool(name) is not None


def _tool(name: str) -> str:
    p = _find_tool(name)
    if not p:
        raise FileNotFoundError(f"{name} not found; set CUDA_HOME")
    return p


def triton_ptx_for_cubin(image: Path, triton_cache: Path) -> Path | None:
    """The PTX Triton compiled into this cubin: the sibling ``.ptx`` of the cache entry whose
    ``.cubin`` has the same bytes (Triton loads the cubin through cuModuleLoadData)."""
    if not triton_cache.is_dir():
        return None
    want = image.read_bytes()
    for cubin in triton_cache.glob("*/*.cubin"):
        try:
            if cubin.stat().st_size == len(want) and cubin.read_bytes() == want:
                ptx = cubin.with_suffix(".ptx")
                return ptx if ptx.exists() else None
        except OSError:
            continue
    return None


def extract_ptx(image: Path, workdir: Path, triton_cache: Path | None = None) -> list[Path]:
    """The PTX modules of a fat binary (cuobjdump writes into the working directory), or of a
    Triton cubin (looked up in the Triton cache)."""
    if image.suffix == ".ptx":
        return [image]
    if image.suffix == ".cubin":
        ptx = triton_ptx_for_cubin(image, triton_cache) if triton_cache else None
        return [ptx] if ptx else []
    subprocess.run(
        [_tool("cuobjdump"), "-xptx", "all", str(image.resolve())],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return sorted(workdir.glob("*.ptx"))


_FUNC_LINE = re.compile(r"^\s*Function\s+(\S+?):\s*$")
_REG_LINE = re.compile(r"\bREG:(\d+)")


def shipped_registers(image: Path, workdir: Path) -> dict[str, int]:
    """Register count per kernel in the image's cubins (empty for PTX-only images)."""
    if image.suffix == ".ptx":
        return {}
    if image.suffix == ".cubin":
        cubins = [image]
    else:
        subprocess.run(
            [_tool("cuobjdump"), "-xelf", "all", str(image.resolve())],
            cwd=workdir,
            check=True,
            capture_output=True,
        )
        cubins = sorted(workdir.glob("*.cubin"))
    regs: dict[str, int] = {}
    for cubin in cubins:
        out = subprocess.run(
            [_tool("cuobjdump"), "--dump-resource-usage", str(cubin)],
            capture_output=True,
            text=True,
        ).stdout
        current = None
        for line in out.splitlines():
            m = _FUNC_LINE.match(line)
            if m:
                current = m.group(1)
                continue
            r = _REG_LINE.search(line)
            if current and r:
                regs[current] = max(regs.get(current, 0), int(r.group(1)))
                current = None
    return regs


def ptxas_registers(ptx_path: Path, arch: str) -> int | None:
    """Register count ptxas assigns to the original PTX (None without ptxas: no pin)."""
    if not _has_tool("ptxas"):
        return None
    out = subprocess.run(
        [_tool("ptxas"), f"-arch={arch}", "-O3", "-v", str(ptx_path), "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    m = re.search(r"Used (\d+) registers", out.stderr)
    return int(m.group(1)) if m else None


def pin_registers(n: int | None) -> int | None:
    return None if n is None else min((n + 7) // 8 * 8, 255)


def ptx_target(ptx: str) -> str:
    m = re.search(r"^\.target\s+(sm_\d+)", ptx, re.M)
    return m.group(1) if m else "sm_80"


def gate(summary, allow_zero_fallback: bool = False) -> tuple[bool, str]:
    """Every KV access must be table-derived on every path and no access may have an unknown
    base. With ``allow_zero_fallback`` an access that is table-derived or a zero offset (the
    kernel's fallback to page 0 for positions beyond the row) on every path is accepted, and the
    reason says so."""
    if summary.unknown_base:
        return False, f"{summary.unknown_base} accesses with unknown base"
    if summary.kv_accesses == 0:
        return False, "no KV access found"
    if summary.kv_must_table == summary.kv_accesses:
        return True, f"{summary.kv_accesses} KV accesses, all table-derived"
    if allow_zero_fallback and summary.kv_must_table_or_zero == summary.kv_accesses:
        return True, (
            f"{summary.kv_accesses} KV accesses, {summary.kv_must_table} table-derived on every "
            f"path, {summary.kv_accesses - summary.kv_must_table} table-derived or page-0 "
            "fallback on every path"
        )
    return False, (
        f"{summary.kv_accesses - summary.kv_must_table} of {summary.kv_accesses} KV accesses "
        "not table-derived on every path"
    )


class StructRoles:
    """Roles for a kernel whose single parameter is a struct, from a ``<field> <offset>`` list
    compiled for the pinned source (FlashAttention-2's ``Flash_fwd_params``, FlashInfer's
    generated ``PagedParams``) and a field-name role map. A class, so that it can be sent to
    worker processes."""

    def __init__(self, offsets_text: str, role_of: dict[str, str]):
        self.offsets, self.size = parse_offsets(offsets_text)
        self.role_of = role_of

    def __call__(self, k: Kernel) -> Roles:
        return roles_by_struct(k, self.offsets, self.role_of, self.size)


def fa2_roles_for(offsets_text: str) -> RolesFor:
    return StructRoles(offsets_text, FA2_FLASH_FWD_PARAMS)


Fa2Roles = StructRoles  # the earlier name


_DEF = re.compile(r"^def\s+(\w+)\s*\((.*?)\)\s*(?:->[^:]*)?:", re.S | re.M)


def triton_arg_names(source: str, kernel: str) -> list[str]:
    """The run-time parameter names of a ``@triton.jit`` kernel in definition order: every
    parameter not annotated ``tl.constexpr``, followed by Triton's ``global_scratch`` and
    ``profile_scratch`` pointers."""
    for m in _DEF.finditer(source):
        if m.group(1) != kernel:
            continue
        body = re.sub(r"#[^\n]*", "", m.group(2))
        names = []
        for raw in body.split(","):
            item = raw.strip()
            if not item:
                continue
            name, _, annotation = item.partition(":")
            name = name.split("=")[0].strip()
            if "constexpr" in annotation:
                continue
            names.append(name)
        return [*names, "global_scratch", "profile_scratch"]
    raise KeyError(f"no def {kernel} in the source")


_TTFUNC = re.compile(r"tt\.func\s+(?:public\s+)?@(\w+)\s*\((.*?)\)\s*(?:attributes|->|\{)", re.S)


def triton_arg_names_from_ttir(ttir: str, kernel: str) -> list[str]:
    """The compiled kernel's run-time arguments in order, from the Triton-IR function signature
    (``tt.func public @k(%output_ptr: !tt.ptr<bf16> {...}, ...)``): Triton names them after the
    Python parameters and has already dropped the constexpr and specialized ones. Triton's two
    scratch pointers follow in the PTX."""
    for m in _TTFUNC.finditer(ttir):
        if m.group(1) != kernel:
            continue
        args = [a.strip() for a in re.split(r",\s*(?=%)", m.group(2)) if a.strip()]
        names = [a.split(":")[0].strip().lstrip("%") for a in args]
        return [*names, "global_scratch", "profile_scratch"]
    raise KeyError(f"no tt.func @{kernel} in the Triton IR")


class TritonRoles:
    """Roles for a Triton kernel by position: ``role_of`` maps the Python parameter names of
    the kernel to roles. The names come from the ``.ttir`` next to the kernel's PTX in Triton's
    cache (exact: specialized arguments are already gone) and, without one, from the Python
    definition; a count that differs from the PTX's parameter count is reported as a mismatch so
    the kernel is left unsupported."""

    wants_ptx_path = True

    def __init__(self, source: str | None, kernel: str, role_of: dict[str, str]):
        self.kernel = kernel
        self.names = triton_arg_names(source, kernel) if source else None
        self.role_of = role_of

    def __call__(self, k: Kernel, ptx_path: Path | None = None) -> Roles:
        names = self.names
        ttir = ptx_path.with_suffix(".ttir") if ptx_path else None
        if ttir is not None and ttir.exists():
            names = triton_arg_names_from_ttir(ttir.read_text(), self.kernel)
        if names is None:
            raise ValueError(f"{k.name}: no Triton IR next to the PTX and no Python source given")
        if len(k.params) != len(names):
            raise ValueError(
                f"{k.name}: {len(k.params)} PTX parameters but {len(names)} named arguments "
                "(argument specialization changed the parameter list)"
            )
        return roles_by_name(k, names, self.role_of)


def build_kernel(
    ptx_path: Path,
    name: str,
    roles_for: RolesFor,
    tier: str,
    target: str,
    out_dir: Path,
    shipped_regs: int | None,
    run_ptxas: bool,
    table_is_write: bool,
    gate_override: str = "",
    allow_zero_fallback: bool = False,
) -> tuple[str, dict]:
    """Analyze, gate, instrument and compile one entry of the module at ``ptx_path``; returns the
    manifest record. Runs in a worker process when the build is parallel. ``gate_override`` names
    evidence that every KV access is must-table-derived when the PTX analysis alone is may-only, as
    for Triton kernels."""
    ptx = ptx_path.read_text()
    funcs = {n: t for kd, n, t in top_level_chunks(ptx) if kd == "func"}
    rec: dict = {"tier": tier, "arch": target, "file": None}
    reduced = reduce_module(ptx, name)
    k = parse_kernel(split_entries(reduced)[name])
    try:
        roles = (
            roles_for(k, ptx_path) if getattr(roles_for, "wants_ptx_path", False) else roles_for(k)
        )
    except (ValueError, KeyError) as exc:
        rec["gate"] = {"ok": False, "reason": f"roles: {exc}"}
        return name, rec
    res, extra = analyze_kernel(k, roles, funcs)
    summary = summarize(res, extra)
    ok, why = gate(summary, allow_zero_fallback)
    if tier == "trace":  # the cross-check wants every kernel, gated or not
        ok, why = True, f"trace tier: no gate ({why})"
    if (
        not ok
        and gate_override
        and summary.kv_table_derived == summary.kv_accesses
        and not summary.unknown_base
    ):
        # Table-derived on some path at PTX, must-derived by the Triton-IR analysis: the caller
        # vouches for the must result.
        ok, why = True, f"{why}; accepted by gate override ({gate_override})"
    callee_kv = sum(
        1 for a in res["accesses"] if a["fn"] != k.name and a["category"].startswith("kv_")
    )
    # KV accesses inside non-inlined callees are instrumented there (the rewriter binds the
    # callee's parameters to the call arguments); a callee whose checks cannot reach a pointer
    # parameter makes the kernel unsupported below.
    rec["gate"] = {"ok": ok, "reason": why, "callee_kv_accesses": callee_kv}
    if not ok:
        return name, rec
    n_regs = shipped_regs
    if n_regs is None:
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as fh:
            fh.write(reduced)
        try:
            n_regs = ptxas_registers(Path(fh.name), target)
        finally:
            os.unlink(fh.name)
        rec["registers_from"] = "ptxas"
    else:
        rec["registers_from"] = "shipped cubin"
    maxnreg = pin_registers(n_regs)
    rec["registers"] = n_regs
    rec["maxnreg"] = maxnreg
    if TIERS[tier] is None:
        text, stats = reduced, {"kv_sites": 0, "table_sites": 0, "control": "recompiled original"}
    else:
        try:
            text, stats = instrument(
                reduced,
                name,
                roles,
                table_is_write=table_is_write,
                maxnreg=maxnreg,
                **TIERS[tier],
            )
        except UnsupportedCallee as exc:
            rec["gate"] = {"ok": False, "reason": str(exc), "callee_kv_accesses": callee_kv}
            return name, rec
    rec["stats"] = stats
    stem = name_hash(name)
    ptx_out = out_dir / f"{stem}.ptx"
    ptx_out.write_text(text)
    rec["file"] = ptx_out.name
    if run_ptxas:
        cubin_out = out_dir / f"{stem}.cubin"
        r = subprocess.run(
            [_tool("ptxas"), f"-arch={target}", "-O3", "-v", str(ptx_out), "-o", str(cubin_out)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            rec["ptxas_error"] = r.stderr[-2000:]
        else:
            m = re.search(r"Used (\d+) registers", r.stderr)
            rec["file"] = cubin_out.name
            rec["instrumented_registers"] = int(m.group(1)) if m else None
            info = r.stderr.strip().splitlines()
            rec["ptxas_info"] = info[-1][:300] if info else ""
    return name, rec


def build_image(
    image: Path,
    cache_dir: Path,
    entry: str,
    roles_for: RolesFor,
    tier: str = "tier1b",
    arch: str | None = None,
    run_ptxas: bool = True,
    table_is_write: bool = False,
    jobs: int = 1,
    triton_cache: Path | None = None,
    gate_override: str = "",
    allow_zero_fallback: bool = False,
    log=print,
) -> dict:
    """Instrument every entry of ``image`` whose name matches the regex ``entry``; write the
    results under ``cache_dir/<image hash>/`` and return the manifest. ``jobs`` worker processes
    build kernels in parallel (``roles_for`` must then be picklable). A ``.cubin`` image is a
    Triton kernel whose PTX is looked up in ``triton_cache``."""
    ihash = image_hash(image)
    out_dir = cache_dir / ihash
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"image": image.name, "kernels": {}}
    )
    rx = re.compile(entry)
    with tempfile.TemporaryDirectory(prefix="kvg-cache-") as tmp:
        work = Path(tmp)
        regs = shipped_registers(image, work)
        tasks = []
        ptx_paths = extract_ptx(image, work, triton_cache)
        if image.suffix == ".cubin" and not ptx_paths:
            manifest["unsupported"] = "SASS only (no matching PTX in the Triton cache)"
        for ptx_path in ptx_paths:
            ptx = ptx_path.read_text()
            target = arch or ptx_target(ptx)
            for name in split_entries(ptx):
                if rx.search(name):
                    tasks.append(
                        (
                            ptx_path,
                            name,
                            roles_for,
                            tier,
                            target,
                            out_dir,
                            regs.get(name),
                            run_ptxas,
                            table_is_write,
                            gate_override,
                            allow_zero_fallback,
                        )
                    )
        if jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
                results = list(pool.map(build_kernel, *zip(*tasks, strict=True)))
        else:
            results = [build_kernel(*t) for t in tasks]
    for name, rec in results:
        manifest["kernels"][name] = rec
        if rec["file"] is None:
            log(f"  skip {name[:80]}…: {rec['gate']['reason']}")
        elif "ptxas_error" in rec:
            tail = rec["ptxas_error"].strip().splitlines()
            log(f"  ptxas failed for {name[:80]}…: {tail[-1] if tail else '?'}")
        else:
            log(
                f"  built {name[:80]}… -> {rec['file']} ({rec['stats']}, regs {rec['registers']}->{rec.get('instrumented_registers')})"
            )
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def build(
    dump_dir: Path,
    cache_dir: Path,
    entry: str,
    roles_for: RolesFor,
    images: list[str] | None = None,
    **kw,
) -> dict[str, dict]:
    """Build every ``.fatbin`` and ``.ptx`` image of ``dump_dir`` (or only those whose content hash
    or file stem is in ``images``), keyed by content hash. Cubin-only images cannot be
    instrumented and are recorded as such."""
    out: dict[str, dict] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for image in sorted(dump_dir.iterdir()):
        if image.suffix not in (".cubin", ".fatbin", ".ptx"):
            continue
        ihash = image_hash(image)
        if images is not None and image.stem not in images and ihash not in images:
            continue
        if image.suffix == ".cubin" and not kw.get("triton_cache"):
            out[ihash] = {"image": image.name, "unsupported": "SASS only (no PTX in the image)"}
            continue
        kw.get("log", print)(f"{image.name}:")
        out[ihash] = build_image(image, cache_dir, entry, roles_for, **kw)
    (cache_dir / "index.json").write_text(
        json.dumps(
            {
                h: {
                    "image": m.get("image"),
                    "kernels": len(m.get("kernels", {})),
                    "unsupported": m.get("unsupported"),
                }
                for h, m in out.items()
            },
            indent=1,
        )
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dump",
        required=True,
        type=Path,
        help="the interposer's KVGUARD_INTERPOSER_DUMP directory",
    )
    ap.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="output cache directory (KVGUARD_INTERPOSER_CACHE)",
    )
    ap.add_argument("--entry", required=True, help="regex over kernel names to instrument")
    ap.add_argument(
        "--family",
        choices=["fa2", "flashinfer-paged", "triton-unified-attention"],
        required=True,
        help="parameter-role source for the matched kernels",
    )
    ap.add_argument(
        "--offsets",
        type=Path,
        help="fa2: the Flash_fwd_params offsets file; flashinfer-paged: the PagedParams offsets file",
    )
    ap.add_argument(
        "--source",
        type=Path,
        help="Triton: the kernel's Python source file (fallback when the cache has no .ttir)",
    )
    ap.add_argument(
        "--kernel",
        default="kernel_unified_attention",
        help="Triton: the def to take the argument names from",
    )
    ap.add_argument(
        "--triton-cache",
        type=Path,
        default=Path.home() / ".triton" / "cache",
        help="Triton cache directory (PTX lookup for cubin images)",
    )
    ap.add_argument(
        "--allow-zero-fallback",
        action="store_true",
        help="accept KV accesses that are table-derived or a zero offset (page 0) on every path",
    )
    ap.add_argument(
        "--gate-override",
        default="",
        help="evidence record proving must-coverage when the PTX analysis is may-only (Triton)",
    )
    ap.add_argument("--tier", choices=sorted(TIERS), default="tier1b")
    ap.add_argument("--arch", help="ptxas target (default: the PTX module's .target)")
    ap.add_argument("--images", help="comma-separated image hashes to restrict the build to")
    ap.add_argument("--no-ptxas", action="store_true", help="write instrumented PTX only")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes")
    a = ap.parse_args(argv)
    if a.family in ("fa2", "flashinfer-paged"):
        if not a.offsets:
            ap.error(f"--family {a.family} needs --offsets")
        role_of = FA2_FLASH_FWD_PARAMS if a.family == "fa2" else FLASHINFER_BATCH_PREFILL_PAGED
        roles_for = StructRoles(a.offsets.read_text(), role_of)
    else:
        roles_for = TritonRoles(
            a.source.read_text() if a.source else None, a.kernel, TRITON_UNIFIED_ATTENTION
        )
    a.cache.mkdir(parents=True, exist_ok=True)
    out = build(
        a.dump,
        a.cache,
        a.entry,
        roles_for,
        images=a.images.split(",") if a.images else None,
        tier=a.tier,
        arch=a.arch,
        run_ptxas=not a.no_ptxas,
        jobs=a.jobs,
        triton_cache=a.triton_cache,
        gate_override=a.gate_override,
        allow_zero_fallback=a.allow_zero_fallback,
    )
    built = sum(1 for m in out.values() for r in m.get("kernels", {}).values() if r.get("file"))
    skipped = sum(
        1 for m in out.values() for r in m.get("kernels", {}).values() if not r.get("file")
    )
    print(f"built {built} kernels, skipped {skipped}, images {len(out)} -> {a.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
