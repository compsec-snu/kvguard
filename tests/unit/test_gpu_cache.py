"""The instrumented-kernel cache and the interposer's Python side: content hashes agree with the
C++ library's FNV-1a, the gate admits only fully table-derived kernels, a PTX image is built into
the layout the interposer looks up, and the Tier 1b descriptor has the rewriter's layout."""

import json

import pytest
from kvguard.gpu import cache
from kvguard.gpu.interposer import tier1b_descriptor
from kvguard.gpu.ptx import KV, TABLE, parse_kernel, roles_by_name, split_entries
from kvguard.gpu.ptx import rewrite as R

from tests.unit.test_ptx_analysis import module

# FNV-1a 64-bit reference values (the C++ side uses the same constants).
FNV_VECTORS = {b"": 0xCBF29CE484222325, b"a": 0xAF63DC4C8601EC8C, b"foobar": 0x85944171F73967E8}


def test_fnv1a_matches_the_reference_vectors():
    for data, h in FNV_VECTORS.items():
        assert cache.fnv1a(data) == h
    assert cache.name_hash("a") == "af63dc4c8601ec8c"


TABLE_KERNEL = module(
    """
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
    ret;""",
    ".param .u64 kv, .param .u64 table",
)

# A kernel whose KV access does not go through the table: the gate must reject it.
DIRECT_KERNEL = module(
    """
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd5, %r1, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
    ret;""",
    ".param .u64 kv, .param .u64 table",
)


def roles_for(k):
    return roles_by_name(k, k.params, {"kv": KV, "table": TABLE})


# The table load and the KV access live in a non-inlined callee that receives both pointers as
# call arguments: the checks are emitted inside the callee.
CALLEE_KERNEL = """//
.version 8.0
.target sm_80
.address_size 64

.func helper(.param .b64 helper_kv, .param .b64 helper_table)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [helper_kv];
    ld.param.u64 %rd2, [helper_table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    ret;
}

.visible .entry k(.param .u64 kv, .param .u64 table)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    {
    .param .b64 param0;
    .param .b64 param1;
    st.param.b64 [param0], %rd1;
    st.param.b64 [param1], %rd2;
    call.uni helper, (param0, param1);
    }
    ret;
}
"""


def test_build_instruments_callee_kv_accesses(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "callee.ptx").write_text(CALLEE_KERNEL)
    out = cache.build(
        dump, tmp_path / "cache", "^k$", roles_for, run_ptxas=False, log=lambda *_: None
    )
    ihash = cache.image_hash(dump / "callee.ptx")
    rec = out[ihash]["kernels"]["k"]
    assert rec["gate"]["ok"] and rec["gate"]["callee_kv_accesses"] == 1
    assert rec["stats"]["table_sites"] == 1 and "helper" in rec["stats"]["callees"]
    text = (tmp_path / "cache" / ihash / rec["file"]).read_text()
    assert "ld.param.u64 %kgd2, [helper_table];" in text  # the check reads the callee's parameter


# Two tables, the same callee: each call is sound on its own, but the callee's table parameter
# is bound to both tables, so a check inside it cannot know which base to subtract.
AMBIGUOUS_CALLEE_KERNEL = CALLEE_KERNEL.replace(
    ".visible .entry k(.param .u64 kv, .param .u64 table)",
    ".visible .entry k(.param .u64 kv, .param .u64 table, .param .u64 table2)",
).replace(
    """    call.uni helper, (param0, param1);
    }""",
    """    call.uni helper, (param0, param1);
    }
    ld.param.u64 %rd3, [table2];
    {
    .param .b64 param2;
    .param .b64 param3;
    st.param.b64 [param2], %rd1;
    st.param.b64 [param3], %rd3;
    call.uni helper, (param2, param3);
    }""",
)


def test_build_marks_a_callee_with_ambiguous_pointer_bindings_unsupported(tmp_path):
    """Called once with (kv, table) and once with (table, kv): the callee's checks cannot tell
    which pointer a parameter holds, so the kernel is recorded as unsupported, not built."""
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "callee.ptx").write_text(AMBIGUOUS_CALLEE_KERNEL)

    def two_tables(k):
        return roles_by_name(k, k.params, {"kv": KV, "table": TABLE, "table2": TABLE})

    out = cache.build(
        dump, tmp_path / "cache", "^k$", two_tables, run_ptxas=False, log=lambda *_: None
    )
    rec = out[cache.image_hash(dump / "callee.ptx")]["kernels"]["k"]
    assert rec["file"] is None and not rec["gate"]["ok"]
    assert "helper: no call argument binds P:table" in rec["gate"]["reason"]


def test_gate_accepts_table_derived_and_rejects_direct_access():
    from kvguard.gpu.ptx import analyze_kernel, summarize

    for ptx, expected in ((TABLE_KERNEL, True), (DIRECT_KERNEL, False)):
        k = parse_kernel(split_entries(ptx)["k"])
        res, extra = analyze_kernel(k, roles_for(k), {})
        ok, why = cache.gate(summarize(res, extra))
        assert ok is expected, why


def test_build_writes_the_layout_the_interposer_looks_up(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    image = dump / "image.ptx"
    image.write_text(TABLE_KERNEL)
    bad = dump / "bad.ptx"
    bad.write_text(DIRECT_KERNEL)
    (dump / "sass_only.cubin").write_bytes(b"\x7fELF")
    out = cache.build(
        dump, tmp_path / "cache", "^k$", roles_for, run_ptxas=False, log=lambda *_: None
    )
    ihash = cache.image_hash(image)
    twin = tmp_path / "cache" / ihash / f"{cache.name_hash('k')}.ptx"
    assert twin.exists()
    text = twin.read_text()
    assert f".const .align 8 .b64 __kvg_desc[{R.N_DESC}];" in text
    assert ".entry k(" in text
    manifest = json.loads((tmp_path / "cache" / ihash / "manifest.json").read_text())
    rec = manifest["kernels"]["k"]
    assert rec["tier"] == "tier1b" and rec["gate"]["ok"] and rec["stats"]["table_sites"] == 1
    assert (
        rec["registers_from"] == "ptxas"
    )  # a PTX image has no shipped cubin to read the count from
    if cache._has_tool("ptxas"):
        assert rec["registers"] >= 1 and rec["maxnreg"] == min((rec["registers"] + 7) // 8 * 8, 255)
    else:
        assert rec["registers"] is None and rec["maxnreg"] is None
    bad_rec = out[cache.image_hash(bad)]["kernels"]["k"]
    assert bad_rec["file"] is None and not bad_rec["gate"]["ok"]
    sass = cache.image_hash(dump / "sass_only.cubin")
    assert out[sass]["unsupported"].startswith("SASS only")
    index = json.loads((tmp_path / "cache" / "index.json").read_text())
    assert set(index) == {ihash, cache.image_hash(bad), sass}


@pytest.mark.skipif(not cache._has_tool("ptxas"), reason="ptxas not installed")
def test_build_compiles_with_ptxas_and_pins_registers(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "image.ptx").write_text(TABLE_KERNEL)
    cache.build(dump, tmp_path / "cache", "^k$", roles_for, arch="sm_80", log=lambda *_: None)
    ihash = cache.image_hash(dump / "image.ptx")
    rec = json.loads((tmp_path / "cache" / ihash / "manifest.json").read_text())["kernels"]["k"]
    assert (
        rec["file"].endswith(".cubin")
        and (tmp_path / "cache" / ihash / rec["file"]).stat().st_size > 0
    )
    assert rec["registers_from"] == "ptxas" and rec["registers"] >= 1
    assert rec["maxnreg"] == min((rec["registers"] + 7) // 8 * 8, 255)
    assert ".maxnreg" in (tmp_path / "cache" / ihash / f"{cache.name_hash('k')}.ptx").read_text()


def test_tier1b_descriptor_layout():
    d = tier1b_descriptor(
        counter_ptr=0x1000,
        record_ptr=0x2000,
        table_row_bytes=256,
        table_rows=7,
        row_len_ptr=0x3000,
        block_size=16,
        num_blocks=4096,
    )
    assert len(d) == R.N_DESC
    assert d[R.D_ENABLE] == R.EN_TABLE and d[R.D_COUNTER] == 0x1000 and d[R.D_RECORD] == 0x2000
    assert d[R.D_TBL_ROW_BYTES] == 256 and d[R.D_TBL_ENTRY_LOG2] == 2 and d[R.D_TBL_NROWS] == 7
    assert (
        d[R.D_ROW_LEN] == 0x3000
        and d[R.D_BS_LOG2] == 4
        and d[R.D_BS_MASK] == 15
        and d[R.D_NBLOCKS] == 4096
    )
    assert (
        d[R.D_TBL_MODE] == 0 and d[R.D_STATE] == 0 and d[R.D_INIT] == 0
    )  # Tier 1b needs no shadow arrays
    assert (
        tier1b_descriptor(
            counter_ptr=1,
            record_ptr=2,
            table_row_bytes=4,
            table_rows=1,
            row_len_ptr=3,
            block_size=1,
            num_blocks=1,
            flat_slot_map=True,
            enabled=False,
        )[R.D_TBL_MODE]
        == 1
    )
    with pytest.raises(ValueError):
        tier1b_descriptor(
            counter_ptr=1,
            record_ptr=2,
            table_row_bytes=4,
            table_rows=1,
            row_len_ptr=3,
            block_size=12,
            num_blocks=1,
        )


TRITON_SOURCE = """
@triton.jit
def kernel_unified_attention(
    output_ptr,  # comment, with: colon
    key_cache_ptr,
    block_tables_ptr,
    scale: tl.constexpr,
    num_seqs,
    BLOCK_SIZE: tl.constexpr,
):
    pass
"""


def test_triton_arg_names_skip_constexprs_and_add_scratch_pointers():
    names = cache.triton_arg_names(TRITON_SOURCE, "kernel_unified_attention")
    assert names == [
        "output_ptr",
        "key_cache_ptr",
        "block_tables_ptr",
        "num_seqs",
        "global_scratch",
        "profile_scratch",
    ]
    with pytest.raises(KeyError):
        cache.triton_arg_names(TRITON_SOURCE, "other")


def test_triton_roles_match_parameters_positionally_and_reject_mismatches():
    roles = cache.TritonRoles(
        TRITON_SOURCE,
        "kernel_unified_attention",
        {"key_cache_ptr": KV, "block_tables_ptr": TABLE},
    )
    six = module(
        "    ret;",
        ".param .u64 a, .param .u64 b, .param .u64 c, .param .u32 d, .param .u64 e, .param .u64 f",
    )
    r = roles(parse_kernel(split_entries(six)["k"]))
    assert r["P:b+0"] == KV and r["P:c+0"] == TABLE and r["P:a+0"] not in (KV, TABLE)
    with pytest.raises(ValueError):
        roles(parse_kernel(split_entries(TABLE_KERNEL)["k"]))


def test_build_finds_triton_ptx_for_a_cubin_through_the_triton_cache(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    cubin = b"\x7fELF-triton-cubin-bytes"
    (dump / "deadbeef.cubin").write_bytes(cubin)
    tcache = tmp_path / "triton-cache"
    entry = tcache / "HASH"
    entry.mkdir(parents=True)
    (entry / "k.cubin").write_bytes(cubin)
    (entry / "k.ptx").write_text(TABLE_KERNEL)
    (tcache / "OTHER").mkdir()
    (tcache / "OTHER" / "k.cubin").write_bytes(b"\x7fELF-other")
    assert cache.triton_ptx_for_cubin(dump / "deadbeef.cubin", tcache) == entry / "k.ptx"
    out = cache.build(
        dump,
        tmp_path / "cache",
        "^k$",
        roles_for,
        run_ptxas=False,
        triton_cache=tcache,
        log=lambda *_: None,
    )
    rec = out[cache.image_hash(dump / "deadbeef.cubin")]["kernels"]["k"]
    assert rec["file"] and rec["gate"]["ok"]
    out2 = cache.build(
        dump, tmp_path / "cache2", "^k$", roles_for, run_ptxas=False, log=lambda *_: None
    )
    assert out2[cache.image_hash(dump / "deadbeef.cubin")]["unsupported"].startswith("SASS only")


TTIR = """
module {
  tt.func public @kernel_unified_attention(%output_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %key_cache_ptr: !tt.ptr<bf16>, %block_tables_ptr: !tt.ptr<i32>, %num_seqs: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    tt.return
  }
}
"""


def test_triton_arg_names_from_the_ir_signature(tmp_path):
    assert cache.triton_arg_names_from_ttir(TTIR, "kernel_unified_attention") == [
        "output_ptr",
        "key_cache_ptr",
        "block_tables_ptr",
        "num_seqs",
        "global_scratch",
        "profile_scratch",
    ]
    with pytest.raises(KeyError):
        cache.triton_arg_names_from_ttir(TTIR, "other")
    # the IR next to the PTX wins over the Python definition
    roles = cache.TritonRoles(
        TRITON_SOURCE, "kernel_unified_attention", {"key_cache_ptr": KV, "block_tables_ptr": TABLE}
    )
    ptx = tmp_path / "k.ptx"
    ptx.write_text("")
    (tmp_path / "k.ttir").write_text(TTIR)
    six = module(
        "    ret;",
        ".param .u64 a, .param .u64 b, .param .u64 c, .param .u32 d, .param .u64 e, .param .u64 f",
    )
    r = roles(parse_kernel(split_entries(six)["k"]), ptx)
    assert r["P:b+0"] == KV and r["P:c+0"] == TABLE


def test_struct_roles_assign_kv_and_table_by_field_offset():
    from kvguard.gpu.ptx.roles import FLASHINFER_BATCH_PREFILL_PAGED

    offsets = "q 0\npaged_kv.k_data 64\npaged_kv.v_data 72\npaged_kv.indices 80\npaged_kv.indptr 88\nsizeof 384\n"
    roles = cache.StructRoles(offsets, FLASHINFER_BATCH_PREFILL_PAGED)
    k = parse_kernel(split_entries(module("    ret;", ".param .align 8 .b8 params[384]"))["k"])
    r = roles(k)
    assert r["P:params+64"] == KV and r["P:params+72"] == KV and r["P:params+80"] == TABLE
    assert r["P:params+88"] == "ptr" and r["P:params+0"] == "ptr" and r["P:params+96"] == "scalar"


# A page offset that is the table entry on one path and zero on the other (the kernel's
# fallback for positions beyond the row): not table-derived on every path, but table-derived or
# page 0 on every path.
ZERO_FALLBACK_KERNEL = module(
    """
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mov.u64 %rd5, 0;
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $DONE;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
$DONE:
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    ret;""",
    ".param .u64 kv, .param .u64 table, .param .u32 n",
)


def test_gate_names_the_zero_fallback_when_allowed():
    from kvguard.gpu.ptx import analyze_kernel, summarize

    k = parse_kernel(split_entries(ZERO_FALLBACK_KERNEL)["k"])
    res, extra = analyze_kernel(k, roles_for(k), {})
    summary = summarize(res, extra)
    assert summary.kv_accesses == 1 and summary.kv_must_table == 0
    assert summary.kv_must_table_or_zero == 1
    ok, why = cache.gate(summary)
    assert not ok
    ok, why = cache.gate(summary, allow_zero_fallback=True)
    assert ok and "page-0 fallback" in why


def test_recompiled_control_tier_inserts_no_check(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "image.ptx").write_text(TABLE_KERNEL)
    out = cache.build(
        dump,
        tmp_path / "cache",
        "^k$",
        roles_for,
        tier="recompiled",
        run_ptxas=False,
        log=lambda *_: None,
    )
    rec = out[cache.image_hash(dump / "image.ptx")]["kernels"]["k"]
    twin = (tmp_path / "cache" / cache.image_hash(dump / "image.ptx") / rec["file"]).read_text()
    assert "__kvg_desc" not in twin and "atom.global.add" not in twin
    assert rec["stats"]["control"] == "recompiled original" and rec["gate"]["ok"]
