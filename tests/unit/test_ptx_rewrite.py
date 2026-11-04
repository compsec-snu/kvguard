"""The PTX rewriter: inserted checks parse and their statistics are as expected."""

import pytest
from kvguard.gpu.ptx import KV, TABLE, parse_kernel, roles_by_name, split_entries
from kvguard.gpu.ptx import rewrite as RW
from kvguard.gpu.ptx.rewrite import N_DESC, instrument

from tests.unit.test_ptx_analysis import module

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


def roles_for(ptx):
    k = parse_kernel(split_entries(ptx)["k"])
    return roles_by_name(k, k.params, {"kv": KV, "table": TABLE})


def cache_has_ptxas():
    from kvguard.gpu import cache

    return cache._has_tool("ptxas")


@pytest.mark.parametrize("tier", ["tier1b", "tier1", "tier2"])
def test_instrumented_module_parses_and_declares_the_descriptor(tier):
    kw = {
        "tier1b": dict(modes=("table",), bounds_only=True),
        "tier1": dict(modes=("table",)),
        "tier2": dict(modes=("table", "kv")),
    }[tier]
    out, stats = instrument(TABLE_KERNEL, "k", roles_for(TABLE_KERNEL), maxnreg=64, **kw)
    assert f".const .align 8 .b64 __kvg_desc[{N_DESC}];" in out
    assert ".maxnreg 64" in out
    k2 = parse_kernel(split_entries(out)["k"])
    assert len(k2.insts) > 11  # checks were inserted
    assert stats["table_sites"] == 1 and stats["kv_sites"] == 2
    if tier == "tier1b":
        assert "__kvg_desc+72" in out  # row length descriptor read by the bounds check
    if tier == "tier2":
        assert out.count("atom.global.add.u32") == 3  # one violation counter update per site


def test_checks_are_gated_by_the_descriptor_enable_word():
    for kw, bit in (
        (dict(modes=("table",), bounds_only=True), RW.EN_TABLE),
        (dict(modes=("table",), bounds_only=True, use_guarded=True), RW.EN_TABLE),
        (dict(modes=("table", "kv")), RW.EN_KV),
    ):
        text, _ = instrument(TABLE_KERNEL, "k", roles_for(TABLE_KERNEL), **kw)
        assert "ld.const.u64 %kgd0, [__kvg_desc+0];" in text
        assert f"and.b64 %kgd0, %kgd0, {bit};" in text
        assert "and.pred %kgp1, %kgp1, %kgp2;" in text


CALLEE_MODULE = """//
// Generated
//
.version 8.0
.target sm_80
.address_size 64

.func helper(.param .b64 helper_p)
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [helper_p];
    st.global.u32 [%rd1], 7;
    ret;
}

.visible .entry k(.param .u64 kv, .param .u64 table)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<4>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    {
    .param .b64 param0;
    st.param.b64 [param0], %rd1;
    call.uni helper, (param0);
    }
    ret;
}
"""


# nvcc's shape for a `const Params&` callee: the entry passes `cvta.param` of its parameter
# block; the callee reads the fields through that generic pointer.
PARAM_BLOCK_MODULE = """//
.version 8.0
.target sm_80
.address_size 64

.func body(.param .b64 body_p, .param .b32 body_n)
{
    .reg .b64 %rd<10>;
    .reg .b32 %r<6>;
    .reg .pred %p<3>;
    ld.param.u64 %rd1, [body_p];
    ld.param.u32 %r4, [body_n];
    ld.u64 %rd2, [%rd1+8];
    ld.u64 %rd7, [%rd1];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd7, %rd5;
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $SKIP;
    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
$SKIP:
    ret;
}

.visible .entry k(.param .align 8 .b8 params[24])
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<2>;
    mov.b64 %rd1, params;
    cvta.param.u64 %rd2, %rd1;
    ld.param.u32 %r1, [params+16];
    {
    .param .b64 param0;
    .param .b32 param1;
    st.param.b64 [param0], %rd2;
    st.param.b32 [param1], %r1;
    call.uni body, (param0, param1);
    }
    ret;
}
"""

PARAM_BLOCK_ROLES = {"P:params+0": KV, "P:params+8": TABLE}


def test_callee_checks_read_the_pointers_through_the_parameter_block():
    """All sites are in the callee: the table check loads the table pointer from the entry's
    parameter block through the callee's pointer parameter; the entry only gets the register
    declarations; site ids continue after the entry's instructions."""
    k = parse_kernel(split_entries(PARAM_BLOCK_MODULE)["k"])
    text, stats = instrument(
        PARAM_BLOCK_MODULE, "k", PARAM_BLOCK_ROLES, modes=("table",), bounds_only=True
    )
    assert stats["table_sites"] == 1 and stats["kv_sites"] == 2  # sites are counted in every mode
    assert set(stats["callees"]) == {"body"} and stats["site_base"] == {"body": len(k.insts)}
    assert stats["callees"]["body"]["table_sites"] == 1
    body = text[text.index(".func body(") : text.index(".visible .entry k(")]
    assert "ld.param.u64 %kgd2, [body_p];\nld.u64 %kgd2, [%kgd2+8];" in body
    assert "__kvg_desc" in body and "ld.global.u32 %r2, [%rd4];" in body
    entry = text[text.index(".visible .entry k(") :]
    assert "__kvg_desc" not in entry.split("{", 1)[1]  # no check in the entry
    kb = parse_kernel(
        split_entries(PARAM_BLOCK_MODULE.replace(".func body", ".entry body"))["body"]
    )
    table_idx = next(i.idx for i in kb.insts if i.text.startswith("ld.global.u32 %r2"))
    assert f"mov.u64 %kgd22, {len(k.insts) + table_idx};" in body  # the record's site id
    assert text.count(".func body(") == 1 and text.count(".entry k(") == 1


def test_callee_tier2_and_use_guarded_checks_stay_inside_the_callee():
    text, stats = instrument(PARAM_BLOCK_MODULE, "k", PARAM_BLOCK_ROLES)  # Tier 2
    body = text[text.index(".func body(") : text.index(".visible .entry k(")]
    assert stats["kv_sites"] == 2 and stats["callees"]["body"]["kv_sites"] == 2
    assert "ld.param.u64 %kgd2, [body_p];\nld.u64 %kgd2, [%kgd2+0];" in body  # the KV pool
    text, stats = instrument(
        PARAM_BLOCK_MODULE,
        "k",
        PARAM_BLOCK_ROLES,
        modes=("table",),
        bounds_only=True,
        use_guarded=True,
    )
    body = text[text.index(".func body(") : text.index(".visible .entry k(")]
    assert stats["guarded_table_sites"] == 1 and stats["guarded_kv_sites"] == 2
    assert ".reg .b32 %kgm<1>;" in body and "mov.b32 %kgm0, 0;" in body
    assert "%kgm" not in text[text.index(".visible .entry k(") :]


# The entry passes the KV pool pointer and a per-thread *row* pointer into the table: the callee
# can recover neither the table base nor the row from its parameters.
UNBOUND_MODULE = """//
.version 8.0
.target sm_80
.address_size 64

.func body(.param .b64 body_kv, .param .b64 body_row)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [body_kv];
    ld.param.u64 %rd4, [body_row];
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    ret;
}

.visible .entry k(.param .u64 kv, .param .u64 table)
{
    .reg .b64 %rd<6>;
    .reg .b32 %r<2>;
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    {
    .param .b64 param0;
    .param .b64 param1;
    st.param.b64 [param0], %rd1;
    st.param.b64 [param1], %rd4;
    call.uni body, (param0, param1);
    }
    ret;
}
"""


def test_callee_without_a_pointer_binding_is_unsupported():
    with pytest.raises(RW.UnsupportedCallee, match="body: no call argument binds P:table\\+0"):
        instrument(
            UNBOUND_MODULE, "k", roles_for(UNBOUND_MODULE), modes=("table",), bounds_only=True
        )
    # Tier 2 reaches the KV pool through its own parameter before the table check is emitted,
    # and still fails at the table site.
    with pytest.raises(RW.UnsupportedCallee):
        instrument(UNBOUND_MODULE, "k", roles_for(UNBOUND_MODULE))


@pytest.mark.skipif(not cache_has_ptxas(), reason="ptxas not installed")
def test_callee_instrumentation_compiles_with_ptxas(tmp_path):
    """ptxas accepts the generic loads through the parameter-block pointer inside the callee."""
    import subprocess

    from kvguard.gpu import cache

    for kw in (dict(modes=("table",), bounds_only=True, use_guarded=True), {}):
        text, _ = instrument(PARAM_BLOCK_MODULE, "k", PARAM_BLOCK_ROLES, **kw)
        src = tmp_path / "callee.ptx"
        src.write_text(text)
        r = subprocess.run(
            [cache._tool("ptxas"), "-arch=sm_80", str(src), "-o", str(tmp_path / "callee.cubin")],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr


def test_instrumented_module_keeps_non_inlined_callees():
    """A module whose entry calls a `.func` keeps the callee's definition, else ptxas rejects the
    call ("requires call prototype"); the instrumented entry replaces the original in place."""
    k = parse_kernel(split_entries(CALLEE_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, stats = instrument(CALLEE_MODULE, "k", roles, modes=("table",), bounds_only=True)
    assert text.count(".func helper(") == 1 and "st.global.u32 [%rd1], 7;" in text
    assert text.index(".func helper(") < text.index(".entry k(")  # original order kept
    assert text.count(".entry k(") == 1 and "__kvg_desc" in text and stats["table_sites"] == 1


GUARDED_MODULE = """//
.version 8.0
.target sm_80
.address_size 64

.visible .entry k(.param .u64 kv, .param .u64 table, .param .u32 n)
{
    .reg .b64 %rd<10>;
    .reg .b32 %r<6>;
    .reg .pred %p<3>;
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mul.wide.u32 %rd5, %r2, 256;
    add.s64 %rd6, %rd1, %rd5;
    setp.ge.u32 %p1, %r1, %r4;
    @%p1 bra $SKIP;
    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
$SKIP:
    ret;
}
"""


def test_use_guarded_tier1b_reports_at_the_consuming_accesses():
    """The table entry is loaded for every thread but only threads below n dereference it: the
    bounds predicate is kept in a flag at the load and reported where the accesses execute."""
    k = parse_kernel(split_entries(GUARDED_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    plain, _st0 = instrument(GUARDED_MODULE, "k", roles, modes=("table",), bounds_only=True)
    text, st = instrument(
        GUARDED_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=True
    )
    assert st["table_sites"] == 1 and st["kv_sites"] == 2
    assert st["guarded_table_sites"] == 1 and st["inexact_table_sites"] == 0
    assert st["guarded_kv_sites"] == 2 and st["unguarded_kv_sites"] == 0
    assert ".reg .b32 %kgm<1>;" in text and "mov.b32 %kgm0, 0;" in text
    assert text.count("selp.b32 %kgr14, 1, 0, %kgp9;") == 1  # the flag set at the table site
    # the load and the store sit in one basic block with the same sources: one guard group
    assert st["guard_groups"] == 1
    assert text.count("and.b32 %kgr15, %kgm0, 0x1;") == 1
    assert text.count("atom.global.add.u32") == 1 and plain.count("atom.global.add.u32") == 1
    assert "$KVG_SKIP_" in text
    body = text[text.index("{") :]
    assert body.index("mov.b32 %kgm0, 0;") < body.index("ld.param.u64")
    assert body.index("selp.b32 %kgr14") < body.index("and.b32 %kgr15, %kgm0")
    assert "__kvg_desc" in text and text.count(".entry k(") == 1


# A chain value (%rd5, from the first table load) is consumed after the table load ran again in
# the loop body: the flag would describe the second load, so the site must stay on the
# immediate check.
INEXACT_MODULE = """//
.version 8.0
.target sm_80
.address_size 64

.visible .entry k(.param .u64 kv, .param .u64 table, .param .u32 n)
{
    .reg .b64 %rd<12>;
    .reg .b32 %r<8>;
    .reg .pred %p<3>;
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    ld.param.u32 %r4, [n];
    mov.u32 %r1, 0;
    mov.u64 %rd5, 0;
$LOOP:
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    mul.wide.u32 %rd5, %r2, 256;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, %r4;
    @%p1 bra $LOOP;
    ret;
}
"""


def test_use_guarded_keeps_the_immediate_check_for_an_inexact_site():
    k = parse_kernel(split_entries(INEXACT_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, st = instrument(
        INEXACT_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=True
    )
    assert st["table_sites"] == 1 and st["kv_sites"] == 1
    assert st["inexact_table_sites"] == 1 and st["guarded_table_sites"] == 0
    assert st["guarded_kv_sites"] == 0 and st["unguarded_kv_sites"] == 1
    assert "%kgm0" not in text  # no flag register, no guard
    assert text.count("atom.global.add.u32") == 1  # the immediate table-site report remains


# The two accesses are separated by a branch (different basic blocks): two guard groups.
SPLIT_MODULE = GUARDED_MODULE.replace(
    """    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
$SKIP:""",
    """    ld.global.u32 %r3, [%rd6];
    setp.eq.u32 %p2, %r3, 0;
    @%p2 bra $SKIP;
    st.global.u32 [%rd6+128], %r3;
$SKIP:""",
)


def test_use_guarded_groups_are_split_at_basic_block_boundaries():
    k = parse_kernel(split_entries(SPLIT_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, st = instrument(
        SPLIT_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=True
    )
    assert st["guarded_kv_sites"] == 2 and st["guard_groups"] == 2
    assert text.count("and.b32 %kgr15, %kgm0, 0x1;") == 2


# The table entry goes through shared memory before it addresses KV: no register chain reaches
# the access, so the table site must keep its immediate check instead of a flag nobody reads.
SMEM_MODULE = """//
.version 8.0
.target sm_80
.address_size 64

.visible .entry k(.param .u64 kv, .param .u64 table)
{
    .reg .b64 %rd<10>;
    .reg .b32 %r<6>;
    .shared .align 4 .b8 buf[64];
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    mov.u32 %r4, buf;
    st.shared.u32 [%r4], %r2;
    bar.sync 0;
    ld.shared.u32 %r5, [%r4];
    mul.wide.u32 %rd5, %r5, 256;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];
    st.global.u32 [%rd6+128], %r3;
    ret;
}
"""


def test_use_guarded_keeps_the_immediate_check_when_no_access_consumes_the_flag():
    k = parse_kernel(split_entries(SMEM_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, st = instrument(
        SMEM_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=True
    )
    assert st["table_sites"] == 1 and st["guarded_table_sites"] == 0
    assert st["unconsumed_table_sites"] == 1 and st["guarded_kv_sites"] == 0
    assert "%kgm0" not in text and text.count("atom.global.add.u32") == 1


# A two-dimensional table: the row base (table + row * row_bytes) is defined once, before the
# per-thread column; the row and its length are computed there, not at the load.
TABLE_2D_MODULE = module(
    """
    ld.param.u64 %rd1, [kv];
    ld.param.u64 %rd2, [table];
    ld.param.u32 %r1, [row];
    mul.wide.u32 %rd3, %r1, 256;
    add.s64 %rd4, %rd2, %rd3;
    mov.u32 %r2, %tid.x;
    mul.wide.u32 %rd5, %r2, 4;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.u32 %r3, [%rd6];
    mul.wide.u32 %rd7, %r3, 256;
    add.s64 %rd8, %rd1, %rd7;
    ld.global.u32 %r4, [%rd8];
    ret;""",
    ".param .u64 kv, .param .u64 table, .param .u32 row",
)


@pytest.mark.parametrize("use_guarded", [False, True])
def test_tier1b_hoists_the_row_computation_to_the_row_base_definition(use_guarded):
    k = parse_kernel(split_entries(TABLE_2D_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, st = instrument(
        TABLE_2D_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=use_guarded
    )
    assert st["table_sites"] == 1 and st["hoisted_table_sites"] == 1 and st["row_groups"] == 1
    assert ".reg .b32 %kgrow<1>;" in text and text.count("div.u32 %kgrow0, %kgr9, %kgr10;") == 1
    body = text[text.index("{") :]
    assert body.index("div.u32 %kgrow0") < body.index("ld.global.u32 %r3, [%rd6];")
    after_load = body[body.index("ld.global.u32 %r3, [%rd6];") :]
    assert "div.u32" not in after_load and "%kgrlen0" in after_load
    assert "ld.global.u32 %kgrlen0, [%kgd11];" in text  # the row length, read once


def test_descriptor_in_registers_reads_the_constant_bank_once_per_thread():
    k = parse_kernel(split_entries(TABLE_2D_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    text, st = instrument(
        TABLE_2D_MODULE,
        "k",
        roles,
        modes=("table",),
        bounds_only=True,
        use_guarded=True,
        desc_in_regs=True,
    )
    body = text[text.index("{") :]
    checks = body[body.index("ld.param.u64 %rd1, [kv];") :]
    stripped = checks.replace("ld.const.u64 %kgd20", "").replace("ld.const.u64 %kgd21", "")
    assert "ld.const.u64 %kgd" not in stripped  # only the report path reads the constant bank
    assert "mov.b64 %kgd0, %kgc0;" in text  # the enable gate from a register
    assert f".reg .b64 %kgc<{N_DESC}>;" in text
    assert text.count("ld.const.u64 %kgc0, [__kvg_desc+0];") == 1
    assert st["desc_slots_in_regs"] >= 5
    assert body.index("ld.const.u64 %kgc0") < body.index("ld.param.u64 %rd1, [kv];")
    plain, _ = instrument(TABLE_2D_MODULE, "k", roles, modes=("table",), bounds_only=True)
    assert "%kgc" not in plain


def test_guard_after_places_the_guard_behind_the_access():
    k = parse_kernel(split_entries(GUARDED_MODULE)["k"])
    roles = roles_by_name(k, k.params, {"kv": KV, "table": TABLE})
    before, _ = instrument(
        GUARDED_MODULE, "k", roles, modes=("table",), bounds_only=True, use_guarded=True
    )
    after, st = instrument(
        GUARDED_MODULE,
        "k",
        roles,
        modes=("table",),
        bounds_only=True,
        use_guarded=True,
        guard_after=True,
    )
    access = "ld.global.u32 %r3, [%rd6];"
    assert before.index("and.b32 %kgr15, %kgm0, 0x1;") < before.index(access)
    assert after.index("and.b32 %kgr15, %kgm0, 0x1;") > after.index(access)
    assert st["guard_groups"] == 1 and after.count("and.b32 %kgr15, %kgm0, 0x1;") == 1


def test_trace_mode_records_every_global_access_against_the_ranges():
    """The dynamic cross-check twin: every global access (table load, KV load, KV store) tests its
    address against the KV and table ranges of ``__kvg_trace`` and sets its site bit."""
    text, stats = instrument(TABLE_KERNEL, "k", roles_for(TABLE_KERNEL), modes=("trace",))
    assert stats["trace_sites"] == 3 and stats["trace_categories"] == {
        "table_load": 1,
        "kv_table": 2,
    }
    assert f".const .align 8 .b64 __kvg_trace[{RW.N_TRACE}];" in text
    assert text.count("atom.global.or.b32") == 6  # two bitmaps per site
    assert (
        "__kvg_desc" in text and "KIND" not in text and "st.global.v2.u64" not in text
    )  # no check
    k = parse_kernel(split_entries(TABLE_KERNEL)["k"])
    kv_store = next(i.idx for i in k.insts if i.text.startswith("st.global.u32"))
    assert f"mov.b32 %kgr1, {1 << (kv_store % 32)};" in text


@pytest.mark.skipif(not cache_has_ptxas(), reason="ptxas not installed")
def test_trace_twin_compiles_with_ptxas(tmp_path):
    import subprocess

    from kvguard.gpu import cache

    text, _ = instrument(PARAM_BLOCK_MODULE, "k", PARAM_BLOCK_ROLES, modes=("trace",))
    src = tmp_path / "trace.ptx"
    src.write_text(text)
    r = subprocess.run(
        [cache._tool("ptxas"), "-arch=sm_80", str(src), "-o", str(tmp_path / "trace.cubin")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
