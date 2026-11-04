"""The PTX provenance analysis on small hand-written kernels: one test per analysis feature plus
the negative controls."""

from kvguard.gpu.ptx import KV, PTR, TABLE, analyze_module, roles_by_name

HEADER = ".version 8.0\n.target sm_80\n.address_size 64\n"


def module(body: str, params: str) -> str:
    return f"{HEADER}.visible .entry k({params})\n{{\n.reg .b64 %rd<32>;\n.reg .b32 %r<32>;\n.reg .pred %p<8>;\n{body}\n}}\n"


def roles(**role_of):
    def roles_for(kernel):
        return roles_by_name(kernel, [p for p in kernel.params], role_of)

    return roles_for


def one(ptx, roles_for):
    (summary,) = analyze_module(ptx, roles_for)
    return summary


def test_store_through_table_entry_is_table_derived_on_every_path():
    ptx = module(
        """
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        mov.u32 %r1, %tid.x;
        mul.wide.u32 %rd3, %r1, 4;
        add.s64 %rd4, %rd2, %rd3;
        ld.global.u32 %r2, [%rd4];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        st.global.u32 [%rd6], %r1;
        ret;""",
        ".param .u64 kv, .param .u64 table",
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.kv_accesses == 1 and s.kv_must_table == 1 and s.unknown_base == 0
    assert s.categories == {"table_load": 1, "kv_table": 1}
    assert s.complete


def test_non_table_index_and_non_kv_access_are_not_misclassified():
    ptx = module(
        """
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [other];
        mov.u32 %r1, %tid.x;
        mul.wide.u32 %rd3, %r1, 4;
        add.s64 %rd4, %rd2, %rd3;
        ld.global.u32 %r2, [%rd4];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        st.global.u32 [%rd6], %r1;
        st.global.u32 [%rd4], %r1;
        ret;""",
        ".param .u64 kv, .param .u64 other",
    )
    s = one(ptx, roles(kv=KV, other=PTR))
    assert s.categories == {"non_kv": 2, "kv_other_index": 1}
    assert s.kv_must_table == 0 and not s.complete


def test_nullness_specialization_prunes_the_unpaged_branch():
    """FA2's shape: `block_table == nullptr` selects a non-paged address; in a paged launch the
    table is non-null, so the branch is dead and every path is table-derived."""
    ptx = module(
        """
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        mov.u32 %r1, %tid.x;
        setp.eq.s64 %p1, %rd2, 0;
        @%p1 bra L_unpaged;
        ld.global.u32 %r2, [%rd2];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        bra L_store;
L_unpaged:
        mul.wide.u32 %rd7, %r1, 256;
        add.s64 %rd6, %rd1, %rd7;
L_store:
        st.global.u32 [%rd6], %r1;
        ret;""",
        ".param .u64 kv, .param .u64 table",
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.kv_accesses == 1 and s.kv_must_table == 1 and s.folded_preds >= 1
    # Without the table marked as a pointer that must be non-null, the unpaged path stays live.
    s2 = one(ptx, roles(kv=KV, table=PTR))
    assert s2.kv_must_table == 0


def test_loop_carried_pointer_stays_derived():
    ptx = module(
        """
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        ld.global.u32 %r2, [%rd2];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        mov.u32 %r3, 0;
L_loop:
        st.global.u32 [%rd6], %r3;
        add.s64 %rd6, %rd6, 16;
        add.s32 %r3, %r3, 1;
        setp.lt.s32 %p1, %r3, 8;
        @%p1 bra L_loop;
        ret;""",
        ".param .u64 kv, .param .u64 table",
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.kv_accesses == 1 and s.kv_must_table == 1


def test_choice_between_derived_and_underived_address_is_not_must():
    ptx = module(
        """
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        ld.param.u32 %r4, [flag];
        mov.u32 %r1, %tid.x;
        ld.global.u32 %r2, [%rd2];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        mul.wide.u32 %rd7, %r1, 256;
        add.s64 %rd8, %rd1, %rd7;
        setp.ne.s32 %p1, %r4, 0;
        selp.b64 %rd9, %rd6, %rd8, %p1;
        st.global.u32 [%rd9], %r1;
        ret;""",
        ".param .u64 kv, .param .u64 table, .param .u32 flag",
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.kv_accesses == 1 and s.kv_table_derived == 1 and s.kv_must_table == 0


def test_shared_memory_staging_is_may_only_at_ptx_level():
    """Triton stages block-table entries through shared memory with cp.async; at PTX level the
    dependence through the shared pool is only "may" (the TTGIR analysis proves "must")."""
    ptx = module(
        """
        .shared .align 16 .b8 smem[64];
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        mov.u32 %r1, %tid.x;
        mov.u64 %rd10, smem;
        cp.async.ca.shared.global [%rd10], [%rd2], 4;
        cp.async.commit_group;
        cp.async.wait_group 0;
        ld.shared.u32 %r2, [smem];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        st.global.u32 [%rd6], %r1;
        ret;""",
        ".param .u64 kv, .param .u64 table",
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.kv_accesses == 1 and s.kv_table_derived == 1  # may: the entry passed through the pool
    assert s.kv_must_table == 0  # not must: the shared pool merges every staged value


def test_callee_parameters_are_seeded_from_the_caller():
    """32 of the 80 FA2 instantiations do their accesses in a .func callee that receives the
    parameter struct by reference."""
    ptx = (
        HEADER
        + """
.func body(.param .b64 body_param_0, .param .b64 body_param_1)
{
.reg .b64 %rd<16>;
.reg .b32 %r<8>;
        ld.param.u64 %rd1, [body_param_0];
        ld.param.u64 %rd2, [body_param_1];
        mov.u32 %r1, %tid.x;
        ld.global.u32 %r2, [%rd2];
        mul.wide.u32 %rd5, %r2, 256;
        add.s64 %rd6, %rd1, %rd5;
        st.global.u32 [%rd6], %r1;
        ret;
}
.visible .entry k(.param .u64 kv, .param .u64 table)
{
.reg .b64 %rd<16>;
        ld.param.u64 %rd1, [kv];
        ld.param.u64 %rd2, [table];
        {
        .param .b64 param0;
        .param .b64 param1;
        st.param.b64 [param0], %rd1;
        st.param.b64 [param1], %rd2;
        call.uni body, (param0, param1);
        }
        ret;
}
"""
    )
    s = one(ptx, roles(kv=KV, table=TABLE))
    assert s.callees == 1
    assert s.kv_accesses == 1 and s.kv_must_table == 1


def test_access_with_no_pointer_parameter_base_is_unknown():
    ptx = module(
        """
        mov.u64 %rd1, 4096;
        mov.u32 %r1, %tid.x;
        st.global.u32 [%rd1], %r1;
        ret;""",
        ".param .u64 kv",
    )
    s = one(ptx, roles(kv=KV))
    assert s.unknown_base == 1 and not s.complete
