"""TTGIR must-provenance on hand-written kernels, with negative controls."""

from kvguard.gpu.ttgir import analyze_ttgir_text

KV = ["%key_cache_ptr", "%value_cache_ptr"]
TABLE = ["%block_tables_ptr"]

SIMPLE = """
module {
  tt.func public @k(%key_cache_ptr: !tt.ptr<bf16>, %value_cache_ptr: !tt.ptr<bf16>, %block_tables_ptr: !tt.ptr<i32>, %seq_lens_ptr: !tt.ptr<i32>) {
    %c16 = arith.constant 16 : i32
    %pid = tt.get_program_id x : i32
    %tptr = tt.addptr %block_tables_ptr, %pid : !tt.ptr<i32>, i32
    %blk = tt.load %tptr : !tt.ptr<i32>
    %off = arith.muli %blk, %c16 : i32
    %kptr = tt.addptr %key_cache_ptr, %off : !tt.ptr<bf16>, i32
    %k = tt.load %kptr : !tt.ptr<bf16>
    %sptr = tt.addptr %seq_lens_ptr, %pid : !tt.ptr<i32>, i32
    %len = tt.load %sptr : !tt.ptr<i32>
    %off2 = arith.muli %len, %c16 : i32
    %vptr = tt.addptr %value_cache_ptr, %off2 : !tt.ptr<bf16>, i32
    %v = tt.load %vptr : !tt.ptr<bf16>
    tt.return
  }
}
"""

STAGED = """
module {
  tt.func public @k(%key_cache_ptr: !tt.ptr<bf16>, %block_tables_ptr: !tt.ptr<i32>) {
    %c16 = arith.constant 16 : i32
    %pid = tt.get_program_id x : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<4xi32>
    %tptr = tt.addptr %block_tables_ptr, %pid : !tt.ptr<i32>, i32
    %tok = ttg.async_copy_global_to_local %tptr, %buf : !tt.ptr<i32> -> <4xi32>
    %w = ttg.async_wait %tok {num = 0 : i32}
    %blk = ttg.local_load %buf : !ttg.memdesc<4xi32> -> i32
    %off = arith.muli %blk, %c16 : i32
    %kptr = tt.addptr %key_cache_ptr, %off : !tt.ptr<bf16>, i32
    %k = tt.load %kptr : !tt.ptr<bf16>
    tt.return
  }
}
"""


def test_direct_and_non_table_kv_accesses():
    r = analyze_ttgir_text(SIMPLE, KV, ["%block_tables_ptr"])
    assert r.kv_accesses == 2 and r.kv_must_table == 1 and r.table_loads == 1
    assert not r.complete


def test_shared_memory_staging_is_must_at_ttgir_level():
    r = analyze_ttgir_text(STAGED, ["%key_cache_ptr"], TABLE)
    assert r.kv_accesses == 1 and r.kv_must_table == 1 and r.complete


def test_wrong_table_argument_is_a_negative_control():
    r = analyze_ttgir_text(SIMPLE, KV, ["%seq_lens_ptr"])
    assert (
        r.kv_accesses == 2 and r.kv_must_table == 1
    )  # only the seq_lens-derived access now counts
