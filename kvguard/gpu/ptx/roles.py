"""Parameter roles: which kernel parameters are KV pools, index tables, other pointers or scalars.

Roles are the one framework-specific input to the analysis, expressed on the provenance token form
``P:<param>+<off>``: one parameter per argument uses offset 0; a single parameter struct (FA2's
``Flash_fwd_params``) is keyed by field offset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .parse import Kernel
from .provenance import Roles

KV = "kv"
TABLE = "table"
PTR = "ptr"
SCALAR = "scalar"


def roles_by_name(kernel: Kernel, names: Iterable[str], role_of: Mapping[str, str]) -> Roles:
    """Roles for a kernel whose PTX parameters correspond one-to-one, in order, to ``names``.
    Names absent from ``role_of`` are scalars; extra PTX parameters beyond ``names`` are scalars."""
    names = list(names)
    out: Roles = {}
    for i, p in enumerate(kernel.params):
        n = names[i] if i < len(names) else f"extra{i}"
        out[f"P:{p}+0"] = role_of.get(n, SCALAR)
    return out


def roles_by_struct(
    kernel: Kernel,
    field_offsets: Mapping[int, str],
    role_of: Mapping[str, str],
    size: int,
    step: int = 4,
) -> Roles:
    """Roles for a kernel whose single parameter is a struct: ``field_offsets`` maps byte offsets
    to field names (from compiling ``offsetof`` for the pinned source), ``size`` is the struct size."""
    if not kernel.params:
        return {}
    pname = kernel.params[0]
    out: Roles = {}
    for off in range(0, size + step, step):
        field = field_offsets.get(off)
        out[f"P:{pname}+{off}"] = role_of.get(field, SCALAR) if field else SCALAR
    return out


def parse_offsets(text: str) -> tuple[dict[int, str], int]:
    """Parse ``<field> <offset>`` lines; the ``sizeof`` line gives the struct size."""
    offsets: dict[int, str] = {}
    size = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        name, value = parts
        if name == "sizeof":
            size = int(value)
        else:
            offsets[int(value)] = name
    return offsets, size


# Role maps for some kernel families, as examples and test fixtures. Adapters own the role maps
# for the kernels they bind.
VLLM_RESHAPE_AND_CACHE_FLASH = (
    ["key", "value", "key_cache", "value_cache", "slot_mapping", "block_stride", "page_stride", "head_stride",
     "key_stride", "value_stride", "num_heads", "head_size", "block_size", "k_scale", "v_scale", "kv_scale_stride"],
    {"key": PTR, "value": PTR, "key_cache": KV, "value_cache": KV, "slot_mapping": TABLE, "k_scale": PTR, "v_scale": PTR},
)  # fmt: skip

FA2_FLASH_FWD_PARAMS = {
    "k_ptr": KV, "v_ptr": KV, "block_table": TABLE,
    "q_ptr": PTR, "o_ptr": PTR, "oaccum_ptr": PTR, "p_ptr": PTR, "softmax_lse_ptr": PTR,
    "softmax_lseaccum_ptr": PTR, "cu_seqlens_q": PTR, "cu_seqlens_k": PTR, "leftpad_k": PTR,
    "seqused_k": PTR, "blockmask": PTR, "knew_ptr": PTR, "vnew_ptr": PTR, "rotary_cos_ptr": PTR,
    "rotary_sin_ptr": PTR, "cache_batch_idx": PTR, "rng_state": PTR, "alibi_slopes_ptr": PTR,
}  # fmt: skip

# FlashInfer's JIT-generated PagedParams (SGLang's default attention backend): one struct
# parameter. The page list (`paged_kv.indices`) is the table; `paged_kv.indptr` and
# `paged_kv.last_page_len` are per-row descriptors the dispatch-time check binds (CSRPageList),
# not tables of KV addresses.
FLASHINFER_BATCH_PREFILL_PAGED = {
    "paged_kv.k_data": KV, "paged_kv.v_data": KV, "paged_kv.indices": TABLE,
    "q": PTR, "o": PTR, "lse": PTR, "q_indptr": PTR, "paged_kv.indptr": PTR, "paged_kv.last_page_len": PTR,
    "paged_kv.rope_pos_offset": PTR, "maybe_custom_mask": PTR, "maybe_mask_indptr": PTR,
    "maybe_alibi_slopes": PTR, "maybe_prefix_len_ptr": PTR, "maybe_token_pos_in_items_ptr": PTR,
    "maybe_max_item_len_ptr": PTR, "maybe_k_cache_sf": PTR, "maybe_v_cache_sf": PTR,
    "request_indices": PTR, "qo_tile_indices": PTR, "kv_tile_indices": PTR, "merge_indptr": PTR,
    "o_indptr": PTR, "block_valid_mask": PTR, "kv_chunk_size_ptr": PTR, "total_num_rows": PTR,
}  # fmt: skip

TRITON_UNIFIED_ATTENTION = {
    "key_cache_ptr": KV, "value_cache_ptr": KV, "block_tables_ptr": TABLE,
    "query_ptr": PTR, "output_ptr": PTR, "seq_lens_ptr": PTR, "query_start_len_ptr": PTR, "sink_ptr": PTR,
    "alibi_slopes_ptr": PTR, "qq_bias_ptr": PTR, "segm_output_ptr": PTR, "segm_max_ptr": PTR,
    "segm_expsum_ptr": PTR, "k_scale": PTR, "v_scale": PTR, "out_scale": PTR, "softcap": SCALAR,
    "mm_prefix_range_ptr": PTR, "rswa_prefix_lens_ptr": PTR, "k_scale_cache_ptr": PTR, "v_scale_cache_ptr": PTR,
    "global_scratch": PTR, "profile_scratch": PTR,
}  # fmt: skip

__all__ = [
    "FA2_FLASH_FWD_PARAMS",
    "FLASHINFER_BATCH_PREFILL_PAGED",
    "KV",
    "PTR",
    "SCALAR",
    "TABLE",
    "TRITON_UNIFIED_ATTENTION",
    "VLLM_RESHAPE_AND_CACHE_FLASH",
    "parse_offsets",
    "roles_by_name",
    "roles_by_struct",
]
