"""Attach the core to a live, unmodified SGLang scheduler through run-time wrappers.

Runs inside the scheduler subprocess, installed by ``hook.py``; apart from its one class-level
patch (``Scheduler.run_event_loop``) every wrapper is an instance attribute on a live object.
It mediates the allocator, snapshots the scheduler before each batch launches, describes the
step's dispatches from the FlashInfer wrappers and ``out_cache_loc``, completes them at
``process_batch_result``, and flushes the record at ``on_idle``. Overlap scheduling is judged by
the core's stream-order rule (``KVGUARD_SGLANG_STREAM_EPOCHS=0`` shows host-order reports).
``KVGUARD_SGLANG_TWINS=1`` publishes the Tier 1b twins' descriptor per step and
``KVGUARD_SGLANG_GPU_VALIDATE=1`` runs the device validators. Newer serving-system features are
opt-in, each behind its own ``KVGUARD_SGLANG_*`` flag and support boundary.
"""

from __future__ import annotations

import cProfile
import dataclasses
import datetime
import importlib.metadata
import importlib.util
import json
import logging
import os
import pstats
import re
import sys
import threading
import time
import traceback
import types
import weakref
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kvguard.adapters.sglang.state import (
    CKPT_POOL,
    STATE_POOL,
    StateMediation,
    ckpt_geometry,
    state_geometry,
)
from kvguard.adapters.streams import current_stream_handle
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    DispatchDescriptor,
    ExternalCopy,
    ExternalFill,
    ExternalRead,
    ExternalWrite,
    Ledger,
    Reporter,
    RequestSnapshot,
    Snapshot,
)
from kvguard.core.violations import (
    OutcomeKind,
    Report,
    Unsupported,
    integration_error,
    unsupported,
)
from kvguard.gpu.validate import PADDING_ROW, report_results

from .translate import (
    DRAFT_POOL,
    POOL,
    SUPPORT_BOUNDARY,
    NodeFacts,
    RowFacts,
    RowSlotCache,
    _ints,
    _np_ints,
    allocated_slots,
    beam_member_rows,
    beam_pending_orphan_slots,
    build_decode_dispatches,
    build_dflash_materialize_dispatch,
    build_eagle_tree_draft_decode_dispatches,
    build_extend_dispatches,
    build_mapping_dispatches,
    build_ngram_copy_dispatch,
    build_snapshot,
    build_verify_dispatches,
    eagle_tree_workspace_positions,
    external_read_groups,
    external_slot_groups,
    free_count_from_allocator,
    free_slots_from_allocator,
    geometry_from_allocator,
    mediated_frees,
    nodes_from_tree,
    page_set,
    page_set_np,
    pages_of,
    pages_of_prefix,
    pre_step_computed,
    row_pages,
    rows_from_requests,
    wrap_allocator_clear,
)

ENV_OUT = "KVGUARD_SGLANG_OUT"
ENV_ENVMETA = "KVGUARD_SGLANG_ENVMETA"
ENV_EARLY_COMPLETE = "KVGUARD_SGLANG_EARLY_COMPLETE"
ENV_STREAM_EPOCHS = "KVGUARD_SGLANG_STREAM_EPOCHS"
ENV_STRICT = "KVGUARD_SGLANG_STRICT"
ENV_FAULT = "KVG_FAULT"
ENV_FAULT_STEP = "KVG_FAULT_STEP"
ENV_TWINS = "KVGUARD_SGLANG_TWINS"
ENV_TWIN_FAULT = "KVG_ENFORCE_FAULT"
ENV_GPU_VALIDATE = "KVGUARD_SGLANG_GPU_VALIDATE"
ENV_GPU_FAULT = "KVGUARD_SGLANG_GPU_FAULT"
ENV_TAGS = "KVGUARD_SGLANG_TAGS"
ENV_NVTX = "KVGUARD_SGLANG_NVTX"
"""The Compute Sanitizer baseline's NVTX memory registrations: the pool's per-layer K and V
tensors as heaps, every page a region per tensor. A baseline, not a KVGuard tier."""
ENV_GPU_FAULT_STEP = "KVGUARD_SGLANG_GPU_FAULT_STEP"
ENV_SYNC = "KVGUARD_SGLANG_SYNC"
ENV_PREKERNEL = "KVGUARD_SGLANG_PREKERNEL"
ENV_TABLES = "KVGUARD_SGLANG_TABLES"
ENV_ROW_CACHE = "KVGUARD_SGLANG_ROW_CACHE"
ENV_AUDIT_EVERY = "KVGUARD_SGLANG_AUDIT_EVERY"
ENV_QUEUED_ROWS = "KVGUARD_SGLANG_QUEUED_ROWS"
ENV_PROFILE = "KVGUARD_SGLANG_PROFILE"
ENV_PROFILE_STEPS = "KVGUARD_SGLANG_PROFILE_STEPS"
ENV_NGRAM = "KVGUARD_SGLANG_NGRAM"
ENV_EAGLE3 = "KVGUARD_SGLANG_EAGLE3"
ENV_MULTI_LAYER_EAGLE = "KVGUARD_SGLANG_MULTI_LAYER_EAGLE"
ENV_NEXTN = "KVGUARD_SGLANG_NEXTN"
ENV_DFLASH = "KVGUARD_SGLANG_DFLASH"
ENV_HICACHE = "KVGUARD_SGLANG_HICACHE"
ENV_EXTERNAL_LINKER = "KVGUARD_SGLANG_EXTERNAL_LINKER"
ENV_BEAM = "KVGUARD_SGLANG_BEAM"
ENV_DISAGG_PREFILL = "KVGUARD_SGLANG_DISAGG_PREFILL"
ENV_DISAGG_DECODE = "KVGUARD_SGLANG_DISAGG_DECODE"
ENV_PD_EXTENT_DIR = "KVGUARD_SGLANG_PD_EXTENT_DIR"
ENV_LIVE_NDJSON = "KVGUARD_SGLANG_LIVE_NDJSON"
"""Optional path. When set, each report is appended as one JSON line as it is emitted (the
``Reporter.sink`` callback) so an out-of-process reader can stream reports live; independent of
the final ``KVGUARD_SGLANG_OUT`` record."""
ENV_PDMUX = "KVGUARD_SGLANG_PDMUX"
SWA_POOL = "swa_kv"


def multi_draft_pool(index: int) -> str:
    return DRAFT_POOL if index == 0 else f"{DRAFT_POOL}_{index}"


FLUSH_INTERVAL_S = 30.0
"""The periodic record flush interval; the record is large JSON, so a short period costs
scheduler-thread time. The idle and exit flushes write it in full anyway."""
TRACE_LIMIT = 600
"""Per-step trace entries kept in the record (seq, mode, batch size, padding, graph, tokens,
tree slots, reports so far)."""
CONFIG_KEYS = (
    "attention_backend", "prefill_attention_backend", "decode_attention_backend", "page_size",
    "kv_cache_dtype", "dtype", "disable_cuda_graph", "disable_radix_cache", "chunked_prefill_size",
    "mem_fraction_static", "max_total_tokens", "max_running_requests", "tp_size", "dp_size",
    "pp_size", "enable_hierarchical_cache", "disable_overlap_schedule", "speculative_algorithm",
    "speculative_num_draft_tokens", "speculative_num_steps", "speculative_eagle_topk",
    "speculative_adaptive",
    "disaggregation_mode", "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill",
    "enable_pdmux", "pdmux_config_path", "enable_unified_cache_external_linker",
    "unified_cache_external_linker_backend", "unified_cache_external_linker_config",
    "context_length", "random_seed", "mamba_track_interval", "enable_int8_mamba_checkpoint",
    "int8_mamba_ckpt_size", "enable_mixed_chunk", "mamba_full_memory_ratio", "max_mamba_cache_size",
)  # fmt: skip

log = logging.getLogger("kvguard.sglang")


def _full_backend(backend: Any) -> Any:
    """The backend whose wrappers hold the KV tables: a hybrid model's composite delegates the
    full-attention layers to ``full_attn_backend``."""
    return getattr(backend, "full_attn_backend", backend)


def _verify_width_supported(facts: dict[str, Any], draft: int) -> bool:
    """Whether a concrete verify width belongs to the admitted static/adaptive boundary."""
    configured = int(facts["speculative_num_draft_tokens"])
    return draft == configured or (
        bool(facts.get("speculative_adaptive")) and 0 < draft <= configured
    )


def _adaptive_runtime_graphs(draft_worker: Any, controller: Any, attribute: str) -> tuple[Any, ...]:
    """Return every distinct graph runner an adaptive EAGLE worker may activate.

    Adaptive EAGLE builds one immutable runtime state per candidate step count and swaps the
    worker's graph-runner attributes at run time. Wrapping only the initially active runner
    therefore misses real draft writes while another state is active.
    """

    candidates = [getattr(draft_worker, attribute, None)]
    state_attribute = {
        "cuda_graph_runner": "cuda_graph_runner",
        "cuda_graph_runner_for_draft_extend": "cuda_graph_runner_for_draft_extend",
    }[attribute]
    for state in getattr(controller, "_states", {}).values():
        candidates.append(getattr(state, state_attribute, None))

    unique: list[Any] = []
    seen: set[int] = set()
    for graph in candidates:
        if graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            unique.append(graph)
    return tuple(unique)


def _replay_full_backend(graph_runner: Any) -> Any:
    """The full-attention backend selected for this replay, including PDMUX's stream choice."""
    return _full_backend(graph_runner._replay_attn_backend())


def _say(msg: str) -> None:
    log.info(msg)
    sys.stderr.write(f"[kvguard-sglang] {msg}\n")
    sys.stderr.flush()


class _TransferReadyHook:
    """Run one KVGuard callback after the linker's concrete readiness waits.

    Wrapping the linker's ``synchronize``/``wait`` object preserves the framework's ordering while
    placing the ledger event at the boundary where the remote store can first touch the device
    pages. ``wait`` is forwarded though FULL-only TP=1 does not use it, to stay transparent.
    """

    def __init__(self, ready: Any, callback: Callable[[], None]):
        self._ready = ready
        self._callback = callback
        self._once = threading.Lock()
        self._called = False

    def _after_ready(self) -> None:
        with self._once:
            if self._called:
                return
            self._called = True
        self._callback()

    def synchronize(self) -> None:
        self._ready.synchronize()
        self._after_ready()

    def wait(self, stream: Any) -> None:
        self._ready.wait(stream)
        self._after_ready()


def _hicache_hybrid_host_mode_supported(
    *, hybrid: bool, unified_hybrid: bool, host_memory_mode: str | None
) -> bool:
    """Whether the live hybrid layout has an admitted HiCache host-pool implementation.

    Legacy hybrid pools remain bounded to file-backed staging. Current unified-memory hybrid
    pools build both attention and recurrent cache-mode host pools from one routed plan.
    """
    return not hybrid or unified_hybrid or host_memory_mode == "buffer_only"


def _pd_state_type_names(scheduler: Any, mode: str) -> list[str]:
    """State payload types published by the live P/D role, in transfer order."""
    if mode.upper() == "PREFILL":
        queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    elif mode.upper() == "DECODE":
        queue = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    else:
        return []
    manager = getattr(queue, "kv_manager", None)
    kv_args = getattr(manager, "kv_args", None)
    return [getattr(item, "name", str(item)) for item in getattr(kv_args, "state_types", ())]


def _pd_payload_ints(payload: Any) -> list[int]:
    """Flatten one transfer component's scalar/array/list indices."""
    if payload is None:
        return []
    tolist = getattr(payload, "tolist", None)
    if callable(tolist):
        return _pd_payload_ints(tolist())
    if isinstance(payload, (list, tuple)):
        return [item for value in payload for item in _pd_payload_ints(value)]
    return [int(payload)]


def _post_verify_graph_observation(graphs: Any, key: tuple) -> tuple[str, int, int]:
    """Classify one completed post-verify call without changing graph state."""
    states = graphs._graphs._states
    state = states.get(key)
    if state is None:
        return "state_limit", 0, len(states)
    slot = state.slots[int(state.turn) ^ 1]
    captured = sum(
        candidate is not None and candidate.graph is not None
        for graph_state in states.values()
        for candidate in graph_state.slots
    )
    return (
        "graph" if slot is not None and slot.graph is not None else "warmup",
        captured,
        len(states),
    )


def wrap_static_batch_loader(runner: Any, observe: Callable[[Any], None]) -> None:
    """Retain a runner's returned batch without changing its ``load_batch`` call shape."""
    orig_load = runner.load_batch

    def load_batch(forward_batch, *args, **kwargs):
        # PrefillCudaGraphRunner takes only keyword extras, while the eager runner selected by
        # --disable-prefill-cuda-graph also receives pp_proxy_tensors as a second positional
        # argument.  This wrapper is observational, so both signatures must pass through.
        static = orig_load(forward_batch, *args, **kwargs)
        observe(static)
        return static

    runner.load_batch = load_batch


# ------------------------------------------------------------------ support boundary


def check_support(scheduler: Any, reporter: Reporter) -> dict[str, Any]:
    """Facts of the live scheduler against ``translate.SUPPORT_BOUNDARY``; every excluded item
    is emitted as ``unsupported`` once all facts are read, and the first is raised, so the record
    of a rejected engine names every class the next adapter extension has to cover. Class names
    are compared exactly: a subclass (paged or hybrid allocator, HiRadixCache, quantized pool,
    hybrid backend) is a different path."""
    facts: dict[str, Any] = {}
    problems: list[Any] = []

    def reject(kind: str, detail: str) -> None:
        problems.append(unsupported(kind, detail, pool=POOL))

    args = scheduler.server_args
    alloc = scheduler.token_to_kv_pool_allocator
    unified_memory = bool(getattr(scheduler, "enable_unified_memory", False))
    facts["allocator"] = type(alloc).__name__
    facts["page_size"] = int(alloc.page_size)
    ps = facts["page_size"]
    unified_allocator = unified_memory and facts["allocator"] == "ChainAllocator"
    facts["unified_memory"] = unified_memory
    facts["unified_allocator"] = unified_allocator
    facts["paged"] = facts["allocator"] == "PagedTokenToKVPoolAllocator" or unified_allocator
    separate_swa = (
        facts["allocator"] == "SWATokenToKVPoolAllocator"
        and ps > 1
        and ps & (ps - 1) == 0
        and ps <= 64
    )
    facts["separate_pool_swa"] = separate_swa
    if not (
        (facts["allocator"] == "TokenToKVPoolAllocator" and ps == 1)
        or (
            facts["allocator"] == "PagedTokenToKVPoolAllocator"
            and ps > 1
            and ps & (ps - 1) == 0
            and ps <= 64
        )
        or (unified_allocator and ps > 1 and ps & (ps - 1) == 0 and ps <= 64)
        or separate_swa
    ):
        reject(
            "allocator",
            f"{facts['allocator']} with page_size {ps}; supported: TokenToKVPoolAllocator at "
            "page size 1, PagedTokenToKVPoolAllocator at a power of two up to 64, or the "
            "bounded unified-memory ChainAllocator, or the independent full/SWA allocator at "
            "the same page sizes",
        )
    kvcache = alloc.get_kvcache()
    facts["kv_pool"] = type(kvcache).__name__
    facts["kv_pool_size"] = int(getattr(kvcache, "size", 0) or 0)
    facts["kv_pool_size_swa"] = int(getattr(kvcache, "size_swa", 0) or 0)
    facts["kv_pool_full_side"] = type(getattr(kvcache, "full_kv_pool", None)).__name__
    facts["kv_pool_swa_side"] = type(getattr(kvcache, "swa_kv_pool", None)).__name__
    facts["kv_pool_full_layers"] = int(getattr(kvcache, "full_layer_nums", 0) or 0)
    facts["kv_pool_swa_layers"] = int(getattr(kvcache, "swa_layer_nums", 0) or 0)
    legacy_hybrid = facts["kv_pool"] == "HybridLinearKVPool"
    unified_hybrid = bool(
        unified_allocator
        and facts["kv_pool"] == "RoutedKVCache"
        and getattr(alloc, "mamba_allocator", None) is not None
        and getattr(alloc, "swa_attn_allocator", None) is None
    )
    hybrid = legacy_hybrid or unified_hybrid
    facts["legacy_hybrid"] = legacy_hybrid
    facts["unified_hybrid"] = unified_hybrid
    facts["hybrid"] = hybrid
    full_pool = getattr(kvcache, "full_kv_pool", None) if legacy_hybrid or separate_swa else kvcache
    facts["full_kv_pool"] = type(full_pool).__name__
    standalone_mla = facts["kv_pool"] == "MLATokenToKVPool"
    hybrid_mla = bool(hybrid and facts["full_kv_pool"] == "MLATokenToKVPool")
    mla = standalone_mla or hybrid_mla
    facts["standalone_mla"] = standalone_mla
    facts["hybrid_mla"] = hybrid_mla
    facts["mla"] = mla
    if facts["kv_pool"] not in ("MHATokenToKVPool", "MLATokenToKVPool") and not (
        (hybrid and facts["full_kv_pool"] in ("MHATokenToKVPool", "MLATokenToKVPool"))
        or (unified_hybrid and facts["full_kv_pool"] == "RoutedKVCache")
        or (
            separate_swa
            and facts["kv_pool"] == "SWAKVPool"
            and facts["kv_pool_full_side"] == "MHATokenToKVPool"
            and facts["kv_pool_swa_side"] == "MHATokenToKVPool"
            and facts["kv_pool_full_layers"] > 0
            and facts["kv_pool_swa_layers"] > 0
        )
    ):
        reject(
            "kv_pool",
            f"{facts['kv_pool']} (full layers: {facts['full_kv_pool']}); supported: "
            "MHATokenToKVPool, MLATokenToKVPool, HybridLinearKVPool over MHA, the bounded "
            "unified-memory RoutedKVCache with full-attention and recurrent-state bands, or "
            "SWAKVPool with independent MHA full/SWA sides",
        )
    tree = scheduler.tree_cache
    facts["tree_cache"] = type(tree).__name__
    facts["radix_cache_disabled"] = bool(getattr(tree, "disable", False))
    facts["hicache_port_enabled"] = os.environ.get(ENV_HICACHE, "0") == "1"
    facts["external_linker_port_enabled"] = os.environ.get(ENV_EXTERNAL_LINKER, "0") == "1"
    if facts["tree_cache"] == "UnifiedRadixCache":
        # The FULL-only unified tree keeps the RadixCache semantics the translation was verified
        # against; other components, an unadmitted HiCache controller, an external linker or
        # session caching are other paths.
        components = tuple(getattr(tree, "tree_components", ()))
        facts["tree_components"] = [getattr(c, "name", str(c)) for c in components]
        allowed = (
            [["FULL"], ["FULL", "MAMBA"]]
            if hybrid
            else ([["FULL", "SWA"]] if separate_swa else [["FULL"]])
        )
        if facts["tree_components"] not in allowed:
            reject(
                "prefix_cache",
                f"UnifiedRadixCache with components {facts['tree_components']}; supported: FULL, or FULL and MAMBA with a hybrid pool (docs/design/recurrent-state-pool.md)",
            )
        controller = getattr(tree, "cache_controller", None)
        decode_backup_controller = bool(
            controller is not None
            and os.environ.get(ENV_DISAGG_DECODE, "0") == "1"
            and getattr(
                scheduler.disaggregation_mode, "name", str(scheduler.disaggregation_mode)
            ).upper()
            == "DECODE"
            and not scheduler.enable_hierarchical_cache
        )
        facts["disagg_decode_backup_controller"] = decode_backup_controller
        if (
            controller is not None
            and not facts["hicache_port_enabled"]
            and not decode_backup_controller
        ):
            reject(
                "prefix_cache",
                "UnifiedRadixCache with a HiCache controller (set the exact-main HiCache port explicitly)",
            )
        linker_wrapper = getattr(tree, "linker", None)
        cache_linker = getattr(linker_wrapper, "cache_linker", None)
        facts["external_linker_wrapper"] = type(linker_wrapper).__name__
        facts["external_linker_backend"] = type(cache_linker).__name__
        facts["external_linker"] = linker_wrapper is not None
        if linker_wrapper is not None and not facts["external_linker_port_enabled"]:
            reject(
                "prefix_cache",
                "UnifiedRadixCache with an external linker (set the exact-main external-linker port explicitly)",
            )
        if linker_wrapper is not None and facts["external_linker_backend"] not in (
            "MooncakeDirectLinker",
            "UMBPDirectLinker",
        ):
            reject(
                "prefix_cache",
                f"UnifiedRadixCache external linker {facts['external_linker_backend']}; supported: MooncakeDirectLinker or UMBPDirectLinker",
            )
        facts["hicache_controller"] = type(controller).__name__
        facts["hicache_write_policy"] = getattr(controller, "write_policy", None)
        facts["hicache_io_backend"] = getattr(controller, "io_backend", None)
        facts["hicache_host_memory_mode"] = getattr(controller, "host_memory_mode", None)
        facts["hicache_storage"] = bool(getattr(controller, "enable_storage", False))
        facts["hicache_storage_backend"] = getattr(controller, "storage_backend_type", None)
        host_pool = getattr(controller, "mem_pool_host", None)
        storage_host_pool = getattr(controller, "storage_host_pool", None)
        storage_config = getattr(controller, "storage_config", None)
        facts["hicache_mem_layout"] = getattr(host_pool, "layout", None)
        facts["hicache_storage_mem_layout"] = getattr(storage_host_pool, "layout", None)
        facts["hicache_unified_storage_plan"] = bool(
            getattr(storage_config, "unified_suffixes", None)
        )
        facts["hicache_unified_storage_chunks"] = len(
            getattr(storage_config, "unified_suffixes", None) or ()
        )
        facts["hicache_host_parallel_alloc_threads"] = int(
            os.environ.get("SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS", "0") or "0"
        )
        if getattr(tree, "enable_session_radix_cache", False):
            reject("prefix_cache", "UnifiedRadixCache with session caching")
        facts["tree_core"] = type(getattr(tree, "tree_core", None)).__name__
    elif facts["tree_cache"] != "RadixCache":
        reject(
            "prefix_cache",
            f"{facts['tree_cache']}; supported: RadixCache or UnifiedRadixCache (FULL only)",
        )
    else:
        facts["external_linker_wrapper"] = "NoneType"
        facts["external_linker_backend"] = "NoneType"
        facts["external_linker"] = False
    facts["tree_strategy"] = tree_strategy(tree)
    facts["lock_refs_exact"] = facts["tree_strategy"] in ("radix_nodes", "python_nodes")  # fmt: skip
    facts["beam_search_port_enabled"] = os.environ.get(ENV_BEAM, "0") == "1"
    req_pool = scheduler.req_to_token_pool
    state_storage = getattr(alloc, "state", None) if unified_hybrid else None
    mamba_pool = (
        getattr(state_storage, "pool", None)
        if state_storage is not None
        else getattr(req_pool, "mamba_pool", None)
    )
    mamba_allocator = (
        getattr(state_storage, "slots", None)
        if state_storage is not None
        else getattr(req_pool, "mamba_allocator", None)
    )
    mamba_ckpt_pool = (
        getattr(state_storage, "ckpt_pool", None)
        if state_storage is not None
        else getattr(req_pool, "mamba_ckpt_pool", None)
    )
    facts["req_to_token_pool"] = type(req_pool).__name__
    facts["mamba_pool"] = type(mamba_pool).__name__
    facts["mamba_allocator"] = type(mamba_allocator).__name__
    facts["mamba_ckpt_pool"] = type(mamba_ckpt_pool).__name__
    facts["mamba_extra_buffer"] = bool(getattr(req_pool, "enable_mamba_extra_buffer", False))
    facts["mamba_replayssm"] = getattr(mamba_pool, "replayssm_write_pos", None) is not None
    facts["linear_layers"] = len(
        getattr(state_storage, "layer_map", None)
        if state_storage is not None
        else (getattr(req_pool, "mamba_map", None) or ())
    )
    if hybrid:
        legacy_state = facts["req_to_token_pool"] == "HybridReqToTokenPool" and (
            facts["mamba_pool"],
            facts["mamba_allocator"],
        ) == ("MambaPool", "MambaSlotAllocator")
        unified_state = (
            unified_hybrid
            and facts["req_to_token_pool"] == "RequestTable"
            and (facts["mamba_pool"], facts["mamba_allocator"])
            == ("StateView", "UnifiedMambaSlotAllocator")
        )
        if not (legacy_state or unified_state):
            reject(
                "req_to_token_pool",
                f"{facts['req_to_token_pool']} with {facts['mamba_pool']} and "
                f"{facts['mamba_allocator']}; supported with a hybrid pool: "
                "HybridReqToTokenPool with MambaPool and MambaSlotAllocator, or the bounded "
                "unified RequestTable with StateView and UnifiedMambaSlotAllocator",
            )
        if facts["mamba_ckpt_pool"] not in ("NoneType", "MambaCheckpointPool"):
            reject(
                "req_to_token_pool",
                f"checkpoint store {facts['mamba_ckpt_pool']}; supported: none, or MambaCheckpointPool",
            )
        if facts["mamba_replayssm"]:
            reject("req_to_token_pool", "ReplaySSM ring buffers (not modelled)")
    elif facts["req_to_token_pool"] != "ReqToTokenPool":
        decode_pool = (
            facts["req_to_token_pool"] == "DecodeReqToTokenPool"
            and os.environ.get(ENV_DISAGG_DECODE, "0") == "1"
            and getattr(
                scheduler.disaggregation_mode, "name", str(scheduler.disaggregation_mode)
            ).upper()
            == "DECODE"
        )
        if not decode_pool:
            reject(
                "req_to_token_pool",
                f"{facts['req_to_token_pool']}; supported: ReqToTokenPool, or the opt-in "
                "P/D decode port's DecodeReqToTokenPool",
            )
    backend = scheduler.tp_worker.model_runner.attn_backend
    facts["attention_backend"] = type(backend).__name__
    full_backend = getattr(backend, "full_attn_backend", backend)
    facts["full_attn_backend"] = type(full_backend).__name__
    facts["linear_attn_backend"] = type(getattr(backend, "linear_attn_backend", None)).__name__
    facts["linear_extend_captured_in_prefill_graph"] = bool(
        getattr(backend, "use_captured_forward_metadata_for_breakable_cuda_graph", False)
    )
    facts["num_wrappers"] = int(getattr(full_backend, "num_wrappers", 0))
    facts["sliding_window_size"] = getattr(
        scheduler.tp_worker.model_runner, "sliding_window_size", None
    )
    facts["sliding_window_kv_pool"] = bool(
        getattr(full_backend, "use_sliding_window_kv_pool", False)
    )
    shared_pool_swa = (
        not hybrid
        and not mla
        and facts["num_wrappers"] == 2
        and facts["sliding_window_size"] is not None
        and not facts["sliding_window_kv_pool"]
        and ps > 1
    )
    facts["shared_pool_swa"] = shared_pool_swa
    if mla and not hybrid:
        if facts["attention_backend"] != "FlashInferMLAAttnBackend" or (
            facts["num_wrappers"] != 0
            or facts["sliding_window_size"] is not None
            or facts["sliding_window_kv_pool"]
        ):
            reject(
                "attention_backend",
                f"{facts['attention_backend']} with {facts['num_wrappers']} wrapper(s), "
                f"sliding window {facts['sliding_window_size']}; supported with an MLA pool: "
                "FlashInferMLAAttnBackend without sliding-window composition",
            )
    elif hybrid:
        hybrid_backend_shape = (
            facts["full_attn_backend"],
            facts["linear_attn_backend"],
            facts["num_wrappers"],
        )
        hybrid_backend_supported = hybrid_backend_shape in (
            ("FlashInferAttnBackend", "GDNAttnBackend", 1),
            ("FlashInferMLAAttnBackend", "KDAAttnBackend", 0),
        )
        facts["hybrid_backend_supported"] = hybrid_backend_supported
        if facts["attention_backend"] != "HybridLinearAttnBackend" or not hybrid_backend_supported:
            reject(
                "attention_backend",
                f"{facts['attention_backend']} (full {facts['full_attn_backend']} with {facts['num_wrappers']} wrapper(s), linear {facts['linear_attn_backend']}); supported with a hybrid pool: HybridLinearAttnBackend over FlashInferAttnBackend plus GDNAttnBackend, or FlashInferMLAAttnBackend plus KDAAttnBackend",
            )
    elif facts["attention_backend"] != "FlashInferAttnBackend" or not (
        facts["num_wrappers"] == 1
        or shared_pool_swa
        or (
            separate_swa
            and facts["num_wrappers"] == 2
            and facts["sliding_window_size"] is not None
            and facts["sliding_window_kv_pool"]
        )
    ):
        reject(
            "attention_backend",
            f"{facts['attention_backend']} with {facts['num_wrappers']} wrapper(s), sliding "
            f"window {facts['sliding_window_size']}, page size {ps}; supported: "
            "FlashInferAttnBackend with one wrapper, or its two-wrapper shared- or "
            "independent-pool sliding-window path at a page size above one",
        )
    if (not hybrid and getattr(backend, "attn_backend_list", None)) or (
        getattr(full_backend, "use_sliding_window_kv_pool", False) and not separate_swa
    ):
        reject("attention_backend", "backend list or sliding-window KV pool")
    facts["speculative_algorithm"] = str(getattr(args, "speculative_algorithm", None))
    facts["speculative_ngram"] = bool(
        not scheduler.spec_algorithm.is_none()
        and callable(getattr(scheduler.spec_algorithm, "is_ngram", None))
        and scheduler.spec_algorithm.is_ngram()
    )
    facts["speculative_ngram_port_enabled"] = os.environ.get(ENV_NGRAM, "0") == "1"
    facts["speculative_eagle3"] = bool(
        not scheduler.spec_algorithm.is_none()
        and callable(getattr(scheduler.spec_algorithm, "is_eagle3", None))
        and scheduler.spec_algorithm.is_eagle3()
    )
    facts["speculative_eagle3_port_enabled"] = os.environ.get(ENV_EAGLE3, "0") == "1"
    facts["speculative_worker"] = type(getattr(scheduler, "draft_worker", None)).__name__
    facts["speculative_multi_layer_eagle"] = (
        facts["speculative_worker"] == "MultiLayerEagleWorkerV2"
    )
    facts["speculative_multi_layer_eagle_port_enabled"] = (
        os.environ.get(ENV_MULTI_LAYER_EAGLE, "0") == "1"
    )
    facts["speculative_nextn"] = bool(
        str(getattr(args, "speculative_algorithm", "")).upper() == "NEXTN"
    )
    facts["speculative_nextn_port_enabled"] = os.environ.get(ENV_NEXTN, "0") == "1"
    facts["speculative_dflash"] = bool(
        str(getattr(args, "speculative_algorithm", "")).upper() == "DFLASH"
    )
    facts["speculative_dflash_port_enabled"] = os.environ.get(ENV_DFLASH, "0") == "1"
    facts["speculative_num_draft_tokens"] = int(
        getattr(args, "speculative_num_draft_tokens", 0) or 0
    )
    facts["speculative_adaptive"] = bool(getattr(args, "speculative_adaptive", False))
    eagle = (
        scheduler.draft_worker
        if (
            facts["speculative_eagle3"]
            or facts["speculative_nextn"]
            or facts["speculative_multi_layer_eagle"]
        )
        else None
    )
    dflash = scheduler.draft_worker if facts["speculative_dflash"] else None
    draft_host = dflash if dflash is not None else getattr(eagle, "draft_worker", None)
    draft_runners = tuple(getattr(draft_host, "draft_runners", ()) or ())
    draft_runner = (
        draft_runners[0]
        if draft_runners
        else getattr(
            draft_host,
            "draft_model_runner" if dflash is not None else "draft_runner",
            None,
        )
    )
    draft_pool = getattr(draft_runner, "token_to_kv_pool", None)
    draft_full_pool = getattr(draft_pool, "full_kv_pool", draft_pool)
    draft_allocator = getattr(draft_runner, "token_to_kv_pool_allocator", None)
    draft_req_pool = getattr(draft_runner, "req_to_token_pool", None)
    draft_decode = (
        getattr(draft_runner, "attn_backend", None)
        if dflash is not None
        else getattr(draft_host, "draft_attn_backend", None)
    )
    draft_decode_leaves = tuple(getattr(draft_decode, "attn_backends", ()) or ())
    draft_extend_list = tuple(getattr(draft_host, "draft_extend_attn_backend_list", ()) or ())
    draft_extend = (
        draft_extend_list[0]
        if draft_extend_list
        else getattr(draft_host, "draft_extend_attn_backend", None)
    )
    facts["speculative_num_steps"] = int(getattr(eagle, "speculative_num_steps", 0) or 0)
    facts["speculative_eagle_topk"] = int(getattr(eagle, "topk", 0) or 0)
    post_verify_graphs = getattr(eagle, "_post_verify_graphs", None)
    facts["nv_mega_runtime_optimization"] = (
        os.environ.get("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0") == "1"
    )
    facts["eagle_post_verify_graphs"] = type(post_verify_graphs).__name__
    facts["eagle_post_verify_graphs_installed"] = post_verify_graphs is not None
    facts["speculative_use_rejection_sampling"] = bool(
        getattr(args, "speculative_use_rejection_sampling", False)
    )
    facts["speculative_use_block_verification"] = bool(
        getattr(args, "speculative_use_block_verification", False)
    )
    facts["dflash_block_size"] = int(getattr(dflash, "block_size", 0) or 0)
    facts["dflash_compact_cache"] = bool(getattr(dflash, "use_compact_draft_cache", False))
    facts["dflash_draft_window_size"] = getattr(dflash, "draft_window_size", None)
    facts["draft_runner"] = type(draft_runner).__name__
    facts["draft_kv_pool"] = type(draft_pool).__name__
    facts["draft_full_kv_pool"] = type(draft_full_pool).__name__
    facts["draft_layers"] = int(getattr(draft_full_pool, "layer_num", 0) or 0)
    facts["draft_full_layers"] = int(getattr(draft_pool, "full_layer_nums", 0) or 0)
    facts["draft_kv_heads"] = int(getattr(draft_full_pool, "head_num", 0) or 0)
    facts["draft_mamba_pool_shared"] = draft_pool is not None and getattr(
        draft_pool, "mamba_pool", None
    ) is getattr(req_pool, "mamba_pool", None)
    facts["draft_pool_independent"] = (
        draft_full_pool is not None and draft_full_pool is not full_pool
    )
    facts["draft_allocator_shared"] = draft_allocator is alloc
    facts["draft_req_pool_shared"] = draft_req_pool is req_pool
    facts["draft_req_table_shared"] = draft_req_pool is not None and getattr(
        draft_req_pool, "req_to_token", None
    ) is getattr(req_pool, "req_to_token", None)
    facts["draft_sub_pool"] = getattr(draft_full_pool, "_sub_pool_name", None)
    facts["owner_sub_pool"] = getattr(
        getattr(alloc, "full_attn_allocator", None), "sub_pool_name", None
    )
    facts["draft_fused_with_owner"] = bool(
        unified_allocator
        and facts["draft_kv_pool"] == "DraftMHAView"
        and facts["draft_sub_pool"] == facts["owner_sub_pool"]
        and facts["draft_allocator_shared"]
        and facts["draft_req_pool_shared"]
        and facts["draft_req_table_shared"]
    )
    facts["draft_storage_fused_with_owner"] = bool(
        unified_allocator
        and facts["draft_kv_pool"] == "DraftMHAView"
        and facts["draft_sub_pool"] == facts["owner_sub_pool"]
        and facts["draft_allocator_shared"]
    )
    facts["draft_decode_backend"] = type(draft_decode).__name__
    facts["draft_decode_wrappers"] = int(getattr(draft_decode, "num_wrappers", 0) or 0)
    facts["draft_sliding_window_size"] = getattr(draft_runner, "sliding_window_size", None)
    facts["draft_decode_leaf_backends"] = [type(item).__name__ for item in draft_decode_leaves]
    facts["draft_extend_backend"] = type(draft_extend).__name__
    facts["draft_runner_count"] = len(draft_runners) or int(draft_runner is not None)
    facts["draft_runner_classes"] = [type(item).__name__ for item in draft_runners]
    facts["draft_kv_pools"] = [
        type(getattr(item, "token_to_kv_pool", None)).__name__ for item in draft_runners
    ]
    facts["draft_kv_pool_sizes"] = [
        int(getattr(getattr(item, "token_to_kv_pool", None), "size", 0) or 0)
        for item in draft_runners
    ]
    facts["draft_kv_pool_swa_sizes"] = [
        int(getattr(getattr(item, "token_to_kv_pool", None), "size_swa", 0) or 0)
        for item in draft_runners
    ]
    facts["draft_full_side_pools"] = [
        type(getattr(getattr(item, "token_to_kv_pool", None), "full_kv_pool", None)).__name__
        for item in draft_runners
    ]
    facts["draft_swa_side_pools"] = [
        type(getattr(getattr(item, "token_to_kv_pool", None), "swa_kv_pool", None)).__name__
        for item in draft_runners
    ]
    facts["draft_full_layer_counts"] = [
        int(getattr(getattr(item, "token_to_kv_pool", None), "full_layer_nums", 0) or 0)
        for item in draft_runners
    ]
    facts["draft_swa_layer_counts"] = [
        int(getattr(getattr(item, "token_to_kv_pool", None), "swa_layer_nums", 0) or 0)
        for item in draft_runners
    ]
    facts["draft_allocator_shared_all"] = bool(draft_runners) and all(
        getattr(item, "token_to_kv_pool_allocator", None) is alloc for item in draft_runners
    )
    facts["draft_req_pool_shared_all"] = bool(draft_runners) and all(
        getattr(item, "req_to_token_pool", None) is req_pool for item in draft_runners
    )
    facts["draft_req_table_shared_all"] = bool(draft_runners) and all(
        getattr(getattr(item, "req_to_token_pool", None), "req_to_token", None)
        is getattr(req_pool, "req_to_token", None)
        for item in draft_runners
    )
    facts["draft_pools_pairwise_distinct"] = len(
        {id(getattr(item, "token_to_kv_pool", None)) for item in draft_runners}
    ) == len(draft_runners)
    facts["draft_backends"] = [
        type(getattr(item, "attn_backend", None)).__name__ for item in draft_runners
    ]
    facts["draft_backend_wrappers"] = [
        int(getattr(getattr(item, "attn_backend", None), "num_wrappers", 0) or 0)
        for item in draft_runners
    ]
    facts["draft_sliding_windows"] = [
        getattr(item, "sliding_window_size", None) for item in draft_runners
    ]
    facts["draft_extend_backends"] = [type(item).__name__ for item in draft_extend_list]
    facts["model_architectures"] = list(
        getattr(
            getattr(scheduler.tp_worker.model_runner.model_config, "hf_config", None),
            "architectures",
            (),
        )
        or ()
    )
    facts["cuda_graph_disabled"] = bool(getattr(args, "disable_cuda_graph", False))
    facts["target_decode_graph_runner"] = type(
        getattr(scheduler.tp_worker.model_runner, "decode_cuda_graph_runner", None)
    ).__name__
    facts["target_prefill_graph_runner"] = type(
        getattr(scheduler.tp_worker.model_runner, "prefill_cuda_graph_runner", None)
    ).__name__
    draft_decode_graph = (
        getattr(draft_runner, "decode_cuda_graph_runner", None)
        if dflash is not None
        else getattr(draft_host, "cuda_graph_runner", None)
    )
    facts["draft_decode_graph_runner"] = type(draft_decode_graph).__name__
    facts["draft_decode_graph_width"] = int(
        getattr(draft_decode_graph, "captured_req_width", 0) or 0
    )
    facts["draft_decode_graph_capture_mode"] = getattr(
        getattr(draft_decode_graph, "capture_forward_mode", None), "name", None
    )
    facts["draft_extend_graph_runner"] = type(
        getattr(draft_host, "cuda_graph_runner_for_draft_extend", None)
    ).__name__
    draft_extend_graph = getattr(draft_host, "cuda_graph_runner_for_draft_extend", None)
    draft_extend_graph_steps = tuple(getattr(draft_extend_graph, "runners", ()) or ())
    facts["draft_extend_graph_step_runners"] = [
        type(item).__name__ for item in draft_extend_graph_steps
    ]
    facts["draft_extend_graph_rotates_in_graph"] = bool(
        getattr(draft_extend_graph, "rotates_in_graph", False)
    )
    facts["draft_extend_graph_captured_req_width"] = int(
        getattr(draft_extend_graph, "captured_req_width", 0) or 0
    )
    facts["draft_extend_graph_disable_padding"] = bool(
        getattr(draft_extend_graph, "disable_padding", False)
    )
    facts["draft_prefill_graph_runner"] = type(
        getattr(draft_runner, "prefill_cuda_graph_runner", None)
    ).__name__
    ngram_supported = False
    eagle_supported = False
    nextn_supported = False
    dflash_supported = False
    multi_layer_eagle_supported = False
    if not scheduler.spec_algorithm.is_none():
        ngram_supported = (
            facts["speculative_ngram"]
            and facts["speculative_ngram_port_enabled"]
            and facts["speculative_worker"] == "NGRAMWorker"
            # NGRAM has no draft model, so a hybrid (GDN) target composes: the full-attention KV
            # pool takes the speculative high-water and the recurrent-state pool keeps its own
            # per-step StateMediation dispatch, unchanged by speculation. MLA stays excluded.
            and not mla
            and (facts["num_wrappers"] == 1 or shared_pool_swa or separate_swa)
            and facts["speculative_num_draft_tokens"] > 0
        )
        # P/D may resolve every speculative graph backend to eager without setting the
        # user-facing disable flag.  The concrete runners, not that input flag, determine
        # which descriptor path executes.
        eager_eagle_graphs = (
            facts["target_decode_graph_runner"] == "NoneType"
            and facts["draft_decode_graph_runner"] == "NoneType"
            and facts["draft_extend_graph_runner"] == "NoneType"
        )
        cuda_eagle_graphs = (
            not facts["cuda_graph_disabled"]
            and facts["target_decode_graph_runner"] == "DecodeCudaGraphRunner"
            and facts["draft_decode_graph_runner"] == "EAGLEDraftCudaGraphRunner"
            and facts["draft_extend_graph_runner"] == "EAGLEDraftExtendCudaGraphRunner"
            and facts["draft_prefill_graph_runner"]
            in ("NoneType", "EagerRunner", "PrefillCudaGraphRunner")
        )
        dflash_graphs = (
            not facts["cuda_graph_disabled"]
            and facts["target_decode_graph_runner"] == "DecodeCudaGraphRunner"
            and facts["draft_decode_graph_runner"] == "DecodeCudaGraphRunner"
            and facts["draft_decode_graph_width"] == facts["dflash_block_size"]
            and facts["draft_decode_graph_capture_mode"] == "TARGET_VERIFY"
            and facts["draft_prefill_graph_runner"] == "NoneType"
        )
        dflash_eager = (
            facts["cuda_graph_disabled"]
            and facts["target_decode_graph_runner"] == "NoneType"
            and facts["draft_decode_graph_runner"]
            == ("EagerRunner" if facts["dflash_compact_cache"] else "NoneType")
            and facts["draft_prefill_graph_runner"] == "NoneType"
        )
        multi_layer_eagle_graphs = (
            not facts["cuda_graph_disabled"]
            and facts["target_decode_graph_runner"] == "DecodeCudaGraphRunner"
            and facts["target_prefill_graph_runner"] == "EagerRunner"
            and facts["draft_decode_graph_runner"] == "NoneType"
            and facts["draft_extend_graph_runner"]
            == "MultiLayerEagleMultiStepDraftExtendCudaGraphRunner"
            and facts["draft_extend_graph_step_runners"]
            == ["MultiLayerEagleDraftExtendCudaGraphRunner"] * 3
            and not facts["draft_extend_graph_rotates_in_graph"]
            and facts["draft_extend_graph_captured_req_width"] == 4
            and not facts["draft_extend_graph_disable_padding"]
            and facts["draft_prefill_graph_runner"] == "NoneType"
        )
        eagle_supported = (
            facts["speculative_eagle3"]
            and facts["speculative_eagle3_port_enabled"]
            and facts["speculative_worker"] == "EAGLEWorkerV2"
            and not hybrid
            and not mla
            and facts["num_wrappers"] == 1
            and facts["speculative_eagle_topk"] in (1, 2, 8)
            and facts["speculative_num_steps"] > 1
            and (
                (
                    facts["speculative_eagle_topk"] == 1
                    and facts["speculative_num_draft_tokens"] == facts["speculative_num_steps"] + 1
                )
                or (
                    facts["speculative_eagle_topk"] > 1
                    and not facts["speculative_use_rejection_sampling"]
                    and facts["speculative_num_draft_tokens"] > 1
                )
            )
            and facts["draft_runner"] == "ModelRunner"
            and facts["draft_kv_pool"] == "MHATokenToKVPool"
            and facts["draft_layers"] > 0
            and facts["draft_kv_heads"] > 0
            and facts["draft_pool_independent"]
            and facts["draft_allocator_shared"]
            and facts["draft_req_pool_shared"]
            and facts["draft_req_table_shared"]
            and draft_decode_leaves
            and all(type(item).__name__ == "FlashInferAttnBackend" for item in draft_decode_leaves)
            and facts["draft_extend_backend"] == "FlashInferAttnBackend"
            and (eager_eagle_graphs or cuda_eagle_graphs)
        )
        legacy_nextn_pool = (
            facts["draft_kv_pool"] == "HybridLinearKVPool"
            and facts["draft_full_kv_pool"] == "MHATokenToKVPool"
            and facts["draft_full_layers"] == 1
            and facts["draft_mamba_pool_shared"]
        )
        fused_nextn_pool = (
            facts["unified_hybrid"]
            and facts["draft_fused_with_owner"]
            and facts["draft_kv_pool"] == "DraftMHAView"
            and facts["draft_full_kv_pool"] == "DraftMHAView"
            and facts["draft_full_layers"] == 0
        )
        nextn_supported = (
            facts["speculative_nextn"]
            and facts["speculative_nextn_port_enabled"]
            and facts["speculative_worker"] == "EAGLEWorkerV2"
            and hybrid
            and facts["num_wrappers"] == 1
            and facts["speculative_eagle_topk"] == 1
            and facts["speculative_num_steps"] > 1
            and facts["speculative_num_draft_tokens"] == facts["speculative_num_steps"] + 1
            and facts["draft_runner"] == "ModelRunner"
            and facts["draft_layers"] == 1
            and facts["draft_kv_heads"] > 0
            and (legacy_nextn_pool or fused_nextn_pool)
            and (not facts["speculative_use_rejection_sampling"] or fused_nextn_pool)
            and facts["draft_pool_independent"]
            and facts["draft_allocator_shared"]
            and facts["draft_req_pool_shared"]
            and facts["draft_req_table_shared"]
            and draft_decode_leaves
            and all(type(item).__name__ == "FlashInferAttnBackend" for item in draft_decode_leaves)
            and facts["draft_extend_backend"] == "FlashInferAttnBackend"
            and (eager_eagle_graphs or cuda_eagle_graphs)
        )
        dflash_compact_view = (
            facts["dflash_compact_cache"]
            and facts["dflash_draft_window_size"] >= facts["dflash_block_size"]
            and facts["draft_storage_fused_with_owner"]
            and not facts["draft_req_pool_shared"]
            and not facts["draft_req_table_shared"]
        )
        dflash_full_view = (
            not facts["dflash_compact_cache"]
            and facts["dflash_draft_window_size"] is None
            and facts["draft_fused_with_owner"]
            and facts["draft_req_pool_shared"]
            and facts["draft_req_table_shared"]
        )
        dflash_supported = (
            facts["speculative_dflash"]
            and facts["speculative_dflash_port_enabled"]
            and facts["speculative_worker"] == "DFlashWorkerV2"
            and facts["model_architectures"] == ["Qwen3_5ForConditionalGeneration"]
            and hybrid
            and facts["unified_hybrid"]
            and facts["num_wrappers"] == 1
            and facts["dflash_block_size"] > 1
            and facts["dflash_block_size"] == facts["speculative_num_draft_tokens"]
            and facts["draft_runner"] == "ModelRunner"
            and facts["draft_kv_pool"] == "DraftMHAView"
            and facts["draft_full_kv_pool"] == "DraftMHAView"
            and facts["draft_layers"] > 0
            and facts["draft_kv_heads"] > 0
            and (dflash_full_view or dflash_compact_view)
            and facts["draft_pool_independent"]
            and facts["draft_allocator_shared"]
            and facts["draft_decode_backend"] == "FlashInferAttnBackend"
            and facts["draft_decode_wrappers"] == 2
            and facts["draft_sliding_window_size"] == 4095
            and (dflash_eager or dflash_graphs)
        )
        multi_layer_eagle_supported = (
            facts["speculative_multi_layer_eagle"]
            and facts["speculative_multi_layer_eagle_port_enabled"]
            and facts["speculative_algorithm"].upper() == "EAGLE"
            and facts["model_architectures"] == ["Step3p5ForCausalLM"]
            and facts["tree_core"] == "RustUnifiedTreeCore"
            and facts["tree_strategy"] == "rust_walk"
            and separate_swa
            and not hybrid
            and not mla
            and facts["num_wrappers"] == 2
            and facts["sliding_window_kv_pool"]
            and facts["kv_pool_full_layers"] == 12
            and facts["kv_pool_swa_layers"] == 33
            and facts["speculative_eagle_topk"] == 1
            and facts["speculative_num_steps"] == 3
            and facts["speculative_num_draft_tokens"] == 4
            and not facts["speculative_use_rejection_sampling"]
            and facts["draft_runner_count"] == 3
            and facts["draft_runner_classes"] == ["ModelRunner"] * 3
            and facts["draft_kv_pools"] == ["SWAKVPool"] * 3
            and facts["draft_full_side_pools"] == ["MHATokenToKVPool"] * 3
            and facts["draft_swa_side_pools"] == ["MHATokenToKVPool"] * 3
            and facts["draft_full_layer_counts"] == [0, 0, 0]
            and facts["draft_swa_layer_counts"] == [1, 1, 1]
            and facts["draft_allocator_shared_all"]
            and facts["draft_req_pool_shared_all"]
            and facts["draft_req_table_shared_all"]
            and facts["draft_pools_pairwise_distinct"]
            and facts["draft_backends"] == ["FlashInferAttnBackend"] * 3
            and facts["draft_backend_wrappers"] == [2, 2, 2]
            and facts["draft_sliding_windows"] == [facts["sliding_window_size"]] * 3
            and facts["draft_extend_backends"] == ["FlashInferAttnBackend"] * 3
            and (eager_eagle_graphs or multi_layer_eagle_graphs)
        )
        if not (
            ngram_supported
            or eagle_supported
            or nextn_supported
            or dflash_supported
            or multi_layer_eagle_supported
        ):
            reject(
                "speculative_decoding",
                f"{facts['speculative_algorithm']} with {facts['speculative_worker']}, "
                f"page size {ps}, hybrid={hybrid}, wrappers={facts['num_wrappers']}; supported: "
                "NGRAMWorker over one full-attention pool (including its shared-pool "
                "sliding/full wrapper pair), or EAGLEWorkerV2 EAGLE3 chain/tree "
                "decoding (greedy tree mode, eager or the exact CUDA-graph runners) over "
                "independent target/draft MHA pools that share one allocator and request table, "
                "the opt-in Qwen3.5 NEXTN top-k-1 path with one independent MTP MHA layer or "
                "the exact unified-memory DraftMHAView fused into its owner pages, "
                "the opt-in eager/graph Qwen3.5 DFlash path over the same fused "
                "DraftMHAView (full or compact suffix table), "
                "or the opt-in eager/per-step-graph Step-3.5 multi-layer EAGLE path over its "
                "exact separate full/SWA target and three SWA-only draft pools",
            )
    elif getattr(scheduler, "draft_worker", None) is not None:
        reject("speculative_decoding", facts["speculative_worker"])
    facts["hierarchical_cache"] = bool(scheduler.enable_hierarchical_cache)
    external_linker_layout_supported = (
        facts.get("tree_components") == ["FULL"] and not hybrid
    ) or (
        set(facts.get("tree_components") or ()) == {"FULL", "MAMBA"}
        and hybrid
        and (unified_hybrid or facts.get("hybrid_mla"))
    )
    facts["external_linker_layout_supported"] = bool(external_linker_layout_supported)
    external_linker_supported = (
        facts.get("external_linker")
        and facts.get("external_linker_port_enabled")
        and facts.get("external_linker_wrapper") == "UnifiedCacheLinkerWrapper"
        and facts.get("external_linker_backend") in ("MooncakeDirectLinker", "UMBPDirectLinker")
        and external_linker_layout_supported
        and not separate_swa
        and str(facts.get("speculative_algorithm") or "none").upper() == "NONE"
        and not facts["hierarchical_cache"]
    )
    facts["external_linker_supported"] = bool(external_linker_supported)
    if facts.get("external_linker") and not external_linker_supported:
        reject(
            "external_linker",
            "the bounded exact-main port requires either one FULL component, the exact unified "
            "FULL plus MAMBA layout, or the exact legacy MLA plus KDA hybrid; no HiCache or "
            "speculative decoding; and the built-in "
            "Mooncake or UMBP direct linker",
        )
    hicache_supported = (
        facts["hierarchical_cache"]
        and facts["hicache_port_enabled"]
        and facts.get("hicache_controller") == "HybridCacheController"
        and facts.get("hicache_write_policy") in ("write_through", "write_back")
        and facts.get("hicache_io_backend") in ("kernel", "direct")
        and (
            (
                facts.get("hicache_host_memory_mode") == "cache"
                and (
                    not facts.get("hicache_storage")
                    or facts.get("hicache_storage_backend") == "file"
                )
            )
            or (
                facts.get("hicache_host_memory_mode") == "buffer_only"
                and facts.get("hicache_storage")
                and facts.get("hicache_storage_backend") == "file"
            )
        )
        and _hicache_hybrid_host_mode_supported(
            hybrid=hybrid,
            unified_hybrid=unified_hybrid,
            host_memory_mode=facts.get("hicache_host_memory_mode"),
        )
        and (
            facts.get("hicache_mem_layout") != "page_unified"
            or (
                facts.get("hicache_io_backend") == "kernel"
                and facts.get("hicache_storage")
                and facts.get("hicache_storage_backend") == "file"
                and facts.get("hicache_storage_mem_layout") == "page_unified"
                and facts.get("hicache_unified_storage_plan")
            )
        )
    )
    if facts["hierarchical_cache"] and not hicache_supported:
        reject(
            "hierarchical_cache",
            "the exact-main port admits a full-attention HybridCacheController "
            "with kernel or direct I/O and write-through or write-back; cache-mode host memory "
            "allows no storage or local file storage, while buffer-only requires local file "
            "storage; page_unified requires kernel I/O, the local file backend and a concrete "
            "unified storage plan; legacy hybrid models are admitted only in buffer-only mode, "
            "while the routed unified-memory hybrid admits cache mode too; the "
            "MLA boundary is one non-speculative FlashInferMLA pool",
        )
    if getattr(scheduler, "enable_hisparse", False):
        reject("hierarchical_cache", "sparse cache")
    if unified_memory and (
        not unified_hybrid
        or facts.get("model_architectures") != ["Qwen3_5ForConditionalGeneration"]
    ):
        reject(
            "unified_memory",
            "the bounded unified-memory port requires one routed full-attention owner band, "
            "one recurrent-state band, no sliding-window band and the Qwen3.5 hybrid backend",
        )
    mode = scheduler.disaggregation_mode
    facts["disaggregation_mode"] = getattr(mode, "name", str(mode))
    facts["disagg_prefill_port_enabled"] = os.environ.get(ENV_DISAGG_PREFILL, "0") == "1"
    facts["disagg_decode_port_enabled"] = os.environ.get(ENV_DISAGG_DECODE, "0") == "1"
    transfer_backend = getattr(scheduler, "transfer_backend", None)
    facts["disaggregation_transfer_backend"] = getattr(
        transfer_backend, "name", str(transfer_backend)
    )
    facts["disaggregation_staging"] = bool(getattr(scheduler, "enable_staging", False))
    facts["disaggregation_state_types"] = _pd_state_type_names(
        scheduler, facts["disaggregation_mode"]
    )
    disagg_backend = facts["disaggregation_transfer_backend"].upper()
    disagg_state_supported = not hybrid or (
        disagg_backend == "MOONCAKE"
        and unified_hybrid
        and facts["disaggregation_state_types"] == ["MAMBA"]
    )
    disagg_backend_supported = disagg_backend in (
        "MOONCAKE",
        "NIXL",
    )
    disagg_prefill_supported = (
        facts["disaggregation_mode"].upper() == "PREFILL"
        and facts["disagg_prefill_port_enabled"]
        and disagg_backend_supported
        and not facts["disaggregation_staging"]
        and disagg_state_supported
        and facts["num_wrappers"] == 1
    )
    facts["disagg_prefill"] = disagg_prefill_supported
    disagg_decode_supported = (
        facts["disaggregation_mode"].upper() == "DECODE"
        and facts["disagg_decode_port_enabled"]
        and disagg_backend_supported
        and not facts["disaggregation_staging"]
        and disagg_state_supported
        and facts["num_wrappers"] == 1
        and facts["req_to_token_pool"]
        == ("RequestTable" if unified_hybrid else "DecodeReqToTokenPool")
    )
    facts["disagg_decode"] = disagg_decode_supported
    if facts["disaggregation_mode"].upper() != "NULL" and not (
        disagg_prefill_supported or disagg_decode_supported
    ):
        reject(
            "disaggregation",
            f"{facts['disaggregation_mode']} via {facts['disaggregation_transfer_backend']}, "
            f"staging={facts['disaggregation_staging']}; supported by the opt-in port: "
            "PREFILL or DECODE via Mooncake or NIXL, no staging, one full-attention wrapper, "
            "and either no state payload or the routed unified Qwen3.5 hybrid's single MAMBA "
            "payload",
        )
    if getattr(scheduler, "dllm_config", None) is not None:
        reject("mode", "dLLM")
    facts["pdmux_enabled"] = bool(getattr(scheduler, "enable_pdmux", False))
    facts["pdmux_port_enabled"] = os.environ.get(ENV_PDMUX, "0") == "1"
    decode_backends = tuple(
        getattr(scheduler.tp_worker.model_runner, "decode_attn_backend_group", ()) or ()
    )
    facts["pdmux_decode_backends"] = [type(item).__name__ for item in decode_backends]
    facts["pdmux_decode_wrappers"] = [
        int(getattr(_full_backend(item), "num_wrappers", 0)) for item in decode_backends
    ]
    pdmux_supported = (
        facts["pdmux_enabled"]
        and facts["pdmux_port_enabled"]
        and facts["disaggregation_mode"].upper() == "NULL"
        and not hybrid
        and (facts["num_wrappers"] == 1 or shared_pool_swa or separate_swa)
        and facts["target_decode_graph_runner"] in ("NoneType", "DecodeCudaGraphRunner")
        and (
            scheduler.spec_algorithm.is_none()
            or ngram_supported
            or eagle_supported
            or nextn_supported
        )
        and decode_backends
        and all(type(item).__name__ == "FlashInferAttnBackend" for item in decode_backends)
        and all(
            int(getattr(item, "num_wrappers", 0)) == facts["num_wrappers"]
            for item in decode_backends
        )
        # Every PDMUX stream backend must use the same physical-id contract as the
        # scheduler backend.  Shared-pool SWA (and ordinary full attention) read
        # full-pool ids, while the independent SWA topology translates wrapper 0
        # through its dedicated window pool on every stream.
        and all(
            bool(getattr(item, "use_sliding_window_kv_pool", False)) == separate_swa
            for item in decode_backends
        )
    )
    facts["pdmux"] = pdmux_supported
    facts["speculative_single_draft_sidecar"] = (
        eagle_supported or nextn_supported or dflash_supported
    )
    facts["speculative_dflash_supported"] = dflash_supported
    facts["speculative_multi_layer_eagle_supported"] = multi_layer_eagle_supported
    facts["speculative_draft_sidecar"] = (
        eagle_supported or nextn_supported or dflash_supported or multi_layer_eagle_supported
    )
    if facts["pdmux_enabled"] and not pdmux_supported:
        reject(
            "pdmux",
            "the opt-in port admits single-rank PDMUX over one full-attention "
            "FlashInfer wrapper or the shared- or independent-pool sliding/full pair, with the "
            "same wrapper count and window-id contract "
            "on every stream backend; only eager execution or the exact DecodeCudaGraphRunner is "
            "admitted; hybrid models are excluded and any speculative worker must independently "
            "satisfy the NGRAM/EAGLE3 boundary",
        )
    facts["tp_size"], facts["dp_size"], facts["pp_size"] = (
        int(args.tp_size),
        int(args.dp_size),
        int(args.pp_size),
    )
    if max(facts["tp_size"], facts["dp_size"], facts["pp_size"]) > 1:
        reject(
            "parallelism",
            f"tp={facts['tp_size']} dp={facts['dp_size']} pp={facts['pp_size']}; supported: 1",
        )
    facts["is_generation"] = bool(scheduler.is_generation)
    if not facts["is_generation"]:
        reject("model", "embedding or reward model")
    facts["overlap_schedule"] = bool(scheduler.enable_overlap)
    facts["chunked_prefill_size"] = getattr(args, "chunked_prefill_size", None)
    facts["layers"] = int(getattr(full_pool, "layer_num", 0) or 0)
    facts["kv_heads"] = int(getattr(full_pool, "head_num", 0) or 0)
    if problems:
        for report in problems:
            reporter.emit(report)
        exc = Unsupported(problems[0])
        exc.facts = facts
        raise exc
    return facts  # fmt: skip


# ------------------------------------------------------------------ the radix tree

FULL_COMPONENT = 0
"""``ComponentType.FULL``: the index of the full-attention KV in a node's ``component_data`` list."""


def tree_strategy(tree: Any) -> str:
    """How the tree's nodes are read: ``radix_nodes`` (``RadixCache.root_node`` with
    ``TreeNode.value``/``lock_ref``), ``python_nodes`` (``UnifiedRadixCache`` on the Python
    ``UnifiedTreeCore``: ``tree_core.root_node`` with per-component data), or ``rust_walk``
    (the Rust core exposes no nodes to Python; ``walk_for_kv_canary`` flattens the device slots
    of all nodes and of the unlocked ones, so lock counts are unknown)."""
    if type(tree).__name__ == "RadixCache":
        return "radix_nodes"
    core = getattr(tree, "tree_core", None)
    if core is None:
        return "none"
    try:
        root = core.root_node
    except NotImplementedError:
        root = None
    if root is not None and hasattr(root, "component_data"):
        return "python_nodes"
    if hasattr(core, "walk_for_kv_canary"):
        return "rust_walk"
    return "none"


def nodes_from_unified_tree(root: Any, full: int = FULL_COMPONENT) -> list[NodeFacts]:
    """``NodeFacts`` for every node below the Python ``UnifiedTreeCore`` root (``children``
    mapping, ``component_data[FULL].value`` and ``.lock_ref``, ``id``). A node whose FULL device
    value is ``None`` (evicted to host) is outside the boundary and skipped, as in
    ``nodes_from_tree``."""
    out = []
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        data = node.component_data[full]
        if data.value is None:
            continue
        out.append(NodeFacts(int(node.id), tuple(_ints(data.value)), int(data.lock_ref)))
    return sorted(out, key=lambda n: n.node_id)


def paged_call_known(name: str, bound: Mapping[str, Any]) -> bool:
    """Whether ``paged_pages_taken`` can derive the pages of this allocator call from its
    bound arguments (the CPU length tensors the paged allocator takes since 0.5.22)."""
    if name == "alloc":
        return True
    if name == "alloc_decode":
        return bound.get("seq_lens_cpu") is not None
    if name == "alloc_extend":
        return bound.get("prefix_lens_cpu") is not None and bound.get("seq_lens_cpu") is not None
    return False


def paged_pages_taken(
    name: str, bound: Mapping[str, Any], out: Any, page_size: int
) -> list[int] | None:
    """The pages a ``PagedTokenToKVPoolAllocator`` call took off its free list, sorted, from
    the call's bound arguments and the slots it returned (``out``): ``alloc`` returns whole
    pages; ``alloc_extend`` opens a page at every page-aligned position of a row's extension
    ``[prefix, seq)`` (the slot of position ``p`` is ``out[start + p - prefix]``, rows
    consecutive); ``alloc_decode`` opens one for a row whose new token's position
    ``seq - 1`` is page-aligned. ``None`` when the slots do not match the lengths."""
    ps = int(page_size)
    slots = _ints(out)
    if name == "alloc":
        return sorted({s // ps for s in slots[::ps]})
    if name == "alloc_decode":
        seq = _ints(bound["seq_lens_cpu"])
        if len(slots) != len(seq):
            return None
        return sorted({slots[r] // ps for r in range(len(seq)) if (seq[r] - 1) % ps == 0})
    if name == "alloc_extend":
        pre, seq = _ints(bound["prefix_lens_cpu"]), _ints(bound["seq_lens_cpu"])
        if len(pre) != len(seq):
            return None
        pages: set[int] = set()
        start = 0
        for p, s in zip(pre, seq, strict=True):
            first = -(-p // ps) * ps  # the first page-aligned position at or after the prefix
            for pos in range(first, s, ps):
                pages.add(slots[start + pos - p] // ps)
            start += s - p
        if start != len(slots):
            return None
        return sorted(pages)
    return None


class WalkStates:
    """The per-slot ``BlockSnapshot`` objects of the walk strategy, kept between snapshots so a
    slot whose state did not change keeps its object (the ledger recognizes a kept object by
    identity). Set differences find the slots that entered or left the tree, changed lock state,
    or entered or left a row's own range; only those get new objects. The rules are those of
    ``slot_states``: a tree slot is PUBLISHED (reserved when unlocked), a row's own slot outside
    the tree is PRIVATE."""

    def __init__(self, pool: str, page_size: int = 1):
        self.pool = pool
        self.page_size = int(page_size)
        self.tree: dict[int, BlockSnapshot] = {}
        self.unlocked: set[int] = set()
        self.private: dict[int, BlockSnapshot] = {}
        self.changed: dict[int, BlockSnapshot] = {}
        """The objects the last ``update`` created (new pages, pages whose state changed)."""
        self.gone: frozenset[int] = frozenset()
        """The pages the last ``update`` dropped from both views."""
        self._tree_mask: np.ndarray | None = None
        self._unlocked_mask: np.ndarray | None = None
        """The tree view of the last mask-form ``update`` as boolean arrays over the page ids."""
        self._masks_stale = True
        """The masks are behind ``tree`` and ``unlocked`` after a set-form ``update``."""
        self._unlocked_stale = False
        """``unlocked`` (the set form) is behind the masks after a mask-form ``update``."""

    def update(
        self,
        rows: list[RowFacts],
        tree_slots: list[int] | None = None,
        unlocked_slots: list[int] | None = None,
        shared_slots: set[int] | None = None,
        reserved_slots: set[int] | None = None,
        *,
        tree_pages: set[int] | None = None,
        unlocked_pages: set[int] | None = None,
        own_pages: set[int] | None = None,
        delta: bool = False,
        tree_mask: np.ndarray | None = None,
        unlocked_mask: np.ndarray | None = None,
    ) -> tuple[tuple[BlockSnapshot, ...], list[int]]:
        """The block entries for one snapshot (tree slots first, then the rows' own slots, in
        first-seen order) and the slots the walk listed twice, sorted. The tree is given as the
        walk's slot lists, page sets, or boolean arrays over the page ids (``tree_mask``,
        ``unlocked_mask``); ``own_pages`` gives the rows' own pages when the caller has them. With
        ``delta`` the entries are the changed objects only and the pages that left are in
        ``gone``."""
        pool = self.pool
        ps = self.page_size
        duplicates: list[int] = []
        if tree_mask is not None:
            blocks = self._update_from_masks(
                rows, tree_mask, unlocked_mask, shared_slots, reserved_slots, own_pages, delta
            )
            return blocks, duplicates
        if self._unlocked_stale:  # the set form after the mask form: its set from the mask
            self.unlocked = set(np.flatnonzero(self._unlocked_mask).tolist())
            self._unlocked_stale = False
        if tree_pages is not None:
            tree_set = tree_pages
            unlocked = unlocked_pages if unlocked_pages is not None else set()
        elif ps == 1:
            tree_set = set(tree_slots)
            if len(tree_set) != len(tree_slots):
                duplicates = sorted(s for s, n in Counter(tree_slots).items() if n > 1)
            unlocked = set(unlocked_slots)
        else:
            # Pages: a node holds whole pages (inserts are page-aligned), so the first slot of
            # each page stands for it; a slot listed twice is a page listed twice.
            tree_set = page_set(tree_slots, ps)
            if len(set(tree_slots)) != len(tree_slots):
                duplicates = sorted(s for s, n in Counter(tree_slots).items() if n > 1)
            unlocked = page_set(unlocked_slots, ps)
        shared = page_set(shared_slots or (), ps)
        pending = page_set(reserved_slots or (), ps)
        if shared or pending:
            published = tree_set | shared | pending
            # Beam-shared history is live even if an overlapping tree node is
            # otherwise evictable.  Member rows do not own radix lock refs.
            unlocked = (unlocked | pending) - shared
        else:
            published = tree_set
        tree = self.tree
        left_tree = tree.keys() - published
        for s in left_tree:
            del tree[s]
        new = published - tree.keys()
        flipped = (unlocked ^ self.unlocked) & tree.keys()
        changed: dict[int, BlockSnapshot] = {}
        for s in new | flipped:
            tree[s] = changed[s] = BlockSnapshot(
                pool, s, BlockState.PUBLISHED, reserved=s in unlocked, refs=None
            )
        self.unlocked = unlocked
        self._masks_stale = True
        if own_pages is None:
            own: set[int] = set()
            for row in rows:
                own.update(row_pages(row.slots, row.cache_protected_len, row.kv_allocated_len, ps))
            own -= published
        else:
            own = own_pages - published
        private = self.private
        left_private = private.keys() - own
        for s in left_private:
            del private[s]
        for s in own - private.keys():
            private[s] = changed[s] = BlockSnapshot(pool, s, BlockState.PRIVATE)
        self.changed = changed
        self.gone = frozenset((left_tree | left_private) - changed.keys())
        if delta:
            return tuple(changed.values()), duplicates
        return (*tree.values(), *private.values()), duplicates

    def _update_from_masks(
        self,
        rows: list[RowFacts],
        tree_mask: np.ndarray,
        unlocked_mask: np.ndarray | None,
        shared_slots: set[int] | None,
        reserved_slots: set[int] | None,
        own_pages: set[int] | None,
        delta: bool,
    ) -> tuple[BlockSnapshot, ...]:
        """``update`` in the mask form: the same objects, states and ``gone`` as the set form
        given the same view; the pages that entered, left or flipped are found with array
        operations over the previous masks (rebuilt from the sets after a set-form update)."""
        pool = self.pool
        ps = self.page_size
        published = tree_mask
        n = published.shape[0]
        unlocked = unlocked_mask if unlocked_mask is not None else np.zeros(n, dtype=np.bool_)
        shared = page_set(shared_slots or (), ps)
        pending = page_set(reserved_slots or (), ps)
        if shared or pending:
            published = published.copy()
            published[list(shared | pending)] = True
            unlocked = unlocked.copy()
            if pending:
                unlocked[list(pending)] = True
            if shared:
                unlocked[list(shared)] = False
        if self._tree_mask is None or self._masks_stale:
            prev_tree = np.zeros(n, dtype=np.bool_)
            prev_unlocked = np.zeros(n, dtype=np.bool_)
            if self.tree:
                prev_tree[list(self.tree)] = True
            if self.unlocked:
                prev_unlocked[list(self.unlocked)] = True
            self._masks_stale = False
        else:
            prev_tree, prev_unlocked = self._tree_mask, self._unlocked_mask
        tree = self.tree
        left_tree = np.flatnonzero(prev_tree & ~published).tolist()
        for s in left_tree:
            del tree[s]
        new = np.flatnonzero(published & ~prev_tree).tolist()
        flipped = np.flatnonzero((unlocked ^ prev_unlocked) & published & prev_tree).tolist()
        changed: dict[int, BlockSnapshot] = {}
        for s in new:
            tree[s] = changed[s] = BlockSnapshot(
                pool, s, BlockState.PUBLISHED, reserved=bool(unlocked[s]), refs=None
            )
        for s in flipped:
            tree[s] = changed[s] = BlockSnapshot(
                pool, s, BlockState.PUBLISHED, reserved=bool(unlocked[s]), refs=None
            )
        self._tree_mask = published.copy()
        self._unlocked_mask = unlocked.copy()
        self._unlocked_stale = True
        if own_pages is None:
            own_pages = set()
            for row in rows:
                own_pages.update(
                    row_pages(row.slots, row.cache_protected_len, row.kv_allocated_len, ps)
                )
        own: set[int] = set()
        if own_pages:
            arr = np.fromiter(own_pages, dtype=np.int64, count=len(own_pages))
            own = set(arr[~published[arr]].tolist())
        private = self.private
        left_private = private.keys() - own
        for s in left_private:
            del private[s]
        for s in own - private.keys():
            private[s] = changed[s] = BlockSnapshot(pool, s, BlockState.PRIVATE)
        self.changed = changed
        self.gone = frozenset((set(left_tree) | left_private) - changed.keys())
        if delta:
            return tuple(changed.values())
        return (*tree.values(), *private.values())


def snapshot_from_walk(
    seq: int,
    pool: str,
    rows: list[RowFacts],
    tree_slots: list[int] | None,
    unlocked_slots: list[int] | None,
    free_slots: list[int],
    states: WalkStates | None = None,
    page_size: int = 1,
    shared_slots: set[int] | None = None,
    reserved_slots: set[int] | None = None,
    *,
    tree_pages: set[int] | None = None,
    unlocked_pages: set[int] | None = None,
    own_pages: set[int] | None = None,
    delta: bool = False,
    tree_mask: np.ndarray | None = None,
    unlocked_mask: np.ndarray | None = None,
) -> tuple[Snapshot, list[int]]:
    """A snapshot from the flattened tree walk (no node identity, no lock counts): every tree slot
    is PUBLISHED, reserved when unlocked; ``refs`` stays ``None`` so ``reference_count_mismatch``
    is not evaluated. Row rules are those of ``slot_states``. ``states`` keeps the per-slot objects
    between calls; with ``delta`` the pool is listed as its change since the previous call. Returns
    the snapshot and the slots the walk listed twice (an integration error the caller reports)."""
    if states is None:
        states = WalkStates(pool, page_size)
    blocks, duplicates = states.update(
        rows,
        tree_slots,
        unlocked_slots,
        shared_slots,
        reserved_slots,
        tree_pages=tree_pages,
        unlocked_pages=unlocked_pages,
        own_pages=own_pages,
        delta=delta,
        tree_mask=tree_mask,
        unlocked_mask=unlocked_mask,
    )
    ps = states.page_size
    requests = tuple(
        RequestSnapshot(
            r.request_id,
            pool,
            pages_of_prefix(r.slots, r.kv_allocated_len, ps),
            int(r.kv_allocated_len),
            pre_step_computed(r.kv_committed_len, r.new_tokens),
        )
        for r in rows
    )
    free = {pool: frozenset(free_slots)}
    if delta:
        gone = {pool: states.gone} if states.gone else None
        snap = Snapshot(seq, requests, blocks, free, delta_pools=frozenset({pool}), gone=gone)
        return snap, duplicates
    return Snapshot(seq, requests, blocks, free), duplicates


def swa_snapshot_from_walk(
    seq: int,
    pool: str,
    rows: Sequence[RowFacts],
    full_to_swa: Sequence[int],
    tree_full_slots: Sequence[int],
    unlocked_full_slots: Sequence[int],
    free_pages: Sequence[int],
    page_size: int,
) -> tuple[Snapshot, list[int]]:
    """Translate one separate SWA rail into its actual physical page namespace.

    ``req_to_token`` and the radix tree retain full-pool token ids, but the attention wrapper and
    SWA writes consume ``full_to_swa``, so the checker uses the translated ids too. Unmapped
    retired prefix pages become ``None`` holes and ``first_token`` starts at the first mapped
    token. Target and MTP draft SWA buffers share this translation but stay independent pools.
    """

    ps = int(page_size)

    def mapped(slot: int) -> int:
        return int(full_to_swa[slot]) if 0 <= int(slot) < len(full_to_swa) else 0

    requests: list[RequestSnapshot] = []
    row_pages: list[tuple[RowFacts, tuple[int | None, ...], int]] = []
    retained_unread_pages: set[int] = set()
    for row in rows:
        n = int(row.kv_allocated_len)
        retired = min(max(int(row.swa_first_token), 0), n)
        raw = [mapped(slot) for slot in row.slots[:n]]
        translated = [0 if position < retired else slot for position, slot in enumerate(raw)]
        retained_unread_pages.update(page_set((slot for slot in raw[:retired] if slot > 0), ps))
        live = [i for i, slot in enumerate(translated) if slot > 0]
        first = live[0] if live else n
        blocks: list[int | None] = []
        for start in range(0, n, ps):
            values = translated[start : min(start + ps, n)]
            positive = [value for value in values if value > 0]
            if not positive:
                blocks.append(None)
                continue
            block = positive[0] // ps
            if any(value // ps != block for value in positive):
                raise ValueError(
                    f"SWA request {row.request_id!r} logical page {start // ps} maps to "
                    f"multiple physical pages"
                )
            for local, value in enumerate(values):
                expected_offset = (start + local) % ps
                if value > 0 and value % ps != expected_offset:
                    raise ValueError(
                        f"SWA request {row.request_id!r} position {start + local} maps to "
                        f"slot {value} with offset {value % ps}"
                    )
            blocks.append(block)
        if any(block is None for block in blocks[first // ps :]):
            raise ValueError(
                f"SWA request {row.request_id!r} has a mapping hole after first live token {first}"
            )
        block_tuple = tuple(blocks)
        requests.append(
            RequestSnapshot(
                row.request_id,
                pool,
                block_tuple,
                n,
                pre_step_computed(row.kv_committed_len, row.new_tokens),
                first_token=first,
            )
        )
        row_pages.append((row, block_tuple, first))

    tree_slots = [mapped(slot) for slot in tree_full_slots]
    tree_slots = [slot for slot in tree_slots if slot > 0]
    unlocked_slots = [mapped(slot) for slot in unlocked_full_slots]
    unlocked_pages = page_set((slot for slot in unlocked_slots if slot > 0), ps)
    # The walk is token-shaped. Re-derive its page sequence so a partial/sparse tail does not
    # depend on list alignment after filtering unmapped full-tree slots. Duplicate-node checks
    # remain on the authoritative full-id walk; repeated token entries inside one page are legal.
    tree_pages = list(dict.fromkeys(slot // ps for slot in tree_slots))
    duplicates: list[int] = []
    states: dict[int, BlockSnapshot] = {
        page: BlockSnapshot(
            pool,
            page,
            BlockState.PUBLISHED,
            reserved=page in unlocked_pages,
            refs=None,
        )
        for page in tree_pages
    }
    # Page-granular eviction may keep a mapped prefix page after every position it holds has
    # fallen left of this step's attention window: allocated but with no request reader. Represent
    # that slack as a reservation until the allocator's free callback retires it; this also covers
    # prefix-cache islands the full rail still locks but the SWA rail no longer consumes.
    for page in retained_unread_pages:
        states[page] = BlockSnapshot(
            pool,
            page,
            BlockState.PUBLISHED,
            reserved=True,
            refs=None,
        )
    for row, blocks, first in row_pages:
        start_page = max(int(row.cache_protected_len), first) // ps
        for block in blocks[start_page:]:
            if block is not None and block not in states:
                states[block] = BlockSnapshot(pool, block, BlockState.PRIVATE)
    return (
        Snapshot(
            seq,
            tuple(requests),
            tuple(states[page] for page in sorted(states)),
            {pool: frozenset(map(int, free_pages))},
        ),
        duplicates,
    )


def held_requests(scheduler: Any, batch: Any, *, queued: bool = True) -> list[tuple[Any, str]]:
    """The ``Req`` objects that can hold a ``req_to_token`` row at the snapshot point, each tagged
    with its source, in source order: ``running_batch.reqs``, the PDMUX split-prefill batch, the
    batch's ``reqs``, the ``chunked_req``, the P/D prefill inflight and decode queues, and, with
    ``queued``, the ``reqs`` of every batch whose result is still queued. A request may appear
    under several sources; the caller de-duplicates by row.

    The queued and P/D-inflight sources matter under overlap: a request whose queued result
    finishes it by length is dropped from the next batch before that result (and thus its
    ``release_kv_cache``) is processed, so until then it holds its row, lock and private slots in
    no running batch. A snapshot without these sources would misreport those blocks as freed and
    re-allocated without mediation.
    """
    out: list[tuple[Any, str]] = []
    running = getattr(scheduler, "running_batch", None)
    if running is not None:
        out.extend((r, "running") for r in running.reqs)
    split_prefill = getattr(scheduler, "split_prefill_batch", None)
    if split_prefill is not None:
        out.extend((r, "split_prefill") for r in split_prefill.reqs)
    out.extend((r, "batch") for r in batch.reqs)
    chunked = getattr(scheduler, "chunked_req", None)
    if chunked is not None:
        out.append((chunked, "chunked"))
    out.extend(
        (r, "pd_inflight") for r in getattr(scheduler, "disagg_prefill_inflight_queue", None) or ()
    )
    decode_prealloc = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    if decode_prealloc is not None:
        for name in ("queue", "pending_reqs"):
            out.extend(
                (entry.req, f"pd_decode_{name}")
                for entry in getattr(decode_prealloc, name, None) or ()
            )
        out.extend(
            (req, "pd_decode_retracted")
            for req in getattr(decode_prealloc, "retracted_queue", None) or ()
        )
        out.extend(
            (req, "pd_decode_rebootstrap")
            for req in getattr(decode_prealloc, "held_rebootstrap_reqs", None) or ()
        )
    decode_transfer = getattr(scheduler, "disagg_decode_transfer_queue", None)
    if decode_transfer is not None:
        out.extend(
            (entry.req, "pd_decode_transfer")
            for entry in getattr(decode_transfer, "queue", None) or ()
        )
        out.extend(
            (entry[0].req, "pd_decode_deferred")
            for entry in getattr(decode_transfer, "_deferred_releases", None) or ()
        )
    out.extend(
        (req, "pd_decode_waiting") for req in getattr(scheduler, "waiting_queue", None) or ()
    )
    if queued:
        for entry in getattr(scheduler, "result_queue", None) or ():
            queued_batch = entry[0] if isinstance(entry, tuple) else entry
            out.extend((r, "queued") for r in getattr(queued_batch, "reqs", ()))
    return out


# ------------------------------------------------------------------ synthetic faults


class Fault:
    """Synthetic fault injection (``KVG_FAULT``) applied once to the live state the kernels
    consume, to confirm the checker fires on a known-bad access. ``KINDS`` lists the supported
    faults; most are an early free of a slot/page a still-live row, descriptor or in-flight
    transfer names, plus ``padding_leak`` (a padded lane's write into a live slot), ``table_swap``
    (a consumed table entry made to disagree with the mapping), and the checker-only
    ``*_shadow_early_free`` controls that touch only the shadow.
    """

    KINDS = (
        "early_free",
        "padding_leak",
        "table_swap",
        "state_early_free",
        "hicache_load_early_free",
        "hicache_state_load_early_free",
        "external_linker_load_early_free",
        "external_linker_store_early_free",
        "eagle_branch_copy_early_free",
        "draft_shadow_early_free",
        "swa_shadow_early_free",
        "beam_member_early_free",
        "pd_decode_receive_early_free",
        "pd_decode_offload_early_free",
    )

    def __init__(self, kind: str, step: int | None = None):
        if kind not in self.KINDS:
            raise ValueError(f"unknown fault {kind!r}; known: {self.KINDS}")
        self.kind = kind
        self.step = step
        self.done = False
        self.note: str | None = None

    def _armed(self, att: SglangAttachment) -> bool:
        return not self.done and (self.step is None or att.seq >= self.step)

    def install(self, att: SglangAttachment) -> None:
        fault, runner = self, att.model_runner
        dcg = getattr(runner, "decode_cuda_graph_runner", None)
        if self.kind == "padding_leak" and dcg is not None:
            registry = dcg.buffer_registry
            orig_fill = registry.fill_from

            def fill_from(forward_batch, *args, **kwargs):
                out = orig_fill(forward_batch, *args, **kwargs)
                raw_bs, padded_bs = kwargs.get("raw_bs"), kwargs.get("padded_bs")
                if fault._armed(att) and raw_bs and padded_bs and padded_bs > raw_bs:
                    buf = registry.get_slot("out_cache_loc").slice_for(
                        padded_bs, kwargs.get("padded_num_tokens", padded_bs)
                    )
                    live = int(forward_batch.out_cache_loc[0].item())
                    buf[raw_bs] = live
                    fault.done = True
                    fault.note = f"step {att.seq}: padded lane {raw_bs} of a {padded_bs}-lane decode graph given live slot {live} (request 0's write target) in the registry's out_cache_loc"
                return out

            registry.fill_from = fill_from
        if self.kind == "table_swap":

            def swap(forward_batch, graph_bs=None):
                bs = int(forward_batch.batch_size)
                if not fault._armed(att) or bs < 2:
                    return
                if graph_bs is not None:
                    # A replay plans into decode_cuda_graph_metadata[padded bs] (see _descriptors).
                    graph_backend = _replay_full_backend(dcg)
                    w = graph_backend.decode_cuda_graph_metadata[int(graph_bs)][0]
                else:
                    w = att.backend.forward_metadata.decode_wrappers[0]
                indptr, indices = w._paged_kv_indptr_buf, w._paged_kv_indices_buf
                a, b = int(indices[0].item()), int(indices[int(indptr[1].item())].item())
                if a != b:
                    indices[0] = b
                    fault.done = True
                    fault.note = f"step {att.seq}: kv_indices entry 0 (row 0) changed from slot {a} to slot {b} (row 1's first entry) in the wrapper's consumed buffer"

            if dcg is not None:
                graph_backend = dcg.backend
                orig_replay = graph_backend.replay

                def replay(key, forward_batch, *args, **kwargs):
                    if forward_batch.forward_mode.is_decode():
                        swap(forward_batch, graph_bs=getattr(dcg, "bs", None))
                    return orig_replay(key, forward_batch, *args, **kwargs)

                graph_backend.replay = replay
            orig_fd = att.backend.forward_decode

            def forward_decode(q, k, v, layer, forward_batch, *args, **kwargs):
                swap(forward_batch)
                return orig_fd(q, k, v, layer, forward_batch, *args, **kwargs)

            att.backend.forward_decode = forward_decode

    def before_forward(self, att: SglangAttachment, batch: Any) -> None:
        if (
            self.kind
            not in (
                "early_free",
                "state_early_free",
                "beam_member_early_free",
                "draft_shadow_early_free",
                "swa_shadow_early_free",
            )
            or not self._armed(att)
            or not batch.forward_mode.is_decode()
            or not batch.reqs
        ):
            return
        import torch

        req = batch.reqs[0]
        kv = req.kv
        if self.kind == "swa_shadow_early_free":
            if not att.separate_swa:
                return
            row = att.req_pool.req_to_token[kv.req_pool_idx]
            pos = int(kv.kv_allocated_len) - 1
            full_slot = int(row[pos].item())
            slot = int(att.allocator.full_to_swa_index_mapping[full_slot].item())
            if slot <= 0:
                return
            page = slot // att.page_size
            att.ledger.free(SWA_POOL, [page], site=f"fault-swa-free@{att.seq}")
            self.done = True
            self.note = (
                f"step {att.seq}: SWA shadow page {page} (slot {slot}, full slot {full_slot}, "
                f"position {pos}) marked free before the next real SWA access"
            )
            return
        if self.kind == "draft_shadow_early_free":
            if not att.speculative_kv:
                return
            row = att.req_pool.req_to_token[kv.req_pool_idx]
            pos = int(kv.kv_allocated_len) - 1
            slot = int(row[pos].item())
            if att.speculative_multi_layer_eagle:
                slot = int(att.allocator.full_to_swa_index_mapping[slot].item())
            page = slot // att.page_size
            att.ledger.free(DRAFT_POOL, [page], site=f"fault-draft-free@{att.seq}")
            self.done = True
            self.note = (
                f"step {att.seq}: draft shadow page {page} (slot {slot}, position {pos}) "
                "marked free before the next real draft access"
            )
            return
        if self.kind == "beam_member_early_free":
            tail = getattr(batch, "beam_tail", None)
            entries = getattr(tail, "entries", None) or ()
            if not entries:
                return
            group = entries[0].group
            member_rows = _ints(group.member_rows_cpu)
            if not member_rows:
                return
            row = att.req_pool.req_to_token[member_rows[0]]
            pos = int(group.leader.kv.kv_allocated_len) - 1
            slot = int(row[pos].item())
            att.allocator.free(torch.tensor([slot], dtype=torch.int64, device=row.device))
            self.done = True
            self.note = (
                f"step {att.seq}: beam member row {member_rows[0]} slot {slot} "
                f"at position {pos} freed before its write and attention access"
            )
            return
        if self.kind == "state_early_free":
            if att.state is None or kv.mamba_pool_idx is None:
                return
            slot_t = kv.mamba_pool_idx.view(-1).clone()
            att.state.allocator.free(slot_t)
            self.done = True
            self.note = f"step {att.seq}: state slot {int(slot_t[0].item())} of request {req.rid} freed before the forward; the row's state index still names it"  # fmt: skip
            return
        row = att.req_pool.req_to_token[kv.req_pool_idx]
        pos = int(kv.kv_allocated_len) - 1
        slot = int(row[pos].item())
        att.allocator.free(torch.tensor([slot], dtype=torch.int64, device=row.device))
        self.done = True
        self.note = f"step {att.seq}: slot {slot} of request {req.rid} (position {pos}, this step's write target) freed before the forward; the row still names it"  # fmt: skip


# ------------------------------------------------------------------ Tier 1b twins (FlashInfer)

TWIN_FAULTS = ("short_entries",)
TWIN_TRACE_LIMIT = 80
TWIN_INTERPOSER_ENV = (
    "LD_PRELOAD",
    "KVGUARD_INTERPOSER_CACHE",
    "KVGUARD_INTERPOSER_MODE",
    "SGLANG_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "FLASHINFER_EXTRA_CUDAFLAGS",
)
_FI_TRAITS = (
    "mask_mode",
    "cta_tile_q",
    "num_mma_q",
    "num_mma_kv",
    "num_mma_d_qk",
    "num_mma_d_vo",
    "num_warps_q",
    "num_warps_kv",
    "pos_encoding_mode",
)


def twin_descriptor(
    *,
    counter_ptr: int,
    record_ptr: int,
    entries_ptr: int,
    indices_bytes: int,
    num_blocks: int,
    entry_bytes: int = 4,
    enabled: bool = True,
) -> list[int]:
    """The Tier 1b descriptor for FlashInfer's paged prefill twins: the kernel's ``paged_kv.indices``
    array is the table, consumed as one flat row of ``indices_bytes``; ``entries_ptr`` is a device
    ``int32[1]`` holding the number of entries the step uses (``kv_indptr[batch_size]``), the bound
    the kernel applies itself. Block size 1 with the flat slot map makes the check "entry index >=
    entries", so any load at or beyond the declared count, or outside the row, is reported."""
    from kvguard.gpu.interposer import tier1b_descriptor

    return tier1b_descriptor(
        counter_ptr=counter_ptr,
        record_ptr=record_ptr,
        table_row_bytes=indices_bytes,
        table_rows=1,
        row_len_ptr=entries_ptr,
        block_size=1,
        num_blocks=num_blocks,
        entry_bytes=entry_bytes,
        flat_slot_map=True,
        enabled=enabled,
    )


def twin_fault_entries(entries: int, fault: str | None) -> int:
    """The entry count to publish under a twin fault: ``short_entries`` declares one entry fewer
    than the kernel reads, which the kernel must report at its load of the last entry."""
    if fault is None:
        return int(entries)
    if fault == "short_entries":
        return max(int(entries) - 1, 0)
    raise ValueError(f"unknown twin fault {fault!r}; known: {TWIN_FAULTS}")


def twin_launch_delta(before: dict, after: dict) -> dict[str, int]:
    """Per-twin launches gained between two ``Interposer.substitutions()`` snapshots."""
    prior = {t["name"]: int(t["launches"]) for t in before.get("twins", [])}
    return {t["name"]: int(t["launches"]) - prior.get(t["name"], 0) for t in after.get("twins", [])}


def flashinfer_kernel_traits(name: str) -> dict[str, int] | None:
    """The ``KernelTraits`` non-type arguments carried by a mangled FlashInfer prefill kernel name
    (``MaskMode``, the MMA/warp tile counts and ``PosEncodingMode``, in template order), or
    ``None`` when the name does not carry them."""
    m = re.search(
        r"KernelTraitsILNS_8MaskModeE(\d+)E((?:Lj\d+E){7})LNS_15PosEncodingModeE(\d+)E", name
    )
    if not m:
        return None
    vals = [
        int(m.group(1)),
        *(int(x) for x in re.findall(r"Lj(\d+)E", m.group(2))),
        int(m.group(3)),
    ]
    return dict(zip(_FI_TRAITS, vals, strict=True))


TWIN_RING = 8


class TwinCounts:
    """The accounting of the twins' readbacks (torch-free): per step, the published entry count
    against the extraction's, and the in-kernel counter, cumulative over the run, as a
    difference to the previous reading."""

    def __init__(self) -> None:
        self.total_seen = 0
        self.trace: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.violations_total = 0
        self.entries_checked = 0
        self.entries_mismatch = 0

    def account(
        self,
        seq: int,
        pending: dict[str, Any] | None,
        kernel_entries: int | None,
        total: int,
        record: list[int],
        published: int,
        fault: str | None,
    ) -> int:
        """Returns the step's report count."""
        n = total - self.total_seen
        self.total_seen = total
        entry = {"seq": seq, **(pending or {"site": None}), "entries": published}
        if kernel_entries is not None:
            self.entries_checked += 1
            entry["kernel_entries"] = int(kernel_entries)
            if published != twin_fault_entries(kernel_entries, fault):
                self.entries_mismatch += 1
                entry["mismatch"] = True
        entry["reports"] = n
        if len(self.trace) < TWIN_TRACE_LIMIT:
            self.trace.append(entry)
        if n:
            self.violations_total += n
            if len(self.violations) < 20:
                self.violations.append(
                    {
                        "seq": seq,
                        "count": n,
                        "record": record,
                        "site_id": record[0],
                        "row": record[1],
                        "column": record[2],
                        "kind": record[3],
                        "entries": published,
                    }
                )
        return n


class TwinPublisher:
    """Per-step descriptor for the Tier 1b twins of FlashInfer's paged prefill kernels and the
    readback of their in-kernel counter. All wrappers are instance attributes on live objects.

    Published at two pre-kernel points: ``backend.init_forward_metadata`` after (eager steps and
    the breakable prefill graph) and the decode graph backend's ``replay`` before. At either point
    the per-step entry count is copied device-to-device into ``entries`` in stream order, and the
    descriptor is published to the interposer. The descriptor carries only the wrapper's constants,
    so it changes only between the eager and graph wrappers; the publisher synchronizes the stream
    before such a change (``desc_waits``).

    The in-kernel counter is read back without a host synchronization through a small pinned ring
    with events; it is cumulative, so a step's reports are the difference to the previous reading,
    and the record is the kernels' first violation of the run.
    """

    def __init__(self, att: SglangAttachment, fault: str | None = None):
        import torch

        from kvguard.gpu.interposer import Interposer

        from . import hook as _hook

        if fault is not None and fault not in TWIN_FAULTS:
            raise ValueError(f"unknown twin fault {fault!r}; known: {TWIN_FAULTS}")
        self.att = att
        self.fault = fault
        self.ip = Interposer()
        self.mode_at_attach = self.ip.mode()
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.record_buf = torch.zeros(4, dtype=torch.int64, device="cuda")
        self.entries = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.at_attach = self.ip.substitutions()
        self.preparation = _hook.TWINS_PREP
        self.publishes: Counter = Counter()
        self.pending: dict[str, Any] | None = None
        self.counts = TwinCounts()
        self.ring = [
            (
                torch.zeros(1, dtype=torch.int32, pin_memory=True),
                torch.zeros(4, dtype=torch.int64, pin_memory=True),
                torch.zeros(1, dtype=torch.int32, pin_memory=True),
                torch.cuda.Event(),
            )
            for _ in range(TWIN_RING)
        ]
        self.in_flight: deque[tuple[int, int, dict[str, Any] | None, int | None]] = deque()
        """(slot, seq, pending publish, kernel entries) of the readbacks not yet read."""
        self.free_slots: deque[int] = deque(range(TWIN_RING))
        self.ring_waits = 0
        self.last_desc: list[int] | None = None
        self.desc_waits = 0
        self.errors: Counter = Counter()
        self.fault_note: str | None = None
        self.graph_replay_wrapped = False

        self._install()

    # ------------------------------------------------------------------ wrap points
    def _install(self) -> None:
        me, att = self, self.att
        backend = att.backend
        orig_init = backend.init_forward_metadata

        def init_forward_metadata(forward_batch, *args, **kwargs):
            out = orig_init(forward_batch, *args, **kwargs)
            me._publish_from_metadata(forward_batch, backend.forward_metadata)
            return out

        backend.init_forward_metadata = init_forward_metadata
        dcg = getattr(att.model_runner, "decode_cuda_graph_runner", None)
        graph_backend = getattr(dcg, "backend", None)
        if graph_backend is not None and hasattr(graph_backend, "replay"):
            orig_replay = graph_backend.replay

            def replay(key, forward_batch, *args, **kwargs):
                me._publish_graph_replay(dcg)
                return orig_replay(key, forward_batch, *args, **kwargs)

            graph_backend.replay = replay
            self.graph_replay_wrapped = True

    def _publish_from_metadata(self, fb: Any, md: Any) -> None:
        site = f"init_forward_metadata:{fb.forward_mode.name.lower()}"
        try:
            if hasattr(md, "decode_wrappers"):
                wrapper = md.decode_wrappers[0]
            elif hasattr(md, "prefill_wrappers"):
                wrapper = md.prefill_wrappers[0]
            else:
                self.errors[f"no_wrappers:{type(md).__name__}"] += 1
                return
            self._publish(wrapper, site)
        except Exception as exc:
            self._error(site, exc)

    def _publish_graph_replay(self, dcg: Any) -> None:
        site = "decode_graph_replay"
        try:
            wrapper = _full_backend(dcg.attn_backend).decode_cuda_graph_metadata[int(dcg.bs)][0]
            self._publish(wrapper, site)
        except Exception as exc:
            self._error(site, exc)

    def _publish(self, wrapper: Any, site: str) -> None:
        import torch

        indptr, indices = wrapper._paged_kv_indptr_buf, wrapper._paged_kv_indices_buf
        last = indptr[-1:]
        if self.fault == "short_entries":
            self.entries.copy_(torch.clamp(last.to(torch.int32) - 1, min=0))
            self.fault_note = (
                "the entry count is published one short of kv_indptr[batch_size] at every step; "
                "the kernel's load of the last page-list entry must be reported"
            )
        else:
            self.entries.copy_(last)
        nbytes = int(indices.numel()) * int(indices.element_size())
        desc = twin_descriptor(
            counter_ptr=self.counter.data_ptr(),
            record_ptr=self.record_buf.data_ptr(),
            entries_ptr=self.entries.data_ptr(),
            indices_bytes=nbytes,
            num_blocks=self.att.geometry.num_blocks,
            entry_bytes=int(indices.element_size()),
        )
        if (
            self.last_desc is not None
            and desc != self.last_desc
            and not self.att.sync_after_forward
        ):
            # An earlier step's refresh copy or memcpy node may still be pending on the mirror.
            torch.cuda.current_stream().synchronize()
            self.desc_waits += 1
        self.ip.set_descriptor(desc)
        self.last_desc = desc
        self.publishes[site] += 1
        self.pending = {"site": site, "indices_bytes": nbytes, "indptr_len": int(indptr.numel())}

    def _error(self, site: str, exc: Exception) -> None:
        self.errors[site] += 1
        log.exception("twin descriptor publish failed at %s", site)
        self.att.reporter.emit(
            integration_error(
                "twin_publish", f"{site}: {type(exc).__name__}: {exc}", pool=POOL, site=site
            )
        )

    # ------------------------------------------------------------------ readback
    def readback(self, seq: int, kernel_entries: int | None) -> None:
        """After the step's forward was launched: read the ring slots whose events completed, then
        copy the in-kernel counter, record and published entry count into a free slot behind this
        step's kernels (no host sync; a full ring waits for its oldest slot). ``kernel_entries`` is
        the ``kv_indptr[batch_size]`` the attachment's own extraction read, compared with the
        published count when the slot is read."""
        self._collect(block=False)
        if not self.free_slots:
            self.ring_waits += 1
            self._collect(block=True, limit=1)
        slot = self.free_slots.popleft()
        counter, record, entries, event = self.ring[slot]
        counter.copy_(self.counter, non_blocking=True)
        record.copy_(self.record_buf, non_blocking=True)
        entries.copy_(self.entries, non_blocking=True)
        event.record()
        self.in_flight.append((slot, seq, self.pending, kernel_entries))
        self.pending = None

    def drain(self) -> None:
        """Wait for every outstanding readback and account it (shutdown)."""
        self._collect(block=True)

    def _collect(self, *, block: bool, limit: int | None = None) -> None:
        done = 0
        while self.in_flight and (limit is None or done < limit):
            slot, seq, pending, kernel_entries = self.in_flight[0]
            counter, record, entries, event = self.ring[slot]
            if block:
                event.synchronize()
            elif not event.query():
                break
            self.in_flight.popleft()
            self.counts.account(
                seq,
                pending,
                kernel_entries,
                int(counter.item()),
                record.tolist(),
                int(entries.item()),
                self.fault,
            )
            self.free_slots.append(slot)
            done += 1

    @property
    def violations_total(self) -> int:
        return self.counts.violations_total

    # ------------------------------------------------------------------ the record
    def outcome(self, subs: dict) -> str:
        launches = sum(int(t["launches"]) for t in subs.get("twins", []))
        if int(subs.get("mode", self.mode_at_attach)) == 0:
            return "pass_through" if launches == 0 else "pass_through_with_launches"
        if launches == 0:
            return "no_twin_launched"
        if self.fault:
            return "detected" if self.violations_total > 0 else "missed"
        return "clean" if self.violations_total == 0 else "reports_on_clean_run"

    def record(self) -> dict[str, Any]:
        from kvguard.gpu.ptx import rewrite as R

        try:
            subs = self.ip.substitutions()
        except Exception as exc:
            subs = {"error": repr(exc), "twins": []}
        return {
            "enabled": True,
            "readback": "event",
            "ring_waits": self.ring_waits,
            "desc_waits": self.desc_waits,
            "unread_slots": len(self.in_flight),
            "outcome": self.outcome(subs),
            "mode_at_attach": self.mode_at_attach,
            "mode": subs.get("mode"),
            "fault": self.fault,
            "fault_note": self.fault_note,
            "preparation": self.preparation,
            "interposer_env": {k: os.environ.get(k) for k in TWIN_INTERPOSER_ENV},
            "graph_replay_wrapped": self.graph_replay_wrapped,
            "publishes": dict(self.publishes),
            "publishes_total": sum(self.publishes.values()),
            "entries_checked": self.counts.entries_checked,
            "entries_mismatch": self.counts.entries_mismatch,
            "kernel_violations_total": self.violations_total,
            "kernel_violation_steps": self.counts.violations,
            "kernel_violation_kind_bounds": R.KIND_BOUNDS,
            "errors": dict(self.errors),
            "trace": self.counts.trace,
            "twin_launches_at_attach": sum(
                int(t["launches"]) for t in self.at_attach.get("twins", [])
            ),
            "twin_launches": sum(int(t["launches"]) for t in subs.get("twins", [])),
            "twin_launches_since_attach": twin_launch_delta(self.at_attach, subs),
            "captured_launches": subs.get("captured_launches"),
            "descriptor_version": subs.get("descriptor_version"),
            "pinned_descriptor": subs.get("pinned_descriptor"),
            "twins": [
                {
                    "name": t["name"],
                    "traits": flashinfer_kernel_traits(t["name"]),
                    "launches": t["launches"],
                    "failed": t.get("failed"),
                    "descriptor_bytes": t.get("descriptor_bytes"),
                }
                for t in subs.get("twins", [])
            ],
            "substitutions_at_attach": self.at_attach,
        }


# ------------------------------------------------------------------ GPU dispatch validators

GPU_FAULTS = ("mirror_free", "table_foreign")
GPU_FAULT_STEP_DEFAULT = 5
GPU_TRACE_LIMIT = 80
GPU_VIOLATION_LIMIT = 20


@dataclass
class DeviceBuffers:
    """The device tensors a step's kernels read (anything with ``data_ptr()`` and
    ``element_size()``): the FlashInfer page list and the KV write's slot map."""

    indptr: Any
    indices: Any
    last_page_len: Any
    loc: Any


class RequestIncarnations:
    """Stable checker request ids for live ``Req`` object incarnations.

    SGLang reuses a public ``rid`` immediately after an abort, so the old and new requests can
    appear in consecutive snapshots. Keying history by the public string would make the new
    request look like the old one rolling its computed prefix backwards; object identity (through
    a weak table that does not extend lifetimes) distinguishes the two.
    """

    def __init__(self) -> None:
        self._next = 0
        self._by_object: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}

    def __call__(self, req: Any) -> str:
        key = id(req)
        found = self._by_object.get(key)
        if found is not None and found[0]() is req:
            return found[1]
        self._next += 1
        label = f"{req.rid}#kvguard-{self._next}"

        def discard(ref: weakref.ReferenceType[Any], *, key: int = key) -> None:
            current = self._by_object.get(key)
            if current is not None and current[0] is ref:
                self._by_object.pop(key, None)

        self._by_object[key] = (weakref.ref(req, discard), label)
        return label

    def renew(self, req: Any) -> str:
        """Start a new checker incarnation for an in-place scheduler retry.

        P/D optimistic retry reuses the same ``Req`` object after releasing its row and resetting
        its computed frontier; treating that as the old execution rolling back would misjudge a
        cache block the completed attempt checkpointed.
        """
        self._by_object.pop(id(req), None)
        return self(req)


@dataclass
class StepPlan:
    """One step's consumed tables: the plain lists the host descriptors were built from, the
    descriptors in kernel order, and the device tensors the validators read. Built at the
    pre-kernel point (GPU validation on) or after the forward (host-only path)."""

    seq: int
    mode: str
    """``decode`` (also idle) or ``extend``."""
    graph: bool
    n: int
    """Real rows."""
    n_pad: int
    """Rows the attention kernel consumes: the padded decode batch, ``n`` on extend."""
    req_ids: list[str]
    seq_lens: list[int]
    prefix: list[int]
    use_ragged: bool
    indptr: list[int]
    indices: list[int]
    loc: list[int]
    descs: list[DispatchDescriptor]
    buffers: DeviceBuffers | None = None
    read_starts: list[int] | None = None
    """Per real row, the first logical position its attention table covers (a sliding-window
    or suffix gather starts above zero); ``None`` when every row reads from position zero."""

    @property
    def attn_first(self) -> bool:
        return bool(self.descs) and self.descs[0].family == "attention"


def validator_rows(
    plan: StepPlan, owner: Callable[[str], int]
) -> tuple[tuple[int, list[int], list[int]], tuple[int, list[int]]]:
    """The validators' row arrays for one step: ``(rows, row_owner, row_end)`` for the CSR
    validator over the page list and ``(n_tokens, tok_owner)`` for the slot-map validator.

    Decode lists one row per (padded) lane, a real row reading ``[0, seq_len)`` and a padding lane
    ``PADDING_ROW``; extend lists ``n`` rows reading the cached prefix (``use_ragged``) or
    ``[0, seq_len)``, with entries beyond the batch's tokens treated as padding tokens.
    """
    n = plan.n
    owners = [owner(r) for r in plan.req_ids[:n]]
    if plan.mode == "extend":
        ends = plan.prefix[:n] if plan.use_ragged else plan.seq_lens[:n]
        rows = (n, owners, [int(e) for e in ends])
        tok: list[int] = []
        for i in range(n):
            tok.extend([owners[i]] * max(int(plan.seq_lens[i]) - int(plan.prefix[i]), 0))
    elif plan.mode == "verify":
        pad = max(plan.n_pad - n, 0)
        rows = (
            n + pad,
            owners + [PADDING_ROW] * pad,
            [int(s) for s in plan.seq_lens[:n]] + [0] * pad,
        )
        tok = []
        for i in range(n):
            tok.extend([owners[i]] * max(int(plan.seq_lens[i]) - int(plan.prefix[i]), 0))
    else:
        pad = max(plan.n_pad - n, 0)
        rows = (
            n + pad,
            owners + [PADDING_ROW] * pad,
            [int(s) for s in plan.seq_lens[:n]] + [0] * pad,
        )
        tok = list(owners)
    n_tokens = len(plan.loc)
    tok = tok[:n_tokens] + [PADDING_ROW] * max(n_tokens - len(tok), 0)
    return rows, (n_tokens, tok)


def validator_targets(
    plan: StepPlan, pages_of: Callable[[str], Sequence[int | None]], page_size: int
) -> tuple[list[int], list[Sequence[int | None]], list[int | None]]:
    """The device validators' mapping arrays for one step, against which the device tables are
    checked (``mapping_mismatch``): per attention row its read start and page table (the ledger's
    mapping, ``None`` for a hole), and per slot-map token the slot the mapping gives its position
    (``None`` for a padding token or a position without a page)."""
    n = plan.n
    pad = max(plan.n_pad - n, 0) if plan.mode != "extend" else 0
    starts = [int(x) for x in plan.read_starts[:n]] if plan.read_starts else [0] * n
    starts = starts + [0] * (n - len(starts)) + [0] * pad
    pages = [tuple(pages_of(rid)) for rid in plan.req_ids[:n]]
    bs = int(page_size)

    def slot(i: int, p: int) -> int | None:
        k = p // bs
        page = pages[i][k] if 0 <= i < n and k < len(pages[i]) else None
        return None if page is None else int(page) * bs + p % bs

    expected: list[int | None] = [None] * len(plan.loc)
    write = next((d for d in plan.descs if d.family == "set_kv_buffer"), None)
    mapping = getattr(write, "mapping", None)
    if mapping is not None and hasattr(mapping, "positions"):
        for t, (r, p) in enumerate(zip(mapping.rows, mapping.positions, strict=True)):
            if t < len(expected) and 0 <= int(r) < n:
                expected[t] = slot(int(r), int(p))
    return starts, pages, expected


class GpuValidation:
    """The device validators under the attachment (``gpu_validate=True``): one ``Validators``, the
    pool's ``DeviceMirror``, the per-step launches in the host's dispatch order, the readback into
    the attachment's reporter, and two synthetic faults.

    ``run_step`` is the order rule: the slot-map validator must see the mirror before this step's
    write and the attention validator after it, so sync, slot map, host ``dispatch(write)``, sync,
    CSR validator, host ``dispatch(attn)``. With ``use_ragged`` the attention reads only the
    cached prefix and runs before the write.

    Faults (``KVGUARD_SGLANG_GPU_FAULT`` from step ``KVGUARD_SGLANG_GPU_FAULT_STEP``, once):
    ``mirror_free`` overwrites the mirror words of a written page to free (visible only to the
    device validator), and ``table_foreign`` overwrites a wrapper ``kv_indices`` entry with a
    foreign page (the host reports the mapping disagreement, the device the foreign access).
    """

    def __init__(
        self,
        ledger: Ledger,
        reporter: Reporter,
        *,
        stats: Counter | None = None,
        timings: Counter | None = None,
        fault: str | None = None,
        fault_step: int | None = None,
        validators: Any = None,
        mirror: Any = None,
    ):
        if fault is not None and fault not in GPU_FAULTS:
            raise ValueError(f"unknown GPU fault {fault!r}; known: {GPU_FAULTS}")
        if validators is None:
            from kvguard.gpu.validate import Validators

            validators = Validators(timing=True)
        if mirror is None:
            mirror = validators.mirror(ledger, POOL)
        self.ledger = ledger
        self.reporter = reporter
        self.v = validators
        self.mirror = mirror
        self.stats = stats if stats is not None else Counter()
        self.timings = timings if timings is not None else Counter()
        self.fault = fault
        self.fault_step = GPU_FAULT_STEP_DEFAULT if fault_step is None else int(fault_step)
        self.fault_done = False
        self.fault_note: str | None = None
        self.fault_info: dict[str, Any] | None = None
        self.errors: Counter = Counter()
        self.trace: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.violations_total = 0
        self.launches_with_reports = 0
        self.results = 0
        self.reports = 0
        self.steps = 0
        self._pending: dict[int, dict[str, Any]] = {}
        self.tags: Any = None
        """The tag tier sharing this loader, whose results the collector routes to it."""

    # ------------------------------------------------------------------ the step
    def run_step(
        self, plan: StepPlan, dispatch: Callable[[DispatchDescriptor], None], stream: int
    ) -> None:
        """Launch the step's validators on ``stream`` around the host dispatches (``dispatch`` is
        the attachment's ``ledger.dispatch`` wrapper), in the order the docstring states."""
        buf = plan.buffers
        if buf is None:
            raise ValueError("the plan carries no device buffers")
        for name, t in (("indptr", buf.indptr), ("indices", buf.indices)):
            if int(t.element_size()) != 4:
                raise ValueError(
                    f"{name} buffer has {t.element_size()}-byte elements; the page-list validator reads int32"
                )
        write = next(d for d in plan.descs if d.family == "set_kv_buffer")
        attn = next(d for d in plan.descs if d.family == "attention")
        (rows, row_owner, row_end), (n_tokens, tok_owner) = validator_rows(
            plan, self.mirror.owners.id_of
        )
        # The mapping the device tables are checked against: the ledger's page table per
        # request (the snapshot this step observed) and the write slots it gives the positions.
        row_start, pages, expected = validator_targets(
            plan, self._pages_of, self.ledger.pools[POOL].block_size
        )
        t0 = time.perf_counter()
        host = 0.0

        def host_dispatch(d: DispatchDescriptor) -> None:
            nonlocal host
            t = time.perf_counter()
            dispatch(d)
            host += time.perf_counter() - t

        def launch_attn() -> None:
            self.v.validate_token_csr(
                self.mirror,
                buf.indptr.data_ptr(),
                buf.indices.data_ptr(),
                rows,
                row_owner,
                row_start,
                row_end,
                pages,
                stream,
                attn.dispatch_id,
            )

        def launch_write() -> None:
            self.v.validate_slot_map(
                self.mirror,
                buf.loc.data_ptr(),
                int(buf.loc.element_size()),
                n_tokens,
                tok_owner,
                stream,
                write.dispatch_id,
                expected=expected,
            )

        self.mirror.sync(stream)
        if plan.attn_first:
            launch_attn()
            launch_write()
            host_dispatch(attn)
            host_dispatch(write)
        else:
            launch_write()
            host_dispatch(write)
            self.mirror.sync(stream)
            self._inject_mirror_fault(plan, stream)
            launch_attn()
            host_dispatch(attn)
        self.steps += 1
        host_us = ((time.perf_counter() - t0) - host) * 1e6
        self.stats["gpu_validator_launches"] += 2
        self.stats["gpu_validate_steps"] += 1
        self.timings["gpu_validate_host_s"] += host_us / 1e6
        self._pending[plan.seq] = {
            "seq": plan.seq,
            "mode": plan.mode,
            "graph": plan.graph,
            "rows": rows,
            "tokens": n_tokens,
            "host_us": round(host_us, 1),
        }

    def _pages_of(self, rid: str) -> Sequence[int | None]:
        """The request's page table as the ledger last observed it (empty when unknown)."""
        shadow = self.ledger.requests.get((rid, POOL))
        return shadow.snapshot.blocks if shadow is not None else ()

    # ------------------------------------------------------------------ faults
    def _armed(self, kind: str, seq: int) -> bool:
        return self.fault == kind and not self.fault_done and seq >= self.fault_step

    def _inject_mirror_fault(self, plan: StepPlan, stream: int) -> None:
        if not self._armed("mirror_free", plan.seq) or plan.n < 1 or len(plan.indptr) < 2:
            return
        lo, hi = int(plan.indptr[0]), int(plan.indptr[1])
        if hi <= lo:
            return
        # The entries are token slots at any page size: the page is the slot's block.
        block = int(plan.indices[hi - 1]) // int(self.ledger.pools[POOL].block_size)
        self.mirror.set_words(block, 0, 0, 0, stream)
        self.fault_done = True
        self.fault_info = {"step": plan.seq, "block": block, "row": 0, "request": plan.req_ids[0]}
        self.fault_note = (
            f"step {plan.seq}: mirror words of slot {block} (row 0's last page, request "
            f"{plan.req_ids[0]!r}, written this step) set to free after the write's sync; the "
            "ledger's shadow is untouched"
        )

    def inject_table_fault(
        self, indptr_t: Any, indices_t: Any, req_ids: list[str], seq: int
    ) -> bool:
        """``table_foreign`` on the wrapper's device buffers (decode, two or more rows): entry 0
        becomes a page private to row 1's request. Returns whether it fired."""
        if not self._armed("table_foreign", seq) or len(req_ids) < 2:
            return False
        indptr = _ints(indptr_t)
        if len(indptr) < 3:
            return False
        lo, hi = int(indptr[1]), int(indptr[2])
        if hi <= lo:
            return False
        geometry = self.ledger.pools[POOL]
        num_slots = geometry.num_blocks * geometry.block_size
        victim = None
        for s in reversed(_ints(indices_t[lo:hi])):
            if 0 <= s < num_slots:  # a token slot at any page size: its page is the block
                sh = self.ledger.block(POOL, s // geometry.block_size)
                if sh.state is BlockState.PRIVATE and sh.owner == req_ids[1]:
                    victim = s
                    break
        if victim is None:
            self.errors["table_foreign_no_private_page"] += 1
            return False
        old = _ints(indices_t[0:1])[0]
        indices_t[0] = victim
        self.fault_done = True
        self.fault_info = {
            "step": seq,
            "row": 0,
            "request": req_ids[0],
            "old": old,
            "block": victim,
            "owner": req_ids[1],
        }
        self.fault_note = (
            f"step {seq}: kv_indices entry 0 (row 0, request {req_ids[0]!r}) changed on the "
            f"device from slot {old} to slot {victim}, private to row 1's request "
            f"{req_ids[1]!r}, before the tables were extracted"
        )
        return True

    # ------------------------------------------------------------------ readback
    def readback(self, seq: int | None = None) -> dict[str, Any]:
        """The results of the launches that completed (no host synchronization), reported."""
        return self._collect(self.v.completed(), seq)

    def drain(self) -> dict[str, Any]:
        """Wait for every outstanding launch and report its result (shutdown)."""
        return self._collect(self.v.drain(), None)

    def _collect(self, results: Sequence[Any], seq: int | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.tags is not None and results:
            results = self.tags.collect(results, self.reporter)
        n_rep = report_results(results, self.reporter, POOL) if results else 0
        self.results += len(results)
        self.reports += n_rep
        self.stats["gpu_results"] += len(results)
        self.stats["gpu_reports"] += n_rep
        gpu_ms = 0.0
        count = 0
        for r in results:
            if r.gpu_ms is not None:
                gpu_ms += r.gpu_ms
            if r.count:
                self.launches_with_reports += 1
                count += r.count
                if len(self.violations) < GPU_VIOLATION_LIMIT:
                    self.violations.append(
                        {
                            "seq": seq,
                            "tag": r.tag,
                            "count": r.count,
                            "kind": r.kind,
                            "name": r.name,
                            "row": r.row,
                            "index": r.index,
                            "entry": r.entry,
                        }
                    )
        self.violations_total += count
        self.stats["gpu_violations"] += count
        self.timings["gpu_validate_device_ms"] += gpu_ms
        self.timings["gpu_readback_s"] += time.perf_counter() - t0
        summary = {
            "gpu_results": len(results),
            "gpu_reports": n_rep,
            "gpu_violations": count,
            "gpu_ms": round(gpu_ms, 4),
        }
        entry = self._pending.pop(seq, None) if seq is not None else None
        if entry is not None:
            entry.update(summary)
            if len(self.trace) < GPU_TRACE_LIMIT:
                self.trace.append(entry)
        return summary

    # ------------------------------------------------------------------ the record
    def outcome(self) -> str:
        if self.steps == 0:
            return "no_launches"
        if self.fault:
            if not self.fault_done:
                return "fault_not_injected"
            return "detected" if self.violations_total else "missed"
        return "clean" if self.violations_total == 0 else "reports_on_clean_run"

    def record(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "outcome": self.outcome(),
            "fault": self.fault,
            "fault_step": self.fault_step,
            "fault_done": self.fault_done,
            "fault_note": self.fault_note,
            "fault_info": self.fault_info,
            "steps": self.steps,
            "launches": getattr(self.v, "launches", None),
            "results": self.results,
            "launches_with_reports": self.launches_with_reports,
            "device_violations_total": self.violations_total,
            "reports_emitted": self.reports,
            "gpu_ms_total": round(float(getattr(self.v, "gpu_ms_total", 0.0) or 0.0), 4),
            "mirror_updates": getattr(self.mirror, "updates", None),
            "mirror_blocks_updated": getattr(self.mirror, "blocks_updated", None),
            "mirror_bytes": getattr(self.mirror, "nbytes", None),
            "validators_device_bytes": (
                self.v.device_bytes() if hasattr(self.v, "device_bytes") else None
            ),
            "owner_ids": len(self.mirror.owners),
            "errors": dict(self.errors),
            "first_violations": self.violations,
            "trace": self.trace,
        }


# ------------------------------------------------------------------ the attachment


@dataclass
class StepEpoch:
    seq: int
    dispatch_ids: list[str]
    event: Any
    forward_iter: int | None = None


class SglangAttachment:
    """Wrappers installed on one live scheduler, feeding one ledger."""

    def __init__(
        self,
        scheduler: Any,
        reporter: Reporter | None = None,
        *,
        out_path: str | None = None,
        early_complete: bool = False,
        fault: str | None = None,
        fault_step: int | None = None,
        environment: Callable[[], dict] | None = None,
        twins: bool = False,
        twin_fault: str | None = None,
        gpu_validate: bool = False,
        tags: bool = False,
        nvtx: bool = False,
        gpu_fault: str | None = None,
        gpu_fault_step: int | None = None,
        validators: Any = None,
        mirror: Any = None,
        sync_after_forward: bool = True,
        prekernel: bool = False,
        tables: str = "device",
        row_cache: bool = True,
        audit_every: int = 64,
        queued_rows: bool = True,
        stream_epochs: bool = True,
        profile: str | None = None,
        profile_steps: tuple[int, int] = (100, 400),
    ):
        self.scheduler = scheduler
        self.stream_epochs = stream_epochs
        """Declare the forward stream in every descriptor and the scheduler stream plus observed
        WAR fence in every snapshot, so the core judges releases by stream order; ``False`` keeps
        host-order judgment."""
        self._fence_seq = 0
        """The last step whose ``_apply_war_barrier`` ran: the scheduler stream has waited behind
        that forward's reads, so the open steps up to it are ordered before the next snapshot's
        table writes."""
        self.sync_after_forward = sync_after_forward
        self.prekernel = prekernel
        if tables not in ("device", "mapping"):
            raise ValueError("tables must be 'device' or 'mapping'")
        self.tables = tables
        self.row_cache_enabled = bool(row_cache)
        self.row_cache: RowSlotCache | None = None
        """The rows' slots kept between snapshots, on the ordinary paths only (not speculative
        decoding, PDMUX, separate SWA pools, recurrent state or beam search); audited with the
        walk. ``tables='device'`` reads back the page lists and write slots the kernels consume;
        ``tables='mapping'`` builds the descriptors from the ledger's page tables alone and leaves
        the device copies to the device validators (plain decode/extend paths only)."""
        self.audit_every = max(1, int(audit_every))
        """The allocator's free set is read every ``audit_every``-th snapshot (the partition audit
        by set); the other snapshots carry its count. The set is read every step while another
        pool shares the snapshot (EAGLE3 draft, a separate SWA rail, a recurrent-state pool)."""
        self.queued_rows = queued_rows
        """Read the rows of the requests whose results are still queued (``held_requests``);
        ``False`` shows the reports of a snapshot that misses them (a diagnostic)."""
        self.allocator = scheduler.token_to_kv_pool_allocator
        self.tree = scheduler.tree_cache
        self.walk_states = WalkStates(POOL)  # the page size is set once the facts are read
        self.req_pool = scheduler.req_to_token_pool
        self.model_runner = scheduler.tp_worker.model_runner
        self.backend = self.model_runner.attn_backend
        """The backend whose wrappers hold the KV tables: the full-attention backend of a hybrid
        model's composite (reassigned once the facts are read)."""
        self.reporter = reporter or Reporter(fail_fast=False)
        self.out_path = out_path
        self.early_complete = early_complete
        self.fault = Fault(fault, fault_step) if fault else None
        self._environment = environment
        self._environment_cache: dict | None = None
        self.facts = check_support(scheduler, self.reporter)
        self.speculative_ngram = bool(self.facts.get("speculative_ngram"))
        self.speculative_nextn = bool(self.facts.get("speculative_nextn"))
        self.speculative_dflash = bool(self.facts.get("speculative_dflash_supported"))
        self.speculative_multi_layer_eagle = bool(
            self.facts.get("speculative_multi_layer_eagle_supported")
        )
        self.separate_swa = bool(self.facts.get("separate_pool_swa"))
        # The existing sidecar machinery is algorithm-agnostic: both EAGLE3 and Qwen3.5
        # NEXTN use EAGLEWorkerV2 with an independently backed draft MHA cache.
        self.speculative_eagle3 = bool(self.facts.get("speculative_single_draft_sidecar"))
        self.speculative_kv = self.speculative_eagle3 or self.speculative_multi_layer_eagle
        self.hicache = bool(self.facts.get("hierarchical_cache"))
        self.external_linker = bool(self.facts.get("external_linker_supported"))
        self.disagg_prefill = bool(self.facts.get("disagg_prefill"))
        self.disagg_decode = bool(self.facts.get("disagg_decode"))
        self.pdmux = bool(self.facts.get("pdmux"))
        if self.pdmux and twins:
            report = unsupported(
                "pdmux_device_checks",
                "the bounded PDMUX port supports its per-stream GPU validators, but its "
                "Tier 1b publisher is not yet supported",
                pool=POOL,
                site="attach",
            )
            self.reporter.emit(report)
            raise Unsupported(report)
        if self.speculative_kv and (twins or gpu_validate):
            report = unsupported(
                "eagle3_device_checks",
                "the bounded speculative ports cover the target and independent draft pools with "
                "host descriptors; GPU validators and Tier 1b twins are not yet supported",
                pool=DRAFT_POOL,
                site="attach",
            )
            self.reporter.emit(report)
            raise Unsupported(report)
        if self.separate_swa and gpu_validate:
            report = unsupported(
                "separate_swa_device_checks",
                "the independent full/SWA port covers both physical pools with host "
                "descriptors; its two-pool GPU validator is not yet supported",
                pool=SWA_POOL,
                site="attach",
            )
            self.reporter.emit(report)
            raise Unsupported(report)
        if self.external_linker and (twins or gpu_validate):
            report = unsupported(
                "external_linker_device_checks",
                "the bounded external-linker port covers host ledger checks; GPU validators and "
                "Tier 1b twins are not yet supported",
                pool=POOL,
                site="attach",
            )
            self.reporter.emit(report)
            raise Unsupported(report)
        if twins and self.facts["num_wrappers"] != 1:
            report = unsupported(
                "twins",
                "Tier 1b publishes one FlashInfer wrapper; multi-wrapper attention needs one "
                "descriptor per wrapper",
                pool=POOL,
                site="attach",
            )
            self.reporter.emit(report)
            raise Unsupported(report)
        self.ledger = Ledger(self.reporter)
        full_allocator = (
            self.allocator.full_attn_allocator
            if self.separate_swa or self.facts.get("unified_allocator")
            else self.allocator
        )
        self.geometry = geometry_from_allocator(
            full_allocator,
            POOL,
            layers=max(self.facts["layers"], 1),
            kv_heads=max(self.facts["kv_heads"], 1),
        )
        if self.facts.get("unified_allocator"):
            # The owner band begins above the arena's state extent.  Those lower virtual
            # page numbers are outside the owner's namespace, just like page zero.
            self.geometry = dataclasses.replace(
                self.geometry,
                num_blocks=int(full_allocator.ids.num_virtual_ids),
                reserved_blocks=frozenset(range(int(full_allocator.min_page_index))),
            )
        self.ledger.register_pool(self.geometry)
        self.swa_geometry = None
        if self.separate_swa:
            self.swa_geometry = geometry_from_allocator(
                self.allocator.swa_attn_allocator,
                SWA_POOL,
                layers=self.facts["kv_pool_swa_layers"],
                kv_heads=max(
                    int(getattr(self.allocator.get_kvcache().swa_kv_pool, "head_num", 0) or 0),
                    1,
                ),
            )
            self.ledger.register_pool(self.swa_geometry)
        self.draft_geometry = None
        self.draft_runner = None
        self.draft_runners: tuple[Any, ...] = ()
        self.draft_pool_ids: tuple[str, ...] = ()
        self.draft_geometries: tuple[Any, ...] = ()
        if self.speculative_eagle3:
            self.draft_runner = (
                scheduler.draft_worker.draft_model_runner
                if self.speculative_dflash
                else scheduler.draft_worker.draft_worker.draft_runner
            )
            draft_allocator = (
                self.allocator.full_attn_allocator
                if self.facts.get("draft_storage_fused_with_owner")
                else self.allocator
            )
            self.draft_geometry = geometry_from_allocator(
                draft_allocator,
                DRAFT_POOL,
                layers=self.facts["draft_layers"],
                kv_heads=self.facts["draft_kv_heads"],
            )
            if self.facts.get("draft_storage_fused_with_owner"):
                self.draft_geometry = dataclasses.replace(
                    self.draft_geometry,
                    num_blocks=int(draft_allocator.ids.num_virtual_ids),
                    reserved_blocks=frozenset(range(int(draft_allocator.min_page_index))),
                )
            self.ledger.register_pool(self.draft_geometry)
            self.draft_runners = (self.draft_runner,)
            self.draft_pool_ids = (DRAFT_POOL,)
            self.draft_geometries = (self.draft_geometry,)
        elif self.speculative_multi_layer_eagle:
            self.draft_runners = tuple(scheduler.draft_worker.draft_worker.draft_runners)
            self.draft_runner = self.draft_runners[0]
            self.draft_pool_ids = tuple(
                multi_draft_pool(index) for index in range(len(self.draft_runners))
            )
            geometries = []
            for pool_id, runner in zip(self.draft_pool_ids, self.draft_runners, strict=True):
                draft_swa = runner.token_to_kv_pool.swa_kv_pool
                geometry = geometry_from_allocator(
                    self.allocator.swa_attn_allocator,
                    pool_id,
                    layers=1,
                    kv_heads=max(int(getattr(draft_swa, "head_num", 0) or 0), 1),
                )
                self.ledger.register_pool(geometry)
                geometries.append(geometry)
            self.draft_geometries = tuple(geometries)
            self.draft_geometry = self.draft_geometries[0]
        self.page_size = int(self.facts["page_size"])
        self.paged = bool(self.facts.get("paged"))
        self.walk_states = WalkStates(POOL, self.page_size)
        self.draft_walk_states = (
            WalkStates(DRAFT_POOL, self.page_size) if self.speculative_eagle3 else None
        )
        self.backend = _full_backend(self.model_runner.attn_backend)
        self.state: StateMediation | None = None
        if self.facts.get("hybrid"):
            state_storage = (
                getattr(self.allocator, "state", None) if self.facts.get("unified_hybrid") else None
            )
            self.state_geometry = state_geometry(
                scheduler.req_to_token_pool,
                layers=self.facts.get("linear_layers", 1),
                state_storage=state_storage,
            )
            self.ledger.register_pool(self.state_geometry)
            ckpt = (
                getattr(state_storage, "ckpt_pool", None)
                if state_storage is not None
                else getattr(scheduler.req_to_token_pool, "mamba_ckpt_pool", None)
            )
            if ckpt is not None:
                self.ledger.register_pool(
                    ckpt_geometry(ckpt, layers=self.facts.get("linear_layers", 1))
                )
        self.seq = 0
        self.cur_batch: Any = None
        self.request_incarnations = RequestIncarnations()
        self.rid_of_row: dict[int, str] = {}
        self._beam_shared_slots: set[int] = set()
        self._beam_reserved_slots: set[int] = set()
        self._beam_unsupported_emitted = False
        self.open: dict[int, StepEpoch] = {}
        self.results: dict[int, tuple[int, Any]] = {}
        self.last_static_fb: Any = None
        self.stats: Counter = Counter()
        self.timings: Counter = Counter()
        self.flush_count = 0
        self.idle_flush_count = 0
        self.events = 0
        self._idle_flushed_at = 0
        self.trace: list[dict[str, Any]] = []
        self._last_dispatch: dict[str, Any] = {}
        self._last_write_blocks: tuple[int, ...] = ()
        self._pdmux_submitted_batches: set[int] = set()
        """Split-prefill batch objects whose write dispatch has been submitted but whose
        completion result may not yet have been admitted to the running decode batch."""
        self.dirty = False
        self._last_flush = 0.0
        self.started = time.time()
        self.gpu: GpuValidation | None = None
        self.tags: Any = None
        """The tag tier over the KV pool (``tags=True``; ``kvguard.adapters.sglang.tags``)."""
        self._prestep: dict[int, list[str]] = {}
        self._verify_requests: list[tuple[str, int]] = []
        """The verify rows in array order as ``(request, candidate_start)``."""
        self._ngram_copy_plan: tuple[list[int], list[int], int] | None = None
        """Accepted lengths, flattened source indices and their row width during compaction."""
        self._draft_rows: list[RowFacts] = []
        self._draft_rows_by_pool: dict[str, list[RowFacts]] = {}
        self._draft_forward_index = 0
        self._draft_prefill_graph_seq: int | None = None
        self._draft_tree_copy_context: tuple[str, ...] | None = None
        self._draft_tree_copy_rolled_back: set[str] = set()
        self._queued_only = 0
        self._hicache_seq = 0
        self._hicache_reads: dict[int, tuple[Any, tuple[str, ...]]] = {}
        self._hicache_store_reservations: dict[int, tuple[int, ...]] = {}
        self._hicache_loads: dict[int, tuple[Any, tuple[tuple[str, tuple[int, ...]], ...]]] = {}
        self._external_linker_seq = 0
        self._external_linker_context: dict[str, Any] | None = None
        self._external_linker_pending_loads: dict[str, dict[str, tuple[int, ...]]] = {}
        self._external_linker_load_jobs: dict[int, dict[str, Any]] = {}
        self._external_linker_store_jobs: deque[dict[str, Any]] = deque()
        self._external_linker_lock = threading.RLock()
        self._decode_offload_reads: dict[int, tuple[Any, tuple[str, ...]]] = {}
        self._decode_offload_owner: str | None = None
        self._decode_offload_trace: list[dict[str, Any]] = []
        self._pd_seq = 0
        self._pd_reads: dict[int, list[str]] = {}
        self._pd_senders: dict[int, Any] = {}
        self._pd_failed_pending: set[int] = set()
        self._pd_reserved_blocks: dict[str, tuple[int, ...]] = {}
        self._pd_trace: list[dict[str, Any]] = []
        self._pd_source_release_trace: list[dict[str, Any]] = []
        self._pd_decode_writes: dict[int, list[tuple[str, str, tuple[int, ...], range, str]]] = {}
        self._pd_decode_receivers: dict[int, Any] = {}
        self._pd_decode_meta: dict[int, dict[str, Any]] = {}
        """Per-receiver destination layout for the bounded-fill at completion: the room, the
        logical destination pages per pool, the published token count, the state slots and the
        owner."""
        self._pd_decode_trace: list[dict[str, Any]] = []
        # The P/D transfer-extent channel: a disaggregated transfer writes only the sender's
        # committed tokens, which the decode side cannot learn from a successful poll alone. With
        # this directory set on both roles, the prefill records the tokens it actually sent per
        # bootstrap room and the decode fills destination initialization only up to that extent,
        # leaving the untransferred reservation uninitialized so a later read reports. Unset: the
        # decode fills the full published destination.
        self._pd_extent_dir = os.environ.get(ENV_PD_EXTENT_DIR) or None
        self._pd_sent_tokens: dict[Any, int] = {}
        self._pd_decode_rooms: dict[int, Any] = {}
        self._pd_retraction_restores: dict[
            int, tuple[tuple[str, tuple[int, ...], range, str], ...]
        ] = {}
        """Host-to-device restore epochs keyed by the framework completion event."""
        self._pd_retraction_deadlock_reported = False
        self._pd_preallocation_capacity_reported = False
        self.profile_path = profile
        self.profile_steps = profile_steps
        self.profiler: cProfile.Profile | None = None
        if profile:
            # The attachment's hot entry points run under cProfile for the steps in the window;
            # the stats are written at the loop exit (``flush``). A diagnostic, not a mode.
            self.profiler = cProfile.Profile()
            for name in (
                "_before_run_batch",
                "_before_kernels",
                "_after_forward",
                "_before_process_result",
            ):
                setattr(self, name, self._profiled(getattr(self, name)))
        if self.facts.get("hybrid"):
            state_storage = (
                getattr(self.allocator, "state", None) if self.facts.get("unified_hybrid") else None
            )
            self.state = StateMediation(
                self,
                scheduler.req_to_token_pool,
                getattr(self.tree, "tree_core", None),
                state_storage=state_storage,
                state_band=(
                    getattr(self.allocator, "mamba_allocator", None)
                    if state_storage is not None
                    else None
                ),
            )
        self._install()
        # The device page-list validator takes FlashInfer's per-token entries at any page size
        # (``kvg_validate_token_csr``), so a paged pool does not refuse it.
        if gpu_validate:
            try:
                self.gpu = GpuValidation(
                    self.ledger,
                    self.reporter,
                    stats=self.stats,
                    timings=self.timings,
                    fault=gpu_fault,
                    fault_step=gpu_fault_step,
                    validators=validators,
                    mirror=mirror,
                )
            except Exception as exc:
                log.exception("GPU validation not installed")
                self.reporter.emit(
                    integration_error(
                        "gpu_validate_unavailable",
                        f"{type(exc).__name__}: {exc}",
                        pool=POOL,
                        site="attach",
                    )
                )
        if tags and (self.separate_swa or self.speculative_kv or self.external_linker):
            # The tag tier launches at the pre-kernel point, which these ports do not take:
            # the independent full/SWA port's pre-kernel dispatch, the EAGLE3/DFlash draft
            # pools' and the external linker's device paths are host-only (as for the device
            # validators above). The attachment stays host-only; the tier is not attached.
            which = (
                "separate_swa" if self.separate_swa
                else "speculative_kv" if self.speculative_kv
                else "external_linker"
            )  # fmt: skip
            self.reporter.emit(
                unsupported(
                    "tag_tier",
                    f"the {which} port covers host ledger checks; the tag tier's pre-kernel "
                    "launches are not yet supported there",
                    pool=POOL,
                    site="attach",
                )
            )
            tags = False
        self.nvtx = None
        """The Compute Sanitizer baseline's NVTX memory registrations (``nvtx=True``): the
        attention pool's per-layer K and V tensors as heaps, every page a region per tensor
        (the page's slab is contiguous in each layer tensor, token-major or page-major)."""
        if nvtx:
            try:
                from kvguard.gpu.nvtxmem import NvtxMemory

                if (
                    self.facts.get("hybrid")
                    or self.facts.get("mla")
                    or self.separate_swa
                    or self.facts.get("unified_allocator")
                ):
                    raise RuntimeError(
                        "the NVTX baseline covers the single full-attention pool only"
                    )
                cache = self.allocator.get_kvcache()
                buffers = [*getattr(cache, "k_buffer", []), *getattr(cache, "v_buffer", [])]
                if not buffers:
                    raise RuntimeError("the NVTX baseline found no K/V buffers on the pool")
                g = self.geometry
                views = []
                for t in buffers:
                    if int(t.shape[0]) == g.num_blocks:
                        views.append(t)  # page-major: the page is the leading axis
                    elif int(t.shape[0]) == g.num_blocks * g.block_size:
                        views.append(
                            t.view(g.num_blocks, g.block_size, *t.shape[1:])
                        )  # token-major
                    else:
                        raise RuntimeError(
                            f"the NVTX baseline cannot map a buffer of shape {tuple(t.shape)} onto "
                            f"{g.num_blocks} pages of {g.block_size}"
                        )
                self.nvtx = NvtxMemory(self.ledger)
                self.nvtx.add_pool(POOL, views, g.num_blocks, reserved=g.reserved_blocks)
                self.stats["nvtx_lanes"] = len(self.nvtx.lanes[POOL])
            except Exception as exc:
                log.exception("NVTX baseline not installed")
                self.reporter.emit(
                    integration_error(
                        f"NVTX baseline not installed: {exc!r}",
                        pool=POOL,
                        site="attach",
                    )
                )
                self.nvtx = None
        if tags:
            try:
                from .tags import SglangTags

                self.tags = SglangTags(self, POOL)
                if self.gpu is not None:
                    self.gpu.tags = self.tags
                self._install_tag_hooks()
            except Exception as exc:
                log.exception("tag tier not installed")
                self.reporter.emit(
                    integration_error(
                        "tags_unavailable",
                        f"{type(exc).__name__}: {exc}"[:300],
                        pool=POOL,
                        site="attach",
                    )
                )
        self.twins: TwinPublisher | None = None
        if twins:
            try:
                self.twins = TwinPublisher(self, twin_fault)
            except Exception as exc:
                log.exception("twin publisher not installed")
                self.reporter.emit(
                    integration_error(
                        "twins_unavailable",
                        f"{type(exc).__name__}: {exc}",
                        pool=POOL,
                        site="attach",
                    )
                )

    # ------------------------------------------------------------------ wrappers
    def _cache_pools(self) -> tuple[str, ...]:
        """Physical KV pools governed by the target allocator's slot lifecycle."""
        return self._full_cache_pools() + self._swa_cache_pools()

    def _full_cache_pools(self) -> tuple[str, ...]:
        if self.speculative_eagle3:
            return (POOL, DRAFT_POOL)
        return (POOL,)

    def _swa_cache_pools(self) -> tuple[str, ...]:
        if not self.separate_swa:
            return ()
        return (SWA_POOL, *self.draft_pool_ids)

    def _install(self) -> None:
        me, alloc, sched, runner = self, self.allocator, self.scheduler, self.model_runner

        if self.facts.get("unified_allocator"):
            self._install_unified_allocator(alloc)
        elif self.separate_swa:
            self._install_paged_allocator(alloc.full_attn_allocator, pools=self._full_cache_pools())
            self._install_paged_allocator(alloc.swa_attn_allocator, pools=self._swa_cache_pools())
        elif self.paged:
            self._install_paged_allocator(alloc)
        orig_alloc = alloc.alloc if not (self.paged or self.separate_swa) else None

        def alloc_(need_size):
            me._reap_pd_reads()
            out = orig_alloc(need_size)
            if out is None:
                me.stats["alloc_refused"] += 1
                return out
            slots = allocated_slots(out)
            for pool in me._cache_pools():
                me.ledger.allocate(pool, slots, site=f"alloc@{me.seq}")
            me.stats["alloc_calls"] += 1
            me.stats["alloc_slots"] += len(slots)
            return out

        if orig_alloc is not None:
            alloc.alloc = alloc_

        orig_free = alloc.free if not (self.paged or self.separate_swa) else None

        def free_(free_index):
            me._reap_pd_reads()
            freed = mediated_frees(free_index, alloc.free_group)
            if freed and me.early_complete:
                me._complete_referencing(freed)
            orig_free(free_index)
            if freed:
                for pool in me._cache_pools():
                    me.ledger.free(pool, freed, site=f"free@{me.seq}")
                me.stats["free_calls"] += 1
                me.stats["free_slots"] += len(freed)
            elif alloc.free_group is not None:
                me.stats["free_deferred_to_group"] += 1

        if orig_free is not None:
            alloc.free = free_

        def allocator_cleared(released: list[int]) -> None:
            if me.early_complete:
                me._complete_referencing(released)
            for pool in me._cache_pools():
                me.ledger.free(pool, released, site=f"clear@{me.seq}")
            me.stats["clear_calls"] += 1
            me.stats["clear_pages" if me.paged else "clear_slots"] += len(released)

        if self.facts.get("unified_allocator"):
            pass  # _install_unified_allocator mediates the owner band's reset.
        elif not self.separate_swa:
            wrap_allocator_clear(
                alloc,
                lambda: free_slots_from_allocator(alloc),
                allocator_cleared,
            )
        else:
            self._install_separate_swa_clear()
        if self.hicache:
            self._install_hicache()
        if self.external_linker:
            self._install_external_linker()
        if getattr(sched, "decode_offload_manager", None) is not None:
            self._install_decode_offload()
        if self.disagg_prefill:
            self._install_disagg_prefill()
        if self.disagg_decode:
            self._install_disagg_decode()

        orig_run_batch = sched.run_batch

        def run_batch(batch, *args, **kwargs):
            me._before_run_batch(batch)
            result = orig_run_batch(batch, *args, **kwargs)
            me.results[id(result)] = (me.seq, result)
            return result

        sched.run_batch = run_batch

        orig_barrier = getattr(sched, "_apply_war_barrier", None)
        if orig_barrier is not None:

            def apply_war_barrier(*args, **kwargs):
                out = orig_barrier(*args, **kwargs)
                me._fence_seq = me.seq
                me.stats["war_barriers"] += 1
                return out

            sched._apply_war_barrier = apply_war_barrier
        else:
            me.stats["war_barrier_missing"] += 1

        orig_process = sched.process_batch_result

        def process_batch_result(batch, result, *args, **kwargs):
            me._before_process_result(result)
            try:
                return orig_process(batch, result, *args, **kwargs)
            finally:
                me.dirty = True
                me.events += 1
                me._maybe_flush()

        sched.process_batch_result = process_batch_result

        orig_idle = sched.on_idle

        def on_idle(*args, **kwargs):
            # One flush per idle period, whatever the periodic flushes did in between.
            me._reap_hicache()
            me._reap_pd_reads()
            me._check_pd_retraction_prealloc_deadlock()
            if me.events != me._idle_flushed_at:
                me.idle_flush_count += 1
                me._idle_flushed_at = me.events
                me.flush("idle")
            return orig_idle(*args, **kwargs)

        sched.on_idle = on_idle

        orig_forward = runner.forward

        def forward(forward_batch, *args, **kwargs):
            out = orig_forward(forward_batch, *args, **kwargs)
            me._after_forward(forward_batch, out)
            return out

        runner.forward = forward

        if self.speculative_ngram:
            self._install_ngram_copy()
        if self.speculative_dflash:
            self._install_dflash_draft()
        elif self.speculative_eagle3:
            self._install_eagle3_draft()
        elif self.speculative_multi_layer_eagle:
            self._install_multi_layer_eagle_drafts()

        pcg = getattr(runner, "prefill_cuda_graph_runner", None)
        self.facts["prefill_graph_runner"] = type(pcg).__name__ if pcg is not None else None
        if pcg is not None and hasattr(pcg, "_prepare_forward_metadata_for_replay"):
            # load_batch calls this at its end with the static batch whose out_cache_loc the
            # replayed prefill consumes, and it calls init_forward_metadata inside: wrapped
            # before the original, so the pre-kernel point sees the static batch too.
            orig_prep = pcg._prepare_forward_metadata_for_replay

            def prepare_replay(forward_batch, static_forward_batch, *args, **kwargs):
                me.last_static_fb = static_forward_batch
                return orig_prep(forward_batch, static_forward_batch, *args, **kwargs)

            pcg._prepare_forward_metadata_for_replay = prepare_replay
            self.facts["prefill_static_batch_source"] = "_prepare_forward_metadata_for_replay"
        elif pcg is not None and hasattr(pcg, "load_batch"):
            wrap_static_batch_loader(pcg, lambda static: setattr(me, "last_static_fb", static))
            self.facts["prefill_static_batch_source"] = "load_batch"
        dcg = getattr(runner, "decode_cuda_graph_runner", None)
        self.facts["decode_graph_runner"] = type(dcg).__name__ if dcg is not None else None
        # The pre-kernel points (GPU validation; inert otherwise): the eager steps and the
        # breakable prefill graph after init_forward_metadata, the full decode graphs before
        # their replay. Installed before the synthetic faults so a fault's wrapper runs first.
        backends = [self.model_runner.attn_backend]
        if self.pdmux:
            backends.extend(getattr(self.model_runner, "decode_attn_backend_group", ()) or ())
        unique_backends = list({id(backend): backend for backend in backends}.values())
        for backend in unique_backends:
            orig_init_md = backend.init_forward_metadata

            def init_forward_metadata(forward_batch, *args, _orig=orig_init_md, **kwargs):
                out = _orig(forward_batch, *args, **kwargs)
                me._before_kernels(forward_batch, "init_forward_metadata")
                return out

            backend.init_forward_metadata = init_forward_metadata
        self.facts["pre_kernel_backends_wrapped"] = len(unique_backends)
        graph_backend = getattr(dcg, "backend", None)
        if graph_backend is not None and hasattr(graph_backend, "replay"):
            orig_replay = graph_backend.replay

            def replay(key, forward_batch, *args, **kwargs):
                me._before_kernels(forward_batch, "decode_graph_replay")
                return orig_replay(key, forward_batch, *args, **kwargs)

            graph_backend.replay = replay
            self.facts["decode_graph_replay_wrapped"] = True
        if self.fault is not None:
            self.fault.install(self)

    def _install_disagg_prefill(self) -> None:
        """Track the concrete KV pages read by the P/D prefill sender."""
        me, sched = self, self.scheduler
        # The chunk sender was renamed across versions: 0.5.20+ calls ``_send_kv_chunk`` (a thin
        # wrapper around ``send_kv_chunk``); 0.5.19 has only ``send_kv_chunk``.  Wrap whichever
        # the scheduler exposes.
        send_chunk_name = "_send_kv_chunk" if hasattr(sched, "_send_kv_chunk") else "send_kv_chunk"
        orig_send_chunk = getattr(sched, send_chunk_name)

        def send_chunk(req, *args, **kwargs):
            me._wrap_disagg_sender(req.disagg_kv_sender, req)
            return orig_send_chunk(req, *args, **kwargs)

        setattr(sched, send_chunk_name, send_chunk)

        # The optimistic bootstrap retry path exists in 0.5.20+ only; wrap it when present.
        orig_retry = getattr(sched, "optimistic_release_and_requeue", None)
        if orig_retry is not None:

            def optimistic_retry(req, *args, **kwargs):
                out = orig_retry(req, *args, **kwargs)
                me.request_incarnations.renew(req)
                me.stats["pd_optimistic_retries"] += 1
                return out

            sched.optimistic_release_and_requeue = optimistic_retry
        self.stats["pd_prefill_wrapped"] += 1

    def _check_pd_retraction_prealloc_deadlock(self) -> None:
        """Report the decode-side circular wait between retraction and preallocation.

        ``process_decode_queue`` refuses to advance capacity-holding transfer entries while any
        retracted request remains, while a retracted request cannot resume because those entries
        hold the remaining capacity; once the other queues are empty, neither side can release
        memory. This is an adapter-level availability invariant, not a timed P1--P4 policy.
        """
        if not self.disagg_decode or self._pd_retraction_deadlock_reported:
            return
        sched = self.scheduler
        prealloc = getattr(sched, "disagg_decode_prealloc_queue", None)
        transfer = getattr(sched, "disagg_decode_transfer_queue", None)
        if prealloc is None or transfer is None:
            return
        retracted = list(getattr(prealloc, "retracted_queue", ()) or ())
        reserved = list(getattr(prealloc, "queue", ()) or ())
        transferred = list(getattr(transfer, "queue", ()) or ())
        if not retracted or not transferred:
            return

        def batch_empty(batch: Any) -> bool:
            return batch is None or batch.is_empty()

        active = (
            not batch_empty(getattr(sched, "running_batch", None))
            or not batch_empty(getattr(sched, "last_batch", None))
            or bool(getattr(sched, "result_queue", None))
            or bool(getattr(sched, "waiting_queue", None))
            or bool(getattr(transfer, "_deferred_releases", None))
        )
        if active:
            return

        retracted_ids = [str(getattr(req, "rid", "<unknown>")) for req in retracted]
        reserved_ids = [
            str(getattr(getattr(entry, "req", None), "rid", "<unknown>"))
            for entry in (*reserved, *transferred)
        ]
        self._pd_retraction_deadlock_reported = True
        self.stats["pd_retraction_prealloc_deadlocks"] += 1
        self.reporter.emit(
            Report(
                OutcomeKind.VIOLATION,
                "pd_retraction_prealloc_deadlock",
                "a retracted decode request could not resume while capacity-holding transfer "
                "entries were forbidden from advancing; no running, result or post-transfer "
                "request remained to release capacity; preallocation entries were also gated "
                f"(retracted={retracted_ids}, reserved={reserved_ids})",
                request=retracted_ids[0],
                site=f"pd-decode-admission@{self.seq}",
            )
        )
        self.dirty = True

    def _report_pd_preallocation_capacity_mismatch(self, detail: str) -> None:
        """Persist a decode preallocation failure before the scheduler process exits."""
        if self._pd_preallocation_capacity_reported:
            return
        self._pd_preallocation_capacity_reported = True
        self.stats["pd_preallocation_capacity_mismatches"] += 1
        self.reporter.emit(
            Report(
                OutcomeKind.VIOLATION,
                "pd_preallocation_capacity_mismatch",
                "decode admission consumed a reservation that no longer fit the KV pool "
                f"({detail})",
                site=f"pd-decode-admission@{self.seq}",
            )
        )
        self.dirty = True
        # The unmodified scheduler raises immediately after this point. Write while its process
        # is still alive instead of relying on the framework's crash-cleanup path.
        self.flush("pd-preallocation-capacity-mismatch")

    def _install_disagg_decode(self) -> None:
        """Track P/D writes into the decode worker's concrete destination pages."""
        me = self
        prealloc = self.scheduler.disagg_decode_prealloc_queue
        transfer = self.scheduler.disagg_decode_transfer_queue
        orig_process_queue = self.scheduler.process_decode_queue

        def process_decode_queue(*args, **kwargs):
            me._check_pd_retraction_prealloc_deadlock()
            try:
                return orig_process_queue(*args, **kwargs)
            except AssertionError as exc:
                detail = str(exc)
                if "KV cache is full! Bug in memory estimation." in detail:
                    me._report_pd_preallocation_capacity_mismatch(detail)
                raise

        self.scheduler.process_decode_queue = process_decode_queue
        self.stats["pd_decode_queue_wrapped"] += 1
        # The receiver is wrapped where the prealloc queue publishes its destination metadata, so
        # a transfer write opens and fills only against that side's shadow. Newer versions route
        # it through ``_send_kv_metadata``; older ones call ``receiver.send_metadata`` directly, so
        # the receiver is wrapped at creation instead. Either way the wrap precedes the first
        # ``send_metadata``.
        if hasattr(prealloc, "_send_kv_metadata"):
            orig_send_metadata = prealloc._send_kv_metadata

            def send_metadata(decode_req, kv_indices, page_size, state_indices=None, **kwargs):
                me._wrap_disagg_receiver(decode_req.kv_receiver, decode_req.req)
                return orig_send_metadata(
                    decode_req,
                    kv_indices,
                    page_size,
                    state_indices,
                    **kwargs,
                )

            prealloc._send_kv_metadata = send_metadata
        elif hasattr(prealloc, "_create_receiver_and_enqueue"):
            orig_create = prealloc._create_receiver_and_enqueue

            def create_receiver_and_enqueue(*args, **kwargs):
                decode_req = orig_create(*args, **kwargs)
                receiver = getattr(decode_req, "kv_receiver", None)
                if receiver is not None:
                    me._wrap_disagg_receiver(receiver, decode_req.req)
                return decode_req

            prealloc._create_receiver_and_enqueue = create_receiver_and_enqueue
        else:
            self.reporter.emit(
                integration_error(
                    "pd_decode_metadata_hook",
                    "the decode prealloc queue exposes neither _send_kv_metadata nor "
                    "_create_receiver_and_enqueue; cannot wrap the receiver",
                    pool=POOL,
                    site="attach",
                )
            )

        # 0.5.19's decode transfer queue frees through ``_do_release`` only; the dedicated
        # ``_release_request`` entry point appears in 0.5.20+.  Wrap it when present.
        if hasattr(transfer, "_release_request"):
            orig_release_request = transfer._release_request

            def release_request(decode_req):
                receiver = decode_req.kv_receiver
                key = id(receiver)
                try:
                    return orig_release_request(decode_req)
                finally:
                    if key in me._pd_decode_writes:
                        # Keep the write epoch open through the allocator release.  If no drain
                        # proof retired it first, that release is precisely the violation.
                        me._retire_disagg_receiver(
                            receiver, success=False, reason="failure_released"
                        )

            transfer._release_request = release_request

        orig_do_release = transfer._do_release

        def do_release(decode_req, idx):
            receiver = decode_req.kv_receiver
            key = id(receiver)
            required_acks = (
                len(getattr(receiver, "bootstrap_infos", None) or ())
                if getattr(receiver, "abort_notified", False)
                else 0
            )
            safe = bool(
                receiver.kv_mgr.is_abort_release_safe(decode_req.req.bootstrap_room, required_acks)
            )
            if safe:
                me._retire_disagg_receiver(receiver, success=False, reason="failure_drained")
            try:
                return orig_do_release(decode_req, idx)
            finally:
                # On the framework's timeout path the free above intentionally occurs while the
                # epoch is open, so the ledger reports it.  Close afterwards to avoid cascading
                # reports on later allocations.
                if key in me._pd_decode_writes:
                    me._retire_disagg_receiver(
                        receiver, success=False, reason="failure_released_unsafe"
                    )

        transfer._do_release = do_release
        self._install_disagg_retraction_backup()
        self.stats["pd_decode_wrapped"] += 1

    def _retraction_groups(self, req: Any) -> tuple[tuple[tuple[int, ...], range], ...]:
        count = max(0, int(req.seqlen) - 1)
        row = self.req_pool.req_to_token[req.kv.req_pool_idx, :count]
        return external_slot_groups(_ints(row), self.page_size)

    def _install_disagg_retraction_backup(self) -> None:
        """Model the host-pool and CPU-tensor retraction copies of newer revisions.

        The host-pool implementation starts H2D asynchronously and either publishes a
        finish event for later admission or orders the consuming stream behind that event.
        """
        tree = self.tree
        if not all(hasattr(tree, name) for name in ("backup_kv_cache", "restore_kv_cache")):
            return
        me = self
        orig_backup = tree.backup_kv_cache

        def backup(req):
            try:
                groups = me._retraction_groups(req)
            except (TypeError, ValueError) as exc:
                me.reporter.emit(
                    integration_error(
                        "pd_decode_retraction_source_shape",
                        str(exc),
                        pool=POOL,
                        site=f"pd-retraction-backup@{me.seq}",
                    )
                )
                groups = ()
            owner = me.request_incarnations(req)
            result = orig_backup(req)
            if result is None:
                return None
            # The host-pool implementation records its D2H start event on the scheduler stream,
            # behind `_apply_war_barrier`, then synchronizes its finish event before returning.
            # The shared reads fenced through this sequence are physically complete here.
            me._complete_through(me._fence_seq, "pd_retraction_backup_sync")
            for blocks, offsets in groups:
                me._pd_seq += 1
                did = f"pd-retraction-backup:{me._pd_seq}"
                me.ledger.external_read(
                    ExternalRead(
                        did,
                        POOL,
                        blocks,
                        offsets,
                        source="sglang_pd_retraction_backup",
                        request=owner,
                    ),
                    site=f"{did}@{me.seq}",
                )
                if did in me.ledger.epochs:
                    me.ledger.complete(did)
            me.stats["pd_retraction_backups"] += 1
            me.stats["pd_retraction_backup_pages"] += sum(
                len(blocks) for blocks, _offsets in groups
            )
            return result

        tree.backup_kv_cache = backup

        def open_restore(req):
            try:
                groups = me._retraction_groups(req)
            except (TypeError, ValueError) as exc:
                me.reporter.emit(
                    integration_error(
                        "pd_decode_retraction_destination_shape",
                        str(exc),
                        pool=POOL,
                        site=f"pd-retraction-restore@{me.seq}",
                    )
                )
                groups = ()
            owner = me.request_incarnations(req)
            engine = getattr(getattr(tree, "cache_controller", None), "l2_transfer_engine", None)
            if getattr(engine, "_h2d_on_current_stream", False):
                stream = me._stream()
            else:
                transfer_stream = getattr(engine, "host_to_device_stream", None)
                stream = (
                    int(transfer_stream.cuda_stream)
                    if getattr(me, "stream_epochs", False) and transfer_stream is not None
                    else None
                )
            after = me._fenced_epochs() if getattr(me, "stream_epochs", False) else ()
            opened = []
            for blocks, offsets in groups:
                me._pd_seq += 1
                did = f"pd-retraction-restore:{me._pd_seq}"
                me.ledger.external_write(
                    ExternalWrite(
                        did,
                        POOL,
                        blocks,
                        offsets,
                        source="sglang_pd_retraction_restore",
                        request=owner,
                        stream=stream,
                        after=after,
                    ),
                    site=f"{did}@{me.seq}",
                )
                opened.append((did, blocks, offsets))
            return owner, opened

        def cancel_restore(opened):
            for did, _blocks, _offsets in opened:
                if did in me.ledger.epochs:
                    me.ledger.complete(did)

        def complete_restore(opened, owner, *, ordered: bool):
            for did, blocks, offsets in opened:
                for block in blocks:
                    shadow = me.ledger.blocks[POOL][block]
                    request = owner if shadow.owner == owner else None
                    me.ledger.external_fill(
                        ExternalFill(
                            POOL,
                            block,
                            offsets,
                            source="sglang_pd_retraction_restore",
                            request=request,
                        ),
                        site=f"{did}-complete@{me.seq}",
                    )
                if did in me.ledger.epochs:
                    me.ledger.complete(did)
            me.stats["pd_retraction_restores"] += 1
            me.stats["pd_retraction_restore_pages"] += sum(
                len(blocks) for _did, blocks, _offsets in opened
            )
            if ordered:
                me.stats["pd_retraction_restores_stream_ordered"] += 1

        has_async_restore = all(
            hasattr(tree, name)
            for name in (
                "start_restore_kv_cache",
                "finish_restore_kv_cache",
                "_finish_restore",
            )
        )
        if has_async_restore:
            orig_start_restore = tree.start_restore_kv_cache

            def start_restore(req, retraction_backup):
                owner, opened = open_restore(req)
                try:
                    result = orig_start_restore(req, retraction_backup)
                except Exception:
                    cancel_restore(opened)
                    raise
                event = req.kv.restore_done_event
                if event is None:
                    cancel_restore(opened)
                    me._reject("asynchronous host restore published no completion event")
                key = id(event)
                if key in me._pd_retraction_restores:
                    cancel_restore(opened)
                    me._reject("asynchronous host restore reused a live completion event")
                me._pd_retraction_restores[key] = tuple(
                    (did, blocks, offsets, owner) for did, blocks, offsets in opened
                )
                me.stats["pd_retraction_restore_starts"] += 1
                me.stats["pd_retraction_restore_start_pages"] += sum(
                    len(blocks) for _did, blocks, _offsets in opened
                )
                return result

            tree.start_restore_kv_cache = start_restore
            orig_finish_restore = tree._finish_restore

            def finish_restore(kv):
                event = kv.restore_done_event
                result = orig_finish_restore(kv)
                if event is not None:
                    pending = me._pd_retraction_restores.pop(id(event), ())
                    if pending:
                        opened = tuple(
                            (did, blocks, offsets) for did, blocks, offsets, _ in pending
                        )
                        complete_restore(opened, pending[0][3], ordered=False)
                return result

            tree._finish_restore = finish_restore
            orig_restore = tree.restore_kv_cache

            def restore(req, retraction_backup):
                result = orig_restore(req, retraction_backup)
                event = req.kv.restore_done_event
                if event is not None:
                    pending = me._pd_retraction_restores.pop(id(event), ())
                    if pending:
                        # Current main queues a wait for this event on the stream that
                        # consumes the row. It is therefore ordered without a host wait.
                        opened = tuple(
                            (did, blocks, offsets) for did, blocks, offsets, _ in pending
                        )
                        complete_restore(opened, pending[0][3], ordered=True)
                return result

            tree.restore_kv_cache = restore
            self.stats["pd_retraction_async_restore_wrapped"] += 1
        else:
            orig_restore = tree.restore_kv_cache

            def restore(req, retraction_backup):
                owner, opened = open_restore(req)
                try:
                    result = orig_restore(req, retraction_backup)
                except Exception:
                    cancel_restore(opened)
                    raise
                complete_restore(opened, owner, ordered=False)
                return result

            tree.restore_kv_cache = restore
        self.stats["pd_retraction_backup_wrapped"] += 1

        allocator = getattr(self, "allocator", None)
        if allocator is None or not all(
            hasattr(allocator, name) for name in ("get_cpu_copy", "load_cpu_copy")
        ):
            return

        def cpu_groups(indices):
            try:
                return external_slot_groups(_ints(indices), me.page_size)
            except (TypeError, ValueError) as exc:
                me.reporter.emit(
                    integration_error(
                        "pd_decode_retraction_cpu_shape",
                        str(exc),
                        pool=POOL,
                        site=f"pd-retraction-cpu@{me.seq}",
                    )
                )
                return ()

        def cpu_owner(req_pool_index):
            if req_pool_index is None:
                return None
            return me.rid_of_row.get(int(req_pool_index))

        def complete_device_work(reason: str) -> None:
            # Both concrete CPU-copy methods globally synchronize before touching KV. Mirror that
            # boundary before opening the external access, so a finished GPU dispatch cannot be
            # mistaken for a concurrent CPU transfer.
            if me.open:
                me._complete_through(max(me.open), reason)

        orig_get_cpu_copy = allocator.get_cpu_copy

        def get_cpu_copy(indices, mamba_indices=None, req_pool_index=None):
            complete_device_work("pd_retraction_cpu_backup_sync")
            groups = cpu_groups(indices)
            owner = cpu_owner(req_pool_index)
            opened = []
            for blocks, offsets in groups:
                me._pd_seq += 1
                did = f"pd-retraction-cpu-backup:{me._pd_seq}"
                me.ledger.external_read(
                    ExternalRead(
                        did,
                        POOL,
                        blocks,
                        offsets,
                        source="sglang_pd_retraction_cpu_backup",
                        request=owner,
                    ),
                    site=f"{did}@{me.seq}",
                )
                opened.append(did)
            try:
                result = orig_get_cpu_copy(
                    indices,
                    mamba_indices=mamba_indices,
                    req_pool_index=req_pool_index,
                )
            finally:
                for did in opened:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
            me.stats["pd_retraction_cpu_backups"] += 1
            me.stats["pd_retraction_cpu_backup_pages"] += sum(
                len(blocks) for blocks, _offsets in groups
            )
            return result

        allocator.get_cpu_copy = get_cpu_copy
        orig_load_cpu_copy = allocator.load_cpu_copy

        def load_cpu_copy(
            kv_cache_cpu,
            indices,
            mamba_indices=None,
            req_pool_index=None,
        ):
            complete_device_work("pd_retraction_cpu_restore_sync")
            groups = cpu_groups(indices)
            owner = cpu_owner(req_pool_index)
            opened = []
            for blocks, offsets in groups:
                me._pd_seq += 1
                did = f"pd-retraction-cpu-restore:{me._pd_seq}"
                me.ledger.external_write(
                    ExternalWrite(
                        did,
                        POOL,
                        blocks,
                        offsets,
                        source="sglang_pd_retraction_cpu_restore",
                        request=owner,
                    ),
                    site=f"{did}@{me.seq}",
                )
                opened.append((did, blocks, offsets))
            try:
                result = orig_load_cpu_copy(
                    kv_cache_cpu,
                    indices,
                    mamba_indices=mamba_indices,
                    req_pool_index=req_pool_index,
                )
            except Exception:
                for did, _blocks, _offsets in opened:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
                raise
            for did, blocks, offsets in opened:
                for block in blocks:
                    shadow = me.ledger.blocks[POOL][block]
                    request = owner if shadow.owner == owner else None
                    me.ledger.external_fill(
                        ExternalFill(
                            POOL,
                            block,
                            offsets,
                            source="sglang_pd_retraction_cpu_restore",
                            request=request,
                        ),
                        site=f"{did}-complete@{me.seq}",
                    )
                if did in me.ledger.epochs:
                    me.ledger.complete(did)
            me.stats["pd_retraction_cpu_restores"] += 1
            me.stats["pd_retraction_cpu_restore_pages"] += sum(
                len(blocks) for _did, blocks, _offsets in opened
            )
            return result

        allocator.load_cpu_copy = load_cpu_copy
        self.stats["pd_retraction_cpu_wrapped"] += 1

    def _wrap_disagg_receiver(self, receiver: Any, req: Any) -> None:
        """Open writes at metadata publication and fill them only at receiver success."""
        if getattr(receiver, "_kvguard_destination_writes_wrapped", False):
            return
        receiver._kvguard_destination_writes_wrapped = True
        me = self
        key = id(receiver)
        me._pd_decode_writes.setdefault(key, [])
        me._pd_decode_receivers[key] = receiver
        orig_send = receiver.send_metadata

        def send(
            page_indices,
            aux_index=None,
            state_indices=None,
            decode_prefix_len=None,
            destination=None,
            device_kv_indices=None,
            *args,
            **kwargs,
        ):
            gpu_open_steps: list[int] = []
            gpu_done_steps: list[int] = []
            for seq, epoch in getattr(me, "open", {}).items():
                query = getattr(epoch.event, "query", None)
                if query is None:
                    continue
                if bool(query()):
                    gpu_done_steps.append(seq)
                else:
                    gpu_open_steps.append(seq)
            if gpu_done_steps:
                # Destination publication is the earliest instant at which the remote peer may
                # write these addresses.  Retire only GPU work that a nonblocking event query
                # proves completed before publication; otherwise ExternalWrite below must judge
                # reuse against the still-running read.
                me._complete_through(max(gpu_done_steps), "pd_receive_publication")
            state_slots: tuple[int, ...] = ()
            if state_indices is not None and any(item is not None for item in state_indices):
                physical = _pd_payload_ints(state_indices)
                if me.state is None:
                    me.reporter.emit(
                        integration_error(
                            "pd_decode_state_payload",
                            "the P/D decode role published state without a mediated state pool",
                            pool=POOL,
                            site=f"pd-receive@{me.seq}",
                        )
                    )
                else:
                    state_slots = tuple(
                        dict.fromkeys(
                            slot for slot in me.state._logical_slots(physical) if slot > 0
                        )
                    )
            destination_name = getattr(destination, "name", None)
            if destination is not None and destination_name not in (None, "DEVICE"):
                me.reporter.emit(
                    integration_error(
                        "pd_decode_host_destination",
                        f"the bounded P/D decode port received destination {destination_name}",
                        pool=POOL,
                        site=f"pd-receive@{me.seq}",
                    )
                )
            if device_kv_indices is not None:
                me.reporter.emit(
                    integration_error(
                        "pd_decode_device_sidecar",
                        "the bounded P/D decode port received an unmodelled device sidecar",
                        pool=POOL,
                        site=f"pd-receive@{me.seq}",
                    )
                )
            prefix = int(decode_prefix_len or 0)
            total = len(req.origin_input_ids) + (
                len(req.output_ids) if getattr(req, "pd_rebootstrap_in_progress", False) else 0
            )
            num_tokens = max(0, total - prefix)
            physical_pages: list[int] = []
            try:
                physical_pages = _ints(page_indices)
                logical_pages = me._logical_kv_pages(physical_pages)
                if any(page < 0 for page in logical_pages):
                    raise ValueError(
                        "P/D decode destination contains an unmapped physical page: "
                        f"physical={physical_pages}, logical={logical_pages}"
                    )
                groups = external_read_groups(logical_pages, num_tokens, me.page_size)
            except (TypeError, ValueError) as exc:
                me.reporter.emit(
                    integration_error(
                        "pd_decode_destination_shape",
                        str(exc),
                        pool=POOL,
                        site=f"pd-receive@{me.seq}",
                    )
                )
                groups = ()
            owner = me.request_incarnations(req)
            has_draft_transfer = getattr(me, "speculative_eagle3", False)
            transfer_pools = (POOL, DRAFT_POOL) if has_draft_transfer else (POOL,)
            for pool in transfer_pools:
                for blocks, offsets in groups:
                    me._pd_seq += 1
                    did = (
                        f"pd-receive:{me._pd_seq}"
                        if pool == POOL
                        else f"pd-receive:{me._pd_seq}:{pool}"
                    )
                    me.ledger.external_write(
                        ExternalWrite(
                            did,
                            pool,
                            blocks,
                            offsets,
                            source="sglang_pd_decode_receive",
                            request=owner,
                        ),
                        site=f"{did}@{me.seq}",
                    )
                    me._pd_decode_writes[key].append((did, pool, blocks, offsets, owner))
            if has_draft_transfer:
                me.stats["pd_receive_draft_submissions"] += 1
                me.stats["pd_receive_draft_pages"] += sum(len(blocks) for blocks, _ in groups)
            if state_slots:
                me._pd_seq += 1
                did = f"pd-receive:{me._pd_seq}:state"
                me.ledger.external_write(
                    ExternalWrite(
                        did,
                        STATE_POOL,
                        state_slots,
                        range(1),
                        source="sglang_pd_decode_receive",
                        request=owner,
                    ),
                    site=f"{did}@{me.seq}",
                )
                me._pd_decode_writes[key].append((did, STATE_POOL, state_slots, range(1), owner))
                me.stats["pd_receive_state_submissions"] += 1
                me.stats["pd_receive_state_slots"] += len(state_slots)
            # The destination layout for the bounded fill at completion (the extent channel): the
            # ordered logical pages, the published token count, the state slots and the owner,
            # keyed by this receiver, plus the request's bootstrap room.
            me._pd_decode_rooms[key] = getattr(req, "bootstrap_room", None)
            me._pd_decode_meta[key] = {
                "pools": tuple(transfer_pools),
                "logical_pages": tuple(logical_pages),
                "num_tokens": int(num_tokens),
                "state_slots": tuple(state_slots),
                "owner": owner,
            }
            fault = getattr(me, "fault", None)
            if (
                fault is not None
                and fault.kind == "pd_decode_receive_early_free"
                and fault._armed(me)
                and groups
            ):
                page = groups[0][0][0]
                me.allocator.free(me.allocator.free_pages.new_tensor([page * me.page_size]))
                fault.done = True
                fault.note = (
                    f"step {me.seq}: decode destination page {page} freed after its "
                    "P/D receive epoch opened and before the remote write completed"
                )
            if len(me._pd_decode_trace) < 80:
                me._pd_decode_trace.append(
                    {
                        "seq": me.seq,
                        "request": str(req.rid),
                        "owner": owner,
                        "blocks": tuple(block for blocks, _offsets in groups for block in blocks),
                        "physical_blocks": tuple(physical_pages),
                        "tokens": num_tokens,
                        "prefix": prefix,
                        "state_slots": state_slots,
                        "gpu_open_steps": tuple(gpu_open_steps),
                        "gpu_done_steps": tuple(gpu_done_steps),
                    }
                )
            me.stats["pd_receive_submissions"] += 1
            me.stats["pd_receive_pages"] += sum(len(blocks) for blocks, _ in groups)
            me.stats["pd_receive_tokens"] += num_tokens
            call_kwargs = dict(kwargs)
            if decode_prefix_len is not None:
                call_kwargs["decode_prefix_len"] = decode_prefix_len
            if destination is not None:
                call_kwargs["destination"] = destination
            if device_kv_indices is not None:
                call_kwargs["device_kv_indices"] = device_kv_indices
            return orig_send(
                page_indices,
                aux_index,
                state_indices,
                *args,
                **call_kwargs,
            )

        receiver.send_metadata = send
        orig_poll = receiver.poll

        def poll(*args, **kwargs):
            status = orig_poll(*args, **kwargs)
            if int(status) == 4:
                me._retire_disagg_receiver(receiver, success=True, reason="success")
            return status

        receiver.poll = poll
        self.stats["pd_receivers_wrapped"] += 1

    def _write_pd_extent(self, room: Any, tokens: int) -> None:
        """Publish the cumulative tokens the prefill sender has actually transferred for a room."""
        directory = self._pd_extent_dir
        if directory is None:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            final = os.path.join(directory, str(room))
            tmp = os.path.join(directory, f".{room}.{os.getpid()}.tmp")
            with open(tmp, "w") as handle:
                handle.write(str(int(tokens)))
            os.replace(tmp, final)
            self.stats["pd_extent_published"] += 1
        except OSError:
            self.stats["pd_extent_write_failed"] += 1

    def _read_pd_extent(self, room: Any) -> int | None:
        """The transferred-token extent the prefill published for a room, or None when the channel
        is off or nothing was published."""
        directory = self._pd_extent_dir
        if directory is None or room is None:
            return None
        try:
            with open(os.path.join(directory, str(room))) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            self.stats["pd_extent_missing"] += 1
            return None

    def _fill_disagg_destination(self, meta: dict[str, Any] | None, room: Any) -> None:
        """Mark the decode destination initialized up to the transferred extent only.

        Without the extent channel, or when the sender published nothing, this fills the full
        published destination. With a published extent shorter than the reservation, the pages
        beyond it stay uninitialized, so the decode forward's own read of them reports."""
        if not meta:
            return
        num_tokens = int(meta["num_tokens"])
        sent = self._read_pd_extent(room)
        fill_tokens = num_tokens if sent is None else max(0, min(int(sent), num_tokens))
        if sent is not None and fill_tokens < num_tokens:
            self.stats["pd_receive_extent_short"] += 1
            self.stats["pd_receive_extent_untransferred_tokens"] += num_tokens - fill_tokens
        logical_pages = meta["logical_pages"]
        ps = self.page_size
        fill_pages = logical_pages[: (fill_tokens + ps - 1) // ps]
        groups = external_read_groups(fill_pages, fill_tokens, ps) if fill_tokens else ()
        owner = meta["owner"]
        for pool in meta["pools"]:
            for blocks, offsets in groups:
                for block in blocks:
                    shadow = self.ledger.blocks[pool][block]
                    request = owner if shadow.owner == owner else None
                    self.ledger.external_fill(
                        ExternalFill(
                            pool,
                            block,
                            offsets,
                            source="sglang_pd_decode_receive",
                            request=request,
                        ),
                        site=f"pd-receive-complete@{self.seq}",
                    )
        state_slots = meta.get("state_slots") or ()
        # The recurrent state is one slot per request; a short attention transfer never carries it,
        # so fill it only when the whole reservation transferred.
        if state_slots and fill_tokens >= num_tokens:
            for slot in state_slots:
                shadow = self.ledger.blocks[STATE_POOL][slot]
                request = owner if shadow.owner == owner else None
                self.ledger.external_fill(
                    ExternalFill(
                        STATE_POOL,
                        slot,
                        range(1),
                        source="sglang_pd_decode_receive",
                        request=request,
                    ),
                    site=f"pd-receive-complete@{self.seq}",
                )
        self.stats["pd_receive_filled_tokens"] += fill_tokens

    def _retire_disagg_receiver(self, receiver: Any, *, success: bool, reason: str) -> None:
        key = id(receiver)
        writes = self._pd_decode_writes.pop(key, [])
        meta = self._pd_decode_meta.pop(key, None)
        room = self._pd_decode_rooms.pop(key, None)
        if success:
            self._fill_disagg_destination(meta, room)
        # Close every opened destination write epoch, filled or not, so lifetimes do not cascade.
        for did, _pool, _blocks, _offsets, _owner in writes:
            if did in self.ledger.epochs:
                self.ledger.complete(did)
        self._pd_decode_receivers.pop(key, None)
        if writes:
            self.stats["pd_receive_completions"] += 1
            self.stats["pd_receive_writes_completed"] += len(writes)
            self.stats[f"pd_receive_completion_{reason}"] += 1

    def _wrap_disagg_sender(self, sender: Any, req: Any) -> None:
        """Wrap one request sender once; all transfer chunks share its terminal status."""
        if type(sender).__name__ == "FakeKVSender":
            self.stats["pd_fake_senders_skipped"] += 1
            return
        if getattr(sender, "_kvguard_source_reads_wrapped", False):
            return
        sender._kvguard_source_reads_wrapped = True
        me = self
        key = id(sender)
        me._pd_reads.setdefault(key, [])
        me._pd_senders[key] = sender
        orig_send = sender.send

        def send(
            kv_indices,
            state_indices=None,
            num_kv_tokens=None,
            indices_event=None,
            *args,
            **kwargs,
        ):
            call_kwargs = dict(kwargs)
            if num_kv_tokens is not None:
                call_kwargs["num_kv_tokens"] = num_kv_tokens
            if indices_event is not None:
                call_kwargs["indices_event"] = indices_event
            if getattr(sender, "conclude_state", None) is not None:
                return orig_send(
                    kv_indices,
                    state_indices,
                    *args,
                    **call_kwargs,
                )
            if indices_event is not None:
                t0 = time.perf_counter()
                indices_event.synchronize()
                me.timings["pd_index_wait_s"] += time.perf_counter() - t0
                me.stats["pd_index_waits"] += 1
            physical_pages: list[int] = []
            try:
                physical_pages = _ints(kv_indices)
                logical_pages = me._logical_kv_pages(physical_pages)
                if any(page < 0 for page in logical_pages):
                    raise ValueError(
                        "P/D prefill source contains an unmapped physical page: "
                        f"physical={physical_pages}, logical={logical_pages}"
                    )
                groups = external_read_groups(logical_pages, int(num_kv_tokens or 0), me.page_size)
            except (TypeError, ValueError) as exc:
                me.reporter.emit(
                    integration_error(
                        "pd_prefill_source_shape",
                        str(exc),
                        pool=POOL,
                        site=f"pd-send@{me.seq}",
                    )
                )
                groups = ()
            me._reap_hicache()
            if len(me._pd_trace) < 80:
                kv = req.kv
                row_slots = _ints(me.req_pool.req_to_token[int(kv.req_pool_idx)])[
                    : int(kv.kv_allocated_len)
                ]
                blocks = tuple(block for group, _offsets in groups for block in group)
                shadows = me.ledger.blocks[POOL]
                me._pd_trace.append(
                    {
                        "seq": me.seq,
                        "request": str(req.rid),
                        "blocks": blocks,
                        "physical_blocks": tuple(physical_pages),
                        "row_blocks": tuple(
                            dict.fromkeys(slot // me.page_size for slot in row_slots)
                        ),
                        "num_tokens": int(num_kv_tokens or 0),
                        "allocated": int(kv.kv_allocated_len),
                        "committed": int(kv.kv_committed_len),
                        "protected": int(kv.cache_protected_len),
                        "prefix": int(getattr(req, "prefix_len", 0) or kv.cache_protected_len),
                        "start_send": int(getattr(req, "start_send_idx", 0)),
                        "filled": {block: shadows[block].filled for block in blocks},
                    }
                )
            opened: list[str] = []
            target_opened: list[str] = []
            owner = me.request_incarnations(req)
            has_draft_transfer = getattr(me, "speculative_eagle3", False)
            transfer_pools = (POOL, DRAFT_POOL) if has_draft_transfer else (POOL,)
            for pool in transfer_pools:
                for blocks, offsets in groups:
                    me._pd_seq += 1
                    did = (
                        f"pd-send:{me._pd_seq}" if pool == POOL else f"pd-send:{me._pd_seq}:{pool}"
                    )
                    me.ledger.external_read(
                        ExternalRead(
                            did,
                            pool,
                            blocks,
                            offsets,
                            source="sglang_pd_prefill_send",
                            request=owner,
                        ),
                        site=f"{did}@{me.seq}",
                    )
                    opened.append(did)
                    if pool == POOL:
                        target_opened.append(did)
            if has_draft_transfer:
                me.stats["pd_send_draft_submissions"] += 1
                me.stats["pd_send_draft_pages"] += sum(len(blocks) for blocks, _ in groups)
            state_slots: tuple[int, ...] = ()
            if state_indices is not None and any(item is not None for item in state_indices):
                physical = _pd_payload_ints(state_indices)
                if me.state is None:
                    me.reporter.emit(
                        integration_error(
                            "pd_prefill_state_payload",
                            "the P/D prefill role published state without a mediated state pool",
                            pool=POOL,
                            site=f"pd-send@{me.seq}",
                        )
                    )
                else:
                    state_slots = tuple(
                        dict.fromkeys(
                            slot for slot in me.state._logical_slots(physical) if slot > 0
                        )
                    )
            if state_slots:
                me._pd_seq += 1
                did = f"pd-send:{me._pd_seq}:state"
                me.ledger.external_read(
                    ExternalRead(
                        did,
                        STATE_POOL,
                        state_slots,
                        range(1),
                        source="sglang_pd_prefill_send",
                        request=owner,
                    ),
                    site=f"{did}@{me.seq}",
                )
                opened.append(did)
                me.stats["pd_send_state_submissions"] += 1
                me.stats["pd_send_state_slots"] += len(state_slots)
            try:
                out = orig_send(
                    kv_indices,
                    state_indices,
                    *args,
                    **call_kwargs,
                )
            except BaseException:
                for did in opened:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
                raise
            # Parallel-sampling siblings can share one sender/room, so a chunk may arrive after a
            # sibling's terminal status already popped this key; re-init rather than KeyError.
            me._pd_reads.setdefault(key, []).extend(opened)
            for did, (blocks, _offsets) in zip(target_opened, groups, strict=True):
                me._pd_reserved_blocks[did] = blocks
            me.stats["pd_send_submissions"] += 1
            me.stats["pd_send_pages"] += sum(len(blocks) for blocks, _ in groups)
            me.stats["pd_send_tokens"] += int(num_kv_tokens or 0)
            # Publish the cumulative tokens actually sent for this request's bootstrap room, so the
            # decode attachment can fill only the transferred extent (the extent channel above).
            if me._pd_extent_dir is not None:
                room = getattr(req, "bootstrap_room", None)
                if room is not None:
                    me._pd_sent_tokens[room] = me._pd_sent_tokens.get(room, 0) + int(
                        num_kv_tokens or 0
                    )
                    me._write_pd_extent(room, me._pd_sent_tokens[room])
            return out

        sender.send = send

        def retire(reason: str) -> None:
            dids = me._pd_reads.pop(key, [])
            for did in dids:
                if did in me.ledger.epochs:
                    me.ledger.complete(did)
                me._pd_reserved_blocks.pop(did, None)
            if dids:
                me.stats["pd_send_completions"] += 1
                me.stats["pd_send_reads_completed"] += len(dids)
                me.stats[f"pd_send_completion_{reason}"] += 1
                # NIXL can reach transport quiescence after the last model result has already
                # caused an idle flush.  Make that lifecycle transition observable in the next
                # idle record even when no further request arrives.
                me.dirty = True
                me.events = getattr(me, "events", 0) + 1
            me._pd_senders.pop(key, None)
            me._pd_failed_pending.discard(key)

        sender._kvguard_retire_source_reads = retire

        def retire_if_terminal(status: Any) -> None:
            if int(status) == 4:
                retire("success")
            elif int(status) == 0:
                # Failure is not a Mooncake source-read completion boundary.  The abort
                # notification marks the room Failed immediately, while a worker that already
                # passed its status check can still be reading these pages.  Reap only when the
                # worker's drain counter reaches zero (see _reap_pd_reads).
                me._reap_pd_reads()

        orig_poll = sender.poll

        def poll(*args, **kwargs):
            status = orig_poll(*args, **kwargs)
            retire_if_terminal(status)
            return status

        sender.poll = poll
        if hasattr(sender, "poll_pp_consensus"):
            orig_poll_pp = sender.poll_pp_consensus

            def poll_pp_consensus(*args, **kwargs):
                status = orig_poll_pp(*args, **kwargs)
                retire_if_terminal(status)
                return status

            sender.poll_pp_consensus = poll_pp_consensus
        self.stats["pd_senders_wrapped"] += 1

    def _reap_pd_reads(self) -> None:
        """Complete P/D source reads only after the transport proves them quiescent.

        A ``Failed`` poll alone is insufficient: the transfer worker increments
        ``_staging_outstanding[room]`` before its final status check and decrements it only after
        an in-progress write returns. Only once the sender concluded Failed and that counter is
        zero can no queued chunk still read the source. Reading the counter on the scheduler thread
        before allocator operations also keeps a real free-vs-transfer race visible to the ledger.
        """
        for key, sender in tuple(self._pd_senders.items()):
            if not self._pd_reads.get(key):
                self._pd_senders.pop(key, None)
                self._pd_failed_pending.discard(key)
                continue
            manager = getattr(sender, "kv_mgr", None)
            room = getattr(sender, "bootstrap_room", None)
            outstanding_map = getattr(manager, "_staging_outstanding", {})
            outstanding = int(outstanding_map.get(room, 0))
            state = getattr(sender, "conclude_state", None)
            statuses = getattr(manager, "request_status", None)
            if state is None and statuses is not None:
                if room in statuses:
                    state = manager.check_status(room)
                elif outstanding == 0:
                    # CommonKVSender.poll() maps an absent room to Failed. With no worker
                    # outstanding, any queued chunk can only observe that absence and skip the
                    # source; this is the same quiescent boundary as a drained failed sender.
                    sender._kvguard_retire_source_reads("status_absent_quiescent")
                    continue
            try:
                state_value = int(state)
            except (TypeError, ValueError):
                state_value = -1
            if state_value == 4:
                deferred_cleanup = bool(
                    getattr(manager, "defer_decode_allocation", False) and room in outstanding_map
                )
                if outstanding == 0 and not deferred_cleanup:
                    sender._kvguard_retire_source_reads("success_quiescent")
                continue
            if state_value != 0:
                continue
            if outstanding > 0:
                if key not in self._pd_failed_pending:
                    self._pd_failed_pending.add(key)
                    self.stats["pd_send_failures_inflight"] += 1
                continue
            reason = "failure_drained" if key in self._pd_failed_pending else "failure_quiescent"
            sender._kvguard_retire_source_reads(reason)

    def _trace_pd_source_release(self, pages: Sequence[int]) -> None:
        """Record transport state when an allocator release overlaps an open P/D source read."""
        released = set(map(int, pages))
        if not released or len(self._pd_source_release_trace) >= 20:
            return
        for key, dids in self._pd_reads.items():
            overlaps = {
                block
                for did in dids
                for block in self._pd_reserved_blocks.get(did, ())
                if block in released
            }
            if not overlaps:
                continue
            sender = self._pd_senders.get(key)
            manager = getattr(sender, "kv_mgr", None)
            room = getattr(sender, "bootstrap_room", None)
            outstanding = getattr(manager, "_staging_outstanding", {})
            active_rooms = getattr(manager, "_kvguard_active_source_rooms", ())
            statuses = getattr(manager, "request_status", {})
            self._pd_source_release_trace.append(
                {
                    "seq": self.seq,
                    "blocks": sorted(overlaps),
                    "room": room,
                    "sender_state": int(getattr(sender, "conclude_state", -1) or 0),
                    "manager_state": (
                        int(statuses.get(room, -1)) if hasattr(statuses, "get") else None
                    ),
                    "worker_outstanding": int(outstanding.get(room, 0)),
                    "worker_active": room in active_rooms,
                    "stack": traceback.format_list(traceback.extract_stack(limit=12)[:-1]),
                }
            )
            self.stats["pd_source_releases_observed"] += 1

    def _install_external_linker(self) -> None:
        """Track direct external-linker reads and writes at the transfer-ready boundary.

        The unified tree allocates load destinations before queuing them and pins offload sources
        until the linker's completion commits. The ``TransferReady`` object's ``synchronize``
        return is the first instant the remote store may touch device memory: source reads and
        destination-write epochs open there, layer-wise load completion fills the logical shadow,
        and completed offloads retire their read epochs. The bounded single-rank port accepts the
        FULL pool, the unified FULL+MAMBA layout and the legacy MLA+KDA hybrid; other routing,
        speculative sidecars and the multi-rank MLA broadcast ``wait`` path are rejected.
        """

        wrapper = self.tree.linker
        linker = wrapper.cache_linker
        me = self
        gate_dir_raw = os.environ.get("KVGUARD_SGLANG_EXTERNAL_LINKER_GATE_DIR")
        gate_kind = os.environ.get("KVGUARD_SGLANG_EXTERNAL_LINKER_GATE_KIND", "load")
        gate_request = os.environ.get("KVGUARD_SGLANG_EXTERNAL_LINKER_GATE_REQUEST")
        gate_timeout = float(os.environ.get("KVGUARD_SGLANG_EXTERNAL_LINKER_GATE_TIMEOUT", "30"))
        gate_lock = threading.Lock()
        gate_used = False

        def transfer_pools(transfers: Sequence[Any]) -> dict[str, tuple[int, ...]]:
            selected: dict[str, list[int]] = {}
            for transfer in transfers:
                indices = getattr(transfer, "device_indices", None)
                if indices is None:
                    continue
                name = str(getattr(getattr(transfer, "name", None), "value", transfer.name))
                if name == "kv":
                    pool = POOL
                    blocks = pages_of(_ints(indices), me.page_size)
                elif name == "mamba" and me.state is not None:
                    pool = STATE_POOL
                    blocks = _ints(indices)
                else:
                    me.reporter.emit(
                        integration_error(
                            "external_linker_pool",
                            f"bounded direct linker received pool {name!r}",
                            pool=POOL,
                            site=f"external-linker@{me.seq}",
                        )
                    )
                    continue
                selected.setdefault(pool, []).extend(blocks)
            return {
                pool: tuple(dict.fromkeys(blocks)) for pool, blocks in selected.items() if blocks
            }

        def new_job(kind: str, pools: dict[str, tuple[int, ...]]) -> dict[str, Any]:
            me._external_linker_seq += 1
            return {
                "id": me._external_linker_seq,
                "kind": kind,
                "pools": pools,
                "submitted_seq": me.seq,
                "begun": False,
                "dids": [],
                "after": (),
            }

        def begin_job(job: dict[str, Any]) -> None:
            with me._external_linker_lock:
                if job["begun"]:
                    return
                job["begun"] = True
                for pool, blocks in job["pools"].items():
                    did = f"external-linker-{job['kind']}:{job['id']}:{pool}"
                    offsets = range(1 if pool == STATE_POOL else me.page_size)
                    if job["kind"] == "store":
                        me.ledger.external_read(
                            ExternalRead(
                                did,
                                pool,
                                blocks,
                                offsets,
                                source="sglang_external_linker_store",
                                after=job["after"],
                            ),
                            site=f"{did}@{job['submitted_seq']}",
                        )
                        me.stats["external_linker_store_blocks"] += len(blocks)
                        if (
                            pool == POOL
                            and me.fault is not None
                            and me.fault.kind == "external_linker_store_early_free"
                            and me.fault._armed(me)
                        ):
                            page = blocks[0]
                            me.allocator.free(
                                me.allocator.free_pages.new_tensor([page * me.page_size])
                            )
                            me.fault.done = True
                            me.fault.note = (
                                f"step {me.seq}: direct-linker store source page {page} freed "
                                "after its external-read epoch opened"
                            )
                    else:
                        me.ledger.external_write(
                            ExternalWrite(
                                did,
                                pool,
                                blocks,
                                offsets,
                                source="sglang_external_linker_load",
                                after=job["after"],
                            ),
                            site=f"{did}@{job['submitted_seq']}",
                        )
                        me.stats["external_linker_load_blocks"] += len(blocks)
                    job["dids"].append(did)
                me.stats[f"external_linker_{job['kind']}_starts"] += 1
            maybe_gate_transfer(job)

        def maybe_gate_transfer(job: dict[str, Any]) -> None:
            """Optionally pause one transfer after readiness and before its first device touch.

            An experiment-only scheduling control: the worker drops a marker after KVGuard opens
            the real epoch and waits for the driver to publish ``release``, so public lifecycle
            operations can run while a source read or destination write is definitely in flight.
            """

            nonlocal gate_used
            if gate_dir_raw is None or job["kind"] != gate_kind:
                return
            requests = tuple(str(rid) for rid in job.get("requests", ()))
            if gate_request is not None and gate_request not in requests:
                return
            with gate_lock:
                if gate_used:
                    return
                gate_used = True
            gate_dir = Path(gate_dir_raw)
            gate_dir.mkdir(parents=True, exist_ok=True)
            marker = gate_dir / "started.json"
            temporary = gate_dir / "started.json.tmp"
            release = gate_dir / "release"
            temporary.write_text(
                json.dumps(
                    {
                        "job_id": job["id"],
                        "kind": job["kind"],
                        "requests": requests,
                        "submitted_seq": job["submitted_seq"],
                        "pools": {pool: list(blocks) for pool, blocks in job["pools"].items()},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            os.replace(temporary, marker)
            me.stats[f"external_linker_{job['kind']}_gate_starts"] += 1
            deadline = time.monotonic() + gate_timeout
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(0.001)
            if release.exists():
                me.stats[f"external_linker_{job['kind']}_gate_releases"] += 1
            else:
                me.stats[f"external_linker_{job['kind']}_gate_timeouts"] += 1

        def credit_load(job: dict[str, Any]) -> None:
            """Publish the layer-wise load's logical contents before its guarded forward.

            The framework exposes one logical prefix while loading physical tensors layer by
            layer; since each layer's first read is ordered after that layer lands, the
            layer-agnostic ledger credits the logical slots once the batch is accepted. The
            separate ``ExternalWrite`` epoch still catches a freed/reused destination at its
            first physical touch.
            """

            with me._external_linker_lock:
                if job.get("credited"):
                    return
                job["credited"] = True
                for pool, blocks in job["pools"].items():
                    for block in blocks:
                        me.ledger.external_fill(
                            ExternalFill(
                                pool,
                                block,
                                range(1 if pool == STATE_POOL else me.page_size),
                                source="sglang_external_linker_layerwise_load",
                            ),
                            site=f"external-linker-load:{job['id']}:{pool}@{me.seq}",
                        )
                me.stats["external_linker_load_credits"] += 1

        def finish_job(job: dict[str, Any], *, success: bool) -> None:
            with me._external_linker_lock:
                for did in job["dids"]:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
                me.stats[f"external_linker_{job['kind']}_completions"] += 1
                if not success:
                    me.stats[f"external_linker_{job['kind']}_failures"] += 1

        orig_record_ready = linker.record_transfer_ready

        def record_transfer_ready():
            ready = orig_record_ready()
            job = me._external_linker_context
            if job is None:
                me.stats["external_linker_unattributed_ready"] += 1
                return ready
            # The new current-head contract records a forward-stream event here.  Every open
            # dispatch already submitted to that stream therefore precedes the later worker
            # touch, even though the worker itself has no CUDA stream identity.
            job["after"] = tuple(
                did
                for epoch in me.open.values()
                for did in epoch.dispatch_ids
                if did in me.ledger.epochs
            )
            return _TransferReadyHook(ready, lambda: begin_job(job))

        linker.record_transfer_ready = record_transfer_ready

        orig_offload = linker.offload

        def offload(transfers, *args, **kwargs):
            job = new_job("store", transfer_pools(transfers))
            previous = me._external_linker_context
            me._external_linker_context = job
            try:
                queued = orig_offload(transfers, *args, **kwargs)
            finally:
                me._external_linker_context = previous
            if queued:
                me._external_linker_store_jobs.append(job)
                me.stats["external_linker_store_submissions"] += 1
            return queued

        linker.offload = offload
        orig_pop_offload = linker.pop_completed_offload

        def pop_completed_offload():
            success = orig_pop_offload()
            if not me._external_linker_store_jobs:
                me.reporter.emit(
                    integration_error(
                        "external_linker_store_completion",
                        "linker returned an offload completion with no tracked submission",
                        pool=POOL,
                        site=f"external-linker@{me.seq}",
                    )
                )
                return success
            finish_job(me._external_linker_store_jobs.popleft(), success=bool(success))
            return success

        linker.pop_completed_offload = pop_completed_offload

        orig_load = linker.load

        def load(rid, transfers):
            queued = orig_load(rid, transfers)
            if queued:
                pools = transfer_pools(transfers)
                if rid in me._external_linker_pending_loads:
                    me.reporter.emit(
                        integration_error(
                            "external_linker_duplicate_load",
                            f"request {rid!r} queued two direct-linker loads",
                            pool=POOL,
                            site=f"external-linker@{me.seq}",
                        )
                    )
                me._external_linker_pending_loads[rid] = pools
                me.stats["external_linker_load_submissions"] += 1
            return queued

        linker.load = load
        orig_cancel_load = linker.cancel_queued_load

        def cancel_queued_load(rid):
            cancelled = orig_cancel_load(rid)
            if cancelled:
                me._external_linker_pending_loads.pop(rid, None)
                me.stats["external_linker_load_cancellations"] += 1
            return cancelled

        linker.cancel_queued_load = cancel_queued_load
        orig_start_load = linker.start_layer_wise_loading

        def start_layer_wise_loading():
            request_ids = tuple(me._external_linker_pending_loads)
            combined: dict[str, list[int]] = {}
            for pools in me._external_linker_pending_loads.values():
                for pool, blocks in pools.items():
                    combined.setdefault(pool, []).extend(blocks)
            pools = {
                pool: tuple(dict.fromkeys(blocks)) for pool, blocks in combined.items() if blocks
            }
            job = new_job("load", pools)
            job["requests"] = request_ids
            if (
                me.fault is not None
                and me.fault.kind == "external_linker_load_early_free"
                and me.fault._armed(me)
                and pools.get(POOL)
            ):
                page = pools[POOL][0]
                me.allocator.free(me.allocator.free_pages.new_tensor([page * me.page_size]))
                me.fault.done = True
                me.fault.note = (
                    f"step {me.seq}: direct-linker load destination page {page} freed after "
                    "reservation and before the transfer-ready device write"
                )
            previous = me._external_linker_context
            me._external_linker_context = job if pools else None
            try:
                index = orig_start_load()
            finally:
                me._external_linker_context = previous
            if index >= 0:
                if index in me._external_linker_load_jobs:
                    me.reporter.emit(
                        integration_error(
                            "external_linker_load_index",
                            f"layer-wise load index {index} was reused before completion",
                            pool=POOL,
                            site=f"external-linker@{me.seq}",
                        )
                    )
                me._external_linker_load_jobs[index] = job
                me._external_linker_pending_loads.clear()
                credit_load(job)
                me.stats["external_linker_load_batches"] += 1
            return index

        linker.start_layer_wise_loading = start_layer_wise_loading
        orig_finish_load = linker.finish_layer_wise_loading

        def finish_layer_wise_loading(index):
            landed = orig_finish_load(index)
            job = me._external_linker_load_jobs.pop(index, None)
            if job is None:
                me.reporter.emit(
                    integration_error(
                        "external_linker_load_completion",
                        f"layer-wise load index {index} completed with no tracked submission",
                        pool=POOL,
                        site=f"external-linker@{me.seq}",
                    )
                )
            else:
                finish_job(job, success=bool(landed))
            return landed

        linker.finish_layer_wise_loading = finish_layer_wise_loading

        def drain_jobs(*, success: bool = False) -> None:
            while me._external_linker_store_jobs:
                finish_job(me._external_linker_store_jobs.popleft(), success=success)
            for job in tuple(me._external_linker_load_jobs.values()):
                finish_job(job, success=success)
            me._external_linker_load_jobs.clear()
            me._external_linker_pending_loads.clear()

        orig_reset = linker.reset

        def reset():
            out = orig_reset()
            drain_jobs()
            return out

        linker.reset = reset
        orig_close = linker.close

        def close():
            out = orig_close()
            drain_jobs()
            return out

        linker.close = close
        self.stats["external_linker_wrapped"] += 1

    def _external_linker_internal_pin_slots(self) -> set[int]:
        """Return the full radix paths locked by pending direct-linker offloads."""

        if not self.external_linker:
            return set()
        core = self.tree.tree_core
        root = core.empty_match_result.last_device_node
        held: set[int] = set()
        for pending in tuple(self.tree.linker.pending_offloads):
            held.update(_ints(core.collect_full_device_indices(pending.lock_node_id, root)))
        return held

    def _install_hicache(self) -> None:
        """Track exact-main HiCache device reads and fills at its L2 submissions.

        The controller queues concrete device slots, merges them at ``start_*``, and attaches one
        CUDA completion event to the resulting ack. D2H is an asynchronous source read whose
        lifecycle stays open through that event. H2D initializes freshly allocated slots only
        after the same event is satisfied. The radix walk already omits host-only nodes.
        """
        controller = self.tree.cache_controller
        me = self
        load_delay_cycles = int(os.environ.get("KVGUARD_SGLANG_HICACHE_LOAD_DELAY_CYCLES", "0"))
        load_gate_dir_raw = os.environ.get("KVGUARD_SGLANG_HICACHE_LOAD_GATE_DIR")
        load_delay_used = False
        store_delay_cycles = int(os.environ.get("KVGUARD_SGLANG_HICACHE_STORE_DELAY_CYCLES", "0"))
        store_gate_dir_raw = os.environ.get("KVGUARD_SGLANG_HICACHE_STORE_GATE_DIR")
        store_delay_used = False

        if self.facts.get("unified_allocator"):
            # HybridCacheController.load deliberately bypasses ChainAllocator and allocates
            # restore destinations from its owner band directly. Mediate that path by the
            # authoritative stable-virtual free-list difference, just as
            # _install_unified_allocator does for ordinary scheduler allocations.
            owner = self.allocator.full_attn_allocator
            governed = self._cache_pools()
            orig_load = controller.load

            def load(*args, **kwargs):
                before = set(_ints(owner.free_virtual_ids))
                out = orig_load(*args, **kwargs)
                if out is None:
                    return out
                taken = sorted(before.difference(_ints(owner.free_virtual_ids)))
                if taken:
                    for pool in governed:
                        me.ledger.allocate(pool, taken, site=f"hicache-load-alloc@{me.seq}")
                    me.stats["hicache_load_allocations"] += 1
                    me.stats["hicache_load_alloc_pages"] += len(taken)
                return out

            controller.load = load
        elif self.separate_swa:
            # HiCache restores allocate the FULL destinations through the ordinary path, but
            # materialize the auxiliary SWA destinations while assembling the component load.
            # Observe the authoritative SWA free-page delta around that composite operation.
            swa_owner = self.allocator.swa_attn_allocator
            orig_load = controller.load

            def load(*args, **kwargs):
                before = set(free_slots_from_allocator(swa_owner))
                queue_before = len(controller.load_queue)
                out = orig_load(*args, **kwargs)
                allocated = {
                    page
                    for operation in controller.load_queue[queue_before:]
                    for transfer in (getattr(operation, "pool_transfers", None) or ())
                    if str(transfer.name) == "swa" and transfer.device_indices is not None
                    for page in pages_of(_ints(transfer.device_indices), me.page_size)
                }
                if allocated:
                    # The controller keeps a bound SWA allocation callback from construction,
                    # so the ordinary suballocator wrapper cannot see this path. It may also
                    # evict and immediately reuse a formerly live page inside the same call;
                    # that page never appears in a before/after free-list delta.
                    not_free_before = allocated.difference(before)
                    me.ledger.allocate(
                        SWA_POOL, sorted(allocated), site=f"hicache-swa-load-alloc@{me.seq}"
                    )
                    me.stats["hicache_swa_load_allocations"] += 1
                    me.stats["hicache_swa_load_alloc_pages"] += len(allocated)
                    me.stats["hicache_swa_load_pages_not_free_before"] += len(not_free_before)
                return out

            controller.load = load

        def queued_pools(queue) -> dict[str, tuple[int, ...]]:
            slots: dict[str, list[int]] = {POOL: []}
            for operation in queue:
                slots[POOL].extend(_ints(operation.device_indices))
                for transfer in getattr(operation, "pool_transfers", None) or ():
                    name = str(transfer.name)
                    if name == "draft" and me.speculative_eagle3:
                        slots.setdefault(DRAFT_POOL, []).extend(_ints(transfer.device_indices))
                    elif name == "swa" and me.separate_swa:
                        slots.setdefault(SWA_POOL, []).extend(_ints(transfer.device_indices))
                    elif name == "mamba" and me.state is not None:
                        slots.setdefault(STATE_POOL, []).extend(_ints(transfer.device_indices))
            return {pool: tuple(dict.fromkeys(values)) for pool, values in slots.items() if values}

        def transfer_blocks(pool: str, slots: tuple[int, ...]) -> tuple[int, ...]:
            if pool == STATE_POOL:
                return slots
            return tuple(dict.fromkeys(pages_of(slots, me.page_size)))

        def transfer_offsets(pool: str) -> range:
            return range(1 if pool == STATE_POOL else me.page_size)

        orig_start_writing = controller.start_writing

        def start_writing(*args, **kwargs):
            nonlocal store_delay_used
            pools = queued_pools(controller.write_queue)
            slots = pools.get(POOL, ())
            store_gate_dir = Path(store_gate_dir_raw) if store_gate_dir_raw is not None else None
            delay_this_store = bool(
                slots
                and store_delay_cycles
                and not store_delay_used
                and (store_gate_dir is None or (store_gate_dir / "arm").exists())
            )
            if delay_this_store:
                import torch

                store_delay_used = True
                stream = controller.l2_transfer_engine.device_to_host_stream
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(store_delay_cycles)
            before = len(controller.ack_write_queue)
            out = orig_start_writing(*args, **kwargs)
            added = controller.ack_write_queue[before:]
            if slots and len(added) == 1:
                ack = added[0]
                me._hicache_seq += 1
                stream = int(controller.l2_transfer_engine.device_to_host_stream.cuda_stream)
                dids = []
                for pool, pool_slots in pools.items():
                    suffix = "kv" if pool == POOL else pool
                    did = f"hicache-store:{me._hicache_seq}:{suffix}"
                    me.ledger.external_read(
                        ExternalRead(
                            did,
                            pool,
                            transfer_blocks(pool, pool_slots),
                            transfer_offsets(pool),
                            source="sglang_hicache_store",
                            stream=stream,
                        ),
                        site=f"{did}@{me.seq}",
                    )
                    dids.append(did)
                    if pool == DRAFT_POOL:
                        me.stats["hicache_draft_store_slots"] += len(pool_slots)
                    elif pool == SWA_POOL:
                        me.stats["hicache_swa_store_slots"] += len(pool_slots)
                    elif pool == STATE_POOL:
                        me.stats["hicache_state_store_slots"] += len(pool_slots)
                me._hicache_reads[id(ack)] = (ack.finish_event, tuple(dids))
                me._hicache_store_reservations[id(ack)] = slots
                me.stats["hicache_store_submissions"] += 1
                me.stats["hicache_store_slots"] += len(slots)
                if delay_this_store:
                    me.stats["hicache_store_delay_starts"] += 1
                    if store_gate_dir is not None:
                        marker = store_gate_dir / "started.json"
                        temporary = store_gate_dir / "started.json.tmp"
                        temporary.write_text(
                            json.dumps(
                                {
                                    "kind": "hicache_store",
                                    "step": me.seq,
                                    "delay_cycles": store_delay_cycles,
                                    "pools": {
                                        pool: len(pool_slots) for pool, pool_slots in pools.items()
                                    },
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        temporary.replace(marker)
            elif slots:
                me.reporter.emit(
                    integration_error(
                        "hicache_store_ack_shape",
                        f"one merged store produced {len(added)} completion acks",
                        pool=POOL,
                        site=f"hicache-store@{me.seq}",
                    )
                )
            return out

        controller.start_writing = start_writing
        orig_start_loading = controller.start_loading

        def start_loading(*args, **kwargs):
            nonlocal load_delay_used
            pools = queued_pools(controller.load_queue)
            slots = pools.get(POOL, ())
            delay_this_load = bool(slots and load_delay_cycles and not load_delay_used)
            if delay_this_load:
                import torch

                load_delay_used = True
                stream = controller.l2_transfer_engine.host_to_device_stream
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(load_delay_cycles)
            fault_indices = None
            fault_pool = None
            if (
                me.fault is not None
                and me.fault.kind in ("hicache_load_early_free", "hicache_state_load_early_free")
                and me.fault._armed(me)
                and controller.load_queue
            ):
                if me.fault.kind == "hicache_load_early_free":
                    fault_indices = controller.load_queue[0].device_indices[:1].clone()
                    fault_pool = POOL
                else:
                    for operation in controller.load_queue:
                        for transfer in getattr(operation, "pool_transfers", None) or ():
                            if str(transfer.name) == "mamba" and len(transfer.device_indices):
                                fault_indices = transfer.device_indices[:1].clone()
                                fault_pool = STATE_POOL
                                break
                        if fault_indices is not None:
                            break
            before = len(controller.ack_load_queue)
            out = orig_start_loading(*args, **kwargs)
            added = controller.ack_load_queue[before:]
            if slots and len(added) == 1:
                ack = added[0]
                me._hicache_loads[id(ack)] = (
                    ack.finish_event,
                    tuple(
                        (pool, transfer_blocks(pool, pool_slots))
                        for pool, pool_slots in pools.items()
                    ),
                )
                me.stats["hicache_load_submissions"] += 1
                me.stats["hicache_load_slots"] += len(slots)
                me.stats["hicache_draft_load_slots"] += len(pools.get(DRAFT_POOL, ()))
                me.stats["hicache_swa_load_slots"] += len(pools.get(SWA_POOL, ()))
                me.stats["hicache_state_load_slots"] += len(pools.get(STATE_POOL, ()))
                if delay_this_load:
                    me.stats["hicache_load_delay_starts"] += 1
                    if load_gate_dir_raw is not None:
                        gate_dir = Path(load_gate_dir_raw)
                        gate_dir.mkdir(parents=True, exist_ok=True)
                        marker = gate_dir / "started.json"
                        temporary = gate_dir / "started.json.tmp"
                        temporary.write_text(
                            json.dumps(
                                {
                                    "kind": "hicache_load",
                                    "step": me.seq,
                                    "delay_cycles": load_delay_cycles,
                                    "pools": {
                                        pool: len(pool_slots) for pool, pool_slots in pools.items()
                                    },
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        temporary.replace(marker)
            elif slots:
                me.reporter.emit(
                    integration_error(
                        "hicache_load_ack_shape",
                        f"one merged load produced {len(added)} completion acks",
                        pool=POOL,
                        site=f"hicache-load@{me.seq}",
                    )
                )
            if fault_indices is not None and slots and len(added) == 1:
                allocator = me.state.allocator if fault_pool == STATE_POOL else me.allocator
                allocator.free(fault_indices)
                me.fault.done = True
                me.fault.note = (
                    f"step {me.seq}: HiCache H2D {fault_pool} destination slot "
                    f"{int(fault_indices[0].item())} freed after submission and before its "
                    "completion event"
                )
            return out

        controller.start_loading = start_loading
        orig_writing_check = self.tree.writing_check

        def writing_check(*args, **kwargs):
            out = orig_writing_check(*args, **kwargs)
            me._reap_hicache()
            live = {id(ack) for ack in controller.ack_write_queue}
            for key in set(me._hicache_store_reservations).difference(live):
                del me._hicache_store_reservations[key]
            return out

        self.tree.writing_check = writing_check
        orig_loading_check = self.tree.loading_check

        def loading_check(*args, **kwargs):
            out = orig_loading_check(*args, **kwargs)
            me._reap_hicache()
            return out

        self.tree.loading_check = loading_check
        self.stats["hicache_wrapped"] += 1

    def _install_decode_offload(self) -> None:
        """Track the decode node's independent asynchronous D2H offload controller."""
        manager = self.scheduler.decode_offload_manager
        controller = manager.cache_controller
        me = self
        orig_offload = manager.offload_kv_cache

        def offload(req):
            previous = me._decode_offload_owner
            me._decode_offload_owner = me.request_incarnations(req)
            try:
                return orig_offload(req)
            finally:
                me._decode_offload_owner = previous

        manager.offload_kv_cache = offload
        orig_start_writing = controller.start_writing

        def start_writing(*args, **kwargs):
            gpu_open_steps = []
            gpu_done_steps = []
            for seq, epoch in me.open.items():
                if bool(epoch.event.query()):
                    gpu_done_steps.append(seq)
                else:
                    gpu_open_steps.append(seq)
            queued = []
            for operation in controller.write_queue:
                try:
                    queued.extend(
                        external_slot_groups(_ints(operation.device_indices), me.page_size)
                    )
                except (TypeError, ValueError) as exc:
                    me.reporter.emit(
                        integration_error(
                            "pd_decode_offload_source_shape",
                            str(exc),
                            pool=POOL,
                            site=f"pd-decode-offload@{me.seq}",
                        )
                    )
            queued_blocks = tuple(block for blocks, _offsets in queued for block in blocks)
            write_overlap = tuple(sorted(set(queued_blocks).intersection(me._last_write_blocks)))
            before = len(controller.ack_write_queue)
            opened = []
            stream = int(controller.l2_transfer_engine.device_to_host_stream.cuda_stream)
            for blocks, offsets in queued:
                me._pd_seq += 1
                did = f"pd-decode-offload:{me._pd_seq}"
                me.ledger.external_read(
                    ExternalRead(
                        did,
                        POOL,
                        blocks,
                        offsets,
                        source="sglang_pd_decode_offload",
                        request=me._decode_offload_owner,
                        stream=stream,
                    ),
                    site=f"{did}@{me.seq}",
                )
                opened.append(did)
            try:
                result = orig_start_writing(*args, **kwargs)
            except Exception:
                for did in opened:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
                raise
            added = controller.ack_write_queue[before:]
            if opened and len(added) == 1:
                ack = added[0]
                me.stats["pd_decode_offload_submissions"] += 1
                me.stats["pd_decode_offload_pages"] += sum(
                    len(blocks) for blocks, _offsets in queued
                )
                me.stats["pd_decode_offload_tokens"] += sum(
                    len(blocks) * me.page_size
                    if len(offsets) == me.page_size
                    else (len(blocks) - 1) * me.page_size + len(offsets)
                    for blocks, offsets in queued
                )
                fault = getattr(me, "fault", None)
                if (
                    fault is not None
                    and fault.kind == "pd_decode_offload_early_free"
                    and fault._armed(me)
                    and queued
                    and not bool(ack.finish_event.query())
                ):
                    page = queued[0][0][0]
                    me.allocator.free(me.allocator.free_pages.new_tensor([page * me.page_size]))
                    fault.done = True
                    fault.note = (
                        f"step {me.seq}: decode offload source page {page} freed after its "
                        "D2H epoch opened and before the completion event"
                    )
                me._decode_offload_reads[id(ack)] = (ack.finish_event, tuple(opened))
                if gpu_open_steps:
                    me.stats["pd_decode_offload_submissions_gpu_open"] += 1
                    if write_overlap:
                        me.stats["pd_decode_offload_write_races"] += 1
                if len(me._decode_offload_trace) < 80:
                    me._decode_offload_trace.append(
                        {
                            "seq": me.seq,
                            "request": me._decode_offload_owner,
                            "blocks": queued_blocks,
                            "forward_write_overlap": write_overlap,
                            "gpu_open_steps": tuple(gpu_open_steps),
                            "gpu_done_steps": tuple(gpu_done_steps),
                            "write_fence": controller.write_fence_stream is not None,
                        }
                    )
            elif opened:
                for did in opened:
                    if did in me.ledger.epochs:
                        me.ledger.complete(did)
                me.reporter.emit(
                    integration_error(
                        "pd_decode_offload_ack_shape",
                        f"one decode offload produced {len(added)} completion acks",
                        pool=POOL,
                        site=f"pd-decode-offload@{me.seq}",
                    )
                )
            return result

        controller.start_writing = start_writing
        self.stats["pd_decode_offload_wrapped"] += 1

    def _reap_hicache(self) -> None:
        """Apply transfers whose exact CUDA completion events have fired, without waiting."""
        for key, (event, dids) in list(self._hicache_reads.items()):
            if event.query():
                for did in dids:
                    self.ledger.complete(did)
                del self._hicache_reads[key]
                self.stats["hicache_store_completions"] += 1
        for key, (event, dids) in list(getattr(self, "_decode_offload_reads", {}).items()):
            if event.query():
                for did in dids:
                    self.ledger.complete(did)
                del self._decode_offload_reads[key]
                self.stats["pd_decode_offload_completions"] += 1
        for key, (event, pools) in list(self._hicache_loads.items()):
            if not event.query():
                continue
            for pool, slots in pools:
                for slot in slots:
                    self.ledger.external_fill(
                        ExternalFill(
                            pool,
                            slot,
                            range(1 if pool == STATE_POOL else self.page_size),
                            source="sglang_hicache_load",
                        ),
                        site=f"hicache-load:{key}:{pool}@{self.seq}",
                    )
            del self._hicache_loads[key]
            self.stats["hicache_load_completions"] += 1

    def _hicache_internal_pin_slots(self) -> set[int]:
        """Return device slots reserved by HiCache's tree-path locks.

        A write-through or load-back pins its complete ancestor path until its acknowledgement;
        those internal locks are reservations even when no request row names the ancestors. The
        Rust walk exposes only that they are locked, so derive the exact paths from the
        controller's operation maps.
        """
        core = self.tree.tree_core
        root = core.empty_match_result.last_device_node
        held: set[int] = set()
        pipeline = getattr(self.tree, "buffer_pipeline", None)
        if pipeline is None:
            write_entries = (
                (pending.node_id, pending.lock_params)
                for pending in getattr(self.tree, "ongoing_write_through", {}).values()
            )
        else:
            write_entries = (
                (entry.intent.snapshot.node_id, entry.lock_params)
                for entry in pipeline.ongoing_write_through.values()
            )
        for node_id, lock_params in write_entries:
            if lock_params is not None:
                held.update(_ints(core.collect_full_device_indices(node_id, root)))
        for pending in getattr(self.tree, "ongoing_load_back", {}).values():
            held.update(_ints(core.collect_full_device_indices(pending.node_id, root)))
        if pipeline is not None:
            for lock in pipeline.anchor_locks.values():
                held.update(_ints(core.collect_full_device_indices(lock.node_id, root)))
        return held

    def _install_ngram_copy(self) -> None:
        """Describe NGRAM's accepted-candidate copy and extend the step event through it.

        The move selects sources from the verify step's ``out_cache_loc`` and copies them into
        the leading candidate slots, preserving mapping and initializedness, so the descriptor
        checks both operands as reads against positions derived from the acceptance lengths.
        """
        import torch
        from sglang.srt.speculative import ngram_worker

        cache = self.allocator.get_kvcache()
        original = cache.move_kv_cache
        original_compact = ngram_worker.move_accept_tokens_to_target_kvcache
        me = self

        def compact(batch, accept_index, num_correct_drafts, allocator):
            lens = [value + 1 for value in _ints(num_correct_drafts)]
            width = int(accept_index.shape[1])
            me._ngram_copy_plan = (lens, _ints(accept_index.reshape(-1)), width)
            try:
                return original_compact(batch, accept_index, num_correct_drafts, allocator)
            finally:
                me._ngram_copy_plan = None

        # NGRAMWorker imported this helper into its module.  This process-local replacement is
        # the only point that exposes the per-request accepted lengths needed to interpret the
        # compacted arrays; the framework source remains untouched.
        ngram_worker.move_accept_tokens_to_target_kvcache = compact

        def move_kv_cache(target_locs, source_locs, *args, **kwargs):
            targets, sources = _ints(target_locs), _ints(source_locs)
            epoch = me.open.get(me.seq)
            plan = me._ngram_copy_plan
            descriptors: list[DispatchDescriptor] = []
            if plan is None:
                me.reporter.emit(
                    integration_error(
                        "spec_copy_without_plan",
                        "accepted-token copy ran outside the NGRAM compaction boundary",
                        pool=POOL,
                        site=f"spec-copy@{me.seq}",
                    )
                )
            else:
                lens, accept_index, width = plan
                try:
                    descriptors.append(
                        build_ngram_copy_dispatch(
                            me.seq,
                            POOL,
                            me._verify_requests,
                            targets,
                            sources,
                            lens,
                            accept_index,
                            width,
                        )
                    )
                    if me.separate_swa:
                        descriptors.append(
                            build_ngram_copy_dispatch(
                                me.seq,
                                SWA_POOL,
                                me._verify_requests,
                                _ints(cache.translate_loc_from_full_to_swa(target_locs)),
                                _ints(cache.translate_loc_from_full_to_swa(source_locs)),
                                lens,
                                accept_index,
                                width,
                                tag="-swa",
                            )
                        )
                except ValueError as exc:
                    me.reporter.emit(
                        integration_error(
                            "spec_copy_shape",
                            str(exc),
                            pool=POOL,
                            site=f"spec-copy@{me.seq}",
                        )
                    )
            pairs = sum(plan[0]) if plan is not None else 0
            if descriptors and epoch is not None:
                for descriptor in descriptors:
                    me._dispatch_one(descriptor, epoch.dispatch_ids)
                me.stats["dispatches"] += len(descriptors)
                me.stats["speculative_copy_sources"] += pairs
                me._last_dispatch["dispatches"] = len(epoch.dispatch_ids)
            elif descriptors:
                me.reporter.emit(
                    integration_error(
                        "spec_copy_without_epoch",
                        "accepted-token copy ran without the target-verify epoch",
                        pool=POOL,
                        site=f"spec-copy@{me.seq}",
                    )
                )
            out = original(target_locs, source_locs, *args, **kwargs)
            if epoch is not None:
                event = torch.cuda.Event()
                event.record()
                epoch.event = event
                me.stats["speculative_copy_events"] += 1
            return out

        cache.move_kv_cache = move_kv_cache

    def _install_eagle3_draft(self) -> None:
        """Observe every eager or CUDA-graph EAGLE3 draft-model forward.

        EAGLEWorkerV2 calls the independent draft ``ModelRunner`` directly for eager prefill,
        chain-decode steps and ``DRAFT_EXTEND_V2``, each selecting its own FlashInfer backend.
        Under CUDA graphs one decode replay holds all inner steps and draft-extend has a separate
        replay; wrap those two graph-runner instances and translate their wrapper metadata.
        """
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )

        if self.page_size > 1 and int(self.facts.get("speculative_eagle_topk", 1)) > 1:
            from sglang.srt.speculative import eagle_worker_common

            draft_pool = self.draft_runner.token_to_kv_pool
            original_move = draft_pool.move_kv_cache
            original_duplicate = eagle_worker_common.duplicate_prefix_tail_to_draft_branches
            me = self

            def move_kv_cache(target_locs, source_locs, *args, **kwargs):
                if me._draft_tree_copy_context is not None:
                    me._record_eagle3_branch_copy(
                        target_locs, source_locs, me._draft_tree_copy_context
                    )
                return original_move(target_locs, source_locs, *args, **kwargs)

            def duplicate_prefix_tail_to_draft_branches(
                token_to_kv_pool,
                rows,
                prefix_base,
                last_page,
                num_new_pages,
                topk,
                page_size,
            ):
                me._draft_tree_copy_context = me._begin_eagle3_branch_copy(
                    prefix_base, last_page, topk, page_size
                )
                try:
                    return original_duplicate(
                        token_to_kv_pool,
                        rows,
                        prefix_base,
                        last_page,
                        num_new_pages,
                        topk,
                        page_size,
                    )
                finally:
                    me._draft_tree_copy_context = None

            draft_pool.move_kv_cache = move_kv_cache
            eagle_worker_common.duplicate_prefix_tail_to_draft_branches = (
                duplicate_prefix_tail_to_draft_branches
            )

        runner = self.draft_runner
        original = runner.forward
        me = self

        def forward(forward_batch, *args, **kwargs):
            backend = get_attn_backend() if has_forward_context() else runner.attn_backend
            out = original(forward_batch, *args, **kwargs)
            if me._draft_prefill_graph_seq == me.seq:
                me.stats["draft_prefill_graph_forwards"] += 1
                me._draft_prefill_graph_seq = None
            me._after_draft_forward(forward_batch, backend)
            return out

        runner.forward = forward

        prefill_graph = getattr(runner, "prefill_cuda_graph_runner", None)
        if prefill_graph is not None and type(prefill_graph).__name__ == "PrefillCudaGraphRunner":
            original_prefill_execute = prefill_graph.execute

            def prefill_execute(*args, **kwargs):
                out = original_prefill_execute(*args, **kwargs)
                me._draft_prefill_graph_seq = me.seq
                me.stats["draft_prefill_graph_replays"] += 1
                return out

            prefill_graph.execute = prefill_execute

        draft_host = self.scheduler.draft_worker.draft_worker
        # Adaptive EAGLE swaps among pre-built graph-runner instances. Wrap every state, not
        # just the instance active when KVGuard attaches, so zero-step catch-up writes remain
        # visible before a later positive-step state consumes them.
        adaptive_controller = getattr(self.scheduler.draft_worker, "adaptive_controller", None)
        decode_graphs = _adaptive_runtime_graphs(
            draft_host, adaptive_controller, "cuda_graph_runner"
        )
        for decode_graph in decode_graphs:
            original_decode_execute = decode_graph.execute

            def decode_execute(
                forward_batch,
                *args,
                _graph=decode_graph,
                _original=original_decode_execute,
                **kwargs,
            ):
                views = me._eagle3_decode_graph_views(forward_batch, draft_host)
                out = _original(forward_batch, *args, **kwargs)
                for view, backend in views:
                    view._kvguard_graph_bs = int(_graph.bs)
                    me._after_draft_forward(view, backend)
                me.stats["draft_graph_decode_replays"] += 1
                return out

            decode_graph.execute = decode_execute

        extend_graphs = _adaptive_runtime_graphs(
            draft_host, adaptive_controller, "cuda_graph_runner_for_draft_extend"
        )
        for extend_graph in extend_graphs:
            original_extend_execute = extend_graph.execute

            def extend_execute(
                forward_batch,
                *args,
                _graph=extend_graph,
                _original=original_extend_execute,
                **kwargs,
            ):
                view = types.SimpleNamespace(
                    forward_mode=forward_batch.forward_mode,
                    batch_size=int(forward_batch.batch_size),
                    req_pool_indices=_ints(forward_batch.req_pool_indices),
                    extend_prefix_lens=_ints(forward_batch.extend_prefix_lens),
                    extend_seq_lens=_ints(forward_batch.extend_seq_lens),
                    out_cache_loc=_ints(forward_batch.out_cache_loc),
                )
                out = _original(forward_batch, *args, **kwargs)
                view._kvguard_graph_bs = int(_graph.bs)
                view._kvguard_use_ragged = False
                me._after_draft_forward(view, draft_host.draft_extend_attn_backend)
                me.stats["draft_graph_extend_replays"] += 1
                return out

            extend_graph.execute = extend_execute

        self.stats["draft_graph_decode_runners_wrapped"] = len(decode_graphs)
        self.stats["draft_graph_extend_runners_wrapped"] = len(extend_graphs)

        post_verify_graphs = getattr(self.scheduler.draft_worker, "_post_verify_graphs", None)
        if post_verify_graphs is not None:
            original_post_verify_run = post_verify_graphs.run

            def post_verify_run(*args, **kwargs):
                key = post_verify_graphs._key(
                    kwargs["batch"],
                    kwargs["verify_input"],
                    kwargs["logits_output"],
                    kwargs["can_run_cuda_graph"],
                    kwargs.get("max_top_k"),
                )
                out = original_post_verify_run(*args, **kwargs)
                me.stats["eagle_post_verify_calls"] += 1
                if key is None:
                    me.stats["eagle_post_verify_ineligible"] += 1
                    return out
                kind, captured, state_keys = _post_verify_graph_observation(post_verify_graphs, key)
                if kind == "state_limit":
                    me.stats["eagle_post_verify_state_limit_fallbacks"] += 1
                    return out
                if kind == "graph":
                    me.stats["eagle_post_verify_graph_replays"] += 1
                else:
                    me.stats["eagle_post_verify_warmups"] += 1
                me.stats["eagle_post_verify_captured_slots_max"] = max(
                    me.stats["eagle_post_verify_captured_slots_max"], captured
                )
                me.stats["eagle_post_verify_state_keys_max"] = max(
                    me.stats["eagle_post_verify_state_keys_max"],
                    state_keys,
                )
                return out

            post_verify_graphs.run = post_verify_run
            self.stats["eagle_post_verify_runner_wrapped"] = 1

    def _install_dflash_draft(self) -> None:
        """Observe eager DFlash block forwards and target-hidden draft-KV materialization."""
        import torch

        worker = self.scheduler.draft_worker
        runner = self.draft_runner
        original_forward = runner.forward
        me = self

        def forward(forward_batch, *args, **kwargs):
            out = original_forward(forward_batch, *args, **kwargs)
            backend = runner.attn_backend
            if out.can_run_graph:
                graph = runner.decode_cuda_graph_runner
                forward_batch._kvguard_graph_bs = int(graph.bs)
                backend = _replay_full_backend(graph)
                me.stats["dflash_draft_graph_replays"] += 1
            me._after_draft_forward(forward_batch, backend)
            return out

        runner.forward = forward

        original_append = worker._append_target_hidden_to_draft_kv_by_loc

        def append_target_hidden(*args, **kwargs):
            positions = _ints(kwargs["positions"])
            logical_locations = me._logical_kv_slots(_ints(kwargs["cache_loc"]))
            commit_lens = kwargs.get("commit_lens")
            cache_loc_2d = kwargs.get("cache_loc_2d")
            extend_lens = kwargs.get("extend_lens")
            descriptor = None
            kind = ""
            tokens = 0
            try:
                if cache_loc_2d is not None and commit_lens is not None:
                    lens = _ints(commit_lens)
                    n = len(lens)
                    width = int(cache_loc_2d.shape[1])
                    req_ids = me._current_dflash_req_ids(n)
                    if len(positions) < n * width or len(logical_locations) < n * width:
                        raise ValueError(
                            f"DFlash commit has positions={len(positions)}, "
                            f"locations={len(logical_locations)}, shape={n}x{width}"
                        )
                    prefixes = [positions[row * width] for row in range(n)]
                    expected = [prefix + offset for prefix in prefixes for offset in range(width)]
                    if positions[: n * width] != expected:
                        raise ValueError(
                            f"DFlash commit positions {positions[: min(len(positions), 16)]} "
                            f"do not form {n} contiguous rows of width {width}"
                        )
                    for rid, prefix in zip(req_ids, prefixes, strict=True):
                        me.ledger.rollback(
                            DRAFT_POOL,
                            rid,
                            prefix,
                            site=f"dflash-commit-rollback@{me.seq}",
                        )
                    descriptor = build_dflash_materialize_dispatch(
                        me.seq,
                        DRAFT_POOL,
                        req_ids,
                        positions,
                        logical_locations,
                        lens,
                        row_width=width,
                        tag="-dflash-commit",
                    )
                    kind = "commit"
                    tokens = sum(lens)
                    me.stats["dflash_rejected_tokens"] += n * width - tokens
                elif extend_lens is not None:
                    lens = _ints(extend_lens)
                    req_ids = me._current_dflash_req_ids(len(lens))
                    descriptor = build_dflash_materialize_dispatch(
                        me.seq,
                        DRAFT_POOL,
                        req_ids,
                        positions,
                        logical_locations,
                        lens,
                        tag="-dflash-prefill",
                    )
                    kind = "prefill"
                    tokens = sum(lens)
            except Exception as exc:
                log.exception("DFlash materialization extraction failed at step %d", me.seq)
                me.reporter.emit(
                    integration_error(
                        "dflash_materialize_extraction",
                        f"{type(exc).__name__}: {exc}"[:300],
                        pool=DRAFT_POOL,
                        site=f"dflash-materialize@{me.seq}",
                    )
                )
                me.stats["dflash_materialize_failed"] += 1
            out = original_append(*args, **kwargs)
            if descriptor is not None:
                dids: list[str] = []
                me._dispatch_one(descriptor, dids)
                me.stats["dispatches"] += len(dids)
                me.stats["draft_dispatches"] += len(dids)
                me.stats[f"dflash_{kind}_calls"] += 1
                me.stats[f"dflash_{kind}_tokens"] += tokens
                event = torch.cuda.Event()
                event.record()
                epoch = me.open.get(me.seq)
                if epoch is None:
                    me.open[me.seq] = StepEpoch(
                        me.seq,
                        dids,
                        event,
                        getattr(me.cur_batch, "forward_iter", None),
                    )
                else:
                    epoch.dispatch_ids.extend(dids)
                    epoch.event = event
                    me.stats["forward_calls_extra"] += 1
                me.events += 1
                me.dirty = True
            return out

        worker._append_target_hidden_to_draft_kv_by_loc = append_target_hidden

    def _install_multi_layer_eagle_drafts(self) -> None:
        """Observe each Step-3.5 MTP depth as an independent SWA KV pool.

        Eager execution calls three distinct ``ModelRunner.forward`` methods in order; graph
        execution replays one captured runner per depth, so retain the real batch at ``prepare``
        and dispatch each depth immediately after its replay, before the next reuses the shared
        buffers. All depths share the target request table and allocator but have physically
        independent SWA buffers.
        """

        me = self
        for index, (runner, pool) in enumerate(
            zip(self.draft_runners, self.draft_pool_ids, strict=True)
        ):
            original = runner.forward

            def forward(
                forward_batch,
                *args,
                _orig=original,
                _runner=runner,
                _pool=pool,
                _index=index,
                **kwargs,
            ):
                out = _orig(forward_batch, *args, **kwargs)
                me._after_draft_forward(
                    forward_batch,
                    _runner.attn_backend,
                    pool=_pool,
                )
                me.stats[f"multi_layer_draft_depth_{_index}_forwards"] += 1
                return out

            runner.forward = forward
        self.stats["multi_layer_draft_runners_wrapped"] = len(self.draft_runners)

        draft_host = self.scheduler.draft_worker.draft_worker
        graph = getattr(draft_host, "cuda_graph_runner_for_draft_extend", None)
        if graph is None:
            return

        original_prepare = graph.prepare

        def prepare(forward_batch, *args, **kwargs):
            out = original_prepare(forward_batch, *args, **kwargs)
            n = int(forward_batch.batch_size)
            extend = forward_batch.extend_seq_lens[:n]
            prefix = getattr(forward_batch, "extend_prefix_lens", None)
            if prefix is None:
                prefix = forward_batch.seq_lens[:n] - extend
            graph._kvguard_forward_view = types.SimpleNamespace(
                forward_mode=forward_batch.forward_mode,
                batch_size=n,
                req_pool_indices=forward_batch.req_pool_indices[:n],
                extend_prefix_lens=prefix[:n],
                extend_seq_lens=extend,
                out_cache_loc=forward_batch.out_cache_loc,
                _kvguard_use_ragged=False,
                _kvguard_graph_width=int(graph.captured_req_width),
            )
            return out

        graph.prepare = prepare
        original_replay = graph.replay

        def replay(step, *args, **kwargs):
            out = original_replay(step, *args, **kwargs)
            view = graph._kvguard_forward_view
            view._kvguard_graph_bs = int(graph.bs)
            index = int(step)
            me._after_draft_forward(
                view,
                graph.runners[index].attn_backend,
                pool=me.draft_pool_ids[index],
            )
            me.stats[f"multi_layer_draft_depth_{index}_forwards"] += 1
            me.stats["multi_layer_draft_graph_replays"] += 1
            return out

        graph.replay = replay
        self.stats["multi_layer_draft_graph_runners_wrapped"] = len(graph.runners)

    def _begin_eagle3_branch_copy(
        self, prefix_base: Any, last_page: Any, topk: int, page_size: int
    ) -> tuple[str, ...]:
        """Roll back before SGLang copies the partial prefix tail into branch pages.

        The framework helper's flattened copy order is request, nonzero branch, then tail
        offset. Returning the parallel owner list lets the wrapped draft-pool mover describe
        each physical copy without reconstructing the GPU request table.
        """
        bases, lasts = _ints(prefix_base), _ints(last_page)
        n = len(bases)
        if len(lasts) != n or int(page_size) != self.page_size:
            self.reporter.emit(
                integration_error(
                    "draft_branch_copy_shape",
                    f"prefix bases={len(bases)}, tails={len(lasts)}, page_size={page_size}, "
                    f"expected {self.page_size}",
                    pool=DRAFT_POOL,
                    site=f"draft-branch-copy@{self.seq}",
                )
            )
            return ()
        row_ids = _ints(self.cur_batch.req_pool_indices)[:n]
        req_ids = [self.rid_of_row.get(row, f"unknown-row-{row}") for row in row_ids]
        owners: list[str] = []
        for rid, base, tail in zip(req_ids, bases, lasts, strict=True):
            if tail < 0 or tail >= self.page_size:
                self.reporter.emit(
                    integration_error(
                        "draft_branch_copy_tail",
                        f"request {rid!r} has last_page={tail} for page size {self.page_size}",
                        pool=DRAFT_POOL,
                        request=rid,
                        site=f"draft-branch-copy@{self.seq}",
                    )
                )
                continue
            if tail:
                frontier = base + tail
                if rid not in self._draft_tree_copy_rolled_back:
                    self.ledger.rollback(
                        DRAFT_POOL,
                        rid,
                        frontier,
                        site=f"draft-tree-rollback@{self.seq}",
                    )
                    self._draft_tree_copy_rolled_back.add(rid)
                    self.stats["draft_branch_copy_rollbacks"] += 1
                owners.extend([rid] * ((int(topk) - 1) * tail))
        return tuple(owners)

    def _record_eagle3_branch_copy(
        self, target_locs: Any, source_locs: Any, owners: Sequence[str]
    ) -> None:
        """Describe the draft pool's per-token prefix-tail copy at page granularity."""
        targets, sources = _ints(target_locs), _ints(source_locs)
        if len(targets) != len(sources) or len(targets) != len(owners):
            self.reporter.emit(
                integration_error(
                    "draft_branch_copy_shape",
                    f"targets={len(targets)}, sources={len(sources)}, owners={len(owners)}",
                    pool=DRAFT_POOL,
                    site=f"draft-branch-copy@{self.seq}",
                )
            )
            return
        fault = getattr(self, "fault", None)
        if (
            fault is not None
            and fault.kind == "eagle_branch_copy_early_free"
            and fault._armed(self)
            and targets
        ):
            import torch

            target = targets[0]
            self.allocator.free(
                torch.tensor([target], dtype=torch.int64, device=self.cur_batch.device)
            )
            fault.done = True
            fault.note = (
                f"step {self.seq}: EAGLE draft branch prefix-tail destination slot {target} "
                "freed before the copy"
            )
        for target, source, owner in zip(targets, sources, owners, strict=True):
            src_offset, dst_offset = source % self.page_size, target % self.page_size
            if src_offset != dst_offset:
                self.reporter.emit(
                    integration_error(
                        "draft_branch_copy_offset",
                        f"source slot {source} offset {src_offset} -> target slot {target} "
                        f"offset {dst_offset}",
                        pool=DRAFT_POOL,
                        request=owner,
                        site=f"draft-branch-copy@{self.seq}",
                    )
                )
                continue
            self.ledger.external_copy(
                ExternalCopy(
                    DRAFT_POOL,
                    source // self.page_size,
                    DRAFT_POOL,
                    target // self.page_size,
                    range(src_offset, src_offset + 1),
                    source="eagle-branch-prefix",
                    request=owner,
                ),
                site=f"draft-branch-copy@{self.seq}",
            )
        self.stats["draft_branch_copy_calls"] += 1
        self.stats["draft_branch_copy_slots"] += len(targets)

    def _eagle3_decode_graph_views(
        self, forward_batch: Any, draft_host: Any
    ) -> list[tuple[Any, Any]]:
        """Freeze the raw rows of one multi-step draft graph before replay mutates buffers."""
        n = int(forward_batch.batch_size)
        steps = int(self.facts["speculative_num_steps"])
        topk = int(self.facts.get("speculative_eagle_topk", 1))
        positions = _ints(forward_batch.positions)[: n * topk]
        seq_lens = _ints(getattr(forward_batch, "seq_lens", positions[::topk]))[:n]
        req_pool_indices = _ints(forward_batch.req_pool_indices)[:n]
        locations = _ints(forward_batch.out_cache_loc)
        if len(positions) != n * topk or len(seq_lens) != n or len(req_pool_indices) != n:
            raise ValueError(
                f"draft graph has positions={len(positions)}, seq_lens={len(seq_lens)}, "
                f"rows={len(req_pool_indices)}, batch={n}, topk={topk}"
            )
        if any(
            positions[request * topk + branch] != seq_lens[request]
            for request in range(n)
            for branch in range(topk)
        ):
            raise ValueError("draft graph's initial branch positions do not equal its seq_lens")
        if len(locations) != n * topk * steps:
            raise ValueError(
                f"draft graph has {len(locations)} locations, expected {n}*{topk}*{steps}"
            )
        backends = tuple(draft_host.draft_attn_backend.attn_backends)
        if len(backends) != steps - 1:
            raise ValueError(f"draft graph has {len(backends)} step backends, expected {steps - 1}")
        views = []
        for step, backend in enumerate(backends):
            views.append(
                (
                    types.SimpleNamespace(
                        forward_mode=forward_batch.forward_mode,
                        batch_size=n,
                        req_pool_indices=req_pool_indices,
                        seq_lens=seq_lens,
                        positions=[
                            seq_lens[request] + step
                            for request in range(n)
                            for _branch in range(topk)
                        ],
                        out_cache_loc=[
                            locations[(request * topk + branch) * steps + step]
                            for request in range(n)
                            for branch in range(topk)
                        ],
                        _kvguard_draft_step=step,
                        _kvguard_draft_topk=topk,
                    ),
                    backend,
                )
            )
        return views

    def _install_paged_allocator(self, alloc: Any, *, pools: Sequence[str] | None = None) -> None:
        """The ``PagedTokenToKVPoolAllocator``'s mediation in pages: an allocation (``alloc``,
        ``alloc_extend``, ``alloc_decode``) takes pages off the free list, derived from the call's
        own arguments and the slots it returns (``paged_pages_taken``), and checked every
        ``audit_every``-th call against the difference of ``get_all_free_pages()`` before and after
        (reading that large free list twice per call is costly, so it is a fallback for a call
        whose arguments are not the ones known here). Every free reaches ``_release_page_ids``,
        the page-level free."""
        import inspect

        me = self
        governed = tuple(pools) if pools is not None else self._cache_pools()
        ps = int(alloc.page_size)

        def free_set() -> set[int]:
            # 0.5.20+ exposes get_all_free_pages() (free plus staged pages).  0.5.19 has no such
            # method: its free pages live in ``free_pages``, with any deferred-release pages of an
            # open free group in ``release_pages``; union them for the same meaning.
            getter = getattr(alloc, "get_all_free_pages", None)
            if getter is not None:
                return set(_ints(getter()))
            pages = set(_ints(getattr(alloc, "free_pages", ())))
            release = getattr(alloc, "release_pages", None)
            if release is not None:
                pages |= set(_ints(release))
            return pages

        def wrap_alloc(name: str) -> None:
            orig = getattr(alloc, name)
            try:
                sig = inspect.signature(orig)
            except (TypeError, ValueError):
                sig = None

            def wrapped(*args, **kwargs):
                me._reap_pd_reads()
                bound = None
                if sig is not None:
                    try:
                        bound = sig.bind(*args, **kwargs).arguments
                    except TypeError:
                        bound = None
                derive = bound is not None and paged_call_known(name, bound)
                audit = not derive or me.stats["alloc_calls"] % me.audit_every == 0
                before = free_set() if audit else None
                out = orig(*args, **kwargs)
                if out is not None:
                    taken = paged_pages_taken(name, bound, out, ps) if derive else None
                    if audit:
                        diff = before - free_set()
                        if taken is None:
                            taken = sorted(diff)
                        elif set(taken) != diff:
                            me.stats["alloc_pages_drift"] += 1
                            me.reporter.emit(
                                integration_error(
                                    "alloc_pages_drift",
                                    f"{name} derived pages {taken[:5]} ({len(taken)}) but the free "
                                    f"list lost {sorted(diff)[:5]} ({len(diff)})",
                                    pool=governed[0],
                                    site=f"alloc@{me.seq}",
                                )
                            )
                            taken = sorted(diff)
                        me.stats["alloc_audited"] += 1
                    elif taken is None:  # the arguments bound but the slots did not read back
                        me.stats["alloc_pages_underived"] += 1
                        taken = []
                    else:
                        me.stats["alloc_pages_derived"] += 1
                    if taken:
                        for pool in governed:
                            me.ledger.allocate(pool, taken, site=f"alloc@{me.seq}")
                        me.stats["alloc_calls"] += 1
                        me.stats["alloc_pages"] += len(taken)
                    else:
                        me.stats[f"{name}_without_new_page"] += 1
                return out

            setattr(alloc, name, wrapped)

        for name in ("alloc", "alloc_extend", "alloc_decode"):
            if hasattr(alloc, name):
                wrap_alloc(name)
        orig_release = alloc._release_page_ids

        def _release_page_ids(*page_ids):
            me._reap_pd_reads()
            pages = [p for t in page_ids for p in _ints(t)]
            me._trace_pd_source_release(pages)
            if pages and me.early_complete:
                me._complete_referencing(pages, governed)
            orig_release(*page_ids)
            if pages:
                for pool in governed:
                    me.ledger.free(pool, pages, site=f"free@{me.seq}")
                me.stats["free_calls"] += 1
                me.stats["free_pages"] += len(pages)

        alloc._release_page_ids = _release_page_ids

    def _install_unified_allocator(self, alloc: Any) -> None:
        """Mediate stable virtual owner pages while physical bands compact underneath them."""

        me = self
        owner = alloc.full_attn_allocator
        governed = self._cache_pools()

        def free_pages() -> set[int]:
            return set(_ints(owner.free_virtual_ids))

        def wrap_alloc(name: str) -> None:
            original = getattr(alloc, name)

            def wrapped(*args, **kwargs):
                me._reap_pd_reads()
                before = free_pages()
                out = original(*args, **kwargs)
                if out is None:
                    me.stats["alloc_refused"] += 1
                    return out
                taken = sorted(before - free_pages())
                if taken:
                    for pool in governed:
                        me.ledger.allocate(pool, taken, site=f"alloc@{me.seq}")
                    me.stats["alloc_calls"] += 1
                    me.stats["alloc_pages"] += len(taken)
                else:
                    me.stats[f"{name}_without_new_page"] += 1
                return out

            setattr(alloc, name, wrapped)

        for name in ("alloc", "alloc_extend", "alloc_decode"):
            wrap_alloc(name)

        original_free = owner.free

        def free(free_index, *args, **kwargs):
            me._reap_pd_reads()
            before = free_pages()
            out = original_free(free_index, *args, **kwargs)
            released = sorted(free_pages() - before)
            if released:
                if me.early_complete:
                    me._complete_referencing(released, governed)
                for pool in governed:
                    me.ledger.free(pool, released, site=f"free@{me.seq}")
                me.stats["free_calls"] += 1
                me.stats["free_pages"] += len(released)
            elif owner.free_group is not None:
                me.stats["free_deferred_to_group"] += 1
            return out

        owner.free = free
        original_clear = owner.clear

        def clear(*args, **kwargs):
            before = free_pages()
            out = original_clear(*args, **kwargs)
            released = sorted(free_pages() - before)
            if released:
                if me.early_complete:
                    me._complete_referencing(released, governed)
                for pool in governed:
                    me.ledger.free(pool, released, site=f"clear@{me.seq}")
                me.stats["clear_calls"] += 1
                me.stats["clear_pages"] += len(released)
            return out

        owner.clear = clear

    def _install_separate_swa_clear(self) -> None:
        """Mediate the composite allocator reset once for each physical sub-pool."""

        alloc = self.allocator
        original = alloc.clear
        me = self

        def used(suballocator: Any) -> set[int]:
            all_pages = set(range(1, int(suballocator.size) // self.page_size + 1))
            return all_pages - set(_ints(suballocator.get_all_free_pages()))

        def clear(*args, **kwargs):
            full_used = used(alloc.full_attn_allocator)
            swa_used = used(alloc.swa_attn_allocator)
            if me.early_complete:
                me._complete_referencing(full_used, me._full_cache_pools())
                me._complete_referencing(swa_used, me._swa_cache_pools())
            out = original(*args, **kwargs)
            for pool in me._full_cache_pools():
                me.ledger.free(pool, sorted(full_used), site=f"clear@{me.seq}")
            for pool in me._swa_cache_pools():
                me.ledger.free(pool, sorted(swa_used), site=f"clear@{me.seq}")
            me.stats["clear_calls"] += 1
            me.stats["clear_full_pages"] += len(full_used)
            me.stats["clear_swa_pages"] += len(swa_used)
            return out

        alloc.clear = clear

    # ------------------------------------------------------------------ snapshots
    def _profiled(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            prof = self.profiler
            if prof is None or not (self.profile_steps[0] <= self.seq < self.profile_steps[1]):
                return fn(*args, **kwargs)
            prof.enable()
            try:
                return fn(*args, **kwargs)
            finally:
                prof.disable()

        return run

    def _write_profile(self) -> None:
        if self.profiler is None or not self.profile_path:
            return
        path = self.profile_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pstats.Stats(self.profiler).dump_stats(path)
        with open(path + ".txt", "w", encoding="utf-8") as fh:
            fh.write(
                f"steps {self.profile_steps[0]}..{self.profile_steps[1] - 1}, pid {os.getpid()}\n"
            )
            text = pstats.Stats(self.profiler, stream=fh)
            text.sort_stats("cumulative").print_stats(60)
            text.sort_stats("tottime").print_stats(40)

    _row_cache_drift_seen = 0

    def _row_slot_cache(self) -> RowSlotCache | None:
        """The row cache when the attachment is on an ordinary path: speculative decoding
        rewrites a row's slots below its length, PDMUX, the separate SWA pools and beam search
        have their own row contracts (the audited walk covers every step there anyway), so
        those read every row at every snapshot. A hybrid model's KV rows follow the ordinary
        contract (its recurrent state has its own pool and snapshot), so the cache applies."""
        if not getattr(self, "row_cache_enabled", False):
            return None
        if (
            getattr(self, "speculative_kv", False)
            or getattr(self, "speculative_ngram", False)
            or getattr(self, "pdmux", False)
            or getattr(self, "separate_swa", False)
            or getattr(self, "_beam_shared_slots", None)
        ):
            self.row_cache = None
            return None
        if getattr(self, "row_cache", None) is None:
            self.row_cache = RowSlotCache()
        return self.row_cache

    def _rows(self, batch: Any):
        """``RowFacts`` for every live request or streaming session holding a row.

        A streaming session owns its ``SessionSlot.kv`` record between turns, so a stable session
        identity is used both while a turn is active and while only the slot remains; treating
        successive ``Req`` objects as owners would make the legal hand-off look like an unmediated
        release and allocation.
        """
        sources = held_requests(self.scheduler, batch, queued=self.queued_rows)
        new_tokens: dict[Any, int] = {}
        mode = batch.forward_mode
        if mode.is_extend():
            lens = getattr(batch, "extend_lens", None) or [r.extend_len for r in batch.reqs]
            decoding = {id(req) for req in (getattr(batch, "decoding_reqs", None) or ())}
            speculative = getattr(self, "speculative_ngram", False) or getattr(
                self,
                "speculative_kv",
                getattr(self, "speculative_eagle3", False),
            )
            for req, n in zip(batch.reqs, lens, strict=False):
                # A speculative decode tail riding an EXTEND/MIXED batch exposes a one-token extend
                # whose ``kv_committed_len`` is still the pre-step frontier; ``decoding_reqs`` is
                # the authoritative distinction.
                new_tokens[req.rid] = 0 if speculative and id(req) in decoding else int(n)
        elif mode.is_decode() and not (
            getattr(self, "speculative_ngram", False)
            or getattr(
                self,
                "speculative_kv",
                getattr(self, "speculative_eagle3", False),
            )
        ):
            for req in batch.reqs:
                new_tokens[req.rid] = 1
        # PDMUX prepares and allocates the next split-prefill batch before it launches the
        # current decode batch. Those rows belong in this snapshot, but their newly allocated
        # suffix has not run yet. Carry the pending batch's own extend widths so
        # ``pre_step_computed`` stops at its authoritative prefix.
        pending_split = getattr(self.scheduler, "split_prefill_batch", None)
        if (
            pending_split is not None
            and pending_split is not batch
            and id(pending_split) not in self._pdmux_submitted_batches
        ):
            split_lens = getattr(pending_split, "extend_lens", None) or [
                req.extend_len for req in pending_split.reqs
            ]
            for req, n in zip(pending_split.reqs, split_lens, strict=False):
                new_tokens[req.rid] = int(n)
        seen: dict[int, tuple[Any, str]] = {}
        reqs = []
        owners: list[str] = []
        pd_decode_pending: set[int] = set()
        queued_only = 0
        for req, source in sources:
            kv = req.kv
            if not kv.holds_kv:
                continue
            idx = int(kv.req_pool_idx)
            owner = self._request_owner(req)
            prior = seen.get(idx)
            if prior is not None:
                other, other_owner = prior
                if other is not req and other_owner != owner:
                    self.reporter.emit(
                        integration_error(
                            "row_shared_by_requests",
                            f"req_to_token row {idx} held by distinct request objects "
                            f"{other.rid!r} and {req.rid!r}",
                            pool=POOL,
                            site=f"snapshot:{self.seq}",
                        )
                    )
                continue
            seen[idx] = (req, owner)
            reqs.append(req)
            owners.append(owner)
            if source in {
                "pd_decode_queue",
                "pd_decode_pending_reqs",
                "pd_decode_transfer",
                "pd_decode_deferred",
            }:
                pd_decode_pending.add(id(req))
            if source == "queued":
                # Held only through a queued result: finishing by length, released when that
                # result is processed (held_requests).
                queued_only += 1

        # Between streaming turns no live Req appears in any batch, but the cache's session slot
        # still owns the row, its private output KV and (for hybrid models) recurrent state.
        session_rows = 0
        session_cache = getattr(self.tree, "session", None)
        for session_id, slot in (getattr(session_cache, "slots", None) or {}).items():
            kv = slot.kv
            if not kv.holds_kv:
                continue
            idx = int(kv.req_pool_idx)
            owner = f"session:{session_id}"
            prior = seen.get(idx)
            if prior is not None:
                if prior[1] != owner:
                    self.reporter.emit(
                        integration_error(
                            "row_shared_by_session",
                            f"req_to_token row {idx} held by {prior[1]!r} and session {session_id!r}",
                            pool=POOL,
                            site=f"snapshot:{self.seq}",
                        )
                    )
                continue
            pseudo = types.SimpleNamespace(rid=owner, kv=kv)
            seen[idx] = (pseudo, owner)
            reqs.append(pseudo)
            owners.append(owner)
            session_rows += 1
        self.stats["snapshot_rows_queued_only"] += queued_only
        self.stats["snapshot_rows_session_only"] += session_rows
        self._queued_only = queued_only
        separate_swa = getattr(self, "separate_swa", False)
        swa_window = max(int(getattr(self, "facts", {}).get("sliding_window_size") or 0), 0)
        rows = [
            dataclasses.replace(
                row,
                request_id=owner,
                # The shared window planner reads from max(committed - window, 0). Speculative
                # decode reserves whole future pages in kv_allocated_len while leaving
                # kv_committed_len at the verify prefix, so the committed frontier (not that
                # page-rounded capacity) defines the contiguous live SWA suffix.
                swa_first_token=(
                    max(row.kv_committed_len - row.new_tokens - swa_window, 0)
                    if separate_swa
                    else 0
                ),
            )
            for owner, row in zip(
                owners,
                rows_from_requests(
                    reqs,
                    self.req_pool.req_to_token,
                    new_tokens,
                    cache=self._row_slot_cache(),
                    audit=self.seq % getattr(self, "audit_every", 64) == 0,
                    owners=owners,
                ),
                strict=True,
            )
        ]
        cache = getattr(self, "row_cache", None)
        if cache is not None and cache.drift != self._row_cache_drift_seen:
            self._row_cache_drift_seen = cache.drift
            self.reporter.emit(
                integration_error(
                    "row_slots_drift",
                    "the audit re-read a row whose kept slots differ from the device table",
                    pool=POOL,
                    site=f"snapshot:{self.seq}",
                )
            )
        if separate_swa and rows:
            self.stats["swa_snapshot_first_token_max"] = max(
                self.stats["swa_snapshot_first_token_max"],
                max(row.swa_first_token for row in rows),
            )
        # Decode preallocation sets ``kv_committed_len`` to the final prompt length before
        # Mooncake has written the destination.  While the request remains in a receiver queue,
        # only its already-resident L1 prefix is initialized; the external-write completion
        # advances the rest before the request moves to ``waiting_queue``.
        rows = [
            (
                dataclasses.replace(
                    row,
                    kv_committed_len=int(getattr(req, "prefix_len", 0) or 0),
                    new_tokens=0,
                )
                if id(req) in pd_decode_pending
                else row
            )
            for req, row in zip(reqs, rows, strict=True)
        ]
        rid_of_row = {
            int(req.kv.req_pool_idx): row.request_id for req, row in zip(reqs, rows, strict=True)
        }

        # Beam search keeps only the leader as a Req; the other physical attention rows live
        # columnarly on BeamGroup. Add those rows as stable virtual requests whose already-computed
        # histories are shared immutable state, while this step's per-row destinations stay private.
        self._beam_shared_slots = set()
        self._beam_reserved_slots = set()
        groups_seen: set[int] = set()
        member_count = 0
        leader_rows = tuple(rows)
        for req, owner, leader_row in zip(reqs, owners, leader_rows, strict=True):
            group = getattr(req, "beam_group", None)
            if (
                group is None
                or id(group) in groups_seen
                or getattr(group, "member_rows_cpu", None) is None
                or getattr(group, "retired", False)
            ):
                continue
            groups_seen.add(id(group))
            if not self.facts.get("beam_search_port_enabled"):
                if not self._beam_unsupported_emitted:
                    self.reporter.emit(
                        unsupported(
                            "beam_search",
                            f"beam member rows require {ENV_BEAM}=1 for the explicit adapter port",
                            pool=POOL,
                            site=f"snapshot:{self.seq}",
                        )
                    )
                    self._beam_unsupported_emitted = True
                continue
            members, member_ids, shared = beam_member_rows(
                req,
                owner,
                self.req_pool.req_to_token,
                new_tokens=leader_row.new_tokens,
            )
            rows.extend(members)
            rid_of_row.update(member_ids)
            self._beam_shared_slots.update(shared)
            self._beam_reserved_slots.update(beam_pending_orphan_slots(req))
            member_count += len(members)
        if member_count:
            self.stats["beam_member_row_snapshots"] += member_count
            self.stats["beam_group_snapshots"] += len(groups_seen)
            self.stats["beam_shared_slot_snapshots"] += len(self._beam_shared_slots)
            self.stats["beam_reserved_slot_snapshots"] += len(self._beam_reserved_slots)
        if getattr(self, "speculative_ngram", False) or getattr(
            self,
            "speculative_kv",
            getattr(self, "speculative_eagle3", False),
        ):
            current_rows = (
                # A decode batch under NGRAM/overlap can still carry a req whose row was
                # released (req_pool_idx is None); guard it like pending_rows below.
                {int(req.kv.req_pool_idx) for req in batch.reqs if req.kv.holds_kv}
                if mode.is_decode()
                else set()
            )
            pending_rows = {
                int(req.kv.req_pool_idx)
                for entry in (getattr(self.scheduler, "result_queue", None) or ())
                for queued_batch in (entry[0] if isinstance(entry, tuple) else entry,)
                for req in getattr(queued_batch, "reqs", ())
                if req.kv.holds_kv
            }

            def keep_high_water(pool: str) -> list[RowFacts]:
                kept = []
                for req, row in zip(reqs, rows, strict=True):
                    shadow = self.ledger.requests.get((row.request_id, pool))
                    high_water = shadow.computed_high_water if shadow is not None else 0
                    committed = (
                        max(row.kv_committed_len, high_water)
                        if int(req.kv.req_pool_idx) in current_rows | pending_rows
                        else row.kv_committed_len
                    )
                    kept.append(
                        dataclasses.replace(
                            row,
                            kv_committed_len=committed,
                            # Only the speculative decode rows have a candidate
                            # suffix whose high-water mark replaces this step's
                            # ordinary new-token credit. PDMUX may already hold a
                            # separate split-prefill batch in the same snapshot;
                            # its allocated suffix is still pending real writes.
                            new_tokens=(
                                0
                                if mode.is_decode() and int(req.kv.req_pool_idx) in current_rows
                                else row.new_tokens
                            ),
                        )
                    )
                return kept

            if getattr(self, "speculative_eagle3", False):
                self._draft_rows = keep_high_water(DRAFT_POOL)
            elif getattr(self, "speculative_multi_layer_eagle", False):
                self._draft_rows_by_pool = {
                    pool: keep_high_water(pool) for pool in self.draft_pool_ids
                }
            rows = keep_high_water(POOL)
        return rows, rid_of_row

    def _request_owner(self, req: Any) -> str:
        """Stable owner shared by an active streaming turn and its retained session slot."""
        session = getattr(req, "session", None)
        if session is not None and getattr(session, "streaming", False):
            return f"session:{session.session_id}"
        return self.request_incarnations(req)

    def _before_run_batch(self, batch: Any) -> None:
        t0 = time.perf_counter()
        self._reap_hicache()
        self._reap_pd_reads()
        pdmux_split_open = False
        if self.pdmux:
            live_split = getattr(self.scheduler, "split_prefill_batch", None)
            self._pdmux_submitted_batches.intersection_update(
                {id(live_split)} if live_split is not None else set()
            )
            pdmux_split_open = (
                live_split is not None and id(live_split) in self._pdmux_submitted_batches
            )
            if pdmux_split_open and batch.forward_mode.is_decode_or_idle():
                self.stats["pdmux_decode_while_split_open"] += 1
        self.seq += 1
        self.events += 1
        self.cur_batch = batch
        self._draft_tree_copy_rolled_back.clear()
        tree_count = unlocked_count = 0
        mode = batch.forward_mode
        self.stats[f"steps_{mode.name.lower()}"] += 1
        rows, rid_of_row = self._rows(batch)
        if self.open:
            held = {r.request_id for r in rows}
            for seq in sorted(self.open):
                consumed = {
                    rid
                    for did in self.open[seq].dispatch_ids
                    if did in self.ledger.epochs
                    for rid, _, _ in self.ledger.epochs[did].consumed
                }
                if consumed - held:
                    if self.early_complete:
                        self._complete_through(seq, "row_released")
                    else:
                        self.stats["row_released_while_open"] += 1
        try:
            free_allocator = (
                self.allocator.full_attn_allocator
                if self.separate_swa or self.facts.get("unified_allocator")
                else self.allocator
            )
            # The recurrent-state pool does not force the walk: its snapshot reads its own
            # allocator and the tree's state values, and the KV pool's view between walks is the
            # ledger's masks as on a dense model.
            audited = (
                self.seq % self.audit_every == 0 or self.speculative_eagle3 or self.separate_swa
            )
            if audited:
                free = free_slots_from_allocator(free_allocator)
                free_count = None
            else:
                free = []
                free_count = free_count_from_allocator(free_allocator)
                self.stats["snapshot_free_counted"] += 1
            strategy = self.facts["tree_strategy"]
            pd_reserved_slots = {
                block * self.page_size
                for blocks in self._pd_reserved_blocks.values()
                for block in blocks
            }
            with self._external_linker_lock:
                external_store_reserved_slots = {
                    block * self.page_size
                    for job in self._external_linker_store_jobs
                    for block in job["pools"].get(POOL, ())
                }
            # The unified-linker wrapper pins every source node from submission until
            # ``pop_completed_offload`` commits its result.  A pinned node can leave the normal
            # Rust tree walk during eviction/cancellation, but its pages are still unavailable
            # for reuse and remain the source of the open external-read epoch.
            reserved_slots = (
                self._beam_reserved_slots | pd_reserved_slots | external_store_reserved_slots
            )
            self.stats["pd_reserved_block_snapshots"] += len(pd_reserved_slots)
            self.stats["external_linker_store_reserved_block_snapshots"] += len(
                external_store_reserved_slots
            )
            if strategy == "rust_walk":
                core = self.tree.tree_core
                # The tree's pages between walks: every non-free page no running row holds
                # privately is the tree's, locked exactly when a running row's protected prefix
                # runs through it. The full walk runs on the audited steps only and checks this
                # view (its drift is an integration error, as a partition difference is).
                ps = self.page_size
                geometry = self.ledger.pools[POOL]
                private_pages: set[int] = set()
                protected_pages: list[int] = []
                for row in rows:
                    private_pages.update(
                        row_pages(row.slots, row.cache_protected_len, row.kv_allocated_len, ps)
                    )
                    protected_pages.extend(row_pages(row.slots, 0, row.cache_protected_len, ps))
                # The view as boolean arrays over the page ids: the ledger's non-free mask less
                # the rows' own and reserved pages is the tree, less the protected pages its
                # unlocked part. Array operations keep the step from growing with the tree.
                published_mask = np.frombuffer(
                    self.ledger.nonfree_mask[POOL], dtype=np.bool_
                ).copy()
                if private_pages:
                    published_mask[list(private_pages)] = False
                if geometry.reserved_blocks:
                    published_mask[list(geometry.reserved_blocks)] = False
                unlocked_mask = published_mask.copy()
                if protected_pages:
                    unlocked_mask[protected_pages] = False
                tree_pages: set[int] | None = None
                unlocked_pages: set[int] | None = None
                tree_slots: list[int] | None = None  # the walk's slot lists, for the draft view
                unlocked: list[int] | None = None
                duplicates: list[int] = []
                if audited:
                    walked = _np_ints(core.walk_for_kv_canary(False, False).slot_indices)
                    walked_free = _np_ints(core.walk_for_kv_canary(True, False).slot_indices)
                    walked_published, duplicates = page_set_np(walked, ps, duplicates=True)
                    walked_unlocked = page_set_np(walked_free, ps)
                    if self.speculative_eagle3:
                        tree_slots, unlocked = walked.tolist(), walked_free.tolist()
                    derived_published = set(np.flatnonzero(published_mask).tolist())
                    derived_unlocked = set(np.flatnonzero(unlocked_mask).tolist())
                    if self.seq > 0 and (
                        walked_published != derived_published or walked_unlocked != derived_unlocked
                    ):
                        extra = sorted(derived_published - walked_published)[:5]
                        missing = sorted(walked_published - derived_published)[:5]
                        self.stats["tree_view_drift"] += 1
                        self.reporter.emit(
                            integration_error(
                                "tree_view_drift",
                                f"the tree view between walks differs from the walk: "
                                f"{len(derived_published - walked_published)} page(s) not in the tree, "
                                f"e.g. {extra}; {len(walked_published - derived_published)} tree page(s) "
                                f"not in the view, e.g. {missing}; unlocked "
                                f"{len(derived_unlocked ^ walked_unlocked)} differ",
                                pool=POOL,
                                site=f"snapshot:{self.seq}",
                            )
                        )
                    # The walk's page sets are the listing, in full, at the audited step.
                    tree_pages, unlocked_pages = walked_published, walked_unlocked
                    tree_count, unlocked_count = int(walked.size), int(walked_free.size)
                else:
                    # Between walks the view is the masks, and the pool is listed as its
                    # change: the kept per-page objects make the ledger's work the size of
                    # the change (`Snapshot.delta_pools`).
                    self.stats["snapshot_tree_derived"] += 1
                    tree_count = int(np.count_nonzero(published_mask)) * ps
                    unlocked_count = int(np.count_nonzero(unlocked_mask)) * ps

                def pin_tree(slots: set[int], stat: str) -> None:
                    # A pinned node is unlocked for the ledger (reserved: no request owns it)
                    # while the transfer holds it; the pins that are in the tree only.
                    pinned = page_set(slots, ps)
                    if tree_pages is not None:
                        pinned &= tree_pages
                        unlocked_pages.update(pinned)
                    else:
                        pinned = {p for p in pinned if published_mask[p]}
                        if pinned:
                            unlocked_mask[list(pinned)] = True
                    self.stats[stat] += len(pinned)

                if self._hicache_store_reservations:
                    # HiCache pins a node while its D2H reads the device slots. It is no longer
                    # mapped by a request, but the transfer owns the node until writing_check
                    # consumes its acknowledgement.  The ExternalRead epoch itself can retire
                    # earlier, at the exact CUDA completion event.
                    held_by_store = {
                        slot
                        for slots in self._hicache_store_reservations.values()
                        for slot in slots
                    }
                    pin_tree(held_by_store, "hicache_store_reserved_slot_snapshots")
                if self.hicache:
                    pin_tree(
                        self._hicache_internal_pin_slots(), "hicache_internal_pin_slot_snapshots"
                    )
                if self.external_linker:
                    pin_tree(
                        self._external_linker_internal_pin_slots(),
                        "external_linker_internal_pin_slot_snapshots",
                    )
                snap, _ = snapshot_from_walk(
                    self.seq,
                    POOL,
                    rows,
                    None,
                    None,
                    free,
                    self.walk_states,
                    shared_slots=self._beam_shared_slots,
                    reserved_slots=reserved_slots,
                    tree_pages=tree_pages,
                    unlocked_pages=unlocked_pages,
                    own_pages=private_pages,
                    delta=not audited,
                    tree_mask=None if audited else published_mask,
                    unlocked_mask=None if audited else unlocked_mask,
                )
                if not audited:
                    self.stats["snapshot_delta"] += 1
                    self.stats["snapshot_delta_blocks"] += len(snap.blocks)
                    self.stats["snapshot_delta_gone"] += len(self.walk_states.gone)
                if duplicates:
                    self.reporter.emit(
                        integration_error(
                            "tree_slot_held_twice",
                            f"the tree walk lists {len(duplicates)} slot(s) twice, e.g. {duplicates[:5]}",
                            pool=POOL,
                            site=f"snapshot:{self.seq}",
                        )
                    )
                nodes = []
                self.stats["snapshot_tree_slots"] += tree_count
                self.stats["snapshot_unlocked_slots"] += unlocked_count
            else:
                if strategy == "python_nodes":
                    nodes = nodes_from_unified_tree(self.tree.tree_core.root_node)
                elif strategy == "radix_nodes":
                    nodes = nodes_from_tree(self.tree.root_node)
                else:
                    nodes = []
                snap = build_snapshot(
                    self.seq,
                    POOL,
                    rows,
                    nodes,
                    free,
                    self.page_size,
                    self._beam_shared_slots,
                    reserved_slots,
                )
                tree_count = sum(len(nd.slots) for nd in nodes)
                unlocked_count = sum(len(nd.slots) for nd in nodes if nd.lock_ref <= 0)
            if self.speculative_eagle3:
                if strategy == "rust_walk":
                    draft_snap, draft_duplicates = snapshot_from_walk(
                        self.seq,
                        DRAFT_POOL,
                        self._draft_rows,
                        tree_slots,
                        unlocked,
                        free,
                        self.draft_walk_states,
                    )
                    if draft_duplicates != duplicates:
                        raise ValueError(
                            "target and draft views of the same radix walk disagree on duplicates"
                        )
                else:
                    draft_snap = build_snapshot(
                        self.seq,
                        DRAFT_POOL,
                        self._draft_rows,
                        nodes,
                        free,
                        self.page_size,
                    )
                merged_free = dict(snap.free or {})
                merged_free[DRAFT_POOL] = frozenset(free)
                snap = dataclasses.replace(
                    snap,
                    requests=snap.requests + draft_snap.requests,
                    blocks=snap.blocks + draft_snap.blocks,
                    free=merged_free,
                )
                self.stats["draft_snapshot_requests"] += len(draft_snap.requests)
                self.stats["draft_snapshot_blocks"] += len(draft_snap.blocks)
            elif self.separate_swa:
                if strategy != "rust_walk":
                    raise ValueError("the independent full/SWA port requires the Rust tree walk")
                core = self.tree.tree_core
                swa_tree_full = _ints(core.walk_for_kv_canary(False, True).slot_indices)
                swa_unlocked_full = _ints(core.walk_for_kv_canary(True, True).slot_indices)
                mapping = _ints(self.allocator.full_to_swa_index_mapping[:-1])
                swa_free = free_slots_from_allocator(self.allocator.swa_attn_allocator)
                views = [(SWA_POOL, rows)] + [
                    (pool, self._draft_rows_by_pool.get(pool, rows)) for pool in self.draft_pool_ids
                ]
                extra_requests = []
                extra_blocks = []
                merged_free = dict(snap.free or {})
                for pool, pool_rows in views:
                    pool_snap, _ = swa_snapshot_from_walk(
                        self.seq,
                        pool,
                        pool_rows,
                        mapping,
                        swa_tree_full,
                        swa_unlocked_full,
                        swa_free,
                        self.page_size,
                    )
                    extra_requests.extend(pool_snap.requests)
                    extra_blocks.extend(pool_snap.blocks)
                    merged_free[pool] = frozenset(swa_free)
                    self.stats[f"{pool}_snapshot_requests"] += len(pool_snap.requests)
                    self.stats[f"{pool}_snapshot_blocks"] += len(pool_snap.blocks)
                snap = dataclasses.replace(
                    snap,
                    requests=snap.requests + tuple(extra_requests),
                    blocks=snap.blocks + tuple(extra_blocks),
                    free=merged_free,
                )
                self.stats["swa_tree_slots"] += len(swa_tree_full)
                self.stats["swa_unlocked_slots"] += len(swa_unlocked_full)
        except ValueError as exc:
            self.reporter.emit(
                integration_error(
                    "adapter_translation", str(exc), pool=POOL, site=f"snapshot:{self.seq}"
                )
            )
            self.stats["snapshots_failed"] += 1
            self.rid_of_row = rid_of_row
            return
        if free_count is not None:
            snap = dataclasses.replace(snap, free=None, free_count={POOL: free_count})
        if self.state is not None:
            try:
                s_requests, s_blocks, s_free = self.state.snapshot(rid_of_row)
            except Exception as exc:
                self.stats["state_snapshots_failed"] += 1
                self.reporter.emit(
                    integration_error(
                        "state_snapshot",
                        f"{type(exc).__name__}: {exc}"[:300],
                        pool=STATE_POOL,
                        site=f"snapshot:{self.seq}",
                    )
                )
            else:
                free = dict(snap.free or {})
                free[STATE_POOL] = s_free
                if self.state.ckpt is not None:
                    free[CKPT_POOL] = self.state.ckpt_free
                snap = dataclasses.replace(
                    snap,
                    requests=snap.requests + tuple(s_requests),
                    blocks=snap.blocks + tuple(s_blocks),
                    free=free,
                )
                self.stats["state_snapshot_requests"] += len(s_requests)
                self.stats["state_snapshot_blocks"] += len(s_blocks)
        if self.stream_epochs:
            snap = dataclasses.replace(snap, stream=self._stream(), after=self._fenced_epochs())
        self.ledger.observe(snap)
        self.rid_of_row = rid_of_row
        self.stats["snapshots"] += 1
        self.stats["snapshot_rows"] += len(rows)
        self.stats["snapshot_nodes"] += len(nodes)
        self.stats["snapshot_blocks"] += len(snap.blocks)
        self.stats["snapshot_free_slots"] += len(free)
        self.timings["snapshot_s"] += time.perf_counter() - t0
        if len(self.trace) < TRACE_LIMIT:
            trace = {
                "seq": self.seq,
                "mode": mode.name,
                "batch": len(batch.reqs),
                "rows": len(rows),
                "tree": tree_count,
                "unlocked": unlocked_count,
                "free": len(free),
                "queued": self._queued_only,
                "reports": len(self.reporter.reports),
            }
            if self.separate_swa:
                trace["swa_rows"] = [
                    {
                        "allocated": int(row.kv_allocated_len),
                        "committed": int(row.kv_committed_len),
                        "protected": int(row.cache_protected_len),
                        "new": int(row.new_tokens),
                        "first": int(row.swa_first_token),
                    }
                    for row in rows
                ]
            if self.pdmux:
                trace["pdmux_split_open"] = pdmux_split_open
            self.trace.append(trace)
        self.dirty = True
        if self.fault is not None:
            self.fault.before_forward(self, batch)

    # ------------------------------------------------------------------ dispatches
    def _after_forward(self, fb: Any, out: Any) -> None:
        import torch

        t0 = time.perf_counter()
        handled = self.seq in self._prestep
        waited = self.sync_after_forward or not handled
        if waited:
            torch.cuda.current_stream().synchronize()
        else:
            self.stats["forward_not_waited"] += 1
        t1 = time.perf_counter()
        self.timings["forward_wait_s"] += t1 - t0
        if waited:
            # A HiCache consumer waits for each restored layer before that layer's kernels.
            # Host-only validation also synchronizes here before it constructs descriptors.
            # In either case every transfer completion event must now be visible, so apply the
            # fills before validating the accesses that followed the waits. Slower direct/page
            # transfers can still be incomplete at the scheduler's pre-forward snapshot.
            self._reap_hicache()
        self.stats["forward_calls"] += 1
        site = f"forward@{self.seq}"
        dids = self._prestep.pop(self.seq, None)
        if dids is not None:
            self.stats["dispatch_pre_kernel"] += 1
        else:
            if self.gpu is not None:
                # The pre-kernel point did not fire for this forward: host-only, after the fact.
                self.stats["dispatch_post_forward_fallback"] += 1
            try:
                dids = self._descriptors(fb, out)
            except Exception as exc:
                log.exception("descriptor extraction failed at step %d", self.seq)
                self.reporter.emit(
                    integration_error(
                        "adapter_extraction", f"{type(exc).__name__}: {exc}", pool=POOL, site=site
                    )
                )
                dids = []
        if self.twins is not None:
            self.twins.readback(self.seq, self._last_dispatch.get("indptr_last") if dids else None)
        gpu_summary = self.gpu.readback(self.seq) if self.gpu is not None else None
        if self.gpu is None and getattr(self, "tags", None) is not None:
            self.tags.readback(self.reporter)
        if self.pdmux and fb.forward_mode.is_split_prefill() and dids:
            self._pdmux_submitted_batches.add(id(self.cur_batch))
            self.stats["pdmux_split_submitted"] += 1
        event = torch.cuda.Event()
        event.record()
        epoch = self.open.get(self.seq)
        if epoch is None:
            self.open[self.seq] = StepEpoch(
                self.seq, dids, event, getattr(self.cur_batch, "forward_iter", None)
            )
        else:
            epoch.dispatch_ids.extend(dids)
            epoch.event = event
            self.stats["forward_calls_extra"] += 1
        self.timings["after_forward_s"] += time.perf_counter() - t1
        self.events += 1
        if self.trace and self.trace[-1]["seq"] == self.seq:
            self.trace[-1].update(self._last_dispatch)
            if gpu_summary is not None:
                self.trace[-1].update(gpu_summary)
            self.trace[-1]["reports"] = len(self.reporter.reports)
        self.dirty = True

    def _after_draft_forward(self, fb: Any, backend: Any, *, pool: str = DRAFT_POOL) -> None:
        """Dispatch the independent draft pool after one eager EAGLE3 forward."""
        import torch

        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        self.timings["draft_forward_wait_s"] += time.perf_counter() - t0
        self.stats["draft_forward_calls"] += 1
        self._draft_forward_index += 1
        site = f"draft-forward@{self.seq}:{self._draft_forward_index}"
        dids: list[str] = []
        try:
            descs, kind, tokens = self._draft_descriptors(
                fb,
                backend,
                tag=f"-draft{self._draft_forward_index}",
                pool=pool,
                separate_swa=self.speculative_multi_layer_eagle,
            )
            for descriptor in descs:
                self._dispatch_one(descriptor, dids)
            self.stats["dispatches"] += len(dids)
            self.stats["draft_dispatches"] += len(dids)
            self.stats[f"draft_forwards_{kind}"] += 1
            self.stats["draft_tokens"] += tokens
        except Exception as exc:
            log.exception("draft descriptor extraction failed at step %d", self.seq)
            self.reporter.emit(
                integration_error(
                    "draft_adapter_extraction",
                    f"{type(exc).__name__}: {exc}"[:300],
                    pool=pool,
                    site=site,
                )
            )
            self.stats["draft_dispatch_failed"] += 1
        event = torch.cuda.Event()
        event.record()
        epoch = self.open.get(self.seq)
        if epoch is None:
            self.open[self.seq] = StepEpoch(
                self.seq, dids, event, getattr(self.cur_batch, "forward_iter", None)
            )
        else:
            epoch.dispatch_ids.extend(dids)
            epoch.event = event
            self.stats["forward_calls_extra"] += 1
        self.timings["after_draft_forward_s"] += time.perf_counter() - t0
        self.events += 1
        self.dirty = True

    def _draft_descriptors(
        self,
        fb: Any,
        backend: Any,
        *,
        tag: str,
        pool: str = DRAFT_POOL,
        separate_swa: bool = False,
    ) -> tuple[tuple[DispatchDescriptor, ...], str, int]:
        """Translate one eager EAGLE3 draft forward from the FlashInfer buffers it consumed."""
        mode = fb.forward_mode
        n = int(fb.batch_size)
        req_ids = self._req_ids(fb, n)
        metadata = backend.forward_metadata
        if getattr(mode, "is_target_verify", lambda: False)() and getattr(
            self, "speculative_dflash", False
        ):
            graph_bs = getattr(fb, "_kvguard_graph_bs", None)
            wrappers = list(
                backend.prefill_cuda_graph_metadata[int(graph_bs)]
                if graph_bs is not None
                else getattr(metadata, "prefill_wrappers", None) or ()
            )
            if len(wrappers) != 2:
                raise RuntimeError(f"DFlash draft verify exposed {len(wrappers)} wrappers")
            width = int(self.facts["dflash_block_size"])
            n_pad = int(graph_bs) if graph_bs is not None else n
            positions = _ints(fb.positions)[: n * width]
            prefixes = (
                [positions[row * width] for row in range(n)]
                if self.facts.get("dflash_compact_cache")
                else _ints(fb.seq_lens)[:n]
            )
            expected_positions = [prefix + offset for prefix in prefixes for offset in range(width)]
            if len(prefixes) != n or positions != expected_positions:
                raise ValueError(
                    f"DFlash draft positions {positions[:16]} do not match "
                    f"prefixes {prefixes[:8]} at width {width}"
                )
            for rid, frontier in zip(req_ids, prefixes, strict=True):
                self.ledger.rollback(
                    pool,
                    rid,
                    frontier,
                    site=f"dflash-draft-rollback@{self.seq}",
                )
            loc_source = fb.out_cache_loc
            if graph_bs is not None:
                graph = self.draft_runner.decode_cuda_graph_runner
                loc_source = graph.buffer_registry.get_slot("out_cache_loc").slice_for(
                    n_pad, n_pad * width
                )
            loc = self._logical_kv_slots(_ints(loc_source)[: n_pad * width])
            if len(loc) != n_pad * width:
                raise ValueError(f"DFlash draft has {len(loc)} write slots for {n_pad}x{width}")
            descriptors: list[DispatchDescriptor] = []
            expected_lens = [prefix + width for prefix in prefixes]
            table_expected_lens = (
                [length + width for length in _ints(fb.seq_lens)[:n]]
                if self.facts.get("dflash_compact_cache")
                else expected_lens
            )
            for wrapper_index, wrapper in enumerate(wrappers):
                indptr = _ints(wrapper._paged_kv_indptr_buf)[: n_pad + 1]
                if len(indptr) != n_pad + 1:
                    raise ValueError(
                        f"DFlash draft wrapper {wrapper_index} indptr has "
                        f"{len(indptr)} entries for {n_pad} rows"
                    )
                table_lens = [indptr[i + 1] - indptr[i] for i in range(n)]
                read_starts = [max(expected_lens[i] - table_lens[i], 0) for i in range(n)]
                if any(
                    table_lens[i] > table_expected_lens[i]
                    or (wrapper_index == 1 and table_lens[i] != table_expected_lens[i])
                    for i in range(n)
                ):
                    raise ValueError(
                        f"DFlash draft wrapper {wrapper_index} table lengths "
                        f"{table_lens[:8]} exceed expected {table_expected_lens[:8]}"
                    )
                indices = self._logical_kv_slots(
                    _ints(wrapper._paged_kv_indices_buf[: max(int(indptr[-1]), 0)])
                )
                lpl = _ints(wrapper._paged_kv_last_page_len_buf)[:n_pad]
                if lpl != [1] * n_pad:
                    raise ValueError(
                        f"DFlash draft wrapper {wrapper_index} last_page_len={lpl[:8]}"
                    )
                write, attention = build_verify_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    prefixes,
                    indptr,
                    indices,
                    loc,
                    width,
                    num_rows_after_padding=n_pad,
                    last_page_len=lpl,
                    page_size=self.page_size,
                    read_starts=read_starts,
                    tag=f"{tag}-{'swa' if wrapper_index == 0 else 'full'}",
                )
                if wrapper_index == 0:
                    descriptors.append(write)
                descriptors.append(attention)
            return (
                tuple(descriptors),
                "dflash_verify",
                n * width,
            )
        if mode.is_decode():
            graph_bs = getattr(fb, "_kvguard_graph_bs", None)
            wrappers = list(
                backend.decode_cuda_graph_metadata[graph_bs]
                if graph_bs is not None
                else getattr(metadata, "decode_wrappers", None) or backend.decode_wrappers
            )
            if len(wrappers) != (2 if separate_swa else 1):
                raise RuntimeError(f"draft decode exposed {len(wrappers)} wrappers")
            wrapper = wrappers[0]
            topk = int(self.facts.get("speculative_eagle_topk", 1))
            if topk > 1:
                padded_requests = int(graph_bs) if graph_bs is not None else n
                positions = _ints(fb.positions)[: n * topk]
                prefixes = _ints(fb.seq_lens)[:n]
                if len(positions) != n * topk or len(prefixes) != n:
                    raise ValueError(
                        f"tree draft decode has positions={len(positions)}, "
                        f"prefixes={len(prefixes)}, requests={n}, topk={topk}"
                    )
                step = int(
                    getattr(
                        fb,
                        "_kvguard_draft_step",
                        positions[0] - prefixes[0] if positions and prefixes else -1,
                    )
                )
                expected_positions = [
                    prefixes[request] + step for request in range(n) for _branch in range(topk)
                ]
                if positions != expected_positions:
                    raise ValueError(
                        f"tree draft positions {positions[:8]} != expected "
                        f"{expected_positions[:8]} at step {step}"
                    )
                if step == 0:
                    # Draft-extend filled the preceding iteration's whole candidate window.
                    # Reconcile its accepted dense prefix once before this iteration's sparse
                    # branch writes; later inner steps must retain earlier branch slots.
                    for rid, frontier in zip(req_ids, prefixes, strict=True):
                        if rid not in self._draft_tree_copy_rolled_back:
                            self.ledger.rollback(
                                pool,
                                rid,
                                frontier,
                                site=f"draft-tree-rollback@{self.seq}",
                            )
                total_rows = padded_requests * topk
                indptr = _ints(wrapper._paged_kv_indptr_buf)[: total_rows + 1]
                if len(indptr) != total_rows + 1:
                    raise ValueError(
                        f"tree draft indptr has {len(indptr)} entries for {total_rows} rows"
                    )
                indices = self._logical_kv_slots(
                    _ints(wrapper._paged_kv_indices_buf[: max(int(indptr[-1]), 0)])
                )
                loc = self._logical_kv_slots(_ints(fb.out_cache_loc)[: n * topk])
                lpl = _ints(wrapper._paged_kv_last_page_len_buf)[:total_rows]
                if len(loc) != n * topk or lpl != [1] * total_rows:
                    raise ValueError(
                        f"tree draft shape loc={len(loc)}, last_page_len={lpl[:8]}, "
                        f"rows={total_rows}"
                    )
                return (
                    build_eagle_tree_draft_decode_dispatches(
                        self.seq,
                        pool,
                        req_ids,
                        prefixes,
                        topk,
                        int(self.facts["speculative_num_steps"]),
                        step,
                        indptr,
                        indices,
                        loc,
                        num_requests_after_padding=padded_requests,
                        page_size=self.page_size,
                        tag=tag,
                    ),
                    "decode_tree",
                    n * topk,
                )
            positions = _ints(fb.positions)[:n]
            if len(positions) != n:
                raise ValueError(f"draft decode has {len(positions)} positions for {n} rows")
            seq_lens = [position + 1 for position in positions]
            for rid, frontier in zip(req_ids, positions, strict=True):
                self.ledger.rollback(pool, rid, frontier, site=f"draft-decode-rollback@{self.seq}")
            indptr = _ints(wrapper._paged_kv_indptr_buf)[: n + 1]
            if len(indptr) != n + 1:
                raise ValueError(f"draft decode indptr has {len(indptr)} entries for {n} rows")
            table_lens = [indptr[i + 1] - indptr[i] for i in range(n)]
            if table_lens != seq_lens:
                raise ValueError(
                    f"draft decode table lengths {table_lens[:8]} != logical lengths {seq_lens[:8]}"
                )
            indices = self._logical_kv_slots(
                _ints(wrapper._paged_kv_indices_buf[: max(int(indptr[-1]), 0)])
            )
            loc = self._logical_kv_slots(
                _ints(metadata.swa_out_cache_loc if separate_swa else fb.out_cache_loc)
            )
            lpl = _ints(wrapper._paged_kv_last_page_len_buf)[:n]
            if len(loc) != n or lpl != [1] * n:
                raise ValueError(
                    f"draft decode shape loc={len(loc)}, last_page_len={lpl[:8]}, rows={n}"
                )
            return (
                build_decode_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    seq_lens,
                    indptr,
                    indices,
                    loc,
                    last_page_len=lpl,
                    page_size=self.page_size,
                    tag=tag,
                ),
                "decode",
                n,
            )
        if mode.is_extend() or mode.is_draft_extend_v2():
            graph_bs = getattr(fb, "_kvguard_graph_bs", None)
            n_pad = int(graph_bs) if graph_bs is not None else n
            wrappers = list(
                backend.draft_extend_cuda_graph_metadata[graph_bs]
                if graph_bs is not None
                else getattr(metadata, "prefill_wrappers", None) or backend.prefill_wrappers
            )
            if len(wrappers) != (2 if separate_swa else 1):
                raise RuntimeError(f"draft extend exposed {len(wrappers)} wrappers")
            wrapper = wrappers[0]
            prefix = _ints(fb.extend_prefix_lens)[:n]
            extend = _ints(fb.extend_seq_lens)[:n]
            if len(prefix) != n or len(extend) != n:
                raise ValueError(
                    f"draft extend has prefix={len(prefix)}, extend={len(extend)} for {n} rows"
                )
            seq_lens = [p + e for p, e in zip(prefix, extend, strict=True)]
            for rid, frontier in zip(req_ids, prefix, strict=True):
                self.ledger.rollback(pool, rid, frontier, site=f"draft-extend-rollback@{self.seq}")
                if (
                    mode.is_draft_extend_v2()
                    and int(self.facts.get("speculative_eagle_topk", 1)) > 1
                ):
                    self.ledger.invalidate_positions(
                        pool,
                        rid,
                        eagle_tree_workspace_positions(
                            frontier,
                            int(self.facts["speculative_eagle_topk"]),
                            int(self.facts["speculative_num_steps"]),
                            self.page_size,
                        ),
                        site=f"draft-tree-invalidate@{self.seq}",
                    )
            indptr = _ints(wrapper._paged_kv_indptr_buf)[: n_pad + 1]
            if len(indptr) != n_pad + 1:
                raise ValueError(f"draft extend indptr has {len(indptr)} entries for {n_pad} rows")
            use_ragged = bool(getattr(fb, "_kvguard_use_ragged", metadata.use_ragged))
            expected = prefix if use_ragged else seq_lens
            table_lens = [indptr[i + 1] - indptr[i] for i in range(n)]
            read_starts = [max(int(expected[i]) - int(table_lens[i]), 0) for i in range(n)]
            if (not separate_swa and table_lens != expected) or any(
                table_lens[i] > expected[i] for i in range(n)
            ):
                raise ValueError(
                    f"draft extend table lengths {table_lens[:8]} != expected {expected[:8]}"
                )
            indices = self._logical_kv_slots(
                _ints(wrapper._paged_kv_indices_buf[: max(int(indptr[-1]), 0)])
            )
            if separate_swa and graph_bs is not None:
                width = int(getattr(fb, "_kvguard_graph_width", 0) or 0)
                if width <= 0:
                    raise ValueError("draft extend graph exposed no captured request width")
                loc_source = backend.cuda_graph_swa_out_cache_loc[: n_pad * width]
            else:
                loc_source = metadata.swa_out_cache_loc if separate_swa else fb.out_cache_loc
            loc = self._logical_kv_slots(_ints(loc_source))
            tokens = sum(extend)
            graph_width = int(getattr(fb, "_kvguard_graph_width", 0) or 0)
            if graph_bs is not None and graph_width <= 0 and n_pad > 0:
                graph_width = len(loc) // n_pad
            expected_locs = tokens if graph_bs is None else n_pad * graph_width
            lpl = _ints(wrapper._paged_kv_last_page_len_buf)[:n_pad]
            if len(loc) != expected_locs or lpl != [1] * n_pad:
                raise ValueError(
                    f"draft extend shape loc={len(loc)}, expected={expected_locs}, "
                    f"last_page_len={lpl[:8]}, rows={n_pad}"
                )
            kind = "draft_extend" if mode.is_draft_extend_v2() else "extend"
            return (
                build_extend_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    seq_lens,
                    prefix,
                    indptr,
                    indices,
                    loc,
                    use_ragged=use_ragged,
                    num_rows_after_padding=n_pad,
                    last_page_len=lpl,
                    page_size=self.page_size,
                    read_starts=read_starts,
                    tag=tag,
                ),
                kind,
                tokens,
            )
        raise RuntimeError(f"unsupported draft forward mode {getattr(mode, 'name', mode)}")

    def _before_kernels(self, fb: Any, site: str) -> None:
        """The pre-kernel point (GPU validation on): the step's tables are complete and none of
        its kernels has run. Extract the tables, validate them on the device and dispatch the
        host descriptors, in the order ``GpuValidation.run_step`` states. ``site`` is
        ``init_forward_metadata`` (eager steps and the breakable prefill graph) or
        ``decode_graph_replay`` (full decode graphs)."""
        gv = self.gpu
        if gv is None and self.tags is None and not self.prekernel:
            return
        if self.seq in self._prestep or self.seq in self.open:
            self.stats["prekernel_extra"] += 1
            return
        import torch

        if torch.cuda.is_current_stream_capturing():
            self.stats["prekernel_capturing"] += 1
            return
        t0 = time.perf_counter()
        where = f"{site}@{self.seq}"
        mode = fb.forward_mode
        graph = site == "decode_graph_replay" or (
            mode.is_extend() and self.last_static_fb is not None
        )
        dids: list[str] = []
        self._prestep[self.seq] = dids
        if self.nvtx is not None:
            # Once per step before its kernels: the previous step's completed publications
            # become read-only now, this step's are queued (nvtxmem.NvtxMemory.step).
            self.nvtx.step()
        try:
            w = self._wrapper(fb, graph)
            if (
                gv is not None
                and gv.fault == "table_foreign"
                and mode.is_decode()
                and w is not None
            ):
                req_ids = self._req_ids(fb, int(fb.batch_size))
                if gv.inject_table_fault(
                    w._paged_kv_indptr_buf, w._paged_kv_indices_buf, req_ids, self.seq
                ):
                    self.stats["gpu_fault_injected"] += 1
            plan = self._plan(fb, graph, w, where)
        except Exception as exc:
            log.exception("pre-kernel extraction failed at step %d", self.seq)
            self.reporter.emit(
                integration_error(
                    "adapter_extraction", f"{type(exc).__name__}: {exc}", pool=POOL, site=where
                )
            )
            self.stats["prekernel_failed"] += 1
            return
        self.timings["extract_s"] += time.perf_counter() - t0
        if plan is None:
            return

        def dispatch(d: DispatchDescriptor) -> None:
            self._dispatch_one(d, dids)

        if self.state is not None:
            self.state.pending_fills(fb)
        stream = torch.cuda.current_stream().cuda_stream
        if self.tags is not None:
            try:
                self.tags.step(plan, stream)
            except Exception as exc:
                log.exception("tag tier failed at step %d", self.seq)
                self.stats["tags_launch_failed"] += 1
                self.reporter.emit(
                    integration_error(
                        "tags_launch", f"{type(exc).__name__}: {exc}"[:300], pool=POOL, site=where
                    )
                )
        if gv is None:
            for d in plan.descs:  # the host dispatches, in the kernel order
                dispatch(d)
        else:
            try:
                gv.run_step(plan, dispatch, stream)
            except Exception as exc:
                log.exception("device validation failed at step %d", self.seq)
                gv.errors[site] += 1
                self.reporter.emit(
                    integration_error(
                        "gpu_validate_launch", f"{type(exc).__name__}: {exc}", pool=POOL, site=where
                    )
                )
                for d in plan.descs:  # the host dispatches stay complete
                    if d.dispatch_id not in dids:
                        dispatch(d)
        self._dispatch_state(fb, dids)
        self._record_plan(plan, dids)
        self.stats[f"prekernel_{site}"] += 1

    def _install_tag_hooks(self) -> None:
        """The publish hook of the tag tier (installed after the tier, which ``_install``
        precedes): a key inserted into the radix cache makes its page-aligned KV indices
        readable by any request."""
        me, tree = self, self.tree
        orig_insert = tree.insert

        def insert_(params, *args, **kwargs):
            out = orig_insert(params, *args, **kwargs)
            try:
                me._publish_tags(params)
            except Exception as exc:
                me.stats["tags_publish_failed"] += 1
                me.reporter.emit(
                    integration_error(
                        "tags_publish",
                        f"{type(exc).__name__}: {exc}"[:300],
                        pool=POOL,
                        site=f"insert@{me.seq}",
                    )
                )
            return out

        tree.insert = insert_

    def _publish_tags(self, params: Any) -> None:
        """The slots ``RadixCache.insert`` published: the value cut at the page-aligned key, as
        the cache's own ``insert`` cuts it (the bigram view for an EAGLE tree)."""
        key, value = params.key, params.value
        if key is None or value is None:
            return
        if getattr(self.tree, "is_eagle", False):
            key, value = key.maybe_to_bigram_view(True, value)
        key = key.page_aligned(self.page_size)
        n = len(key)
        if n == 0:
            return
        self.tags.publish(value[:n])

    def _req_ids(self, fb: Any, n: int) -> list[str]:
        """The checker ids of the forward batch's rows. The scheduler batch whose forward this
        is (``cur_batch``, the same object the forward batch was built from, in its order)
        gives the rows' table indices on the host; the device ``req_pool_indices`` is read
        only when that batch is not the forward's (a copy to the host synchronizes with the
        previous forward on the stream)."""
        reqs = getattr(self.cur_batch, "reqs", None)
        if reqs is not None and len(reqs) == n:
            try:
                # A freed/relaunched row leaves req.kv.req_pool_idx None (int() -> TypeError);
                # fall through to the device req_pool_indices, which hold the row the kernels
                # actually read, so the consuming access is still checked against it.
                row_ids = [int(r.kv.req_pool_idx) for r in reqs]
            except (AttributeError, TypeError):
                row_ids = None
            if row_ids is not None:
                self.stats["req_ids_from_batch"] += 1
                return [self.rid_of_row.get(i, f"unknown-row-{i}") for i in row_ids]
        self.stats["req_ids_from_device"] += 1
        row_ids = _ints(fb.req_pool_indices)[:n]
        return [self.rid_of_row.get(i, f"unknown-row-{i}") for i in row_ids]

    def _current_dflash_req_ids(self, n: int) -> list[str]:
        reqs = list(getattr(self.cur_batch, "reqs", ()) or ())[: int(n)]
        if len(reqs) != int(n):
            raise ValueError(f"DFlash materialization has {n} rows but {len(reqs)} requests")
        rows = [int(req.kv.req_pool_idx) for req in reqs]
        req_ids = [self.rid_of_row.get(row) for row in rows]
        if any(rid is None for rid in req_ids):
            raise ValueError(f"DFlash materialization rows {rows} are absent from the snapshot")
        return [str(rid) for rid in req_ids]

    def _wrapper(self, fb: Any, graph: bool) -> Any:
        """The FlashInfer wrapper whose buffers the step's attention kernels consume."""
        wrappers = self._wrappers(fb, graph)
        return wrappers[0] if wrappers else None

    def _wrappers(self, fb: Any, graph: bool) -> list[Any]:
        """Every FlashInfer wrapper consumed by the step, in layer-group order."""
        mode = fb.forward_mode
        backend = self._backend_for(fb)
        md = backend.forward_metadata
        if getattr(self, "facts", {}).get("mla"):
            if mode.is_target_verify():
                raise RuntimeError("speculative verify is outside the bounded MLA port")
            if mode.is_decode_or_idle():
                if graph:
                    dcg = self.model_runner.decode_cuda_graph_runner
                    graph_backend = _replay_full_backend(dcg)
                    return [graph_backend.decode_cuda_graph_metadata[int(dcg.bs)]]
                return [md.decode_wrapper]
            if mode.is_extend():
                return [md.prefill_wrapper]
            return []
        if mode.is_target_verify():
            if graph:
                dcg = self.model_runner.decode_cuda_graph_runner
                # PDMUX captures one verify graph per stream backend.  ``dcg.attn_backend``
                # names the construction-time backend, while replay selects the active stream's
                # group member and refreshes that member's metadata buffers.
                graph_backend = _replay_full_backend(dcg)
                return list(graph_backend.prefill_cuda_graph_metadata[int(dcg.bs)])
            if hasattr(md, "prefill_wrappers"):
                return list(md.prefill_wrappers)
            return list(backend.prefill_wrappers)
        if mode.is_decode_or_idle():
            # A graph replay plans into decode_cuda_graph_metadata[padded bs] (the wrappers the
            # captured kernels hold); backend.forward_metadata is assigned only at capture
            # (_prepare_cuda_graph_metadata) and in the eager init_forward_metadata, so after a
            # prefill it still names that step's PrefillMetadata during replays.
            if graph:
                dcg = self.model_runner.decode_cuda_graph_runner
                graph_backend = _replay_full_backend(dcg)
                return list(graph_backend.decode_cuda_graph_metadata[int(dcg.bs)])
            if hasattr(md, "decode_wrappers"):
                return list(md.decode_wrappers)
            return list(backend.decode_wrappers)
        if mode.is_extend():
            if not hasattr(md, "prefill_wrappers"):
                raise RuntimeError(f"extend step with {type(md).__name__} forward metadata")
            return list(md.prefill_wrappers)
        return []

    def _backend_for(self, fb: Any) -> Any:
        """Return the backend the current runtime state selected for this forward."""
        if self.pdmux and fb.forward_mode.is_decode_or_idle():
            return _full_backend(self.model_runner.decode_attn_backend)
        if getattr(self, "facts", {}).get("speculative_adaptive"):
            # Each adaptive step count owns a separately planned target backend.  The worker
            # swaps ModelRunner.attn_backend with the runtime state; self.backend is only the
            # state that happened to be active when KVGuard attached.
            return _full_backend(self.model_runner.attn_backend)
        return self.backend

    def _table_buffers(self, wrapper: Any) -> tuple[Any, Any, Any]:
        """Return the concrete CSR buffers consumed by the admitted FlashInfer wrapper."""
        if self.facts.get("mla"):
            return (
                wrapper._kv_indptr_buf,
                wrapper._kv_indices_buf,
                wrapper._kv_len_arr_buf,
            )
        return (
            wrapper._paged_kv_indptr_buf,
            wrapper._paged_kv_indices_buf,
            wrapper._paged_kv_last_page_len_buf,
        )

    def _last_page_lens(self, buffer: Any, rows: int) -> list[int]:
        """MLA's expanded CSR has one token per entry; MHA publishes explicit tail lengths."""
        return [1] * rows if self.facts.get("mla") else _ints(buffer)[:rows]

    def _pages_of(self, rid: str) -> Sequence[int | None]:
        """The request's page table as the ledger last observed it (empty when unknown)."""
        shadow = self.ledger.requests.get((rid, POOL))
        return shadow.snapshot.blocks if shadow is not None else ()

    def _logical_kv_slots(self, slots: Sequence[int]) -> list[int]:
        """Inverse-map physical storage write locations to stable virtual token ids."""
        values = list(map(int, slots))
        if not getattr(self, "facts", {}).get("unified_allocator"):
            return values
        p2v = _ints(self.allocator.full_attn_allocator.ids.physical_to_virtual)
        out: list[int] = []
        for value in values:
            if value < 0:
                out.append(value)
                continue
            physical_page, offset = divmod(value, self.page_size)
            virtual_page = p2v[physical_page] if physical_page < len(p2v) else -1
            out.append(virtual_page * self.page_size + offset if virtual_page >= 0 else -1)
        self.stats["unified_write_slots"] += len(out)
        self.stats["unified_write_slots_relocated"] += sum(
            logical != physical
            for logical, physical in zip(out, values, strict=True)
            if logical >= 0
        )
        return out

    def _logical_kv_pages(self, pages: Sequence[int]) -> list[int]:
        """Inverse-map physical transfer pages to stable unified-memory owner pages."""
        values = list(map(int, pages))
        if not getattr(self, "facts", {}).get("unified_allocator"):
            return values
        p2v = _ints(self.allocator.full_attn_allocator.ids.physical_to_virtual)
        out = [p2v[page] if 0 <= page < len(p2v) else -1 for page in values]
        self.stats["unified_pd_pages"] += len(out)
        self.stats["unified_pd_pages_relocated"] += sum(
            logical != physical
            for logical, physical in zip(out, values, strict=True)
            if logical >= 0
        )
        return out

    def _plan(
        self,
        fb: Any,
        graph: bool,
        w: Any,
        site: str,
        *,
        tag: str = "",
        pool: str = POOL,
        loc_override: Any = None,
    ) -> StepPlan | None:
        """Extract the step's consumed tables from the forward batch and the wrapper the kernels
        read, and build the host descriptors in kernel order."""
        mode = fb.forward_mode
        n = int(fb.batch_size)
        req_ids = self._req_ids(fb, n)
        seq_lens = _ints(fb.seq_lens_cpu if fb.seq_lens_cpu is not None else fb.seq_lens)[:n]
        prefix: list[int] = []
        use_ragged = False
        read_starts: list[int] = []
        indptr_buf, indices_buf, last_page_len_buf = self._table_buffers(w)
        if self.facts.get("mla") and graph and mode.is_decode_or_idle():
            # MLA's sync-free graph replan updates the capture-stable KV-index buffer but
            # deliberately leaves the wrapper's device indptr at its capture shape. The
            # executed FA2/FA3 plan is built from this exact host indptr instead.
            dcg = self.model_runner.decode_cuda_graph_runner
            graph_backend = _replay_full_backend(dcg)
            indptr_buf = graph_backend.cuda_graph_kv_indptr_cpu[: int(dcg.bs) + 1]
        mapping_tables = (
            getattr(self, "tables", "device") == "mapping"
            and pool == POOL
            and loc_override is None
            and (mode.is_decode_or_idle() or mode.is_extend())
            and not self.facts.get("mla")
            and not self.facts.get("unified_allocator")
            and self.facts.get("sliding_window_size") is None
            and not getattr(self, "separate_swa", False)
            and not self.pdmux
            and not (self.speculative_ngram or self.speculative_kv)
        )
        if mapping_tables:
            # The bookkeeping tier: the descriptors from the ledger's page tables, no device
            # read; the step's lists are the mapping's, and the device tensors go to the
            # validators as before.
            self.stats["mapping_tables_steps"] += 1
            if mode.is_extend():
                md = self._backend_for(fb).forward_metadata
                use_ragged = bool(md.use_ragged)
                prefix = (
                    [int(p) for p in fb.extend_prefix_lens_cpu]
                    if fb.extend_prefix_lens_cpu is not None
                    else _ints(fb.extend_prefix_lens)
                )
                loc_t = fb.out_cache_loc
                if graph:
                    self.stats["graph_replays_extend"] += 1
                    if self.last_static_fb is not None:
                        loc_t = self.last_static_fb.out_cache_loc
                self.last_static_fb = None
                n_pad = n
                kind = "extend"
                (first, second), loc = build_mapping_dispatches(
                    self.seq, pool, req_ids, seq_lens, prefix, self._pages_of, self.page_size,
                    mode="extend", use_ragged=use_ragged, tag=tag,
                )  # fmt: skip
                total = sum(s - p for s, p in zip(seq_lens, prefix, strict=False))
                self.stats["dispatch_tokens"] += total
                self.stats["use_ragged_true" if use_ragged else "use_ragged_false"] += 1
            else:
                loc_t = fb.out_cache_loc
                n_pad = n
                if graph:
                    self.stats["graph_replays_decode"] += 1
                    dcg = self.model_runner.decode_cuda_graph_runner
                    n_pad = int(dcg.bs)
                    try:
                        width = int(getattr(dcg, "captured_req_width", 1) or 1)
                        loc_t = dcg.buffer_registry.get_slot("out_cache_loc").slice_for(
                            n_pad, n_pad * width
                        )
                        self.stats["padded_loc_from_registry"] += 1
                    except Exception:
                        self.stats["padded_loc_fallback"] += 1
                kind = "decode"
                (first, second), loc = build_mapping_dispatches(
                    self.seq, pool, req_ids, seq_lens, None, self._pages_of, self.page_size,
                    mode="decode", num_rows_after_padding=n_pad, tag=tag,
                )  # fmt: skip
                self.stats["padding_rows"] += n_pad - n
                self.stats["dispatch_tokens"] += n
            descs = [first, second]
            indptr = [0]
            for L in seq_lens:
                indptr.append(indptr[-1] + int(L))
            indptr.extend([indptr[-1]] * (n_pad - n))
            n_loc = int(loc_t.numel()) if hasattr(loc_t, "numel") else len(loc_t)
            loc = list(loc) + [0] * max(n_loc - len(loc), 0)
            return StepPlan(
                seq=self.seq,
                mode=kind,
                graph=graph,
                n=n,
                n_pad=n_pad,
                req_ids=req_ids,
                seq_lens=list(seq_lens),
                prefix=list(prefix)[:n],
                use_ragged=use_ragged,
                indptr=indptr,
                indices=[],
                loc=loc,
                descs=descs,
                buffers=DeviceBuffers(indptr_buf, indices_buf, last_page_len_buf, loc_t),
                read_starts=None,
            )
        if mode.is_target_verify():
            if not (self.speculative_ngram or self.speculative_kv):
                raise RuntimeError("target verify reached outside the speculative boundary")
            spec = fb.spec_info
            draft = int(getattr(spec, "draft_token_num", 0) or 0)
            if not _verify_width_supported(self.facts, draft):
                raise RuntimeError(
                    f"verify draft width {draft}, expected {self.facts['speculative_num_draft_tokens']}"
                )
            if self.facts.get("speculative_adaptive"):
                self.stats[f"adaptive_verify_width_{draft}"] += 1
            if self.speculative_dflash:
                # DFlash publishes the verify host bound after adding its dense block, unlike
                # EAGLE/NGRAM whose verify ForwardBatch lengths remain at the committed prefix.
                seq_lens = [int(length) - draft for length in seq_lens]
                if any(length < 0 for length in seq_lens):
                    raise ValueError(f"DFlash verify lengths underflow after width {draft}")
            for rid, frontier in zip(req_ids, seq_lens, strict=True):
                self.ledger.rollback(pool, rid, frontier, site=f"verify-rollback@{self.seq}")
            indptr = _ints(indptr_buf)
            n_pad = len(indptr) - 1
            # SGLang's FlashInfer translator keeps one physical token slot per position,
            # including at page sizes above one.
            indices = self._logical_kv_slots(_ints(indices_buf[: max(int(indptr[-1]), 0)]))
            table_lens = [int(indptr[i + 1]) - int(indptr[i]) for i in range(n)]
            read_starts = [max(int(seq_lens[i]) + draft - table_lens[i], 0) for i in range(n)]
            loc_t = fb.out_cache_loc if loc_override is None else loc_override
            if graph and loc_override is None:
                dcg = self.model_runner.decode_cuda_graph_runner
                try:
                    bs = int(dcg.bs)
                    width = int(getattr(dcg, "captured_req_width", draft) or draft)
                    loc_t = dcg.buffer_registry.get_slot("out_cache_loc").slice_for(bs, bs * width)
                    self.stats["padded_loc_from_registry"] += 1
                except Exception:
                    self.stats["padded_loc_fallback"] += 1
            self.last_static_fb = None
            loc = self._logical_kv_slots(_ints(loc_t))
            self._verify_requests = list(zip(req_ids, map(int, seq_lens), strict=True))
            lpl = self._last_page_lens(last_page_len_buf, n_pad)
            if any(x != 1 for x in lpl):
                self.reporter.emit(
                    integration_error(
                        "kv_last_page_len_not_one",
                        f"verify kv_last_page_len {lpl[:8]}",
                        pool=POOL,
                        site=site,
                    )
                )
            descs = list(
                build_verify_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    seq_lens,
                    indptr,
                    indices,
                    loc,
                    draft,
                    num_rows_after_padding=n_pad,
                    last_page_len=lpl,
                    page_size=self.page_size,
                    read_starts=read_starts,
                    tag=tag or "-verify",
                )
            )
            self.stats["speculative_verify_steps"] += 1
            self.stats["speculative_verify_tokens"] += n * draft
            self.stats["speculative_verify_padding_rows"] += max(n_pad - n, 0)
            self.stats["speculative_rollbacks"] = self.ledger.counters["speculative_rollbacks"]
            prefix = list(seq_lens)
            seq_lens = [s + draft for s in seq_lens]
            use_ragged = False
            kind = "verify"
        elif mode.is_decode_or_idle():
            indptr = _ints(indptr_buf)
            n_pad = len(indptr) - 1
            indices = self._logical_kv_slots(_ints(indices_buf[: max(int(indptr[-1]), 0)]))
            table_lens = [int(indptr[i + 1]) - int(indptr[i]) for i in range(n)]
            read_starts = [max(int(seq_lens[i]) - table_lens[i], 0) for i in range(n)]
            suffix = [start for start in read_starts if start]
            self.stats["suffix_table_rows"] += len(suffix)
            if suffix:
                self.stats["suffix_table_max_start"] = max(
                    self.stats["suffix_table_max_start"], max(suffix)
                )
            lpl = self._last_page_lens(last_page_len_buf, n_pad)
            if any(
                x != 1 for x in lpl
            ):  # one entry per position at every page size (translate._token_table)
                self.reporter.emit(
                    integration_error(
                        "kv_last_page_len_not_one",
                        f"kv_last_page_len {lpl[:8]}",
                        pool=POOL,
                        site=site,
                    )
                )
            loc_t = fb.out_cache_loc if loc_override is None else loc_override
            if graph:
                self.stats["graph_replays_decode"] += 1
                dcg = self.model_runner.decode_cuda_graph_runner
                bs = int(dcg.bs)
                if loc_override is None:
                    try:
                        width = int(getattr(dcg, "captured_req_width", 1) or 1)
                        loc_t = dcg.buffer_registry.get_slot("out_cache_loc").slice_for(
                            bs, bs * width
                        )
                        self.stats["padded_loc_from_registry"] += 1
                    except Exception:
                        self.stats["padded_loc_fallback"] += 1
                if bs != n_pad:
                    self.reporter.emit(
                        integration_error(
                            "padded_batch_mismatch",
                            f"graph bs {bs}, wrapper rows {n_pad}",
                            pool=pool,
                            site=site,
                        )
                    )
            loc = self._logical_kv_slots(_ints(loc_t))
            descs = list(
                build_decode_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    seq_lens,
                    indptr,
                    indices,
                    loc,
                    num_rows_after_padding=n_pad,
                    last_page_len=lpl,
                    page_size=self.page_size,
                    read_starts=read_starts,
                    tag=tag,
                )
            )
            self.stats["padding_rows"] += n_pad - n
            self.stats["dispatch_tokens"] += n
            kind = "decode"
        elif mode.is_extend():
            md = self._backend_for(fb).forward_metadata
            use_ragged = bool(md.use_ragged)
            if self.facts.get("mla") and use_ragged:
                # MLA's ragged prefill consumes the new-token batch directly and never plans
                # the paged wrapper. Represent its empty cached-prefix read explicitly rather
                # than inspecting the wrapper's absent/stale plan buffers.
                indptr = [0] * (n + 1)
                indices = []
            else:
                indptr = _ints(indptr_buf)[: n + 1]
                indices = self._logical_kv_slots(_ints(indices_buf[: max(int(indptr[-1]), 0)]))
            prefix = (
                [int(p) for p in fb.extend_prefix_lens_cpu]
                if fb.extend_prefix_lens_cpu is not None
                else _ints(fb.extend_prefix_lens)
            )
            if self.speculative_ngram or self.speculative_kv:
                # Mixed scheduling and decode-to-extend present a speculative decode tail as a
                # one-token extend.  The physical candidate window can still be initialized
                # beyond its authoritative prefix, so invalidate that rejected suffix before
                # the target rewrites the tail (the draft-pool path does the same in
                # ``_draft_descriptors``).
                for rid, frontier in zip(req_ids, prefix, strict=True):
                    self.ledger.rollback(pool, rid, frontier, site=f"extend-rollback@{self.seq}")
            loc_t = fb.out_cache_loc if loc_override is None else loc_override
            if graph:
                self.stats["graph_replays_extend"] += 1
                if self.last_static_fb is not None:
                    loc_t = self.last_static_fb.out_cache_loc
                    self.stats["padded_loc_from_static_batch"] += 1
                else:
                    self.stats["padded_loc_fallback"] += 1
            self.last_static_fb = None
            loc = self._logical_kv_slots(_ints(loc_t))
            table_lens = [int(indptr[i + 1]) - int(indptr[i]) for i in range(n)]
            read_ends = prefix if use_ragged else seq_lens
            read_starts = [max(int(read_ends[i]) - table_lens[i], 0) for i in range(n)]
            suffix = [start for start in read_starts if start]
            self.stats["suffix_table_rows"] += len(suffix)
            if suffix:
                self.stats["suffix_table_max_start"] = max(
                    self.stats["suffix_table_max_start"], max(suffix)
                )
            descs = list(
                build_extend_dispatches(
                    self.seq,
                    pool,
                    req_ids,
                    seq_lens,
                    prefix,
                    indptr,
                    indices,
                    loc,
                    use_ragged=use_ragged,
                    last_page_len=self._last_page_lens(last_page_len_buf, n),
                    page_size=self.page_size,
                    read_starts=read_starts,
                    tag=tag,
                )
            )
            self.stats["use_ragged_true" if use_ragged else "use_ragged_false"] += 1
            if getattr(md, "extend_no_prefix", False):
                self.stats["extend_no_prefix"] += 1
            total = sum(s - p for s, p in zip(seq_lens, prefix, strict=False))
            self.stats["dispatch_tokens"] += total
            self.stats["padded_tokens"] += max(len(loc) - total, 0)
            n_pad = n
            kind = "extend"
        else:
            self.reporter.emit(
                integration_error(
                    "unsupported_forward_mode", f"forward mode {mode.name}", pool=pool, site=site
                )
            )
            return None
        return StepPlan(
            seq=self.seq,
            mode=kind,
            graph=graph,
            n=n,
            n_pad=n_pad,
            req_ids=req_ids,
            seq_lens=seq_lens,
            prefix=prefix[:n],
            use_ragged=use_ragged,
            indptr=indptr,
            indices=indices,
            loc=loc,
            descs=descs,
            buffers=DeviceBuffers(
                indptr_buf,
                indices_buf,
                last_page_len_buf,
                loc_t,
            ),
            read_starts=list(read_starts) if read_starts else None,
        )

    def _stream(self) -> int | None:
        return current_stream_handle() if self.stream_epochs else None

    def _fenced_epochs(self) -> tuple[str, ...]:
        """The dispatches of the open steps the scheduler stream has waited behind."""
        return tuple(
            did
            for seq, ep in self.open.items()
            if seq <= self._fence_seq
            for did in ep.dispatch_ids
        )

    def _dispatch_state(self, fb: Any, dids: list[str]) -> None:
        """The recurrent-state pool's dispatch of the step (a hybrid model), from the GDN
        backend's metadata; failures are integration errors, never exceptions in the engine."""
        if self.state is None:
            return
        try:
            n = int(fb.batch_size)
            rids = [self.rid_of_row.get(i) for i in _ints(fb.req_pool_indices)[:n]]
            d = self.state.descriptor(
                fb, rids, getattr(self.model_runner.attn_backend, "linear_attn_backend", None)
            )
        except Exception as exc:
            log.exception("state descriptor failed at step %d", self.seq)
            self.stats["state_dispatch_failed"] += 1
            self.reporter.emit(
                integration_error(
                    "state_extraction",
                    f"{type(exc).__name__}: {exc}"[:300],
                    pool=STATE_POOL,
                    site=f"state@{self.seq}",
                )
            )
            return
        if d is not None:
            self._dispatch_one(d, dids)
            self.stats["state_dispatches"] += 1
            if self.tags is not None:
                try:
                    import torch

                    self.tags.state_step(d, torch.cuda.current_stream().cuda_stream)
                except Exception as exc:
                    self.stats["tags_state_failed"] += 1
                    self.reporter.emit(
                        integration_error(
                            "tags_state",
                            f"{type(exc).__name__}: {exc}"[:300],
                            pool=STATE_POOL,
                            site=f"state@{self.seq}",
                        )
                    )

    def _dispatch_one(self, d: DispatchDescriptor, dids: list[str]) -> None:
        t0 = time.perf_counter()
        d = dataclasses.replace(d, snapshot_seq=self.seq, stream=self._stream())
        self.ledger.dispatch(d)
        dids.append(d.dispatch_id)
        self.timings["dispatch_s"] += time.perf_counter() - t0

    def _record_plan(self, plan: StepPlan, dids: list[str]) -> None:
        self.stats["dispatches"] += len(dids)
        self._last_dispatch = {
            "n": plan.n,
            "lanes": plan.n_pad,
            "graph": plan.graph,
            "requests": plan.req_ids,
            "loc": len(plan.loc),
            "dispatches": len(dids),
            "indptr_last": int(plan.indptr[-1]) if len(plan.indptr) else None,
        }
        self._last_write_blocks = tuple(dict.fromkeys(pages_of(plan.loc, self.page_size)))

    def _descriptors(self, fb: Any, out: Any) -> list[str]:
        """The host-only path: consumed descriptors extracted and dispatched after the forward.

        A shared-pool sliding-window model has two tables: wrapper 0 gathers the live suffix for
        sliding-window layers and wrapper 1 gathers the full prefix for full-attention layers.
        The pool-level write is represented once, while both independently consumed tables are
        checked.
        """
        t0 = time.perf_counter()
        graph = bool(getattr(out, "can_run_graph", False))
        wrappers = self._wrappers(fb, graph)
        # A prefill graph's load_batch result belongs to the whole forward.  _plan consumes the
        # saved reference so it cannot accidentally leak into the next step; restore that same
        # reference for each wrapper of this step, then clear it once all tables are extracted.
        static_fb = self.last_static_fb
        plans = []
        for index, wrapper in enumerate(wrappers):
            self.last_static_fb = static_fb
            separate_swa_wrapper = getattr(self, "separate_swa", False) and index == 0
            plan_pool = SWA_POOL if separate_swa_wrapper else POOL
            loc_override = None
            if separate_swa_wrapper:
                if graph:
                    dcg = self.model_runner.decode_cuda_graph_runner
                    # Match _wrappers(): PDMUX chooses the replay backend from the
                    # stream-local graph key. model_runner.decode_attn_backend is a
                    # scheduler-side convenience pointer and may already name a
                    # different stream when extraction runs after replay.
                    active_backend = _replay_full_backend(dcg)
                    bs = int(dcg.bs)
                    width = int(getattr(dcg, "captured_req_width", 1) or 1)
                    graph_loc = getattr(active_backend, "cuda_graph_swa_out_cache_loc", None)
                    loc_override = graph_loc[: bs * width] if graph_loc is not None else None
                else:
                    active_backend = self._backend_for(fb)
                    metadata = active_backend.forward_metadata
                    loc_override = getattr(metadata, "swa_out_cache_loc", None)
                if loc_override is None:
                    raise RuntimeError("separate SWA wrapper exposed no SWA write locations")
            plan_kwargs = {
                "tag": ("" if len(wrappers) == 1 else ("-swa" if index == 0 else f"-full{index}"))
            }
            if getattr(self, "separate_swa", False):
                plan_kwargs.update(pool=plan_pool, loc_override=loc_override)
            plan = self._plan(
                fb,
                graph,
                wrapper,
                f"forward@{self.seq}:{index}",
                **plan_kwargs,
            )
            if plan is not None:
                plans.append(plan)
        self.last_static_fb = None
        self.timings["extract_s"] += time.perf_counter() - t0
        if not plans:
            return []
        primary = plans[0]
        if getattr(self, "separate_swa", False):
            descs = [descriptor for plan in plans for descriptor in plan.descs]
        else:
            descs = list(primary.descs)
            for plan in plans[1:]:
                extra = next(d for d in plan.descs if d.family == "attention")
                insert_at = len(descs) - 1 if primary.attn_first else len(descs)
                descs.insert(insert_at, extra)
        dids: list[str] = []
        for d in descs:
            self._dispatch_one(d, dids)
        self._dispatch_state(fb, dids)
        self._record_plan(primary, dids)
        self.stats["attention_wrappers"] += len(plans)
        self._last_dispatch["wrappers"] = len(plans)
        return dids

    # ------------------------------------------------------------------ completion
    def _complete_through(self, seq: int, reason: str) -> None:
        """Complete every open step up to ``seq`` after waiting for its forward."""
        for s in sorted(self.open):
            if s > seq:
                break
            epoch = self.open.pop(s)
            t0 = time.perf_counter()
            epoch.event.synchronize()
            self.timings["complete_wait_s"] += time.perf_counter() - t0
            for did in epoch.dispatch_ids:
                self.ledger.complete(did)
            self.stats[f"completed_{reason}"] += 1

    def _complete_referencing(
        self, slots: Sequence[int], pools: Sequence[str] | None = None
    ) -> None:
        if not self.open:
            return
        governed = tuple(pools) if pools is not None else self._cache_pools()
        keys = {(pool, int(s)) for pool in governed for s in slots}
        for seq in sorted(self.open):
            epoch = self.open[seq]
            if any(
                keys & self.ledger.epochs[d].blocks
                for d in epoch.dispatch_ids
                if d in self.ledger.epochs
            ):
                self._complete_through(seq, "on_free")

    def _before_process_result(self, result: Any) -> None:
        entry = self.results.pop(id(result), None)
        if entry is None:
            self.stats["results_without_step"] += 1
            return
        self._complete_through(entry[0], "result")

    # ------------------------------------------------------------------ the record
    def config(self) -> dict[str, Any]:
        args = self.scheduler.server_args
        return {k: str(getattr(args, k, None)) for k in CONFIG_KEYS}

    def environment(self) -> dict:
        if self._environment_cache is None:
            try:
                self._environment_cache = (
                    self._environment() if self._environment else fallback_environment()
                )
            except Exception as exc:
                self._environment_cache = {"error": repr(exc)}
        return self._environment_cache

    def _unified_allocator_record(self) -> dict[str, Any] | None:
        if not self.facts.get("unified_allocator"):
            return None
        out: dict[str, Any] = {}
        for role in ("full_attn_allocator", "mamba_allocator"):
            band = getattr(self.allocator, role, None)
            if band is None:
                continue
            out[role] = {
                "sub_pool": band.sub_pool_name,
                "lazy_compaction": bool(band.lazy_compaction),
                "min_page_index": int(band.min_page_index),
                "num_pages": int(band.num_pages),
                "live_pages": int(band.live_page_count),
                "free_physical_holes": len(band._free_phys_pages),
                "pending_reuse_pages": len(band._pending_reuse_pages_cpu),
                "watermark_physical": int(band.watermark_physical),
                "free_lazy_calls": int(band._stats_n_free_lazy),
                "flush_calls": int(band._stats_n_flush_calls),
                "flush_work_calls": int(band._stats_n_flush_did_work),
                "flush_moves": int(band._stats_n_flush_moves),
                "pages_absorbed": int(band._stats_n_pages_absorbed),
                "drain_calls": int(band._stats_n_drain_calls),
                "drain_work_calls": int(band._stats_n_drain_did_work),
                "drained_pages": int(band._stats_n_drained_pages_total),
            }
        return out

    def shadow_bytes(self) -> dict[str, Any]:
        """The attachment's memory for the memory table: the host shadow (the ledger's
        structures and this adapter's caches, deep size, each object once) and the device
        tiers' allocations (the dispatch validators' mirror and buffers, the tag tier's tags).
        The recurrent-state pool's records are in the ledger; its mediation objects reference
        the framework's and are not walked."""
        from kvguard.core.memsize import deep_size

        seen: set[int] = set()
        ledger = self.ledger.memory_bytes(seen)
        caches = {
            "walk_states": deep_size(self.walk_states, seen),
            "draft_walk_states": (
                deep_size(self.draft_walk_states, seen)
                if getattr(self, "draft_walk_states", None) is not None
                else 0
            ),
            "row_cache": deep_size(self.row_cache, seen) if self.row_cache is not None else 0,
        }
        gpu = self.gpu
        device = {
            "mirror": int(getattr(getattr(gpu, "mirror", None), "nbytes", 0) or 0),
            "validators": (
                int(gpu.v.device_bytes())
                if gpu is not None and hasattr(gpu.v, "device_bytes")
                else 0
            ),
            "tags": int((self.tags.stats() or {}).get("tag_bytes", 0))
            if self.tags is not None
            else 0,
        }
        return {
            "host_ledger": ledger,
            "host_caches": caches,
            "host_total": ledger["total"] + sum(caches.values()),
            "device": device,
            "device_total": sum(device.values()),
        }

    def record(self) -> dict[str, Any]:
        reports = self.reporter.reports
        if self.reporter.violations():
            status = "violations"
        elif self.reporter.integration_errors():
            status = "integration_errors"
        else:
            status = "pass"
        try:
            shadow_bytes = self.shadow_bytes()
        except Exception as exc:  # the record must still be written
            shadow_bytes = {"error": repr(exc)}
        return {
            "framework": "sglang",
            "shadow_bytes": shadow_bytes,
            "attached": True,
            "pid": os.getpid(),
            "status": status,
            "geometry": {
                "num_blocks": self.geometry.num_blocks,
                "block_size": self.geometry.block_size,
                "reserved_blocks": sorted(self.geometry.reserved_blocks),
                "layers": self.geometry.layers,
                "kv_heads": self.geometry.kv_heads,
            },
            "draft_geometry": (
                {
                    "num_blocks": self.draft_geometry.num_blocks,
                    "block_size": self.draft_geometry.block_size,
                    "reserved_blocks": sorted(self.draft_geometry.reserved_blocks),
                    "layers": self.draft_geometry.layers,
                    "kv_heads": self.draft_geometry.kv_heads,
                }
                if self.draft_geometry is not None
                else None
            ),
            "support_boundary": SUPPORT_BOUNDARY,
            "boundary_facts": self.facts,
            "config": self.config(),
            "early_complete": self.early_complete,
            "stream_epochs": self.stream_epochs,
            "ledger_counters": dict(self.ledger.counters),
            "sync_after_forward": self.sync_after_forward,
            "prekernel": self.prekernel,
            "row_cache": self.row_cache_enabled,
            "row_cache_stats": dict(self.row_cache.stats) if self.row_cache else None,
            "row_cache_drift": (
                [self.row_cache.drift, self.row_cache.drift_examples] if self.row_cache else None
            ),
            "tables": self.tables,
            "audit_every": self.audit_every,
            "queued_rows": self.queued_rows,
            "fault": self.fault.kind if self.fault else None,
            "fault_note": self.fault.note if self.fault else None,
            "twins": self.twins.record() if self.twins is not None else None,
            "gpu_validation": self.gpu.record() if self.gpu is not None else None,
            "tags": self.tags.stats() if self.tags is not None else None,
            "nvtx": self.nvtx.record() if self.nvtx is not None else None,
            "state": self.state.record() if self.state is not None else None,
            "unified_allocator": self._unified_allocator_record(),
            "stats": dict(sorted(self.stats.items())),
            "timings": {k: round(v, 4) for k, v in sorted(self.timings.items())},
            "report_kinds": dict(Counter(r.kind for r in reports)),
            "report_kinds_device": dict(Counter(r.kind for r in reports if _device_report(r))),
            "report_kinds_tags": dict(Counter(r.kind for r in reports if _tag_report(r))),
            "report_kinds_host": dict(
                Counter(r.kind for r in reports if not (_device_report(r) or _tag_report(r)))
            ),
            "reports_first": [str(r) for r in reports[:40]],
            "duplicates": self.reporter.duplicates,
            "open_epochs": len(self.ledger.epochs),
            "open_steps": len(self.open),
            "hicache_pending_reads": len(self._hicache_reads),
            "hicache_store_reservations": len(self._hicache_store_reservations),
            "hicache_pending_loads": len(self._hicache_loads),
            "external_linker_pending_loads": len(self._external_linker_load_jobs),
            "external_linker_pending_stores": len(self._external_linker_store_jobs),
            "pd_decode_offload_pending_reads": len(self._decode_offload_reads),
            "pd_trace": self._pd_trace,
            "pd_source_release_trace": self._pd_source_release_trace,
            "pd_decode_trace": self._pd_decode_trace,
            "pd_decode_offload_trace": self._decode_offload_trace,
            "pd_decode_pending_receivers": len(self._pd_decode_writes),
            "pd_retraction_pending_restores": len(self._pd_retraction_restores),
            "flush_count": self.flush_count,
            "idle_flush_count": self.idle_flush_count,
            "trace": self.trace[:TRACE_LIMIT],
            "uptime_s": round(time.time() - self.started, 1),
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "environment": self.environment(),
        }

    def flush(self, reason: str) -> None:
        if not self.out_path:
            return
        try:
            self._reap_pd_reads()
            if reason in ("loop_exit", "idle"):
                # The GPU has nothing queued: the outstanding readbacks complete at once, and
                # the record then covers every step (the last record of a run is the idle one).
                if self.gpu is not None:
                    self.gpu.drain()
                elif self.tags is not None:
                    self.tags.drain_into(self.reporter)
                if self.twins is not None:
                    self.twins.drain()
                self._write_profile()
            rec = self.record()
            rec["flush_reason"] = reason
            tmp = f"{self.out_path}.tmp"
            os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, default=str)
            os.replace(tmp, self.out_path)
            self.flush_count += 1
            self.dirty = False
            self._last_flush = time.time()
        except Exception:
            log.exception("record flush failed")

    def _maybe_flush(self) -> None:
        if self.dirty and time.time() - self._last_flush >= FLUSH_INTERVAL_S:
            self.flush("periodic")


# ------------------------------------------------------------------ environment and entry points


def _device_report(report: Any) -> bool:
    """A report the device validators emitted (``report_results`` sets the site prefix)."""
    return str(getattr(report, "site", "") or "").startswith("gpu-validate:")


def _tag_report(r: Any) -> bool:
    """A report of the tag tier (site ``gpu-tags:<dispatch id>``)."""
    return (r.site or "").startswith("gpu-tags:")


def fallback_environment() -> dict:
    """A minimal environment block when ``envmeta`` is not available."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {},
    }
    for name in ("torch", "triton", "flashinfer-python", "sglang", "sglang-kernel"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    return info


def environment_from_env() -> dict:
    """``envmeta.collect`` from the module file named by ``KVGUARD_SGLANG_ENVMETA``."""
    path = os.environ.get(ENV_ENVMETA)
    if not path or not os.path.exists(path):
        return fallback_environment()
    spec = importlib.util.spec_from_file_location("kvguard_envmeta", path)
    if spec is None or spec.loader is None:
        return fallback_environment()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect(["sglang", "sglang-kernel", "flash-attn-4", "cuda-tile"])


def _report_record(report: Any) -> dict[str, Any]:
    """One report as a plain dict for the live NDJSON feed."""

    def val(x: Any) -> Any:
        return getattr(x, "value", x)

    return {
        "outcome": val(report.outcome),
        "kind": report.kind,
        "detail": report.detail,
        "policy": val(report.policy) if report.policy is not None else None,
        "pool": report.pool,
        "block": report.block,
        "slot": report.slot,
        "generation": report.generation,
        "request": report.request,
        "site": report.site,
        "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
    }


def _install_live_ndjson_sink(reporter: Reporter) -> None:
    """If ``KVGUARD_SGLANG_LIVE_NDJSON`` is set, stream each emitted report as a JSON line so an
    out-of-process reader can show detection live. Truncates the file once per attachment; never
    raises into the checker."""
    path = os.environ.get(ENV_LIVE_NDJSON) or None
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        handle = open(path, "w", encoding="utf-8")  # noqa: SIM115 (kept open for the run)
    except OSError as exc:
        _say(f"live ndjson disabled: {exc}")
        return

    def sink(report: Any, _fh: Any = handle) -> None:
        try:
            _fh.write(json.dumps(_report_record(report), default=str) + "\n")
            _fh.flush()
        except Exception:  # pragma: no cover - never disturb the checker
            pass

    reporter.sink = sink


def attach(scheduler: Any, reporter: Reporter | None = None, **options) -> SglangAttachment:
    """Install the adapter on a live ``Scheduler`` (inside the scheduler process)."""
    return SglangAttachment(scheduler, reporter, **options)


def attach_from_env(scheduler: Any) -> SglangAttachment | None:
    """Attach with options from the environment; never raises unless ``KVGUARD_SGLANG_STRICT=1``.
    ``KVGUARD_SGLANG_PROFILE=<path>`` runs the attachment's entry points under cProfile for the
    steps ``KVGUARD_SGLANG_PROFILE_STEPS`` (``100:400``) and writes the stats there at the loop
    exit (a diagnostic for the host-cost work).

    On ``Unsupported`` the record is still written (``attached: false`` and the report) so the
    probe sees why, and the scheduler runs unattached.
    """
    out = os.environ.get(ENV_OUT) or None
    fault = os.environ.get(ENV_FAULT) or None
    step = os.environ.get(ENV_FAULT_STEP)
    twins = bool(os.environ.get(ENV_TWINS))
    twin_fault = os.environ.get(ENV_TWIN_FAULT) or None
    gpu_validate = os.environ.get(ENV_GPU_VALIDATE, "0") not in ("", "0")
    tags = os.environ.get(ENV_TAGS, "0") not in ("", "0")
    nvtx = os.environ.get(ENV_NVTX, "0") not in ("", "0")
    gpu_fault = os.environ.get(ENV_GPU_FAULT) or None
    gpu_step = os.environ.get(ENV_GPU_FAULT_STEP)
    profile = os.environ.get(ENV_PROFILE) or None
    lo, _, hi = os.environ.get(ENV_PROFILE_STEPS, "100:400").partition(":")
    reporter = Reporter(fail_fast=False)
    _install_live_ndjson_sink(reporter)
    try:
        att = SglangAttachment(
            scheduler,
            reporter,
            out_path=out,
            early_complete=os.environ.get(ENV_EARLY_COMPLETE, "0") == "1",
            fault=fault,
            fault_step=int(step) if step else None,
            environment=environment_from_env,
            twins=twins,
            twin_fault=twin_fault,
            gpu_validate=gpu_validate,
            tags=tags,
            nvtx=nvtx,
            gpu_fault=gpu_fault,
            gpu_fault_step=int(gpu_step) if gpu_step else None,
            sync_after_forward=os.environ.get(ENV_SYNC, "1") != "0",
            prekernel=os.environ.get(ENV_PREKERNEL, "0") == "1",
            tables=os.environ.get(ENV_TABLES, "device"),
            row_cache=os.environ.get(ENV_ROW_CACHE, "1") == "1",
            audit_every=int(os.environ.get(ENV_AUDIT_EVERY, "64")),
            queued_rows=os.environ.get(ENV_QUEUED_ROWS, "1") != "0",
            stream_epochs=os.environ.get(ENV_STREAM_EPOCHS, "1") != "0",
            profile=profile,
            profile_steps=(int(lo or 100), int(hi or 400)),
        )
    except Unsupported as exc:
        _say(f"not attached: {exc.report}")
        if out:
            rec = {
                "framework": "sglang",
                "attached": False,
                "pid": os.getpid(),
                "status": "unsupported",
                "report_kinds": dict(Counter(r.kind for r in reporter.reports)),
                "reports_first": [str(r) for r in reporter.reports],
                "support_boundary": SUPPORT_BOUNDARY,
                "boundary_facts": getattr(exc, "facts", None),
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            }
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, default=str)
        if os.environ.get(ENV_STRICT, "0") == "1":
            raise
        return None
    _say(
        f"attached to Scheduler pid {os.getpid()}: pool {att.geometry.num_blocks} slots, "
        f"{att.facts['attention_backend']}, overlap={att.facts['overlap_schedule']}, "
        f"early_complete={att.early_complete}, stream_epochs={att.stream_epochs}, "
        f"sync={att.sync_after_forward}, prekernel={att.prekernel}, tables={att.tables}, "
        f"queued_rows={att.queued_rows}, fault={fault or None}, "
        f"twins={att.twins.outcome(att.twins.at_attach) if att.twins else None}, "
        f"twin_fault={twin_fault if att.twins else None}, "
        f"gpu_validate={att.gpu is not None}, gpu_fault={gpu_fault if att.gpu else None}, "
        f"tags={att.tags is not None}, record -> {out}"
    )
    att.flush("attach")
    return att


def flush_attachment(att: SglangAttachment, reason: str) -> None:
    att.flush(reason)


__all__ = [
    "CONFIG_KEYS",
    "GPU_FAULTS",
    "TWIN_FAULTS",
    "DeviceBuffers",
    "Fault",
    "GpuValidation",
    "SglangAttachment",
    "StepEpoch",
    "StepPlan",
    "TwinPublisher",
    "attach",
    "attach_from_env",
    "check_support",
    "environment_from_env",
    "fallback_environment",
    "flashinfer_kernel_traits",
    "flush_attachment",
    "held_requests",
    "twin_descriptor",
    "twin_fault_entries",
    "twin_launch_delta",
    "validator_rows",
]
