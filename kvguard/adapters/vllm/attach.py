"""Attach the core to a live, in-process vLLM engine through run-time wrappers.

No vLLM file is modified: the wrappers are instance attributes set on the live objects. The
attachment covers the support boundary in ``translate.SUPPORT_BOUNDARY`` and raises
``Unsupported`` for anything else it can detect at attach time.

The main wrap points are the block pool's allocation/free/eviction/reset methods (the mediation
point), each single-type manager's ``allocate_new_blocks`` (which group an allocation belongs
to), ``Scheduler.schedule`` (the step's snapshot and copy-on-write copies),
``GPUModelRunner.prepare_attn`` (the block tables and slot mappings the kernels consume) and
``Scheduler.update_from_output`` (where a step's dispatches complete; under asynchronous
scheduling a step's dispatches stay open across the next step and the core's stream-order rule
judges them). Newer serving-system features (speculative decoders, KV offload/tiering, the
Mooncake Store, hybrid recurrent state, NIXL P/D) are opt-in and wrapped at their own points.
Hooks let an experiment observe or perturb the state at each point.

With ``gpu_validate=True`` the dispatch-time validators also run on the device at
``prepare_attn`` (slot-map and dense-table validators launched before the step's kernels, read
back at the next ``update_from_output``); they cover the full-attention group, the
recurrent-state groups being checked on the host. With ``tags=True`` the tag tier runs at the
same points with or without the validators; its reports carry the site ``gpu-tags:<dispatch id>``.
"""

from __future__ import annotations

import cProfile
import hashlib
import inspect
import json
import os
import pstats
import re
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from kvguard.adapters.streams import current_stream_handle
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    DispatchDescriptor,
    DispatchRow,
    ExternalFill,
    ExternalRead,
    ExternalWrite,
    Ledger,
    PoolGeometry,
    Reporter,
    RequestSnapshot,
    SlotMap,
    Snapshot,
)
from kvguard.core.violations import (
    OutcomeKind,
    Policy,
    Report,
    Unsupported,
    integration_error,
    unsupported,
)
from kvguard.core.worker import LedgerWorker

from .snapshot_cache import SnapshotCache
from .translate import (
    BlockFacts,
    RequestFacts,
    build_dispatches,
    build_positioned_dispatches,
    build_snapshot,
    build_state_dispatch,
    cow_copy,
    dense_table,
    mapping_tables,
    mediated_frees,
    state_columns,
    window_start,
)

DRAFT_POOL = "draft_kv"

POOL = "gpu"
"""The attention group's core pool: the full-attention group, or the first sliding-window group
of a model without one."""


def mtp_pool(module_index: int) -> str:
    """Logical draft pool for an MTP module sharing the target's physical allocation."""
    return f"mtp_draft_{module_index}"


def _grouped_gpu_blocks(spec: Any, group_ids: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    """Split a managed-offload GPU specification by its authoritative group metadata."""
    block_ids = tuple(int(v) for v in np.asarray(spec.block_ids).reshape(-1))
    group_sizes = tuple(int(v) for v in spec.group_sizes)
    block_indices = tuple(int(v) for v in spec.block_indices)
    if (
        len(group_sizes) != len(group_ids)
        or len(block_indices) != len(group_ids)
        or sum(group_sizes) != len(block_ids)
        or any(size < 0 for size in group_sizes)
    ):
        raise ValueError(
            "invalid grouped GPU transfer specification: "
            f"{group_sizes=}, {block_indices=}, {group_ids=}, blocks={len(block_ids)}"
        )
    cursor = 0
    grouped: dict[int, tuple[int, ...]] = {}
    for group_id, size in zip(group_ids, group_sizes, strict=True):
        grouped[group_id] = block_ids[cursor : cursor + size]
        cursor += size
    return grouped


def state_pool(group: int) -> str:
    """The core pool of KV cache group ``group``, a recurrent-state (Mamba) group."""
    return f"state{group}"


def window_pool(group: int) -> str:
    """The core pool of KV cache group ``group``, a sliding-window group beside the ``gpu`` one."""
    return f"swa{group}"


def attention_pool(group: int) -> str:
    """The core pool of an additional full-attention group beside the ``gpu`` anchor."""
    return f"attention{group}"


class _MooncakeStoreProxy:
    """Delegate a native Store handle while observing successful multi-buffer operations."""

    def __init__(
        self,
        target: Any,
        *,
        put: Callable[[Any, Any, Any, Any], None] | None = None,
        get: Callable[[Any, Any, Any, Any], None] | None = None,
        before_put: Callable[[Any, Any, Any], None] | None = None,
        before_get: Callable[[Any, Any, Any], None] | None = None,
    ) -> None:
        self._target = target
        self._put = put
        self._get = get
        self._before_put = before_put
        self._before_get = before_get

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *args, **kwargs):
        if self._before_put is not None:
            self._before_put(keys, addrs, sizes)
        try:
            result = self._target.batch_put_from_multi_buffers(keys, addrs, sizes, *args, **kwargs)
        except Exception:
            if self._put is not None:
                self._put(keys, addrs, sizes, ())
            raise
        if self._put is not None:
            self._put(keys, addrs, sizes, result)
        return result

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, *args, **kwargs):
        if self._before_get is not None:
            self._before_get(keys, addrs, sizes)
        try:
            result = self._target.batch_get_into_multi_buffers(keys, addrs, sizes, *args, **kwargs)
        except Exception:
            if self._get is not None:
                self._get(keys, addrs, sizes, ())
            raise
        if self._get is not None:
            self._get(keys, addrs, sizes, result)
        return result


class _ZeroResourceTracker:
    """Shared tracker for historical vLLM zeroing-ID staging buffers."""

    def _install_zero_resources(self, runner: Any) -> bool:
        zeroer = getattr(runner, "kv_block_zeroer", None)
        if zeroer is None:
            zeroer = getattr(runner, "_kv_block_zeroer", None)
        if zeroer is None or not _zeroer_resources(zeroer):
            return False
        me = self
        orig_zero = zeroer.zero_block_ids

        def zero_block_ids(block_ids, *args, **kwargs):
            me._reap_zero_uses()
            resources = _zeroer_resources(zeroer)
            # An empty request or an uninitialized zeroer launches no asynchronous work.
            if not block_ids or getattr(zeroer, "_meta", None) is None or not resources:
                return orig_zero(block_ids, *args, **kwargs)
            me._zero_seq += 1
            operation = f"vllm-zero-{me._zero_seq}"
            site = f"zero@{me._zero_seq}"
            me.ledger.begin_resource_use(operation, resources, site=site)
            try:
                out = orig_zero(block_ids, *args, **kwargs)
                import torch

                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream())
            except BaseException:
                me.ledger.end_resource_use(operation, site=site)
                raise
            me._zero_pending.append((done, operation))
            me.stats["zero_resource_uses"] += 1
            me.stats["zero_resource_pending_max"] = max(
                me.stats["zero_resource_pending_max"], len(me._zero_pending)
            )
            return out

        zeroer.zero_block_ids = zero_block_ids
        self.stats["zero_resource_wrapped"] += 1
        return True

    def _reap_zero_uses(self, *, wait: bool = False) -> None:
        """End zeroing-resource uses whose actual GPU consumers have completed."""
        pending = []
        for event, operation in self._zero_pending:
            if wait:
                event.synchronize()
                complete = True
            else:
                complete = event.query()
            if complete:
                self.ledger.end_resource_use(operation, site="zero-complete")
                self.stats["zero_resource_completions"] += 1
            else:
                pending.append((event, operation))
        self._zero_pending = pending


class VllmAttachment(_ZeroResourceTracker):
    """Wrappers installed on one engine, feeding one ledger."""

    def __init__(
        self,
        llm_or_engine: Any,
        reporter: Reporter | None = None,
        *,
        after_prepare_attn: Callable[[Any, Any, Any], None] | None = None,
        after_dispatch: Callable[[Any, Any, Any], None] | None = None,
        audit_every: int = 64,
        gpu_validate: bool = False,
        tags: bool = False,
        nvtx: bool = False,
        tables: str = "device",
        delta_listing: bool = True,
        checker: str = "inline",
        stream_epochs: bool = True,
        profile: str | None = None,
        profile_steps: tuple[int, int] = (100, 400),
        snapshot: str = "incremental",
    ):
        engine = getattr(
            getattr(llm_or_engine, "llm_engine", llm_or_engine), "engine_core", llm_or_engine
        )
        engine = getattr(engine, "engine_core", engine)  # InprocClient -> EngineCore
        self.engine = engine
        self.scheduler = engine.scheduler
        self.kvm = self.scheduler.kv_cache_manager
        self.pool = self.kvm.block_pool
        self.reporter = reporter or Reporter(fail_fast=True)
        self.runner = engine.model_executor.driver_worker.worker.model_runner
        self.managers = tuple(self.kvm.coordinator.single_type_managers)
        self.groups = self._classify_groups()
        """Per KV cache group: kind, core pool, block size, layers and (Mamba) cache mode."""
        self.kv_gid: int = next(
            g["group"] for g in self.groups if g["pool"] == POOL and not g.get("draft", False)
        )
        self.draft_gids: tuple[int, ...] = tuple(
            g["group"] for g in self.groups if g.get("draft", False)
        )
        self.mtp_gids: tuple[int, ...] = ()
        self.mtp_pools: tuple[str, ...] = ()
        self.mtp_layers: tuple[str, ...] = ()
        self.mtp_shared = False
        self.state_gids: tuple[int, ...] = tuple(
            g["group"] for g in self.groups if g["kind"] == "state" and not g.get("draft", False)
        )
        self.window_gids: tuple[int, ...] = tuple(
            g["group"]
            for g in self.groups
            if g["kind"] == "window" and g["pool"] != POOL and not g.get("draft", False)
        )
        """The sliding-window groups beside the ``gpu`` group (their own pools)."""
        self.aux_attention_gids: tuple[int, ...] = tuple(
            g["group"]
            for g in self.groups
            if g["kind"] in ("attention", "window")
            and g["group"] != self.kv_gid
            and not g.get("draft", False)
        )
        """Every additional target attention group, full or sliding-window."""
        self.window_of: dict[int, int | None] = {
            g["group"]: g.get("window") for g in self.groups if g["kind"] != "state"
        }
        """Per attention group, its sliding window (None for full attention)."""
        self.extra_of: dict[int, int] = {
            g["group"]: int(g.get("extra_retained", 0) or 0)
            for g in self.groups
            if g["kind"] != "state"
        }
        self.state_mode: str = next((g["mode"] for g in self.groups if g["kind"] == "state"), "")
        self.hybrid = len(self.groups) > 1
        self.pool_of_group: dict[int, str] = {g["group"]: g["pool"] for g in self.groups}
        self.manager = self.managers[self.kv_gid]
        self.block_size = self.manager.block_size
        spec_config = getattr(self.runner, "speculative_config", None)
        self.speculative = spec_config is not None
        self.eagle3 = False
        self.standalone_draft = False
        self.dflash = False
        self.dflash_separate = False
        self.dflash_gid = self.kv_gid
        self.dflash_pool = DRAFT_POOL
        self.dflash_graphs = False
        self.multi_module_mtp = False
        self.eagle_lookahead = 0
        self.eagle_fault = os.environ.get("KVGUARD_VLLM_EAGLE_FAULT", "")
        self.eagle_fault_done = False
        if self.eagle_fault not in ("", "draft_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_EAGLE_FAULT {self.eagle_fault!r}")
        self.dflash_fault = os.environ.get("KVGUARD_VLLM_DFLASH_FAULT", "")
        self.dflash_fault_done = False
        if self.dflash_fault not in ("", "draft_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_DFLASH_FAULT {self.dflash_fault!r}")
        self.standalone_fault = os.environ.get("KVGUARD_VLLM_STANDALONE_FAULT", "")
        self.standalone_fault_done = False
        if self.standalone_fault not in ("", "draft_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_STANDALONE_FAULT {self.standalone_fault!r}")
        self.mtp_fault = os.environ.get("KVGUARD_VLLM_MTP_FAULT", "")
        self.mtp_fault_done = False
        if self.mtp_fault not in ("", "draft_early_free", "draft_graph_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_MTP_FAULT {self.mtp_fault!r}")
        self.offload_fault = os.environ.get("KVGUARD_VLLM_OFFLOAD_FAULT", "")
        self.offload_fault_done = False
        if self.offload_fault not in ("", "swa_load_early_free", "state_load_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_OFFLOAD_FAULT {self.offload_fault!r}")
        self.pd_fault = os.environ.get("KVGUARD_VLLM_PD_FAULT", "")
        self.pd_fault_done = False
        if self.pd_fault not in ("", "content_digest"):
            raise ValueError(f"unknown KVGUARD_VLLM_PD_FAULT {self.pd_fault!r}")
        self.mooncake_fault = os.environ.get("KVGUARD_VLLM_MOONCAKE_FAULT", "")
        self.mooncake_fault_done = False
        if self.mooncake_fault not in ("", "load_early_free", "store_early_free"):
            raise ValueError(f"unknown KVGUARD_VLLM_MOONCAKE_FAULT {self.mooncake_fault!r}")
        # Opt-in (off by default): cross-check each worker block table (the GPU mirror of the
        # manager's req_to_blocks) against the manager mapping, flagging a row that still names a
        # block the manager has released. The kernels consume a gathered copy that masks such an
        # entry, so it is a latent mirror divergence; the gate keeps ordinary runs quiet.
        self.worker_table_check = os.environ.get("KVGUARD_VLLM_WORKER_TABLE", "0") == "1"
        if self.speculative:
            use_ngram = getattr(spec_config, "use_ngram", None)
            method = getattr(spec_config, "method", type(spec_config).__name__)
            self.eagle3 = method == "eagle3"
            if self.eagle3:
                speculator = getattr(self.runner, "speculator", None)
                draft_layers = tuple(getattr(speculator, "draft_attn_layer_names", ()))
                if (
                    self.hybrid
                    or type(speculator).__name__ != "EagleSpeculator"
                    or not draft_layers
                ):
                    self._reject(
                        f"EAGLE3 over {len(self.groups)} scheduler group(s), speculator "
                        f"{type(speculator).__name__} with {len(draft_layers)} KV layer(s); "
                        "the adapter requires one target group and Model Runner V2's standard "
                        "EAGLE speculator"
                    )
                self.eagle_lookahead = int(spec_config.num_speculative_tokens)
                if gpu_validate:
                    self._reject("EAGLE3 with device validators; the host-only port is measured")
            elif method == "draft_model":
                speculator = getattr(self.runner, "speculator", None)
                draft_layers = self._draft_layer_names(speculator)
                layer_gid = {
                    layer: group["group"]
                    for group in self.groups
                    for layer in self.kvm.kv_cache_config.kv_cache_groups[
                        group["group"]
                    ].layer_names
                }
                draft_layer_gids = {
                    layer_gid[layer] for layer in draft_layers if layer in layer_gid
                }
                compilation_config = getattr(
                    getattr(self.runner, "vllm_config", None), "compilation_config", None
                )
                cg_mode = getattr(compilation_config, "cudagraph_mode", None)
                graphs = getattr(cg_mode, "name", str(cg_mode)) != "NONE"
                if (
                    self.hybrid
                    or type(speculator).__name__ != "StandaloneARSpeculator"
                    or not draft_layers
                    or set(draft_layers) - set(layer_gid)
                    or draft_layer_gids != {self.kv_gid}
                    or graphs
                ):
                    self._reject(
                        f"standalone draft model over {len(self.groups)} scheduler group(s), "
                        f"speculator {type(speculator).__name__} with {len(draft_layers)} KV "
                        f"layer(s), graphs={getattr(cg_mode, 'name', str(cg_mode))}; the bounded "
                        "port requires one shared target/draft group and eager execution"
                    )
                if gpu_validate:
                    self._reject(
                        "standalone draft model with device validators; the host-only port is "
                        "measured"
                    )
                self.standalone_draft = True
                self.eagle_lookahead = int(
                    getattr(getattr(self.runner, "vllm_config", None), "num_lookahead_tokens", 0)
                )
                if self.eagle_lookahead != int(spec_config.num_speculative_tokens):
                    self._reject(
                        f"standalone draft lookahead {self.eagle_lookahead} differs from its "
                        f"{int(spec_config.num_speculative_tokens)} configured slots (the first "
                        "slot carries the correction token)"
                    )
            elif method == "dflash":
                speculator = getattr(self.runner, "speculator", None)
                draft_layers = self._draft_layer_names(speculator)
                layer_gid = {
                    layer: group["group"]
                    for group in self.groups
                    for layer in self.kvm.kv_cache_config.kv_cache_groups[
                        group["group"]
                    ].layer_names
                }
                draft_layer_gids = {
                    layer_gid[layer] for layer in draft_layers if layer in layer_gid
                }
                shared_draft = bool(draft_layers) and draft_layer_gids == {self.kv_gid}
                separate_draft = (
                    bool(draft_layers)
                    and bool(self.draft_gids)
                    and draft_layer_gids <= set(self.draft_gids)
                )
                scheduler_config = getattr(
                    getattr(self.runner, "vllm_config", None), "scheduler_config", None
                )
                compilation_config = getattr(
                    getattr(self.runner, "vllm_config", None), "compilation_config", None
                )
                cg_mode = getattr(compilation_config, "cudagraph_mode", None)
                query_cg = getattr(speculator, "query_cudagraph_manager", None)
                query_cg_mode = getattr(query_cg, "cudagraph_mode", None)
                graphs = getattr(cg_mode, "name", str(cg_mode)) != "NONE"
                async_scheduling = bool(getattr(scheduler_config, "async_scheduling", False))
                if (
                    type(speculator).__name__ not in ("DFlashSpeculator", "DFlash2Speculator")
                    or not draft_layers
                    or set(draft_layers) - set(layer_gid)
                    or not (shared_draft or separate_draft)
                    or len(draft_layer_gids) != 1
                    or (
                        graphs
                        and (
                            type(query_cg).__name__ != "DFlashCudaGraphManager"
                            or getattr(query_cg_mode, "name", str(query_cg_mode))
                            != "FULL_DECODE_ONLY"
                        )
                    )
                ):
                    self._reject(
                        f"DFlash over {len(self.groups)} scheduler group(s), speculator "
                        f"{type(speculator).__name__} with {len(draft_layers)} KV layer(s), "
                        f"async={async_scheduling}, "
                        f"graphs={getattr(cg_mode, 'name', str(cg_mode))}/"
                        f"{getattr(query_cg_mode, 'name', str(query_cg_mode))}; the bounded port "
                        "requires one shared target/draft group or one independent draft group "
                        "and DFlash's full-decode graph manager when graphs are enabled"
                    )
                if gpu_validate:
                    self._reject("DFlash with device validators; the host-only port is measured")
                self.dflash = True
                self.dflash_separate = separate_draft
                self.dflash_gid = next(iter(draft_layer_gids))
                self.dflash_pool = (
                    self.pool_of_group[self.dflash_gid] if separate_draft else DRAFT_POOL
                )
                self.dflash_graphs = graphs
                self.eagle_lookahead = int(
                    getattr(getattr(self.runner, "vllm_config", None), "num_lookahead_tokens", 0)
                )
                if self.eagle_lookahead != int(spec_config.num_speculative_tokens) + 1:
                    self._reject(
                        f"DFlash lookahead {self.eagle_lookahead} does not include the bonus plus "
                        f"{int(spec_config.num_speculative_tokens)} draft tokens"
                    )
            elif method == "mtp":
                speculator = getattr(self.runner, "speculator", None)
                # The speculator exposes its draft layers as a set, while multi-module MTP selects
                # its numerically named ModuleDict layers in ascending order. Scheduler groups may
                # have a different order, so bind modules to groups by the layer name, not by gid.
                draft_layers = tuple(sorted(getattr(speculator, "draft_attn_layer_names", ())))
                draft_layer_gid = {
                    layer: group["group"]
                    for group in self.groups
                    for layer in self.kvm.kv_cache_config.kv_cache_groups[
                        group["group"]
                    ].layer_names
                    if layer in draft_layers
                }
                draft_architectures = tuple(
                    getattr(
                        getattr(
                            getattr(spec_config, "draft_model_config", None), "hf_config", None
                        ),
                        "architectures",
                        (),
                    )
                    or ()
                )
                transformers_mtp = draft_architectures in (
                    ("TransformersForMTP",),
                    ("TransformersMoEForMTP",),
                )
                separate_layout = (
                    bool(self.draft_gids)
                    and set(draft_layers) == set(draft_layer_gid)
                    and set(draft_layer_gid.values()) == set(self.draft_gids)
                )
                shared_layout = (
                    transformers_mtp
                    and not self.draft_gids
                    and set(draft_layers) == set(draft_layer_gid)
                    and set(draft_layer_gid.values()) == {self.kv_gid}
                )
                if (
                    type(speculator).__name__ != "MultiModuleMTPSpeculator"
                    or draft_architectures
                    not in (
                        ("Step3p5MTP",),
                        ("TransformersForMTP",),
                        ("TransformersMoEForMTP",),
                    )
                    or not (separate_layout or shared_layout)
                    or len(draft_layers) != int(spec_config.num_speculative_tokens)
                    or any(
                        group["kind"] not in ("attention", "window")
                        for group in self.groups
                        if group.get("draft", False)
                    )
                ):
                    self._reject(
                        f"MTP architecture {draft_architectures!r}, speculator "
                        f"{type(speculator).__name__}, {len(self.draft_gids)} independent draft "
                        f"group(s), shared={shared_layout}, "
                        f"and {len(draft_layers)} draft layer(s); the adapter requires "
                        "Step3p5MTP or Transformers-backend MTP on Model Runner V2's "
                        "multi-module speculator"
                    )
                self.mtp_gids = tuple(draft_layer_gid[layer] for layer in draft_layers)
                self.mtp_layers = draft_layers
                self.mtp_shared = shared_layout
                self.mtp_pools = tuple(
                    mtp_pool(index) if shared_layout else self.pool_of_group[gid]
                    for index, gid in enumerate(self.mtp_gids)
                )
                cg_mode = getattr(
                    getattr(getattr(self.runner, "vllm_config", None), "compilation_config", None),
                    "cudagraph_mode",
                    None,
                )
                self.mtp_cuda_graphs = getattr(cg_mode, "name", str(cg_mode)) != "NONE"
                if self.mtp_cuda_graphs and not callable(
                    getattr(getattr(speculator, "cudagraph_manager", None), "run_fullgraph", None)
                ):
                    self._reject(
                        "multi-module MTP CUDA graphs without an observable full-graph replay point"
                    )
                if gpu_validate:
                    self._reject(
                        "multi-module MTP with device validators; the host-only port is measured"
                    )
                self.multi_module_mtp = True
                self.eagle_lookahead = int(spec_config.num_speculative_tokens)
            elif self.hybrid or not callable(use_ngram) or not use_ngram():
                self._reject(
                    f"speculative method {method!r} over {len(self.groups)} KV cache group(s); "
                    "the adapter models n-gram, bounded EAGLE3/DFlash/standalone-draft, and "
                    "multi-module MTP"
                )
        self.offload_kind: str | None = None
        self.offload_tier_types: tuple[str, ...] = ()
        self.pd_kind: str | None = None
        self.pd: Any | None = None
        self.mooncake_kind: str | None = None
        self.mooncake: Any | None = None
        self.offload = self._classify_offload()
        self.core_ledger = Ledger(self.reporter)
        self.worker: LedgerWorker | None = None
        if checker == "thread":
            # The ledger's events applied on a background thread (`kvguard/core/worker.py`);
            # the engine's thread pays the extraction and the enqueue only.
            if gpu_validate:
                raise ValueError(
                    "checker='thread' with gpu_validate: the device mirror reads the ledger's "
                    "shadow on the engine's thread, which the worker does not serve"
                )
            self.worker = LedgerWorker(self.core_ledger)
        elif checker != "inline":
            raise ValueError(f"unknown checker {checker!r}")
        self.checker = checker
        self.ledger: Any = self.worker or self.core_ledger
        self.stream_epochs = stream_epochs
        """Declare the engine's current stream in every descriptor and snapshot; under
        asynchronous scheduling the frees of a finished request land while the next step runs and
        the stream-order rule judges their reuse."""
        null = self.pool.null_block.block_id
        # The groups' pools share vLLM's one BlockPool: the core audits their partition once.
        allocator = "block_pool" if self.hybrid else None
        self.geometry = PoolGeometry(
            POOL,
            num_blocks=len(self.pool.blocks),
            block_size=self.block_size,
            reserved_blocks=frozenset({null}),
            allocator=allocator,
        )
        self.ledger.register_pool(self.geometry)
        if self._shared_draft_pool():
            draft_layers = self._draft_layer_names(self.runner.speculator)
            self.ledger.register_pool(
                PoolGeometry(
                    DRAFT_POOL,
                    num_blocks=len(self.pool.blocks),
                    block_size=self.block_size,
                    layers=max(len(draft_layers), 1),
                    reserved_blocks=frozenset({null}),
                    rewritable=self.dflash or self.standalone_draft,
                )
            )
        if self.mtp_shared:
            for pool in self.mtp_pools:
                self.ledger.register_pool(
                    PoolGeometry(
                        pool,
                        num_blocks=len(self.pool.blocks),
                        block_size=self.block_size,
                        layers=1,
                        reserved_blocks=frozenset({null}),
                        rewritable=True,
                    )
                )
        for g in self.groups:
            if g["kind"] in ("attention", "window") and g["pool"] != POOL:
                self.ledger.register_pool(
                    PoolGeometry(
                        g["pool"],
                        num_blocks=len(self.pool.blocks),
                        block_size=int(g["block_size"]),
                        layers=max(int(g["layers"]), 1),
                        reserved_blocks=frozenset({null}),
                        allocator=allocator,
                        # Multi-module MTP re-prefills rejected positions in
                        # each draft layer's private KV workspace.  Those
                        # updates are writes to initialized slots by design;
                        # reads must still observe an initialized generation.
                        rewritable=bool(g.get("draft", False)),
                    )
                )
            if g["kind"] == "state":
                # One block per state column; the state is rewritten in place each step.
                self.ledger.register_pool(
                    PoolGeometry(
                        g["pool"],
                        num_blocks=len(self.pool.blocks),
                        block_size=1,
                        layers=max(int(g["layers"]), 1),
                        components=("conv", "ssm"),
                        reserved_blocks=frozenset({null}),
                        rewritable=True,
                        allocator=allocator,
                    )
                )
        self._mutated: dict[int, set[str]] = {}
        """Per group, the requests whose block lists were changed in place since the last
        snapshot (the managers' retirements), for the caches' selective rebuild."""
        self.group_of: dict[int, int] = {}
        """Which KV cache group holds each non-free block (its allocating manager's), kept from
        the mediated allocations and frees: a block id is physical and belongs to one group at
        a time; the snapshot lists it in that group's pool and counts it free for the others."""
        self._alloc_group: int | None = None
        self._deferred_rows: dict[tuple[str, str], RequestSnapshot] = {}
        self._deferred_counts: dict[tuple[str, str], Counter[int]] = {}
        self._deferred_by_block: dict[int, set[tuple[str, str]]] = {}
        self._simple_store_owners: dict[tuple[str, int], str] = {}
        self._draining_deferred = False
        """Request mappings popped while a connector write is still in flight: vLLM removes the
        rows from ``req_to_blocks`` but retains the blocks' references until the producing GPU
        step completes, so keep the rows in the snapshot for that interval."""
        self._offload_filled: set[int] = set()
        """CPU-offload destinations known to contain a complete block, until reallocation."""
        self._offload_filled_pools: set[tuple[str, int]] = set()
        """The logical pool attribution of completed managed multi-group loads."""
        self._pd_pending: dict[str, tuple[int, tuple[tuple[int, int], ...]]] = {}
        """NIXL pull request -> local frontier and destination block/fill pairs."""
        self._pd_send_dispatches: dict[str, tuple[str, ...]] = {}
        """Producer request -> source-read epochs, open until ``finished_sending``."""
        self._pd_send_seq = 0
        self._mooncake_events: deque[dict[str, Any]] = deque()
        """Native Store completions appended by transfer threads and drained on the engine thread."""
        self._mooncake_pending: dict[str, tuple[int, int, dict[str, int]]] = {}
        """Request -> local frontier, remote frontier and the last destination per logical pool."""
        self._mooncake_store_owners: dict[tuple[str, int], str] = {}
        """Last producer identity for a physical block retained only by a Store transfer pin."""
        pd_content_dir = os.environ.get("KVGUARD_VLLM_PD_MANIFEST_DIR")
        self._pd_content_dir = Path(pd_content_dir) if pd_content_dir else None
        if self._pd_content_dir is not None:
            self._pd_content_dir.mkdir(parents=True, exist_ok=True)
        self._pd_content_expected: dict[str, tuple[tuple[int, str], ...]] = {}
        """Optional E2E source-page digests paired with pending decoder destinations."""
        self._offload_recompute_tails: dict[str, tuple[int, dict[str, int], bool]] = {}
        """Full-hit tails vLLM invalidated for its one-token logits replay, as
        ``(position, concrete pool blocks, admitted)``. Admission waits for the first snapshot and
        requires the block to have exactly one mapper, so earlier sharing stays a real violation."""
        self._offload_pending: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._offload_pending_by_req: dict[str, int] = {}
        self._offload_flushed: set[int] = set()
        """Asynchronous loads from schedule metadata until the worker reports completion."""
        self._offload_store_jobs: dict[int, tuple[str, tuple[int, ...]]] = {}
        self._offload_store_epochs: set[int] = set()
        self._offload_store_dispatches: dict[int, tuple[str, ...]] = {}
        self._offload_store_events: dict[int, Any] = {}
        self._offload_flushing: set[int] = set()
        self._offload_transfer_worker: Any = None
        """GPU-to-CPU jobs known from scheduler metadata and reads open on the worker."""
        self._executing_output: Any = None
        """The scheduler output the worker thread is executing (``runner.execute_model``)."""
        self._managed_offload_routes: dict[int, dict[str, tuple[int, ...]]] = {}
        """Managed load/store job ids to the logical pools named by ``group_sizes``."""
        self._managed_offload_store_generations: dict[int, dict[tuple[str, int], int]] = {}
        """Store-source generations captured when managed metadata publishes a job: with GPU
        prefix caching off, a finished request's blocks may be freed while the offloader fence
        retains their bytes, so matching the generation at submission distinguishes that retained-
        free source from an unsafe reuse."""
        self.offload_events: deque[dict[str, Any]] | None = (
            deque(maxlen=4096) if os.environ.get("KVGUARD_VLLM_TRACE_OFFLOAD", "0") == "1" else None
        )
        """Optional bounded transfer metadata for discovery-only request attribution."""
        self.dflash_events: deque[dict[str, Any]] | None = (
            deque(maxlen=1024) if os.environ.get("KVGUARD_VLLM_TRACE_DFLASH", "0") == "1" else None
        )
        """Optional bounded metadata for DFlash writes that touch published draft blocks."""
        self.standalone_events: deque[dict[str, Any]] | None = (
            deque(maxlen=1024)
            if os.environ.get("KVGUARD_VLLM_TRACE_STANDALONE", "0") == "1"
            else None
        )
        """Optional bounded metadata for standalone-draft writes to published blocks."""
        self.mtp_events: deque[dict[str, Any]] | None = (
            deque(maxlen=1024) if os.environ.get("KVGUARD_VLLM_TRACE_MTP", "0") == "1" else None
        )
        """Optional bounded metadata for multi-module MTP writes to published blocks."""
        self._simple_pending_loads: dict[int, tuple[tuple[str, tuple[int, ...], int], ...]] = {}
        """SimpleCPU event -> request, GPU destinations and exact loaded frontier."""
        self._simple_completed_load_pins: Counter[int] = Counter()
        """DMA-complete load blocks still pinned until scheduler-side cleanup."""
        self._published_blocks: set[tuple[str, int]] = set()
        self._normalized_cache: dict[tuple[str, int], tuple[Any, Any, Any, Any]] = {}
        """Per listed block: the snapshot object last normalized, its normalized object, and the
        pool block's hash and hash boundary it was normalized under."""
        self._normalized_external: tuple[set, set, set] = (set(), set(), set())
        """Blocks already observed immutable, until their next mediated allocation."""
        self._zeroed_blocks: set[tuple[str, int]] = set()
        """Attention blocks whose current generation was physically cleared before use."""
        self._cow_pins: Counter[int] = Counter()
        """Extra physical references retaining CoW endpoints until their worker copy runs."""
        self._cow_cache_blocks: set[tuple[str, int]] = set()
        """CoW destinations that hold a prefix-cache copy rather than a request mapping."""
        self._cow_request_blocks: set[tuple[str, int]] = set()
        """CoW destinations privately mapped by the request that will fill their tail."""
        self._cow_deferred_batches: set[int] = set()
        self._deferred_release_kinds: deque[tuple[bool, int]] = deque()
        self._releasing_cow_pins = False
        """Exact scheduler deferred-free entries that release CoW execution pins."""
        self.seq = 0
        self.open: dict[int, list[str]] = {}
        """The open dispatch ids per step, keyed by the identity of the step's
        ``SchedulerOutput``: completed when that step's output is processed, which under
        asynchronous scheduling (the batch queue) is one step behind the latest scheduled."""
        self._step_key: int | None = None
        self._speculative_steps: dict[int, dict[str, int]] = {}
        self._speculative_caps: dict[str, tuple[int, int]] = {}
        self._speculative_retained: dict[tuple[str, str], int] = {}
        """Target positions actually written by each in-flight speculative step: with async
        scheduling the CPU request may still include rejected draft tokens, so a step's last
        executed write frontier is the conservative computed claim until its output is processed."""
        self._zero_pending: list[tuple[Any, str]] = []
        """Completion events and ledger operations for mutable zeroing-ID buffers, which stay in
        use until the nonblocking copy and zeroing kernel finish, not when ``zero_block_ids``
        returns."""
        self._zero_seq = 0
        self._eagle_dispatch_seq = 0
        self._eagle_prepare_lookahead: int | None = None
        self._eagle_prepared_batches: dict[int, int] = {}
        """EAGLE descriptors are published with the target batch, before its GPU job starts:
        under async structured decoding, publishing only from ``propose`` would land the host
        callback after an optimistic release even though target and draft are one stream job."""
        self._dflash_dispatch_seq = 0
        self._dflash_input_batch: Any | None = None
        self._dflash_frontiers: dict[str, int] = {}
        """Dense initialized draft frontier from the last concrete draft execution."""
        self._dflash_pending_writes: dict[str, set[int]] = {}
        """Concrete writes above a gap, retained until earlier positions fill it."""
        self._standalone_dispatch_seq = 0
        self._standalone_input_batch: Any | None = None
        self._standalone_dummy_run = False
        self._standalone_calls = 0
        """Draft forwards of the current standalone proposal (0: the expanded prefill)."""
        self._standalone_scratch: dict[str, tuple[int, int]] = {}
        """Per request, the positions ``[from, to)`` its last proposal wrote beyond the target's
        committed frontier: the correction slot and the multi-step draft positions, rewritten by
        the request's next draft forward and never committed content."""
        self._mtp_dispatch_seq = 0
        self._mtp_input_batch: Any | None = None
        self._mtp_frontiers: dict[tuple[str, str], int] = {}
        """Last concrete write frontier per multi-module MTP pool and request: each module has an
        independent cache, so its snapshot claim is capped by the positions observed at its real
        forward rather than the scheduler's optimistic count."""
        self.stats: Counter = Counter()
        self.timings: Counter = Counter()
        self.after_prepare_attn = after_prepare_attn
        self.after_dispatch = after_dispatch
        # Every ``audit_every``-th snapshot walks the whole pool and free queue (the P1 partition
        # audit and leak checks); the others list only the referenced and cached blocks, at a cost
        # proportional to the live state instead of the pool size.
        self.audit_every = max(1, int(audit_every))
        if snapshot not in ("incremental", "full"):
            raise ValueError("snapshot must be 'incremental' or 'full'")
        if snapshot == "full" and self.hybrid:
            raise ValueError("snapshot='full' reads one KV cache group; use 'incremental'")
        self.snapshot_mode = snapshot
        """``incremental``: the snapshot's parts are kept between steps and an ordinary step reads
        only what the pool's mediated paths touched; ``full``: every step reads every live
        request's block list and held block's facts. Audited steps walk the free queue in both."""
        self.caches: dict[int, SnapshotCache] = {}
        for g in self.groups:
            gid = g["group"]
            member = self._member(gid) if self.hybrid else None
            window = (int(g["window"]), self.extra_of[gid]) if g["kind"] == "window" else None
            self.caches[gid] = SnapshotCache(
                g["pool"],
                int(g["block_size"]),
                self.stats,
                member=member,
                columns=g["kind"] == "state",
                rebuild=g["kind"] != "attention",
                window=window,
            )
        self.cache = self.caches[self.kv_gid]
        if tables not in ("device", "mapping"):
            raise ValueError("tables must be 'device' or 'mapping'")
        self.tables = tables
        self.delta_listing = delta_listing
        """Ordinary (unaudited) snapshots list the pools as their change since the previous one
        where possible, so normalization and the ledger's observation scale with the change."""
        """``device``: the host dispatch checks the gathered device block table and slot map
        copied back (one synchronizing copy per step); ``mapping``: it checks tables derived from
        the KV manager's block lists and the scheduled positions, with no device copy."""
        self._columns: list[tuple[int, int]] = []
        """The state columns derived for the last dispatch, per batch row (hybrid models)."""
        self.validators = None
        self.mirror = None
        self.mirrors: dict[int, Any] = {}
        """Per attention-like group, the device mirror of its pool (``gpu_validate``)."""
        if gpu_validate:
            from kvguard.gpu.validate import Validators

            self.validators = Validators(timing=True)
            self.mirror = self.validators.mirror(self.ledger, POOL)
            self.mirrors[self.kv_gid] = self.mirror
            for g in (*self.aux_attention_gids, *self.state_gids):
                # One mirror per pool over the same block ids; the owner ids are shared.
                self.mirrors[g] = self.validators.mirror(
                    self.ledger, self.pool_of_group[g], self.mirror.owners
                )
        self.tags = None
        """The tag tier over the attention-like groups (``tags=True``; ``kvguard.adapters.vllm.tags``)."""
        if tags:
            from .tags import VllmTags

            self.tags = VllmTags(self)
        self.nvtx = None
        """The Compute Sanitizer baseline's NVTX memory registrations (``nvtx=True``;
        ``kvguard.gpu.nvtxmem``): the attention pool's tensors as heaps, its blocks as regions."""
        if nvtx:
            from kvguard.gpu.nvtxmem import NvtxMemory

            if self.hybrid:
                raise RuntimeError("the NVTX baseline covers the single attention group only")
            caches = [c for c in getattr(self.runner, "kv_caches", ()) if c.ndim >= 2]
            if not caches:
                raise RuntimeError("the NVTX baseline found no KV cache tensors on the runner")
            self.nvtx = NvtxMemory(self.ledger)
            self.nvtx.add_pool(
                POOL, caches, self.geometry.num_blocks, reserved=self.geometry.reserved_blocks
            )
            self.stats["nvtx_lanes"] = len(self.nvtx.lanes[POOL])
        self._install()
        self.profile_path = profile
        self.profile_steps = profile_steps
        self.profiler: cProfile.Profile | None = None
        self._profile_written = False
        if profile:
            # The attachment's two entry points run under cProfile for the steps in the window;
            # the stats are written at finish() (a diagnostic for the host-cost work).
            self.profiler = cProfile.Profile()
            for name in ("_snapshot", "_dispatch"):
                setattr(self, name, self._profiled(getattr(self, name)))

    # ------------------------------------------------------------------ groups
    def _classify_offload(self) -> Any | None:
        """Admit the supported CPU offload connectors at an explicit, source-derived boundary."""
        connector = getattr(self.scheduler, "connector", None)
        if connector is None:
            return None
        connector_kind = type(connector).__name__
        if connector_kind == "NixlPullConnector":
            pd = getattr(connector, "connector_scheduler", None)
            config = getattr(connector, "kv_transfer_config", None)
            role = getattr(config, "kv_role", None)
            extra = getattr(config, "kv_connector_extra_config", {}) or {}
            supported = (
                type(pd).__name__ == "NixlPullConnectorScheduler"
                and len(self.groups) == 1
                and self.groups[0]["kind"] == "attention"
                and not self.hybrid
                and not self.eagle3
                and not self.dflash
                and not self.standalone_draft
                and not self.multi_module_mtp
                and role in ("kv_producer", "kv_consumer")
                and getattr(config, "kv_buffer_device", None) == "cuda"
                and not bool(extra.get("bidirectional_kv_xfer", False))
            )
            if not supported:
                self._reject(
                    "NIXL P/D outside the admitted boundary: pull mode, one ordinary "
                    "full-attention group, direct CUDA buffers, distinct producer/consumer "
                    "roles, and no speculative, hybrid or bidirectional transfer"
                )
            self.pd_kind = "nixl_pull"
            self.pd = pd
            return None
        if connector_kind == "MooncakeStoreConnector":
            mooncake = getattr(connector, "connector_scheduler", None)
            store_group_ids = tuple(int(g) for g in getattr(mooncake, "_store_group_ids", ()))
            supported_groups = tuple(int(group["group"]) for group in self.groups)
            if (
                type(mooncake).__name__ != "MooncakeStoreScheduler"
                or store_group_ids != supported_groups
                or getattr(mooncake, "kv_role", None) != "kv_both"
                or self.dflash
                or self.standalone_draft
                or self.multi_module_mtp
                or (self.speculative and not self.eagle3)
            ):
                self._reject(
                    "Mooncake Store outside the admitted boundary: rank-local kv_both over "
                    "every full/sliding/recurrent group, optionally with bounded EAGLE3"
                )
            self.mooncake_kind = "store"
            self.mooncake = mooncake
            return None
        if connector_kind == "SimpleCPUOffloadConnector":
            offload = getattr(connector, "scheduler_manager", None)
            if (
                offload is None
                or type(offload).__name__ != "SimpleCPUOffloadScheduler"
                or int(getattr(offload, "block_size", -1)) != self.block_size
                or int(getattr(offload, "fa_gidx", -1)) != self.kv_gid
            ):
                self._reject(
                    "SimpleCPU offload outside the admitted boundary: one full-attention "
                    "anchor whose scheduler blocks match the GPU pool"
                )
            self.offload_kind = "simple"
            return offload
        if connector_kind != "OffloadingConnector":
            self._reject(
                f"KV connector {connector_kind}; the adapter models only the "
                "OffloadingConnector and SimpleCPUOffloadConnector"
            )
        offload = getattr(connector, "connector_scheduler", None)
        config = getattr(offload, "config", None)
        manager = getattr(offload, "manager", None)
        manager_kind = type(manager).__name__
        if manager_kind == "TieringOffloadingManager":
            primary = getattr(manager, "primary_tier", None)
            tiers = tuple(getattr(manager, "secondary_tiers", ()))
            self.offload_tier_types = tuple(type(tier).__name__ for tier in tiers)
            supported_secondary_tiers = {
                "FileSystemTierManager",
                "KVCRSecondaryTierManager",
                "P2PSecondaryTierManager",
            }
            tiering_supported = (
                type(primary).__name__ == "CPUPrimaryTierOffloadingManager"
                and bool(tiers)
                and all(type(tier).__name__ in supported_secondary_tiers for tier in tiers)
            )
        else:
            tiering_supported = False
        group_configs = tuple(getattr(config, "kv_group_configs", ()))
        config_by_gid = {int(getattr(group, "group_idx", -1)): group for group in group_configs}
        supported_groups = tuple(group["group"] for group in self.groups)
        if (
            offload is None
            or (manager_kind != "CPUOffloadingManager" and not tiering_supported)
            or len(group_configs) != len(self.groups)
            or set(config_by_gid) != set(supported_groups)
            or any(
                int(getattr(config_by_gid[gid], "tokens_per_block", -1))
                != int(self.managers[gid].block_size)
                for gid in supported_groups
            )
            or int(getattr(config, "blocks_per_chunk", 0)) < 1
            or int(getattr(config, "num_workers", 0)) != 1
        ):
            self._reject(
                "KV offload outside the admitted CPU boundary: full/sliding-window/recurrent "
                "groups whose transfer metadata matches the scheduler geometry, a positive "
                "number of GPU blocks per chunk, and one worker; TieringOffloadingManager is "
                "bounded to a CPUPrimaryTierOffloadingManager plus built-in filesystem, KVCR "
                "or P2P secondary tiers"
            )
        self.offload_kind = "managed"
        return offload

    def _classify_groups(self) -> list[dict[str, Any]]:
        """The KV cache groups the attachment can model, or ``Unsupported``.

        Full attention, sliding-window attention, and recurrent-state groups are represented as
        independent logical pools over vLLM's shared physical allocator.  Recurrent groups use
        the ``align`` or ``none`` Mamba mode without speculative blocks.
        """
        groups = self.kvm.kv_cache_config.kv_cache_groups
        draft_layers = set(
            self._draft_layer_names(getattr(getattr(self, "runner", None), "speculator", None))
        )
        out: list[dict[str, Any]] = []
        for gid, (manager, group) in enumerate(zip(self.managers, groups, strict=True)):
            spec = group.kv_cache_spec
            raw_kind = type(spec).__name__
            layer_specs = (spec,)
            if raw_kind == "UniformTypeKVCacheSpecs":
                layer_specs = tuple(getattr(spec, "kv_cache_specs", {}).values())
                if not layer_specs:
                    self._reject(f"group {gid} has an empty UniformTypeKVCacheSpecs wrapper")
            layer_kinds = {type(layer_spec).__name__ for layer_spec in layer_specs}
            if layer_kinds and layer_kinds <= {"FullAttentionSpec", "MLAAttentionSpec"}:
                kind = (
                    "MLAAttentionSpec"
                    if layer_kinds == {"MLAAttentionSpec"}
                    else "FullAttentionSpec"
                )
            elif len(layer_kinds) == 1:
                kind = next(iter(layer_kinds))
            else:
                self._reject(
                    f"group {gid} is a {raw_kind} over heterogeneous layer specs "
                    f"{sorted(layer_kinds)}"
                )
            logical_spec = layer_specs[0]
            entry: dict[str, Any] = {
                "group": gid,
                "spec": raw_kind if raw_kind == kind else f"{raw_kind}[{kind}]",
                "manager": type(manager).__name__,
                "block_size": int(manager.block_size),
                "layers": len(group.layer_names),
            }
            group_layers = set(group.layer_names)
            draft_only = bool(group_layers) and bool(draft_layers) and group_layers <= draft_layers
            if not draft_layers:
                draft_only = bool(getattr(spec, "is_draft", False)) or any(
                    bool(getattr(layer_spec, "is_draft", False)) for layer_spec in layer_specs
                )
            if draft_only:
                entry["draft"] = True
            elif group_layers & draft_layers:
                entry["draft_layers"] = len(group_layers & draft_layers)
            if kind in ("FullAttentionSpec", "MLAAttentionSpec"):
                entry.update(kind="attention", pool=POOL)
            elif kind == "SlidingWindowSpec":
                entry.update(
                    kind="window",
                    pool=window_pool(gid),
                    window=int(logical_spec.sliding_window),
                    extra_retained=int(getattr(logical_spec, "extra_retained_tokens", 0) or 0),
                )
            elif kind == "MambaSpec":
                mode = str(getattr(logical_spec, "mamba_cache_mode", "none"))
                spec_blocks = int(getattr(logical_spec, "num_speculative_blocks", 0) or 0)
                if mode not in ("align", "none") or spec_blocks:
                    self._reject(
                        f"group {gid}: Mamba cache mode {mode!r} with {spec_blocks} speculative "
                        "block(s); the adapter models 'align' and 'none' without speculation"
                    )
                entry.update(kind="state", pool=state_pool(gid), mode=mode)
            else:
                self._reject(
                    f"group {gid} is a {kind}; the adapter models full-attention, sliding-window "
                    "and Mamba groups (including MLA's full-attention specialization)"
                )
            out.append(entry)
        attention = [g for g in out if g["kind"] == "attention" and not g.get("draft", False)]
        windows = [g for g in out if g["kind"] == "window" and not g.get("draft", False)]
        if not attention:
            if not windows:
                self._reject(
                    f"no attention group among {len(out)}; the adapter needs one; "
                    f"classified groups: {out!r}"
                )
            # No full-attention group: the first sliding-window group is the "gpu" pool.
            windows[0]["pool"] = POOL
        else:
            # Hybrid planning can split one attention type into several physical groups to
            # balance their layer counts against a small draft/window group.  Their mappings
            # and allocations are independent even though their token semantics are identical.
            for group in attention[1:]:
                group["pool"] = attention_pool(group["group"])
        modes = {g["mode"] for g in out if g["kind"] == "state" and not g.get("draft", False)}
        if len(modes) > 1:
            self._reject(f"Mamba groups in several cache modes {sorted(modes)}")
        return out

    @staticmethod
    def _draft_layer_names(speculator: Any) -> tuple[str, ...]:
        """Return the runtime draft KV layers exposed by current Model Runner V2.

        EAGLE and multi-module MTP publish the names directly on the speculator. DFlash
        publishes them on its draft model because one physical scheduler group contains both
        target and draft layers.
        """
        direct = tuple(getattr(speculator, "draft_attn_layer_names", ()) or ())
        if direct:
            return direct
        getter = getattr(getattr(speculator, "model", None), "get_draft_kv_cache_layer_names", None)
        return tuple(getter() or ()) if callable(getter) else ()

    def _shared_draft_pool(self) -> bool:
        """Whether one physical attention group carries independent target and draft layers."""
        return (
            bool(getattr(self, "eagle3", False))
            or bool(getattr(self, "standalone_draft", False))
            or (
                bool(getattr(self, "dflash", False))
                and not bool(getattr(self, "dflash_separate", False))
            )
        )

    def _shared_logical_pools(self, pool: str) -> tuple[str, ...]:
        """Logical draft pools over the target group's physical block allocation."""
        if pool != POOL:
            return ()
        pools = list(getattr(self, "mtp_pools", ()) if getattr(self, "mtp_shared", False) else ())
        if self._shared_draft_pool():
            pools.append(DRAFT_POOL)
        return tuple(pools)

    def _reject(self, detail: str) -> None:
        report = unsupported("kv_cache_groups", detail)
        self.reporter.emit(report)
        raise Unsupported(report)

    def _member(self, gid: int) -> Callable[[int], bool]:
        group_of = self.group_of

        def member(b: int) -> bool:
            return group_of.get(b) == gid

        return member

    def _forget_allocated_publications(self, pool: str, block_ids) -> None:
        """Drop publication memory for every logical generation an allocation replaces."""
        for block_id in block_ids:
            self._published_blocks.discard((pool, block_id))
            getattr(self, "_cow_cache_blocks", set()).discard((pool, block_id))
            getattr(self, "_cow_request_blocks", set()).discard((pool, block_id))
            for logical_pool in self._shared_logical_pools(pool):
                self._published_blocks.discard((logical_pool, block_id))
                getattr(self, "_cow_cache_blocks", set()).discard((logical_pool, block_id))
                getattr(self, "_cow_request_blocks", set()).discard((logical_pool, block_id))

    def _mark(self, block_ids) -> None:
        """Mark the blocks dirty in the cache of the group each belongs to (every cache for a
        block whose group is not recorded): a block is listed by one group's cache, so the other
        caches need not re-read it."""
        if not self.hybrid:
            self.caches[self.kv_gid].mark(list(block_ids))
            return
        group_of = self.group_of
        by_group: dict[int | None, list[int]] = {}
        for b in block_ids:
            by_group.setdefault(group_of.get(b), []).append(b)
        for g, ids in by_group.items():
            if g is None:
                for cache in self.caches.values():
                    cache.mark(ids)
            else:
                self.caches[g].mark(ids)

    def _release(self, block_ids, site: str) -> None:
        """Free the blocks in the ledger, each in the pool of the group that held it."""
        by_pool: dict[str, list[int]] = {}
        for b in block_ids:
            g = self.group_of.pop(b, None)
            if g is None:
                g = self.kv_gid
                if self.hybrid:
                    self.stats["free_without_group"] += 1
                    self.reporter.emit(
                        integration_error(
                            "free_without_group",
                            f"block {b} was freed without a recorded KV cache group",
                            pool=POOL,
                            block=b,
                            site=site,
                        )
                    )
            by_pool.setdefault(self.pool_of_group[g], []).append(b)
        for pool, ids in by_pool.items():
            self.ledger.free(pool, ids, site=site)
            for logical_pool in self._shared_logical_pools(pool):
                self.ledger.free(logical_pool, ids, site=f"{site}:{logical_pool}")

    # ------------------------------------------------------------------ wrappers
    def _install(self) -> None:
        pool, sched, runner, me = self.pool, self.scheduler, self.runner, self
        if self.pd_kind == "nixl_pull":
            self._install_pd_scheduler()
        if self.mooncake_kind == "store":
            self._install_mooncake_worker()
        if getattr(self, "offload_kind", None) == "simple":
            self._install_simple_offload_scheduler()
        elif self.offload_tier_types:
            self._install_tiering_observer()
        if getattr(self, "offload_kind", None) == "managed":
            self._install_external_load_capacity_guard()

        for gid, manager in enumerate(self.managers):
            orig_alloc = manager.allocate_new_blocks

            def allocate_new_blocks(*args, _orig=orig_alloc, _gid=gid, **kwargs):
                prev = me._alloc_group
                me._alloc_group = _gid
                try:
                    return _orig(*args, **kwargs)
                finally:
                    me._alloc_group = prev

            manager.allocate_new_blocks = allocate_new_blocks

            # Connector hits allocate their GPU destinations through a separate manager entry
            # point before the ordinary token-slot allocation. It still draws from the shared
            # physical block pool and therefore needs the same logical-group attribution. Only
            # some revisions expose this entry point, so wrap it when present.
            if hasattr(manager, "allocate_external_computed_blocks"):
                orig_external_alloc = manager.allocate_external_computed_blocks

                def allocate_external_computed_blocks(
                    *args, _orig=orig_external_alloc, _gid=gid, **kwargs
                ):
                    prev = me._alloc_group
                    me._alloc_group = _gid
                    try:
                        return _orig(*args, **kwargs)
                    finally:
                        me._alloc_group = prev

                manager.allocate_external_computed_blocks = allocate_external_computed_blocks

            # The paths that change a request's list in place (a sliding window's and the
            # align mode's retirements): the request's tuple is rebuilt at the next snapshot,
            # the others extend.
            if gid != self.kv_gid or self.window_of.get(gid) is not None:
                orig_skip = manager.remove_skipped_blocks

                def remove_skipped_blocks(request_id, *args, _orig=orig_skip, _gid=gid, **kwargs):
                    me._mutated.setdefault(_gid, set()).add(request_id)
                    return _orig(request_id, *args, **kwargs)

                manager.remove_skipped_blocks = remove_skipped_blocks

        orig_get_new = pool.get_new_blocks

        def get_new_blocks(num_blocks):
            blocks = orig_get_new(num_blocks)
            ids = [b.block_id for b in blocks]
            me._offload_filled.difference_update(ids)
            g = me._alloc_group
            if g is None:
                g = me.kv_gid
                if me.hybrid:
                    me.stats["alloc_outside_manager"] += 1
                    me.reporter.emit(
                        integration_error(
                            "alloc_outside_manager",
                            f"{len(ids)} block(s) allocated outside any manager's allocate_new_blocks",
                            pool=POOL,
                            site=f"alloc@{me.seq}",
                        )
                    )
            for b in ids:
                me.group_of[b] = g
                me._offload_filled_pools.discard((me.pool_of_group[g], b))
                me._zeroed_blocks.discard((me.pool_of_group[g], b))
                for logical_pool in me._shared_logical_pools(me.pool_of_group[g]):
                    me._offload_filled_pools.discard((logical_pool, b))
                    me._zeroed_blocks.discard((logical_pool, b))
            me._forget_allocated_publications(me.pool_of_group[g], ids)
            me._mark(ids)
            me.ledger.allocate(me.pool_of_group[g], ids, site=f"alloc@{me.seq}")
            for logical_pool in me._shared_logical_pools(me.pool_of_group[g]):
                me.ledger.allocate(logical_pool, ids, site=f"alloc-{logical_pool}@{me.seq}")
            me.stats["alloc_blocks"] += len(blocks)
            return blocks

        pool.get_new_blocks = get_new_blocks

        orig_evict = pool._maybe_evict_cached_block

        def maybe_evict(block):
            evicted = orig_evict(block)
            if evicted:
                me._mark((block.block_id,))
            if evicted and block.ref_cnt == 0 and not block.is_null:
                me._release([block.block_id], f"evict@{me.seq}")
                me.stats["evicted_blocks"] += 1
            return evicted

        pool._maybe_evict_cached_block = maybe_evict

        orig_reset = pool.reset_prefix_cache

        def reset_prefix_cache():
            # Clears every cached block's hash at once, without the eviction path; succeeds only
            # when no request holds a block, so the cached set is exactly the reserved published
            # blocks, freed here as one event.
            cached = []
            for entry in pool.cached_block_hash_to_block._cache.values():
                for b in entry.values() if isinstance(entry, dict) else (entry,):
                    if b.ref_cnt == 0 and not b.is_null:
                        cached.append(b.block_id)
            ok = orig_reset()
            if ok:
                me._mark(cached)
            if ok and cached:
                me._release(sorted(set(cached)), f"reset_prefix_cache@{me.seq}")
                me.stats["reset_prefix_cache"] += 1
                me.stats["reset_evicted_blocks"] += len(set(cached))
            return ok

        pool.reset_prefix_cache = reset_prefix_cache

        orig_free = pool.free_blocks

        def free_blocks(ordered_blocks, *args, **kwargs):
            # Older vLLM revisions expose queue-order controls such as
            # ``prepend=True`` here.  They do not change which blocks are
            # released, so preserve them while interposing on the block list.
            blocks = _call_free_blocks(orig_free, ordered_blocks, *args, **kwargs)
            if me._draining_deferred:
                is_cow, batch_id = me._deferred_release_kinds.popleft()
                if is_cow:
                    me._release_cow_pins(blocks)
                    me._cow_deferred_batches.discard(batch_id)
                else:
                    me._release_deferred_rows(blocks)
            elif me._releasing_cow_pins:
                me._release_cow_pins(blocks)
            me._mark(b.block_id for b in blocks)
            freed = mediated_frees(
                {
                    "block_id": b.block_id,
                    "ref_cnt": b.ref_cnt,
                    "has_hash": b.block_hash is not None,
                    "is_null": b.is_null,
                }
                for b in blocks
            )
            if freed:
                me._release(freed, f"free@{me.seq}")
            me.stats["free_blocks"] += len(freed)
            me.stats["released_cached_blocks"] += sum(
                1 for b in blocks if b.ref_cnt == 0 and b.block_hash is not None
            )

        pool.free_blocks = free_blocks

        orig_pop_for_free = getattr(self.kvm, "pop_blocks_for_free", None)
        if orig_pop_for_free is not None:

            def pop_blocks_for_free(request):
                blocks = orig_pop_for_free(request)
                me._defer_request_row(request, blocks)
                return blocks

            self.kvm.pop_blocks_for_free = pop_blocks_for_free

        orig_drain_deferred = getattr(sched, "_drain_deferred_frees", None)
        if orig_drain_deferred is not None:

            def drain_deferred_frees():
                me._deferred_release_kinds.clear()
                processed = int(sched.processed_step_seq)
                for fence, blocks in sched.deferred_frees:
                    if int(fence) > processed:
                        break
                    batch_id = id(blocks)
                    me._deferred_release_kinds.append(
                        (batch_id in me._cow_deferred_batches, batch_id)
                    )
                me._draining_deferred = True
                try:
                    return orig_drain_deferred()
                finally:
                    me._draining_deferred = False
                    me._deferred_release_kinds.clear()

            sched._drain_deferred_frees = drain_deferred_frees

        orig_free_cow = getattr(sched, "_free_cow_retained_blocks", None)
        if orig_free_cow is not None:

            def free_cow_retained_blocks(blocks, fence_seq):
                ids = tuple(int(block.block_id) for block in blocks)
                me._cow_pins.update(ids)
                me.stats["cow_pins"] += len(ids)
                me._trace_simple_offload(
                    "cow_retain",
                    request=None,
                    event_id=int(fence_seq),
                    block_ids=ids,
                )
                before = {id(entry_blocks) for _, entry_blocks in sched.deferred_frees}
                me._releasing_cow_pins = True
                try:
                    return orig_free_cow(blocks, fence_seq)
                finally:
                    me._releasing_cow_pins = False
                    for _, entry_blocks in sched.deferred_frees:
                        batch_id = id(entry_blocks)
                        if batch_id not in before:
                            me._cow_deferred_batches.add(batch_id)
                    me._trace_simple_offload(
                        "cow_retain_done",
                        request=None,
                        event_id=int(fence_seq),
                        block_ids=ids,
                        deferred=any(
                            id(entry_blocks) not in before
                            for _, entry_blocks in sched.deferred_frees
                        ),
                    )

            sched._free_cow_retained_blocks = free_cow_retained_blocks

        # The pool's other paths that change a block's reference count or hash: the caches re-read
        # the blocks they name at the next snapshot.
        for name, which in (
            ("touch", "blocks"),
            ("unpin_blocks", "blocks"),
            ("cache_full_blocks", "full"),
            ("cache_partial_block", "block"),
            ("move_block_hashes", "pair"),
        ):
            orig = getattr(pool, name, None)
            if orig is None:
                me.stats[f"pool_without_{name}"] += 1
                continue
            setattr(pool, name, _marking(orig, which, me))

        orig_schedule = sched.schedule

        def schedule(*args, **kwargs):
            out = orig_schedule(*args, **kwargs)
            me._step_key = id(out)
            me._zero_fills(out)
            me._copies(out)
            me._offload_loads(out)
            me._pd_loads(out)
            me._mooncake_loads(out)
            me._snapshot(out)
            return out

        sched.schedule = schedule

        orig_update = sched.update_from_output

        def update_from_output(scheduler_output, model_output):
            step_key = id(scheduler_output)
            for did in me.open.pop(step_key, ()):
                me.ledger.complete(did)
            if me.open:
                me.stats["steps_in_flight_max"] = max(me.stats["steps_in_flight_max"], len(me.open))
            me._collect_device()
            me._offload_completions(model_output)
            me._pd_completions(model_output)
            me._mooncake_completions(model_output)
            try:
                return orig_update(scheduler_output, model_output)
            finally:
                me._complete_speculative_step(step_key)

        sched.update_from_output = update_from_output

        orig_execute_model = runner.execute_model

        def execute_model(scheduler_output, *args, **kwargs):
            # The worker's own view of the step it executes: the scheduler thread may already
            # have scheduled the next step, so nothing here reads the attachment's latest
            # scheduler-side fields for this step's identity.
            me._executing_output = scheduler_output
            if me.eagle3:
                me._begin_eagle_target(scheduler_output)
            try:
                return orig_execute_model(scheduler_output, *args, **kwargs)
            finally:
                me._executing_output = None
                me._eagle_prepare_lookahead = None

        runner.execute_model = execute_model

        orig_prepare_attn = runner.prepare_attn

        def prepare_attn(input_batch):
            block_tables, slot_mappings = orig_prepare_attn(input_batch)
            me._flush_offload_store_epochs(me._executing_output)
            me._reap_offload_store_epochs()
            me.last_slot_mappings = slot_mappings  # for the hooks (fault injection)
            if me.after_prepare_attn is not None:
                me.after_prepare_attn(me, input_batch, block_tables)
            me._dispatch(input_batch, block_tables, slot_mappings)
            if me.eagle3:
                me._prepare_eagle_descriptors(input_batch)
            if me.after_dispatch is not None:
                me.after_dispatch(me, input_batch, block_tables)
            return block_tables, slot_mappings

        runner.prepare_attn = prepare_attn

        if self.eagle3:
            speculator = runner.speculator
            orig_propose = speculator.propose

            def propose(input_batch, *args, **kwargs):
                requested = kwargs.get("num_speculative_tokens")
                lookahead = me.eagle_lookahead if requested is None else int(requested)
                me.stats[f"eagle_requested_k_{lookahead}"] += 1
                me.stats["eagle_propose_batches"] += 1
                if bool(getattr(input_batch, "has_prefill", False)):
                    me.stats["eagle_prefill_batches"] += 1
                else:
                    me.stats["eagle_decode_only_batches"] += 1
                if bool(getattr(input_batch, "decode_graph_eligible", False)):
                    me.stats["eagle_decode_graph_eligible_batches"] += 1
                num_tokens = int(input_batch.num_tokens)
                num_tokens_padded = int(
                    getattr(input_batch, "num_tokens_after_padding", num_tokens)
                )
                if num_tokens_padded > num_tokens:
                    me.stats["eagle_padded_batches"] += 1
                    me.stats["eagle_padding_tokens"] += num_tokens_padded - num_tokens
                me._match_eagle_proposal(input_batch, lookahead)
                return orig_propose(input_batch, *args, **kwargs)

            speculator.propose = propose
            prefill_graph_manager = getattr(speculator, "prefill_cudagraph_manager", None)
            if prefill_graph_manager is not None:
                orig_prefill_fullgraph = prefill_graph_manager.run_fullgraph

                def run_prefill_fullgraph(descriptor):
                    me.stats["eagle_prefill_graph_replays"] += 1
                    me.stats["eagle_prefill_graph_tokens"] += int(descriptor.num_tokens)
                    return orig_prefill_fullgraph(descriptor)

                prefill_graph_manager.run_fullgraph = run_prefill_fullgraph
            decode_graph_manager = getattr(speculator, "decode_cudagraph_manager", None)
            if decode_graph_manager is not None:
                orig_decode_fullgraph = decode_graph_manager.run_fullgraph

                def run_decode_fullgraph(descriptor):
                    me.stats["eagle_decode_graph_replays"] += 1
                    me.stats["eagle_decode_graph_tokens"] += int(descriptor.num_tokens)
                    return orig_decode_fullgraph(descriptor)

                decode_graph_manager.run_fullgraph = run_decode_fullgraph
        elif self.standalone_draft:
            speculator = runner.speculator
            orig_propose = speculator.propose
            orig_run_model = speculator._run_model

            def propose(input_batch, *args, **kwargs):
                requested = kwargs.get("num_speculative_tokens")
                requested_k = me.eagle_lookahead if requested is None else int(requested)
                me.stats[f"standalone_requested_k_{requested_k}"] += 1
                me.stats["standalone_propose_batches"] += 1
                me._standalone_input_batch = input_batch
                me._standalone_dummy_run = bool(kwargs.get("dummy_run", False))
                me._standalone_calls = 0
                try:
                    return orig_propose(input_batch, *args, **kwargs)
                finally:
                    me._standalone_input_batch = None
                    me._standalone_dummy_run = False

            def run_model(*args, **kwargs):
                input_ids = args[0] if args else kwargs["input_ids"]
                positions = args[1] if len(args) > 1 else kwargs["positions"]
                attn_metadata = args[2] if len(args) > 2 else kwargs.get("attn_metadata")
                slot_mappings = args[3] if len(args) > 3 else kwargs.get("slot_mappings")
                if not me._standalone_dummy_run:
                    me._dispatch_standalone_draft(
                        speculator,
                        input_ids,
                        positions,
                        attn_metadata,
                        slot_mappings,
                    )
                return orig_run_model(*args, **kwargs)

            speculator.propose = propose
            speculator._run_model = run_model
        elif self.dflash:
            speculator = runner.speculator
            orig_propose = speculator.propose
            orig_precompute = speculator._precompute_context_kv
            orig_run_model = speculator._run_model

            def propose(input_batch, *args, **kwargs):
                me._dflash_input_batch = input_batch
                requested = kwargs.get("num_speculative_tokens")
                requested_k = me.eagle_lookahead - 1 if requested is None else int(requested)
                me.stats[f"dflash_requested_k_{requested_k}"] += 1
                try:
                    return orig_propose(input_batch, *args, **kwargs)
                finally:
                    me._dflash_input_batch = None

            def precompute_context_kv(start, end, dummy_run=False):
                if not dummy_run:
                    me._dispatch_dflash_context(speculator, int(start), int(end))
                return orig_precompute(start, end, dummy_run)

            def run_model(*args, **kwargs):
                num_tokens = int(args[0] if args else kwargs["num_tokens"])
                slot_mappings = args[2] if len(args) > 2 else kwargs.get("slot_mappings")
                me._dispatch_dflash_query(speculator, num_tokens, slot_mappings)
                return orig_run_model(*args, **kwargs)

            speculator.propose = propose
            speculator._precompute_context_kv = precompute_context_kv
            speculator._run_model = run_model
            if self.dflash_graphs:
                orig_prepare_graph_replay = speculator._prepare_graph_replay
                graph_manager = speculator.query_cudagraph_manager
                orig_run_fullgraph = graph_manager.run_fullgraph

                def prepare_graph_replay(
                    input_batch,
                    num_graph_reqs,
                    last_hidden_states,
                    aux_hidden_states,
                    num_rejected,
                    dummy_run,
                ):
                    result = orig_prepare_graph_replay(
                        input_batch,
                        num_graph_reqs,
                        last_hidden_states,
                        aux_hidden_states,
                        num_rejected,
                        dummy_run,
                    )
                    if not dummy_run:
                        graph_context = min(
                            int(input_batch.num_tokens),
                            int(speculator._num_graph_context_tokens(num_graph_reqs)),
                        )
                        me._dispatch_dflash_context(speculator, 0, graph_context)
                    return result

                def run_fullgraph(descriptor):
                    me._dispatch_dflash_graph_query(speculator, int(descriptor.num_tokens))
                    return orig_run_fullgraph(descriptor)

                speculator._prepare_graph_replay = prepare_graph_replay
                graph_manager.run_fullgraph = run_fullgraph
        elif self.multi_module_mtp:
            speculator = runner.speculator
            orig_propose = speculator.propose
            orig_run_model = speculator._run_model

            def propose(input_batch, *args, **kwargs):
                requested = kwargs.get("num_speculative_tokens")
                requested_k = len(me.mtp_gids) if requested is None else int(requested)
                me.stats[f"mtp_requested_k_{requested_k}"] += 1
                me._mtp_input_batch = input_batch
                try:
                    return orig_propose(input_batch, *args, **kwargs)
                finally:
                    me._mtp_input_batch = None

            def run_model(*args, **kwargs):
                num_tokens = int(args[0] if args else kwargs["num_tokens"])
                slot_mappings = args[2] if len(args) > 2 else kwargs.get("slot_mappings")
                module_index = int(args[4] if len(args) > 4 else kwargs.get("spec_module_idx", 0))
                me._dispatch_mtp(speculator, num_tokens, slot_mappings, module_index)
                return orig_run_model(*args, **kwargs)

            speculator.propose = propose
            speculator._run_model = run_model
            graph_manager = getattr(speculator, "cudagraph_manager", None)
            if graph_manager is not None:
                orig_run_fullgraph = graph_manager.run_fullgraph

                def run_fullgraph(desc):
                    me._dispatch_mtp_full_graph(speculator, desc)
                    return orig_run_fullgraph(desc)

                graph_manager.run_fullgraph = run_fullgraph

        # Some hybrid-cache runners zero new attention blocks by staging the block IDs through
        # mutable pinned and device tensors; retain those resources in the ledger until a CUDA
        # event says the consumer is done. Runners using ``async_tensor_h2d`` have no such
        # staging tensors for this wrapper to track.
        self._install_zero_resources(runner)
        self._install_offload_worker()

        model_state = getattr(runner, "model_state", None)
        if (
            self.hybrid
            and self.state_mode == "align"
            and self.tables == "device"
            and getattr(model_state, "_align_mode", False)
            and hasattr(model_state, "preprocess_state")
        ):
            # The worker's state columns (its GPU buffers, advanced by
            # preprocess_mamba_align_fused_kernel) against the adapter's derivation: an
            # observation of the framework's bookkeeping, read back with the device tables.
            orig_preprocess = model_state.preprocess_state

            def preprocess_state(input_batch, *args, **kwargs):
                out = orig_preprocess(input_batch, *args, **kwargs)
                me._check_state_columns(model_state, input_batch)
                return out

            model_state.preprocess_state = preprocess_state
            self.stats["state_columns_wrapped"] += 1

    def _mooncake_decode_buffers(
        self, worker: Any, keys: Any, addrs: Any, sizes: Any
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Decode native rank-local scatter/gather descriptors to group/block pairs."""
        decoded: list[tuple[int, tuple[int, ...]]] = []
        num_blocks = len(self.pool.blocks)
        for key, item_addrs, item_sizes in zip(keys, addrs, sizes, strict=True):
            key_groups = tuple(
                group for group in range(len(worker.token_dbs)) if f"@group:{group}@" in str(key)
            )
            if len(key_groups) != 1:
                raise ValueError(
                    f"Mooncake key does not name exactly one KV group: {str(key)[:160]!r}"
                )
            expected_group = key_groups[0]
            candidates: list[tuple[int, set[int]]] = []
            for store_group, db in enumerate(worker.token_dbs):
                if store_group != expected_group:
                    continue
                layout = db.store_layout
                bases = tuple(int(v) for v in getattr(layout, "kv_caches_base_addr", ()))
                strides = tuple(int(v) for v in getattr(layout, "block_stride", ()))
                lengths = tuple(int(v) for v in getattr(layout, "block_len", ()))
                if not bases or not (len(bases) == len(strides) == len(lengths)):
                    continue
                blocks: set[int] = set()
                matched_all = True
                for raw_addr, raw_size in zip(item_addrs, item_sizes, strict=True):
                    addr, size = int(raw_addr), int(raw_size)
                    segment_matches: list[tuple[int, ...]] = []
                    for base, stride, length in zip(bases, strides, lengths, strict=True):
                        delta = addr - base
                        if (
                            stride <= 0
                            or length <= 0
                            or length > stride
                            or delta < 0
                            or delta % stride
                            or size < length
                            or (size - length) % stride
                        ):
                            continue
                        first = delta // stride
                        count = 1 + (size - length) // stride
                        if first + count > num_blocks:
                            continue
                        segment_matches.append(tuple(range(first, first + count)))
                    if len(segment_matches) != 1:
                        matched_all = False
                        break
                    blocks.update(segment_matches[0])
                if matched_all:
                    candidates.append((store_group, blocks))
            if len(candidates) != 1:
                raise ValueError(
                    "Mooncake scatter/gather item does not identify exactly one rank-local "
                    f"KV group (candidates={[group for group, _ in candidates]!r})"
                )
            group, blocks = candidates[0]
            decoded.append((group, tuple(sorted(blocks))))
        return tuple(decoded)

    def _install_mooncake_worker(self) -> None:
        """Observe exact native Store operations without calling the ledger off-thread."""
        if self.mooncake_kind != "store":
            return
        try:
            from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

            connector = get_kv_transfer_group()
            worker = connector.connector_worker
        except (AttributeError, ImportError) as exc:
            self._reject(f"Mooncake Store worker is unavailable: {exc}")
        store_group_ids = tuple(int(g) for g in self.mooncake._store_group_ids)
        if (
            type(connector).__name__ != "MooncakeStoreConnector"
            or type(worker).__name__ != "MooncakeStoreWorker"
            or int(getattr(worker, "tp_size", -1)) != 1
            or len(worker.token_dbs) != len(store_group_ids)
            or any(
                type(db.store_layout).__name__ != "RankLocalStoreLayout" for db in worker.token_dbs
            )
        ):
            self._reject(
                "Mooncake Store worker outside the admitted single-rank rank-local boundary"
            )

        send_thread = getattr(worker, "kv_send_thread", None)
        if send_thread is not None:
            active: dict[str, Any] = {"request": None, "operation": None}
            original_handle = send_thread._handle_request

            def handle_store(req_meta, _active=active, _original=original_handle):
                _active["request"] = req_meta
                try:
                    return _original(req_meta)
                finally:
                    _active["request"] = None

            def before_put(keys, addrs, sizes, _active=active):
                req_meta = _active["request"]
                try:
                    decoded = self._mooncake_decode_buffers(worker, keys, addrs, sizes)
                    token = f"put:{time.monotonic_ns()}:{id(req_meta)}"
                    _active["operation"] = (token, decoded)
                    self._mooncake_events.append(
                        {
                            "kind": "read_start",
                            "token": token,
                            "request": str(getattr(req_meta, "req_id", "")),
                            "job": getattr(req_meta, "store_job_id", None),
                            "items": decoded,
                        }
                    )
                except Exception as exc:
                    self._mooncake_events.append({"kind": "error", "error": repr(exc)})

            def observe_put(keys, addrs, sizes, result, _active=active):
                operation = _active.pop("operation", None)
                if operation is None:
                    return
                token, decoded = operation
                successful = tuple(
                    decoded[index] for index, status in enumerate(result) if int(status) >= 0
                )
                self._mooncake_events.append(
                    {
                        "kind": "read_complete",
                        "token": token,
                        "request": str(getattr(_active["request"], "req_id", "")),
                        "items": decoded,
                        "successful": successful,
                    }
                )

            send_thread._handle_request = handle_store
            send_thread.store = _MooncakeStoreProxy(
                send_thread.store,
                put=observe_put,
                before_put=before_put,
            )
            self.stats["mooncake_store_sender_wrapped"] += 1

        for recv_thread in worker.kv_recv_threads:
            active = {"request": None, "operation": None}
            original_handle = recv_thread._handle_request
            gate_dir_value = os.environ.get("KVGUARD_VLLM_MOONCAKE_GATE_DIR")
            gate_dir = Path(gate_dir_value) if gate_dir_value else None
            gate_request = os.environ.get("KVGUARD_VLLM_MOONCAKE_GATE_REQUEST", "")
            gate_state = {"used": False}

            def handle_load(req_meta, _active=active, _original=original_handle):
                _active["request"] = req_meta
                try:
                    return _original(req_meta)
                finally:
                    _active["request"] = None

            def observe_get(keys, addrs, sizes, result, _active=active):
                operation = _active.pop("operation", None)
                if operation is None:
                    return
                token, decoded = operation
                successful = tuple(
                    decoded[index] for index, status in enumerate(result) if int(status) >= 0
                )
                self._mooncake_events.append(
                    {
                        "kind": "write_complete",
                        "token": token,
                        "request": str(getattr(_active["request"], "req_id", "")),
                        "items": decoded,
                        "successful": successful,
                    }
                )

            def before_get(
                keys,
                addrs,
                sizes,
                _active=active,
                _gate_dir=gate_dir,
                _gate_request=gate_request,
                _gate_state=gate_state,
            ):
                req_meta = _active["request"]
                try:
                    decoded = self._mooncake_decode_buffers(worker, keys, addrs, sizes)
                    token = f"get:{time.monotonic_ns()}:{id(req_meta)}"
                    _active["operation"] = (token, decoded)
                    self._mooncake_events.append(
                        {
                            "kind": "write_start",
                            "token": token,
                            "request": str(getattr(req_meta, "req_id", "")),
                            "job": None,
                            "items": decoded,
                        }
                    )
                except Exception as exc:
                    self._mooncake_events.append({"kind": "error", "error": repr(exc)})
                request = str(getattr(req_meta, "req_id", ""))
                if (
                    _gate_dir is None
                    or _gate_state["used"]
                    or not (_gate_dir / "arm").exists()
                    or (_gate_request and request != _gate_request)
                ):
                    return
                _gate_state["used"] = True
                _gate_dir.mkdir(parents=True, exist_ok=True)
                (_gate_dir / "entered.json").write_text(
                    json.dumps(
                        {
                            "request": request,
                            "keys": len(keys),
                            "buffer_lists": len(addrs),
                            "size_lists": len(sizes),
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                deadline = time.monotonic() + 60
                while not (_gate_dir / "release").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Mooncake timing gate expired for request {request!r}")
                    time.sleep(0.005)

            recv_thread._handle_request = handle_load
            recv_thread.store = _MooncakeStoreProxy(
                recv_thread.store,
                get=observe_get,
                before_get=before_get,
            )
        self.stats["mooncake_store_receivers_wrapped"] += len(worker.kv_recv_threads)

    def _mooncake_pools(self, store_group: int) -> tuple[str, ...]:
        """Logical pools whose layer tensors one Store-group descriptor transfers."""
        store_group_ids = tuple(int(g) for g in self.mooncake._store_group_ids)
        gid = store_group_ids[int(store_group)]
        pool = self.pool_of_group[gid]
        return (pool, *self._shared_logical_pools(pool))

    def _mooncake_loads(self, scheduler_output: Any) -> None:
        """Retain load frontiers and bind Store jobs to their pinned source owners."""
        if self.mooncake_kind != "store":
            return
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        if metadata is None:
            return
        if type(metadata).__name__ != "MooncakeStoreConnectorMetadata":
            self._reject(
                "Mooncake Store scheduler published "
                f"{type(metadata).__name__}, expected MooncakeStoreConnectorMetadata"
            )
        store_group_ids = tuple(int(g) for g in self.mooncake._store_group_ids)
        for req_meta in metadata.requests:
            trace = getattr(self, "offload_events", None)
            if trace is not None:
                load_spec = getattr(req_meta, "load_spec", None)
                trace.append(
                    {
                        "seq": self.seq + 1,
                        "kind": "mooncake_metadata",
                        "request": str(req_meta.req_id),
                        "can_save": bool(getattr(req_meta, "can_save", False)),
                        "token_len_chunk": int(getattr(req_meta, "token_len_chunk", 0) or 0),
                        "completed_token_len": getattr(req_meta, "completed_token_len", None),
                        "load_local": (
                            None if load_spec is None else int(load_spec.vllm_cached_tokens)
                        ),
                        "load_remote": (
                            None if load_spec is None else int(load_spec.kvpool_cached_tokens)
                        ),
                        "blocks": [
                            [int(block) for block in blocks] for blocks in req_meta.block_ids
                        ],
                    }
                )
            if bool(getattr(req_meta, "can_save", False)):
                rid = str(req_meta.req_id)
                for store_group, blocks in enumerate(req_meta.block_ids):
                    for pool in self._mooncake_pools(store_group):
                        for block in blocks:
                            block_id = int(block)
                            if block_id != self.pool.null_block.block_id:
                                self._mooncake_store_owners[(pool, block_id)] = rid
            load_spec = getattr(req_meta, "load_spec", None)
            if load_spec is None or not bool(load_spec.can_load):
                continue
            rid = str(req_meta.req_id)
            if rid in self._mooncake_pending:
                self._reject(f"Mooncake request {rid!r} published two pending loads")
            local_tokens = int(load_spec.vllm_cached_tokens)
            remote_tokens = int(load_spec.kvpool_cached_tokens)
            tails: dict[str, int] = {}
            for store_group, blocks in enumerate(req_meta.block_ids):
                gid = store_group_ids[store_group]
                manager_block_size = int(self.managers[gid].block_size)
                first = local_tokens // manager_block_size
                end = (remote_tokens + manager_block_size - 1) // manager_block_size
                selected = tuple(int(block) for block in blocks[first:end])
                if selected:
                    for pool in self._mooncake_pools(store_group):
                        tails[pool] = selected[-1]
                        self.stats["mooncake_load_planned_blocks"] += len(selected)
            self._mooncake_pending[rid] = (local_tokens, remote_tokens, tails)
            self.stats["mooncake_load_jobs"] += 1
            self.stats["mooncake_load_planned_tokens"] += remote_tokens - local_tokens

    def _normalize_mooncake_reservations(self, snap: Snapshot) -> Snapshot:
        """Remove Store-only physical pins from request reference counts.

        A completed request can disappear while its native PUT still owns the sole reference.
        Keep that source visible as a sparse row of its producer until the scheduler releases
        the pin; this authorizes the exact Store read without pretending the pin is a new tenant.
        """
        self._mooncake_pin_private = set()
        if self.mooncake_kind != "store":
            return snap
        pins: Counter[int] = Counter()
        for block_ids, _remaining in getattr(self.mooncake, "_pinned_saves", {}).values():
            pins.update(int(block) for block in block_ids)
        if not pins:
            self._mooncake_store_owners.clear()
            return snap
        mapped: Counter[tuple[str, int]] = Counter()
        fully_computed: set[tuple[str, int]] = set()
        for row in snap.requests:
            block_size = self.core_ledger.pools[row.pool].block_size
            for index, block in enumerate(row.blocks):
                if block is None:
                    continue
                key = (row.pool, int(block))
                mapped[key] += 1
                if row.num_computed >= (index + 1) * block_size:
                    fully_computed.add(key)
        normalized = []
        reservations: dict[tuple[str, str], list[int]] = {}
        live_keys: set[tuple[str, int]] = set()
        adjusted = 0
        for block in snap.blocks:
            count = pins[block.block]
            if not count:
                normalized.append(block)
                continue
            key = (block.pool, block.block)
            live_keys.add(key)
            refs = None if block.refs is None else max(0, block.refs - count)
            geometry = self.core_ledger.pools[block.pool]
            if not mapped[key] and refs == 0:
                owner = self._mooncake_store_owners.get(key)
                if owner is None:
                    owner = f"__kvguard_mooncake_store__:{block.pool}:{block.block}"
                reservations.setdefault((owner, block.pool), []).append(block.block)
                refs = 1
                state = BlockState.PRIVATE if geometry.rewritable else BlockState.PUBLISHED
            elif geometry.rewritable and (refs is None or refs <= 1):
                state = BlockState.PRIVATE
            elif (
                block.state is BlockState.PUBLISHED
                and refs is not None
                and refs <= 1
                and mapped[key] <= 1
                and key not in fully_computed
            ):
                # The raw allocator count made a partial immutable block look shared solely
                # because the Store owned an I/O pin. Removing that pin restores PRIVATE.
                state = BlockState.PRIVATE
                self._mooncake_pin_private.add(key)
            else:
                state = block.state
            normalized.append(replace(block, state=state, reserved=False, refs=refs))
            adjusted += 1
        self._mooncake_store_owners = {
            key: owner for key, owner in self._mooncake_store_owners.items() if key in live_keys
        }
        rows: list[RequestSnapshot] = []
        for (owner, pool), blocks in reservations.items():
            geometry = self.core_ledger.pools[pool]
            unique_blocks = tuple(dict.fromkeys(blocks))
            prior = getattr(self.core_ledger, "requests", {}).get((owner, pool))
            source = None if prior is None else prior.snapshot
            remaining = Counter(unique_blocks)
            sparse = []
            if source is not None:
                for block_id in source.blocks:
                    if block_id is not None and remaining[block_id]:
                        sparse.append(block_id)
                        remaining[block_id] -= 1
                    else:
                        sparse.append(None)
            if source is not None and not +remaining:
                reservation = replace(source, blocks=tuple(sparse))
                if not geometry.rewritable:
                    present = [index for index, value in enumerate(sparse) if value is not None]
                    # Store reads remain live after sliding-window attention retires the
                    # corresponding prefix.  Describe the Store's physical source span, not
                    # the producer's current attention window, while sparse holes keep every
                    # other retired block unmapped.
                    first = present[0] * geometry.block_size
                    last = min(source.num_tokens, (present[-1] + 1) * geometry.block_size)
                    reservation = replace(
                        reservation,
                        num_tokens=last,
                        num_computed=min(source.num_computed, last),
                        first_token=first,
                    )
                    # The pinned subset shrinks as independent Store jobs finish.  Give each
                    # immutable transfer view its own reader identity so that removing a pin is
                    # not interpreted as the producer request rolling back published tokens.
                    shape = hashlib.sha256(
                        repr(
                            (
                                reservation.first_token,
                                reservation.num_tokens,
                                reservation.num_computed,
                                reservation.blocks,
                            )
                        ).encode("utf-8")
                    ).hexdigest()[:12]
                    reservation = replace(
                        reservation,
                        request=(f"__kvguard_mooncake_store__:{source.request}:{pool}:{shape}"),
                    )
                    self.stats["mooncake_source_rows_versioned"] += 1
                rows.append(reservation)
                self.stats["mooncake_source_rows_retained"] += 1
            else:
                rows.append(
                    RequestSnapshot(
                        owner,
                        pool,
                        unique_blocks,
                        len(unique_blocks) * geometry.block_size,
                        0,
                    )
                )
                self.stats["mooncake_synthetic_rows"] += 1
        requests = list(snap.requests)
        request_index = {(row.request, row.pool): index for index, row in enumerate(requests)}
        for reservation in rows:
            key = (reservation.request, reservation.pool)
            index = request_index.get(key)
            if index is None:
                request_index[key] = len(requests)
                requests.append(reservation)
                continue
            current = requests[index]
            geometry = self.core_ledger.pools[reservation.pool]
            width = max(len(current.blocks), len(reservation.blocks))
            current_blocks = current.blocks + (None,) * (width - len(current.blocks))
            reserved_blocks = reservation.blocks + (None,) * (width - len(reservation.blocks))
            collision = any(
                left is not None and right is not None and left != right
                for left, right in zip(current_blocks, reserved_blocks, strict=True)
            )
            if not geometry.rewritable or collision:
                # Immutable store sources are readers, not private owners. Preserve their
                # logical positions under a distinct row when a live beam child has already
                # replaced the producer's row.
                synthetic = replace(
                    reservation,
                    request=f"__kvguard_mooncake_store__:{reservation.request}:{reservation.pool}",
                )
                requests.append(synthetic)
                self.stats["mooncake_source_rows_split"] += 1
                continue
            requests[index] = replace(
                current,
                blocks=tuple(
                    left if left is not None else right
                    for left, right in zip(current_blocks, reserved_blocks, strict=True)
                ),
                num_tokens=max(current.num_tokens, reservation.num_tokens),
                num_computed=max(current.num_computed, reservation.num_computed),
                first_token=min(current.first_token, reservation.first_token),
            )
            self.stats["mooncake_source_rows_merged"] += 1
        self.stats["mooncake_pinned_blocks_normalized"] += adjusted
        self.stats["mooncake_pin_rows"] += len(rows)
        return replace(
            snap,
            requests=tuple(requests),
            blocks=tuple(normalized),
        )

    def _drain_mooncake_events(self) -> None:
        """Open and complete native operations while vLLM still holds their block pins."""
        if self.mooncake_kind != "store":
            return
        while self._mooncake_events:
            event = self._mooncake_events.popleft()
            trace = getattr(self, "offload_events", None)
            if trace is not None:
                trace.append(
                    {
                        "seq": self.seq,
                        "kind": f"mooncake_{event['kind']}",
                        "request": event.get("request"),
                        "items": [
                            [
                                pool,
                                [int(block) for block in blocks],
                            ]
                            for store_group, blocks in event.get("items", ())
                            for pool in self._mooncake_pools(store_group)
                        ],
                    }
                )
            if event["kind"] == "error":
                self._reject(f"cannot decode Mooncake Store buffers: {event['error']}")
            if event["kind"] in ("read_complete", "write_complete"):
                rid = event["request"] or None
                completed_pools: set[str] = set()
                for store_group, _blocks in event["items"]:
                    for pool in self._mooncake_pools(store_group):
                        if pool not in completed_pools:
                            self.ledger.complete(f"mooncake-{event['token']}:{pool}")
                            completed_pools.add(pool)
                if event["kind"] == "write_complete":
                    for store_group, blocks in event["successful"]:
                        for pool in self._mooncake_pools(store_group):
                            offsets = range(self.core_ledger.pools[pool].block_size)
                            for block in blocks:
                                self.ledger.external_fill(
                                    ExternalFill(
                                        pool,
                                        int(block),
                                        offsets,
                                        source="mooncake_store_get",
                                        request=rid,
                                    ),
                                    site=f"mooncake-{event['token']}:{pool}@{self.seq}",
                                )
                                self.stats["mooncake_load_filled_blocks"] += 1
                operation = "read" if event["kind"] == "read_complete" else "write"
                self.stats[f"mooncake_native_{operation}_calls"] += 1
                continue
            rid = event["request"] or None
            by_pool: dict[str, set[int]] = {}
            for store_group, blocks in event["items"]:
                for pool in self._mooncake_pools(store_group):
                    by_pool.setdefault(pool, set()).update(int(block) for block in blocks)
            for pool, block_set in by_pool.items():
                blocks = tuple(sorted(block_set))
                if not blocks:
                    continue
                did = f"mooncake-{event['token']}:{pool}"
                if (
                    not self.mooncake_fault_done
                    and blocks
                    and self.mooncake_fault
                    == ("store_early_free" if event["kind"] == "read_start" else "load_early_free")
                ):
                    self.mooncake_fault_done = True
                    self.ledger.free(pool, blocks[:1], site=f"{did}:fault")
                    self.stats["mooncake_faults_injected"] += 1
                offsets = range(self.core_ledger.pools[pool].block_size)
                if event["kind"] == "read_start":
                    self.ledger.external_read(
                        ExternalRead(
                            did,
                            pool,
                            blocks,
                            offsets,
                            source="mooncake_store_put",
                            request=rid,
                        ),
                        site=f"{did}@{self.seq}",
                    )
                    self.stats["mooncake_store_read_blocks"] += len(blocks)
                else:
                    self.ledger.external_write(
                        ExternalWrite(
                            did,
                            pool,
                            blocks,
                            offsets,
                            source="mooncake_store_get",
                            request=rid,
                        ),
                        site=f"{did}@{self.seq}",
                    )
            gate_dir_value = os.environ.get("KVGUARD_VLLM_MOONCAKE_GATE_DIR")
            if gate_dir_value and event["kind"] in ("read_start", "write_start"):
                gate_dir = Path(gate_dir_value)
                entered_path = gate_dir / "entered.json"
                observed_path = gate_dir / "observed.json"
                entered = (
                    json.loads(entered_path.read_text(encoding="utf-8"))
                    if entered_path.exists()
                    else None
                )
                if (
                    entered is not None
                    and event["request"] == entered.get("request")
                    and not observed_path.exists()
                    and not (gate_dir / "release").exists()
                ):
                    observed_path.write_text(
                        json.dumps(
                            {
                                "kind": event["kind"],
                                "request": event["request"],
                                "token": event["token"],
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    self.stats["mooncake_gate_epochs_observed"] += 1

    def _mooncake_completions(self, model_output: Any) -> None:
        """Drain native operations and retire completed request-load frontiers."""
        if self.mooncake_kind != "store":
            return
        self._drain_mooncake_events()
        output = getattr(model_output, "kv_connector_output", None)
        if output is None:
            return
        failed = {str(rid) for rid in (getattr(output, "failed_recving", ()) or ())}
        completed = {str(rid) for rid in (getattr(output, "finished_recving", ()) or ())}
        for rid in failed | completed:
            pending = self._mooncake_pending.pop(rid, None)
            if pending is None:
                self.stats["mooncake_load_completions_without_pending"] += 1
                continue
            if rid in failed:
                self.stats["mooncake_load_failures"] += 1
                continue
            local_tokens, remote_tokens, tails = pending
            request = self.scheduler.requests.get(rid)
            if (
                request is not None
                and tails
                and remote_tokens > 0
                and remote_tokens == int(request.num_tokens)
            ):
                self._offload_recompute_tails[rid] = (
                    remote_tokens - 1,
                    tails,
                    False,
                )
                self.stats["mooncake_full_hit_tail_recomputes"] += 1
            self.stats["mooncake_load_completions"] += 1
            self.stats["mooncake_load_completed_tokens"] += remote_tokens - local_tokens

    def _install_offload_worker(self) -> None:
        """Open GPU-source read epochs at the CPU worker's real transfer submission.

        Store metadata is deferred one step, so the scheduler hook alone is too early; the worker
        exposes both the submission and its transfer stream. A job named in ``jobs_to_flush`` is
        the read-before-new-owner-write path, waited in the same call, so it is counted rather
        than revalidated against a possibly reallocated source generation.
        """
        if self.offload is None:
            return
        try:
            from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

            connector = get_kv_transfer_group()
            connector_worker = (
                connector
                if getattr(self, "offload_kind", None) == "simple"
                else connector.connector_worker
            )
        except (AttributeError, ImportError) as exc:
            self._reject(f"CPU offload worker is unavailable for store-read epochs: {exc}")
        if getattr(self, "offload_kind", None) == "simple":
            self._install_simple_offload_worker(connector, connector_worker)
            return
        try:
            transfer_worker = connector_worker.worker
        except AttributeError as exc:
            self._reject(f"managed CPU offload worker is unavailable: {exc}")
        if (
            type(connector).__name__ != "OffloadingConnector"
            or type(connector_worker).__name__ != "OffloadingConnectorWorker"
            or type(transfer_worker).__name__ != "CPUOffloadingWorker"
        ):
            self._reject(
                "CPU offload worker is outside the admitted store-read boundary: "
                f"{type(connector).__name__}/{type(connector_worker).__name__}/"
                f"{type(transfer_worker).__name__}"
            )

        orig_handle_preemptions = connector_worker.handle_preemptions

        def handle_preemptions(metadata):
            self._offload_flushing = {
                int(job_id) for job_id in (getattr(metadata, "jobs_to_flush", ()) or ())
            }
            try:
                return orig_handle_preemptions(metadata)
            finally:
                self._offload_flushing.clear()

        connector_worker.handle_preemptions = handle_preemptions

        orig_submit_store = transfer_worker.submit_store

        def submit_store(job_id, src_spec, dst_spec):
            success = orig_submit_store(job_id, src_spec, dst_spec)
            if success:
                self._offload_store_started(
                    int(job_id), transfer_worker, int(job_id) in self._offload_flushing
                )
            return success

        transfer_worker.submit_store = submit_store

        orig_wait = transfer_worker.wait

        def wait(job_ids):
            out = orig_wait(job_ids)
            for job_id in job_ids:
                self._offload_store_complete(int(job_id), waited=True)
            return out

        transfer_worker.wait = wait
        self._offload_transfer_worker = transfer_worker
        self.stats["offload_store_worker_wrapped"] += 1

    def _install_tiering_observer(self) -> None:
        """Count tier work and detect a source-derived promotion capacity cycle.

        A single full-attention group needs every key in its accepted prefix resident in the
        primary CPU tier before ``prepare_load`` can pin it. If one request promotes more distinct
        keys than the primary can hold and returns to the same promotion key set and primary-cache
        state before any ``prepare_load``, the lookup has cycled: no completion or passage of time
        can make that working set fit. The check stays within the single full-attention-group
        boundary, where every probed key belongs to the final load set.
        """
        manager = self.offload.manager
        primary_capacity = int(getattr(manager.primary_tier, "_num_chunks", 0))
        cycle_guard_enabled = (
            primary_capacity > 0
            and len(self.groups) == 1
            and self.groups[0]["kind"] == "attention"
            and not self.groups[0].get("draft", False)
        )
        promoted_without_load: dict[str, set[Any]] = {}
        promotion_states: dict[str, Counter[Any]] = {}
        reported_cycles: set[str] = set()
        promotion_jobs: dict[int, tuple[str, tuple[Any, ...]]] = {}

        orig_prepare_load = manager.prepare_load

        def prepare_load(keys, req_context, *args, **kwargs):
            out = orig_prepare_load(keys, req_context, *args, **kwargs)
            request_id = str(req_context.req_id)
            promoted_without_load.pop(request_id, None)
            promotion_states.pop(request_id, None)
            return out

        manager.prepare_load = prepare_load

        orig_request_finished = manager.on_request_finished

        def on_request_finished(req_context, *args, **kwargs):
            request_id = str(req_context.req_id)
            promoted_without_load.pop(request_id, None)
            promotion_states.pop(request_id, None)
            reported_cycles.discard(request_id)
            return orig_request_finished(req_context, *args, **kwargs)

        manager.on_request_finished = on_request_finished

        for index, tier in enumerate(manager.secondary_tiers):
            tier_name = type(tier).__name__
            orig_submit_store = tier.submit_store
            orig_submit_load = tier.submit_load
            orig_finished = tier.get_finished_jobs

            def submit_store(job, *args, _orig=orig_submit_store, _name=tier_name, **kwargs):
                self.stats[f"tiering_{_name}_store_jobs"] += 1
                self.stats[f"tiering_{_name}_store_chunks"] += len(job.keys)
                return _orig(job, *args, **kwargs)

            def submit_load(job, *args, _orig=orig_submit_load, _name=tier_name, **kwargs):
                self.stats[f"tiering_{_name}_load_jobs"] += 1
                self.stats[f"tiering_{_name}_load_chunks"] += len(job.keys)
                if cycle_guard_enabled and bool(getattr(job, "is_promotion", False)):
                    request_id = str(job.req_context.req_id)
                    job_keys = tuple(job.keys)
                    promotion_jobs[int(job.job_id)] = (request_id, job_keys)
                    primary = manager.primary_tier
                    policy = getattr(primary, "_policy", None)
                    chunks = getattr(policy, "chunks", None)
                    if isinstance(chunks, dict):
                        ranks = getattr(policy, "_ranks", {})
                        evictable = getattr(policy, "_evictable", set())
                        primary_state = frozenset(
                            (
                                key,
                                int(chunk.ref_cnt),
                                int(getattr(chunk, "chunk_id", -1)),
                                ranks.get(key),
                                key in evictable,
                            )
                            for key, chunk in chunks.items()
                        )
                        request_state = getattr(self.offload, "_req_status", {}).get(request_id)
                        request = getattr(request_state, "req", None)
                        request_frontier = (
                            int(getattr(request, "num_computed_tokens", -1)),
                            int(getattr(request_state, "num_locally_computed_tokens", -1)),
                            getattr(request_state, "partial_tail_boundary", None),
                        )
                        primary_meta = (
                            int(getattr(primary, "_num_allocated_chunks", -1)),
                            tuple(getattr(primary, "_free_list", ())),
                            int(getattr(primary, "_num_evictable_cache_chunks", -1)),
                            int(getattr(primary, "_num_write_pending_chunks", -1)),
                            int(getattr(policy, "_next_rank", -1)),
                        )
                        signature = (
                            frozenset(job_keys),
                            primary_state,
                            primary_meta,
                            request_frontier,
                        )
                        state_counts = promotion_states.setdefault(request_id, Counter())
                        state_counts[signature] += 1
                        active_requests = {
                            str(active_id) for active_id in getattr(self.offload, "_req_status", {})
                        }
                        distinct = len(promoted_without_load.get(request_id, ()))
                        if (
                            state_counts[signature] >= 2
                            and distinct > primary_capacity
                            and active_requests == {request_id}
                            and request_id not in reported_cycles
                        ):
                            reported_cycles.add(request_id)
                            self.stats["tiering_promotion_capacity_cycles"] += 1
                            self.reporter.emit(
                                Report(
                                    OutcomeKind.VIOLATION,
                                    "tiering_promotion_capacity_cycle",
                                    "a secondary-tier promotion returned to the same pending "
                                    "key set and complete primary-cache state before this "
                                    "request initiated any GPU load, after its distinct "
                                    "promoted working set exceeded primary capacity and while "
                                    "it was the only active request; this is a closed cache "
                                    "state cycle "
                                    f"(distinct={distinct}, capacity={primary_capacity})",
                                    request=request_id,
                                    site=f"secondary-promotion@{self.seq}",
                                )
                            )
                return _orig(job, *args, **kwargs)

            def get_finished_jobs(*args, _orig=orig_finished, _name=tier_name, **kwargs):
                results = list(_orig(*args, **kwargs))
                for result in results:
                    outcome = "success" if result.success else "failure"
                    self.stats[f"tiering_{_name}_{outcome}_jobs"] += 1
                    if getattr(result, "deferred_release", None) is not None:
                        self.stats[f"tiering_{_name}_deferred_release_jobs"] += 1
                    promotion = promotion_jobs.pop(int(result.job_id), None)
                    if promotion is None:
                        continue
                    request_id, job_keys = promotion
                    successful_keys = (
                        job_keys
                        if bool(result.success)
                        else tuple(getattr(result, "successful_keys", ()) or ())
                    )
                    seen = promoted_without_load.setdefault(request_id, set())
                    seen.update(successful_keys)
                return results

            tier.submit_store = submit_store
            tier.submit_load = submit_load
            tier.get_finished_jobs = get_finished_jobs
            self.stats[f"tiering_secondary_{index}_{tier_name}"] = 1

    def _install_external_load_capacity_guard(self) -> None:
        """Report an async external load that cannot fit even in an empty GPU pool.

        A transient allocation refusal is healthy when another request owns blocks (completion or
        preemption can make room), but a refusal while every usable block is already free can
        never improve: the request stays remotely visible but unschedulable. This is an
        adapter-level availability invariant, not one of the core P1--P4 policies, checked at the
        real allocation decision without a timer.
        """
        manager = self.kvm
        pool = self.pool
        orig_allocate_slots = manager.allocate_slots
        reported: set[str] = set()

        def allocate_slots(request, num_new_tokens, *args, **kwargs):
            out = orig_allocate_slots(request, num_new_tokens, *args, **kwargs)
            request_id = str(request.request_id)
            if (
                out is None
                and request_id not in reported
                and int(num_new_tokens) == 0
                and int(kwargs.get("num_external_computed_tokens", 0)) > 0
                and bool(kwargs.get("delay_cache_blocks", False))
                and int(kwargs.get("reserved_blocks", 0)) > 0
                and not bool(kwargs.get("has_scheduled_reqs", True))
            ):
                free_blocks = int(pool.get_num_free_blocks())
                usable_blocks = int(pool.num_gpu_blocks) - 1
                if free_blocks == usable_blocks:
                    reported.add(request_id)
                    self.stats["external_load_capacity_deadlocks"] += 1
                    self.reporter.emit(
                        Report(
                            OutcomeKind.VIOLATION,
                            "external_load_capacity_deadlock",
                            "an asynchronous external load was rejected while every usable GPU "
                            "block was free; no cache transition can make this admitted request "
                            f"schedulable (free={free_blocks}, physical={pool.num_gpu_blocks}, "
                            f"external_tokens={kwargs['num_external_computed_tokens']}, "
                            f"reserved_blocks={kwargs['reserved_blocks']})",
                            request=request_id,
                            site=f"external-load-admission@{self.seq}",
                        )
                    )
            return out

        manager.allocate_slots = allocate_slots
        self.stats["external_load_capacity_guard"] += 1

    def _install_pd_scheduler(self) -> None:
        """Open NIXL producer reads before their coordinates reach the remote decoder."""
        connector = self.scheduler.connector
        orig_finished = connector.request_finished_all_groups

        def request_finished_all_groups(request, block_ids):
            delay_free, params = orig_finished(request, block_ids)
            if delay_free and params and params.get("remote_block_ids"):
                self._open_pd_send(
                    str(request.request_id),
                    block_ids,
                    int(params.get("remote_num_tokens", 0)),
                    params.get("remote_block_ids"),
                )
            return delay_free, params

        connector.request_finished_all_groups = request_finished_all_groups
        self.stats["pd_scheduler_wrapped"] += 1

    def _open_pd_send(
        self,
        rid: str,
        block_groups,
        num_tokens: int,
        remote_block_groups,
    ) -> None:
        """Represent a producer's remotely readable pages until NIXL acknowledges them."""
        if rid in self._pd_send_dispatches:
            self._reject(f"NIXL producer request {rid!r} opened its source read twice")
        groups = tuple(tuple(int(block) for block in group) for group in block_groups)
        remote_groups = tuple(tuple(int(block) for block in group) for group in remote_block_groups)
        if len(groups) != 1 or groups != remote_groups:
            self._reject(
                f"NIXL producer request {rid!r} changed its one-group transfer shape: "
                f"local={groups!r}, remote={remote_groups!r}"
            )
        block_ids = groups[0]
        expected = (num_tokens + self.block_size - 1) // self.block_size
        if num_tokens <= 0 or len(block_ids) != expected:
            self._reject(
                f"NIXL producer request {rid!r} exposes {len(block_ids)} blocks for "
                f"{num_tokens} tokens at block size {self.block_size}"
            )
        self._pd_send_seq += 1
        did = f"nixl-send:{self._pd_send_seq}"
        self._pd_publish_content(rid, block_ids, num_tokens)
        # NIXL moves whole physical pages, including the unused suffix of the last page.  Judge
        # only the positions the decoder can consume: the suffix is a transport over-read, like
        # a predicated attention tile's dead loads, and is overwritten before it becomes KV.
        # Each semantic read still reserves its complete source block for the transfer epoch.
        dispatches: list[str] = []
        full_blocks, tail = divmod(num_tokens, self.block_size)
        if full_blocks:
            full_did = f"{did}:full" if tail else did
            self.ledger.external_read(
                ExternalRead(
                    full_did,
                    POOL,
                    block_ids[:full_blocks],
                    range(self.block_size),
                    source="nixl_pull_send",
                    request=rid,
                ),
                site=f"{full_did}@{self.seq}",
            )
            dispatches.append(full_did)
        if tail:
            tail_did = f"{did}:tail"
            self.ledger.external_read(
                ExternalRead(
                    tail_did,
                    POOL,
                    (block_ids[-1],),
                    range(tail),
                    source="nixl_pull_send",
                    request=rid,
                ),
                site=f"{tail_did}@{self.seq}",
            )
            dispatches.append(tail_did)
        self._pd_send_dispatches[rid] = tuple(dispatches)
        self.stats["pd_send_jobs"] += 1
        self.stats["pd_send_blocks"] += len(block_ids)
        self.stats["pd_send_tokens"] += num_tokens

    def _pd_loads(self, scheduler_output) -> None:
        """Record decoder destinations without crediting bytes before NIXL completion."""
        if self.pd_kind != "nixl_pull":
            return
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        if metadata is None:
            return
        if type(metadata).__name__ != "NixlConnectorMetadata":
            self._reject(
                f"NIXL P/D metadata is {type(metadata).__name__}, expected NixlConnectorMetadata"
            )
        for rid, req_meta in metadata.reqs_to_recv.items():
            rid = str(rid)
            if not bool(req_meta.awaiting_kvs):
                self.stats["pd_notify_only_receives"] += 1
                continue
            if rid in self._pd_pending:
                self._reject(f"NIXL decoder request {rid!r} published two pending pulls")
            request = self.scheduler.requests.get(rid)
            remote = getattr(req_meta, "remote", None)
            remote_tokens = getattr(remote, "num_tokens", None)
            groups = tuple(
                tuple(int(block) for block in group) for group in req_meta.local_block_ids
            )
            local_cached = tuple(int(count) for count in req_meta.local_num_computed_blocks)
            if (
                request is None
                or remote_tokens is None
                or len(groups) != 1
                or len(local_cached) != 1
            ):
                self._reject(
                    f"NIXL decoder request {rid!r} has unsupported receive metadata: "
                    f"request={request is not None}, remote_tokens={remote_tokens!r}, "
                    f"groups={len(groups)}, cached={local_cached!r}"
                )
            local_tokens = local_cached[0] * self.block_size
            computed = int(request.num_computed_tokens)
            external_tokens = computed - local_tokens
            block_ids = groups[0]
            expected = (external_tokens + self.block_size - 1) // self.block_size
            if external_tokens <= 0 or computed > int(remote_tokens) or len(block_ids) != expected:
                self._reject(
                    f"NIXL decoder request {rid!r} maps {len(block_ids)} blocks for "
                    f"local={local_tokens}, computed={computed}, remote={remote_tokens}"
                )
            fills = tuple(
                (
                    block_id,
                    min(self.block_size, external_tokens - index * self.block_size),
                )
                for index, block_id in enumerate(block_ids)
            )
            self._pd_pending[rid] = (local_tokens, fills)
            self._pd_expect_content(rid, remote, block_ids, int(remote_tokens))
            self.stats["pd_receive_jobs"] += 1
            self.stats["pd_receive_blocks"] += len(block_ids)
            self.stats["pd_receive_tokens"] += external_tokens

    def _pd_completions(self, model_output) -> None:
        """Credit NIXL destinations and close producer epochs at exact worker completion."""
        if self.pd_kind != "nixl_pull":
            return
        output = getattr(model_output, "kv_connector_output", None)
        if output is None:
            return
        failed = {str(rid) for rid in (getattr(output, "failed_recving", ()) or ())}
        for rid_value in getattr(output, "finished_recving", ()) or ():
            rid = str(rid_value)
            pending = self._pd_pending.pop(rid, None)
            if pending is None:
                self.stats["pd_receive_completions_without_pending"] += 1
                continue
            if rid in failed:
                self.stats["pd_receive_failures"] += 1
                self._pd_content_expected.pop(rid, None)
                continue
            self._pd_check_content(rid)
            local_tokens, fills = pending
            for block_id, fill in fills:
                self.ledger.external_fill(
                    ExternalFill(
                        POOL,
                        block_id,
                        range(fill),
                        source="nixl_pull_load",
                        request=rid,
                    ),
                    site=f"nixl-load:{rid}@{self.seq}",
                )
            request = self.scheduler.requests.get(rid)
            loaded_tokens = local_tokens + sum(fill for _, fill in fills)
            if (
                request is not None
                and fills
                and loaded_tokens > 0
                and loaded_tokens == int(request.num_tokens)
            ):
                # The scheduler publishes the whole externally loaded prompt, then retreats by
                # one token so the decoder can project that token for logits.  Reuse the guarded
                # full-hit-tail normalization used by managed offload: it permits exactly this
                # private rewrite and still reports if another request shares the block first.
                self._offload_recompute_tails[rid] = (
                    loaded_tokens - 1,
                    {POOL: fills[-1][0]},
                    False,
                )
                self.stats["pd_full_hit_tail_recomputes"] += 1
            self.stats["pd_receive_completions"] += 1
            self.stats["pd_receive_completed_blocks"] += len(fills)
        for rid_value in getattr(output, "finished_sending", ()) or ():
            rid = str(rid_value)
            dispatches = self._pd_send_dispatches.pop(rid, ())
            for did in dispatches:
                self.ledger.complete(did)
            if dispatches:
                self.stats["pd_send_completions"] += 1
            else:
                self.stats["pd_send_completions_without_epoch"] += 1

    def _pd_manifest_path(self, remote_request_id: str) -> Path | None:
        content_dir = getattr(self, "_pd_content_dir", None)
        if content_dir is None:
            return None
        name = hashlib.sha256(remote_request_id.encode()).hexdigest()
        return content_dir / f"{name}.json"

    def _pd_block_digests(self, block_ids: tuple[int, ...]) -> tuple[str, ...]:
        """Hash every layer's exact physical bytes for each logical NIXL page."""
        import torch

        caches = tuple(getattr(self.runner, "kv_caches", ()))
        if not caches or any(cache.ndim < 1 for cache in caches):
            self._reject("NIXL content validation cannot identify Model Runner V2 KV tensors")
        if block_ids and any(int(cache.shape[0]) <= max(block_ids) for cache in caches):
            self._reject(
                "NIXL content validation requires LBHNC per-layer views with blocks on axis 0"
            )
        digests = []
        for block_id in block_ids:
            digest = hashlib.sha256()
            for cache in caches:
                # ``view(uint8)`` makes BF16 and quantized caches equally byte-exact.  The CPU
                # copy deliberately synchronizes this discovery-only E2E witness at the source
                # publication or completed destination pull.
                raw = cache[block_id].contiguous().view(torch.uint8).cpu().numpy().tobytes()
                digest.update(raw)
            digests.append(digest.hexdigest())
        self.stats["pd_content_blocks_hashed"] += len(block_ids)
        return tuple(digests)

    def _pd_publish_content(self, rid: str, block_ids: tuple[int, ...], num_tokens: int) -> None:
        path = self._pd_manifest_path(rid)
        if path is None:
            return
        digests = list(self._pd_block_digests(block_ids))
        if self.pd_fault == "content_digest" and not self.pd_fault_done and digests:
            self.pd_fault_done = True
            digests[0] = ("0" if digests[0][0] != "0" else "1") + digests[0][1:]
            self.stats["pd_content_faults_injected"] += 1
        payload = {
            "request": rid,
            "num_tokens": num_tokens,
            "block_size": self.block_size,
            "blocks": block_ids,
            "digests": digests,
        }
        temporary = path.with_suffix(f".{os.getpid()}.{self._pd_send_seq}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(temporary, path)
        self.stats["pd_content_manifests_published"] += 1

    def _pd_expect_content(
        self,
        rid: str,
        remote,
        local_blocks: tuple[int, ...],
        remote_tokens: int,
    ) -> None:
        if getattr(self, "_pd_content_dir", None) is None:
            return
        remote_rid = str(getattr(remote, "request_id", ""))
        path = self._pd_manifest_path(remote_rid)
        if path is None or not path.exists():
            self._reject(
                f"NIXL decoder request {rid!r} has no KVGuard source manifest for {remote_rid!r}"
            )
        payload = json.loads(path.read_text())
        remote_blocks = tuple(int(block) for group in remote.block_ids for block in group)
        manifest_blocks = tuple(int(block) for block in payload.get("blocks", ()))
        digests = tuple(str(digest) for digest in payload.get("digests", ()))
        if (
            int(payload.get("num_tokens", -1)) != remote_tokens
            or int(payload.get("block_size", -1)) != self.block_size
            or manifest_blocks != remote_blocks
            or len(digests) != len(manifest_blocks)
            or len(local_blocks) > len(manifest_blocks)
        ):
            self.reporter.emit(
                Report(
                    OutcomeKind.VIOLATION,
                    "mapping_mismatch",
                    "NIXL decoder metadata does not name the producer page sequence recorded "
                    f"at publication (remote_tokens={remote_tokens}, "
                    f"source_tokens={payload.get('num_tokens')}, local_pages={len(local_blocks)}, "
                    f"remote_pages={len(remote_blocks)}, source_pages={len(manifest_blocks)})",
                    policy=Policy.P3,
                    pool=POOL,
                    request=rid,
                    site=f"nixl-content-plan@{self.seq}",
                )
            )
            return
        # The worker's ordinary one-group prefix-cache rule end-trims P's full page list to D's
        # unhashed suffix.  Pair exactly that suffix with the decoder destinations.
        suffix = digests[-len(local_blocks) :] if local_blocks else ()
        self._pd_content_expected[rid] = tuple(zip(local_blocks, suffix, strict=True))
        self.stats["pd_content_plans"] += 1

    def _pd_check_content(self, rid: str) -> None:
        expected = getattr(self, "_pd_content_expected", {}).pop(rid, None)
        if expected is None:
            return
        blocks = tuple(block for block, _ in expected)
        observed = self._pd_block_digests(blocks)
        for (block_id, source), destination in zip(expected, observed, strict=True):
            if source == destination:
                continue
            self.stats["pd_content_mismatches"] += 1
            self.reporter.emit(
                Report(
                    OutcomeKind.VIOLATION,
                    "mapping_mismatch",
                    "NIXL completed a decoder destination page with bytes different from the "
                    "producer page paired to that logical suffix position",
                    policy=Policy.P3,
                    pool=POOL,
                    block=block_id,
                    request=rid,
                    site=f"nixl-content-complete@{self.seq}",
                )
            )
        self.stats["pd_content_completions"] += 1

    def _install_simple_offload_scheduler(self) -> None:
        """Retain the producer identity for finish-time SimpleCPU store pins.

        The store metadata drops the finished requests' ids, so capture the exact blocks at the
        scheduler method that creates each transfer, keeping a private recurrent-state source's
        logical owner until the DMA pin is released.
        """
        queue_finished = getattr(self.offload, "_queue_finished_eager_store", None)
        if queue_finished is None:
            self._reject("SimpleCPU scheduler has no finish-time eager-store boundary")

        def queue_finished_eager_store(request, block_ids):
            pending = self.offload._pending_finished_stores
            before = len(pending)
            out = queue_finished(request, block_ids)
            owner = str(request.request_id)
            for transfer in pending[before:]:
                routed = self._offload_pool_blocks(tuple(int(b) for b in transfer.gpu_block_ids))
                for pool, pool_blocks in routed.items():
                    for block_id in pool_blocks:
                        self._simple_store_owners[(pool, block_id)] = owner
                        self.stats["simple_offload_finish_owner_blocks"] += 1
            return out

        self.offload._queue_finished_eager_store = queue_finished_eager_store
        cleanup_load = getattr(self.offload, "_cleanup_load_request", None)
        if cleanup_load is not None:

            def cleanup_load_request(req_id):
                state = self.offload._reqs_to_load.get(req_id)
                state_source = "active"
                if state is None:
                    state = self.offload._abandoned_reqs_to_load.get(req_id)
                    state_source = "abandoned"
                block_ids = tuple(
                    int(block)
                    for block in getattr(getattr(state, "transfer_meta", None), "gpu_block_ids", ())
                )
                self._trace_simple_offload(
                    "load_cleanup_start",
                    request=str(req_id),
                    event_id=getattr(state, "load_event", None),
                    block_ids=block_ids,
                    state_source=state_source if state is not None else "missing",
                )
                try:
                    return cleanup_load(req_id)
                finally:
                    completed = getattr(self, "_simple_completed_load_pins", Counter())
                    for block_id in block_ids:
                        if completed[block_id] <= 0:
                            continue
                        completed[block_id] -= 1
                        if completed[block_id] <= 0:
                            del completed[block_id]
                        self.stats["offload_load_pins_released"] += 1
                    self._trace_simple_offload(
                        "load_cleanup_done",
                        request=str(req_id),
                        event_id=getattr(state, "load_event", None),
                        block_ids=block_ids,
                        state_source=state_source if state is not None else "missing",
                    )

            self.offload._cleanup_load_request = cleanup_load_request
        self.stats["simple_offload_scheduler_wrapped"] += 1

    def _install_simple_offload_worker(self, connector, connector_worker) -> None:
        """Bind SimpleCPU transfers to their worker CUDA-event high-watermarks."""
        transfer_worker = getattr(connector_worker, "worker_handler", None)
        if (
            type(connector).__name__ != "SimpleCPUOffloadConnector"
            or type(connector_worker).__name__ != "SimpleCPUOffloadConnector"
            or type(transfer_worker).__name__ != "SimpleCPUOffloadWorker"
        ):
            self._reject(
                "SimpleCPU offload worker is outside the admitted event boundary: "
                f"{type(connector).__name__}/{type(connector_worker).__name__}/"
                f"{type(transfer_worker).__name__}"
            )

        orig_wait_for_save = transfer_worker.wait_for_save

        def wait_for_save():
            metadata = transfer_worker._connector_metadata
            out = orig_wait_for_save()
            if metadata is None or int(metadata.store_event) < 0:
                return out
            event_id = int(metadata.store_event)
            pending = self._offload_store_jobs.get(event_id)
            if pending is None:
                self._reject(f"SimpleCPU store event {event_id} started without scheduler metadata")
            event = next(
                (event for idx, event in transfer_worker._store_events if int(idx) == event_id),
                None,
            )
            if transfer_worker.store_stream is None:
                self._reject(f"SimpleCPU store event {event_id} has no CUDA stream")
            rid, block_ids = pending
            self.stats["offload_store_starts"] += 1
            self._open_offload_store(
                event_id,
                block_ids,
                request=rid,
                stream=int(transfer_worker.store_stream.cuda_stream),
                event=event,
            )
            return out

        transfer_worker.wait_for_save = wait_for_save
        orig_poll = transfer_worker._poll_stream_events

        def poll_stream_events(is_store: bool):
            hwm = int(orig_poll(is_store))
            if is_store:
                for event_id in sorted(event for event in self._offload_store_jobs if event <= hwm):
                    self._offload_store_complete(event_id, waited=False)
                    self._offload_store_jobs.pop(event_id, None)
                    self.stats["offload_store_completions"] += 1
            else:
                for event_id in sorted(
                    event for event in self._simple_pending_loads if event <= hwm
                ):
                    self._complete_simple_offload_load(event_id)
            return hwm

        transfer_worker._poll_stream_events = poll_stream_events
        self.stats["simple_offload_worker_wrapped"] += 1

    # ------------------------------------------------------------------ extraction
    def _trace_offload(
        self,
        kind: str,
        job_id: int,
        request: str | None,
        block_ids: tuple[int, ...],
        job: Any | None = None,
    ) -> None:
        """Retain bounded attribution data without changing serving behavior.

        Discovery-only telemetry read from the scheduler's private job status at completion; an
        unexpected shape or representation must never make the attachment fail.
        """
        trace = getattr(self, "offload_events", None)
        if trace is None:
            return
        try:
            jobs = getattr(getattr(self, "offload", None), "_jobs", {})
            status = jobs.get(int(job_id)) if hasattr(jobs, "get") else None
            keys = getattr(status, "keys", ()) or ()
            cpu_spec = (
                getattr(job, "dst_spec", None)
                if kind == "store"
                else getattr(job, "src_spec", None)
            )
            cpu_chunks = [int(value) for value in getattr(cpu_spec, "block_ids", ())]
            manager = getattr(getattr(self, "offload", None), "manager", None)
            if type(manager).__name__ == "TieringOffloadingManager":
                manager = getattr(manager, "primary_tier", None)
            policy = getattr(manager, "_policy", None)
            key_chunks = []
            for key in keys:
                chunk = policy.get(key) if policy is not None else None
                key_chunks.append(
                    {
                        "cpu_chunk": (int(chunk.chunk_id) if chunk is not None else None),
                        "key": repr(key),
                    }
                )
            key_chunks.sort(
                key=lambda row: (
                    row["cpu_chunk"] is None,
                    -1 if row["cpu_chunk"] is None else row["cpu_chunk"],
                    row["key"],
                )
            )
            trace.append(
                {
                    "seq": self.seq + 1,
                    "kind": kind,
                    "job_id": int(job_id),
                    "request": request,
                    "blocks": list(block_ids),
                    "keys": sorted(repr(key) for key in (keys or ())),
                    "cpu_chunks": cpu_chunks,
                    "key_chunks": key_chunks,
                }
            )
        except Exception:
            # Attribution is not part of enforcement.  Losing one trace row is
            # preferable to perturbing or terminating the serving workload.
            stats = getattr(self, "stats", None)
            if stats is not None:
                stats["offload_trace_errors"] += 1

    def _trace_simple_offload(
        self,
        kind: str,
        *,
        request: str | None,
        event_id: int | None,
        block_ids: tuple[int, ...],
        **facts: Any,
    ) -> None:
        """Append bounded SimpleCPU lifecycle facts for discovery diagnostics only."""
        trace = getattr(self, "offload_events", None)
        if trace is None:
            return
        try:
            rows = []
            for block_id in block_ids:
                obj = self.pool.blocks[block_id]
                rows.append(
                    {
                        "block": block_id,
                        "refs": int(obj.ref_cnt),
                        "group": self.group_of.get(block_id),
                        "hash_tokens": getattr(obj, "block_hash_num_tokens", None),
                        "completed_pins": int(
                            getattr(self, "_simple_completed_load_pins", {}).get(block_id, 0)
                        ),
                    }
                )
            trace.append(
                {
                    "seq": self.seq + 1,
                    "kind": kind,
                    "event_id": None if event_id is None else int(event_id),
                    "request": request,
                    "blocks": rows,
                    **facts,
                }
            )
        except Exception:
            stats = getattr(self, "stats", None)
            if stats is not None:
                stats["offload_trace_errors"] += 1

    def _offload_loads(self, scheduler_output) -> None:
        """Record CPU-offload loads, crediting synchronous destinations immediately.

        A synchronous job completes its CPU-to-GPU copy before the forward, so its destinations
        are initialized before the snapshot's dispatch; an asynchronous job leaves its blocks
        uninitialized until the worker reports it complete.
        """
        if self.offload is None:
            return
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        if metadata is None:
            return
        if self.offload_kind == "simple":
            self._simple_offload_loads(metadata)
            return
        if type(metadata).__name__ != "OffloadingConnectorMetadata":
            self._reject(
                f"offload metadata is {type(metadata).__name__}, expected "
                "OffloadingConnectorMetadata"
            )
        for job_id in getattr(metadata, "jobs_to_flush", ()) or ():
            if job_id in self._offload_pending:
                self._offload_flushed.add(int(job_id))
        for job_id, job in metadata.load_jobs.items():
            try:
                block_ids, routed = self._managed_offload_blocks(job.dst_spec)
            except (AttributeError, TypeError, ValueError) as exc:
                self._reject(f"offload load job {job_id} has an unsupported GPU shape: {exc}")
            if bool(getattr(job, "sync", False)):
                self._credit_offload_load(int(job_id), block_ids, self.seq + 1, routed=routed)
                self.stats["offload_sync_load_jobs"] += 1
                self._trace_offload("load_sync", int(job_id), str(job.req_id), block_ids, job)
            else:
                rid = str(job.req_id)
                if rid in self._offload_pending_by_req:
                    self._reject(f"request {rid!r} has two asynchronous offload loads")
                self._offload_pending[int(job_id)] = (rid, block_ids)
                self._offload_pending_by_req[rid] = int(job_id)
                self._managed_offload_routes[int(job_id)] = routed
                self.stats["offload_async_load_jobs"] += 1
                self._trace_offload("load_async", int(job_id), rid, block_ids, job)
            self.stats["offload_load_jobs"] += 1
            self.stats["offload_load_blocks"] += len(block_ids)
        self.stats["offload_store_jobs"] += len(metadata.store_jobs)
        for job_id, job in metadata.store_jobs.items():
            try:
                block_ids, routed = self._managed_offload_blocks(job.src_spec)
            except (AttributeError, TypeError, ValueError) as exc:
                self._reject(f"offload store job has an unsupported GPU shape: {exc}")
            job_id = int(job_id)
            if job_id in self._offload_store_jobs:
                self._reject(f"offload store job {job_id} was published twice")
            self._offload_store_jobs[job_id] = (str(job.req_id), block_ids)
            self._managed_offload_routes[job_id] = routed
            self._managed_offload_store_generations[job_id] = {
                (pool, block_id): self.core_ledger.block(pool, block_id).generation
                for pool, pool_blocks in routed.items()
                for block_id in pool_blocks
            }
            self._trace_offload("store", job_id, str(job.req_id), block_ids, job)
            self.stats["offload_store_blocks"] += len(block_ids)

    def _simple_offload_loads(self, metadata) -> None:
        """Record SimpleCPU event metadata, preserving per-request load destinations."""
        if type(metadata).__name__ != "SimpleCPUOffloadMetadata":
            self._reject(
                f"SimpleCPU offload metadata is {type(metadata).__name__}, expected "
                "SimpleCPUOffloadMetadata"
            )
        load_event = int(metadata.load_event)
        if load_event >= 0:
            if load_event in self._simple_pending_loads:
                self._reject(f"SimpleCPU load event {load_event} was published twice")
            requests = []
            all_blocks = []
            for rid in metadata.load_event_to_reqs.get(load_event, ()):
                state = self.offload._reqs_to_load.get(rid)
                if state is None:
                    self._reject(
                        f"SimpleCPU load event {load_event} has no state for request {rid!r}"
                    )
                block_ids = tuple(int(block) for block in state.transfer_meta.gpu_block_ids)
                requests.append((str(rid), block_ids, int(state.request.num_computed_tokens)))
                all_blocks.extend(block_ids)
                if rid in self._offload_pending_by_req:
                    self._reject(f"request {rid!r} has two asynchronous offload loads")
                self._offload_pending_by_req[str(rid)] = load_event
                self._trace_simple_offload(
                    "load_published",
                    request=str(rid),
                    event_id=load_event,
                    block_ids=block_ids,
                    computed=int(state.request.num_computed_tokens),
                    finished=bool(getattr(state.request, "is_finished", lambda: False)()),
                )
            if tuple(all_blocks) != tuple(int(block) for block in metadata.load_gpu_blocks):
                self._reject(
                    f"SimpleCPU load event {load_event}'s request blocks disagree with metadata"
                )
            self._simple_pending_loads[load_event] = tuple(requests)
            self.stats["offload_async_load_jobs"] += len(requests)
            self.stats["offload_load_jobs"] += 1
            self.stats["offload_load_blocks"] += len(all_blocks)

        store_event = int(metadata.store_event)
        if store_event >= 0:
            if store_event in self._offload_store_jobs:
                self._reject(f"SimpleCPU store event {store_event} was published twice")
            block_ids = tuple(int(block) for block in metadata.store_gpu_blocks)
            self._offload_store_jobs[store_event] = (None, block_ids)
            self.stats["offload_store_jobs"] += 1
            self.stats["offload_store_blocks"] += len(block_ids)

    def _complete_simple_offload_load(self, event_id: int) -> None:
        pending = self._simple_pending_loads.pop(event_id, ())
        credited = 0
        for rid, block_ids, computed in pending:
            request = self.scheduler.requests.get(rid)
            self._trace_simple_offload(
                "load_complete_start",
                request=rid,
                event_id=event_id,
                block_ids=block_ids,
                computed=computed,
                scheduler_request=bool(request is not None),
                finished=bool(request is None or request.is_finished()),
            )
            if request is None or request.is_finished():
                self.stats["offload_async_loads_completed_after_abort"] += 1
                self.stats["offload_async_load_blocks_completed_after_abort"] += len(block_ids)
            # The event is the physical DMA completion boundary even if the request was
            # cancelled while the copy ran. Credit the destination before the scheduler
            # releases its transfer pins: it may preserve the loaded blocks in prefix cache,
            # while a later reallocation will clear the generation in the usual path.
            self._credit_offload_load(
                event_id,
                block_ids,
                self.seq + 1,
                request=rid,
                computed=computed,
            )
            completed_pins = getattr(self, "_simple_completed_load_pins", None)
            if completed_pins is None:
                completed_pins = self._simple_completed_load_pins = Counter()
            completed_pins.update(block_ids)
            self._trace_simple_offload(
                "load_complete_done",
                request=rid,
                event_id=event_id,
                block_ids=block_ids,
                computed=computed,
                scheduler_request=bool(request is not None),
                finished=bool(request is None or request.is_finished()),
            )
            credited += 1
            self._offload_pending_by_req.pop(rid, None)
        if credited:
            self.stats["offload_async_loads_completed"] += credited

    def _offload_pool_blocks(self, block_ids: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Route physical SimpleCPU blocks to the logical pools whose layers were copied."""
        if not self.hybrid:
            routed = {POOL: block_ids}
            for logical_pool in self._shared_logical_pools(POOL):
                routed[logical_pool] = block_ids
            return routed
        by_pool: dict[str, list[int]] = {}
        for block_id in block_ids:
            gid = self.group_of.get(block_id)
            if gid is None:
                self._reject(f"SimpleCPU transfer block {block_id} has no recorded KV cache group")
            pool = self.pool_of_group[gid]
            by_pool.setdefault(pool, []).append(block_id)
            for logical_pool in self._shared_logical_pools(pool):
                by_pool.setdefault(logical_pool, []).append(block_id)
        return {pool: tuple(ids) for pool, ids in by_pool.items()}

    def _managed_offload_blocks(
        self, spec: Any
    ) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
        """Decode a managed job's physical blocks into KVGuard logical pools."""
        group_configs = tuple(self.offload.config.kv_group_configs)
        group_ids = tuple(int(config.group_idx) for config in group_configs)
        grouped = _grouped_gpu_blocks(spec, group_ids)
        routed: dict[str, list[int]] = {}
        for gid, block_ids in grouped.items():
            pool = self.pool_of_group.get(gid)
            if pool is None:
                raise ValueError(f"transfer names unknown KV cache group {gid}")
            for block_id in block_ids:
                allocated_gid = self.group_of.get(block_id)
                if allocated_gid is not None and allocated_gid != gid:
                    raise ValueError(
                        f"block {block_id} belongs to group {allocated_gid}, metadata says {gid}"
                    )
            routed.setdefault(pool, []).extend(block_ids)
            for logical_pool in self._shared_logical_pools(pool):
                routed.setdefault(logical_pool, []).extend(block_ids)
        flat = tuple(int(v) for v in np.asarray(spec.block_ids).reshape(-1))
        return flat, {pool: tuple(ids) for pool, ids in routed.items()}

    def _credit_offload_load(
        self,
        job_id: int,
        block_ids: tuple[int, ...],
        seq: int,
        *,
        request: str | None = None,
        computed: int | None = None,
        routed: dict[str, tuple[int, ...]] | None = None,
        track_last_token: bool = False,
    ) -> dict[str, int]:
        # Each hybrid block belongs to one scheduler group. In the EAGLE case, the one physical
        # attention-group transfer carries both target and draft layers, which KVGuard represents
        # as independently initialized logical pools over the same block ids.
        routed = self._offload_pool_blocks(block_ids) if routed is None else routed
        tail_blocks: dict[str, int] = {}
        for pool, pool_blocks in routed.items():
            block_size = self.core_ledger.pools[pool].block_size
            for block_id in pool_blocks:
                fill = block_size
                if request is not None and computed is not None and block_size > 1:
                    gid = self.group_of[block_id] if self.hybrid else self.kv_gid
                    manager = self.managers[gid] if self.hybrid else self.manager
                    blocks = manager.req_to_blocks.get(request, ())
                    index = next(
                        (
                            idx
                            for idx, block in enumerate(blocks)
                            if not block.is_null and int(block.block_id) == block_id
                        ),
                        None,
                    )
                    if index is None:
                        self._reject(
                            f"SimpleCPU load block {block_id} is absent from request {request!r}"
                        )
                    physical_size = int(manager.block_size)
                    fill = max(0, min(block_size, int(computed) - index * physical_size))
                    if (
                        track_last_token
                        and int(computed) > 0
                        and index == (int(computed) - 1) // physical_size
                        and block_size == physical_size
                    ):
                        tail_blocks[pool] = block_id
                self.ledger.external_fill(
                    ExternalFill(
                        pool,
                        block_id,
                        range(fill),
                        source="cpu_offload_load",
                        request=request,
                    ),
                    site=f"offload-load:{job_id}@{seq}",
                )
        self._offload_filled.update(block_ids)
        filled_pools = getattr(self, "_offload_filled_pools", None)
        if filled_pools is None:
            filled_pools = self._offload_filled_pools = set()
        filled_pools.update(
            (pool, block_id) for pool, pool_blocks in routed.items() for block_id in pool_blocks
        )
        self.stats["offload_load_pool_blocks"] += sum(len(ids) for ids in routed.values())
        return tail_blocks

    def _offload_completions(self, model_output) -> None:
        if self.offload is None or getattr(self, "offload_kind", None) == "simple":
            return
        connector_output = getattr(model_output, "kv_connector_output", None)
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        completed = getattr(metadata, "completed_jobs", {})
        for job_id, count in completed.items():
            job_id = int(job_id)
            if job_id in self._offload_store_jobs:
                self._offload_store_complete(job_id, waited=False)
                self._offload_store_jobs.pop(job_id, None)
                getattr(self, "_managed_offload_routes", {}).pop(job_id, None)
                self.stats["offload_store_completions"] += 1
            pending = self._offload_pending.get(job_id)
            if pending is None or int(count) < 1:
                continue
            rid, block_ids = pending
            request = self.scheduler.requests.get(rid)
            finished = bool(request is None or request.is_finished())
            if request is not None and request.is_finished():
                self.stats["offload_async_loads_completed_after_abort"] += 1
            if job_id in self._offload_flushed:
                self._offload_flushed.discard(job_id)
                self.stats["offload_async_loads_flushed"] += 1
            else:
                computed = None if finished else int(request.num_computed_tokens)
                track_last_token = bool(
                    not finished
                    and computed is not None
                    and computed > 0
                    and computed == int(request.num_tokens)
                )
                tail_blocks = self._credit_offload_load(
                    job_id,
                    block_ids,
                    self.seq + 1,
                    request=rid if request is not None else None,
                    computed=computed,
                    routed=getattr(self, "_managed_offload_routes", {}).get(job_id),
                    track_last_token=track_last_token,
                )
                if track_last_token and tail_blocks:
                    tails = getattr(self, "_offload_recompute_tails", None)
                    if tails is None:
                        tails = self._offload_recompute_tails = {}
                    tails[rid] = (computed - 1, tail_blocks, False)
                    self.stats["offload_full_hit_tail_recomputes"] += 1
                self.stats["offload_async_loads_completed"] += 1
            del self._offload_pending[job_id]
            getattr(self, "_managed_offload_routes", {}).pop(job_id, None)
            self._offload_pending_by_req.pop(rid, None)

    def _offload_store_started(self, job_id: int, transfer_worker, flushed: bool) -> None:
        pending = self._offload_store_jobs.get(job_id)
        if pending is None:
            self._reject(f"offload store job {job_id} started without scheduler metadata")
        self.stats["offload_store_starts"] += 1
        if flushed:
            self.stats["offload_store_flushes"] += 1
            return
        transfers = transfer_worker._store_handler._transfers
        transfer = transfers[-1] if transfers else None
        if transfer is None or int(transfer.job_id) != job_id:
            self._reject(f"offload store job {job_id} has no matching transfer-stream record")
        event = transfer_worker._store_handler._transfer_events.get(job_id)
        if event is None:
            self._reject(f"offload store job {job_id} has no matching completion event")
        stream = int(transfer.stream.cuda_stream)
        rid, block_ids = pending
        self._open_offload_store(
            job_id,
            block_ids,
            request=rid,
            stream=stream,
            event=event,
            routed=getattr(self, "_managed_offload_routes", {}).get(job_id),
        )

    def _open_offload_store(
        self,
        job_id: int,
        block_ids: tuple[int, ...],
        *,
        request: str | None,
        stream: int,
        event: Any,
        routed: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        """Open the logical GPU-source reads for one submitted CPU store."""
        if job_id in self._offload_store_epochs:
            self._reject(f"offload store job {job_id} opened its source read twice")
        dispatches = []
        routed = self._offload_pool_blocks(block_ids) if routed is None else routed
        published_generations = getattr(self, "_managed_offload_store_generations", {}).get(
            job_id, {}
        )
        block_method = getattr(self.core_ledger, "block", None)
        for pool, pool_blocks in routed.items():
            suffix = "" if pool == POOL else ":draft" if pool == DRAFT_POOL else f":{pool}"
            did = f"offload-store:{job_id}{suffix}"
            preinitialized = frozenset(
                block_id
                for block_id in pool_blocks
                if (pool, block_id) in getattr(self, "_zeroed_blocks", set())
            )
            retained_free = frozenset(
                block_id
                for block_id in pool_blocks
                if callable(block_method)
                and (shadow := block_method(pool, block_id)).state is BlockState.FREE
                and published_generations.get((pool, block_id)) == shadow.generation
            )
            self.ledger.external_read(
                ExternalRead(
                    did,
                    pool,
                    pool_blocks,
                    range(self.core_ledger.pools[pool].block_size),
                    source="cpu_offload_store",
                    request=request,
                    stream=stream,
                    preinitialized_blocks=preinitialized,
                    retained_free_blocks=retained_free,
                ),
                site=f"{did}@{self.seq}",
            )
            dispatches.append(did)
            self.stats["offload_store_retained_free_blocks"] += len(retained_free)
        self._offload_store_epochs.add(job_id)
        self._offload_store_dispatches[job_id] = tuple(dispatches)
        self._offload_store_events[job_id] = event
        self.stats["offload_store_read_epochs"] += 1
        self.stats["offload_store_pool_read_epochs"] += len(routed)

    def _offload_store_complete(self, job_id: int, *, waited: bool) -> None:
        getattr(self, "_managed_offload_store_generations", {}).pop(job_id, None)
        if job_id not in self._offload_store_epochs:
            return
        for dispatch_id in self._offload_store_dispatches.pop(job_id, ()):
            self.ledger.complete(dispatch_id)
        self._offload_store_epochs.remove(job_id)
        self._offload_store_events.pop(job_id, None)
        key = "offload_store_epochs_waited" if waited else "offload_store_epochs_completed"
        self.stats[key] += 1

    def _flush_offload_store_epochs(self, scheduler_output) -> None:
        """Close the store reads the worker waits before this step's kernels.

        The worker submits any deferred store and then waits every job in the step's
        ``jobs_to_flush`` from ``pre_forward``, which the runner calls after ``prepare_attn``. The
        step's writes are therefore ordered after those stores, so the attachment, which validates
        the step at ``prepare_attn``, must not judge them against a read epoch the same step
        retires before any of its kernels run.
        """
        if getattr(self, "offload_kind", None) != "managed" or not self._offload_store_epochs:
            return
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        for job_id in getattr(metadata, "jobs_to_flush", ()) or ():
            job_id = int(job_id)
            if job_id in self._offload_store_epochs:
                self._offload_store_complete(job_id, waited=True)
                self.stats["offload_store_epochs_flushed_before_forward"] += 1

    def _reap_offload_store_epochs(self) -> None:
        """Close store reads whose completion the worker has already observed before forward.

        The handler recycles one pooled CUDA event per transfer, so a held event's ``query`` may
        lag; a job the handler no longer tracks has been observed complete by ``get_finished``.
        """
        handler = getattr(getattr(self, "_offload_transfer_worker", None), "_store_handler", None)
        tracked = getattr(handler, "_transfer_events", None)
        for job_id, event in list(self._offload_store_events.items()):
            if tracked is not None and job_id not in tracked:
                self._offload_store_complete(job_id, waited=False)
                self.stats["offload_store_epochs_handler_finished"] += 1
            elif event is not None and event.query():
                self._offload_store_complete(job_id, waited=False)
                self.stats["offload_store_epochs_queried_complete"] += 1

    def _offload_computed(self, rid: str, computed: int) -> int:
        mooncake_pending = getattr(self, "_mooncake_pending", {}).get(rid)
        if mooncake_pending is not None:
            # The external frontier is optimistic until the native GET reports completion.
            return min(int(computed), int(mooncake_pending[0]))
        pd_pending = getattr(self, "_pd_pending", {}).get(rid)
        if pd_pending is not None:
            # The scheduler records the remote frontier optimistically while the request is
            # parked.  Only its block-aligned local hit exists before ``finished_recving``.
            return min(int(computed), int(pd_pending[0]))
        if rid not in self._offload_pending_by_req:
            return int(computed)
        if getattr(self, "offload_kind", None) == "simple":
            # SimpleCPU exposes the aggregate external frontier but not the local-prefix split
            # in scheduler metadata. The blocked request cannot dispatch yet, so zero is the
            # conservative claim until its exact load event completes.
            return 0
        status = self.offload._req_status.get(rid)
        if status is None:
            return 0
        return int(status.num_locally_computed_tokens)

    def _snapshot_computed(self, rid: str, computed: int, scheduled: int) -> int:
        """Return a post-schedule count whose pre-step portion is authoritative now.

        A pending speculative result leaves vLLM's CPU counter optimistic, so cap the pre-step
        portion at the last target position actually written by its newest in-flight step while
        retaining this step's scheduled count.
        """
        computed = max(self._offload_computed(rid, int(computed)), int(scheduled))
        pending = self._speculative_caps.get(rid)
        if pending is None:
            return computed
        step_key, cap = pending
        if int(scheduled) > 0 and step_key not in self._speculative_steps:
            # The producing step has retired and this request is executing again.  A connector
            # can leave the CPU counter at its local-prefix frontier for the intervening idle
            # snapshot, but once the scheduler admits new tokens that completed cap is stale.
            # Keeping it would make the new concrete write range appear outside the request.
            self._speculative_caps.pop(rid, None)
            self.stats["speculative_completed_caps_retired"] += 1
            return computed
        pre = computed - int(scheduled)
        if pre <= cap:
            return computed
        self.stats["speculative_snapshot_caps"] += 1
        self.stats["speculative_snapshot_capped_tokens"] += pre - cap
        return cap + int(scheduled)

    def _record_speculative_step(
        self, step_key: int, req_ids, computed: np.ndarray, scheduled: np.ndarray
    ) -> None:
        frontiers = {
            str(rid): int(frontier) + int(count)
            for rid, frontier, count in zip(req_ids, computed, scheduled, strict=True)
        }
        self._speculative_steps.setdefault(step_key, {}).update(frontiers)
        for rid, frontier in frontiers.items():
            self._speculative_caps[rid] = (step_key, frontier)

    def _complete_speculative_step(self, step_key: int) -> None:
        # Keep the concrete target-write frontier until that request executes again: a connector
        # can leave the CPU count at an older local-prefix frontier for one snapshot, and dropping
        # the frontier here would look like a physical KV rollback. The next dispatch supersedes
        # it and performs the exact device-derived rollback first.
        self._speculative_steps.pop(step_key, None)

    def _restore_speculative_target_writes(self, snap: Snapshot) -> None:
        """Retain concrete target writes across a lagging scheduler snapshot.

        A connector can keep ``num_computed_tokens`` at its local-prefix frontier for one snapshot
        after speculative target kernels ran; re-credit only the mapped slots inside the accepted
        live range. The next execution descriptor still rolls every rejected position back.
        """
        if not self._speculative_caps:
            return
        target_pools = {
            group["pool"]
            for group in self.groups
            if group["kind"] in ("attention", "window") and not group.get("draft", False)
        }
        for row in snap.requests:
            if row.pool not in target_pools:
                continue
            pending = self._speculative_caps.get(str(row.request))
            if pending is None:
                continue
            stop = min(int(row.num_tokens), int(pending[1]))
            start = int(row.num_computed)
            if stop <= start:
                continue
            geometry = self.core_ledger.pools[row.pool]
            restored = 0
            for position in range(start, stop):
                slot = row.slot_at(geometry, position)
                if slot is None:
                    continue
                block = geometry.block_of(slot)
                offset = slot - block * geometry.block_size
                if self.ledger.block(row.pool, block).is_filled(offset):
                    continue
                self.ledger.external_fill(
                    ExternalFill(
                        row.pool,
                        block,
                        range(offset, offset + 1),
                        source="speculative_target_write",
                        request=row.request,
                    ),
                    site=f"speculative-target-retain@{self.seq}",
                )
                restored += 1
            if restored:
                self._speculative_retained[(row.pool, str(row.request))] = stop
                self.stats["speculative_target_slots_retained"] += restored
                self.stats["speculative_target_rows_retained"] += 1

    def _copies(self, scheduler_output) -> None:
        """The worker's copy-on-write block copies of this step (a partial prefix hit's shared
        tail block copied into the request's private block, run before the step's kernels):
        external copies in the pool of the blocks' group, over the positions the destination's
        request has computed (the step's new or resumed requests carry the counts), recorded
        before the snapshot so that the request's computed claim finds them filled."""
        copies = getattr(scheduler_output, "kv_cache_block_copies", None)
        if not copies:
            return
        owner: dict[int, tuple[str, int]] = {}
        for r in scheduler_output.scheduled_new_reqs:
            for ids in r.block_ids:
                for b in ids:
                    owner[int(b)] = (r.req_id, int(r.num_computed_tokens))
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            new_ids = cached.new_block_ids[i] if i < len(cached.new_block_ids) else None
            if not new_ids:
                continue
            for ids in new_ids:
                for b in ids:
                    owner[int(b)] = (rid, int(cached.num_computed_tokens[i]))
        site = f"cow@{self.seq + 1}"
        for c in copies:
            src, dst = int(c.src_block_id), int(c.dst_block_id)
            g = self.group_of.get(dst)
            gs = self.group_of.get(src)
            if g is None or (gs is not None and gs != g):
                self.stats["cow_copies_rejected"] += 1
                self.reporter.emit(
                    integration_error(
                        "cow_group_unknown",
                        f"copy of block {src} (group {gs}) into block {dst} (group {g})",
                        pool=POOL,
                        block=dst,
                        site=site,
                    )
                )
                continue
            rid, pre = owner.get(dst, (None, None))
            if rid is None:
                self.stats["cow_copies_without_request"] += 1
            pool = self.pool_of_group[g]
            if rid is None:
                # Mamba align keeps writing its running source and moves the prefix-cache
                # entry to this copied destination.  The destination consequently has no
                # request row; after the copy it is an immutable cached checkpoint.
                self._cow_cache_blocks.add((pool, dst))
                for logical_pool in self._shared_logical_pools(pool):
                    self._cow_cache_blocks.add((logical_pool, dst))
            else:
                self._cow_request_blocks.add((pool, dst))
                for logical_pool in self._shared_logical_pools(pool):
                    self._cow_request_blocks.add((logical_pool, dst))
            self._trace_simple_offload(
                "cow_copy",
                request=rid,
                event_id=None,
                block_ids=(src, dst),
                pool=pool,
                computed=pre,
            )
            copy = cow_copy(pool, src, dst, self.core_ledger.pools[pool].block_size, pre)
            self.ledger.external_copy(copy, site=site)
            for logical_pool in self._shared_logical_pools(pool):
                self.ledger.external_copy(
                    cow_copy(logical_pool, src, dst, self.block_size, pre),
                    site=f"{site}:{logical_pool}",
                )
            self.stats["cow_copies"] += 1

    def _zero_fills(self, scheduler_output) -> None:
        """Record vLLM's pre-forward zeroing of new attention blocks.

        This is a full external fill (a CPU-offload store can later copy the whole physical page),
        not a logical KV fill: the first model write to each token position must stay legal.
        Recurrent-state groups are absent from ``new_block_ids_to_zero``.
        """
        block_ids = getattr(scheduler_output, "new_block_ids_to_zero", None)
        if not block_ids:
            return
        for block_id in (int(value) for value in block_ids):
            gid = self.group_of.get(block_id, self.kv_gid if not self.hybrid else None)
            if gid is None:
                self._reject(f"zeroed block {block_id} has no recorded KV cache group")
            pool = self.pool_of_group[gid]
            self._zeroed_blocks.add((pool, block_id))
            for logical_pool in self._shared_logical_pools(pool):
                self._zeroed_blocks.add((logical_pool, block_id))
            self.stats["zero_filled_blocks"] += 1

    def _release_cow_pins(self, blocks) -> None:
        """Forget CoW endpoint references released by the scheduler's execution fence."""
        block_ids = tuple(int(block.block_id) for block in blocks)
        self._trace_simple_offload(
            "cow_release_start",
            request=None,
            event_id=None,
            block_ids=block_ids,
        )
        for block in blocks:
            block_id = int(block.block_id)
            if self._cow_pins[block_id] <= 0:
                continue
            self._cow_pins[block_id] -= 1
            if self._cow_pins[block_id] <= 0:
                del self._cow_pins[block_id]
            self.stats["cow_pins_released"] += 1
        self._trace_simple_offload(
            "cow_release_done",
            request=None,
            event_id=None,
            block_ids=block_ids,
        )

    def _trace_snapshot_refs(self, snap: Snapshot) -> None:
        """Record only suspicious/shared partial rows under the opt-in discovery trace."""
        trace = getattr(self, "offload_events", None)
        if trace is None:
            return
        try:
            holders: dict[tuple[str, int], list[str]] = {}
            for request in snap.requests:
                for block_id in request.blocks:
                    if block_id is not None:
                        holders.setdefault((request.pool, int(block_id)), []).append(
                            str(request.request)
                        )
            for block in snap.blocks:
                mapped = holders.get((block.pool, block.block), [])
                mismatch = block.refs is not None and int(block.refs) != len(mapped)
                partial_shared = (
                    block.state is BlockState.PUBLISHED
                    and self._partial_attention_hash(block.pool, block.block)
                )
                if not mismatch and not partial_shared:
                    continue
                obj = self.pool.blocks[block.block]
                trace.append(
                    {
                        "seq": self.seq,
                        "kind": "snapshot_ref",
                        "pool": block.pool,
                        "block": block.block,
                        "state": block.state.value,
                        "refs": block.refs,
                        "raw_refs": int(obj.ref_cnt),
                        "holders": mapped,
                        "hash_tokens": getattr(obj, "block_hash_num_tokens", None),
                        "cow_pins": int(self._cow_pins[block.block]),
                        "cow_cache": (block.pool, block.block) in self._cow_cache_blocks,
                        "cow_request": (block.pool, block.block) in self._cow_request_blocks,
                        "completed_load_pins": int(self._simple_completed_load_pins[block.block]),
                    }
                )
        except Exception:
            self.stats["offload_trace_errors"] += 1

    def _normalize_cow_reservations(self, snap: Snapshot) -> Snapshot:
        """Remove execution-fence CoW pins from framework reference counts."""
        if not self._cow_pins:
            return snap
        normalized = []
        reservations = []
        mapped = {
            (request.pool, int(block))
            for request in snap.requests
            for block in request.blocks
            if block is not None
        }
        adjusted = 0
        for block in snap.blocks:
            pins = self._cow_pins[block.block]
            if not pins or block.refs is None:
                normalized.append(block)
                continue
            key = (block.pool, block.block)
            if key in self._cow_cache_blocks:
                # The cache-copy destination is not mapped by a request; keep its retention
                # explicit as one synthetic published reader (overlapping copies add physical pins
                # but not logical mappings). It is not in the free queue, so marking it reserved
                # would make the free-count partition wrong.
                block = replace(
                    block,
                    state=BlockState.PUBLISHED,
                    reserved=False,
                    refs=1,
                )
                reservations.append(
                    RequestSnapshot(
                        f"__kvguard_cow_pin__:{block.pool}:{block.block}",
                        block.pool,
                        (block.block,),
                        1,
                        1,
                    )
                )
                normalized.append(block)
                adjusted += 1
                continue
            refs = max(0, block.refs - pins)
            obj = self.pool.blocks[block.block]
            if key in self._cow_request_blocks:
                block = replace(
                    block,
                    state=BlockState.PRIVATE,
                    reserved=False,
                    refs=refs,
                )
            elif refs == 0 and obj.block_hash is not None:
                if key not in mapped:
                    # An ordinary partial-hit source leaves the request's table but retains
                    # its hit reference until the copy fence. It is physically outside the
                    # free queue for that interval, represented by a synthetic published
                    # reader rather than a reserved/free-queue cache entry.
                    geometry = self.core_ledger.pools[block.pool]
                    block = replace(
                        block,
                        state=BlockState.PUBLISHED,
                        reserved=False,
                        refs=1,
                    )
                    reservations.append(
                        RequestSnapshot(
                            f"__kvguard_cow_pin__:{block.pool}:{block.block}",
                            block.pool,
                            (block.block,),
                            geometry.block_size,
                            0,
                        )
                    )
                else:
                    block = replace(
                        block,
                        state=BlockState.PUBLISHED,
                        reserved=True,
                        refs=0,
                    )
            elif (
                refs <= 1
                and key in mapped
                and self._partial_attention_hash(block.pool, block.block)
            ):
                # The raw count made the partial block look shared, but the other reference is
                # only this copy's source retention. Its sole mapped request may legally resume
                # at the hashed boundary and fill the physical block's tail.
                block = replace(
                    block,
                    state=BlockState.PRIVATE,
                    reserved=False,
                    refs=refs,
                )
            else:
                block = replace(block, refs=refs)
            normalized.append(block)
            adjusted += 1
        self.stats["cow_pinned_blocks_normalized"] += adjusted
        self.stats["cow_pin_rows"] += len(reservations)
        return replace(
            snap,
            requests=snap.requests + tuple(reservations),
            blocks=tuple(normalized),
        )

    def _snapshot(self, scheduler_output) -> None:
        t0 = time.perf_counter()
        self.seq += 1
        live_request_ids = set(self.scheduler.requests)
        for rid in self._speculative_caps.keys() - live_request_ids:
            del self._speculative_caps[rid]
            self.stats["speculative_frontiers_pruned"] += 1
        for key in self._speculative_retained.keys() - {
            (pool, rid) for rid in live_request_ids for pool in self.core_ledger.pools
        }:
            del self._speculative_retained[key]
        scheduled = scheduler_output.num_scheduled_tokens
        audit = self.seq % self.audit_every == 0
        stream = current_stream_handle() if self.stream_epochs else None
        if self.snapshot_mode == "full":
            snap = self._snapshot_full(scheduled, audit, stream)
        else:
            snap = self._snapshot_incremental(scheduled, audit, stream)
        if self._deferred_rows:
            snap = replace(snap, requests=snap.requests + self._deferred_rows_for_snapshot())
        if self._shared_draft_pool():
            snap = self._with_eagle_snapshot(snap)
        elif self.dflash and self.dflash_separate:
            snap = self._with_dflash_separate_snapshot(snap)
        if self.multi_module_mtp:
            snap = self._with_mtp_frontiers(snap)
        snap = self._normalize_cow_reservations(snap)
        snap = self._normalize_offload_reservations(snap)
        snap = self._normalize_mooncake_reservations(snap)
        snap = self._normalize_publications(snap)
        snap, invalidations = self._normalize_offload_recompute_tails(snap)
        if self.offload_events is not None:
            for row in snap.requests:
                if row.pool != POOL or None not in row.blocks:
                    continue
                request = self.scheduler.requests.get(row.request)
                self.offload_events.append(
                    {
                        "seq": self.seq,
                        "kind": "snapshot_hole",
                        "request": row.request,
                        "blocks": list(row.blocks),
                        "num_tokens": row.num_tokens,
                        "num_computed": row.num_computed,
                        "scheduled": int(scheduled.get(row.request, 0)),
                        "scheduler_computed": (
                            None if request is None else int(request.num_computed_tokens)
                        ),
                        "scheduler_finished": bool(request is None or request.is_finished()),
                        "pending_load": row.request in self._offload_pending_by_req,
                    }
                )
        self._trace_snapshot_refs(snap)
        if self.mooncake_kind == "store" and self.offload_events is not None:
            request_index = snap.request_index()
            for block in snap.blocks:
                if block.pool != DRAFT_POOL or block.state is not BlockState.PUBLISHED:
                    continue
                shadow = self.core_ledger.block(block.pool, block.block)
                geometry = self.core_ledger.pools[block.pool]
                full_mask = (1 << geometry.block_size) - 1
                if shadow.state is not BlockState.PRIVATE or shadow.filled == full_mask:
                    continue
                holders = [
                    {
                        "request": str(request),
                        "num_tokens": row.num_tokens,
                        "num_computed": row.num_computed,
                        "positions": [
                            index for index, value in enumerate(row.blocks) if value == block.block
                        ],
                    }
                    for (request, pool), row in request_index.items()
                    if pool == block.pool and block.block in row.blocks
                ]
                self.offload_events.append(
                    {
                        "seq": self.seq,
                        "kind": "mooncake_draft_publish_candidate",
                        "block": block.block,
                        "generation": shadow.generation,
                        "filled": shadow.filled,
                        "pending_publish": shadow.pending_publish,
                        "refs": block.refs,
                        "holders": holders,
                    }
                )
        self.ledger.observe(snap)
        self._restore_speculative_target_writes(snap)
        # The completion can extend the loaded request beyond the prior snapshot's local
        # frontier.  Adopt this scheduler snapshot first, then invalidate its exact tail while
        # the normalization above still keeps the one-holder block private.
        for pool, request, position in invalidations:
            self.ledger.invalidate_positions(
                pool,
                request,
                (position,),
                site=f"offload-full-hit-recompute@{self.seq}",
            )
        self.stats["snapshots"] += 1
        self.stats["snapshot_requests"] += len(snap.requests)
        self.stats["snapshot_blocks"] += len(snap.blocks)
        dt = time.perf_counter() - t0
        self.timings["snapshot_s"] += dt
        if audit:
            # The audited snapshots' whole cost, apart from the ordinary steps' (a part of
            # snapshot_s; no _s suffix: the records sum the _s timings per step).
            self.timings["snapshot_audit_steps"] += dt

    def _normalize_offload_recompute_tails(
        self, snap: Snapshot
    ) -> tuple[Snapshot, tuple[tuple[str, str, int], ...]]:
        """Keep vLLM's exact full-hit replay tail private until its one real rewrite.

        On completing a full-prompt restore the scheduler retreats the semantic frontier by one
        token so that token is projected again for logits. This is a private invalidation only
        while the block has one mapper and one physical reference; if another request adopts it
        first, no normalization is applied and the core reports the published overwrite.
        """
        tails = getattr(self, "_offload_recompute_tails", None)
        if not tails:
            return snap, ()
        requests = snap.request_index()
        request_indices = {
            (row.request, row.pool): index for index, row in enumerate(snap.requests)
        }
        normalized_requests = list(snap.requests)
        blocks = {(block.pool, block.block): block for block in snap.blocks}
        mappers: Counter[tuple[str, int]] = Counter(
            (row.pool, int(block))
            for row in snap.requests
            for block in row.blocks
            if block is not None
        )
        normalized = {(block.pool, block.block): block for block in snap.blocks}
        invalidations: list[tuple[str, str, int]] = []
        for request, (position, tail_blocks, admitted) in list(tails.items()):
            scheduler_request = self.scheduler.requests.get(request)
            if scheduler_request is None or scheduler_request.is_finished():
                tails.pop(request, None)
                continue
            selected: list[tuple[str, int]] = []
            safe = True
            for pool, expected_block in tail_blocks.items():
                row = requests.get((request, pool))
                geometry = self.core_ledger.pools[pool]
                index = position // geometry.block_size
                actual = None if row is None or index >= len(row.blocks) else row.blocks[index]
                block = blocks.get((pool, expected_block))
                if (
                    actual != expected_block
                    or block is None
                    or block.reserved
                    or block.refs != 1
                    or mappers[(pool, expected_block)] != 1
                ):
                    safe = False
                    break
                selected.append((pool, expected_block))
            if not safe or not selected:
                # Never relax a tail after it was observed shared or moved: publication remains
                # sticky, and its later rewrite is judged by the ordinary policy.
                tails.pop(request, None)
                self.stats["offload_full_hit_tail_recompute_rejected"] += 1
                continue
            for pool, block_id in selected:
                normalized[(pool, block_id)] = replace(
                    normalized[(pool, block_id)],
                    state=BlockState.PRIVATE,
                    reserved=False,
                )
                row_index = request_indices[(request, pool)]
                row = normalized_requests[row_index]
                if row.num_tokens <= position:
                    # build_snapshot normally ends an unscheduled row at its computed frontier.
                    # The scheduler nevertheless owns this restored token and its mapped slot;
                    # expose that one-token live tail while it waits under pressure.
                    normalized_requests[row_index] = replace(row, num_tokens=position + 1)
                    self.stats["offload_full_hit_tail_live_extensions"] += 1
                if not admitted:
                    invalidations.append((pool, request, position))
            tails[request] = (position, tail_blocks, True)
            self.stats["offload_full_hit_tail_private_snapshots"] += 1
        return (
            replace(
                snap,
                requests=tuple(normalized_requests),
                blocks=tuple(normalized.values()),
            ),
            tuple(invalidations),
        )

    def _normalize_offload_reservations(self, snap: Snapshot) -> Snapshot:
        """Remove SimpleCPU's transfer pins from request reference counts.

        A load destination or store source in flight raises the block refcount, but those pins
        are not request mappings. KV store sources are published; a recurrent-state source stays
        private and rewritable while its pin keeps it reserved after its request row leaves.
        """
        if getattr(self, "offload_kind", None) != "simple":
            return snap
        load_pins: Counter[int] = Counter()
        for states_name in ("_reqs_to_load", "_abandoned_reqs_to_load"):
            for state in getattr(self.offload, states_name, {}).values():
                transfer = getattr(state, "transfer_meta", None)
                load_pins.update(int(block) for block in getattr(transfer, "gpu_block_ids", ()))
        for block_id, count in getattr(self, "_simple_completed_load_pins", {}).items():
            load_pins[block_id] = max(load_pins[block_id], int(count))
        # Eager mode maintains ``_in_flight_store_gpu_blocks`` for deduplication. Lazy mode
        # deliberately does not, because it selects already-cached blocks from the free queue.
        # The transfer records are authoritative in both modes, including jobs queued at request
        # finish and jobs abandoned by a connector reset while their DMA is still running.
        store_pins = {
            int(block) for block in getattr(self.offload, "_in_flight_store_gpu_blocks", ())
        }
        for transfers_name in ("_store_event_to_blocks", "_abandoned_store_event_to_blocks"):
            for transfer in getattr(self.offload, transfers_name, {}).values():
                store_pins.update(int(block) for block in transfer.gpu_block_ids)
        for transfer in getattr(self.offload, "_pending_finished_stores", ()):
            store_pins.update(int(block) for block in transfer.gpu_block_ids)
        if not load_pins and not store_pins:
            self._simple_store_owners = {}
            return snap
        normalized = []
        reservation_blocks: dict[tuple[str, str], list[int]] = {}
        mapped = {
            (request.pool, int(block))
            for request in snap.requests
            for block in request.blocks
            if block is not None
        }
        pinned_ids = tuple(sorted(store_pins | load_pins.keys()))
        routed = self._offload_pool_blocks(pinned_ids)
        store_keys = {
            (pool, block_id)
            for pool, pool_blocks in self._offload_pool_blocks(tuple(sorted(store_pins))).items()
            for block_id in pool_blocks
        }
        remembered_owners = getattr(self, "_simple_store_owners", {})
        self._simple_store_owners = {
            key: owner for key, owner in remembered_owners.items() if key in store_keys
        }

        def reserve_store_source(pool: str, block_id: int) -> None:
            owner = self._simple_store_owners.get((pool, block_id))
            if owner is None:
                block_method = getattr(self.core_ledger, "block", None)
                shadow = None if block_method is None else block_method(pool, block_id)
                owner = getattr(shadow, "owner", None)
                if owner is not None:
                    self.stats["simple_offload_shadow_owner_blocks"] += 1
            if owner is None:
                owner = f"__kvguard_simple_store__:{pool}:{block_id}"
            reservation_blocks.setdefault((owner, pool), []).append(block_id)

        offload_pools = set(self.pool_of_group.values())
        offload_pools.update(self._shared_logical_pools(POOL))
        seen: set[tuple[str, int]] = set()
        adjusted = 0
        for block in snap.blocks:
            seen.add((block.pool, block.block))
            if block.pool not in offload_pools:
                normalized.append(block)
                continue
            pins = load_pins[block.block] + (block.block in store_pins)
            if not pins:
                normalized.append(block)
                continue
            refs = None if block.refs is None else max(0, block.refs - pins)
            if block.block in store_pins:
                # A completed request can leave the connector's DMA pin as the only physical
                # reference. Keep that pin visible through a sparse copy of the producer's last
                # row: it occupies the allocator block and authorizes the source read without
                # changing the pinned block's logical positions.
                copied_cache = (block.pool, block.block) in getattr(
                    self, "_cow_cache_blocks", set()
                ) and self.pool.blocks[block.block].block_hash is not None
                if refs == 0:
                    refs = 1
                    if (block.pool, block.block) not in mapped:
                        reserve_store_source(block.pool, block.block)
                geometry = self.core_ledger.pools[block.pool]
                # ``ref_cnt > 1`` normally means two request mappings, but here one reference is
                # the connector pin. A rewritable state block with one real mapper stays private;
                # one with two real mappers is a shared immutable checkpoint, kept published so
                # both requests may read it while a write stays illegal.
                state = (
                    BlockState.PUBLISHED
                    if copied_cache
                    else (
                        BlockState.PRIVATE
                        if geometry.rewritable and (refs is None or refs <= 1)
                        else BlockState.PUBLISHED
                    )
                )
                block = replace(block, state=state, reserved=False, refs=refs)
            else:
                block = replace(block, refs=refs)
            normalized.append(block)
            adjusted += 1
        for pool, pool_blocks in routed.items():
            for block_id in pool_blocks:
                if (pool, block_id) in seen:
                    continue
                obj = self.pool.blocks[block_id]
                pins = load_pins[block_id] + (block_id in store_pins)
                refs = max(0, int(obj.ref_cnt) - pins)
                if block_id in store_pins:
                    if refs == 0:
                        refs = 1
                        if (pool, block_id) not in mapped:
                            reserve_store_source(pool, block_id)
                    geometry = self.core_ledger.pools[pool]
                    normalized.append(
                        BlockSnapshot(
                            pool,
                            block_id,
                            (
                                BlockState.PRIVATE
                                if geometry.rewritable and refs <= 1
                                else BlockState.PUBLISHED
                            ),
                            reserved=False,
                            refs=refs,
                        )
                    )
                elif obj.ref_cnt > 0:
                    normalized.append(BlockSnapshot(pool, block_id, BlockState.PRIVATE, refs=refs))
                adjusted += 1
        self.stats["simple_offload_reserved_blocks"] += adjusted
        reservations = []
        for (owner, pool), blocks in reservation_blocks.items():
            prior = getattr(self.core_ledger, "requests", {}).get((owner, pool))
            source = None if prior is None else prior.snapshot
            remaining = Counter(blocks)
            sparse = []
            if source is not None:
                for block_id in source.blocks:
                    if block_id is not None and remaining[block_id]:
                        sparse.append(block_id)
                        remaining[block_id] -= 1
                    else:
                        sparse.append(None)
            if source is not None and not +remaining:
                # Preserve the pinned source blocks at their original logical positions. A
                # zero-token row with the producer's id would look like a speculative rollback
                # and would incorrectly invalidate the slots the DMA is still reading.
                reservation = replace(source, blocks=tuple(sparse))
                geometry = self.core_ledger.pools[pool]
                if not geometry.rewritable:
                    # Eager finish-time stores may retain only the immutable suffix block after
                    # the request's preceding blocks have already been released. Keep the sparse
                    # physical positions, but make those released prefix/suffix holes dead. State
                    # pools deliberately keep their column holes under the rewritable rule.
                    present = [index for index, block in enumerate(sparse) if block is not None]
                    first = max(source.first_token, present[0] * geometry.block_size)
                    last = min(source.num_tokens, (present[-1] + 1) * geometry.block_size)
                    reservation = replace(
                        reservation,
                        num_tokens=last,
                        num_computed=min(source.num_computed, last),
                        first_token=first,
                    )
                reservations.append(reservation)
                self.stats["simple_offload_source_rows_retained"] += 1
            else:
                block_size = self.core_ledger.pools[pool].block_size
                reservations.append(
                    RequestSnapshot(
                        owner,
                        pool,
                        tuple(blocks),
                        len(blocks) * block_size,
                        0,
                    )
                )
                self.stats["simple_offload_synthetic_rows"] += 1
        requests = list(snap.requests)
        request_index = {(row.request, row.pool): idx for idx, row in enumerate(requests)}
        for reservation in reservations:
            key = (reservation.request, reservation.pool)
            idx = request_index.get(key)
            if idx is None:
                request_index[key] = len(requests)
                requests.append(reservation)
                continue
            current = requests[idx]
            geometry = self.core_ledger.pools[reservation.pool]
            width = max(len(current.blocks), len(reservation.blocks))
            current_blocks = current.blocks + (None,) * (width - len(current.blocks))
            reserved_blocks = reservation.blocks + (None,) * (width - len(reservation.blocks))
            if not geometry.rewritable or any(
                left is not None and right is not None and left != right
                for left, right in zip(current_blocks, reserved_blocks, strict=True)
            ):
                self._reject(
                    "SimpleCPU's retained store source overlaps a live request row: "
                    f"{key!r}, live={current.blocks!r}, retained={reservation.blocks!r}"
                )
            requests[idx] = replace(
                current,
                blocks=tuple(
                    left if left is not None else right
                    for left, right in zip(current_blocks, reserved_blocks, strict=True)
                ),
                num_tokens=max(current.num_tokens, reservation.num_tokens),
                num_computed=max(current.num_computed, reservation.num_computed),
                first_token=min(current.first_token, reservation.first_token),
            )
            self.stats["simple_offload_source_rows_merged"] += 1
        return replace(snap, requests=tuple(requests), blocks=tuple(normalized))

    def _normalize_publications(self, snap: Snapshot) -> Snapshot:
        """Keep a hashed generation immutable once publication has been observed.

        Once publication is observed, a later reference-count drop cannot make that generation
        writable again (including a recurrent checkpoint shared by online-beam children and later
        left with one mapper); a completed external load establishes the same fact. EAGLE's target
        and draft pools remember generations separately even though one allocation replaces both.
        """
        draft_mapped: Counter[tuple[str, int]] = Counter()
        draft_complete: set[tuple[str, int]] = set()
        for row in snap.requests:
            if row.pool != DRAFT_POOL:
                continue
            block_size = self.core_ledger.pools[row.pool].block_size
            for index, block_id in enumerate(row.blocks):
                if block_id is None:
                    continue
                key = (row.pool, int(block_id))
                draft_mapped[key] += 1
                if row.num_computed >= (index + 1) * block_size:
                    draft_complete.add(key)
        # Incremental: a listed block whose snapshot object the pool's wrapped paths left
        # untouched (the snapshot cache keeps the object) and whose external facts (the offload
        # fills, the Store pins) did not change keeps its normalized object, avoiding a
        # per-block partial-hash lookup over the whole prefix cache.
        cache = getattr(self, "_normalized_cache", None)
        if cache is None:
            cache = self._normalized_cache = {}
            self._normalized_external = (set(), set(), set())
        offload_pools = getattr(self, "_offload_filled_pools", set())
        offload_filled = getattr(self, "_offload_filled", set())
        mooncake_pins = getattr(self, "_mooncake_pin_private", set())
        changed_keys: set[tuple[str, int]] = set()
        prev = self._normalized_external
        changed_keys |= offload_pools ^ prev[0]
        changed_keys |= {(POOL, b) for b in offload_filled ^ prev[1]}
        changed_keys |= mooncake_pins ^ prev[2]
        self._normalized_external = (set(offload_pools), set(offload_filled), set(mooncake_pins))
        published = self._published_blocks
        normalized = []
        listed: set[tuple[str, int]] = set()
        pool_blocks = self.pool.blocks
        delta = bool(snap.delta_pools)
        if delta:
            # A kept block (listed before, unchanged in the cache) whose external facts
            # changed is re-normalized from its cached source and joins the change.
            kept = [
                (key, cache[key][0])
                for key in changed_keys
                if key in cache and key[0] in snap.delta_pools
            ]
            entries = [(block, False) for block in snap.blocks] + [
                (block, True) for _, block in kept
            ]
        else:
            entries = [(block, False) for block in snap.blocks]
        for block, kept_block in entries:
            key = (block.pool, block.block)
            if not kept_block:
                listed.add(key)
            cached = cache.get(key)
            obj = pool_blocks[block.block]
            if (
                not kept_block
                and cached is not None
                and cached[0] is block
                and cached[2] is obj.block_hash
                and cached[3] == getattr(obj, "block_hash_num_tokens", None)
                and block.pool != DRAFT_POOL
                and key not in changed_keys
            ):
                normalized.append(cached[1])
                continue
            partial_attention = self._partial_attention_hash(block.pool, block.block)
            known_complete = not partial_attention and (
                key in self._published_blocks
                or key in getattr(self, "_offload_filled_pools", set())
                or (block.pool == POOL and block.block in self._offload_filled)
            )
            # Mooncake Store pins every allocated request block while a PUT can retry, including
            # an EAGLE lookahead tail that the PUT does not transfer.  Its extra allocator
            # reference must neither establish nor resurrect publication for that partial
            # logical block.
            known_complete = known_complete and key not in getattr(
                self, "_mooncake_pin_private", set()
            )
            if (
                block.pool == DRAFT_POOL
                and block.refs is not None
                and block.refs <= 1
                and draft_mapped[key] == 1
                and key not in draft_complete
            ):
                # EAGLE target and draft tensors share the allocation and target block hash,
                # but their initialization frontiers are independent. A remembered target
                # publication cannot make the sole draft owner's partial lookahead tail
                # immutable after a transient Store/CoW reference disappears.
                known_complete = False
                block = replace(block, state=BlockState.PRIVATE, reserved=False)
            source = block
            if block.state is BlockState.PRIVATE and known_complete and obj.block_hash is not None:
                block = replace(block, state=BlockState.PUBLISHED)
            cache[key] = (
                source,
                block,
                obj.block_hash,
                getattr(obj, "block_hash_num_tokens", None),
            )
            if block.state is BlockState.PUBLISHED and not partial_attention:
                published.add(key)
            else:
                published.discard(key)
            normalized.append(block)
        if delta:
            for pool, ids in (snap.gone or {}).items():
                for b in ids:
                    cache.pop((pool, b), None)
                    published.discard((pool, b))
        else:
            for key in cache.keys() - listed:
                del cache[key]
                published.discard(key)
        return replace(snap, blocks=tuple(normalized))

    def _partial_attention_hash(self, pool: str, block_id: int) -> bool:
        """Whether a full-attention cache entry ends inside its physical block.

        Such an entry can return from cached/shared to private when its exclusive request resumes
        and fills its tail, so publication is not sticky for that generation.
        """
        gid = getattr(self, "group_of", {}).get(block_id)
        pool_of_group = getattr(self, "pool_of_group", {gid: POOL})
        if gid is None or pool_of_group.get(gid) != pool:
            return False
        group = next(
            (candidate for candidate in getattr(self, "groups", ()) if candidate["group"] == gid),
            None,
        )
        if group is not None and group["kind"] != "attention":
            return False
        obj = self.pool.blocks[block_id]
        boundary = getattr(obj, "block_hash_num_tokens", None)
        if boundary is None:
            return False
        block_size = int(self.managers[gid].block_size)
        return int(boundary) % block_size != 0

    def _with_eagle_snapshot(self, snap: Snapshot) -> Snapshot:
        """Mirror the shared block allocation into the draft layers' logical KV pool.

        EAGLE3, DFlash and standalone AR share physical block ids and request tables between
        target and draft layers but initialize their contents by different forwards. Extending
        the draft mapping by the lookahead admits the positions the proposer fills later this
        execution without crediting those fills to the target layers.
        """
        draft_requests = []
        live_dflash: set[str] = set()
        for request in snap.requests:
            if request.pool != POOL:
                continue
            capacity = len(request.blocks) * self.block_size
            computed = request.num_computed
            if (str(request.request), POOL) in getattr(self, "_deferred_rows", {}):
                # A finished request's draft writes land with its target's, in the same
                # worker-side preparation; the draft claim follows what its own pool credited.
                draft_shadow = self.core_ledger.requests.get((request.request, DRAFT_POOL))
                if draft_shadow is not None:
                    computed = min(computed, max(int(draft_shadow.computed_high_water), 0))
            if getattr(self, "dflash", False) or getattr(self, "standalone_draft", False):
                rid = str(request.request)
                live_dflash.add(rid)
                frontier = self._dflash_frontiers.setdefault(rid, computed)
                if computed > frontier:
                    prefix = "standalone" if getattr(self, "standalone_draft", False) else "dflash"
                    self.stats[f"{prefix}_snapshot_caps"] += 1
                    self.stats[f"{prefix}_snapshot_capped_tokens"] += computed - frontier
                    computed = frontier
            draft_requests.append(
                replace(
                    request,
                    pool=DRAFT_POOL,
                    num_tokens=min(capacity, request.num_tokens + self.eagle_lookahead),
                    num_computed=computed,
                )
            )
        if getattr(self, "dflash", False) or getattr(self, "standalone_draft", False):
            for rid in self._dflash_frontiers.keys() - live_dflash:
                del self._dflash_frontiers[rid]
                self._dflash_pending_writes.pop(rid, None)
                getattr(self, "_standalone_scratch", {}).pop(rid, None)
        draft_blocks = tuple(
            replace(block, pool=DRAFT_POOL) for block in snap.blocks if block.pool == POOL
        )
        free = snap.free
        if free is not None and POOL in free:
            free = dict(free)
            free[DRAFT_POOL] = free[POOL]
        free_count = snap.free_count
        if free_count is not None and POOL in free_count:
            free_count = dict(free_count)
            free_count[DRAFT_POOL] = free_count[POOL]
        return replace(
            snap,
            requests=snap.requests + tuple(draft_requests),
            blocks=snap.blocks + draft_blocks,
            free=free,
            free_count=free_count,
        )

    def _with_dflash_separate_snapshot(self, snap: Snapshot) -> Snapshot:
        """Expose an independent DFlash draft group through its concrete write frontier."""
        pool = self.dflash_pool
        geometry = self.core_ledger.pools[pool]
        window = self.window_of[self.dflash_gid]
        extra = self.extra_of[self.dflash_gid]
        requests = []
        live: set[str] = set()
        for row in snap.requests:
            if row.pool != pool:
                requests.append(row)
                continue
            rid = str(row.request)
            live.add(rid)
            frontier = self._dflash_frontiers.setdefault(rid, row.num_computed)
            computed = min(row.num_computed, frontier)
            if row.num_computed > computed:
                self.stats["dflash_snapshot_caps"] += 1
                self.stats["dflash_snapshot_capped_tokens"] += row.num_computed - computed
            capacity = len(row.blocks) * geometry.block_size
            requests.append(
                replace(
                    row,
                    num_tokens=min(capacity, row.num_tokens + self.eagle_lookahead),
                    num_computed=computed,
                    first_token=(
                        row.first_token if window is None else window_start(computed, window, extra)
                    ),
                )
            )
        for rid in self._dflash_frontiers.keys() - live:
            del self._dflash_frontiers[rid]
            self._dflash_pending_writes.pop(rid, None)
        return replace(snap, requests=tuple(requests))

    def _with_mtp_frontiers(self, snap: Snapshot) -> Snapshot:
        """Cap draft claims at each module's last concrete write frontier.

        Target and MTP groups share one scheduler count, which under the async loop can still
        include rejected verification tokens when the snapshot is built, so the speculator's
        module-specific positions are the authoritative draft frontier.
        """
        if self.mtp_shared:
            requests = []
            blocks = []
            free = None if snap.free is None else dict(snap.free)
            free_count = None if snap.free_count is None else dict(snap.free_count)
            for pool in self.mtp_pools:
                geometry = self.core_ledger.pools[pool]
                requests.extend(
                    replace(
                        row,
                        pool=pool,
                        num_tokens=min(
                            len(row.blocks) * geometry.block_size,
                            row.num_tokens + self.eagle_lookahead,
                        ),
                    )
                    for row in snap.requests
                    if row.pool == POOL
                )
                blocks.extend(
                    replace(block, pool=pool) for block in snap.blocks if block.pool == POOL
                )
                if free is not None and POOL in free:
                    free[pool] = free[POOL]
                if free_count is not None and POOL in free_count:
                    free_count[pool] = free_count[POOL]
            snap = replace(
                snap,
                requests=snap.requests + tuple(requests),
                blocks=snap.blocks + tuple(blocks),
                free=free,
                free_count=free_count,
            )

        gid_of_pool = dict(zip(self.mtp_pools, self.mtp_gids, strict=True))
        draft_pools = set(gid_of_pool)
        requests = []
        capped = 0
        tokens = 0
        for row in snap.requests:
            if row.pool not in draft_pools:
                requests.append(row)
                continue
            frontier = self._mtp_frontiers.get((row.pool, str(row.request)))
            if frontier is None or row.num_computed <= frontier:
                requests.append(row)
                continue
            tokens += row.num_computed - frontier
            capped += 1
            gid = gid_of_pool[row.pool]
            window = self.window_of[gid]
            requests.append(
                replace(
                    row,
                    num_computed=frontier,
                    first_token=(
                        0 if window is None else window_start(frontier, window, self.extra_of[gid])
                    ),
                )
            )
        self.stats["mtp_snapshot_caps"] += capped
        self.stats["mtp_snapshot_capped_tokens"] += tokens
        return replace(snap, requests=tuple(requests))

    def _defer_request_row(self, request, blocks) -> None:
        if not blocks:
            return
        rid = str(request.request_id)
        by_pool: dict[str, list[int]] = {}
        for block in blocks:
            if block.is_null:
                continue
            block_id = int(block.block_id)
            if self.hybrid:
                gid = self.group_of.get(block_id)
                if gid is None:
                    self._reject(f"deferred block {block_id} has no recorded KV cache group")
                pool = self.pool_of_group[gid]
            else:
                pool = POOL
            by_pool.setdefault(pool, []).append(block_id)
        for pool, pool_ids in by_pool.items():
            key = (rid, pool)
            if key in self._deferred_rows:
                self.reporter.emit(
                    integration_error(
                        "deferred_row_reused",
                        f"request {rid!r} acquired a second deferred row in pool {pool!r}",
                        pool=pool,
                        request=rid,
                        site=f"defer@{self.seq}",
                    )
                )
                continue
            ids = tuple(pool_ids)
            shadow = self.core_ledger.requests.get(key)
            if shadow is not None and Counter(
                block for block in shadow.snapshot.blocks if block is not None
            ) == Counter(ids):
                # The ledger's row as the last snapshot read it: its computed count is raised to
                # the shadow's high water at each snapshot once that step's writes land. Kept at
                # the pre-step count, the draft mirror of the row would read as a rollback and
                # invalidate the step's draft writes.
                row = shadow.snapshot
            else:
                geometry = self.core_ledger.pools[pool]
                capacity = len(ids) * geometry.block_size
                computed = 0
                if not geometry.rewritable:
                    computed = self._snapshot_computed(rid, int(request.num_computed_tokens), 0)
                    if shadow is not None:
                        computed = max(computed, int(shadow.computed_high_water))
                row = RequestSnapshot(rid, pool, ids, capacity, min(computed, capacity))
            self._deferred_rows[key] = row
            counts = self._deferred_counts[key] = Counter(
                block for block in row.blocks if block is not None
            )
            for block_id in counts:
                self._deferred_by_block.setdefault(block_id, set()).add(key)
            self.stats["deferred_rows"] += 1
            self.stats["deferred_blocks"] += sum(counts.values())

    def _deferred_rows_for_snapshot(self) -> tuple[RequestSnapshot, ...]:
        """The deferred rows with their computed claim raised to the shadow's high water.

        A finished request's last step may still be executing when its row is deferred, so the
        claim follows the high water of writes the ledger has actually credited.
        """
        rows = []
        for key, row in self._deferred_rows.items():
            shadow = self.core_ledger.requests.get(key)
            if shadow is not None and not self.core_ledger.pools[key[1]].rewritable:
                claim = min(max(row.num_computed, int(shadow.computed_high_water)), row.num_tokens)
                if claim != row.num_computed:
                    row = replace(row, num_computed=claim)
                    self._deferred_rows[key] = row
                    self.stats["deferred_rows_claim_raised"] += 1
            rows.append(row)
        return tuple(rows)

    def _release_deferred_rows(self, blocks) -> None:
        completed: set[tuple[str, str]] = set()
        for block in blocks:
            block_id = int(block.block_id)
            for key in self._deferred_by_block.pop(block_id, ()):
                counts = self._deferred_counts[key]
                counts.pop(block_id, None)
                if not counts:
                    completed.add(key)
        for key in completed:
            row = self._deferred_rows[key]
            if getattr(self, "offload_kind", None) == "simple":
                for block_id in row.blocks:
                    if block_id is not None:
                        self._simple_store_owners[(row.pool, int(block_id))] = row.request
            del self._deferred_counts[key]
            del self._deferred_rows[key]
            self.stats["deferred_rows_released"] += 1

    def _delta_listing(self) -> bool:
        """Whether the ordinary steps may list the pools as their change: the plain attention
        attachment (one cache group, no speculative, offload, Store or copy-on-write
        normalization of the listing, no discovery trace), where every pass after the cache
        keeps an unchanged block's object. The other compositions list in full."""
        return (
            self.delta_listing
            and not self.hybrid
            and not self.speculative
            and not self.dflash
            and not self.multi_module_mtp
            and not self.standalone_draft
            and getattr(self, "offload_kind", None) is None
            and getattr(self, "mooncake_kind", None) is None
            and not self._cow_pins
            and not self._offload_recompute_tails
            and self.offload_events is None
        )

    def _snapshot_incremental(self, scheduled, audit: bool, stream: int | None) -> Snapshot:
        """The default: the caches' request tuples and block snapshots, refreshed for what the
        pool's mediated paths and the requests' progress touched; an audited step re-reads the
        listed blocks (drift check) and walks the free queue for the partition audit by set,
        an ordinary step passes the allocator's free count for the audit by count. Each KV
        cache group's cache serves its core pool; the pools share the allocator, so the free
        set and the count are given once and the core audits the union."""
        requests = []
        live = list(self.scheduler.requests.items())
        for gid, cache in self.caches.items():
            req_to_blocks = self.managers[gid].req_to_blocks
            entries = []
            for rid, req in live:
                blocks = req_to_blocks.get(rid)
                if blocks:
                    count = int(scheduled.get(rid, 0))
                    computed = self._snapshot_computed(rid, req.num_computed_tokens, count)
                    entries.append((rid, blocks, computed, count))
            mutated = self._mutated.pop(gid, None) if cache.rebuild else None
            requests.extend(cache.requests_step(entries, mutated if mutated is not None else set()))
        pool = self.pool
        block_of = pool.blocks.__getitem__
        caches = list(self.caches.values())
        if not (audit or any(not c.valid for c in caches)):
            for c in caches:
                c.blocks_step(block_of)
            # One count for the shared allocator, given under the attention pool.
            free_count = {POOL: int(pool.get_num_free_blocks())}
            if self._delta_listing():
                # The pools listed as their change since the last snapshot: the caches'
                # new objects and dropped blocks (`Snapshot.delta_pools`); the audited steps
                # list in full and re-check the ledger's kept listing.
                changed: list[BlockSnapshot] = []
                gone: dict[str, frozenset[int]] = {}
                for c in caches:
                    objects, dropped = c.delta()
                    changed.extend(objects)
                    if dropped:
                        gone[c.pool] = dropped
                self.stats["snapshot_delta"] += 1
                self.stats["snapshot_delta_blocks"] += len(changed)
                return Snapshot(
                    self.seq,
                    tuple(requests),
                    tuple(changed),
                    None,
                    free_count=free_count,
                    stream=stream,
                    delta_pools=frozenset(c.pool for c in caches),
                    gone=gone or None,
                )
            blocks = tuple(b for c in caches for b in c.blocks())
            return Snapshot(
                self.seq, tuple(requests), blocks, None, free_count=free_count, stream=stream
            )
        ta = time.perf_counter()
        if any(not c.valid for c in caches):
            objects = pool.blocks
        else:
            listed: set[int] = set()
            for c in caches:
                listed.update(c.listed)
            objects = [block_of(b) for b in listed]
        for c in caches:
            drift, example = c.resync(block_of, objects)
            if drift:
                self.reporter.emit(
                    integration_error(
                        "snapshot_cache_drift",
                        f"the full read contradicted {drift} cached block snapshot(s), e.g. {example}",
                        pool=c.pool,
                        site=f"snapshot:{self.seq}",
                    )
                )
        queue = pool.free_block_queue.get_all_free_blocks()
        free = frozenset(b.block_id for b in queue if not b.is_null and b.block_hash is None)
        if self.hybrid:
            self._audit_groups(block_of, f"snapshot:{self.seq}")
        blocks = tuple(b for c in caches for b in c.blocks())
        self.stats["audited_snapshots"] += 1
        self.timings["snapshot_audit_read"] += time.perf_counter() - ta
        # The free set once, under the attention pool: the pools share the allocator.
        return Snapshot(self.seq, tuple(requests), blocks, {POOL: free}, stream=stream)

    def _audit_groups(self, block_of, site: str) -> None:
        """The recorded groups against the pool's own facts: a recorded block must be non-free,
        and a hashed block's hash carries its group id (``make_block_hash_with_group_id``: the
        last four bytes, big-endian)."""
        self.stats["group_audits"] += 1
        bad = 0
        example = None
        for b, g in self.group_of.items():
            obj = block_of(b)
            h = obj.block_hash
            if obj.ref_cnt <= 0 and h is None:
                bad += 1
                example = example or f"block {b} recorded in group {g} is free"
            elif isinstance(h, (bytes, bytearray)) and len(h) >= 4:
                hg = int.from_bytes(h[-4:], "big", signed=False)
                if hg != g:
                    bad += 1
                    example = (
                        example or f"block {b} recorded in group {g} carries group {hg}'s hash"
                    )
        if bad:
            self.stats["group_drift"] += bad
            self.reporter.emit(
                integration_error(
                    "group_drift",
                    f"{bad} block(s) recorded in a group the pool contradicts, e.g. {example}",
                    pool=POOL,
                    site=site,
                )
            )

    def _snapshot_full(self, scheduled, audit: bool, stream: int | None) -> Snapshot:
        """The complete read every step (``snapshot="full"``): every live request's block list
        and the facts of every block it or the prefix cache holds, the whole pool and the free
        queue at an audited step. One KV cache group."""
        requests = []
        for rid, req in self.scheduler.requests.items():
            blocks = self.manager.req_to_blocks.get(rid)
            if not blocks:
                continue
            count = int(scheduled.get(rid, 0))
            computed = self._snapshot_computed(rid, req.num_computed_tokens, count)
            requests.append(
                RequestFacts(
                    rid,
                    tuple(None if b.is_null else b.block_id for b in blocks),
                    computed,
                    count,
                )
            )
        if audit:
            ta = time.perf_counter()
            blocks = [
                BlockFacts(b.block_id, b.ref_cnt, b.block_hash is not None, b.is_null)
                for b in self.pool.blocks
            ]
            queue = self.pool.free_block_queue.get_all_free_blocks()
            free_ids = [b.block_id for b in queue if not b.is_null]
            free_hashed = [b.block_id for b in queue if b.block_hash is not None]
            self.stats["audited_snapshots"] += 1
            self.timings["snapshot_audit_read"] += time.perf_counter() - ta
        else:
            seen: dict[int, Any] = {}
            for rid in self.scheduler.requests:
                for b in self.manager.req_to_blocks.get(rid) or ():
                    if not b.is_null:
                        seen[b.block_id] = b
            # Cached (hashed) blocks waiting in the free queue are reserved, not free; the
            # pool's hash map lists them without walking the queue.
            for entry in self.pool.cached_block_hash_to_block._cache.values():
                for b in entry.values() if isinstance(entry, dict) else (entry,):
                    seen[b.block_id] = b
            blocks = [
                BlockFacts(b.block_id, b.ref_cnt, b.block_hash is not None, b.is_null)
                for b in seen.values()
            ]
            free_ids, free_hashed = None, ()
        snap = build_snapshot(
            self.seq,
            POOL,
            self.block_size,
            requests,
            blocks,
            free_queue_ids=free_ids,
            free_queue_hashed=free_hashed,
        )
        return replace(snap, stream=stream) if stream is not None else snap

    def _dispatch(self, input_batch, block_tables, slot_mappings) -> None:
        t0 = time.perf_counter()
        n = int(input_batch.num_reqs)
        if n == 0:
            return
        n_pad = int(input_batch.num_reqs_after_padding)
        total = int(input_batch.num_tokens)
        total_pad = int(getattr(input_batch, "num_tokens_after_padding", total))
        req_ids = list(input_batch.req_ids[:n])
        scheduled, adaptive = _input_execution_counts(input_batch, n)
        if adaptive:
            upper = np.asarray(input_batch.num_scheduled_tokens[:n], dtype=np.int64)
            trimmed = upper - scheduled
            self.stats["adaptive_verification_batches"] += 1
            self.stats["adaptive_verification_tokens"] += int(scheduled.sum())
            if np.any(trimmed):
                self.stats["adaptive_verification_trimmed_batches"] += 1
                self.stats["adaptive_verification_trimmed_tokens"] += int(trimmed.sum())
        computed, used_device_fallback = _input_pre_step_computed(
            input_batch, n, scheduled, force_device=self.speculative
        )
        if used_device_fallback:
            self.stats["pre_step_from_seq_lens"] += n
        if self.speculative:
            # With async speculative decoding the CPU mirror is intentionally optimistic while
            # prepare_inputs forms positions and slot mappings from the corrected GPU counter.
            # Reconcile that authoritative execution-time retreat before checking the write.
            pools = {
                self.pool_of_group[g["group"]]
                for g in self.groups
                if g["kind"] in ("attention", "window")
            }
            if self._shared_draft_pool():
                pools.add(DRAFT_POOL)
            pools.update(self.mtp_pools)
            for pool in pools:
                for rid, frontier in zip(req_ids, computed, strict=True):
                    retained = self._speculative_retained.pop((pool, str(rid)), None)
                    if retained is not None and retained > int(frontier):
                        shadow = self.ledger.requests.get((rid, pool))
                        stop = (
                            retained
                            if shadow is None
                            else min(retained, int(shadow.snapshot.num_tokens))
                        )
                        self.ledger.invalidate_positions(
                            pool,
                            rid,
                            range(int(frontier), stop),
                            site=f"speculative-retained-rollback@{self.seq}",
                        )
                    self.ledger.rollback(
                        pool,
                        rid,
                        int(frontier),
                        site=f"speculative-rollback@{self.seq}",
                    )
                    if self.dflash and pool == self.dflash_pool:
                        draft_rid = str(rid)
                        self._dflash_frontiers[draft_rid] = int(frontier)
                        self._dflash_pending_writes.pop(draft_rid, None)
                    if self.multi_module_mtp and pool in self.mtp_pools:
                        self._mtp_frontiers[(pool, str(rid))] = int(frontier)
        kv = self.kv_gid
        null = self.pool.null_block.block_id
        if self.tables == "mapping":
            # The mapping as the step's snapshot read it (the cache's tuples; the schedule
            # allocated this step's blocks before the snapshot).
            ids = self.caches[kv].ids
            block_lists = [ids(rid) for rid in req_ids]
            table_np, slots_np = mapping_tables(
                block_lists, computed, scheduled, self.block_size, n_pad, null
            )
        else:
            table_np = block_tables[kv][:n_pad].cpu().numpy()
            slots_np = slot_mappings[kv][:total_pad].cpu().numpy().astype(np.int64)
        write, attn = build_dispatches(
            self.seq,
            POOL,
            self.block_size,
            req_ids,
            computed,
            scheduled,
            table_np,
            n_pad,
            slots_np,
            window=self.window_of.get(kv),
        )
        stream = current_stream_handle() if self.stream_epochs else None
        if stream is not None:
            write, attn = replace(write, stream=stream), replace(attn, stream=stream)
        if self.tags is not None:
            self.tags.dispatch(kv, block_tables[kv], slot_mappings[kv], write, attn, req_ids)
        if self.validators is None:
            for d in (write, attn):
                self.ledger.dispatch(d)
                self._opened(d.dispatch_id)
        else:
            self._dispatch_gpu(
                input_batch, block_tables[kv], slot_mappings[kv], write, attn, self.mirror
            )
        if self.worker_table_check and input_batch is not None:
            self._check_stale_worker_blocks(kv, req_ids, input_batch, null)
        step_key = self._current_step_key()
        if self.speculative and step_key is not None:
            self._record_speculative_step(step_key, req_ids, computed, scheduled)
        self.stats["dispatches"] += 2
        self.stats["dispatch_tokens"] += total
        self.stats["padding_rows"] += n_pad - n
        self.stats["padding_tokens"] += total_pad - total
        for g in self.aux_attention_gids:
            self._dispatch_window(
                g,
                req_ids,
                computed,
                scheduled,
                block_tables,
                slot_mappings,
                n_pad,
                total_pad,
                null,
                stream,
                input_batch,
            )
        if self.state_gids:
            self._dispatch_state(req_ids, computed, scheduled, block_tables, n_pad, null, stream)
        tails = getattr(self, "_offload_recompute_tails", None)
        if tails:
            for rid, frontier, count in zip(req_ids, computed, scheduled, strict=True):
                tail = tails.get(str(rid))
                if tail is None:
                    continue
                position = tail[0]
                if int(frontier) <= position < int(frontier) + int(count):
                    tails.pop(str(rid), None)
                    self.stats["offload_full_hit_tail_recomputes_dispatched"] += 1
        if self.nvtx is not None:
            self.nvtx.step()
        self.timings["dispatch_s"] += time.perf_counter() - t0

    def _begin_eagle_target(self, scheduler_output: Any) -> None:
        """Remember the proposal width before the target job prepares its descriptors."""
        self._eagle_prepare_lookahead = (
            0
            if bool(getattr(scheduler_output, "skip_drafts", False))
            else int(getattr(scheduler_output, "num_spec_tokens_to_schedule", 0))
        )

    def _prepare_eagle_descriptors(self, input_batch: Any) -> None:
        """Publish the target job's draft accesses before any of its kernels run."""
        lookahead = self._eagle_prepare_lookahead
        if lookahead is None:
            self._reject("EAGLE3 attention preparation outside its target execution")
        # K=0 suppresses token proposals, not the draft prefill.  The proposer
        # still executes that first forward with ``skip_sample=True`` so its KV
        # remains aligned with the target for a later step.
        self._dispatch_eagle(input_batch, lookahead)
        self._eagle_prepared_batches[id(input_batch)] = lookahead

    def _match_eagle_proposal(self, input_batch: Any, lookahead: int) -> None:
        """Require the later proposer callback to match the descriptors already published."""
        prepared = self._eagle_prepared_batches.pop(id(input_batch), None)
        if prepared is None or prepared != lookahead:
            self._reject(
                f"EAGLE3 proposer K={lookahead} without its matching prepared descriptor "
                f"(prepared={prepared})"
            )

    def _dispatch_eagle(self, input_batch: Any, lookahead: int) -> None:
        """Validate EAGLE3's draft prefill and its one-token autoregressive forwards.

        The runner reuses the target batch shape for draft prefill, then advances every live row
        by one position per remaining proposal; derive the slots from the scheduler-owned request
        lists without synchronizing mutable CUDA-graph metadata back to the host.
        """
        n = int(input_batch.num_reqs)
        if n == 0:
            return
        req_ids = list(input_batch.req_ids[:n])
        scheduled, _ = _input_execution_counts(input_batch, n)
        computed, used_device_fallback = _input_pre_step_computed(
            input_batch, n, scheduled, force_device=True
        )
        if used_device_fallback:
            self.stats["eagle_pre_step_from_seq_lens"] += n
        block_lists = [self.cache.ids(rid) for rid in req_ids]
        if self.eagle_fault == "draft_early_free" and not self.eagle_fault_done:
            victim = next(
                (
                    int(blocks[(int(frontier) + int(count) - 1) // self.block_size])
                    for blocks, frontier, count in zip(
                        block_lists, computed, scheduled, strict=True
                    )
                    if int(count) > 0
                    and (int(frontier) + int(count) - 1) // self.block_size < len(blocks)
                    and blocks[(int(frontier) + int(count) - 1) // self.block_size]
                ),
                None,
            )
            if victim is not None:
                self.ledger.free(DRAFT_POOL, (victim,), site=f"eagle-fault@{self.seq}")
                self.eagle_fault_done = True
                self.stats["eagle_fault_draft_early_free"] += 1
                self.stats["eagle_fault_block"] = victim
        stream = current_stream_handle() if self.stream_epochs else None

        def dispatch(frontiers: np.ndarray, counts: np.ndarray, draft_index: int) -> None:
            table, slots = mapping_tables(
                block_lists,
                frontiers,
                counts,
                self.block_size,
                n,
                self.pool.null_block.block_id,
            )
            self._eagle_dispatch_seq += 1
            site = f"eagle3-{draft_index}-{self._eagle_dispatch_seq}"
            write, attn = build_dispatches(
                self.seq,
                DRAFT_POOL,
                self.block_size,
                req_ids,
                frontiers,
                counts,
                table,
                n,
                slots,
                site=site,
            )
            if stream is not None:
                write = replace(write, stream=stream)
                attn = replace(attn, stream=stream)
            for descriptor in (write, attn):
                self.ledger.dispatch(descriptor)
                self._opened(descriptor.dispatch_id)
            total = int(counts.sum())
            self.stats["eagle_draft_forwards"] += 1
            self.stats["eagle_draft_tokens"] += total
            self.stats["dispatches"] += 2
            self.stats["dispatch_tokens"] += total

        # The draft prefill writes the same query positions the target just processed.
        dispatch(computed, scheduled, 0)
        frontier = computed + scheduled
        ones = np.ones(n, dtype=np.int64)
        for draft_index in range(1, max(lookahead, 1)):
            dispatch(frontier + draft_index - 1, ones, draft_index)

    def _inject_standalone_fault(self, req_ids: list[str]) -> None:
        if self.standalone_fault != "draft_early_free" or self.standalone_fault_done:
            return
        victim = next(
            (int(block) for rid in req_ids for block in reversed(self.cache.ids(rid)) if block),
            None,
        )
        if victim is None:
            return
        self.ledger.free(DRAFT_POOL, (victim,), site=f"standalone-fault@{self.seq}")
        self.standalone_fault_done = True
        self.stats["standalone_fault_draft_early_free"] += 1
        self.stats["standalone_fault_block"] = victim

    def _dispatch_standalone_draft(
        self,
        speculator: Any,
        input_ids: Any,
        positions: Any,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, Any] | None,
    ) -> None:
        """Validate one eager standalone-AR forward from its concrete draft layout.

        The first call may contain accepted target tokens, a correction token and rejected
        positions; later calls carry one token per request. The draft metadata is authoritative,
        so consume its actual positions, sequence lengths, block table and slot map.
        """
        input_batch = self._standalone_input_batch
        if input_batch is None:
            self._reject("standalone draft execution outside its proposer call")
        if attn_metadata is None or slot_mappings is None:
            self._reject("standalone draft execution without attention metadata and slot maps")
        draft_layers = self._draft_layer_names(speculator)
        missing_metadata = tuple(layer for layer in draft_layers if layer not in attn_metadata)
        missing_slots = tuple(layer for layer in draft_layers if layer not in slot_mappings)
        if not draft_layers or missing_metadata or missing_slots:
            self._reject(
                f"standalone draft layers {draft_layers!r} have missing metadata "
                f"{missing_metadata!r} or slot mappings {missing_slots!r}"
            )
        metadata = attn_metadata[draft_layers[0]]
        if type(metadata).__name__ != "FlashAttentionMetadata":
            self._reject(
                f"standalone draft metadata {type(metadata).__name__}; the bounded port uses "
                "FlashAttentionMetadata"
            )
        if any(attn_metadata[layer] is not metadata for layer in draft_layers[1:]):
            self._reject("standalone draft layers do not share one attention mapping")
        first_slots = slot_mappings[draft_layers[0]]
        if any(
            int(slot_mappings[layer].data_ptr()) != int(first_slots.data_ptr())
            for layer in draft_layers[1:]
        ):
            self._reject("standalone draft layers do not share one slot mapping")

        n = int(input_batch.num_reqs)
        if n == 0:
            return
        req_ids = [str(value) for value in input_batch.req_ids[:n]]
        qsl = metadata.query_start_loc[: n + 1].detach().cpu().numpy()
        total = int(qsl[-1])
        num_tokens = int(input_ids.shape[0])
        if int(metadata.num_actual_tokens) != total or total != num_tokens:
            self._reject(
                f"standalone draft metadata names {int(metadata.num_actual_tokens)}/{total} "
                f"tokens but the model executes {num_tokens}"
            )
        concrete_positions = positions[:total].detach().cpu().numpy()
        concrete_tokens = input_ids[:total].detach().cpu().numpy()
        seq_lens = metadata.seq_lens[:n].detach().cpu().numpy()
        table = metadata.block_table[:n].detach().cpu().numpy()
        slots = metadata.slot_mapping[:total].detach().cpu().numpy()
        if int(metadata.slot_mapping.data_ptr()) != int(first_slots.data_ptr()):
            self._reject("standalone draft metadata and model slot mappings disagree")

        self._inject_standalone_fault(req_ids)
        self._standalone_dispatch_seq += 1
        try:
            write, attention = build_positioned_dispatches(
                self.seq,
                DRAFT_POOL,
                self.block_size,
                req_ids,
                qsl,
                concrete_positions,
                seq_lens,
                table,
                slots,
                site=f"standalone-draft-{self._standalone_dispatch_seq}",
            )
        except ValueError as exc:
            self._reject(f"standalone draft layout: {exc}")
        trace = getattr(self, "standalone_events", None)
        if trace is not None:
            published = []
            shadows = self.core_ledger.requests
            for token_index, (slot, position, token) in enumerate(
                zip(slots, concrete_positions, concrete_tokens, strict=True)
            ):
                if int(slot) < 0:
                    continue
                block = int(slot) // self.block_size
                shadow = self.core_ledger.block(DRAFT_POOL, block)
                offset = int(slot) % self.block_size
                if shadow.state is not BlockState.PUBLISHED or (
                    shadow.pending_publish >> offset & 1
                ):
                    continue
                row = int(np.searchsorted(qsl[1:], token_index, side="right"))
                published.append(
                    {
                        "token_index": token_index,
                        "row": row,
                        "request": req_ids[row],
                        "position": int(position),
                        "token": int(token),
                        "slot": int(slot),
                        "block": block,
                        "generation": shadow.generation,
                        "filled": shadow.filled,
                        "holders": sorted(
                            str(request_shadow.snapshot.request)
                            for (_, pool), request_shadow in shadows.items()
                            if pool == DRAFT_POOL and block in request_shadow.snapshot.blocks
                        ),
                    }
                )
            if published:
                trace.append(
                    {
                        "seq": self.seq,
                        "dispatch": write.dispatch_id,
                        "published_writes": published,
                    }
                )
        self._standalone_scratch_rewrites(speculator, req_ids, qsl, concrete_positions, write)
        stream = current_stream_handle() if self.stream_epochs else None
        if stream is not None:
            write = replace(write, stream=stream)
            attention = replace(attention, stream=stream)
        for descriptor in (write, attention):
            self.ledger.dispatch(descriptor)
            self._opened(descriptor.dispatch_id)
        self._advance_dflash_frontiers(write.rows)
        self.stats["standalone_draft_forwards"] += 1
        self.stats["standalone_draft_tokens"] += total
        self.stats["dispatches"] += 2
        self.stats["dispatch_tokens"] += total

    def _standalone_scratch_rewrites(
        self, speculator: Any, req_ids: list[str], qsl: np.ndarray, positions: np.ndarray, write
    ) -> None:
        """Declare the provisional draft writes of this forward and re-open the last ones.

        ``StandaloneARSpeculator`` writes, in the proposal's first forward, the accepted target
        tokens, one correction token at the position after them and the rejected positions, then
        one draft token per later forward; everything from the correction slot on is beyond the
        target's committed frontier and is rewritten by the request's next proposal. The draft
        pool is rewritable, so the rewrite is legal while the block is private; when the scheduler
        has published the block in between, the slots are re-opened for exactly the request that
        wrote them (``Ledger.authorize_scratch_rewrite``).
        """
        call = self._standalone_calls
        self._standalone_calls += 1
        n = len(req_ids)
        last = None
        if call == 0:
            indices = getattr(speculator, "last_token_indices", None)
            if indices is None:
                self._reject("standalone draft speculator exposes no last_token_indices")
            last = np.asarray(indices[:n].detach().cpu().numpy(), dtype=np.int64)
        for row_index, (rid, row) in enumerate(zip(req_ids, write.rows, strict=True)):
            writes = row.writes
            if not len(writes):
                continue
            previous = self._standalone_scratch.get(rid)
            if previous is not None:
                lo, hi = max(previous[0], writes.start), min(previous[1], writes.stop)
                if lo < hi:
                    reopened = self.core_ledger.authorize_scratch_rewrite(
                        DRAFT_POOL,
                        rid,
                        range(lo, hi),
                        site=f"standalone-scratch@{self.seq}",
                    )
                    self.stats["standalone_scratch_rewrites"] += hi - lo
                    self.stats["standalone_scratch_reopened_slots"] += reopened
            if last is not None:
                begin, limit = int(qsl[row_index]), int(qsl[row_index + 1])
                correction = int(last[row_index])
                if not begin <= correction < limit:
                    self._reject(
                        f"standalone draft row {row_index} correction index {correction} is "
                        f"outside its query [{begin}, {limit})"
                    )
                scratch_from = int(positions[correction])
                if not writes.start <= scratch_from < writes.stop:
                    self._reject(
                        f"standalone draft row {row_index} correction position {scratch_from} "
                        f"is outside its writes [{writes.start}, {writes.stop})"
                    )
                self._standalone_scratch[rid] = (scratch_from, writes.stop)
                self.stats["standalone_scratch_positions"] += writes.stop - scratch_from
            else:
                current = self._standalone_scratch.get(rid)
                if current is None or writes.start < current[0]:
                    self._reject(
                        f"standalone draft forward {call} writes {writes} for request {rid!r} "
                        f"below its correction slot {current}"
                    )
                self._standalone_scratch[rid] = (current[0], max(current[1], writes.stop))
                self.stats["standalone_scratch_positions"] += len(writes)

    def _inject_dflash_fault(self, req_ids: list[str]) -> None:
        if self.dflash_fault != "draft_early_free" or self.dflash_fault_done:
            return
        cache = self.caches[self.dflash_gid]
        victim = next(
            (int(block) for rid in req_ids for block in reversed(cache.ids(rid)) if block),
            None,
        )
        if victim is None:
            return
        self.ledger.free(self.dflash_pool, (victim,), site=f"dflash-fault@{self.seq}")
        self.dflash_fault_done = True
        self.stats["dflash_fault_draft_early_free"] += 1
        self.stats["dflash_fault_block"] = victim

    def _advance_dflash_frontiers(self, rows: tuple[DispatchRow, ...]) -> None:
        """Advance only across the dense prefix established by concrete DFlash writes."""
        for row in rows:
            if row.request is None:
                continue
            rid = str(row.request)
            frontier = self._dflash_frontiers.get(rid, 0)
            pending = self._dflash_pending_writes.setdefault(rid, set())
            pending.update(int(value) for value in row.writes if int(value) >= frontier)
            while frontier in pending:
                pending.remove(frontier)
                frontier += 1
            self._dflash_frontiers[rid] = frontier
            if not pending:
                del self._dflash_pending_writes[rid]

    def _dispatch_dflash_context(self, speculator: Any, start: int, end: int) -> None:
        """Validate DFlash's target-context projection writes into its draft KV layers.

        ``prepare_dflash_inputs`` has already published the exact context positions and slots.
        Rejected suffix positions and evicted/null blocks carry the padding sentinel and are
        omitted, exactly as the framework's projection kernel omits them.
        """
        input_batch = self._dflash_input_batch
        if input_batch is None:
            self._reject("DFlash context projection outside its proposer call")
        n = int(input_batch.num_reqs)
        if n == 0 or end <= start:
            return
        req_ids = [str(value) for value in input_batch.req_ids[:n]]
        qsl = input_batch.query_start_loc[: n + 1].detach().cpu().numpy()
        if len(qsl) != n + 1 or int(qsl[0]) != 0 or end > int(qsl[-1]):
            self._reject(
                f"DFlash context slice [{start}, {end}) is outside target query rows "
                f"{tuple(int(value) for value in qsl)}"
            )
        positions = speculator.context_positions[start:end].detach().cpu().numpy()
        draft_gid = getattr(self, "dflash_gid", getattr(self, "kv_gid", 0))
        draft_pool = getattr(self, "dflash_pool", DRAFT_POOL)
        draft_gids = tuple(getattr(speculator, "draft_kv_cache_group_ids", (draft_gid,)))
        draft_index = draft_gids.index(draft_gid)
        slots = speculator._context_slot_mappings[draft_index][start:end].detach().cpu().numpy()
        row_positions: list[list[int]] = [[] for _ in range(n)]
        map_slots: list[int] = []
        map_rows: list[int] = []
        map_positions: list[int] = []
        for offset, (slot, position) in enumerate(zip(slots, positions, strict=True), start=start):
            if int(slot) < 0:
                continue
            row = int(np.searchsorted(qsl[1:], offset, side="right"))
            if not 0 <= row < n:
                self._reject(f"DFlash context token {offset} has no request row")
            logical = int(position)
            row_positions[row].append(logical)
            map_slots.append(int(slot))
            map_rows.append(row)
            map_positions.append(logical)
        if not map_slots:
            return
        self._inject_dflash_fault(req_ids)
        self._dflash_dispatch_seq += 1
        descriptor = DispatchDescriptor(
            f"dflash-context-{self._dflash_dispatch_seq}@{self.seq}",
            draft_pool,
            "dflash_context_kv",
            tuple(
                DispatchRow(request, writes=tuple(row_positions[row]))
                for row, request in enumerate(req_ids)
            ),
            SlotMap(map_slots, map_rows, map_positions),
        )
        if getattr(self, "dflash_events", None) is not None:
            draft_block_size = int(self.managers[draft_gid].block_size)
            for row, request in enumerate(req_ids):
                touched = sorted(
                    {
                        slot // draft_block_size
                        for slot, mapped_row in zip(map_slots, map_rows, strict=True)
                        if mapped_row == row
                    }
                )
                published = [
                    block
                    for block in touched
                    if self.ledger.block(self.dflash_pool, block).state is BlockState.PUBLISHED
                ]
                if not published:
                    continue
                request_shadow = self.ledger.requests.get((request, self.dflash_pool))
                snapshot = request_shadow.snapshot if request_shadow is not None else None
                target_shadow = self.ledger.requests.get((request, POOL))
                target_snapshot = target_shadow.snapshot if target_shadow is not None else None
                self.dflash_events.append(
                    {
                        "seq": self.seq,
                        "dispatch": descriptor.dispatch_id,
                        "row": row,
                        "request": request,
                        "positions": row_positions[row],
                        "slots": [
                            slot
                            for slot, mapped_row in zip(map_slots, map_rows, strict=True)
                            if mapped_row == row
                        ],
                        "frontier": self._dflash_frontiers.get(request),
                        "snapshot": (
                            {
                                "blocks": list(snapshot.blocks),
                                "num_tokens": snapshot.num_tokens,
                                "num_computed": snapshot.num_computed,
                            }
                            if snapshot is not None
                            else None
                        ),
                        "target_snapshot": (
                            {
                                "blocks": list(target_snapshot.blocks),
                                "num_tokens": target_snapshot.num_tokens,
                                "num_computed": target_snapshot.num_computed,
                            }
                            if target_snapshot is not None
                            else None
                        ),
                        "blocks": [
                            {
                                "block": block,
                                "ref_cnt": int(self.pool.blocks[block].ref_cnt),
                                "hashed": self.pool.blocks[block].block_hash is not None,
                                "hash_num_tokens": getattr(
                                    self.pool.blocks[block], "block_hash_num_tokens", None
                                ),
                                "holders": sorted(
                                    str(shadow.snapshot.request)
                                    for (rid, pool), shadow in self.ledger.requests.items()
                                    if pool == self.dflash_pool and block in shadow.snapshot.blocks
                                ),
                                "cow_pin": int(self._cow_pins[block]),
                                "cow_request": (
                                    self.dflash_pool,
                                    block,
                                )
                                in self._cow_request_blocks,
                                "cow_cache": (
                                    self.dflash_pool,
                                    block,
                                )
                                in self._cow_cache_blocks,
                                "target_state": (
                                    None
                                    if self.dflash_separate
                                    else self.ledger.block(POOL, block).state.value
                                ),
                                "target_filled": (
                                    None
                                    if self.dflash_separate
                                    else self.ledger.block(POOL, block).filled
                                ),
                                "target_pending": (
                                    None
                                    if self.dflash_separate
                                    else self.ledger.block(POOL, block).pending_publish
                                ),
                                "draft_state": self.ledger.block(
                                    self.dflash_pool, block
                                ).state.value,
                                "draft_filled": self.ledger.block(self.dflash_pool, block).filled,
                                "draft_pending": self.ledger.block(
                                    self.dflash_pool, block
                                ).pending_publish,
                            }
                            for block in published
                        ],
                    }
                )
        stream = current_stream_handle() if self.stream_epochs else None
        if stream is not None:
            descriptor = replace(descriptor, stream=stream)
        self.ledger.dispatch(descriptor)
        self._opened(descriptor.dispatch_id)
        self._advance_dflash_frontiers(descriptor.rows)
        self.stats["dflash_context_forwards"] += 1
        self.stats["dflash_context_tokens"] += len(map_slots)
        self.stats["dispatches"] += 1
        self.stats["dispatch_tokens"] += len(map_slots)

    def _dispatch_dflash_query(
        self, speculator: Any, num_tokens: int, slot_mappings: dict[str, Any] | None
    ) -> None:
        """Validate one eager DFlash bonus-plus-mask query from its concrete GPU layout."""
        input_batch = self._dflash_input_batch
        if input_batch is None:
            self._reject("DFlash query execution outside its proposer call")
        if slot_mappings is None:
            self._reject("DFlash query execution without per-layer slot mappings")
        draft_layers = self._draft_layer_names(speculator)
        missing = tuple(layer for layer in draft_layers if layer not in slot_mappings)
        if not draft_layers or missing:
            self._reject(
                f"DFlash draft layers {draft_layers!r} have missing slot mappings {missing!r}"
            )
        n = int(input_batch.num_reqs)
        if n == 0:
            return
        req_ids = [str(value) for value in input_batch.req_ids[:n]]
        buffers = speculator.input_buffers
        qsl = buffers.query_start_loc[: n + 1].detach().cpu().numpy()
        total = int(qsl[-1])
        if total > num_tokens:
            self._reject(f"DFlash query has {total} tokens but the model executes {num_tokens}")
        positions = buffers.positions[:total].detach().cpu().numpy()
        seq_lens = buffers.seq_lens[:n].detach().cpu().numpy()
        slots = slot_mappings[draft_layers[0]][:total].detach().cpu().numpy()
        draft_gid = getattr(self, "dflash_gid", getattr(self, "kv_gid", 0))
        draft_pool = getattr(self, "dflash_pool", DRAFT_POOL)
        managers = getattr(self, "managers", ())
        draft_block_size = int(managers[draft_gid].block_size) if managers else int(self.block_size)
        table = speculator.block_tables.input_block_tables[draft_gid][:n].detach().cpu().numpy()
        self._inject_dflash_fault(req_ids)
        self._dflash_dispatch_seq += 1
        try:
            write, attention = build_positioned_dispatches(
                self.seq,
                draft_pool,
                draft_block_size,
                req_ids,
                qsl,
                positions,
                seq_lens,
                table,
                slots,
                site=f"dflash-query-{self._dflash_dispatch_seq}",
            )
        except ValueError as exc:
            self._reject(f"DFlash query layout: {exc}")
        stream = current_stream_handle() if self.stream_epochs else None
        if stream is not None:
            write = replace(write, stream=stream)
            attention = replace(attention, stream=stream)
        for descriptor in (write, attention):
            self.ledger.dispatch(descriptor)
            self._opened(descriptor.dispatch_id)
        self._advance_dflash_frontiers(write.rows)
        self.stats["dflash_query_forwards"] += 1
        self.stats["dflash_query_tokens"] += total
        self.stats["dispatches"] += 2
        self.stats["dispatch_tokens"] += total

    def _dispatch_dflash_graph_query(self, speculator: Any, num_tokens: int) -> None:
        """Describe a DFlash query whose model forward is inside a captured full graph."""
        draft_layers = self._draft_layer_names(speculator)
        gid = int(speculator.draft_kv_cache_group_id)
        slot_mapping = speculator.block_tables.slot_mappings[gid]
        self._dispatch_dflash_query(
            speculator,
            num_tokens,
            {layer: slot_mapping for layer in draft_layers},
        )
        self.stats["dflash_graph_replays"] += 1

    def _dispatch_mtp(
        self,
        speculator: Any,
        num_tokens: int,
        slot_mappings: dict[str, Any] | None,
        module_index: int,
        *,
        full_graph: bool = False,
    ) -> None:
        """Validate one eager multi-module MTP forward against its concrete GPU layout.

        Every module processes the same prepared query window, including any rejected-token
        re-prefill prefix. Read/write positions, the block table and per-layer slot map come from
        the speculator's device buffers rather than target scheduling counts, which do not
        describe that re-prefill layout.
        """
        input_batch = self._mtp_input_batch
        if input_batch is None:
            self._reject("multi-module MTP model execution outside its proposer call")
        if not 0 <= module_index < len(self.mtp_gids):
            self._reject(f"MTP module {module_index} has no draft KV group among {self.mtp_gids!r}")
        if slot_mappings is None:
            self._reject("multi-module MTP execution without per-layer slot mappings")

        gid = self.mtp_gids[module_index]
        layer_names = (self.mtp_layers[module_index],)
        if layer_names[0] not in slot_mappings:
            self._reject(
                f"MTP group {gid} has layers {layer_names!r}, but the execution "
                f"published mappings for {tuple(slot_mappings)!r}"
            )
        n = int(input_batch.num_reqs)
        if n == 0:
            return
        req_ids = list(input_batch.req_ids[:n])
        buffers = speculator.input_buffers
        qsl = buffers.query_start_loc[: n + 1].detach().cpu().numpy()
        total = int(qsl[-1])
        if total > num_tokens:
            self._reject(f"MTP query has {total} tokens but the model executes {num_tokens}")
        positions = buffers.positions[:total].detach().cpu().numpy()
        seq_lens = buffers.seq_lens[:n].detach().cpu().numpy()
        slots = slot_mappings[layer_names[0]][:total].detach().cpu().numpy()
        table = speculator.block_tables.input_block_tables[gid][:n].detach().cpu().numpy()
        pool = self.mtp_pools[module_index]
        inject_fault = self.mtp_fault == "draft_early_free" or (
            self.mtp_fault == "draft_graph_early_free" and full_graph
        )
        if inject_fault and not self.mtp_fault_done:
            victim = next(
                (
                    int(block)
                    for rid in req_ids
                    for block in reversed(self.caches[gid].ids(rid))
                    if block
                ),
                None,
            )
            if victim is not None:
                self.ledger.free(pool, (victim,), site=f"mtp-fault@{self.seq}")
                self.mtp_fault_done = True
                self.stats["mtp_fault_draft_early_free"] += 1
                self.stats["mtp_fault_block"] = victim

        self._mtp_dispatch_seq += 1
        site = f"mtp-{module_index}-{self._mtp_dispatch_seq}"
        try:
            write, attn = build_positioned_dispatches(
                self.seq,
                pool,
                int(self.managers[gid].block_size),
                req_ids,
                qsl,
                positions,
                seq_lens,
                table,
                slots,
                window=self.window_of.get(gid),
                site=site,
            )
        except ValueError as exc:
            self._reject(f"MTP group {gid} layout: {exc}")
        # The EAGLE/MTP cache contract treats the last accepted position of an offloaded,
        # block-aligned prefix as a volatile proof slot that a later request re-prefills. Keep
        # the exception exact: only Transformers MTP's shared group, the concrete previous
        # frontier, a block-terminal position, and only when the core confirms one mapper.
        # Any other shared or non-terminal published write still reaches the ordinary P2 rule.
        if self.mtp_shared and self.offload is not None:
            for token_index, (slot, position) in enumerate(zip(slots, positions, strict=True)):
                if int(slot) < 0:
                    continue
                row = int(np.searchsorted(qsl[1:], token_index, side="right"))
                rid = str(req_ids[row])
                frontier = self._mtp_frontiers.get((pool, rid))
                if (
                    frontier is not None
                    and int(position) + 1 == int(frontier)
                    and int(position) % self.block_size == self.block_size - 1
                    and int(slot) % self.block_size == self.block_size - 1
                    and self.core_ledger.authorize_volatile_rewrite(
                        pool,
                        rid,
                        int(position),
                        site=f"mtp-offload-volatile-tail@{self.seq}",
                    )
                ):
                    self.stats["mtp_offload_volatile_tail_rewrites"] += 1
        trace = getattr(self, "mtp_events", None)
        if trace is not None:
            published = []
            request_shadows = self.core_ledger.requests
            for token_index, (slot, position) in enumerate(zip(slots, positions, strict=True)):
                if int(slot) < 0:
                    continue
                block = int(slot) // self.block_size
                offset = int(slot) % self.block_size
                shadow = self.core_ledger.block(pool, block)
                if shadow.state is not BlockState.PUBLISHED or (
                    shadow.pending_publish >> offset & 1
                ):
                    continue
                row = int(np.searchsorted(qsl[1:], token_index, side="right"))
                published.append(
                    {
                        "token_index": token_index,
                        "row": row,
                        "request": str(req_ids[row]),
                        "position": int(position),
                        "slot": int(slot),
                        "block": block,
                        "generation": shadow.generation,
                        "owner": None if shadow.owner is None else str(shadow.owner),
                        "readers": sorted(str(value) for value in shadow.readers),
                        "reserved": shadow.reserved,
                        "filled": shadow.filled,
                        "pending_publish": shadow.pending_publish,
                        "frontier": self._mtp_frontiers.get((pool, str(req_ids[row]))),
                        "holders": sorted(
                            str(request_shadow.snapshot.request)
                            for (_, shadow_pool), request_shadow in request_shadows.items()
                            if shadow_pool == pool and block in request_shadow.snapshot.blocks
                        ),
                    }
                )
            if published:
                trace.append(
                    {
                        "seq": self.seq,
                        "module": module_index,
                        "pool": pool,
                        "site": site,
                        "published_writes": published,
                    }
                )
        stream = current_stream_handle() if self.stream_epochs else None
        if stream is not None:
            write = replace(write, stream=stream)
            attn = replace(attn, stream=stream)
        for descriptor in (write, attn):
            self.ledger.dispatch(descriptor)
            self._opened(descriptor.dispatch_id)
        for rid, row in zip(req_ids, write.rows, strict=True):
            if row.writes:
                self._mtp_frontiers[(pool, str(rid))] = int(row.writes.stop)
        self.stats["mtp_draft_forwards"] += 1
        self.stats["mtp_draft_tokens"] += sum(1 for value in slots if int(value) >= 0)
        self.stats["dispatches"] += 2
        self.stats["dispatch_tokens"] += total

    def _dispatch_mtp_full_graph(self, speculator: Any, descriptor: Any) -> None:
        """Validate every multi-module MTP module immediately before one full-graph replay.

        Full replay bypasses ``_run_model``, but ``propose`` has already rebuilt the persistent
        positions, tables and per-group slot mappings the captured graph consumes.
        """
        slot_buffer = speculator.block_tables.slot_mappings
        slot_mappings = {
            layer_name: slot_buffer[gid]
            for layer_name, gid in zip(self.mtp_layers, self.mtp_gids, strict=True)
        }
        for module_index in range(len(self.mtp_gids)):
            self._dispatch_mtp(
                speculator,
                int(descriptor.num_tokens),
                slot_mappings,
                module_index,
                full_graph=True,
            )
        self.stats["mtp_full_graph_replays"] += 1

    def _dispatch_window(
        self,
        g,
        req_ids,
        computed,
        scheduled,
        block_tables,
        slot_mappings,
        n_pad,
        total_pad,
        null,
        stream,
        input_batch=None,
    ):
        """An auxiliary attention group's dispatches over its own table and slot map."""
        manager = self.managers[g]
        bs = manager.block_size
        if self.tables == "mapping":
            ids = self.caches[g].ids
            lists = [ids(rid) for rid in req_ids]
            table_g, slots_g = mapping_tables(lists, computed, scheduled, bs, n_pad, null)
        else:
            table_g = block_tables[g][:n_pad].cpu().numpy()
            slots_g = slot_mappings[g][:total_pad].cpu().numpy().astype(np.int64)
            if self.worker_table_check and input_batch is not None:
                self._check_stale_worker_blocks(g, req_ids, input_batch, null)
        write, attn = build_dispatches(
            self.seq,
            self.pool_of_group[g],
            bs,
            req_ids,
            computed,
            scheduled,
            table_g,
            n_pad,
            slots_g,
            window=self.window_of[g],
            site=self.pool_of_group[g],
        )
        self._inject_offload_window_fault(self.pool_of_group[g], table_g)
        if stream is not None:
            write, attn = replace(write, stream=stream), replace(attn, stream=stream)
        if self.tags is not None:
            self.tags.dispatch(g, block_tables[g], slot_mappings[g], write, attn, req_ids)
        if self.validators is None:
            for d in (write, attn):
                self.ledger.dispatch(d)
                self._opened(d.dispatch_id)
        else:
            # The device validators over this group's own tables and mirror.
            self._dispatch_gpu(
                _Rows(req_ids, len(req_ids)),
                block_tables[g],
                slot_mappings[g],
                write,
                attn,
                self.mirrors[g],
            )
        stat = "window_dispatches" if self.window_of[g] is not None else "attention_dispatches"
        self.stats[stat] += 2

    def _check_stale_worker_blocks(self, g, req_ids, input_batch, null) -> None:
        """A worker block table is the GPU mirror of the manager's ``req_to_blocks``. Where the
        manager has released a block -- a null entry in its mapping, as a sliding-window eviction
        produces -- but the worker row still names a block, the worker references a block the
        request no longer holds. The per-step gathered table the kernels consume masks such an
        entry, so it is a latent mirror divergence reported here against the manager mapping. The
        append-only worker table is only inconsistent where the manager released something, so a
        request whose mapping has no null needs no device read."""
        idsfn = self.caches[g].ids
        mappings = [idsfn(rid) for rid in req_ids]
        if not any(block is None for mapping in mappings for block in mapping):
            return
        pool = self.pool_of_group[g]
        try:
            # The runner's append-only per-group worker tables (persistent-row indexed); the
            # per-step gathered tables are a masked copy, so read the raw ones.
            raw_t = self.runner.block_tables.block_tables[g]
            raw = np.asarray(raw_t.np if hasattr(raw_t, "np") else raw_t.gpu.cpu().numpy())
            idx_t = input_batch.idx_mapping
            idx_map = np.asarray(idx_t.cpu().numpy() if hasattr(idx_t, "cpu") else idx_t).astype(
                np.int64
            )
        except Exception:  # raw worker table absent on an unexpected build
            self.stats["worker_table_unavailable"] = (
                self.stats.get("worker_table_unavailable", 0) + 1
            )
            return
        for i, (rid, mapping) in enumerate(zip(req_ids, mappings, strict=True)):
            if i >= len(idx_map):
                continue
            prow = int(idx_map[i])
            if prow < 0 or prow >= len(raw):
                continue
            row = raw[prow]
            for j in range(min(len(mapping), len(row))):
                if mapping[j] is not None:
                    continue  # the request still holds this block; its live region is in flux
                dev = int(row[j])
                if dev != null:
                    self.stats["stale_worker_block"] = self.stats.get("stale_worker_block", 0) + 1
                    self.reporter.emit(
                        Report(
                            OutcomeKind.VIOLATION,
                            "mapping_mismatch",
                            f"worker block_table slot {j} names block {dev} that request {rid!r} "
                            f"no longer holds (the manager released it)",
                            policy=Policy.P3,
                            pool=pool,
                            request=rid,
                            site=f"worker-table@{self.seq}",
                        )
                    )
                    break

    def _inject_offload_window_fault(self, pool: str, table: Any) -> None:
        """Free one real loaded SWA block in the shadow for the matched liveness control."""
        if self.offload_fault != "swa_load_early_free" or self.offload_fault_done:
            return
        filled = getattr(self, "_offload_filled_pools", set())
        victim = next(
            (int(block) for block in np.asarray(table).reshape(-1) if (pool, int(block)) in filled),
            None,
        )
        if victim is None:
            return
        self.ledger.free(pool, (victim,), site=f"offload-fault@{self.seq}")
        self.offload_fault_done = True
        self.stats["offload_fault_swa_load_early_free"] += 1
        self.stats["offload_fault_block"] = victim

    def _inject_offload_state_fault(self, pool: str, table: Any) -> None:
        """Free one loaded recurrent block that this state dispatch will read or write."""
        if self.offload_fault != "state_load_early_free" or self.offload_fault_done:
            return
        filled = getattr(self, "_offload_filled_pools", set())
        rows = np.asarray(table)
        victim = next(
            (
                int(rows[row, column])
                for row, (source, destination) in enumerate(self._columns)
                for column in ((source,) if source == destination else (source, destination))
                if column >= 0 and (pool, int(rows[row, column])) in filled
            ),
            None,
        )
        if victim is None:
            return
        self.ledger.free(pool, (victim,), site=f"offload-fault@{self.seq}")
        self.offload_fault_done = True
        self.stats["offload_fault_state_load_early_free"] += 1
        self.stats["offload_fault_block"] = victim

    def _dispatch_state(self, req_ids, computed, scheduled, block_tables, n_pad, null, stream):
        """One descriptor per recurrent-state group over its gathered block table (the device
        copy, or the manager's block lists), the same columns for every group. With the device
        validators, two dense launches per group over the gathered table before the host
        dispatch: the read columns and the written columns (``row_write``)."""
        mode = self.state_mode
        bs = self.managers[self.state_gids[0]].block_size
        self._columns = [
            state_columns(int(computed[i]), int(scheduled[i]), bs, mode)
            for i in range(len(req_ids))
        ]
        for g in self.state_gids:
            manager = self.managers[g]
            if self.tables == "mapping":
                ids = self.caches[g].ids
                table_g = dense_table([ids(rid) for rid in req_ids], n_pad, null)
            else:
                table_g = block_tables[g][:n_pad].cpu().numpy()
            self._inject_offload_state_fault(self.pool_of_group[g], table_g)
            d = build_state_dispatch(
                self.seq,
                self.pool_of_group[g],
                manager.block_size,
                mode,
                req_ids,
                computed,
                scheduled,
                table_g,
                n_pad,
                site=state_pool(g),
            )
            if stream is not None:
                d = replace(d, stream=stream)
            if self.validators is not None:
                self._validate_state_columns(g, block_tables[g], req_ids, n_pad, d.dispatch_id)
            self.ledger.dispatch(d)
            self._opened(d.dispatch_id)
            self.stats["state_dispatches"] += 1
        self.stats["state_migrations"] += sum(1 for s, d in self._columns if s >= 0 and s != d)

    def _validate_state_columns(self, g, bt, req_ids, n_pad, tag) -> None:
        """The device validators over one state group's gathered table: one launch for the
        columns the rows read (the previous state; rows without one read nothing), one for
        the columns they write (the running state), both before the host dispatch so the
        mirror shows the step's starting state; padding rows carry their whole width."""
        import torch

        from kvguard.gpu.validate import PADDING_ROW

        t0 = time.perf_counter()
        stream = torch.cuda.current_stream().cuda_stream
        mirror = self.mirrors[g]
        owner = mirror.owners.id_of
        n = len(req_ids)
        owners = [owner(rid) for rid in req_ids] + [PADDING_ROW] * (n_pad - n)
        pad = [0] * (n_pad - n)
        mirror.sync(stream)
        reads_end = [s + 1 if s >= 0 else 0 for s, _ in self._columns] + pad
        reads_start = [max(s, 0) for s, _ in self._columns] + pad
        self.validators.validate_dense(
            mirror,
            bt.data_ptr(),
            int(bt.stride(0)),
            n_pad,
            int(bt.shape[1]),
            owners,
            reads_end,
            stream,
            f"read-{tag}",
            reads_start,
        )
        writes_end = [d + 1 for _, d in self._columns] + pad
        writes_start = [d for _, d in self._columns] + pad
        self.validators.validate_dense(
            mirror,
            bt.data_ptr(),
            int(bt.stride(0)),
            n_pad,
            int(bt.shape[1]),
            owners,
            writes_end,
            stream,
            f"write-{tag}",
            writes_start,
            [1] * n + pad,
        )
        self.stats["gpu_validator_launches"] += 2
        self.timings["gpu_validate_host_s"] += time.perf_counter() - t0

    def _check_state_columns(self, model_state, input_batch) -> None:
        """The worker's source and running columns after ``preprocess_state`` against the
        adapter's derivation for the same batch rows."""
        n = int(input_batch.num_reqs)
        if n == 0 or len(self._columns) != n:
            return
        idx = input_batch.idx_mapping[:n]
        src = model_state._mamba_src_col_gpu[idx].cpu().numpy()
        dst = model_state._mamba_state_idx_gpu[idx].cpu().numpy()
        self.stats["state_columns_checked"] += n
        for i, (s, d) in enumerate(self._columns):
            ws, wd = int(src[i]), int(dst[i])
            if ws != s or wd != d:
                self.stats["state_column_mismatches"] += 1
                self.reporter.emit(
                    integration_error(
                        "state_column_mismatch",
                        f"row {i} ({input_batch.req_ids[i]!r}): the worker reads column {ws} and "
                        f"writes column {wd}, the adapter derived {s} and {d}",
                        pool=self.pool_of_group[self.state_gids[0]],
                        request=input_batch.req_ids[i],
                        site=f"dispatch:state@{self.seq}",
                    )
                )

    # ------------------------------------------------------------------ device validators
    def _dispatch_gpu(self, input_batch, bt, slots, write, attn, mirror) -> None:
        """One group's two dispatches with the device validators launched before the step's
        kernels, in the host order: the mirror shows the write's slots filled when the table
        is checked. ``bt`` and ``slots`` are the group's gathered block table and slot map,
        ``mirror`` the device mirror of its pool; a row's reads start at its window."""
        import torch

        from kvguard.gpu.validate import PADDING_ROW

        t0 = time.perf_counter()
        stream = torch.cuda.current_stream().cuda_stream
        owner = mirror.owners.id_of
        req_ids = list(input_batch.req_ids[: int(input_batch.num_reqs)])
        mirror.sync(stream)
        self.validators.validate_slot_map(
            mirror,
            slots.data_ptr(),
            slots.element_size(),
            len(write.mapping.slots),
            [PADDING_ROW if r >= len(req_ids) else owner(req_ids[r]) for r in write.mapping.rows],
            stream,
            write.dispatch_id,
        )
        host = time.perf_counter()
        self.ledger.dispatch(write)
        self._opened(write.dispatch_id)
        t1 = time.perf_counter()
        mirror.sync(stream)
        n_pad = len(attn.rows)
        row_owner = [PADDING_ROW if r.padding else owner(r.request) for r in attn.rows]
        row_end = [r.reads.stop for r in attn.rows]
        row_start = [r.reads.start for r in attn.rows]
        self.validators.validate_dense(
            mirror,
            bt.data_ptr(),
            int(bt.stride(0)),
            n_pad,
            int(bt.shape[1]),
            row_owner,
            row_end,
            stream,
            attn.dispatch_id,
            row_start,
        )
        t2 = time.perf_counter()
        self.ledger.dispatch(attn)
        self._opened(attn.dispatch_id)
        self.stats["gpu_validator_launches"] += 2
        self.timings["gpu_validate_host_s"] += (host - t0) + (t2 - t1)

    def _collect_device(self, final: bool = False) -> None:
        """Read back the device validators' and the tag tier's completed launches (all of them
        when ``final``) and report them."""
        v = self.validators if self.validators is not None else getattr(self.tags, "v", None)
        if v is None:
            return
        self._collect_gpu(v.drain() if final else v.completed())

    def _collect_gpu(self, results) -> None:
        from kvguard.gpu.validate import report_results

        if self.tags is not None:
            results = self.tags.collect(results)
        if not results:
            return
        self.stats["gpu_results"] += len(results)
        # A result belongs to the pool its dispatch tag names (``write-swa3@12``: ``swa3``), the
        # attention pool otherwise.
        by_pool: dict[str, list] = {}
        for r in results:
            m = re.match(r"(?:write|attn|read)-([^@]+)@", r.tag)
            by_pool.setdefault(m.group(1) if m else POOL, []).append(r)
        for pool, rs in by_pool.items():
            self.stats["gpu_reports"] += report_results(rs, self.reporter, pool)
        for r in results:
            if r.count:
                self.stats["gpu_violations"] += r.count
            if r.gpu_ms is not None:
                self.timings["gpu_validate_device_ms"] += r.gpu_ms

    def _profiled(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            prof = self.profiler
            if prof is None:
                return fn(*args, **kwargs)
            if not (self.profile_steps[0] <= self.seq < self.profile_steps[1]):
                if self.seq >= self.profile_steps[1] and not self._profile_written:
                    # The stats are written when the window closes: a server stopped by signal
                    # (the serving harness) does not reach finish().
                    self._profile_written = True
                    self._write_profile()
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
            fh.write(f"steps {self.profile_steps[0]}..{self.profile_steps[1] - 1}\n")
            text = pstats.Stats(self.profiler, stream=fh)
            text.sort_stats("cumulative").print_stats(60)
            text.sort_stats("tottime").print_stats(40)

    def _current_step_key(self) -> int | None:
        """The step the worker thread executes, when inside ``execute_model``; the scheduler
        thread's latest ``schedule()`` otherwise (asynchronous scheduling runs the next
        ``schedule()`` while the worker still prepares the previous step)."""
        executing = getattr(self, "_executing_output", None)
        return id(executing) if executing is not None else self._step_key

    def _opened(self, dispatch_id: str) -> None:
        self.open.setdefault(self._current_step_key() or 0, []).append(dispatch_id)

    def finish(self) -> None:
        """Read back every outstanding validator result and apply every queued event (call once
        the engine is idle)."""
        self._write_profile()
        self.stats["offload_store_epochs_pending"] = len(self._offload_store_epochs)
        if self.nvtx is not None:
            for key, value in self.nvtx.record().items():
                if isinstance(value, (int, bool)):
                    self.stats[f"nvtx_{key}"] = int(value)
        self._drain_mooncake_events()
        self.stats["mooncake_loads_pending"] = len(self._mooncake_pending)
        self._reap_zero_uses(wait=True)
        for dids in self.open.values():
            for did in dids:
                self.ledger.complete(did)
        self.open.clear()
        if self.worker is not None:
            self.worker.flush()
        self._collect_device(final=True)
        if self.tags is not None:
            for k, v in self.tags.stats().items():
                self.stats[k] = v
        for k, v in self.core_ledger.counters.items():
            self.stats[f"ledger_{k}"] = v


class _Rows:
    """The two ``InputBatch`` facts ``_dispatch_gpu`` reads, for a window group's dispatches."""

    def __init__(self, req_ids, num_reqs: int):
        self.req_ids = req_ids
        self.num_reqs = num_reqs


def _zeroer_resources(zeroer: Any) -> tuple[str, ...]:
    """Return the concrete staging tensors selected by a ``KVBlockZeroer``.

    An older layout has one tensor per field; the current one makes each a list and advances
    ``_id_buffer_index`` after selecting the pair, so the pre-call index names the pair this call
    will mutate.
    """

    def selected(value: Any) -> Any | None:
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            index = int(getattr(zeroer, "_id_buffer_index", 0))
            return value[index % len(value)]
        return value

    resources = []
    for kind, value in (
        ("pinned", selected(getattr(zeroer, "_ids_pinned", None))),
        ("device", selected(getattr(zeroer, "_ids_gpu", None))),
    ):
        if value is not None and callable(getattr(value, "data_ptr", None)):
            resources.append(f"vllm-zero-{kind}:{int(value.data_ptr())}")
    return tuple(resources)


class VllmZeroResourceAttachment(_ZeroResourceTracker):
    """Narrow attachment for mutable zeroing descriptors outside the full adapter boundary.

    This is used for speculative historical runners whose cache layouts the full vLLM adapter
    does not claim to model. It checks only the actual zeroer's transient staging resources;
    it does not imply P1--P4 coverage of the runner's other cache operations.
    """

    def __init__(self, llm_or_engine: Any, reporter: Reporter | None = None):
        engine = getattr(
            getattr(llm_or_engine, "llm_engine", llm_or_engine), "engine_core", llm_or_engine
        )
        engine = getattr(engine, "engine_core", engine)
        self.engine = engine
        self.runner = engine.model_executor.driver_worker.worker.model_runner
        self.reporter = reporter or Reporter(fail_fast=True)
        self.core_ledger = Ledger(self.reporter)
        self.ledger = self.core_ledger
        self.stats: Counter = Counter()
        self._zero_pending: list[tuple[Any, str]] = []
        self._zero_seq = 0
        if not self._install_zero_resources(self.runner):
            report = unsupported(
                "zeroing_descriptor_resources",
                "the model runner has no persistent pinned/device zeroing-ID staging tensors",
            )
            self.reporter.emit(report)
            raise Unsupported(report)

    def finish(self) -> None:
        """Wait for tracked zeroing operations and publish ledger counters."""
        self._reap_zero_uses(wait=True)
        for key, value in self.core_ledger.counters.items():
            self.stats[f"ledger_{key}"] = value


def _call_free_blocks(orig: Callable[..., Any], ordered_blocks, *args, **kwargs) -> list[Any]:
    """Materialize a possibly lazy block iterable and preserve the framework's call shape."""
    blocks = list(ordered_blocks)
    orig(blocks, *args, **kwargs)
    return blocks


def _input_pre_step_computed(
    input_batch: Any,
    n: int,
    scheduled: np.ndarray,
    *,
    force_device: bool = False,
) -> tuple[np.ndarray, bool]:
    """Read the per-row pre-step position across Model Runner V2 revisions.

    Current vLLM carries the CPU mirror directly; earlier revisions expose only the device
    ``seq_lens`` (``computed + scheduled``), so recover the value from it. The boolean lets the
    caller account for that compatibility readback separately.
    """
    direct = getattr(input_batch, "num_computed_tokens_np", None)
    if direct is not None and not force_device:
        return np.asarray(direct[:n]), False
    seq_lens = np.asarray(input_batch.seq_lens[:n].cpu().numpy(), dtype=np.int64)
    computed = seq_lens - np.asarray(scheduled[:n], dtype=np.int64)
    if np.any(computed < 0):
        raise ValueError("vLLM input batch has a sequence length below its scheduled length")
    return computed, True


def _input_execution_counts(input_batch: Any, n: int) -> tuple[np.ndarray, bool]:
    """Return the concrete per-row query counts consumed by the target/drafter.

    Adaptive verification keeps ``num_scheduled_tokens`` as a CPU upper bound and reallocates the
    real draft budget on the GPU, so its ``query_start_loc`` names the real row lengths; other
    batches keep the cheap CPU counts.
    """
    upper = np.asarray(input_batch.num_scheduled_tokens[:n], dtype=np.int64)
    if getattr(input_batch, "max_query_len", None) is None:
        return upper, False
    qsl = np.asarray(input_batch.query_start_loc[: n + 1].cpu().numpy(), dtype=np.int64)
    counts = np.diff(qsl)
    if np.any(counts < 0) or np.any(counts > upper):
        raise ValueError(
            "vLLM adaptive-verification query lengths are outside their CPU upper bounds"
        )
    if int(counts.sum()) != int(input_batch.num_tokens):
        raise ValueError(
            "vLLM adaptive-verification query lengths do not sum to the executed token count"
        )
    return counts, True


def _marking(orig: Callable[..., Any], which: str, me: VllmAttachment) -> Callable[..., Any]:
    """``orig`` wrapped to mark the blocks it names: its ``blocks`` argument, for
    ``cache_full_blocks`` the slice ``blocks[num_cached_blocks:num_full_blocks]`` it hashes
    (``full``), the one ``block`` of ``cache_partial_block`` (``block``), or the source and
    destination of ``move_block_hashes`` (``pair``)."""
    sig = inspect.signature(orig)

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs).arguments
            if which == "block":
                blocks = (bound["block"],)
            elif which == "pair":
                blocks = (bound["src_block"], bound["dst_block"])
            else:
                blocks = bound["blocks"]
                if which == "full":
                    blocks = blocks[bound["num_cached_blocks"] : bound["num_full_blocks"]]
            me._mark(b.block_id for b in blocks)
            if which == "full" and me.tags is not None:
                me.tags.publish([b.block_id for b in blocks])
        except (TypeError, KeyError) as exc:
            me.stats["mark_failures"] += 1
            for cache in me.caches.values():
                cache.invalidate()
            me.reporter.emit(
                integration_error(
                    "snapshot_cache_mark",
                    f"{which}: {exc!r}"[:200],
                    pool=POOL,
                    site=f"snapshot:{me.seq}",
                )
            )
        return out

    return wrapped


def attach(llm_or_engine: Any, reporter: Reporter | None = None, **hooks) -> VllmAttachment:
    """Install the adapter on a live engine (an ``LLM``, an ``InprocClient`` or an ``EngineCore``)."""
    return VllmAttachment(llm_or_engine, reporter, **hooks)


def attach_zero_resources(
    llm_or_engine: Any, reporter: Reporter | None = None
) -> VllmZeroResourceAttachment:
    """Install only the zeroing descriptor-resource check on a live vLLM engine."""
    return VllmZeroResourceAttachment(llm_or_engine, reporter)


__all__ = [
    "POOL",
    "VllmAttachment",
    "VllmZeroResourceAttachment",
    "attach",
    "attach_zero_resources",
    "state_pool",
    "window_pool",
]
