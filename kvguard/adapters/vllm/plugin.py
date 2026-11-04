"""Attach the vLLM adapter inside the engine-core process through vLLM's plugin mechanism.

vLLM loads the entry points of the group ``vllm.general_plugins`` (filtered by ``VLLM_PLUGINS``)
in every process that builds an engine core or worker. The engine core of a default ``LLM`` or
``vllm serve`` runs in a subprocess that :func:`attach` cannot reach from the user's process;
this plugin attaches there. :func:`register`, the entry point, wraps ``EngineCore.step``,
``step_with_batch_queue`` and ``shutdown`` on the class object (in memory only) with wrappers
that attach the adapter at the first step and write the record. The record is also written every
``KVGUARD_VLLM_FLUSH_EVERY`` steps and once per idle period, since a client may terminate the
process rather than shut it down. Any failure to attach is recorded and the engine runs
unmodified.

:func:`dist_info` writes a ``*.dist-info`` whose ``entry_points.txt`` names this module, so the
plugin is discoverable by putting its parent on ``PYTHONPATH`` without installing a package.

Environment (read in the engine-core process): ``KVGUARD_VLLM_OUT`` (record path; without it
nothing is written), ``KVGUARD_VLLM_GPU_VALIDATE``, ``KVGUARD_VLLM_AUDIT_EVERY``,
``KVGUARD_VLLM_TABLES`` (``device`` or ``mapping``), ``KVGUARD_VLLM_CHECKER`` (``inline`` or
``thread``), ``KVGUARD_VLLM_FLUSH_EVERY``, ``KVGUARD_VLLM_TAGS`` (the tag tier), and
``KVGUARD_VLLM_TWINS`` (substitute the Tier 1b twins through the preloaded interposer, which
needs ``LD_PRELOAD`` and ``KVGUARD_INTERPOSER_CACHE`` in its environment).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Any

ENTRY_POINT_GROUP = "vllm.general_plugins"
ENTRY_POINT_NAME = "kvguard"
ENTRY_POINT_VALUE = "kvguard.adapters.vllm.plugin:register"
DIST_NAME = "kvguard_vllm_plugin"
DIST_VERSION = "0.0.1"
CORE_MODULE = "vllm.v1.engine.core"
ENV_OUT = "KVGUARD_VLLM_OUT"
ENV_GPU_VALIDATE = "KVGUARD_VLLM_GPU_VALIDATE"
ENV_AUDIT_EVERY = "KVGUARD_VLLM_AUDIT_EVERY"
ENV_TABLES = "KVGUARD_VLLM_TABLES"
ENV_CHECKER = "KVGUARD_VLLM_CHECKER"
ENV_FLUSH_EVERY = "KVGUARD_VLLM_FLUSH_EVERY"
ENV_TWINS = "KVGUARD_VLLM_TWINS"
ENV_TAGS = "KVGUARD_VLLM_TAGS"
ENV_NVTX = "KVGUARD_VLLM_NVTX"
ENV_PROFILE = "KVGUARD_VLLM_PROFILE"
"""A path: the attachment's entry points run under cProfile for the steps in
``KVGUARD_VLLM_PROFILE_STEPS`` (``lo:hi``, default 100:400) and the stats are written there."""
ENV_PROFILE_STEPS = "KVGUARD_VLLM_PROFILE_STEPS"
STEP_METHODS = ("step", "step_with_batch_queue")
REPORT_FLUSHES = 3
"""Record writes triggered by reports (the first ones); the periodic flush covers the rest."""

log = logging.getLogger("kvguard.vllm.plugin")


def dist_info(directory: str) -> str:
    """Write the ``*.dist-info`` that declares the entry point; returns the directory to put on
    ``PYTHONPATH`` (``directory`` itself)."""
    d = os.path.join(directory, f"{DIST_NAME}-{DIST_VERSION}.dist-info")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "METADATA"), "w", encoding="utf-8") as fh:
        fh.write(
            "Metadata-Version: 2.1\n"
            f"Name: {DIST_NAME.replace('_', '-')}\n"
            f"Version: {DIST_VERSION}\n"
            "Summary: KVGuard vLLM plugin entry point (generated; see kvguard/adapters/vllm/plugin.py)\n"
        )
    with open(os.path.join(d, "entry_points.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"[{ENTRY_POINT_GROUP}]\n{ENTRY_POINT_NAME} = {ENTRY_POINT_VALUE}\n")
    return directory


def _shadow_bytes(att: Any) -> dict[str, Any]:
    """The attachment's memory: the host shadow (ledger and snapshot-cache containers, deep size)
    and the device tiers (validator mirrors and buffers, tag-tier tags)."""
    from kvguard.core.memsize import deep_size

    seen: set[int] = set()
    ledger = att.ledger.memory_bytes(seen)
    caches: dict[str, int] = {}
    for gid, c in (getattr(att, "caches", None) or {}).items():
        own = [c.listed, c.changed, c.gone, c.facts, c.cover, c.dirty]
        req = {k: (v[1], v[3], v[4]) for k, v in c._req.items()}
        caches[f"snapshot_cache_{gid}"] = deep_size(own, seen) + deep_size(req, seen)
    validators = getattr(att, "validators", None)
    mirrors = getattr(att, "mirrors", None) or {}
    tags = getattr(att, "tags", None)
    device = {
        "mirror": sum(int(getattr(m, "nbytes", 0) or 0) for m in mirrors.values()),
        "validators": (
            int(validators.device_bytes())
            if validators is not None and hasattr(validators, "device_bytes")
            else 0
        ),
        "tags": int((tags.stats() or {}).get("tag_bytes", 0)) if tags is not None else 0,
    }
    return {
        "host_ledger": ledger,
        "host_caches": caches,
        "host_total": ledger["total"] + sum(caches.values()),
        "device": device,
        "device_total": sum(device.values()),
    }


class PluginState:
    """What the plugin holds in the engine-core process: the attachment once made, its reporter,
    the record path and the step and flush counters."""

    def __init__(self, env: dict[str, str] | None = None):
        env = os.environ if env is None else env
        self.out = env.get(ENV_OUT) or None
        self.gpu_validate = env.get(ENV_GPU_VALIDATE, "0") == "1"
        self.tags = env.get(ENV_TAGS, "0") == "1"
        self.nvtx = env.get(ENV_NVTX, "0") == "1"
        self.profile = env.get(ENV_PROFILE) or None
        lo, _, hi = env.get(ENV_PROFILE_STEPS, "100:400").partition(":")
        self.profile_steps = (int(lo or 100), int(hi or 400))
        self.audit_every = int(env.get(ENV_AUDIT_EVERY, "64"))
        self.tables = env.get(ENV_TABLES, "device")
        self.checker = env.get(ENV_CHECKER, "inline")
        self.flush_every = max(1, int(env.get(ENV_FLUSH_EVERY, "200")))
        self.twins = env.get(ENV_TWINS, "0") == "1"
        self.interposer: Any = None
        self.publisher: Any = None
        self.twins_error: str | None = None
        self.attachment: Any = None
        self.reporter: Any = None
        self.attempted = False
        self.attach_error: str | None = None
        self.attached_at = time.time()
        self.flushes = 0
        self.last_flushed_step = 0
        self.idle_flushed_at: int | None = None
        self.report_flushes = 0

    # ------------------------------------------------------------------ twins
    def enable_twins(self) -> None:
        """Switch the preloaded interposer to substitution (before the engine loads its
        kernels); a missing library is recorded and the engine runs without twins."""
        try:
            from .twins import enable_substitution

            self.interposer = enable_substitution()
        except Exception as exc:
            self.twins_error = f"{type(exc).__name__}: {exc}"[:400]
            log.warning("kvguard: twins requested but not enabled: %s", self.twins_error)

    def _publish_twins(self) -> None:
        if not self.twins or self.interposer is None or self.attachment is None:
            return
        try:
            from .twins import DescriptorPublisher

            att = self.attachment
            self.publisher = DescriptorPublisher(
                self.interposer, att.block_size, att.geometry.num_blocks, max_rows=4096
            )
            att.after_prepare_attn = self.publisher
        except Exception as exc:
            self.twins_error = f"{type(exc).__name__}: {exc}"[:400]
            log.exception("kvguard: the twin publisher could not be installed")
            self.publisher = None

    # ------------------------------------------------------------------ attaching
    def ensure_attached(self, engine: Any) -> None:
        if self.attempted:
            return
        self.attempted = True
        try:
            from kvguard.core import Reporter

            from .attach import attach

            # The first reports are written out as they happen: an engine that dies on the
            # condition the report describes never reaches its shutdown flush.
            self.reporter = Reporter(fail_fast=False, sink=self._on_report)
            self.attachment = attach(
                engine,
                self.reporter,
                gpu_validate=self.gpu_validate,
                tags=self.tags,
                nvtx=self.nvtx,
                audit_every=self.audit_every,
                tables=self.tables,
                checker=self.checker,
                profile=self.profile,
                profile_steps=self.profile_steps,
            )
            self.attached_at = time.time()
            log.info(
                "kvguard attached to the engine core (pid %d, gpu_validate=%s, tables=%s)",
                os.getpid(),
                self.gpu_validate,
                self.tables,
            )
        except Exception as exc:
            # Never break the engine: record why and leave it unmodified.
            self.attach_error = f"{type(exc).__name__}: {exc}"[:400]
            log.exception("kvguard could not attach to the engine core")
            self.attachment = None
        self._publish_twins()
        self.flush("attach")

    def _on_report(self, report: Any) -> None:
        if self.report_flushes < REPORT_FLUSHES:
            self.report_flushes += 1
            self.flush("report")

    # ------------------------------------------------------------------ the record
    def record(self) -> dict[str, Any]:
        att, reporter = self.attachment, self.reporter
        if att is not None:
            try:
                shadow_bytes = _shadow_bytes(att)
            except Exception as exc:  # the record must still be written
                shadow_bytes = {"error": repr(exc)}
        else:
            shadow_bytes = None
        rec: dict[str, Any] = {
            "plugin": ENTRY_POINT_VALUE,
            "pid": os.getpid(),
            "attached": att is not None,
            "attach_error": self.attach_error,
            "gpu_validate": self.gpu_validate,
            "tags": self.tags,
            "nvtx": self.nvtx,
            "audit_every": self.audit_every,
            "tables": self.tables,
            "checker": self.checker,
            "twins": self.twins,
            "twins_error": self.twins_error,
            "flushes": self.flushes,
        }
        if att is None:
            return rec
        steps = max(att.seq, 1)
        timings = dict(att.timings)
        gpu_reports = [r for r in reporter.reports if (r.site or "").startswith("gpu-validate:")]
        tag_reports = [r for r in reporter.reports if (r.site or "").startswith("gpu-tags:")]
        rec.update(
            steps=att.seq,
            geometry={"num_blocks": att.geometry.num_blocks, "block_size": att.block_size},
            adapter_stats=dict(att.stats),
            nvtx=att.nvtx.record() if getattr(att, "nvtx", None) is not None else None,
            adapter_timings_s={k: round(v, 4) for k, v in timings.items()},
            adapter_host_ms_per_step=round(
                1000 * sum(v for k, v in timings.items() if k.endswith("_s")) / steps, 3
            ),
            report_kinds=dict(Counter(r.kind for r in reporter.reports)),
            device_report_kinds=dict(Counter(r.kind for r in gpu_reports)),
            tag_report_kinds=dict(Counter(r.kind for r in tag_reports)),
            host_report_kinds=dict(
                Counter(
                    r.kind
                    for r in reporter.reports
                    if r not in gpu_reports and r not in tag_reports
                )
            ),
            reports_first=[str(r) for r in reporter.reports[:20]],
            reports_last=[str(r) for r in reporter.reports[-20:]],
            violations=len(reporter.violations()),
        )
        rec["shadow_bytes"] = shadow_bytes
        if offload_events := getattr(att, "offload_events", None):
            rec["offload_events"] = list(offload_events)
        if dflash_events := getattr(att, "dflash_events", None):
            rec["dflash_events"] = list(dflash_events)
        if standalone_events := getattr(att, "standalone_events", None):
            rec["standalone_events"] = list(standalone_events)
        if mtp_events := getattr(att, "mtp_events", None):
            rec["mtp_events"] = list(mtp_events)
        if self.publisher is not None:
            rec.update(self.publisher.summary())
        worker = getattr(att, "worker", None)
        if worker is not None:
            rec["checker_stats"] = worker.stats()
        if att.validators is not None:
            rec["validator_gpu_us_per_step"] = round(1000 * att.validators.gpu_ms_total / steps, 2)
            rec["validator_host_us_per_step"] = round(
                1e6 * timings.get("gpu_validate_host_s", 0.0) / steps, 1
            )
        return rec

    def flush(self, reason: str) -> None:
        if not self.out:
            return
        try:
            rec = self.record()
            rec["flush_reason"] = reason
            rec["written_at"] = time.time()
            os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
            tmp = f"{self.out}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, default=str)
            os.replace(tmp, self.out)
            self.flushes += 1
        except Exception:
            log.exception("kvguard could not write its record to %s", self.out)

    def after_step(self, executed: bool | None = None) -> None:
        """After an engine step: the periodic flush plus one flush per idle period, so the record
        covers a finished workload even when the process is terminated instead of shut down."""
        att = self.attachment
        if att is None:
            return
        if executed is False:
            if self.idle_flushed_at != att.seq:  # once per idle period
                self.idle_flushed_at = att.seq
                self.flush("idle")
            return
        if att.seq - self.last_flushed_step >= self.flush_every:
            self.last_flushed_step = att.seq
            self.flush("periodic")

    def shutdown(self) -> None:
        att = self.attachment
        if att is not None:
            try:
                att.finish()
            except Exception:
                log.exception("kvguard: finishing the attachment failed")
        if self.publisher is not None:
            try:
                self.publisher.drain()
                self.reporter.extend(self.publisher.reports())
            except Exception:
                log.exception("kvguard: draining the twin publisher failed")
        self.flush("shutdown")


_STATE: PluginState | None = None


def install(core_cls: Any, state: PluginState | None = None) -> PluginState:
    """Wrap the engine-core class's step methods and ``shutdown`` (idempotent per class)."""
    global _STATE
    state = state or _STATE or PluginState()
    _STATE = state
    if state.twins and state.interposer is None and state.twins_error is None:
        state.enable_twins()
    if getattr(core_cls, "__kvguard_plugin__", None) is state:
        return state
    for name in STEP_METHODS:
        orig = getattr(core_cls, name, None)
        if orig is None:
            continue

        def make(orig_fn):
            @functools.wraps(orig_fn)
            def step(self, *args, **kwargs):
                state.ensure_attached(self)
                out = orig_fn(self, *args, **kwargs)
                # Both step methods return (outputs, model_executed).
                executed = out[1] if isinstance(out, tuple) and len(out) == 2 else None
                state.after_step(executed)
                return out

            return step

        setattr(core_cls, name, make(orig))
    orig_shutdown = getattr(core_cls, "shutdown", None)
    if orig_shutdown is not None:

        @functools.wraps(orig_shutdown)
        def shutdown(self, *args, **kwargs):
            state.shutdown()  # guarded inside: it cannot raise into the engine's shutdown
            return orig_shutdown(self, *args, **kwargs)

        core_cls.shutdown = shutdown
    core_cls.__kvguard_plugin__ = state
    return state


def register() -> None:
    """The entry point: wrap ``EngineCore`` once the core module is importable. Loaded in every
    vLLM process; in a worker process (no engine core) it wraps the class too, harmlessly: no
    step runs there."""
    try:
        module = sys.modules.get(CORE_MODULE)
        if module is None:
            import importlib

            module = importlib.import_module(CORE_MODULE)
        install(module.EngineCore)
    except Exception:
        log.exception("kvguard vLLM plugin: could not wrap %s.EngineCore", CORE_MODULE)


__all__ = [
    "DIST_NAME",
    "ENTRY_POINT_GROUP",
    "ENTRY_POINT_NAME",
    "ENTRY_POINT_VALUE",
    "ENV_TAGS",
    "ENV_TWINS",
    "PluginState",
    "dist_info",
    "install",
    "register",
]
