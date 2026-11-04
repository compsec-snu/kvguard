"""Import-time attachment of the SGLang adapter.

SGLang runs its scheduler in a spawned subprocess that imports a ``sitecustomize`` found on
``PYTHONPATH``; the injected ``sitecustomize`` calls :func:`install` when ``KVGUARD_SGLANG_ATTACH``
is set. :func:`install` registers a post-import hook on ``sglang.srt.managers.scheduler`` that,
once the module executes, replaces ``Scheduler.run_event_loop`` on the imported class object (in
memory only, no file modified) with a wrapper that attaches the core to the live scheduler and
then runs the original loop. Everything else the adapter installs is an instance attribute on
live objects (see ``attach.py``).

When ``KVGUARD_SGLANG_TWINS`` is set, :func:`prepare_twins` publishes the inert Tier 1b descriptor
to the preloaded interposer before CUDA-graph capture, so twins captured into graphs carry a
descriptor memcpy node; the checks stay inert until the attachment publishes the real descriptor.

Imports neither torch nor SGLang, so it is unit-testable with a synthetic module.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.abc
import logging
import os
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

SCHEDULER_MODULE = "sglang.srt.managers.scheduler"
ENV_ATTACH = "KVGUARD_SGLANG_ATTACH"
ENV_TWINS = "KVGUARD_SGLANG_TWINS"

log = logging.getLogger("kvguard.sglang")


class PostImportHook(importlib.abc.MetaPathFinder):
    """Run ``callback(module)`` once, right after ``module_name`` has been executed."""

    def __init__(self, module_name: str, callback: Callable[[ModuleType], None]):
        self.module_name = module_name
        self.callback = callback
        self.done = False
        self.error: BaseException | None = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.module_name or self.done:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader
        orig_exec = loader.exec_module
        hook = self

        def exec_module(module):
            orig_exec(module)
            hook._fire(module)

        loader.exec_module = exec_module
        return spec

    def _fire(self, module: ModuleType) -> None:
        if self.done:
            return
        self.done = True
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)
        try:
            self.callback(module)
        except Exception as exc:
            self.error = exc
            log.exception("post-import hook for %s failed", self.module_name)

    def install(self) -> PostImportHook:
        """Register the finder, or fire immediately when the module is already imported."""
        module = sys.modules.get(self.module_name)
        if module is not None:
            self._fire(module)
        elif self not in sys.meta_path:
            sys.meta_path.insert(0, self)
        return self


def wrap_run_event_loop(
    scheduler_cls: type, attach: Callable[[Any], Any], flush: Callable[[Any, str], None]
) -> bool:
    """Replace ``scheduler_cls.run_event_loop`` with a wrapper that calls ``attach(scheduler)``
    before the loop and ``flush(attachment, reason)`` when it exits. Idempotent; returns whether
    the patch was applied now. ``attach`` must not raise: it returns the attachment or ``None``."""
    orig = scheduler_cls.run_event_loop
    if getattr(orig, "_kvguard_wrapped", False):
        return False

    @functools.wraps(orig)
    def run_event_loop(self, *args, **kwargs):
        attachment = attach(self)
        try:
            return orig(self, *args, **kwargs)
        finally:
            if attachment is not None:
                flush(attachment, "loop_exit")

    run_event_loop._kvguard_wrapped = True  # type: ignore[attr-defined]
    scheduler_cls.run_event_loop = run_event_loop
    return True


def _patch_scheduler_module(module: ModuleType) -> None:
    from .attach import attach_from_env, flush_attachment

    applied = wrap_run_event_loop(module.Scheduler, attach_from_env, flush_attachment)
    log.info(
        "Scheduler.run_event_loop %s (pid %d)",
        "wrapped" if applied else "already wrapped",
        os.getpid(),
    )


_HOOK: PostImportHook | None = None
TWINS_PREP: dict[str, Any] | None = None
"""What :func:`prepare_twins` did in this process (read into the attachment's record)."""


def prepare_twins(force: bool = False) -> dict[str, Any] | None:
    """Publish the inert Tier 1b descriptor to the preloaded interposer when
    ``KVGUARD_SGLANG_TWINS`` is set (or ``force``), once per process, and record the outcome.
    Returns ``None`` when the variable is unset.
    """
    global TWINS_PREP
    if not force and not os.environ.get(ENV_TWINS):
        return None
    if TWINS_PREP is not None:
        return TWINS_PREP
    rec: dict[str, Any] = {
        "pid": os.getpid(),
        "interposer": False,
        "inert_descriptor_published": False,
        "mode_at_start": None,
        "error": None,
    }
    try:
        from kvguard.gpu import interposer as ip_mod

        if ip_mod.available():
            ip = ip_mod.Interposer()
            ip.set_descriptor(
                ip_mod.tier1b_descriptor(
                    counter_ptr=0,
                    record_ptr=0,
                    table_row_bytes=4,
                    table_rows=0,
                    row_len_ptr=0,
                    block_size=1,
                    num_blocks=1,
                    flat_slot_map=True,
                    enabled=False,
                )
            )
            rec.update(interposer=True, inert_descriptor_published=True, mode_at_start=ip.mode())
            log.info(
                "inert twin descriptor published (pid %d, mode %s)",
                os.getpid(),
                rec["mode_at_start"],
            )
    except Exception as exc:
        rec["error"] = repr(exc)
        log.exception("twin preparation failed")
    TWINS_PREP = rec
    return rec


def install(force: bool = False) -> PostImportHook | None:
    """Install the post-import hook when ``KVGUARD_SGLANG_ATTACH`` is set (or ``force``), and
    prepare the twins when ``KVGUARD_SGLANG_TWINS`` is set."""
    global _HOOK
    if not force and not os.environ.get(ENV_ATTACH):
        return None
    if _HOOK is None:
        _HOOK = PostImportHook(SCHEDULER_MODULE, _patch_scheduler_module).install()
    prepare_twins()
    return _HOOK


__all__ = [
    "ENV_ATTACH",
    "ENV_TWINS",
    "SCHEDULER_MODULE",
    "PostImportHook",
    "install",
    "prepare_twins",
    "wrap_run_event_loop",
]
