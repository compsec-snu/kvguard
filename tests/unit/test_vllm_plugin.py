"""The vLLM plugin (``kvguard.adapters.vllm.plugin``) without vLLM: the generated entry point,
the class-level wrap of a synthetic engine core, the lazy attachment and the record."""

from __future__ import annotations

import json
import sys
import types
from importlib.metadata import distributions

import pytest
from kvguard.adapters.vllm import plugin


def test_dist_info_declares_the_entry_point(tmp_path):
    d = plugin.dist_info(str(tmp_path))
    eps = [
        ep
        for dist in distributions(path=[d])
        for ep in dist.entry_points
        if ep.group == plugin.ENTRY_POINT_GROUP
    ]
    assert [(ep.name, ep.value) for ep in eps] == [
        (plugin.ENTRY_POINT_NAME, plugin.ENTRY_POINT_VALUE)
    ]
    assert plugin.dist_info(str(tmp_path)) == d, "idempotent"


class FakeAttachment:
    def __init__(self, engine, reporter, **options):
        self.engine, self.reporter, self.options = engine, reporter, options
        self.seq = 0
        self.timings = {"snapshot_s": 0.0, "dispatch_s": 0.0}
        self.stats = {}
        self.block_size = 16
        self.geometry = types.SimpleNamespace(num_blocks=100)
        self.validators = None
        self.finished = 0

    def step(self):
        self.seq += 1
        self.timings["snapshot_s"] += 0.001

    def finish(self):
        self.finished += 1


class FakeCore:
    def __init__(self):
        self.steps = 0
        self.shutdowns = 0

    idle = False

    def step(self):
        self.steps += 1
        if self.idle:
            return {}, False
        if self.__kvguard_plugin__.attachment is not None:
            self.__kvguard_plugin__.attachment.step()  # stands for the wrapped hooks firing
        return {"out": self.steps}, True

    def step_with_batch_queue(self):
        return self.step()

    def shutdown(self):
        self.shutdowns += 1


def test_install_attaches_at_the_first_step_and_records_at_shutdown(tmp_path, monkeypatch):
    import kvguard.adapters.vllm.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach", FakeAttachment)
    out = tmp_path / "core.json"
    state = plugin.PluginState(
        {plugin.ENV_OUT: str(out), plugin.ENV_GPU_VALIDATE: "1", plugin.ENV_FLUSH_EVERY: "3"}
    )

    class Core(FakeCore):
        pass

    assert plugin.install(Core, state) is state
    assert plugin.install(Core, state) is state, "idempotent per class"
    core = Core()
    assert core.step() == ({"out": 1}, True), "the original step's result is returned"
    att = state.attachment
    assert isinstance(att, FakeAttachment) and att.engine is core
    assert att.options == {
        "gpu_validate": True,
        "tags": False,
        "nvtx": False,
        "audit_every": 64,
        "tables": "device",
        "checker": "inline",
        "profile": None,
        "profile_steps": (100, 400),
    }
    assert json.loads(out.read_text())["flush_reason"] == "attach"
    for _ in range(5):
        core.step_with_batch_queue()
    rec = json.loads(out.read_text())
    assert rec["flush_reason"] == "periodic" and rec["steps"] in (3, 6)
    core.idle = True
    core.step()
    core.step()
    rec = json.loads(out.read_text())
    assert rec["flush_reason"] == "idle" and rec["steps"] == 6, "one flush per idle period"
    flushes = rec["flushes"]
    core.shutdown()
    assert core.shutdowns == 1 and att.finished == 1
    rec = json.loads(out.read_text())
    assert rec["flush_reason"] == "shutdown" and rec["attached"] and rec["steps"] == 6
    assert rec["flushes"] == flushes + 1
    assert rec["adapter_host_ms_per_step"] == pytest.approx(1.0)
    assert rec["report_kinds"] == {} and rec["violations"] == 0
    assert rec["geometry"] == {"num_blocks": 100, "block_size": 16}


def test_an_attach_failure_leaves_the_engine_running(tmp_path, monkeypatch):
    import kvguard.adapters.vllm.attach as attach_mod

    def failing(engine, reporter, **options):
        raise RuntimeError("no such engine")

    monkeypatch.setattr(attach_mod, "attach", failing)
    out = tmp_path / "core.json"
    state = plugin.PluginState({plugin.ENV_OUT: str(out)})

    class Core(FakeCore):
        pass

    plugin.install(Core, state)
    core = Core()
    assert core.step() == ({"out": 1}, True) and core.step() == ({"out": 2}, True)
    assert state.attachment is None and "RuntimeError" in state.attach_error
    core.shutdown()
    rec = json.loads(out.read_text())
    assert rec["attached"] is False and rec["attach_error"].startswith("RuntimeError")


def test_register_wraps_the_core_module_class(monkeypatch):
    module = types.ModuleType(plugin.CORE_MODULE)

    class EngineCore(FakeCore):
        pass

    module.EngineCore = EngineCore
    monkeypatch.setitem(sys.modules, plugin.CORE_MODULE, module)
    monkeypatch.setattr(plugin, "_STATE", None)
    monkeypatch.setenv(plugin.ENV_OUT, "")
    plugin.register()
    assert isinstance(EngineCore.__kvguard_plugin__, plugin.PluginState)
    assert EngineCore.step.__wrapped__ is FakeCore.step
    plugin.register()
    assert EngineCore.step.__wrapped__ is FakeCore.step, "wrapped once"


def test_twins_without_the_preloaded_interposer_are_recorded_and_the_engine_runs(
    tmp_path, monkeypatch
):
    import kvguard.adapters.vllm.attach as attach_mod

    monkeypatch.setattr(attach_mod, "attach", FakeAttachment)
    out = tmp_path / "core.json"
    state = plugin.PluginState({plugin.ENV_OUT: str(out), plugin.ENV_TWINS: "1"})

    class Core(FakeCore):
        pass

    plugin.install(Core, state)
    assert state.twins and state.interposer is None
    assert "preloaded" in (state.twins_error or ""), state.twins_error
    core = Core()
    assert core.step() == ({"out": 1}, True)
    assert state.attachment is not None and state.publisher is None
    core.shutdown()
    rec = json.loads(out.read_text())
    assert rec["attached"] and rec["twins"] and rec["twins_error"] == state.twins_error
    assert "twin_launches" not in rec and rec["report_kinds"] == {}


def test_the_first_reports_flush_the_record(tmp_path, monkeypatch):
    import kvguard.adapters.vllm.attach as attach_mod
    from kvguard.core import OutcomeKind, Policy, Report

    monkeypatch.setattr(attach_mod, "attach", FakeAttachment)
    out = tmp_path / "core.json"
    state = plugin.PluginState({plugin.ENV_OUT: str(out), plugin.ENV_FLUSH_EVERY: "1000"})

    class Core(FakeCore):
        pass

    plugin.install(Core, state)
    core = Core()
    core.step()
    flushes = json.loads(out.read_text())["flushes"]
    for i in range(plugin.REPORT_FLUSHES + 2):
        state.reporter.emit(
            Report(OutcomeKind.VIOLATION, "mapping_to_free_block", "x", policy=Policy.P2, block=i)
        )
    # The record on disk is the one written at the last report flush: it holds the reports
    # emitted up to then.
    rec = json.loads(out.read_text())
    assert state.report_flushes == plugin.REPORT_FLUSHES
    assert rec["flush_reason"] == "report" and rec["flushes"] == flushes + plugin.REPORT_FLUSHES
    assert rec["report_kinds"] == {"mapping_to_free_block": plugin.REPORT_FLUSHES}
