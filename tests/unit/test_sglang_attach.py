"""The SGLang adapter's import-time attachment, without SGLang or torch.

``hook.PostImportHook`` is exercised on a synthetic package written to a temporary directory:
the callback must run exactly once, after the module body, and the ``Scheduler.run_event_loop``
patch must attach before the loop and flush when it exits. ``attach.Fault`` and the record
helpers are plain Python and are checked for their contracts.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import threading
from collections import Counter, deque
from types import SimpleNamespace

import pytest
from kvguard.adapters.sglang import attach as attach_mod
from kvguard.adapters.sglang.hook import PostImportHook, wrap_run_event_loop
from kvguard.core import BlockState, Reporter

PKG = "kvg_fake_sched_pkg"


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    pkg = tmp_path / PKG
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "scheduler.py").write_text(
        textwrap.dedent(
            """
            EXECUTED = []

            class Scheduler:
                def __init__(self):
                    self.calls = []

                def run_event_loop(self, n=1):
                    self.calls.append(("loop", n))
                    return "done"

            EXECUTED.append("body")
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == PKG or m.startswith(PKG + ".")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield f"{PKG}.scheduler"
    for name in [m for m in sys.modules if m == PKG or m.startswith(PKG + ".")]:
        sys.modules.pop(name, None)
    for finder in list(sys.meta_path):
        if isinstance(finder, PostImportHook):
            sys.meta_path.remove(finder)


def test_post_import_hook_fires_once_after_module_body(fake_package):
    seen = []
    hook = PostImportHook(fake_package, lambda m: seen.append(list(m.EXECUTED))).install()
    assert hook in sys.meta_path
    mod = importlib.import_module(fake_package)
    assert seen == [["body"]], "callback runs after the module body, once"
    assert hook.done and hook not in sys.meta_path
    importlib.reload(mod)
    assert seen == [["body"]], "a reload does not fire the hook again"
    assert hook.error is None


def test_post_import_hook_fires_immediately_when_already_imported(fake_package):
    importlib.import_module(fake_package)
    seen = []
    hook = PostImportHook(fake_package, lambda m: seen.append(m.__name__)).install()
    assert seen == [fake_package] and hook not in sys.meta_path


def test_post_import_hook_swallows_callback_errors(fake_package):
    def bad(_module):
        raise RuntimeError("boom")

    hook = PostImportHook(fake_package, bad).install()
    mod = importlib.import_module(fake_package)  # the framework's import must still succeed
    assert mod.EXECUTED == ["body"]
    assert isinstance(hook.error, RuntimeError)


def test_unified_kv_locations_inverse_map_physical_pages_to_virtual_tokens():
    att = object.__new__(attach_mod.SglangAttachment)
    att.facts = {"unified_allocator": True}
    att.page_size = 16
    att.allocator = SimpleNamespace(
        full_attn_allocator=SimpleNamespace(ids=SimpleNamespace(physical_to_virtual=[-1, 7, 3]))
    )
    att.stats = Counter()

    assert att._logical_kv_slots([19, 32, -1, 65]) == [115, 48, -1, -1]
    assert att.stats == {"unified_write_slots": 4, "unified_write_slots_relocated": 2}


def test_adaptive_verify_width_uses_the_concrete_positive_width_below_capture_maximum():
    static = {"speculative_num_draft_tokens": 4, "speculative_adaptive": False}
    adaptive = {"speculative_num_draft_tokens": 4, "speculative_adaptive": True}

    assert attach_mod._verify_width_supported(static, 4)
    assert not attach_mod._verify_width_supported(static, 1)
    assert attach_mod._verify_width_supported(adaptive, 1)
    assert attach_mod._verify_width_supported(adaptive, 4)
    assert not attach_mod._verify_width_supported(adaptive, 0)
    assert not attach_mod._verify_width_supported(adaptive, 5)


def test_adaptive_runtime_graphs_include_every_distinct_candidate_state():
    initial_decode = object()
    initial_extend = object()
    zero_decode = object()
    zero_extend = object()
    controller = SimpleNamespace(
        _states={
            3: SimpleNamespace(
                cuda_graph_runner=initial_decode,
                cuda_graph_runner_for_draft_extend=initial_extend,
            ),
            0: SimpleNamespace(
                cuda_graph_runner=zero_decode,
                cuda_graph_runner_for_draft_extend=zero_extend,
            ),
        }
    )
    worker = SimpleNamespace(
        cuda_graph_runner=initial_decode,
        cuda_graph_runner_for_draft_extend=initial_extend,
    )

    assert attach_mod._adaptive_runtime_graphs(worker, controller, "cuda_graph_runner") == (
        initial_decode,
        zero_decode,
    )
    assert attach_mod._adaptive_runtime_graphs(
        worker, controller, "cuda_graph_runner_for_draft_extend"
    ) == (initial_extend, zero_extend)


def test_adaptive_backend_selection_follows_the_live_runtime_state():
    startup = object()
    active = object()
    att = object.__new__(attach_mod.SglangAttachment)
    att.pdmux = False
    att.facts = {"speculative_adaptive": True}
    att.backend = startup
    att.model_runner = SimpleNamespace(attn_backend=active)

    assert att._backend_for(SimpleNamespace(forward_mode=None)) is active

    att.facts["speculative_adaptive"] = False
    assert att._backend_for(SimpleNamespace(forward_mode=None)) is startup


def test_wrap_run_event_loop_attaches_before_and_flushes_after(fake_package):
    mod = importlib.import_module(fake_package)
    events = []

    def attach(scheduler):
        events.append(("attach", scheduler.calls[:]))
        return "attachment"

    def flush(attachment, reason):
        events.append(("flush", attachment, reason))

    assert wrap_run_event_loop(mod.Scheduler, attach, flush) is True
    assert wrap_run_event_loop(mod.Scheduler, attach, flush) is False, "idempotent"
    s = mod.Scheduler()
    assert s.run_event_loop(n=3) == "done"
    assert events == [("attach", []), ("flush", "attachment", "loop_exit")]
    assert s.calls == [("loop", 3)]


def test_wrap_run_event_loop_skips_flush_when_not_attached(fake_package):
    mod = importlib.import_module(fake_package)
    events = []
    wrap_run_event_loop(mod.Scheduler, lambda s: None, lambda a, r: events.append(r))
    mod.Scheduler().run_event_loop()
    assert events == []


def test_fault_kinds_and_arming():
    with pytest.raises(ValueError):
        attach_mod.Fault("nonsense")
    f = attach_mod.Fault("early_free", step=5)

    class Att:
        seq = 3

    assert f._armed(Att()) is False
    Att.seq = 5
    assert f._armed(Att()) is True
    f.done = True
    assert f._armed(Att()) is False
    assert attach_mod.Fault("external_linker_load_early_free").kind == (
        "external_linker_load_early_free"
    )
    assert attach_mod.Fault("external_linker_store_early_free").kind == (
        "external_linker_store_early_free"
    )
    assert attach_mod.Fault("swa_shadow_early_free").kind == "swa_shadow_early_free"


def test_transfer_ready_hook_runs_once_after_the_framework_orders_the_transfer():
    calls = []

    class Ready:
        def synchronize(self):
            calls.append("synchronize")

        def wait(self, stream):
            calls.append(("wait", stream))

    hook = attach_mod._TransferReadyHook(Ready(), lambda: calls.append("callback"))
    hook.synchronize()
    hook.wait(17)

    assert calls == ["synchronize", "callback", ("wait", 17)]


def test_hicache_cache_mode_is_admitted_only_for_the_routed_unified_hybrid():
    supported = attach_mod._hicache_hybrid_host_mode_supported

    assert supported(hybrid=False, unified_hybrid=False, host_memory_mode="cache")
    assert supported(hybrid=True, unified_hybrid=True, host_memory_mode="cache")
    assert supported(hybrid=True, unified_hybrid=False, host_memory_mode="buffer_only")
    assert not supported(hybrid=True, unified_hybrid=False, host_memory_mode="cache")


def test_request_incarnations_distinguish_reused_public_ids():
    class Req:
        def __init__(self, rid):
            self.rid = rid

    ids = attach_mod.RequestIncarnations()
    old = Req("reused")
    replacement = Req("reused")
    other = Req("other")

    assert ids(old) == ids(old)
    assert ids(old) != ids(replacement)
    assert ids(old).startswith("reused#kvguard-")
    assert ids(replacement).startswith("reused#kvguard-")
    assert ids(other).startswith("other#kvguard-")


def test_request_incarnations_renew_an_in_place_retry():
    class Req:
        rid = "retry"

    req = Req()
    ids = attach_mod.RequestIncarnations()
    first = ids(req)
    second = ids.renew(req)

    assert first != second
    assert ids(req) == second


def test_streaming_turn_and_retained_session_share_one_owner():
    class Req:
        def __init__(self, rid, session):
            self.rid = rid
            self.session = session

    att = object.__new__(attach_mod.SglangAttachment)
    att.request_incarnations = attach_mod.RequestIncarnations()
    session = SimpleNamespace(session_id="conversation", streaming=True)
    first_turn = Req("turn-1", session)
    second_turn = Req("turn-2", session)
    ordinary = Req("ordinary", None)

    assert att._request_owner(first_turn) == "session:conversation"
    assert att._request_owner(second_turn) == "session:conversation"
    assert att._request_owner(ordinary).startswith("ordinary#kvguard-")


def test_fallback_environment_lists_packages():
    env = attach_mod.fallback_environment()
    assert set(env["packages"]) >= {"torch", "sglang"}
    assert env["python"]


def test_support_boundary_and_config_keys_are_exported():
    assert attach_mod.SUPPORT_BOUNDARY["framework"] == "sglang"
    assert "disable_overlap_schedule" in attach_mod.CONFIG_KEYS
    assert "max_total_tokens" in attach_mod.CONFIG_KEYS


def test_prefill_load_batch_wrapper_forwards_positional_runner_arguments():
    """The live eager runner passes pp_proxy_tensors positionally when prefill graphs are off."""

    calls = []

    class Runner:
        def load_batch(self, forward_batch, pp_proxy_tensors, *, marker=None):
            calls.append((forward_batch, pp_proxy_tensors, marker))
            return "static-batch"

    runner = Runner()
    observed = []
    attach_mod.wrap_static_batch_loader(runner, observed.append)
    assert runner.load_batch("forward", "proxy", marker=7) == "static-batch"
    assert calls == [("forward", "proxy", 7)]
    assert observed == ["static-batch"]


def test_multi_wrapper_descriptors_share_one_prefill_graph_static_batch():
    """Each table in a multi-wrapper prefill graph uses the same captured output locations."""
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 7
    att.last_static_fb = object()
    static_fb = att.last_static_fb
    att.timings = Counter()
    att.stats = Counter()
    att._last_dispatch = {}
    att._wrappers = lambda _fb, _graph: ["swa", "full"]
    seen = []

    def plan(_fb, graph, wrapper, _site, *, tag):
        seen.append((graph, wrapper, tag, att.last_static_fb))
        att.last_static_fb = None  # _plan consumes it after reading out_cache_loc.
        desc = SimpleNamespace(family="attention", dispatch_id=tag)
        return SimpleNamespace(descs=[desc], attn_first=True)

    att._plan = plan
    dispatched = []
    att._dispatch_one = lambda desc, dids: (
        dispatched.append(desc.dispatch_id),
        dids.append(desc.dispatch_id),
    )
    att._dispatch_state = lambda _fb, _dids: None
    att._record_plan = lambda _plan, _dids: None
    dids = att._descriptors(object(), SimpleNamespace(can_run_graph=True))

    assert [item[:3] for item in seen] == [
        (True, "swa", "-swa"),
        (True, "full", "-full1"),
    ]
    assert all(item[3] is static_fb for item in seen)
    assert att.last_static_fb is None
    assert dids == dispatched == ["-full1", "-swa"]


def test_pdmux_verify_graph_uses_the_active_stream_backend_metadata():
    class Mode:
        @staticmethod
        def is_target_verify():
            return True

        @staticmethod
        def is_decode_or_idle():
            return False

    active_wrapper = object()
    active = SimpleNamespace(prefill_cuda_graph_metadata={2: [active_wrapper]})
    construction = SimpleNamespace(prefill_cuda_graph_metadata={})
    graph = SimpleNamespace(
        bs=2,
        attn_backend=construction,
        _replay_attn_backend=lambda: active,
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.pdmux = True
    att.backend = SimpleNamespace(forward_metadata=object())
    att.model_runner = SimpleNamespace(decode_cuda_graph_runner=graph)

    assert attach_mod._replay_full_backend(graph) is active
    assert att._wrappers(SimpleNamespace(forward_mode=Mode()), graph=True) == [active_wrapper]


def test_pdmux_separate_swa_graph_uses_active_stream_write_locations():
    active = SimpleNamespace(cuda_graph_swa_out_cache_loc=[71, 72, 73, 74])
    construction = SimpleNamespace(cuda_graph_swa_out_cache_loc=[11, 12, 13, 14])
    graph = SimpleNamespace(
        bs=2,
        captured_req_width=1,
        attn_backend=construction,
        _replay_attn_backend=lambda: active,
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 8
    att.separate_swa = True
    att.last_static_fb = None
    att.timings = Counter()
    att.stats = Counter()
    att._last_dispatch = {}
    att.model_runner = SimpleNamespace(decode_cuda_graph_runner=graph)
    att._wrappers = lambda _fb, _graph: ["swa", "full"]
    observed = []

    def plan(_fb, graph_enabled, wrapper, _site, **kwargs):
        observed.append((graph_enabled, wrapper, kwargs.get("pool"), kwargs.get("loc_override")))
        return SimpleNamespace(descs=[])

    att._plan = plan
    att._dispatch_state = lambda _fb, _dids: None
    att._record_plan = lambda _plan, _dids: None

    assert att._descriptors(object(), SimpleNamespace(can_run_graph=True)) == []
    assert observed == [
        (True, "swa", attach_mod.SWA_POOL, [71, 72]),
        (True, "full", attach_mod.POOL, None),
    ]


def test_mla_wrapper_and_table_buffers_use_the_singular_metadata_contract():
    class Mode:
        @staticmethod
        def is_target_verify():
            return False

        @staticmethod
        def is_decode_or_idle():
            return True

        @staticmethod
        def is_extend():
            return False

    eager = object()
    graph = object()
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrapper=eager),
        decode_cuda_graph_metadata={4: graph},
    )
    wrapper = SimpleNamespace(
        _kv_indptr_buf=[0, 3],
        _kv_indices_buf=[11, 12, 13],
        _kv_len_arr_buf=[3],
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.facts = {"mla": True}
    att.pdmux = False
    att.backend = backend
    att.model_runner = SimpleNamespace(
        decode_cuda_graph_runner=SimpleNamespace(bs=4, _replay_attn_backend=lambda: backend)
    )

    fb = SimpleNamespace(forward_mode=Mode())
    assert att._wrappers(fb, graph=False) == [eager]
    assert att._wrappers(fb, graph=True) == [graph]
    assert att._table_buffers(wrapper) == ([0, 3], [11, 12, 13], [3])
    assert att._last_page_lens([99], 3) == [1, 1, 1]


def test_mla_graph_plan_uses_the_host_replan_indptr_not_the_stale_capture_buffer():
    class Mode:
        name = "DECODE"

        @staticmethod
        def is_target_verify():
            return False

        @staticmethod
        def is_decode_or_idle():
            return True

        @staticmethod
        def is_extend():
            return False

    wrapper = SimpleNamespace(
        _kv_indptr_buf=[0, 1],
        _kv_indices_buf=[10, 11, 13],
        _kv_len_arr_buf=[1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrapper=wrapper),
        decode_cuda_graph_metadata={1: wrapper},
        cuda_graph_kv_indptr_cpu=[0, 3],
    )
    graph = SimpleNamespace(
        bs=1,
        _replay_attn_backend=lambda: backend,
        buffer_registry=SimpleNamespace(),
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.facts = {"mla": True}
    att.pdmux = False
    att.backend = backend
    att.model_runner = SimpleNamespace(decode_cuda_graph_runner=graph)
    att.seq = 7
    att.page_size = 16
    att.stats = Counter()
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att._req_ids = lambda _fb, _n: ["request"]

    plan = att._plan(
        SimpleNamespace(
            forward_mode=Mode(),
            batch_size=1,
            seq_lens_cpu=[3],
            seq_lens=[3],
            out_cache_loc=[13],
        ),
        True,
        wrapper,
        "mla-graph",
    )

    assert plan.indptr == [0, 3]
    assert plan.indices == [10, 11, 13]
    attention = plan.descs[1]
    assert attention.mapping.table == [[10, 11, 13]]


class _DraftMode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_extend(self):
        return self.name == "EXTEND"

    def is_draft_extend_v2(self):
        return self.name == "DRAFT_EXTEND_V2"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"


class _RollbackRecorder:
    def __init__(self):
        self.calls = []
        self.invalidations = []
        self.copies = []

    def rollback(self, pool, rid, frontier, *, site):
        self.calls.append((pool, rid, frontier, site))

    def invalidate_positions(self, pool, rid, positions, *, site):
        self.invalidations.append((pool, rid, tuple(positions), site))

    def external_copy(self, copy, *, site):
        self.copies.append((copy, site))


def _draft_attachment():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 7
    att.rid_of_row = {4: "a", 5: "b"}
    att.ledger = _RollbackRecorder()
    att.facts = {"speculative_eagle_topk": 1, "speculative_num_steps": 3}
    att.speculative_dflash = False
    att.page_size = 1
    att._draft_tree_copy_rolled_back = set()
    att.stats = Counter()
    att.reporter = SimpleNamespace(emit=lambda report: None)
    att.cur_batch = SimpleNamespace(req_pool_indices=[4, 5])
    return att


def test_post_verify_graph_observation_distinguishes_warmup_replay_and_state_limit():
    warmup = SimpleNamespace(graph=None)
    replay = SimpleNamespace(graph=object())
    states = {
        ("a",): SimpleNamespace(turn=1, slots=[warmup, None]),
        ("b",): SimpleNamespace(turn=0, slots=[None, replay]),
    }
    graphs = SimpleNamespace(_graphs=SimpleNamespace(_states=states))

    assert attach_mod._post_verify_graph_observation(graphs, ("a",)) == ("warmup", 1, 2)
    assert attach_mod._post_verify_graph_observation(graphs, ("b",)) == ("graph", 1, 2)
    assert attach_mod._post_verify_graph_observation(graphs, ("missing",)) == (
        "state_limit",
        0,
        2,
    )


def test_dflash_draft_verify_describes_one_write_and_both_window_views():
    att = _draft_attachment()
    att.speculative_dflash = True
    att.page_size = 16
    att.facts = {"dflash_block_size": 2}
    swa = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4, 8],
        _paged_kv_indices_buf=list(range(100, 108)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    full = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 5, 12],
        _paged_kv_indices_buf=list(range(200, 212)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(prefill_wrappers=[swa, full]))
    fb = SimpleNamespace(
        forward_mode=_DraftMode("TARGET_VERIFY"),
        batch_size=2,
        req_pool_indices=[4, 5],
        seq_lens=[3, 5],
        positions=[3, 4, 5, 6],
        out_cache_loc=[13, 14, 25, 26],
    )

    (write, swa_attention, full_attention), kind, tokens = att._draft_descriptors(
        fb, backend, tag="-dflash"
    )

    assert (kind, tokens) == ("dflash_verify", 4)
    assert [row.writes for row in write.rows] == [range(3, 5), range(5, 7)]
    assert [row.reads for row in swa_attention.rows] == [range(1, 5), range(3, 7)]
    assert [row.reads for row in full_attention.rows] == [range(0, 5), range(0, 7)]
    assert att.ledger.calls == [
        (attach_mod.DRAFT_POOL, "a", 3, "dflash-draft-rollback@7"),
        (attach_mod.DRAFT_POOL, "b", 5, "dflash-draft-rollback@7"),
    ]


def test_dflash_compact_draft_maps_local_suffix_table_to_global_positions():
    att = _draft_attachment()
    att.speculative_dflash = True
    att.page_size = 16
    att.facts = {"dflash_block_size": 2, "dflash_compact_cache": True}
    swa = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 6, 14],
        _paged_kv_indices_buf=list(range(116, 122)) + list(range(224, 232)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    full = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 6, 14],
        _paged_kv_indices_buf=list(range(116, 122)) + list(range(224, 232)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(prefill_wrappers=[swa, full]))
    fb = SimpleNamespace(
        forward_mode=_DraftMode("TARGET_VERIFY"),
        batch_size=2,
        req_pool_indices=[4, 5],
        seq_lens=[4, 6],
        positions=[20, 21, 30, 31],
        out_cache_loc=[120, 121, 230, 231],
    )

    (write, swa_attention, full_attention), kind, tokens = att._draft_descriptors(
        fb, backend, tag="-dflash-compact"
    )

    assert (kind, tokens) == ("dflash_verify", 4)
    assert [row.writes for row in write.rows] == [range(20, 22), range(30, 32)]
    assert [row.reads for row in swa_attention.rows] == [range(16, 22), range(24, 32)]
    assert [row.reads for row in full_attention.rows] == [range(16, 22), range(24, 32)]
    assert swa_attention.mapping.first == [16, 24]
    assert full_attention.mapping.first == [16, 24]
    assert att.ledger.calls == [
        (attach_mod.DRAFT_POOL, "a", 20, "dflash-draft-rollback@7"),
        (attach_mod.DRAFT_POOL, "b", 30, "dflash-draft-rollback@7"),
    ]


def test_dflash_draft_graph_uses_replay_buffers_and_checks_padding_rows():
    att = _draft_attachment()
    att.speculative_dflash = True
    att.page_size = 16
    att.facts = {"dflash_block_size": 2}
    swa = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4, 5, 6],
        _paged_kv_indices_buf=[100, 101, 102, 103, 0, 0],
        _paged_kv_last_page_len_buf=[1, 1, 1],
    )
    full = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 5, 6, 7],
        _paged_kv_indices_buf=[200, 201, 202, 203, 204, 0, 0],
        _paged_kv_last_page_len_buf=[1, 1, 1],
    )

    class Slot:
        @staticmethod
        def slice_for(rows, tokens):
            assert (rows, tokens) == (3, 6)
            return [13, 14, 0, 0, 0, 0]

    att.draft_runner = SimpleNamespace(
        decode_cuda_graph_runner=SimpleNamespace(
            buffer_registry=SimpleNamespace(
                get_slot=lambda name: Slot() if name == "out_cache_loc" else None
            )
        )
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(prefill_wrappers=[object(), object()]),
        prefill_cuda_graph_metadata={3: [swa, full]},
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("TARGET_VERIFY"),
        batch_size=1,
        req_pool_indices=[4],
        seq_lens=[3],
        positions=[3, 4],
        out_cache_loc=[999, 999],
        _kvguard_graph_bs=3,
    )

    (write, swa_attention, full_attention), kind, tokens = att._draft_descriptors(
        fb, backend, tag="-dflash-graph"
    )

    assert (kind, tokens) == ("dflash_verify", 2)
    assert write.mapping.slots == [13, 14, 0, 0, 0, 0]
    assert [row.padding for row in write.rows] == [False, True]
    assert [row.padding for row in swa_attention.rows] == [False, True, True]
    assert [row.padding for row in full_attention.rows] == [False, True, True]
    assert swa_attention.mapping.table == [[100, 101, 102, 103], [0], [0]]
    assert full_attention.mapping.table == [[200, 201, 202, 203, 204], [0], [0]]


def test_eagle3_draft_decode_uses_position_plus_one_and_its_own_pool():
    att = _draft_attachment()
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4, 10],
        _paged_kv_indices_buf=list(range(100, 110)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrappers=[wrapper]),
        decode_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=2,
        req_pool_indices=[4, 5],
        positions=[3, 5],
        out_cache_loc=[13, 25],
    )

    (write, attention), kind, tokens = att._draft_descriptors(fb, backend, tag="-draft9")

    assert (kind, tokens) == ("decode", 2)
    assert write.pool == attention.pool == attach_mod.DRAFT_POOL
    assert [row.writes for row in write.rows] == [range(3, 4), range(5, 6)]
    assert [row.reads for row in attention.rows] == [range(0, 4), range(0, 6)]
    assert att.ledger.calls == [
        (attach_mod.DRAFT_POOL, "a", 3, "draft-decode-rollback@7"),
        (attach_mod.DRAFT_POOL, "b", 5, "draft-decode-rollback@7"),
    ]


def test_eagle3_draft_extend_rolls_back_then_describes_ragged_prefix():
    att = _draft_attachment()
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 3, 5],
        _paged_kv_indices_buf=[10, 11, 12, 20, 21],
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(prefill_wrappers=[wrapper], use_ragged=True),
        prefill_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DRAFT_EXTEND_V2"),
        batch_size=2,
        req_pool_indices=[4, 5],
        extend_prefix_lens=[3, 2],
        extend_seq_lens=[4, 4],
        out_cache_loc=list(range(30, 38)),
    )

    (attention, write), kind, tokens = att._draft_descriptors(fb, backend, tag="-draft10")

    assert (kind, tokens) == ("draft_extend", 8)
    assert [row.reads for row in attention.rows] == [range(0, 3), range(0, 2)]
    assert [row.writes for row in write.rows] == [range(3, 7), range(2, 6)]
    assert write.mapping.positions == [3, 4, 5, 6, 2, 3, 4, 5]
    assert att.ledger.calls == [
        (attach_mod.DRAFT_POOL, "a", 3, "draft-extend-rollback@7"),
        (attach_mod.DRAFT_POOL, "b", 2, "draft-extend-rollback@7"),
    ]


def test_eagle3_decode_graph_expands_one_replay_into_per_step_views():
    att = _draft_attachment()
    backends = [object(), object()]
    draft_host = SimpleNamespace(draft_attn_backend=SimpleNamespace(attn_backends=backends))
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=2,
        req_pool_indices=[4, 5],
        positions=[3, 8],
        # The framework layout is request-major, then speculative step.
        out_cache_loc=[30, 31, 32, 80, 81, 82],
    )

    views = att._eagle3_decode_graph_views(fb, draft_host)

    assert [backend for _view, backend in views] == backends
    assert [view.positions for view, _backend in views] == [[3, 8], [4, 9]]
    assert [view.out_cache_loc for view, _backend in views] == [[30, 80], [31, 81]]
    assert all(view.req_pool_indices == [4, 5] for view, _backend in views)


def test_eagle3_tree_decode_uses_branch_positions_without_dense_rollback():
    att = _draft_attachment()
    att.facts = {"speculative_eagle_topk": 2, "speculative_num_steps": 3}
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 5, 10, 17, 24],
        _paged_kv_indices_buf=list(range(100, 124)),
        _paged_kv_last_page_len_buf=[1, 1, 1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrappers=[wrapper]),
        decode_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=2,
        req_pool_indices=[4, 5],
        seq_lens=[3, 5],
        positions=[4, 4, 6, 6],
        out_cache_loc=[14, 17, 26, 29],
        _kvguard_draft_step=1,
        _kvguard_draft_topk=2,
    )

    (write, attention), kind, tokens = att._draft_descriptors(fb, backend, tag="-tree")

    assert (kind, tokens) == ("decode_tree", 4)
    assert [row.writes for row in write.rows] == [
        range(4, 5),
        range(7, 8),
        range(6, 7),
        range(9, 10),
    ]
    assert [row.reads for row in attention.rows] == [
        (0, 1, 2, 3, 4),
        (0, 1, 2, 6, 7),
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 8, 9),
    ]
    assert att.ledger.calls == []


def test_eagle3_tree_decode_rolls_back_once_at_the_first_inner_step():
    att = _draft_attachment()
    att.facts = {"speculative_eagle_topk": 2, "speculative_num_steps": 3}
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4, 8],
        _paged_kv_indices_buf=list(range(100, 108)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrappers=[wrapper]),
        decode_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=1,
        req_pool_indices=[4],
        seq_lens=[3],
        positions=[3, 3],
        out_cache_loc=[14, 17],
        _kvguard_draft_step=0,
    )

    (write, _attention), _kind, _tokens = att._draft_descriptors(fb, backend, tag="-tree0")

    assert not write.advances_computed
    assert att.ledger.calls == [(attach_mod.DRAFT_POOL, "a", 3, "draft-tree-rollback@7")]


def test_eagle3_tree_draft_extend_invalidates_the_sparse_workspace():
    att = _draft_attachment()
    att.facts = {"speculative_eagle_topk": 2, "speculative_num_steps": 3}
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 7],
        _paged_kv_indices_buf=list(range(100, 107)),
        _paged_kv_last_page_len_buf=[1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(prefill_wrappers=[wrapper], use_ragged=False),
        prefill_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DRAFT_EXTEND_V2"),
        batch_size=1,
        req_pool_indices=[4],
        extend_prefix_lens=[3],
        extend_seq_lens=[4],
        out_cache_loc=[14, 15, 16, 17],
    )

    att._draft_descriptors(fb, backend, tag="-tree-extend")

    assert att.ledger.calls == [(attach_mod.DRAFT_POOL, "a", 3, "draft-extend-rollback@7")]
    assert att.ledger.invalidations == [
        (attach_mod.DRAFT_POOL, "a", (3, 4, 5, 6, 7, 8), "draft-tree-invalidate@7")
    ]


def test_eagle3_paged_branch_copy_rolls_back_then_copies_tail_offsets():
    att = _draft_attachment()
    att.page_size = 16
    att.facts = {"speculative_eagle_topk": 2, "speculative_num_steps": 3}
    att.cur_batch = SimpleNamespace(req_pool_indices=[4])

    owners = att._begin_eagle3_branch_copy([16], [2], 2, 16)
    assert owners == ("a", "a")
    assert att.ledger.calls == [(attach_mod.DRAFT_POOL, "a", 18, "draft-tree-rollback@7")]

    att._record_eagle3_branch_copy([48, 49], [32, 33], owners)
    copies = [copy for copy, _site in att.ledger.copies]
    assert [(copy.block, copy.dst_block, copy.offsets, copy.request) for copy in copies] == [
        (2, 3, range(0, 1), "a"),
        (2, 3, range(1, 2), "a"),
    ]
    assert att.stats["draft_branch_copy_rollbacks"] == 1
    assert att.stats["draft_branch_copy_slots"] == 2


def test_eagle3_paged_tree_decode_and_invalidation_use_page_aligned_branches():
    att = _draft_attachment()
    att.page_size = 16
    att.facts = {"speculative_eagle_topk": 2, "speculative_num_steps": 3}
    table = [[*range(16, 34), 34, 35], [*range(16, 34), 50, 51]]
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 20, 40],
        _paged_kv_indices_buf=[slot for row_slots in table for slot in row_slots],
        _paged_kv_last_page_len_buf=[1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrappers=[wrapper]),
        decode_wrappers=[wrapper],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=1,
        req_pool_indices=[4],
        seq_lens=[18],
        positions=[19, 19],
        out_cache_loc=[35, 51],
        _kvguard_draft_step=1,
    )

    (write, attention), kind, tokens = att._draft_descriptors(fb, backend, tag="-paged")
    assert (kind, tokens) == ("decode_tree", 2)
    assert [row.writes for row in write.rows] == [range(19, 20), range(35, 36)]
    assert [row.reads for row in attention.rows] == [
        (*range(18), 18, 19),
        (*range(18), 34, 35),
    ]

    extend_wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 22],
        _paged_kv_indices_buf=list(range(22)),
        _paged_kv_last_page_len_buf=[1],
    )
    extend_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(prefill_wrappers=[extend_wrapper], use_ragged=False),
        prefill_wrappers=[extend_wrapper],
    )
    extend_fb = SimpleNamespace(
        forward_mode=_DraftMode("DRAFT_EXTEND_V2"),
        batch_size=1,
        req_pool_indices=[4],
        extend_prefix_lens=[18],
        extend_seq_lens=[4],
        out_cache_loc=[34, 35, 36, 37],
    )
    att._draft_descriptors(extend_fb, extend_backend, tag="-paged-extend")
    assert att.ledger.invalidations == [
        (
            attach_mod.DRAFT_POOL,
            "a",
            (18, 19, 20, 32, 33, 34, 35, 36),
            "draft-tree-invalidate@7",
        )
    ]


def test_eagle3_tree_decode_graph_expands_request_branch_step_layout():
    att = _draft_attachment()
    att.facts = {"speculative_num_steps": 3, "speculative_eagle_topk": 2}
    backends = [object(), object()]
    draft_host = SimpleNamespace(draft_attn_backend=SimpleNamespace(attn_backends=backends))
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=2,
        req_pool_indices=[4, 5],
        seq_lens=[3, 8],
        positions=[3, 3, 8, 8],
        out_cache_loc=[30, 31, 32, 40, 41, 42, 80, 81, 82, 90, 91, 92],
    )

    views = att._eagle3_decode_graph_views(fb, draft_host)

    assert [view.positions for view, _backend in views] == [[3, 3, 8, 8], [4, 4, 9, 9]]
    assert [view.out_cache_loc for view, _backend in views] == [
        [30, 40, 80, 90],
        [31, 41, 81, 91],
    ]
    assert [view._kvguard_draft_step for view, _backend in views] == [0, 1]


def test_eagle3_graph_descriptors_use_the_replay_bucket_and_paged_extend():
    att = _draft_attachment()
    decode_wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4, 10],
        _paged_kv_indices_buf=list(range(100, 110)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    stale_wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 4],
        _paged_kv_indices_buf=list(range(4)),
        _paged_kv_last_page_len_buf=[1],
    )
    decode_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(decode_wrappers=[stale_wrapper]),
        decode_wrappers=[stale_wrapper],
        decode_cuda_graph_metadata={2: [decode_wrapper]},
    )
    decode_fb = SimpleNamespace(
        forward_mode=_DraftMode("DECODE"),
        batch_size=2,
        req_pool_indices=[4, 5],
        positions=[3, 5],
        out_cache_loc=[13, 25],
        _kvguard_graph_bs=2,
    )
    (_write, attention), _kind, _tokens = att._draft_descriptors(
        decode_fb, decode_backend, tag="-draft-graph"
    )
    assert [row.reads for row in attention.rows] == [range(0, 4), range(0, 6)]

    extend_wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 7, 13],
        _paged_kv_indices_buf=list(range(200, 213)),
        _paged_kv_last_page_len_buf=[1, 1],
    )
    extend_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(prefill_wrappers=[stale_wrapper], use_ragged=True),
        prefill_wrappers=[stale_wrapper],
        draft_extend_cuda_graph_metadata={2: [extend_wrapper]},
    )
    extend_fb = SimpleNamespace(
        forward_mode=_DraftMode("DRAFT_EXTEND_V2"),
        batch_size=2,
        req_pool_indices=[4, 5],
        extend_prefix_lens=[3, 2],
        extend_seq_lens=[4, 4],
        out_cache_loc=list(range(30, 38)),
        _kvguard_graph_bs=2,
        _kvguard_use_ragged=False,
    )
    (write, attention), _kind, _tokens = att._draft_descriptors(
        extend_fb, extend_backend, tag="-draft-extend-graph"
    )
    assert [row.reads for row in attention.rows] == [range(0, 7), range(0, 6)]
    assert [row.writes for row in write.rows] == [range(3, 7), range(2, 6)]


def test_multilayer_eagle_graph_uses_swa_write_buffer_and_checks_every_padding_row():
    att = _draft_attachment()
    att.page_size = 16
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=[0, 7, 8, 9],
        _paged_kv_indices_buf=list(range(100, 109)),
        _paged_kv_last_page_len_buf=[1, 1, 1],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            prefill_wrappers=[object(), object()],
            use_ragged=True,
            swa_out_cache_loc=[999] * 4,
        ),
        draft_extend_cuda_graph_metadata={3: [wrapper, object()]},
        cuda_graph_swa_out_cache_loc=[
            30,
            31,
            32,
            33,
            0,
            0,
            0,
            0,
            90,
            91,
            92,
            93,
        ],
    )
    fb = SimpleNamespace(
        forward_mode=_DraftMode("DRAFT_EXTEND_V2"),
        batch_size=1,
        req_pool_indices=[4],
        extend_prefix_lens=[3],
        extend_seq_lens=[4],
        out_cache_loc=[700, 701, 702, 703],
        _kvguard_graph_bs=3,
        _kvguard_graph_width=4,
        _kvguard_use_ragged=False,
    )

    (write, attention), kind, tokens = att._draft_descriptors(
        fb,
        backend,
        tag="-step-graph",
        pool="draft-depth-0",
        separate_swa=True,
    )

    assert (kind, tokens) == ("draft_extend", 4)
    assert write.mapping.slots == [30, 31, 32, 33]
    assert [row.padding for row in attention.rows] == [False, True, True]
    assert attention.mapping.table[1] == [107, 0, 0, 0, 0]
    assert attention.mapping.table[2] == [108, 90, 91, 92, 93]


def test_validator_rows_for_ngram_verify_uses_full_window_and_draft_owners():
    plan = attach_mod.StepPlan(
        seq=3,
        mode="verify",
        graph=True,
        n=2,
        n_pad=3,
        req_ids=["a", "b"],
        seq_lens=[8, 7],
        prefix=[5, 4],
        use_ragged=False,
        indptr=[0, 8, 15, 16],
        indices=list(range(16)),
        loc=list(range(9)),
        descs=[],
    )
    rows, tokens = attach_mod.validator_rows(plan, {"a": 10, "b": 20}.__getitem__)
    assert rows == (3, [10, 20, attach_mod.PADDING_ROW], [8, 7, 0])
    assert tokens == (
        9,
        [
            10,
            10,
            10,
            20,
            20,
            20,
            attach_mod.PADDING_ROW,
            attach_mod.PADDING_ROW,
            attach_mod.PADDING_ROW,
        ],
    )


# tree extraction


class _Data:
    def __init__(self, value, lock_ref=0):
        self.value, self.lock_ref = value, lock_ref


class _UNode:
    """A ``UnifiedTreeNode`` lookalike: ``children``, ``id``, ``component_data[FULL]``."""

    def __init__(self, nid, value, lock_ref=0, children=()):
        self.id = nid
        self.children = {i: c for i, c in enumerate(children)}
        self.component_data = [_Data(value, lock_ref)]


class _PyCore:
    def __init__(self, root):
        self.root_node = root


class _RustCore:
    @property
    def root_node(self):
        raise NotImplementedError("not ported")

    def walk_for_kv_canary(self, unlocked_only, swa_resident_only):
        raise AssertionError("not called by tree_strategy")


class _Tree:
    def __init__(self, core):
        self.tree_core = core


def test_nodes_from_unified_tree_reads_full_component_and_skips_evicted():
    leaf = _UNode(3, [7, 8], lock_ref=0)
    evicted = _UNode(4, None, lock_ref=1)
    mid = _UNode(2, [5, 6], lock_ref=2, children=[leaf, evicted])
    root = _UNode(1, [], lock_ref=1, children=[mid])
    nodes = attach_mod.nodes_from_unified_tree(root)
    assert [(n.node_id, n.slots, n.lock_ref) for n in nodes] == [(2, (5, 6), 2), (3, (7, 8), 0)]


def test_tree_strategy_distinguishes_the_cores():
    assert attach_mod.tree_strategy(_Tree(_PyCore(_UNode(1, [])))) == "python_nodes"
    assert attach_mod.tree_strategy(_Tree(_RustCore())) == "rust_walk"
    assert attach_mod.tree_strategy(_Tree(None)) == "none"

    class RadixCache:  # the class name is the strategy's key
        pass

    assert attach_mod.tree_strategy(RadixCache()) == "radix_nodes"


def test_snapshot_from_walk_states_and_duplicates():
    rows = [
        attach_mod.RowFacts("a", (5, 6, 9, 10), 4, 4, 2, new_tokens=1),  # tree prefix 5,6; own 9,10
        attach_mod.RowFacts("b", (5, 6, 11), 3, 3, 2, new_tokens=1),
    ]
    snap, dup = attach_mod.snapshot_from_walk(
        7, "kv", rows, tree_slots=[5, 6, 7, 8, 7], unlocked_slots=[7, 8], free_slots=[12, 13]
    )
    assert dup == [7], "a slot the walk lists twice is reported, once"
    by_slot = {b.block: b for b in snap.blocks}
    assert by_slot[5].state is BlockState.PUBLISHED and not by_slot[5].reserved
    assert by_slot[5].refs is None, "the walk carries no lock counts"
    assert by_slot[7].state is BlockState.PUBLISHED and by_slot[7].reserved
    assert by_slot[9].state is BlockState.PRIVATE and by_slot[11].state is BlockState.PRIVATE
    assert snap.seq == 7 and snap.free == {"kv": frozenset({12, 13})}
    req = {r.request: r for r in snap.requests}
    assert req["a"].blocks == (5, 6, 9, 10) and req["a"].num_computed == 3
    assert req["b"].num_tokens == 3 and req["b"].num_computed == 2


def test_snapshot_from_walk_marks_beam_history_shared_not_private():
    rows = [
        attach_mod.RowFacts("leader", (5, 6, 9), 3, 3, 0, new_tokens=1),
        attach_mod.RowFacts("leader#beam1", (5, 6, 10), 3, 3, 0, new_tokens=1),
    ]
    snap, dup = attach_mod.snapshot_from_walk(
        8,
        "kv",
        rows,
        tree_slots=[],
        unlocked_slots=[],
        free_slots=[11],
        shared_slots={5, 6},
    )
    assert dup == []
    by_slot = {b.block: b for b in snap.blocks}
    assert by_slot[5].state is BlockState.PUBLISHED and not by_slot[5].reserved
    assert by_slot[6].state is BlockState.PUBLISHED and not by_slot[6].reserved
    assert by_slot[9].state is BlockState.PRIVATE
    assert by_slot[10].state is BlockState.PRIVATE


def test_swa_snapshot_translates_retired_prefix_and_separate_physical_pages():
    mapping = [0] * 64
    mapping[32:48] = range(64, 80)
    mapping[48:64] = range(80, 96)
    rows = [
        attach_mod.RowFacts(
            "a",
            tuple(range(16, 64)),
            kv_allocated_len=48,
            kv_committed_len=48,
            cache_protected_len=32,
            new_tokens=1,
        )
    ]

    snap, duplicates = attach_mod.swa_snapshot_from_walk(
        9,
        "swa_kv",
        rows,
        mapping,
        tree_full_slots=range(32, 48),
        unlocked_full_slots=range(32, 48),
        free_pages=[1, 2, 3, 6],
        page_size=16,
    )

    assert duplicates == []
    assert snap.free == {"swa_kv": frozenset({1, 2, 3, 6})}
    request = snap.requests[0]
    assert request.blocks == (None, 4, 5)
    assert request.first_token == 16
    assert request.num_computed == 47
    by_page = {block.block: block for block in snap.blocks}
    assert by_page[4].state is BlockState.PUBLISHED and by_page[4].reserved
    assert by_page[5].state is BlockState.PRIVATE and not by_page[5].reserved


def test_swa_snapshot_discards_a_retained_prefix_island_before_evicted_gap():
    mapping = [0] * 64
    mapping[16:32] = range(16, 32)
    mapping[48:64] = range(64, 80)
    row = attach_mod.RowFacts(
        "A",
        tuple(range(16, 64)),
        48,
        47,
        16,
        1,
        swa_first_token=32,
    )

    snap, duplicates = attach_mod.swa_snapshot_from_walk(
        10,
        "swa_kv",
        [row],
        mapping,
        tree_full_slots=range(16, 32),
        unlocked_full_slots=range(16, 32),
        free_pages=[1, 2, 3],
        page_size=16,
    )

    assert duplicates == []
    request = snap.requests[0]
    assert request.blocks == (None, None, 4)
    assert request.first_token == 32
    assert {block.block for block in snap.blocks} == {1, 4}
    by_page = {block.block: block for block in snap.blocks}
    assert by_page[1].state is BlockState.PUBLISHED and by_page[1].reserved


# HiCache transfers


class _QueryEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


class _HiCacheController:
    def __init__(self):
        self.write_event = _QueryEvent()
        self.load_event = _QueryEvent()
        self.write_queue = [SimpleNamespace(device_indices=[7, 8, 7], pool_transfers=None)]
        self.load_queue = [SimpleNamespace(device_indices=[20, 21], pool_transfers=None)]
        self.ack_write_queue = []
        self.ack_load_queue = []
        self.write_fence_stream = None
        self.l2_transfer_engine = SimpleNamespace(
            device_to_host_stream=SimpleNamespace(cuda_stream=101),
            host_to_device_stream=SimpleNamespace(cuda_stream=202),
        )

    def start_writing(self):
        self.write_queue.clear()
        self.ack_write_queue.append(SimpleNamespace(finish_event=self.write_event))

    def start_loading(self):
        self.load_queue.clear()
        self.ack_load_queue.append(SimpleNamespace(finish_event=self.load_event))
        return 9


class _TransferLedger:
    def __init__(self):
        self.reads = []
        self.completions = []
        self.fills = []
        self.writes = []
        self.allocations = []
        self.epochs = {}
        self.blocks = {}
        self.frees = []

    def allocate(self, pool, blocks, *, site):
        self.allocations.append((pool, tuple(blocks), site))

    def external_read(self, read, *, site):
        self.reads.append((read, site))
        self.epochs[read.dispatch_id] = object()

    def complete(self, dispatch_id):
        self.completions.append(dispatch_id)

    def external_fill(self, fill, *, site):
        self.fills.append((fill, site))

    def external_write(self, write, *, site):
        self.writes.append((write, site))
        self.epochs[write.dispatch_id] = object()

    def free(self, pool, blocks, *, site):
        self.frees.append((pool, tuple(blocks), site))


class _DirectLinker:
    def __init__(self):
        self.store_ready = None
        self.load_ready = None

    def record_transfer_ready(self):
        return SimpleNamespace(synchronize=lambda: None, wait=lambda _stream: None)

    def offload(self, _transfers, *args, **kwargs):
        self.store_ready = self.record_transfer_ready()
        return True

    def pop_completed_offload(self):
        return True

    def load(self, _rid, _transfers):
        return True

    def cancel_queued_load(self, _rid):
        return False

    def start_layer_wise_loading(self):
        self.load_ready = self.record_transfer_ready()
        return 4

    def finish_layer_wise_loading(self, _index):
        return True

    def reset(self):
        return None

    def close(self):
        return None


def _external_linker_attachment():
    linker = _DirectLinker()
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = SimpleNamespace(linker=SimpleNamespace(cache_linker=linker))
    att.page_size = 16
    att.state = None
    att.seq = 7
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.epochs["forward:1"] = object()
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.fault = None
    att.open = {1: SimpleNamespace(dispatch_ids=("forward:1",))}
    att._external_linker_seq = 0
    att._external_linker_context = None
    att._external_linker_pending_loads = {}
    att._external_linker_load_jobs = {}
    att._external_linker_store_jobs = deque()
    att._external_linker_lock = threading.RLock()
    att._install_external_linker()
    return att, linker


def test_external_linker_store_and_load_follow_the_concrete_readiness_boundary():
    att, linker = _external_linker_attachment()
    transfer = SimpleNamespace(
        name=SimpleNamespace(value="kv"),
        device_indices=list(range(16, 32)),
    )

    assert linker.offload([transfer]) is True
    assert att.ledger.reads == []
    linker.store_ready.synchronize()
    [(read, site)] = att.ledger.reads
    assert read.blocks == (1,) and read.offsets == range(16)
    assert read.after == ("forward:1",)
    assert site == "external-linker-store:1:kv@7"
    assert linker.pop_completed_offload() is True
    assert att.ledger.completions == ["external-linker-store:1:kv"]

    assert linker.load("request", [transfer]) is True
    assert linker.start_layer_wise_loading() == 4
    assert [(fill.block, fill.offsets) for fill, _site in att.ledger.fills] == [(1, range(16))]
    assert att.ledger.writes == []
    linker.load_ready.synchronize()
    [(write, site)] = att.ledger.writes
    assert write.blocks == (1,) and write.offsets == range(16)
    assert write.after == ("forward:1",)
    assert site == "external-linker-load:2:kv@7"
    assert linker.finish_layer_wise_loading(4) is True
    assert att.ledger.completions[-1] == "external-linker-load:2:kv"
    assert att.stats["external_linker_store_completions"] == 1
    assert att.stats["external_linker_load_completions"] == 1


def test_external_linker_internal_pins_cover_the_locked_ancestor_path():
    class Core:
        empty_match_result = SimpleNamespace(last_device_node=0)

        @staticmethod
        def collect_full_device_indices(node, root):
            assert root == 0
            return {5: [1, 2, 5], 8: [1, 2, 6, 8]}[node]

    att = object.__new__(attach_mod.SglangAttachment)
    att.external_linker = True
    att.tree = SimpleNamespace(
        tree_core=Core(),
        linker=SimpleNamespace(
            pending_offloads=[
                SimpleNamespace(lock_node_id=5),
                SimpleNamespace(lock_node_id=8),
            ]
        ),
    )

    assert att._external_linker_internal_pin_slots() == {1, 2, 5, 6, 8}


class _PdSender:
    def __init__(self):
        self.conclude_state = None
        self.sent = []
        self.status = 3
        self.bootstrap_room = 17
        self.kv_mgr = SimpleNamespace(_staging_outstanding={})

    def send(self, kv_indices, state_indices=None, **kwargs):
        self.sent.append((tuple(kv_indices), state_indices, kwargs))
        return "sent"

    def poll(self):
        return self.status

    def poll_pp_consensus(self):
        return self.status


def test_pd_prefill_source_reads_stay_open_until_terminal_poll():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 9
    att.page_size = 16
    att.stats = Counter()
    att.timings = Counter()
    att.ledger = _TransferLedger()
    att.ledger.epochs = {}
    att.ledger.blocks = {
        attach_mod.POOL: {
            7: SimpleNamespace(filled=0xFFFF),
            11: SimpleNamespace(filled=0xFFFF),
            19: SimpleNamespace(filled=0x7),
        }
    }
    record_read = att.ledger.external_read

    def external_read(read, *, site):
        record_read(read, site=site)
        att.ledger.epochs[read.dispatch_id] = object()

    att.ledger.external_read = external_read
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_reads = {}
    att._pd_senders = {}
    att._pd_extent_dir = None
    att._pd_sent_tokens = {}
    att._pd_failed_pending = set()
    att._pd_reserved_blocks = {}
    att._pd_trace = []
    att.req_pool = SimpleNamespace(req_to_token={3: list(range(64))})
    att._reap_hicache = lambda: None
    sender = _PdSender()
    req = SimpleNamespace(
        rid="A",
        prefix_len=0,
        start_send_idx=0,
        kv=SimpleNamespace(
            req_pool_idx=3,
            kv_allocated_len=35,
            kv_committed_len=35,
            cache_protected_len=0,
        ),
    )

    class Event:
        waits = 0

        def synchronize(self):
            self.waits += 1

    event = Event()
    att._wrap_disagg_sender(sender, req)
    assert sender.send([7, 11, 19], num_kv_tokens=35, indices_event=event) == "sent"
    assert event.waits == 1
    assert [(r.blocks, r.offsets, r.request) for r, _ in att.ledger.reads] == [
        ((7, 11), range(16), "A#1"),
        ((19,), range(3), "A#1"),
    ]
    assert att.ledger.completions == []
    assert att._pd_reserved_blocks == {"pd-send:1": (7, 11), "pd-send:2": (19,)}
    assert sender.poll() == 3 and att.ledger.completions == []
    sender.status = 4
    assert sender.poll() == 4
    assert att.ledger.completions == ["pd-send:1", "pd-send:2"]
    assert att._pd_reserved_blocks == {}
    assert att.stats["pd_send_reads_completed"] == 2


def test_pd_prefill_speculative_source_reads_target_and_draft_pools():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 9
    att.page_size = 16
    att.speculative_eagle3 = True
    att.stats = Counter()
    att.timings = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {7: SimpleNamespace(filled=0xFFFF)},
        attach_mod.DRAFT_POOL: {7: SimpleNamespace(filled=0xFFFF)},
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_reads = {}
    att._pd_senders = {}
    att._pd_extent_dir = None
    att._pd_sent_tokens = {}
    att._pd_failed_pending = set()
    att._pd_reserved_blocks = {}
    att._pd_trace = []
    att.req_pool = SimpleNamespace(req_to_token={3: list(range(16))})
    att._reap_hicache = lambda: None
    sender = _PdSender()
    req = SimpleNamespace(
        rid="spec",
        prefix_len=0,
        start_send_idx=0,
        kv=SimpleNamespace(
            req_pool_idx=3,
            kv_allocated_len=16,
            kv_committed_len=16,
            cache_protected_len=0,
        ),
    )

    att._wrap_disagg_sender(sender, req)
    assert sender.send([7], num_kv_tokens=16) == "sent"
    assert [(read.pool, read.blocks) for read, _site in att.ledger.reads] == [
        (attach_mod.POOL, (7,)),
        (attach_mod.DRAFT_POOL, (7,)),
    ]
    assert att._pd_reserved_blocks == {"pd-send:1": (7,)}
    sender.status = 4
    assert sender.poll() == 4
    assert att.ledger.completions == ["pd-send:1", "pd-send:2:draft_kv"]
    assert att.stats["pd_send_draft_pages"] == 1


def test_pd_prefill_hybrid_state_read_shares_sender_lifecycle():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 9
    att.page_size = 16
    att.stats = Counter()
    att.timings = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {55: SimpleNamespace(filled=0xFFFF)},
    }
    att.facts = {"unified_allocator": True}
    att.allocator = SimpleNamespace(
        full_attn_allocator=SimpleNamespace(
            ids=SimpleNamespace(physical_to_virtual=[-1] * 7 + [55])
        )
    )
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_reads = {}
    att._pd_senders = {}
    att._pd_extent_dir = None
    att._pd_sent_tokens = {}
    att._pd_failed_pending = set()
    att._pd_reserved_blocks = {}
    att._pd_trace = []
    att.req_pool = SimpleNamespace(req_to_token={3: list(range(32))})
    att.state = SimpleNamespace(
        _logical_slots=lambda slots: [102 if slot == 5 else -1 for slot in slots]
    )
    att._reap_hicache = lambda: None
    sender = _PdSender()
    req = SimpleNamespace(
        rid="hybrid",
        prefix_len=0,
        start_send_idx=0,
        kv=SimpleNamespace(
            req_pool_idx=3,
            kv_allocated_len=16,
            kv_committed_len=16,
            cache_protected_len=0,
        ),
    )

    att._wrap_disagg_sender(sender, req)
    assert sender.send([7], state_indices=[[5]], num_kv_tokens=16) == "sent"
    assert [(read.pool, read.blocks, read.offsets) for read, _site in att.ledger.reads] == [
        (attach_mod.POOL, (55,), range(16)),
        (attach_mod.STATE_POOL, (102,), range(1)),
    ]
    assert att._pd_reserved_blocks == {"pd-send:1": (55,)}
    sender.status = 4
    assert sender.poll() == 4
    assert att.ledger.completions == ["pd-send:1", "pd-send:2:state"]
    assert att.stats["pd_send_state_slots"] == 1


def test_pd_failed_sender_stays_open_until_mooncake_worker_drains():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 9
    att.page_size = 16
    att.stats = Counter()
    att.timings = Counter()
    att.ledger = _TransferLedger()
    att.ledger.epochs = {}
    att.ledger.blocks = {attach_mod.POOL: {7: SimpleNamespace(filled=0xFFFF)}}
    record_read = att.ledger.external_read

    def external_read(read, *, site):
        record_read(read, site=site)
        att.ledger.epochs[read.dispatch_id] = object()

    att.ledger.external_read = external_read
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_reads = {}
    att._pd_senders = {}
    att._pd_extent_dir = None
    att._pd_sent_tokens = {}
    att._pd_failed_pending = set()
    att._pd_reserved_blocks = {}
    att._pd_trace = []
    att.req_pool = SimpleNamespace(req_to_token={3: list(range(64))})
    att._reap_hicache = lambda: None
    sender = _PdSender()
    req = SimpleNamespace(
        rid="A",
        prefix_len=0,
        start_send_idx=0,
        kv=SimpleNamespace(
            req_pool_idx=3,
            kv_allocated_len=16,
            kv_committed_len=16,
            cache_protected_len=0,
        ),
    )

    att._wrap_disagg_sender(sender, req)
    sender.send([7], num_kv_tokens=16)
    sender.conclude_state = sender.status = 0
    sender.kv_mgr._staging_outstanding[sender.bootstrap_room] = 1

    assert sender.poll() == 0
    assert att.ledger.completions == []
    assert att.stats["pd_send_failures_inflight"] == 1

    sender.kv_mgr._staging_outstanding[sender.bootstrap_room] = 0
    att._reap_pd_reads()
    assert att.ledger.completions == ["pd-send:1"]
    assert att._pd_reserved_blocks == {}
    assert att.stats["pd_send_completion_failure_drained"] == 1


@pytest.mark.parametrize(
    ("statuses", "reason"),
    [({17: 4}, "success_quiescent"), ({}, "status_absent_quiescent")],
)
def test_pd_sender_reaps_after_authoritative_worker_quiescence(statuses, reason):
    att = object.__new__(attach_mod.SglangAttachment)
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.epochs = {"pd-send:1": object()}
    att._pd_reads = {17: ["pd-send:1"]}
    att._pd_failed_pending = set()
    att._pd_reserved_blocks = {"pd-send:1": (7,)}
    sender = _PdSender()
    sender.kv_mgr.request_status = statuses
    sender.kv_mgr.check_status = statuses.__getitem__
    sender._kvguard_retire_source_reads = lambda reason: (
        att.ledger.complete("pd-send:1"),
        att._pd_reads.pop(17),
        att._pd_reserved_blocks.pop("pd-send:1"),
        att._pd_senders.pop(17),
        att.stats.update({f"pd_send_completion_{reason}": 1}),
    )
    att._pd_senders = {17: sender}

    att._reap_pd_reads()

    assert att.ledger.completions == ["pd-send:1"]
    assert att._pd_senders == {}
    assert att.stats[f"pd_send_completion_{reason}"] == 1


def test_pd_source_release_trace_records_active_transport_overlap():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 9
    att.stats = Counter()
    att._pd_reads = {17: ["pd-send:1"]}
    att._pd_reserved_blocks = {"pd-send:1": (7,)}
    att._pd_source_release_trace = []
    manager = SimpleNamespace(
        _staging_outstanding={41: 1},
        _kvguard_active_source_rooms={41},
        request_status={41: 0},
    )
    sender = SimpleNamespace(
        bootstrap_room=41,
        conclude_state=0,
        kv_mgr=manager,
    )
    att._pd_senders = {17: sender}

    att._trace_pd_source_release([7, 8])

    assert len(att._pd_source_release_trace) == 1
    trace = att._pd_source_release_trace[0]
    assert trace["seq"] == 9
    assert trace["blocks"] == [7]
    assert trace["room"] == 41
    assert trace["sender_state"] == 0
    assert trace["manager_state"] == 0
    assert trace["worker_outstanding"] == 1
    assert trace["worker_active"] is True
    assert trace["stack"]
    assert att.stats["pd_source_releases_observed"] == 1


class _PdReceiver:
    def __init__(self):
        self.status = 3
        self.sent = []
        self.bootstrap_room = 41
        self.kv_mgr = SimpleNamespace()

    def send_metadata(self, page_indices, aux_index=None, state_indices=None, **kwargs):
        self.sent.append((tuple(page_indices), aux_index, state_indices, kwargs))
        return "metadata"

    def poll(self):
        return self.status


def test_pd_decode_destinations_stay_uninitialized_until_success_poll():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 2
    att.page_size = 16
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {
            block: SimpleNamespace(owner=None, generation=1, filled=0) for block in (7, 11, 19)
        }
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_decode_writes = {}
    att._pd_decode_receivers = {}
    att._pd_decode_meta = {}
    att._pd_decode_rooms = {}
    att._pd_extent_dir = None
    att._pd_decode_trace = []
    receiver = _PdReceiver()
    req = SimpleNamespace(rid="A", origin_input_ids=list(range(35)), output_ids=[])

    att._wrap_disagg_receiver(receiver, req)
    assert receiver.send_metadata([7, 11, 19], decode_prefix_len=0) == "metadata"
    assert [(w.blocks, w.offsets) for w, _site in att.ledger.writes] == [
        ((7, 11), range(16)),
        ((19,), range(3)),
    ]
    assert att.ledger.fills == [] and att.ledger.completions == []
    assert receiver.poll() == 3 and att.ledger.fills == []

    receiver.status = 4
    assert receiver.poll() == 4
    assert [(f.block, f.offsets) for f, _site in att.ledger.fills] == [
        (7, range(16)),
        (11, range(16)),
        (19, range(3)),
    ]
    assert att.ledger.completions == ["pd-receive:1", "pd-receive:2"]
    assert att.stats["pd_receive_pages"] == 3
    assert att.stats["pd_receive_tokens"] == 35


def test_pd_decode_speculative_destination_fills_target_and_draft_pools():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 2
    att.page_size = 16
    att.speculative_eagle3 = True
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        pool: {7: SimpleNamespace(owner=None, generation=1, filled=0)}
        for pool in (attach_mod.POOL, attach_mod.DRAFT_POOL)
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_decode_writes = {}
    att._pd_decode_receivers = {}
    att._pd_decode_meta = {}
    att._pd_decode_rooms = {}
    att._pd_extent_dir = None
    att._pd_decode_trace = []
    receiver = _PdReceiver()
    req = SimpleNamespace(rid="spec", origin_input_ids=list(range(16)), output_ids=[])

    att._wrap_disagg_receiver(receiver, req)
    assert receiver.send_metadata([7], decode_prefix_len=0) == "metadata"
    assert [(write.pool, write.blocks) for write, _site in att.ledger.writes] == [
        (attach_mod.POOL, (7,)),
        (attach_mod.DRAFT_POOL, (7,)),
    ]
    assert att.ledger.fills == []
    receiver.status = 4
    assert receiver.poll() == 4
    assert [(fill.pool, fill.block) for fill, _site in att.ledger.fills] == [
        (attach_mod.POOL, 7),
        (attach_mod.DRAFT_POOL, 7),
    ]
    assert att.ledger.completions == ["pd-receive:1", "pd-receive:2:draft_kv"]
    assert att.stats["pd_receive_draft_pages"] == 1


def test_pd_decode_hybrid_state_destination_fills_only_after_success():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 2
    att.page_size = 16
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {55: SimpleNamespace(owner=None, generation=1, filled=0)},
        attach_mod.STATE_POOL: {102: SimpleNamespace(owner=None, generation=1, filled=0)},
    }
    att.facts = {"unified_allocator": True}
    att.allocator = SimpleNamespace(
        full_attn_allocator=SimpleNamespace(
            ids=SimpleNamespace(physical_to_virtual=[-1] * 7 + [55])
        )
    )
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_decode_writes = {}
    att._pd_decode_receivers = {}
    att._pd_decode_meta = {}
    att._pd_decode_rooms = {}
    att._pd_extent_dir = None
    att._pd_decode_trace = []
    att.state = SimpleNamespace(
        _logical_slots=lambda slots: [102 if slot == 5 else -1 for slot in slots]
    )
    receiver = _PdReceiver()
    req = SimpleNamespace(rid="hybrid", origin_input_ids=list(range(16)), output_ids=[])

    att._wrap_disagg_receiver(receiver, req)
    assert receiver.send_metadata([7], state_indices=[[5]], decode_prefix_len=0) == "metadata"
    assert [(write.pool, write.blocks) for write, _site in att.ledger.writes] == [
        (attach_mod.POOL, (55,)),
        (attach_mod.STATE_POOL, (102,)),
    ]
    assert att.ledger.fills == []
    receiver.status = 4
    assert receiver.poll() == 4
    assert [(fill.pool, fill.block) for fill, _site in att.ledger.fills] == [
        (attach_mod.POOL, 55),
        (attach_mod.STATE_POOL, 102),
    ]
    assert att.stats["pd_receive_state_slots"] == 1


def test_pd_decode_publication_retires_only_cuda_events_already_done():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 6
    att.page_size = 16
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {attach_mod.POOL: {7: SimpleNamespace(owner=None, generation=1, filled=0)}}
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_decode_writes = {}
    att._pd_decode_receivers = {}
    att._pd_decode_meta = {}
    att._pd_decode_rooms = {}
    att._pd_extent_dir = None
    att._pd_decode_trace = []
    att.open = {
        4: SimpleNamespace(event=SimpleNamespace(query=lambda: True)),
        5: SimpleNamespace(event=SimpleNamespace(query=lambda: False)),
    }
    completed = []
    att._complete_through = lambda seq, reason: completed.append((seq, reason))
    receiver = _PdReceiver()
    req = SimpleNamespace(rid="A", origin_input_ids=list(range(16)), output_ids=[])

    att._wrap_disagg_receiver(receiver, req)
    assert receiver.send_metadata([7], decode_prefix_len=0) == "metadata"

    assert completed == [(4, "pd_receive_publication")]
    assert att._pd_decode_trace[0]["gpu_done_steps"] == (4,)
    assert att._pd_decode_trace[0]["gpu_open_steps"] == (5,)


def test_pd_decode_receive_early_free_control_releases_live_destination_page():
    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 2
    att.page_size = 16
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {
            block: SimpleNamespace(owner=None, generation=1, filled=0) for block in (7, 11)
        }
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_decode_writes = {}
    att._pd_decode_receivers = {}
    att._pd_decode_meta = {}
    att._pd_decode_rooms = {}
    att._pd_extent_dir = None
    att._pd_decode_trace = []
    att.fault = attach_mod.Fault("pd_decode_receive_early_free")
    released = []
    att.allocator = SimpleNamespace(
        free_pages=SimpleNamespace(new_tensor=lambda values: values),
        free=lambda pages: released.extend(attach_mod._ints(pages)),
    )
    receiver = _PdReceiver()
    req = SimpleNamespace(rid="A", origin_input_ids=list(range(20)), output_ids=[])

    att._wrap_disagg_receiver(receiver, req)
    assert receiver.send_metadata([7, 11], decode_prefix_len=0) == "metadata"

    assert released == [7 * 16]
    assert att.fault.done
    assert "before the remote write completed" in att.fault.note
    assert att.ledger.completions == []


def test_pd_decode_host_pool_retraction_checks_backup_and_fills_restore():
    class _Rows:
        def __getitem__(self, key):
            row, positions = key
            assert row == 3
            return list(range(16, 52))[positions]

    calls = []
    backup_handle = object()
    tree = SimpleNamespace(
        backup_kv_cache=lambda req: calls.append(("backup", req.rid)) or backup_handle,
        restore_kv_cache=lambda req, handle: (
            calls.append(("restore", req.rid, handle)) or "restored"
        ),
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = tree
    att.req_pool = SimpleNamespace(req_to_token=_Rows())
    att.page_size = 16
    att.seq = 7
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {
            block: SimpleNamespace(owner=None, generation=1, filled=0) for block in (1, 2, 3)
        }
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._fence_seq = 6
    completed_through = []
    att._complete_through = lambda seq, reason: completed_through.append((seq, reason))
    req = SimpleNamespace(rid="A", seqlen=36, kv=SimpleNamespace(req_pool_idx=3))

    att._install_disagg_retraction_backup()
    assert tree.backup_kv_cache(req) is backup_handle
    assert [(read.blocks, read.offsets) for read, _site in att.ledger.reads] == [
        ((1, 2), range(16)),
        ((3,), range(3)),
    ]
    assert att.ledger.completions == ["pd-retraction-backup:1", "pd-retraction-backup:2"]
    assert completed_through == [(6, "pd_retraction_backup_sync")]

    assert tree.restore_kv_cache(req, backup_handle) == "restored"
    assert [(write.blocks, write.offsets) for write, _site in att.ledger.writes] == [
        ((1, 2), range(16)),
        ((3,), range(3)),
    ]
    assert [(fill.block, fill.offsets) for fill, _site in att.ledger.fills] == [
        (1, range(16)),
        (2, range(16)),
        (3, range(3)),
    ]
    assert calls == [("backup", "A"), ("restore", "A", backup_handle)]
    assert att.stats["pd_retraction_backups"] == 1
    assert att.stats["pd_retraction_restores"] == 1


def test_pd_decode_async_host_pool_restore_stays_open_until_framework_finish():
    class _Rows:
        def __getitem__(self, key):
            row, positions = key
            assert row == 3
            return list(range(16, 52))[positions]

    event = object()
    calls = []
    tree = SimpleNamespace()

    def start_restore(req, handle):
        calls.append(("start", req.rid, handle))
        req.kv.restore_done_event = event

    def finish_restore(kv):
        calls.append(("finish", kv.restore_done_event))
        kv.restore_done_event = None

    tree.backup_kv_cache = lambda req: object()
    tree.start_restore_kv_cache = start_restore
    tree._finish_restore = finish_restore
    tree.finish_restore_kv_cache = lambda req: tree._finish_restore(req.kv)
    tree.restore_kv_cache = lambda req, handle: tree.start_restore_kv_cache(req, handle)
    tree.cache_controller = SimpleNamespace(
        l2_transfer_engine=SimpleNamespace(
            _h2d_on_current_stream=False,
            host_to_device_stream=SimpleNamespace(cuda_stream=123),
        )
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = tree
    att.req_pool = SimpleNamespace(req_to_token=_Rows())
    att.page_size = 16
    att.seq = 7
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {
            block: SimpleNamespace(owner=None, generation=1, filled=0) for block in (1, 2, 3)
        }
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._pd_retraction_restores = {}
    att._fence_seq = 6
    att.stream_epochs = True
    att._fenced_epochs = lambda: ("attn@6",)
    att._complete_through = lambda seq, reason: None
    req = SimpleNamespace(
        rid="A",
        seqlen=36,
        kv=SimpleNamespace(req_pool_idx=3, restore_done_event=None),
    )
    backup_handle = object()

    att._install_disagg_retraction_backup()
    assert tree.start_restore_kv_cache(req, backup_handle) is None
    assert len(att.ledger.writes) == 2
    assert att.ledger.writes[0][0].stream == 123
    assert att.ledger.writes[0][0].after == ("attn@6",)
    assert att.ledger.fills == []
    assert att.ledger.completions == []
    assert id(event) in att._pd_retraction_restores

    tree.finish_restore_kv_cache(req)
    assert [(fill.block, fill.offsets) for fill, _site in att.ledger.fills] == [
        (1, range(16)),
        (2, range(16)),
        (3, range(3)),
    ]
    assert att.ledger.completions == [
        "pd-retraction-restore:1",
        "pd-retraction-restore:2",
    ]
    assert att._pd_retraction_restores == {}
    assert calls == [("start", "A", backup_handle), ("finish", event)]
    assert att.stats["pd_retraction_restore_starts"] == 1
    assert att.stats["pd_retraction_restores"] == 1


def test_pd_decode_cpu_tensor_retraction_checks_backup_and_fills_restore():
    calls = []
    tree = SimpleNamespace(
        backup_kv_cache=lambda req: object(),
        restore_kv_cache=lambda req, handle: None,
    )
    allocator = SimpleNamespace(
        get_cpu_copy=lambda indices, **kwargs: (
            calls.append(("backup", tuple(indices), kwargs)) or "cpu-copy"
        ),
        load_cpu_copy=lambda copy, indices, **kwargs: (
            calls.append(("restore", copy, tuple(indices), kwargs)) or "loaded"
        ),
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = tree
    att.allocator = allocator
    att.page_size = 16
    att.seq = 9
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.ledger.blocks = {
        attach_mod.POOL: {
            block: SimpleNamespace(owner="A#1", generation=1, filled=0) for block in (1, 2, 3)
        }
    }
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att._pd_seq = 0
    att._fence_seq = 8
    att.rid_of_row = {4: "A#1"}
    att.open = {}
    att.req_pool = SimpleNamespace(req_to_token=[])
    att._complete_through = lambda seq, reason: None

    att._install_disagg_retraction_backup()
    assert allocator.get_cpu_copy(list(range(16, 51)), req_pool_index=4) == "cpu-copy"
    assert [(read.blocks, read.offsets) for read, _site in att.ledger.reads] == [
        ((1, 2), range(16)),
        ((3,), range(3)),
    ]

    assert allocator.load_cpu_copy("cpu-copy", list(range(16, 51)), req_pool_index=4) == "loaded"
    assert [(write.blocks, write.offsets) for write, _site in att.ledger.writes] == [
        ((1, 2), range(16)),
        ((3,), range(3)),
    ]
    assert [(fill.block, fill.offsets) for fill, _site in att.ledger.fills] == [
        (1, range(16)),
        (2, range(16)),
        (3, range(3)),
    ]
    assert calls == [
        ("backup", tuple(range(16, 51)), {"mamba_indices": None, "req_pool_index": 4}),
        (
            "restore",
            "cpu-copy",
            tuple(range(16, 51)),
            {"mamba_indices": None, "req_pool_index": 4},
        ),
    ]
    assert att.stats["pd_retraction_cpu_backups"] == 1
    assert att.stats["pd_retraction_cpu_restores"] == 1


def _pd_retraction_deadlock_attachment(*, running_active=False, reserved=True, transferred=True):
    empty_batch = SimpleNamespace(is_empty=lambda: True)
    prealloc = SimpleNamespace(
        retracted_queue=[SimpleNamespace(rid="retracted")],
        queue=[SimpleNamespace(req=SimpleNamespace(rid="reserved"))] if reserved else [],
    )
    transfer = SimpleNamespace(
        queue=[SimpleNamespace(req=SimpleNamespace(rid="transferred"))] if transferred else [],
        _deferred_releases=[],
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.disagg_decode = True
    att._pd_retraction_deadlock_reported = False
    att._pd_preallocation_capacity_reported = False
    att.scheduler = SimpleNamespace(
        disagg_decode_prealloc_queue=prealloc,
        disagg_decode_transfer_queue=transfer,
        running_batch=(SimpleNamespace(is_empty=lambda: False) if running_active else empty_batch),
        last_batch=None,
        result_queue=[],
        waiting_queue=[],
    )
    att.reporter = Reporter(fail_fast=False)
    att.stats = Counter()
    att.seq = 17
    att.dirty = False
    return att


def test_pd_retraction_prealloc_circular_wait_is_reported_once():
    att = _pd_retraction_deadlock_attachment()

    att._check_pd_retraction_prealloc_deadlock()
    att._check_pd_retraction_prealloc_deadlock()

    assert [report.kind for report in att.reporter.reports] == ["pd_retraction_prealloc_deadlock"]
    assert att.reporter.reports[0].policy is None
    assert att.reporter.reports[0].request == "retracted"
    assert att.stats["pd_retraction_prealloc_deadlocks"] == 1
    assert att.dirty


def test_pd_retraction_transfer_circular_wait_is_reported():
    att = _pd_retraction_deadlock_attachment(reserved=False, transferred=True)

    att._check_pd_retraction_prealloc_deadlock()

    assert [report.kind for report in att.reporter.reports] == ["pd_retraction_prealloc_deadlock"]
    assert "transferred" in att.reporter.reports[0].detail


def test_pd_retraction_preallocation_without_transfer_is_not_circular():
    att = _pd_retraction_deadlock_attachment(transferred=False)

    att._check_pd_retraction_prealloc_deadlock()

    assert att.reporter.reports == []


def test_pd_retraction_prealloc_wait_allows_an_active_request_to_release_capacity():
    att = _pd_retraction_deadlock_attachment(running_active=True)

    att._check_pd_retraction_prealloc_deadlock()

    assert att.reporter.reports == []
    assert att.stats["pd_retraction_prealloc_deadlocks"] == 0


def test_pd_preallocation_capacity_mismatch_is_persisted_once():
    att = _pd_retraction_deadlock_attachment()
    flushes = []
    att.flush = flushes.append

    att._report_pd_preallocation_capacity_mismatch("available=256, required_alloc=304")
    att._report_pd_preallocation_capacity_mismatch("duplicate")

    assert [report.kind for report in att.reporter.reports] == [
        "pd_preallocation_capacity_mismatch"
    ]
    assert att.reporter.reports[0].policy is None
    assert att.stats["pd_preallocation_capacity_mismatches"] == 1
    assert flushes == ["pd-preallocation-capacity-mismatch"]


def _hicache_attachment(page_size=1, *, unified=False):
    controller = _HiCacheController()
    if page_size > 1:
        controller.write_queue = [SimpleNamespace(device_indices=list(range(16, 32)))]
        controller.load_queue = [SimpleNamespace(device_indices=list(range(32, 48)))]

    def writing_check():
        if controller.write_event.done:
            controller.ack_write_queue.clear()
        return "write-checked"

    tree = SimpleNamespace(
        cache_controller=controller,
        writing_check=writing_check,
        loading_check=lambda: "load-checked",
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = tree
    att.page_size = page_size
    att.seq = 4
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.fault = None
    att.facts = {"unified_allocator": unified}
    att.speculative_eagle3 = False
    att.separate_swa = False
    att.state = None
    if unified:
        owner = SimpleNamespace(free_virtual_ids=[1, 2, 3])
        att.allocator = SimpleNamespace(full_attn_allocator=owner)

        def load(*_args, **_kwargs):
            page = owner.free_virtual_ids.pop(0)
            return list(range(page * page_size, (page + 1) * page_size))

        controller.load = load
    att._hicache_seq = 0
    att._hicache_reads = {}
    att._hicache_store_reservations = {}
    att._hicache_loads = {}
    att._install_hicache()
    return att, controller


def test_hicache_store_is_an_external_read_until_the_exact_completion_event():
    att, controller = _hicache_attachment()

    assert controller.start_writing() is None
    [(read, site)] = att.ledger.reads
    assert read.blocks == (7, 8), "merged transfer slots are de-duplicated in queue order"
    assert read.offsets == range(1) and read.stream == 101
    assert read.source == "sglang_hicache_store" and site == "hicache-store:1:kv@4"
    assert att.stats["hicache_store_submissions"] == 1
    assert att.ledger.completions == []
    assert len(att._hicache_store_reservations) == 1

    assert att.tree.writing_check() == "write-checked"
    assert att.ledger.completions == [], "polling an incomplete CUDA event does not retire it"
    controller.write_event.done = True
    assert att.tree.writing_check() == "write-checked"
    assert att.ledger.completions == ["hicache-store:1:kv"]
    assert att._hicache_store_reservations == {}, "the framework consumed the transfer ack"
    assert att.stats["hicache_store_completions"] == 1


def test_hicache_load_fills_allocated_slots_only_after_completion():
    att, controller = _hicache_attachment()

    assert controller.start_loading() == 9
    assert att.ledger.fills == []
    assert att.stats["hicache_load_submissions"] == 1
    controller.load_event.done = True
    assert att.tree.loading_check() == "load-checked"
    assert [fill.block for fill, _site in att.ledger.fills] == [20, 21]
    assert all(fill.offsets == range(1) for fill, _site in att.ledger.fills)
    assert all(fill.source == "sglang_hicache_load" for fill, _site in att.ledger.fills)
    assert att.stats["hicache_load_completions"] == 1


def test_unified_hicache_load_mediates_direct_owner_allocation():
    att, controller = _hicache_attachment(page_size=16, unified=True)

    assert controller.load([100] * 16) == list(range(16, 32))

    assert att.ledger.allocations == [(attach_mod.POOL, (1,), "hicache-load-alloc@4")]
    assert att.stats["hicache_load_allocations"] == 1
    assert att.stats["hicache_load_alloc_pages"] == 1


def test_host_fallback_reaps_hicache_after_its_forward_synchronize(monkeypatch):
    """A load that completes during forward must be credited before host validation."""

    order = []

    class Stream:
        @staticmethod
        def synchronize():
            order.append("synchronize")

    class Event:
        def record(self):
            order.append("record")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: Stream(), Event=Event)),
    )

    att = object.__new__(attach_mod.SglangAttachment)
    att.seq = 7
    att._prestep = {}
    att.sync_after_forward = False
    att.stats = Counter()
    att.timings = Counter()
    att._reap_hicache = lambda: order.append("reap")
    att._descriptors = lambda _fb, _out: order.append("descriptors") or []
    att.twins = None
    att.gpu = None
    att.pdmux = False
    att.cur_batch = SimpleNamespace(forward_iter=None)
    att.open = {}
    att._last_dispatch = {}
    att.trace = []
    att.events = 0
    att.dirty = False

    att._after_forward(SimpleNamespace(), None)

    assert order[:3] == ["synchronize", "reap", "descriptors"]


def test_hicache_transfers_are_page_granular_at_page_size_sixteen():
    att, controller = _hicache_attachment(page_size=16)

    controller.start_writing()
    [(read, _site)] = att.ledger.reads
    assert read.blocks == (1,) and read.offsets == range(16)
    controller.start_loading()
    controller.load_event.done = True
    att.tree.loading_check()
    [(fill, _site)] = att.ledger.fills
    assert fill.block == 2 and fill.offsets == range(16)


def test_hicache_tracks_eagle_draft_sidecar_reads_and_fills():
    att, controller = _hicache_attachment()
    att.speculative_eagle3 = True
    controller.write_queue[0].pool_transfers = [
        SimpleNamespace(name="draft", device_indices=[7, 8, 7])
    ]
    controller.load_queue[0].pool_transfers = [
        SimpleNamespace(name="draft", device_indices=[20, 21])
    ]

    controller.start_writing()
    assert [(read.pool, read.blocks) for read, _site in att.ledger.reads] == [
        ("kv", (7, 8)),
        ("draft_kv", (7, 8)),
    ]
    controller.write_event.done = True
    att.tree.writing_check()
    assert att.ledger.completions == [
        "hicache-store:1:kv",
        "hicache-store:1:draft_kv",
    ]

    controller.start_loading()
    controller.load_event.done = True
    att.tree.loading_check()
    assert [(fill.pool, fill.block) for fill, _site in att.ledger.fills] == [
        ("kv", 20),
        ("kv", 21),
        ("draft_kv", 20),
        ("draft_kv", 21),
    ]


def test_hicache_tracks_hybrid_mamba_state_reads_and_fills():
    att, controller = _hicache_attachment(page_size=16)
    att.state = object()
    controller.write_queue[0].pool_transfers = [SimpleNamespace(name="mamba", device_indices=[7])]
    controller.load_queue[0].pool_transfers = [SimpleNamespace(name="mamba", device_indices=[20])]

    controller.start_writing()
    assert [(read.pool, read.blocks, read.offsets) for read, _site in att.ledger.reads] == [
        ("kv", (1,), range(16)),
        ("state", (7,), range(1)),
    ]

    controller.start_loading()
    controller.load_event.done = True
    att.tree.loading_check()
    assert [(fill.pool, fill.block, fill.offsets) for fill, _site in att.ledger.fills] == [
        ("kv", 2, range(16)),
        ("state", 20, range(1)),
    ]


def test_hicache_state_load_fault_frees_the_recurrent_destination():
    class OneSlot:
        def __len__(self):
            return 1

        def __getitem__(self, key):
            return self

        def __iter__(self):
            return iter((20,))

        def clone(self):
            return self

        def item(self):
            return 20

    att, controller = _hicache_attachment()
    freed = []
    att.state = SimpleNamespace(allocator=SimpleNamespace(free=lambda slots: freed.extend(slots)))
    att.fault = attach_mod.Fault("hicache_state_load_early_free")
    controller.load_queue[0].pool_transfers = [
        SimpleNamespace(name="mamba", device_indices=OneSlot())
    ]

    controller.start_loading()

    assert freed == [20]
    assert att.fault.done and "state destination slot 20" in att.fault.note


def _decode_offload_attachment(*, fault=None):
    controller = _HiCacheController()
    controller.write_queue = [SimpleNamespace(device_indices=[7, 8])]
    manager = SimpleNamespace(cache_controller=controller)

    def offload(req):
        controller.start_writing()
        return f"offloaded-{req.rid}"

    manager.offload_kv_cache = offload
    att = object.__new__(attach_mod.SglangAttachment)
    att.scheduler = SimpleNamespace(decode_offload_manager=manager)
    att.page_size = 1
    att.seq = 9
    att.stats = Counter()
    att.ledger = _TransferLedger()
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.fault = fault
    att.open = {
        8: SimpleNamespace(event=SimpleNamespace(query=lambda: False)),
    }
    att._last_write_blocks = (10,)
    att._decode_offload_reads = {}
    att._decode_offload_owner = None
    att._decode_offload_trace = []
    att._hicache_reads = {}
    att._hicache_loads = {}
    att._pd_seq = 0
    att.request_incarnations = lambda req: f"{req.rid}#1"
    att.allocator = SimpleNamespace(
        free_pages=SimpleNamespace(new_tensor=lambda values: values),
        free=lambda _slots: None,
    )
    att._install_decode_offload()
    return att, manager, controller


def test_decode_offload_is_an_external_read_until_the_exact_completion_event():
    att, manager, controller = _decode_offload_attachment()

    assert manager.offload_kv_cache(SimpleNamespace(rid="A")) == "offloaded-A"
    [(read, site)] = att.ledger.reads
    assert read.blocks == (7, 8) and read.offsets == range(1)
    assert read.request == "A#1" and read.stream == 101
    assert read.source == "sglang_pd_decode_offload"
    assert site == "pd-decode-offload:1@9"
    assert att.ledger.completions == []
    assert att.stats["pd_decode_offload_submissions_gpu_open"] == 1
    assert not att.stats["pd_decode_offload_write_races"]

    att._reap_hicache()
    assert att.ledger.completions == []
    controller.write_event.done = True
    att._reap_hicache()
    assert att.ledger.completions == ["pd-decode-offload:1"]
    assert att.stats["pd_decode_offload_completions"] == 1


def test_decode_offload_early_free_control_releases_a_live_source_page():
    fault = attach_mod.Fault("pd_decode_offload_early_free")
    att, manager, _controller = _decode_offload_attachment(fault=fault)
    released = []
    att.allocator.free = lambda pages: released.extend(attach_mod._ints(pages))

    manager.offload_kv_cache(SimpleNamespace(rid="A"))

    assert released == [7]
    assert fault.done
    assert "before the completion event" in fault.note
    assert att.ledger.completions == []


def test_hicache_internal_pins_cover_the_complete_locked_tree_paths():
    class Core:
        empty_match_result = SimpleNamespace(last_device_node=0)

        @staticmethod
        def collect_full_device_indices(node, root):
            assert root == 0
            return {5: [1, 2, 3, 4, 5], 8: [1, 2, 6, 7, 8]}[node]

    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = SimpleNamespace(
        tree_core=Core(),
        ongoing_write_through={
            5: SimpleNamespace(node_id=5, lock_params=object()),
            9: SimpleNamespace(node_id=8, lock_params=None),
        },
        ongoing_load_back={8: SimpleNamespace(node_id=8)},
    )

    assert att._hicache_internal_pin_slots() == {1, 2, 3, 4, 5, 6, 7, 8}


def test_buffer_only_hicache_internal_pins_come_from_pipeline():
    class Core:
        empty_match_result = SimpleNamespace(last_device_node=0)

        @staticmethod
        def collect_full_device_indices(node, root):
            assert root == 0
            return {5: [1, 2, 5], 8: [1, 6, 8]}[node]

    pipeline = SimpleNamespace(
        ongoing_write_through={
            -1: SimpleNamespace(
                intent=SimpleNamespace(snapshot=SimpleNamespace(node_id=5)),
                lock_params=object(),
            )
        },
        anchor_locks={"request": SimpleNamespace(node_id=8)},
    )
    att = object.__new__(attach_mod.SglangAttachment)
    att.tree = SimpleNamespace(
        tree_core=Core(),
        buffer_pipeline=pipeline,
        ongoing_load_back={},
    )

    assert att._hicache_internal_pin_slots() == {1, 2, 5, 6, 8}


# Tier 1b twins


def test_twin_descriptor_is_one_flat_row_bounded_by_the_entry_count():
    from kvguard.gpu.ptx import rewrite as R

    d = attach_mod.twin_descriptor(
        counter_ptr=0x10,
        record_ptr=0x20,
        entries_ptr=0x30,
        indices_bytes=4 * 1000,
        num_blocks=16385,
    )
    assert len(d) == R.N_DESC
    assert d[R.D_ENABLE] == R.EN_TABLE
    assert d[R.D_COUNTER] == 0x10 and d[R.D_RECORD] == 0x20 and d[R.D_ROW_LEN] == 0x30
    assert d[R.D_TBL_ROW_BYTES] == 4000 and d[R.D_TBL_NROWS] == 1, "the page list is one row"
    assert d[R.D_TBL_ENTRY_LOG2] == 2, "int32 entries"
    assert d[R.D_TBL_MODE] == 1 and d[R.D_BS_LOG2] == 0 and d[R.D_BS_MASK] == 0, (
        "flat slot map with block size 1: the check is entry index >= entries"
    )
    assert d[R.D_NBLOCKS] == 16385
    inert = attach_mod.twin_descriptor(
        counter_ptr=0, record_ptr=0, entries_ptr=0, indices_bytes=4, num_blocks=1, enabled=False
    )
    assert inert[R.D_ENABLE] == 0 and inert[R.D_COUNTER] == 0


def test_twin_fault_entries():
    assert attach_mod.twin_fault_entries(10, None) == 10
    assert attach_mod.twin_fault_entries(10, "short_entries") == 9
    assert attach_mod.twin_fault_entries(0, "short_entries") == 0, "never negative"
    with pytest.raises(ValueError):
        attach_mod.twin_fault_entries(1, "nonsense")


def test_twin_launch_delta_between_substitution_snapshots():
    before = {"twins": [{"name": "a", "launches": 5}]}
    after = {"twins": [{"name": "a", "launches": 8}, {"name": "b", "launches": 2}]}
    assert attach_mod.twin_launch_delta(before, after) == {"a": 3, "b": 2}
    assert attach_mod.twin_launch_delta({}, {}) == {}


def test_flashinfer_kernel_traits_from_the_mangled_name():
    name = (
        "_ZN10flashinfer34BatchPrefillWithPagedKVCacheKernelILb1ENS_12KernelTraitsILNS_8MaskModeE2E"
        "Lj16ELj1ELj8ELj4ELj4ELj1ELj4ELNS_15PosEncodingModeE0E13__nv_bfloat16S4_S4_fi"
        "NS_16DefaultAttentionILb1ELb0ELb0ELb0EEELb1EEE11PagedParamsEEvT1_"
    )
    assert attach_mod.flashinfer_kernel_traits(name) == {
        "mask_mode": 2,
        "cta_tile_q": 16,
        "num_mma_q": 1,
        "num_mma_kv": 8,
        "num_mma_d_qk": 4,
        "num_mma_d_vo": 4,
        "num_warps_q": 1,
        "num_warps_kv": 4,
        "pos_encoding_mode": 0,
    }
    assert attach_mod.flashinfer_kernel_traits("flash_fwd_splitkv_kernel") is None


def test_prepare_twins_without_the_preloaded_interposer(monkeypatch):
    from kvguard.adapters.sglang import hook

    monkeypatch.setattr(hook, "TWINS_PREP", None)
    monkeypatch.delenv(hook.ENV_TWINS, raising=False)
    assert hook.prepare_twins() is None, "unset variable: nothing happens"
    rec = hook.prepare_twins(force=True)
    assert rec["interposer"] is False and rec["inert_descriptor_published"] is False
    assert rec["error"] is None
    assert hook.prepare_twins(force=True) is rec, "once per process"


# held rows


class _Batch:
    def __init__(self, *reqs):
        self.reqs = list(reqs)


class _Sched:
    def __init__(
        self, running, chunked=None, queue=None, with_queue=True, pd_inflight=(), split=None
    ):
        self.running_batch = running
        self.split_prefill_batch = split
        self.chunked_req = chunked
        self.disagg_prefill_inflight_queue = list(pd_inflight)
        if with_queue:
            self.result_queue = queue if queue is not None else []


def test_held_requests_adds_the_requests_of_queued_results_after_the_batches():
    a, b, c, d, e = object(), object(), object(), object(), object()
    running, batch, queued = _Batch(a, b), _Batch(a, b), _Batch(a, b, c)
    sched = _Sched(running, chunked=d, queue=[(queued, "result")], pd_inflight=[e])
    held = attach_mod.held_requests(sched, batch)
    assert held == [(a, "running"), (b, "running"), (a, "batch"), (b, "batch"), (d, "chunked"), (e, "pd_inflight"), (a, "queued"), (b, "queued"), (c, "queued")]  # fmt: skip
    assert attach_mod.held_requests(sched, batch, queued=False) == held[:6], (
        "queued=False is the snapshot that misses a finishing request's row (the diagnostic)"
    )
    # The normal loop: an empty deque, or no attribute at all on an older object.
    assert attach_mod.held_requests(_Sched(running, queue=[]), batch)[-1] == (b, "batch")
    assert attach_mod.held_requests(_Sched(running, with_queue=False), batch)[-1] == (b, "batch")
    # A scheduler without a running batch (not yet formed) and a queue of bare batches.
    sched = _Sched(None, queue=[queued])
    assert attach_mod.held_requests(sched, _Batch(c)) == [(c, "batch"), (a, "queued"), (b, "queued"), (c, "queued")]  # fmt: skip


def test_held_requests_includes_pdmux_split_prefill_allocations_before_decode():
    running, split, current = object(), object(), object()
    sched = _Sched(_Batch(running), split=_Batch(split), queue=[])
    assert attach_mod.held_requests(sched, _Batch(current)) == [
        (running, "running"),
        (split, "split_prefill"),
        (current, "batch"),
    ]


def test_held_requests_includes_pd_decode_destination_owners():
    a, b, c, d = object(), object(), object(), object()
    sched = _Sched(_Batch())
    sched.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[SimpleNamespace(req=a)],
        pending_reqs=[SimpleNamespace(req=b)],
        retracted_queue=[],
        held_rebootstrap_reqs=[],
    )
    sched.disagg_decode_transfer_queue = SimpleNamespace(
        queue=[SimpleNamespace(req=c)],
        _deferred_releases=[(SimpleNamespace(req=d), 0, 0, 0, 1)],
    )
    sched.waiting_queue = [c]

    assert attach_mod.held_requests(sched, _Batch()) == [
        (a, "pd_decode_queue"),
        (b, "pd_decode_pending_reqs"),
        (c, "pd_decode_transfer"),
        (d, "pd_decode_deferred"),
        (c, "pd_decode_waiting"),
    ]


def test_streaming_session_row_keeps_one_owner_during_and_between_turns():
    class Mode:
        @staticmethod
        def is_extend():
            return True

        @staticmethod
        def is_decode():
            return False

    class Reporter:
        def __init__(self):
            self.reports = []

        def emit(self, report):
            self.reports.append(report)

    session = SimpleNamespace(session_id="s1", streaming=True)
    kv = SimpleNamespace(
        holds_kv=True,
        req_pool_idx=1,
        kv_allocated_len=3,
        kv_committed_len=3,
        cache_protected_len=1,
    )
    req = SimpleNamespace(rid="turn-2", session=session, kv=kv, extend_len=1)
    slot = SimpleNamespace(kv=kv)
    att = object.__new__(attach_mod.SglangAttachment)
    att.scheduler = _Sched(None)
    att.tree = SimpleNamespace(session=SimpleNamespace(slots={"s1": slot}))
    att.req_pool = SimpleNamespace(req_to_token=[[0, 0, 0], [5, 6, 7]])
    att.reporter = Reporter()
    att.seq = 4
    att.queued_rows = True
    att.request_incarnations = lambda request: f"{request.rid}#inc"
    att.stats = Counter()

    active = _Batch(req)
    active.forward_mode = Mode()
    active.extend_lens = [1]
    rows, owners = att._rows(active)
    assert [(row.request_id, row.slots, row.new_tokens) for row in rows] == [
        ("session:s1", (5, 6, 7), 1)
    ]
    assert owners == {1: "session:s1"}
    assert att.reporter.reports == []

    between = _Batch()
    between.forward_mode = Mode()
    between.extend_lens = []
    rows, owners = att._rows(between)
    assert [(row.request_id, row.slots, row.new_tokens) for row in rows] == [
        ("session:s1", (5, 6, 7), 0)
    ]
    assert owners == {1: "session:s1"}
    assert att.stats["snapshot_rows_session_only"] == 1


def test_separate_swa_row_frontier_matches_flashinfer_decode_window():
    class Mode:
        @staticmethod
        def is_extend():
            return False

        @staticmethod
        def is_decode():
            return True

    kv = SimpleNamespace(
        holds_kv=True,
        req_pool_idx=1,
        kv_allocated_len=640,
        kv_committed_len=640,
        cache_protected_len=256,
    )
    req = SimpleNamespace(rid="long", session=None, kv=kv)
    batch = _Batch(req)
    batch.forward_mode = Mode()
    att = object.__new__(attach_mod.SglangAttachment)
    att.scheduler = _Sched(None)
    att.tree = SimpleNamespace(session=SimpleNamespace(slots={}))
    att.req_pool = SimpleNamespace(req_to_token=[[0] * 640, list(range(640))])
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.seq = 4
    att.queued_rows = True
    att.request_incarnations = lambda request: request.rid
    att.stats = Counter()
    att.separate_swa = True
    att.facts = {"sliding_window_size": 511}

    rows, _ = att._rows(batch)

    assert rows[0].new_tokens == 1
    assert rows[0].swa_first_token == 128
    assert att.stats["swa_snapshot_first_token_max"] == 128


def test_speculative_mixed_decode_tail_uses_committed_len_as_pre_step_frontier():
    class Mode:
        @staticmethod
        def is_extend():
            return True

        @staticmethod
        def is_decode():
            return False

    def req(rid, row, allocated, committed):
        return SimpleNamespace(
            rid=rid,
            session=None,
            kv=SimpleNamespace(
                holds_kv=True,
                req_pool_idx=row,
                kv_allocated_len=allocated,
                kv_committed_len=committed,
                cache_protected_len=0,
            ),
        )

    prefill = req("prefill", 1, 6, 6)
    decode = req("decode", 2, 6, 5)
    batch = _Batch(prefill, decode)
    batch.forward_mode = Mode()
    batch.extend_lens = [2, 1]
    batch.decoding_reqs = [decode]

    att = object.__new__(attach_mod.SglangAttachment)
    att.scheduler = _Sched(None)
    att.tree = SimpleNamespace(session=SimpleNamespace(slots={}))
    att.req_pool = SimpleNamespace(
        req_to_token=[[0] * 6, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
    )
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.seq = 9
    att.queued_rows = True
    att.request_incarnations = lambda request: request.rid
    att.stats = Counter()
    att.speculative_ngram = False
    att.speculative_eagle3 = True
    att.ledger = SimpleNamespace(requests={})
    att._draft_rows = []

    rows, _ = att._rows(batch)

    by_id = {row.request_id: row for row in rows}
    assert by_id["prefill"].new_tokens == 2
    assert by_id["decode"].new_tokens == 0
    assert (
        attach_mod.pre_step_computed(by_id["decode"].kv_committed_len, by_id["decode"].new_tokens)
        == 5
    )


def test_speculative_decode_keeps_pending_pdmux_split_prefill_width():
    class Mode:
        @staticmethod
        def is_extend():
            return False

        @staticmethod
        def is_decode():
            return True

    def req(rid, row, allocated, committed):
        return SimpleNamespace(
            rid=rid,
            session=None,
            extend_len=allocated - committed,
            kv=SimpleNamespace(
                holds_kv=True,
                req_pool_idx=row,
                kv_allocated_len=allocated,
                kv_committed_len=committed,
                cache_protected_len=0,
            ),
        )

    pending = req("pending-prefill", 1, 6, 4)
    decoding = req("decode", 2, 6, 5)
    split = _Batch(pending)
    split.extend_lens = [2]
    batch = _Batch(decoding)
    batch.forward_mode = Mode()

    att = object.__new__(attach_mod.SglangAttachment)
    att.scheduler = _Sched(None, split=split)
    att.tree = SimpleNamespace(session=SimpleNamespace(slots={}))
    att.req_pool = SimpleNamespace(
        req_to_token=[[0] * 6, [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
    )
    att.reporter = SimpleNamespace(emit=lambda _report: None)
    att.seq = 10
    att.queued_rows = True
    att.request_incarnations = lambda request: request.rid
    att.stats = Counter()
    att.speculative_ngram = True
    att.speculative_eagle3 = False
    att.ledger = SimpleNamespace(requests={})
    att._pdmux_submitted_batches = set()

    rows, _ = att._rows(batch)

    by_id = {row.request_id: row for row in rows}
    assert by_id["pending-prefill"].new_tokens == 2
    assert by_id["decode"].new_tokens == 0


def _observe_rows(eng, rids):
    """A snapshot from the rows of ``rids`` only (the attachment's row sources), the tree and
    the free set, as ``Engine.observe`` builds it from every live request."""
    from kvguard.adapters.sglang.translate import (
        build_snapshot,
        free_slots_from_allocator,
        nodes_from_tree,
        rows_from_requests,
    )

    eng.seq += 1
    snap = build_snapshot(
        eng.seq,
        "kv",
        rows_from_requests([eng.reqs[r] for r in rids], eng.rows.req_to_token, eng.new_tokens),
        nodes_from_tree(eng.root),
        free_slots_from_allocator(eng.allocator),
    )
    eng.ledger.observe(snap)
    return snap


def _finishing_in_flight(rows_at_gap):
    """The overlap loop's finishing-by-length sequence on the fake engine of the translate
    tests: ``a`` and ``b`` prefill and decode together; at the gap step ``a``'s queued result
    will commit its last token, so ``update_running_batch`` excludes it from the decode batch
    while it still holds its row (``release_kv_cache`` runs when that result is processed,
    after the step's launch); then ``a`` is released with its KV checkpointed into the tree and
    ``b`` decodes on. ``rows_at_gap`` names the rows the snapshot of the gap step reads."""
    from collections import Counter

    from tests.unit.test_sglang_translate import Engine

    def kinds(eng):
        return dict(Counter(r.kind for r in eng.ledger.reporter.reports))

    eng = Engine(size=40, max_reqs=4, max_context_len=16)
    eng.prefill("a", 4)
    eng.prefill("b", 4)
    eng.step()
    eng.decode("a", "b")
    eng.step()
    assert kinds(eng) == {}
    # The gap step: the queued result finishes "a" by length; the batch is "b" alone.
    eng.decode("b")
    _observe_rows(eng, rows_at_gap)
    eng.forward()
    eng.complete()
    at_gap = kinds(eng)
    # The queued result is processed: "a" checkpoints its row into the tree and releases.
    eng.finish("a", checkpoint=True)
    eng.decode("b")
    eng.step()
    return at_gap, kinds(eng)


def test_a_finishing_request_held_through_the_queued_result_keeps_the_ledger_silent():
    at_gap, after = _finishing_in_flight(rows_at_gap=["a", "b"])
    assert at_gap == {} and after == {}


def test_a_snapshot_missing_the_finishing_request_reports_its_blocks_as_unmediated():
    # The attachment before held_requests read the running batch and the batch only.
    at_gap, after = _finishing_in_flight(rows_at_gap=["b"])
    assert at_gap["unmediated_free"] == 5, "a's five private slots left the listing without a free"
    assert at_gap["block_unaccounted"] == 1, "they are neither free nor in any state"
    assert "published_unreferenced" not in at_gap, "a's prefix was not cached in this scenario"
    assert after["unmediated_allocation"] == 5, "they re-enter as PUBLISHED without an alloc"
    assert set(after) == {"unmediated_free", "block_unaccounted", "unmediated_allocation"}


# walk states


def test_walk_states_keep_unchanged_objects_between_snapshots():
    """Only the slots whose state changed get new objects; the rest keep theirs (the ledger
    recognizes a kept object by identity), and the rules are those of ``slot_states``."""
    ws = attach_mod.WalkStates("kv")
    rows = [attach_mod.RowFacts("a", (5, 6, 9, 10), 4, 4, 2, new_tokens=1)]
    blocks, dup = ws.update(rows, tree_slots=[5, 6, 7], unlocked_slots=[7])
    assert dup == []
    first = {b.block: b for b in blocks}
    assert {s: (b.state, b.reserved) for s, b in first.items()} == {
        5: (BlockState.PUBLISHED, False),
        6: (BlockState.PUBLISHED, False),
        7: (BlockState.PUBLISHED, True),
        9: (BlockState.PRIVATE, False),
        10: (BlockState.PRIVATE, False),
    }
    # Next step: slot 7 leaves the tree, slot 11 enters it, slot 6 becomes unlocked, the row
    # grows by its own slot 12 and its own slot 10 moves into the tree (a checkpoint).
    rows = [attach_mod.RowFacts("a", (5, 6, 9, 10, 11, 12), 6, 6, 2, new_tokens=1)]
    blocks, dup = ws.update(rows, tree_slots=[5, 6, 10, 11], unlocked_slots=[6])
    second = {b.block: b for b in blocks}
    assert set(second) == {5, 6, 9, 10, 11, 12}
    assert second[5] is first[5] and second[9] is first[9], "unchanged slots keep their objects"
    assert second[6] is not first[6] and second[6].reserved, "a lock-state change is a new object"
    assert second[10] is not first[10] and second[10].state is BlockState.PUBLISHED
    assert second[11].state is BlockState.PUBLISHED and not second[11].reserved
    assert second[12].state is BlockState.PRIVATE
    # A slot the walk lists twice is reported once, sorted, and still gets one object.
    blocks, dup = ws.update(rows, tree_slots=[5, 6, 10, 11, 5, 11], unlocked_slots=[6])
    assert dup == [5, 11] and len({b.block for b in blocks}) == len(blocks)
    assert {b.block: b for b in blocks}[5] is first[5]


def test_walk_states_list_the_change_between_walks():
    """Between walks the view is page sets and the pool is listed as its change: the objects
    created this step and the pages that left, which together with the kept objects are the
    complete listing; a page that moves between the tree and a row is a change, not gone."""
    pool = "kv"
    ws = attach_mod.WalkStates(pool, 16)
    rows = [attach_mod.RowFacts("a", tuple(range(16, 48)) + tuple(range(64, 70)), 38, 38, 32)]
    tree = {1, 2, 5}
    own = {4}
    blocks, dup = ws.update(rows, tree_pages=tree, unlocked_pages={5}, own_pages=own, delta=True)
    assert dup == [] and {b.block for b in blocks} == {1, 2, 5, 4} and ws.gone == frozenset()
    kept = {
        b.block: b for b in ws.update(rows, tree_pages=tree, unlocked_pages={5}, own_pages=own)[0]
    }
    # No change: an empty delta, the complete form unchanged.
    blocks, _ = ws.update(rows, tree_pages=tree, unlocked_pages={5}, own_pages=own, delta=True)
    assert blocks == () and ws.gone == frozenset()
    # Page 5 leaves the tree, page 4 (the row's own) enters it, page 2 becomes unlocked, the
    # row's own page 7 appears: three new objects, one page gone.
    rows = [
        attach_mod.RowFacts(
            "a", tuple(range(16, 48)) + tuple(range(64, 80)) + tuple(range(112, 118)), 54, 54, 32
        )
    ]
    blocks, _ = ws.update(
        rows, tree_pages={1, 2, 4}, unlocked_pages={2}, own_pages={4, 7}, delta=True
    )
    changed = {b.block: b for b in blocks}
    assert set(changed) == {2, 4, 7} and ws.gone == frozenset({5})
    assert changed[2].reserved and changed[4].state is BlockState.PUBLISHED
    assert changed[7].state is BlockState.PRIVATE
    assert changed[2] is not kept[2] and changed[4] is not kept[4]
    complete = {
        b.block: b
        for b in ws.update(rows, tree_pages={1, 2, 4}, unlocked_pages={2}, own_pages={4, 7})[0]
    }
    assert set(complete) == {1, 2, 4, 7} and complete[1] is kept[1]
    # The snapshot form carries the delta fields.
    snap, _ = attach_mod.snapshot_from_walk(
        4,
        pool,
        rows,
        None,
        None,
        [9],
        ws,
        page_size=16,
        tree_pages={1, 2},
        unlocked_pages={2},
        own_pages={4, 7},
        delta=True,
    )
    # Page 4 moved from the tree to the row: a changed object, nothing gone.
    assert snap.delta_pools == frozenset({pool}) and snap.gone is None
    assert {b.block for b in snap.blocks} == {4} and snap.blocks[0].state is BlockState.PRIVATE
    assert snap.requests[0].blocks == (1, 2, 4, 7) and snap.free == {pool: frozenset({9})}


def test_paged_pages_taken_from_the_call():
    """The pages a paged allocator call takes, from its CPU lengths and returned slots: whole
    pages for ``alloc``, a page per page-aligned position of a row's extension, one for a
    decode row whose new token starts a page; mismatched slots give ``None``."""
    ps = 16
    taken = attach_mod.paged_pages_taken
    assert taken("alloc", {}, list(range(48, 64)) + list(range(96, 112)), ps) == [3, 6]
    # Two rows: [10, 40) opens pages at positions 16 and 32; [32, 33) opens one at 32;
    # [5, 12) opens none. Slots laid out row by row.
    rows = [(10, 40), (32, 33), (5, 12)]
    slots = []
    for p, s in rows:
        for pos in range(p, s):
            slots.append(
                {0: 100, 1: 7, 2: 9}[pos // ps] * ps + pos % ps
                if (p, s) == (10, 40)
                else 50 * ps + pos % ps
                if (p, s) == (32, 33)
                else 60 * ps + pos % ps
            )
    bound = {"prefix_lens_cpu": [p for p, _ in rows], "seq_lens_cpu": [s for _, s in rows]}
    assert taken("alloc_extend", bound, slots, ps) == [7, 9, 50]
    assert taken("alloc_extend", bound, slots[:-1], ps) is None
    assert attach_mod.paged_call_known("alloc_extend", bound)
    assert not attach_mod.paged_call_known("alloc_extend", {"prefix_lens": [1]})
    # Decode: rows whose new token's position (seq - 1) is page-aligned open a page.
    bound = {"seq_lens_cpu": [17, 18, 33]}
    assert taken("alloc_decode", bound, [8 * ps, 9 * ps + 1, 11 * ps], ps) == [8, 11]
    assert taken("alloc_decode", bound, [8 * ps], ps) is None
    assert taken("other", {}, [], ps) is None


def test_paged_allocator_wrapper_derives_and_audits():
    """The wrapper takes the derived pages on ordinary calls and reads the free list before
    and after on the audited calls (every ``audit_every``-th), reporting a disagreement."""
    from collections import Counter

    from kvguard.core import Ledger, PoolGeometry, Reporter

    ps = 16

    class Alloc:
        page_size = ps

        def __init__(self):
            self.free = list(range(1, 64))
            self.reads = 0

        def get_all_free_pages(self):
            self.reads += 1
            return list(self.free)

        def alloc(self, need_size):
            pages = self.free[: need_size // ps]
            del self.free[: need_size // ps]
            return [p * ps + o for p in pages for o in range(ps)]

        lie = False  # return a slot of page 50 while taking another page off the list

        def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
            out = []
            for s, last in zip(seq_lens_cpu, last_loc, strict=True):
                if (s - 1) % ps == 0:
                    page = self.free.pop(0)
                    out.append((50 if self.lie else page) * ps)
                else:
                    out.append(last + 1)
            return out

        def _release_page_ids(self, *page_ids):
            for t in page_ids:
                self.free.extend(t)

    alloc = Alloc()
    reporter = Reporter(fail_fast=False)
    ledger = Ledger(reporter)
    ledger.register_pool(PoolGeometry("kv", 64, ps, reserved_blocks=frozenset({0})))
    me = SimpleNamespace(
        ledger=ledger,
        reporter=reporter,
        stats=Counter(),
        seq=1,
        audit_every=4,
        early_complete=False,
        _reap_pd_reads=lambda: None,
        _cache_pools=lambda: ("kv",),
        _trace_pd_source_release=lambda pages: None,
    )
    attach_mod.SglangAttachment._install_paged_allocator(me, alloc)
    out = alloc.alloc(32)  # call 0: audited (0 % 4 == 0): two free-list reads
    assert out[0] == ps and alloc.reads == 2 and me.stats["alloc_audited"] == 1
    assert (
        ledger.block("kv", 1).state.name == "PRIVATE"
        and ledger.block("kv", 2).state.name == "PRIVATE"
    )
    alloc.alloc_decode([17, 18], [17, 18], [1 * ps + 15, 2 * ps + 1])  # call 1: derived, no read
    assert (
        alloc.reads == 2 and me.stats["alloc_pages_derived"] == 1 and me.stats["alloc_calls"] == 2
    )
    assert ledger.block("kv", 3).state.name == "PRIVATE"
    alloc.alloc_decode([18], [18], [3 * ps])  # no new page
    assert me.stats["alloc_decode_without_new_page"] == 1 and alloc.reads == 2
    # The audited call (every fourth) reads the free list before and after: page 40 moved to
    # the front of the list is what the call takes, and the derivation agrees.
    alloc.alloc_decode([33], [33], [3 * ps + 15])  # call 2: derived
    alloc.alloc_decode([49], [49], [4 * ps + 15])  # call 3: derived
    alloc.free.remove(40)
    alloc.free.insert(0, 40)
    alloc.alloc_decode([65], [65], [5 * ps + 15])  # call 4: audited
    assert alloc.reads == 4 and me.stats["alloc_audited"] == 2 and reporter.reports == []
    assert ledger.block("kv", 40).state.name == "PRIVATE"
    alloc._release_page_ids([40])
    assert ledger.block("kv", 40).state.name == "FREE" and me.stats["free_pages"] == 1
    # A disagreement on an audited call is reported and the free list wins: the fake returns a
    # slot of page 50 while taking page 41 off its list.
    for _ in range(3):
        alloc.alloc_decode([18], [18], [6 * ps])  # no new page: not counted as a call
    alloc.free.remove(41)
    alloc.free.insert(0, 41)
    me.stats["alloc_calls"] = 8  # the next counted call is audited
    alloc.lie = True
    alloc.alloc_decode([81], [81], [7 * ps + 15])
    assert [r.kind for r in reporter.reports] == ["alloc_pages_drift"]
    assert me.stats["alloc_pages_drift"] == 1 and me.stats["alloc_calls"] == 9
    assert ledger.block("kv", 41).state.name == "PRIVATE"
    assert ledger.block("kv", 50).state.name == "FREE"


def test_row_pages_read_one_slot_per_page():
    from kvguard.adapters.sglang.translate import page_set, row_pages

    slots = tuple(range(16, 48)) + tuple(range(64, 80)) + tuple(range(112, 118))
    for start, stop in ((0, 54), (5, 20), (16, 16), (31, 33), (48, 54), (53, 54), (0, 1)):
        assert row_pages(slots, start, stop, 16) == page_set(slots[start:stop], 16), (start, stop)
    assert row_pages(slots, 3, 9, 1) == set(slots[3:9])


def test_snapshot_from_walk_with_kept_states_equals_the_one_shot_form():
    rows = [attach_mod.RowFacts("a", (5, 6, 9), 3, 3, 2, new_tokens=1)]
    ws = attach_mod.WalkStates("kv")
    one_shot, _ = attach_mod.snapshot_from_walk(1, "kv", rows, [5, 6, 7], [7], [8])
    kept, _ = attach_mod.snapshot_from_walk(1, "kv", rows, [5, 6, 7], [7], [8], ws)
    assert set(one_shot.blocks) == set(kept.blocks) and one_shot.requests == kept.requests
    again, _ = attach_mod.snapshot_from_walk(2, "kv", rows, [5, 6, 7], [7], [8], ws)
    assert all(
        x is y
        for x, y in zip(
            sorted(kept.blocks, key=lambda b: b.block),
            sorted(again.blocks, key=lambda b: b.block),
            strict=True,
        )
    )


# twin readback


def test_twin_counts_account_cumulative_counter_and_entry_checks():
    """The in-kernel counter is cumulative over the run; a step's reports are the difference to
    the previous reading, the record is kept with the first reporting step, and the published
    entry count is checked against the extraction's (shifted by the injected fault)."""
    c = attach_mod.TwinCounts()
    assert c.account(1, {"site": "decode_graph_replay"}, 700, 0, [0, 0, 0, 0], 700, None) == 0
    assert c.account(2, {"site": "decode_graph_replay"}, 732, 3, [7, 0, 5, 1], 732, None) == 3
    assert c.account(3, None, None, 3, [7, 0, 5, 1], 732, None) == 0, "no new reports"
    assert c.account(4, {"site": "x"}, 740, 5, [7, 0, 5, 1], 740, None) == 2
    assert c.violations_total == 5 and [v["seq"] for v in c.violations] == [2, 4]
    assert c.violations[0]["count"] == 3 and c.violations[0]["kind"] == 1
    assert c.entries_checked == 3 and c.entries_mismatch == 0
    assert [e["reports"] for e in c.trace] == [0, 3, 0, 2] and c.trace[2]["site"] is None
    # A published count that is not the extraction's is a mismatch, unless the fault explains it.
    assert c.account(5, {"site": "x"}, 750, 5, [7, 0, 5, 1], 749, None) == 0
    assert c.entries_mismatch == 1
    c.account(6, {"site": "x"}, 750, 5, [7, 0, 5, 1], 749, "short_entries")
    assert c.entries_mismatch == 1, "short_entries publishes one short by design"


def test_walk_states_mask_form_equals_the_set_form():
    """The mask form of ``WalkStates.update`` (the view as boolean arrays over the page ids,
    the adapter's derivation between walks) lists the same objects, states and gone pages as the
    set form given the same view, through a random sequence of tree, lock and own-page changes,
    beam-shared and pending pages, interleaved set-form (audited) steps and complete listings."""
    import random

    import numpy as np

    pool = "kv"
    n = 64
    rng = random.Random(7)
    ws_set = attach_mod.WalkStates(pool, 16)
    ws_mask = attach_mod.WalkStates(pool, 16)
    kept_set: dict[int, object] = {}
    kept_mask: dict[int, object] = {}

    def masks(tree, unlocked):
        t = np.zeros(n, dtype=np.bool_)
        u = np.zeros(n, dtype=np.bool_)
        t[list(tree)] = True
        u[list(unlocked)] = True
        return t, u

    def view(blocks):
        return {b.block: (b.state, b.reserved, b.refs) for b in blocks}

    for step in range(60):
        tree = set(rng.sample(range(1, n), rng.randint(0, 40)))
        unlocked = {p for p in tree if rng.random() < 0.5}
        own = set(rng.sample(range(1, n), rng.randint(0, 12))) - tree
        shared = (
            {p * 16 for p in rng.sample(range(1, n), rng.randint(0, 2))} if step % 7 == 3 else set()
        )
        pending = (
            {p * 16 + 3 for p in rng.sample(range(1, n), rng.randint(0, 2))}
            if step % 5 == 4
            else set()
        )
        delta = step % 9 != 8
        audited = step % 11 == 10
        t, u = masks(tree, unlocked)
        blocks_set, _ = ws_set.update(
            [],
            shared_slots=shared,
            reserved_slots=pending,
            tree_pages=set(tree),
            unlocked_pages=set(unlocked),
            own_pages=set(own),
            delta=delta,
        )
        if audited:  # the mask-form states take an audited (set-form) step
            blocks_mask, _ = ws_mask.update(
                [],
                shared_slots=shared,
                reserved_slots=pending,
                tree_pages=set(tree),
                unlocked_pages=set(unlocked),
                own_pages=set(own),
                delta=delta,
            )
        else:
            blocks_mask, _ = ws_mask.update(
                [],
                shared_slots=shared,
                reserved_slots=pending,
                own_pages=set(own),
                delta=delta,
                tree_mask=t,
                unlocked_mask=u,
            )
        assert view(blocks_set) == view(blocks_mask), step
        assert ws_set.gone == ws_mask.gone, step
        assert set(ws_set.changed) == set(ws_mask.changed), step
        # The kept objects: a page whose state did not change keeps its object in both forms.
        for b, obj in ws_mask.tree.items():
            if b in kept_mask and b not in ws_mask.changed:
                assert obj is kept_mask[b], (step, b)
        for b, obj in ws_set.tree.items():
            if b in kept_set and b not in ws_set.changed:
                assert obj is kept_set[b], (step, b)
        kept_set = {**ws_set.tree, **ws_set.private}
        kept_mask = {**ws_mask.tree, **ws_mask.private}
        assert view(kept_set.values()) == view(kept_mask.values()), step
    # The complete form after the sequence agrees too.
    full_set, _ = ws_set.update(
        [], tree_pages=set(tree), unlocked_pages=set(unlocked), own_pages=set(own)
    )
    full_mask, _ = ws_mask.update([], own_pages=set(own), tree_mask=t, unlocked_mask=u)
    assert view(full_set) == view(full_mask)


def test_walk_states_at_page_size_sixteen_list_pages():
    """At a page size above one the walk's slots stand for their pages: a node holds whole
    pages, a row's own range its pages, and the request maps pages."""
    ws = attach_mod.WalkStates("kv", 16)
    rows = [
        attach_mod.RowFacts(
            "a", tuple(range(16, 48)) + tuple(range(64, 70)), 38, 38, 32, new_tokens=1
        )
    ]
    blocks, dup = ws.update(
        rows,
        tree_slots=list(range(16, 48)) + list(range(80, 96)),
        unlocked_slots=list(range(80, 96)),
    )
    assert dup == []
    assert {b.block: (b.state, b.reserved) for b in blocks} == {
        1: (BlockState.PUBLISHED, False),
        2: (BlockState.PUBLISHED, False),
        5: (BlockState.PUBLISHED, True),
        4: (BlockState.PRIVATE, False),
    }
    snap, _ = attach_mod.snapshot_from_walk(
        3, "kv", rows, list(range(16, 48)), [], [6, 7], ws, page_size=16
    )
    assert snap.requests[0].blocks == (1, 2, 4) and snap.requests[0].num_tokens == 38
    assert snap.free == {"kv": frozenset({6, 7})}
