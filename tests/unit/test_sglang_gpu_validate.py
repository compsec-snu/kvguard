"""The SGLang adapter's GPU-validation path without a GPU: ``GpuValidation.run_step`` over fake
validators and a fake mirror, driven by a real ledger with an SGLang-like pool (page size 1).

Checked: the launch order around the host dispatches (the mirror must show the step's write
before the attention validator and not before the slot-map validator), the row, owner and
token arrays for decode (padded lanes) and extend (ragged and paged, padded tokens), the
readback-to-report path, and the two synthetic faults.
"""

from __future__ import annotations

import pytest
from kvguard.adapters.sglang import attach as A
from kvguard.adapters.sglang.translate import (
    POOL,
    RowFacts,
    build_decode_dispatches,
    build_extend_dispatches,
    build_snapshot,
    geometry,
)
from kvguard.core import Ledger, Reporter
from kvguard.gpu.validate import PADDING_ROW, OwnerIds, Result

SIZE = 64  # slots 1..64; slot 0 reserved


class FakeTensor:
    """What the validators need of a device tensor, over a Python list."""

    def __init__(self, values, elem=4, ptr=0x1000):
        self.values = [int(v) for v in values]
        self.elem = elem
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr

    def element_size(self):
        return self.elem

    def tolist(self):
        return list(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item], self.elem, self.ptr)
        return self.values[item]

    def __setitem__(self, i, v):
        self.values[i] = int(v)

    def __len__(self):
        return len(self.values)


class FakeValidators:
    """Records every launch; ``completed`` hands back whatever the test scripted."""

    def __init__(self):
        self.calls = []
        self.targets = []
        """Per launch, the mapping arrays the validators were given (row starts and page
        tables; the expected write slots)."""
        self.launches = 0
        self.gpu_ms_total = 0.0
        self.scripted = []

    def validate_token_csr(
        self, mirror, indptr, indices, rows, row_owner, row_start, row_end, pages, stream, tag
    ):
        self.launches += 1
        self.calls.append(("csr", tag, rows, list(row_owner), list(row_end), stream))
        self.targets.append((tag, list(row_start), [tuple(p) for p in pages]))

    def validate_slot_map(
        self, mirror, slots, elem, n_tokens, tok_owner, stream, tag, expected=None
    ):
        self.launches += 1
        self.calls.append(("slot_map", tag, n_tokens, list(tok_owner), elem, stream))
        self.targets.append((tag, None if expected is None else list(expected)))

    def completed(self):
        out, self.scripted = self.scripted, []
        return out

    def drain(self):
        return self.completed()


class FakeMirror:
    """Records each sync with the ledger's dirty set at that moment (what a real sync uploads)."""

    def __init__(self, ledger, validators):
        self.ledger = ledger
        self.owners = OwnerIds()
        self.updates = 0
        self.blocks_updated = 0
        self.v = validators

    def sync(self, stream):
        dirty = sorted(self.ledger.drain_dirty(POOL))
        self.v.calls.append(("sync", dirty, stream))
        self.updates += 1
        self.blocks_updated += len(dirty)
        return len(dirty)

    def set_words(self, block, state, filled, pending, stream):
        self.v.calls.append(("set_words", block, (state, filled, pending), stream))


@pytest.fixture
def ledger():
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(geometry(SIZE, 1))
    return lg


def observe(ledger, seq, rows, free):
    ledger.observe(build_snapshot(seq, POOL, rows, [], free))


def gv(ledger, **kw):
    v = FakeValidators()
    return A.GpuValidation(
        ledger, ledger.reporter, validators=v, mirror=FakeMirror(ledger, v), **kw
    ), v


def plan_of(seq, mode, descs, **kw):
    defaults = {
        "graph": False,
        "prefix": [],
        "use_ragged": False,
        "indptr": [],
        "indices": [],
        "loc": [],
    }
    defaults.update(kw)
    n = len(defaults["req_ids"])
    p = A.StepPlan(
        seq=seq,
        mode=mode,
        n=n,
        n_pad=defaults.pop("n_pad", n),
        descs=list(descs),
        buffers=A.DeviceBuffers(
            FakeTensor(defaults["indptr"]),
            FakeTensor(defaults["indices"]),
            FakeTensor([1] * max(len(defaults["indptr"]) - 1, 0)),
            FakeTensor(defaults["loc"], elem=8),
        ),
        **defaults,
    )
    return p


def dispatcher(ledger, seq, dids):
    import dataclasses

    def dispatch(d):
        d = dataclasses.replace(d, snapshot_seq=seq)
        ledger.dispatch(d)
        dids.append(d.dispatch_id)

    return dispatch


# decode


def decode_world(ledger, n_pad=None):
    """Two decode rows: A holds slots 1..3 (positions 0..2 written), B slots 4..5; this step
    writes position 2 (A) and position 1 (B) and reads [0, 3) / [0, 2)."""
    ledger.allocate(POOL, [1, 2, 3, 4, 5])
    for s in (1, 2, 4):  # the positions written before this step
        ledger.block(POOL, s).filled = 1
    rows = [
        RowFacts("A", (1, 2, 3), 3, 3, 0, new_tokens=1),
        RowFacts("B", (4, 5), 2, 2, 0, new_tokens=1),
    ]
    observe(ledger, 1, rows, list(range(6, SIZE + 1)))
    ledger.drain_dirty(POOL)
    indptr, indices = [0, 3, 5], [1, 2, 3, 4, 5]
    lanes = n_pad or 2
    if lanes > 2:
        indptr = indptr + [5] * (lanes - 2)
    loc = [3, 5] + [0] * (lanes - 2)
    descs = build_decode_dispatches(
        1, POOL, ["A", "B"], [3, 2], indptr, indices, loc, num_rows_after_padding=lanes
    )
    return plan_of(
        1,
        "decode",
        descs,
        req_ids=["A", "B"],
        seq_lens=[3, 2],
        indptr=indptr,
        indices=indices,
        loc=loc,
        n_pad=lanes,
    )


def test_decode_order_slot_map_before_the_write_and_csr_after(ledger):
    g, v = gv(ledger)
    plan = decode_world(ledger)
    dids = []
    g.run_step(plan, dispatcher(ledger, 1, dids), stream=7)
    kinds = [c[0] for c in v.calls]
    assert kinds == ["sync", "slot_map", "sync", "csr"], v.calls
    assert dids == ["write@1", "attn@1"], "host order: the write dispatch between the launches"
    assert v.calls[0][1] == [], "nothing dirty before the step"
    assert v.calls[2][1] == [3, 5], "the second sync carries the slots the write dispatch marked"
    a, b = g.mirror.owners.id_of("A"), g.mirror.owners.id_of("B")
    assert v.calls[1][1:] == ("write@1", 2, [a, b], 8, 7)
    assert v.calls[3][1:] == ("attn@1", 2, [a, b], [3, 2], 7)
    assert ledger.reporter.reports == []
    assert set(ledger.epochs) == {"write@1", "attn@1"}
    assert g.stats["gpu_validator_launches"] == 2 and v.launches == 2


def test_decode_padded_lanes_are_padding_rows_and_tokens(ledger):
    g, v = gv(ledger)
    plan = decode_world(ledger, n_pad=4)
    g.run_step(plan, dispatcher(ledger, 1, []), stream=0)
    a, b = g.mirror.owners.id_of("A"), g.mirror.owners.id_of("B")
    slot_map = next(c for c in v.calls if c[0] == "slot_map")
    assert slot_map[2] == 4 and slot_map[3] == [a, b, PADDING_ROW, PADDING_ROW]
    csr = next(c for c in v.calls if c[0] == "csr")
    assert csr[2] == 4
    assert csr[3] == [a, b, PADDING_ROW, PADDING_ROW] and csr[4] == [3, 2, 0, 0]
    assert ledger.reporter.reports == [], "the padded lanes name slot 0 (reserved)"


# extend


def extend_world(ledger, use_ragged, padded_tokens=0):
    """Two prefill rows: A has a cached prefix of 2 positions (slots 1, 2, filled) and writes 2
    new tokens into slots 3, 4; B has no prefix and writes 3 tokens into slots 5, 6, 7."""
    ledger.allocate(POOL, [1, 2, 3, 4, 5, 6, 7])
    for s in (1, 2):  # the cached prefix
        ledger.block(POOL, s).filled = 1
    rows = [
        RowFacts("A", (1, 2, 3, 4), 4, 4, 0, new_tokens=2),
        RowFacts("B", (5, 6, 7), 3, 3, 0, new_tokens=3),
    ]
    observe(ledger, 1, rows, list(range(8, SIZE + 1)))
    ledger.drain_dirty(POOL)
    if use_ragged:
        indptr, indices = [0, 2, 2], [1, 2]
    else:
        indptr, indices = [0, 4, 7], [1, 2, 3, 4, 5, 6, 7]
    loc = [3, 4, 5, 6, 7] + [0] * padded_tokens
    descs = build_extend_dispatches(
        1, POOL, ["A", "B"], [4, 3], [2, 0], indptr, indices, loc, use_ragged=use_ragged
    )
    return plan_of(
        1,
        "extend",
        descs,
        req_ids=["A", "B"],
        seq_lens=[4, 3],
        prefix=[2, 0],
        use_ragged=use_ragged,
        indptr=indptr,
        indices=indices,
        loc=loc,
    )


def test_extend_ragged_reads_the_prefix_before_the_write(ledger):
    g, v = gv(ledger)
    plan = extend_world(ledger, use_ragged=True)
    dids = []
    g.run_step(plan, dispatcher(ledger, 1, dids), stream=3)
    assert [c[0] for c in v.calls] == ["sync", "csr", "slot_map"]
    assert dids == ["attn@1", "write@1"]
    a, b = g.mirror.owners.id_of("A"), g.mirror.owners.id_of("B")
    assert v.calls[1][1:] == ("attn@1", 2, [a, b], [2, 0], 3), "row ends are the prefix lengths"
    assert v.calls[2][1:] == ("write@1", 5, [a, a, b, b, b], 8, 3)
    assert ledger.reporter.reports == []


def test_extend_paged_writes_first_then_reads_whole_rows_with_padded_tokens(ledger):
    g, v = gv(ledger)
    plan = extend_world(ledger, use_ragged=False, padded_tokens=3)
    dids = []
    g.run_step(plan, dispatcher(ledger, 1, dids), stream=0)
    assert [c[0] for c in v.calls] == ["sync", "slot_map", "sync", "csr"]
    assert dids == ["write@1", "attn@1"]
    a, b = g.mirror.owners.id_of("A"), g.mirror.owners.id_of("B")
    assert v.calls[1][1:] == (
        "write@1",
        8,
        [a, a, b, b, b, PADDING_ROW, PADDING_ROW, PADDING_ROW],
        8,
        0,
    )
    assert v.calls[2][1] == [3, 4, 5, 6, 7], (
        "the written slots reach the mirror before the CSR check"
    )
    assert v.calls[3][1:] == ("attn@1", 2, [a, b], [4, 3], 0), (
        "real rows only: the padding row is the slot map's"
    )
    assert ledger.reporter.reports == []


# readback


def test_readback_turns_device_results_into_reports(ledger):
    g, v = gv(ledger)
    plan = decode_world(ledger)
    g.run_step(plan, dispatcher(ledger, 1, []), stream=0)
    v.scripted = [
        Result("write@1", 0, 0, 0, 0, 0, gpu_ms=0.01),
        Result("attn@1", 2, 3, 1, 0, 4, gpu_ms=0.02),
    ]
    summary = g.readback(1)
    assert summary == {"gpu_results": 2, "gpu_reports": 1, "gpu_violations": 2, "gpu_ms": 0.03}
    (rep,) = ledger.reporter.reports
    assert rep.kind == "access_free_block" and rep.site == "gpu-validate:attn@1" and rep.block == 4
    assert (
        g.stats["gpu_reports"] == 1
        and g.stats["gpu_violations"] == 2
        and g.stats["gpu_results"] == 2
    )
    assert g.timings["gpu_validate_device_ms"] == pytest.approx(0.03)
    assert g.trace[0]["seq"] == 1 and g.trace[0]["gpu_reports"] == 1 and g.trace[0]["rows"] == 2
    assert g.violations[0]["name"] == "access_free_block" and g.violations[0]["row"] == 1
    assert g.outcome() == "reports_on_clean_run"
    rec = g.record()
    assert rec["device_violations_total"] == 2 and rec["launches"] == 2 and rec["steps"] == 1
    assert g.drain() == {"gpu_results": 0, "gpu_reports": 0, "gpu_violations": 0, "gpu_ms": 0.0}


def test_clean_outcome_and_no_launches(ledger):
    g, _ = gv(ledger)
    assert g.outcome() == "no_launches"
    g.run_step(decode_world(ledger), dispatcher(ledger, 1, []), stream=0)
    g.readback(1)
    assert g.outcome() == "clean"


# faults


def test_unknown_fault_is_rejected(ledger):
    with pytest.raises(ValueError):
        gv(ledger, fault="nonsense")


def test_mirror_free_fault_after_the_write_sync_before_the_csr(ledger):
    g, v = gv(ledger, fault="mirror_free", fault_step=1)
    plan = decode_world(ledger)
    g.run_step(plan, dispatcher(ledger, 1, []), stream=5)
    assert [c[0] for c in v.calls] == ["sync", "slot_map", "sync", "set_words", "csr"]
    assert v.calls[3][1:] == (3, (0, 0, 0), 5), "row 0's last page (slot 3, written this step)"
    assert g.fault_done and g.fault_info["block"] == 3 and "slot 3" in g.fault_note
    assert ledger.block(POOL, 3).filled == 1, "the ledger's shadow is untouched"
    assert ledger.reporter.reports == [], "the host sees nothing"
    v.scripted = [Result("attn@1", 1, 3, 0, 2, 3)]
    g.readback(1)
    assert g.outcome() == "detected"


def test_mirror_free_fault_is_not_armed_before_its_step_nor_on_ragged_extend(ledger):
    g, v = gv(ledger, fault="mirror_free", fault_step=9)
    g.run_step(decode_world(ledger), dispatcher(ledger, 1, []), stream=0)
    assert not g.fault_done and "set_words" not in [c[0] for c in v.calls]
    assert g.outcome() == "fault_not_injected"
    lg2 = Ledger(Reporter(fail_fast=False))
    lg2.register_pool(geometry(SIZE, 1))
    g2, _ = gv(lg2, fault="mirror_free", fault_step=1)
    g2.run_step(extend_world(lg2, use_ragged=True), dispatcher(lg2, 1, []), stream=0)
    assert not g2.fault_done, (
        "the ragged attention reads the prefix only; no page is written and read"
    )


def test_table_foreign_fault_rewrites_entry_0_with_row_1s_private_page(ledger):
    g, _ = gv(ledger, fault="table_foreign", fault_step=1)
    ledger.allocate(POOL, [1, 2, 3, 4, 5])
    rows = [
        RowFacts("A", (1, 2, 3), 3, 3, 0, new_tokens=1),
        RowFacts("B", (4, 5), 2, 2, 0, new_tokens=1),
    ]
    observe(ledger, 1, rows, list(range(6, SIZE + 1)))
    indptr, indices = FakeTensor([0, 3, 5]), FakeTensor([1, 2, 3, 4, 5])
    assert g.inject_table_fault(indptr, indices, ["A", "B"], seq=1) is True
    assert indices.tolist() == [5, 2, 3, 4, 5], "row 1's last page, private to B"
    assert g.fault_info == {"step": 1, "row": 0, "request": "A", "old": 1, "block": 5, "owner": "B"}
    assert g.inject_table_fault(indptr, indices, ["A", "B"], seq=2) is False, "once"
    # The host descriptor built from the overwritten table disagrees with A's mapping.
    descs = build_decode_dispatches(
        1, POOL, ["A", "B"], [3, 2], indptr.tolist(), indices.tolist(), [3, 5]
    )
    for d in descs:
        ledger.dispatch(d)
    assert "mapping_mismatch" in [r.kind for r in ledger.reporter.reports]


def test_table_foreign_needs_two_rows_and_a_private_page(ledger):
    g, _ = gv(ledger, fault="table_foreign", fault_step=1)
    assert g.inject_table_fault(FakeTensor([0, 3]), FakeTensor([1, 2, 3]), ["A"], seq=1) is False
    # Row 1's pages are published (shared): nothing private to overwrite with.
    from kvguard.adapters.sglang.translate import NodeFacts

    ledger.allocate(POOL, [1, 4, 5])
    rows = [
        RowFacts("A", (1,), 1, 1, 0, new_tokens=1),
        RowFacts("B", (4, 5), 2, 2, 2, new_tokens=0),
    ]
    ledger.observe(
        build_snapshot(1, POOL, rows, [NodeFacts(1, (4, 5), 1)], list(range(6, SIZE + 1)))
    )
    assert (
        g.inject_table_fault(FakeTensor([0, 1, 3]), FakeTensor([1, 4, 5]), ["A", "B"], seq=1)
        is False
    )
    assert g.errors["table_foreign_no_private_page"] == 1 and not g.fault_done


# the pure helper


def test_validator_targets_from_the_plan_and_the_mapping():
    """The mapping arrays the device validators check the device tables against: per row the
    read start and the ledger's page table, per write token the slot the mapping gives the
    position the write descriptor names; padding lanes and tokens get no expectation, nor does
    a position whose page is a hole."""
    from kvguard.adapters.sglang.translate import build_decode_dispatches, build_extend_dispatches

    ps = 4
    pages = {"A": (3, 4), "B": (5, None)}
    # Decode: A at seq_len 6 (page 4, offset 1), B at 5 (page None: hole); two padding lanes.
    write, attn = build_decode_dispatches(
        1,
        POOL,
        ["A", "B"],
        [6, 5],
        [0, 2, 3, 3, 3],
        [20, 17, 21],
        [17, 21, 0, 0],
        num_rows_after_padding=4,
        page_size=ps,
        read_starts=[4, 0],
    )
    decode = A.StepPlan(
        1,
        "decode",
        True,
        2,
        4,
        ["A", "B"],
        [6, 5],
        [],
        False,
        [0, 2, 3, 3, 3],
        [20, 17, 21],
        [17, 21, 0, 0],
        [write, attn],
        read_starts=[4, 0],
    )
    starts, tables, expected = A.validator_targets(decode, pages.__getitem__, ps)
    assert starts == [4, 0, 0, 0] and tables == [(3, 4), (5, None)]
    assert expected == [4 * ps + 1, None, None, None]
    # Extend: A writes positions [2, 4) of pages (3, 4); B writes [0, 2) of page 5; one
    # padded token.
    write, attn = build_extend_dispatches(
        2,
        POOL,
        ["A", "B"],
        [4, 2],
        [2, 0],
        [0, 2, 2],
        [12, 13],
        [14, 15, 20, 21, 0],
        use_ragged=True,
        page_size=ps,
    )
    extend = A.StepPlan(
        2,
        "extend",
        False,
        2,
        2,
        ["A", "B"],
        [4, 2],
        [2, 0],
        True,
        [0, 2, 2],
        [12, 13],
        [14, 15, 20, 21, 0],
        [write, attn],
    )
    starts, tables, expected = A.validator_targets(extend, pages.__getitem__, ps)
    assert starts == [0, 0] and expected == [14, 15, 20, 21, None]
    # A request the ledger does not know maps nothing: no expectation, an empty page table.
    starts, tables, expected = A.validator_targets(extend, lambda rid: (), ps)
    assert tables == [(), ()] and expected == [None] * 5


def test_validator_rows_decode_and_extend():
    owner = {"A": 1, "B": 2}.__getitem__
    decode = A.StepPlan(
        1,
        "decode",
        True,
        2,
        4,
        ["A", "B"],
        [3, 2],
        [],
        False,
        [0, 3, 5, 5, 5],
        [1, 2, 3, 4, 5],
        [3, 5, 0, 0],
        [],
    )
    assert A.validator_rows(decode, owner) == (
        (4, [1, 2, PADDING_ROW, PADDING_ROW], [3, 2, 0, 0]),
        (4, [1, 2, PADDING_ROW, PADDING_ROW]),
    )
    ragged = A.StepPlan(
        1,
        "extend",
        False,
        2,
        2,
        ["A", "B"],
        [4, 3],
        [2, 0],
        True,
        [0, 2, 2],
        [1, 2],
        [3, 4, 5, 6, 7],
        [],
    )
    assert A.validator_rows(ragged, owner) == ((2, [1, 2], [2, 0]), (5, [1, 1, 2, 2, 2]))
    paged = A.StepPlan(
        1,
        "extend",
        True,
        2,
        2,
        ["A", "B"],
        [4, 3],
        [2, 0],
        False,
        [0, 4, 7],
        [1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 0],
        [],
    )
    assert A.validator_rows(paged, owner) == (
        (2, [1, 2], [4, 3]),
        (6, [1, 1, 2, 2, 2, PADDING_ROW]),
    )


def test_run_step_rejects_non_int32_page_list(ledger):
    g, _ = gv(ledger)
    plan = decode_world(ledger)
    plan.buffers.indices = FakeTensor([1, 2, 3, 4, 5], elem=8)
    with pytest.raises(ValueError, match="int32"):
        g.run_step(plan, dispatcher(ledger, 1, []), stream=0)


def test_env_names_and_exports():
    assert A.ENV_GPU_VALIDATE == "KVGUARD_SGLANG_GPU_VALIDATE"
    assert A.ENV_GPU_FAULT == "KVGUARD_SGLANG_GPU_FAULT"
    assert set(A.GPU_FAULTS) == {"mirror_free", "table_foreign"}
    assert "GpuValidation" in A.__all__ and "validator_rows" in A.__all__
