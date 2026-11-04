"""The SGLang adapter's pure translation rules (no SGLang or torch needed).

The fakes carry the attribute names the adapter reads from SGLang 0.5.22: ``Req.rid`` and
``Req.kv`` (``req_pool_idx``, ``kv_allocated_len``, ``kv_committed_len``, ``cache_protected_len``),
``TreeNode.value``/``lock_ref``/``children``/``id``, and the allocator's ``free_pages``,
``release_pages`` and ``free_group``. ``Engine`` replays the stock path's host-side bookkeeping
(page_size 1, radix cache, FlashInfer) step by step and feeds a ledger the way an attachment would:
wrapped ``alloc``/``free``, a snapshot after the batch is prepared, descriptors in kernel order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from kvguard.adapters.sglang.translate import (
    PADDING_SLOT,
    SUPPORT_BOUNDARY,
    NodeFacts,
    RowFacts,
    allocated_slots,
    beam_member_rows,
    beam_pending_orphan_slots,
    build_decode_dispatches,
    build_dflash_materialize_dispatch,
    build_eagle_tree_draft_decode_dispatches,
    build_extend_dispatches,
    build_ngram_copy_dispatch,
    build_snapshot,
    build_token_table,
    build_verify_dispatches,
    eagle_tree_branch_starts,
    eagle_tree_workspace_positions,
    external_read_groups,
    external_slot_groups,
    free_slots_from_allocator,
    geometry,
    geometry_from_allocator,
    mediated_frees,
    nodes_from_tree,
    pre_step_computed,
    rows_from_requests,
    slot_states,
    wrap_allocator_clear,
)
from kvguard.core import (
    BlockState,
    CSRPageList,
    DispatchDescriptor,
    ExternalFill,
    Ledger,
    PositionedTokenTable,
    Reporter,
    SlotMap,
    TokenTable,
    Unsupported,
)
from kvguard.core.violations import Policy

from tests.unit.conftest import kinds, row

POOL = "kv"
SIZE = 12


def test_external_read_groups_split_a_partial_pd_tail():
    assert external_read_groups([7, 11, 19], 35, 16) == (
        ((7, 11), range(16)),
        ((19,), range(3)),
    )
    assert external_read_groups([2, 4], 32, 16) == (((2, 4), range(16)),)
    assert external_read_groups([3, 8], 2, 1) == (((3, 8), range(1)),)
    assert external_read_groups([], 0, 16) == ()


def test_external_read_groups_reject_sender_shape_disagreement():
    with pytest.raises(ValueError, match="expected 2"):
        external_read_groups([7], 17, 16)


def test_external_slot_groups_derive_full_pages_and_partial_tail_from_a_row():
    assert external_slot_groups(range(16, 51), 16) == (
        ((1, 2), range(16)),
        ((3,), range(3)),
    )
    assert external_slot_groups([7, 8], 1) == (((7, 8), range(1)),)
    assert external_slot_groups([], 16) == ()


def test_external_slot_groups_reject_non_allocator_row_shape():
    with pytest.raises(ValueError, match="expected 1"):
        external_slot_groups([16, 18], 16)


def test_allocator_clear_reports_only_ids_added_to_the_free_set():
    class Allocator:
        def __init__(self):
            self.free_ids = [1, 4]
            self.calls = []

        def clear(self, marker=None):
            self.calls.append(marker)
            self.free_ids = [1, 2, 3, 4]
            return "cleared"

    allocator = Allocator()
    released = []
    wrap_allocator_clear(
        allocator,
        lambda: allocator.free_ids,
        released.append,
    )
    assert allocator.clear(marker=7) == "cleared"
    assert allocator.calls == [7]
    assert released == [[2, 3]]
    allocator.clear(marker=8)
    assert allocator.calls == [7, 8]
    assert released == [[2, 3]], "an id already free is not released twice"


def test_beam_member_rows_are_virtual_requests_with_shared_history():
    from types import SimpleNamespace

    leader = FakeReq("leader", FakeKV(1, 3, 3, 0))
    leader.beam_group = SimpleNamespace(member_rows_cpu=FakeTensor([2]), retired=False)
    table = [
        [0, 0, 0],
        [10, 11, 20],
        [10, 11, 21],
    ]

    members, row_ids, shared = beam_member_rows(leader, "leader@1", table, new_tokens=1)

    assert row_ids == {2: "leader@1#beam1"}
    assert members == [RowFacts("leader@1#beam1", (10, 11, 21), 3, 3, 0, 1)]
    assert shared == {10, 11}

    rows = [RowFacts("leader@1", (10, 11, 20), 3, 3, 0, 1), *members]
    states = slot_states(POOL, rows, [], shared_slots=shared)
    assert states[10].state is BlockState.PUBLISHED
    assert states[11].state is BlockState.PUBLISHED
    assert states[20].state is BlockState.PRIVATE
    assert states[21].state is BlockState.PRIVATE

    leader.beam_group.pending_orphans = [
        SimpleNamespace(
            old_mapping=FakeTensor([10, 11, 20, 21]),
            new_mapping=FakeTensor([10, 11, 20, 20]),
        )
    ]
    assert beam_pending_orphan_slots(leader) == {21}
    states = slot_states(POOL, rows, [], reserved_slots={21})
    assert states[21].state is BlockState.PUBLISHED and states[21].reserved


# fakes


@dataclass
class FakeKV:
    req_pool_idx: int | None = None
    kv_allocated_len: int = 0
    kv_committed_len: int = 0
    cache_protected_len: int = 0


@dataclass
class FakeReq:
    rid: str
    kv: FakeKV = field(default_factory=FakeKV)
    prefix_len: int = 0
    last_node: FakeNode | None = None


class FakeNode:
    counter = 0

    def __init__(self, parent=None, value=None):
        self.id = FakeNode.counter
        FakeNode.counter += 1
        self.children: dict = {}
        self.parent = parent
        self.value = value
        self.lock_ref = 0
        self.key = None


class FakeTensor:
    """Enough of a tensor for the adapter: ``tolist`` and indexing."""

    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def __getitem__(self, i):
        return FakeTensor(self.values[i]) if isinstance(i, slice) else self.values[i]

    def __len__(self):
        return len(self.values)


class FakeAllocator:
    """``TokenToKVPoolAllocator``: ``free_pages`` 1..size, ``release_pages`` (need_sort), a free group."""

    def __init__(self, size, page_size=1):
        self.size, self.page_size = size, page_size
        self.free_pages = list(range(1, size + 1))
        self.release_pages: list[int] = []
        self.free_group: list | None = None

    def get_all_free_pages(self):
        return self.free_pages + self.release_pages

    def merge_and_sort_free(self):
        self.free_pages = sorted(self.free_pages + self.release_pages)
        self.release_pages = []

    def alloc(self, need_size):
        if need_size > len(self.free_pages):
            self.merge_and_sort_free()
        if need_size > len(self.free_pages):
            return None
        out, self.free_pages = self.free_pages[:need_size], self.free_pages[need_size:]
        return FakeTensor(out)

    def free(self, free_index):
        if self.free_group is None:
            self.release_pages.extend(free_index)
        else:
            self.free_group.append(list(free_index))

    def free_group_begin(self):
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free([s for group in pending for s in group])


class FakeReqToTokenPool:
    """Row 0 is the padding row (all zeros); rows 1..size are handed out."""

    def __init__(self, size, max_context_len):
        self.req_to_token = [[0] * max_context_len for _ in range(size + 1)]
        self.free_slots = list(range(1, size + 1))

    def alloc_rows(self, need_size):
        out = self.free_slots[-need_size:]
        del self.free_slots[-need_size:]
        return out

    def free_rows(self, indices):
        self.free_slots.extend(indices)


class Engine:
    """SGLang's host-side bookkeeping for the stock path, driving a ledger."""

    def __init__(self, size=SIZE, max_reqs=4, max_context_len=16, use_ragged=True):
        self.allocator = FakeAllocator(size)
        self.rows = FakeReqToTokenPool(max_reqs, max_context_len)
        self.root = FakeNode(value=[])
        self.root.lock_ref = 1
        self.reqs: dict[str, FakeReq] = {}
        self.ledger = Ledger(Reporter(fail_fast=False))
        self.ledger.register_pool(geometry_from_allocator(self.allocator, POOL))
        self.seq = 0
        self.use_ragged = use_ragged
        self.new_tokens: dict[str, int] = {}
        self.batch: list[str] = []
        self.out_cache_loc: list[int] = []
        self.mode: str | None = None
        self.open: list[str] = []
        self._wrap_allocator()

    # -- allocator mediation, as an attachment would install it
    def _wrap_allocator(self):
        orig_alloc, orig_free, me = self.allocator.alloc, self.allocator.free, self

        def alloc(need_size):
            out = orig_alloc(need_size)
            me.ledger.allocate(POOL, allocated_slots(out), site=f"alloc@{me.seq}")
            return out

        def free(free_index):
            freed = mediated_frees(free_index, me.allocator.free_group)
            orig_free(free_index)
            if freed:
                me.ledger.free(POOL, freed, site=f"free@{me.seq}")

        self.allocator.alloc, self.allocator.free = alloc, free

    # -- radix tree
    def path(self, node):
        values: list[int] = []
        while node is not self.root:
            values = list(node.value) + values
            node = node.parent
        return values

    def lock(self, node):
        while node is not self.root:
            node.lock_ref += 1
            node = node.parent

    def unlock(self, node):
        while node is not self.root:
            node.lock_ref -= 1
            node = node.parent

    def insert_under(self, parent, slots):
        child = FakeNode(parent, list(slots))
        child.key = tuple(slots)
        parent.children[child.key] = child
        return child

    def evict(self, node):
        """``RadixCache.evict``: an unlocked leaf's slots go back through ``free_segment``."""
        assert node.lock_ref == 0 and not node.children
        self.allocator.free(list(node.value))
        del node.parent.children[node.key]

    # -- scheduler steps
    def prefill(self, rid, extend_len, prefix_node=None):
        """Admit ``rid`` with the path to ``prefix_node`` as its matched prefix (or continue a
        chunked request) and extend it by ``extend_len`` tokens: ``prepare_for_extend``."""
        req = self.reqs.get(rid)
        if req is None:
            req = FakeReq(rid)
            self.reqs[rid] = req
            prefix_node = prefix_node or self.root
            prefix = self.path(prefix_node)
            req.prefix_len, req.last_node = len(prefix), prefix_node
            self.lock(prefix_node)  # _commit_prefill_admission: req.lock = tree_cache.lock(...)
            req.kv.cache_protected_len = req.prefix_len  # init_next_round_input
            (req.kv.req_pool_idx,) = self.rows.alloc_rows(1)  # alloc_req_slots
            self.rows.req_to_token[req.kv.req_pool_idx][: len(prefix)] = prefix
        else:
            req.prefix_len = req.kv.kv_allocated_len
        new = allocated_slots(self.allocator.alloc(extend_len))  # alloc_token_slots
        row = self.rows.req_to_token[req.kv.req_pool_idx]
        row[req.prefix_len : req.prefix_len + extend_len] = new  # write_cache_indices
        req.kv.kv_allocated_len = req.kv.kv_committed_len = req.prefix_len + extend_len
        self.new_tokens[rid] = extend_len
        self.batch.append(rid)
        self.out_cache_loc.extend(new)
        self.mode = "extend"
        return req

    def decode(self, *rids):
        """``prepare_for_decode``: one new slot per request at position ``seq_len``."""
        self.mode = "decode"
        for rid in rids:
            req = self.reqs[rid]
            (slot,) = allocated_slots(self.allocator.alloc(1))
            self.rows.req_to_token[req.kv.req_pool_idx][req.kv.kv_allocated_len] = slot
            req.kv.kv_allocated_len += 1
            req.kv.kv_committed_len += 1
            self.new_tokens[rid] = 1
            self.batch.append(rid)
            self.out_cache_loc.append(slot)

    def checkpoint(self, rid):
        """``checkpoint_kv_cache`` of a chunked request: its own slots join the tree under
        ``last_node``, the row keeps pointing at them, the lock moves to the new node."""
        req = self.reqs[rid]
        row = self.rows.req_to_token[req.kv.req_pool_idx]
        own = row[req.kv.cache_protected_len : req.kv.kv_committed_len]
        node = self.insert_under(req.last_node, own)
        self.unlock(req.last_node)
        self.lock(node)
        req.last_node = node
        req.kv.cache_protected_len = req.kv.kv_committed_len
        return node

    def finish(self, rid, *, checkpoint=True):
        """``release_kv_cache`` inside the result processor's free group: with ``checkpoint`` the
        row's own slots join the tree; otherwise they are freed. Then the lock and the row go."""
        req = self.reqs.pop(rid)
        row = self.rows.req_to_token[req.kv.req_pool_idx]
        own = row[req.kv.cache_protected_len : req.kv.kv_committed_len]
        node = req.last_node
        self.allocator.free_group_begin()
        if checkpoint and own:
            node = self.insert_under(req.last_node, own)
        else:
            self.allocator.free(own)
        self.unlock(req.last_node)
        self.rows.free_rows([req.kv.req_pool_idx])
        req.kv.req_pool_idx = None
        self.allocator.free_group_end()
        return node

    # -- snapshot and dispatch
    def observe(self):
        self.seq += 1
        snap = build_snapshot(
            self.seq,
            POOL,
            rows_from_requests(self.reqs.values(), self.rows.req_to_token, self.new_tokens),
            nodes_from_tree(self.root),
            free_slots_from_allocator(self.allocator),
        )
        self.ledger.observe(snap)
        return snap

    def gather(self, lanes):
        """``create_flashinfer_kv_indices_triton``: ``req_to_token[row, :len]`` per lane."""
        indptr, indices = [0], []
        for row_idx, length in lanes:
            indices.extend(self.rows.req_to_token[row_idx][:length])
            indptr.append(len(indices))
        return indptr, indices

    def forward(
        self,
        *,
        kv_indptr=None,
        kv_indices=None,
        out_cache_loc=None,
        num_rows_after_padding=None,
    ):
        rids = list(self.batch)
        seq_lens = [self.reqs[r].kv.kv_allocated_len for r in rids]
        rows = [self.reqs[r].kv.req_pool_idx for r in rids]
        loc = list(self.out_cache_loc)
        if self.mode == "decode":
            n_pad = num_rows_after_padding or len(rids)
            lanes = list(zip(rows, seq_lens, strict=True)) + [(0, 1)] * (n_pad - len(rids))
            loc += [PADDING_SLOT] * (n_pad - len(loc))
            indptr, indices = self.gather(lanes)
            ds = build_decode_dispatches(
                self.seq, POOL, rids, seq_lens,
                kv_indptr if kv_indptr is not None else indptr,
                kv_indices if kv_indices is not None else indices,
                out_cache_loc if out_cache_loc is not None else loc,
                num_rows_after_padding=n_pad,
            )  # fmt: skip
        else:
            prefix = [seq_lens[i] - self.new_tokens[r] for i, r in enumerate(rids)]
            paged = prefix if self.use_ragged else seq_lens
            indptr, indices = self.gather(list(zip(rows, paged, strict=True)))
            ds = build_extend_dispatches(
                self.seq, POOL, rids, seq_lens, prefix,
                kv_indptr if kv_indptr is not None else indptr,
                kv_indices if kv_indices is not None else indices,
                out_cache_loc if out_cache_loc is not None else loc,
                use_ragged=self.use_ragged,
            )  # fmt: skip
        for d in ds:
            self.ledger.dispatch(d)
            self.open.append(d.dispatch_id)
        self.batch, self.out_cache_loc, self.new_tokens = [], [], {}
        return ds

    def complete(self):
        for did in self.open:
            self.ledger.complete(did)
        self.open.clear()

    def step(self, **kw):
        self.observe()
        ds = self.forward(**kw)
        self.complete()
        return ds

    def slots_of(self, rid):
        req = self.reqs[rid]
        return list(self.rows.req_to_token[req.kv.req_pool_idx][: req.kv.kv_allocated_len])


# geometry, states


def test_geometry_page_size_one_reserves_the_padding_slot():
    g = geometry(SIZE, 1, POOL)
    assert (g.num_blocks, g.block_size, g.reserved_blocks) == (SIZE + 1, 1, frozenset({0}))
    assert geometry_from_allocator(FakeAllocator(SIZE), POOL) == g
    assert SUPPORT_BOUNDARY["page_size"].startswith("1 (TokenToKVPoolAllocator)")


def test_ngram_verify_descriptors_cover_candidates_and_graph_padding():
    write, attn = build_verify_dispatches(
        9,
        POOL,
        ["a", "b"],
        [3, 2],
        [0, 5, 9, 10],
        [11, 12, 13, 21, 22, 31, 32, 41, 42, 0],
        [13, 21, 31, 22, 32, 41, 0, 0, 0],
        3,
        num_rows_after_padding=3,
        last_page_len=[1, 1, 1],
    )

    assert write.dispatch_id == "write-verify@9"
    assert [(r.request, r.writes, r.padding) for r in write.rows] == [
        ("a", range(3, 6), False),
        ("b", range(2, 5), False),
        (None, range(0), True),
    ]
    assert isinstance(write.mapping, SlotMap)
    assert write.mapping.rows == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert write.mapping.positions == [3, 4, 5, 2, 3, 4, 0, 1, 2]
    assert isinstance(attn.mapping, CSRPageList)
    assert attn.mapping.indptr == [0, 5, 9, 10]
    assert attn.rows[0].reads == range(0, 5)
    assert attn.rows[1].reads == range(0, 4)
    assert attn.rows[2].padding


def test_ngram_verify_rejects_malformed_width_or_indptr():
    with pytest.raises(ValueError, match="positive"):
        build_verify_dispatches(1, POOL, ["a"], [3], [0, 1], [1], [1], 0)
    with pytest.raises(ValueError, match="indptr"):
        build_verify_dispatches(
            1,
            POOL,
            ["a"],
            [3],
            [0, 1],
            [1],
            [1],
            1,
            num_rows_after_padding=2,
        )


def test_dflash_materialize_selects_only_each_committed_prefix():
    dispatch = build_dflash_materialize_dispatch(
        11,
        "draft_kv",
        ["a", "b"],
        [10, 11, 12, 13, 20, 21, 22, 23],
        [110, 111, 112, 113, 120, 121, 122, 123],
        [3, 1],
        row_width=4,
        tag="-dflash-commit",
    )

    assert dispatch.dispatch_id == "write-dflash-commit@11"
    assert [row.writes for row in dispatch.rows] == [range(10, 13), range(20, 21)]
    assert isinstance(dispatch.mapping, SlotMap)
    assert dispatch.mapping.slots == [110, 111, 112, 120]
    assert dispatch.mapping.rows == [0, 0, 0, 1]
    assert dispatch.mapping.positions == [10, 11, 12, 20]


def test_dflash_materialize_packed_prefill_and_shape_checks():
    dispatch = build_dflash_materialize_dispatch(
        4,
        "draft_kv",
        ["a", "b"],
        [5, 6, 9],
        [15, 16, 19],
        [2, 1],
    )
    assert [row.writes for row in dispatch.rows] == [range(5, 7), range(9, 10)]
    assert dispatch.mapping.rows == [0, 0, 1]

    with pytest.raises(ValueError, match="exceed row width"):
        build_dflash_materialize_dispatch(1, "draft_kv", ["a"], [5, 6], [15, 16], [3], row_width=2)


def test_eagle_tree_draft_decode_describes_only_each_branch_and_detects_a_swap():
    rows = [
        RowFacts("a", tuple(range(1, 11)), 10, 4, 0, new_tokens=4),
        RowFacts("b", tuple(range(11, 23)), 12, 6, 0, new_tokens=6),
    ]
    ledger = Ledger(Reporter(fail_fast=False), mediated=False)
    ledger.register_pool(geometry(30, 1, POOL))
    ledger.observe(build_snapshot(1, POOL, rows, [], range(23, 31)))
    for request, start, length in (("a", 1, 4), ("b", 11, 6)):
        for slot in range(start, start + length):
            ledger.external_fill(ExternalFill(POOL, slot, range(1), request=request))
    rows = [
        RowFacts("a", tuple(range(1, 11)), 10, 4, 0),
        RowFacts("b", tuple(range(11, 23)), 12, 6, 0),
    ]
    ledger.observe(build_snapshot(2, POOL, rows, [], range(23, 31)))

    # Step zero: prefix plus the first slot of each of two branches. Request-major, then branch.
    step0_table = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 8],
        [11, 12, 13, 14, 15, 16, 17],
        [11, 12, 13, 14, 15, 16, 20],
        [0],
        [0],
    ]
    step0_indptr = [0]
    for entries in step0_table:
        step0_indptr.append(step0_indptr[-1] + len(entries))
    step0_write, step0_attn = build_eagle_tree_draft_decode_dispatches(
        1,
        POOL,
        ["a", "b"],
        [4, 6],
        2,
        3,
        0,
        step0_indptr,
        [slot for entries in step0_table for slot in entries],
        [5, 8, 17, 20],
        num_requests_after_padding=3,
        tag="-tree0",
    )
    assert isinstance(step0_attn.mapping, PositionedTokenTable)
    assert [row.reads for row in step0_attn.rows[:4]] == [
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 7),
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 5, 9),
    ]
    assert [row.writes for row in step0_write.rows[:4]] == [
        range(4, 5),
        range(7, 8),
        range(6, 7),
        range(9, 10),
    ]
    assert not step0_write.advances_computed
    ledger.dispatch(step0_write)
    ledger.dispatch(step0_attn)
    ledger.complete(step0_write.dispatch_id)
    ledger.complete(step0_attn.dispatch_id)

    # Step one consumes the preceding slot of the same branch, never the other branch's slots.
    step1_table = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 8, 9],
        [11, 12, 13, 14, 15, 16, 17, 18],
        [11, 12, 13, 14, 15, 16, 20, 21],
    ]
    step1_indptr = [0]
    for entries in step1_table:
        step1_indptr.append(step1_indptr[-1] + len(entries))
    step1_write, step1_attn = build_eagle_tree_draft_decode_dispatches(
        1,
        POOL,
        ["a", "b"],
        [4, 6],
        2,
        3,
        1,
        step1_indptr,
        [slot for entries in step1_table for slot in entries],
        [6, 9, 18, 21],
        tag="-tree1",
    )
    ledger.dispatch(step1_write)
    ledger.dispatch(step1_attn)
    assert kinds(ledger) == []
    assert ledger.requests[("a", POOL)].computed_high_water == 4
    assert ledger.requests[("b", POOL)].computed_high_water == 6

    bad = [list(entries) for entries in step1_table]
    bad[1][-1] = 6  # branch one position 8 names branch zero's position 5
    bad_attn = DispatchDescriptor(
        "attn-tree-bad@1",
        POOL,
        "attention",
        step1_attn.rows,
        PositionedTokenTable(bad, step1_attn.mapping.positions),
    )
    ledger.dispatch(bad_attn)
    assert kinds(ledger) == ["mapping_mismatch"]
    ledger.invalidate_positions(POOL, "a", range(4, 10), site="tree-to-extend")
    assert ledger.counters["speculative_sparse_invalidations"] == 4


def test_eagle_tree_paged_layout_uses_branch_pages_and_sparse_workspace():
    assert eagle_tree_branch_starts(18, 2, 3, 16) == (18, 34)
    assert eagle_tree_workspace_positions(18, 2, 3, 16) == (
        18,
        19,
        20,
        32,
        33,
        34,
        35,
        36,
    )

    rows = [[*range(16, 34), 34, 35], [*range(16, 34), 50, 51]]
    indptr = [0, len(rows[0]), len(rows[0]) + len(rows[1])]
    write, attention = build_eagle_tree_draft_decode_dispatches(
        2,
        POOL,
        ["a"],
        [18],
        2,
        3,
        1,
        indptr,
        [slot for row_slots in rows for slot in row_slots],
        [35, 51],
        page_size=16,
        tag="-paged-tree",
    )
    assert [row.writes for row in write.rows] == [range(19, 20), range(35, 36)]
    assert [row.reads for row in attention.rows] == [
        (*range(18), 18, 19),
        (*range(18), 34, 35),
    ]
    assert isinstance(attention.mapping, PositionedTokenTable)
    assert attention.mapping.positions == [
        (*range(18), 18, 19),
        (*range(18), 34, 35),
    ]


def test_eagle_tree_topk8_paged_layout_scales_branch_regions():
    starts = eagle_tree_branch_starts(18, 8, 3, 16)
    assert starts == (18, 34, 50, 66, 82, 98, 114, 130)
    workspace = eagle_tree_workspace_positions(18, 8, 3, 16)
    assert len(workspace) == 38
    assert workspace[:3] == (18, 19, 20)
    assert workspace[3:8] == (32, 33, 34, 35, 36)
    assert workspace[-5:] == (128, 129, 130, 131, 132)
    assert 16 not in workspace and 17 not in workspace


def test_ngram_verify_uses_token_table_for_paged_allocator():
    _, attn = build_verify_dispatches(
        2,
        POOL,
        ["a"],
        [17],
        [0, 3],
        [16, 17, 32],
        [32, 33],
        2,
        page_size=16,
    )
    assert isinstance(attn.mapping, TokenTable)
    assert attn.mapping.table == [[16, 17, 32]]
    assert attn.rows[0].reads == range(0, 3)


def test_verify_descriptor_can_name_a_sliding_window_suffix():
    _, attn = build_verify_dispatches(
        10,
        POOL,
        ["a"],
        [192],
        [0, 4],
        [688, 689, 690, 691],
        [704, 705, 706, 707],
        4,
        page_size=16,
        read_starts=[192],
        tag="-swa",
    )

    assert isinstance(attn.mapping, TokenTable)
    assert attn.mapping.first == [192]
    assert attn.rows[0].reads == range(192, 196)
    assert attn.dispatch_id == "attn-swa@10"


def test_ngram_copy_uses_accepted_lengths_and_source_indices():
    copy = build_ngram_copy_dispatch(
        7,
        POOL,
        [("a", 10), ("b", 20)],
        [101, 102, 201, 0, 0, 0],
        [103, 101, 204, 0, 0, 0],
        [2, 1],
        [2, 0, -1, 3, -1, -1],
        3,
    )

    assert copy.dispatch_id == "spec-copy@7"
    assert isinstance(copy.mapping, TokenTable)
    assert copy.mapping.table == [
        [101],
        [103],
        [102],
        [101],
        [201],
        [204],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]
    assert copy.mapping.first == [10, 12, 11, 10, 20, 20, 0, 0, 0, 0, 0, 0]
    assert [row.request for row in copy.rows[:6]] == ["a", "a", "a", "a", "b", "b"]
    assert all(row.padding for row in copy.rows[6:])

    swa_copy = build_ngram_copy_dispatch(
        7,
        "sglang-swa-kv",
        [("a", 10), ("b", 20)],
        [101, 102, 201, 0, 0, 0],
        [103, 101, 204, 0, 0, 0],
        [2, 1],
        [2, 0, -1, 3, -1, -1],
        3,
        tag="-swa",
    )
    assert swa_copy.dispatch_id == "spec-copy-swa@7"


def test_ngram_copy_rejects_malformed_compaction_shape():
    with pytest.raises(ValueError, match="nonnegative indices"):
        build_ngram_copy_dispatch(1, POOL, [("a", 4)], [11, 0], [12, 0], [2], [0, -1], 2)


@pytest.mark.parametrize("corrupt", ["target", "source"])
def test_ngram_copy_detects_a_cross_request_operand(corrupt):
    eng = Engine()
    eng.prefill("a", 3)
    eng.prefill("b", 3)
    eng.step()
    a, b = eng.slots_of("a"), eng.slots_of("b")
    targets = [a[0], b[0], 0, 0]
    sources = [a[0], b[0], 0, 0]
    (targets if corrupt == "target" else sources)[0] = b[0]
    copy = build_ngram_copy_dispatch(
        eng.seq,
        POOL,
        [("a", 0), ("b", 0)],
        targets,
        sources,
        [1, 1],
        [0, -1, 2, -1],
        2,
    )

    eng.ledger.dispatch(copy)
    assert kinds(eng.ledger) == ["mapping_mismatch"]


def test_geometry_page_size_sixteen_makes_pages_the_blocks():
    g = geometry_from_allocator(FakeAllocator(SIZE, page_size=16), POOL)
    assert (g.num_blocks, g.block_size, g.reserved_blocks) == (SIZE // 16 + 1, 16, frozenset({0}))
    assert g.slot(3, 5) == 53 and g.block_of(53) == 3


def test_geometry_rejects_a_page_size_that_is_not_a_power_of_two():
    with pytest.raises(Unsupported) as info:
        geometry_from_allocator(FakeAllocator(SIZE, page_size=12), POOL)
    assert info.value.report.kind == "page_size"


def test_pages_of_a_rows_slots():
    from kvguard.adapters.sglang.translate import page_set, pages_of

    slots = tuple(range(48, 48 + 20))  # 20 tokens from page 3's first slot: pages 3, 4
    assert pages_of(slots, 16) == (3, 4) and pages_of(slots, 1) == slots
    assert page_set(slots[16:], 16) == {4} and page_set(slots, 1) == set(slots)


def test_slot_states_at_page_size_sixteen():
    nodes = [NodeFacts(1, tuple(range(16, 48)), lock_ref=1)]  # pages 1, 2
    rows = [
        RowFacts(
            "B",
            tuple(range(16, 48)) + tuple(range(64, 69)),
            37,
            37,
            cache_protected_len=32,
            new_tokens=5,
        )
    ]
    states = slot_states(POOL, rows, nodes, page_size=16)
    assert {b: (s.state, s.reserved) for b, s in states.items()} == {
        1: (BlockState.PUBLISHED, False),
        2: (BlockState.PUBLISHED, False),
        4: (BlockState.PRIVATE, False),
    }


def test_csr_carries_the_last_page_lengths_and_pages_the_extras():
    from kvguard.adapters.sglang.translate import _csr, _token_table

    csr = _csr([0, 2, 3], [3, 4, 7], 3, extra={2: [128]}, last_page_len=[16, 5], page_size=16)
    assert list(csr.indptr) == [0, 2, 3, 4] and list(csr.indices) == [3, 4, 7, 8]
    assert list(csr.last_page_len) == [16, 5, 1]
    table = _token_table([0, 2, 3], [48, 49, 64], 3, extra={2: [80]})
    assert [list(r) for r in table.table] == [[48, 49], [64], [80]]
    assert list(table.seq_len) == [2, 1, 1]


def test_decode_dispatches_at_page_size_sixteen_against_the_ledger():
    from kvguard.core import Ledger, PoolGeometry, Reporter

    g = geometry(256, 16, POOL)
    assert isinstance(g, PoolGeometry)
    ledger = Ledger(Reporter(fail_fast=False))
    ledger.register_pool(g)
    ledger.allocate(POOL, [3, 4])
    # Request A holds 21 slots on pages 3 and 4 (slots 48..68). First snapshot: nothing computed
    # yet; then the 20 positions of the earlier steps are written; second snapshot: 20 computed.
    free = [1, 2, *list(range(5, 17))]
    rows = [RowFacts("A", tuple(range(48, 69)), 21, 1, cache_protected_len=0, new_tokens=1)]
    snap = build_snapshot(1, POOL, rows, [], free_slots=free, page_size=16)
    assert snap.requests[0].blocks == (3, 4) and snap.requests[0].num_computed == 0
    ledger.observe(snap)
    for p in range(20):
        ledger.external_fill(
            ExternalFill(POOL, 3 + p // 16, range(p % 16, p % 16 + 1), request="A")
        )
    rows = [RowFacts("A", tuple(range(48, 69)), 21, 21, cache_protected_len=0, new_tokens=1)]
    snap = build_snapshot(2, POOL, rows, [], free_slots=free, page_size=16)
    assert snap.requests[0].num_computed == 20
    ledger.observe(snap)
    # This decode step writes position 20 into slot 68 and reads positions 0..21 over pages 3
    # and 4 (last page length 5), as FlashInfer's page list says.
    # FlashInfer's page list at page size 16 still has one entry per position (slots 48..68).
    entries = list(range(48, 69))
    write, attn = build_decode_dispatches(
        2, POOL, ["A"], [21], [0, 21], entries, [68], last_page_len=[1], page_size=16
    )
    from kvguard.core import TokenTable

    assert isinstance(attn.mapping, TokenTable)
    ledger.dispatch(write)
    ledger.dispatch(attn)
    assert [r.kind for r in ledger.reporter.reports] == []
    # An entry naming another page's slot is a mapping mismatch.
    bad_entries = [*entries[:10], 80, *entries[11:]]
    _, bad = build_decode_dispatches(
        3, POOL, ["A"], [21], [0, 21], bad_entries, [68], last_page_len=[1], page_size=16
    )
    ledger.dispatch(bad)
    kinds = [r.kind for r in ledger.reporter.reports]
    assert kinds and all(k in ("mapping_mismatch", "stale_table_entry") for k in kinds), kinds


def test_paged_decode_descriptor_can_name_a_sliding_window_suffix():
    entries = list(range(53, 69))
    _, attn = build_decode_dispatches(
        7,
        POOL,
        ["A"],
        [21],
        [0, len(entries)],
        entries,
        [68],
        last_page_len=[1],
        page_size=16,
        read_starts=[5],
        tag="-swa",
    )
    from kvguard.core import TokenTable

    assert isinstance(attn.mapping, TokenTable)
    assert attn.mapping.first == [5]
    assert attn.rows[0].reads == range(5, 21)
    assert attn.dispatch_id == "attn-swa@7"


def test_page_size_one_decode_descriptor_can_name_a_hicache_suffix():
    entries = list(range(105, 125))
    _, attn = build_decode_dispatches(
        8,
        POOL,
        ["A"],
        [125],
        [0, len(entries)],
        entries,
        [125],
        last_page_len=[1],
        page_size=1,
        read_starts=[105],
        tag="-hicache-suffix",
    )

    assert isinstance(attn.mapping, TokenTable)
    assert attn.mapping.table == [entries]
    assert attn.mapping.first == [105]
    assert attn.rows[0].reads == range(105, 125)
    assert attn.dispatch_id == "attn-hicache-suffix@8"


def test_pre_step_computed_undoes_the_allocation_time_advance():
    assert pre_step_computed(kv_committed_len=6, new_tokens=6) == 0  # first prefill chunk
    assert pre_step_computed(7, 1) == 6  # decode
    assert pre_step_computed(7, 0) == 7  # not in this batch


def test_slot_states_rules():
    nodes = [NodeFacts(1, (3, 4), lock_ref=2), NodeFacts(2, (5,), lock_ref=0)]
    rows = [
        RowFacts("B", (3, 4, 7), 3, 3, cache_protected_len=2, new_tokens=1),
        RowFacts("C", (3, 4, 5, 9), 4, 4, cache_protected_len=3, new_tokens=1),
        RowFacts("D", (8, 10), 2, 2, cache_protected_len=1),  # claims slot 8 is the tree's
        RowFacts("E", (5, 11), 2, 2, cache_protected_len=0),  # claims tree slot 5 as its own
    ]
    st = slot_states(POOL, rows, nodes)
    assert st[3].state is BlockState.PUBLISHED and st[3].refs == 2 and not st[3].reserved
    assert st[5].state is BlockState.PUBLISHED and st[5].refs == 0 and st[5].reserved
    assert st[7].state is BlockState.PRIVATE and st[7].refs is None
    assert st[9].state is BlockState.PRIVATE and st[10].state is BlockState.PRIVATE
    assert st[11].state is BlockState.PRIVATE
    assert 8 not in st  # a claim without a holder is left to the core
    with pytest.raises(ValueError):
        slot_states(POOL, [], [NodeFacts(1, (3,), 1), NodeFacts(2, (3,), 0)])


def test_build_snapshot_rows_states_and_free_set():
    rows = [RowFacts("A", (3, 4, 5), 3, 3, 0, new_tokens=3)]
    snap = build_snapshot(1, POOL, rows, [], free_slots=FakeTensor([1, 2, 6]))
    assert [(r.request, r.blocks, r.num_tokens, r.num_computed) for r in snap.requests] == [
        ("A", (3, 4, 5), 3, 0)
    ]
    assert [b.block for b in snap.blocks] == [3, 4, 5]
    assert snap.free == {POOL: frozenset({1, 2, 6})}


def test_mediated_frees_defers_inside_a_free_group():
    assert mediated_frees(FakeTensor([4, 5]), free_group=None) == [4, 5]
    assert mediated_frees(FakeTensor([4, 5]), free_group=[]) == []
    assert allocated_slots(None) == [] and allocated_slots(FakeTensor([7])) == [7]


def test_extraction_reads_sglang_attribute_names():
    class Table:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, i):
            return FakeTensor(self.rows[i])

    table = Table([[0, 0, 0], [3, 4, 9], [6, 0, 0]])
    a = FakeReq("A", FakeKV(req_pool_idx=1, kv_allocated_len=2, kv_committed_len=2))
    b = FakeReq("B", FakeKV(req_pool_idx=2, kv_allocated_len=1, kv_committed_len=1))
    waiting = FakeReq("W")  # no row yet
    rows = rows_from_requests([a, b, waiting], table, new_tokens={"B": 1})
    assert rows == [RowFacts("A", (3, 4), 2, 2, 0, 0), RowFacts("B", (6,), 1, 1, 0, 1)]
    root = FakeNode(value=[])
    n1 = FakeNode(root, FakeTensor([3, 4]))
    n1.lock_ref = 1
    n2 = FakeNode(n1, None)  # device copy evicted (hierarchical cache): skipped
    root.children = {"a": n1}
    n1.children = {"b": n2}
    assert nodes_from_tree(root) == [NodeFacts(n1.id, (3, 4), 1)]
    alloc = FakeAllocator(5)
    alloc.free_pages, alloc.release_pages = [1, 2], [5]
    assert free_slots_from_allocator(alloc) == [1, 2, 5]


def test_unified_allocator_exposes_stable_virtual_free_ids():
    alloc = SimpleNamespace(free_virtual_ids=[3, 8], get_all_free_pages=lambda: [1, 2])
    assert free_slots_from_allocator(alloc) == [3, 8]


# clean scenarios


def test_clean_prefill_and_decode_steps_report_nothing():
    eng = Engine()
    eng.prefill("A", 3)
    attn, write = eng.step()  # ragged extend: attention over the (empty) prefix, then the write
    assert attn.family == "attention" and write.family == "set_kv_buffer"
    assert [r.writes for r in write.rows] == [range(0, 3)]
    for _ in range(2):
        eng.decode("A")
        write, attn = eng.step()
        assert write.family == "set_kv_buffer" and attn.family == "attention"
    assert kinds(eng.ledger) == []
    slots = eng.slots_of("A")
    assert len(slots) == 5 and all(eng.ledger.block(POOL, s).filled == 1 for s in slots)
    shadow = eng.ledger.requests[("A", POOL)].snapshot
    assert (shadow.num_tokens, shadow.num_computed) == (5, 4)


def test_prefix_sharing_through_the_radix_tree():
    eng = Engine()
    eng.prefill("A", 4)
    eng.step()
    eng.decode("A")
    eng.step()
    node = eng.finish("A")  # checkpoint: A's five slots join the tree
    eng.observe()
    assert kinds(eng.ledger) == []
    cached = list(node.value)
    assert len(cached) == 5
    for s in cached:
        b = eng.ledger.block(POOL, s)
        assert b.state is BlockState.PUBLISHED and b.reserved and not b.readers
    eng.prefill("B", 1, prefix_node=node)  # prefix hit of five tokens, one new token
    eng.step()
    assert kinds(eng.ledger) == []
    assert all(eng.ledger.block(POOL, s).readers == {"B"} for s in cached)
    assert not eng.ledger.block(POOL, cached[0]).reserved
    eng.prefill("C", 2, prefix_node=node)
    eng.step()
    eng.decode("B", "C")
    eng.step()
    assert kinds(eng.ledger) == []
    assert all(eng.ledger.block(POOL, s).readers == {"B", "C"} for s in cached)
    assert node.lock_ref == 2


def test_finish_without_checkpoint_frees_the_own_slots_through_the_free_group():
    eng = Engine()
    eng.prefill("A", 4)
    eng.step()
    node = eng.finish("A")
    eng.prefill("B", 2, prefix_node=node)
    eng.step()
    own = eng.slots_of("B")[4:]
    eng.finish("B", checkpoint=False)  # abort path: the two own slots are freed at group end
    assert set(own) <= set(eng.allocator.release_pages)
    eng.observe()
    assert kinds(eng.ledger) == []
    assert all(eng.ledger.block(POOL, s).state is BlockState.FREE for s in own)
    assert node.lock_ref == 0 and eng.ledger.block(POOL, node.value[0]).reserved


def test_eviction_of_an_unlocked_node_is_a_mediated_free():
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    node = eng.finish("A")
    eng.observe()
    eng.evict(node)
    eng.observe()
    assert kinds(eng.ledger) == []
    assert set(node.value) <= frozenset(free_slots_from_allocator(eng.allocator))
    assert all(eng.ledger.block(POOL, s).state is BlockState.FREE for s in node.value)
    eng.prefill("B", SIZE)  # exhausts free_pages: the evicted slots are reallocated, new generation
    eng.step()
    assert kinds(eng.ledger) == []
    assert set(node.value) <= set(eng.slots_of("B"))
    assert all(eng.ledger.block(POOL, s).generation == 2 for s in node.value)


def test_chunked_prefill_checkpoint_publishes_the_running_rows_slots():
    eng = Engine()
    eng.prefill("A", 4)
    eng.step()
    node = eng.checkpoint("A")  # stash_chunked_request -> checkpoint_kv_cache
    assert node.lock_ref == 1
    eng.observe()
    assert kinds(eng.ledger) == []
    assert all(eng.ledger.block(POOL, s).readers == {"A"} for s in node.value)
    eng.prefill("A", 2)  # second chunk reads the published prefix and writes positions 4, 5
    attn, write = eng.step()
    assert [r.reads for r in attn.rows] == [range(0, 4)]
    assert [r.writes for r in write.rows] == [range(4, 6)]
    assert kinds(eng.ledger) == []


def test_paged_extend_order_writes_before_reading_everything():
    eng = Engine(use_ragged=False)
    eng.prefill("A", 3)
    write, attn = eng.step()
    assert write.family == "set_kv_buffer" and [r.reads for r in attn.rows] == [range(0, 3)]
    eng.decode("A")
    eng.step()
    assert kinds(eng.ledger) == []


# descriptors


def test_decode_descriptors_slot_map_csr_and_padding_lanes():
    indptr, indices = [0, 3, 5, 6, 7], [3, 4, 9, 6, 7, 0, 0]
    write, attn = build_decode_dispatches(
        5, POOL, ["A", "B"], FakeTensor([3, 2]), indptr, indices, FakeTensor([9, 7, 0, 0]),
        num_rows_after_padding=4,
    )  # fmt: skip
    assert isinstance(write.mapping, SlotMap)
    assert (write.mapping.slots, write.mapping.rows, write.mapping.positions) == (
        [9, 7],
        [0, 1],
        [2, 1],
    )
    assert [r.writes for r in write.rows] == [range(2, 3), range(1, 2)]
    assert isinstance(attn.mapping, CSRPageList)
    assert [r.reads for r in attn.rows[:2]] == [range(0, 3), range(0, 2)]
    assert [r.padding for r in attn.rows] == [False, False, True, True]
    assert [list(attn.mapping.entries(i)) for i in range(4)] == [[3, 4, 9], [6, 7], [0, 0], [0, 0]]
    assert attn.mapping.last_page_len == [1, 1, 1, 1]
    assert (write.dispatch_id, attn.dispatch_id) == ("write@5", "attn@5")


def test_extend_descriptors_ragged_reads_the_prefix_only_and_paged_reads_everything():
    args = (2, POOL, ["A", "B"], [5, 3], [4, 0], [0, 4, 4], [3, 4, 5, 6], [7, 8, 9, 10])
    attn, write = build_extend_dispatches(*args, use_ragged=True)
    assert [r.reads for r in attn.rows] == [range(0, 4), range(0, 0)]
    assert [r.writes for r in write.rows] == [range(4, 5), range(0, 3)]
    assert write.mapping.slots == [7, 8, 9, 10] and write.mapping.rows == [0, 1, 1, 1]
    assert write.mapping.positions == [4, 0, 1, 2]
    assert [list(attn.mapping.entries(i)) for i in range(2)] == [[3, 4, 5, 6], []]
    write, attn = build_extend_dispatches(
        2, POOL, ["A", "B"], [5, 3], [4, 0], [0, 5, 8], [3, 4, 5, 6, 7, 8, 9, 10], [7, 8, 9, 10],
        use_ragged=False,
    )  # fmt: skip
    assert [r.reads for r in attn.rows] == [range(0, 5), range(0, 3)]


def test_extend_descriptors_split_fixed_width_graph_padding_by_request_row():
    write, attn = build_extend_dispatches(
        3,
        POOL,
        ["A"],
        [7],
        [3],
        [0, 7, 8, 9],
        list(range(24)),
        [40, 41, 42, 43, 0, 0, 0, 0, 90, 91, 92, 93],
        use_ragged=False,
        num_rows_after_padding=3,
    )

    assert [row.padding for row in attn.rows] == [False, True, True]
    assert [list(attn.mapping.entries(row)) for row in range(3)] == [
        list(range(7)),
        [7, 0, 0, 0, 0],
        [8, 90, 91, 92, 93],
    ]
    assert write.mapping.slots == [40, 41, 42, 43]


def test_extend_padded_token_tail_becomes_a_padding_row():
    attn, write = build_extend_dispatches(
        1, POOL, ["A"], [2], [0], [0, 0], [], [5, 6, 0, 0, 0, 0], use_ragged=True
    )
    assert write.mapping.slots == [5, 6]
    assert [r.padding for r in attn.rows] == [False, True]
    assert list(attn.mapping.entries(1)) == [0, 0, 0, 0]


def test_token_table_representation_for_row_readers():
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    eng.observe()
    rows = [eng.rows.req_to_token[eng.reqs["A"].kv.req_pool_idx]]
    good = build_token_table(rows, [3])
    assert good.table == [eng.slots_of("A")] and good.seq_len == [3]
    eng.ledger.dispatch(
        DispatchDescriptor("t1", POOL, "attention", (row("A", reads=range(0, 3)),), good)
    )
    assert kinds(eng.ledger) == []
    bad = build_token_table([[s if i != 1 else 11 for i, s in enumerate(rows[0])]], [3])
    eng.ledger.dispatch(
        DispatchDescriptor("t2", POOL, "attention", (row("A", reads=range(0, 3)),), bad)
    )
    assert kinds(eng.ledger) == ["mapping_mismatch"]


# padding and holes


def test_decode_padding_lanes_touching_slot_zero_are_fine():
    eng = Engine()
    eng.prefill("A", 2)
    eng.step()
    eng.decode("A")
    _, attn = eng.step(num_rows_after_padding=4)
    assert [r.padding for r in attn.rows] == [False, True, True, True]
    assert [list(attn.mapping.entries(i)) for i in range(1, 4)] == [[0, 0]] * 3
    assert kinds(eng.ledger) == []


def test_decode_padding_lane_writing_a_live_slot_is_reported():
    """A padded lane's out_cache_loc names a request-managed slot."""
    eng = Engine()
    eng.prefill("A", 2)
    eng.prefill("B", 2)
    eng.step()
    eng.decode("A")
    eng.observe()
    loc = [*eng.out_cache_loc, eng.slots_of("B")[0], PADDING_SLOT]
    eng.forward(num_rows_after_padding=3, out_cache_loc=loc)
    assert kinds(eng.ledger) == ["padding_row_names_managed_block"]
    assert eng.ledger.reporter.reports[0].policy is Policy.P2


def test_stale_row_entry_beyond_the_request_is_reported():
    """A lane whose kv_indptr runs one past the request's tokens reads the row's stale entry (a
    slot of the row's previous occupant, here one that another request owns)."""
    eng = Engine()
    eng.prefill("A", 2)
    eng.prefill("B", 2)
    eng.step()
    eng.decode("A")
    eng.observe()
    stale = [*eng.slots_of("A"), eng.slots_of("B")[0]]
    eng.forward(kv_indptr=[0, len(stale)], kv_indices=stale)
    assert "stale_table_entry" in kinds(eng.ledger)


# corrupted inputs


def test_use_after_free_of_an_own_slot_fails_the_partition_audit():
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    eng.allocator.free([eng.slots_of("A")[2]])  # freed while A's row still names it
    eng.observe()
    found = kinds(eng.ledger)
    assert "block_free_and_allocated" in found and "unmediated_allocation" in found
    report = next(r for r in eng.ledger.reporter.reports if r.kind == "block_free_and_allocated")
    assert report.policy is Policy.P1


def test_freed_prefix_slot_still_mapped_is_a_mapping_to_a_free_block():
    """A node evicted under a request that locked it: the row's tree claim has no holder."""
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    node = eng.finish("A")
    eng.prefill("B", 1, prefix_node=node)
    eng.step()
    node.lock_ref = 0  # the lock is lost ...
    eng.evict(node)  # ... and the node is evicted while B maps its slots
    eng.observe()
    assert "mapping_to_free_block" in kinds(eng.ledger)
    report = next(r for r in eng.ledger.reporter.reports if r.kind == "mapping_to_free_block")
    assert report.policy is Policy.P2 and report.request == "B"


def test_lock_ref_that_disagrees_with_the_rows_is_reported():
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    node = eng.finish("A")
    eng.prefill("B", 1, prefix_node=node)
    eng.prefill("C", 1, prefix_node=node)
    eng.step()
    assert kinds(eng.ledger) == []
    node.lock_ref = 1  # two requests map the node, the tree counts one
    eng.observe()
    assert kinds(eng.ledger) == ["reference_count_mismatch"] * len(node.value)
    assert eng.ledger.reporter.reports[0].policy is Policy.P2


def test_corrupted_kv_indices_is_a_mapping_mismatch():
    eng = Engine()
    eng.prefill("A", 2)
    eng.prefill("B", 2)
    eng.step()
    eng.decode("A")
    eng.observe()
    wrong = eng.slots_of("A")
    wrong[1] = eng.slots_of("B")[0]  # A's lane names B's private slot
    eng.forward(kv_indices=wrong)
    assert kinds(eng.ledger)[0] == "mapping_mismatch"
    assert eng.ledger.reporter.reports[0].policy is Policy.P3


def test_corrupted_out_cache_loc_is_a_slot_map_mismatch():
    eng = Engine()
    eng.prefill("A", 3)
    eng.step()
    eng.decode("A")
    eng.observe()
    eng.forward(out_cache_loc=[eng.slots_of("A")[0]])  # the decode token writes the first slot
    assert kinds(eng.ledger)[0] == "slot_map_mismatch"


def test_committed_positions_nobody_wrote_are_reported():
    """kv_committed_len counts positions no forward wrote."""
    eng = Engine()
    eng.prefill("A", 3)
    eng.new_tokens["A"] = 0  # the batch never ran, yet the request claims the positions
    eng.observe()
    assert kinds(eng.ledger) == ["computed_beyond_filled"]
    assert eng.ledger.reporter.reports[0].policy is Policy.P4


# row reads


class Fake2D:
    """A two-dimensional tensor as ``rows_from_requests`` reads it: ``dim()``, indexing by a
    list of rows and a column slice, ``tolist``."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.gathers = 0

    def dim(self):
        return 2

    def __getitem__(self, key):
        idx, cols = key
        self.gathers += 1
        return Fake2D([self.rows[i][cols] for i in idx])

    def tolist(self):
        return [list(r) for r in self.rows]


def test_rows_from_requests_reads_a_tensor_table_with_one_gather():
    from kvguard.adapters.sglang.translate import _rows_of, rows_from_requests

    table = Fake2D([[0] * 8, [11, 12, 13, 0, 0, 0, 0, 0], [21, 22, 0, 0, 0, 0, 0, 0]])
    assert _rows_of(table, [1, 2], [3, 2]) == [[11, 12, 13], [21, 22]]
    assert table.gathers == 1, "one gather of the held rows, as wide as the longest"
    assert _rows_of(table, [], []) == []
    # A plain sequence table is read row by row, the same result.
    plain = [[0] * 8, [11, 12, 13, 0, 0, 0, 0, 0], [21, 22, 0, 0, 0, 0, 0, 0]]
    assert _rows_of(plain, [1, 2], [3, 2]) == [[11, 12, 13], [21, 22]]

    class KV:
        def __init__(self, idx, n):
            self.req_pool_idx, self.kv_allocated_len, self.kv_committed_len = idx, n, n
            self.cache_protected_len = 0

    class Req:
        def __init__(self, rid, idx, n):
            self.rid, self.kv = rid, KV(idx, n)

    rows = rows_from_requests([Req("A", 1, 3), Req("W", None, 0), Req("B", 2, 2)], table, {"A": 1})
    assert [(r.request_id, r.slots, r.new_tokens) for r in rows] == [
        ("A", (11, 12, 13), 1),
        ("B", (21, 22), 0),
    ]


def test_free_count_matches_the_free_set_without_reading_it():
    from types import SimpleNamespace

    from kvguard.adapters.sglang.translate import (
        free_count_from_allocator,
        free_slots_from_allocator,
    )

    class T:  # a tensor-like with numel and tolist
        def __init__(self, xs):
            self.xs = list(xs)

        def numel(self):
            return len(self.xs)

        def tolist(self):
            return list(self.xs)

        def __len__(self):
            return len(self.xs)

        def __iter__(self):
            return iter(self.xs)

    alloc = SimpleNamespace(free_pages=T([3, 5, 9]), release_pages=T([11]))
    assert free_slots_from_allocator(alloc) == [3, 5, 9, 11]
    assert free_count_from_allocator(alloc) == 4
    paged = SimpleNamespace(get_all_free_pages=lambda: T([16, 32, 64]))
    assert free_count_from_allocator(paged) == len(free_slots_from_allocator(paged)) == 3
    virtual = SimpleNamespace(free_virtual_ids=T([1, 2]))
    assert free_count_from_allocator(virtual) == 2


def test_mapping_dispatches_come_from_the_page_tables_alone():
    """The host tier's mapping tables: the attention over each request's pages as a dense
    block table, the write over the slots the mapping gives the written positions, padding
    lanes as empty padding rows; a hole in the mapping is a hole in the table and a -1 slot."""
    from kvguard.adapters.sglang.translate import build_mapping_dispatches
    from kvguard.core import DenseBlockTable, SlotMap

    pages = {"A": (3, 4), "B": (5, None, 7)}
    (write, attn), loc = build_mapping_dispatches(
        9,
        "kv",
        ["A", "B"],
        [20, 38],
        None,
        pages.__getitem__,
        16,
        mode="decode",
        num_rows_after_padding=4,
        tag="-t",
    )
    assert write.dispatch_id == "write-t@9" and attn.dispatch_id == "attn-t@9"
    assert [r.writes for r in write.rows] == [range(19, 20), range(37, 38)]
    assert isinstance(write.mapping, SlotMap) and loc == [4 * 16 + 3, 7 * 16 + 5]
    assert write.mapping.slots == loc and write.mapping.positions == [19, 37]
    assert isinstance(attn.mapping, DenseBlockTable)
    assert attn.mapping.table == [[3, 4], [5, None, 7], [], []] and attn.mapping.valid == [
        2,
        3,
        0,
        0,
    ]
    assert [r.reads for r in attn.rows[:2]] == [range(0, 20), range(0, 38)]
    assert attn.rows[2].padding and attn.rows[3].padding
    # Extend: B writes [2, 20) into page 5 and the hole, ragged attention reads the prefix.
    (first, second), loc = build_mapping_dispatches(
        10,
        "kv",
        ["B"],
        [20],
        [2],
        pages.__getitem__,
        16,
        mode="extend",
        use_ragged=True,
    )
    assert first.family == "attention" and second.family == "set_kv_buffer"
    assert loc[:14] == [5 * 16 + p for p in range(2, 16)] and loc[14:] == [-1] * 4
    assert first.mapping.table == [[5]] and first.mapping.valid == [1]
    assert second.rows[0].writes == range(2, 20)
    (first, second), _ = build_mapping_dispatches(
        11,
        "kv",
        ["B"],
        [20],
        [2],
        pages.__getitem__,
        16,
        mode="extend",
        use_ragged=False,
    )
    assert first.family == "set_kv_buffer" and second.mapping.valid == [2]


def test_page_set_np_matches_page_set_and_lists_duplicates():
    """The array form of ``page_set`` (the audited walk's hundreds of thousands of slots) gives
    the set form's pages at any page size, and the slots listed twice, sorted."""
    import numpy as np
    from kvguard.adapters.sglang.translate import _np_ints, page_set, page_set_np

    slots = [5, 17, 18, 40, 5, 1000, 1001, 17, 31]
    arr = np.asarray(slots, dtype=np.int64)
    assert page_set_np(arr, 16) == page_set(slots, 16) == {0, 1, 2, 62}
    pages, dups = page_set_np(arr, 16, duplicates=True)
    assert pages == page_set(slots, 16) and dups == [5, 17]
    assert page_set_np(arr, 1) == set(slots)
    assert page_set_np(np.asarray([], dtype=np.int64), 16, duplicates=True) == (set(), [])
    assert page_set_np(np.asarray([], dtype=np.int64), 16) == set()
    # Lists and arrays become int64 arrays without a per-element conversion of an array.
    assert _np_ints(slots).tolist() == slots and _np_ints(arr) is arr
    assert _np_ints(np.asarray(slots, dtype=np.int32)).dtype == np.int64


class _Table:
    """A two-dimensional table with a tensor's reading interface (``dim``, ``tolist`` and the
    ``[rows, cols]`` gathers ``_rows_of``, ``_gather_cells`` and ``_band_of`` use), counting
    the gathers."""

    def __init__(self, rows):
        self.rows = rows
        self.gathers = 0

    def dim(self):
        return 2

    def tolist(self):
        return [list(r) if isinstance(r, list) else r for r in self.rows]

    def __getitem__(self, key):
        rows, cols = key
        self.gathers += 1
        if isinstance(cols, slice):
            return _Table([self.rows[r][cols] for r in rows])
        return _Table([self.rows[r][c] for r, c in zip(rows, cols, strict=True)])


def test_row_slot_cache_reads_only_what_grew_and_audits():
    """The row cache re-reads a new or shrunk row in full, a decode row at its one new position,
    a chunked row over its new columns, keeps an unchanged row, drops a row that left, and gives
    the same slots as a full read, in one gather per kind of read; an audit re-reads every row
    and counts a kept prefix the device changed under it."""
    from types import SimpleNamespace as NS

    from kvguard.adapters.sglang.translate import RowSlotCache, rows_from_requests

    table = _Table([[1000 + 40 * r + c for c in range(40)] for r in range(8)])

    def req(rid, row, n):
        return NS(
            rid=rid,
            kv=NS(req_pool_idx=row, kv_allocated_len=n, kv_committed_len=n, cache_protected_len=0),
        )

    def plain(reqs):
        return [list(r.slots) for r in rows_from_requests(reqs, table)]

    cache = RowSlotCache()
    reqs = [req("a", 3, 5), req("b", 6, 12)]
    got = [list(r.slots) for r in rows_from_requests(reqs, table, cache=cache)]
    assert got == plain(reqs) and cache.stats["rows_full"] == 2
    # a decodes (+1), b chunk-prefills (+7), c is new: three gathers, the table's slots.
    reqs = [req("a", 3, 6), req("b", 6, 19), req("c", 1, 4)]
    table.gathers = 0
    rows = rows_from_requests(reqs, table, cache=cache)
    assert table.gathers == 3 and [list(r.slots) for r in rows] == plain(reqs)
    assert cache.stats["rows_one"] == 1 and cache.stats["rows_more"] == 1
    assert cache.stats["rows_full"] == 3
    # Eight decode rows: one gather. Unchanged rows are kept (the same list objects); a row
    # re-owned or shrunk is read in full; a row that left is dropped.
    reqs = [req(chr(97 + r), r, 10 + r) for r in range(8)]
    rows_from_requests(reqs, table, cache=cache)
    reqs = [req(chr(97 + r), r, 11 + r) for r in range(8)]
    table.gathers = 0
    got = [list(r.slots) for r in rows_from_requests(reqs, table, cache=cache)]
    assert table.gathers == 1 and cache.stats["rows_one"] == 9 and got == plain(reqs)
    kept = cache.rows[3][1]
    reqs = [req("d", 3, 14), req("z", 6, 3), req("c", 2, 2)]
    rows = rows_from_requests(reqs, table, cache=cache)
    assert [list(r.slots) for r in rows] == plain(reqs) and rows[0].slots is kept
    assert set(cache.rows) == {2, 3, 6}
    # An audit re-reads everything; a prefix the device changed under the cache is drift.
    table.rows[3][2] = -1
    rows = rows_from_requests(reqs, table, cache=cache, audit=True)
    assert cache.drift == 1 and list(rows[0].slots) == table.rows[3][:14]
    rows_from_requests(reqs, table, cache=cache, audit=True)
    assert cache.drift == 1 and cache.stats["audited"] == 2
    # A table read row by row (no two-dimensional gather) gives the same answers.
    lists = table.tolist()
    cache2 = RowSlotCache()
    for step in (
        [req("a", 3, 5)],
        [req("a", 3, 6), req("b", 6, 12)],
        [req("a", 3, 6), req("b", 6, 20)],
    ):
        assert [list(r.slots) for r in rows_from_requests(step, lists, cache=cache2)] == plain(step)


def test_row_slot_cache_reads_a_new_owner_or_a_re_prefilled_row_in_full():
    """A later request reusing a public request id on the same row at the same length is a new
    owner (the checker identity) and is read in full; a request whose committed count went
    backwards (retracted and re-prefilled onto its row) is read in full too; with no owners
    given the request id is the owner."""
    from types import SimpleNamespace as NS

    from kvguard.adapters.sglang.translate import RowSlotCache, rows_from_requests

    table = _Table([[1000 + 40 * r + c for c in range(40)] for r in range(4)])

    def req(rid, row, n, committed=None):
        return NS(
            rid=rid,
            kv=NS(
                req_pool_idx=row,
                kv_allocated_len=n,
                kv_committed_len=n if committed is None else committed,
                cache_protected_len=0,
            ),
        )

    cache = RowSlotCache()
    first = rows_from_requests([req("r", 2, 10)], table, cache=cache, owners=["r#kvguard-1"])
    assert list(first[0].slots) == table.rows[2][:10]
    table.rows[2] = [5000 + c for c in range(40)]  # the row rewritten for a new request
    again = rows_from_requests([req("r", 2, 10)], table, cache=cache, owners=["r#kvguard-2"])
    assert list(again[0].slots) == table.rows[2][:10] and cache.stats["rows_full"] == 2
    # The same owner, the same length, a committed count that went backwards: a re-prefill.
    table.rows[2] = [7000 + c for c in range(40)]
    rows_from_requests([req("r", 2, 10, committed=10)], table, cache=cache, owners=["r#kvguard-2"])
    assert cache.stats["rows_full"] == 2  # unchanged: kept
    back = rows_from_requests(
        [req("r", 2, 10, committed=4)], table, cache=cache, owners=["r#kvguard-2"]
    )
    assert list(back[0].slots) == table.rows[2][:10] and cache.stats["rows_full"] == 3
    # Without owners the request id stands for the owner.
    cache2 = RowSlotCache()
    rows_from_requests([req("s", 1, 6)], table, cache=cache2)
    assert cache2.rows[1][0] == "s"


def test_row_slot_cache_rereads_a_checkpointed_row_from_its_old_protected_length():
    """The tree's checkpoint of an unfinished request repoints the row from the old protected
    length to the new one (the canonical slots of an identical prefix): a row whose protected
    length grew is re-read from the old protected length, with its growth, in the same band
    read as the chunked rows; the audit does not count that repointing as drift."""
    from types import SimpleNamespace as NS

    from kvguard.adapters.sglang.translate import RowSlotCache, rows_from_requests

    table = _Table([[1000 + 40 * r + c for c in range(40)] for r in range(4)])

    def req(rid, row, n, protected):
        return NS(
            rid=rid,
            kv=NS(
                req_pool_idx=row,
                kv_allocated_len=n,
                kv_committed_len=n,
                cache_protected_len=protected,
            ),
        )

    cache = RowSlotCache()
    rows_from_requests([req("a", 1, 10, 0), req("b", 2, 12, 4)], table, cache=cache)
    # a is checkpointed to 8 (positions 0..7 repointed) and decodes one position; b's
    # checkpoint moves from 4 to 12 while it grows by three.
    table.rows[1][0:8] = [300 + c for c in range(8)]
    table.rows[2][4:12] = [500 + c for c in range(8)]
    table.gathers = 0
    rows = rows_from_requests([req("a", 1, 11, 8), req("b", 2, 15, 12)], table, cache=cache)
    assert table.gathers == 1 and cache.stats["rows_more"] == 2
    assert cache.stats["rows_repointed"] == 2
    assert list(rows[0].slots) == table.rows[1][:11]
    assert list(rows[1].slots) == table.rows[2][:15]
    # Audited: the repointed prefixes are what the device holds, so no drift.
    rows_from_requests([req("a", 1, 12, 8), req("b", 2, 15, 12)], table, cache=cache, audit=True)
    assert cache.drift == 0
    # A protected length that shrank is a new life of the row: read in full.
    table.rows[1][0] = -7
    rows = rows_from_requests([req("a", 1, 12, 0)], table, cache=cache)
    assert list(rows[0].slots) == table.rows[1][:12]


def test_pages_of_prefix_matches_pages_of_the_prefix():
    from kvguard.adapters.sglang.translate import pages_of, pages_of_prefix

    slots = list(range(100, 200))
    for n in (0, 1, 15, 16, 17, 33, 100):
        assert pages_of_prefix(slots, n, 16) == pages_of(slots[:n], 16)
        assert pages_of_prefix(slots, n, 1) == pages_of(slots[:n], 1)
