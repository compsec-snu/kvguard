"""Shared builders for the core tests: a small pool, snapshots and dispatches in a few lines."""

from __future__ import annotations

import pytest
from kvguard.core import (
    BlockSnapshot,
    BlockState,
    DenseBlockTable,
    DispatchDescriptor,
    DispatchRow,
    Ledger,
    PoolGeometry,
    Reporter,
    RequestSnapshot,
    Snapshot,
)

BS = 4  # slots per block in the test pool
POOL = "kv"


@pytest.fixture
def geometry() -> PoolGeometry:
    return PoolGeometry(pool=POOL, num_blocks=16, block_size=BS, reserved_blocks=frozenset({0}))


@pytest.fixture
def ledger(geometry) -> Ledger:
    lg = Ledger(Reporter(fail_fast=False))
    lg.register_pool(geometry)
    return lg


class World:
    """Builds snapshots incrementally so tests read like a scheduler's steps."""

    def __init__(self, ledger: Ledger, pool: str = POOL):
        self.ledger = ledger
        self.pool = pool
        self.seq = 0
        self.requests: dict[str, RequestSnapshot] = {}
        self.blocks: dict[int, BlockSnapshot] = {}
        self.free: set[int] | None = None

    # -- state construction
    def request(self, rid, blocks, num_tokens, num_computed=0, first_token=0):
        """Declare a request's mapping. ``num_computed`` defaults to 0: the framework may only
        count positions as computed after a write dispatch (see ``write``)."""
        self.requests[rid] = RequestSnapshot(
            rid, self.pool, tuple(blocks), num_tokens, num_computed, first_token
        )
        return self

    def write(self, rid, start, stop, did=None):
        """Run a write dispatch for ``rid`` over positions [start, stop) and complete it, then
        record the positions as computed in the request's next snapshot."""
        did = did or f"w{self.seq}-{rid}-{start}-{stop}"
        self.dispatch(did, [row(rid, writes=range(start, stop))]).complete(did)
        rs = self.requests[rid]
        self.requests[rid] = RequestSnapshot(
            rs.request,
            rs.pool,
            rs.blocks,
            rs.num_tokens,
            max(rs.num_computed, stop),
            rs.first_token,
        )
        return self

    def drop_request(self, rid):
        self.requests.pop(rid, None)
        return self

    def private(self, *blocks):
        for b in blocks:
            self.blocks[b] = BlockSnapshot(self.pool, b, BlockState.PRIVATE)
        return self

    def published(self, *blocks, reserved=False):
        for b in blocks:
            self.blocks[b] = BlockSnapshot(self.pool, b, BlockState.PUBLISHED, reserved=reserved)
        return self

    def freed(self, *blocks):
        for b in blocks:
            self.blocks.pop(b, None)
        return self

    # -- events
    def alloc(self, *blocks):
        self.ledger.allocate(self.pool, blocks)
        self.private(*blocks)
        return self

    def release(self, *blocks):
        self.ledger.free(self.pool, blocks)
        self.freed(*blocks)
        return self

    def observe(self, stream=None, after=(), free_count=None):
        self.seq += 1
        snap = Snapshot(
            self.seq,
            tuple(self.requests.values()),
            tuple(self.blocks.values()),
            {self.pool: frozenset(self.free)} if self.free is not None else None,
            free_count={self.pool: free_count} if free_count is not None else None,
            stream=stream,
            after=tuple(after),
        )
        self.ledger.observe(snap)
        return self

    def dispatch(self, did, rows, mapping=None, family="attn", stream=None, after=()):
        if mapping is None:
            table, valid = [], []
            for r in rows:
                rs = self.requests.get(r.request) if r.request else None
                if rs is None:
                    table.append([])
                    valid.append(0)
                else:
                    table.append(list(rs.blocks))
                    valid.append(-(-rs.num_tokens // BS))
            mapping = DenseBlockTable(table, valid)
        self.ledger.dispatch(
            DispatchDescriptor(
                did, self.pool, family, tuple(rows), mapping, stream=stream, after=tuple(after)
            )
        )
        return self

    def complete(self, did):
        self.ledger.complete(did)
        return self


@pytest.fixture
def world(ledger) -> World:
    return World(ledger)


def kinds(ledger: Ledger) -> list[str]:
    return [r.kind for r in ledger.reporter.reports]


def row(rid, reads=range(0), writes=range(0), **kw) -> DispatchRow:
    return DispatchRow(rid, reads, writes, **kw)
