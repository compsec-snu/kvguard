"""Tier 1b twins under the vLLM attachment: the interposer set-up and the descriptor publisher.

The twins, substituted at launch by the preloaded interposer, read a descriptor naming the
step's block-table geometry, a device vector of row lengths, and the counter and record they
write violations to. :class:`DescriptorPublisher` fills it at every ``prepare_attn`` from the
adapter's view of the batch and reads the counter back at a later step, without synchronizing
the host with the stream.

Two hazards of a host that runs ahead of the stream are handled here:

* **The descriptor's source.** The interposer copies the descriptor into a twin module's
  constant bank from one fixed host buffer (a captured graph holds a memcpy node reading that
  address at replay), so a queued graph from the previous step would read a rewritten
  descriptor. The publisher waits for the stream before a publish that changes the descriptor
  (``desc_waits``).
* **The row lengths' source.** They travel through a ring of pinned host buffers, each with an
  event recorded after its copy was enqueued; a buffer is rewritten only once that copy ran.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from kvguard.core import OutcomeKind, Policy, Report
from kvguard.gpu.interposer import MODE_SUBSTITUTE, Interposer, available, tier1b_descriptor

ROW_RING = 8
"""Pinned row-length buffers in flight at most; the host waits when all are queued."""

KIND_NAMES = {1: "kv_read", 2: "kv_write", 3: "table_read", 4: "table_write", 5: "bounds"}


def enable_substitution() -> Interposer:
    """Put the preloaded interposer into substitution mode with a disabled descriptor (the twins
    check nothing until the first publish). Call before the engine loads its kernels, so the
    twins are what the CUDA graphs capture. Raises ``RuntimeError`` when the library is not
    preloaded."""
    if not available():
        raise RuntimeError(
            "the KVGuard interposer is not preloaded (LD_PRELOAD=build/interposer/libkvg_interposer.so)"
        )
    ip = Interposer()
    ip.set_descriptor(
        tier1b_descriptor(
            counter_ptr=0,
            record_ptr=0,
            table_row_bytes=4,
            table_rows=0,
            row_len_ptr=0,
            block_size=16,
            num_blocks=1,
            enabled=False,
        )
    )
    ip.set_mode(MODE_SUBSTITUTE)
    return ip


class DescriptorPublisher:
    """Fills the twins' Tier 1b descriptor at every ``prepare_attn`` and reads the in-kernel
    counter back without blocking the host (``sync_read=True`` restores the synchronizing read
    of the first measurements)."""

    def __init__(
        self,
        ip: Interposer,
        block_size: int,
        num_blocks: int,
        max_rows: int = 4096,
        *,
        sync_read: bool = False,
        ring: int = ROW_RING,
    ):
        import torch

        self.torch = torch
        self.ip = ip
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.record = torch.zeros(4, dtype=torch.int64, device="cuda")
        self.row_len = torch.zeros(max_rows, dtype=torch.int32, device="cuda")
        self.steps = 0
        self.violation_steps: list[tuple[int, int, list[int]]] = []
        self.stats: Counter = Counter()
        # The row lengths: a ring of pinned buffers with one event each (see the module docstring).
        self.row_host = [
            torch.zeros(max_rows, dtype=torch.int32, pin_memory=True) for _ in range(ring)
        ]
        self.row_events = [torch.cuda.Event() for _ in range(ring)]
        self.row_used = [False] * ring
        self.row_slot = 0
        self.last_desc: list[int] | None = None
        # The counter and record come back through pinned memory and an event; at most one
        # readback is in flight, checked at the next publish.
        self.sync_read = sync_read
        self.pinned = torch.zeros(5, dtype=torch.int64, pin_memory=True)
        self.event = torch.cuda.Event()
        self.pending_step: int | None = None

    # ------------------------------------------------------------------ readback
    def _note(self, step: int, n: int, record: list[int]) -> None:
        self.violation_steps.append((step, n, record))
        self.counter.zero_()
        self.record.zero_()

    def check_previous(self, step: int) -> None:
        if self.sync_read:
            n = int(self.counter.item())
            if n:
                self._note(step - 1, n, self.record.tolist())
            return
        if self.pending_step is not None and self.event.query():
            n = int(self.pinned[0].item())
            if n:
                self._note(self.pending_step, n, self.pinned[1:].tolist())
            self.pending_step = None
        if self.pending_step is None:
            self.pinned[0].copy_(self.counter[0], non_blocking=True)
            self.pinned[1:].copy_(self.record, non_blocking=True)
            self.event.record()
            self.pending_step = step - 1

    def drain(self) -> None:
        """Wait for the stream and read what the last steps left (call once the engine is idle)."""
        self.torch.cuda.synchronize()
        self.pending_step = None
        n = int(self.counter.item())
        if n:
            self._note(-1, n, self.record.tolist())

    # ------------------------------------------------------------------ publish
    def __call__(self, att: Any, input_batch: Any, block_tables: Any) -> None:
        torch = self.torch
        self.check_previous(att.seq)
        n = int(input_batch.num_reqs)
        n_pad = int(input_batch.num_reqs_after_padding)
        if n == 0:
            return
        lens = np.asarray(input_batch.num_computed_tokens_np[:n]) + np.asarray(
            input_batch.num_scheduled_tokens[:n]
        )
        slot = self.row_slot % len(self.row_host)
        event = self.row_events[slot]
        if self.row_used[slot] and not event.query():
            event.synchronize()  # every buffer's copy is still queued: wait for the oldest
            self.stats["row_ring_waits"] += 1
        host = self.row_host[slot]
        host[:n] = torch.from_numpy(np.asarray(lens, dtype=np.int32))
        host[n:n_pad] = 0
        self.row_len[:n_pad].copy_(host[:n_pad], non_blocking=True)
        event.record()
        self.row_used[slot] = True
        self.row_slot += 1
        bt = block_tables[0]
        desc = tier1b_descriptor(
            counter_ptr=self.counter.data_ptr(),
            record_ptr=self.record.data_ptr(),
            table_row_bytes=bt.stride(0) * bt.element_size(),
            table_rows=n_pad,
            row_len_ptr=self.row_len.data_ptr(),
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            entry_bytes=bt.element_size(),
        )
        if desc != self.last_desc:
            # Queued launches (a graph of the previous step) still read the descriptor's host
            # source at execution: let them finish before it changes.
            torch.cuda.current_stream().synchronize()
            self.stats["desc_waits"] += 1
            self.ip.set_descriptor(desc)
            self.last_desc = desc
        self.steps += 1

    # ------------------------------------------------------------------ outcome
    def reports(self) -> list[Report]:
        """One violation report per step whose readback found in-kernel violations (the
        reporter keeps the first; the record carries the total)."""
        out = []
        for step, n, rec in self.violation_steps:
            site, row, col, kind = [*rec, 0, 0, 0, 0][:4]
            out.append(
                Report(
                    OutcomeKind.VIOLATION,
                    "twin_kernel_violation",
                    f"{n} in-kernel violation(s) read back for step {step}; first record: "
                    f"site {site}, row {row}, column {col}, kind {KIND_NAMES.get(kind, kind)}",
                    policy=Policy.P2 if kind in (1, 2) else Policy.P3,
                    site=f"twin@{step}",
                )
            )
        return out

    def summary(self) -> dict[str, Any]:
        subs = self.ip.substitutions()
        return {
            "twin_launches": sum(t["launches"] for t in subs["twins"]),
            "twin_kernels": [t.get("name", "")[:90] for t in subs["twins"]],
            "twin_publishes": self.steps,
            "twin_stats": dict(self.stats),
            "kernel_violations_total": sum(n for _, n, _ in self.violation_steps),
            "kernel_violation_steps": self.violation_steps[:10],
        }


__all__ = ["ROW_RING", "DescriptorPublisher", "enable_substitution"]
