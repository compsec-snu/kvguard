"""The tag tier under the SGLang attachment.

One :class:`~kvguard.gpu.tags.TagMirror` over the KV pool, kept from: the stamp of the step's
``out_cache_loc`` and the check of the FlashInfer page list the wrapper reads (:meth:`step`); the
ledger's listener events (mediated frees, host-tier external fills); and the radix cache's
``insert`` publish hook (:meth:`publish`). Runs with or without the device validators
(``tags=True`` beside ``gpu_validate``); reports carry the site ``gpu-tags:<dispatch id>``.
"""

from __future__ import annotations

import time
from typing import Any

from kvguard.gpu.tags import TagTier
from kvguard.gpu.validate import PADDING_ROW


class SglangTags(TagTier):
    def __init__(self, att: Any, pool: str):
        gpu = att.gpu
        pools = [pool]
        if att.state is not None:
            # A hybrid model's recurrent-state pool and int8 checkpoint store: rewritable slots
            # stamped at the state dispatch, copies and quantizations checked via ledger events.
            from .state import CKPT_POOL, STATE_POOL

            pools += [p for p in (STATE_POOL, CKPT_POOL) if p in att.ledger.pools]
        super().__init__(
            att.ledger,
            pools,
            gpu.v if gpu is not None else None,
            gpu.mirror.owners if gpu is not None else None,
        )
        self.att = att
        self.pool = pool
        self.mirror = self.mirrors[pool]
        self.page_size = int(att.page_size)
        self.publishes = 0
        self.published_slots = 0
        self.state_steps = 0

    def state_step(self, desc: Any, stream: int) -> None:
        """Stamp the recurrent-state dispatch's slot map (the active slot each row rewrites in
        place, a tracked row's tracking slot) with the rows' owners, rewriting allowed."""
        m = self.mirrors.get(desc.pool)
        if m is None:
            return
        t0 = time.perf_counter()
        owner = self.owners.id_of
        rows = desc.rows
        sm = desc.mapping
        tok_owner = [PADDING_ROW if rows[r].padding else owner(rows[r].request) for r in sm.rows]
        tok_pos = [int(p) & 0xFFFFFFFF for p in sm.positions]
        m.stamp_host([int(x) for x in sm.slots], tok_owner, tok_pos, stream, desc.dispatch_id, True)
        self.state_steps += 1
        self.host_s += time.perf_counter() - t0

    def step(self, plan: Any, stream: int) -> None:
        """Stamp the step's ``out_cache_loc`` and check its page list on ``stream`` in the
        kernel order (attention first with ``use_ragged``), before the step's kernels."""
        from .attach import validator_rows

        buf = plan.buffers
        if buf is None:
            raise ValueError("the plan carries no device buffers")
        t0 = time.perf_counter()
        owner = self.owners.id_of
        write = next(d for d in plan.descs if d.family == "set_kv_buffer")
        attn = next(d for d in plan.descs if d.family == "attention")
        (rows, row_owner, row_end), (n_tokens, tok_owner) = validator_rows(plan, owner)
        # A sliding-window wrapper's row starts at its window: the descriptor's token table
        # carries the start per row (``TokenTable.first``); a page list starts at zero.
        mapping = attn.mapping
        first_of = getattr(mapping, "first_position", None)
        row_start = [
            int(first_of(r)) if first_of is not None and r < len(attn.rows) else 0
            for r in range(rows)
        ]
        positions = [int(p) & 0xFFFFFFFF for p in write.mapping.positions]
        tok_pos = positions[:n_tokens] + [0] * max(n_tokens - len(positions), 0)
        tok_owner = list(tok_owner)
        for t in range(len(positions), n_tokens):
            tok_owner[t] = PADDING_ROW

        def stamp() -> None:
            self.mirror.stamp(
                buf.loc.data_ptr(), int(buf.loc.element_size()), n_tokens, tok_owner, tok_pos,
                stream, write.dispatch_id,
            )  # fmt: skip

        def check() -> None:
            self.mirror.check_csr(
                buf.indptr.data_ptr(), buf.indices.data_ptr(), buf.last_page_len.data_ptr(),
                rows, row_owner, row_end, stream, attn.dispatch_id, entry_slots=1,
                row_start=row_start,
            )  # fmt: skip

        if plan.attn_first:
            check()
            stamp()
        else:
            stamp()
            check()
        self.host_s += time.perf_counter() - t0

    def publish(self, value: Any) -> None:
        """The KV indices of a key inserted into the radix cache (a device tensor of slots)
        become readable by any request."""
        n = int(value.numel()) if hasattr(value, "numel") else len(value)
        if n == 0:
            return
        if hasattr(value, "data_ptr"):
            v = value.contiguous()
            self.mirror.update_slots(
                2, self.stream(), self._site("publish"), slots_ptr=v.data_ptr(),
                elem_bytes=int(v.element_size()), n=n,
            )  # fmt: skip
        else:
            self.mirror.update_slots(2, self.stream(), self._site("publish"), slots=list(value))
        self.publishes += 1
        self.published_slots += n

    def stats(self) -> dict[str, Any]:
        out = TagTier.stats(self)
        out["tag_publishes"] = self.publishes
        out["tag_published_slots"] = self.published_slots
        out["tag_state_steps"] = self.state_steps
        return out
