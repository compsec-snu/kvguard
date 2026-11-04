"""The tag tier under the vLLM attachment.

One :class:`~kvguard.gpu.tags.TagMirror` per attention-like KV cache group (the recurrent-state
groups are a boundary of this first wiring), kept from: the stamp and dense check at
``prepare_attn`` (:meth:`dispatch`); the ledger's listener events (mediated frees, offload-path
external fills and copies); and ``BlockPool.cache_full_blocks``'s publish hook (:meth:`publish`).
Runs with or without the device validators (``tags=True`` beside ``gpu_validate``); reports carry
the site ``gpu-tags:<dispatch id>``.
"""

from __future__ import annotations

import time
from typing import Any

from kvguard.gpu.tags import TagTier
from kvguard.gpu.validate import PADDING_ROW


class VllmTags(TagTier):
    def __init__(self, att: Any):
        self.att = att
        groups = (att.kv_gid, *att.aux_attention_gids)
        super().__init__(
            att.ledger,
            [att.pool_of_group[g] for g in groups],
            att.validators,
            att.mirror.owners if att.mirror is not None else None,
        )
        self.group_mirror = {g: self.mirrors[att.pool_of_group[g]] for g in groups}
        self.kv_pool = att.pool_of_group[att.kv_gid]

    def pool_of_tag(self, tag: str) -> str:
        m = TagTier.pool_of_tag(self, tag)
        return m if "-" in tag.split("@", 1)[0] else self.kv_pool

    # -- the dispatch ----------------------------------------------------------------------
    def dispatch(self, g: int, bt: Any, slots: Any, write: Any, attn: Any, req_ids: list) -> None:
        """Stamp the step's slot map and check its block table for group ``g`` on the current
        stream, before the step's kernels (``bt`` and ``slots`` are the group's gathered device
        tensors)."""
        t0 = time.perf_counter()
        mirror = self.group_mirror[g]
        stream = self.stream()
        owner = self.owners.id_of
        n = len(req_ids)
        sm = write.mapping
        tok_owner = [PADDING_ROW if r >= n else owner(req_ids[r]) for r in sm.rows]
        tok_pos = [p & 0xFFFFFFFF for p in sm.positions]
        mirror.stamp(
            slots.data_ptr(), slots.element_size(), len(sm.slots), tok_owner, tok_pos, stream,
            write.dispatch_id,
        )  # fmt: skip
        mirror.check_dense(
            bt.data_ptr(),
            int(bt.stride(0)),
            len(attn.rows),
            int(bt.shape[1]),
            [PADDING_ROW if r.padding else owner(r.request) for r in attn.rows],
            [r.reads.stop for r in attn.rows],
            stream,
            attn.dispatch_id,
            [r.reads.start for r in attn.rows],
        )
        self.host_s += time.perf_counter() - t0

    # -- the publish hook ------------------------------------------------------------------
    def publish(self, block_ids: list) -> None:
        """The blocks ``cache_full_blocks`` hashed, each in the pool of its group."""
        by_pool: dict[str, list[int]] = {}
        group_of, pool_of = self.att.group_of, self.att.pool_of_group
        for b in block_ids:
            by_pool.setdefault(pool_of[group_of.get(b, self.att.kv_gid)], []).append(b)
        stream = None
        for pool, ids in by_pool.items():
            m = self.mirrors.get(pool)
            if m is not None:
                stream = self.stream() if stream is None else stream
                m.publish(ids, stream, self._site("publish"))

    def collect(self, results, reporter=None):  # type: ignore[override]
        return TagTier.collect(self, results, self.att.reporter if reporter is None else reporter)
