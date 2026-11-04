"""Current CUDA stream handle for a thread, used by the stream-order epoch rule; ``None`` when
there is no CUDA, which keeps the core's host-order judgment."""

from __future__ import annotations


def current_stream_handle() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.current_stream().cuda_stream)
    except Exception:
        return None


__all__ = ["current_stream_handle"]
