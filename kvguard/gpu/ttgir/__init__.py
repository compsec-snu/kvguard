"""Must-provenance over Triton's TTGIR (JIT kernels are analyzed at their
compiler IR, where shared buffers are explicit SSA values, so staging a block-table entry through
shared memory keeps its provenance)."""

from .analyze import TTGIRResult, analyze_ttgir, analyze_ttgir_text, parse_ttgir

__all__ = ["TTGIRResult", "analyze_ttgir", "analyze_ttgir_text", "parse_ttgir"]
