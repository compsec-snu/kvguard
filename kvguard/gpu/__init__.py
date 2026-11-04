"""GPU instrumentation toolchain: static provenance analysis of loaded kernels, check insertion
and launch-time activation.

``kvguard.gpu.ptx`` analyzes and rewrites PTX modules, ``kvguard.gpu.ttgir`` analyzes Triton IR,
``kvguard.gpu.cache`` builds the content-addressed cache of instrumented twins from the images the
driver interposer dumps, and ``kvguard.gpu.interposer`` is the ctypes control API of the preloaded
library.
"""
