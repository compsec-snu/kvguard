"""Minimal ctypes bindings to the CUDA driver API: load a PTX module, allocate device and pinned
host memory, copy asynchronously, launch kernels and record events on a caller-supplied stream.
No torch dependency, so the same code runs in every framework environment. The caller passes the
stream handle (``torch.cuda.current_stream().cuda_stream`` under torch; 0 for the default stream).
"""

from __future__ import annotations

import ctypes as C
from collections.abc import Sequence

CU_JIT_OPTIMIZATION_LEVEL = 7
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_EVENT_DEFAULT = 0
CUDA_ERROR_NOT_READY = 600


class CudaError(RuntimeError):
    pass


class Driver:
    """One handle on ``libcuda``; methods raise :class:`CudaError` on failure."""

    def __init__(self) -> None:
        try:
            self.cu = C.CDLL("libcuda.so.1")
        except OSError as exc:  # pragma: no cover - no driver on the host
            raise CudaError(f"libcuda.so.1 not loadable: {exc}") from None
        cu = self.cu
        cu.cuGetErrorString.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        cu.cuInit.argtypes = [C.c_uint]
        cu.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
        cu.cuCtxSetCurrent.argtypes = [C.c_void_p]
        cu.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
        cu.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
        cu.cuModuleLoadDataEx.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_uint,
            C.POINTER(C.c_int),
            C.POINTER(C.c_void_p),
        ]
        cu.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        cu.cuModuleGetGlobal_v2.argtypes = [
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_size_t),
            C.c_void_p,
            C.c_char_p,
        ]
        cu.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
        cu.cuMemFree_v2.argtypes = [C.c_uint64]
        cu.cuMemAllocHost_v2.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        cu.cuMemFreeHost.argtypes = [C.c_void_p]
        cu.cuMemcpyHtoD_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
        cu.cuMemcpyDtoH_v2.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t]
        cu.cuMemcpyHtoDAsync_v2.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t, C.c_void_p]
        cu.cuMemcpyDtoHAsync_v2.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p]
        cu.cuMemsetD32Async.argtypes = [C.c_uint64, C.c_uint, C.c_size_t, C.c_void_p]
        cu.cuLaunchKernel.argtypes = [
            C.c_void_p,
            C.c_uint,
            C.c_uint,
            C.c_uint,
            C.c_uint,
            C.c_uint,
            C.c_uint,
            C.c_uint,
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        cu.cuEventCreate.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
        cu.cuEventRecord.argtypes = [C.c_void_p, C.c_void_p]
        cu.cuEventQuery.argtypes = [C.c_void_p]
        cu.cuEventSynchronize.argtypes = [C.c_void_p]
        cu.cuEventElapsedTime.argtypes = [C.POINTER(C.c_float), C.c_void_p, C.c_void_p]
        cu.cuEventDestroy_v2.argtypes = [C.c_void_p]
        cu.cuStreamSynchronize.argtypes = [C.c_void_p]
        cu.cuFuncGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_void_p]
        self.check(cu.cuInit(0), "cuInit")

    def check(self, r: int, what: str) -> None:
        if r != 0:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise CudaError(f"{what}: CUDA error {r} {s.value.decode() if s.value else ''}")

    def ensure_context(self, device: int = 0) -> None:
        """Make the device's primary context current when no context is (torch makes one)."""
        ctx = C.c_void_p()
        self.check(self.cu.cuCtxGetCurrent(C.byref(ctx)), "cuCtxGetCurrent")
        if ctx.value:
            return
        dev = C.c_int()
        self.check(self.cu.cuDeviceGet(C.byref(dev), device), "cuDeviceGet")
        self.check(self.cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self.check(self.cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def load_ptx(self, ptx: str, opt_level: int = 4) -> int:
        mod = C.c_void_p()
        log = C.create_string_buffer(1 << 16)
        opts = (C.c_int * 3)(
            CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        )
        vals = (C.c_void_p * 3)(C.c_void_p(opt_level), C.cast(log, C.c_void_p), C.c_void_p(1 << 16))
        r = self.cu.cuModuleLoadDataEx(C.byref(mod), ptx.encode() + b"\0", 3, opts, vals)
        if r != 0:
            raise CudaError(f"cuModuleLoadDataEx failed ({r}): {log.value.decode()[:4000]}")
        return mod.value

    def function(self, module: int, name: str) -> int:
        f = C.c_void_p()
        self.check(
            self.cu.cuModuleGetFunction(C.byref(f), C.c_void_p(module), name.encode()),
            f"cuModuleGetFunction {name}",
        )
        return f.value

    def global_ptr(self, module: int, name: str) -> tuple[int, int]:
        p, sz = C.c_uint64(), C.c_size_t()
        self.check(
            self.cu.cuModuleGetGlobal_v2(
                C.byref(p), C.byref(sz), C.c_void_p(module), name.encode()
            ),
            f"cuModuleGetGlobal {name}",
        )
        return p.value, sz.value

    def num_regs(self, f: int) -> int:
        v = C.c_int()
        self.check(self.cu.cuFuncGetAttribute(C.byref(v), 4, C.c_void_p(f)), "NUM_REGS")
        return v.value

    def mem_alloc(self, nbytes: int) -> int:
        p = C.c_uint64()
        self.check(self.cu.cuMemAlloc_v2(C.byref(p), max(nbytes, 1)), "cuMemAlloc")
        return p.value

    def mem_free(self, ptr: int) -> None:
        self.check(self.cu.cuMemFree_v2(C.c_uint64(ptr)), "cuMemFree")

    def mem_alloc_host(self, nbytes: int) -> int:
        """Page-locked host memory (the source and target of asynchronous copies)."""
        p = C.c_void_p()
        self.check(self.cu.cuMemAllocHost_v2(C.byref(p), max(nbytes, 1)), "cuMemAllocHost")
        return p.value

    def mem_free_host(self, ptr: int) -> None:
        self.check(self.cu.cuMemFreeHost(C.c_void_p(ptr)), "cuMemFreeHost")

    def memcpy_htod(self, dst: int, src: int, nbytes: int) -> None:
        self.check(self.cu.cuMemcpyHtoD_v2(C.c_uint64(dst), C.c_void_p(src), nbytes), "HtoD")

    def memcpy_dtoh(self, dst: int, src: int, nbytes: int) -> None:
        self.check(self.cu.cuMemcpyDtoH_v2(C.c_void_p(dst), C.c_uint64(src), nbytes), "DtoH")

    def memcpy_htod_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self.check(
            self.cu.cuMemcpyHtoDAsync_v2(
                C.c_uint64(dst), C.c_void_p(src), nbytes, C.c_void_p(stream)
            ),
            "HtoDAsync",
        )

    def memcpy_dtoh_async(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        self.check(
            self.cu.cuMemcpyDtoHAsync_v2(
                C.c_void_p(dst), C.c_uint64(src), nbytes, C.c_void_p(stream)
            ),
            "DtoHAsync",
        )

    def memset_d32_async(self, dst: int, value: int, n_words: int, stream: int) -> None:
        self.check(
            self.cu.cuMemsetD32Async(
                C.c_uint64(dst), value & 0xFFFFFFFF, n_words, C.c_void_p(stream)
            ),
            "MemsetD32Async",
        )

    def launch(
        self,
        f: int,
        grid: Sequence[int],
        block: Sequence[int],
        args: Sequence[C._SimpleCData],
        stream: int,
        smem: int = 0,
    ) -> None:
        """Launch ``f`` with ``args`` (ctypes scalars: ``c_void_p`` for pointers, ``c_uint32`` for
        32-bit values) on ``stream``."""
        ptrs = (C.c_void_p * len(args))(*[C.addressof(a) for a in args])
        gx, gy, gz = ([*list(grid), 1, 1])[:3]
        bx, by, bz = ([*list(block), 1, 1])[:3]
        self.check(
            self.cu.cuLaunchKernel(
                C.c_void_p(f), gx, gy, gz, bx, by, bz, smem, C.c_void_p(stream), ptrs, None
            ),
            "cuLaunchKernel",
        )

    def event_create(self) -> int:
        e = C.c_void_p()
        self.check(self.cu.cuEventCreate(C.byref(e), CU_EVENT_DEFAULT), "cuEventCreate")
        return e.value

    def event_record(self, event: int, stream: int) -> None:
        self.check(self.cu.cuEventRecord(C.c_void_p(event), C.c_void_p(stream)), "cuEventRecord")

    def event_done(self, event: int) -> bool:
        r = self.cu.cuEventQuery(C.c_void_p(event))
        if r == CUDA_ERROR_NOT_READY:
            return False
        self.check(r, "cuEventQuery")
        return True

    def event_synchronize(self, event: int) -> None:
        self.check(self.cu.cuEventSynchronize(C.c_void_p(event)), "cuEventSynchronize")

    def event_elapsed_ms(self, start: int, end: int) -> float:
        ms = C.c_float()
        self.check(
            self.cu.cuEventElapsedTime(C.byref(ms), C.c_void_p(start), C.c_void_p(end)),
            "cuEventElapsedTime",
        )
        return float(ms.value)

    def event_destroy(self, event: int) -> None:
        self.check(self.cu.cuEventDestroy_v2(C.c_void_p(event)), "cuEventDestroy")

    def stream_synchronize(self, stream: int) -> None:
        self.check(self.cu.cuStreamSynchronize(C.c_void_p(stream)), "cuStreamSynchronize")


__all__ = ["CudaError", "Driver"]
