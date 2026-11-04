/* A thin C shim over the NVTX memory API (nvtx3, header-only) so that KVGuard's ledger can tell
 * NVIDIA Compute Sanitizer about the KV cache's sub-allocations: the pool tensors as sub-allocator
 * heaps, the blocks as regions registered at allocation and unregistered at the free, read-only
 * permissions for published blocks. Under compute-sanitizer the NVTX calls reach the tool through
 * its injection library (NVTX_INJECTION64_PATH); without a tool they are no-ops. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtMem.h>
#include <nvtx3/nvToolsExtMemCudaRt.h>

static nvtxDomainHandle_t kvg_domain = NULL;

int kvg_nvtx_init(const char* name) {
    if (kvg_domain == NULL) kvg_domain = nvtxDomainCreateA(name);
    return kvg_domain != NULL;
}

void* kvg_nvtx_heap_register(const void* ptr, size_t size) {
    nvtxMemVirtualRangeDesc_t range;
    nvtxMemHeapDesc_t desc;
    memset(&range, 0, sizeof range);
    memset(&desc, 0, sizeof desc);
    range.ptr = ptr;
    range.size = size;
    desc.extCompatID = NVTX_EXT_COMPATID_MEM;
    desc.structSize = sizeof desc;
    desc.usage = NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
    desc.type = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
    desc.typeSpecificDescSize = sizeof range;
    desc.typeSpecificDesc = &range;
    return (void*)nvtxMemHeapRegister(kvg_domain, &desc);
}

void kvg_nvtx_heap_unregister(void* heap) {
    nvtxMemHeapUnregister(kvg_domain, (nvtxMemHeapHandle_t)heap);
}

void kvg_nvtx_regions_register(void* heap, const uint64_t* ptrs, const uint64_t* sizes, size_t n) {
    nvtxMemVirtualRangeDesc_t* ranges = (nvtxMemVirtualRangeDesc_t*)calloc(n, sizeof *ranges);
    nvtxMemRegionsRegisterBatch_t batch;
    size_t i;
    if (ranges == NULL) return;
    for (i = 0; i < n; i++) {
        ranges[i].ptr = (const void*)(uintptr_t)ptrs[i];
        ranges[i].size = (size_t)sizes[i];
    }
    memset(&batch, 0, sizeof batch);
    batch.extCompatID = NVTX_EXT_COMPATID_MEM;
    batch.structSize = sizeof batch;
    batch.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
    batch.heap = (nvtxMemHeapHandle_t)heap;
    batch.regionCount = n;
    batch.regionDescElementSize = sizeof *ranges;
    batch.regionDescElements = ranges;
    batch.regionHandleElementsOut = NULL;
    nvtxMemRegionsRegister(kvg_domain, &batch);
    free(ranges);
}

void kvg_nvtx_regions_unregister(const uint64_t* ptrs, size_t n) {
    nvtxMemRegionRef_t* refs = (nvtxMemRegionRef_t*)calloc(n, sizeof *refs);
    nvtxMemRegionsUnregisterBatch_t batch;
    size_t i;
    if (refs == NULL) return;
    for (i = 0; i < n; i++) refs[i].pointer = (const void*)(uintptr_t)ptrs[i];
    memset(&batch, 0, sizeof batch);
    batch.extCompatID = NVTX_EXT_COMPATID_MEM;
    batch.structSize = sizeof batch;
    batch.refType = NVTX_MEM_REGION_REF_TYPE_POINTER;
    batch.refCount = n;
    batch.refElementSize = sizeof *refs;
    batch.refElements = refs;
    nvtxMemRegionsUnregister(kvg_domain, &batch);
    free(refs);
}

/* flags: NVTX_MEM_PERMISSIONS_REGION_FLAGS_READ 0x1, _WRITE 0x2, _ATOMIC 0x4, _RESET 0x8;
 * applied to the process-wide permissions (kernel launches on every stream). */
void kvg_nvtx_permissions_assign(const uint64_t* ptrs, size_t n, uint32_t flags) {
    nvtxMemPermissionsAssignRegionDesc_t* regions =
        (nvtxMemPermissionsAssignRegionDesc_t*)calloc(n, sizeof *regions);
    nvtxMemPermissionsAssignBatch_t batch;
    size_t i;
    if (regions == NULL) return;
    for (i = 0; i < n; i++) {
        regions[i].flags = flags;
        regions[i].regionRefType = NVTX_MEM_REGION_REF_TYPE_POINTER;
        regions[i].region.pointer = (const void*)(uintptr_t)ptrs[i];
    }
    memset(&batch, 0, sizeof batch);
    batch.extCompatID = NVTX_EXT_COMPATID_MEM;
    batch.structSize = sizeof batch;
    batch.permissions = nvtxMemCudaGetProcessWidePermissions(kvg_domain);
    batch.regionCount = n;
    batch.regionElementSize = sizeof *regions;
    batch.regionElements = regions;
    nvtxMemPermissionsAssign(kvg_domain, &batch);
    free(regions);
}

/* The default access of registered regions for kernels on `device` (applied where no region
 * permission is assigned). Unused: on CUDA 13.2's tool either call kills the sanitized process,
 * so nvtxmem.py assigns every region's permission explicitly instead. Kept for a tool that
 * accepts it. */
void kvg_nvtx_set_default_access(int device, uint32_t flags) {
    nvtxMemCudaSetPeerAccess(kvg_domain, nvtxMemCudaGetProcessWidePermissions(kvg_domain), device, flags);
    nvtxMemCudaSetPeerAccess(kvg_domain, nvtxMemCudaGetDeviceWidePermissions(kvg_domain, device), device, flags);
}

void kvg_nvtx_mark_initialized(const uint64_t* ptrs, const uint64_t* sizes, size_t n, void* stream) {
    nvtxMemVirtualRangeDesc_t* ranges = (nvtxMemVirtualRangeDesc_t*)calloc(n, sizeof *ranges);
    nvtxMemMarkInitializedBatch_t batch;
    size_t i;
    if (ranges == NULL) return;
    for (i = 0; i < n; i++) {
        ranges[i].ptr = (const void*)(uintptr_t)ptrs[i];
        ranges[i].size = (size_t)sizes[i];
    }
    memset(&batch, 0, sizeof batch);
    batch.extCompatID = NVTX_EXT_COMPATID_MEM;
    batch.structSize = sizeof batch;
    batch.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
    batch.regionDescCount = n;
    batch.regionDescElementSize = sizeof *ranges;
    batch.regionDescElements = ranges;
    nvtxMemCudaMarkInitialized(kvg_domain, (cudaStream_t)stream, 0, &batch);
    free(ranges);
}
