// KVGuard driver interposer: intercept every CUDA module and library load, enumerate the kernels
// each one carries, count launches per kernel, and write a JSON summary at exit. This is the
// "collection" half: the Python toolchain analyzes the recorded images offline; the
// "enforcement" half (swapping instrumented binaries per launch, graph node updates) builds on
// the same hooks.
//
// Interception works at three levels so that every path to a kernel is covered:
//   1. Direct dynamic linking against libcuda.so.1: this library exports the same symbol names,
//      so LD_PRELOAD interposition routes PLT calls here.
//   2. Entry points obtained at run time: the statically linked CUDA runtime in torch, Triton's
//      driver shim and cuda-python all dlopen libcuda and fetch functions with dlsym or
//      cuGetProcAddress. Both are interposed: dlsym returns this library's wrappers for the
//      intercepted names, and cuGetProcAddress does the same while remembering the real entry
//      point (per symbol and per default-stream flavour) for the wrapper to call.
//   3. The CUDA runtime (libcudart) does not look driver entry points up by name at all: it uses
//      the driver's private export tables, so its module loads and launches are invisible at the
//      driver level. Its own public entry points are PLT calls from torch and every extension,
//      so the library exports __cudaRegisterFatBinary / __cudaRegisterFunction (every fat binary
//      a host object registers, with the kernel names it maps to host stubs), the launch entries
//      that CUDA 13 host stubs call (__cudaGetKernel resolves a stub to a cudaKernel_t once, then
//      __cudaLaunchKernel / __cudaLaunchKernel_ptsz launch by handle; see crt/device_functions.h)
//      and the public cudaLaunchKernel / cudaLaunchKernelExC that older stubs and hand-written
//      launches use (FlashInfer's dispatchers launch through the public entry, so substitution
//      covers it too). Every runtime launch is attributed to its kernel name through those maps.
//
// Environment:
//   KVGUARD_INTERPOSER_OUT   path of the JSON summary (default: kvg-interposer.json in cwd)
//   KVGUARD_INTERPOSER_DUMP  directory to save the images carrying KV-family kernels as <hash>.<ptx|cubin|fatbin>
//   KVGUARD_INTERPOSER_FULL  1 to list every kernel name per module and dump every image
//   KVGUARD_INTERPOSER_LOG   1 to log each intercepted call to stderr
//   KVGUARD_INTERPOSER_CACHE directory of instrumented kernels built by `python -m kvguard.gpu.cache`
//                            (<cache>/<image hash>/<FNV-1a of the kernel name>.cubin|.ptx)
//   KVGUARD_INTERPOSER_MODE  initial substitution mode: 0 pass-through (default), 1 substitute
//
// Enforcement half: with a cache configured and the mode on, a runtime launch
// of a kernel that has an instrumented twin in the cache is redirected to the twin, which takes
// the same parameters; its `__kvg_desc` descriptor is refreshed on the launch stream whenever the
// host copy changed. The host copy lives in page-locked memory: while a stream is being captured
// into a CUDA graph the refresh is always issued, so the graph holds a memcpy node that re-reads
// the current descriptor at every replay, and a twin is never loaded during capture (the warm-up
// launches load it before vLLM captures). The control API (kvg_set_mode, kvg_set_descriptor,
// kvg_substitutions) is called from Python through ctypes (`kvguard.gpu.interposer`).
//
// In pass-through mode nothing here changes kernel behaviour: every wrapper calls the real
// function with the original arguments and only records what it saw.

#include <cuda.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/stat.h>
#include <unistd.h>

// cuda.h maps the unversioned names to their current versions; this library defines the real
// exported names of both the legacy and the current entry points, so the macros must not apply.
#undef cuGetProcAddress
#undef cuLaunchKernel
#undef cuLaunchKernelEx

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define KVG_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

std::mutex g_mu;
bool g_log = false;

struct ModuleInfo {
    std::string kind;     // ptx | cubin | fatbin | file | unknown
    std::string origin;   // which API loaded it
    size_t size = 0;
    uint64_t hash = 0;
    size_t kernel_count = 0;
    std::vector<std::string> kernels;  // KV-pattern names only, unless KVGUARD_INTERPOSER_FULL=1
};

// Kernel names worth listing per module: the KV-touching families of the three frameworks.
bool kv_pattern(const std::string& n) {
    static const char* pats[] = {"flash_fwd", "reshape_and_cache", "concat_and_cache", "unified_attention", "paged",
                                 "BatchPrefill", "BatchDecode", "xqa", "mla", "gather_cache", "cp_gather", "attention",
                                 "fmha", "kv_cache", "copy_blocks"};
    for (const char* p : pats)
        if (n.find(p) != std::string::npos) return true;
    return false;
}

std::map<const void*, ModuleInfo> g_modules;  // CUmodule or CUlibrary handle -> info
std::map<std::string, uint64_t> g_launches;   // kernel name -> launch count
std::map<const void*, std::string> g_func_names;  // CUfunction / CUkernel handle -> name
std::map<const void*, const void*> g_func_module;  // CUfunction / CUkernel handle -> CUmodule / CUlibrary
std::map<std::string, uint64_t> g_calls;      // intercepted API -> call count
std::atomic<bool> g_summary_written{false};

uint64_t fnv1a(const void* data, size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    uint64_t h = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// glibc's own dlsym and dlvsym, found by walking libc's dynamic symbol table through its link
// map. This library interposes both functions, so neither can be used to find the other without
// recursion; dlopen and dlinfo are not interposed and suffice for the walk.
void* libc_symbol(const char* name) {
    void* h = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return nullptr;
    struct link_map* lm = nullptr;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || !lm) return nullptr;
    const ElfW(Sym)* symtab = nullptr;
    const char* strtab = nullptr;
    const uint32_t* gnu_hash = nullptr;
    const uint32_t* hash = nullptr;
    for (const ElfW(Dyn)* dyn = lm->l_ld; dyn->d_tag != DT_NULL; ++dyn) {
        switch (dyn->d_tag) {
            case DT_SYMTAB: symtab = reinterpret_cast<const ElfW(Sym)*>(dyn->d_un.d_ptr); break;
            case DT_STRTAB: strtab = reinterpret_cast<const char*>(dyn->d_un.d_ptr); break;
            case DT_GNU_HASH: gnu_hash = reinterpret_cast<const uint32_t*>(dyn->d_un.d_ptr); break;
            case DT_HASH: hash = reinterpret_cast<const uint32_t*>(dyn->d_un.d_ptr); break;
            default: break;
        }
    }
    if (!symtab || !strtab) return nullptr;
    size_t nsyms = 0;
    if (hash) {
        nsyms = hash[1];
    } else if (gnu_hash) {
        uint32_t nbuckets = gnu_hash[0], symoffset = gnu_hash[1], bloom_size = gnu_hash[2];
        const ElfW(Addr)* bloom = reinterpret_cast<const ElfW(Addr)*>(gnu_hash + 4);
        const uint32_t* buckets = reinterpret_cast<const uint32_t*>(bloom + bloom_size);
        const uint32_t* chains = buckets + nbuckets;
        uint32_t maxsym = 0;
        for (uint32_t i = 0; i < nbuckets; ++i) if (buckets[i] > maxsym) maxsym = buckets[i];
        if (maxsym >= symoffset) {
            while (!(chains[maxsym - symoffset] & 1)) ++maxsym;
        }
        nsyms = maxsym + 1;
    }
    for (size_t i = 0; i < nsyms; ++i) {
        if (symtab[i].st_shndx == SHN_UNDEF || symtab[i].st_value == 0) continue;
        if (std::strcmp(strtab + symtab[i].st_name, name) == 0)
            return reinterpret_cast<void*>(lm->l_addr + symtab[i].st_value);
    }
    return nullptr;
}

using dlsym_t = void* (*)(void*, const char*);
using dlvsym_t = void* (*)(void*, const char*, const char*);

dlsym_t real_dlsym() {
    static dlsym_t fn = reinterpret_cast<dlsym_t>(libc_symbol("dlsym"));
    return fn;
}

dlvsym_t real_dlvsym() {
    static dlvsym_t fn = reinterpret_cast<dlvsym_t>(libc_symbol("dlvsym"));
    return fn;
}

void* libcuda_handle() {
    static void* h = nullptr;
    if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY);
    return h;
}

template <typename F>
F real_fn(const char* name) {
    void* h = libcuda_handle();
    return h ? reinterpret_cast<F>(real_dlsym()(h, name)) : nullptr;
}

// Entry points recorded from cuGetProcAddress, keyed by symbol; the per-thread-default-stream
// flavour (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) is stored separately.
std::map<std::string, void*> g_real[2];

template <typename F>
F real_for(const char* symbol, int ptsz, const char* fallback_name) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_real[ptsz].find(symbol);
        if (it != g_real[ptsz].end()) return reinterpret_cast<F>(it->second);
    }
    return real_fn<F>(fallback_name);
}

// Recording.

std::string image_kind(const void* image, size_t* size_out) {
    const unsigned char* p = static_cast<const unsigned char*>(image);
    if (!p) return "unknown";
    if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
        // ELF64 cubin: the image ends where its last section, its section header table or its
        // program header table ends (nvcc puts the program headers after the section headers, so
        // the section-table end alone was 224 bytes short and dumped cubins did not match
        // Triton's cache files).
        if (size_out) {
            uint64_t phoff, shoff; uint16_t phentsize, phnum, shentsize, shnum;
            std::memcpy(&phoff, p + 0x20, 8);
            std::memcpy(&shoff, p + 0x28, 8);
            std::memcpy(&phentsize, p + 0x36, 2);
            std::memcpy(&phnum, p + 0x38, 2);
            std::memcpy(&shentsize, p + 0x3a, 2);
            std::memcpy(&shnum, p + 0x3c, 2);
            uint64_t end = shoff + static_cast<uint64_t>(shentsize) * shnum;
            uint64_t ph_end = phoff + static_cast<uint64_t>(phentsize) * phnum;
            if (phnum && ph_end > end) end = ph_end;
            for (uint16_t i = 0; i < shnum; ++i) {
                const unsigned char* sh = p + shoff + static_cast<uint64_t>(i) * shentsize;
                uint32_t type; uint64_t off, size;
                std::memcpy(&type, sh + 0x04, 4);
                std::memcpy(&off, sh + 0x18, 8);
                std::memcpy(&size, sh + 0x20, 8);
                if (type != 8 /* SHT_NOBITS */ && off + size > end) end = off + size;
            }
            *size_out = static_cast<size_t>(end);
        }
        return "cubin";
    }
    uint32_t magic;
    std::memcpy(&magic, p, 4);
    if (magic == 0xBA55ED50u) {  // fatbin wrapper
        // struct { u32 magic; u32 version; u64 size; } header, data follows
        uint64_t sz;
        std::memcpy(&sz, p + 8, 8);
        if (size_out) *size_out = static_cast<size_t>(sz + 16);
        return "fatbin";
    }
    // PTX is text; size is the string length.
    if (size_out) *size_out = std::strlen(reinterpret_cast<const char*>(p));
    return "ptx";
}

// Saves one copy of each distinct image that carries a KV-family kernel (every image with
// KVGUARD_INTERPOSER_FULL). Runtime fat binaries are dumped from __cudaRegisterFatBinaryEnd,
// once their kernels are registered.
void dump_image(const ModuleInfo& m, const void* image) {
    const char* dir = std::getenv("KVGUARD_INTERPOSER_DUMP");
    if (!dir || !image || !m.size || m.kernels.empty()) return;
    mkdir(dir, 0755);
    char name[512];
    std::snprintf(name, sizeof(name), "%s/%016llx.%s", dir, static_cast<unsigned long long>(m.hash), m.kind.c_str());
    std::ifstream probe(name);
    if (probe.good()) return;  // already dumped
    std::ofstream out(name, std::ios::binary);
    out.write(static_cast<const char*>(image), static_cast<std::streamsize>(m.size));
}

void record_module(const void* handle, const void* image, const char* origin, const std::vector<std::string>& kernels) {
    ModuleInfo m;
    m.origin = origin;
    if (image) {
        m.kind = image_kind(image, &m.size);
        m.hash = fnv1a(image, m.size);
    } else {
        m.kind = "file";
    }
    m.kernel_count = kernels.size();
    static const bool full = std::getenv("KVGUARD_INTERPOSER_FULL") != nullptr;
    for (const std::string& k : kernels)
        if (full || kv_pattern(k)) m.kernels.push_back(k);
    dump_image(m, image);
    std::lock_guard<std::mutex> lk(g_mu);
    g_modules[handle] = m;
    g_calls[origin] += 1;
    if (g_log) std::fprintf(stderr, "[kvg] %s: %s %zu bytes, %zu kernels, hash %016llx\n", origin, m.kind.c_str(), m.size, kernels.size(), static_cast<unsigned long long>(m.hash));
}

std::vector<std::string> enumerate_module(CUmodule mod) {
    std::vector<std::string> names;
    auto count_fn = real_fn<CUresult (*)(unsigned int*, CUmodule)>("cuModuleGetFunctionCount");
    auto enum_fn = real_fn<CUresult (*)(CUfunction*, unsigned int, CUmodule)>("cuModuleEnumerateFunctions");
    auto name_fn = real_fn<CUresult (*)(const char**, CUfunction)>("cuFuncGetName");
    if (!count_fn || !enum_fn || !name_fn) return names;
    unsigned int n = 0;
    if (count_fn(&n, mod) != CUDA_SUCCESS || n == 0) return names;
    std::vector<CUfunction> fns(n);
    if (enum_fn(fns.data(), n, mod) != CUDA_SUCCESS) return names;
    for (CUfunction f : fns) {
        const char* nm = nullptr;
        if (name_fn(&nm, f) == CUDA_SUCCESS && nm) {
            names.emplace_back(nm);
            std::lock_guard<std::mutex> lk(g_mu);
            g_func_names[f] = nm;
        }
    }
    return names;
}

std::vector<std::string> enumerate_library(CUlibrary lib) {
    std::vector<std::string> names;
    auto count_fn = real_fn<CUresult (*)(unsigned int*, CUlibrary)>("cuLibraryGetKernelCount");
    auto enum_fn = real_fn<CUresult (*)(CUkernel*, unsigned int, CUlibrary)>("cuLibraryEnumerateKernels");
    auto name_fn = real_fn<CUresult (*)(const char**, CUkernel)>("cuKernelGetName");
    if (!count_fn || !enum_fn || !name_fn) return names;
    unsigned int n = 0;
    if (count_fn(&n, lib) != CUDA_SUCCESS || n == 0) return names;
    std::vector<CUkernel> ks(n);
    if (enum_fn(ks.data(), n, lib) != CUDA_SUCCESS) return names;
    for (CUkernel k : ks) {
        const char* nm = nullptr;
        if (name_fn(&nm, k) == CUDA_SUCCESS && nm) {
            names.emplace_back(nm);
            std::lock_guard<std::mutex> lk(g_mu);
            g_func_names[k] = nm;
        }
    }
    return names;
}

void record_launch(CUfunction f) {
    std::string name;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_func_names.find(f);
        if (it != g_func_names.end()) name = it->second;
    }
    if (name.empty()) {
        auto name_fn = real_fn<CUresult (*)(const char**, CUfunction)>("cuFuncGetName");
        const char* nm = nullptr;
        if (name_fn && name_fn(&nm, f) == CUDA_SUCCESS && nm) name = nm;
        if (name.empty()) name = "<unnamed>";
        std::lock_guard<std::mutex> lk(g_mu);
        g_func_names[f] = name;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_launches[name] += 1;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { o += ' '; }
        else o += c;
    }
    return o;
}

std::string substitutions_json_locked();  // enforcement section, below

// Written at exit. Processes that recorded nothing (torch's compile workers inherit LD_PRELOAD)
// do not touch the file, so the engine process's summary survives.
void write_summary_locked() {
    if (g_modules.empty() && g_launches.empty()) return;
    const char* path = std::getenv("KVGUARD_INTERPOSER_OUT");
    std::ofstream out(path ? path : "kvg-interposer.json");
    out << "{\n  \"pid\": " << static_cast<long>(getpid()) << ",\n  \"calls\": {";
    bool first = true;
    for (auto& [k, v] : g_calls) { out << (first ? "" : ", ") << "\"" << k << "\": " << v; first = false; }
    out << "},\n  \"modules\": [\n";
    first = true;
    for (auto& [h, m] : g_modules) {
        out << (first ? "" : ",\n") << "    {\"kind\": \"" << m.kind << "\", \"origin\": \"" << m.origin << "\", \"size\": " << m.size
            << ", \"hash\": \"" << std::hex << m.hash << std::dec << "\", \"kernel_count\": " << m.kernel_count << ", \"kv_kernels\": [";
        for (size_t i = 0; i < m.kernels.size(); ++i) out << (i ? ", " : "") << "\"" << json_escape(m.kernels[i]) << "\"";
        out << "]}";
        first = false;
    }
    out << "\n  ],\n  \"substitutions\": " << substitutions_json_locked() << ",\n  \"launches\": {\n";
    first = true;
    for (auto& [k, v] : g_launches) { out << (first ? "" : ",\n") << "    \"" << json_escape(k) << "\": " << v; first = false; }
    out << "\n  }\n}\n";
}

void write_summary() {
    if (g_summary_written.exchange(true)) return;
    std::lock_guard<std::mutex> lk(g_mu);
    write_summary_locked();
}

}  // namespace

// Wrappers.

KVG_EXPORT CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    auto real = real_for<CUresult (*)(CUmodule*, const void*)>("cuModuleLoadData", 0, "cuModuleLoadData");
    CUresult r = real(module, image);
    if (r == CUDA_SUCCESS) record_module(*module, image, "cuModuleLoadData", enumerate_module(*module));
    return r;
}

KVG_EXPORT CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    auto real = real_for<CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**)>("cuModuleLoadDataEx", 0, "cuModuleLoadDataEx");
    CUresult r = real(module, image, numOptions, options, optionValues);
    if (r == CUDA_SUCCESS) record_module(*module, image, "cuModuleLoadDataEx", enumerate_module(*module));
    return r;
}

KVG_EXPORT CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    auto real = real_for<CUresult (*)(CUmodule*, const void*)>("cuModuleLoadFatBinary", 0, "cuModuleLoadFatBinary");
    CUresult r = real(module, fatCubin);
    if (r == CUDA_SUCCESS) record_module(*module, fatCubin, "cuModuleLoadFatBinary", enumerate_module(*module));
    return r;
}

KVG_EXPORT CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    auto real = real_for<CUresult (*)(CUmodule*, const char*)>("cuModuleLoad", 0, "cuModuleLoad");
    CUresult r = real(module, fname);
    if (r == CUDA_SUCCESS) record_module(*module, nullptr, "cuModuleLoad", enumerate_module(*module));
    return r;
}

KVG_EXPORT CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions,
                                      CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    auto real = real_for<CUresult (*)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int)>("cuLibraryLoadData", 0, "cuLibraryLoadData");
    CUresult r = real(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    if (r == CUDA_SUCCESS) record_module(*library, code, "cuLibraryLoadData", enumerate_library(*library));
    return r;
}

KVG_EXPORT CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions,
                                          CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    auto real = real_for<CUresult (*)(CUlibrary*, const char*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int)>("cuLibraryLoadFromFile", 0, "cuLibraryLoadFromFile");
    CUresult r = real(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    if (r == CUDA_SUCCESS) record_module(*library, nullptr, "cuLibraryLoadFromFile", enumerate_library(*library));
    return r;
}

KVG_EXPORT CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    auto real = real_for<CUresult (*)(CUfunction*, CUmodule, const char*)>("cuModuleGetFunction", 0, "cuModuleGetFunction");
    CUresult r = real(hfunc, hmod, name);
    if (r == CUDA_SUCCESS && name) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_func_names[*hfunc] = name;
        g_func_module[*hfunc] = hmod;
        g_calls["cuModuleGetFunction"] += 1;
    }
    return r;
}

KVG_EXPORT CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    auto real = real_for<CUresult (*)(CUkernel*, CUlibrary, const char*)>("cuLibraryGetKernel", 0, "cuLibraryGetKernel");
    CUresult r = real(pKernel, library, name);
    if (r == CUDA_SUCCESS && name) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_func_names[*pKernel] = name;
        g_func_module[*pKernel] = library;
        g_calls["cuLibraryGetKernel"] += 1;
    }
    return r;
}

namespace {
bool substitute_driver_launch(int ptsz, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                              unsigned smem, CUstream stream, void** params, void** extra, CUresult* rc);
bool substitute_driver_launch_ex(int ptsz, const CUlaunchConfig* config, CUfunction f, void** params, void** extra, CUresult* rc);
}

static CUresult launch_kernel_impl(int ptsz, CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
                                   unsigned int smem, CUstream stream, void** params, void** extra) {
    auto real = real_for<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**)>(
        "cuLaunchKernel", ptsz, ptsz ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    record_launch(f);
    CUresult rc = CUDA_SUCCESS;
    if (substitute_driver_launch(ptsz, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra, &rc)) return rc;
    return real(f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
}

KVG_EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
                                   unsigned int smem, CUstream stream, void** params, void** extra) {
    return launch_kernel_impl(0, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
}

KVG_EXPORT CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
                                        unsigned int smem, CUstream stream, void** params, void** extra) {
    return launch_kernel_impl(1, f, gx, gy, gz, bx, by, bz, smem, stream, params, extra);
}

static CUresult launch_kernel_ex_impl(int ptsz, const CUlaunchConfig* config, CUfunction f, void** params, void** extra) {
    auto real = real_for<CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**)>("cuLaunchKernelEx", ptsz, ptsz ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx");
    record_launch(f);
    CUresult rc = CUDA_SUCCESS;
    if (substitute_driver_launch_ex(ptsz, config, f, params, extra, &rc)) return rc;
    return real(config, f, params, extra);
}

KVG_EXPORT CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** params, void** extra) {
    return launch_kernel_ex_impl(0, config, f, params, extra);
}

KVG_EXPORT CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** params, void** extra) {
    return launch_kernel_ex_impl(1, config, f, params, extra);
}

// CUDA runtime (libcudart) hooks.

namespace {

std::map<const void*, std::string> g_stub_names;   // host stub address -> kernel name
std::map<const void*, const void*> g_fatbin_of;     // registration handle -> fatbin image
std::map<const void*, const void*> g_kernel_stubs;  // cudaKernel_t (a CUkernel) -> host stub address
std::map<const void*, const void*> g_stub_module;   // host stub address -> registration handle (module key)

// Enforcement: instrumented twins.

struct Replacement {
    std::string name;
    CUlibrary lib = nullptr;
    CUkernel kernel = nullptr;
    CUdeviceptr desc = 0;      // the twin's __kvg_desc, 0 when the module has none
    size_t desc_size = 0;
    uint64_t desc_version = 0; // version of g_desc last copied into the module
    int max_dyn_smem = -1;     // the original's dynamic-shared-memory limit, mirrored on growth
    bool failed = false;
    uint64_t launches = 0;
};

std::map<const void*, Replacement*> g_repl_by_stub;  // host stub or CUfunction -> twin, or nullptr once looked up without one
std::vector<Replacement*> g_replacements;
std::atomic<int> g_mode{0};
std::vector<uint64_t> g_desc;                        // host copy of the descriptor (u64 slots)
std::atomic<uint64_t> g_desc_version{0};
uint64_t* g_desc_pinned = nullptr;                   // page-locked mirror of g_desc (64 slots)
size_t g_desc_pinned_n = 0;                          // slots currently valid in the mirror
std::atomic<uint64_t> g_captured_launches{0};        // twin launches issued while capturing

// The page-locked mirror is allocated on first use (a CUDA context exists at the first launch).
// A captured memcpy node copies from this fixed address, so a later kvg_set_descriptor is seen by
// every replay. Falls back to the pageable copy when the allocation fails.
uint64_t* desc_pinned() {
    static bool tried = false;
    if (!tried) {
        tried = true;
        auto alloc = real_fn<CUresult (*)(void**, size_t, unsigned int)>("cuMemHostAlloc");
        void* p = nullptr;
        if (alloc && alloc(&p, 64 * sizeof(uint64_t), CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS) {
            std::memset(p, 0, 64 * sizeof(uint64_t));
            g_desc_pinned = static_cast<uint64_t*>(p);
        } else {
            std::fprintf(stderr, "[kvg] cuMemHostAlloc failed; descriptors use pageable memory (not capturable)\n");
        }
    }
    return g_desc_pinned;
}

bool read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

// The twin of a kernel identified by its host stub (runtime launches) or its CUfunction (driver
// launches), loaded from the cache on first use (a current context exists then, which the
// library and global lookups need). Negative results are cached too.
Replacement* replacement_for_key(const void* key, bool driver) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_repl_by_stub.find(key);
        if (it != g_repl_by_stub.end()) return it->second;
    }
    const char* dir = std::getenv("KVGUARD_INTERPOSER_CACHE");
    Replacement* r = nullptr;
    std::string name;
    uint64_t ihash = 0;
    bool known = false;
    if (dir) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto& names = driver ? g_func_names : g_stub_names;
        auto& mods = driver ? g_func_module : g_stub_module;
        auto sn = names.find(key);
        auto sm = mods.find(key);
        if (sn != names.end() && sm != mods.end()) {
            auto mod = g_modules.find(sm->second);
            if (mod != g_modules.end()) { name = sn->second; ihash = mod->second.hash; known = true; }
        }
    }
    if (known) {
        char base[1024];
        std::snprintf(base, sizeof(base), "%s/%016llx/%016llx", dir, static_cast<unsigned long long>(ihash),
                      static_cast<unsigned long long>(fnv1a(name.data(), name.size())));
        std::vector<char> code;
        std::string kind;
        if (read_file(std::string(base) + ".cubin", code)) kind = "cubin";
        else if (read_file(std::string(base) + ".ptx", code)) { kind = "ptx"; code.push_back('\0'); }
        if (!kind.empty()) {
            r = new Replacement;
            r->name = name;
            auto load = real_fn<CUresult (*)(CUlibrary*, const void*, CUjit_option*, void**, unsigned, CUlibraryOption*, void**, unsigned)>("cuLibraryLoadData");
            auto getk = real_fn<CUresult (*)(CUkernel*, CUlibrary, const char*)>("cuLibraryGetKernel");
            auto getg = real_fn<CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*)>("cuLibraryGetGlobal");
            CUresult rc = (load && getk && getg) ? load(&r->lib, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0) : CUDA_ERROR_NOT_FOUND;
            if (rc == CUDA_SUCCESS) rc = getk(&r->kernel, r->lib, name.c_str());
            if (rc == CUDA_SUCCESS && getg(&r->desc, &r->desc_size, r->lib, "__kvg_desc") != CUDA_SUCCESS) { r->desc = 0; r->desc_size = 0; }
            r->failed = rc != CUDA_SUCCESS;
            if (g_log || r->failed)
                std::fprintf(stderr, "[kvg] twin %s %s%s (%s, %zu bytes, desc %zu bytes)\n", r->failed ? "FAILED to load" : "loaded", base, kind == "ptx" ? ".ptx" : ".cubin", name.c_str(), code.size(), r->desc_size);
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_repl_by_stub[key] = r;
    if (r) g_replacements.push_back(r);
    return r;
}


struct kvg_dim3 { unsigned int x, y, z; };

void refresh_descriptor(Replacement* r, CUstream s, bool capturing);
void count_twin_launch(Replacement* r, CUresult res);

// Capture state of a stream (a CUDA graph being recorded).
bool stream_capturing(CUstream s) {
    auto iscap = real_fn<CUresult (*)(CUstream, CUstreamCaptureStatus*)>("cuStreamIsCapturing");
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    return iscap && iscap(s, &st) == CUDA_SUCCESS && st != CU_STREAM_CAPTURE_STATUS_NONE;
}

// The twin for a key, respecting capture: inside a capture only an already loaded twin is used
// (no library load or global lookup there).
Replacement* twin_for(const void* key, bool driver, bool capturing) {
    if (capturing) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_repl_by_stub.find(key);
        if (it == g_repl_by_stub.end() || !it->second || it->second->failed) {
            if (it == g_repl_by_stub.end()) g_calls["twin_not_loaded_at_capture"] += 1;
            return nullptr;
        }
    }
    Replacement* r = replacement_for_key(key, driver);
    return (r && !r->failed) ? r : nullptr;
}

// The host code may have raised the original's dynamic-shared-memory limit (cudaFuncSetAttribute,
// cuFuncSetAttribute); the twin needs the same limit. `original_limit` reads it from the original.
void mirror_smem_limit(Replacement* r, size_t smem, int (*original_limit)(const void*, CUdevice), const void* original) {
    if (static_cast<int>(smem) <= r->max_dyn_smem) return;
    CUdevice dev = 0;
    auto getdev = real_fn<CUresult (*)(CUdevice*)>("cuCtxGetDevice");
    auto seta = real_fn<CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice)>("cuKernelSetAttribute");
    if (!getdev || !seta || getdev(&dev) != CUDA_SUCCESS) return;
    int v = original_limit(original, dev);
    if (v < 0) return;
    if (v < static_cast<int>(smem)) v = static_cast<int>(smem);
    seta(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, v, r->kernel, dev);
    r->max_dyn_smem = v;
}

int kernel_smem_limit(const void* kernel, CUdevice dev) {
    auto geta = real_fn<CUresult (*)(int*, CUfunction_attribute, CUkernel, CUdevice)>("cuKernelGetAttribute");
    int v = 0;
    return (geta && geta(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<CUkernel>(const_cast<void*>(kernel)), dev) == CUDA_SUCCESS) ? v : -1;
}

int function_smem_limit(const void* f, CUdevice) {
    auto geta = real_fn<CUresult (*)(int*, CUfunction_attribute, CUfunction)>("cuFuncGetAttribute");
    int v = 0;
    return (geta && geta(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<CUfunction>(const_cast<void*>(f))) == CUDA_SUCCESS) ? v : -1;
}

// Redirect a runtime launch to the kernel's twin. Returns true when it did; `rc` is then the
// launch result as a cudaError_t (0 on success).
bool substitute_launch(int ptsz, void* kernel, kvg_dim3 grid, kvg_dim3 block, void** args, size_t smem, void* stream, int* rc) {
    if (g_mode.load(std::memory_order_relaxed) == 0) return false;
    const void* stub = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto k = g_kernel_stubs.find(kernel);
        if (k == g_kernel_stubs.end()) return false;
        stub = k->second;
    }
    CUstream s = stream ? static_cast<CUstream>(stream) : (ptsz ? CU_STREAM_PER_THREAD : nullptr);
    bool capturing = stream_capturing(s);
    Replacement* r = twin_for(stub, false, capturing);
    if (!r) return false;
    mirror_smem_limit(r, smem, kernel_smem_limit, kernel);
    refresh_descriptor(r, s, capturing);
    auto launch = real_for<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**)>(
        "cuLaunchKernel", ptsz, ptsz ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    if (!launch) return false;
    CUresult res = launch(reinterpret_cast<CUfunction>(r->kernel), grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          static_cast<unsigned>(smem), s, args, nullptr);
    count_twin_launch(r, res);
    *rc = res == CUDA_SUCCESS ? 0 : 719;  // cudaErrorLaunchFailure
    return true;
}

// Redirect a driver launch (cuLaunchKernel from Triton, cuda-python, ...) to the twin of the
// CUfunction, if the function was obtained through cuModuleGetFunction / cuLibraryGetKernel and
// its module's image has a twin in the cache.
bool substitute_driver_launch(int ptsz, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                              unsigned smem, CUstream stream, void** params, void** extra, CUresult* rc) {
    if (g_mode.load(std::memory_order_relaxed) == 0 || extra != nullptr) return false;
    CUstream s = stream ? stream : (ptsz ? CU_STREAM_PER_THREAD : nullptr);
    bool capturing = stream_capturing(s);
    Replacement* r = twin_for(f, true, capturing);
    if (!r) return false;
    mirror_smem_limit(r, smem, function_smem_limit, f);
    refresh_descriptor(r, s, capturing);
    auto launch = real_for<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**)>(
        "cuLaunchKernel", ptsz, ptsz ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    if (!launch) return false;
    *rc = launch(reinterpret_cast<CUfunction>(r->kernel), gx, gy, gz, bx, by, bz, smem, s, params, nullptr);
    count_twin_launch(r, *rc);
    return true;
}

// cuLaunchKernelEx (Triton's launcher): the same substitution with the launch configuration
// passed through, attributes included, and the twin's kernel in place of the function.
bool substitute_driver_launch_ex(int ptsz, const CUlaunchConfig* config, CUfunction f, void** params, void** extra, CUresult* rc) {
    if (g_mode.load(std::memory_order_relaxed) == 0 || extra != nullptr || !config) return false;
    CUstream s = config->hStream ? config->hStream : (ptsz ? CU_STREAM_PER_THREAD : nullptr);
    bool capturing = stream_capturing(s);
    Replacement* r = twin_for(f, true, capturing);
    if (!r) return false;
    mirror_smem_limit(r, config->sharedMemBytes, function_smem_limit, f);
    refresh_descriptor(r, s, capturing);
    auto launch = real_for<CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**)>("cuLaunchKernelEx", ptsz, ptsz ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx");
    if (!launch) return false;
    CUlaunchConfig cfg = *config;
    cfg.hStream = s;
    *rc = launch(&cfg, reinterpret_cast<CUfunction>(r->kernel), params, nullptr);
    count_twin_launch(r, *rc);
    return true;
}

void count_twin_launch(Replacement* r, CUresult res) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        r->launches += 1;
        g_calls["substituted"] += 1;
        g_launches[r->name + " [kvg twin]"] += 1;
    }
    if (res != CUDA_SUCCESS) std::fprintf(stderr, "[kvg] twin launch of %s failed: CUresult %d\n", r->name.c_str(), static_cast<int>(res));
}

// Copy the descriptor into the twin's symbol when the host copy changed (eager) or always while
// capturing, so the graph gets a memcpy node from the page-locked mirror.
void refresh_descriptor(Replacement* r, CUstream s, bool capturing) {
    uint64_t ver = g_desc_version.load();
    if (r->desc && (capturing || ver != r->desc_version)) {
        // Eager: refresh when the host copy changed. Capturing: always, so the graph gets a
        // memcpy node from the page-locked mirror that re-reads the descriptor at every replay.
        auto cp = real_fn<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream)>("cuMemcpyHtoDAsync_v2");
        uint64_t* pinned = desc_pinned();
        size_t n = 0;
        uint64_t buf[64];
        {
            std::lock_guard<std::mutex> lk(g_mu);
            n = std::min({g_desc.size(), r->desc_size / 8, sizeof(buf) / 8});
            if (pinned) {
                std::copy_n(g_desc.begin(), n, pinned);
                g_desc_pinned_n = n;
            } else {
                std::copy_n(g_desc.begin(), n, buf);
            }
        }
        if (n && cp) {
            // The copy covers the twin's whole descriptor symbol (never more: it is a .const
            // array of N_DESC slots), from the mirror or, without one, from the stack buffer.
            size_t bytes = std::min(r->desc_size, size_t(64 * 8));
            if (pinned) cp(r->desc, pinned, bytes, s);
            else if (!capturing) cp(r->desc, buf, std::min(bytes, n * 8), s);
        }
        r->desc_version = ver;
        if (capturing) g_captured_launches.fetch_add(1);
    }
}

std::string substitutions_json_locked() {
    std::string o = "{\"mode\": " + std::to_string(g_mode.load()) + ", \"descriptor_version\": " + std::to_string(g_desc_version.load()) +
                    ", \"captured_launches\": " + std::to_string(g_captured_launches.load()) + ", \"pinned_descriptor\": " + (g_desc_pinned ? "true" : "false") + ", \"twins\": [";
    bool first = true;
    for (Replacement* r : g_replacements) {
        o += std::string(first ? "" : ", ") + "{\"name\": \"" + json_escape(r->name) + "\", \"launches\": " + std::to_string(r->launches) +
             ", \"failed\": " + (r->failed ? "true" : "false") + ", \"descriptor_bytes\": " + std::to_string(r->desc_size) + "}";
        first = false;
    }
    return o + "]}";
}

std::string substitutions_json() {
    std::lock_guard<std::mutex> lk(g_mu);
    return substitutions_json_locked();
}

template <typename F>
F next_fn(const char* name) {
    static std::map<std::string, void*> cache;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = cache.find(name);
    if (it != cache.end()) return reinterpret_cast<F>(it->second);
    void* p = real_dlsym() ? real_dlsym()(RTLD_NEXT, name) : nullptr;
    cache[name] = p;
    return reinterpret_cast<F>(p);
}

struct FatBinWrapper {  // crt/fatbinary.h __fatBinC_Wrapper_t
    int magic;
    int version;
    const unsigned long long* data;
    void* filename_or_fatbins;
};

void record_runtime_launch(const void* func, const char* via) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_stub_names.find(func);
    g_launches[it != g_stub_names.end() ? it->second : "<unregistered stub>"] += 1;
    g_calls[via] += 1;
}

void record_handle_launch(const void* kernel, const char* via) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto k = g_kernel_stubs.find(kernel);
    auto it = k != g_kernel_stubs.end() ? g_stub_names.find(k->second) : g_stub_names.end();
    g_launches[it != g_stub_names.end() ? it->second : "<unresolved cudaKernel_t>"] += 1;
    g_calls[via] += 1;
}

}  // namespace

KVG_EXPORT void** __cudaRegisterFatBinary(void* fatCubin) {
    auto real = next_fn<void** (*)(void*)>("__cudaRegisterFatBinary");
    void** handle = real ? real(fatCubin) : nullptr;
    const FatBinWrapper* w = static_cast<const FatBinWrapper*>(fatCubin);
    const void* image = (w && w->magic == 0x466243b1) ? static_cast<const void*>(w->data) : nullptr;
    if (handle && image) {
        record_module(handle, image, "__cudaRegisterFatBinary", {});
        std::lock_guard<std::mutex> lk(g_mu);
        g_fatbin_of[handle] = image;
    }
    return handle;
}

KVG_EXPORT void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun, const char* deviceName, int thread_limit,
                                       void* tid, void* bid, void* bDim, void* gDim, int* wSize) {
    auto real = next_fn<void (*)(void**, const char*, char*, const char*, int, void*, void*, void*, void*, int*)>("__cudaRegisterFunction");
    if (real) real(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    if (deviceName) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_stub_names[hostFun] = deviceName;
        g_stub_module[hostFun] = fatCubinHandle;
        g_calls["__cudaRegisterFunction"] += 1;
        auto it = g_modules.find(fatCubinHandle);
        if (it != g_modules.end()) {
            it->second.kernel_count += 1;
            static const bool full = std::getenv("KVGUARD_INTERPOSER_FULL") != nullptr;
            if (full || kv_pattern(deviceName)) it->second.kernels.emplace_back(deviceName);
        }
    }
}

KVG_EXPORT void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    auto real = next_fn<void (*)(void**)>("__cudaRegisterFatBinaryEnd");
    if (real) real(fatCubinHandle);
    ModuleInfo m;
    const void* image = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_modules.find(fatCubinHandle);
        auto f = g_fatbin_of.find(fatCubinHandle);
        if (it == g_modules.end() || f == g_fatbin_of.end()) return;
        m = it->second;
        image = f->second;
    }
    dump_image(m, image);
}

// The dynamic-shared-memory limit of a host stub's kernel, through the runtime's own query.
struct kvg_func_attributes {  // cudaFuncAttributes (driver_types.h); only the leading fields matter
    size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
    int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA, maxDynamicSharedSizeBytes, preferredShmemCarveout;
    int reserved[16];
};

int stub_smem_limit(const void* stub, CUdevice) {
    auto geta = next_fn<int (*)(kvg_func_attributes*, const void*)>("cudaFuncGetAttributes");
    kvg_func_attributes a{};
    return (geta && geta(&a, stub) == 0) ? a.maxDynamicSharedSizeBytes : -1;
}

// Libraries built with older toolkits, and hand-written launches (FlashInfer's dispatchers), call
// the public cudaLaunchKernel with the host stub itself: the stub is already the twin lookup key.
bool substitute_stub_launch(const void* func, kvg_dim3 grid, kvg_dim3 block, void** args, size_t smem, void* stream, int* rc) {
    if (g_mode.load(std::memory_order_relaxed) == 0) return false;
    CUstream s = static_cast<CUstream>(stream);  // the runtime's legacy default stream is the driver's null stream
    bool capturing = stream_capturing(s);
    Replacement* r = twin_for(func, false, capturing);
    if (!r) return false;
    mirror_smem_limit(r, smem, stub_smem_limit, func);
    refresh_descriptor(r, s, capturing);
    auto launch = real_for<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**)>(
        "cuLaunchKernel", 0, "cuLaunchKernel");
    if (!launch) return false;
    CUresult res = launch(reinterpret_cast<CUfunction>(r->kernel), grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          static_cast<unsigned>(smem), s, args, nullptr);
    count_twin_launch(r, res);
    *rc = res == CUDA_SUCCESS ? 0 : 719;  // cudaErrorLaunchFailure
    return true;
}

struct kvg_launch_config {  // cudaLaunchConfig_t (driver_types.h)
    kvg_dim3 gridDim, blockDim;
    size_t dynamicSmemBytes;
    void* stream;
    void* attrs;
    unsigned int numAttrs;
};

KVG_EXPORT int cudaLaunchKernel(const void* func, kvg_dim3 gridDim, kvg_dim3 blockDim, void** args, size_t sharedMem, void* stream) {
    auto real = next_fn<int (*)(const void*, kvg_dim3, kvg_dim3, void**, size_t, void*)>("cudaLaunchKernel");
    record_runtime_launch(func, "cudaLaunchKernel");
    int rc = 0;
    if (substitute_stub_launch(func, gridDim, blockDim, args, sharedMem, stream, &rc)) return rc;
    return real ? real(func, gridDim, blockDim, args, sharedMem, stream) : 999;
}

KVG_EXPORT int cudaLaunchKernelExC(const void* config, const void* func, void** args) {
    auto real = next_fn<int (*)(const void*, const void*, void**)>("cudaLaunchKernelExC");
    record_runtime_launch(func, "cudaLaunchKernelExC");
    const kvg_launch_config* c = static_cast<const kvg_launch_config*>(config);
    int rc = 0;
    if (c && c->numAttrs == 0 && substitute_stub_launch(func, c->gridDim, c->blockDim, args, c->dynamicSmemBytes, c->stream, &rc)) return rc;
    if (c && c->numAttrs != 0 && g_mode.load(std::memory_order_relaxed) != 0) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_calls["launch_with_attributes_not_substituted"] += 1;
    }
    return real ? real(config, func, args) : 999;
}

// CUDA 13 host stubs: resolve the stub to a kernel handle once, then launch by handle.
KVG_EXPORT int __cudaGetKernel(void** kernel, const void* hostFun) {
    auto real = next_fn<int (*)(void**, const void*)>("__cudaGetKernel");
    int rc = real ? real(kernel, hostFun) : 999;
    if (rc == 0 && kernel && *kernel) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_kernel_stubs[*kernel] = hostFun;
        g_calls["__cudaGetKernel"] += 1;
    }
    return rc;
}

KVG_EXPORT int __cudaLaunchKernel(void* kernel, kvg_dim3 gridDim, kvg_dim3 blockDim, void** args, size_t sharedMem, void* stream) {
    auto real = next_fn<int (*)(void*, kvg_dim3, kvg_dim3, void**, size_t, void*)>("__cudaLaunchKernel");
    record_handle_launch(kernel, "__cudaLaunchKernel");
    int rc = 0;
    if (substitute_launch(0, kernel, gridDim, blockDim, args, sharedMem, stream, &rc)) return rc;
    return real ? real(kernel, gridDim, blockDim, args, sharedMem, stream) : 999;
}

KVG_EXPORT int __cudaLaunchKernel_ptsz(void* kernel, kvg_dim3 gridDim, kvg_dim3 blockDim, void** args, size_t sharedMem, void* stream) {
    auto real = next_fn<int (*)(void*, kvg_dim3, kvg_dim3, void**, size_t, void*)>("__cudaLaunchKernel_ptsz");
    record_handle_launch(kernel, "__cudaLaunchKernel_ptsz");
    int rc = 0;
    if (substitute_launch(1, kernel, gridDim, blockDim, args, sharedMem, stream, &rc)) return rc;
    return real ? real(kernel, gridDim, blockDim, args, sharedMem, stream) : 999;
}

// Control API (ctypes from kvguard.gpu.interposer).

KVG_EXPORT int kvg_set_mode(int mode) { return g_mode.exchange(mode); }
KVG_EXPORT int kvg_get_mode() { return g_mode.load(); }

KVG_EXPORT int kvg_set_descriptor(const uint64_t* vals, size_t n) {
    if (!vals || n == 0 || n > 64) return -1;
    std::lock_guard<std::mutex> lk(g_mu);
    g_desc.assign(vals, vals + n);
    g_desc_version.fetch_add(1);
    if (g_desc_pinned) {
        // Graph replays copy from the mirror; make the new descriptor visible to them at once.
        std::copy_n(g_desc.begin(), n, g_desc_pinned);
        for (size_t i = n; i < 64; ++i) g_desc_pinned[i] = 0;
        g_desc_pinned_n = n;
    }
    return 0;
}

// JSON of the twins loaded so far and their launch counts; returns the length needed.
KVG_EXPORT size_t kvg_substitutions(char* buf, size_t n) {
    std::string j = substitutions_json();
    if (buf && n) { size_t m = std::min(n - 1, j.size()); std::memcpy(buf, j.data(), m); buf[m] = 0; }
    return j.size();
}

// Entry-point interposition.

namespace {

struct Hook { const char* symbol; void* wrapper; void* wrapper_ptsz; };

const Hook kHooks[] = {
    {"cuModuleLoadData", reinterpret_cast<void*>(&cuModuleLoadData), nullptr},
    {"cuModuleLoadDataEx", reinterpret_cast<void*>(&cuModuleLoadDataEx), nullptr},
    {"cuModuleLoadFatBinary", reinterpret_cast<void*>(&cuModuleLoadFatBinary), nullptr},
    {"cuModuleLoad", reinterpret_cast<void*>(&cuModuleLoad), nullptr},
    {"cuLibraryLoadData", reinterpret_cast<void*>(&cuLibraryLoadData), nullptr},
    {"cuLibraryLoadFromFile", reinterpret_cast<void*>(&cuLibraryLoadFromFile), nullptr},
    {"cuModuleGetFunction", reinterpret_cast<void*>(&cuModuleGetFunction), nullptr},
    {"cuLibraryGetKernel", reinterpret_cast<void*>(&cuLibraryGetKernel), nullptr},
    {"cuLaunchKernel", reinterpret_cast<void*>(&cuLaunchKernel), reinterpret_cast<void*>(&cuLaunchKernel_ptsz)},
    {"cuLaunchKernelEx", reinterpret_cast<void*>(&cuLaunchKernelEx), reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz)},
};

const Hook* find_hook(const char* symbol) {
    if (!symbol) return nullptr;
    for (const Hook& h : kHooks)
        if (std::strcmp(h.symbol, symbol) == 0) return &h;
    // "_ptsz" / "_v2" suffixes name the same entry points
    std::string s(symbol);
    for (const char* suf : {"_ptsz", "_v2", "_v3"}) {
        size_t n = std::strlen(suf);
        if (s.size() > n && s.compare(s.size() - n, n, suf) == 0) {
            std::string base = s.substr(0, s.size() - n);
            for (const Hook& h : kHooks)
                if (base == h.symbol) return &h;
        }
    }
    return nullptr;
}

}  // namespace

// cuGetProcAddress (CUDA 12+: _v2 with a status out-parameter; older signature kept for 11.x).
KVG_EXPORT CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* status) {
    auto real = real_fn<CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*)>("cuGetProcAddress_v2");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult r = real(symbol, pfn, cudaVersion, flags, status);
    if (r != CUDA_SUCCESS || !pfn || !*pfn) return r;
    const Hook* h = find_hook(symbol);
    if (h) {
        int ptsz = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) ? 1 : 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_real[ptsz][h->symbol] = *pfn;
            g_calls["cuGetProcAddress"] += 1;
        }
        *pfn = (ptsz && h->wrapper_ptsz) ? h->wrapper_ptsz : h->wrapper;
        if (g_log) std::fprintf(stderr, "[kvg] cuGetProcAddress(%s, flags=%llu) -> wrapper\n", symbol, static_cast<unsigned long long>(flags));
    }
    return r;
}

KVG_EXPORT CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
    auto real = real_fn<CUresult (*)(const char*, void**, int, cuuint64_t)>("cuGetProcAddress");
    if (!real) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult r = real(symbol, pfn, cudaVersion, flags);
    if (r != CUDA_SUCCESS || !pfn || !*pfn) return r;
    const Hook* h = find_hook(symbol);
    if (h) {
        int ptsz = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) ? 1 : 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_real[ptsz][h->symbol] = *pfn;
            g_calls["cuGetProcAddress"] += 1;
        }
        *pfn = (ptsz && h->wrapper_ptsz) ? h->wrapper_ptsz : h->wrapper;
    }
    return r;
}

// dlsym and dlvsym: route lookups of cuGetProcAddress and of the intercepted driver symbols to
// this library. The CUDA runtime fetches cuGetProcAddress_v2 with dlvsym on the libcuda handle;
// Triton and cuda-python use dlsym.
namespace {
void* route(const char* symbol, void* target, const char* via) {
    if (!symbol || !target) return target;
    if (g_log && symbol[0] == 'c' && symbol[1] == 'u') std::fprintf(stderr, "[kvg] %s(%s)\n", via, symbol);
    if (std::strcmp(symbol, "cuGetProcAddress_v2") == 0) return reinterpret_cast<void*>(&cuGetProcAddress_v2);
    if (std::strcmp(symbol, "cuGetProcAddress") == 0) return reinterpret_cast<void*>(&cuGetProcAddress);
    if (symbol[0] == 'c' && symbol[1] == 'u') {
        const Hook* h = find_hook(symbol);
        if (h) {
            std::lock_guard<std::mutex> lk(g_mu);
            g_calls[via] += 1;
            int ptsz = std::strstr(symbol, "_ptsz") != nullptr;
            g_real[ptsz][h->symbol] = target;
            return (ptsz && h->wrapper_ptsz) ? h->wrapper_ptsz : h->wrapper;
        }
    }
    return target;
}
}  // namespace

KVG_EXPORT void* dlsym(void* handle, const char* symbol) {
    dlsym_t real = real_dlsym();
    if (!real) return nullptr;
    return route(symbol, real(handle, symbol), "dlsym");
}

KVG_EXPORT void* dlvsym(void* handle, const char* symbol, const char* version) {
    dlvsym_t real = real_dlvsym();
    if (!real) return nullptr;
    return route(symbol, real(handle, symbol, version), "dlvsym");
}

__attribute__((constructor)) static void kvg_init() {
    g_log = std::getenv("KVGUARD_INTERPOSER_LOG") != nullptr;
    if (const char* m = std::getenv("KVGUARD_INTERPOSER_MODE")) g_mode.store(std::atoi(m));
    if (g_log) std::fprintf(stderr, "[kvg] interposer loaded\n");
}

// Exit.

namespace {

// A static object defined after every global this file declares: its destructor runs before
// theirs (reverse order of construction), so the summary is written while the maps are alive. A
// constructor-attribute function registering std::atexit would run before the globals are
// constructed and its handler after they are destroyed (a use-after-free in the first version);
// defining this object before the enforcement globals crashed at exit the same way.
struct AtExit {
    ~AtExit() { write_summary(); }
} g_atexit;

}  // namespace
