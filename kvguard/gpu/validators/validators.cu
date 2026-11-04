// Dispatch-time validators: one kernel per mapping representation, run on the launch stream
// before the kernels that consume the table. Every entry a row consumes is checked against a
// device mirror of the shadow (kept in step with the core's ledger):
//
//   state[b]            bit 0 allocated (not free), bit 1 published, bit 2 shared (readable by
//                       any request), bit 3 reserved (outside the request-managed namespace);
//                       bits 8.. the owner id
//   filled[b*W .. +W)   bitmask of slot offsets that hold initialized entries, W = ceil(bs / 32)
//                       words per block (offset o is bit o % 32 of word o / 32)
//   pending[b*W .. +W)  bitmask of slot offsets a published block may still be written at
//
// Rows carry an owner id (KVG_PADDING_ROW for a CUDA-graph padding row, which may touch reserved
// blocks only) and the end of the positions they read (the dense table also their start, for a
// sliding window's rows). Violations increment a counter; the first one is recorded as (kind,
// row or token, entry index or slot offset, block or slot).

#include <stdint.h>

#define KVG_ALLOC 1u
#define KVG_PUBLISHED 2u
#define KVG_SHARED 4u
#define KVG_RESERVED 8u
#define KVG_OWNER_SHIFT 8
#define KVG_PADDING_ROW 0xFFFFFFFFu

enum {
  K_OUT_OF_POOL = 1,     // entry outside the pool
  K_RESERVED_LIVE = 2,   // a reserved block at a live position
  K_FREE = 3,            // a free block
  K_FOREIGN = 4,         // a private block of another request
  K_UNDER_FILLED = 5,    // a read of slots never written
  K_PADDING_MANAGED = 6, // a padding row names a request-managed block
  K_WRITE_PUBLISHED = 7, // a write into a published block outside its pending slots
  K_DOUBLE_WRITE = 8,    // a write into an initialized slot
  K_PAGE_LENGTH = 9,     // CSR: the pages and last_page_len do not cover the row's positions
  K_MISMATCH = 11,       // the device entry is not the slot the mapping gives the position
};
#define KVG_HOLE 0xFFFFFFFFu  // a position of the mapping with no page (a retired window)

__device__ __forceinline__ void report(uint32_t* counter, uint32_t* record, uint32_t kind,
                                       uint32_t row, uint32_t index, uint32_t entry) {
  if (atomicAdd(counter, 1u) == 0u) {
    record[0] = kind;
    record[1] = row;
    record[2] = index;
    record[3] = entry;
  }
}

// Whether every offset of [lo, hi) of block blk has its bit set in `mask` (W words per block).
__device__ __forceinline__ bool bits_all_set(const uint32_t* mask, uint32_t words, int64_t blk,
                                             uint32_t lo, uint32_t hi) {
  const uint32_t* m = mask + (size_t)blk * words;
  for (uint32_t w = lo >> 5; w <= (hi - 1u) >> 5; ++w) {
    uint32_t a = w << 5, b = a + 32u;
    uint32_t from = lo > a ? lo - a : 0u;
    uint32_t to = hi < b ? hi - a : 32u;
    uint32_t need = to - from >= 32u ? 0xFFFFFFFFu : (((1u << (to - from)) - 1u) << from);
    if ((m[w] & need) != need) return false;
  }
  return true;
}

__device__ __forceinline__ bool bit_set(const uint32_t* mask, uint32_t words, int64_t blk,
                                        uint32_t o) {
  return (mask[(size_t)blk * words + (o >> 5)] >> (o & 31u)) & 1u;
}

// One block entry of a row that reads the slot offsets [lo, hi) of it (hi <= lo: the entry is
// listed but no position of it is read). Returns the violation kind or 0.
__device__ __forceinline__ uint32_t check_block(int64_t blk, uint32_t owner, uint32_t lo,
                                                uint32_t hi, const uint32_t* state,
                                                const uint32_t* filled, uint32_t words,
                                                uint32_t num_blocks) {
  if (blk < 0 || blk >= (int64_t)num_blocks) return K_OUT_OF_POOL;
  uint32_t st = state[blk];
  if (owner == KVG_PADDING_ROW) return (st & KVG_RESERVED) ? 0u : K_PADDING_MANAGED;
  if (st & KVG_RESERVED) return K_RESERVED_LIVE;
  if (!(st & KVG_ALLOC)) return K_FREE;
  if (!(st & KVG_SHARED) && (st >> KVG_OWNER_SHIFT) != owner) return K_FOREIGN;
  if (hi > lo && !bits_all_set(filled, words, blk, lo, hi)) return K_UNDER_FILLED;
  return 0u;
}

// One block entry of a row that writes the slot offsets [lo, hi) of it, in a pool whose slots
// are rewritten in place (a recurrent-state pool): the block must be allocated and the row's
// own, or published with those offsets pending; initialized slots may be written again.
__device__ __forceinline__ uint32_t check_block_write(int64_t blk, uint32_t owner, uint32_t lo,
                                                      uint32_t hi, const uint32_t* state,
                                                      const uint32_t* pending, uint32_t words,
                                                      uint32_t num_blocks) {
  if (blk < 0 || blk >= (int64_t)num_blocks) return K_OUT_OF_POOL;
  uint32_t st = state[blk];
  if (owner == KVG_PADDING_ROW) return (st & KVG_RESERVED) ? 0u : K_PADDING_MANAGED;
  if (st & KVG_RESERVED) return K_RESERVED_LIVE;
  if (!(st & KVG_ALLOC)) return K_FREE;
  if (st & KVG_PUBLISHED)
    return (hi > lo && !bits_all_set(pending, words, blk, lo, hi)) ? K_WRITE_PUBLISHED : 0u;
  if ((st >> KVG_OWNER_SHIFT) != owner) return K_FOREIGN;
  return 0u;
}

// Dense block table [rows, width] (int32): row r reads positions [row_start[r], row_end[r]) and
// consumes the entries from row_start[r] / bs up to ceil(row_end[r] / bs) (a sliding-window row's
// entries below the window are retired ids the kernel does not read). A padding row with end == 0
// is checked over its whole width; a positive end checks only that consumed prefix. A row with
// row_write[r] set writes those positions instead (a rewritable pool's state columns): ownership
// is checked, fills are not.
extern "C" __global__ void kvg_validate_dense(const int32_t* table, uint32_t row_stride,
                                              uint32_t rows, uint32_t width,
                                              const uint32_t* row_owner, const uint32_t* row_end,
                                              const uint32_t* row_start,
                                              const uint32_t* row_write, uint32_t bs,
                                              uint32_t words, const uint32_t* state,
                                              const uint32_t* filled, const uint32_t* pending,
                                              uint32_t num_blocks, uint32_t* counter,
                                              uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  bool writes = owner != KVG_PADDING_ROW && row_write[r] != 0u;
  uint32_t start = owner == KVG_PADDING_ROW ? 0u : row_start[r];
  uint32_t j0 = owner == KVG_PADDING_ROW ? 0u : start / bs;
  uint32_t n = owner == KVG_PADDING_ROW && end == 0u
                   ? width
                   : min(width, (end + bs - 1u) / bs);
  for (uint32_t j = j0 + threadIdx.x; j < n; j += blockDim.x) {
    int64_t blk = table[(size_t)r * row_stride + j];
    uint32_t first = j * bs;
    uint32_t lo = start > first ? start - first : 0u;
    uint32_t hi = end < first + bs ? (end > first ? end - first : 0u) : bs;
    uint32_t k = writes ? check_block_write(blk, owner, lo, hi, state, pending, words, num_blocks)
                        : check_block(blk, owner, lo, hi, state, filled, words, num_blocks);
    if (k) report(counter, record, k, r, j, (uint32_t)blk);
  }
}

// TensorRT-LLM block-offsets table: row r holds `width` (K, V) int32 pairs, K at
// table[r * row_stride + j] and V at table[r * row_stride + kv_stride + j] (the batch table
// [max_num_sequences, 2, max_blocks_per_seq] of one pool). A pair names block K / divisor when
// K is a non-negative multiple of `divisor` (the pool's numLayers * kvFactor: setOffsets writes
// flat_index3(block, 0, field, numLayers, kvFactor)) and V == K + 1; a negative offset carries
// the secondary-pool flag (bit 31). Anything else is out of pool (kind 1, recorded with the raw
// K offset as the entry); a decoded block is checked as the dense table's entries are.
extern "C" __global__ void kvg_validate_dense_offsets(
    const int32_t* table, uint32_t row_stride, uint32_t kv_stride, uint32_t rows, uint32_t width,
    uint32_t divisor, const uint32_t* row_owner, const uint32_t* row_end, uint32_t bs,
    uint32_t words, const uint32_t* state, const uint32_t* filled, uint32_t num_blocks,
    uint32_t* counter, uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  uint32_t n = owner == KVG_PADDING_ROW && end == 0u
                   ? width
                   : min(width, (end + bs - 1u) / bs);
  const int32_t* krow = table + (size_t)r * row_stride;
  const int32_t* vrow = krow + kv_stride;
  for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
    int32_t k = krow[j], v = vrow[j];
    uint32_t first = j * bs;
    uint32_t need = first < end ? min(bs, end - first) : 0u;
    bool decodable = k >= 0 && v >= 0 && divisor != 0u && ((uint32_t)k % divisor) == 0u &&
                     (int64_t)v == (int64_t)k + 1;
    if (!decodable) {
      report(counter, record, K_OUT_OF_POOL, r, j, (uint32_t)k);
      continue;
    }
    int64_t blk = (int64_t)((uint32_t)k / divisor);
    uint32_t kind = check_block(blk, owner, 0u, need, state, filled, words, num_blocks);
    if (kind) report(counter, record, kind, r, j, (uint32_t)blk);
  }
}

// CSR page list (FlashInfer): pages of row r are indices[indptr[r]:indptr[r+1]], the last page
// holds last_page_len[r] positions; together they must cover exactly the row's read positions.
extern "C" __global__ void kvg_validate_csr(const int32_t* indptr, const int32_t* indices,
                                            const int32_t* last_page_len, uint32_t rows,
                                            const uint32_t* row_owner, const uint32_t* row_end,
                                            uint32_t bs, uint32_t words, const uint32_t* state,
                                            const uint32_t* filled, uint32_t num_blocks,
                                            uint32_t* counter, uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  int32_t lo = indptr[r], hi = indptr[r + 1];
  uint32_t npages = hi > lo ? (uint32_t)(hi - lo) : 0u;
  if (threadIdx.x == 0 && owner != KVG_PADDING_ROW && end > 0) {
    int32_t lpl = last_page_len[r];
    bool ok = npages > 0 && lpl >= 1 && (uint32_t)lpl <= bs &&
              (npages - 1u) * bs + (uint32_t)lpl == end;
    if (!ok) report(counter, record, K_PAGE_LENGTH, r, npages, (uint32_t)lpl);
  }
  for (uint32_t j = threadIdx.x; j < npages; j += blockDim.x) {
    int64_t blk = indices[lo + j];
    uint32_t first = j * bs;
    uint32_t need = first < end ? min(bs, end - first) : 0u;
    uint32_t k = check_block(blk, owner, 0u, need, state, filled, words, num_blocks);
    if (k) report(counter, record, k, r, j, (uint32_t)blk);
  }
}

// Per-token table (SGLang req_to_token, int32): table[r][p] is the slot of position p for
// p < row_end[r]; the slot's block must be readable by the row and the slot initialized.
extern "C" __global__ void kvg_validate_token_table(const int32_t* table, uint32_t row_stride,
                                                    uint32_t rows, const uint32_t* row_owner,
                                                    const uint32_t* row_end, uint32_t bs,
                                                    uint32_t words, const uint32_t* state,
                                                    const uint32_t* filled, uint32_t num_blocks,
                                                    uint32_t* counter, uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  int64_t num_slots = (int64_t)num_blocks * bs;
  for (uint32_t p = threadIdx.x; p < end; p += blockDim.x) {
    int64_t s = table[(size_t)r * row_stride + p];
    if (s < 0 || s >= num_slots) {
      report(counter, record, K_OUT_OF_POOL, r, p, (uint32_t)s);
      continue;
    }
    int64_t b = s / bs;
    uint32_t o = (uint32_t)(s - b * bs);
    uint32_t st = state[b], k = 0u;
    if (owner == KVG_PADDING_ROW) k = (st & KVG_RESERVED) ? 0u : K_PADDING_MANAGED;
    else if (st & KVG_RESERVED) k = K_RESERVED_LIVE;
    else if (!(st & KVG_ALLOC)) k = K_FREE;
    else if (!(st & KVG_SHARED) && (st >> KVG_OWNER_SHIFT) != owner) k = K_FOREIGN;
    else if (!bit_set(filled, words, b, o)) k = K_UNDER_FILLED;
    if (k) report(counter, record, k, r, p, (uint32_t)s);
  }
}

// Per-token page list at any page size (SGLang's FlashInfer kv_indices: one slot per position):
// row r's entries indices[indptr[r]:indptr[r+1]] are the slots of its positions
// [row_start[r], row_end[r]) in order, and the slot the mapping gives position p is
// pages[page_indptr[r] + p / bs] * bs + p % bs (the row's page table, uploaded; KVG_HOLE for a
// position without a page). Per entry: in the pool, equal to the mapping's slot (K_MISMATCH:
// the device list disagrees with the framework's bookkeeping, the host's mapping_mismatch),
// then the token-table permission checks on the slot's block and offset. The entry count must
// be the row's position count (K_PAGE_LENGTH). A padding row (KVG_PADDING_ROW) reads nothing
// and may name reserved blocks only.
extern "C" __global__ void kvg_validate_token_csr(const int32_t* indptr, const int32_t* indices,
                                                  uint32_t rows, const uint32_t* row_owner,
                                                  const uint32_t* row_start,
                                                  const uint32_t* row_end,
                                                  const uint32_t* page_indptr,
                                                  const uint32_t* pages, uint32_t bs,
                                                  uint32_t words, const uint32_t* state,
                                                  const uint32_t* filled, uint32_t num_blocks,
                                                  uint32_t* counter, uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], start = row_start[r], end = row_end[r];
  int32_t lo = indptr[r], hi = indptr[r + 1];
  uint32_t n = hi > lo ? (uint32_t)(hi - lo) : 0u;
  int64_t num_slots = (int64_t)num_blocks * bs;
  if (owner == KVG_PADDING_ROW) {
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
      int64_t s = indices[lo + j];
      if (s < 0 || s >= num_slots) {
        report(counter, record, K_OUT_OF_POOL, r, j, (uint32_t)s);
        continue;
      }
      if (!(state[s / bs] & KVG_RESERVED)) report(counter, record, K_PADDING_MANAGED, r, j, (uint32_t)s);
    }
    return;
  }
  uint32_t need = end > start ? end - start : 0u;
  if (threadIdx.x == 0 && n != need) report(counter, record, K_PAGE_LENGTH, r, n, need);
  uint32_t pbase = page_indptr[r], npages = page_indptr[r + 1] - pbase;
  for (uint32_t j = threadIdx.x; j < n && j < need; j += blockDim.x) {
    uint32_t p = start + j;
    int64_t s = indices[lo + j];
    if (s < 0 || s >= num_slots) {
      report(counter, record, K_OUT_OF_POOL, r, j, (uint32_t)s);
      continue;
    }
    uint32_t pi = p / bs, o = p - pi * bs;
    uint32_t page = pi < npages ? pages[pbase + pi] : KVG_HOLE;
    if (page == KVG_HOLE || s != (int64_t)page * bs + o) {
      report(counter, record, K_MISMATCH, r, j, (uint32_t)s);
      continue;
    }
    uint32_t st = state[page], k = 0u;
    if (st & KVG_RESERVED) k = K_RESERVED_LIVE;
    else if (!(st & KVG_ALLOC)) k = K_FREE;
    else if (!(st & KVG_SHARED) && (st >> KVG_OWNER_SHIFT) != owner) k = K_FOREIGN;
    else if (!bit_set(filled, words, page, o)) k = K_UNDER_FILLED;
    if (k) report(counter, record, k, r, j, (uint32_t)s);
  }
}

// Flat slot map (vLLM slot_mapping int64, SGLang out_cache_loc): token t writes slots[t]; -1
// writes nothing. The slot's block must be private to the token's request (or published with the
// slot pending), and the slot not yet initialized. With `expected` (null otherwise), token t's
// slot must be expected[t], the slot the mapping gives its position (K_MISMATCH); an expected
// value of KVG_HOLE leaves the token unchecked against the mapping.
extern "C" __global__ void kvg_validate_slot_map(const void* slots, uint32_t elem_bytes,
                                                 uint32_t n_tokens, const uint32_t* tok_owner,
                                                 const uint32_t* expected, uint32_t bs,
                                                 uint32_t words, const uint32_t* state,
                                                 const uint32_t* filled, const uint32_t* pending,
                                                 uint32_t num_blocks, uint32_t* counter,
                                                 uint32_t* record) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tokens) return;
  int64_t s = elem_bytes == 8 ? ((const int64_t*)slots)[t] : (int64_t)((const int32_t*)slots)[t];
  if (s == -1) return;
  uint32_t owner = tok_owner[t];
  int64_t num_slots = (int64_t)num_blocks * bs;
  if (s < 0 || s >= num_slots) {
    report(counter, record, K_OUT_OF_POOL, t, 0u, (uint32_t)s);
    return;
  }
  if (expected != 0 && owner != KVG_PADDING_ROW) {
    uint32_t e = expected[t];
    if (e != KVG_HOLE && s != (int64_t)e) {
      report(counter, record, K_MISMATCH, t, 0u, (uint32_t)s);
      return;
    }
  }
  int64_t b = s / bs;
  uint32_t o = (uint32_t)(s - b * bs);
  uint32_t st = state[b], k = 0u;
  if (owner == KVG_PADDING_ROW) k = (st & KVG_RESERVED) ? 0u : K_PADDING_MANAGED;
  else if (st & KVG_RESERVED) k = K_RESERVED_LIVE;
  else if (!(st & KVG_ALLOC)) k = K_FREE;
  else if (st & KVG_PUBLISHED) k = bit_set(pending, words, b, o) ? 0u : K_WRITE_PUBLISHED;
  else if ((st >> KVG_OWNER_SHIFT) != owner) k = K_FOREIGN;
  if (!k && owner != KVG_PADDING_ROW && bit_set(filled, words, b, o)) k = K_DOUBLE_WRITE;
  if (k) report(counter, record, k, t, o, (uint32_t)b);
}

// Mirror update: vals[(1 + 2W) * i ..] = state of block ids[i], its W filled words, then its W
// pending words.
extern "C" __global__ void kvg_mirror_update(uint32_t* state, uint32_t* filled, uint32_t* pending,
                                             const uint32_t* ids, const uint32_t* vals,
                                             uint32_t n, uint32_t words) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t b = ids[i];
  const uint32_t* v = vals + (size_t)i * (1u + 2u * words);
  state[b] = v[0];
  for (uint32_t w = 0; w < words; ++w) {
    filled[(size_t)b * words + w] = v[1 + w];
    pending[(size_t)b * words + w] = v[1 + words + w];
  }
}

// The tag tier: one 8-byte tag per slot, kept by the device itself from the
// write (the stamp beside the slot-map validator), the free, the publish and the external fill,
// and compared at every read with the owner and the logical position the row asks for. The tags
// need no shadow: ``tags[2*s]`` holds the flags and the 24-bit owner id, ``tags[2*s + 1]`` the
// logical position (KVG_ANY_POS when the writer did not know one; then no position is compared).
//
//   KVG_TAG_WRITTEN   the slot holds a value written since its block was last freed
//   KVG_TAG_SHARED    the slot's block is published: any owner may read it
//   KVG_TAG_FREED     the slot's block was freed after the write: the owner and position stay for
//                     the diagnosis, the next write clears the flag
//   KVG_TAG_ALLOC     the slot's block is allocated (the allocation event set it, the free
//                     cleared it): a write into a slot without it is a write into free memory
//
// Owner id 0 is "unknown" (an external engine acting for no request, an adapter that does not
// know the owner): a tag held by owner 0 is readable and rewritable by anyone, and a reader or
// writer with owner 0 is checked against no ownership.
//
// ``reserved[b]`` (one byte per block) marks the geometry's reserved blocks (the null block); a
// padding row may name those only, a live row never. The kernels report as the validators do
// (counter, first record), with the record (kind, row or token, position, slot, tag word 0,
// tag word 1).

#define KVG_TAG_WRITTEN 1u
#define KVG_TAG_SHARED 2u
#define KVG_TAG_FREED 4u
#define KVG_TAG_ALLOC 8u
#define KVG_ANY_POS 0xFFFFFFFFu

enum {
  K_TAG_POSITION = 10, // the slot holds another logical position than the row asks for
};

enum {
  TAG_OP_FREE = 1,         // clear written and shared, set freed, on [lo, hi)
  TAG_OP_PUBLISH = 2,      // set shared on [lo, hi)
  TAG_OP_FILL = 3,         // stamp [lo, hi) with owner and pos0 + (o - lo); written slots reported
  TAG_OP_FILL_REWRITE = 4, // the same, written slots of the owner's own rewritten in place
  TAG_OP_ALLOC = 5,        // set allocated, clear freed, on [lo, hi)
  TAG_OP_INVALIDATE = 6,   // clear written on [lo, hi): a speculative rollback's rejected slots
};

__device__ __forceinline__ void report_tag(uint32_t* counter, uint32_t* record, uint32_t kind,
                                           uint32_t row, uint32_t pos, uint32_t slot,
                                           uint32_t w0, uint32_t w1) {
  if (atomicAdd(counter, 1u) == 0u) {
    record[0] = kind;
    record[1] = row;
    record[2] = pos;
    record[3] = slot;
    record[4] = w0;
    record[5] = w1;
  }
}

// A read of slot `s` (inside the pool, not reserved) by `owner` at logical position `pos`
// (KVG_ANY_POS: no position asked). Returns the violation kind or 0 and the tag words.
__device__ __forceinline__ uint32_t tag_read(const uint32_t* tags, int64_t s, uint32_t owner,
                                             uint32_t pos, uint32_t* w0, uint32_t* w1) {
  uint32_t a = tags[2 * (size_t)s], b = tags[2 * (size_t)s + 1];
  *w0 = a;
  *w1 = b;
  if ((a & KVG_TAG_FREED) || !(a & KVG_TAG_ALLOC)) return K_FREE;
  if (!(a & KVG_TAG_WRITTEN)) return K_UNDER_FILLED;
  uint32_t held = a >> KVG_OWNER_SHIFT;
  if (!(a & KVG_TAG_SHARED) && owner != 0u && held != 0u && held != owner) return K_FOREIGN;
  if (pos != KVG_ANY_POS && b != KVG_ANY_POS && b != pos) return K_TAG_POSITION;
  return 0u;
}

// A write of slot `s` by `owner` at `pos`: the slot's block must be allocated (else the write
// goes into free memory, K_FREE); a written slot may not be written again unless `rewrite` (a
// rewritable pool), and then only by its owner and not when it is published. Returns the
// violation kind or 0; stamps the slot either way, so the diagnosis sees the writer.
__device__ __forceinline__ uint32_t tag_write(uint32_t* tags, int64_t s, uint32_t owner,
                                              uint32_t pos, uint32_t rewrite, uint32_t* w0,
                                              uint32_t* w1) {
  uint32_t a = tags[2 * (size_t)s], b = tags[2 * (size_t)s + 1];
  *w0 = a;
  *w1 = b;
  uint32_t k = 0u;
  uint32_t shared = (a & KVG_TAG_FREED) ? 0u : (a & KVG_TAG_SHARED);
  if (!(a & KVG_TAG_ALLOC)) {
    k = K_FREE;
  } else if ((a & KVG_TAG_WRITTEN) && !(a & KVG_TAG_FREED)) {
    uint32_t held = a >> KVG_OWNER_SHIFT;
    if (!rewrite) k = K_DOUBLE_WRITE;
    else if (a & KVG_TAG_SHARED) k = K_WRITE_PUBLISHED;
    else if (held != 0u && owner != 0u && held != owner) k = K_FOREIGN;
  }
  tags[2 * (size_t)s] = (owner << KVG_OWNER_SHIFT) | KVG_TAG_WRITTEN | (a & KVG_TAG_ALLOC) | shared;
  tags[2 * (size_t)s + 1] = pos;
  return k;
}

// The stamp beside the slot-map validator: token t writes slots[t] (-1 writes nothing) on behalf
// of tok_owner[t] at logical position tok_pos[t]. A padding token may name a reserved slot only
// and stamps nothing; a live token may not name a reserved slot.
extern "C" __global__ void kvg_tag_stamp(const void* slots, uint32_t elem_bytes,
                                         uint32_t n_tokens, const uint32_t* tok_owner,
                                         const uint32_t* tok_pos, uint32_t bs,
                                         uint32_t num_blocks, const uint8_t* reserved_blocks,
                                         uint32_t rewrite, uint32_t* tags, uint32_t* counter,
                                         uint32_t* record) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tokens) return;
  int64_t s = elem_bytes == 8 ? ((const int64_t*)slots)[t] : (int64_t)((const int32_t*)slots)[t];
  if (s == -1) return;
  uint32_t owner = tok_owner[t], pos = tok_pos[t];
  int64_t num_slots = (int64_t)num_blocks * bs;
  if (s < 0 || s >= num_slots) {
    report_tag(counter, record, K_OUT_OF_POOL, t, pos, (uint32_t)s, 0u, 0u);
    return;
  }
  bool is_reserved = reserved_blocks[s / bs] != 0u;
  if (owner == KVG_PADDING_ROW) {
    if (!is_reserved) report_tag(counter, record, K_PADDING_MANAGED, t, pos, (uint32_t)s, 0u, 0u);
    return;
  }
  if (is_reserved) {
    report_tag(counter, record, K_RESERVED_LIVE, t, pos, (uint32_t)s, 0u, 0u);
    return;
  }
  uint32_t w0, w1;
  uint32_t k = tag_write(tags, s, owner, pos, rewrite, &w0, &w1);
  if (k) report_tag(counter, record, k, t, pos, (uint32_t)s, w0, w1);
}

// One consumed position `pos` of row `r` whose block entry `j` is `blk`: the block must be inside
// the pool and not reserved (a padding row: reserved only, no tag read), then the slot's tag
// must agree with the row.
__device__ __forceinline__ void tag_check_position(int64_t blk, uint32_t j, uint32_t o,
                                                   uint32_t r, uint32_t owner, uint32_t pos,
                                                   uint32_t bs, uint32_t num_blocks,
                                                   const uint8_t* reserved_blocks, const uint32_t* tags,
                                                   uint32_t* counter, uint32_t* record) {
  if (blk < 0 || blk >= (int64_t)num_blocks) {
    report_tag(counter, record, K_OUT_OF_POOL, r, pos, j, 0u, 0u);
    return;
  }
  bool reserved = reserved_blocks[blk] != 0u;
  if (owner == KVG_PADDING_ROW) {
    if (!reserved) report_tag(counter, record, K_PADDING_MANAGED, r, pos, (uint32_t)blk, 0u, 0u);
    return;
  }
  if (reserved) {
    report_tag(counter, record, K_RESERVED_LIVE, r, pos, (uint32_t)blk, 0u, 0u);
    return;
  }
  int64_t s = blk * bs + o;
  uint32_t w0, w1;
  uint32_t k = tag_read(tags, s, owner, pos, &w0, &w1);
  if (k) report_tag(counter, record, k, r, pos, (uint32_t)s, w0, w1);
}

// Dense block table: row r reads positions [row_start[r], row_end[r]) through entry pos / bs. A
// padding row with end == 0 is checked over its whole width (reserved entries only).
extern "C" __global__ void kvg_tag_check_dense(const int32_t* table, uint32_t row_stride,
                                               uint32_t rows, uint32_t width,
                                               const uint32_t* row_owner, const uint32_t* row_end,
                                               const uint32_t* row_start, uint32_t bs,
                                               uint32_t num_blocks, const uint8_t* reserved_blocks,
                                               const uint32_t* tags, uint32_t* counter,
                                               uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  const int32_t* row = table + (size_t)r * row_stride;
  if (owner == KVG_PADDING_ROW && end == 0u) {
    for (uint32_t j = threadIdx.x; j < width; j += blockDim.x)
      tag_check_position(row[j], j, 0u, r, owner, j * bs, bs, num_blocks, reserved_blocks, tags,
                         counter, record);
    return;
  }
  uint32_t start = owner == KVG_PADDING_ROW ? 0u : row_start[r];
  uint32_t limit = min(end, width * bs);
  for (uint32_t i = start + threadIdx.x; i < limit; i += blockDim.x) {
    uint32_t j = i / bs, o = i - j * bs;
    tag_check_position(row[j], j, o, r, owner, i, bs, num_blocks, reserved_blocks, tags, counter,
                       record);
  }
}

// TensorRT-LLM block-offsets table, decoded as kvg_validate_dense_offsets decodes it.
extern "C" __global__ void kvg_tag_check_dense_offsets(
    const int32_t* table, uint32_t row_stride, uint32_t kv_stride, uint32_t rows, uint32_t width,
    uint32_t divisor, const uint32_t* row_owner, const uint32_t* row_end, uint32_t bs,
    uint32_t num_blocks, const uint8_t* reserved_blocks, const uint32_t* tags, uint32_t* counter,
    uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  const int32_t* krow = table + (size_t)r * row_stride;
  const int32_t* vrow = krow + kv_stride;
  uint32_t limit = owner == KVG_PADDING_ROW && end == 0u ? width * bs : min(end, width * bs);
  for (uint32_t i = threadIdx.x; i < limit; i += blockDim.x) {
    uint32_t j = i / bs, o = i - j * bs;
    int32_t k = krow[j], v = vrow[j];
    bool decodable = k >= 0 && v >= 0 && divisor != 0u && ((uint32_t)k % divisor) == 0u &&
                     (int64_t)v == (int64_t)k + 1;
    if (!decodable) {
      if (o == 0u) report_tag(counter, record, K_OUT_OF_POOL, r, i, (uint32_t)k, 0u, 0u);
      continue;
    }
    int64_t blk = (int64_t)((uint32_t)k / divisor);
    tag_check_position(blk, j, o, r, owner, i, bs, num_blocks, reserved_blocks, tags, counter,
                       record);
  }
}

// CSR page list: positions [row_start[r], row_end[r]) through entry (pos - row_start) /
// entry_slots of the row's list (a sliding-window wrapper's row starts at its window). An
// entry covers `entry_slots` positions: bs when it names a block, 1 when it names a slot
// (FlashInfer's lists at SGLang's page sizes above one are one entry per position).
extern "C" __global__ void kvg_tag_check_csr(const int32_t* indptr, const int32_t* indices,
                                             const int32_t* last_page_len, uint32_t rows,
                                             const uint32_t* row_owner, const uint32_t* row_end,
                                             const uint32_t* row_start,
                                             uint32_t entry_slots, uint32_t bs,
                                             uint32_t num_blocks, const uint8_t* reserved_blocks,
                                             const uint32_t* tags, uint32_t* counter,
                                             uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r];
  uint32_t start = owner == KVG_PADDING_ROW ? 0u : row_start[r];
  int32_t lo = indptr[r], hi = indptr[r + 1];
  uint32_t npages = hi > lo ? (uint32_t)(hi - lo) : 0u;
  uint32_t es = entry_slots;
  uint32_t span = end > start ? end - start : 0u;
  if (threadIdx.x == 0 && owner != KVG_PADDING_ROW && span > 0) {
    int32_t lpl = last_page_len[r];
    bool ok = npages > 0 && lpl >= 1 && (uint32_t)lpl <= es &&
              (npages - 1u) * es + (uint32_t)lpl == span;
    if (!ok) report_tag(counter, record, K_PAGE_LENGTH, r, end, npages, (uint32_t)lpl, 0u);
  }
  uint32_t limit = owner == KVG_PADDING_ROW && end == 0u ? npages * es : min(span, npages * es);
  for (uint32_t k = threadIdx.x; k < limit; k += blockDim.x) {
    uint32_t i = start + k;
    uint32_t j = k / es, o = k - j * es;
    int64_t entry = indices[lo + j];
    if (es == bs) {
      tag_check_position(entry, j, o, r, owner, i, bs, num_blocks, reserved_blocks, tags,
                         counter, record);
    } else {
      int64_t s = entry * (int64_t)es + o;
      if (s < 0 || s >= (int64_t)num_blocks * bs) {
        report_tag(counter, record, K_OUT_OF_POOL, r, i, (uint32_t)s, 0u, 0u);
        continue;
      }
      int64_t b = s / bs;
      tag_check_position(b, j, (uint32_t)(s - b * bs), r, owner, i, bs, num_blocks,
                         reserved_blocks, tags, counter, record);
    }
  }
}

// Per-token table: table[r][c] is the slot of logical position row_first[r] + c for
// row_first[r] + c < row_end[r] (a sliding-window wrapper's suffix row starts at its window).
extern "C" __global__ void kvg_tag_check_token_table(
    const int32_t* table, uint32_t row_stride, uint32_t rows, const uint32_t* row_owner,
    const uint32_t* row_end, const uint32_t* row_first, uint32_t bs, uint32_t num_blocks,
    const uint8_t* reserved_blocks, const uint32_t* tags, uint32_t* counter, uint32_t* record) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  uint32_t owner = row_owner[r], end = row_end[r], first = row_first[r];
  int64_t num_slots = (int64_t)num_blocks * bs;
  for (uint32_t p = first + threadIdx.x; p < end; p += blockDim.x) {
    int64_t s = table[(size_t)r * row_stride + (p - first)];
    if (s < 0 || s >= num_slots) {
      report_tag(counter, record, K_OUT_OF_POOL, r, p, (uint32_t)s, 0u, 0u);
      continue;
    }
    int64_t b = s / bs;
    tag_check_position(b, (uint32_t)b, (uint32_t)(s - b * bs), r, owner, p, bs, num_blocks,
                       reserved_blocks, tags, counter, record);
  }
}

// Ranges read by a non-kernel engine (an external copy's source): record i is (block, owner,
// pos0, lo, hi): slots [lo, hi) of the block, at positions pos0 + (o - lo) (KVG_ANY_POS: none).
extern "C" __global__ void kvg_tag_check_ranges(const uint32_t* recs, uint32_t n, uint32_t bs,
                                                uint32_t num_blocks, const uint8_t* reserved_blocks,
                                                const uint32_t* tags, uint32_t* counter,
                                                uint32_t* record) {
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  const uint32_t* rec = recs + (size_t)i * 5;
  int64_t blk = (int64_t)rec[0];
  uint32_t owner = rec[1], pos0 = rec[2], lo = rec[3], hi = min(rec[4], bs);
  for (uint32_t o = lo + threadIdx.x; o < hi; o += blockDim.x) {
    uint32_t pos = pos0 == KVG_ANY_POS ? KVG_ANY_POS : pos0 + (o - lo);
    tag_check_position(blk, (uint32_t)blk, o, i, owner, pos, bs, num_blocks, reserved_blocks, tags,
                       counter, record);
  }
}

// Tag maintenance: record i is (block, op, owner, pos0, lo, hi) over slots [lo, hi) of the block
// (see TAG_OP_*). A fill into a written slot is reported as the stamp reports it.
extern "C" __global__ void kvg_tag_update(const uint32_t* recs, uint32_t n, uint32_t bs,
                                          uint32_t num_blocks, uint32_t* tags,
                                          uint32_t* counter, uint32_t* record) {
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  const uint32_t* rec = recs + (size_t)i * 6;
  int64_t blk = (int64_t)rec[0];
  uint32_t op = rec[1], owner = rec[2], pos0 = rec[3], lo = rec[4], hi = min(rec[5], bs);
  if (blk < 0 || blk >= (int64_t)num_blocks) {
    if (threadIdx.x == 0) report_tag(counter, record, K_OUT_OF_POOL, i, pos0, (uint32_t)blk, 0u, 0u);
    return;
  }
  for (uint32_t o = lo + threadIdx.x; o < hi; o += blockDim.x) {
    size_t s = (size_t)blk * bs + o;
    uint32_t a = tags[2 * s];
    if (op == TAG_OP_FREE) {
      tags[2 * s] = (a & ~(KVG_TAG_WRITTEN | KVG_TAG_SHARED | KVG_TAG_ALLOC)) |
                    ((a & KVG_TAG_WRITTEN) ? KVG_TAG_FREED : 0u);
    } else if (op == TAG_OP_PUBLISH) {
      tags[2 * s] = a | KVG_TAG_SHARED;
    } else if (op == TAG_OP_ALLOC) {
      tags[2 * s] = (a & ~KVG_TAG_FREED) | KVG_TAG_ALLOC;
    } else if (op == TAG_OP_INVALIDATE) {
      tags[2 * s] = a & ~KVG_TAG_WRITTEN;
    } else {
      uint32_t pos = pos0 == KVG_ANY_POS ? KVG_ANY_POS : pos0 + (o - lo);
      uint32_t w0, w1;
      uint32_t k = tag_write(tags, (int64_t)s, owner, pos, op == TAG_OP_FILL_REWRITE ? 1u : 0u,
                             &w0, &w1);
      if (k) report_tag(counter, record, k, i, pos, (uint32_t)s, w0, w1);
    }
  }
}

// Tag maintenance by slot list (a free or a publish of pages named one slot each, as SGLang's
// page-size-1 allocator hands them out, or a request's KV indices at its insertion into the
// radix cache): slots[i] (int32 or int64 at `slots`, -1 skipped) gets the op over its own slot
// only. Reads the list from the device, so a framework tensor can be passed as it is.
extern "C" __global__ void kvg_tag_update_slots(const void* slots, uint32_t elem_bytes,
                                                uint32_t n, uint32_t op, uint32_t bs,
                                                uint32_t num_blocks, uint32_t* tags,
                                                uint32_t* counter, uint32_t* record) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t s = elem_bytes == 8 ? ((const int64_t*)slots)[i] : (int64_t)((const int32_t*)slots)[i];
  if (s == -1) return;
  int64_t num_slots = (int64_t)num_blocks * bs;
  if (s < 0 || s >= num_slots) {
    report_tag(counter, record, K_OUT_OF_POOL, i, KVG_ANY_POS, (uint32_t)s, 0u, 0u);
    return;
  }
  uint32_t a = tags[2 * (size_t)s];
  if (op == TAG_OP_FREE) {
    tags[2 * (size_t)s] = (a & ~(KVG_TAG_WRITTEN | KVG_TAG_SHARED | KVG_TAG_ALLOC)) |
                          ((a & KVG_TAG_WRITTEN) ? KVG_TAG_FREED : 0u);
  } else if (op == TAG_OP_PUBLISH) {
    tags[2 * (size_t)s] = a | KVG_TAG_SHARED;
  } else if (op == TAG_OP_ALLOC) {
    tags[2 * (size_t)s] = (a & ~KVG_TAG_FREED) | KVG_TAG_ALLOC;
  } else if (op == TAG_OP_INVALIDATE) {
    tags[2 * (size_t)s] = a & ~KVG_TAG_WRITTEN;
  }
}
