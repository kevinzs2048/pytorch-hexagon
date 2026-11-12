// SPDX-License-Identifier: BSD-3-Clause
// Experimental FP16 prefill: validation, scheduling and storage only.
#include "attention_memref.h"
#include "temporary_buffer_pool.h"
#include "worker-pool.h"
#include <atomic>
#include <cstdlib>
#include <limits>
#include <qurt.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#ifndef QWEN_PREFILL_POOL_BYTES
#define QWEN_PREFILL_POOL_BYTES 0
#endif
#ifndef QWEN_PREFILL_VTCM_BYTES
#define QWEN_PREFILL_VTCM_BYTES (1024 * 1024)
#endif
static_assert(QWEN_PREFILL_POOL_BYTES <= 1024 * 1024,
              "Prefill pool limit is per logical worker, at most 1 MiB");

using qwen_attention::Memref;
using qwen_attention::FlatMemref;
using qwen_attention::KVMemref;
using qwen_attention::FlatKVMemref;
extern "C" void attention_forward(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int);
#ifdef QWEN_TRITON_HMX_DECODE
extern "C" void attention_forward_q1_hmx_g2(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int);
constexpr int MinimumQueryLength = 1;
#else
constexpr int MinimumQueryLength = 2;
#endif
extern "C" int hexagon_runtime_configure_vtcm_pool_dsp(unsigned);
extern "C" int hexagon_runtime_configure_hmx_performance_dsp(unsigned);

#ifdef QWEN_ATTENTION_KV_DMA
// Multi-token attention stages its K/V heads through VTCM with the user DMA
// engine and overlaps the second head's transfer with the first head's kernel
// (the llama.cpp HMX flash-attention pattern; descriptor layout and
// dmstart/dmlink/dmpoll usage follow llama.cpp hex-dma.h, MIT). The staged
// planes keep the source element strides, so every kernel reads the same
// values from VTCM: arithmetic and results are unchanged. Sources are read
// through L2, which holds rows the graph has just appended.
#include "HAP_compute_res.h"
#ifndef QWEN_ATTENTION_KV_DMA_MIN_KV
#define QWEN_ATTENTION_KV_DMA_MIN_KV 256
#endif
static_assert(sizeof(qwen_attention::KVElement) == 2, "VTCM staging is sized for FP16 KV");
namespace kv_dma {
constexpr unsigned PlaneBytes = 1024 * 1024 * 2; // 1024 tokens x 8 heads x 128 FP16
// K and V planes plus a guard: kernel tile prefetch/reads may touch up to one
// 128-row tile past the final row; in DDR that stays mapped, in VTCM it must
// stay inside the lease (KV length 1024 faulted without it).
constexpr unsigned GuardBytes = 512 * 1024;
constexpr unsigned LeaseBytes = 2 * PlaneBytes + GuardBytes;
struct alignas(64) Descriptor { // v73 2D descriptor (16-bit fields)
  void *next;
  uint32_t reserved0 : 24, desc_size : 2, dst_comp : 1, src_comp : 1,
      dst_bypass : 1, src_bypass : 1, order : 1, done : 1;
  const void *src;
  void *dst;
  uint32_t desc_type : 8, reserved1 : 24;
  uint32_t row_size : 16, nrows : 16;
  uint32_t src_stride : 16, dst_stride : 16;
  uint32_t src_offset : 16, dst_offset : 16;
};
static_assert(sizeof(Descriptor) == 64);
inline void start(Descriptor *d) {
  asm volatile(" release(%0):at" : : "r"(d));
  asm volatile(" dmstart(%0)" : : "r"(d));
}
inline void link(Descriptor *tail, Descriptor *d) {
  asm volatile(" release(%0):at" : : "r"(d));
  asm volatile(" dmlink(%0, %1)" : : "r"(tail), "r"(d));
}
inline void wait(const Descriptor &d) {
  while (!static_cast<const volatile Descriptor &>(d).done) {
    unsigned status;
    asm volatile(" %0 = dmpoll" : "=r"(status) : : "memory");
  }
}
// rows x 256 bytes (one head) from a strided source to the staged plane.
inline void fill(Descriptor &d, const void *src, void *dst, unsigned rows,
                 unsigned srcStride, unsigned dstStride) {
  d = {};
  d.desc_size = 1; // 2D
  d.dst_bypass = 1;
  d.src = src;
  d.dst = dst;
  d.row_size = 256;
  d.nrows = rows;
  d.src_stride = srcStride;
  d.dst_stride = dstStride;
}
Descriptor descriptors[4][4];
unsigned resource;
uint8_t *base;
// Cache the descriptor; hold the physical VTCM lease only during one call.
bool acquire() {
  if (!resource) {
    compute_res_attr_t attr;
    void *pointer = nullptr;
    unsigned size = LeaseBytes;
    if (HAP_compute_res_attr_init(&attr) ||
        HAP_compute_res_attr_set_serialize(&attr, 0) ||
        HAP_compute_res_attr_set_cache_mode(&attr, 1) ||
        HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size))
      return false;
    unsigned id = HAP_compute_res_acquire(&attr, 1000);
    if (!id) return false;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &pointer, &size) ||
        !pointer || size < LeaseBytes) {
      HAP_compute_res_release(id);
      return false;
    }
    resource = id;
    base = static_cast<uint8_t *>(pointer);
  }
  // As the INT8 runtime: the cached descriptor is leased per operation.
  return !HAP_compute_res_acquire_cached(resource, 1000);
}
void release() {
  if (resource) HAP_compute_res_release_cached(resource);
}
__attribute__((destructor)) void cleanup() {
  if (resource) HAP_compute_res_release(resource);
  resource = 0;
}
} // namespace kv_dma
#endif

namespace {
using Pool = pytorch_hexagon::TemporaryBufferPool;
struct WorkerStorage { Pool pool{QWEN_PREFILL_POOL_BYTES, 256}; };
WorkerStorage storage[4];
worker_pool_context_t workers;
int allocationKey = -1;
std::atomic_flag busy = ATOMIC_FLAG_INIT;

__attribute__((destructor)) void cleanup() {
  if (workers)
    worker_pool_release(&workers);
  if (allocationKey >= 0)
    qurt_tls_delete_key(allocationKey);
  workers = nullptr;
  allocationKey = -1;
}

Pool *currentPool() {
  return allocationKey < 0 ? nullptr :
      static_cast<Pool *>(qurt_tls_get_specific(allocationKey));
}

struct Scope {
  Pool &pool;
  explicit Scope(unsigned id) : pool(storage[id].pool) {
    if (!pool.begin() || qurt_tls_set_specific(allocationKey, &pool))
      std::abort();
  }
  ~Scope() {
    // Outputs still live at return are detached, preserving malloc ownership.
    pool.end();
    if (qurt_tls_set_specific(allocationKey, nullptr))
      std::abort();
  }
};
struct Job {
  FlatMemref q;
  FlatKVMemref k, v;
  FlatMemref out;
  int m, n, qh, qt, kh, kt, vh, vt;
  bool staged = false; // K/V heads staged through VTCM (QWEN_ATTENTION_KV_DMA)
};
void worker(unsigned, unsigned id, void *opaque) {
  if (id >= 4)
    std::abort();
  Scope scope(id); // Job IDs are exclusive; physical threads may change.
  auto &j = *static_cast<Job *>(opaque);
  auto kernel = attention_forward;
#ifdef QWEN_TRITON_HMX_DECODE
  if (j.m == 1) {
    attention_forward_q1_hmx_g2(1, &j.q, 1, &j.k, 1, &j.v, 1, &j.out,
        j.m, j.n, j.qh, j.qt, j.kh, j.kt, j.vh, j.vt,
        1, 4, 1, 0, id, 0);
    return;
  }
#endif
#ifdef QWEN_ATTENTION_KV_DMA
  if (j.staged) {
    // Both heads' K and V on one chain; head id+4 transfers during head id.
    using namespace kv_dma;
    auto *k = static_cast<uint8_t *>(static_cast<void *>(base));
    auto *v = k + PlaneBytes;
    Descriptor *d = descriptors[id];
    for (unsigned i = 0; i < 2; ++i) {
      unsigned head = id + 4 * i;
      fill(d[2 * i], j.k.aligned + head * j.kh, k + head * 256, j.n, j.kt * 2, 2048);
      fill(d[2 * i + 1], j.v.aligned + head * j.vh, v + head * 256, j.n, j.vt * 2, 2048);
    }
    start(&d[0]);
    for (unsigned i = 1; i < 4; ++i) link(&d[i - 1], &d[i]);
    using Element = qwen_attention::KVElement;
    FlatKVMemref sk{reinterpret_cast<Element *>(k), reinterpret_cast<Element *>(k), 0, int64_t(j.n) * 1024, 1};
    FlatKVMemref sv{reinterpret_cast<Element *>(v), reinterpret_cast<Element *>(v), 0, int64_t(j.n) * 1024, 1};
    for (unsigned i = 0; i < 2; ++i) {
      wait(d[2 * i]);
      wait(d[2 * i + 1]);
      kernel(1, &j.q, 1, &sk, 1, &sv, 1, &j.out, j.m, j.n, j.qh, j.qt,
             128, 1024, 128, 1024, 1, 8, 1, 0, id + 4 * i, 0);
    }
    return;
  }
#endif
  for (unsigned head = id; head < 8; head += 4)
    kernel(1, &j.q, 1, &j.k, 1, &j.v, 1, &j.out,
                      j.m, j.n, j.qh, j.qt, j.kh, j.kt, j.vh, j.vt,
                      1, 8, 1, 0, head, 0);
}
} // namespace

// Only generated kernel calls use these hooks. Runtime bitcode keeps libc.
// Outlined HMX output allocations run outside the submitting worker's TLS and
// use malloc; a later worker release safely delegates foreign pointers to free.
extern "C" void *hexagon_kernel_malloc(uint64_t bytes) {
  if (bytes > std::numeric_limits<size_t>::max())
    return nullptr;
  auto *pool = currentPool();
  return pool ? pool->allocate(size_t(bytes)) : std::malloc(size_t(bytes));
}
extern "C" void hexagon_kernel_free(void *pointer) {
  if (auto *pool = currentPool())
    pool->release(pointer);
  else
    std::free(pointer);
}

// The Triton kernels construct the causal mask themselves, so the model's mask
// operand may only be that mask: row r sees the first n - m + r + 1 keys.
// Visible entries must be +0 or -0; hidden entries -inf or the most negative
// finite FP32. Returns nonzero for any other bit pattern. On the DSP, whole
// 32-lane groups of visible entries are tested on HVX; the remaining visible
// entries and the at most m - 1 hidden entries of a row are tested one by one.
// Host builds (unit tests) apply the same rule to every element.
static unsigned causal_mask_invalid(const float *rows, int64_t row_stride, int m, int n) {
  auto element = [](const float *value, bool visible) -> unsigned {
    uint32_t bits;
    __builtin_memcpy(&bits, value, sizeof(bits));
    return visible ? bits & 0x7fffffffU  // Both signed zeros are accepted.
                   : unsigned((bits != 0xff800000U) & (bits != 0xff7fffffU));
  };
  unsigned invalid = 0;
#if defined(__hexagon__)
  const HVX_Vector magnitude = Q6_V_vsplat_R(0x7fffffff);
  HVX_Vector wrong = Q6_V_vzero();
#endif
  for (int row = 0; row < m; ++row) {
    const float *values = rows + row * row_stride;
    const int visible = n - m + row + 1;
    int col = 0;
#if defined(__hexagon__)
    for (; col + 32 <= visible; col += 32)
      wrong = Q6_V_vor_VV(wrong, Q6_V_vand_VV(*reinterpret_cast<const HVX_UVector *>(values + col), magnitude));
#endif
    for (; col < visible; ++col)
      invalid |= element(values + col, true);
    for (; col < n; ++col)
      invalid |= element(values + col, false);
  }
#if defined(__hexagon__)
  alignas(128) uint32_t lanes[32];
  *reinterpret_cast<HVX_Vector *>(lanes) = wrong;
  for (uint32_t lane : lanes)
    invalid |= lane;
#endif
  return invalid;
}

extern "C" int qwen_triton_prefill(const Memref *q, const KVMemref *k,
                                  const KVMemref *v, const Memref *mask,
                                  Memref *out) {
  using namespace qwen_attention;
  if (!q || !k || q->sizes[2] < MinimumQueryLength || q->sizes[2] > 32 ||
      k->sizes[2] < q->sizes[2] || k->sizes[2] > 1024)
    return 2;
  int m = int(q->sizes[2]), n = int(k->sizes[2]);
  if (!valid(q, 1, 16, m, 128) || !valid(k, 1, 8, n, 128) ||
      !valid(v, 1, 8, n, 128) || !valid(mask, 1, 1, m, n) ||
      !valid(out, 1, m, 16, 128) || out->strides[1] != 2048 ||
      out->strides[2] != 128)
    return 2;
  if (busy.test_and_set())
    return 3;
  struct Unlock { ~Unlock() { busy.clear(); } } unlock;
  // Generated HMX kernels stage operands directly into VTCM with DSP/HVX
  // loads (enableConvertToHexagonmem=False). The caller owns external RPC/DMA
  // synchronization; the operator requires no raw-buffer flush/invalidate.
  // The Triton kernel constructs its causal mask. Reject padding/additive masks
  // rather than silently ignoring model inputs with different semantics.
  if (causal_mask_invalid(mask->aligned + mask->offset, mask->strides[2], m, n))
    return 2;
  if (hexagon_runtime_configure_hmx_performance_dsp(1) ||
      hexagon_runtime_configure_vtcm_pool_dsp(QWEN_PREFILL_VTCM_BYTES))
    return 8;
  if (!workers) {
    if (qurt_tls_create_key(&allocationKey, nullptr) ||
        worker_pool_init_with_stack_size(&workers, 4, 32768)) {
      cleanup();
      return 4;
    }
  }
  Job job{flatten(*q), flatten(*k), flatten(*v), flatten(*out), m, n,
          int(q->strides[1]), int(q->strides[2]), int(k->strides[1]),
          int(k->strides[2]), int(v->strides[1]), int(v->strides[2])};
#ifdef QWEN_ATTENTION_KV_DMA
  // 16-bit descriptor strides: element strides up to 32767 FP16 values.
  job.staged = m > 1 && n >= QWEN_ATTENTION_KV_DMA_MIN_KV &&
               job.kt <= 32767 && job.vt <= 32767 && kv_dma::acquire();
  struct Lease {
    bool held;
    ~Lease() { if (held) kv_dma::release(); }
  } lease{job.staged};
#endif
  if (worker_pool_run_func(workers, worker, &job, 4))
    return 6;
  return 0;
}
