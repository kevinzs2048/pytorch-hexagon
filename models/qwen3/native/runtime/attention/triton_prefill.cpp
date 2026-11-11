// SPDX-License-Identifier: BSD-3-Clause
// Experimental FP16 prefill: validation, scheduling and storage only.
#include "attention_memref.h"
#include "temporary_buffer_pool.h"
#include "worker-pool.h"
#include <atomic>
#include <cstdlib>
#include <limits>
#include <qurt.h>

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
  unsigned invalid = 0;
  for (int row = 0; row < m; ++row) {
    const float *values = mask->aligned + mask->offset + row * mask->strides[2];
    const int visible = n - m + row + 1;
    for (int col = 0; col < visible; ++col) {
      uint32_t bits;
      __builtin_memcpy(&bits, values + col, sizeof(bits));
      invalid |= bits & 0x7fffffffU; // Both signed zeros are accepted.
    }
    for (int col = visible; col < n; ++col) {
      uint32_t bits;
      __builtin_memcpy(&bits, values + col, sizeof(bits));
      invalid |= (bits != 0xff800000U) & (bits != 0xff7fffffU);
    }
  }
  if (invalid)
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
  if (worker_pool_run_func(workers, worker, &job, 4))
    return 6;
  return 0;
}
