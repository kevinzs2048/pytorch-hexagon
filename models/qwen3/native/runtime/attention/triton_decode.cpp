// SPDX-License-Identifier: BSD-3-Clause
// ABI, validation and scheduling only. q1 attention arithmetic is in Triton.
#include "worker-pool.h"
#include "attention_memref.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>

extern "C" void *memalign(size_t, size_t);
using qwen_attention::Memref;
using qwen_attention::FlatMemref;
using qwen_attention::KVMemref;
using qwen_attention::FlatKVMemref;
extern "C" void attention_decode(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int64_t, void *, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int);
#ifdef QWEN_TRITON_PREFILL
extern "C" int qwen_triton_prefill(const Memref *, const KVMemref *,
                                    const KVMemref *, const Memref *, Memref *);
#endif
#ifdef QWEN_KV_FP16
#ifndef QWEN_TRITON_PREFILL
#error "FP16 KV storage requires the Triton prefill kernels"
#endif
#define QWEN_ATTENTION_ENTRY _mlir_ciface_qwen_ordered_fp16kv_attention_v1
#else
#define QWEN_ATTENTION_ENTRY _mlir_ciface_qwen_ordered_fp32_attention_v1
extern "C" int qwen_strict_prefill(const Memref *, const Memref *,
                                    const Memref *, const Memref *, Memref *);
#endif

namespace {
// CMake reads this bound from the hash-checked kernel receipt. The larger
// fallback retains compatibility with callers that link older decode objects.
#ifndef QWEN_DECODE_SCRATCH_ELEMENTS
#define QWEN_DECODE_SCRATCH_ELEMENTS 1064960
#endif
constexpr size_t ScratchElements = QWEN_DECODE_SCRATCH_ELEMENTS;
static_assert(ScratchElements >= 16384 && ScratchElements <= 1064960);
worker_pool_context_t pool;
float *scratch;
std::atomic_flag busy = ATOMIC_FLAG_INIT;

__attribute__((destructor)) void cleanup() {
  if (pool)
    worker_pool_release(&pool);
  free(scratch);
  pool = nullptr;
  scratch = nullptr;
}

using qwen_attention::valid;
using qwen_attention::flatten;

struct Job {
  FlatMemref q;
  FlatKVMemref k, v;
  FlatMemref out, scratch;
  int length, qhead, qtoken, khead, ktoken, vhead, vtoken;
};

void worker(unsigned, unsigned id, void *opaque) {
  auto &j = *static_cast<Job *>(opaque);
  attention_decode(1, &j.q, 1, &j.k, 1, &j.v, 1, &j.out, 1, &j.scratch,
                   1, j.length, j.qhead, j.qtoken, j.khead, j.ktoken,
                   j.vhead, j.vtoken, 1, 4, 1, 0, id, 0);
}
} // namespace

extern "C" int QWEN_ATTENTION_ENTRY(
    const Memref *q, const KVMemref *k, const KVMemref *v, const Memref *mask,
    Memref *out) {
  if (!q || !k || q->sizes[2] < 1 || q->sizes[2] > 32)
    return 2;
#ifdef QWEN_TRITON_HMX_DECODE
  // Effective length, never cache capacity. Prefill validates the q1 ABI/mask
  // and shares its worker pool and HMX owner with the long decode path.
  if (q->sizes[2] == 1 && k->sizes[2] >= QWEN_HMX_DECODE_MIN_KV)
    return qwen_triton_prefill(q, k, v, mask, out);
#endif
  if (q->sizes[2] > 1) {
#ifdef QWEN_TRITON_PREFILL
    return qwen_triton_prefill(q, k, v, mask, out);
#else
    return qwen_strict_prefill(q, k, v, mask, out);
#endif
  }
  if (k->sizes[2] < 1 || k->sizes[2] > 1024)
    return 2;
  int length = int(k->sizes[2]);
  if (!valid(q, 1, 16, 1, 128) || !valid(k, 1, 8, length, 128) ||
      !valid(v, 1, 8, length, 128) || !valid(mask, 1, 1, 1, length) ||
      !valid(out, 1, 1, 16, 128) || out->strides[1] != 2048 ||
      out->strides[2] != 128)
    return 2;
  if (busy.test_and_set())
    return 3;
  struct Unlock {
    ~Unlock() { busy.clear(); }
  } unlock;
  // All occupied keys are visible for the final q1 position. Do not silently
  // ignore an arbitrary additive or padding mask supplied by a different graph.
  for (int i = 0; i < length; ++i)
    if (mask->aligned[mask->offset + i] != 0.0f)
      return 2;
  if (!pool) {
    scratch = static_cast<float *>(memalign(128, ScratchElements * sizeof(float)));
    if (!scratch || worker_pool_init_with_stack_size(&pool, 4, 32768)) {
      cleanup();
      return 4;
    }
  }
  Job job{flatten(*q), flatten(*k), flatten(*v), flatten(*out),
          {scratch, scratch, 0, ScratchElements, 1}, length,
          int(q->strides[1]), int(q->strides[2]), int(k->strides[1]),
          int(k->strides[2]), int(v->strides[1]), int(v->strides[2])};
  if (worker_pool_run_func(pool, worker, &job, 4))
    return 6;
  return 0;
}
