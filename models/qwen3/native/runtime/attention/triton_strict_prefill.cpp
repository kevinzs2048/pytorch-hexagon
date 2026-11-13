// SPDX-License-Identifier: BSD-3-Clause
// ABI, validation and scheduling only. prefill attention arithmetic is in
// Triton.
#include "attention_memref.h"
#include "worker-pool.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>

extern "C" void *memalign(size_t, size_t);
using qwen_attention::FlatMemref;
using qwen_attention::Memref;
extern "C" void qwen_prefill_qk(int64_t, void *, int64_t, void *, int64_t,
                                void *, int64_t, void *, int64_t, void *, int,
                                int, int, int, int, int, int, int, int, int,
                                int, int, int, int, int);
extern "C" void qwen_prefill_softmax(int64_t, void *, int64_t, void *, int64_t,
                                     void *, int64_t, void *, int64_t, void *,
                                     int, int, int, int, int, int, int, int,
                                     int, int, int, int, int, int, int);
extern "C" void qwen_prefill_pv(int64_t, void *, int64_t, void *, int64_t,
                                void *, int64_t, void *, int64_t, void *, int,
                                int, int, int, int, int, int, int, int, int,
                                int, int, int, int, int);
extern "C" void qwen_prefill_pack(int64_t, void *, int64_t, void *, int64_t,
                                  void *, int64_t, void *, int64_t, void *,
                                  int64_t, void *, int, int, int, int, int, int,
                                  int, int, int, int, int, int, int, int, int);
extern "C" int qwen_strict_prefill(const Memref *, const Memref *,
                                   const Memref *, const Memref *, Memref *);

namespace {
constexpr unsigned MaximumKV = 256;
constexpr unsigned ScoreElements = 16 * 32 * MaximumKV;
constexpr size_t ScratchElements = ScoreElements + 32;
constexpr unsigned PackedElements = 8 * MaximumKV * 128;
worker_pool_context_t pool;
float *scratch, *packed_k, *packed_v;
std::atomic_flag busy = ATOMIC_FLAG_INIT;

__attribute__((destructor)) void cleanup() {
  if (pool)
    worker_pool_release(&pool);
  free(scratch);
  free(packed_k);
  free(packed_v);
  packed_k = packed_v = nullptr;
  pool = nullptr;
  scratch = nullptr;
}

using qwen_attention::flatten;
using qwen_attention::valid;

struct Job {
  FlatMemref q, k, v, out, scratch, pk, pv;
  int query, length, qhead, qtoken, khead, ktoken, vhead, vtoken;
};

void worker(unsigned, unsigned id, void *opaque) {
  auto &j = *static_cast<Job *>(opaque);
  // Each worker owns the same four query heads throughout all three stages.
  // Keep intermediate scores local to that worker instead of dispatching and
  // synchronizing a separate worker group for each numerical stage.
  decltype(&qwen_prefill_qk) stages[] = {
      qwen_prefill_qk, qwen_prefill_softmax, qwen_prefill_pv};
  for (auto invoke : stages)
    invoke(1, &j.q, 1, &j.pk, 1, &j.pv, 1, &j.out, 1, &j.scratch, j.query,
           j.length, j.qhead, j.qtoken, j.khead, j.ktoken, j.vhead, j.vtoken,
           4, 1, 4, 1, 0, id, 0);
}
void pack_worker(unsigned, unsigned id, void *opaque) {
  auto &j = *static_cast<Job *>(opaque);
  qwen_prefill_pack(1, &j.scratch, 1, &j.k, 1, &j.v, 1, &j.pk, 1, &j.pv, 1,
                    &j.q, j.query, j.length, j.qhead, j.qtoken, j.khead,
                    j.ktoken, j.vhead, j.vtoken, 2, 1, 4, 1, 0, id, 0);
}
} // namespace

extern "C" int qwen_triton_prefill(const Memref *q, const Memref *k,
                                   const Memref *v, const Memref *mask,
                                   Memref *out) {
  if (!q || !k || q->sizes[2] < 1 || q->sizes[2] > 32)
    return 2;
  if (q->sizes[2] < 2 || k->sizes[2] < q->sizes[2] || k->sizes[2] > 1024)
    return 2;
  int query = int(q->sizes[2]);
  int length = int(k->sizes[2]);
  if (!valid(q, 1, 16, query, 128) || !valid(k, 1, 8, length, 128) ||
      !valid(v, 1, 8, length, 128) || !valid(mask, 1, 1, query, length) ||
      !valid(out, 1, query, 16, 128) || out->strides[1] != 2048 ||
      out->strides[2] != 128)
    return 2;
  // Packing and worker dispatch pay off once the query/key tile is large
  // enough. Smaller tails retain the reference path.
  if (query < 16 || query * length < 1024 || length > int(MaximumKV))
    return qwen_strict_prefill(q, k, v, mask, out);
  if (busy.test_and_set())
    return 3;
  struct Unlock {
    ~Unlock() { busy.clear(); }
  } unlock;
  // Reject arbitrary additive/padding masks; the kernels implement causality.
  for (int row = 0; row < query; ++row)
    for (int col = 0; col < length; ++col) {
      float v = mask->aligned[mask->offset + row * mask->strides[2] + col];
      if (col <= length - query + row
              ? v != 0.0f
              : !(v == -__builtin_inff() || v == -3.4028234663852886e38f))
        return 2;
    }
  if (!pool) {
    scratch =
        static_cast<float *>(memalign(128, ScratchElements * sizeof(float)));
    packed_k =
        static_cast<float *>(memalign(128, PackedElements * sizeof(float)));
    packed_v =
        static_cast<float *>(memalign(128, PackedElements * sizeof(float)));
    if (!scratch || !packed_k || !packed_v ||
        worker_pool_init_with_stack_size(&pool, 4, 32768)) {
      cleanup();
      return 4;
    }
  }
  Job job{flatten(*q),
          flatten(*k),
          flatten(*v),
          flatten(*out),
          {scratch, scratch, 0, ScratchElements, 1},
          {packed_k, packed_k, 0, PackedElements, 1},
          {packed_v, packed_v, 0, PackedElements, 1},
          query,
          length,
          int(q->strides[1]),
          int(q->strides[2]),
          int(k->strides[1]),
          int(k->strides[2]),
          int(v->strides[1]),
          int(v->strides[2])};
  if (worker_pool_run_func(pool, pack_worker, &job, 4))
    return 6;
  // The Triton packing stage checks the numeric bound before any output is
  // written. Unsafe/nonfinite inputs keep the exact reference behavior.
  for (int head = 0; head < 16; ++head)
    if (scratch[ScoreElements + head] != 1.0f)
      return qwen_strict_prefill(q, k, v, mask, out);
  if (worker_pool_run_func(pool, worker, &job, 4))
    return 6;
  return 0;
}
