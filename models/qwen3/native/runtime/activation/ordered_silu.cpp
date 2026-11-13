// SPDX-License-Identifier: BSD-3-Clause
// Strict SiLU, distributed over independent elements; optional Triton kernel.
#pragma clang fp contract(off)
#include "worker-pool.h"
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <qurt_memory.h>
struct Memref {
  float *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" float __hexagon_silu_exp_compat(float);
static worker_pool_context_t pool;
static std::atomic_flag busy = ATOMIC_FLAG_INIT;
__attribute__((destructor)) static void cleanup() {
  if (pool)
    worker_pool_release(&pool);
  pool = nullptr;
}
struct Job {
  const float *in;
  float *out;
  int rows, istride;
#ifdef QWEN_TRITON_SILU_UP
  const int32_t *up = nullptr;
  const float *scale = nullptr, *weight = nullptr;
#endif
};
#ifdef QWEN_TRITON_SILU
// ABI/grid adapter only; all activation arithmetic is generated from Triton.
struct Flat {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
extern "C" void silu_forward(int64_t, void *, int64_t, void *, int, int, int,
                             int, int, int, int, int);
#ifdef QWEN_TRITON_SILU_UP
extern "C" void silu_product(int64_t, void *, int64_t, void *, int64_t, void *,
                              int64_t, void *, int64_t, void *, int, int, int,
                              int, int, int, int);
#endif
static void worker(unsigned, unsigned id, void *opaque) {
  const auto &j = *static_cast<Job *>(opaque);
  Flat x{const_cast<float *>(j.in), const_cast<float *>(j.in), 0,
         int64_t(j.rows) * j.istride, 1};
  Flat y{j.out, j.out, 0, int64_t(j.rows) * 3072, 1};
#ifdef QWEN_TRITON_SILU_UP
  if (j.up) {
    Flat up{const_cast<int32_t *>(j.up), const_cast<int32_t *>(j.up), 0, x.size, 1};
    Flat scale{const_cast<float *>(j.scale), const_cast<float *>(j.scale), 0, x.size, 1};
    Flat weight{const_cast<float *>(j.weight), const_cast<float *>(j.weight), 0, 3072, 1};
    silu_product(1, &x, 1, &up, 1, &scale, 1, &weight, 1, &y,
                 j.rows, 4, 1, 1, id, 0, 0);
    return;
  }
#endif
  silu_forward(1, &x, 1, &y, j.rows, j.istride, 4, 1, 1, id, 0, 0);
}

#ifdef QWEN_TRITON_SILU_UP
// Same owner, busy guard and worker pool as ordered SiLU. Numerical operations
// remain in the Triton object; this ABI is only for the admitted quantizer graph.
template <typename T, unsigned Rank> struct ProductMemref {
  T *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
template <typename T, unsigned Rank>
static bool productInput(const ProductMemref<T, Rank> *input, int rows) {
  if (!input || !input->aligned || input->offset < 0 ||
      input->offset > INT32_MAX / 4 - int64_t(rows) * 3072)
    return false;
  if constexpr (Rank == 2)
    return input->sizes[0] == rows && input->sizes[1] == 3072 &&
           input->strides[0] == 3072 && input->strides[1] == 1;
  return rows == 1 && input->sizes[0] == 3072 && input->strides[0] == 1;
}
extern "C" int _mlir_ciface_qwen_quantized_silu_up_v1(
    const Memref *gate, const ProductMemref<int32_t, 2> *up,
    const ProductMemref<float, 2> *scale,
    const ProductMemref<float, 1> *weight, Memref *out) {
  if (!gate || !out || gate->sizes[1] < 4 || gate->sizes[1] > 32)
    return 2;
  int rows = int(gate->sizes[1]);
  for (const auto *tensor : {gate, static_cast<const Memref *>(out)})
    if (!tensor->aligned || tensor->sizes[0] != 1 || tensor->sizes[1] != rows ||
        tensor->sizes[2] != 3072 || tensor->strides[1] != 3072 ||
        tensor->strides[2] != 1 || tensor->offset < 0 ||
        tensor->offset > INT32_MAX / 4 - int64_t(rows) * 3072)
      return 2;
  if (!productInput(up, rows) || !productInput(scale, rows) ||
      !productInput(weight, 1))
    return 2;
  unsigned bytes = unsigned(rows * 3072 * 4);
  Job job{gate->aligned + gate->offset, out->aligned + out->offset, rows, 3072,
          up->aligned + up->offset, scale->aligned + scale->offset,
          weight->aligned + weight->offset};
  const void *inputs[] = {job.in, job.up, job.scale, job.weight};
  unsigned lengths[] = {bytes, bytes, bytes, 3072 * 4};
  uintptr_t output = reinterpret_cast<uintptr_t>(job.out);
  if (bytes > UINTPTR_MAX - output)
    return 2;
  for (unsigned i = 0; i < 4; ++i) {
    uintptr_t start = reinterpret_cast<uintptr_t>(inputs[i]);
    if (lengths[i] > UINTPTR_MAX - start ||
        (start < output + bytes && output < start + lengths[i]))
      return 2; // The three-pass kernel must not overwrite any input.
  }
  if (busy.test_and_set(std::memory_order_acquire))
    return 3;
  struct Guard { ~Guard() { busy.clear(std::memory_order_release); } } guard;
  if (!pool && worker_pool_init(&pool, 4))
    return 4;
  for (unsigned i = 0; i < 4; ++i)
    if (qurt_mem_cache_clean(reinterpret_cast<uintptr_t>(inputs[i]), lengths[i],
                             QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE))
      return 5;
  if (qurt_mem_cache_clean(output, bytes, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE))
    return 5;
  if (worker_pool_run_func(pool, worker, &job, 4))
    return 6;
  return qurt_mem_cache_clean(output, bytes, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE)
             ? 5 : 0;
}
#endif
#else
static void worker(unsigned, unsigned id, void *opaque) {
  const auto &j = *static_cast<Job *>(opaque);
  for (int row = 0; row < j.rows; ++row)
    for (int z = int(id) * 768; z < (int(id) + 1) * 768; ++z) {
      float x = j.in[row * j.istride + z];
      float denominator = 1.0f + __hexagon_silu_exp_compat(-x);
      j.out[row * 3072 + z] = x / denominator;
    }
}
#endif
extern "C" int _mlir_ciface_qwen_ordered_silu_v1(const Memref *in,
                                                 Memref *out) {
  if (!in || !out || !in->aligned || !out->aligned || in->sizes[0] != 1 ||
      out->sizes[0] != 1 || in->sizes[1] < 1 || in->sizes[1] > 32 ||
      out->sizes[1] != in->sizes[1] || in->sizes[2] != 3072 ||
      out->sizes[2] != 3072 || in->offset < 0 || out->offset < 0 ||
      in->strides[2] != 1 || out->strides[2] != 1 || in->strides[1] < 3072 ||
      in->strides[1] > INT32_MAX / 32 || out->strides[1] != 3072 ||
      in->offset > INT32_MAX || out->offset > INT32_MAX ||
      in->offset + (in->sizes[1] - 1) * in->strides[1] + 3072 > INT32_MAX / 4 ||
      out->offset + out->sizes[1] * 3072 > INT32_MAX / 4)
    return 2;
#ifdef QWEN_APPROX_SILU
  // Exceptional vectors are repaired in a second pass, so the input must
  // survive the first pass. Reject aliases before starting any workers.
  uintptr_t src = reinterpret_cast<uintptr_t>(in->aligned + in->offset);
  uintptr_t dst = reinterpret_cast<uintptr_t>(out->aligned + out->offset);
  uintptr_t srcBytes = ((in->sizes[1] - 1) * in->strides[1] + 3072) * 4;
  uintptr_t dstBytes = out->sizes[1] * 3072 * 4;
  if (srcBytes > UINTPTR_MAX - src || dstBytes > UINTPTR_MAX - dst ||
      (src < dst + dstBytes && dst < src + srcBytes))
    return 2;
#endif
  if (busy.test_and_set(std::memory_order_acquire))
    return 3;
  struct Guard {
    ~Guard() { busy.clear(std::memory_order_release); }
  } guard;
  if (!pool && worker_pool_init(&pool, 4))
    return 4;
  Job j{in->aligned + in->offset, out->aligned + out->offset, int(in->sizes[1]),
        int(in->strides[1])};
  // Ordinary DSP memory is scalar/HVX coherent (V73 HVX manual 1.2.3/3.9).
  // The caller owns RPC/DMA synchronization; worker join orders local stores.
  return worker_pool_run_func(pool, worker, &j, 4) ? 6 : 0;
}
