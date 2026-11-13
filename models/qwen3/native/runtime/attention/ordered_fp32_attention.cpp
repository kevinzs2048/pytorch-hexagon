// SPDX-License-Identifier: BSD-3-Clause
#include "ordered_kv_pack.h"
// Ordered FP32 attention for the admitted Qwen policy: q=1..32, KV=q..1024.
// Preserve 128-term QK FMA, four-lane Softmax sum, and native ARM PV blocks.
// Only independent heads/columns are parallelized. KV remains FP32 and owned
// by the caller; the packed buffers below are reusable temporary workspaces.
#pragma clang fp contract(off)
#include "worker-pool.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#ifndef __hexagon__
#include <malloc.h>
#endif
struct Memref {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
#include "../qwen3_sleef_exp_candidate.h"
struct Job {
  const float *Q, *K, *V, *mask;
  float *output, *scratch;
  int m, n, qhead, qrow, maskrow, vhead, vrow;
  const float *rawK, *rawV;
  float *keyWorkspace, *valueWorkspace;
  int khead, krow;
};
static constexpr unsigned MaxWorkers = 8;
static worker_pool_context_t pool;
static float *kp, *vp, *scratch;
static std::atomic_flag busy = ATOMIC_FLAG_INIT;
__attribute__((destructor)) static void cleanup() {
  if (pool)
    worker_pool_release(&pool);
  free(kp);
  free(vp);
  free(scratch);
  kp = vp = scratch = nullptr;
  pool = nullptr;
}
// Independent output lanes; each lane retains the original FMA reduction order.
template <int Fixed>
static __attribute__((noinline)) void
ordered_dot_eight(const float *a, const float *b, int length, float *out) {
  float sums[8] = {};
#pragma clang loop unroll(disable)
  for (int z = 0; z < (Fixed ? Fixed : length); ++z) {
    float value = a[z];
#pragma clang loop unroll(full)
    for (int lane = 0; lane < 8; ++lane)
      sums[lane] = __builtin_fmaf(value, b[z * 8 + lane], sums[lane]);
  }
#pragma clang loop unroll(full)
  for (int lane = 0; lane < 8; ++lane)
    out[lane] = sums[lane];
}

// Two independent query rows or GQA query heads, eight output columns.
// BStep=0 reads strided V directly for short decode; otherwise V/K is packed.
template <int Fixed, int BStep = 8>
static __attribute__((noinline)) void
ordered_dot_pair(const float *a, const float *b, int stride, int length,
                 float *out, int bstride = 0) {
  float sums[16] = {};
#pragma clang loop unroll(disable)
  for (int z = 0; z < (Fixed ? Fixed : length); ++z) {
    float a0 = a[z], a1 = a[stride + z];
#pragma clang loop unroll(full)
    for (int lane = 0; lane < 8; ++lane) {
      float v = b[z * (BStep ? BStep : bstride) + lane];
      sums[lane] = __builtin_fmaf(a0, v, sums[lane]);
      sums[8 + lane] = __builtin_fmaf(a1, v, sums[8 + lane]);
    }
  }
#pragma clang loop unroll(full)
  for (int lane = 0; lane < 16; ++lane)
    out[lane] = sums[lane];
}

// Four workers pack disjoint KV-head ranges. Four-worker compute consumes
// its own range; eight-worker compute starts after packing joins.
static __attribute__((noinline)) void
pack_worker_heads(const Job &j, unsigned workers, unsigned id) {
  int first = 8 * id / workers, last = 8 * (id + 1) / workers;
  qwen_ordered::pack_keys(j.keyWorkspace, j.rawK, j.khead, j.krow, j.n, first,
                          last);
  if (j.m != 1 || j.n > 128)
    qwen_ordered::pack_values(j.valueWorkspace, j.rawV, j.vhead, j.vrow, j.n,
                              first, last);
}

static void packing_worker(unsigned workers, unsigned id, void *raw) {
  pack_worker_heads(*static_cast<Job *>(raw), workers, id);
}

static void worker(unsigned workers, unsigned id, void *raw) {
  const auto &j = *static_cast<Job *>(raw);
  const float *Q = j.Q, *K = j.K, *V = j.V;
  const int m = j.m, n = j.n;
  // Decode pairs the two query heads that share one K/V head. Prefill pairs
  // query rows within a head. Each lane keeps its original reduction order.
  const int rows = m == 1 ? 2 : m;
  if (workers == 4)
    pack_worker_heads(j, workers, id);
  // Each worker owns complete cache lines; unused query capacity need not be
  // flushed between calls. Every consumed score is written by this worker.
  float *scores = j.scratch + id * ((rows * n + 31) / 32 * 32);
  for (int h = int(16 * id / workers); h < int(16 * (id + 1) / workers);
       h += m == 1 ? 2 : 1) {
    // Keep one packed key tile hot while applying it to every query row.
    for (int col = 0; col < n; col += 8)
      for (int row = 0; row < rows; ++row) {
        float accum[8] = {};
        int width = std::min(8, n - col);
        const float *a = Q + h * j.qhead + row * (m == 1 ? j.qhead : j.qrow);
        const float *b = K + ((h / 2) * ((n + 7) / 8) + col / 8) * 128 * 8;
        if (width == 8 && row + 1 < rows) {
          float pair[16];
          ordered_dot_pair<128>(a, b, m == 1 ? j.qhead : j.qrow, 128, pair);
          for (int rr = 0; rr < 2; ++rr)
            for (int lane = 0; lane < 8; ++lane)
              scores[(row + rr) * n + col + lane] =
                  pair[rr * 8 + lane] * float(0.088388347648318447) +
                  j.mask[(m == 1 ? 0 : row + rr) * j.maskrow + col + lane];
          ++row;
          continue;
        }
        if (width == 8)
          ordered_dot_eight<128>(a, b, 128, accum);
        else
          for (int z = 0; z < 128; ++z) {
#pragma clang loop unroll(full)
            for (int lane = 0; lane < 8; ++lane)
              if (lane < width) {
                if (m == 1 && n < 4)
                  accum[lane] += a[z] * b[z * 8 + lane];
                else
                  accum[lane] =
                      __builtin_fmaf(a[z], b[z * 8 + lane], accum[lane]);
              }
          }
        for (int lane = 0; lane < width; ++lane)
          scores[row * n + col + lane] =
              accum[lane] * float(0.088388347648318447) +
              j.mask[(m == 1 ? 0 : row) * j.maskrow + col + lane];
      }
    for (int row = 0; row < rows; ++row) {
      float *p = scores + row * n, maximum = -INFINITY;
      for (int col = 0; col < n; ++col)
        if (p[col] > maximum || std::isnan(p[col]))
          maximum = p[col];
      for (int col = 0; col < n; ++col)
        p[col] = qwen3_sleef_exp_candidate(p[col] - maximum);
      float sum = 0;
      if (n < 4)
        for (int col = 0; col < n; ++col)
          sum += p[col];
      else {
        float lanes[4] = {};
        int end = n / 4 * 4;
        for (int col = 0; col < end; col += 4)
          for (int lane = 0; lane < 4; ++lane)
            lanes[lane] += p[col + lane];
        for (int lane = 0; end + lane < n; ++lane)
          lanes[lane] += p[end + lane];
        sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
      }
      float inverse = 1.0f / sum;
      for (int col = 0; col < n; ++col)
        p[col] *= inverse;
    }
    // Likewise reuse a value tile across query rows, preserving each PV block.
    for (int col = 0; col < 128; col += 8)
      for (int row = 0; row < rows; ++row) {
        const float *p = scores + row * n;
        if (row + 1 < rows) {
          float total[16] = {};
          for (int start = 0; start < n;) {
            int remaining = n - start, width = remaining;
            if (m > 1 && remaining >= 704)
              width = 352;
            else if (m > 1 && remaining > 352)
              width = m * n >= 4096 ? (remaining + 1) / 2
                                    : ((remaining / 2 + 15) / 16) * 16;
            float pair[16];
            if (m == 1 && n <= 128)
              ordered_dot_pair<0, 0>(p, V + (h / 2) * j.vhead + col,
                                    n, width, pair, j.vrow);
            else
              ordered_dot_pair<0>(
                  p + start,
                  V + (((h / 2) * 16 + col / 8) * ((n + 3) / 4 * 4) + start) * 8,
                  n, width, pair);
            for (int lane = 0; lane < 16; ++lane)
              if (m == 1)
                total[lane] = pair[lane];
              else
                total[lane] += pair[lane];
            start += width;
          }
          for (int rr = 0; rr < 2; ++rr)
            for (int lane = 0; lane < 8; ++lane)
              j.output[((m == 1 ? 0 : row + rr) * 16 + h +
                        (m == 1 ? rr : 0)) * 128 + col + lane] =
                  total[rr * 8 + lane];
          ++row;
          continue;
        }
        // Only an odd prefill row reaches this path; decode uses head pairs.
        float total[8] = {};
        for (int start = 0; start < n;) {
          int remaining = n - start, width = remaining;
          if (m > 1 && remaining >= 704)
            width = 352;
          else if (m > 1 && remaining > 352)
            width = m * n >= 4096 ? (remaining + 1) / 2
                                  : ((remaining / 2 + 15) / 16) * 16;
          float partial[8];
          ordered_dot_eight<0>(
              p + start,
              V + (((h / 2) * 16 + col / 8) * ((n + 3) / 4 * 4) + start) * 8,
              width, partial);
          for (int lane = 0; lane < 8; ++lane)
            total[lane] += partial[lane];
          start += width;
        }
        for (int lane = 0; lane < 8; ++lane)
          j.output[(row * 16 + h) * 128 + col + lane] = total[lane];
      }
  }
}

static bool valid(const Memref *x, int a, int b, int c, int d) {
  if (!x || !x->aligned || x->offset < 0 || x->offset > INT32_MAX ||
      x->sizes[0] != a || x->sizes[1] != b || x->sizes[2] != c ||
      x->sizes[3] != d || x->strides[3] != 1)
    return false;
  int64_t span = x->offset;
  for (int z = 0; z < 4; ++z) {
    if (x->strides[z] < 0 || x->strides[z] > INT32_MAX ||
        (x->strides[z] == 0 && x->sizes[z] != 1))
      return false;
    span += (x->sizes[z] - 1) * x->strides[z];
  }
  return span < INT32_MAX / 4;
}
extern "C" int _mlir_ciface_qwen_ordered_fp32_attention_v1(const Memref *q,
                                                           const Memref *k,
                                                           const Memref *v,
                                                           const Memref *mask,
                                                           Memref *out) {
  if (!q || !k)
    return 2;
  if (q->sizes[2] < 1 || q->sizes[2] > 32 || k->sizes[2] < q->sizes[2] ||
      k->sizes[2] > 1024)
    return 2;
  int m = int(q->sizes[2]), n = int(k->sizes[2]);
  if (!valid(q, 1, 16, m, 128) || !valid(k, 1, 8, n, 128) ||
      !valid(v, 1, 8, n, 128) || !valid(mask, 1, 1, m, n) ||
      !valid(out, 1, m, 16, 128) || out->strides[1] != 2048 ||
      out->strides[2] != 128)
    return 2;
  if (busy.test_and_set())
    return 3;
  struct Unlock {
    ~Unlock() { busy.clear(); }
  } unlock;
  if (!pool) {
    kp = (float *)memalign(128, 8 * 1024 * 128 * 4);
    vp = (float *)memalign(128, 8 * 1024 * 128 * 4);
    scratch = (float *)memalign(128, MaxWorkers * 32 * 1024 * 4);
    if (!kp || !vp || !scratch || worker_pool_init(&pool, MaxWorkers)) {
      cleanup();
      return 4;
    }
  }
  const auto *Q = q->aligned + q->offset;
  const auto *K = k->aligned + k->offset;
  const auto *V = v->aligned + v->offset;
  // More workers help long multi-row attention, but regress long decode.
  const unsigned workers = m >= 8 && n >= 256 ? 8 : 4;
  Job job{Q,
          kp,
          m == 1 && n <= 128 ? V : vp,
          mask->aligned + mask->offset,
          out->aligned + out->offset,
          scratch,
          m,
          n,
          int(q->strides[1]),
          int(q->strides[2]),
          int(mask->strides[2]),
          int(v->strides[1]),
          int(v->strides[2]),
          K,
          V,
          kp,
          vp,
          int(k->strides[1]),
          int(k->strides[2])};
  // Eight concurrent pack-and-compute workers regress long prefill on this
  // target. Pack with four first, then join before the eight compute workers.
  // Ordinary DSP memory is scalar/HVX coherent; RPC/DMA synchronization is the
  // caller's responsibility, not a per-stage cache flush (80-N2040-54 AB 3.9).
  if (workers == 8 && worker_pool_run_func(pool, packing_worker, &job, 4))
    return 6;
  if (worker_pool_run_func(pool, worker, &job, workers))
    return 6;
  return 0;
}
