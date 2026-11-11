// SPDX-License-Identifier: BSD-3-Clause
// Triton handles activation layout and signed-bias cancellation; HexKL executes
// u8*i8 HMX; llama.cpp supplies DMA transport. Existing immutable HVX weights
// are read through a 2D DMA view, without another packed weight allocation.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#ifndef restrict
#define restrict __restrict
#endif
#include "hexkl_micro.h"

extern "C" {
void *memalign(size_t, size_t);
void *qwen_int8_dma_init();
int qwen_int8_dma_push(void *, void *, const void *, unsigned);
void *qwen_int8_dma_pop(void *);
void qwen_int8_dma_drain(void *);
}

struct MemI8 {
  int8_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct MemI32 {
  int32_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct Flat {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
extern "C" void qwen_hvx_matmul(MemI8 *, MemI8 *, MemI32 *);
extern "C" void qwen_int8_pack_activation(
    int64_t, void *, int64_t, void *, int, int, int, int, int, int, int, int);
extern "C" void qwen_int8_accumulate_result(
    int64_t, void *, int64_t, void *, int, int, int, int, int, int, int, int, int);
#ifdef QWEN_INT8_FUSED_DEQUANT
// Fused output stages (Triton): dequantize each HMX result tile in the order
// of the graph they replace; see models/qwen3/kernels/int8_prefill.py.
extern "C" void qwen_int8_dequant_result(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int, int, int, int, int, int, int, int, int, int, int, int, int);
extern "C" void qwen_int8_dequant_residual_result(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int64_t, void *, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
extern "C" void qwen_int8_dequant_product_result(
    int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
    int64_t, void *, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
struct MemF32 {
  float *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct MemF32Vector {
  float *allocated, *aligned;
  int64_t offset, size, stride;
};
#endif

#ifdef QWEN_SHARED_HMX_RUNTIME
// Generated attention and INT8 projections must use one HMX owner thread.
// A second cached HAP lease can prevent either operator acquiring the device.
extern "C" void hexagon_runtime_dispatch_hmx_dsp(void (*)(void *), void *);
extern "C" void *hexagon_runtime_acquire_hmx_workspace_dsp(size_t, uint64_t, bool);
extern "C" void hexagon_runtime_release_hmx_workspace_dsp(void *);
extern "C" int hexagon_runtime_configure_vtcm_pool_dsp(unsigned);
extern "C" int hexagon_runtime_configure_hmx_performance_dsp(unsigned);
#endif

namespace {
constexpr unsigned WorkspaceBytes = 1048576;
constexpr unsigned WeightCount = 28 * 7;
static std::atomic_flag busy = ATOMIC_FLAG_INIT;
static unsigned storage_size;
#ifndef QWEN_SHARED_HMX_RUNTIME
static unsigned resource, power_client;
static bool initialized, powered;
#endif
static uint8_t *storage;
struct Weight {
  const int8_t *base;
  unsigned k, n;
};
static Weight weights[WeightCount];
static unsigned weight_count;

#ifdef QWEN_SHARED_HMX_RUNTIME
bool acquire() {
  storage = static_cast<uint8_t *>(
      hexagon_runtime_acquire_hmx_workspace_dsp(WorkspaceBytes, 128, true));
  storage_size = WorkspaceBytes;
  return storage != nullptr;
}
#else
__attribute__((destructor)) void cleanup() {
  if (resource) HAP_compute_res_release(resource);
  if (powered) HAP_power_destroy(&power_client);
}

// Cache the resource descriptor, but release its physical VTCM/HMX lease after
// every operation. Attention and the HVX fallback share those same resources.
bool acquire() {
  if (!initialized) {
    initialized = true;
    HAP_power_request_t p{};
    p.type = HAP_power_set_HMX;
    p.hmx.power_up = TRUE;
    if (HAP_power_set(&power_client, &p)) return false;
    powered = true;
    compute_res_attr_t attr;
    void *base = nullptr;
    unsigned size = WorkspaceBytes;
    if (HAP_compute_res_attr_init(&attr) ||
        HAP_compute_res_attr_set_serialize(&attr, 0) ||
        HAP_compute_res_attr_set_cache_mode(&attr, 1) ||
        HAP_compute_res_attr_set_hmx_param(&attr, 1) ||
        HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size)) return false;
    resource = HAP_compute_res_acquire(&attr, 1000);
    if (!resource) return false;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &base, &size) ||
        !base || size < WorkspaceBytes) {
      HAP_compute_res_release(resource);
      resource = 0;
      return false;
    }
    storage = static_cast<uint8_t *>(base);
    storage_size = size;
  }
  return resource && !HAP_compute_res_acquire_cached(resource, 1000);
}
#endif

bool eligible(const MemI8 *a, const MemI8 *b, const MemI32 *c) {
  if (!a || !b || !c || !a->aligned || !b->aligned || !c->aligned ||
      a->sizes[0] < 16 || a->sizes[0] > 32 ||
      c->sizes[0] != a->sizes[0] ||
      (a->sizes[0] < 32 && c->sizes[1] < 1024) ||
      (a->sizes[1] != 1024 && a->sizes[1] != 2048 && a->sizes[1] != 3072) ||
      b->sizes[0] != a->sizes[1] || b->sizes[1] != c->sizes[1] ||
      c->sizes[1] < 128 || c->sizes[1] > 3072 || c->sizes[1] % 128 ||
      a->strides[1] != 1 || b->strides[1] != 1 || c->strides[1] != 1 ||
      a->strides[0] != a->sizes[1] || b->strides[0] != b->sizes[1] ||
      c->strides[0] != c->sizes[1] ||
      a->offset < 0 || a->offset > INT32_MAX - a->sizes[0] * a->sizes[1] ||
      b->offset < 0 || b->offset > INT32_MAX - b->sizes[0] * b->sizes[1] ||
      c->offset < 0 || c->offset > INT32_MAX - c->sizes[0] * c->sizes[1]) return false;
  return reinterpret_cast<uintptr_t>(b->aligned + b->offset) % 128 == 0;
}

bool publish_weights(const int8_t *base, unsigned k, unsigned n) {
  // The prepacked ABI owns immutable weight storage for the entire session.
  for (unsigned i = 0; i < weight_count; ++i)
    if (weights[i].base == base && weights[i].k == k && weights[i].n == n)
      return true;
  if (weight_count == WeightCount ||
      qurt_mem_cache_clean(reinterpret_cast<uintptr_t>(base), k * n,
                           QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE)) return false;
  weights[weight_count++] = {base, k, n};
  return true;
}

#ifdef QWEN_INT8_FUSED_DEQUANT
extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t),
                                          void *, uint64_t);
// Raw accumulator tiles of one projection: up to 3072 / 32 tiles of 33 x 32.
alignas(128) int32_t staged_tiles[96 * 33 * 32];

struct Dequant {
  float *row_scale, *column_scale, *residual, *out;
  int scale_stride, residual_stride;
  bool product = false;  // residual operand multiplies instead of adds
  std::atomic<unsigned> *ready = nullptr;  // tiles published by the HMX owner
  // Whole-vector byte copy: scalar loads from VTCM are very slow. Both the
  // HMX result tile and the staging rows are 128-byte aligned.
  void stage(const int32_t *tile, unsigned rows, unsigned column) const {
    typedef int32_t Row __attribute__((vector_size(128), aligned(128)));
    auto *source = reinterpret_cast<const Row *>(tile);
    auto *target = reinterpret_cast<Row *>(staged_tiles + (column / 32) * 33 * 32);
    for (unsigned row = 0; row < rows; ++row) target[row] = source[row];
    target[32] = source[32];
    if (ready) {
      __asm__ __volatile__("barrier" ::: "memory");
      ready->store(column / 32 + 1, std::memory_order_release);
    }
  }
  // Runs the Triton output stage over all staged tiles on the worker pool.
  void finish(unsigned rows, unsigned n) const {
    struct Job { const Dequant *dequant; unsigned rows, n; } job{this, rows, n};
    __hexagon_parallel_for_v1([](void *opaque, uint64_t start, uint64_t count) {
      auto &j = *static_cast<Job *>(opaque);
      for (uint64_t t = start; t < start + count; ++t)
        j.dequant->tile(staged_tiles + t * 33 * 32, j.rows, j.n, unsigned(t * 32));
    }, &job, n / 32);
  }
  // Writes rows x (32 * tiles) FP32 outputs starting at `column` from INT32
  // accumulators `stride` apart; `bias` is the zero-point row for every tile.
  void run(const int32_t *accumulator, const int32_t *bias, unsigned rows, unsigned n,
           unsigned column, unsigned tiles, unsigned stride) const {
    Flat acc{const_cast<int32_t *>(accumulator), const_cast<int32_t *>(accumulator), 0, 1, 1};
    Flat zero{const_cast<int32_t *>(bias), const_cast<int32_t *>(bias), 0, 32, 1};
    Flat output{out, out, 0, rows * n, 1};
    Flat scales{row_scale, row_scale, 0, 1, 1};
    Flat weights{column_scale, column_scale, 0, n, 1};
    int r = rows, c = n, col = column, t = tiles, s = stride;
    if (residual) {
      Flat source{residual, residual, 0, 1, 1};
      (product ? qwen_int8_dequant_product_result : qwen_int8_dequant_residual_result)(
          1, &acc, 1, &zero, 1, &output, 1, &scales, 1, &weights, 1, &source,
          r, c, col, t, s, 0, scale_stride, residual_stride, 1, 1, 1, 0, 0, 0);
    } else {
      qwen_int8_dequant_result(1, &acc, 1, &zero, 1, &output, 1, &scales, 1, &weights,
          r, c, col, t, s, 0, scale_stride, 1, 1, 1, 0, 0, 0);
    }
  }
  // One staged HMX tile: rows x 32 accumulators, zero-point bias in row 32.
  void tile(const int32_t *accumulator, unsigned rows, unsigned n, unsigned column) const {
    run(accumulator, accumulator + 32 * 32, rows, n, column, 1, 32);
  }
};
#else
struct Dequant {
  void stage(const int32_t *, unsigned, unsigned) const {}
};
#endif

bool execute(MemI8 *a, MemI8 *b, MemI32 *c, const Dequant *dequant = nullptr) {
  if (!eligible(a, b, c) || busy.test_and_set()) return false;
  struct BusyGuard { ~BusyGuard() { busy.clear(); } } busy_guard;
  if (!acquire()) return false;
  struct Lease {
    bool hvx = false;
#ifndef QWEN_SHARED_HMX_RUNTIME
    bool hmx = false;
#endif
    ~Lease() {
      if (hvx) qurt_hvx_unlock();
#ifdef QWEN_SHARED_HMX_RUNTIME
      hexagon_runtime_release_hmx_workspace_dsp(storage);
#else
      if (hmx) HAP_compute_res_hmx_unlock(resource);
      HAP_compute_res_release_cached(resource);
#endif
    }
  } lease;
#ifndef QWEN_SHARED_HMX_RUNTIME
  if (HAP_compute_res_hmx_lock(resource)) return false;
  lease.hmx = true;
#endif
  if (qurt_hvx_get_lock_val() != QURT_HVX_LOCKED) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return false;
    lease.hvx = true;
  }

  unsigned rows = a->sizes[0], k = a->sizes[1], n = c->sizes[1];
  auto *ap = a->aligned + a->offset;
  auto *bp = b->aligned + b->offset;
  auto *cp = c->aligned + c->offset;
  if (!publish_weights(bp, k, n)) return false;
  auto *vtcm = storage;
  unsigned cfg = (storage_size - hexkl_micro_hmx_config_size()) &
                 ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1U);
  unsigned result = cfg - 8192, weight = 64 * k;
  if (weight + 64 * k > result ||
      hexkl_micro_hmx_setup_acc_read_int32(vtcm, cfg)) return false;
  Flat input{ap, ap, 0, rows * k, 1}, packed{vtcm, vtcm, 0, 64 * k, 1};
  qwen_int8_pack_activation(1, &input, 1, &packed, rows, k, 1, 1, 1, 0, 0, 0);

  void *queue = qwen_int8_dma_init();
  struct Drain {
    void *queue;
    ~Drain() { qwen_int8_dma_drain(queue); }
  } drain{queue};
  auto push = [&](void *dst, unsigned tile) {
    return qwen_int8_dma_push(queue, dst,
        bp + (tile / 4) * k * 128 + (tile % 4) * 128, k / 4);
  };
  unsigned pushed = 0;
  for (unsigned slot = 0; slot < 2 && pushed < n / 32; ++slot, ++pushed)
    if (!push(vtcm + weight + slot * 32 * k, pushed)) std::abort();

  // Once output accumulation starts, errors cannot fall back: repeating the
  // operation would double-add a partially completed result.
  for (unsigned col = 0; col < n; col += 32) {
    void *ready = qwen_int8_dma_pop(queue);
    if (!ready) std::abort();
    unsigned active = static_cast<uint8_t *>(ready) - vtcm;
    hexkl_micro_hmx_acc_clear_int32();
    for (unsigned z = 0; z < k / 32; ++z)
      if (hexkl_micro_hmx_mm_u8i8(vtcm, z * 2048, active + z * 1024)) std::abort();
    if (pushed < n / 32 && !push(ready, pushed++)) std::abort();
    if (hexkl_micro_hmx_acc_read_int32(vtcm, cfg, result)) std::abort();
    if (dequant) {
      // Stage the raw tile (rows plus the bias row); the floating-point
      // output stage runs afterwards on HVX workers, off the HMX path.
      dequant->stage(reinterpret_cast<const int32_t *>(vtcm + result), rows, col);
      continue;
    }
    Flat acc{vtcm + result, vtcm + result, 0, 2048, 1};
    Flat output{cp, cp, 0, rows * n, 1};
    qwen_int8_accumulate_result(1, &acc, 1, &output, rows, n, col, 1, 1, 1, 0, 0, 0);
  }
  return true;
}
} // namespace

extern "C" void _mlir_ciface___hexagon_i8_prepacked_v1_matmul(
    MemI8 *a, MemI8 *b, MemI32 *c) {
#ifdef QWEN_SHARED_HMX_RUNTIME
  if (!eligible(a, b, c)) {
    qwen_hvx_matmul(a, b, c);
    return;
  }
  // One MiB INT8 scratch plus attention preparation slots share the pool.
  if (hexagon_runtime_configure_hmx_performance_dsp(1) ||
      hexagon_runtime_configure_vtcm_pool_dsp(2 * 1024 * 1024))
    std::abort();
  struct Request {
    MemI8 *a, *b;
    MemI32 *c;
    bool succeeded = false;
  } request{a, b, c};
  hexagon_runtime_dispatch_hmx_dsp([](void *opaque) {
    auto &r = *static_cast<Request *>(opaque);
    r.succeeded = execute(r.a, r.b, r.c);
  }, &request);
  if (!request.succeeded) qwen_hvx_matmul(a, b, c);
#else
  if (!execute(a, b, c)) qwen_hvx_matmul(a, b, c);
#endif
}

#ifdef QWEN_INT8_FUSED_DEQUANT
namespace {
// HVX fallback for the fused entries: exact INT32 product first, then the same
// Triton output stage per 32-column tile (bias row zero), so floating-point
// arithmetic stays in the maintained kernel.
alignas(128) int32_t fallback_product[32 * 3072];
alignas(128) const int32_t zero_bias[32] = {};

void dequantized_matmul(MemI8 *a, MemI8 *b, MemF32 *scale, MemF32Vector *weight,
                        MemF32 *residual, MemF32 *out, bool multiply = false) {
  int64_t rows = a ? a->sizes[0] : 0, n = out ? out->sizes[1] : 0;
  if (!a || !b || !scale || !weight || !out || rows < 1 || rows > 32 ||
      n < 32 || n % 32 || (n > 3072 && (rows != 1 || n > 151936)) || out->sizes[0] != rows ||
      out->strides[0] != n || out->strides[1] != 1 || b->sizes[1] != n ||
      (scale->sizes[0] != 1 && scale->sizes[0] < rows) || scale->sizes[1] != 1 ||
      weight->size != n || weight->stride != 1 ||
      (residual && (residual->sizes[0] != rows || residual->sizes[1] != n ||
                    residual->strides[1] != 1)))
    std::abort();
  Dequant dequant{scale->aligned + scale->offset, weight->aligned + weight->offset,
                  residual ? residual->aligned + residual->offset : nullptr,
                  out->aligned + out->offset,
                  scale->sizes[0] == 1 ? 0 : static_cast<int>(scale->strides[0]),
                  residual ? static_cast<int>(residual->strides[0]) : 0, multiply};
  // Shape-only descriptor for the existing eligibility check; never written.
  MemI32 shape{reinterpret_cast<int32_t *>(dequant.out),
               reinterpret_cast<int32_t *>(dequant.out), 0, {rows, n}, {n, 1}};
#ifdef QWEN_SHARED_HMX_RUNTIME
  if (eligible(a, b, &shape)) {
    if (hexagon_runtime_configure_hmx_performance_dsp(1) ||
        hexagon_runtime_configure_vtcm_pool_dsp(2 * 1024 * 1024))
      std::abort();
    // Pipeline (as llama.cpp overlaps HVX work with its HMX queue): the worker
    // owning row 0 submits the HMX job and blocks; the others run the output
    // stage on each tile as soon as the HMX owner has staged it, and the
    // submitting worker joins them afterwards. One pool dispatch per call.
    std::atomic<unsigned> ready{0};
    dequant.ready = &ready;
    struct Pipeline {
      MemI8 *a, *b;
      MemI32 *c;
      const Dequant *dequant;
      unsigned rows, n, tiles;
      std::atomic<unsigned> next{0};
      std::atomic<bool> finished{false};
      bool succeeded = false;
    } pipeline{a, b, &shape, &dequant, unsigned(rows), unsigned(n), unsigned(n / 32)};
    __hexagon_parallel_for_v1([](void *opaque, uint64_t start, uint64_t) {
      auto &p = *static_cast<Pipeline *>(opaque);
      if (start == 0) {
        hexagon_runtime_dispatch_hmx_dsp([](void *opaque) {
          auto &q = *static_cast<Pipeline *>(opaque);
          q.succeeded = execute(q.a, q.b, q.c, q.dequant);
        }, &p);
        p.finished.store(true, std::memory_order_release);
      }
      for (;;) {
        unsigned tile = p.next.fetch_add(1, std::memory_order_relaxed);
        if (tile >= p.tiles) return;
        while (p.dequant->ready->load(std::memory_order_acquire) <= tile) {
          // Only a failure before the first tile ends the job without it.
          if (p.finished.load(std::memory_order_acquire) &&
              p.dequant->ready->load(std::memory_order_acquire) <= tile)
            return;
          __asm__ __volatile__("pause(#64)");
        }
        p.dequant->tile(staged_tiles + tile * 33 * 32, p.rows, p.n, tile * 32);
      }
    }, &pipeline, 64);
    dequant.ready = nullptr;
    if (pipeline.succeeded) return;
  }
#else
  if (execute(a, b, &shape, &dequant)) {
    dequant.finish(rows, n);
    return;
  }
#endif
  // Small queries (decode, prefill tails) and the single-row LM head: exact
  // INT32 product on the HVX path, then the output stage split by column
  // tiles over the workers (a serial call when there are few tiles).
  int32_t *buffer = fallback_product;
  if (rows * n > 32 * 3072) {
    static int32_t *wide = nullptr;  // one row of up to 151936 columns
    if (!wide && !(wide = static_cast<int32_t *>(memalign(128, 151936 * sizeof(int32_t)))))
      std::abort();
    buffer = wide;
  }
  std::memset(buffer, 0, size_t(rows * n) * sizeof(int32_t));
  MemI32 product{buffer, buffer, 0, {rows, n}, {n, 1}};
  qwen_hvx_matmul(a, b, &product);
  unsigned tiles = unsigned(n / 32);
  if (tiles < 16) {
    dequant.run(buffer, zero_bias, unsigned(rows), unsigned(n), 0, tiles, unsigned(n));
    return;
  }
  struct Columns { const Dequant *dequant; const int32_t *product; unsigned rows, n; } job{
      &dequant, buffer, unsigned(rows), unsigned(n)};
  __hexagon_parallel_for_v1([](void *opaque, uint64_t start, uint64_t count) {
    auto &j = *static_cast<Columns *>(opaque);
    unsigned column = unsigned(start * 32);
    j.dequant->run(j.product + column, zero_bias, j.rows, j.n, column, unsigned(count), j.n);
  }, &job, tiles);
}
} // namespace

extern "C" void _mlir_ciface___hexagon_i8_prepacked_v1_matmul_dequant(
    MemI8 *a, MemI8 *b, MemF32 *scale, MemF32Vector *weight, MemF32 *out) {
  dequantized_matmul(a, b, scale, weight, nullptr, out);
}

extern "C" void _mlir_ciface___hexagon_i8_prepacked_v1_matmul_dequant_residual(
    MemI8 *a, MemI8 *b, MemF32 *scale, MemF32Vector *weight, MemF32 *residual,
    MemF32 *out) {
  dequantized_matmul(a, b, scale, weight, residual, out);
}
#endif
#ifdef QWEN_INT8_FUSED_DEQUANT
extern "C" void _mlir_ciface___hexagon_i8_prepacked_v1_matmul_dequant_product(
    MemI8 *a, MemI8 *b, MemF32 *scale, MemF32Vector *weight, MemF32 *factor,
    MemF32 *out) {
  dequantized_matmul(a, b, scale, weight, factor, out, true);
}
#endif
