// SPDX-License-Identifier: BSD-3-Clause
// Triton handles activation layout and signed-bias cancellation; HexKL executes
// u8*i8 HMX; llama.cpp supplies DMA transport. Existing immutable HVX weights
// are read through a 2D DMA view, without another packed weight allocation.
#include <atomic>
#include <cstdint>
#include <cstdlib>
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

bool execute(MemI8 *a, MemI8 *b, MemI32 *c) {
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
