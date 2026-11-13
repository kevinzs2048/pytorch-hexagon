// SPDX-License-Identifier: BSD-3-Clause
#ifndef QWEN_PREPACKED_DMA_H
#define QWEN_PREPACKED_DMA_H
#include "hex-dma.h"
#include <HAP_compute_res.h>
#include <qurt_thread.h>

// Two complete 128-column tiles per worker, bounded by Qwen's largest K.
// Only immutable, already packed weights are transferred. Activations and
// accumulators keep the original exact integer kernel and DDR layout.
#define QWEN_DMA_TILE_BYTES (3072u * 128u)
static unsigned qwen_dma_resource;
static unsigned char *qwen_dma_storage;
static int qwen_dma_initialized;
// The resident graph is serialized. Keep this bounded lease only while its
// synchronous forward runs; standalone matmul callers retain per-op release.
static int qwen_dma_forward_active, qwen_dma_held;
static unsigned qwen_dma_forward_owner;

void qwen_prepacked_forward_begin(void) {
  if (qwen_dma_forward_active || qwen_dma_held) abort();
  qwen_dma_forward_owner = qurt_thread_get_id();
  qwen_dma_forward_active = 1;
}

void qwen_prepacked_forward_end(void) {
  if (!qwen_dma_forward_active ||
      qwen_dma_forward_owner != qurt_thread_get_id()) abort();
  if (qwen_dma_held) HAP_compute_res_release_cached(qwen_dma_resource);
  qwen_dma_held = 0;
  qwen_dma_forward_active = 0;
}
static struct {
  dma_queue queue;
  dma_descriptor_2d descriptors[4] __attribute__((aligned(64)));
  dma_ptr pointers[4];
} qwen_dma_workers[4] __attribute__((aligned(128)));

static void qwen_dma_destroy(void) {
  if (qwen_dma_held) abort();
  if (qwen_dma_resource) HAP_compute_res_release(qwen_dma_resource);
  qwen_dma_resource = 0;
  qwen_dma_storage = NULL;
}

// Called under the existing prepacked_busy guard. Resource unavailability
// retains the DDR kernel. Do not retry expensive allocation for every layer.
static int qwen_dma_acquire(size_t k) {
  if (qwen_dma_forward_active &&
      qwen_dma_forward_owner != qurt_thread_get_id()) abort();
  if (k > 3072) return 0;
  if (qwen_dma_held) return 1;
  if (!qwen_dma_initialized) {
    qwen_dma_initialized = 1;
    compute_res_attr_t attr;
    unsigned size = 8 * QWEN_DMA_TILE_BYTES;
    void *storage = NULL;
    if (HAP_compute_res_attr_init(&attr) ||
        HAP_compute_res_attr_set_serialize(&attr, 0) ||
        HAP_compute_res_attr_set_cache_mode(&attr, 1) ||
        HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size)) return 0;
    qwen_dma_resource = HAP_compute_res_acquire(&attr, 1000);
    if (!qwen_dma_resource) return 0;
    if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &storage, &size) ||
        !storage || size < 8 * QWEN_DMA_TILE_BYTES) {
      qwen_dma_destroy();
      return 0;
    }
    qwen_dma_storage = storage;
    for (unsigned i = 0; i < 4; ++i) {
      dma_queue *queue = &qwen_dma_workers[i].queue;
      queue->desc = qwen_dma_workers[i].descriptors;
      queue->dptr = qwen_dma_workers[i].pointers;
      queue->capacity = 4;
      queue->idx_mask = 3;
      queue->tail = &queue->desc[3];
    }
  }
  int acquired = qwen_dma_resource &&
                 !HAP_compute_res_acquire_cached(qwen_dma_resource, 1000);
  if (acquired && qwen_dma_forward_active) qwen_dma_held = 1;
  return acquired;
}

static void qwen_dma_release(void) {
  // Workers have joined and consumed every queued transfer before release.
  // A resident forward releases once after every operator has joined. Outside
  // that scope, preserve the original per-matmul resource lifetime.
  if (!qwen_dma_held) HAP_compute_res_release_cached(qwen_dma_resource);
}
#endif
