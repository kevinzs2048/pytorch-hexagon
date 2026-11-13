// SPDX-License-Identifier: BSD-3-Clause
// DMA queue is owned by the serialized HMX matmul runtime; no numerical work.
#include "hex-dma.h"
#include <string.h>

static dma_descriptor_2d descriptors[4] __attribute__((aligned(64)));
static dma_ptr pointers[4];
static dma_queue queue;

void *qwen_int8_dma_init(void) {
  memset(&queue, 0, sizeof(queue));
  memset(descriptors, 0, sizeof(descriptors));
  queue.desc = descriptors;
  queue.dptr = pointers;
  queue.capacity = 4;
  queue.idx_mask = 3;
  queue.tail = &descriptors[3];
  return &queue;
}

int qwen_int8_dma_push(void *q, void *dst, const void *src, unsigned rows) {
  // [K/4,128 columns,4 bytes] -> [K/4,32 columns,4 bytes].
  return dma_queue_push_single_2d(q, dma_make_ptr(dst, src), 128, 512, 128, rows);
}

void *qwen_int8_dma_pop(void *q) { return dma_queue_pop(q).dst; }

void qwen_int8_dma_drain(void *q) {
  dma_queue *p = q;
  while (p->pop_idx != p->push_idx) dma_queue_pop(p);
}
