// SPDX-License-Identifier: BSD-3-Clause
#ifndef QWEN_ATTENTION_MEMREF_H
#define QWEN_ATTENTION_MEMREF_H
#include <cstdint>
namespace qwen_attention {
// Arguments are ordinary DSP-visible memory. The caller completes external
// RPC/DMA synchronization before invoking an operator. V73 scalar/HVX accesses
// are coherent (80-N2040-54 AB, sections 1.2.3 and 3.9); worker-pool joins order
// accesses between stages, so local operators need no cache clean/invalidate.
struct Memref {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
struct FlatMemref {
  float *allocated, *aligned;
  int64_t offset, size, stride;
};
inline int64_t span(const Memref &m) {
  int64_t result = 1;
  for (unsigned i = 0; i < 4; ++i)
    result += (m.sizes[i] - 1) * m.strides[i];
  return result;
}

inline bool valid(const Memref *m, int a, int b, int c, int d) {
  if (!m || !m->aligned || m->offset < 0 || m->offset > INT32_MAX / 4 ||
      m->sizes[0] != a || m->sizes[1] != b ||
      m->sizes[2] != c || m->sizes[3] != d || m->strides[3] != 1)
    return false;
  for (unsigned i = 0; i < 4; ++i)
    if (m->strides[i] < 0 || m->strides[i] > INT32_MAX / 4 ||
        (m->strides[i] == 0 && m->sizes[i] != 1))
      return false;
  return m->offset + span(*m) <= INT32_MAX / 4;
}

inline FlatMemref flatten(const Memref &m) {
  return {m.allocated, m.aligned + m.offset, 0, span(m), 1};
}

} // namespace qwen_attention
#endif
