// SPDX-License-Identifier: BSD-3-Clause
#ifndef QWEN_ATTENTION_MEMREF_H
#define QWEN_ATTENTION_MEMREF_H
#include <cstdint>
namespace qwen_attention {
// Arguments are ordinary DSP-visible memory. The caller completes external
// RPC/DMA synchronization before invoking an operator. V73 scalar/HVX accesses
// are coherent (80-N2040-54 AB, sections 1.2.3 and 3.9); worker-pool joins order
// accesses between stages, so local operators need no cache clean/invalidate.
template <typename T> struct MemrefT {
  T *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
template <typename T> struct FlatMemrefT {
  T *allocated, *aligned;
  int64_t offset, size, stride;
};
using Memref = MemrefT<float>;
using FlatMemref = FlatMemrefT<float>;
// KV operand storage. QWEN_KV_FP16 selects IEEE binary16 planes (the llama.cpp
// default F16 KV cache type); kernels declare K/V as *fp16 and the elements
// are only moved here, never converted. Q, mask and output remain FP32.
#ifdef QWEN_KV_FP16
using KVElement = uint16_t;
#else
using KVElement = float;
#endif
using KVMemref = MemrefT<KVElement>;
using FlatKVMemref = FlatMemrefT<KVElement>;

template <typename T> inline int64_t span(const MemrefT<T> &m) {
  int64_t result = 1;
  for (unsigned i = 0; i < 4; ++i)
    result += (m.sizes[i] - 1) * m.strides[i];
  return result;
}

template <typename T>
inline bool valid(const MemrefT<T> *m, int a, int b, int c, int d) {
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

template <typename T> inline FlatMemrefT<T> flatten(const MemrefT<T> &m) {
  return {m.allocated, m.aligned + m.offset, 0, span(m), 1};
}

} // namespace qwen_attention
#endif
