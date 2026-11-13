// SPDX-License-Identifier: BSD-3-Clause
// Bit-preserving 128-byte shuffles. No floating-point conversion or arithmetic.
#ifndef QWEN_ORDERED_KV_PACK_H
#define QWEN_ORDERED_KV_PACK_H
#include <cstdint>
#include <cstring>
#include <utility>
namespace qwen_ordered {
using Words = int32_t __attribute__((vector_size(128)));
template <int Block, int Start, size_t... I>
static inline __attribute__((always_inline)) Words
zip_impl(Words a, Words b, std::index_sequence<I...>) {
  return __builtin_shufflevector(a, b,
                                 int(Start + (I / (2 * Block)) * Block +
                                     (I % Block) + ((I / Block) % 2) * 32)...);
}
template <int Block, int Start>
static inline __attribute__((always_inline)) Words zip(Words a, Words b) {
  return zip_impl<Block, Start>(a, b, std::make_index_sequence<32>{});
}
static inline __attribute__((always_inline)) Words load(const float *p) {
  Words result;
  std::memcpy(&result, p, 128);
  return result;
}
static inline __attribute__((always_inline)) void store(float *p, Words value) {
  std::memcpy(p, &value, 128);
}
// [head,token,128] -> [head,ceil(tokens/8),128,8].
static inline void pack_keys(float *dst, const float *src, int64_t head_stride,
                             int64_t token_stride, int n, int first = 0,
                             int last = 8) {
  int blocks = (n + 7) / 8;
  for (int h = first; h < last; ++h)
    for (int block = 0; block < blocks; ++block)
      for (int z = 0; z < 128; z += 32) {
        Words pair[4][2], quad[2][4];
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
          int token = block * 8 + i * 2;
          Words a = token < n
                        ? load(src + h * head_stride + token * token_stride + z)
                        : Words{};
          Words b =
              token + 1 < n
                  ? load(src + h * head_stride + (token + 1) * token_stride + z)
                  : Words{};
          pair[i][0] = zip<1, 0>(a, b);
          pair[i][1] = zip<1, 16>(a, b);
        }
#pragma clang loop unroll(full)
        for (int i = 0; i < 2; ++i) {
#pragma clang loop unroll(full)
          for (int half = 0; half < 2; ++half) {
            quad[i][2 * half] =
                zip<2, 0>(pair[2 * i][half], pair[2 * i + 1][half]);
            quad[i][2 * half + 1] =
                zip<2, 16>(pair[2 * i][half], pair[2 * i + 1][half]);
          }
        }
#pragma clang loop unroll(full)
        for (int chunk = 0; chunk < 4; ++chunk) {
          store(dst + ((h * blocks + block) * 128 + z + chunk * 8) * 8,
                zip<4, 0>(quad[0][chunk], quad[1][chunk]));
          store(dst + ((h * blocks + block) * 128 + z + chunk * 8 + 4) * 8,
                zip<4, 16>(quad[0][chunk], quad[1][chunk]));
        }
      }
}
// [head,token,128] -> [head,16,ceil(tokens/4)*4,8].
static inline void pack_values(float *dst, const float *src,
                               int64_t head_stride, int64_t token_stride, int n,
                               int first = 0, int last = 8) {
  int padded = (n + 3) / 4 * 4;
  for (int h = first; h < last; ++h)
    for (int token = 0; token < padded; token += 4)
      for (int z = 0; z < 128; z += 32) {
        Words row[4];
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i)
          row[i] =
              token + i < n
                  ? load(src + h * head_stride + (token + i) * token_stride + z)
                  : Words{};
        Words a = zip<8, 0>(row[0], row[1]), b = zip<8, 16>(row[0], row[1]);
        Words c = zip<8, 0>(row[2], row[3]), d = zip<8, 16>(row[2], row[3]);
        store(dst + ((h * 16 + z / 8) * padded + token) * 8, zip<16, 0>(a, c));
        store(dst + ((h * 16 + z / 8 + 1) * padded + token) * 8,
              zip<16, 16>(a, c));
        store(dst + ((h * 16 + z / 8 + 2) * padded + token) * 8,
              zip<16, 0>(b, d));
        store(dst + ((h * 16 + z / 8 + 3) * padded + token) * 8,
              zip<16, 16>(b, d));
      }
}
} // namespace qwen_ordered
#endif
