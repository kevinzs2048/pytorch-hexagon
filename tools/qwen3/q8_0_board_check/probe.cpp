// SPDX-License-Identifier: BSD-3-Clause
// DSP side of tools/qwen3/q8_0_board_check: one Q8_0 projection of the linked
// weight table per packet, timed with HAP_perf (best of five after a warm-up).
// Packet QWQ8002: u32 m, n, k, mode, id (pad to 64 bytes); x f32 [m*k], side f32 [m*n].
// mode: 0 matmul, 1 matmul_residual, 2 matmul_product,
//       3 matvec_q8, 4 matvec_q8_residual, 5 matvec_q8_product.
// Reply: magic, u64 best us at 16, then y f32 [m*n] at 64.
#include <cstdint>
#include <cstring>
#include "HAP_perf.h"
extern "C" void *memalign(size_t, size_t);
extern "C" void free(void *);
struct MemF32 { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; };
using Plain = void (*)(MemF32 *, int64_t, MemF32 *);
using Side = void (*)(MemF32 *, int64_t, MemF32 *, MemF32 *);
extern "C" {
void _mlir_ciface___hexagon_q8_0_matmul(MemF32 *, int64_t, MemF32 *);
void _mlir_ciface___hexagon_q8_0_matmul_residual(MemF32 *, int64_t, MemF32 *, MemF32 *);
void _mlir_ciface___hexagon_q8_0_matmul_product(MemF32 *, int64_t, MemF32 *, MemF32 *);
void _mlir_ciface___hexagon_q8_0_matvec_q8(MemF32 *, int64_t, MemF32 *);
void _mlir_ciface___hexagon_q8_0_matvec_q8_residual(MemF32 *, int64_t, MemF32 *, MemF32 *);
void _mlir_ciface___hexagon_q8_0_matvec_q8_product(MemF32 *, int64_t, MemF32 *, MemF32 *);
const uint8_t *qwen_q8_0_weight(int64_t id, uint32_t *n, uint32_t *k);
}
int main() { return 2; }
extern "C" int hexagon_forward_buffers(const unsigned char *in, int in_len, unsigned char *out, int out_len, uint64_t *us) {
  if (!in || in_len < 64 || std::memcmp(in, "QWQ8002\0", 8)) return 2;
  uint32_t h[5];
  std::memcpy(h, in + 8, sizeof h);
  const uint32_t m = h[0], n = h[1], k = h[2], mode = h[3], id = h[4];
  uint32_t tn = 0, tk = 0;
  if (mode > 5 || !qwen_q8_0_weight(id, &tn, &tk) || tn != n || tk != k) return 4;
  if (in_len < int(64 + 4ull * m * k + 4ull * m * n) || out_len < int(64 + 4ull * m * n)) return 3;
  const float *xs = reinterpret_cast<const float *>(in + 64), *ss = xs + size_t(m) * k;
  auto *x = static_cast<float *>(memalign(128, 4ull * m * k)), *s = static_cast<float *>(memalign(128, 4ull * m * n)),
       *y = static_cast<float *>(memalign(128, 4ull * m * n));
  std::memcpy(x, xs, 4ull * m * k);
  std::memcpy(s, ss, 4ull * m * n);
  MemF32 X{x, x, 0, {m, k}, {k, 1}}, S{s, s, 0, {m, n}, {n, 1}}, Y{y, y, 0, {m, n}, {n, 1}};
  const Plain plain[] = {_mlir_ciface___hexagon_q8_0_matmul, _mlir_ciface___hexagon_q8_0_matvec_q8};
  const Side side[] = {_mlir_ciface___hexagon_q8_0_matmul_residual, _mlir_ciface___hexagon_q8_0_matmul_product,
                       _mlir_ciface___hexagon_q8_0_matvec_q8_residual, _mlir_ciface___hexagon_q8_0_matvec_q8_product};
  uint64_t best = ~0ull;
  for (int i = 0; i < 6; ++i) {
    const uint64_t t0 = HAP_perf_get_time_us();
    if (mode % 3 == 0) plain[mode / 3](&X, id, &Y);
    else side[(mode / 3) * 2 + mode % 3 - 1](&X, id, &S, &Y);
    const uint64_t t = HAP_perf_get_time_us() - t0;
    if (i && t < best) best = t;
  }
  std::memset(out, 0, 64);
  std::memcpy(out, "QWQ8002\0", 8);
  std::memcpy(out + 16, &best, 8);
  std::memcpy(out + 64, y, 4ull * m * n);
  free(x); free(s); free(y);
  *us = 0;
  return 0;
}
