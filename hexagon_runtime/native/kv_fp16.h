// SPDX-License-Identifier: BSD-3-Clause
#ifndef PYTORCH_HEXAGON_KV_FP16_H
#define PYTORCH_HEXAGON_KV_FP16_H
// IEEE binary16 KV rows (llama.cpp's default F16 KV cache type). Both helpers
// are bitwise equal to scalar (__fp16) round-to-nearest-even conversion and
// its exact widening, including subnormal, tie and overflow inputs; this is
// also the HVX vcvt conversion the generated graph and attention kernels use.
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pytorch_hexagon {
inline uint16_t kv_round_f16(float value) {
  __fp16 half = static_cast<__fp16>(value);
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

inline float kv_widen_f16(uint16_t bits) {
  __fp16 half;
  std::memcpy(&half, &bits, sizeof(half));
  return static_cast<float>(half);
}

// Row conversions (kv_fp16.cpp). Build that file with -mhvx -mhvx-ieee-fp
// -DPYTORCH_HEXAGON_KV_FP16_HVX for the HVX path; elsewhere it is scalar.
void kv_round_f16(uint16_t *destination, const float *source, size_t count);
void kv_widen_f16(float *destination, const uint16_t *source, size_t count);
} // namespace pytorch_hexagon
#endif
