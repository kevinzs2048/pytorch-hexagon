// SPDX-License-Identifier: BSD-3-Clause
// HVX IEEE conversion is confined to this file: the compiler exposes no macro
// for -mhvx-ieee-fp, so the build defines PYTORCH_HEXAGON_KV_FP16_HVX with it.
// Lane order: vdeal(vcvt(lo, hi)) narrows and vcvt(vshuff(h)) widens; both
// match the scalar conversion bit for bit (hexagon-sim, 1.28M values).
#include "kv_fp16.h"
#ifdef PYTORCH_HEXAGON_KV_FP16_HVX
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

namespace pytorch_hexagon {
void kv_round_f16(uint16_t *destination, const float *source, size_t count) {
  size_t i = 0;
#ifdef PYTORCH_HEXAGON_KV_FP16_HVX
  for (; i + 64 <= count; i += 64) {
    HVX_Vector low = *reinterpret_cast<const HVX_UVector *>(source + i);
    HVX_Vector high = *reinterpret_cast<const HVX_UVector *>(source + i + 32);
    *reinterpret_cast<HVX_UVector *>(destination + i) =
        Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(low, high));
  }
  if (i < count) { // Pad the tail into one vector: no scalar half libcalls.
    float wide[64] __attribute__((aligned(128))) = {};
    uint16_t half[64] __attribute__((aligned(128)));
    std::memcpy(wide, source + i, (count - i) * sizeof(float));
    *reinterpret_cast<HVX_Vector *>(half) = Q6_Vh_vdeal_Vh(Q6_Vhf_vcvt_VsfVsf(
        *reinterpret_cast<const HVX_Vector *>(wide), *reinterpret_cast<const HVX_Vector *>(wide + 32)));
    std::memcpy(destination + i, half, (count - i) * sizeof(uint16_t));
  }
#else
  for (; i < count; ++i) destination[i] = kv_round_f16(source[i]);
#endif
}

void kv_widen_f16(float *destination, const uint16_t *source, size_t count) {
  size_t i = 0;
#ifdef PYTORCH_HEXAGON_KV_FP16_HVX
  for (; i + 64 <= count; i += 64) {
    HVX_VectorPair wide = Q6_Wsf_vcvt_Vhf(
        Q6_Vh_vshuff_Vh(*reinterpret_cast<const HVX_UVector *>(source + i)));
    *reinterpret_cast<HVX_UVector *>(destination + i) = Q6_V_lo_W(wide);
    *reinterpret_cast<HVX_UVector *>(destination + i + 32) = Q6_V_hi_W(wide);
  }
  if (i < count) {
    uint16_t half[64] __attribute__((aligned(128))) = {};
    float wide[64] __attribute__((aligned(128)));
    std::memcpy(half, source + i, (count - i) * sizeof(uint16_t));
    HVX_VectorPair pair = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(*reinterpret_cast<const HVX_Vector *>(half)));
    *reinterpret_cast<HVX_Vector *>(wide) = Q6_V_lo_W(pair);
    *reinterpret_cast<HVX_Vector *>(wide + 32) = Q6_V_hi_W(pair);
    std::memcpy(destination + i, wide, (count - i) * sizeof(float));
  }
#else
  for (; i < count; ++i) destination[i] = kv_widen_f16(source[i]);
#endif
}
} // namespace pytorch_hexagon
