// SPDX-License-Identifier: BSD-3-Clause
// FP16 resident KV append (qwen-fp16-kv-storage): rounds new FP32 K or V rows
// into the FP16 plane view with the HVX IEEE conversion of kv_fp16.cpp, which is
// bitwise equal to the scalar round-to-nearest-even arith.truncf it replaces.
// The graph compiler emits that truncf as scalar code (no -mhvx-ieee-fp).
// Prefill rows are split over the worker pool (one 128-value head row each),
// which hides the DDR write latency of the plane; decode's 8 rows stay serial.
#include <cstdint>
#include <cstdlib>
#include "kv_fp16.h"

namespace {
template <typename T> struct Mem {
  T *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
struct Job {
  const float *source;
  uint16_t *destination;
  int64_t tokens, source_head, source_token, destination_head, destination_token;
};
} // namespace

extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t), void *, uint64_t);

// Rows[1, 8, T, 128] FP32 into Plane[1, 8, T, 128] FP16, unit inner strides.
extern "C" void _mlir_ciface_qwen_kv_append_f16_v1(Mem<float> *rows, Mem<uint16_t> *plane) {
  if (!rows || !plane) std::abort();
  int64_t tokens = rows->sizes[2];
  if (rows->sizes[0] != 1 || rows->sizes[1] != 8 || rows->sizes[3] != 128 ||
      plane->sizes[0] != 1 || plane->sizes[1] != 8 || plane->sizes[3] != 128 ||
      plane->sizes[2] != tokens || tokens < 0 || rows->strides[3] != 1 || plane->strides[3] != 1)
    std::abort();
  if (tokens == 0) return;
  Job job{rows->aligned + rows->offset, plane->aligned + plane->offset, tokens,
          rows->strides[1], rows->strides[2], plane->strides[1], plane->strides[2]};
  auto body = [](void *opaque, uint64_t start, uint64_t count) {
    auto &j = *static_cast<Job *>(opaque);
    for (uint64_t unit = start; unit < start + count; ++unit) {
      int64_t head = int64_t(unit) / j.tokens, token = int64_t(unit) % j.tokens;
      pytorch_hexagon::kv_round_f16(j.destination + head * j.destination_head + token * j.destination_token,
                                    j.source + head * j.source_head + token * j.source_token, 128);
    }
  };
  uint64_t units = uint64_t(8 * tokens);
  if (units < 64)
    body(&job, 0, units);
  else
    __hexagon_parallel_for_v1(body, &job, units);
}
