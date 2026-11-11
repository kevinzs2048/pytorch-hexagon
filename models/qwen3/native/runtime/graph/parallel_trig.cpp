// SPDX-License-Identifier: BSD-3-Clause
// Row-parallel driver for the batched ARM-compatible RoPE trig tables. It calls
// the existing four-lane runtime entries (SLEEF xsinf_u1/xcosf_u1, pure and
// reentrant) with exactly the groups the graph loop used, so results are
// bitwise identical; only the rows are split over the runtime worker pool.
#include <cstdint>
#include <cstdlib>

namespace {
struct Group {
  float *allocated, *aligned;
  int64_t offset, size, stride;
};
template <int R> struct Mem {
  float *allocated, *aligned;
  int64_t offset, sizes[R], strides[R];
};
struct Job {
  const float *in;
  float *out;
  int64_t in_row;
  int which;
};
} // namespace

extern "C" void _mlir_ciface___hexagon_arm_sin4(Group *, float, float, float, float);
extern "C" void _mlir_ciface___hexagon_arm_cos4(Group *, float, float, float, float);
extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t),
                                          void *, uint64_t);

// out[0, r, j + k] = out[0, r, 64 + j + k] = f(in[0, r, j .. j + 3])[k] for
// j = 0, 4, ..., 60; f = cos (which == 0) or sin (which == 1).
extern "C" void _mlir_ciface_qwen_arm_trig_rows_v1(Mem<3> *in, Mem<3> *out, int32_t which) {
  if (!in || !out || (which != 0 && which != 1)) std::abort();
  int64_t rows = in->sizes[1];
  if (in->sizes[0] != 1 || in->sizes[2] != 128 || in->strides[2] != 1 || out->sizes[0] != 1 ||
      out->sizes[1] != rows || out->sizes[2] != 128 || out->strides[2] != 1 ||
      out->strides[1] != 128 || rows < 1)
    std::abort();
  Job job{in->aligned + in->offset, out->aligned + out->offset, in->strides[1], which};
  auto body = [](void *opaque, uint64_t start, uint64_t count) {
    auto &j = *static_cast<Job *>(opaque);
    auto f = j.which ? _mlir_ciface___hexagon_arm_sin4 : _mlir_ciface___hexagon_arm_cos4;
    float group[4];
    Group g{group, group, 0, 4, 1};
    for (uint64_t r = start; r < start + count; ++r) {
      const float *x = j.in + r * j.in_row;
      float *y = j.out + r * 128;
      for (int c = 0; c < 64; c += 4) {
        f(&g, x[c], x[c + 1], x[c + 2], x[c + 3]);
        for (int k = 0; k < 4; ++k) y[c + k] = y[64 + c + k] = group[k];
      }
    }
  };
  // Decode has a single row: a worker dispatch would cost more than the work.
  if (rows < 4)
    body(&job, 0, uint64_t(rows));
  else
    __hexagon_parallel_for_v1(body, &job, uint64_t(rows));
}
