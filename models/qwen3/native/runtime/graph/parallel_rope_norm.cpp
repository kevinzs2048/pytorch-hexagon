// SPDX-License-Identifier: BSD-3-Clause
// Row-parallel driver for the per-head RMSNorm + rotary embedding kernel
// (Triton qwen_rope_heads_norm). Numerical work stays in the Triton kernel; this
// file validates layouts and partitions (head, row) units like parallel_rope.cpp.
#include <cstdint>
#include <cstdlib>

namespace {
struct Flat {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
struct MemF32x4 {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
struct MemF32x1 {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
struct Job {
  float *y, *w, *cos, *sin, *out;
  int rows, y_row, y_head, c_row, o_head, o_row;
  float eps;
};
} // namespace

extern "C" void qwen_rope_heads_norm(int64_t, void *, int64_t, void *, int64_t, void *, int64_t, void *,
                                     int64_t, void *, int, int, int, int, int, int, int, int, float,
                                     int, int, int, int, int, int);
extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t), void *, uint64_t);

// Y[1, rows, heads, 128] (any strides, unit inner), W[128], Cos/Sin[1, 1, rows, 128],
// Out[1, heads, rows, 128] contiguous.
extern "C" void _mlir_ciface_qwen_rope_heads_norm_v1(MemF32x4 *y, MemF32x1 *w, MemF32x4 *cos,
                                                     MemF32x4 *sin, MemF32x4 *out, float eps) {
  if (!y || !w || !cos || !sin || !out) std::abort();
  int64_t rows = y->sizes[1], heads = y->sizes[2];
  if (y->sizes[0] != 1 || y->sizes[3] != 128 || y->strides[3] != 1 ||
      w->sizes[0] != 128 || w->strides[0] != 1 ||
      cos->sizes[0] != 1 || cos->sizes[1] != 1 || cos->sizes[2] != rows ||
      cos->sizes[3] != 128 || cos->strides[3] != 1 ||
      sin->sizes[0] != 1 || sin->sizes[1] != 1 || sin->sizes[2] != rows ||
      sin->sizes[3] != 128 || sin->strides[3] != 1 || cos->strides[2] != sin->strides[2] ||
      out->sizes[0] != 1 || out->sizes[1] != heads || out->sizes[2] != rows ||
      out->sizes[3] != 128 || out->strides[3] != 1 || out->strides[2] != 128 ||
      out->strides[1] != rows * 128 || rows < 1 || heads < 1)
    std::abort();
  Job job{y->aligned + y->offset, w->aligned + w->offset, cos->aligned + cos->offset,
          sin->aligned + sin->offset, out->aligned + out->offset, int(rows), int(y->strides[1]),
          int(y->strides[2]), int(cos->strides[2]), int(out->strides[1]), int(out->strides[2]), eps};
  auto body = [](void *opaque, uint64_t start, uint64_t count) {
    auto &j = *static_cast<Job *>(opaque);
    Flat y{j.y, j.y, 0, 1, 1}, w{j.w, j.w, 0, 1, 1}, c{j.cos, j.cos, 0, 1, 1},
        s{j.sin, j.sin, 0, 1, 1}, o{j.out, j.out, 0, 1, 1};
    qwen_rope_heads_norm(1, &y, 1, &w, 1, &c, 1, &s, 1, &o, int(start), int(count), j.rows,
                         j.y_row, j.y_head, j.c_row, j.o_head, j.o_row, j.eps, 1, 1, 1, 0, 0, 0);
  };
  uint64_t units = uint64_t(rows * heads);
  if (units < 64)
    body(&job, 0, units);
  else
    __hexagon_parallel_for_v1(body, &job, units);
}
