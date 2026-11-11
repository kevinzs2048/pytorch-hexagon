// SPDX-License-Identifier: BSD-3-Clause
// Row-parallel driver for the additive attention mask (Triton
// qwen_causal_mask). It only validates layouts and partitions query rows.
#include <cstdint>
#include <cstdlib>

namespace {
struct Flat {
  void *allocated, *aligned;
  int64_t offset, size, stride;
};
template <typename T, int N> struct Mem {
  T *allocated, *aligned;
  int64_t offset, sizes[N], strides[N];
};
struct Job {
  int64_t *pos, *valid;
  float *out;
  int kv, pos_step;
};
} // namespace

extern "C" void qwen_causal_mask(int64_t, void *, int64_t, void *, int64_t, void *, int, int,
                                 int, int, int, int, int, int, int, int);
extern "C" void __hexagon_parallel_for_v1(void (*)(void *, uint64_t, uint64_t),
                                          void *, uint64_t);

// Pos[n] with n == rows or 1; Valid[1, width >= kv]; Out[1, 1, rows, kv] contiguous.
extern "C" void _mlir_ciface_qwen_causal_mask_v1(Mem<int64_t, 1> *pos, Mem<int64_t, 2> *valid,
                                                 Mem<float, 4> *out) {
  if (!pos || !valid || !out) std::abort();
  int64_t rows = out->sizes[2], kv = out->sizes[3];
  if (out->sizes[0] != 1 || out->sizes[1] != 1 || out->strides[3] != 1 ||
      (rows > 1 && out->strides[2] != kv) || rows < 1 || kv < 1 || kv > (1 << 30) ||
      (pos->sizes[0] != rows && pos->sizes[0] != 1) || pos->strides[0] != 1 ||
      valid->sizes[0] != 1 || valid->sizes[1] < kv || valid->strides[1] != 1)
    std::abort();
  Job job{pos->aligned + pos->offset, valid->aligned + valid->offset, out->aligned + out->offset,
          int(kv), pos->sizes[0] == 1 ? 0 : 1};
  auto body = [](void *opaque, uint64_t start, uint64_t count) {
    auto &j = *static_cast<Job *>(opaque);
    Flat p{j.pos, j.pos, 0, 1, 1}, v{j.valid, j.valid, 0, 1, 1}, o{j.out, j.out, 0, 1, 1};
    qwen_causal_mask(1, &p, 1, &v, 1, &o, int(start), int(count), j.kv, j.pos_step,
                     1, 1, 1, 0, 0, 0);
  };
  // Decode has one query row: run it on the calling thread.
  if (rows < 4)
    body(&job, 0, uint64_t(rows));
  else
    __hexagon_parallel_for_v1(body, &job, uint64_t(rows));
}
