// SPDX-License-Identifier: BSD-3-Clause
// Token embedding rows (qwen-embedding-rows): Out[0, t, :] = Table[Ids[0, t], :].
// Each id is bounds-checked once (the graph asserted it per element); rows are
// copied unchanged.
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
template <typename T, int N> struct Mem {
  T *allocated, *aligned;
  int64_t offset, sizes[N], strides[N];
};
} // namespace

extern "C" void _mlir_ciface_qwen_embedding_rows_v1(Mem<int64_t, 2> *ids, Mem<float, 2> *table,
                                                    Mem<float, 3> *out) {
  if (!ids || !table || !out) std::abort();
  int64_t tokens = ids->sizes[1], width = table->sizes[1], vocab = table->sizes[0];
  if (ids->sizes[0] != 1 || out->sizes[0] != 1 || out->sizes[1] != tokens || out->sizes[2] != width ||
      table->strides[1] != 1 || out->strides[2] != 1 || width < 1 || tokens < 0)
    std::abort();
  const int64_t *id = ids->aligned + ids->offset;
  const float *rows = table->aligned + table->offset;
  float *destination = out->aligned + out->offset;
  for (int64_t t = 0; t < tokens; ++t) {
    int64_t row = id[t * ids->strides[1]];
    if (row < 0 || row >= vocab) std::abort();
    std::memcpy(destination + t * out->strides[1], rows + row * table->strides[0], size_t(width) * sizeof(float));
  }
}
