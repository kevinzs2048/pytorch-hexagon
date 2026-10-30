// SPDX-License-Identifier: BSD-3-Clause
// Exact byte transfer for the measured 512-byte memcpy hot path.
// The standard memcpy precondition remains: source and destination do not overlap.
#include <cstddef>

#if !__has_builtin(__builtin_memcpy_inline)
#error "An always-inline constant-size byte copy is required"
#endif

extern "C" void *__real_memcpy(void *, const void *, std::size_t);

extern "C" __attribute__((visibility("default"), used))
void *__wrap_memcpy(void *destination, const void *source, std::size_t bytes) {
  if (bytes == 512) {
    // LLVM guarantees that this intrinsic does not become another memcpy call.
    // No alignment assumption or floating-point load/arithmetic is introduced.
    __builtin_memcpy_inline(destination, source, 512);
    return destination;
  }
  return __real_memcpy(destination, source, bytes);
}
