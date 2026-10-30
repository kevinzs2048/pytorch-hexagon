// Q9075/Linux DSP shim: the cDSP libc lacks C11 aligned_alloc, which libc++'s
// std::vector operator-new path calls. Without it, any std::vector use crashes
// (e.g. the ThreadManager-based multi-program-grid wrapper). Provide it as a
// memalign wrapper. WEAK so it never clashes with the identical definition that
// libhexagon_mlir_async_runtime.a (HexagonCustomDefs) already ships.
#include <cstddef>
extern "C" void *memalign(size_t, size_t);
extern "C" __attribute__((weak)) void *aligned_alloc(size_t alignment,
                                                     size_t size) {
  return memalign(alignment, size);
}
