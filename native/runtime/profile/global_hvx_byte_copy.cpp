// Exact global memcpy dispatch, reusing the already accepted HVX byte routine.
// Source and destination must satisfy the standard valid, nonoverlapping ranges.
// The original fixed512 implementation remains the complete fallback.
#include <cstddef>
#include <cstdint>

extern "C" __attribute__((visibility("hidden")))
void *qwen_caller_byte_copy(void *, const void *, std::size_t);
extern "C" __attribute__((visibility("hidden")))
void *qwen_global_copy_preserved_memcpy(void *, const void *, std::size_t);

extern "C" __attribute__((visibility("hidden"), used, noinline))
void *__wrap_memcpy(void *destination, const void *source, std::size_t bytes) {
  // This is exactly the accepted helper's admission test. Its fallback to
  // memcpy therefore cannot recursively re-enter this fast branch.
  if (bytes >= 1024 &&
      !((reinterpret_cast<std::uintptr_t>(destination) ^
         reinterpret_cast<std::uintptr_t>(source)) & 3u))
    return qwen_caller_byte_copy(destination, source, bytes);
  return qwen_global_copy_preserved_memcpy(destination, source, bytes);
}
