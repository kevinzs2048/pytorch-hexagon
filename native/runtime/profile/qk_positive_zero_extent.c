/* Integer-only guard for the observed q1 QK recurrence. The caller evaluates
 * one ORIGINAL 128-column tile after the returned prefix before copying its
 * completed bit patterns to the remaining identical tiles. Returning 1024
 * means the entire original recurrence must execute. No query is inspected or
 * modified, no floating-point comparison is performed, and -0 is not +0. */
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

struct qk_m2 { void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; };
struct qk_m3 { void *allocated, *aligned; int64_t offset, sizes[3], strides[3]; };
extern bool qwen_zero_words(const void *, size_t);

static const unsigned char *canonical_base(void *pointer, int64_t offset,
                                           size_t words) {
  uintptr_t address = (uintptr_t)pointer;
  /* Exact canonical offset only: no speculative pointer arithmetic on an
   * unsupported descriptor, even when its dimensions happen to agree. */
  if (!pointer || offset != 0 || (address & 3) ||
      words > SIZE_MAX / 4 || words * 4 > UINTPTR_MAX - address)
    return NULL;
  return (const unsigned char *)pointer;
}

int64_t _mlir_ciface_qwen_qk_positive_zero_extent(struct qk_m3 *keys,
                                                struct qk_m2 *seed,
                                                bool broadcast) {
  if (!keys || !seed || broadcast ||
      keys->sizes[0] != 8 || keys->sizes[1] != 128 || keys->sizes[2] != 1024 ||
      keys->strides[0] != 128 * 1024 || keys->strides[1] != 1024 ||
      keys->strides[2] != 1 || seed->sizes[0] != 16 || seed->sizes[1] != 1024 ||
      seed->strides[0] != 1024 || seed->strides[1] != 1)
    return 1024;
  const unsigned char *k = canonical_base(keys->aligned, keys->offset, 8 * 128 * 1024);
  const unsigned char *s = canonical_base(seed->aligned, seed->offset, 16 * 1024);
  if (!k || !s) return 1024;

  /* A single non-+0 bit pattern extends the prefix, including negative zero,
   * subnormals, NaNs and infinities. Scan backwards in exactly the original
   * 128-column tile size. Every checked byte is inside its descriptor extent.
   * High occupancy is cost-ineligible and conservatively falls back. */
  for (int64_t start = 896; start >= 0; start -= 128) {
    for (int64_t head = 0; head < 16; ++head)
      if (!qwen_zero_words(s + 4 * (head * 1024 + start), 128 * 4))
        return start + 128 >= 768 ? 1024 : start + 128;
    for (int64_t row = 0; row < 8 * 128; ++row)
      if (!qwen_zero_words(k + 4 * (row * 1024 + start), 128 * 4))
        return start + 128 >= 768 ? 1024 : start + 128;
  }
  return 0;
}
