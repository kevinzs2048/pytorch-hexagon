/* SPDX-License-Identifier: BSD-3-Clause
 * Exact caller byte transfer, retaining owned input/output buffers.
 * Standard memcpy precondition: both byte ranges are valid and do not overlap.
 * No allocation, floating-point operation, input mutation or padding access.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef QWEN_COPY_HOST_TEST
typedef unsigned char HVX_Vector __attribute__((vector_size(128), aligned(128)));
typedef unsigned char HVX_UVector __attribute__((vector_size(128), aligned(4)));
#else
#include <hexagon_types.h>
#if !defined(__HVX__) || __HVX_LENGTH__ != 128
#error "128-byte HVX target required"
#endif
#endif

__attribute__((visibility("hidden"), noinline))
void *qwen_caller_byte_copy(void *destination, const void *source, size_t bytes) {
  unsigned char *d = (unsigned char *)destination;
  const unsigned char *s = (const unsigned char *)source;
  /* The SDK's unaligned vector type still requires four-byte alignment.
   * Equal low two bits guarantee s is aligned to four after aligning d to128.
   * Preserve the existing memcpy/fixed512 path for all other cases.
   */
  if (bytes < 1024 || (((uintptr_t)d ^ (uintptr_t)s) & 3u))
    return memcpy(destination, source, bytes);
  while (((uintptr_t)d & 127u) && bytes) {
    *(volatile unsigned char *)d++ = *(const volatile unsigned char *)s++;
    --bytes;
  }
  /* Every iteration reads/writes exactly128 bytes in the remaining ranges.
   * s is four-byte aligned; d is128-byte aligned. Never round the byte count.
   */
  while (bytes >= 128) {
    *(HVX_Vector *)d = *(const HVX_UVector *)s;
    d += 128; s += 128; bytes -= 128;
  }
  while (bytes) {
    *(volatile unsigned char *)d++ = *(const volatile unsigned char *)s++;
    --bytes;
  }
  return destination;
}
