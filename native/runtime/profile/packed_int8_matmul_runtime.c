/* Narrow external Linalg library-call ABI. No allocation or FP operations. */
#include <stdint.h>
#include <stdlib.h>
#include "exact_int8_matmul_tile256.h"

struct qwen_i8_memref2 { int8_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };
struct qwen_i32_memref2 { int32_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };

static int fits_index(int64_t value) { return value>=PTRDIFF_MIN && value<=PTRDIFF_MAX; }

#ifdef QWEN_RECORD_PROBE_CALLS
static volatile uint64_t qwen_probe_calls;
uint64_t qwen_packed_probe_call_count(void) { return qwen_probe_calls; }
#endif

void _mlir_ciface___qwen_i8_matmul(struct qwen_i8_memref2 *a,
                                  struct qwen_i8_memref2 *b,
                                  struct qwen_i32_memref2 *c) {
#ifdef QWEN_RECORD_PROBE_CALLS
  ++qwen_probe_calls;
#endif
  /* An invalid descriptor is a failed primitive, never a fabricated result. */
  if (!a || !b || !c || !a->aligned || !b->aligned || !c->aligned
      || a->sizes[0]!=1 || c->sizes[0]!=1 || a->sizes[1]!=b->sizes[0]
      || b->sizes[1]!=c->sizes[1] || a->sizes[1]<1 || a->sizes[1]>131071
      || c->sizes[1]<1 || c->sizes[1]>151936
      || !fits_index(a->offset) || !fits_index(b->offset) || !fits_index(c->offset)
      || !fits_index(a->strides[1]) || !fits_index(b->strides[0])
      || !fits_index(b->strides[1]) || !fits_index(c->strides[1])) abort();
  if (qwen_exact_int8_matmul_tile256(a->aligned+a->offset,b->aligned+b->offset,
      c->aligned+c->offset,(size_t)a->sizes[1],(size_t)c->sizes[1],
      (ptrdiff_t)a->strides[1],(ptrdiff_t)b->strides[0],
      (ptrdiff_t)b->strides[1],(ptrdiff_t)c->strides[1])) abort();
}
