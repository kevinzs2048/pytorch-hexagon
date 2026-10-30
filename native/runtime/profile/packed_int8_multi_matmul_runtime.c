/* Explicit integer-only ABI for the existing bounded multi-token graph. */
#include <stdint.h>
#include <stdlib.h>
#include "exact_int8_multi_matmul.h"

struct qwen_multi_i8_memref2 { int8_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };
struct qwen_multi_i32_memref2 { int32_t *allocated,*aligned; int64_t offset,sizes[2],strides[2]; };

static int valid_span(int64_t offset,int64_t rows,int64_t columns,const int64_t strides[2]) {
  if (offset<INT32_MIN || offset>INT32_MAX || rows<1 || columns<1
      || strides[0]<INT32_MIN || strides[0]>INT32_MAX
      || strides[1]<INT32_MIN || strides[1]>INT32_MAX) return 0;
  /* All dimension bounds are checked first. Products fit signed int64 even
   * on native DSP32; their final pointer-index range is checked separately.
   */
  int64_t row=(rows-1)*strides[0],column=(columns-1)*strides[1];
  int64_t lo=offset+(row<0?row:0)+(column<0?column:0);
  int64_t hi=offset+(row>0?row:0)+(column>0?column:0);
  return lo>=INT32_MIN && hi<=INT32_MAX;
}

#ifdef QWEN_RECORD_PROBE_CALLS
static volatile uint64_t qwen_multi_probe_calls;
uint64_t qwen_multi_packed_probe_call_count(void) { return qwen_multi_probe_calls; }
#endif

void _mlir_ciface___qwen_i8_multi_matmul(struct qwen_multi_i8_memref2 *a,
    struct qwen_multi_i8_memref2 *b,struct qwen_multi_i32_memref2 *c) {
#ifdef QWEN_RECORD_PROBE_CALLS
  ++qwen_multi_probe_calls;
#endif
  if (!a || !b || !c || !a->aligned || !b->aligned || !c->aligned
      || a->sizes[0]<1 || a->sizes[0]>32 || c->sizes[0]!=a->sizes[0]
      || a->sizes[1]<1 || a->sizes[1]>131071 || a->sizes[1]!=b->sizes[0]
      || c->sizes[1]<1 || c->sizes[1]>151936 || b->sizes[1]!=c->sizes[1]
      || !valid_span(a->offset,a->sizes[0],a->sizes[1],a->strides)
      || !valid_span(b->offset,b->sizes[0],b->sizes[1],b->strides)
      || !valid_span(c->offset,c->sizes[0],c->sizes[1],c->strides)) abort();
  if (qwen_exact_int8_multi_matmul(a->aligned+a->offset,b->aligned+b->offset,
      c->aligned+c->offset,(size_t)a->sizes[0],(size_t)a->sizes[1],(size_t)c->sizes[1],
      (ptrdiff_t)a->strides[0],(ptrdiff_t)a->strides[1],(ptrdiff_t)b->strides[0],
      (ptrdiff_t)b->strides[1],(ptrdiff_t)c->strides[0],(ptrdiff_t)c->strides[1])) abort();
}
