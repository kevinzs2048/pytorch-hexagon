/* Diagnostic link wrappers. Original model instructions and callees are kept.
 * Only the caller's model invocation is counted; no power votes or FP changes.
 * The accepted model entry is single-threaded. This is not a production runtime.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "HAP_perf.h"

#define SLOTS 4096u
struct row {
  uint32_t kind, return_address;
  uint64_t size, calls, bytes, microseconds, cycles;
};
_Static_assert(sizeof(struct row)==48,"Profile packet ABI changed");
extern char model_entry[] __asm__("_mlir_ciface_Qwen3DynamicBoundProbe");
static struct row rows[SLOTS];
static uint64_t overflow, invocations;
/* Link wrapping is invisible to the compiler's libc alias analysis. */
static volatile int enabled, inside;
#ifndef QWEN_PROFILE_MEMCPY_TARGET
#define QWEN_PROFILE_MEMCPY_TARGET __real_memcpy
#endif
extern void *QWEN_PROFILE_MEMCPY_TARGET(void *, const void *, size_t);
extern void *__real_memset(void *, int, size_t);
/* HexagonSelectionDAGInfo.cpp declares this aligned helper's return as void. */
extern void __real___hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(
    void *, const void *, size_t);

void qwen_memory_profile_begin(void) {
  enabled=0;
  memset(rows,0,sizeof(rows));
  overflow=0; ++invocations; inside=0; enabled=1;
}
void qwen_memory_profile_end(void) {enabled=0;}

static void record(uint32_t kind, uintptr_t site, size_t bytes,
    uint64_t elapsed, uint64_t cycles) {
  uint32_t slot=(uint32_t)((site>>2) ^ bytes ^ (bytes>>12) ^ (kind*2654435761u)) & (SLOTS-1);
  for (unsigned step=0;step<SLOTS;++step,slot=(slot+1)&(SLOTS-1)) {
    struct row *r=&rows[slot];
    if (!r->kind || (r->kind==kind && r->return_address==(uint32_t)site && r->size==bytes)) {
      r->kind=kind;r->return_address=(uint32_t)site;r->size=bytes;
      ++r->calls;r->bytes+=bytes;r->microseconds+=elapsed;r->cycles+=cycles;
      return;
    }
  }
  ++overflow;
}
#define START() \
  int tracked=enabled && !inside; \
  uint64_t start_us=0,start_cycles=0; \
  if(tracked){inside=1;start_us=HAP_perf_get_time_us();start_cycles=HAP_perf_get_pcycles();}
#define STOP(kind) \
  if(tracked){uint64_t cycles=HAP_perf_get_pcycles()-start_cycles; \
    uint64_t elapsed=HAP_perf_get_time_us()-start_us; \
    record(kind,(uintptr_t)__builtin_return_address(0),n,elapsed,cycles);inside=0;}

__attribute__((noinline)) void *__wrap_memcpy(void *d,const void *s,size_t n) {
  START();void *result=QWEN_PROFILE_MEMCPY_TARGET(d,s,n);STOP(1);return result;
}
__attribute__((noinline)) void *__wrap_memset(void *d,int c,size_t n) {
  START();void *result=__real_memset(d,c,n);STOP(2);return result;
}
__attribute__((noinline)) void
__wrap___hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(void *d,const void *s,size_t n) {
  START();__real___hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(d,s,n);STOP(3);
}

/* Extra diagnostic RPC: it never executes the model and cannot be a model input.
 * The normal model packet remains byte-for-byte unchanged. */
int qwen_memory_profile_query(const unsigned char *input,int input_len,
    unsigned char *output,int output_len,uint64_t *kernel_us) {
  if (!input || input_len<16 || memcmp(input,"QWMEM01\0",8)) return -1;
  if (!output || !kernel_us || output_len<(int)(128+sizeof(rows))) return 2;
  enabled=0; memset(output,0,(size_t)output_len);*kernel_us=0;
  memcpy(output,"QWMEM01\0",8);
  uint32_t version=1, slots=SLOTS, row_bytes=sizeof(struct row);
  memcpy(output+8,&version,4);memcpy(output+12,&slots,4);
  memcpy(output+16,&overflow,8);memcpy(output+24,&invocations,8);
  memcpy(output+32,&row_bytes,4);memcpy(output+128,rows,sizeof(rows));
  uint32_t address=(uint32_t)(uintptr_t)model_entry;
  memcpy(output+36,&address,4);
  return 0;
}
