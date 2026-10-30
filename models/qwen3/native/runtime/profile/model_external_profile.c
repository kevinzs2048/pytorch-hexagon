/* Diagnostic delegation only: every callee and FP/INT return is unchanged.
 * The existing top-level tracking gate excludes nested calls from double counts.
 */
#include "model_memory_profile.c"

extern float __real___hexagon_sqrtf(float);
extern float __real_expf(float);
extern float __real__mlir_ciface___hexagon_silu_exp_compat(float);
extern float __real__mlir_ciface___hexagon_softmax_exp_compat(float);
extern float __real__mlir_ciface___hexagon_arm_sin4_lane(float,float,float,float,int32_t);
extern float __real__mlir_ciface___hexagon_arm_cos4_lane(float,float,float,float,int32_t);
#if defined(QWEN_PROFILE_MULTI_INT8)
extern void __real__mlir_ciface___qwen_i8_multi_matmul(void *,void *,void *);
#elif !defined(QWEN_PROFILE_NO_PACKED_INT8)
extern void __real__mlir_ciface___qwen_i8_matmul(void *,void *,void *);
#endif
extern int64_t __real___hexagon_divdi3(int64_t,int64_t);
extern uint64_t __real___hexagon_udivdi3(uint64_t,uint64_t);

#define SCALAR_WRAPPER(symbol,kind) \
__attribute__((noinline)) float __wrap_##symbol(float x) { \
  size_t n=0;START();float result=__real_##symbol(x);STOP(kind);return result; }
SCALAR_WRAPPER(__hexagon_sqrtf,4)
SCALAR_WRAPPER(expf,5)
SCALAR_WRAPPER(_mlir_ciface___hexagon_silu_exp_compat,6)
SCALAR_WRAPPER(_mlir_ciface___hexagon_softmax_exp_compat,7)
#define LANE_WRAPPER(symbol,kind) \
__attribute__((noinline)) float __wrap_##symbol(float a,float b,float c,float d,int32_t lane) { \
  size_t n=0;START();float result=__real_##symbol(a,b,c,d,lane);STOP(kind);return result; }
LANE_WRAPPER(_mlir_ciface___hexagon_arm_sin4_lane,8)
LANE_WRAPPER(_mlir_ciface___hexagon_arm_cos4_lane,9)
#if defined(QWEN_PROFILE_MULTI_INT8)
__attribute__((noinline)) void __wrap__mlir_ciface___qwen_i8_multi_matmul(void *a,void *b,void *c) {
  size_t n=0;START();__real__mlir_ciface___qwen_i8_multi_matmul(a,b,c);STOP(10);
}
#elif !defined(QWEN_PROFILE_NO_PACKED_INT8)
__attribute__((noinline)) void __wrap__mlir_ciface___qwen_i8_matmul(void *a,void *b,void *c) {
  size_t n=0;START();__real__mlir_ciface___qwen_i8_matmul(a,b,c);STOP(10);
}
#endif
__attribute__((noinline)) int64_t __wrap___hexagon_divdi3(int64_t a,int64_t b) {
  size_t n=0;START();int64_t result=__real___hexagon_divdi3(a,b);STOP(11);return result;
}
__attribute__((noinline)) uint64_t __wrap___hexagon_udivdi3(uint64_t a,uint64_t b) {
  size_t n=0;START();uint64_t result=__real___hexagon_udivdi3(a,b);STOP(12);return result;
}

#ifdef QWEN_PROFILE_ALLOCATIONS
#include <errno.h>
extern void *__real_malloc(size_t);
extern void *__real_calloc(size_t,size_t);
extern int __real_posix_memalign(void **,size_t,size_t);
extern void __real_free(void *);
/* Keep these rows aggregate: the existing model uses 3764 of 4096 slots.
 * Incoming and resulting errno must survive all diagnostic timer calls.
 */
#define ALLOCATION_START() \
  int incoming_errno=errno; START(); errno=incoming_errno
#define ALLOCATION_STOP(kind) \
  int result_errno=errno; \
  if(tracked){uint64_t cycles=HAP_perf_get_pcycles()-start_cycles; \
    uint64_t elapsed=HAP_perf_get_time_us()-start_us; \
    record(kind,0,n,elapsed,cycles);inside=0;} \
  errno=result_errno
__attribute__((noinline,visibility("hidden"))) void *__wrap_malloc(size_t n) {
  ALLOCATION_START();void *result=__real_malloc(n);ALLOCATION_STOP(13);return result;
}
__attribute__((noinline,visibility("hidden"))) void *__wrap_calloc(size_t count,size_t width) {
  size_t n;int size_overflow=__builtin_mul_overflow(count,width,&n);
  if(size_overflow)n=0;
  ALLOCATION_START();void *result=__real_calloc(count,width);
  if(tracked && size_overflow)++overflow;
  ALLOCATION_STOP(14);return result;
}
__attribute__((noinline,visibility("hidden"))) int __wrap_posix_memalign(void **out,size_t alignment,size_t n) {
  ALLOCATION_START();int result=__real_posix_memalign(out,alignment,n);
  ALLOCATION_STOP(15);return result;
}
__attribute__((noinline,visibility("hidden"))) void __wrap_free(void *pointer) {
  size_t n=0;ALLOCATION_START();__real_free(pointer);ALLOCATION_STOP(16);
}
#endif
