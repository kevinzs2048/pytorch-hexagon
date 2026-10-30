/* Diagnostic timeline around the unchanged, single-threaded INT8 callee.
 * No model, parameter, FP operation, allocation or buffer ownership is changed.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include "HAP_perf.h"

#define TIMELINE_SLOTS 256u
struct timeline_row {
  uint32_t ordinal, return_address;
  int64_t m, k, n;
  uint64_t enter_us, leave_us, enter_cycles, leave_cycles;
};
_Static_assert(sizeof(struct timeline_row)==64,"Timeline ABI changed");
struct timeline_memref2 {
  void *allocated,*aligned;
  int64_t offset,sizes[2],strides[2];
};
extern char model_entry[] __asm__("_mlir_ciface_Qwen3DynamicBoundProbe");
extern void __real__mlir_ciface___qwen_i8_multi_matmul(void *,void *,void *);
static struct timeline_row timeline[TIMELINE_SLOTS];
static uint64_t invocations, overflow, begin_us, end_us, begin_cycles, end_cycles;
static uint32_t count;
static volatile int enabled, inside;
#ifdef QWEN_TIMELINE_ATTENTION_PHASES
struct attention_phase {
  uint32_t layer,calls;
  uint64_t first_enter_us,last_leave_us,first_enter_cycles,last_leave_cycles;
  uint64_t expected_calls,m,reserved;
};
_Static_assert(sizeof(struct attention_phase)==64,"Attention timeline ABI changed");
static struct attention_phase phases[28];
extern float __real__mlir_ciface___hexagon_softmax_exp_compat(float);
#endif

void qwen_memory_profile_begin(void) {
  int saved_errno=errno;
  enabled=0;inside=0;count=0;overflow=0;
  memset(timeline,0,sizeof(timeline));
#ifdef QWEN_TIMELINE_ATTENTION_PHASES
  memset(phases,0,sizeof(phases));
#endif
  ++invocations;end_us=0;end_cycles=0;
  begin_us=HAP_perf_get_time_us();begin_cycles=HAP_perf_get_pcycles();
  enabled=1;errno=saved_errno;
}
void qwen_memory_profile_end(void) {
  int saved_errno=errno;
  enabled=0;end_cycles=HAP_perf_get_pcycles();end_us=HAP_perf_get_time_us();
  errno=saved_errno;
}

__attribute__((noinline,visibility("hidden"))) void
__wrap__mlir_ciface___qwen_i8_multi_matmul(void *a,void *b,void *c) {
  int incoming_errno=errno,tracked=enabled && !inside;
  struct timeline_row row={0};
  if(tracked) {
    inside=1;
    const struct timeline_memref2 *left=a,*right=b,*out=c;
    row.ordinal=count;
    row.return_address=(uint32_t)(uintptr_t)__builtin_return_address(0);
    row.m=left->sizes[0];row.k=left->sizes[1];row.n=right->sizes[1];
    if(right->sizes[0]!=row.k || out->sizes[0]!=row.m || out->sizes[1]!=row.n)
      ++overflow;
    row.enter_us=HAP_perf_get_time_us();row.enter_cycles=HAP_perf_get_pcycles();
  }
  errno=incoming_errno;
  __real__mlir_ciface___qwen_i8_multi_matmul(a,b,c);
  int result_errno=errno;
  if(tracked) {
    row.leave_cycles=HAP_perf_get_pcycles();row.leave_us=HAP_perf_get_time_us();
    if(count<TIMELINE_SLOTS) {
      timeline[count++]=row;
#ifdef QWEN_TIMELINE_ATTENTION_PHASES
      if(count<=196 && count%7==3) {
        unsigned layer=(count-3)/7;
        phases[layer].layer=layer;
        phases[layer].m=(uint64_t)row.m;
        phases[layer].expected_calls=(uint64_t)row.m*16u*1024u;
      }
#endif
    }
    else ++overflow;
    inside=0;
  }
  errno=result_errno;
}

#ifdef QWEN_TIMELINE_ATTENTION_PHASES
/* Only the first and last existing exponential of each layer read a timer.
 * Every intervening call still delegates exactly once, in original order.
 */
__attribute__((noinline,visibility("hidden"))) float
__wrap__mlir_ciface___hexagon_softmax_exp_compat(float x) {
  struct attention_phase *phase=NULL;
  int first=0,last=0;
  if(enabled && !inside) {
    if(count>=3 && count<=192 && count%7==3) {
      phase=&phases[(count-3)/7];
      if(!phase->expected_calls || phase->calls>=phase->expected_calls)++overflow;
      else {
        first=phase->calls==0;
        ++phase->calls;
        last=phase->calls==phase->expected_calls;
      }
    }else ++overflow;
  }
  if(first) {
    int incoming_errno=errno;
    phase->first_enter_us=HAP_perf_get_time_us();
    phase->first_enter_cycles=HAP_perf_get_pcycles();
    errno=incoming_errno;
  }
  float result=__real__mlir_ciface___hexagon_softmax_exp_compat(x);
  if(last) {
    int result_errno=errno;
    phase->last_leave_cycles=HAP_perf_get_pcycles();
    phase->last_leave_us=HAP_perf_get_time_us();
    errno=result_errno;
  }
  return result;
}
#endif

int qwen_memory_profile_query(const unsigned char *input,int input_len,
    unsigned char *output,int output_len,uint64_t *kernel_us) {
#ifdef QWEN_TIMELINE_ATTENTION_PHASES
  const char *magic="QWTIM02\0";
  uint32_t version=2,phase_count=28;
  const size_t extra_bytes=sizeof(phases);
#else
  const char *magic="QWTIM01\0";
  uint32_t version=1,phase_count=0;
  const size_t extra_bytes=0;
#endif
  if(!input || input_len<16 || memcmp(input,magic,8))return -1;
  if(input_len!=16 || !output || !kernel_us || enabled || inside || !end_us
      || output_len!=(int)(128+sizeof(timeline)+extra_bytes))return 2;
  for(int i=8;i<16;++i)if(input[i])return 2;
  memset(output,0,(size_t)output_len);*kernel_us=0;
  memcpy(output,magic,8);
  uint32_t slots=TIMELINE_SLOTS,row_bytes=sizeof(struct timeline_row);
  uint32_t address=(uint32_t)(uintptr_t)model_entry;
  memcpy(output+8,&version,4);memcpy(output+12,&slots,4);
  memcpy(output+16,&overflow,8);memcpy(output+24,&invocations,8);
  memcpy(output+32,&row_bytes,4);memcpy(output+36,&address,4);
  memcpy(output+40,&count,4);
  memcpy(output+44,&phase_count,4);
  memcpy(output+48,&begin_us,8);memcpy(output+56,&end_us,8);
  memcpy(output+64,&begin_cycles,8);memcpy(output+72,&end_cycles,8);
  memcpy(output+128,timeline,sizeof(timeline));
#ifdef QWEN_TIMELINE_ATTENTION_PHASES
  memcpy(output+128+sizeof(timeline),phases,sizeof(phases));
#endif
  return 0;
}
