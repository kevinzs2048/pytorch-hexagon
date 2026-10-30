/* SPDX-License-Identifier: BSD-3-Clause
 * Diagnostic delegation of the existing Q2 integer guard, without inner-call
 * wrappers. The packet uses the established external-profile ABI (kind 13).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "HAP_perf.h"
struct memref3 {void *allocated,*aligned;int64_t offset,sizes[3],strides[3];};
struct row {uint32_t kind,site;uint64_t size,calls,bytes,us,cycles;};
_Static_assert(sizeof(struct row)==48,"profile row ABI");
extern char model_entry[] __asm__("_mlir_ciface_Qwen3DynamicBoundProbe");
extern int64_t __real__mlir_ciface_qwen_q2_qk_positive_zero_extent(struct memref3*,struct memref3*,bool);
static struct row rows[4096];
static uint64_t invocations;
static uint32_t count,overflow;
static int enabled;
void qwen_memory_profile_begin(void){
 int e=errno;enabled=0;memset(rows,0,sizeof(rows));count=0;overflow=0;++invocations;enabled=1;errno=e;
}
void qwen_memory_profile_end(void){enabled=0;}
int64_t __wrap__mlir_ciface_qwen_q2_qk_positive_zero_extent(struct memref3*k,struct memref3*s,bool b){
 int e=errno;uint64_t t=0,c=0;int tracked=enabled;
 if(tracked){t=HAP_perf_get_time_us();c=HAP_perf_get_pcycles();}
 errno=e;int64_t answer=__real__mlir_ciface_qwen_q2_qk_positive_zero_extent(k,s,b);int result_errno=errno;
 if(tracked){uint64_t c2=HAP_perf_get_pcycles(),t2=HAP_perf_get_time_us();
  if(count<4096){rows[count]=(struct row){13,(uint32_t)(uintptr_t)__builtin_return_address(0),0,1,0,t2-t,c2-c};++count;}
  else ++overflow;
 }
 errno=result_errno;return answer;
}
int qwen_memory_profile_query(const unsigned char*in,int il,unsigned char*out,int ol,uint64_t*us){
 if(!in || il<16 || memcmp(in,"QWMEM01\0",8))return -1;
 if(il!=16 || !out || ol!=128+(int)sizeof(rows) || !us || enabled)return 2;
 int e=errno;memset(out,0,(size_t)ol);memcpy(out,"QWMEM01\0",8);
 uint32_t version=1,slots=4096,rowbytes=48,address=(uint32_t)(uintptr_t)model_entry;
 uint64_t overflow64=overflow;
 memcpy(out+8,&version,4);memcpy(out+12,&slots,4);memcpy(out+16,&overflow64,8);
 memcpy(out+24,&invocations,8);memcpy(out+32,&rowbytes,4);memcpy(out+36,&address,4);
 memcpy(out+128,rows,sizeof(rows));*us=0;errno=e;return 0;
}
