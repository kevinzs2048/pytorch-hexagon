"""Host ABI/ownership checks; actual Triton arithmetic is tested on the DSP."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("hmx_decode", [False, True])
def test_prefill_validation_and_job_owned_storage(tmp_path, hmx_decode):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    (tmp_path/'qurt.h').write_text('''
#pragma once
inline thread_local const void *current_pool=nullptr;
inline int qurt_tls_create_key(int *key,void(*)(void*)){*key=0;return 0;}
inline int qurt_tls_delete_key(int){current_pool=nullptr;return 0;}
inline int qurt_tls_set_specific(int,const void*p){current_pool=p;return 0;}
inline void *qurt_tls_get_specific(int){return const_cast<void*>(current_pool);}
''')
    (tmp_path/'worker-pool.h').write_text('''
#pragma once
using worker_pool_context_t=void*;
inline int worker_pool_init_with_stack_size(void**p,unsigned n,unsigned stack){
 *p=(void*)1;return n!=4||stack!=32768;
}
inline void worker_pool_release(void**p){*p=nullptr;}
inline int worker_pool_run_func(void*,void(*f)(unsigned,unsigned,void*),void*p,unsigned n){
 // All logical jobs execute on the same physical thread, in reverse order.
 for(unsigned i=n;i>0;--i)f(n,i-1,p);return 0;
}
''')
    (tmp_path/'check.cpp').write_text(r'''
#include "attention_memref.h"
#include <cassert>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace qwen_attention;
extern "C" int qwen_triton_prefill(const Memref*,const Memref*,const Memref*,const Memref*,Memref*);
extern "C" void *hexagon_kernel_malloc(uint64_t);
extern "C" void hexagon_kernel_free(void*);
extern "C" int hexagon_runtime_configure_vtcm_pool_dsp(unsigned){return 0;}
extern "C" int hexagon_runtime_configure_hmx_performance_dsp(unsigned){return 0;}
extern "C" int _mlir_ciface_qwen_ordered_fp32_attention_v1(
 const Memref*,const Memref*,const Memref*,const Memref*,Memref*);
unsigned calls, hmx_calls, hvx_calls, hmx_head_mask;
extern "C" void attention_forward_q1_hmx_g2(int64_t,void*,int64_t,void*,int64_t,void*,int64_t,void*,
 int m,int n,int,int,int,int,int,int,int gx,int gy,int gz,int tile,int head,int z){
 assert(m==1&&n==4&&gx==1&&gy==4&&gz==1&&tile==0&&head>=0&&head<4&&z==0);
 ++hmx_calls;hmx_head_mask|=1u<<head;
}
extern "C" void attention_decode(int64_t,void*,int64_t,void*,int64_t,void*,int64_t,void*,int64_t,void*,
 int m,int n,int,int,int,int,int,int,int gx,int gy,int gz,int tile,int head,int z){
 assert(m==1&&(n==3||n==4)&&gx==1&&gy==4&&gz==1&&tile==0&&head>=0&&head<4&&z==0);
 ++hvx_calls;
}
unsigned *escaped[8];
extern "C" void attention_forward(int64_t,void*,int64_t,void*,int64_t,void*,int64_t,void*out,
 int m,int n,int qh,int qt,int kh,int kt,int vh,int vt,
 int gx,int gy,int gz,int tile,int head,int z){
 assert(m==2&&n==4&&qh==256&&qt==128&&kh==512&&kt==128&&vh==512&&vt==128);
 assert(gx==1&&gy==8&&gz==1&&tile==0&&head>=0&&head<8&&z==0);
 ++calls;
 auto *temporary=static_cast<unsigned*>(hexagon_kernel_malloc(32));
 assert(temporary);*temporary=77;hexagon_kernel_free(temporary);
 // An allocation returned by an HMX owner outside this pool may be freed here.
 hexagon_kernel_free(std::malloc(32));
 escaped[head]=static_cast<unsigned*>(hexagon_kernel_malloc(sizeof(unsigned)));
 *escaped[head]=head+123;
 static_cast<FlatMemref*>(out)->aligned[head*128]=float(head);
}
Memref descriptor(std::vector<float>&data,int b,int c,int d){
 return {data.data(),data.data(),0,{1,b,c,d},{0,c*d,d,1}};
}
int main(){
 std::vector<float> q(16*2*128),k(8*4*128),v(k.size()),mask(8),out(2*16*128,-99);
 auto Q=descriptor(q,16,2,128),K=descriptor(k,8,4,128),V=descriptor(v,8,4,128);
 auto M=descriptor(mask,1,2,4),O=descriptor(out,2,16,128);
 mask[3]=-std::numeric_limits<float>::infinity();
 mask[0]=-0.0f; // Signed zero is a valid visible entry.
 for(unsigned repeat=0;repeat<3;++repeat){
  assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==0);
  assert(calls==(repeat+1)*8);
  for(unsigned head=0;head<8;++head){assert(*escaped[head]==head+123);hexagon_kernel_free(escaped[head]);}
  mask[3]=std::numeric_limits<float>::lowest();
 }
 auto prior=out;
 for(float invalid : {0.0f,-0.0f,-1.0f,std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}){
  mask[3]=invalid;assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==2);
  assert(out==prior&&calls==24);
 }
 mask[3]=-std::numeric_limits<float>::infinity();
 for(float invalid : {-1.0f,std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
  mask[0]=invalid;assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==2);
  assert(out==prior&&calls==24);
 }
 mask[0]=0;
 V.strides[3]=2;assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==2);V.strides[3]=1;
 Q.offset=-1;assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==2);Q.offset=0;
 assert(qwen_triton_prefill(nullptr,&K,&V,&M,&O)==2);
 assert(out==prior&&calls==24);
 // Public dispatch uses effective length, and q1 reuses the prefill owner only
 // at the selected threshold. Padding/additive masks still fail before launch.
 Q=descriptor(q,16,1,128);O=descriptor(out,1,16,128);
 M=descriptor(mask,1,1,4);mask[3]=0;
 assert(_mlir_ciface_qwen_ordered_fp32_attention_v1(&Q,&K,&V,&M,&O)==0);
#ifdef QWEN_TRITON_HMX_DECODE
 assert(hmx_calls==4&&hvx_calls==0&&hmx_head_mask==15);
#else
 assert(hmx_calls==0&&hvx_calls==4);
 assert(qwen_triton_prefill(&Q,&K,&V,&M,&O)==2);
#endif
 unsigned before=hmx_calls+hvx_calls;
 mask[2]=-1;
 assert(_mlir_ciface_qwen_ordered_fp32_attention_v1(&Q,&K,&V,&M,&O)==2);
 assert(hmx_calls+hvx_calls==before);mask[2]=0;
 K.sizes[2]=V.sizes[2]=M.sizes[3]=3;
 assert(_mlir_ciface_qwen_ordered_fp32_attention_v1(&Q,&K,&V,&M,&O)==0);
 assert(hmx_calls+hvx_calls==before+4);
}
''')
    native = ROOT/'models/qwen3/native/runtime/attention'
    common = ROOT/'hexagon_runtime/native'
    for limit in (0, 1048576):
        binary = tmp_path/f'check-{limit}'
        subprocess.run([compiler, '-std=c++17', '-O1', '-g',
                        '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                        f'-DQWEN_PREFILL_POOL_BYTES={limit}', '-DQWEN_TRITON_PREFILL',
                        *(['-DQWEN_TRITON_HMX_DECODE', '-DQWEN_HMX_DECODE_MIN_KV=4'] if hmx_decode else []),
                        '-I'+str(tmp_path), '-I'+str(native), '-I'+str(common),
                        str(native/'triton_prefill.cpp'), str(native/'triton_decode.cpp'),
                        str(common/'temporary_buffer_pool.cpp'),
                        str(tmp_path/'check.cpp'), '-o', str(binary)],
                       check=True, capture_output=True)
        subprocess.run([str(binary)], check=True, capture_output=True, timeout=15)
