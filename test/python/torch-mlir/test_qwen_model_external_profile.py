"""Verify diagnostic wrappers preserve actual C ABI bits and nested-call scope."""
from qwen3.compiler_dependency import compiler_root
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[3]


def test_native_external_wrappers_preserve_FP_bits_lane_arguments_and_integer_results(tmp_path):
    gcc=shutil.which('gcc')
    if not gcc:pytest.skip('Native C compiler unavailable')
    (tmp_path/'HAP_perf.h').write_text('''#include <stdint.h>
static inline uint64_t HAP_perf_get_time_us(void){static uint64_t t;return ++t;}
static inline uint64_t HAP_perf_get_pcycles(void){static uint64_t t;return ++t;}
''')
    declarations='''#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
extern float __hexagon_sqrtf(float),expf(float);
extern float _mlir_ciface___hexagon_silu_exp_compat(float),_mlir_ciface___hexagon_softmax_exp_compat(float);
extern float _mlir_ciface___hexagon_arm_sin4_lane(float,float,float,float,int32_t);
extern float _mlir_ciface___hexagon_arm_cos4_lane(float,float,float,float,int32_t);
extern void _mlir_ciface___qwen_i8_matmul(void *,void *,void *);
extern void __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(void *,const void *,size_t);
extern int64_t __hexagon_divdi3(int64_t,int64_t);
extern uint64_t __hexagon_udivdi3(uint64_t,uint64_t);
'''
    (tmp_path/'helpers.c').write_text(declarations+'''
float __hexagon_sqrtf(float x){unsigned char scratch[4];memcpy(scratch,&x,4);return x;}
float expf(float x){return x;}
float _mlir_ciface___hexagon_silu_exp_compat(float x){return x;}
float _mlir_ciface___hexagon_softmax_exp_compat(float x){return x;}
float _mlir_ciface___hexagon_arm_sin4_lane(float a,float b,float c,float d,int32_t lane){float x[]={a,b,c,d};return x[lane];}
float _mlir_ciface___hexagon_arm_cos4_lane(float a,float b,float c,float d,int32_t lane){float x[]={a,b,c,d};return x[3-lane];}
void _mlir_ciface___qwen_i8_matmul(void *a,void *b,void *c){*(uint64_t *)c=*(uint64_t *)a+*(uint64_t *)b;}
void __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(void *d,const void *s,size_t n){memcpy(d,s,n);}
int64_t __hexagon_divdi3(int64_t a,int64_t b){return a/b;}
uint64_t __hexagon_udivdi3(uint64_t a,uint64_t b){return a/b;}
''')
    (tmp_path/'harness.c').write_text(declarations+r'''
char _mlir_ciface_Qwen3DynamicBoundProbe;
extern void qwen_memory_profile_begin(void),qwen_memory_profile_end(void);
extern int qwen_memory_profile_query(const unsigned char *,int,unsigned char *,int,uint64_t *);
struct row{uint32_t kind,site;uint64_t size,calls,bytes,us,cycles;};
static uint32_t bits(float x){uint32_t r;memcpy(&r,&x,4);return r;}
int main(void){
 unsigned char query[16]="QWMEM01",*out=malloc(128+4096*48);uint64_t elapsed;
 uint32_t values[]={0,0x80000000,0x7fc12345,0xffa54321};float x[4];memcpy(x,values,16);
 uint64_t a=UINT64_MAX-10,b=7,c=0;
 qwen_memory_profile_begin();
 for(unsigned i=0;i<4;++i){
   if(bits(__hexagon_sqrtf(x[i]))!=values[i] || bits(expf(x[i]))!=values[i] ||
      bits(_mlir_ciface___hexagon_silu_exp_compat(x[i]))!=values[i] ||
      bits(_mlir_ciface___hexagon_softmax_exp_compat(x[i]))!=values[i])return 1;
   if(bits(_mlir_ciface___hexagon_arm_sin4_lane(x[0],x[1],x[2],x[3],i))!=values[i] ||
      bits(_mlir_ciface___hexagon_arm_cos4_lane(x[0],x[1],x[2],x[3],i))!=values[3-i])return 2;
 }
 _mlir_ciface___qwen_i8_matmul(&a,&b,&c);if(c!=UINT64_MAX-3)return 3;
 if(__hexagon_divdi3(INT64_MIN,7)!=INT64_MIN/7 ||
    __hexagon_divdi3(INT64_MAX,-3)!=INT64_MAX/-3 ||
    __hexagon_udivdi3(UINT64_MAX,7)!=UINT64_MAX/7 ||
    __hexagon_udivdi3(UINT64_MAX,UINT64_MAX)!=1)return 4;
 qwen_memory_profile_end();__hexagon_sqrtf(x[0]);
 if(qwen_memory_profile_query(query,16,out,128+4096*48,&elapsed))return 5;
 uint64_t counts[13]={0};struct row *rows=(struct row *)(out+128);
 for(unsigned i=0;i<4096;++i){if(rows[i].kind>12)return 6;counts[rows[i].kind]+=rows[i].calls;}
 for(unsigned kind=4;kind<=9;++kind)if(counts[kind]!=4)return 7;
 if(counts[10]!=1 || counts[11]!=2 || counts[12]!=2)return 8;
 /* bits() invokes memcpy at top level; copies inside sqrt delegate are excluded. */
 if(counts[1]!=24 || counts[2] || counts[3])return 9;
 free(out);return 0;
}
''')
    symbols=['memcpy','memset','__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes',
        '__hexagon_sqrtf','expf','_mlir_ciface___hexagon_silu_exp_compat',
        '_mlir_ciface___hexagon_softmax_exp_compat','_mlir_ciface___hexagon_arm_sin4_lane',
        '_mlir_ciface___hexagon_arm_cos4_lane','_mlir_ciface___qwen_i8_matmul',
        '__hexagon_divdi3','__hexagon_udivdi3']
    exe=tmp_path/'check'
    subprocess.run([gcc,'-O2','-fno-builtin','-U_FORTIFY_SOURCE','-D_FORTIFY_SOURCE=0',
        '-I',str(tmp_path),str(compiler_root()/'tools/qwen3/runtime/profile/model_external_profile.c'),
        str(tmp_path/'harness.c'),str(tmp_path/'helpers.c'),
        *['-Wl,--wrap='+s for s in symbols],'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
