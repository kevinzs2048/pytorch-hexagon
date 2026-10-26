"""Diagnostic isolation and native link-wrapper checks, not DSP acceptance."""
from qwen3.compiler_dependency import compiler_root
import importlib.util
from pathlib import Path
import shutil
import subprocess
import struct

import pytest

ROOT=Path(__file__).resolve().parents[3]
SOURCE=compiler_root()/'tools/qwen3/runtime/profile/model_memory_profile.c'
spec=importlib.util.spec_from_file_location('qwen_model_memory_profile',ROOT/'tools/qwen3/profile_model_memory.py')


def test_instrumentation_keeps_model_call_and_normal_packet_guards(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'tools/qwen3'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original='''extern "C" int hexagon_forward_buffers(const unsigned char *input,int input_len,
unsigned char *output,int output_len,uint64_t *kernel_us) {
  if (!input || !output || !kernel_us || input_len < 16) return 2;
  *kernel_us = 0;
  if (memcmp(input,"HXDY",4)) return 2;
  *kernel_us = benchmark_time_us(1, [&]() {
    _mlir_ciface_Qwen3DynamicBoundProbe(r,dt0,dt1,1024,1);
  });
  return 0;
}
'''
    instrumented=module.instrument(original)
    assert instrumented.count('_mlir_ciface_Qwen3DynamicBoundProbe(r,dt0,dt1,1024,1);')==1
    assert instrumented.index('if (!input')<instrumented.index('int diagnostic')
    for removed in ('extern "C" void qwen_memory_profile_begin();\n',
        'extern "C" void qwen_memory_profile_end();\n',
        'extern "C" int qwen_memory_profile_query(const unsigned char *,int,unsigned char *,int,uint64_t *);\n',
        '  int diagnostic = qwen_memory_profile_query(input,input_len,output,output_len,kernel_us);\n',
        '  if (diagnostic >= 0) return diagnostic;\n',
        '    qwen_memory_profile_begin();\n','\n    qwen_memory_profile_end();'):
        instrumented=instrumented.replace(removed,'')
    assert instrumented==original
    with pytest.raises(ValueError):module.instrument(original.replace('1024,1);','1024,1);\n _mlir_ciface_Qwen3DynamicBoundProbe(r);'))


def test_native_wrappers_preserve_bytes_returns_and_tracking_boundary(tmp_path):
    gcc=shutil.which('gcc')
    if not gcc:pytest.skip('Native C compiler unavailable')
    (tmp_path/'HAP_perf.h').write_text('''#include <stdint.h>
static inline uint64_t HAP_perf_get_time_us(void){static uint64_t t;return ++t;}
static inline uint64_t HAP_perf_get_pcycles(void){static uint64_t t;return ++t;}
''')
    harness=tmp_path/'harness.c'
    harness.write_text(r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
char _mlir_ciface_Qwen3DynamicBoundProbe;
extern void __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(void *,const void *,size_t);
extern void qwen_memory_profile_begin(void),qwen_memory_profile_end(void);
extern int qwen_memory_profile_query(const unsigned char *,int,unsigned char *,int,uint64_t *);
struct row{uint32_t kind,site;uint64_t size,calls,bytes,us,cycles;};
int main(void){
  unsigned char a[512],b[512],c[512];
  unsigned char query[16]="QWMEM01",bad[16]="HXDY";
  unsigned char *out=malloc(128+4096*48);uint64_t us=7;
  if(qwen_memory_profile_query(bad,16,out,128+4096*48,&us)!=-1 || us!=7)return 1;
  if(qwen_memory_profile_query(query,16,out,10,&us)!=2)return 2;
  for(int i=0;i<512;++i)a[i]=(unsigned char)(i*79+5);
  qwen_memory_profile_begin();
  unsigned sizes[]={0,1,31,32,33,127,128,129,255};
  uint64_t expected=0;
  for(unsigned j=0;j<9;++j){size_t n=sizes[j];expected+=n;
    if(memcpy(b+3,a+1,n)!=b+3)return 3;
    for(size_t k=0;k<n;++k)if(b[3+k]!=a[1+k])return 4;
    __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(c,a,n);
    for(size_t k=0;k<n;++k)if(c[k]!=a[k])return 5;
    if(memset(b+7,0xA3,n)!=b+7)return 6;
    for(size_t k=0;k<n;++k)if(b[7+k]!=0xA3)return 7;
  }
  qwen_memory_profile_end();memcpy(c,a,512);
  if(qwen_memory_profile_query(query,16,out,128+4096*48,&us) || us)return 8;
  struct row *rows=(struct row *)(out+128);uint64_t calls[4]={0},bytes[4]={0};
  for(unsigned i=0;i<4096;++i){struct row *r=rows+i;if(r->kind>3)return 9;
    calls[r->kind]+=r->calls;bytes[r->kind]+=r->bytes;}
  for(unsigned i=1;i<=3;++i)if(calls[i]!=9 || bytes[i]!=expected){
    fprintf(stderr,"kind=%u calls=%llu bytes=%llu expected=%llu\n",i,
      (unsigned long long)calls[i],(unsigned long long)bytes[i],(unsigned long long)expected);return 10;}
  free(out);return 0;
}
''')
    exe=tmp_path/'check'
    helper=tmp_path/'helper.c'
    helper.write_text('#include <string.h>\nvoid __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes(void *d,const void *s,size_t n){memcpy(d,s,n);}\n')
    subprocess.run([gcc,'-O2','-fno-builtin','-U_FORTIFY_SOURCE','-D_FORTIFY_SOURCE=0',
        '-I',str(tmp_path),str(SOURCE),str(harness),str(helper),
        '-Wl,--wrap=memcpy','-Wl,--wrap=memset',
        '-Wl,--wrap=__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes','-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_profile_parser_rejects_partial_or_fabricated_coverage():
    target=ROOT/'tools/qwen3/probe_model_memory_arm.py'
    local=importlib.util.spec_from_file_location('memory_profile_arm_parser',target)
    module=importlib.util.module_from_spec(local);local.loader.exec_module(module)
    data=bytearray(128+4096*48);data[:8]=b'QWMEM01\0'
    struct.pack_into('<IIQQII',data,8,1,4096,0,1,48,0x10000)
    for index,kind in enumerate((1,2,3)):
        struct.pack_into('<IIQQQQQ',data,128+index*48,kind,0x10100+index*4,128,7,896,9,10000)
    parsed=module.parse_profile(bytes(data))
    assert parsed['totals']['3']['bytes']==896 and parsed['totals']['1']['calls']==7
    for offset,value in ((16,1),(32,32),(128+24,895),(128,4)):
        corrupted=data[:];struct.pack_into('<I',corrupted,offset,value)
        with pytest.raises(ValueError):module.parse_profile(bytes(corrupted))
    with pytest.raises(ValueError):module.parse_profile(bytes(data[:-1]))
    extended=data[:]
    struct.pack_into('<IIQQQQQ',extended,128+3*48,10,0x10500,0,197,0,1000,1000000)
    parsed=module.parse_profile(bytes(extended),tuple(range(1,13)),(1,2,3,10))
    assert parsed['totals']['10']['calls']==197 and parsed['totals']['5']['calls']==0
    with pytest.raises(ValueError):module.parse_profile(bytes(extended))
    with pytest.raises(ValueError):module.parse_profile(bytes(data),tuple(range(1,13)),(10,))
    with pytest.raises(ValueError):module.parse_profile(bytes(data),(1,2,3,17))
