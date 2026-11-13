"""Host oracle for the persistent packed-weight ABI's exact integer contract."""
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_prepacked_matmul_against_logical_matrix(tmp_path):
    compiler = shutil.which('cc')
    if not compiler:
        pytest.skip('C compiler required')
    source = tmp_path/'check.c'
    source.write_text(r'''
#include <stdlib.h>
#include "exact_int8_prepacked.h"
static unsigned state=71;
static unsigned next(void){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
int main(void){
 for(size_t m=1;m<=32;++m)for(size_t k=4;k<=68;k+=32)for(size_t n=128;n<=256;n+=128){
  int8_t*a=malloc(m*k),*b=malloc(k*n),*p=malloc(k*n);
  int32_t*c=malloc(m*n*4),*ref=malloc(m*n*4);
  if(!a||!b||!p||!c||!ref)return 2;
  for(size_t i=0;i<m*k;++i)a[i]=(int8_t)next();
  for(size_t i=0;i<k*n;++i)b[i]=(int8_t)next();
  for(size_t col=0;col<n;++col)for(size_t red=0;red<k;++red)
   p[(col/128)*k*128+(red/4)*512+(col%128)*4+red%4]=b[red*n+col];
  for(size_t i=0;i<m*n;++i){unsigned x=next();memcpy(c+i,&x,4);memcpy(ref+i,&x,4);}
  if(hexagon_i8_prepacked_v1(a,p,c,m,k,n,k,n))return 3;
  for(size_t row=0;row<m;++row)for(size_t col=0;col<n;++col){
   uint32_t x;memcpy(&x,ref+row*n+col,4);
   for(size_t red=0;red<k;++red)x+=(uint32_t)((int32_t)a[row*k+red]*(int32_t)b[red*n+col]);
   memcpy(ref+row*n+col,&x,4);
  }
  if(memcmp(c,ref,m*n*4))return 4;
  if(!hexagon_i8_prepacked_v1(a,p,c,0,k,n,k,n))return 5;
  if(!hexagon_i8_prepacked_v1(a,p,c,m,k-1,n,k,n))return 6;
  free(a);free(b);free(p);free(c);free(ref);
 }
 return 0;
}
''')
    binary = tmp_path/'check'
    subprocess.run([compiler, '-O2', '-Wall', '-Wextra', '-I',
                    str(ROOT/'models/qwen3/native/runtime/profile'), str(source),
                    '-o', str(binary)], check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_dma_resource_failure_falls_back_without_leaking_or_repeated_allocation(tmp_path):
    compiler = shutil.which('cc')
    if not compiler:
        pytest.skip('C compiler required')
    (tmp_path/'qurt_thread.h').write_text('static unsigned thread_id=7;\nstatic unsigned qurt_thread_get_id(void){return thread_id;}\n')
    (tmp_path/'hex-dma.h').write_text('''
#include <stddef.h>
typedef struct { void *next; } dma_descriptor_2d;
typedef struct { void *dst; const void *src; } dma_ptr;
typedef struct {
 dma_descriptor_2d *desc,*tail; dma_ptr *dptr;
 unsigned capacity,idx_mask;
} dma_queue;
''')
    (tmp_path/'HAP_compute_res.h').write_text('''
typedef int compute_res_attr_t;
static int mode,allocations,releases,leases,attempts;
static char storage;
static int HAP_compute_res_attr_init(int*a){*a=0;return 0;}
static int HAP_compute_res_attr_set_serialize(int*a,int n){return 0;}
static int HAP_compute_res_attr_set_cache_mode(int*a,int n){return 0;}
static int HAP_compute_res_attr_set_vtcm_param_v2(int*a,unsigned n,unsigned m,unsigned p){return 0;}
static unsigned HAP_compute_res_acquire(int*a,unsigned timeout){++allocations;return mode==1?0:7;}
static int HAP_compute_res_attr_get_vtcm_ptr_v2(int*a,void**p,unsigned*n){*p=&storage;*n=mode==2?128:3145728;return 0;}
static int HAP_compute_res_acquire_cached(unsigned id,unsigned timeout){++attempts;if(mode==3)return 1;++leases;return 0;}
static void HAP_compute_res_release_cached(unsigned id){--leases;}
static void HAP_compute_res_release(unsigned id){++releases;}
''')
    source=tmp_path/'resource.c'
    source.write_text('''
#include <stdlib.h>
#include <assert.h>
#include "prepacked_dma.h"
int main(int argc,char**argv){
 mode=atoi(argv[1]);
 int scope_first=argc>2;
 if(scope_first)qwen_prepacked_forward_begin();
 assert(!qwen_dma_acquire(3076) && allocations==0);
 int acquired=qwen_dma_acquire(3072);
 assert(acquired==(mode==0));
 if(acquired)qwen_dma_release();
 if(mode==1 || mode==2){
   assert(!qwen_dma_acquire(1024));
   assert(allocations==1);
 }
 if(scope_first)qwen_prepacked_forward_end();
 int before=attempts;
 qwen_prepacked_forward_begin();
 for(int i=0;i<197;++i){
   int ok=qwen_dma_acquire(1024);
   assert(ok==(mode==0));
   if(ok)qwen_dma_release();
 }
 if(mode==0)assert(leases==1 && attempts==before+1);
 qwen_prepacked_forward_end();
 assert(leases==0);
 qwen_prepacked_forward_begin();
 assert(!qwen_dma_acquire(3076));
 qwen_prepacked_forward_end();
 assert(leases==0);
 if(mode==0){
   assert(qwen_dma_acquire(1024));qwen_dma_release();
   assert(leases==0 && attempts==before+2);
 }
 qwen_dma_destroy();
 assert(releases==(mode==1?0:1));
 if(mode==0)assert(leases==0);
 return 0;
}
''')
    binary=tmp_path/'resource'
    subprocess.run([compiler,'-std=c11','-I'+str(tmp_path),'-I'+str(
        ROOT/'models/qwen3/native/runtime/profile'),str(source),'-o',str(binary)],
        check=True,capture_output=True,timeout=30)
    for mode in range(4):
        for extra in ([], ['scope-first']):
            subprocess.run([str(binary),str(mode),*extra],check=True,timeout=5)
