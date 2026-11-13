"""ABI/layout checks for ordered attention; numerical device fixtures live separately."""
import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Memref(c.Structure):
    _fields_ = [('allocated', c.POINTER(c.c_float)), ('aligned', c.POINTER(c.c_float)),
                ('offset', c.c_int64), ('sizes', c.c_int64 * 4), ('strides', c.c_int64 * 4)]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    folder = tmp_path_factory.mktemp('ordered-fp32')
    (folder / 'worker-pool.h').write_text('''
#include <stdint.h>
typedef void* worker_pool_context_t;
static int worker_pool_init(void** p, unsigned n) {*p=(void*)1; return n!=8;}
static void worker_pool_release(void** p) {*p=0;}
static int worker_pool_run_func(void*, void(*f)(unsigned,unsigned,void*),void*p,unsigned n) {
  for(unsigned i=0;i<n;++i)f(n,i,p); return 0;
}
''')
    (folder / 'qurt_memory.h').write_text('''
#include <stdint.h>
#define QURT_MEM_CACHE_FLUSH_INVALIDATE 0
#define QURT_MEM_DCACHE 0
static int qurt_mem_cache_clean(uintptr_t, unsigned, int, int) {return 0;}
''')
    lib = folder / 'kernel.so'
    subprocess.run([compiler, '-O2', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off',
                    '-I'+str(folder), str(ROOT/'models/qwen3/native/runtime/attention/ordered_fp32_attention.cpp'),
                    '-o', str(lib)], check=True, capture_output=True)
    library = c.CDLL(str(lib))
    fn = library._mlir_ciface_qwen_ordered_fp32_attention_v1
    fn.argtypes = [c.POINTER(Memref)] * 5
    fn.restype = c.c_int
    return fn


def tensor(shape, strides, offset=7):
    size = offset + sum((n-1)*s for n, s in zip(shape, strides)) + 19
    storage = (c.c_float * size)(*([-9876.0] * size))
    return storage, Memref(storage, storage, offset, (c.c_int64*4)(*shape),
                           (c.c_int64*4)(*strides))


@pytest.mark.parametrize('query,length', [(2,4), (1,2), (1,4), (2,2), (1,128), (1,256), (3,8), (8,256), (9,512)])
def test_offsets_strides_mask_gqa_and_workspace_reuse(kernel, query, length):
    q, qd = tensor((1,16,query,128), (9999,303,145,1))
    k, kd = tensor((1,8,length,128), (9999,length*137+5,137,1))
    v, vd = tensor((1,8,length,128), (9999,length*141+3,141,1))
    mask, md = tensor((1,1,query,length), (9999,9999,length+3,1))
    out, od = tensor((1,query,16,128), (4096,2048,128,1), offset=3)
    qd.strides[0] = kd.strides[0] = vd.strides[0] = od.strides[0] = 0
    md.strides[0] = md.strides[1] = 0
    for h in range(16):
        for row in range(query):
            for lane in range(128):
                q[qd.offset+h*303+row*145+lane] = 0.0
    for h in range(8):
        for token in range(length):
            for lane in range(128):
                k[kd.offset+h*kd.strides[1]+token*137+lane] = 1.0
                v[vd.offset+h*vd.strides[1]+token*141+lane] = h*32+token*4+lane%4
    for row in range(query):
        count = length//2 if query == 2 and row == 0 else length
        for token in range(length):
            mask[md.offset+row*md.strides[2]+token] = 0.0 if token < count else -float('inf')
    assert kernel(qd,kd,vd,md,od) == 0
    for row in range(query):
        count = length//2 if query == 2 and row == 0 else length
        for h in range(16):
            for lane in range(128):
                assert out[od.offset+row*2048+h*128+lane] == (h//2)*32+2*(count-1)+lane%4
    assert list(out[:od.offset]) == [-9876.0]*od.offset
    assert list(out[od.offset+query*2048:]) == [-9876.0]*(len(out)-od.offset-query*2048)
    vd.strides[3] = 2
    before = bytes(out)
    assert kernel(qd,kd,vd,md,od) == 2
    assert bytes(out) == before
    vd.strides[3] = 1
    kd.offset = -1
    assert kernel(qd,kd,vd,md,od) == 2


def test_vector_pack_is_a_byte_permutation_with_zero_padding(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    source = tmp_path/'pack.cpp'
    source.write_text(r'''
#include "ordered_kv_pack.h"
#include <vector>
#include <cassert>
int main() {
  for(int n : {1,3,8,353,1024}) for(int layout : {0,1}) {
    int hs=layout?128:n*128+13, ts=layout?1024:128;
    std::vector<uint32_t> input(3+7*hs+(n-1)*ts+128);
    std::vector<uint32_t> key(8*((n+7)/8*8)*128), value(8*((n+3)/4*4)*128);
    // Arbitrary bits include signed zero and NaN encodings; packing cannot round them.
    for(size_t i=0;i<input.size();++i) input[i]=(uint32_t(i)*0x9e3779b9u)^0x80000000u;
    for(int first=0;first<8;first+=2) {
      qwen_ordered::pack_keys((float*)key.data(),(float*)input.data()+3,hs,ts,n,first,first+2);
      qwen_ordered::pack_values((float*)value.data(),(float*)input.data()+3,hs,ts,n,first,first+2);
    }
    for(int h=0;h<8;++h) for(int t=0;t<(n+7)/8*8;++t) for(int z=0;z<128;++z) {
      uint32_t expected=t<n?input[3+h*hs+t*ts+z]:0;
      assert(key[((h*((n+7)/8)+t/8)*128+z)*8+t%8]==expected);
    }
    for(int h=0;h<8;++h) for(int t=0;t<(n+3)/4*4;++t) for(int z=0;z<128;++z) {
      uint32_t expected=t<n?input[3+h*hs+t*ts+z]:0;
      assert(value[((h*16+z/8)*((n+3)/4*4)+t)*8+z%8]==expected);
    }
  }
}
''')
    binary = tmp_path/'pack'
    subprocess.run([compiler, '-O2', '-std=c++17', '-Wno-psabi',
                    '-I'+str(ROOT/'models/qwen3/native/runtime/attention'),
                    str(source), '-o', str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)


def test_triton_decode_adapter_contract(tmp_path):
    """Exercise the real adapter; a stub kernel checks ABI/grid, not numerics."""
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    (tmp_path/'worker-pool.h').write_text('''
#include <cassert>
typedef void* worker_pool_context_t;
static int worker_pool_init_with_stack_size(void**p,unsigned n,unsigned stack) {
  assert(n==4 && stack==32768); *p=(void*)1; return 0;
}
static void worker_pool_release(void**p) {*p=0;}
static int worker_pool_run_func(void*,void(*f)(unsigned,unsigned,void*),void*p,unsigned n) {
  assert(n==4);for(unsigned i=0;i<n;++i)f(n,i,p);return 0;
}
''')
    (tmp_path/'qurt_memory.h').write_text('''
#include <cstdint>
#define QURT_MEM_CACHE_FLUSH_INVALIDATE 0
#define QURT_MEM_DCACHE 0
static int qurt_mem_cache_clean(uintptr_t, unsigned, int, int) {return 0;}
''')
    (tmp_path/'stub.cpp').write_text(r'''
#include <cassert>
#include <cstdint>
struct D {float *allocated,*aligned;int64_t offset,size,stride;};
extern "C" int qwen_strict_prefill(const void*,const void*,const void*,const void*,void*) {return 27;}
extern "C" void attention_decode(int64_t qr,void*q,int64_t kr,void*k,int64_t vr,void*v,
 int64_t orank,void*out,int64_t sr,void*scratch,int m,int n,int qh,int qt,int kh,int kt,
 int vh,int vt,int gx,int gy,int gz,int px,int py,int pz) {
 assert(qr==1&&kr==1&&vr==1&&orank==1&&sr==1);
 assert(m==1&&n==4&&gx==1&&gy==4&&gz==1&&px==0&&pz==0&&py>=0&&py<4);
 assert(qh==145&&kh==128&&kt==1024&&vh==128&&vt==1024);
 auto*s=(D*)scratch;auto*o=(D*)out;
 assert(s->size==1064960&&s->offset==0&&s->stride==1&&o->size==2048);
 for(int h=py*4;h<py*4+4;++h)for(int z=0;z<128;++z)o->aligned[h*128+z]=h*128+z;
}
''')
    library_path = tmp_path/'adapter.so'
    subprocess.run([compiler, '-O2', '-std=c++17', '-shared', '-fPIC',
                    '-I'+str(tmp_path),
                    str(ROOT/'models/qwen3/native/runtime/attention/triton_decode.cpp'),
                    str(tmp_path/'stub.cpp'), '-o', str(library_path)],
                   check=True, capture_output=True)
    library = c.CDLL(str(library_path))
    fn = library._mlir_ciface_qwen_ordered_fp32_attention_v1
    fn.argtypes, fn.restype = [c.POINTER(Memref)]*5, c.c_int
    q, qd = tensor((1,16,1,128), (0,145,0,1))
    k, kd = tensor((1,8,4,128), (0,128,1024,1))
    v, vd = tensor((1,8,4,128), (0,128,1024,1))
    mask, md = tensor((1,1,1,4), (0,0,0,1))
    out, od = tensor((1,1,16,128), (0,2048,128,1), offset=3)
    for i in range(4):
        mask[md.offset+i] = 0.0
    assert fn(qd,kd,vd,md,od) == 0
    assert list(out[od.offset:od.offset+2048]) == list(range(2048))
    assert list(out[:od.offset]) == [-9876.0]*od.offset
    assert list(out[od.offset+2048:]) == [-9876.0]*(len(out)-od.offset-2048)
    before = bytes(out)
    for invalid in (-float('inf'), float('nan'), 0.25):
        mask[md.offset+2] = invalid
        assert fn(qd,kd,vd,md,od) == 2
        assert bytes(out) == before
    mask[md.offset+2] = 0.0
    vd.strides[2] = 0
    assert fn(qd,kd,vd,md,od) == 2
    vd.strides[2] = 1024
    qd.offset = -1
    assert fn(qd,kd,vd,md,od) == 2
    qd.offset = 7
    assert fn(qd,kd,vd,md,od) == 0
    qd.sizes[2] = 2
    assert fn(qd,kd,vd,md,od) == 27  # Dispatch prefill to the retained implementation.
