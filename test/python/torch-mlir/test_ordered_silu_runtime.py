"""Check strided SiLU ABI, work partition and invalid-descriptor rejection."""
import ctypes as c
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Memref(c.Structure):
    _fields_ = [('allocated', c.POINTER(c.c_float)), ('aligned', c.POINTER(c.c_float)),
                ('offset', c.c_int64), ('sizes', c.c_int64 * 3), ('strides', c.c_int64 * 3)]


@pytest.fixture(scope='module')
def kernel(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    folder = tmp_path_factory.mktemp('ordered-silu')
    (folder/'worker-pool.h').write_text('''
#include <stdint.h>
typedef void* worker_pool_context_t;
static int worker_pool_init(void** p, unsigned n) {*p=(void*)1;return n!=4;}
static void worker_pool_release(void** p) {*p=0;}
static int worker_pool_run_func(void*,void(*f)(unsigned,unsigned,void*),void*p,unsigned n){
 for(unsigned i=0;i<n;++i)f(n,i,p);return 0;
}
''')
    (folder/'qurt_memory.h').write_text('''
#include <stdint.h>
#define QURT_MEM_CACHE_FLUSH_INVALIDATE 0
#define QURT_MEM_DCACHE 0
static int qurt_mem_cache_clean(uintptr_t,unsigned,int,int){return 0;}
''')
    # A fixed helper result isolates indexing/partitioning from exp accuracy.
    # Real exp arithmetic is checked against ARM in the full-model DSP replay.
    (folder/'helper.cpp').write_text('extern "C" float __hexagon_silu_exp_compat(float){return 1.0f;}\n')
    lib = folder/'kernel.so'
    subprocess.run([compiler, '-O2', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off',
                    '-I'+str(folder), str(ROOT/'models/qwen3/native/runtime/activation/ordered_silu.cpp'),
                    str(folder/'helper.cpp'), '-o', str(lib)], check=True, capture_output=True)
    library = c.CDLL(str(lib))
    fn = library._mlir_ciface_qwen_ordered_silu_v1
    fn.argtypes = [c.POINTER(Memref)] * 2
    fn.restype = c.c_int
    return fn


@pytest.mark.parametrize('rows', [1, 2, 3, 32])
def test_padded_input_offsets_worker_partitions_and_repeated_calls(kernel, rows):
    stride, offset = 3091, 7
    source = (c.c_float*(offset+rows*stride))()
    output = (c.c_float*(offset+rows*3072+11))(*([-999.0]*(offset+rows*3072+11)))
    a = Memref(source,source,offset,(c.c_int64*3)(1,rows,3072),(c.c_int64*3)(0,stride,1))
    b = Memref(output,output,offset,(c.c_int64*3)(1,rows,3072),(c.c_int64*3)(0,3072,1))
    for r in range(rows):
        for z in range(3072):
            source[offset+r*stride+z] = r*3072+z-2048
    assert kernel(a,b) == 0
    for r in range(rows):
        for z in range(3072):
            assert output[offset+r*3072+z] == (r*3072+z-2048)*0.5
    assert list(output[:offset]) == [-999.0]*offset
    assert list(output[offset+rows*3072:]) == [-999.0]*11
    before = bytes(output)
    a.strides[2] = 2
    assert kernel(a,b) == 2 and bytes(output) == before
    a.strides[2] = 1
    a.offset = 2**63-1
    assert kernel(a,b) == 2 and bytes(output) == before


class MatrixI32(c.Structure):
    _fields_ = [('allocated', c.POINTER(c.c_int32)), ('aligned', c.POINTER(c.c_int32)),
                ('offset', c.c_int64), ('sizes', c.c_int64 * 2), ('strides', c.c_int64 * 2)]


class MatrixF32(c.Structure):
    _fields_ = [('allocated', c.POINTER(c.c_float)), ('aligned', c.POINTER(c.c_float)),
                ('offset', c.c_int64), ('sizes', c.c_int64 * 2), ('strides', c.c_int64 * 2)]


class Weight(c.Structure):
    _fields_ = [('allocated', c.POINTER(c.c_float)), ('aligned', c.POINTER(c.c_float)),
                ('offset', c.c_int64), ('sizes', c.c_int64 * 1), ('strides', c.c_int64 * 1)]


@pytest.fixture(scope='module')
def shared_adapter(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    folder = tmp_path_factory.mktemp('quantized-silu-adapter')
    (folder/'worker-pool.h').write_text("""
#include <stdint.h>
typedef void* worker_pool_context_t;
extern unsigned pool_initializations;
static int worker_pool_init(void** p,unsigned n){++pool_initializations;*p=(void*)1;return n!=4;}
static void worker_pool_release(void**p){*p=0;}
static int worker_pool_run_func(void*,void(*f)(unsigned,unsigned,void*),void*p,unsigned n){
 for(unsigned i=0;i<n;++i)f(n,i,p);return 0;
}
""")
    (folder/'qurt_memory.h').write_text("""
#include <stdint.h>
#define QURT_MEM_CACHE_FLUSH_INVALIDATE 0
#define QURT_MEM_DCACHE 0
static int qurt_mem_cache_clean(uintptr_t,unsigned,int,int){return 0;}
""")
    # Stub arithmetic deliberately differs from SiLU: test descriptors, grid and
    # shared ownership here. Actual math/output equality is tested on the DSP.
    (folder/'stub.cpp').write_text("""
#include <stdint.h>
#include <cstdlib>
extern "C" {
unsigned pool_initializations=0, product_workers=0, strict_workers=0;
int reentrant_status=-1;
struct Flat {void*allocated,*aligned;int64_t offset,size,stride;};
struct Memref {float*allocated,*aligned;int64_t offset,sizes[3],strides[3];};
int _mlir_ciface_qwen_ordered_silu_v1(const Memref*,Memref*);
void silu_forward(int64_t,void*x,int64_t,void*y,int rows,int stride,
                  int gx,int gy,int gz,int id,int py,int pz){
 if(gx!=4||gy!=1||gz!=1||py||pz)std::abort();++strict_workers;
 auto a=(float*)((Flat*)x)->aligned,b=(float*)((Flat*)y)->aligned;
 for(int r=id;r<rows;r+=4)for(int k=0;k<3072;++k)b[r*3072+k]=a[r*stride+k]*.5f;
}
void silu_product(int64_t,void*g,int64_t,void*u,int64_t,void*s,int64_t,void*w,
                  int64_t,void*y,int rows,int gx,int gy,int gz,int id,int py,int pz){
 if(gx!=4||gy!=1||gz!=1||py||pz)std::abort();++product_workers;
 auto a=(float*)((Flat*)g)->aligned, b=(float*)((Flat*)y)->aligned;
 auto acc=(int32_t*)((Flat*)u)->aligned;
 auto scale=(float*)((Flat*)s)->aligned, weight=(float*)((Flat*)w)->aligned;
 Memref x{a,a,0,{1,rows,3072},{rows*3072,3072,1}};
 Memref out{b,b,0,{1,rows,3072},{rows*3072,3072,1}};
 reentrant_status=_mlir_ciface_qwen_ordered_silu_v1(&x,&out);
 for(int r=id;r<rows;r+=4)for(int k=0;k<3072;++k){int j=r*3072+k;
   b[j]=a[j]+float(acc[j])+scale[j]+weight[k];}
}
}
""")
    lib=folder/'adapter.so'
    subprocess.run([compiler,'-O2','-std=c++17','-shared','-fPIC','-ffp-contract=off',
                    '-DQWEN_TRITON_SILU','-DQWEN_APPROX_SILU','-DQWEN_TRITON_SILU_UP','-I'+str(folder),
                    str(ROOT/'models/qwen3/native/runtime/activation/ordered_silu.cpp'),
                    str(folder/'stub.cpp'),'-o',str(lib)],check=True,capture_output=True)
    library=c.CDLL(str(lib))
    library._mlir_ciface_qwen_ordered_silu_v1.argtypes=[c.POINTER(Memref)]*2
    library._mlir_ciface_qwen_quantized_silu_up_v1.argtypes=[c.POINTER(Memref),
        c.POINTER(MatrixI32),c.POINTER(MatrixF32),c.POINTER(Weight),c.POINTER(Memref)]
    return library


@pytest.mark.parametrize('rows',[4,5,31,32])
def test_quantized_adapter_uses_shared_owner_and_preserves_inputs(shared_adapter,rows):
    lib=shared_adapter;size=rows*3072;offset=3
    arrays=[(c.c_float*(size+offset+5))(*([-99.]*(size+offset+5))) for _ in range(3)]
    gate,scale,output=arrays
    up=(c.c_int32*(size+offset+5))();weight=(c.c_float*(3072+offset))()
    for i in range(size):gate[offset+i]=i%23;up[offset+i]=i%17;scale[offset+i]=i%11
    for i in range(3072):weight[offset+i]=i%5
    shape=(c.c_int64*3)(1,rows,3072);strides=(c.c_int64*3)(size,3072,1)
    g=Memref(gate,gate,offset,shape,strides);out=Memref(output,output,offset,shape,strides)
    u=MatrixI32(up,up,offset,(c.c_int64*2)(rows,3072),(c.c_int64*2)(3072,1))
    s=MatrixF32(scale,scale,offset,(c.c_int64*2)(rows,3072),(c.c_int64*2)(3072,1))
    w=Weight(weight,weight,offset,(c.c_int64*1)(3072),(c.c_int64*1)(1))
    saved=[bytes(x) for x in [gate,up,scale,weight]]
    strict=lib._mlir_ciface_qwen_ordered_silu_v1
    product=lib._mlir_ciface_qwen_quantized_silu_up_v1
    assert strict(g,out)==0
    assert product(g,u,s,w,out)==0
    assert c.c_uint.in_dll(lib,'pool_initializations').value==1
    assert c.c_int.in_dll(lib,'reentrant_status').value==3
    assert list(output[offset:offset+size])==[float(i%23+i%17+i%11+(i%3072)%5) for i in range(size)]
    assert list(output[:offset])+list(output[offset+size:])==[-99.]*8
    assert saved==[bytes(x) for x in [gate,up,scale,weight]]
    assert strict(g,out)==0 # busy guard released after product
    before=bytes(output)
    for descriptor,field,index,value in [(g,'sizes',1,3),(u,'strides',0,3073),
                                        (s,'sizes',1,3071),(w,'sizes',0,3071)]:
        old=getattr(descriptor,field)[index];getattr(descriptor,field)[index]=value
        assert product(g,u,s,w,out)==2 and bytes(output)==before
        getattr(descriptor,field)[index]=old
    # Every input must remain distinct from the output, including partial aliases.
    for source in [gate,up,scale,weight]:
        alias=c.cast(source,c.POINTER(c.c_float))
        aliased=Memref(alias,alias,offset+1,shape,strides)
        assert product(g,u,s,w,aliased)==2
    assert saved==[bytes(x) for x in [gate,up,scale,weight]]
    w.offset=2**63-1
    assert product(g,u,s,w,out)==2 and bytes(output)==before


def test_approximate_silu_rejects_alias_before_writing(shared_adapter):
    storage = (c.c_float * 6144)(*range(6144))
    shape = (c.c_int64 * 3)(1, 1, 3072)
    strides = (c.c_int64 * 3)(3072, 3072, 1)
    source = Memref(storage, storage, 0, shape, strides)
    saved = bytes(storage)
    for offset in (0, 1, 3071):
        output = Memref(storage, storage, offset, shape, strides)
        assert shared_adapter._mlir_ciface_qwen_ordered_silu_v1(source, output) == 2
        assert bytes(storage) == saved
