"""Check exact integer guard equivalence and full strided byte-copy behavior."""
from qwen3.compiler_dependency import compiler_root
import ctypes
import itertools
import random
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME=compiler_root()/'tools/qwen3/runtime/profile'


class Unranked(ctypes.Structure):
    _fields_=[('rank',ctypes.c_int64),('descriptor',ctypes.c_void_p)]


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    folder=tmp_path_factory.mktemp('copy-guard');compiler=shutil.which('g++')
    assert compiler
    result=[]
    for name in ('original_memref_copy','guarded_memref_copy'):
        target=folder/(name+'.so')
        # The former saved reference was byte-identical to this maintained ABI implementation.
        source = (compiler_root()/'qcom_hexagon_backend/bin/runtime/src/HexagonMemrefCopy.cpp'
                  if name == 'original_memref_copy' else RUNTIME/(name+'.cpp'))
        subprocess.run([compiler,'-O2','-Wall','-Wextra','-Werror','-fPIC','-shared','-DQWEN_COPY_CHECKS_TEST_ONLY',
            str(source),'-o',str(target)],check=True,capture_output=True)
        lib=ctypes.CDLL(str(target));copy=getattr(lib,'memrefCopy' if name.startswith('original') else '__wrap_memrefCopy')
        copy.argtypes=[ctypes.c_int64,ctypes.POINTER(Unranked),ctypes.POINTER(Unranked)];copy.restype=None
        result.append((lib,copy))
    return result


def test_positive_product_classifier_matches_unbounded_integer_math(libraries):
    check=libraries[1][0].qwen_copy_positive_product
    check.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_int64)];check.restype=ctypes.c_bool
    edges={1,2,3,(1<<63)-1}
    for bit in (15,16,30,31,32,33,47,48,62):
        edges.update([(1<<bit)-1,1<<bit,(1<<bit)+1])
    rng=random.Random(20261006)
    pairs=list(itertools.product(sorted(edges),repeat=2))+[(rng.randint(1,(1<<63)-1),rng.randint(1,(1<<63)-1)) for _ in range(1024)]
    for a,b in pairs:
        out=ctypes.c_int64(-123);admitted=check(a,b,ctypes.byref(out));expected=a*b<=(1<<63)-1
        assert admitted==expected,(a,b)
        assert out.value==(a*b if expected else -123)


def test_byte_extent_classifier_matches_original_native_size_bound(libraries):
    check=libraries[1][0].qwen_copy_block_extent
    check.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_size_t)];check.restype=ctypes.c_bool
    bound=(1<<(8*ctypes.sizeof(ctypes.c_size_t)))-1
    edges=[1,2,128,512,(1<<31)-1,1<<31,(1<<32)-1,1<<32,(1<<32)+1,(1<<63)-1]
    for a,b in itertools.product(edges,repeat=2):
        out=ctypes.c_size_t(123);admitted=check(a,b,ctypes.byref(out));expected=a*b<=bound
        assert admitted==expected,(a,b)
        assert out.value==(a*b if expected else 123)


def descriptor(data,offset,sizes,strides):
    n=len(sizes)
    class Ranked(ctypes.Structure):
        _fields_=[('allocated',ctypes.c_void_p),('aligned',ctypes.c_void_p),('offset',ctypes.c_int64),
            ('sizes',ctypes.c_int64*n),('strides',ctypes.c_int64*n)]
    ranked=Ranked(ctypes.addressof(data),ctypes.addressof(data),offset,(ctypes.c_int64*n)(*sizes),(ctypes.c_int64*n)(*strides))
    return ranked,Unranked(n,ctypes.addressof(ranked))


@pytest.mark.parametrize('sizes,source_strides,destination_strides,source_offset,destination_offset',[
    ([],[],[],3,7),([8],[1],[1],0,0),([8],[2],[3],1,2),
    ([2,3,8],[24,8,1],[24,8,1],2,3),([2,1,8],[16,123,1],[24,99,1],2,3),
    ([2,3,8],[0,8,1],[24,8,1],2,3),([2,3,8],[24,0,1],[24,8,1],2,3),
    ([2,3,8],[24,8,1],[0,8,1],2,3),([2,3,8],[24,8,1],[24,0,1],2,3),
    ([2,3,8],[-24,8,-1],[24,-8,1],64,64),([2,0,8],[24,8,1],[24,8,1],0,0),
    ([1,2,2,3],[100,24,8,1],[100,24,8,1],1,5),
])
@pytest.mark.parametrize('element_size',[1,2,4,8,16])
def test_copy_preserves_all_addressed_bits_and_unwritten_bytes(libraries,sizes,source_strides,destination_strides,source_offset,destination_offset,element_size):
    amount=4096
    source=(ctypes.c_ubyte*amount)(*[(i*37+19)%256 for i in range(amount)])
    expected=bytearray([0xa5]*amount)
    indices=itertools.product(*(range(s) for s in sizes)) if sizes else [()]
    for index in indices:
        read=(source_offset+sum(i*s for i,s in zip(index,source_strides)))*element_size
        write=(destination_offset+sum(i*s for i,s in zip(index,destination_strides)))*element_size
        expected[write:write+element_size]=bytes(source[read:read+element_size])
    for _,copy in libraries:
        dest=(ctypes.c_ubyte*amount)(*[0xa5]*amount)
        ranked_src,src=descriptor(source,source_offset,sizes,source_strides)
        ranked_dst,dst=descriptor(dest,destination_offset,sizes,destination_strides)
        copy(element_size,ctypes.byref(src),ctypes.byref(dst))
        assert bytes(dest)==expected


@pytest.mark.parametrize('name',['qwen_copy_positive_product','qwen_copy_block_extent'])
@pytest.mark.parametrize('a,b',[(-1,1),(1,-1),(0,1),(1,0)])
def test_test_hooks_reject_outside_positive_precondition(libraries,name,a,b):
    check=getattr(libraries[1][0],name)
    assert not check(a,b,None)
