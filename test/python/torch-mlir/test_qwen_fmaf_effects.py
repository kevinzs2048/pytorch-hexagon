"""Runtime identity and packet checks; this is not DSP purity acceptance."""
from qwen3.compiler_dependency import native_root
import ctypes
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[3]


def driver():
    spec=importlib.util.spec_from_file_location('qwen_fmaf_driver',ROOT/'tools/qwen3/probe_model_memory_arm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_query(tmp_path):
    compiler=shutil.which('gcc')
    if not compiler:pytest.skip('Native compiler unavailable')
    lib=tmp_path/'effects.so'
    subprocess.run([compiler,'-O2','-fPIC','-fno-builtin','-shared','-D_GNU_SOURCE',
        '-Wall','-Wextra','-Werror',str(native_root()/'runtime/profile/model_fmaf_effects.c'),
        '-lm','-ldl','-o',str(lib)],check=True)
    owner=ctypes.CDLL(str(lib),mode=ctypes.RTLD_GLOBAL,use_errno=True)
    fn=owner.qwen_fmaf_effects_query
    fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
    fn.restype=ctypes.c_int
    def call(payload,output_len):
        buf=ctypes.create_string_buffer(payload);out=ctypes.create_string_buffer(output_len)
        time=ctypes.c_uint64(99);ctypes.set_errno(73)
        rc=fn(buf,len(payload),out,output_len,ctypes.byref(time))
        assert ctypes.get_errno()==73
        return rc,out.raw,time.value
    return call


def normal_payload():
    triples=((1.0,2.0,3.0),(1.0000001192092896,0.9999999403953552,-1.0),
             (-0.0,1.0,-0.0),(0.5,0.5,-0.25))
    return b'QWFMA01\0'+struct.pack('<II',len(triples),0)+b''.join(struct.pack('<fff',*t) for t in triples)


def test_actual_native_query_results_provider_and_errno(native_query):
    payload=normal_payload();rc,response,t=native_query(payload,256+4*4)
    assert rc==0 and t==0
    parsed=driver().parse_fmaf_effects(response,payload)
    assert parsed['finite_test_triples']==4 and parsed['errno_unchanged']
    assert parsed['independent_ARM_libm_result_bits_identical']
    # An IFUNC may resolve to a hidden dispatch target without a public name.
    # The query reports that fact; equality of imported/dlsym addresses remains
    # mandatory, and the provider must be observable in this native test.
    assert parsed['dladdr_result'] and parsed['provider']


@pytest.mark.parametrize('payload,size,expected',[
    (b'HXDY'+bytes(12),272,-1),
    (b'QWFMA01\0',272,2),
    (b'QWFMA01\0'+struct.pack('<II',0,0),256,2),
    (b'QWFMA01\0'+struct.pack('<II',8193,0),256,2),
    (b'QWFMA01\0'+struct.pack('<II',4,1)+bytes(48),272,2),
    (normal_payload()[:-1],272,2),
    (normal_payload(),271,2),
])
def test_native_query_rejects_wrong_format_without_errno_change(native_query,payload,size,expected):
    assert native_query(payload,size)[0]==expected


@pytest.mark.parametrize('offset,value',[(8,2),(12,3),(16,0),(20,0),(40,7),(44,8),(240,1),(256,0)])
def test_parser_rejects_wrong_identity_errno_or_actual_results(native_query,offset,value):
    payload=normal_payload();_,response,_=native_query(payload,272)
    corrupt=bytearray(response);struct.pack_into('<I',corrupt,offset,value)
    with pytest.raises(ValueError):driver().parse_fmaf_effects(bytes(corrupt),payload)


def test_query_input_contains_bounded_finite_binary32_triples():
    module=driver();payload=module.fmaf_inputs();count,reserved=struct.unpack_from('<II',payload,8)
    assert payload==module.fmaf_inputs() and count==6344 and reserved==0
    assert len(payload)==16+count*12
    for (bits,) in struct.iter_unpack('<I',payload[16:]):assert bits&0x7f800000!=0x7f800000


def test_new_query_preserves_original_normal_model_call(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools/qwen3'))
    spec=importlib.util.spec_from_file_location('profile_fmaf_instrument',ROOT/'tools/qwen3/profile_model_memory.py')
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
    baseline=module.instrument(original)
    query=module.instrument(original,True)
    for extra in ('extern "C" int qwen_fmaf_effects_query(const unsigned char *,int,unsigned char *,int,uint64_t *);\n',
        '  int effects = qwen_fmaf_effects_query(input,input_len,output,output_len,kernel_us);\n',
        '  if (effects >= 0) return effects;\n'):
        query=query.replace(extra,'')
    assert query==baseline
