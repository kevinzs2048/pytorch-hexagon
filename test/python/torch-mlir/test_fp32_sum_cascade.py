"""Admission and structural guards for the explicit, bounded FP32 sum policy."""
import pytest
from test_matmul_schedule_safety import run


def module(k=128,outer=(4,),keepdim=True,dtype='f32',body=None,buffer=False):
    shape=(*outer,k);result=(*outer,1) if keepdim else outer
    kind='memref' if buffer else 'tensor'
    a=kind+'<'+'x'.join(map(str,shape))+'x'+dtype+'>'
    b=kind+'<'+'x'.join(map(str,result))+('x' if result else '')+dtype+'>'
    dims=','.join(f'd{i}' for i in range(len(shape)))
    output=','.join([*(f'd{i}' for i in range(len(outer))),*(['0'] if keepdim else [])])
    iters=','.join(['"parallel"']*len(outer)+['"reduction"'])
    body=body or f'%sum = arith.addf %x,%acc : {dtype}\n linalg.yield %sum : {dtype}'
    assign='' if buffer else '%result = '
    ret='return' if buffer else 'return %result : '+b
    sig='' if buffer else ' -> '+b
    return f'''func.func @probe(%a:{a},%init:{b}){sig} {{
      {assign}linalg.generic {{indexing_maps=[affine_map<({dims})->({dims})>,affine_map<({dims})->({output})>],iterator_types=[{iters}]}}
        ins(%a:{a}) outs(%init:{b}) {{
          ^bb0(%x:{dtype},%acc:{dtype}): {body}
        }}{sig}
      {ret}
    }}'''


def transform(text):return run(text,'--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))',schedule=False)


@pytest.mark.parametrize('k',[128,1024])
@pytest.mark.parametrize('outer',[(4,),(2,3),(1,4,8)])
@pytest.mark.parametrize('keepdim',[False,True])
def test_explicit_policy_is_valid_and_initial_accumulator_used_once(k,outer,keepdim):
    output=transform(module(k=k,outer=outer,keepdim=keepdim))
    assert output.count('hexagon.fp32_sum_cascade_partial')==1
    assert output.count('hexagon.fp32_sum_cascade =')==1 or output.count('hexagon.fp32_sum_cascade,')==1 or 'hexagon.fp32_sum_cascade}' in output
    assert output.count('%arg1')==2 # signature and one arithmetic/reshape use
    assert 'fastmath' not in output


@pytest.mark.parametrize('k',[128,1024])
def test_reapplying_does_not_change_an_existing_cascade(k):
    first=transform(module(k=k));second=transform(first)
    assert first==second


@pytest.mark.parametrize('k',[1,64,127,129,256,1023,1025,'?'])
def test_unsupported_reduction_lengths_unchanged(k):
    assert 'hexagon.fp32_sum_cascade' not in transform(module(k=k))


@pytest.mark.parametrize('outer',[('?',),(0,),(1048577,)])
def test_dynamic_empty_or_excessive_outer_size_not_admitted(outer):
    assert 'hexagon.fp32_sum_cascade' not in transform(module(outer=outer))


@pytest.mark.parametrize('dtype',['bf16','f16','f64'])
def test_other_element_types_not_admitted(dtype):
    assert 'hexagon.fp32_sum_cascade' not in transform(module(dtype=dtype))


@pytest.mark.parametrize('body',[
    '%v = arith.mulf %x,%acc : f32\n linalg.yield %v : f32',
    '%v = arith.addf %acc,%x : f32\n linalg.yield %v : f32',
    '%v = arith.addf %x,%acc fastmath<fast> : f32\n linalg.yield %v : f32',
    '%v = arith.mulf %x,%x : f32\n %s = arith.addf %v,%acc : f32\n linalg.yield %s : f32',
])
def test_custom_bodies_not_admitted(body):
    assert 'hexagon.fp32_sum_cascade' not in transform(module(body=body))


def test_buffer_semantics_not_admitted():
    assert 'hexagon.fp32_sum_cascade' not in transform(module(buffer=True))


def test_rank_one_keepdim_is_valid():
    assert 'hexagon.fp32_sum_cascade_partial' in transform(module(outer=()))


def test_scalar_result_not_admitted():
    assert 'hexagon.fp32_sum_cascade' not in transform(module(outer=(),keepdim=False))


@pytest.mark.parametrize('k',[128,1024])
def test_cascade_survives_full_lowering(k):
    output=run(transform(module(k=k)),
        '--linalg-to-llvm=enable-split-reduction=true add-fastmath=false far=false enable-vtcm-tiling=false enable-convert-to-hexagonmem=false',schedule=False,
        optimizer_env='QWEN_HEXAGON_LOWERING_OPT')
    assert 'llvm.func' in output and 'linalg.generic' not in output


def test_default_pipeline_does_not_enable_cascade():
    output=run(module(),'--linalg-to-llvm=enable-split-reduction=true',schedule=False,
               optimizer_env='QWEN_HEXAGON_LOWERING_OPT')
    assert 'fp32_sum_cascade' not in output
