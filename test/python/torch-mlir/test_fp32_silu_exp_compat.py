"""Real optimizer structural admission tests, not device/model acceptance."""
import pytest
from test_matmul_schedule_safety import run


DECL='func.func private @__hexagon_silu_exp_compat(f32) -> f32'
MARKER='hexagon.fp32_silu_exp_compat'


def module(batch=1,queries=2,width=3072,dtype='f32'):
    typ=f'tensor<{batch}x{queries}x{width}x{dtype}>'
    def generic(name,args,body):
        maps=', '.join(['affine_map<(d0,d1,d2)->(d0,d1,d2)>']*(len(args)+1))
        ins=', '.join(args);types=', '.join([typ]*len(args))
        bb=', '.join(f'%in{i}: {dtype}' for i in range(len(args)))+f', %out: {dtype}'
        return f'''%{name} = linalg.generic {{indexing_maps=[{maps}],iterator_types=["parallel","parallel","parallel"]}}
          ins({ins}: {types}) outs(%empty: {typ}) {{
            ^bb0({bb}): {body}
            linalg.yield %v : {dtype}
          }} -> {typ}\n'''
    body=generic('negative',['%a'],f'%v = arith.negf %in0 : {dtype}')
    body+=generic('exponent',['%negative'],f'%v = math.exp %in0 : {dtype}')
    body+=generic('addition',['%exponent'],f'%v = arith.addf %in0, %one : {dtype}')
    body+=generic('division',['%a','%addition'],f'%v = arith.divf %in0, %in1 : {dtype}')
    return f'''module {{
    {DECL}
    func.func @probe(%a: {typ}, %other: {typ}) -> {typ} {{
      %one = arith.constant 1.0 : {dtype}
      %empty = tensor.empty() : {typ}
      {body}
      return %division : {typ}
    }}
    }}'''


def transform(text,enabled=True):
    flag='{fp32-silu-exp-compat=true}' if enabled else ''
    return run(text,'--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum'+flag+'))',schedule=False, optimizer_env='QWEN_SINGLETON_SUM_OPT')


@pytest.mark.parametrize('batch,queries',[(1,1),(1,2),(28,2)])
def test_only_exp_changes_with_explicit_policy(batch,queries):
    output=transform(module(batch=batch,queries=queries))
    assert output.count(MARKER)==1 and 'math.exp' not in output
    assert all(op in output for op in ('arith.negf','arith.addf','arith.divf'))
    assert '__hexagon_softmax' not in output and '__wrap_expf' not in output
    assert transform(output)==output


def test_default_retains_sdk_exp_and_needs_no_declaration():
    output=transform(module().replace(DECL,''),enabled=False)
    assert 'math.exp' in output and MARKER not in output


@pytest.mark.parametrize('batch,queries,width',[(0,2,3072),(29,2,3072),(1,3,3072),(1,2,1024)])
def test_unvalidated_shapes_unchanged(batch,queries,width):
    assert MARKER not in transform(module(batch,queries,width))


@pytest.mark.parametrize('dtype',['bf16','f16','f64'])
def test_other_precision_unchanged(dtype):
    assert MARKER not in transform(module(dtype=dtype))


@pytest.mark.parametrize('before,after',[
    ('arith.constant 1.0','arith.constant 2.0'),
    ('ins(%a, %addition:','ins(%other, %addition:'),
    ('arith.negf %in0 :','arith.negf %in0 fastmath<fast> :'),
    ('math.exp %in0 :','math.exp %in0 fastmath<fast> :'),
    ('arith.addf %in0, %one :','arith.addf %in0, %one fastmath<fast> :'),
    ('arith.divf %in0, %in1 :','arith.divf %in0, %in1 fastmath<fast> :'),
    ('arith.divf %in0, %in1 :','arith.divf %in1, %in0 :'),
    ('arith.negf %in0 :','math.absf %in0 :'),
])
def test_not_a_strict_silu_chain_is_not_rewritten(before,after):
    assert before in module()
    assert MARKER not in transform(module().replace(before,after))


def test_an_exponent_shared_with_an_unrelated_consumer_is_not_rewritten():
    text=module();typ='tensor<1x2x3072xf32>'
    text=text.replace(f') -> {typ} {{',f') -> ({typ}, {typ}) {{')
    text=text.replace(f'return %division : {typ}',f'return %division, %exponent : {typ}, {typ}')
    assert MARKER not in transform(text)


@pytest.mark.parametrize('declaration',['', 'func.func private @__hexagon_silu_exp_compat(f64) -> f64',
    'func.func @__hexagon_silu_exp_compat(%x:f32) -> f32 { return %x : f32 }'])
def test_missing_or_nonexternal_abi_fails_closed(declaration):
    # The common helper raises an AssertionError for optimizer rejection.
    with pytest.raises(AssertionError,match='external __hexagon_silu_exp_compat'):
        transform(module().replace(DECL,declaration))
