"""Run the real C++ contract tool on admitted and rejected original IR."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS=Path(__file__).resolve().parents[3]/'tools/qwen3'
sys.path.insert(0,str(TOOLS))
from inspect_helper_effects import module


@pytest.fixture(scope='module')
def optimizer():
    path=os.environ.get('QWEN_SILU_EFFECT_OPT')
    if not path:pytest.skip('Explicit built isolated effect optimizer required')
    assert Path(path).is_file()
    return path


def admitted():
    text=module('silu-chain','unknown').replace('@probe(', '@Qwen3DynamicBoundProbe(')
    text=text.replace('iterator_types = ["parallel", "parallel", "parallel"]} ins(%x :',
        'iterator_types = ["parallel", "parallel", "parallel"], hexagon.fp32_arm_silu_denominator} ins(%x :',1)
    text=text.replace('func.call @__hexagon_silu_exp_compat(%negative) :',
        'func.call @__hexagon_silu_exp_compat(%negative) {hexagon.fp32_silu_exp_compat} :')
    return text


def invoke(optimizer,text,enabled=True,count=1,multi=False):
    argv=[optimizer,'-','--verify-each']
    if enabled:argv+=['--qwen-silu-effects='+('multi-row=true ' if multi else '')+'expected-count='+str(count)]
    return subprocess.run(argv,input=text,text=True,capture_output=True,timeout=30)


def test_changes_only_original_helper_declaration(optimizer):
    text=admitted()
    original=invoke(optimizer,text,False);candidate=invoke(optimizer,text)
    assert original.returncode==candidate.returncode==0,candidate.stderr
    attrs=' attributes {llvm.readnone, passthrough = ["nounwind", "willreturn"]}'
    assert candidate.stdout.count(attrs)==1
    assert candidate.stdout.replace(attrs,'',1)==original.stdout
    assert 'qwen scoped query1 SiLU effect calls: 1' in candidate.stderr
    assert 'qwen preserved original operations:' in candidate.stderr


@pytest.mark.parametrize('change',[
    lambda s:s.replace('hexagon.fp32_silu_exp_compat','unknown.tag'),
    lambda s:s.replace('hexagon.fp32_arm_silu_denominator','unknown.tag'),
    lambda s:s.replace('@Qwen3DynamicBoundProbe','@other_model'),
    lambda s:s.replace('tensor<1x1x3072xf32>','tensor<1x2x3072xf32>'),
    lambda s:s.replace('tensor<1x1x3072xf32>','tensor<1x1x1024xf32>'),
    lambda s:s.replace('private @__hexagon_silu_exp_compat(f32) -> f32',
        'private @__hexagon_silu_exp_compat(f32) -> f32 attributes {llvm.readnone}'),
    lambda s:s.replace('private @__hexagon_silu_exp_compat(f32) -> f32',
        'private @__hexagon_silu_exp_compat(f32) -> f32 attributes {passthrough = ["nounwind"]}'),
    lambda s:s.replace('private @__hexagon_silu_exp_compat(f32) -> f32',
        'private @__hexagon_silu_exp_compat(%x:f32) -> f32 {return %x : f32}'),
    lambda s:s.replace('%one = arith.constant 1.0 : f32',
        '%one = arith.constant 1.0 : f32\n%escape = func.constant @__hexagon_silu_exp_compat : (f32) -> f32'),
    lambda s:s.replace('%one = arith.constant 1.0 : f32',
        '%one = arith.constant 1.0 : f32\n%outside = func.call @__hexagon_silu_exp_compat(%one) {hexagon.fp32_silu_exp_compat} : (f32) -> f32'),
])
def test_rejects_foreign_unmarked_escaped_changed_shape_or_existing_contract(optimizer,change):
    result=invoke(optimizer,change(admitted()))
    assert result.returncode!=0 and result.returncode!=-11
    assert 'error:' in result.stderr


@pytest.mark.parametrize('count',[0,2,28])
def test_requires_exact_nonzero_coverage(optimizer,count):
    result=invoke(optimizer,admitted(),count=count)
    assert result.returncode!=0


def admitted_multi(rows):
    text=admitted().replace('tensor<1x1x3072xf32>',f'tensor<1x{rows}x3072xf32>')
    if rows=='?':
        text=text.replace('%empty = tensor.empty() : tensor<1x?x3072xf32>',
            '%axis = arith.constant 1 : index\n%query = tensor.dim %x, %axis : tensor<1x?x3072xf32>\n'
            '%empty = tensor.empty(%query) : tensor<1x?x3072xf32>')
    return text


@pytest.mark.parametrize('rows',[2,13,32,'?'])
def test_explicit_multi_scope_changes_only_scalar_declaration_and_preserves_original_FP(optimizer,rows):
    text=admitted_multi(rows);original=invoke(optimizer,text,False);candidate=invoke(optimizer,text,multi=True)
    assert original.returncode==candidate.returncode==0,candidate.stderr
    attrs=' attributes {llvm.readnone, passthrough = ["nounwind", "willreturn"]}'
    assert candidate.stdout.count(attrs)==1 and candidate.stdout.replace(attrs,'',1)==original.stdout
    assert 'qwen scoped query2..32 SiLU effect calls: 1' in candidate.stderr
    assert invoke(optimizer,text).returncode!=0


@pytest.mark.parametrize('rows',[0,1,33,1024])
def test_multi_scope_rejects_other_extents(optimizer,rows):
    result=invoke(optimizer,admitted_multi(rows),multi=True)
    assert result.returncode!=0 and 'error:' in result.stderr


@pytest.mark.parametrize('change',[
    lambda s:s.replace('hexagon.fp32_silu_exp_compat','unknown.tag'),
    lambda s:s.replace('hexagon.fp32_arm_silu_denominator','unknown.tag'),
    lambda s:s.replace('@Qwen3DynamicBoundProbe','@other_model'),
    lambda s:s.replace('3072xf32','1024xf32'),
    lambda s:s.replace('tensor<1x?x','tensor<2x?x'),
])
def test_multi_scope_rejects_foreign_helpers_shapes_and_markers(optimizer,change):
    result=invoke(optimizer,change(admitted_multi('?')),multi=True)
    assert result.returncode!=0 and 'error:' in result.stderr
