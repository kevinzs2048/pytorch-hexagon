"""Share identical checked reciprocals without weakening their first check."""
import os
import subprocess

import pytest

OPT = os.environ.get('QWEN_PACKED_I8_OPT')
pytestmark = pytest.mark.skipif(not OPT, reason='QWEN_PACKED_I8_OPT required')


def source(second_input='%a', second_init='%init', predicate='one', numerator='%one'):
    def reciprocal(name, value, init, pred, num):
        return f'''
        %{name} = linalg.generic {{indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel"]}}
          ins({value} : tensor<2x4xf32>) outs({init} : tensor<2x4xf32>) {{
          ^bb0(%x: f32, %unused: f32):
            %ok = arith.cmpf {pred}, %x, %zero : f32
            cf.assert %ok, "nonzero scale required"
            %inverse = arith.divf {num}, %x : f32
            linalg.yield %inverse : f32
        }} -> tensor<2x4xf32>
        '''
    return '''#id = affine_map<(d0,d1) -> (d0,d1)>
      module {
        func.func @test(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>,
                       %init: tensor<2x4xf32>, %other: tensor<2x4xf32>)
          -> (tensor<2x4xf32>, tensor<2x4xf32>) {
          %zero = arith.constant 0.0 : f32
          %one = arith.constant 1.0 : f32
          %two = arith.constant 2.0 : f32
      ''' + reciprocal('first', '%a', '%init', 'one', '%one') + reciprocal(
          'second', second_input, second_init, predicate, numerator) + '''
          return %first, %second : tensor<2x4xf32>, tensor<2x4xf32>
        }
      }'''


def run(text):
    return subprocess.run([OPT, '-', '--qwen-reuse-activation=expected-count=1',
                           '--cse', '--verify-each'], input=text,
                          text=True, capture_output=True, timeout=30)


def test_reuses_division_and_keeps_first_assertion():
    result = run(source())
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('arith.divf') == 1
    assert result.stdout.count('cf.assert') == 1
    assert 'qwen reused activation reciprocals: 1' in result.stderr


@pytest.mark.parametrize('change', [dict(second_input='%b'), dict(second_init='%other'),
                                  dict(predicate='ogt'), dict(numerator='%two')])
def test_rejects_different_values_bounds_check_or_arithmetic(change):
    result = run(source(**change))
    assert result.returncode != 0
    assert 'activation reuse coverage differs: 0 != 1' in result.stderr
