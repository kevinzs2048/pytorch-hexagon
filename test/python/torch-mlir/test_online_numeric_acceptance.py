"""Hash differences must preserve the established numerical acceptance rule."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/qwen3'))
import numpy as np
import pytest
from online_packet_audit import difference,accept_metrics,numerical_outputs


def values():
    return [np.zeros((1,1,2,1),dtype='f4') for _ in range(56)]+[np.zeros((1,1,3),dtype='f4')]


def test_non_bitexact_difference_below_original_tolerance_passes():
    a,b=values(),values();a[-1][0,0,1]=5e-5;a[27][0,0,0,0]=5e-5
    metrics=numerical_outputs(a,b,1)
    assert not metrics[-1]['bitexact'] and not metrics[27]['bitexact']
    assert max(x['max_abs_error'] for x in metrics)<1e-4


@pytest.mark.parametrize('index',[0,27,55,56])
def test_large_difference_in_any_layer_or_logits_fails(index):
    a,b=values(),values();a[index].reshape(-1)[0]=0.25
    with pytest.raises(ValueError,match='1e-4'):numerical_outputs(a,b,1)


def test_original_threshold_is_strict():
    accept_metrics([{'max_abs_error':float(np.nextafter(1e-4,0))}])
    with pytest.raises(ValueError):accept_metrics([{'max_abs_error':1e-4}])


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf])
def test_nonfinite_tensor_cannot_be_accepted(invalid):
    with pytest.raises(ValueError,match='Nonfinite'):
        difference(np.array([invalid],dtype='f4'),np.zeros(1,dtype='f4'))


def test_zero_sign_difference_is_a_numerical_pass():
    m=difference(np.array([-0.],dtype='f4'),np.array([0.],dtype='f4'))
    assert not m['bitexact'] and m['max_abs_error']==0
    accept_metrics([m])
