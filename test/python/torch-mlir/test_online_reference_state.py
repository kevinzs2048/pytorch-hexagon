"""Independent CPU state must be audited, including numerical replay branches."""
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/qwen3'))
import pytest
from online_packet_audit import validate_reference_cache_trace


def trace():
    zero=hashlib.sha256(bytes(4194304)).hexdigest();rows=[]
    for step,request,position,incoming in [(0,0,0,zero),(1,0,1,'actual-first-output'),(2,1,0,zero)]:
        ins=[{'sha256':'native-token-position-mask'} for _ in range(60)]
        for index in [3,4,*range(6,60)]:ins[index]={'sha256':incoming}
        outs=[{'sha256':f'actual-{step}-output'} for _ in range(57)]
        if step==0:outs=[{'sha256':'actual-first-output'} for _ in range(57)]
        rows.append(dict(step=step,request_index=request,query=1,positions=[position],input=dict(fields=ins),output=dict(fields=outs)))
    return rows


def test_independent_native_reset_and_continuity_pass():
    validate_reference_cache_trace(trace())


@pytest.mark.parametrize('fault',['initial_reset','second_request_reset','continuity','position','ownership'])
def test_corrupted_reference_state_is_rejected(fault):
    rows=trace()
    if fault=='initial_reset':rows[0]['input']['fields'][3]['sha256']='polluted'
    elif fault=='second_request_reset':rows[2]['input']['fields'][59]['sha256']='polluted'
    elif fault=='continuity':rows[1]['input']['fields'][34]['sha256']='different-history'
    elif fault=='position':rows[1]['positions']=[2]
    else:rows[2]['request_index']=2
    with pytest.raises(ValueError):validate_reference_cache_trace(rows)
