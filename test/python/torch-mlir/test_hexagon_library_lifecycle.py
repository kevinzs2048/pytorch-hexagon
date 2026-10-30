import copy
import pytest
from hexagon_runtime.hexagon_library_lifecycle import audit_library_events


def events():
    def operation(stage,**kwargs):
        return dict(stage=stage,rpc_result=0,seconds=0.1,**kwargs)
    def invoke(index,slot):
        return operation('invoke',ordinal=index,library_slot=slot,DSP_result=0,output_result=0,
            kernel_us=1,input_bytes=64,output_bytes=128,
            AP_input_fnv1a32=123,DSP_input_fnv1a32=123,RPC_input_fnv1a32=123)
    return [operation('open'),operation('load',loader_result=0),
        dict(stage='ready',ARM_pid=10,dynamic_sizes=True,library_switching=True,input_bytes=128,output_bytes=256),
        invoke(1,0),operation('entry_open',library_slot=1),operation('entry_load',library_slot=1,loader_result=0),
        operation('select',library_slot=1,cached=False),invoke(2,1),
        operation('select',library_slot=0,cached=True),invoke(3,0),
        operation('unload',library_slot=1,loader_result=0),operation('close',library_slot=1),
        operation('unload',library_slot=0,loader_result=0),operation('close',library_slot=0)]


def test_owned_library_trace_preserves_call_order_and_closes_both():
    result=audit_library_events(events(),[0,1,0])
    assert result['one_AP_process'] and result['loaded_libraries']==2
    assert result['all_loaded_owners_closed'] and not result['physical_weight_sharing_verified']


@pytest.mark.parametrize('index,field,value',[
    (4,'library_slot',2),(5,'loader_result',1),(6,'cached',True),
    (7,'library_slot',0),(7,'library_slot',True),(7,'ordinal',1),
    (7,'DSP_result',False),(7,'RPC_input_fnv1a32',124),(7,'input_bytes',129),
    (8,'library_slot',3),(8,'cached',False),(11,'library_slot',0),
    (12,'rpc_result',1),(13,'seconds',float('nan')),
])
def test_invalid_execution_or_cleanup_is_not_accepted(index,field,value):
    rows=events();rows[index][field]=value
    with pytest.raises(ValueError):audit_library_events(rows,[0,1,0])


@pytest.mark.parametrize('slots',[[0,1],[0,0,0],[0,True,0],[]])
def test_every_forward_must_bind_the_actual_selected_library(slots):
    with pytest.raises(ValueError):audit_library_events(events(),slots)


def test_native_session_table_cannot_lie_about_owned_libraries():
    from audit_qwen_native_request_sequence import audit_sessions
    rows=events()
    record=dict(library_switching=True,DSP_calls=3,stream_events=rows,
        DSP_sessions=[dict(index=0,first_forward=0,end_forward=3,process_exit_code=0,
            events=rows,library='prefill.so',sha256='prefill')],
        DSP_library_slots=[dict(index=0,library='prefill.so',sha256='prefill'),
                           dict(index=1,library='decode.so',sha256='decode')],
        forwards=[dict(DSP_session=0,DSP_library_slot=i,DSO_sha256=('prefill','decode')[i]) for i in [0,1,0]],
        selected_DSO_sha256='prefill',loaded_library_source='prefill.so')
    assert len(audit_sessions(record))==1
    bad=copy.deepcopy(record);bad['forwards'][1]['DSO_sha256']='prefill'
    with pytest.raises(ValueError,match='different owned library'):audit_sessions(bad)
    bad=copy.deepcopy(record);bad['DSP_library_slots'].pop()
    with pytest.raises(ValueError,match='table differs'):audit_sessions(bad)
