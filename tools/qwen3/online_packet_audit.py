"""Observe actual DSP packets against independently executed native CPU tensors.

Hash equality proves byte equality, not a new numerical tolerance. Differences
require independently replayed tensor bytes; they never replace DSP values.
"""
import hashlib
import numpy as np
from packet_trace import describe_bytes, validate_qwen_layout, TOLERANCE


def difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype or not left.size:
        raise ValueError('Equal nonempty tensor layouts required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite numerical reference/output')
    error = np.abs(left.astype(np.float64)-right.astype(np.float64))
    return dict(max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                bitexact=left.tobytes()==right.tobytes())


def accept_metrics(metrics):
    if not metrics or any(not np.isfinite(m['max_abs_error']) or
                          m['max_abs_error'] >= TOLERANCE for m in metrics):
        raise ValueError('Original max_abs < 1e-4 numerical standard failed: '+str(metrics))


def numerical_outputs(actual, reference, valid):
    metrics = [difference(a[:,:,:valid,:], b[:,:,:valid,:])
               for a,b in zip(actual[:56],reference[:56],strict=True)]
    metrics.append(difference(actual[-1],reference[-1]))
    accept_metrics(metrics)
    return metrics


def validate_reference_cache_trace(forwards):
    """Validate independent CPU reset/continuity even on tolerated differences."""
    zero=hashlib.sha256(bytes(1*8*1024*128*4)).hexdigest()
    request=-1;previous=None;end=0
    for step,e in enumerate(forwards):
        positions=e['positions'];q=e['query'];index=e['request_index']
        if e['step']!=step or not 1<=q<=32 or len(positions)!=q:
            raise ValueError('Independent CPU forward ownership/query differs')
        incoming=[f['sha256'] for f in e['input']['fields'][3:5]+e['input']['fields'][6:]]
        outgoing=[f['sha256'] for f in e['output']['fields'][:56]]
        if len(incoming)!=56 or len(outgoing)!=56:
            raise ValueError('Complete independent CPU cache tensor descriptions required')
        if index!=request:
            if index!=request+1 or incoming!=[zero]*56:
                raise ValueError('Independent native CPU request did not reset every cache byte')
            end=0
        elif incoming!=previous:
            raise ValueError('Independent native CPU cache continuity differs')
        if positions!=list(range(end,end+q)) or end+q>1024:
            raise ValueError('Independent native CPU cache positions differ')
        request=index;previous=outgoing;end+=q


class OnlinePacketObserver:
    def __init__(self, trace, layout_for, replay):
        if (trace['status'] != 'independent_native_packet_trace_completed' or
                trace['full_native_generate'] is not True or
                trace['CPU_reference_execution'] is not True or trace['DSP_calls'] != 0 or
                trace['CPU_model_fallback'] is not False or not trace['forwards']):
            raise ValueError('Completed independent native W8A8 CPU trace required')
        validate_reference_cache_trace(trace['forwards'])
        self.trace,self.layout_for,self.replay = trace,layout_for,replay
        self.rows=[];self.pending=None;self.previous=None;self.request=-1

    def input(self, payload, contract, request_index):
        if self.pending is not None or len(self.rows)>=len(self.trace['forwards']):
            raise ValueError('Unexpected/extra actual DSP forward')
        expected=self.trace['forwards'][len(self.rows)];q=expected['query']
        if expected['step']!=len(self.rows) or expected['contract']!=contract or expected['request_index']!=request_index:
            raise ValueError('Actual graph ABI/request ownership differs from independent trace')
        layout=self.layout_for(contract,{'query':q});validate_qwen_layout(layout,q)
        description,values=describe_bytes(layout,'inputs',payload,query=q)
        for index in (0,1,2,5):
            if description['fields'][index]!=expected['input']['fields'][index]:
                raise ValueError('Actual native token/position/mask tensor differs at input index '+str(index))
        ids=values[0].reshape(-1).tolist();positions=values[1].reshape(-1).tolist()
        if ids!=expected['token_ids'] or positions!=expected['positions'] or values[2].reshape(-1).tolist()!=positions:
            raise ValueError('Native token/position trajectory diverged')
        if positions!=list(range(positions[0],positions[0]+q)) or not 0<=positions[0]<=positions[-1]<1024:
            raise ValueError('Actual cache positions outside native capacity')
        kv=values[3:5]+values[6:]
        digests=[f['sha256'] for f in description['fields'][3:5]+description['fields'][6:]]
        reset=request_index!=self.request
        if reset:
            if request_index!=self.request+1 or positions[0]!=0 or any(np.any(v.view('<u4')) for v in kv):
                raise ValueError('New native request did not reset every cache byte')
        elif positions[0]!=self.rows[-1]['positions'][-1]+1 or digests!=self.previous:
            raise ValueError('Native cache did not receive previous actual DSP outputs')
        self.pending=dict(expected=expected,layout=layout,payload=payload,values=values,
                          input=description,reset=reset)
        self.request=request_index

    def output(self,payload):
        if self.pending is None:raise ValueError('DSP response without observed actual input')
        p=self.pending;e=p['expected'];layout=p['layout'];positions=e['positions']
        description,values=describe_bytes(layout,'outputs',payload)
        p['output']=description
        incoming=p['values'][3:5]+p['values'][6:];start,end=positions[0],positions[-1]+1
        for before,after in zip(incoming,values[:56],strict=True):
            if not np.isfinite(after).all():raise ValueError('DSP KV contains nonfinite values')
            for region in (slice(0,start),slice(end,1024)):
                if not np.array_equal(before.view('<u4')[:,:,region,:],after.view('<u4')[:,:,region,:]):
                    raise ValueError('DSP modified KV outside active positions')
        if not np.isfinite(values[-1]).all():raise ValueError('DSP logits contain nonfinite values')
        exact_input=p['input']['sha256']==e['input']['sha256']
        exact_output=description['sha256']==e['output']['sha256']
        row=dict(step=e['step'],request_index=e['request_index'],query=e['query'],positions=positions,
                 input=p['input'],output=description,independent_input_bitexact=exact_input,
                 independent_output_bitexact=exact_output,KV_reset_all_zero=p['reset'],
                 KV_continuity_bitexact=not p['reset'],unchanged_KV_bitexact=True,
                 all_outputs_finite=True,same_input_numerical_acceptance=True,
                 independent_native_numerical_acceptance=True,logits_top_token=int(values[-1].argmax()))
        row['differing_input_tensor_indices']=[i for i,(a,b) in enumerate(zip(p['input']['fields'],e['input']['fields'],strict=True)) if a!=b]
        row['differing_output_tensor_indices']=[i for i,(a,b) in enumerate(zip(description['fields'],e['output']['fields'],strict=True)) if a!=b]
        if not (exact_input and exact_output):
            # Replay returns immutable independently computed raw tensors, not
            # asserted metrics or substitutions for production DSP outputs.
            ref=self.replay(e,p['payload'],payload,layout)
            ri,iv=describe_bytes(layout,'inputs',ref['native_input'],query=e['query'])
            ro,ov=describe_bytes(layout,'outputs',ref['native_output'])
            if ri['sha256']!=e['input']['sha256'] or ro['sha256']!=e['output']['sha256']:
                raise ValueError('Independent CPU snapshot is not deterministic against its trace')
            for index in (0,1,2,5):
                if p['input']['fields'][index]!=ri['fields'][index]:
                    raise ValueError('Actual native tokens/positions/mask differ')
            row['independent_valid_input_KV']=[difference(a[:,:,:start,:],b[:,:,:start,:])
                for a,b in zip(incoming,iv[3:5]+iv[6:],strict=True)] if start else []
            if start:accept_metrics(row['independent_valid_input_KV'])
            row['independent_valid_output_KV_and_logits']=numerical_outputs(values,ov,end)
            if not exact_input:
                if ref['same_input_sha256']!=p['input']['sha256']:
                    raise ValueError('Same-input CPU forward did not use actual DSP bytes')
                metrics=[difference(v[:,:,start:end,:],r) for v,r in zip(values[:56],ref['same_input_KV'],strict=True)]
                metrics.append(difference(values[-1],ref['same_input_logits']));accept_metrics(metrics)
                row['same_input_updated_KV_and_logits']=metrics
            row['replay_evidence']=ref['evidence']
        self.rows.append(row);self.previous=[f['sha256'] for f in description['fields'][:56]];self.pending=None
        return row

    def finish(self,run):
        if (self.pending is not None or len(self.rows)!=len(self.trace['forwards']) or
                run['DSP_calls']!=len(self.rows) or len(run['forwards'])!=len(self.rows) or
                run['status']!='ARM_native_dynamic_compiler_handoff_generate_completed' or
                run['process_exit_code']!=0 or not run['native_generate'] or run['CPU_model_fallback']):
            raise ValueError('Native DSP/CPU trajectory or process lifetime incomplete')
        for actual,observed in zip(run['forwards'],self.rows,strict=True):
            e=self.trace['forwards'][observed['step']]
            if run['captures'][actual['AOT_capture_index']]['contract']!=e['contract']:
                raise ValueError('Actual executed capture ABI differs from independent CPU trace')
            if any(actual[key]!=value for key,value in dict(step=observed['step'],
                request_index=e['request_index'],query=e['query'],cache_positions=e['positions'],
                input_token_ids=e['token_ids'],input_sha256=observed['input']['sha256'],
                output_sha256=observed['output']['sha256'],logits_top_token=observed['logits_top_token']).items()):
                raise ValueError('Native runner telemetry differs from actual observed packets')
        if len(run['requests'])!=len(self.trace['requests']):raise ValueError('Native request count differs')
        for actual,expected in zip(run['requests'],self.trace['requests'],strict=True):
            if any(actual[key]!=expected[key] for key in expected):
                raise ValueError('Native generated tokens, stopping or cache ownership differ')
        from hexagon_library_lifecycle import audit_library_events
        return audit_library_events(run['stream_events'],[f['DSP_library_slot'] for f in run['forwards']])
