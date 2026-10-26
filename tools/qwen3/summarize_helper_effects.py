"""Revisit raw helper-effect evidence and recompute bounded ABBA timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from build import sha,write
from probe_helper_effects_arm import parse_response


def event(row,input_bytes,output_bytes):
    e=row['actual_invoke']
    if (e['stage']!='invoke' or any(e[k] for k in ('rpc_result','DSP_result','output_result'))
            or e['input_bytes']!=input_bytes or e['output_bytes']!=output_bytes
            or not e['kernel_us'] or not math.isfinite(e['seconds']) or e['seconds']<=0
            or len({e[k] for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')})!=1):
        raise ValueError('Incomplete actual invocation or exact checksum evidence')
    return e


def paired(rows,baseline,candidate,key):
    old=statistics.mean(row[key] for row in rows if row['library']==baseline)
    new=statistics.mean(row[key] for row in rows if row['library']==candidate)
    if old<=0 or new<=0:raise ValueError('Positive paired measurements required')
    return dict(baseline=old,candidate=new,ratio=new/old,reduction_percent=100*(1-new/old))


def primitive(path,d):
    if (d['status']!='actual_DSP_scoped_SiLU_effects_original_bits_and_finite_ARM_sequence_passed'
            or d['process_exit_code'] or d['DSP_calls']!=26 or d['model_inference']
            or d['power_votes_changed'] or len(d['rows'])!=24 or len(d['diagnostic_rows'])!=2):
        raise ValueError('Complete actual primitive evidence required')
    dynamic=d.get('dynamic_row',False)
    extent=d.get('row_extent',1)
    if (type(dynamic) is not bool or type(extent) is not int or not 1<=extent<=32
            or (not dynamic and extent!=1)):
        raise ValueError('Explicit bounded dynamic primitive extent required')
    elements=3072*extent
    input_bytes,output_bytes=128+8*elements,128+4*elements
    if dynamic and d.get('FP32_words_per_iteration')!=elements:
        raise ValueError('Complete dynamic FP word coverage required')
    measurements={};slots={};ordinals=[]
    for pattern in dict.fromkeys(r['pattern'] for r in d['rows']):
        rows=[r for r in d['rows'] if r['pattern']==pattern]
        if [r['library'] for r in rows]!=['unknown','declaration-contract','declaration-contract','unknown']:
            raise ValueError('Original/candidate ABBA order changed')
        packet=(path.parent/(pattern+'-input.bin')).read_bytes();arm=(path.parent/(pattern+'-ARM.bin')).read_bytes()
        if dynamic:
            import struct
            if (len(packet)!=input_bytes or len(arm)!=4*elements or packet[:8]!=b'QWSILU2\0'
                    or struct.unpack_from('<II',packet,8)!=(rows[0]['repeats'],extent) or any(packet[16:128])):
                raise ValueError('Dynamic packet header, all words or independent ARM extent changed')
        original=None;values=[]
        for i,r in enumerate(rows):
            if r['index']!=i or r['input_sha256']!=hashlib.sha256(packet).hexdigest() or r['diagnostic_call_counter']:
                raise ValueError('ABBA input binding or unwrapped timing changed')
            raw=path.parent/(pattern+'-'+str(i)+'-DSP.bin')
            data=raw.read_bytes()
            if sha(raw)!=r['full_output_sha256'] or hashlib.sha256(data[128:]).hexdigest()!=r['value_sha256']:
                raise ValueError('Raw DSP result changed')
            parsed=parse_response(data,r['repeats'],0,extent)
            if any(parsed[k]!=r[k] for k in parsed):raise ValueError('Raw actual timing differs from report')
            if original is None:original=data[128:]
            if (data[128:]!=original or (r['independent_ARM_match_required'] and data[128:]!=arm)
                    or r['independent_ARM_bits_identical']!=(data[128:]==arm)):
                raise ValueError('Original DSP or independent finite ARM bits changed')
            if dynamic and (r.get('all_FP32_words_identical_to_original_DSP') is not True
                    or r.get('FP32_words_per_iteration')!=elements):
                raise ValueError('Reported dynamic row word coverage changed')
            e=event(r,input_bytes,output_bytes);ordinals.append(e['ordinal'])
            if r['library'] in slots and slots[r['library']]!=e['library_slot']:raise ValueError('Actual library slot changed')
            slots[r['library']]=e['library_slot']
            values.append(dict(library=r['library'],us=r['elapsed_us']/r['repeats']))
        measurements[pattern]=paired(values,'unknown','declaration-contract','us')
    for row,calls in zip(d['diagnostic_rows'],[3*elements,elements]):
        if row['library'] not in ('unknown-counted','declaration-contract-counted') or not row['diagnostic_call_counter']:
            raise ValueError('Separate diagnostic count evidence required')
        raw=(path.parent/(row['library']+'-DSP.bin')).read_bytes();parsed=parse_response(raw,1,calls,extent)
        if any(parsed[k]!=row[k] for k in parsed):raise ValueError('Raw diagnostic count changed')
        baseline=next(r for r in d['rows'] if r['pattern']=='finite-seeded')
        if hashlib.sha256(raw[128:]).hexdigest()!=baseline['value_sha256']:raise ValueError('Counted result changed')
        e=event(row,input_bytes,output_bytes);ordinals.append(e['ordinal']);slots[row['library']]=e['library_slot']
    if sorted(ordinals)!=list(range(1,27)) or len(set(slots.values()))!=4:raise ValueError('Actual calls or distinct libraries incomplete')
    result=dict(status='primitive_raw_bytes_events_ABBA_and_separate_counts_revisited',DSP_calls=26,
        all_original_DSP_output_words_identical=True,all_finite_independent_ARM_output_words_identical=True,
        nonfinite_original_DSP_preserved_preexisting_ARM_NaN_bits_differ=True,
        actual_calls_per_3072_element_iteration=dict(original=9216,candidate=3072),
        timing_unit='microseconds_per_iteration_including_result_allocation_compare_copy_free',
        measurements=measurements,model_inference_acceptance=False)
    if dynamic:
        finite=['finite-boundaries','finite-range','finite-adjacent','finite-seeded','signed-zero-subnormal']
        if set(measurements)!=set(finite+['nonfinite-original-DSP-contract']):
            raise ValueError('Original complete finite and special primitive patterns required')
        if any(r['independent_ARM_match_required']!=(r['pattern'] in finite) for r in d['rows']):
            raise ValueError('Finite ARM oracle requirement changed')
        result.pop('actual_calls_per_3072_element_iteration')
        result.update(status='dynamic_primitive_raw_all_row_bytes_events_ABBA_and_separate_counts_revisited',
            dynamic_row=True,row_extent=extent,FP32_words_per_iteration=elements,
            actual_calls_per_iteration=dict(original=3*elements,candidate=elements),
            positive_primitive_gain=all(measurements[name]['ratio']<0.995 for name in finite))
    return result


def model(d,guarded_copy=False,dynamic=False,integer=False,silu=False,fixed512=False):
    expected_status=('actual_same_input_dynamic_fixed512_copy_model_full_packet_gate_passed' if fixed512 else
                     'actual_same_input_dynamic_SiLU_effect_model_full_packet_gate_passed' if silu else
                     'actual_same_input_dynamic_integer_model_full_packet_gate_passed' if integer else
                     'actual_same_input_dynamic_guarded_copy_model_full_packet_gate_passed' if dynamic else
                     'actual_same_input_q1_guarded_copy_model_full_packet_gate_passed' if guarded_copy
                     else 'actual_same_input_q1_SiLU_effect_model_full_packet_gate_passed')
    if (d['status']!=expected_status
            or d['process_exit_code'] or d['inline_CPU_observer'] or d['diagnostic_instrumentation']
            or d['power_votes_changed'] or d['DSP_calls']!=4*d['rounds'] or len(d['rows'])!=d['DSP_calls']):
        raise ValueError('Complete unwrapped model early gate required')
    if dynamic and (d.get('candidate_kind')!=('fixed512-copy' if fixed512 else 'multi-silu' if silu else 'multi-integer' if integer else 'guarded-copy-dynamic') or type(d.get('query')) is not int
                    or not 2<=d['query']<=32 or d['output_bytes']!=235488768
                    or not 234885520<d['input_bytes']<=235012880):
        raise ValueError('Original bounded complete dynamic packet identity required')
    values=[];slots={}
    for ordinal,row in enumerate(d['rows'],1):
        expected=['baseline','candidate','candidate','baseline'][(ordinal-1)%4]
        if (row['library']!=expected or row['round']!=(ordinal-1)//4 or row['index']!=(ordinal-1)%4
                or not row['all_logits_and_56_KV_words_bitexact']
                or row['full_output_sha256']!=d['original_complete_output_sha256']):raise ValueError('Full original model output or ABBA order changed')
        e=event(row,d['input_bytes'] if dynamic else 234885520,235488768)
        if e['ordinal']!=ordinal:raise ValueError('Actual model ordinal changed')
        if expected in slots and slots[expected]!=e['library_slot']:raise ValueError('Model library slot changed')
        slots[expected]=e['library_slot']
        values.append(dict(library=expected,kernel_us=e['kernel_us'],rpc_seconds=e['seconds'],
            callback_seconds=e['profile']['AP_total_seconds']))
    if len(set(slots.values()))!=2:raise ValueError('Two actual model libraries required')
    status=('same_input_full_q1_output_and_unwrapped_guarded_copy_ABBA_revisited' if guarded_copy
            else 'same_input_full_q1_output_and_unwrapped_model_ABBA_revisited')
    if dynamic:status='same_input_full_dynamic_output_and_unwrapped_guarded_copy_ABBA_revisited'
    if integer:status='same_input_full_dynamic_output_and_unwrapped_integer_ABBA_revisited'
    if silu:status='same_input_full_dynamic_output_and_unwrapped_SiLU_effect_ABBA_revisited'
    if fixed512:status='same_input_full_dynamic_output_and_unwrapped_fixed512_copy_ABBA_revisited'
    result=dict(status=status,DSP_calls=d['DSP_calls'],
        full_original_logits_and_56_KV_byteexact=True,
        measurements={k:paired(values,'baseline','candidate',k) for k in ('kernel_us','rpc_seconds','callback_seconds')},
        scope=('raw_dynamic_replay_early_gate_NOT_native_generation_or_original_state_acceptance' if dynamic
               else 'raw_q1_replay_early_gate_NOT_capture_startup_native_generation_or_original_state_acceptance'),
        model_inference_acceptance=False)
    if dynamic:result['query']=d['query']
    return result


def copy_primitive(path,d,fixed512=False):
    from probe_guarded_copy_arm import cases,packet_and_reference,guard_packet_and_reference
    from probe_guarded_copy_arm import parse_response as parse_copy
    count=82 if fixed512 else 66
    status=('actual_DSP_fixed512_copy_accepted_baseline_distinct_blocks_and_DSP32_guards_passed' if fixed512 else
            'actual_DSP_guarded_copy_original_and_independent_bytes_and_DSP32_guard_queries_passed')
    if (d['status']!=status
            or d['process_exit_code'] or d['DSP_calls']!=count or d['FP_evaluation_changed']
            or d['power_votes_changed'] or len(d['rows'])!=count-2 or len(d['diagnostic_rows'])!=2):
        raise ValueError('Complete actual byte-copy and DSP32 integer guard evidence required')
    if fixed512 and (d.get('fixed512') is not True or d.get('source_blocks_distinct') is not True
                     or d.get('baseline_kind')!='actual_accepted_guarded_copy_runtime'):
        raise ValueError('Actual accepted guarded baseline and distinct vector block inputs required')
    measurements={};slots={};ordinals=[]
    for case in cases(fixed512):
        rows=[r for r in d['rows'] if r['case']==case[0]]
        if [r['library'] for r in rows]!=['original','candidate','candidate','original']:
            raise ValueError('Byte-copy original/candidate ABBA order differs')
        payload,expected=packet_and_reference(case,rows[0]['repeats'],fixed512)
        if ((path.parent/(case[0]+'-input.bin')).read_bytes()!=payload
                or (path.parent/(case[0]+'-Python-integer-oracle.bin')).read_bytes()!=expected):
            raise ValueError('Recorded descriptor or independent byte oracle differs')
        values=[]
        for index,row in enumerate(rows):
            raw=path.parent/(case[0]+'-'+str(index)+'-DSP.bin');data=raw.read_bytes()
            parsed=parse_copy(data,row['repeats'])
            if (row['index']!=index or row['descriptor']!=list(case[1:]) or row['diagnostic_hooks']
                    or row['input_sha256']!=hashlib.sha256(payload).hexdigest()
                    or sha(raw)!=row['full_output_sha256'] or data[128:]!=expected
                    or row['value_sha256']!=hashlib.sha256(expected).hexdigest()
                    or any(parsed[k]!=row[k] for k in parsed)):
                raise ValueError('Actual full destination bytes, sentinel or timing differs')
            e=event(row,4352,4224);ordinals.append(e['ordinal'])
            if e['kernel_us']!=parsed['elapsed_us']:raise ValueError('Reported kernel time differs from raw output')
            if row['library'] in slots and slots[row['library']]!=e['library_slot']:raise ValueError('Actual copy library slot differs')
            slots[row['library']]=e['library_slot']
            values.append(dict(library=row['library'],us=parsed['elapsed_us']/parsed['repeats']))
        measurements[case[0]]=paired(values,'original','candidate','us')
    payload,expected=guard_packet_and_reference(1024)
    if ((path.parent/'integer-checks-input.bin').read_bytes()!=payload
            or (path.parent/'integer-checks-Python-integer-oracle.bin').read_bytes()!=expected):
        raise ValueError('Exact integer guard diagnostic input or oracle differs')
    for index,row in enumerate(d['diagnostic_rows']):
        raw=(path.parent/('integer-checks-'+str(index)+'-DSP.bin')).read_bytes()
        parsed=parse_copy(raw,1024,guard=True)
        if (row['index']!=index or row['library']!='candidate-checks' or not row['diagnostic_hooks']
                or row['integer_pairs']!=32 or raw[128:]!=expected or any(parsed[k]!=row[k] for k in parsed)):
            raise ValueError('Actual native DSP32 overflow classification or products differ')
        e=event(row,640,896);ordinals.append(e['ordinal']);slots[row['library']]=e['library_slot']
    if sorted(ordinals)!=list(range(1,count+1)) or len(set(slots.values()))!=3:
        raise ValueError('Actual copy and separate integer diagnostic calls incomplete')
    result=dict(status='guarded_copy_raw_destination_bytes_sentinels_DSP32_guards_and_ABBA_revisited',DSP_calls=count,
        all_4096_original_and_independent_destination_bytes_exact=True,all_unwritten_sentinel_bytes_exact=True,
        native_DSP_size_t_bytes=4,all_32_integer_guard_pairs_exact_to_unbounded_math=True,
        FP_values_not_interpreted_or_changed=True,diagnostic_hooks_absent_from_unwrapped_timing_libraries=True,
        measurements=measurements,model_inference_acceptance=False)
    if fixed512:
        result.update(status='fixed512_copy_distinct_blocks_destination_sentinels_DSP32_guards_and_ABBA_revisited',
            baseline_kind='actual_accepted_guarded_copy_runtime',source_blocks_distinct=True,
            positive_primitive_gain=all(v['ratio']<.995 for k,v in measurements.items() if k.startswith('hot-512')),
            scope='tiny_copy_primitive_NOT_full_model_or_native_state_acceptance')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('primitive','model','copy-primitive','fixed512-copy-primitive','copy-model','copy-dynamic-model','multi-integer-model','multi-silu-model','fixed512-copy-model'),required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--run-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or sha(a.run)!=a.run_sha256:raise ValueError('Fresh summary and pinned completed proof required')
    d=json.loads(a.run.read_text())
    if a.kind=='primitive':result=primitive(a.run,d)
    elif a.kind in ('copy-primitive','fixed512-copy-primitive'):result=copy_primitive(a.run,d,a.kind=='fixed512-copy-primitive')
    else:result=model(d,a.kind in ('copy-model','copy-dynamic-model'),a.kind in ('copy-dynamic-model','multi-integer-model','multi-silu-model','fixed512-copy-model'),a.kind=='multi-integer-model',a.kind=='multi-silu-model',a.kind=='fixed512-copy-model')
    result['run']=dict(path=str(a.run),sha256=a.run_sha256);result['source_sha256']=sha(__file__)
    write(a.output,result);print(json.dumps(result['measurements'],indent=2))


if __name__=='__main__':main()
