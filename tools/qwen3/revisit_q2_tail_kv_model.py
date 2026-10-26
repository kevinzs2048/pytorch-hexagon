"""Re-read all complete Q2 model inputs/outputs, raw RPC traces and normal owners."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import struct
import subprocess
import tempfile
import time
from build import ROOT, HERE, entry, sha, write
from profile_model_memory import checked
from native_model_acceptance import require_model_acceptance
from revisit_q2_dispatch_preflight import actual_owner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','run','controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    git = lambda *args: subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh frozen independent model reviewer required')
    result = read(a.run/'result.json')
    actual_owner(a.controller,result,'run_q2_tail_kv_model.py',a.run)
    preparation_path = checked(result['preparation'])
    preparation = read(preparation_path)
    actual_owner(checked(result['preparation_controller']),preparation,
                 'prepare_q2_tail_kv_model.py',preparation_path.parent)
    m = read(checked(result['manifest']))
    model,_=require_model_acceptance(read(checked(m['index'])))
    if model!=m['qualified_baseline_model']:raise ValueError('Qualified numerical baseline differs')
    link_path = checked(m['link']); link = read(link_path)
    actual_owner(checked(m['link_controller']),link,'link_q2_tail_model.py',link_path.parent)
    code_path=checked(link['model_code']);code=read(code_path)
    actual_owner(checked(link['model_controller']),code,'compile_q2_tail_model_candidate.py',code_path.parent)
    sdk_path=checked(code['SDK']);sdk=read(sdk_path)
    actual_owner(checked(code['SDK_controller']),sdk,'build_q2_tail_model_sdk.py',sdk_path.parent)
    if (link['status']!='Q2_identical_tail_current_model_replayed_new_primary_and_tested_guard_linked_NOT_numerical_acceptance'
            or code['status']!='one_existing_Q2_identical_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code['model_PRIMARY_compilations']!=1 or len(code['unchanged_constant_objects'])!=3
            or link['current_baseline_byte_replayed']is not True
            or link['actual_tested_physical_library_byte_replayed']is not True
            or link['original_caller_runtime_FP_helpers_and_constants_unchanged']is not True):
        raise ValueError('Actual one-primary candidate and exact baseline/runtime/constant replay required')
    for ref in link['original_link_inputs']+link['guard_objects']:checked(ref)
    for ref in m['libraries'].values():
        checked(ref)
    if (m['libraries']['baseline'] != link['baseline_library']
            or m['unchanged_q1_library'] != read(checked(m['index']))['candidate_libraries']['(1, 1)']
            or result['status'] != 'actual_Q2_nonempty_KV_tail_model_ABBA12_completed_PENDING_independent_review'
            or result['DSP_calls'] != 12 or result['board_returncode'] or result['model_PRIMARY_compilations']
            or result['model_inference_acceptance'] or result['performance_acceptance']):
        raise ValueError('Actual one-primary candidate and complete normally owned ABBA12 required')
    board = read(checked(result['board']))
    checked(result['board_archive'])
    if (board['status'] != 'actual_Q2_nonempty_KV_tail_model_ABBA12_full_logits_and_56KV_bitexact_NOT_generation_acceptance'
            or board['DSP_calls'] != 12 or board['process_exit_code'] or board['cases'] != [6,8,32]
            or board['manifest_sha256'] != result['manifest']['sha256']
            or board['source_sha256'] != result['board_source']['sha256'] or not board['workdir_removed']
            or board['diagnostic_instrumentation'] or board['power_votes_changed']
            or board['model_inference_acceptance'] or board['performance_acceptance']):
        raise ValueError('Exact actual complete numerical comparison, original votes and normal close required')
    row=m['row_guard'];build=read(checked(row['build']));review=read(checked(row['review']))
    actual_owner(checked(row['build_controller']),build,'build_q2_tail_row_guard.py',Path(row['build']['path']).parent)
    actual_owner(checked(row['review_controller']),review,'revisit_q2_tail_row_guard.py',Path(row['review']['path']))
    original_model=read(checked(row['original_model_review']))
    actual_owner(checked(row['original_model_review_controller']),original_model,'revisit_q2_tail_model_comparison.py',Path(row['original_model_review']['path']))
    original_manifest=read(checked(row['original_manifest']))
    if (review['DSP_calls']!=256 or review['receipt']!=row['build'] or review['primitive_positive_gain']is not True
            or original_model['DSP_calls']!=12 or original_model['manifest']!=row['original_manifest']
            or original_manifest['link']!=m['link'] 
            or original_manifest['libraries']!=dict(baseline=link['baseline_library'],candidate=link['library'])
            or row['primary']!=link['candidate_primary'] or row['guard_object']!=build['integer_guard_object']
            or row['replay']['sha256']!=link['library']['sha256'] or build['isolated_operator_compilations']):
        raise ValueError('Exact target-proved new guard and original actually compared full model required')
    checked(row['primary']);checked(row['guard_object']);checked(row['replay']);checked(build['integer_guard_variant'])
    commands=row['commands'];expected=link['commands'][-1]['argv'][:]
    replay=expected[:];replay[replay.index('-o')+1]=row['replay']['path']
    old=next(r['path']for r in link['guard_objects']if Path(r['path']).name=='q2_qk_positive_zero_extent.c.o')
    if expected.count(old)!=1:raise ValueError('Exactly one replaced integer guard object required')
    expected[expected.index(old)]=row['guard_object']['path'];expected[expected.index('-o')+1]=m['libraries']['candidate']['path']
    if (len(commands)!=2 or any(r['returncode']for r in commands)or commands[0]['argv']!=replay or commands[1]['argv']!=expected
            or build['integer_guard_variant']['sha256']!=sha(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent_row_major.c')):
        raise ValueError('Only the exact newly target-proved guard object may change; original FP primary and caller remain exact')
    previous=read(checked(m['previous_model_review']))
    actual_owner(checked(m['previous_model_review_controller']),previous,'revisit_q2_tail_row_model.py',Path(m['previous_model_review']['path']))
    previous_manifest=read(checked(m['previous_model_manifest']))
    if (previous['manifest']!=m['previous_model_manifest']or previous['DSP_calls']!=12
            or previous_manifest['libraries']!=m['libraries']or previous_manifest['row_guard']!=m['row_guard']
            or previous_manifest['contract']!=m['contract']or m['packet_transform']!=entry(HERE/'nonempty_prefill_packet.py')
            or m['original_nonempty_positions']!=list(range(8,14))or m['occupied_KV_positions']!=8
            or m['all_original_56_KV_inputs_preserved']is not True):
        raise ValueError('Prior checked identical model and explicitly preserved actual nonempty prefix required')
    native=read(checked(m['accepted_run']));audit=read(checked(m['accepted_audit']))
    forward=next(r for r in native['forwards']if r['query']==6 and r['cache_positions']==list(range(8,14)))
    packet=m['packets'][0]
    if (audit['numerical_acceptance']is not True or audit['run_sha256']!=m['accepted_run']['sha256']
            or native['process_exit_code']or native['CPU_model_fallback']
            or forward['DSO_sha256']!=m['libraries']['baseline']['sha256']or packet['query']!=6
            or packet['input']['sha256']!=forward['input_sha256']or packet['output']['sha256']!=forward['output_sha256']):
        raise ValueError('Actual independently ARM-checked nonempty original forward required')
    source = checked(result['board_source'])
    blob = subprocess.check_output(['git','-C',result['source_root'],'show',
            result['source_revision']+':tools/qwen3/probe_q2_tail_kv_model_arm.py'])
    if hashlib.sha256(blob).hexdigest() != sha(source):
        raise ValueError('Actually deployed probe must match its original committed source')
    raw = a.run/'board/stream/stream.stdout.txt'
    events = [json.loads(line) for line in raw.read_text().splitlines()]
    if (events != board['stream_events'] or events[-1]['stage'] != 'close'
            or [e for e in events if e.get('stage')=='invoke'] != [r['actual_invoke'] for r in board['rows']]
            or len(board['rows']) != 12
            or any(e.get(k,0) for e in events for k in ('rpc_result','DSP_result','output_result','loader_result'))):
        raise ValueError('Every actual invoke, original raw RPC trace and normal close required')
    ready = [e for e in events if e.get('stage')=='ready']
    if len(ready) != 1 or ready[0] != board['ready']:
        raise ValueError('Exact real AP owner identity required')
    import sys
    sys.path[:0] = [str(ROOT),str(ROOT/'qwen3/runtime/arm')]
    from nonempty_prefill_packet import resize_nonempty_prefill_packet
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for
    original = checked(next(r for r in m['packets'] if r['query']==6)['input']).read_bytes()
    fnv_source = compiler_tools()/'runtime/profile/physical_gqa_pv_reference.c'
    proofs = []; timings = []
    with tempfile.TemporaryDirectory(prefix='qwen3-q2-model-fnv-') as temporary:
        library = Path(temporary)/'reference.so'
        subprocess.run(['cc','-O2','-fPIC','-shared',str(fnv_source),'-lm','-o',str(library)],check=True,capture_output=True)
        fnv = ctypes.CDLL(str(library)).qwen_physical_scalar_fnv32
        fnv.argtypes = [ctypes.c_void_p,ctypes.c_uint]; fnv.restype = ctypes.c_uint32
        for case,query in enumerate(m['queries']):
            group = board['rows'][case*4:case*4+4]
            if [(r['query'],r['index'],r['role']) for r in group] != [
                    (query,i,role) for i,role in enumerate(('baseline','candidate','candidate','baseline'))]:
                raise ValueError('Exact complete ABBA order required for each actual runtime query')
            input_path = a.run/'board'/group[0]['input_file']
            payload = input_path.read_bytes()
            accepted = next((r for r in m['packets'] if r['query']==query),None)
            if accepted:
                checked(accepted['input']); checked(accepted['output'])
                if sha(input_path) != accepted['input']['sha256']:
                    raise ValueError('Original independently CPU-accepted full input changed')
                expected = accepted['output']['sha256']
            else:
                if payload != resize_nonempty_prefill_packet(m['contract'],original,query):
                    raise ValueError('Derived q8/q32 must preserve the actual prior KV and original positions')
                expected = sha(a.run/'board'/group[0]['output_file'])
            layout = layout_for(m['contract'],{'query':query})
            scalar = int(fnv(ctypes.c_char_p(payload),len(payload)))
            if (payload[:16] != b'HXDY\1\0\0\0'+struct.pack('<q',query)
                    or len(payload) != 16+layout['input_bytes'] or layout['output_bytes'] != m['output_bytes']):
                raise ValueError('Original full dynamic model ABI required')
            outputs = []
            for row in group:
                output = a.run/'board'/row['output_file']; event = row['actual_invoke']
                if (row['input_file'] != input_path.name or row['input_sha256'] != sha(input_path)
                        or row['output_sha256'] != sha(output) or sha(output) != expected
                        or output.stat().st_size != layout['output_bytes']
                        or row['all_logits_and_56_KV_byteexact'] is not True
                        or event['input_bytes'] != len(payload) or event['output_bytes'] != layout['output_bytes']
                        or any(event[k] != scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))
                        or not event['kernel_us']):
                    raise ValueError('Every logits/KV byte, complete input/output and independently computed FNV must match')
                outputs.append(entry(output))
            proofs.append(dict(query=query,input=entry(input_path),outputs=outputs,
                               scalar_FNV=scalar,recorded_qualified_ARM_W8A8_reference_preserved=accepted is not None))
            baseline = statistics.median(r['actual_invoke']['kernel_us'] for r in group if r['role']=='baseline')
            candidate = statistics.median(r['actual_invoke']['kernel_us'] for r in group if r['role']=='candidate')
            timings.append(dict(query=query,baseline_kernel_us=baseline,candidate_kernel_us=candidate,
                                candidate_over_baseline=candidate/baseline))
    pid = ready[0]['ARM_pid']; b = read(a.config)['board']
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',b['ssh_target'],shlex.join([b['python'],'-c',
        'from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(pid)])],check=True,timeout=15)
    write(a.output,dict(status='actual_Q2_nonempty_KV_tail_model_ABBA12_logits_56KV_and_3FNV_independently_revisited_NOT_generation_acceptance',
          source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
          run=entry(a.run/'result.json'),controller=entry(a.controller),manifest=result['manifest'],
          qualified_baseline_model=m['qualified_baseline_model'],link=m['link'],row_guard=m['row_guard'],raw=entry(raw),cases=proofs,
          private_engineering_timings=timings,original_ARM_W8A8_queries=[6],derived_DSP_reference_queries=[8,32],
          unchanged_q1_library=m['unchanged_q1_library'],DSP_calls=12,new_DSP_calls=0,model_PRIMARY_compilations=0,
          complete_model_packet_numerical_comparison=True,occupied_KV_positions=8,all_original_56_KV_inputs_preserved=True,model_inference_acceptance=False,
          generation_regression=False,performance_acceptance=False,actual_AP_owner_gone=pid,
          independent_FNV_source=entry(fnv_source),completed_unix=time.time()))


if __name__ == '__main__':
    main()
