"""Revisit unchanged-model INT8 timestamps and exact original ARM packets."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from build import ROOT, HERE, entry, sha, write
from profile_model_memory import checked, linked_caller, instrument
from integer_call_timeline import revisit_timeline_packets
from revisit_native_compile_session_abba import require_normal_owner
from native_model_acceptance import require_model_acceptance


def historical_owner(path, script, output, source_sha, revision):
    owner=json.loads(Path(path).read_text())
    require_normal_owner(path,owner['authorized_deadline_unix'])
    command=owner['command'];root=Path(owner['source_root']);source=root/'tools/qwen3'/script
    if (owner['source_revision']!=revision
            or command[1]!=str(source) or command[command.index('--output')+1]!=str(output)):
        raise ValueError('Exact historical producer, normal exit and output required')
    blob=subprocess.check_output(['git','-C',str(root),'show',revision+':tools/qwen3/'+script])
    if hashlib.sha256(blob).hexdigest()!=source_sha:
        raise ValueError('Historical producer executable bytes changed')
    return owner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','index',
                 'build','build-controller','plan','run','run-controller','accepted-run','accepted-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--query',type=int,default=8)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    revision=git('rev-parse','HEAD')
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh frozen independent reviewer required')
    index=read(a.index);model,closure=require_model_acceptance(index)
    if index['deployment_selection']['release_key']!='qualified_global_copy_native_release':
        raise ValueError('The actual independently qualified byte-copy model is required')
    cold=Path(closure['cold_run']);acceptance=read(cold/'acceptance.json')
    if (not a.accepted_run.is_relative_to(cold/'audit')
            or checked(acceptance['audit'])!=a.accepted_audit):
        raise ValueError('Own accepted cold9 ARM-reference evidence required')
    native=read(a.accepted_run);audit=read(a.accepted_audit)
    if (audit['status']!='framework_native_numerical_acceptance_passed'
            or audit['numerical_acceptance'] is not True or audit['run_sha256']!=sha(a.accepted_run)
            or native['process_exit_code'] or native['CPU_model_fallback'] is not False):
        raise ValueError('Actual same-input fresh ARM W8A8 comparison required')
    forward=next(row for row in native['forwards'] if row['query']==a.query)
    build=read(a.build/'result.json');process=read(a.run/'process.json');plan=read(a.plan)
    build_owner=historical_owner(a.build_controller,'profile_model_memory.py',a.build,
                     build['source_sha256'],build['source_revision'])
    historical_owner(a.run_controller,'run_staged_probe.py',a.run,
                     process['source_sha256'],process['source_revision'])
    if (process['status']!='complete' or process['returncode'] or process['output_copy_returncode']
            or process['actual_DSP_calls']!=6 or process['plan']!=entry(a.plan)
            or build['status']!='diagnostic_model_memory_library_built_NOT_device_acceptance'
            or build['DSP_calls'] or build['model_graph_changed'] or build['power_votes_changed']
            or any(build[k] is not True for k in ('model_object_unchanged',
                'profile_state_and_counters_no_preemptible_relocations',
                'all_wrapped_model_callees_resolve'))):
        raise ValueError('Normally completed unchanged diagnostic build and exact6-call board run required')
    if (build.get('integer_call_timeline')is not True
            or build.get('all_original_link_objects_unchanged')is not True
            or build.get('preserved_existing_memcpy')is not None
            or build.get('inspect_fmaf')is not False or build.get('DSP_calls')!=0
            or build.get('query_bounds')!=[2,32] or type(a.query)is not int or not 2<=a.query<=32):
        raise ValueError('Original dynamic model and INT8-only timeline required')
    original_link=read(checked(build['original_link_receipt']))
    if (build['uninstrumented_replay']['byte_identical'] is not True
            or build['uninstrumented_replay']['expected_library']!=original_link['library']
            or sha(checked(build['uninstrumented_replay']['library']))!=forward['DSO_sha256']
            or forward['DSO_sha256']!=index['candidate_libraries'][str((1,1)if a.query==1 else(2,32))]
            or original_link['library']['sha256']!=forward['DSO_sha256']):
        raise ValueError('Byte-identical current accepted baseline must precede diagnostic instrumentation')
    for row in build['original_link_inputs']+build['constants']:
        checked(row)
    checked(build['model_object']);checked(build['library'])
    checked(build['original_wrapper']);checked(build['diagnostic_wrapper'])
    caller=read(Path(build['original_link_receipt']['path']).parents[1]/'caller/result.json')
    if linked_caller(caller,original_link)!=checked(build['original_wrapper']):
        raise ValueError('Actual production caller, transfer helpers and ownership guards changed')
    expected=original_link['command'][:]
    original=checked(build['original_wrapper']);wrapper=checked(build['diagnostic_wrapper'])
    runtime=checked(build['profiler_source']);obj=Path(build['compile_command'][build['compile_command'].index('-o')+1])
    if wrapper.read_text()!=instrument(original.read_text()):
        raise ValueError('Only the original unchanged model invocation may be bracketed')
    expected[expected.index(str(original))]=str(wrapper)
    expected[expected.index('-o')+1]=str(checked(build['library']))
    wrappers=['_mlir_ciface___qwen_i8_multi_matmul']
    if build.get('attention_phase_timeline')is True:
        wrappers+=['_mlir_ciface___hexagon_softmax_exp_compat']
    expected[1:1]=[str(obj)]+['-Wl,--wrap='+symbol for symbol in wrappers]
    if (expected!=build['link_command'] or runtime.read_bytes()!=(compiler_tools()/'runtime/profile/model_integer_timeline.c').read_bytes()
            or runtime not in map(Path,build['compile_command'])
            or any(flag in build['compile_command']for flag in ('-ffast-math','-Ofast'))):
        raise ValueError('Exact original link inputs and only the authenticated INT8 timeline object required')
    for row in plan['files']:
        if sha(row['source'])!=row['sha256']:raise ValueError('Staged source or model bytes changed')
    board=a.run/'board';result=read(board/'result.json')
    parser_source=HERE/'integer_call_timeline.py'
    producer_parser=Path(build_owner['source_root'])/'tools/qwen3/integer_call_timeline.py'
    if (result.get('timeline_parser_sha256')!=sha(parser_source)
            or next((r for r in plan['files']if r['name']=='integer_call_timeline.py'),None)
                !=dict(name='integer_call_timeline.py',source=str(producer_parser),sha256=sha(parser_source))):
        raise ValueError('The actual board parser must be the pinned independently revisited source')
    if result['source_sha256']!=next(row['sha256']for row in plan['files']
            if row['name']=='probe_model_memory_arm.py'):
        raise ValueError('Actually deployed probe differs from its pinned source')
    proof=revisit_timeline_packets(board,build,forward,a.query)
    full=board/Path(result['preserved_complete_profile_output']['path']).name
    original=a.accepted_run.parent/f"forward-{forward['step']:02d}-q{a.query}/output.bin"
    payload=a.accepted_run.parent/f"forward-{forward['step']:02d}-q{a.query}/input.bin"
    if (sha(payload)!=forward['input_sha256'] or sha(full)!=forward['output_sha256']
            or full.stat().st_size!=235488768 or full.read_bytes()!=original.read_bytes()):
        raise ValueError('Retained complete input/logits/56KV must match the original ARM-exact packet')
    with tempfile.TemporaryDirectory(prefix='qwen-current-profile-fnv-')as temporary:
        library=Path(temporary)/'fnv.so';source=compiler_tools()/'runtime/profile/physical_gqa_pv_reference.c'
        subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
        fnv=ctypes.CDLL(str(library)).qwen_physical_scalar_fnv32
        fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
        data=payload.read_bytes();scalar=int(fnv(ctypes.c_char_p(data),len(data)))
        if scalar!=forward['invoke']['AP_input_fnv1a32']:raise ValueError('Independent full-input FNV differs')
    config=read(a.config);pid=result['ready']['ARM_pid']
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],
        shlex.join([config['board']['python'],'-c',
        'from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(pid)])],check=True)
    if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):
        raise ValueError('Independent reviewer source changed')
    proof.update(status='current_global_copy_INT8_call_timeline_original_ARM_packets_raw_RPC_and_normal_owners_independently_revisited_DIAGNOSTIC',
        source_root=str(ROOT),source_revision=revision,source=entry(__file__),model_index=entry(a.index),qualified_model=model,query=a.query,
        build=entry(a.build/'result.json'),build_controller=entry(a.build_controller),
        run_controller=entry(a.run_controller),board_process=entry(a.run/'process.json'),
        complete_diagnostic_output_file_retained=True,complete_output=entry(full),input=entry(payload),
        original_ARM_exact_output=entry(original),independent_full_input_FNV=scalar,
        independent_FNV_source=entry(source),performance_acceptance=False,shared_default_changed=False,
        completed_unix=time.time())
    write(a.output,proof)


if __name__=='__main__':main()
