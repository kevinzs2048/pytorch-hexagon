"""Recompute byte reference, full raw packets, FNV and real global-copy owners."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
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
from revisit_q2_dispatch_preflight import actual_owner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','build','build-controller','packet','packet-controller','plan','run','controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    revision=git('rev-parse','HEAD')
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh frozen independent operator review required')
    build=read(a.build/'result.json');packet=read(a.packet/'result.json');process=read(a.run/'process.json')
    actual_owner(a.build_controller,build,'build_global_hvx_copy.py',a.build)
    actual_owner(a.packet_controller,dict(source_root=str(Path(packet['source']['path']).parents[2]),
        source_revision=read(a.packet_controller)['source_revision'],source=packet['source']),
        'prepare_caller_byte_copy_packet.py',a.packet)
    actual_owner(a.controller,dict(source_root=str(ROOT),source_revision=process['source_revision'],
        source=dict(path=str(HERE/'run_staged_probe.py'),sha256=process['source_sha256'])),
        'run_staged_probe.py',a.run)
    if (build['status']!='model_local_global_HVX_copy_both_accepted_primaries_relinked_NOT_device_or_generation_acceptance'
            or build['new_model_PRIMARY_compilations'] or build['DSP_calls']
            or build['all_original_model_primaries_and_constants_unchanged'] is not True
            or build['original_fixed512_text_unchanged'] is not True
            or packet['case_count']!=498 or packet['DSP_calls'] or process['status']!='complete'
            or process['returncode'] or process['output_copy_returncode'] or process['actual_DSP_calls']!=4
            or process['plan']!=entry(a.plan)):
        raise ValueError('Unchanged models and normally completed exact4-copy operator scope required')
    for model in build['models']:
        for row in model['original_link_inputs']+model['constants']:
            checked(row)
        for name in ('primary','baseline','baseline_replay','candidate','accepted_HVX_runtime','accepted_HVX_source'):
            checked(model[name])
        if (model['baseline_replay']['sha256']!=model['baseline']['sha256']
                or model['delegate']['original_text_instructions_identical'] is not True
                or model['delegate']['delegate_symbol']!='qwen_global_copy_preserved_memcpy'
                or model['all_copy_symbols_locally_bound'] is not True):
            raise ValueError('Original fallback instructions, both baselines and local bindings must be preserved')
    payload=checked(packet['packet']).read_bytes();expected_file=checked(packet['expected'])
    count,reserved=struct.unpack_from('<II',payload,8)
    if payload[:8]!=b'QWCPY1\0\0' or count!=498 or reserved or len(packet['cases'])!=count:
        raise ValueError('The complete original498 alignment/length classes are required')
    expected=bytearray(b'\xa7'*expected_file.stat().st_size)
    for index,case in enumerate(packet['cases']):
        source,destination,size=struct.unpack_from('<III',payload,16+12*index)
        if (source<16+12*count or source+size>len(payload) or destination+size>len(expected)
                or size!=case['bytes'] or source%128!=case['source_alignment']
                or destination%128!=case['destination_alignment']):
            raise ValueError('Actual complete byte ranges and canonical alignment cases changed')
        expected[destination:destination+size]=payload[source:source+size]
    if bytes(expected)!=expected_file.read_bytes():raise ValueError('Independent full CPU destination/canary reference differs')
    plan=read(a.plan)
    for row in plan['files']:
        if sha(row['source'])!=row['sha256']:raise ValueError('Actually staged source or byte data changed')
    board=a.run/'board';result=read(board/'result.json');events_path=board/'stream.stdout.txt'
    events=[json.loads(line)for line in events_path.read_text().splitlines()]
    if (result['status']!='model_local_global_HVX_copy_ABBA4_all_bytes_canaries_exact_NOT_model_acceptance'
            or result['DSP_calls']!=4 or result['DSP_model_calls'] or result['process_exit_code']
            or result['model_inference_acceptance'] or result['performance_acceptance']
            or result['stream_events']!=events or result['build_sha256']!=sha(a.build/'result.json')
            or result['source_sha256']!=sha(HERE/'probe_global_copy_arm.py')
            or result['packet_sha256']!=packet['packet']['sha256']
            or result['expected_sha256']!=packet['expected']['sha256']
            or events[-1]['stage']!='close' or any(e.get(k,0)for e in events
                for k in ('rpc_result','DSP_result','output_result','loader_result'))):
        raise ValueError('Actual exact operator packets, raw RPC trace and normal close required')
    with tempfile.TemporaryDirectory(prefix='qwen-global-copy-fnv-')as temporary:
        source=compiler_tools()/'runtime/profile/physical_gqa_pv_reference.c';library=Path(temporary)/'reference.so'
        subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
        fnv=ctypes.CDLL(str(library)).qwen_physical_scalar_fnv32
        fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
        scalar=int(fnv(ctypes.c_char_p(payload),len(payload)))
    invokes=[e for e in events if e['stage']=='invoke'];outputs=[]
    if len(invokes)!=4 or len(result['rows'])!=4:raise ValueError('Four actual unique operator invokes required')
    for index,(row,event)in enumerate(zip(result['rows'],invokes)):
        role=('baseline','candidate','candidate','baseline')[index];path=board/row['output_file']
        if (row['index']!=index or row['role']!=role or row['actual_invoke']!=event
                or event['ordinal']!=index+1 or event['library_slot']!=(0,1,1,0)[index]
                or event['input_bytes']!=len(payload) or event['output_bytes']!=len(expected)
                or any(event[k]!=scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32'))
                or row['output_sha256']!=packet['expected']['sha256']
                or sha(path)!=packet['expected']['sha256'] or path.read_bytes()!=bytes(expected)):
            raise ValueError('Every original/candidate byte, canary, role, raw event and independently calculated FNV must match')
        outputs.append(entry(path))
    timing={role:statistics.mean(row['actual_invoke']['kernel_us']for row in result['rows']if row['role']==role)
            for role in ('baseline','candidate')}
    ratio=timing['candidate']/timing['baseline'];config=read(a.config)
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',config['board']['ssh_target'],
        shlex.join([config['board']['python'],'-c',
        'from pathlib import Path;import sys;assert not Path("/proc",sys.argv[1]).exists()',str(result['ready']['ARM_pid'])])],check=True)
    if git('rev-parse','HEAD')!=revision or git('status','--porcelain'):raise ValueError('Reviewer source changed')
    write(a.output,dict(status='actual_DSP_model_local_global_copy_498_cases_ABBA4_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance',
        source_root=str(ROOT),source_revision=revision,source=entry(__file__),build=entry(a.build/'result.json'),
        build_controller=entry(a.build_controller),packet=entry(a.packet/'result.json'),packet_controller=entry(a.packet_controller),
        run=entry(a.run/'process.json'),controller=entry(a.controller),raw_stdout=entry(events_path),outputs=outputs,
        independent_input_FNV=scalar,independent_FNV_source=entry(source),case_count=498,
        kernel_microseconds=timing,candidate_over_baseline_ratio=ratio,positive_primitive_gain=ratio<.98,
        DSP_primitive_calls=4,DSP_model_calls=0,new_DSP_calls=0,model_inference_acceptance=False,
        performance_acceptance=False,shared_default_changed=False,completed_unix=time.time()))


if __name__=='__main__':main()
