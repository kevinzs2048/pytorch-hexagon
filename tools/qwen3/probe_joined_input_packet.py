"""Time host packing of an actual accepted complete model input, without DSP."""
from qwen3.compiler_dependency import compiler_root
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import time

from build import ROOT, entry, sha, write
sys.path.insert(0, str(ROOT))
from qwen3.runtime.support.joined_input_packet import pack_torch_request_joined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-source','native-run','input','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--forward-index',type=int,default=0)
    p.add_argument('--require-aarch64',action='store_true')
    p.add_argument('--source-revision')
    p.add_argument('--candidate-sha256')
    p.add_argument('--original-sha256')
    a=p.parse_args()
    arm=platform.machine()=='aarch64'
    if (not a.output.is_absolute() or a.output.exists()
            or a.require_aarch64 and platform.machine()!='aarch64'
            or not arm and subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()):
        raise ValueError('Fresh output, committed candidate and selected actual host required')
    original=a.reference_source/'qcom_hexagon_backend/backend/hexagon_dynamic_buffer_io.py'
    own=compiler_root()/'qcom_hexagon_backend/backend/hexagon_dynamic_buffer_io.py'
    if sha(original)!=sha(own):
        raise ValueError('Frozen original layout, validation and wrapper generator must be unchanged')
    candidate=ROOT/'qwen3/runtime/support/joined_input_packet.py'
    if arm:
        if (not a.source_revision or not re.fullmatch('[0-9a-f]{40}',a.source_revision)
                or sha(candidate)!=a.candidate_sha256 or sha(original)!=a.original_sha256):
            raise ValueError('Actual ARM requires the committed deployment revision and pinned formatter bytes')
    spec=importlib.util.spec_from_file_location('qcom_hexagon_backend.backend._original_pack',original)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    run=json.loads(a.native_run.read_text());forward=run['forwards'][a.forward_index]
    if (run['status']!='ARM_native_dynamic_compiler_handoff_generate_completed'
            or run['process_exit_code']!=0 or sha(a.input)!=forward['input_sha256']):
        raise ValueError('Actual normally completed accepted forward input required')
    contract=run['captures'][forward['AOT_capture_index']]['contract']
    packet=a.input.read_bytes();names=tuple(contract['variables']);header=8+8*len(names)
    if packet[:8]!=reference.MAGIC:
        raise ValueError('Original HXDY header required')
    values=dict(zip(names,struct.unpack_from('<'+'q'*len(names),packet,8),strict=True))
    layout=reference.layout_for(contract,values)
    if len(packet)!=header+layout['input_bytes']:
        raise ValueError('Full accepted packet size differs')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    owned=bytearray(packet);args=[];index=0
    for row in contract['inputs']:
        if row['kind']=='scalar':
            args.append(values.get(row['value'],row['value']))
        else:
            field=layout['inputs'][index];index+=1
            array=np.frombuffer(owned,dtype=np.dtype('<f4' if field['dtype']=='float32' else '<i8'),
                count=field['bytes']//(4 if field['dtype']=='float32' else 8),offset=header+field['offset'])
            args.append(torch.from_numpy(array.reshape(field['shape'])))
    for fn in (reference.pack_torch_request,pack_torch_request_joined):
        result=fn(contract,args)
        if result!=(packet,layout,values):
            raise ValueError('Entire model input or canonical layout differs')
        del result
    order=('established','joined','joined','established')*3;rows=[]
    for position,role in enumerate(order):
        fn=reference.pack_torch_request if role=='established' else pack_torch_request_joined
        start=time.perf_counter_ns();actual,actual_layout,dimensions=fn(contract,args);elapsed=time.perf_counter_ns()-start
        if actual!=packet or actual_layout!=layout or dimensions!=values:
            raise ValueError('Actual timed input is not byte exact')
        rows.append(dict(position=position,role=role,packing_ns=elapsed,
            complete_input_sha256=hashlib.sha256(actual).hexdigest()))
        del actual
    medians={role:statistics.median(r['packing_ns'] for r in rows if r['role']==role) for role in ('established','joined')}
    write(a.output,dict(status='actual_complete_model_input_joined_packing_ABBA12_byteexact_HOST_ONLY',
        machine=platform.machine(),input=entry(a.input),native_run=entry(a.native_run),
        original_formatter=entry(original),candidate_formatter=entry(candidate),
        query=forward['query'],input_bytes=len(packet),all_padding_and_tensor_bytes_identical=True,
        rows=rows,median_packing_ns=medians,joined_over_established_ratio=medians['joined']/medians['established'],
        host_only=True,DSP_calls=0,model_inference_acceptance=False,production_speedup_claimed=False,
        source_revision=(a.source_revision if arm else subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()),completed_unix=time.time()))
    print('whole-input byte exact; host packing ratio',medians['joined']/medians['established'])


if __name__=='__main__':main()
