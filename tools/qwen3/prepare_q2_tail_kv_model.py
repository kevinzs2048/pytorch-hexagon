"""Use the actual ARM-checked second prefill chunk, retaining its prior eight KV positions."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from build import ROOT,HERE,entry,sha,write
from profile_model_memory import checked
from native_model_acceptance import require_model_acceptance
from revisit_q2_dispatch_preflight import actual_owner
from nonempty_prefill_packet import resize_nonempty_prefill_packet

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-review','model-review-controller','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists()or git('status','--porcelain'):raise ValueError('Fresh frozen nonempty model comparison required')
    review=read(a.model_review)
    actual_owner(a.model_review_controller,review,'revisit_q2_tail_row_model.py',a.model_review)
    if (review['status']!='actual_Q2_row_major_guard_model_ABBA12_logits_56KV_and_3FNV_independently_revisited_NOT_generation_acceptance'
            or review['DSP_calls']!=12 or review['model_inference_acceptance']or review['generation_regression']):
        raise ValueError('Actual separately checked row-guard full model12 required')
    original_manifest=checked(review['manifest']);m=read(original_manifest)
    model,closure=require_model_acceptance(read(checked(m['index'])))
    cold=Path(closure['cold_run']);acceptance=read(cold/'acceptance.json');native_path=checked(m['accepted_run']);audit_path=checked(m['accepted_audit'])
    if not native_path.is_relative_to(cold/'audit')or checked(acceptance['audit'])!=audit_path:
        raise ValueError('Own qualified current model original ARM comparison required')
    native=read(native_path);audit=read(audit_path)
    forward=next(r for r in native['forwards']if r['query']==6 and r['cache_positions']==list(range(8,14)))
    if (audit['status']!='framework_native_numerical_acceptance_passed'or audit['numerical_acceptance']is not True
            or audit['run_sha256']!=sha(native_path)or native['process_exit_code']or native['CPU_model_fallback']
            or forward['DSO_sha256']!=m['libraries']['baseline']['sha256']):
        raise ValueError('Actual original q6@8 ARM-W8A8 reference and same baseline required')
    folder=native_path.parent/f"forward-{forward['step']:02d}-q6";packet=folder/'input.bin';reference=folder/'output.bin'
    if sha(packet)!=forward['input_sha256']or sha(reference)!=forward['output_sha256']:
        raise ValueError('Original full input/output bits must remain available')
    sys.path[:0]=[str(ROOT),str(ROOT/'qwen3/runtime/arm')]
    payload=packet.read_bytes()
    for query in (6,8,32):
        derived=resize_nonempty_prefill_packet(m['contract'],payload,query)
        if query==6 and derived!=payload:raise ValueError('Original nonempty packet must round-trip byte-identically')
    with tempfile.TemporaryDirectory(prefix='qwen-kv-prefill-fnv-')as temporary:
        source=compiler_tools()/'runtime/profile/dynamic_qk_reference.c';library=Path(temporary)/'fnv.so'
        subprocess.run(['cc','-O2','-shared','-fPIC',str(source),'-lm','-o',str(library)],check=True,capture_output=True)
        fnv=ctypes.CDLL(str(library)).qwen_dynamic_qk_fnv;fnv.argtypes=[ctypes.c_void_p,ctypes.c_uint];fnv.restype=ctypes.c_uint32
        scalar=int(fnv(ctypes.c_char_p(payload),len(payload)))
        if any(forward['invoke'][k]!=scalar for k in ('AP_input_fnv1a32','DSP_input_fnv1a32','RPC_input_fnv1a32')):
            raise ValueError('Independent complete original q6 request FNV changed')
    a.output.mkdir()
    for name,path in [('baseline.so',checked(m['libraries']['baseline'])),('candidate.so',checked(m['libraries']['candidate'])),('input-q6.bin',packet),('reference-q6.bin',reference)]:os.link(path,a.output/name)
    m.update(status='Q2_nonempty_KV_tail_model_comparison_manifest_prepared_NOT_executed',
        packets=[dict(query=6,step=forward['step'],input=entry(packet),output=entry(reference),raw_FNV_recomputed=scalar,original_ARM_reference_source=model)],
        queries=[6,8,32],original_ARM_W8A8_queries=[6],derived_DSP_reference_queries=[8,32],
        original_nonempty_positions=list(range(8,14)),occupied_KV_positions=8,all_original_56_KV_inputs_preserved=True,
        previous_model_review=entry(a.model_review),previous_model_review_controller=entry(a.model_review_controller),
        previous_model_manifest=entry(original_manifest),packet_transform=entry(HERE/'nonempty_prefill_packet.py'))
    write(a.output/'manifest.json',m)
    write(a.output/'result.json',dict(status='Q2_nonempty_KV_tail_model_comparison_prepared_NOT_executed',source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        manifest=entry(a.output/'manifest.json'),DSP_calls=0,model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False,completed_unix=time.time()))

if __name__=='__main__':main()
