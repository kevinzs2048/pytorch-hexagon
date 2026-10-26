"""Derive an original warm catalog from the accepted dynamic compiler replies.

The original ARM native code matches complete captured graph identity and all
immutable input bytes. This never selects a model by prompt or query length.
The result is preparation only, not an executed or accepted new runtime path.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from build import ROOT,entry,sha,write


def derive_entries(native,compiled,remote_root,expected_libraries):
    if len(compiled)!=2 or len(native['captures'])!=2:
        raise ValueError('Both actual original compiled query domains required')
    rows=[]
    for capture,result in zip(native['captures'],compiled,strict=True):
        candidate=copy.deepcopy(result['compiled_entry']);bounds=tuple(candidate['contract']['variables']['query'])
        if (result['status']!='carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP'
                or result['CPU_model_forward'] is not False or result['DSP_calls']!=0
                or candidate['graph']!=capture['graph'] or candidate['contract']!=capture['contract']
                or candidate['retained_input_indices']!=capture['retained_input_indices']
                or candidate['frontend_key']!=capture['frontend_key'] or candidate['machine_key']!=capture['machine_key']
                or expected_libraries.get(str(bounds))!=candidate['library']['sha256']
                or candidate['library']['sha256']!=capture['DSO_sha256']):
            raise ValueError('Exact executed graph, retained input binding, ABI, keys and DSO required')
        candidate['library']['remote_path']=str(remote_root/(candidate['library']['sha256']+'.so'))
        rows.append(candidate)
    if ({tuple(row['contract']['variables']['query']) for row in rows}!={(1,1),(2,32)}
            or len({row['graph']['sha256'] for row in rows})!=2
            or len({row['binding_content_sha256'] for row in rows})!=1):
        raise ValueError('Two distinct actual graphs for one complete immutable model required')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--board-library-root',type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if (not a.output.is_absolute() or a.output.exists()
            or a.board_library_root.parent!=Path('/dev/shm')
            or not a.board_library_root.name.startswith('qwen3-')
            or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()):
        p.error('Fresh owned preparation and a scoped original ARM model deployment required')
    from native_model_acceptance import require_model_acceptance,accepted_cold
    from revisit_release_index import indexed_files
    index=read(a.index);model,closure=require_model_acceptance(index)
    refs=[];indexed_files(model,refs)
    cold=accepted_cold(model,closure);native=read(cold/'native-result.json');commands=read(cold/'commands.json')
    numerical=Path(model['source_root']);sys.path[:0]=[str(numerical/'qwen3/runtime/arm'),str(numerical)]
    from audit_arm_dynamic_warm_run import audit_compile_handoff
    worker=commands['worker'];mailbox=Path(worker[worker.index('--mailbox')+1]);chain=audit_compile_handoff(native,mailbox,rebuilt=True)
    compiled=[read(mailbox/f'capture-{i:02d}/compiled/result.json') for i in range(2)]
    rows=derive_entries(native,compiled,a.board_library_root,index['candidate_libraries'])
    runtime=read(index['runtime_catalog']['path']);catalog=copy.deepcopy(runtime);catalog['entries']=rows
    catalog['source_run']=entry(cold/'native-result.json');catalog['source_runs']=[catalog['source_run']]
    catalog['compiled_reply_provenance']=[entry(mailbox/f'capture-{i:02d}/compiled/result.json') for i in range(2)]
    catalog['base_runtime_catalog']=index['runtime_catalog'];catalog['source_sha256']=sha(__file__)
    catalog.update(numerical_acceptance=False,automatic_cold_compilation=False)
    a.output.mkdir();write(a.output/'catalog.json',catalog)
    write(a.output/'result.json',dict(status='accepted_dynamic_model_artifact_catalog_prepared_NOT_device_acceptance',
        source=entry(__file__),source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        accepted_model=model,current_version=index['current_default_accepted_version'],catalog=entry(a.output/'catalog.json'),
        candidate_libraries=index['candidate_libraries'],
        original_compile_chain=chain,original_runtime_artifacts_preserved=catalog['runtime']==runtime['runtime'],
        full_graph_and_immutable_binding_selector_source=entry(numerical/'qwen3/runtime/arm/hexagon_arm_dynamic_catalog.py'),
        DSP_calls=0,new_model_PRIMARY_compilations=0,new_frontend_invocations=0,
        compile_requests_skipped_on_device=False,model_inference_acceptance=False,performance_acceptance=False,completed_unix=time.time()))


if __name__=='__main__':main()
