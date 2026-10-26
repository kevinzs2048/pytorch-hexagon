"""Build a separately typed, stop-before-codegen SDK for identical QK tails."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from build import ROOT,HERE,entry,write
from build_zero_tail_pv_preflight_sdk import relocate
from profile_model_memory import checked
from revisit_native_compile_session_abba import require_normal_owner


def same_pinned_source(reference,candidate):
    return checked(reference).read_bytes()==Path(candidate).read_bytes()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','base-sdk','base-sdk-controller','model-code','model-code-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=int,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text());c=read(a.config)
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain') or a.deadline_unix-time.time()<1800:
        raise ValueError('Fresh frozen diagnostic SDK and explicit budget required')
    base=read(a.base_sdk);model=read(a.model_code)
    for controller,result,script,output in (
        (a.base_sdk_controller,base,'build_q2_dispatch_model_sdk.py',a.base_sdk.parent),
        (a.model_code_controller,model,'compile_q2_dispatch_model_candidate.py',a.model_code.parent)):
        owner=read(controller);require_normal_owner(controller,owner['authorized_deadline_unix'])
        if (owner['source_revision']!=result['source_revision']
                or Path(owner['command'][1]).name!=script
                or owner['command'][owner['command'].index('--output')+1]!=str(output)):
            raise ValueError('Actual normally completed original producer required')
    if (base['status']!='Q2_dispatch_target_proved_model_SDK_built_NOT_model_acceptance'
            or model['status']!='one_existing_Q2_runtime_dispatch_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or model['SDK']!=entry(a.base_sdk) or model['SDK_controller']!=entry(a.base_sdk_controller)
            or model['query_bounds']!=[2,32] or model['model_PRIMARY_compilations']!=1
            or not base['unmodified_SDK_replay_byte_identical']
            or not base['original_PV_objects_unchanged'] or not base['original_link_inputs_unchanged']
            or not same_pinned_source(base['proven_recurrence'],compiler_tools()/'buffer_reuse/DynamicQKRegisterProven.inc')
            or base['original_shared_sdk']!=entry(c['compiler'])):
        raise ValueError('Exact normally tested Q2 compiler and unchanged register recurrence required')
    checked(base['candidate']);checked(model['primary'])
    old=next(row for row in base['commands']if '-c'in row['argv'])
    old_object=old['argv'][old['argv'].index('-o')+1];link=base['commands'][-1]
    if link['argv'].count(old_object)!=1:raise ValueError('Exactly one existing QK pass object required')
    inputs={str((Path(link['cwd'])/word).resolve()):entry(Path(link['cwd'])/word)['sha256']
        for word in link['argv']if word.endswith('.o')}
    a.output.mkdir();record=dict(status='building_Q2_identical_tail_structural_SDK_NOT_target',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        base_sdk=entry(a.base_sdk),base_sdk_controller=entry(a.base_sdk_controller),model_code=entry(a.model_code),
        model_code_controller=entry(a.model_code_controller),original_shared_sdk=entry(c['compiler']),
        dispatch=entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp'),
        identical_tail=entry(compiler_tools()/'buffer_reuse/DynamicQKIdenticalTail.inc'),
        integer_guard=entry(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent.c'),
        proven_recurrence=base['proven_recurrence'],original_link_objects=inputs,commands=[],
        model_PRIMARY_compilations=0,DSP_calls=0,model_inference_acceptance=False,
        performance_acceptance=False,deadline_unix=a.deadline_unix)
    save=lambda:write(a.output/'result.json',record)
    def run(argv,cwd,label):
        row=dict(argv=argv,cwd=cwd,started_unix=time.time());record['commands'].append(row);save()
        with (a.output/(label+'.log')).open('xb')as log:
            r=subprocess.run(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=300,
                env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path'])))
        row.update(returncode=r.returncode,completed_unix=time.time());save();r.check_returncode()
    save()
    try:
        replay=a.output/'base-replay.so';run(relocate(link['argv'],replay),link['cwd'],'base-replay')
        if entry(replay)['sha256']!=base['candidate']['sha256']:
            raise ValueError('The exact original SDK must replay before changing the QK pass')
        temporary=a.output/'base-replay-dedup.tmp';os.link(base['candidate']['path'],temporary);os.replace(temporary,replay)
        record.update(base_replay=entry(replay),base_replay_byte_identical=True);save()
        obj=a.output/'DynamicQKRegisterDispatch.cpp.o'
        argv=relocate(old['argv'],obj,compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')
        argv+=['-DQWEN_Q2_IDENTICAL_TAIL=1','-DQWEN_Q2_DISPATCH_PREFLIGHT=1']
        run(argv,old['cwd'],'identical-tail-compile')
        candidate=a.output/'libtriton.so';argv=relocate(link['argv'],candidate)
        argv[argv.index(old_object)]=str(obj);run(argv,link['cwd'],'identical-tail-link')
        for path,digest in inputs.items():
            if entry(path)['sha256']!=digest:raise ValueError('An original compiler object changed')
        if record['original_shared_sdk']!=entry(c['compiler']):raise ValueError('Shared SDK changed')
        record.update(status='Q2_identical_tail_structural_SDK_STOP_BEFORE_MODEL_CODEGEN_NOT_target',
            candidate=entry(candidate),original_PV_objects_unchanged=True,
            original_link_inputs_unchanged=True,completed_unix=time.time());save()
    except BaseException as error:record.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
