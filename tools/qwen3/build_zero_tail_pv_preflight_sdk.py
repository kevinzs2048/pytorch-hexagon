"""Replay the original q1 compiler, add a guard and STOP before model codegen."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from build import ROOT,HERE,entry,write
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner

def relocate(command,target,source=None):
    command=command.copy()
    for flag,suffix in [('-o',''),('-MT',''),('-MF','.d')]:
        if flag in command:command[command.index(flag)+1]=str(target)+suffix
    command=['--dependency-file='+str(target)+'.d' if x.startswith('--dependency-file=') else x for x in command]
    if source is not None:command[command.index('-c')+1]=str(source)
    return command

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','original-sdk','physical-review','physical-review-controller','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-candidate',action='store_true')
    parser.add_argument('--all28-review',type=Path)
    parser.add_argument('--all28-review-controller',type=Path)
    parser.add_argument('--deadline-unix',type=int,required=True);a=parser.parse_args()
    read=lambda path:json.loads(Path(path).read_text());c=read(a.config)
    git=lambda *x:subprocess.check_output(['git','-C',str(ROOT),*x],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain') or a.deadline_unix-time.time()<1800:
        raise ValueError('Frozen source, fresh output and admitted half hour required')
    proof=read(a.physical_review);owner=require_normal_owner(a.physical_review_controller,a.deadline_unix)
    refs=[];indexed_files(proof,refs)
    if (proof['status']!='isolated_actual_q1_physical_GQA_guard24_full_words_FP_flags_raw_FNV_and_owner_independently_revisited_NOT_model'
            or not proof['numerical']['all_target_FP32_words_equal'] or not proof['numerical']['all_target_exception_flags_equal']
            or proof['model_calls'] or proof['DSP_calls']!=1
            or owner['source_revision']!=proof['reviewer_revision']
            or owner['command'][owner['command'].index('--output')+1]!=str(a.physical_review)):
        raise ValueError('Actual independent physical precision and normal owner required')
    linked=read(proof['linked_build']['path']);lower=read(linked['build']['path'])
    if not lower['transform'].get('actual_P_derived_extent') or linked['integer_guard_source']['sha256']!=entry(compiler_tools()/'runtime/profile/zero_tail_hvx_guard.c')['sha256']:
        raise ValueError('Exact target-tested runtime-derived integer helper required')
    if a.model_candidate:
        all28=read(a.all28_review);all28_owner=require_normal_owner(a.all28_review_controller,a.deadline_unix)
        refs=[];indexed_files(all28,refs)
        if (all28['status']!='actual_CPP_target24_and_original_q1_all28_preflight_independently_closed_NOT_model_acceptance'
                or all28_owner['source_revision']!=all28['source_revision']
                or all28['CPP_target_review']!=entry(a.physical_review)
                or lower['transform'].get('actual_CPP_generator',{}).get('sha256')!=entry(compiler_tools()/'buffer_reuse/ZeroTailPVRegister.cpp')['sha256']):
            raise ValueError('Actual target-tested identical CPP pass and independent all28 closure required for model candidate mode')
    elif a.all28_review or a.all28_review_controller:
        raise ValueError('All28 closure flags are reserved for explicit model candidate mode')
    original=read(a.original_sdk);refs=[];indexed_files(original,refs)
    model=read(lower['actual_full_model']['path'])
    if (original['status']!='isolated_ordered_PV_register_q1_model_compiler_built_NOT_DSP'
            or original['candidate']['sha256']!=model['inputs']['compiler']['sha256']
            or entry(c['compiler'])!=original['accepted_sdk']
            or not all(original[k] for k in ['unmodified_SDK_replay_byte_identical','original_link_inputs_unchanged','original_SDK_unchanged'])):
        raise ValueError('Exact originally compiled q1 model SDK and unchanged shared SDK required')
    old_compile=next(row for row in original['commands'] if '-c' in row['argv'] and row['argv'][row['argv'].index('-c')+1].endswith('OrderedPVRegister.cpp'))
    old_object=old_compile['argv'][old_compile['argv'].index('-o')+1];link=original['commands'][-1]
    if link['argv'].count(old_object)!=1:raise ValueError('Exactly one original register pass object required')
    for path,digest in original['link_inputs'].items():
        if entry(path)['sha256']!=digest:raise ValueError('Original SDK link input changed')
    a.output.mkdir();record=dict(status='building_zero_tail_q1_SDK_PREFLIGHT_ONLY',source_root=str(ROOT),
        source_revision=git('rev-parse','HEAD'),source=entry(__file__),physical_review=entry(a.physical_review),
        physical_review_controller=entry(a.physical_review_controller),original_sdk=entry(a.original_sdk),
        original_shared_sdk=original['accepted_sdk'],runtime_helper=entry(compiler_tools()/'runtime/profile/zero_tail_hvx_guard.c'),
        pass_source=entry(compiler_tools()/'buffer_reuse/ZeroTailPVRegister.cpp'),register_source=entry(compiler_tools()/'buffer_reuse/OrderedPVRegister.cpp'),
        commands=[],DSP_calls=0,model_PRIMARY_compilations=0,model_acceptance=False,performance_acceptance=False,
        deadline_unix=a.deadline_unix,started_unix=time.time())
    if a.model_candidate:record.update(all28_review=entry(a.all28_review),all28_review_controller=entry(a.all28_review_controller))
    save=lambda:write(a.output/'result.json',record)
    def run(command,cwd,label):
        row=dict(argv=command,cwd=cwd,started_unix=time.time());record['commands'].append(row);save()
        with (a.output/(label+'.log')).open('xb') as log:
            done=subprocess.run(command,cwd=cwd,env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path'])),
                stdout=log,stderr=subprocess.STDOUT,timeout=min(300,a.deadline_unix-time.time()-60))
        row.update(returncode=done.returncode,completed_unix=time.time());save();done.check_returncode()
    try:
        replay=a.output/'original-q1-replay.so';run(relocate(link['argv'],replay),link['cwd'],'original-q1-replay')
        if entry(replay)['sha256']!=original['candidate']['sha256']:raise ValueError('Original q1 SDK replay must be byte-identical')
        register=a.output/'OrderedPVRegister.cpp.o';command=relocate(old_compile['argv'],register,compiler_tools()/'buffer_reuse/OrderedPVRegister.cpp')
        command.append('-DQWEN_Q1_ZERO_TAIL_MODEL_CANDIDATE=1' if a.model_candidate else '-DQWEN_Q1_ZERO_TAIL_PREFLIGHT=1');run(command,old_compile['cwd'],'register-preflight')
        guard=a.output/'ZeroTailPVRegister.cpp.o';command=relocate(old_compile['argv'],guard,compiler_tools()/'buffer_reuse/ZeroTailPVRegister.cpp')
        run(command,old_compile['cwd'],'zero-tail-transform')
        candidate=a.output/'libtriton.so';command=relocate(link['argv'],candidate)
        command[command.index(old_object)]=str(register);command.insert(command.index(str(register)),str(guard))
        run(command,link['cwd'],'preflight-sdk-link')
        for path,digest in original['link_inputs'].items():
            if entry(path)['sha256']!=digest:raise ValueError('Original SDK link input changed during build')
        if entry(c['compiler'])!=original['accepted_sdk']:raise ValueError('Shared SDK changed')
        record.update(status=('zero_tail_runtime_extent_q1_MODEL_CANDIDATE_SDK_built_NOT_model_acceptance' if a.model_candidate else 'zero_tail_runtime_extent_q1_SDK_PREFLIGHT_built_STOP_BEFORE_MODEL_CODEGEN'),
            candidate=entry(candidate),original_replay=entry(replay),unmodified_q1_SDK_replay_byte_identical=True,
            original_link_inputs_unchanged=True,original_shared_SDK_unchanged=True,completed_unix=time.time())
    except BaseException as exc:record.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':main()
