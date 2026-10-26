"""Authenticate unchanged preparation inputs before copying an accepted stage.

This does not run a frontend, optimizer, model compiler, linker or DSP. A new
legitimate frontend/policy is a miss; drift in an accepted artifact is an error.
The existing numerical checkout and its original cache are read only.
"""
import argparse
import copy
import importlib.util
import hashlib
import json
import math
import errno
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from build import ROOT, entry, sha, verify_dependencies, write
from run_regression import accepted_base, require_configured_python
from reuse_accepted_compiler_objects import ReadOnlyCompilerCache


class PreparedStageInputMismatch(ValueError):
    """The original preparation pipeline must process new legitimate inputs."""


def stage_artifacts(files, destination, share_immutable=False):
    """Share only immutable graph blobs; every mutable receipt remains owned."""
    immutable={'hexagon-linalg.mlirbc','adapted-source.mlir.gz'}
    staged=[]
    for row in files:
        source=Path(row['path']);target=destination/source.name
        if source.is_symlink() or not source.is_file() or sha(source)!=row['sha256']:
            raise ValueError('Authenticated prepared source artifact changed')
        mode='copy'
        if share_immutable and source.name in immutable:
            try:
                os.link(source,target);mode='hardlink'
            except OSError as error:
                if error.errno not in (errno.EXDEV,errno.EOPNOTSUPP):raise
                shutil.copyfile(source,target)
        else:shutil.copyfile(source,target)
        if sha(target)!=row['sha256']:raise ValueError('Owned prepared artifact staging changed')
        staged.append(dict(source=str(source),target=str(target),sha256=row['sha256'],mode=mode,
            source_inode=source.stat().st_ino,target_inode=target.stat().st_ino))
    return staged


def preparation_policy(recipe, bounds):
    for name in ('packed_i8_query1','packed_multi_i8_query2_to32','silu_effects_query1',
                 'silu_effects_query2_to32','arm_rope_trig_compat'):
        if type(recipe.get(name,False)) is not bool:
            raise ValueError('Explicit preparation policy booleans required')
    names=['attention_optimizer','sums_optimizer','knorm_optimizer']
    for enabled,name in ((recipe.get('packed_i8_query1',False) and bounds==[1,1],'packed_i8_optimizer'),
                         (recipe.get('packed_multi_i8_query2_to32',False) and bounds==[2,32],'packed_multi_i8_optimizer'),
                         (recipe.get('silu_effects_query1',False) and bounds==[1,1],'silu_effects_optimizer'),
                         (recipe.get('silu_effects_query2_to32',False) and bounds==[2,32],'multi_silu_effects_optimizer')):
        if enabled:names.append(name)
    return dict(arm_rope_trig=recipe.get('arm_rope_trig_compat',False),
        inputs={name:recipe['inputs'][name] for name in names})


def frontend_entry(path):
    from hexagon_compiled_artifact_cache import canonical,describe_artifacts
    value=json.loads(path.read_text());folder=path.parent/value['build']
    if (value['status']!='captured_AOT_Linalg_cached_NOT_machine_code'
            or value['key']!=hashlib.sha256(canonical(value['request'])).hexdigest()
            or value['request']['function_name']!='Qwen3DynamicBoundProbe'
            or Path(value['build']).name!=value['build']
            or not value['build'].startswith(value['key']+'-')
            or folder.is_symlink() or not folder.is_dir()
            or value['artifacts']!=describe_artifacts(folder,['captured-linalg.mlirbc','lowering.json'])):
        raise ValueError('The current actual frontend artifacts changed')
    return value


def preparation_frontend_identity(old,old_path,new,new_path):
    """Ignore only the measured duration, after authenticating both raw files."""
    from hexagon_compiled_artifact_cache import canonical
    if (any(new[key]!=old[key] for key in ('status','key','request'))
            or new['artifacts'][0]!=old['artifacts'][0]):
        raise PreparedStageInputMismatch('Current captured graph, weights or frontend differs')
    paths=[path.parent/value['build']/'lowering.json'
           for value,path in ((old,old_path),(new,new_path))]
    reports=[json.loads(path.read_text()) for path in paths]
    for report in reports:
        duration=report.get('frontend_seconds')
        if type(duration) not in (int,float) or not math.isfinite(duration) or duration<0:
            raise ValueError('Finite nonnegative actual frontend duration required')
    if (canonical({k:v for k,v in reports[0].items() if k!='frontend_seconds'})
            !=canonical({k:v for k,v in reports[1].items() if k!='frontend_seconds'})):
        raise PreparedStageInputMismatch('Current complete lowering evidence differs')
    return dict(status='identical_captured_Linalg_and_lowering_except_measured_frontend_seconds',
        accepted_lowering=entry(paths[0]),current_lowering=entry(paths[1]),
        excluded_identity_fields=['frontend_seconds'],
        accepted_frontend_seconds=reports[0]['frontend_seconds'],
        current_frontend_seconds=reports[1]['frontend_seconds'])


def authenticate(a):
    base=accepted_base(a.accepted_cold)
    native=json.loads((a.accepted_cold/'native-result.json').read_text())
    acceptance=json.loads((a.accepted_cold/'acceptance.json').read_text())
    audit=json.loads(Path(acceptance['audit']['path']).read_text())
    if (sha(a.accepted_cold/'native-result.json')!=audit['run_sha256']
            or native['status']!='ARM_native_dynamic_compiler_handoff_generate_completed'
            or native['DSP_calls']!=9 or native['process_exit_code'] or native['CPU_model_fallback']
            or native['frontend_invocations']!=2 or native['machine_code_compilations']!=2):
        raise ValueError('Original accepted cold9 with both actual query domains required')
    captures=[row for row in native['captures'] if row['contract']['variables']['query']==a.query_bounds]
    if len(captures)!=1:raise ValueError('Exactly one original query domain required')
    capture=captures[0];worker=base['worker']
    def argument(flag):
        if worker.count(flag)!=1:raise ValueError('Unambiguous original worker required')
        return worker[worker.index(flag)+1]
    scripts=Path(base['source_root'])/'qwen3/runtime/arm'
    current=ROOT/'qwen3/runtime/arm'
    sources=('prepare_captured_qwen_hexagon.py','apply_dynamic_bound_qk_policy.py',
        'repair_dynamic_scatter_mlir_text.py','streamed_mlir_interchange.py',
        'hexagon_qwen_numerical_policy.py','hexagon_compiled_artifact_cache.py')
    for name in sources:
        if sha(current/name)!=sha(scripts/name):
            raise PreparedStageInputMismatch('Preparation source changed: '+name)
    original_recipe=json.loads(Path(base['recipe']['path']).read_text())
    requested_recipe=json.loads(a.recipe.read_text())
    policy=preparation_policy(requested_recipe,a.query_bounds)
    for row in policy['inputs'].values():
        if not Path(row['path']).is_absolute() or sha(row['path'])!=row['sha256']:
            raise ValueError('Current pinned optimizer bytes drifted')
    if policy!=preparation_policy(original_recipe,a.query_bounds):
        raise PreparedStageInputMismatch('Current numerical preparation policy differs')
    old_frontend_path=Path(argument('--frontend-cache'))/(capture['frontend_key']+'.json')
    old=frontend_entry(old_frontend_path);new=frontend_entry(a.frontend_entry)
    frontend_identity=preparation_frontend_identity(old,old_frontend_path,new,a.frontend_entry)
    spec=importlib.util.spec_from_file_location('accepted_preparation_stage_source',
        scripts/'hexagon_qwen_captured_machine_cache.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache=module.CapturedQwenMachineCache(Path(argument('--machine-cache')),
        frontend_root=Path(argument('--frontend-cache')),recipe_path=Path(base['recipe']['path']),
        recipe_sha256=base['recipe']['sha256'])
    cache.cache=ReadOnlyCompilerCache(cache.cache,capture['machine_key'])
    actual=cache.get_or_compile(old,runtime_contract=capture['contract'])
    folder=Path(actual['directory'])/'prepared';result_path=folder/'result.json'
    result=json.loads(result_path.read_text())
    if (result['status']!='actual_captured_Qwen_prepared_for_Hexagon_NOT_machine_code'
            or result['source_sha256']!=sha(scripts/'prepare_captured_qwen_hexagon.py')
            or result['frontend_key']!=old['key'] or result['source']!=old['artifacts'][0]
            or result['inputs']['entry']!=entry(old_frontend_path)
            or {k:v for k,v in result['inputs'].items() if k!='entry'}!=policy['inputs']
            or Path(result['output']['path'])!=folder/'hexagon-linalg.mlirbc'
            or sha(result['output']['path'])!=result['output']['sha256']
            or result['DSP_calls'] or result['machine_code_compilations']):
        raise ValueError('Authenticated original prepared graph binding changed')
    for key,name in (('packed_integer_policy','packed-optimizer.log'),
                     ('packed_multi_integer_policy','packed-multi-optimizer.log'),
                     ('silu_effects','silu-effects-optimizer.log'),
                     ('multi_silu_effects','multi-silu-effects-optimizer.log')):
        if key in result and sha(folder/name)!=result[key]['log_sha256']:
            raise ValueError('Original policy proof log changed')
    if sha(folder/'operator-policy-inspection.mlir')!=result['operator_policy']['inspection_sha256']:
        raise ValueError('Original actual operator policy inspection changed')
    files=[path for path in folder.iterdir() if path.is_file()]
    if any(path.is_symlink() or not path.is_file() for path in folder.iterdir()):
        raise ValueError('Regular complete original prepared stage required')
    return folder,result,dict(accepted_cold=entry(a.accepted_cold/'acceptance.json'),
        original_prepared=entry(result_path),current_frontend=entry(a.frontend_entry),
        frontend_identity=frontend_identity,
        policy=policy,source_pins={name:sha(current/name) for name in sources},
        files=[dict(**entry(path),bytes=path.stat().st_size) for path in sorted(files)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','accepted-cold','frontend-entry','recipe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recipe-sha256',required=True)
    p.add_argument('--stage-output',action='store_true')
    p.add_argument('--share-immutable-artifacts',action='store_true')
    p.add_argument('--query-bounds',type=int,nargs=2,required=True)
    a=p.parse_args()
    if (not a.output.is_absolute() or a.output.exists() or a.output.is_symlink()
            or a.query_bounds not in ([1,1],[2,32])):p.error('Fresh original-domain prepared stage required')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed prototype source required')
    if sha(a.recipe)!=a.recipe_sha256:raise ValueError('Current complete preparation recipe changed')
    config=json.loads(a.config.read_text());require_configured_python(config)
    sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
    verify_dependencies(config,json.loads(Path(__file__).with_name('dependencies.lock.json').read_text()))
    start=time.monotonic();record=dict(status='checking_exact_accepted_preparation_stage',DSP_calls=0,
        new_frontend_invocations=0,new_prepare_optimizer_invocations=0,model_inference_acceptance=False,
        source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__))
    a.output.mkdir()
    try:
        folder,result,proof=authenticate(a)
        destination=a.output if a.stage_output else a.output/'prepared'
        if not a.stage_output:destination.mkdir()
        staged=stage_artifacts(proof['files'],destination,a.share_immutable_artifacts)
        current_result=copy.deepcopy(result)
        current_result['inputs']['entry']=entry(a.frontend_entry)
        current_result['output']['path']=str(destination/'hexagon-linalg.mlirbc')
        current_result['historical_preparation_elapsed_seconds']=current_result.pop('elapsed_seconds')
        current_result['elapsed_seconds']=time.monotonic()-start
        current_result['prepared_stage_reuse']=dict(path=str(a.output/'reuse-result.json'))
        write(destination/'result.json',current_result)
        record.update(status=('exact_accepted_prepared_Linalg_staged_NOT_pipeline_or_DSP_acceptance'
                              if a.share_immutable_artifacts else
                              'exact_accepted_prepared_Linalg_copied_NOT_pipeline_or_DSP_acceptance'),
            **proof,prepared=entry(destination/'result.json'),query_bounds=a.query_bounds,
            shared_immutable_artifacts=a.share_immutable_artifacts,staged_artifacts=staged,
            current_recipe=entry(a.recipe),
            historical_preparation_seconds=result['elapsed_seconds'])
    except PreparedStageInputMismatch as e:
        record.update(status='prepared_stage_cache_miss',error=str(e));raise SystemExit(10)
    except BaseException as e:record.update(status='prepared_stage_reuse_rejected',error=str(e));raise
    finally:
        record['reuse_seconds']=time.monotonic()-start;write(a.output/'reuse-result.json',record)
    print(record['status'])


if __name__=='__main__':main()
