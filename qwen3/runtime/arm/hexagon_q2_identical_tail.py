"""Reuse the exact tested dynamic QK tail model above the accepted byte-copy link.

The proof is runtime integer bit inspection; original ordered FP recurrences,
query2..7 fallback, caller, parameter bytes and both execution domains remain.
"""
from qwen3.compiler_dependency import compiler_root
import copy
import json
from pathlib import Path
import shutil
import subprocess
from hexagon_compiled_artifact_cache import canonical,digest_file
from hexagon_global_memcpy import checked,require_owner

ROOT=Path(__file__).resolve().parents[3]
CONTRACT='original_Qwen_q2to32_QK_M8to32_identical_FMA_tail_row_contiguous_positive_zero_proof_v1'
INPUTS={'q2_tail_model_review','q2_tail_model_review_owner','q2_tail_nonempty_review','q2_tail_nonempty_review_owner','q2_tail_linker'}
MODEL_STATUS='actual_Q2_row_major_guard_model_ABBA12_logits_56KV_and_3FNV_independently_revisited_NOT_generation_acceptance'
NONEMPTY_STATUS='actual_Q2_nonempty_KV_tail_model_ABBA12_logits_56KV_and_3FNV_independently_revisited_NOT_generation_acceptance'


def require_scope(model,nonempty):
    for review,status,queries,original,derived in ((model,MODEL_STATUS,[8,2,32],[8],[2,32]),(nonempty,NONEMPTY_STATUS,[6,8,32],[6],[8,32])):
        if (review.get('status')!=status or any(type(review.get(k))is not int or review[k]!=n for k,n in [('DSP_calls',12),('new_DSP_calls',0),('model_PRIMARY_compilations',0)])
                or review.get('complete_model_packet_numerical_comparison')is not True
                or any(review.get(k)is not False for k in ('model_inference_acceptance','generation_regression','performance_acceptance'))
                or review.get('original_ARM_W8A8_queries')!=original or review.get('derived_DSP_reference_queries')!=derived
                or [r['query']for r in review['cases']]!=queries or any(len(r['outputs'])!=4 for r in review['cases'])):
            raise ValueError('Independent complete model12 for cold and actual nonempty KV required')
    if (nonempty.get('occupied_KV_positions')!=8 or nonempty.get('all_original_56_KV_inputs_preserved')is not True
            or model['row_guard']!=nonempty['row_guard']or model['link']!=nonempty['link']
            or model['unchanged_q1_library']!=nonempty['unchanged_q1_library']):
        raise ValueError('Actual retained prefix, exact same new guard/model and unchanged q1 required')
    timing={r['query']:r['candidate_over_baseline']for r in model['private_engineering_timings']}
    if set(timing)!={2,8,32}or any(type(x)not in (int,float)or not 0<x for x in timing.values())or timing[8]>=0.97 or timing[32]>=0.97 or timing[2]>1.03:
        raise ValueError('Measured complete-model q8/q32 gain without small-query regression required')


def authenticate(inputs):
    if not INPUTS<=set(inputs):raise ValueError('Explicit model and nonempty proof inputs required')
    for name in INPUTS:checked(inputs[name])
    if checked(inputs['q2_tail_linker']).resolve()!=Path(__file__).resolve():
        raise ValueError('Only the actual authenticated tail linker may run')
    model=json.loads(checked(inputs['q2_tail_model_review']).read_text());nonempty=json.loads(checked(inputs['q2_tail_nonempty_review']).read_text())
    require_scope(model,nonempty)
    for name,review,script in [('model',model,'revisit_q2_tail_row_model.py'),('nonempty',nonempty,'revisit_q2_tail_kv_model.py')]:
        require_owner(checked(inputs['q2_tail_'+name+'_review_owner']),review,script,Path(inputs['q2_tail_'+name+'_review']['path']))
    manifest=json.loads(checked(model['manifest']).read_text());kv_manifest=json.loads(checked(nonempty['manifest']).read_text())
    if (manifest['libraries']!=kv_manifest['libraries']or manifest['contract']!=kv_manifest['contract']
            or manifest['row_guard']!=model['row_guard']or kv_manifest['row_guard']!=model['row_guard']
            or manifest['link']!=model['link']or kv_manifest['link']!=model['link']):
        raise ValueError('Both actually executed complete model manifests must bind one exact library')
    link=json.loads(checked(model['link']).read_text());row=manifest['row_guard']
    require_owner(checked(manifest['link_controller']),link,'link_q2_tail_model.py',Path(model['link']['path']).parent)
    code=json.loads(checked(link['model_code']).read_text())
    require_owner(checked(link['model_controller']),code,'compile_q2_tail_model_candidate.py',Path(link['model_code']['path']).parent)
    sdk=json.loads(checked(code['SDK']).read_text());require_owner(checked(code['SDK_controller']),sdk,'build_q2_tail_model_sdk.py',Path(code['SDK']['path']).parent)
    build=json.loads(checked(row['build']).read_text());physical=json.loads(checked(row['review']).read_text())
    require_owner(checked(row['build_controller']),build,'build_q2_tail_row_guard.py',Path(row['build']['path']).parent)
    require_owner(checked(row['review_controller']),physical,'revisit_q2_tail_row_guard.py',Path(row['review']['path']))
    if (code['status']!='one_existing_Q2_identical_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code['model_PRIMARY_compilations']!=1 or code['query_bounds']!=[2,32]or code['options']['addFastMath']!='False'
            or code['DSP_calls']or code['model_acceptance']or code['performance_acceptance']
            or len(code['unchanged_constant_objects'])!=3 or row['primary']!=code['primary']
            or physical['status']!='actual_Q2_row_major_guard_ABBA256_full_bits_four_rounding_FP_flags_and_extents_independently_revisited_NOT_model'
            or physical['DSP_calls']!=256 or physical['new_DSP_calls']or physical['primitive_positive_gain']is not True
            or physical['four_rounding_modes_original_target_bits_and_FP_flags_equal']is not True
            or physical['all_runtime_guard_counts_and_independent_positive_zero_extents_equal']is not True
            or physical['receipt']!=row['build']or row['guard_object']!=build['integer_guard_object']
            or build['isolated_operator_compilations']or build['new_model_PRIMARY_compilations']
            or row['replay']['sha256']!=link['library']['sha256']):
        raise ValueError('Strict unchanged FP primary, constants and actual newly target-proved guard required')
    for ref in (code['primary'],code['actual_linalg'],code['compiler_diagnostics'],sdk['candidate'],row['guard_object'],build['integer_guard_variant'],manifest['libraries']['candidate'],manifest['libraries']['baseline']):checked(ref)
    if digest_file(compiler_root()/'tools/qwen3/runtime/profile/q2_qk_positive_zero_extent_row_major.c')!=build['integer_guard_variant']['sha256']:
        raise ValueError('Exact target-proved integer predicate source required')
    return dict(model=model,nonempty=nonempty,manifest=manifest,link=link,code=code,sdk=sdk,row=row,build=build)


def relink(linked,bounds,inputs):
    if tuple(bounds)!=(2,32):raise ValueError('QK prefill tail applies only to existing q2..32 domain')
    proof=authenticate(inputs);link,code,row=proof['link'],proof['code'],proof['row']
    folder=Path(linked).parent;path=Path(linked)/'result.json';base=json.loads(path.read_text());objects_path=folder/'objects/result.json';objects=json.loads(objects_path.read_text())
    if ('q2_identical_tail'in base or base['library']['sha256']!=proof['manifest']['libraries']['baseline']['sha256']
            or digest_file(checked(base['library']))!=link['baseline_library']['sha256']
            or base.get('global_copy',{}).get('all_copy_symbols_locally_bound')is not True
            or objects['objects'][0]['sha256']!=link['original_primary']['sha256']
            or objects['objects'][1:]!=code['unchanged_constant_objects']
            or objects['inputs']['linalg']['sha256']!=code['actual_linalg']['sha256']):
        raise ValueError('Exact accepted global-copy q2 library, old primary and original three constants must precede tail reuse')
    primary_old=checked(objects['objects'][0]);command=base['command'][:]
    if command.count(str(primary_old))!=1:raise ValueError('Exactly one original primary link input required')
    base_link=Path(linked)/'before-q2-identical-tail-result.json';base_objects=folder/'objects/before-q2-identical-tail-result.json'
    if base_link.exists()or base_objects.exists():raise ValueError('Fresh final tail stage required')
    base_link.write_bytes(path.read_bytes());base_objects.write_bytes(objects_path.read_bytes())
    primary=folder/'objects/q2-identical-tail-primary.o';shutil.copyfile(checked(code['primary']),primary)
    diagnostics=folder/'objects/q2-identical-tail-compiler-diagnostics.txt';shutil.copyfile(checked(code['compiler_diagnostics']),diagnostics)
    zero=next(ref for ref in link['guard_objects']if Path(ref['path']).name=='zero_tail_hvx_guard.c.o')
    helpers=[]
    for name,ref in [('q2-identical-tail-guard.o',row['guard_object']),('q2-identical-tail-zero-words.o',zero)]:
        target=folder/'runtime'/name;shutil.copyfile(checked(ref),target)
        if digest_file(target)!=ref['sha256']:raise ValueError('Owned exact target-tested guard bytes changed')
        helpers.append(target)
    candidate=Path(linked)/'libqwen_q2_identical_tail.so'
    command[command.index(str(primary_old))]=str(primary);command[command.index('-o')+1]=str(candidate);command[1:1]=list(map(str,helpers))
    with (Path(linked)/'q2-identical-tail-link.log').open('xb')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if digest_file(candidate)!=proof['manifest']['libraries']['candidate']['sha256']:
        raise ValueError('Public cache must reproduce the exact complete-model-tested row-guard DSO bytes')
    result=copy.deepcopy(objects);result['objects'][0].update(path=str(primary),sha256=digest_file(primary),bytes=primary.stat().st_size)
    result['inputs']['compiler']=proof['sdk']['candidate'];result['compiler_diagnostics']=dict(path=str(diagnostics),sha256=digest_file(diagnostics))
    result.update(q2_identical_tail_producer=link['model_code'],q2_identical_tail_model_review=inputs['q2_tail_model_review'],q2_identical_tail_base_objects=dict(path=str(base_objects),sha256=digest_file(base_objects)),new_model_object_codegen_invocations=0)
    objects_path.write_bytes(canonical(result)+b'\n')
    result=dict(base,command=command,source_sha256=digest_file(__file__),library=dict(path=str(candidate),sha256=digest_file(candidate),bytes=candidate.stat().st_size))
    result['q2_identical_tail']=dict(contract=CONTRACT,base_link=dict(path=str(base_link),sha256=digest_file(base_link)),base_objects=dict(path=str(base_objects),sha256=digest_file(base_objects)),
        model_review=inputs['q2_tail_model_review'],nonempty_review=inputs['q2_tail_nonempty_review'],linker=inputs['q2_tail_linker'],
        primary=dict(path=str(primary),sha256=digest_file(primary)),guard_objects=[dict(path=str(p),sha256=digest_file(p))for p in helpers],
        original_FMA_order_caller_FP_helpers_and_constants_unchanged=True,model_acceptance=False)
    path.write_bytes(canonical(result)+b'\n')
    # The cache registers the final linked library itself; list only additional artifacts.
    return [str(p.relative_to(folder))for p in (primary,diagnostics,*helpers,base_link,base_objects,Path(linked)/'q2-identical-tail-link.log')]
