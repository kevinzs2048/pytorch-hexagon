"""Numerical closure helpers for the separately named global byte-copy candidate."""
import json
from pathlib import Path
import subprocess
import sys
from build import ROOT,entry
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner

sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
from hexagon_zero_tail import checked
from hexagon_global_memcpy import authenticate

COLD_STATUS = 'actual_public_model_local_global_copy_cold9_fresh_ARM_CPU_bitwise_logits_KV_complete_packets_raw_RPC_3FNV_and_normal_owners_independently_revisited_NOT_full393_or_delivery_acceptance'
FULL_STATUS = 'global_copy_original393_complete_NOT_independently_accepted'
REVIEW_STATUS = 'global_copy_original393_independently_revisited'


def tested_model(build):
    recipe=json.loads(checked(build['recipe']).read_text())
    from hexagon_global_memcpy import authenticate as authenticate_copy
    review,model=authenticate_copy(recipe['inputs'])
    if recipe.get('global_hvx_memcpy') is not True:
        raise ValueError('Explicit model-local byte-copy recipe required')
    expected={tuple(row['query_bounds']):row['candidate']['sha256'] for row in model['models']}
    if set(expected)!={(1,1),(2,32)}:
        raise ValueError('Exactly the two reviewed dynamic model domains required')
    return model,expected


def require_cold_review(proof, *, cold, recipe, libraries, baseline):
    if (proof.get('status') != COLD_STATUS or type(proof.get('DSP_calls')) is not int or proof['DSP_calls'] != 9
            or proof.get('new_DSP_calls') != 0 or proof.get('new_model_object_codegen_invocations') != 0
            or proof.get('generation_regression') is not True or proof.get('model_inference_acceptance') is not False
            or proof.get('performance_acceptance') is not False
            or proof.get('actual_native') != entry(cold/'native-result.json')
            or proof.get('cold_acceptance') != entry(cold/'acceptance.json')
            or proof.get('baseline') != entry(baseline/'native-result.json')):
        raise ValueError('Exact independently bitwise CPU-reviewed global-copy cold9 required')
    from run_integer_regressions import candidate_libraries
    base,actual = candidate_libraries(cold)
    if actual != libraries or base['recipe'] != entry(recipe):
        raise ValueError('global-copy cold9 must bind this exact recipe and both libraries')
    refs = []; indexed_files(proof,refs)
    owner = json.loads(checked(proof['controller']).read_text())
    require_normal_owner(proof['controller']['path'],owner['authorized_deadline_unix'])
    if owner['source_root'] != base['source_root'] or owner['source_revision'] != base['revision']:
        raise ValueError('Actual normal frozen numerical producer required')
    return base


def require_reviewer_owner(proof,controller):
    owner = json.loads(Path(controller).read_text())
    require_normal_owner(controller,owner['authorized_deadline_unix'])
    source = Path(proof['source']['path']); root = source.parents[2]
    git = lambda *args: subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if (owner['source_root'] != str(root) or owner['source_revision'] != proof['source_revision']
            or owner['command'][1] != str(source) or git('rev-parse','HEAD') != proof['source_revision']
            or git('status','--porcelain') or entry(source) != proof['source']):
        raise ValueError('Exact frozen independently closed global-copy cold reviewer required')


def revisit_numerical_cold_cache(cold,base):
    # This helper is generic: it constructs the genuine numerical producer's
    # cache class in a fresh interpreter and verifies read-only hits only.
    from zero_tail_release import revisit_numerical_cold_cache as revisit
    return revisit(cold,base)
