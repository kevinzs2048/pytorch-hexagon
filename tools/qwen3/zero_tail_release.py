"""Bind the existing release queue to one already compiled q1 candidate."""
import json
from pathlib import Path
import subprocess
import sys

from build import ROOT, entry, sha
from revisit_release_index import indexed_files
from revisit_native_compile_session_abba import require_normal_owner

sys.path.insert(0, str(ROOT/'qwen3/runtime/arm'))
from hexagon_zero_tail import authenticate, checked

COLD_STATUS = 'actual_public_cold9_fresh_ARM_CPU_bitwise_logits_KV_complete_original_packets_live_handoff_raw_RPC_3FNV_and_normal_owners_independently_revisited_NOT_full393_or_delivery_acceptance'
FULL_STATUS = 'zero_tail_original393_complete_NOT_independently_accepted'
REVIEW_STATUS = 'zero_tail_original393_independently_revisited'
RELEASE_KEY = 'qualified_zero_tail_native_release'


def tested_model(build):
    recipe = json.loads(checked(build['recipe']).read_text())
    authenticate(recipe['inputs'])
    link = json.loads(checked(recipe['inputs']['zero_tail_link']).read_text())
    prior = json.loads(checked(link['caller_review']).read_text())
    closure = json.loads(checked(prior['closure']).read_text())
    caller_build = json.loads(checked(closure['recipe_build']).read_text())
    model = json.loads(checked(caller_build['tested_model']).read_text())
    expected = {tuple(row['query_bounds']): row['library']['sha256'] for row in model['models']}
    expected[(1, 1)] = link['library']['sha256']
    return model, expected


def require_cold_review(proof, *, cold, recipe, libraries, baseline):
    if (proof.get('status') != COLD_STATUS or type(proof.get('DSP_calls')) is not int
            or proof['DSP_calls'] != 9 or proof.get('new_DSP_calls') != 0
            or proof.get('new_model_object_codegen_invocations') != 0
            or proof.get('generation_regression') is not True
            or proof.get('model_inference_acceptance') is not False
            or proof.get('performance_acceptance') is not False
            or proof.get('actual_native') != entry(cold/'native-result.json')
            or proof.get('cold_acceptance') != entry(cold/'acceptance.json')
            or proof.get('baseline') != entry(baseline/'native-result.json')):
        raise ValueError('Only the exact bitwise CPU-reviewed original cold9 may be reused')
    from run_integer_regressions import candidate_libraries
    base, actual = candidate_libraries(cold)
    if actual != libraries or base['recipe'] != entry(recipe):
        raise ValueError('Reused cold9 must retain this exact recipe and both tested DSOs')
    refs = []; indexed_files(proof, refs)
    owner = json.loads(checked(proof['controller']).read_text())
    require_normal_owner(proof['controller']['path'], owner['authorized_deadline_unix'])
    if owner['source_root'] != base['source_root'] or owner['source_revision'] != base['revision']:
        raise ValueError('Cold9 must retain its actual frozen normal numerical producer')
    return base


def require_reviewer_owner(proof, controller):
    owner = json.loads(Path(controller).read_text())
    require_normal_owner(controller, owner['authorized_deadline_unix'])
    source = Path(proof['source']['path'])
    root = source.parents[2]
    git = lambda *args: subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if (owner['source_root'] != str(root) or owner['source_revision'] != proof['source_revision']
            or owner['command'][1] != str(source) or git('rev-parse', 'HEAD') != proof['source_revision']
            or git('status', '--porcelain') or entry(source) != proof['source']):
        raise ValueError('Cold independent review must have its actual frozen normal owner')


def main():
    import argparse
    import copy
    from build import write
    from native_model_acceptance import require_model_acceptance, selected_dynamic_model
    p = argparse.ArgumentParser(description='Prepare an opt-in delivery index only after original full393 qualification')
    for key in ('parent-index', 'closure', 'review', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.is_absolute():
        raise ValueError('Fresh owned selection required')
    read = lambda path: json.loads(Path(path).read_text())
    closure = read(a.closure/'result.json'); proof = read(a.review)
    if closure['status'] != FULL_STATUS or proof['status'] != REVIEW_STATUS:
        raise ValueError('Complete independently qualified zero-tail393 required')
    parent = copy.deepcopy(read(a.parent_index))
    version = 'ordered_q1_zero_tail_dynamic_Qwen_W8A8'
    parent[RELEASE_KEY] = dict(version=version, source_root=closure['numerical_source_root'],
        source_revision=closure['numerical_source_revision'], closure=entry(a.closure/'result.json'),
        independent_audit=entry(a.review), candidate_libraries=closure['candidate_libraries'],
        recipe=closure['recipe'], model_inference_acceptance=proof['model_inference_acceptance'])
    a.output.mkdir(); write(a.output/'parent-index.json', parent)
    index = copy.deepcopy(parent)
    index['deployment_selection'] = dict(parent_index=entry(a.output/'parent-index.json'), release_key=RELEASE_KEY)
    index['current_default_accepted_version'] = version
    index['accepted_versions'][version] = selected_dynamic_model(parent, RELEASE_KEY)
    index['candidate_libraries'] = closure['candidate_libraries']
    index['machine_recipe'] = closure['recipe']
    require_model_acceptance(index)
    write(a.output/'index.json', index)
    write(a.output/'result.json', dict(status='independently_qualified_zero_tail_opt_in_index_prepared_NOT_warm_delivery',
        parent_index=entry(a.parent_index), index=entry(a.output/'index.json'),
        shared_default_changed=False, DSP_calls=0))


def revisit_numerical_cold_cache(cold, base):
    """Use the actual producer's cache class in a fresh read-only interpreter.

    Cache authentication owns absolute reuse-source paths. A maintenance
    checkout may review receipts, but cannot impersonate that numerical owner.
    """
    source = Path(base['source_root'])
    git = lambda *args: subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != base['revision'] or git('status', '--porcelain'):
        raise ValueError('The actual frozen cold cache producer is required')
    script = '''
import json,sys
from pathlib import Path
root,cold=map(Path,sys.argv[1:3])
sys.path[:0]=[str(root/'tools/qwen3'),str(root/'qwen3/runtime/arm'),str(root)]
from run_integer_regressions import candidate_libraries
from audit_prepared_stage_reuse import cold_cache
base,_=candidate_libraries(cold)
assert base == json.loads(sys.argv[3])
print(json.dumps(cold_cache(cold,base)))
'''
    import os
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    proof = json.loads(subprocess.check_output([sys.executable, '-c', script,
        str(source), str(cold), json.dumps(base)], text=True, env=environment, cwd=source))
    proof['numerical_cache_source'] = dict(source_root=str(source), source_revision=base['revision'],
        validator=entry(source/'tools/qwen3/audit_prepared_stage_reuse.py'),
        cache_class=entry(source/'qwen3/runtime/arm/hexagon_qwen_captured_machine_cache.py'))
    return proof


if __name__ == '__main__':
    main()
