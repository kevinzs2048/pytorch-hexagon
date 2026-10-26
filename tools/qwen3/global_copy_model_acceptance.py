"""Bind byte-copy delivery to its own full numerical closure and normal owners."""
import hashlib
import json
from pathlib import Path
import subprocess

from build import entry, sha
from caller_copy_native_gain import positive_native_gain
from revisit_global_copy_native393 import require_complete
from revisit_native_compile_session_abba import require_normal_owner
from revisit_q2_dispatch_preflight import actual_owner


def require_scope(closure, proof):
    if (require_complete(closure) is not True
            or proof.get('status') != 'global_copy_original393_independently_revisited'
            or proof.get('global_copy_candidate') is not True
            or proof.get('fresh_cold9_independently_reviewed') is not True
            or proof.get('model_inference_acceptance') is not True
            or proof.get('positive_native_callback_gain') is not True
            or proof.get('model_math_and_owned_lifetimes_unchanged') is not True
            or proof.get('ordered_FMA_precision_and_owned_lifetimes_preserved') is not True
            or proof.get('production_TUI_speedup_claimed') is not False
            or len(proof.get('raw_invocation_traces', [])) != 12
            or len(proof.get('compile_chains', [])) != 12
            or len(proof.get('states', {}).get('proofs', [])) != 6
            or proof.get('states', {}).get('DSP_calls') != 304
            or not positive_native_gain(proof.get('native', {}).get('raw_native_measurements', {}))):
        raise ValueError('Own independently accepted global-copy full393 and positive native gain required')
    for row in (closure, proof):
        if any(type(row.get(k)) is not int or row[k] != n for k, n in (
                ('DSP_calls',393), ('candidate_DSP_calls',361), ('paired_baseline_DSP_calls',32),
                ('newly_executed_DSP_calls',384), ('reused_independently_verified_DSP_calls',9))):
            raise ValueError('Exact full393 scope and separate new/reused counts required')
    if type(proof.get('new_DSP_calls')) is not int or proof['new_DSP_calls'] != 0:
        raise ValueError('Only an independent zero-DSP revisit can accept the model')


def require_global_copy(index, model, closure, proof):
    require_scope(closure, proof)
    recipe = json.loads(Path(index['machine_recipe']['path']).read_text())
    if (recipe.get('global_hvx_memcpy') is not True
            or sha(index['machine_recipe']['path']) != index['machine_recipe']['sha256']
            or proof.get('closure') != entry(model['acceptance']['path'])
            or proof.get('candidate_libraries') != index['candidate_libraries']
            or closure.get('candidate_libraries') != index['candidate_libraries']
            or closure.get('recipe') != index['machine_recipe']
            or any(row.get('numerical_source_root') != model['source_root']
                   or row.get('numerical_source_revision') != model['source_revision'] for row in (closure, proof))):
        raise ValueError('Delivery must bind this exact numerical source, recipe and both model libraries')
    source = Path(model['source_root'])
    git = lambda root,*args: subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git(source,'rev-parse','HEAD') != model['source_revision'] or git(source,'status','--porcelain'):
        raise ValueError('Accepted numerical source must remain frozen')
    owner_path = Path(proof['controller']['path'])
    if entry(owner_path) != proof['controller']:
        raise ValueError('Full numerical controller changed')
    actual_owner(owner_path,closure,'run_global_copy_native393.py',Path(model['acceptance']['path']).parent)
    reviewer = Path(proof['reviewer_source']['path']); root = reviewer.parents[2]
    blob = subprocess.check_output(['git','-C',str(root),'show',
        proof['reviewer_revision']+':'+str(reviewer.relative_to(root))])
    if entry(reviewer) != proof['reviewer_source'] or hashlib.sha256(blob).hexdigest() != proof['reviewer_source']['sha256']:
        raise ValueError('Exact committed independently qualified reviewer required')
    from global_copy_release import require_cold_review, require_reviewer_owner
    cold_proof = json.loads(Path(closure['reused_cold_review']['path']).read_text())
    from run_integer_regressions import candidate_libraries
    _,libraries = candidate_libraries(Path(closure['cold_run']))
    if {str(k):v for k,v in libraries.items()} != index['candidate_libraries']:
        raise ValueError('Own fresh ARM-reviewed cold run must contain the selected libraries')
    require_cold_review(cold_proof,cold=Path(closure['cold_run']),recipe=Path(index['machine_recipe']['path']),
        libraries=libraries,baseline=Path(closure['baseline_cold']))
    require_reviewer_owner(cold_proof,closure['reused_cold_reviewer_controller']['path'])
