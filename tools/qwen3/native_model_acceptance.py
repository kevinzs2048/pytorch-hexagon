"""Bind deployment selection to a frozen, independently qualified model.

An opt-in index snapshot selects the qualified dynamic PV release without
changing the shared default or borrowing the old model's cache receipts.
"""
import json
from pathlib import Path
import subprocess

from build import entry, sha


def selected_dynamic_model(parent, release_key='qualified_dynamic_PV_native_release'):
    release = parent[release_key]
    return dict(source_root=release['source_root'], source_revision=release['source_revision'],
                acceptance=release['closure'], numerical_audit=release['independent_audit'])


def require_selected_index(index):
    selection = index.get('deployment_selection')
    if selection is None:
        return False
    reference = selection['parent_index']
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('The frozen parent release index changed')
    parent = json.loads(Path(reference['path']).read_text())
    release_key = selection.get('release_key')
    if release_key not in ('qualified_dynamic_PV_native_release', 'qualified_zero_tail_native_release',
                           'qualified_q2_qk_dispatch_native_release', 'qualified_global_copy_native_release', 'qualified_q2_identical_tail_native_release'):
        raise ValueError('Only explicitly qualified release selection is allowed')
    release = parent[release_key]
    version = release['version']
    if (release.get('model_inference_acceptance') is not True
            or index['current_default_accepted_version'] != version
            or index['accepted_versions'][version] != selected_dynamic_model(parent, release_key)
            or index['candidate_libraries'] != release['candidate_libraries']
            or index['machine_recipe'] != release['recipe']
            or index['runtime_catalog'] != parent['runtime_catalog']):
        raise ValueError('Selection must bind the exact independently qualified dynamic PV release')
    return True


def require_model_acceptance(index):
    """Keep the existing ordered-PV gate; separately admit its qualified successor."""
    from revisit_release_index import indexed_files
    version = index['current_default_accepted_version']
    model = index['accepted_versions'][version]
    refs = []
    indexed_files(model, refs)
    closure = json.loads(Path(model['acceptance']['path']).read_text())
    proof = json.loads(Path(model['numerical_audit']['path']).read_text())
    if not require_selected_index(index):
        from ordered_pv_release_acceptance import require_pv393, require_pv_sources
        require_pv393(closure, proof)
        require_pv_sources(closure, proof, model['source_root'], closure['baseline_closure'])
        return model, closure
    if index['deployment_selection']['release_key']=='qualified_q2_identical_tail_native_release':
        from q2_tail_model_acceptance import require_q2_tail
        require_q2_tail(index,model,closure,proof)
        return model,closure
    if index['deployment_selection']['release_key'] == 'qualified_global_copy_native_release':
        from global_copy_model_acceptance import require_global_copy
        require_global_copy(index, model, closure, proof)
        return model, closure
    zero_tail = index['deployment_selection']['release_key'] == 'qualified_zero_tail_native_release'
    q2_dispatch = index['deployment_selection']['release_key'] == 'qualified_q2_qk_dispatch_native_release'
    prefix = 'q2_dispatch' if q2_dispatch else 'zero_tail' if zero_tail else 'dynamic_PV'
    if (type(closure.get('Q2_QK_dispatch_candidate',False)) is not bool
            or closure.get('Q2_QK_dispatch_candidate',False) is not q2_dispatch):
        raise ValueError('Q2 candidate identity must belong to its separately qualified release key')
    qk=closure.get('QK_guard_candidate',False)
    if type(qk) is not bool or (qk and not (zero_tail or q2_dispatch)):
        raise ValueError('QK successor must retain original full zero-tail qualification')
    if qk:
        recipe=json.loads(Path(index['machine_recipe']['path']).read_text())
        if proof.get('QK_guard_candidate') is not True or recipe.get('qk_tail_query1') is not True:
            raise ValueError('New QK library requires its own exact full393 proof and explicit recipe')
    if q2_dispatch:
        if (not qk or proof.get('Q2_QK_dispatch_candidate') is not True
                or proof.get('positive_prefill_native_gain') is not True
                or proof.get('ordered_FMA_precision_and_owned_lifetimes_preserved') is not True
                or recipe.get('q2_qk_dispatch') is not True):
            raise ValueError('Q2 dispatch requires its own full393, prefill gain, unchanged precision and explicit recipe')
    if (closure.get('status') != prefix+'_original393_complete_NOT_independently_accepted'
            or closure.get('candidate_kind') != ('ordered-q2-runtime-QK-dispatch' if q2_dispatch else
                                                'ordered-q1-positive-zero-tail' if zero_tail else 'ordered-pv-register-q2to32')
            or closure.get('model_inference_acceptance') is not False
            or closure.get('original_state304_executed') is not True
            or proof.get('status') != prefix+'_original393_independently_revisited'
            or proof.get('model_inference_acceptance') is not True
            or proof.get('positive_native_callback_gain' if zero_tail or q2_dispatch else 'positive_native_gain') is not True
            or proof.get('production_TUI_speedup_claimed') is not False
            or proof.get('closure') != entry(model['acceptance']['path'])
            or proof.get('candidate_libraries') != index['candidate_libraries']
            or closure.get('candidate_libraries') != index['candidate_libraries']
            or closure.get('numerical_source_root') != model['source_root']
            or proof.get('numerical_source_root') != model['source_root']
            or closure.get('numerical_source_revision') != model['source_revision']
            or proof.get('numerical_source_revision') != model['source_revision']
            or len(proof.get('raw_invocation_traces', [])) != 12
            or len(proof.get('compile_chains', [])) != 12
            or len(proof.get('states', {}).get('proofs', [])) != 6
            or proof.get('states', {}).get('DSP_calls') != 304):
        raise ValueError('Actual independent full393 dynamic PV acceptance required')
    for row in (closure, proof):
        if any(type(row.get(k)) is not int or row[k] != n for k, n in
               (('DSP_calls', 393), ('candidate_DSP_calls', 361), ('paired_baseline_DSP_calls', 32))):
            raise ValueError('All original dynamic PV calls and their scope required')
    if type(proof.get('new_DSP_calls')) is not int or proof['new_DSP_calls'] != 0:
        raise ValueError('The numerical acceptance must be an independent audit')
    source = Path(model['source_root'])
    git = lambda root, *args: subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git(source, 'rev-parse', 'HEAD') != model['source_revision'] or git(source, 'status', '--porcelain'):
        raise ValueError('The qualified numerical source must stay frozen')
    reviewer = proof['reviewer_source']
    review_root = Path(reviewer['path']).resolve().parents[2]
    relative = str(Path(reviewer['path']).relative_to(review_root))
    blob = subprocess.check_output(['git', '-C', str(review_root), 'show',
                                    proof['reviewer_revision'] + ':' + relative])
    import hashlib
    if sha(reviewer['path']) != reviewer['sha256'] or hashlib.sha256(blob).hexdigest() != reviewer['sha256']:
        raise ValueError('The independent audit must match its actual committed reviewer')
    from revisit_native_compile_session_abba import require_normal_owner
    controller = proof['controller']
    owner = json.loads(Path(controller['path']).read_text())
    if entry(controller['path']) != controller:
        raise ValueError('Actual full393 normal controller receipt changed')
    require_normal_owner(controller['path'], owner['authorized_deadline_unix'])
    return model, closure


def accepted_cold(model, closure):
    if closure.get('candidate_kind') in ('ordered-pv-register-q2to32', 'ordered-q1-positive-zero-tail',
                                       'ordered-q2-runtime-QK-dispatch', 'model-local-exact-global-byte-copy'):
        return Path(closure['cold_run'])
    return Path(model['acceptance']['path']).parent / 'cold'
