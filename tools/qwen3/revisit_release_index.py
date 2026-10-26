"""Rehash indexed receipts and explicit frozen sources without compiler or DSP.

Historical receipts remain historical; the supplied protection manifest names
the current source heads. The dirty checkout scope is HEAD and index only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from build import ROOT, entry, sha, write


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def indexed_files(value, rows):
    if isinstance(value, dict):
        if isinstance(value.get('path'),str) and 'sha256' in value:
            path=Path(value['path'])
            if not path.is_absolute() or not path.is_file() or sha(path)!=value['sha256']:
                raise ValueError('Indexed receipt missing or changed: '+str(path))
            if 'bytes' in value and path.stat().st_size!=value['bytes']:
                raise ValueError('Indexed file size changed: '+str(path))
            rows.append(dict(path=str(path),sha256=value['sha256'],bytes=path.stat().st_size))
        for item in value.values(): indexed_files(item,rows)
    elif isinstance(value,list):
        for item in value: indexed_files(item,rows)


def current_metadata(index, references):
    iteration = index.get('current_8h_iteration')
    if iteration is not None:
        count = iteration['completed_model_calls_independently_verified']
        if type(count) is not int or count < 393 or count != iteration['completed_model_DSP_calls']:
            raise ValueError('Current completed model count aliases disagree')
    backup = index.get('current_maintenance_source_backup')
    if backup is not None:
        proof = json.loads(Path(backup['path']).read_text())
        root = index['maintenance_delivery_source_root']
        if (proof['status'] != 'source_base_delta_actual_restore_HEAD_tree_clean_fsck_passed'
                or proof['source_root'] != root
                or proof['source_revision'] != index['maintenance_delivery_revision']
                or proof['source_revision'] != index['maintenance_revision']
                or git(root, 'rev-parse', 'HEAD') != proof['source_revision']
                or git(root, 'rev-parse', 'HEAD^{tree}') != proof['tree']
                or index['maintenance_source_backup'] != backup
                or index['maintenance_delivery_source_backup'] != backup):
            raise ValueError('Current maintenance heads, tree and actual recovery pointers disagree')
        indexed_files(proof, references)


def accepted_transport(index, references):
    record = index.get('accepted_HVX_transport')
    if record is None:
        return None
    if record.get('acceptance_mode') == 'model_native393_common_HVX_runtime':
        from common_hvx_model_acceptance import revisit
        return revisit(index, record, references, indexed_files, accepted_transport_parts, git)
    if record.get('acceptance_mode') == 'completed_parts_after_pre_DSP_admission_refusal':
        return accepted_transport_parts(index, record, references)
    current = index['current_default_accepted_version']
    numerical = index['accepted_versions'][current]
    result = json.loads(Path(record['acceptance']['path']).read_text())
    proof = json.loads(Path(record['independent_audit']['path']).read_text())
    if (record['numerical_version'] != current or record['runtime_catalog'] != index['runtime_catalog']
            or record['numerical_baseline_acceptance'] != numerical['acceptance']
            or result['status'] != 'unchanged_model_exact_HVX_block_transport_native_timing_and_original_states_accepted'
            or result['DSP_calls'] != 384 or result['candidate_DSP_calls'] != 352
            or result['paired_baseline_DSP_calls'] != 32 or result['model_inference_acceptance'] is not True
            or result['actual_model_compilations'] != 0 or result['model_DSOs_and_recipe_unchanged'] is not True
            or result['hvx_block_protocol'] is not True
            or result['source_revision'] != record['source_revision']
            or result['numerical_revision'] != numerical['source_revision']
            or result['numerical_source_root'] != numerical['source_root']
            or result['runtime_catalog'] != {k:record['runtime_catalog'][k] for k in ('path','sha256')}
            or git(record['source_root'], 'rev-parse', 'HEAD') != record['source_revision']
            or git(record['source_root'], 'status', '--porcelain')):
        raise ValueError('Default runtime must bind complete unchanged-model HVX384 and the current numerical393')
    if (proof['status'] != 'unchanged_model_HVX_transport384_all_states_raw_native_and_constant_bytes_independently_revisited'
            or proof['DSP_calls'] != 384 or proof['new_DSP_calls'] != 0
            or proof['model_inference_acceptance'] is not True or proof['actual_model_compilations'] != 0
            or proof['source_revision'] != record['source_revision']
            or proof['numerical_source_revision'] != numerical['source_revision']
            or proof['closure'] != {k:record['acceptance'][k] for k in ('path','sha256')}
            or proof['controller'] != {k:record['controller_process'][k] for k in ('path','sha256')}
            or proof['all_original_and_candidate_constant_bytes_reverified'] is not True
            or len(proof['sessions']) != 11 or sum(s['DSP_calls'] for s in proof['sessions']) != 384):
        raise ValueError('Complete independently rechecked runtime packets, states, owners and constant bytes required')
    baseline = Path(numerical['acceptance']['path']).parent
    old_runtime = json.loads((baseline/'warm16/commands.json').read_text())['runtime_override']
    rollback = index['runtime_rollback']
    if (rollback['numerical_version'] != current
            or rollback['runtime_catalog']['sha256'] != old_runtime['sha256']
            or rollback['runtime_catalog']['sha256'] == record['runtime_catalog']['sha256']):
        raise ValueError('Runtime rollback must retain the actually accepted same-model original393 transport')
    source_backup = json.loads(Path(record['source_backup']['path']).read_text())
    if (source_backup['status'] != 'source_full_bundle_actual_restore_HEAD_tree_clean_fsck_passed'
            or source_backup['source_root'] != record['source_root']
            or source_backup['source_revision'] != record['source_revision']
            or source_backup['tree'] != git(record['source_root'], 'rev-parse', 'HEAD^{tree}')
            or source_backup['temporary_restore_removed'] is not True):
        raise ValueError('Accepted transport source recovery must match its frozen source')
    indexed_files(source_backup, references)
    return dict(source_revision=record['source_revision'], numerical_version=current,
                DSP_calls=384, independent_audit=record['independent_audit'])


def accepted_transport_parts(index, record, references):
    """Eleven complete sessions; original failed/unrecorded controllers stay so."""
    current=index['current_default_accepted_version'];numerical=index['accepted_versions'][current]
    proof=json.loads(Path(record['independent_audit']['path']).read_text())
    def identity(value):return {k:value[k] for k in ('path','sha256')}
    if (record['acceptance'] != record['independent_audit']
            or record['numerical_version'] != current
            or record['numerical_baseline_acceptance'] != numerical['acceptance']
            or record['runtime_catalog'] != index['runtime_catalog']
            or proof['status'] != 'unchanged_model_HVX_transport384_completed_parts_all_states_raw_native_and_constant_bytes_independently_revisited'
            or proof['acceptance_mode'] != record['acceptance_mode']
            or proof['DSP_calls'] != 384 or proof['candidate_DSP_calls'] != 352
            or proof['paired_baseline_DSP_calls'] != 32 or proof['new_DSP_calls'] != 0
            or proof['model_inference_acceptance'] is not True or proof['actual_model_compilations'] != 0
            or proof['model_DSOs_and_recipe_unchanged'] is not True or proof['hvx_block_protocol'] is not True
            or proof['source_revision'] != record['source_revision']
            or proof['numerical_source_revision'] != numerical['source_revision']
            or proof['numerical_source_root'] != numerical['source_root']
            or proof['baseline_closure'] != identity(numerical['acceptance'])
            or proof['runtime_catalog'] != identity(record['runtime_catalog'])
            or proof['controller'] != identity(record['controller_process'])
            or proof['original_failed_controller'] != identity(record['original_failed_controller'])
            or proof['original_failed_closure'] != identity(record['original_failed_closure'])
            or proof['failed_decode_DSP_calls'] != 0
            or proof['all_original_and_candidate_constant_bytes_reverified'] is not True
            or len(proof['sessions']) != 11 or sum(s['DSP_calls'] for s in proof['sessions']) != 384
            or proof['states']['DSP_calls'] != 304 or len(proof['states']['proofs']) != 6
            or proof['loader_failure_fixed'] is not False or proof['production_launcher_switched'] is not False
            or git(record['source_root'],'rev-parse','HEAD') != record['source_revision']
            or git(record['source_root'],'status','--porcelain')
            or sha(proof['reviewer_source']['path']) != proof['reviewer_source']['sha256']):
        raise ValueError('Explicit complete-parts384 must bind unchanged model393, six states and the exact reviewed source')
    failed=json.loads(Path(record['original_failed_controller']['path']).read_text())
    old=json.loads(Path(record['original_failed_closure']['path']).read_text())
    controller=json.loads(Path(record['controller_process']['path']).read_text())
    remaining=json.loads(Path(proof['continuation_states']['path']).read_text())
    if (failed['status'] != 'failed' or failed['returncode'] != 1
            or old['status'] != 'failed' or old['DSP_calls'] != 80
            or failed['authorized_deadline_unix']-failed['start_unix'] < 10800
            or controller['authorized_deadline_unix'] != failed['authorized_deadline_unix']
            or controller['original_failed_controller'] != identity(record['original_failed_controller'])
            or controller['source_revision'] != numerical['source_revision']
            or remaining['status'] != 'all_integer_candidate_state_regressions_passed'
            or remaining['DSP_calls'] != 223
            or [(c['case'],c['DSP_calls']) for c in remaining['cases']] != [('decode993',95),('code64',128)]):
        raise ValueError('Preserved pre-DSP refusal and exact original remaining223 required')
    if proof['continuation_completion']=='outer_exit_unrecorded_owners_gone_complete_model_sessions_revisited':
        if (controller['status'] != 'running' or 'returncode' in controller
                or proof['outer_controller_normal_exit_claimed'] is not False):
            raise ValueError('Unrecorded outer exit must remain explicitly unrecorded')
    elif proof['continuation_completion']=='recorded_normal_outer_exit':
        if controller['status'] not in ('completed','complete') or controller['returncode'] != 0:
            raise ValueError('Actual recorded normal continuation exit required')
    else:raise ValueError('Unknown continuation completion evidence')
    for owner in (failed,controller):
        for key in ('controller_pid','child_pid'):
            pid=owner[key]
            if type(pid) is not int or pid<=0 or Path('/proc',str(pid)).exists():
                raise ValueError('Actual old and continuation host owners must be gone')
    old_runtime=json.loads((Path(numerical['acceptance']['path']).parent/'warm16/commands.json').read_text())['runtime_override']
    rollback=index['runtime_rollback']
    if (rollback['numerical_version'] != current
            or rollback['runtime_catalog']['sha256'] != old_runtime['sha256']
            or rollback['runtime_catalog']['sha256'] == record['runtime_catalog']['sha256']):
        raise ValueError('Same-model accepted scalar transport rollback required')
    backup=json.loads(Path(record['source_backup']['path']).read_text())
    if (backup['status'] != 'source_full_bundle_actual_restore_HEAD_tree_clean_fsck_passed'
            or backup['source_root'] != record['source_root']
            or backup['source_revision'] != record['source_revision']
            or backup['tree'] != git(record['source_root'],'rev-parse','HEAD^{tree}')
            or backup['temporary_restore_removed'] is not True):
        raise ValueError('Exact accepted transport actual source recovery required')
    indexed_files(proof,references);indexed_files(backup,references)
    return dict(source_revision=record['source_revision'],numerical_version=current,DSP_calls=384,
        acceptance_mode=record['acceptance_mode'],independent_audit=record['independent_audit'],
        outer_controller_normal_exit_claimed=proof['outer_controller_normal_exit_claimed'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source-protection','dirty-source','original-sdk','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or git(ROOT,'status','--porcelain'):
        raise ValueError('Fresh absolute result and clean committed reviewer required')
    index=json.loads(args.index.read_text())
    protection=json.loads(args.source_protection.read_text())
    current=index['current_default_accepted_version']
    rollback=index['rollback']['previous_accepted_version']
    if (current not in index['accepted_versions'] or rollback not in index['accepted_versions']
            or index['complete_DSP_calls']!=393 or index['production_launcher_switched'] is not False):
        raise ValueError('Current and rollback must retain full original393 acceptance')
    references=[]
    indexed_files(index,references)
    current_metadata(index,references)
    runtime = accepted_transport(index,references)
    for version in (current,rollback):
        record=index['accepted_versions'][version]
        closure=json.loads(Path(record['acceptance']['path']).read_text())
        from ordered_pv_release_acceptance import KIND as PV_KIND, require_pv393
        pv = closure.get('candidate_kind') == PV_KIND
        if pv:
            require_pv393(closure,json.loads(Path(record['numerical_audit']['path']).read_text()))
        if (closure.get('DSP_calls')!=393 or closure.get('model_inference_acceptance') is not (False if pv else True)
                or git(record['source_root'],'rev-parse','HEAD')!=record['source_revision']):
            raise ValueError('Current and rollback must bind actual complete frozen model acceptance')
        if version==current and closure['candidate_libraries']!=index['candidate_libraries']:
            raise ValueError('Default model library identities differ from actual accepted closure')
    host_packing=None
    if index.get('accepted_host_input_packing') is not None:
        from host_input_packing_acceptance import revisit as revisit_packing
        host_packing=revisit_packing(index,index['accepted_host_input_packing'],references,indexed_files,git)
    trees=[]
    for row in protection['trees']:
        root=Path(row['path'])
        if git(root,'rev-parse','HEAD')!=row['HEAD'] or git(root,'status','--porcelain'):
            raise ValueError('Explicit frozen source changed: '+str(root))
        trees.append(dict(path=str(root),HEAD=row['HEAD'],clean=True))
    dirty_head=git(args.dirty_source,'rev-parse','HEAD')
    index_path=Path(git(args.dirty_source,'rev-parse','--git-path','index'))
    if not index_path.is_absolute(): index_path=args.dirty_source/index_path
    dirty_index=sha(index_path)
    original_sdk=sha(args.original_sdk)
    if (dirty_head!=protection['dirty_HEAD'] or dirty_index!=protection['dirty_index_sha256']
            or original_sdk!=protection['original_SDK_sha256']):
        raise ValueError('Original dirty HEAD/index or original SDK changed')
    write(args.output,dict(status='current_release_index_all_receipts_and_frozen_sources_rehashed_without_new_DSP',
        index=entry(args.index),source_protection=entry(args.source_protection),current_version=current,
        rollback_version=rollback,indexed_references=references,trees=trees,
        dirty_scope='HEAD_and_index_only_not_all_untracked_contents',dirty_HEAD=dirty_head,
        dirty_index_sha256=dirty_index,original_SDK_sha256=original_sdk,
        production_launcher_switched=False,new_DSP_calls=0,new_compiler_calls=0,
        accepted_HVX_transport=runtime,accepted_host_input_packing=host_packing,reviewer_revision=git(ROOT,'rev-parse','HEAD'),completed_unix=time.time()))
    print('current and rollback index: %d file references and %d frozen sources passed' % (len(references),len(trees)))


if __name__=='__main__': main()
