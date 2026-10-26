"""Read-only authentication of copied or immutably staged preparation artifacts."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from build import entry, sha


def read(path):return json.loads(Path(path).read_text())


def validate_prepared_reuse(receipt, destination, frontend, recipe, bounds):
    import reuse_accepted_prepared_linalg as producer
    authenticate = producer.authenticate
    shared = receipt.get('shared_immutable_artifacts')
    if read(destination/'reuse-result.json') != receipt:
        raise ValueError('Actual current preparation reuse receipt differs')
    expected_status = ('exact_accepted_prepared_Linalg_staged_NOT_pipeline_or_DSP_acceptance'
                       if shared is True else
                       'exact_accepted_prepared_Linalg_copied_NOT_pipeline_or_DSP_acceptance')
    if (destination.is_symlink() or type(shared) is not bool or receipt.get('status') != expected_status
            or receipt.get('source_sha256') != sha(producer.__file__)
            or receipt.get('query_bounds') != bounds
            or receipt.get('current_frontend') != entry(frontend)
            or receipt.get('current_recipe') != entry(recipe)
            or receipt.get('model_inference_acceptance') is not False
            or any(type(receipt.get(k)) is not int or receipt[k] != 0 for k in
                   ('DSP_calls', 'new_prepare_optimizer_invocations', 'new_frontend_invocations'))):
        raise ValueError('Exact preparation reuse kind, identity and zero new execution required')
    for key in ('reuse_seconds', 'historical_preparation_seconds'):
        if type(receipt.get(key)) not in (int,float) or not math.isfinite(receipt[key]) or receipt[key] < 0:
            raise ValueError('Finite actual preparation timing required')
    accepted = Path(receipt['accepted_cold']['path']).parent
    original, result, proof = authenticate(SimpleNamespace(accepted_cold=accepted,
        query_bounds=bounds, recipe=recipe, frontend_entry=frontend))
    if any(receipt.get(k) != value for k,value in proof.items()):
        raise ValueError('Recomputed complete accepted preparation lineage differs')
    if receipt['historical_preparation_seconds'] != result['elapsed_seconds']:
        raise ValueError('Original actual preparation timing differs')
    rows=receipt.get('staged_artifacts',[])
    if len(rows) != len(proof['files']):raise ValueError('Every original preparation artifact required')
    for row, source_row in zip(rows,proof['files'],strict=True):
        source=Path(source_row['path']);target=destination/source.name
        if (row.get('source') != str(source) or row.get('target') != str(target)
                or row.get('sha256') != source_row['sha256']
                or row.get('source_inode') != source.stat().st_ino
                or row.get('target_inode') != target.stat().st_ino
                or target.is_symlink() or not target.is_file()):
            raise ValueError('Actual owned staging paths and inode proof changed')
        expected_mode=('hardlink' if source.stat().st_dev == target.stat().st_dev
                       and source.stat().st_ino == target.stat().st_ino else 'copy')
        if (row.get('mode') != expected_mode or (expected_mode == 'hardlink' and
                (shared is not True or source.name not in ('hexagon-linalg.mlirbc','adapted-source.mlir.gz')))):
            raise ValueError('Only immutable original graph artifacts may be shared')
        # These two owned receipts replace the copied historical receipts. Their
        # full current content is authenticated separately, while the original
        # versions remain authenticated by the accepted lineage above.
        if source.name not in ('result.json','reuse-result.json') and sha(target) != source_row['sha256']:
            raise ValueError('Owned staged preparation content differs')
    if receipt.get('prepared') != entry(destination/'result.json'):
        raise ValueError('Owned preparation report changed')
    actual=read(destination/'result.json');expected=copy.deepcopy(result)
    expected['inputs']['entry']=entry(frontend)
    expected['output']['path']=str(destination/'hexagon-linalg.mlirbc')
    expected['historical_preparation_elapsed_seconds']=expected.pop('elapsed_seconds')
    expected['prepared_stage_reuse']=dict(path=str(destination/'reuse-result.json'))
    duration=actual.get('elapsed_seconds')
    if (type(duration) not in (int,float) or not math.isfinite(duration)
            or not 0 <= duration <= receipt['reuse_seconds']
            or {k:v for k,v in actual.items() if k!='elapsed_seconds'} != expected):
        raise ValueError('Complete current preparation report rebinding changed')


def cold_cache(cold,base):
    from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
    from reuse_accepted_compiler_objects import ReadOnlyCompilerCache
    from reuse_accepted_prepared_linalg import frontend_entry
    native=read(cold/'native-result.json');recipe=Path(base['recipe']['path']);rows=[]
    for capture in native['captures']:
        bounds=capture['contract']['variables']['query']
        frontend_path=cold/'frontend-cache'/(capture['frontend_key']+'.json')
        frontend=frontend_entry(frontend_path)
        cache=CapturedQwenMachineCache(cold/'machine-cache',recipe,sha(recipe),cold/'frontend-cache')
        cache.cache=ReadOnlyCompilerCache(cache.cache,capture['machine_key'])
        replay=cache.get_or_compile(frontend,runtime_contract=capture['contract']);folder=Path(replay['directory'])
        object_reuse=read(folder/'objects/reuse-result.json')
        linked=read(folder/'linked/result.json')
        if (not replay['cache_hit'] or replay['machine_code_compilations'] or replay['model_object_codegen_invocations']
                or object_reuse['status']!='authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance'
                or object_reuse['new_machine_code_compilations'] or object_reuse['DSP_calls']
                or object_reuse['query_bounds']!=bounds
                or object_reuse['original_complete_policy_key_and_all_retained_artifacts_rehashed'] is not True
                or sha(linked['library']['path'])!=capture['DSO_sha256']):
            raise ValueError('Actual cold cache and exact primary object reuse required')
        prep=folder/'prepared/reuse-result.json';prepared=None
        if prep.exists():
            receipt=read(prep)
            validate_prepared_reuse(receipt, folder/'prepared', frontend_path, recipe, bounds)
            prepared=dict(receipt=entry(prep),actual_reuse_seconds=receipt['reuse_seconds'],
                historical_preparation_seconds=receipt['historical_preparation_seconds'])
        else:
            miss=read(folder/'prepared-reuse-miss/reuse-result.json')
            if miss['status']!='prepared_stage_cache_miss':raise ValueError('Explicit legitimate original preparation fallback required')
        rows.append(dict(query_bounds=bounds,machine_key=capture['machine_key'],DSO=linked['library'],
            primary_object_reuse=entry(folder/'objects/reuse-result.json'),prepared_stage=prepared))
    if len(rows)!=2 or {tuple(r['query_bounds']) for r in rows}!={(1,1),(2,32)}:
        raise ValueError('Exactly both original cold query graphs required')
    return dict(rows=rows,new_primary_model_codegen_invocations=0,
        new_SDK_caller_link_pipeline_invocations=2,actual_fresh_frontend_invocations=2,
        both_prepared_stages_actually_reused=all(r['prepared_stage'] is not None for r in rows),DSP_calls=0)

