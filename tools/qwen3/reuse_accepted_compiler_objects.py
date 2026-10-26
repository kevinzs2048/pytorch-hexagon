"""Authenticate reusable compiler objects independently of a later link recipe.

The accepted numerical source remains frozen. A read-only replay reconstructs
its original cache key and rehashes every retained artifact. Reuse requires
identical lowered Linalg bytes, compiler, options, reference constants and
Triton source; a changed graph is a rejection, never an approximate match.
No compiler, linker, model forward or DSP session is launched by this tool.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import copy
import shutil
import sys
import time

from run_regression import accepted_base, require_configured_python, sha, write


class CompilerStageInputMismatch(ValueError):
    """A legitimate new compiler input needs normal exact compilation."""


def same_linalg_input(original,current,allow_relocation=False):
    if original==current:
        return True
    if (not allow_relocation or set(original)!={'path','sha256'} or set(current)!={'path','sha256'} or
            original['sha256']!=current['sha256']):
        return False
    paths=[Path(row['path']) for row in (original,current)]
    if any(not path.is_absolute() or path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('Relocated prepared graph must be an absolute retained regular file')
    if any(sha(path)!=original['sha256'] for path in paths) or paths[0].stat().st_size!=paths[1].stat().st_size:
        raise ValueError('Relocated prepared graph bytes changed')
    return True


def cse_producer(a, original, variant='empty-cse'):
    """Authenticate one already compiled q1 object, without changing its history."""
    if variant not in ('empty-cse','gqa-value-view','gqa-value-forward','gqa-key-compact','gqa-dead-repeat','ordered-pv-register'):
        raise ValueError('Explicit isolated compiler variant required')
    for name in ('cse_sdk_receipt', 'cse_objects_receipt'):
        if sha(getattr(a, name)) != getattr(a, name + '_sha256'):
            raise ValueError('Pinned query1 CSE producer changed: ' + name)
    sdk = json.loads(a.cse_sdk_receipt.read_text())
    code = json.loads(a.cse_objects_receipt.read_text())
    pv=variant=='ordered-pv-register'
    dynamic_pv=getattr(a,'dynamic_pv_register',False)
    if type(dynamic_pv) is not bool or dynamic_pv and (variant!='empty-cse' or
            getattr(a,'dynamic_cse',False) or getattr(a,'dynamic_qk_invariant',False)):
        raise ValueError('Dynamic ordered PV must have a distinct original multi-token producer')
    dynamic_qk=getattr(a,'dynamic_qk_invariant',False)
    if type(dynamic_qk) is not bool or dynamic_qk and variant!='empty-cse':
        raise ValueError('Distinct ordered QK producer selection required')
    dynamic_cse=getattr(a,'dynamic_cse',False) or dynamic_qk or dynamic_pv
    if type(dynamic_cse) is not bool or dynamic_cse and variant!='empty-cse':
        raise ValueError('Dynamic CSE requires its own explicit unchanged multi-token producer')
    dead=variant=='gqa-dead-repeat'
    tiled=getattr(a,'gqa_key_tile_n32',False)
    scalar_key=getattr(a,'gqa_key_scalar_forward',False)
    if type(scalar_key) is not bool or scalar_key and (not dead or tiled):
        raise ValueError('Exact scalar K source forwarding is a distinct final contract after original proof stages')
    if type(tiled) is not bool or tiled and not dead:
        raise ValueError('N32 scheduling requires the paired original dead-repeat producer')
    key=variant in ('gqa-key-compact','gqa-dead-repeat')
    forward=variant in ('gqa-value-forward','gqa-key-compact','gqa-dead-repeat')
    view=variant=='gqa-value-view' or forward
    expected=('isolated_current_dynamic_QK_PV_register_model_candidate_SDK_built_NOT_DSP_or_model_acceptance' if dynamic_pv else
              'isolated_ordered_PV_register_q1_model_compiler_built_NOT_DSP' if pv else
              'isolated_dynamic_QK_invariant_vector_and_CSE_SDK_built_NOT_model_or_DSP' if dynamic_qk else
              'isolated_scalar_K_forward_and_original_dead_repeat_SDK_built_NOT_model_or_DSP' if scalar_key else
              'isolated_tiled_K_copy_and_original_dead_repeat_SDK_built_NOT_model_or_DSP' if tiled else
              'isolated_dead_V_repeat_and_original_K_SDK_built_DIAGNOSTIC_NOT_model_or_DSP' if dead else
              'isolated_gqa_key_compact_V_forward_view_CSE_SDK_built_DIAGNOSTIC_NOT_model_or_DSP' if key else
              'isolated_gqa_value_forward_and_view_CSE_SDK_built_DIAGNOSTIC_NOT_model_or_DSP' if forward else
              'isolated_gqa_value_view_and_CSE_SDK_built_NOT_model_or_DSP' if view
              else 'isolated_empty_tensor_CSE_SDK_built_NOT_model_or_DSP')
    if (a.query_bounds != ([2,32] if dynamic_cse else [1,1])
            or sdk['status'] != expected
            or not all(sdk[k] for k in ('unmodified_SDK_replay_byte_identical',
                'original_link_inputs_unchanged', 'original_SDK_unchanged'))
            or sdk['accepted_sdk']['sha256'] != original['inputs']['compiler']['sha256']
            or code['status'] != original['status'] or code['source_sha256'] != original['source_sha256']
            or code['machine_code_compilations'] != 1 or code['DSP_calls'] != 0
            or code['model_execution'] is not False or code['graph_kind'] != 'dynamic-bound'
            or code['allow_code_difference'] is not True or len(code['objects']) != 4
            or code['inputs']['compiler']['sha256'] != sdk['candidate']['sha256']
            or code['inputs']['options_result'] != original['inputs']['options_result']
            or not same_linalg_input(original['inputs']['linalg'],code['inputs']['linalg'],allow_relocation=forward or dynamic_cse or pv)
            or any(code[k] != original[k] for k in ('overrides', 'option_reconciliation',
                'expected_objects_result', 'triton_python'))
            or any((code['objects'][i]['sha256'], code['objects'][i]['bytes']) !=
                   (original['objects'][i]['sha256'], original['objects'][i]['bytes']) for i in (1, 2, 3))):
        raise ValueError('Original q1 graph/options/constants and exactly isolated CSE compiler required')
    for row in (sdk['accepted_sdk'], sdk['candidate'], sdk['source'], code['objects'][0],
                code['compiler_diagnostics']):
        if sha(row['path']) != row['sha256']:
            raise ValueError('CSE producer artifact changed: ' + row['path'])
    if pv:
        from ordered_pv_model_contract import require_compiler
        if sha(Path(__file__).with_name('ordered_pv_model_contract.py'))!=a.ordered_pv_contract_source_sha256:
            raise ValueError('Pinned distinct ordered PV model contract required')
        diagnostics=Path(code['compiler_diagnostics']['path']).read_text()
        require_compiler(sdk,diagnostics)
        from gqa_dead_repeat_contract import validate_dead_repeat
        from gqa_shared_copy_contract import compiler_counts
        validate_dead_repeat(sdk,diagnostics);compiler_counts(diagnostics,'gqa-key-compact-q1')
        tile='QWEN_GQA_KEY_TILE positions=32 head-position-component-order fresh-complete-destination-proved=1 FP-arithmetic-changed=0'
        if diagnostics.splitlines().count(tile)!=28:
            raise ValueError('Original complete N32 key schedule must remain in all28 layers')
        for row in [sdk[name] for name in ('source','gqa_value_view_source','gqa_value_forward_source',
                    'gqa_repeat_slice_source','gqa_key_compact_source')]+sdk['only_added_translation_units']:
            if sha(row['path'])!=row['sha256']:
                raise ValueError('Actually compiled original GQA or PV source changed')
    if dynamic_qk:
        from dynamic_qk_invariant_contract import require_compiler
        if sha(Path(__file__).with_name('dynamic_qk_invariant_contract.py'))!=a.dynamic_qk_invariant_contract_source_sha256:
            raise ValueError('Pinned distinct QK contract required')
        require_compiler(sdk,Path(code['compiler_diagnostics']['path']).read_text())
        for row in sdk['only_added_translation_units']:
            if sha(row['path'])!=row['sha256']:
                raise ValueError('Actually compiled QK source changed')
    if dynamic_pv:
        from dynamic_pv_native_contract import require_compiler
        if sha(Path(__file__).with_name('dynamic_pv_native_contract.py'))!=a.dynamic_pv_register_contract_source_sha256:
            raise ValueError('Pinned original partial-order PV native contract required')
        require_compiler(sdk,Path(code['compiler_diagnostics']['path']).read_text())
    if view:
        row=sdk['gqa_value_view_source']
        if (sha(row['path'])!=row['sha256'] or sdk['floating_arithmetic_changed']
                or sdk['production_model_acceptance']):
            raise ValueError('Pinned pure identity V compiler required')
        diagnostics=Path(code['compiler_diagnostics']['path']).read_text()
        if (diagnostics.count('QWEN_GQA_VALUE_VIEW eliminated_identity_copy ')!=28 or
                'QWEN_GQA_VALUE_VIEW eliminated=28 FP-arithmetic-changed=0' not in diagnostics):
            raise ValueError('Exactly 28 original identity V copies required')
        if forward:
            from gqa_shared_copy_contract import compiler_counts
            compiler_counts(diagnostics,'gqa-key-compact-q1' if key else 'gqa-value-forward-q1')
            if dead:
                from gqa_dead_repeat_contract import validate_dead_repeat
                validate_dead_repeat(sdk,diagnostics)
            if scalar_key:
                from key_forward_model_contract import validate_key_forward
                if sha(Path(__file__).with_name('key_forward_model_contract.py')) != a.gqa_key_scalar_forward_contract_source_sha256:
                    raise ValueError('Exact immutable K source reuse validator required')
                validate_key_forward(sdk,diagnostics)
            if tiled:
                from key_tile_model_contract import validate_key_tile
                if sha(Path(__file__).with_name('key_tile_model_contract.py')) != a.gqa_key_tile_contract_source_sha256:
                    raise ValueError('Exact pinned N32 reuse validator required')
                validate_key_tile(sdk,diagnostics)
            for name in ('gqa_value_forward_source','gqa_repeat_slice_source', *(['gqa_key_compact_source'] if key else [])):
                row=sdk[name]
                if sha(row['path'])!=row['sha256']:
                    raise ValueError('Pinned repeat-slice compiler source changed')
            required_commands={'CSE-wrapper-compile':sdk['source'],
                'gqa-value-view-compile':sdk['gqa_value_view_source'],
                'gqa-value-forward-compile':sdk['gqa_value_forward_source'],
                'gqa-repeat-slice-compile':sdk['gqa_repeat_slice_source']}
            if key:required_commands['gqa-key-compact-compile']=sdk['gqa_key_compact_source']
            compiled=[]
            for label,row in required_commands.items():
                matches=[r for r in sdk['commands'] if '-c' in r['argv'] and
                    r['argv'][r['argv'].index('-c')+1]==row['path']]
                if len(matches)!=1 or type(matches[0]['returncode']) is not int or matches[0]['returncode']!=0:
                    raise ValueError('All selected actual added compiler units must complete')
                argv=matches[0]['argv']
                if argv[argv.index('-c')+1]!=row['path']:
                    raise ValueError('Compiler command and source proof differ')
                compiled.append(argv[argv.index('-o')+1])
            links=[r for r in sdk['commands'] if '-c' not in r['argv'] and '-o' in r['argv'] and
                   r['argv'][r['argv'].index('-o')+1]==sdk['candidate']['path']]
            if len(links)!=1 or type(links[0]['returncode']) is not int or links[0]['returncode']!=0 or any(p not in links[0]['argv'] for p in compiled):
                raise ValueError('All selected compiler units must actually be linked')
    return code


class ReadOnlyCompilerCache:
    def __init__(self, cache, expected_key):
        self.cache = cache
        self.expected_key = expected_key

    def get_or_compile(self, request, compiler):
        # Deliberately do not invoke the callback, including on a cache miss.
        from hexagon_compiled_artifact_cache import cache_key
        key = cache_key(request)
        if key != self.expected_key:
            raise ValueError('Original complete compiler policy key changed')
        hit = self.cache._read(key, request)
        if hit is None:
            raise ValueError('Original compiler artifact cache entry missing')
        return dict(hit, cache_hit=True, compiler_invocations=0)


def reused_objects(a):
    base = accepted_base(a.accepted_cold)
    native = json.loads((a.accepted_cold / 'native-result.json').read_text())
    if (native['status'] != 'ARM_native_dynamic_compiler_handoff_generate_completed'
            or native['DSP_calls'] != 9 or native['frontend_invocations'] != 2
            or native['machine_code_compilations'] != 2 or native['process_exit_code'] != 0
            or native['CPU_model_fallback']):
        raise ValueError('Actual accepted original two-graph cold9 required')
    captures = [c for c in native['captures'] if c['contract']['variables']['query'] == a.query_bounds]
    if len(captures) != 1:
        raise ValueError('Exactly one actual selected query category required')
    capture = captures[0]
    worker = base['worker']
    def arg(flag):
        if worker.count(flag) != 1:
            raise ValueError('Unambiguous original compiler worker required')
        return worker[worker.index(flag) + 1]
    numerical_scripts = Path(base['source_root']) / 'qwen3/runtime/arm'
    sys.path.insert(0, str(numerical_scripts))
    path = numerical_scripts / 'hexagon_qwen_captured_machine_cache.py'
    spec = importlib.util.spec_from_file_location('accepted_compiler_stage_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = module.CapturedQwenMachineCache(Path(arg('--machine-cache')),
        frontend_root=Path(arg('--frontend-cache')), recipe_path=Path(base['recipe']['path']),
        recipe_sha256=base['recipe']['sha256'])
    cache.cache = ReadOnlyCompilerCache(cache.cache, capture['machine_key'])
    frontend_path = Path(arg('--frontend-cache')) / (capture['frontend_key'] + '.json')
    frontend = json.loads(frontend_path.read_text())
    actual = cache.get_or_compile(frontend, runtime_contract=capture['contract'])
    folder = Path(actual['directory'])
    producer_path = folder / 'objects/result.json'
    producer = json.loads(producer_path.read_text())
    if (producer.get('status') != 'dynamic-bound_q17_lifted_graph_Hexagon_code_candidate_constants_identical_NOT_DSP'
            or producer.get('source_sha256') != sha(numerical_scripts / 'probe_hexagon_dynamic_linalg.py')
            or producer.get('machine_code_compilations') != 1 or producer.get('DSP_calls') != 0
            or producer.get('model_execution') is not False
            or producer.get('graph_kind') != 'dynamic-bound' or len(producer.get('objects', [])) != 4
            or producer.get('allow_code_difference') is not True):
        raise ValueError('Original complete code-only compiler producer required')
    original_producer_path = producer_path
    dead=bool(getattr(a,'gqa_dead_repeat_sdk_receipt',None))
    key=bool(getattr(a,'gqa_key_compact_sdk_receipt',None)) or dead
    forward=bool(getattr(a,'gqa_value_forward_sdk_receipt',None)) or key
    view=bool(getattr(a,'gqa_value_view_sdk_receipt',None)) or forward
    pv=bool(getattr(a,'ordered_pv_sdk_receipt',None))
    if pv:
        from types import SimpleNamespace
        scoped=SimpleNamespace(**vars(a))
        for name in ('sdk','objects'):
            for suffix in ('receipt','receipt_sha256'):
                setattr(scoped,'cse_'+name+'_'+suffix,getattr(a,'ordered_pv_'+name+'_'+suffix))
        producer=cse_producer(scoped,producer,variant='ordered-pv-register')
        producer_path=a.ordered_pv_objects_receipt
    elif view:
        from types import SimpleNamespace
        scoped=SimpleNamespace(**vars(a))
        for name in ('sdk','objects'):
            for suffix in ('receipt','receipt_sha256'):
                prefix='gqa_dead_repeat' if dead else 'gqa_key_compact' if key else 'gqa_value_forward' if forward else 'gqa_value_view'
                setattr(scoped,'cse_'+name+'_'+suffix,getattr(a,prefix+'_'+name+'_'+suffix))
        producer=cse_producer(scoped,producer,variant='gqa-dead-repeat' if dead else 'gqa-key-compact' if key else 'gqa-value-forward' if forward else 'gqa-value-view')
        producer_path=getattr(a,prefix+'_objects_receipt')
    elif a.cse_sdk_receipt:
        producer = cse_producer(a, producer)
        producer_path = a.cse_objects_receipt
    pairs = (('linalg', a.linalg), ('compiler', a.compiler), ('options_result', a.options_result))
    for name, path in pairs:
        if sha(path) != producer['inputs'][name]['sha256']:
            raise CompilerStageInputMismatch('Compiler stage input differs: ' + name)
    if (sha(a.expected_objects_result) != producer['expected_objects_result']['sha256']
            or a.triton_python.resolve() != Path(producer['triton_python']['root']).resolve()
            or sha(a.triton_python / 'triton/__init__.py') != producer['triton_python']['package_source_sha256']):
        raise CompilerStageInputMismatch('Reference constants or Triton source changed')
    expected = json.loads(a.expected_objects_result.read_text())
    for index, row in enumerate(producer['objects']):
        if index:
            reference = expected['objects'][index]
            if (row['sha256'], row['bytes']) != (reference['sha256'], reference['bytes']):
                raise ValueError('Original compiler constant object differs')
        elif (not row['retained'] or sha(row['path']) != row['sha256']
                or Path(row['path']).stat().st_size != row['bytes']):
            raise ValueError('Retained compiler object drifted')
    # Bytecode includes the original weights; no prompt or token whitelist.
    result = dict(status='authenticated_exact_compiler_stage_objects_reused_NOT_model_acceptance',
        accepted_cold=dict(path=str(a.accepted_cold / 'acceptance.json'),
            sha256=sha(a.accepted_cold / 'acceptance.json'), source_revision=base['revision']),
        original_cache_key=capture['machine_key'], frontend_key=capture['frontend_key'],
        query_bounds=a.query_bounds, producer_result=dict(path=str(producer_path), sha256=sha(producer_path)),
        current_stage_inputs={name: dict(path=str(path), sha256=sha(path)) for name, path in pairs},
        historical_compiler_elapsed_seconds=producer['elapsed_seconds'],
        historical_machine_code_compilations=1, new_machine_code_compilations=0,
        new_frontend_compilations=0, DSP_calls=0, model_inference_acceptance=False,
        original_complete_policy_key_and_all_retained_artifacts_rehashed=True,
        compiler_objects=producer['objects'])
    if a.cse_sdk_receipt or view or pv:
        result_key='query2_ordered_PV_register' if getattr(a,'dynamic_pv_register',False) else 'query1_ordered_PV_register' if pv else 'query2_QK_invariant_vector' if getattr(a,'dynamic_qk_invariant',False) else 'query2_cse' if getattr(a,'dynamic_cse',False) else 'query1_gqa_dead_repeat' if dead else 'query1_gqa_key_compact' if key else 'query1_gqa_value_forward' if forward else 'query1_gqa_value_view' if view else 'query1_empty_tensor_CSE'
        prefix='ordered_pv' if pv else 'gqa_dead_repeat' if dead else 'gqa_key_compact' if key else 'gqa_value_forward' if forward else 'gqa_value_view' if view else 'cse'
        result[result_key] = dict(
            original_producer=dict(path=str(original_producer_path), sha256=sha(original_producer_path)),
            compiler_receipt=dict(path=str(getattr(a,prefix+'_sdk_receipt')), sha256=getattr(a,prefix+'_sdk_receipt_sha256')),
            objects_receipt=dict(path=str(getattr(a,prefix+'_objects_receipt')), sha256=getattr(a,prefix+'_objects_receipt_sha256')),
            original_graph_options_and_constants_identical=True)
        if getattr(a,'gqa_key_scalar_forward',False):
            result['query1_gqa_key_scalar_forward']=dict(result[result_key],
                contract='original_Qwen_query1_immutable_K_scalar_source_index_equivalence_v1',
                all28_original_K_scalar_sources_proved=True,complete_K_copy_producers_removed=28)
        if getattr(a,'gqa_key_tile_n32',False):
            result['query1_gqa_key_tile']=dict(result[result_key],
                contract='original_Qwen_query1_pure_compact_K_copy_partition_N32_v1',
                all28_complete_K_copy_tiles_proved=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'accepted-cold', 'linalg', 'compiler', 'options-result',
                 'expected-objects-result', 'triton-python', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--query-bounds', nargs=2, type=int, required=True)
    parser.add_argument('--stage-output', action='store_true',
        help='Retain an owned code object and an explicit historical producer receipt for the linker')
    parser.add_argument('--dynamic-cse',action='store_true')
    parser.add_argument('--dynamic-qk-invariant',action='store_true')
    parser.add_argument('--dynamic-qk-invariant-contract-source-sha256')
    parser.add_argument('--dynamic-pv-register',action='store_true')
    parser.add_argument('--dynamic-pv-register-contract-source-sha256')
    parser.add_argument('--gqa-key-scalar-forward',action='store_true')
    parser.add_argument('--gqa-key-scalar-forward-contract-source-sha256')
    parser.add_argument('--gqa-key-tile-n32',action='store_true')
    parser.add_argument('--gqa-key-tile-contract-source-sha256')
    for name in ('cse-sdk-receipt', 'cse-objects-receipt'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--' + name + '-sha256')
    for name in ('gqa-value-view-sdk-receipt','gqa-value-view-objects-receipt'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('gqa-value-forward-sdk-receipt','gqa-value-forward-objects-receipt'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('gqa-key-compact-sdk-receipt','gqa-key-compact-objects-receipt'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('gqa-dead-repeat-sdk-receipt','gqa-dead-repeat-objects-receipt'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('ordered-pv-sdk-receipt','ordered-pv-objects-receipt'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--ordered-pv-contract-source-sha256')
    a = parser.parse_args()
    pv_values=[a.ordered_pv_sdk_receipt,a.ordered_pv_sdk_receipt_sha256,
               a.ordered_pv_objects_receipt,a.ordered_pv_objects_receipt_sha256,
               a.ordered_pv_contract_source_sha256]
    if (len([v for v in pv_values if v]) not in (0,5)
            or a.ordered_pv_sdk_receipt and (a.query_bounds!=[1,1]
                or any((a.cse_sdk_receipt,a.gqa_value_view_sdk_receipt,
                        a.gqa_value_forward_sdk_receipt,a.gqa_key_compact_sdk_receipt,
                        a.gqa_dead_repeat_sdk_receipt,a.dynamic_cse,a.dynamic_qk_invariant,
                        a.gqa_key_tile_n32,a.gqa_key_scalar_forward)))):
        parser.error('Distinct original ordered PV producer requires its own paired q1 contract')
    if (len([v for v in (a.cse_sdk_receipt, a.cse_sdk_receipt_sha256,
                         a.cse_objects_receipt, a.cse_objects_receipt_sha256) if v]) not in (0, 4)
            or (a.cse_sdk_receipt and a.query_bounds != ([2,32] if a.dynamic_cse or a.dynamic_qk_invariant or a.dynamic_pv_register else [1,1]))
            or (a.dynamic_cse or a.dynamic_qk_invariant or a.dynamic_pv_register) and not a.cse_sdk_receipt):
        parser.error('Paired isolated CSE SDK and complete object receipts apply only to query1')
    if (bool(a.dynamic_pv_register_contract_source_sha256)!=a.dynamic_pv_register or a.dynamic_pv_register and
            (a.dynamic_cse or a.dynamic_qk_invariant or a.ordered_pv_sdk_receipt or a.query_bounds!=[2,32])):
        parser.error('Dynamic ordered PV uses its own multi-token compiler and exact partial-order contract')
    if (bool(a.dynamic_qk_invariant_contract_source_sha256)!=a.dynamic_qk_invariant or
            a.dynamic_qk_invariant and (a.dynamic_cse or a.query_bounds!=[2,32])):
        parser.error('Distinct dynamic QK selection and pinned contract required; cannot claim CSE-only')
    view_values=[a.gqa_value_view_sdk_receipt,a.gqa_value_view_sdk_receipt_sha256,
                 a.gqa_value_view_objects_receipt,a.gqa_value_view_objects_receipt_sha256]
    if (len([v for v in view_values if v]) not in (0,4) or
            (a.gqa_value_view_sdk_receipt and (a.query_bounds!=[1,1] or a.cse_sdk_receipt))):
        parser.error('One paired scoped GQA value-view producer applies only to query1')
    forward_values=[a.gqa_value_forward_sdk_receipt,a.gqa_value_forward_sdk_receipt_sha256,
                    a.gqa_value_forward_objects_receipt,a.gqa_value_forward_objects_receipt_sha256]
    if (len([v for v in forward_values if v]) not in (0,4) or
            a.gqa_value_forward_sdk_receipt and (a.query_bounds!=[1,1] or a.cse_sdk_receipt or a.gqa_value_view_sdk_receipt)):
        parser.error('One paired proved repeat-slice producer applies only to query1')
    key_values=[a.gqa_key_compact_sdk_receipt,a.gqa_key_compact_sdk_receipt_sha256,
                a.gqa_key_compact_objects_receipt,a.gqa_key_compact_objects_receipt_sha256]
    if (len([v for v in key_values if v]) not in (0,4) or
            a.gqa_key_compact_sdk_receipt and (a.query_bounds!=[1,1] or a.cse_sdk_receipt or
                a.gqa_value_view_sdk_receipt or a.gqa_value_forward_sdk_receipt)):
        parser.error('One paired proved original SCF K producer applies only to query1')
    dead_values=[a.gqa_dead_repeat_sdk_receipt,a.gqa_dead_repeat_sdk_receipt_sha256,
                 a.gqa_dead_repeat_objects_receipt,a.gqa_dead_repeat_objects_receipt_sha256]
    if (len([v for v in dead_values if v]) not in (0,4) or
            a.gqa_dead_repeat_sdk_receipt and (a.query_bounds!=[1,1] or a.cse_sdk_receipt or
                a.gqa_value_view_sdk_receipt or a.gqa_value_forward_sdk_receipt or a.gqa_key_compact_sdk_receipt)):
        parser.error('One paired proved late dead-repeat producer applies only to query1')
    if bool(a.gqa_key_tile_contract_source_sha256)!=a.gqa_key_tile_n32 or a.gqa_key_tile_n32 and (
            not a.gqa_dead_repeat_sdk_receipt or a.query_bounds != [1,1]):
        parser.error('N32 scheduling requires the explicit complete query1 dead-repeat producer')
    if bool(a.gqa_key_scalar_forward_contract_source_sha256)!=a.gqa_key_scalar_forward or a.gqa_key_scalar_forward and (
            not a.gqa_dead_repeat_sdk_receipt or a.gqa_key_tile_n32 or a.query_bounds != [1,1]):
        parser.error('Scalar K forwarding requires its distinct query1 final proof and paired producer')
    if (a.query_bounds not in ([1, 1], [2, 32]) or not a.output.is_absolute()
            or a.output.exists() or a.output.is_symlink()):
        parser.error('Original query category and fresh absolute result directory required')
    require_configured_python(json.loads(a.config.read_text()))
    a.output.mkdir(parents=True)
    start = time.monotonic()
    result = dict(status='checking_exact_compiler_stage_reuse', DSP_calls=0,
        new_machine_code_compilations=0, model_inference_acceptance=False)
    try:
        result = reused_objects(a)
        # Keep the compiler receipt byte-identical. Its compilation count is
        # historical; this operation's explicit new count remains zero.
        producer = Path(result['producer_result']['path'])
        destination = a.output / 'producer-result.json'
        destination.write_bytes(producer.read_bytes())
        result['objects_result'] = dict(path=str(destination), sha256=sha(destination))
        if a.stage_output:
            original = json.loads(destination.read_text())
            owned = a.output / 'dynamic-hexagon-0.o'
            shutil.copyfile(original['objects'][0]['path'], owned)
            if sha(owned) != original['objects'][0]['sha256']:
                raise ValueError('Owned reused compiler object differs')
            diagnostic = a.output / 'compiler-diagnostics.txt'
            shutil.copyfile(original['compiler_diagnostics']['path'], diagnostic)
            if sha(diagnostic) != original['compiler_diagnostics']['sha256']:
                raise ValueError('Original compiler diagnostics changed')
            linked = copy.deepcopy(original)
            linked['inputs'] = result['current_stage_inputs']
            linked['expected_objects_result'] = dict(path=str(a.expected_objects_result), sha256=sha(a.expected_objects_result))
            linked['objects'][0]['path'] = str(owned)
            linked['compiler_diagnostics']['path'] = str(diagnostic)
            linked['historical_producer_result'] = result['objects_result']
            linked['new_model_object_codegen_invocations'] = 0
            linked['compiler_stage_reuse'] = dict(path=str(a.output / 'reuse-result.json'))
            write(a.output / 'result.json', linked)
            result['owned_objects_result'] = dict(path=str(a.output / 'result.json'), sha256=sha(a.output / 'result.json'))
    except CompilerStageInputMismatch as error:
        result.update(status='compiler_stage_cache_miss', error=str(error))
        raise SystemExit(10)
    except BaseException as error:
        result.update(status='compiler_stage_reuse_rejected', error=str(error)); raise
    finally:
        result['source_sha256'] = sha(__file__)
        result['reuse_elapsed_seconds'] = time.monotonic() - start
        write(a.output / ('reuse-result.json' if a.stage_output else 'result.json'), result)
    print(json.dumps({k: result[k] for k in ('status', 'reuse_elapsed_seconds',
        'historical_compiler_elapsed_seconds', 'new_machine_code_compilations', 'DSP_calls')}), flush=True)


if __name__ == '__main__':
    main()
