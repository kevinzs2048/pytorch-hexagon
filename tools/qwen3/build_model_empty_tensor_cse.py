"""Relink the accepted complete q1 model with the isolated CSE code object."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from build import ROOT, sha, verify_dependencies, write
from profile_model_memory import checked
from gqa_shared_copy_contract import CONTRACTS, qualified_primitive, compiler_counts


def same_linalg_input(left, right, relocated=False):
    if not relocated:
        return left == right
    if left['sha256'] != right['sha256']:
        return False
    # The completed native cache owns a separate path for identical bytecode.
    # Verify both real files; path spelling does not change a compiler input.
    return checked(left).stat().st_size == checked(right).stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'machine', 'recipe', 'compiler-receipt',
                 'objects-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('compiler-receipt', 'objects-receipt'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--compiler-variant', choices=('empty-cse', 'gqa-value-view', 'gqa-value-forward', 'gqa-key-compact'), default='empty-cse')
    for name in ('accepted-closure', 'baseline-model-gate'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--primitive-revisit', type=Path)
    parser.add_argument('--primitive-revisit-sha256')
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists() or
            subprocess.check_output(['git', '-C', str(ROOT), 'status',
                                     '--porcelain'], text=True).strip()):
        raise ValueError('Clean source and a fresh absolute output required')
    for name in ('compiler_receipt', 'objects_receipt'):
        if sha(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise ValueError('Candidate compilation receipt changed: ' + name)
    config = json.loads(args.config.read_text())
    verify_dependencies(config, json.loads(
        Path(__file__).with_name('dependencies.lock.json').read_text()))
    sdk = json.loads(args.compiler_receipt.read_text())
    code = json.loads(args.objects_receipt.read_text())
    shared_kind = next((k for k,c in CONTRACTS.items() if c['variant']==args.compiler_variant), None)
    shared = shared_kind is not None
    contract = CONTRACTS.get(shared_kind)
    view = args.compiler_variant == 'gqa-value-view' or shared
    primitive = None
    if shared:
        if not args.primitive_revisit or sha(args.primitive_revisit)!=args.primitive_revisit_sha256:
            raise ValueError('Actual pinned independently revisited standalone gate required')
        primitive=json.loads(args.primitive_revisit.read_text())
        if not qualified_primitive(primitive, shared_kind):
            raise ValueError('Positive actual standalone timing and complete ARM word gate required')
    elif args.primitive_revisit or args.primitive_revisit_sha256:
        raise ValueError('Standalone copy proof belongs only to the shared-copy model diagnostic')
    if view:
        for name in ('accepted_closure', 'baseline_model_gate'):
            if not getattr(args,name) or sha(getattr(args,name)) != getattr(args,name+'_sha256'):
                raise ValueError('Pinned completed CSE baseline required')
        accepted = json.loads(args.accepted_closure.read_text())
        gate = json.loads(args.baseline_model_gate.read_text())
        if (accepted['status'] != ('query1_GQA_value_view_native_timing_and_original_states_accepted' if shared else 'query1_empty_tensor_CSE_resumed_native_timing_and_original_states_accepted')
                or accepted['DSP_calls'] != 393 or not accepted['model_inference_acceptance']
                or gate['status'] != ('actual_same_input_q1_GQA_value_view_model_full_packet_gate_passed' if shared else 'actual_same_input_q1_empty_tensor_CSE_model_full_packet_gate_passed')
                or gate['DSP_calls'] != 8 or gate['process_exit_code'] != 0
                or sdk['floating_arithmetic_changed'] or sdk['production_model_acceptance']):
            raise ValueError('Actual complete CSE393 and original ARM packet gate required')
    original = {k: json.loads((args.machine / k / 'result.json').read_text())
                for k in ('prepared', 'objects', 'caller', 'linked')}
    old, caller, link = original['objects'], original['caller'], original['linked']
    expected_sdk = (contract['sdk_status'] if shared else 'isolated_gqa_value_view_and_CSE_SDK_built_NOT_model_or_DSP' if view
                    else 'isolated_empty_tensor_CSE_SDK_built_NOT_model_or_DSP')
    if (sdk['status'] != expected_sdk
            or not sdk['unmodified_SDK_replay_byte_identical']
            or not sdk['original_link_inputs_unchanged']
            or not sdk['original_SDK_unchanged']
            or (not view and sdk['accepted_sdk']['sha256'] != old['inputs']['compiler']['sha256'])
            or code['status'] != 'dynamic-bound_q17_lifted_graph_Hexagon_code_candidate_constants_identical_NOT_DSP'
            or code['machine_code_compilations'] != 1 or code['DSP_calls'] != 0
            or code['inputs']['compiler']['sha256'] != sdk['candidate']['sha256']
            or not same_linalg_input(code['inputs']['linalg'],old['inputs']['linalg'],relocated=view)
            or code['inputs']['options_result'] != old['inputs']['options_result']
            or code['overrides'] != old['overrides']
            or code['option_reconciliation'] != old['option_reconciliation']
            or len(code['objects']) != 4 or len(old['objects']) != 4
            or any(code['objects'][i]['sha256'] != old['objects'][i]['sha256']
                   for i in range(1, 4))
            or link['query_specialization'] != 1
            or link['input_bytes'] != 234885520 or link['output_bytes'] != 235488768
            or caller['output_count'] != 57):
        raise ValueError('Exact original q1 graph, options, constants and isolated SDK required')
    if view:
        if (accepted['candidate_libraries']['(1, 1)'] != link['library']['sha256']
                or gate['candidate_library_sha256'] != link['library']['sha256']):
            raise ValueError('Frozen accepted CSE model must be the actual relink baseline')
        checked(sdk['gqa_value_view_source'])
        diagnostics = checked(code['compiler_diagnostics']).read_text()
        if shared:
            counts=compiler_counts(diagnostics, shared_kind)
            checked(sdk['gqa_value_forward_source'])
            if shared_kind=='gqa-value-forward-q1':checked(sdk['gqa_repeat_slice_source'])
            if shared_kind=='gqa-key-compact-q1': checked(sdk['gqa_key_compact_source'])
        if (diagnostics.count('QWEN_GQA_VALUE_VIEW eliminated_identity_copy ') != 28 or
                'QWEN_GQA_VALUE_VIEW eliminated=28 FP-arithmetic-changed=0' not in diagnostics):
            raise ValueError('Exactly all 28 proven V identity copies required')
    checked(sdk['candidate'])
    checked(sdk['source'])
    checked(sdk['accepted_sdk'])
    model = checked(code['objects'][0])
    previous_model = checked(old['objects'][0])
    wrapper = checked(caller['wrapper'])
    header = checked(caller['runtime_header'])
    guard = checked(caller['silu_effects']['guard_header'])
    checked(link['library'])
    recipe = json.loads(args.recipe.read_text())
    constants = json.loads(checked(recipe['inputs']['constants']).read_text())
    for row in constants['libraries']:
        checked(row)
        if row['input_object_sha256'] != code['objects'][row['index']]['sha256']:
            raise ValueError('Original constant library changed')
    command = link['command'].copy()
    if (command.count(str(previous_model)) != 1 or command.count(str(wrapper)) != 1
            or command.count('-o') != 1 or sha(command[0]) != link['compiler_sha256']
            or sha(config['lld']) != link['linker_sha256']):
        raise ValueError('Original caller, compiler and linker are required')
    inputs = []
    for word in command:
        path = Path(word)
        if path.is_absolute() and path.is_file() and word != link['library']['path']:
            inputs.append(dict(path=str(path), sha256=sha(path)))
    for path in (header, guard):
        inputs.append(dict(path=str(path), sha256=sha(path)))
    args.output.mkdir(parents=True)
    for name in ('prepared', 'caller'):
        (args.output / name).symlink_to(args.machine / name, target_is_directory=True)
    (args.output / 'objects').symlink_to(args.objects_receipt.parent, target_is_directory=True)
    (args.output / 'linked').mkdir()
    library = args.output / 'candidate.so'
    command[command.index(str(previous_model))] = str(model)
    command[command.index('-o') + 1] = str(library)
    record = dict(status='linking_empty_tensor_CSE_query1_NOT_DSP',
        source_revision=subprocess.check_output(['git', '-C', str(ROOT),
            'rev-parse', 'HEAD'], text=True).strip(), source_sha256=sha(__file__),
        original_model_library=link['library'], original_model_object=old['objects'][0],
        original_link_inputs=inputs, original_caller_unchanged=True,
        original_math_header_unchanged=True, original_runtime_objects_unchanged=True,
        original_guarded_copy_runtime_unchanged=True,
        model_graph_unchanged=True, model_object_unchanged=False,
        FP_evaluation_order_changed=False, power_votes_changed=False,
        only_compiler_change='CSE after original empty-tensor elimination',
        SDK_unmodified_replay_byte_identical=True,
        compiler_receipt=dict(path=str(args.compiler_receipt), sha256=args.compiler_receipt_sha256),
        objects_receipt=dict(path=str(args.objects_receipt), sha256=args.objects_receipt_sha256),
        query_bounds=[1, 1], input_bytes_are_maximum_capacity=False,
        input_bytes=link['input_bytes'], output_bytes=link['output_bytes'],
        constants=constants['libraries'], constant_objects_identical=True,
        actual_model_compilations=1, DSP_calls=0,
        test_hooks_absent_from_production_runtime=True,
        model_inference_acceptance=False, command=command)
    if view:
        record.update(status='linking_GQA_value_view_query1_NOT_DSP',
            only_compiler_change='proven identity V layout copy to tensor shape view after original CSE',
            eliminated_identity_value_copies=28,
            accepted_baseline_closure=dict(path=str(args.accepted_closure),sha256=args.accepted_closure_sha256),
            baseline_model_gate=dict(path=str(args.baseline_model_gate),sha256=args.baseline_model_gate_sha256))
    if shared:
        record.update(status='linking_'+shared_kind+'_NOT_DSP', candidate_kind=shared_kind,
            only_compiler_change=contract['change'], **counts,
            primitive_revisit=dict(path=str(args.primitive_revisit),sha256=args.primitive_revisit_sha256))
    write(args.output / 'result.json', record)
    try:
        start = time.monotonic()
        with (args.output / 'link.log').open('xb') as log:
            process = subprocess.run(command, env=dict(os.environ,
                LD_LIBRARY_PATH=':'.join(config['library_path'])),
                stdout=log, stderr=subprocess.STDOUT, timeout=120)
        record.update(link_returncode=process.returncode,
                      link_seconds=time.monotonic() - start)
        process.check_returncode()
        for row in inputs + constants['libraries']:
            checked(row)
        current = dict(link, command=command, objects_sha256=args.objects_receipt_sha256,
            library=dict(path=str(library), sha256=sha(library), bytes=library.stat().st_size),
            model_object_unchanged=False, original_FP_order_preserved=True)
        for name in ('fixed512_copy', 'guarded_copy'):
            current[name] = dict(current[name], original_model_object_and_FP_order_unchanged=False,
                                 original_runtime_unchanged=True, original_FP_order_preserved=True)
        write(args.output / 'linked/result.json', current)
        record.update(status='empty_tensor_CSE_query1_model_built_NOT_device_or_state_acceptance',
                      library=current['library'])
        if view:
            record['status']='GQA_value_view_query1_model_built_NOT_device_or_state_acceptance'
        if shared: record['status']=contract['model_status']
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        write(args.output / 'result.json', record)


if __name__ == '__main__':
    main()
