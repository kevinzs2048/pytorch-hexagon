"""Compiler-host transaction for a trusted ARM-captured Qwen graph.

Actual restored AOT -> existing torch-mlir cache/lowering -> existing Hexagon
machine cache/compiler. No graph recapture, generation loop or DSP execution.
This command is not yet an automatic live ARM compiler-request service.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from capture_qwen_symbolic_graph_carrier import sha, CONTEXT_SHA
from restore_qwen_symbolic_graph_carrier import restore_carrier
from hexagon_compiled_artifact_cache import canonical
from hexagon_qwen_shape_policy import query_bounds


def derive_runtime_contract(graph, binding, frontend, catalog):
    import torch
    from qcom_hexagon_backend.backend.hexagon_captured_buffer_contract import (
        scalar_policy_from_tensor_contract, derive_captured_contract)
    candidates = [entry['contract'] for entry in catalog['entries']
                  if entry['contract']['variables'] == {'query': [2, 32]}
                  and len(entry['contract']['inputs']) == 63]
    if len(candidates) != 1:
        raise ValueError('One admitted base Qwen dynamic tensor policy required')
    base = candidates[0]
    evidence = json.loads((Path(frontend['directory'])/'lowering.json').read_text())
    if len(evidence['signatures']) != 1:
        raise ValueError('One actual Linalg entry required')
    placeholders = [node for node in graph.graph.nodes if node.op == 'placeholder']
    tensors = [placeholders[i].meta['val'] for i in binding.retained
               if isinstance(placeholders[i].meta['val'], torch.Tensor)]
    if len(tensors) != 60 or tensors[0].ndim != 2 or tensors[0].dtype != torch.int64:
        raise ValueError('Selected Qwen runtime tensor profile required')
    extent = tensors[0].shape[1]
    bounds = {'query': query_bounds(extent)}
    literals = {1, 1024}
    if bounds['query'][0] == bounds['query'][1]:
        literals.add(bounds['query'][0])
    policy = scalar_policy_from_tensor_contract(graph, binding.retained,
        [row for row in base['inputs'] if row['kind'] == 'tensor'],
        variables=bounds, allowed_literals=literals)
    contract = derive_captured_contract(graph, binding.retained, evidence['signatures'][0],
        variables=bounds, scalar_bindings=policy)
    normalized = json.loads(json.dumps(contract))
    return normalized, normalized != base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--producer-sha256', required=True)
    parser.add_argument('--trusted-local-producer', action='store_true')
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--frontend-cache', type=Path, required=True)
    parser.add_argument('--machine-cache', type=Path, required=True)
    parser.add_argument('--machine-recipe', type=Path, required=True)
    parser.add_argument('--machine-recipe-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.trusted_local_producer:
        parser.error('Only explicitly trusted local capture artifacts may be loaded')
    if (platform.machine() != 'x86_64' or sha(args.context) != CONTEXT_SHA
            or sha(args.catalog) != args.catalog_sha256):
        raise ValueError('Pinned compiler-host model and ABI catalog required')
    args.output.mkdir(exist_ok=False)
    record = dict(status='admitting_carried_graph_compilation', source_sha256=sha(__file__),
        producer=dict(path=str(args.producer), sha256=args.producer_sha256),
        catalog=dict(path=str(args.catalog), sha256=args.catalog_sha256),
        context_sha256=CONTEXT_SHA, CPU_model_forward=False, DSP_calls=0,
        numerical_acceptance=False, live_ARM_compile_request_service=False)
    def save():
        (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        from qwen3_generate_compile_common import load_model
        from hexagon_carrier_parameter_binding import CarrierParameterBinding
        from hexagon_captured_linalg_cache import CapturedLinalgCache
        from hexagon_qwen_captured_machine_cache import CapturedQwenMachineCache
        from hexagon_module_scalar_binding import binding_content_signature
        from hexagon_buffer_transport import buffer_capacities_for_contract
        graph, _, producer = restore_carrier(args.producer, args.producer_sha256, trusted_local=True)
        catalog = json.loads(args.catalog.read_text())
        selected_content=producer['binding_content_sha256']
        if producer.get('guarded_module_constants') is not None:
            from tools.qwen3.guarded_module_constants import selected_model_content_signature
            selected_content=selected_model_content_signature(producer['immutable_inputs'],
                producer['guarded_module_constants']['records'])
        if {entry['binding_content_sha256'] for entry in catalog['entries']} != {selected_content}:
            raise ValueError('Carried graph changes selected immutable model content')
        model, _, _ = load_model(args.context)
        binding = CarrierParameterBinding(model, producer)
        frontend = CapturedLinalgCache(args.frontend_cache).get_or_lower(
            graph, binding, name='Qwen3DynamicBoundProbe')
        record.update(status='selecting_actual_Hexagon_compilation',
                      frontend_cache=frontend, producer_machine=producer['host_machine'])
        save()
        contract, generated_caller = derive_runtime_contract(graph, binding, frontend, catalog)
        caps = buffer_capacities_for_contract(contract)
        machine = CapturedQwenMachineCache(args.machine_cache, args.machine_recipe,
            args.machine_recipe_sha256, args.frontend_cache).get_or_compile(
                frontend, runtime_contract=contract)
        product = json.loads(Path(machine['linked_result']['path']).read_text())
        library = product['library']
        if (sha(library['path']) != library['sha256']
                or machine['request']['graph_sha256'] != frontend['request']['graph']['sha256']
                or machine['request']['parameters_sha256'] != frontend['request']['parameters_sha256']):
            raise ValueError('Compiled product does not bind actual carried graph and weights')
        relative = str(Path(library['path']).relative_to(args.machine_cache))
        entry = dict(graph=producer['graph_identity'],
            parameters_sha256=hashlib.sha256(canonical(binding.records)).hexdigest(),
            binding_content_sha256=binding_content_signature(binding.records),
            retained_input_indices=binding.retained, contract=contract,
            frontend_key=frontend['key'], machine_key=machine['key'],
            library=dict(path=library['path'], relative_to_machine_cache=relative, sha256=library['sha256']))
        record.update(status='carried_ARM_Qwen_graph_bound_to_Hexagon_library_NOT_DSP',
            machine_cache=machine, compiled_entry=entry, buffer_capacities=caps,
            generated_caller=True, contract_differs_from_base=generated_caller,
            immutable_tensors_verified=len(binding.records),
            frontend_invocations=frontend['frontend_invocations'],
            machine_code_compilations=machine['machine_code_compilations'])
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
