"""Warm-only ARM admission by actual captured graph, never by prompt length.

The catalog is derived from an existing pinned native DSP run. It is not a
compiler: unknown graphs fail closed and require compiler-host admission.
"""
import hashlib
import json
import copy
from pathlib import Path

from hexagon_runtime.hexagon_compiled_artifact_cache import canonical, digest_file


def catalog_from_run(run):
    if (run['status'] != 'native_Qwen_requests_on_one_DSP_process'
            or run['native_generate'] is not True or run['q1_dynamic_library_bindings']
            or run['process_exit_code'] != 0
            or run['library_lifecycle']['all_loaded_owners_closed'] is not True):
        raise ValueError('Completed strict-capture native DSP control required')
    count = len(run['AOT_compiles'])
    if not 1 <= count <= 4 or any(len(run[key]) != count for key in
            ('frontend_cache', 'machine_cache', 'captured_buffer_contracts', 'module_scalar_bindings')):
        raise ValueError('Every actual capture needs its own complete compiled binding')
    slots = {row['sha256']: row for row in run['DSP_library_slots']}
    entries = []
    for index, (capture, frontend, machine, contract_row, scalars) in enumerate(zip(
            run['AOT_compiles'], run['frontend_cache'], run['machine_cache'],
            run['captured_buffer_contracts'], run['module_scalar_bindings'])):
        request = frontend['request']
        compiled = machine['request']
        contract = contract_row['contract']
        if (compiled['graph_sha256'] != request['graph']['sha256']
                or compiled['parameters_sha256'] != request['parameters_sha256']
                or compiled['options']['retained_input_indices'] != request['retained_input_indices']
                or compiled['options']['function_name'] != request['function_name']
                or capture['retained_inputs'] != len(request['retained_input_indices'])
                or len(contract['inputs']) != capture['retained_inputs']
                or contract['variables'] not in ({'query': [1, 1]}, {'query': [2, 32]})
                or len(contract['outputs']) != 57):
            raise ValueError('Capture, weights, compiled entry or bounded ABI mismatch')
        libraries = {row['DSO_sha256'] for row in run['forwards'] if row['AOT_capture_index'] == index}
        if len(libraries) != 1 or not libraries <= slots.keys():
            raise ValueError('Each capture must have one actually executed library')
        library = slots[next(iter(libraries))]
        if (not library['remote_path'].startswith('/dev/shm/qwen3-')
                or Path(library['remote_path']).name != library['sha256']+'.so'):
            raise ValueError('Content-addressed control deployment required')
        entries.append(dict(graph=request['graph'], parameters_sha256=request['parameters_sha256'],
            retained_input_indices=request['retained_input_indices'], contract=contract,
            binding_content_sha256=scalars['all_frozen_content_signature'],
            library=library, frontend_key=frontend['key'], machine_key=machine['key'],
            linked_result=machine['linked_result']))
    if len({row['graph']['sha256'] for row in entries}) != len(entries):
        raise ValueError('Ambiguous captured graph catalog')
    return dict(schema=1, status='warm_ARM_dynamic_catalog_NOT_executed', entries=entries,
                numerical_acceptance=False, automatic_cold_compilation=False)


def select_actual_capture(catalog, graph_identity, binding):
    from hexagon_module_scalar_binding import binding_content_signature
    matches = [entry for entry in catalog['entries'] if entry['graph'] == graph_identity]
    if len(matches) != 1:
        raise ValueError('New or ambiguous actual graph: compiler-host admission required; no fallback')
    entry = matches[0]
    if (hashlib.sha256(canonical(binding.records)).hexdigest() != entry['parameters_sha256']
            or binding.retained != entry['retained_input_indices']
            or binding_content_signature(binding.records) != entry['binding_content_sha256']):
        raise ValueError('Actual captured model weights or retained inputs differ from compiled entry')
    return entry


def build_catalog(run_path, run_sha, runtime_path, runtime_sha):
    if digest_file(run_path) != run_sha or digest_file(runtime_path) != runtime_sha:
        raise ValueError('Pinned run and runtime records required')
    run = json.loads(Path(run_path).read_text())
    runtime = json.loads(Path(runtime_path).read_text())
    if (runtime['library_switch_protocol'] != 'bounded-library-slots-v1'
            or runtime['dynamic_size_protocol'] != 'bounded-input-output-v1'):
        raise ValueError('Existing multiple-entry dynamic runtime required')
    if run['runtime_result_sha256'] != runtime_sha:
        raise ValueError('Runtime differs from actually executed control')
    result = catalog_from_run(run)
    for entry in result['entries']:
        linked = entry['linked_result']
        if digest_file(linked['path']) != linked['sha256']:
            raise ValueError('Linked product metadata changed')
        product = json.loads(Path(linked['path']).read_text())
        if (product['library']['sha256'] != entry['library']['sha256']
                or digest_file(entry['library']['library']) != entry['library']['sha256']):
            raise ValueError('Actually executed machine library changed')
    artifacts = []
    for row in runtime['artifacts']:
        if digest_file(row['path']) != row['sha256']:
            raise ValueError('Runtime artifact bytes changed')
        artifacts.append(dict(name=Path(row['path']).name, sha256=row['sha256']))
    roots = {str(Path(entry['library']['remote_path']).parent) for entry in result['entries']}
    if len(roots) != 1:
        raise ValueError('One known control deployment required')
    result.update(source_sha256=digest_file(__file__),
        source_run=dict(path=str(run_path), sha256=run_sha),
        runtime=dict(path=str(runtime_path), sha256=runtime_sha, artifacts=artifacts),
        control_deployment=next(iter(roots)))
    return result


def merge_catalogs(primary, additional):
    """Merge actual capture identities from controls, not a token-length table."""
    result = copy.deepcopy(primary)
    result['source_runs'] = [primary['source_run']]
    for other in additional:
        if other['runtime']['sha256'] != primary['runtime']['sha256']:
            raise ValueError('All controls must use the same actual runtime')
        result['source_runs'].append(other['source_run'])
        for entry in other['entries']:
            matches = [row for row in result['entries'] if row['graph'] == entry['graph']]
            if matches:
                previous = matches[0]
                keys = ('parameters_sha256', 'retained_input_indices', 'contract',
                        'binding_content_sha256', 'frontend_key', 'machine_key')
                if (any(previous[key] != entry[key] for key in keys)
                        or previous['library']['sha256'] != entry['library']['sha256']):
                    raise ValueError('Same captured graph has conflicting compiled provenance')
            else:
                result['entries'].append(copy.deepcopy(entry))
    if (not 1 <= len(result['entries']) <= 4
            or len({entry['binding_content_sha256'] for entry in result['entries']}) != 1):
        raise ValueError('Bounded entry set for one immutable model required')
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--run-sha256', required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--additional-run', type=Path, action='append', default=[])
    parser.add_argument('--additional-run-sha256', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.additional_run) != len(args.additional_run_sha256):
        parser.error('Every additional completed control needs its explicit digest')
    result = build_catalog(args.run, args.run_sha256, args.runtime, args.runtime_sha256)
    if args.additional_run:
        result = merge_catalogs(result, [build_catalog(path, digest, args.runtime, args.runtime_sha256)
            for path, digest in zip(args.additional_run, args.additional_run_sha256, strict=True)])
    args.output.mkdir(exist_ok=False)
    (args.output/'catalog.json').write_bytes(canonical(result)+b'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
