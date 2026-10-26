"""Generate a q1 or bounded caller from the live AOT-derived contract.

The filename preserves the original q1 CLI; scalar arity is no longer fixed.
"""
import argparse
import json
from pathlib import Path

from hexagon_compiled_artifact_cache import digest_file,canonical
from prepare_qwen_static_query_entry import write_caller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry', 'contract', 'prepared-graph', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--triton-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--silu-effects-guard',type=Path)
    parser.add_argument('--silu-effects-guard-sha256')
    parser.add_argument('--silu-effects-dynamic',action='store_true')
    args = parser.parse_args()
    records = {}
    for name in ('entry', 'contract', 'prepared_graph', 'reference'):
        path = getattr(args, name)
        if digest_file(path) != getattr(args, name+'_sha256'):
            raise ValueError('Captured caller dependency changed: '+name)
        records[name] = json.loads(path.read_text())
    entry, contract, prepared, reference = [records[name] for name in ('entry', 'contract', 'prepared_graph', 'reference')]
    if (entry['status'] != 'captured_AOT_Linalg_cached_NOT_machine_code'
            or prepared['status'] != 'actual_captured_Qwen_prepared_for_Hexagon_NOT_machine_code'
            or prepared['frontend_key'] != entry['key']
            or prepared['source']['sha256'] != entry['artifacts'][0]['sha256']
            or digest_file(prepared['output']['path']) != prepared['output']['sha256']):
        raise ValueError('Actual captured frontend and prepared graph must agree')
    effects_key='multi_silu_effects' if args.silu_effects_dynamic else 'silu_effects'
    other_effects_key='silu_effects' if args.silu_effects_dynamic else 'multi_silu_effects'
    if (bool(args.silu_effects_guard)!=bool(args.silu_effects_guard_sha256)
            or bool(args.silu_effects_guard)!=(effects_key in prepared)
            or other_effects_key in prepared or (args.silu_effects_dynamic and not args.silu_effects_guard)):
        raise ValueError('Scoped effect graph and actual runtime guard must be paired')
    if args.silu_effects_guard and (digest_file(args.silu_effects_guard)!=args.silu_effects_guard_sha256
            or prepared[effects_key]['coverage']!=28
            or prepared[effects_key]['all_original_operations_properties_and_SSA_preserved'] is not True):
        raise ValueError('Scoped effect coverage or guard changed')
    write_caller(contract, reference, args.output, args.triton_python,
        '/dev/shm/qwen3-captured-q1-automatic', provenance=dict(
            frontend_entry_sha256=args.entry_sha256, contract_sha256=args.contract_sha256,
            prepared_graph_sha256=args.prepared_graph_sha256,
            graph_sha256=prepared['output']['sha256'], reference_sha256=args.reference_sha256,
            frontend_key=entry['key']))
    if args.silu_effects_guard:
        from hexagon_silu_helper_effects import guard_caller
        result=args.output/'result.json';caller=json.loads(result.read_text())
        guard_caller(caller,args.silu_effects_guard,multi=args.silu_effects_dynamic)
        result.write_bytes(canonical(caller)+b'\n')


if __name__ == '__main__':
    main()
