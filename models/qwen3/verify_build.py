"""Compare rebuilt model DSOs through the original native generation entry.

This is a development check, not release promotion. The accepted asset bundle,
framework guards, graph contracts, constants and model parameters are verified
normally. Only explicitly hashed model libraries may be replaced for this run.
"""
import argparse
import json
from pathlib import Path
import struct

from .bundle import sha, write


def reference_generation_options(reference):
    """Replay the requested budget, including when EOS stopped a request early."""
    options = reference.get('generation_options')
    if options is not None:
        budget, chunk = options['max_new_tokens'], options['prefill_chunk_size']
        if options.get('do_sample') is not False:
            raise ValueError('The reference must use deterministic native generation')
    else:
        # The compact reference16 intentionally omits generation_options. Its
        # completed requests all reached the same original token limit.
        budgets = {len(row['generated_ids']) for row in reference['requests']}
        if (len(budgets) != 1 or not all(row['terminal_observation'].get(
                'reached_requested_token_limit') is True for row in reference['requests'])):
            raise ValueError('Early-EOS references must retain their original generation_options')
        budget, chunk = budgets.pop(), 32
    if type(budget) is not int or budget < 1 or type(chunk) is not int or not 1 <= chunk <= 32:
        raise ValueError('Positive generation budget and supported prefill chunk required')
    return budget, chunk


def stage_candidates(catalog_path, libraries):
    catalog = json.loads(catalog_path.read_text())
    if not libraries or not set(libraries) <= {'q1', 'q2'}:
        raise ValueError('Explicit q1 and/or q2 candidate libraries required')
    identity = {}
    for entry in catalog['entries']:
        bounds = entry['contract']['variables']['query']
        profile = {(1, 1): 'q1', (2, 32): 'q2'}[tuple(bounds)]
        if profile not in libraries:
            continue
        row = libraries[profile]
        source = Path(row['path']).resolve(strict=True)
        if sha(source) != row['sha256']:
            raise ValueError('Candidate library changed: ' + profile)
        with source.open('rb') as stream:
            header = stream.read(20)
        if header[:6] != b'\x7fELF\x01\x01' or struct.unpack_from('<HH', header, 16) != (3, 164):
            raise ValueError('Expected an ELF32 Hexagon shared library')
        destination = Path(catalog['control_deployment']) / (row['sha256'] + '.so')
        if not destination.exists():
            destination.symlink_to(source)
        if sha(destination) != row['sha256']:
            raise ValueError('Candidate staging differs')
        entry['library'] = dict(sha256=row['sha256'], remote_path=str(destination))
        identity[profile] = row['sha256']
    if set(identity) != set(libraries):
        raise ValueError('Candidate domain is absent from the accepted catalog')
    write(catalog_path, catalog)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--libraries', type=Path,
                        help='Optional JSON mapping q1/q2 to candidate path and SHA256; otherwise use the bundle')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet-recording', choices=('full', 'hashes'), default='full',
                        help='Both compare complete packet hashes; full also retains the raw bytes')
    parser.add_argument('--capture-output', type=Path,
                        help='Fresh existing directory for the actual AOT build inputs')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    libraries = json.loads(args.libraries.read_text()) if args.libraries else {}
    budget, chunk = reference_generation_options(reference)
    from .cli import generate
    from .runtime_checks import compare_integer
    args.requests_json = json.dumps([r['prompt_ids'] for r in reference['requests']])
    args.prompt = args.prompts_json = None
    args.chat_template = False
    args.max_new_tokens = budget
    args.chunk_size = chunk
    args.cache_dir = args.output.parent / 'candidate-aot-cache'
    if args.capture_output is not None:
        args.cache_dir = args.output.parent / (args.output.name+'-capture-cache')
        if args.cache_dir.exists():
            parser.error('Capture export needs a fresh AOT cache so both compiler callbacks execute')
    prepare = (lambda catalog: stage_candidates(catalog, libraries)) if args.libraries else None
    generate(args, prepare_candidate=prepare)
    actual = json.loads((args.output/'result.json').read_text())
    if args.capture_output is not None and len(actual.get('exported_captures', [])) != len(actual['captures']):
        raise ValueError('Every actual compiler callback must produce a build capture')
    expected_libraries = {tuple(c['contract']['variables']['query']): c['DSO_sha256']
                          for c in reference['captures']}
    for profile, row in libraries.items():
        expected_libraries[(1, 1) if profile == 'q1' else (2, 32)] = row['sha256']
    compare_integer(reference, actual, expected_libraries)
    write(args.output/'comparison.json', dict(status='identical', DSP_calls=actual['DSP_calls'],
        reference_sha256=sha(args.reference), result_sha256=sha(args.output/'result.json'),
        libraries=libraries, compared='all logits and 56 KV tensors, inputs, positions, tokens and termination',
        release_promoted=False))


if __name__ == '__main__':
    main()
