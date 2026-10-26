"""Relink a freshly compiled object using a pinned, previously tested link recipe.

This is a host-only operation: it never configures a device transport or asks
for an SSH password. It admits only the same lowered graph/ABI and identical
constant objects as the reference; it is not an arbitrary-model ABI generator.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from hexagon_compiled_artifact_cache import canonical, digest_file


def replace_object(argv, old_object, new_object, new_output):
    argv = list(argv)
    if (argv.count(str(old_object)) != 1 or argv.count('-o') != 1
            or len(argv) < 3 or argv[-2] != '-o'
            or not Path(argv[0]).is_absolute()):
        raise ValueError('Unique original object and terminal link output required')
    argv[argv.index(str(old_object))] = str(new_object)
    argv[-1] = str(new_output)
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('objects', 'reference', 'manifest', 'constants'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('objects', 'reference', 'manifest', 'constants'):
        if digest_file(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise ValueError('Pinned relink input changed: ' + name)
    objects = json.loads(args.objects.read_text())
    reference = json.loads(args.reference.read_text())
    manifest = json.loads(args.manifest.read_text())
    constants = json.loads(args.constants.read_text())
    if (objects.get('machine_code_compilations') != 1
            or objects.get('status') !=
            'dynamic-bound_q17_lifted_graph_Hexagon_code_candidate_constants_identical_NOT_DSP'
            or reference.get('status') != 'dynamic_q17_file_caller_linked_NOT_DSP_forward'
            or objects['inputs']['linalg']['sha256'] != reference['graph_sha256']
            or len(objects['objects']) != 4
            or reference['constants_result_sha256'] != args.constants_sha256
            or digest_file(reference['library']['path']) != reference['library']['sha256']
            or digest_file(reference['wrapper']['path']) != reference['wrapper']['sha256']):
        raise ValueError('Same compiled graph, original ABI and verified reference required')
    old_object = args.reference.parent / 'bounded_rmsnorm_qwen.o'
    new_object = Path(objects['objects'][0]['path'])
    if digest_file(new_object) != objects['objects'][0]['sha256']:
        raise ValueError('Freshly compiled code changed')
    # The reference object's identity is logged even if code generation is
    # nondeterministic. Graph equality above, not object equality, guards ABI.
    old_object_sha = digest_file(old_object)
    argv = shlex.split(manifest['link_command'])
    if (argv[-1] != reference['library']['path']
            or argv.count(reference['wrapper']['path']) != 1):
        raise ValueError('Original link command does not bind the reference library')
    directories = [Path(arg[2:]) for arg in argv if arg.startswith('-L')]
    libraries = sorted((row for row in constants['libraries'] if row['index']),
                       key=lambda row: row['index'])
    if [row['index'] for row in libraries] != [1, 2, 3]:
        raise ValueError('Three original Qwen constant libraries required')
    for row in libraries:
        path = Path(row['path'])
        candidates = [folder/path.name for folder in directories if (folder/path.name).is_file()]
        if (objects['objects'][row['index']]['sha256'] != row['input_object_sha256']
                or digest_file(path) != row['sha256'] or not candidates
                or candidates[0].resolve() != path.resolve()
                or argv.count('-l' + path.name[3:-3]) != 1):
            raise ValueError('Linked constant library changed or search path shadowed it')
    linker = constants['inputs']['lld']
    if (argv.count('--ld-path=' + linker['path']) != 1
            or digest_file(linker['path']) != linker['sha256']):
        raise ValueError('Pinned linker changed')
    header = Path(reference['wrapper']['path']).parent/'qwen3_sleef_exp_candidate.h'
    if digest_file(header) != reference['silu_runtime']['header_copy_sha256']:
        raise ValueError('Pinned numerical runtime header changed')
    args.output.mkdir(exist_ok=False)
    target = args.output / Path(reference['library']['path']).name
    command = replace_object(argv, old_object, new_object, target)
    record = dict(reference)
    record.update(status='relinking_actual_captured_Hexagon_object',
        source_sha256=digest_file(__file__), objects_result_sha256=args.objects_sha256,
        reference_result_sha256=args.reference_sha256,
        reference_link_manifest_sha256=args.manifest_sha256,
        compiler_sha256=digest_file(argv[0]), linker_sha256=linker['sha256'],
        reference_object_sha256=old_object_sha,
        new_object=dict(path=str(new_object), sha256=objects['objects'][0]['sha256']),
        host_only_link=True, DSP_calls=0, model_execution=False,
        numerical_acceptance=False, command=command)
    try:
        with (args.output/'link.log').open('xb') as stream:
            subprocess.run(command, cwd=args.reference.parent, check=True,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        with target.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('Linker did not emit an ELF library')
        record.update(status='dynamic_q17_file_caller_linked_NOT_DSP_forward',
                      library=dict(path=str(target), sha256=digest_file(target),
                                   bytes=target.stat().st_size))
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output/'result.json').write_bytes(canonical(record)+b'\n')
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
