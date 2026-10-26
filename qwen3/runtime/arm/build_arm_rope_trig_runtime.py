"""Build the opt-in four-lane ARM RoPE runtime; no device execution."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hexagon-clang', type=Path, required=True)
    parser.add_argument('--lld', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = compiler_tools() / 'arm_rope_trig_runtime'
    args.output.mkdir(exist_ok=False, parents=True)
    commands = []
    inputs = {str(path): digest(path) for path in sorted(source.iterdir()) if path.is_file()}
    for stem in ('trig_lane', 'rempitab'):
        command = [str(args.hexagon_clang.resolve()), '-O2', '-ffp-contract=off',
                   '-fPIC', '-mv73', '-mhvx', '-Wno-unknown-pragmas', '-Wno-gcc-compat',
                   '-I', str(source), '-c', str(source / (stem + '.c')),
                   '-o', str(args.output / (stem + '.o'))]
        with (args.output / (stem + '.log')).open('xb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        commands.append(command)
    runtime = args.output / 'trig_runtime.o'
    command = [str(args.lld.resolve()), '-r', str(args.output / 'trig_lane.o'),
               str(args.output / 'rempitab.o'), '-o', str(runtime)]
    subprocess.run(command, check=True)
    commands.append(command)
    record = dict(status='ARM_four_lane_RoPE_runtime_built_NOT_DSP',
                  source_sha256=digest(__file__), inputs=inputs,
                  compiler=dict(path=str(args.hexagon_clang), sha256=digest(args.hexagon_clang)),
                  linker=dict(path=str(args.lld), sha256=digest(args.lld)), commands=commands,
                  runtime=dict(path=str(runtime), sha256=digest(runtime)), DSP_calls=0)
    (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(record['status'])


if __name__ == '__main__':
    main()
