# SPDX-License-Identifier: BSD-3-Clause
"""Resident-builder steps of the Q8_0 projections (models/qwen3/build_model.py).

rewrite_domain       --qwen-q8-0-projections on a prepared domain (after the
                     fused INT8 dequantization), with the INT8 weight dump
weights_and_library  every domain dumped the same id -> weight bytes; Q8_0 tiles
                     from the HF checkpoint (gguf-py), the weight library and table
runtime_object       runtime/q8_0/q8_0_hmx.cpp for the model link
"""
import json
import subprocess
from pathlib import Path

from . import library, weights
from .weights import sha

PROJECTIONS = 197                      # 28 layers x 7 + LM head
SINGLE_ROW = ('initial_single', 'decode')


def rewrite_domain(tool, source, target, domain, dump, env, log, passes=()):
    """Rewrite one prepared domain after the given fused-graph passes, in one
    policy invocation; returns the pass record."""
    dump.mkdir(parents=True)
    options = f'expected-count={PROJECTIONS}' + (' single-row=true' if domain in SINGLE_ROW else '')
    with open(log, 'wb') as out:
        subprocess.run([str(tool), str(source), *passes, f'--qwen-q8-0-projections={options} dump-dir={dump}',
                        '--canonicalize', '--verify-each', '--emit-bytecode', '-o', str(target)],
                       env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    text = Path(log).read_text()
    if f'Q8_0 projections: {PROJECTIONS}, activation quantizations removed: 113' not in text:
        raise ValueError('Q8_0 projection coverage differs: ' + domain)
    return dict(options=options, single_row=domain in SINGLE_ROW, passes=list(passes),
                manifest_sha256=sha(dump / 'manifest.json'))


def same_dumps(dumps):
    """All domains map every id to byte-identical INT8 weights and column scales."""
    names = sorted(dumps)
    first = dumps[names[0]]
    manifest = (first / 'manifest.json').read_bytes()
    for name in names[1:]:
        if (dumps[name] / 'manifest.json').read_bytes() != manifest:
            raise ValueError('Q8_0 projection order differs: ' + name)
        for row in json.loads(manifest):
            for suffix in ('w', 's'):
                if sha(first / f"{row['id']}.{suffix}") != sha(dumps[name] / f"{row['id']}.{suffix}"):
                    raise ValueError(f"Q8_0 id {row['id']} weight differs: {name}")
    return first


def weights_and_library(dumps, hf_checkpoint, hf_sha256, gguf_py, out, hexagon_tools, lld, python, root, env, log):
    if sha(hf_checkpoint) != hf_sha256:
        raise ValueError('HF checkpoint differs from the resident plan')
    dump = same_dumps(dumps)
    tiles = out / 'q8_0-tiles'
    record = weights.build(dump, hf_checkpoint, gguf_py, tiles, log=log)
    info = library.build(tiles, out / 'q8_0-library', hexagon_tools, lld, python, root, env)
    for row in record['projections']:  # the tiles are inputs of the library only
        (tiles / f"{row['id']}.q8").unlink()
    for name in dumps.values():
        for path in name.glob('*.[ws]'):
            path.unlink()  # weight-sized dumps; manifest and hashes stay in the record
    return dict(weights=dict(json=str(tiles / 'weights.json'), sha256=sha(tiles / 'weights.json'),
                             source=record['source'], hf_sha256=record['hf_sha256']),
                library=info)


def runtime_object(hexagon_tools, includes, hexkl_include, out, env, run):
    source = Path(__file__).resolve().parents[1] / 'native/runtime/q8_0/q8_0_hmx.cpp'
    obj = out / 'q8_0_hmx.o'
    run('q8_0-runtime', [Path(hexagon_tools) / 'bin/hexagon-clang++', '-O3', '-mv73', '-mhvx', '-mhvx-length=128B',
                         '-mhmx', '-mhvx-ieee-fp', '-fPIC', '-G0', '-mlong-calls', '-std=c++17', '-fno-math-errno',
                         '-Drestrict=__restrict', *['-I' + str(p) for p in includes], '-I' + str(hexkl_include),
                         '-c', source, '-o', obj])
    return dict(object=str(obj), object_sha256=sha(obj), source_sha256=sha(source))
