#!/usr/bin/env python3
"""Compatibility entry for the unified source build.

Use python -m qwen3.build. Small metadata helpers below remain for historical
evidence/profiling tools; no reference-wrapper recipe is generated here.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Also support the existing direct build command and its Python subprocesses.
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def entry(path):
    return dict(path=str(Path(path).absolute()), sha256=sha(path))


def verify_dependencies(config, lock):
    # Hexagon clang selects the LLD argument dialect by executable basename.
    # Renaming identical bytes can select the GNU Hexagon linker dialect.
    if Path(config['lld']).name != 'ld.lld':
        raise ValueError('Hexagon clang requires the pinned linker basename ld.lld')
    checked = []
    for row in lock['files']:
        root = Path(config[row['root']])
        path = root / row['relative'] if row['relative'] else root
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f"Pinned SDK dependency changed: {row['root']}/{row['relative']}")
        checked.append(dict(root=row['root'], relative=row['relative'], sha256=row['sha256']))
    for row in json.loads((HERE/'model-constants.lock.json').read_text())['libraries']:
        path = Path(config['constants_dir']) / row['name']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f"Immutable model constants changed: {row['name']}")
    return checked



if __name__ == "__main__":
    from qwen3.build import main
    main()
