#!/usr/bin/env python3
"""Verify the pinned ARM W8A8 implementation before executing any DSP graph."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from build import HERE, write

PROBE = r'''
import hashlib,importlib,json,platform,sys
from pathlib import Path
assert platform.machine() == 'aarch64'
rows=[];versions={}
for name in ('torch','torchao','transformers'):
    m=importlib.import_module(name);versions[name]=m.__version__
    root=Path(m.__file__).parent
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix in ('.py','.so') or '.so.' in p.name):
            with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            rows.append(dict(root=name,relative=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha))
import torch
for name,value in zip(('context','decompositions'),sys.argv[1:]):
    p=Path(value)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    rows.append(dict(root=name,relative='',bytes=p.stat().st_size,sha256=sha))
print(json.dumps(dict(schema=1,architecture=platform.machine(),python=sys.version.split()[0],versions=versions,torch_git=torch.version.git_version,files=rows)))
'''


def inspect(config):
    b = config['board']
    command = ['env', 'PYTHONPATH='+b['pythonpath'], 'PYTHONDONTWRITEBYTECODE=1',
               b['python'], '-c', PROBE, b['context'], b['decompositions']]
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
        b['ssh_target'], shlex.join(command)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    expected = json.loads((HERE/'board-framework.lock.json').read_text())
    actual = inspect(json.loads(a.config.read_text()))
    write(a.output, dict(passed=actual == expected, actual=actual))
    if actual != expected:
        raise ValueError('ARM framework/source dependencies changed; inspect the saved actual manifest')
    print('Pinned ARM W8A8 framework and decompositions verified', flush=True)


if __name__ == '__main__':
    main()
