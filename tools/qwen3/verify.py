#!/usr/bin/env python3
"""Run the retained host suite and IR checks against the unified source build."""
from qwen3.compiler_dependency import compiler_root
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from build import ROOT, entry, write

FIXTURES = ['arm-rmsnorm-order.mlir', 'bounded-dynamic-rmsnorm.mlir',
            'bounded-dynamic-k-norm.mlir', 'bounded-dynamic-softmax.mlir',
            'arm-silu-order.mlir', 'arm-rope-trig-order.mlir',
            'bounded-dynamic-rmsnorm-invalid.mlir', 'bounded-dynamic-softmax-invalid.mlir']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compiler-source', type=Path)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    source = a.compiler_source.resolve() if a.compiler_source else compiler_root()
    a.build = a.build.resolve()
    from qwen3.build import verify_products
    verify_products(a.build, source, a.config)
    a.output.mkdir(parents=True, exist_ok=False)
    scripts = ROOT/'qwen3/runtime/arm'
    sums = str(a.build/'policy/qwen-policy-sums-opt')
    first = str(a.build/'policy/qwen-policy-first-opt')
    lowering = str(a.build/'compiler/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt')
    llvm_bin = Path(c['llvm_build'])/'bin'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(map(str,(scripts,ROOT/'tools/qwen3',ROOT,source))),
        LD_LIBRARY_PATH=':'.join(c['library_path']), QWEN_MATMUL_SCHEDULE_OPT=lowering,
        QWEN_SINGLETON_SUM_OPT=sums, QWEN_SINGLETON_ATTENTION_OPT=first,
        QWEN_HEXAGON_LOWERING_OPT=lowering, QWEN_MLIR_OPT=str(llvm_bin/'mlir-opt'),
        QWEN_SILU_EFFECT_OPT=str(a.build/'policy/qwen-silu-effects-opt'))
    result = dict(status='checking', source_root=str(ROOT), tests={}, filechecks=[])
    try:
        name = 'retained-suite'
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   '--junitxml='+str(a.output/(name+'.xml')),
                   str(ROOT/'test/python/qwen3'), str(ROOT/'test/python/torch-mlir')]
        with (a.output/(name+'.log')).open('xb') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        suite = ET.parse(a.output/(name+'.xml')).getroot()
        counts = {key: sum(int(node.get(key, '0')) for node in suite.iter('testsuite'))
                  for key in ['tests', 'failures', 'errors', 'skipped']}
        if counts['skipped'] or counts['failures'] or counts['errors']:
            raise ValueError('Verification requires all tests to execute and pass')
        result['tests'][name] = dict(counts, command=command)
        checks = [(source/'qcom_hexagon_backend/test/Transforms'/name, sums,
                   'builtin.module(func.func(stabilize-fp32-sum))') for name in FIXTURES]
        checks.append((source/'qcom_hexagon_backend/test/Conversion/LinalgToLLVM/fdiv_sqrt_to_rsqrt_hexagon_routines.mlir',
                       lowering, 'builtin.module(llvm-intr-to-hexagon-routines)'))
        for fixture, optimizer, pipeline in checks:
            name = fixture.name
            invalid = '-invalid.' in name
            command = [optimizer, str(fixture), '--verify-each', '--pass-pipeline='+pipeline]
            opt = subprocess.run(command, env=env, capture_output=True)
            if bool(opt.returncode) != invalid:
                raise ValueError('Unexpected policy admission result: '+name)
            check = subprocess.run([str(llvm_bin/'FileCheck'), str(fixture)],
                input=opt.stdout+opt.stderr if invalid else opt.stdout, capture_output=True, env=env)
            (a.output/(name+'.log')).write_bytes(opt.stderr+check.stderr)
            if check.returncode:
                raise ValueError('IR FileCheck failed: '+name)
            result['filechecks'].append(dict(fixture=entry(fixture), expected_rejection=invalid, passed=True))
        result['status'] = 'policy_framework_and_IR_checks_passed'
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(a.output/'result.json', result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
