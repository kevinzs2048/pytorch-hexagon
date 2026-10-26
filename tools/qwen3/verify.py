#!/usr/bin/env python3
"""Run policy, framework and IR admission checks against newly built tools."""
from qwen3.compiler_dependency import compiler_root
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from build import ROOT, entry, write

POLICY = ['test_arm_rope_trig_policy.py', 'test_fp32_sum_cascade.py',
          'test_fp32_silu_decomposition.py', 'test_fp32_silu_exp_compat.py',
          'test_bounded_softmax_singleton_policy.py', 'test_singleton_attention_policy.py',
          'test_unregistered_policy_stream.py', 'test_captured_machine_cache.py']
FRAMEWORK = ['test_arm_dynamic_warm_audit.py', 'test_captured_machine_cache.py',
             'test_compile_qwen_symbolic_carrier.py', 'test_live_compile_mailbox.py',
             'test_qwen_query_shape_scope.py', 'test_static_query_caller_profiles.py',
             'test_qwen_generation_budget.py', 'test_online_packet_trace.py',
             'test_online_numeric_acceptance.py', 'test_online_reference_state.py']
FIXTURES = ['arm-rmsnorm-order.mlir', 'bounded-dynamic-rmsnorm.mlir',
            'bounded-dynamic-k-norm.mlir', 'bounded-dynamic-softmax.mlir',
            'arm-silu-order.mlir', 'arm-rope-trig-order.mlir',
            'bounded-dynamic-rmsnorm-invalid.mlir', 'bounded-dynamic-softmax-invalid.mlir']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    scripts = ROOT/'qwen3/runtime/arm'
    sums = str(a.build/'policy/qwen-policy-sums-opt')
    first = str(a.build/'policy/qwen-policy-first-opt')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(scripts)+':'+str(ROOT),
        LD_LIBRARY_PATH=':'.join(c['library_path']), QWEN_MATMUL_SCHEDULE_OPT=sums,
        QWEN_SINGLETON_SUM_OPT=sums, QWEN_SINGLETON_ATTENTION_OPT=first,
        QWEN_HEXAGON_LOWERING_OPT=c['lowering_opt'])
    result = dict(status='checking', source_root=str(ROOT), tests={}, filechecks=[])
    try:
        for name, files in [('policy', POLICY), ('framework', FRAMEWORK)]:
            command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                       '--junitxml='+str(a.output/(name+'.xml')), *[str(scripts/f) for f in files]]
            with (a.output/(name+'.log')).open('xb') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            suite = ET.parse(a.output/(name+'.xml')).getroot()
            counts = {key: sum(int(node.get(key, '0')) for node in suite.iter('testsuite'))
                      for key in ['tests', 'failures', 'errors', 'skipped']}
            if counts['skipped'] or counts['failures'] or counts['errors']:
                raise ValueError('Verification requires all tests to execute and pass')
            result['tests'][name] = dict(counts, command=command)
        for name in FIXTURES:
            fixture = compiler_root()/'qcom_hexagon_backend/test/Transforms'/name
            invalid = '-invalid.' in name
            command = [sums, str(fixture), '--verify-each',
                       '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))']
            opt = subprocess.run(command, env=env, capture_output=True)
            if bool(opt.returncode) != invalid:
                raise ValueError('Unexpected policy admission result: '+name)
            check = subprocess.run([c['filecheck'], str(fixture)],
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
