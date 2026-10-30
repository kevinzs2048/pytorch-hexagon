"""Apply the existing isolated Qwen numerical policy to actual captured Linalg.

This consumes a verified frontend cache entry, not a recaptured q17 model. The
three explicit compiler stages preserve the existing candidate's order. Their
intermediates stream through pipes so weights are not copied to three files.
This is still the bounded Qwen W8A8 candidate policy, not a universal backend.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from hexagon_runtime.hexagon_compiled_artifact_cache import canonical, describe_artifacts, digest_file
from apply_dynamic_bound_qk_policy import FunctionPolicyPatch
from repair_dynamic_scatter_mlir_text import ScatterPropertyRepair
from streamed_mlir_interchange import stream_mlir_pipeline


class CandidateText:
    def __init__(self, operation, *, arm_rope_trig=False):
        self.operation = operation
        self.arm_rope_trig = arm_rope_trig
        self.scatter_count = None

    def print(self, *, file, binary):
        if not binary:
            raise ValueError('Binary complete MLIR printing required')
        repair = ScatterPropertyRepair(file.write)
        policy = FunctionPolicyPatch(repair.write, silu=True, softmax_exp=True,
                                     bounded_rmsnorm=True, bounded_softmax=True,
                                     arm_rope_trig=self.arm_rope_trig)

        class Sink:
            def write(self, data):
                policy.write(data)
                return len(data)

            def flush(self):
                file.flush()

        self.operation.print(file=Sink(), binary=True)
        policy.finish()
        repair.finish()
        self.scatter_count = repair.count
        if repair.count != 56:
            raise ValueError('Current Qwen candidate requires exactly 56 KV scatters')
        file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry', 'attention-optimizer', 'sums-optimizer', 'knorm-optimizer'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--arm-rope-trig", action="store_true",
                        help="Use the explicitly linked four-lane ARM RoPE runtime")
    parser.add_argument('--packed-i8-optimizer',type=Path)
    parser.add_argument('--packed-i8-optimizer-sha256')
    parser.add_argument('--packed-multi-i8-optimizer',type=Path)
    parser.add_argument('--packed-multi-i8-optimizer-sha256')
    parser.add_argument('--silu-effects-optimizer',type=Path)
    parser.add_argument('--silu-effects-optimizer-sha256')
    parser.add_argument('--multi-silu-effects-optimizer',type=Path)
    parser.add_argument('--multi-silu-effects-optimizer-sha256')
    args = parser.parse_args()
    inputs = {}
    if bool(args.multi_silu_effects_optimizer)!=bool(args.multi_silu_effects_optimizer_sha256):
        parser.error('Optional query2..32 SiLU optimizer requires its explicit identity')
    if args.multi_silu_effects_optimizer:
        if not args.packed_multi_i8_optimizer or args.silu_effects_optimizer or args.packed_i8_optimizer:
            parser.error('Query2..32 SiLU requires the existing multi-row integer route and excludes query1 transforms')
        if (not args.multi_silu_effects_optimizer.is_absolute()
                or digest_file(args.multi_silu_effects_optimizer)!=args.multi_silu_effects_optimizer_sha256):
            raise ValueError('Query2..32 SiLU optimizer dependency changed')
        inputs['multi_silu_effects_optimizer']=dict(path=str(args.multi_silu_effects_optimizer),sha256=args.multi_silu_effects_optimizer_sha256)
    if bool(args.packed_multi_i8_optimizer)!=bool(args.packed_multi_i8_optimizer_sha256):
        parser.error('Optional multi-row integer optimizer requires its explicit identity')
    if args.packed_multi_i8_optimizer:
        if args.packed_i8_optimizer or args.silu_effects_optimizer:
            parser.error('The existing multi-token policy cannot select query1-only transforms')
        if (not args.packed_multi_i8_optimizer.is_absolute()
                or digest_file(args.packed_multi_i8_optimizer)!=args.packed_multi_i8_optimizer_sha256):
            raise ValueError('Multi-row integer optimizer dependency changed')
        inputs['packed_multi_i8_optimizer']=dict(path=str(args.packed_multi_i8_optimizer),sha256=args.packed_multi_i8_optimizer_sha256)
    if bool(args.packed_i8_optimizer)!=bool(args.packed_i8_optimizer_sha256):
        parser.error('Optional integer optimizer requires its explicit identity')
    if args.packed_i8_optimizer:
        if (not args.packed_i8_optimizer.is_absolute()
                or digest_file(args.packed_i8_optimizer)!=args.packed_i8_optimizer_sha256):
            raise ValueError('Integer optimizer dependency changed')
        inputs['packed_i8_optimizer']=dict(path=str(args.packed_i8_optimizer),sha256=args.packed_i8_optimizer_sha256)
    if bool(args.silu_effects_optimizer)!=bool(args.silu_effects_optimizer_sha256):
        parser.error('Optional SiLU effect optimizer requires its explicit identity')
    if args.silu_effects_optimizer:
        if (not args.silu_effects_optimizer.is_absolute()
                or digest_file(args.silu_effects_optimizer)!=args.silu_effects_optimizer_sha256):
            raise ValueError('SiLU effect optimizer dependency changed')
        inputs['silu_effects_optimizer']=dict(path=str(args.silu_effects_optimizer),sha256=args.silu_effects_optimizer_sha256)
    for name in ('entry', 'attention_optimizer', 'sums_optimizer', 'knorm_optimizer'):
        path, expected = getattr(args, name), getattr(args, name + '_sha256')
        if digest_file(path) != expected:
            raise ValueError('Compiler input changed: ' + name)
        inputs[name] = dict(path=str(path), sha256=expected)
    entry = json.loads(args.entry.read_text())
    build = entry.get('build', '')
    if (entry.get('status') != 'captured_AOT_Linalg_cached_NOT_machine_code'
            or not isinstance(build, str) or Path(build).name != build
            or not build.startswith(entry.get('key', '') + '-')
            or entry['request']['function_name'] != 'Qwen3DynamicBoundProbe'):
        raise ValueError('Actual captured Qwen frontend entry required')
    folder = args.entry.parent / build
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Original frontend transaction required')
    rows = entry['artifacts']
    if rows != describe_artifacts(folder, ['captured-linalg.mlirbc', 'lowering.json']):
        raise ValueError('Captured Linalg artifact bytes changed')
    source = folder / 'captured-linalg.mlirbc'
    if shutil.disk_usage(args.output.parent).free < source.stat().st_size * 2 + 256 * 1024**2:
        raise ValueError('Insufficient space for final bytecode plus complete text archive')
    args.output.mkdir(exist_ok=False)
    result = dict(status='preparing_actual_captured_Qwen_Linalg', inputs=inputs,
                  frontend_key=entry['key'], source=rows[0], DSP_calls=0,
                  machine_code_compilations=0, numerical_acceptance=False,
                  policy='existing_bounded_Qwen_candidate_NOT_generic_model_policy',
                  source_sha256=digest_file(__file__))
    started = time.monotonic()
    try:
        from torch_mlir import ir
        # Register the same importer dialects as the frontend before parsing.
        from torch_mlir import fx
        context = ir.Context()
        context.allow_unregistered_dialects = True
        with context:
            module = ir.Module.parse(source.read_bytes())
            if not module.operation.verify():
                raise ValueError('Captured Linalg did not verify')
            target = args.output / 'hexagon-linalg.mlirbc'
            first = [str(args.attention_optimizer), '-', '--verify-each', '--emit-bytecode',
                '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{'
                'fp32-attention-dot-order=true fp32-silu-exp-compat=true '
                'short-softmax-exp-compat=true}))', '-o', '-']
            def sum_command(optimizer, output):
                return [str(optimizer), '-', '--verify-each', '--emit-bytecode',
                    '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum))',
                    '-o', str(output)]
            printed = CandidateText(module.operation, arm_rope_trig=args.arm_rope_trig)
            result['interchange'] = stream_mlir_pipeline(printed,
                [first, sum_command(args.sums_optimizer, '-'),
                 sum_command(args.knorm_optimizer, target)],
                archive_path=args.output/'adapted-source.mlir.gz',
                log_path=args.output/'optimizer.log', timeout=1800)
            result['scatter_properties_adapted'] = printed.scatter_count
            integer_optimizer=args.packed_multi_i8_optimizer or args.packed_i8_optimizer
            if integer_optimizer:
                multi=bool(args.packed_multi_i8_optimizer)
                log_name='packed-multi-optimizer.log' if multi else 'packed-optimizer.log'
                baseline_sha=digest_file(target)
                replacement=args.output/'packed-linalg.tmp.mlirbc'
                with (args.output/log_name).open('xb') as log:
                    subprocess.run([str(integer_optimizer),str(target),'--verify-each',
                        '--emit-bytecode','--qwen-packed-i8-matmul='+('multi-row=true ' if multi else '')+'expected-count=197','-o',str(replacement)],
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
                text=(args.output/log_name).read_text()
                if ('qwen packed integer contractions: 197' not in text
                        or 'qwen preserved original operations:' not in text):
                    raise ValueError('Complete integer coverage/preservation proof missing')
                replacement.replace(target)
                result['packed_multi_integer_policy' if multi else 'packed_integer_policy']=dict(coverage=197,
                    baseline_prepared_graph_sha256=baseline_sha,
                    optimizer=inputs['packed_multi_i8_optimizer' if multi else 'packed_i8_optimizer'],
                    all_surviving_original_operations_and_SSA_dependencies_preserved=True,
                    log_sha256=digest_file(args.output/log_name),numerical_acceptance=False)
                if multi:
                    result['packed_multi_integer_policy']['all_original_operations_properties_and_SSA_preserved']=True
            effects_optimizer=args.multi_silu_effects_optimizer or args.silu_effects_optimizer
            if effects_optimizer:
                multi_effects=bool(args.multi_silu_effects_optimizer)
                effects_log='multi-silu-effects-optimizer.log' if multi_effects else 'silu-effects-optimizer.log'
                baseline_sha=digest_file(target)
                replacement=args.output/'silu-effects.tmp.mlirbc'
                with (args.output/effects_log).open('xb') as log:
                    subprocess.run([str(effects_optimizer),str(target),'--verify-each',
                        '--emit-bytecode','--qwen-silu-effects='+('multi-row=true ' if multi_effects else '')+'expected-count=28','-o',str(replacement)],
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
                text=(args.output/effects_log).read_text()
                if ('qwen scoped '+('query2..32' if multi_effects else 'query1')+' SiLU effect calls: 28' not in text
                        or 'qwen preserved original operations:' not in text):
                    raise ValueError('Complete original SiLU coverage/preservation proof missing')
                replacement.replace(target)
                result['multi_silu_effects' if multi_effects else 'silu_effects']=dict(coverage=28,baseline_prepared_graph_sha256=baseline_sha,
                    optimizer=inputs['multi_silu_effects_optimizer' if multi_effects else 'silu_effects_optimizer'],
                    all_original_operations_properties_and_SSA_preserved=True,
                    log_sha256=digest_file(args.output/effects_log),numerical_acceptance=False)
            # Inspect the actual transformed graph before spending time on DSP
            # code generation. Query specialization must not skip a policy.
            from hexagon_qwen_numerical_policy import audit_policy
            inspection = args.output/'operator-policy-inspection.mlir'
            # The frozen numerical tools predate the explicit-call buffer
            # dialects. The integer tool registers those dialects; invoke it
            # without a rewrite pass to verify/print the actual final graph.
            inspector = effects_optimizer or integer_optimizer or args.knorm_optimizer
            result['inspection_optimizer'] = inputs[
                'multi_silu_effects_optimizer' if args.multi_silu_effects_optimizer else
                ('silu_effects_optimizer' if args.silu_effects_optimizer else
                ('packed_multi_i8_optimizer' if args.packed_multi_i8_optimizer else
                 ('packed_i8_optimizer' if args.packed_i8_optimizer else 'knorm_optimizer')))]
            with (args.output/'policy-inspection.log').open('xb') as log:
                subprocess.run([str(inspector), str(target), '--verify-each',
                    '--mlir-elide-elementsattrs-if-larger=16',
                    '--mlir-elide-resource-strings-if-larger=16', '-o', str(inspection)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            result['operator_policy'] = audit_policy(inspection.read_text(),
                                                     arm_rope_trig=args.arm_rope_trig)
            result['operator_policy']['inspection_sha256'] = digest_file(inspection)
        with target.open('rb') as stream:
            if stream.read(4) != b'ML\xefR':
                raise ValueError('Expected verified Hexagon MLIR bytecode')
        result.update(status='actual_captured_Qwen_prepared_for_Hexagon_NOT_machine_code',
                      output=dict(path=str(target), sha256=digest_file(target),
                                  bytes=target.stat().st_size))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        (args.output/'result.json').write_bytes(canonical(result) + b'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
