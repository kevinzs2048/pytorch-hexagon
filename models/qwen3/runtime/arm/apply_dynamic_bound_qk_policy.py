"""Stream a pinned dynamic Qwen graph through the isolated QK-only optimizer.

This is a structural compiler experiment. It never executes the model or
claims numerical acceptance; PV, softmax, and SiLU remain outside this patch.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import traceback

from repair_dynamic_scatter_mlir_text import ScatterPropertyRepair, sha


FUNCTION = b'  func.func @Qwen3DynamicBoundProbe('
POLICY = (b') attributes {hexagon.fp32_prefill_query = 17 : i64, '
          b'hexagon.fp32_prefill_capacity = 1024 : i64, '
          b'hexagon.fp32_prefill_order = "qk-fma-pv-openblas-armv8-k1024"} {\n')
SILU_DECLARATION = b'  func.func private @__hexagon_silu_exp_compat(f32) -> f32\n'
SOFTMAX_DECLARATION = b'  func.func private @__hexagon_softmax_exp_compat(f32) -> f32\n'


class FunctionPolicyPatch:
    def __init__(self, emit, *, silu=False, softmax_exp=False,
                 bounded_rmsnorm=False, bounded_softmax=False, arm_rope_trig=False):
        self.emit = emit
        self.silu = silu
        self.softmax_exp = softmax_exp
        self.bounded_rmsnorm = bounded_rmsnorm
        self.bounded_softmax = bounded_softmax
        self.arm_rope_trig = arm_rope_trig
        self.pending = b''
        self.patched = False

    def write(self, data):
        if self.patched:
            self.emit(data)
            return
        self.pending += data
        start = self.pending.find(FUNCTION)
        if start < 0:
            if len(self.pending) > 256 * 1024:
                raise ValueError('Qwen function signature exceeded bounded prefix')
            return
        end = self.pending.find(b'\n', start)
        if end < 0:
            if len(self.pending) > 256 * 1024:
                raise ValueError('Qwen function signature exceeded bounded prefix')
            return
        line = self.pending[start:end + 1]
        if not line.endswith(b') {\n') or line.count(b') {\n') != 1:
            raise ValueError('Unexpected Qwen function signature boundary')
        policy = POLICY
        extra = []
        if self.arm_rope_trig:
            extra.append(b"hexagon.fp32_arm_rope_trig4 = true")
        if self.bounded_rmsnorm:
            extra.append(b'hexagon.fp32_dynamic_rmsnorm_max_q = 32 : i64')
        if self.bounded_softmax:
            extra.append(b'hexagon.fp32_dynamic_softmax_max_q = 32 : i64')
        if extra:
            policy = policy.replace(
                b'} {\n',
                b', ' + b', '.join(extra) + b'} {\n',
            )
        changed = self.pending[:start] + line[:-4] + policy + self.pending[end + 1:]
        if self.silu or self.softmax_exp or self.arm_rope_trig:
            module = b'module {\n'
            declarations = ((SILU_DECLARATION if self.silu else b'') +
                            (SOFTMAX_DECLARATION if self.softmax_exp else b''))
            if self.arm_rope_trig:
                declarations += b"  func.func private @__hexagon_arm_sin4_lane(f32,f32,f32,f32,i32) -> f32\n  func.func private @__hexagon_arm_cos4_lane(f32,f32,f32,f32,i32) -> f32\n"
            if (changed.count(module) != 1 or
                    (self.silu and SILU_DECLARATION in changed) or
                    (self.softmax_exp and SOFTMAX_DECLARATION in changed)):
                raise ValueError('Unique module declaration insertion point required')
            changed = changed.replace(module, module + declarations, 1)
        self.emit(changed)
        self.pending = b''
        self.patched = True

    def finish(self):
        if not self.patched or self.pending:
            raise ValueError('Exactly one Qwen function must be annotated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--source-archive-sha256', required=True)
    parser.add_argument('--optimizer', type=Path, required=True)
    parser.add_argument('--optimizer-sha256', required=True)
    parser.add_argument('--bytecode-output', type=Path, required=True)
    parser.add_argument('--record-output', type=Path, required=True)
    parser.add_argument('--silu', action='store_true',
                        help='Also enable strict symbolic SiLU exp compatibility')
    parser.add_argument('--softmax-exp', action='store_true',
                        help='Enable strict dynamic q17 softmax exponent diagnostic, not sum/PV policy')
    parser.add_argument('--pv', action='store_true',
                        help='Record isolated dynamic q17 PV block policy optimizer')
    parser.add_argument('--sum', action='store_true',
                        help='Record guarded dynamic q17 softmax sum cascade optimizer')
    parser.add_argument('--rmsnorm', action='store_true',
                        help='Record q17-only dynamic RMSNorm sum cascade extension')
    parser.add_argument('--bounded-rmsnorm', action='store_true',
                        help='Annotate max query 32 for a later generic RMSNorm pass')
    parser.add_argument('--bounded-softmax', action='store_true',
                        help='Annotate max query 32 for a later generic Softmax sum pass')
    args = parser.parse_args()
    if args.bytecode_output.exists() or args.record_output.exists():
        raise FileExistsError('Fresh graph and record paths required')
    if args.pv and not (args.silu and args.softmax_exp):
        raise ValueError('PV candidate is layered atop both explicit exponent policies')
    if args.sum and not (args.pv and args.silu and args.softmax_exp):
        raise ValueError('Sum candidate is layered atop dynamic PV and both exponent policies')
    if args.rmsnorm and not args.sum:
        raise ValueError('RMSNorm candidate is layered atop guarded dynamic softmax sum')
    if args.bounded_rmsnorm and (not args.pv or args.rmsnorm):
        raise ValueError('Bounded RMSNorm annotation requires PV, without q17 RMSNorm rewrite')
    if args.bounded_softmax and (not args.pv or args.sum):
        raise ValueError('Bounded Softmax annotation requires PV, without q17 Softmax sum rewrite')
    if sha(args.source_archive) != args.source_archive_sha256 or sha(args.optimizer) != args.optimizer_sha256:
        raise ValueError('Pinned input or optimizer drifted')
    if shutil.disk_usage(args.bytecode_output.parent).free < 2 * 1024**3:
        raise ValueError('At least 2 GiB scratch output required')
    args.record_output.mkdir(exist_ok=False)
    result = dict(status='admitting_dynamic_QK_SiLU_softmax_exp_PV_sum_RMSNorm_policy' if args.rmsnorm else
                      'admitting_dynamic_QK_SiLU_softmax_exp_PV_sum_policy' if args.sum else
                      'admitting_dynamic_QK_SiLU_softmax_exp_PV_policy' if args.pv else
                      'admitting_dynamic_QK_SiLU_softmax_exp_policy' if args.silu and args.softmax_exp else
                      'admitting_dynamic_QK_SiLU_policy' if args.silu else
                      'admitting_dynamic_QK_only_policy', source_sha256=sha(__file__),
                  input_archive_sha256=args.source_archive_sha256,
                  optimizer_sha256=args.optimizer_sha256, DSP_calls=0,
                  machine_code_compilations=0, numerical_acceptance=False,
                  silu_compat=args.silu, softmax_exp_diagnostic=args.softmax_exp,
                  pv_block_candidate=args.pv, softmax_sum_candidate=args.sum,
                  rmsnorm_sum_candidate=args.rmsnorm,
                  bounded_rmsnorm_annotation=args.bounded_rmsnorm,
                  bounded_softmax_annotation=args.bounded_softmax)

    def save():
        (args.record_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    process = None
    timer = None
    expired = threading.Event()
    try:
        raw_hash, patched_hash, repaired_hash = (hashlib.sha256() for _ in range(3))
        raw_bytes = patched_bytes = repaired_bytes = 0
        flags = ('fp32-attention-dot-order=true' +
                 (' fp32-silu-exp-compat=true' if args.silu else '') +
                 (' short-softmax-exp-compat=true' if args.softmax_exp else ''))
        argv = [str(args.optimizer), '-', '--verify-each', '--emit-bytecode',
            '--pass-pipeline=builtin.module(func.func(stabilize-fp32-sum{' + flags + '}))',
            '-o', str(args.bytecode_output)]
        with (args.record_output / 'optimizer.log').open('xb') as log:
            process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT)

            def expire():
                expired.set()
                if process.poll() is None:
                    process.kill()

            timer = threading.Timer(1800, expire)
            timer.daemon = True
            timer.start()

            def emit_repaired(data):
                nonlocal repaired_bytes
                process.stdin.write(data)
                repaired_hash.update(data)
                repaired_bytes += len(data)

            repair = ScatterPropertyRepair(emit_repaired)

            def emit_patched(data):
                nonlocal patched_bytes
                patched_hash.update(data)
                patched_bytes += len(data)
                repair.write(data)

            annotate = FunctionPolicyPatch(emit_patched, silu=args.silu,
                                            softmax_exp=args.softmax_exp,
                                            bounded_rmsnorm=args.bounded_rmsnorm,
                                            bounded_softmax=args.bounded_softmax)
            with gzip.open(args.source_archive, 'rb') as source:
                while chunk := source.read(1024 * 1024):
                    raw_hash.update(chunk)
                    raw_bytes += len(chunk)
                    annotate.write(chunk)
            annotate.finish()
            repair.finish()
            if repair.count != 56:
                raise ValueError(f'Expected 56 KV scatter repairs, got {repair.count}')
            process.stdin.close()
            status = process.wait(timeout=1800)
            if expired.is_set() or status:
                raise RuntimeError(f'Isolated optimizer failed: status={status}, timed_out={expired.is_set()}')
        with args.bytecode_output.open('rb') as stream:
            if stream.read(4) != b'ML\xefR':
                raise ValueError('Optimizer did not produce MLIR bytecode')
        result.update(status=('dynamic_bound_QK_SiLU_softmax_exp_PV_sum_RMSNorm_bytecode_NOT_numerical_or_DSP'
                              if args.rmsnorm else
                              'dynamic_bound_QK_SiLU_softmax_exp_PV_sum_bytecode_NOT_numerical_or_DSP'
                              if args.sum else
                              'dynamic_bound_QK_SiLU_softmax_exp_PV_bytecode_NOT_numerical_or_DSP'
                              if args.pv else
                              'dynamic_bound_QK_SiLU_softmax_exp_bytecode_NOT_numerical_or_DSP'
                              if args.silu and args.softmax_exp else
                              'dynamic_bound_QK_SiLU_policy_bytecode_NOT_numerical_or_DSP'
                              if args.silu else 'dynamic_bound_QK_only_policy_bytecode_NOT_numerical_or_DSP'),
            argv=argv, interchange=dict(raw_text_sha256=raw_hash.hexdigest(),
                raw_text_bytes=raw_bytes, annotated_text_sha256=patched_hash.hexdigest(),
                annotated_text_bytes=patched_bytes, repaired_text_sha256=repaired_hash.hexdigest(),
                repaired_text_bytes=repaired_bytes, scatter_repairs=repair.count,
                function_annotations=1, external_SiLU_declarations=int(args.silu),
                external_softmax_declarations=int(args.softmax_exp)),
            output=dict(path=str(args.bytecode_output), sha256=sha(args.bytecode_output),
                        bytes=args.bytecode_output.stat().st_size))
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error)[:2000])
        (args.record_output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        save()
    print(result['status'])


if __name__ == '__main__':
    main()
