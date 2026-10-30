"""Validate byte-copy semantics and actual DSP32 integer guards, then time ABBA.

The oracle enumerates descriptor addresses using Python integers. It never calls
the candidate C++ implementation. Separate diagnostic hooks are not timed as
production performance or linked into the production model runtime.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import traceback

INPUT_BYTES = 256 + 4096
OUTPUT_BYTES = 128 + 4096


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cases(fixed512=False):
    # Models' 512-byte contiguous hot path, other extents, and distinct ABI cases.
    original = [
        ('hot-512', [128], [1], [1], 0, 0, 4),
        ('contiguous-8', [1], [1], [1], 0, 0, 8),
        ('contiguous-256', [64], [1], [1], 0, 0, 4),
        ('contiguous-2048', [64, 8], [8, 1], [8, 1], 0, 0, 4),
        ('scalar', [], [], [], 3, 7, 16),
        ('strided', [8], [2], [3], 1, 2, 1),
        ('rank-three', [2, 3, 8], [24, 8, 1], [24, 8, 1], 2, 3, 4),
        ('size-one-gap', [2, 1, 8], [16, 123, 1], [24, 99, 1], 2, 3, 16),
        ('source-outer-broadcast', [2, 3, 8], [0, 8, 1], [24, 8, 1], 2, 3, 4),
        ('source-inner-broadcast', [2, 3, 8], [24, 0, 1], [24, 8, 1], 2, 3, 8),
        ('destination-outer-broadcast', [2, 3, 8], [24, 8, 1], [0, 8, 1], 2, 3, 4),
        ('destination-inner-broadcast', [2, 3, 8], [24, 8, 1], [24, 0, 1], 2, 3, 8),
        ('negative-strides', [2, 3, 8], [-24, 8, -1], [24, -8, 1], 64, 64, 16),
        ('empty', [2, 0, 8], [24, 8, 1], [24, 8, 1], 0, 0, 4),
        ('rank-four', [1, 2, 2, 3], [100, 24, 8, 1], [100, 24, 8, 1], 1, 5, 2),
        ('rank-eight', [1, 1, 1, 1, 1, 1, 2, 8], [999, 7, 21, 43, 0, -4, 8, 1],
         [777, -7, 39, 54, 0, -3, 8, 1], 0, 0, 4),
    ]
    if fixed512:
        original += [
            ('hot-512-offset31', [512], [1], [1], 31, 1, 1),
            ('hot-512-offset16', [512], [1], [1], 16, 16, 1),
            ('hot-512-strided-blocks', [3, 512], [640, 1], [700, 1], 1, 3, 1),
            ('hot-512-negative-outer', [2, 512], [-200, 1], [600, 1], 201, 1, 1),
        ]
    return original


def packet_and_reference(case, repeats, distinct_blocks=False):
    _, sizes, src_strides, dst_strides, src_offset, dst_offset, element_size = case
    if not 1 <= repeats <= 100000 or len(sizes) > 8 or not 1 <= element_size <= 16:
        raise ValueError('Invalid bounded copy case')
    if len(sizes) != len(src_strides) or len(sizes) != len(dst_strides):
        raise ValueError('Descriptor ranks differ')
    source = bytes((i * 37 + 19 + ((i//128)*53+(i//257)*11 if distinct_blocks else 0)) % 256 for i in range(4096))
    expected = bytearray([0xa5] * 4096)
    # Repetitions are idempotent: all reads use the same immutable source.
    indices = itertools.product(*(range(s) for s in sizes)) if sizes else [()]
    for index in indices:
        read = (src_offset + sum(i * s for i, s in zip(index, src_strides))) * element_size
        write = (dst_offset + sum(i * s for i, s in zip(index, dst_strides))) * element_size
        if read < 0 or write < 0 or max(read, write) + element_size > 4096:
            raise ValueError('Oracle descriptor exceeds packet extent')
        expected[write:write + element_size] = source[read:read + element_size]
    metadata = bytearray(256)
    metadata[:8] = b'QWCOPY1\0'
    struct.pack_into('<IIqqq', metadata, 8, repeats, len(sizes), element_size, src_offset, dst_offset)
    for offset, values in ((64, sizes), (128, src_strides), (192, dst_strides)):
        struct.pack_into('<8q', metadata, offset, *values, *([0] * (8 - len(values))))
    return bytes(metadata) + source, bytes(expected)


def guard_pairs():
    m32, m63 = (1 << 32) - 1, (1 << 63) - 1
    return [
        (1, 1), (128, 4), (512, 1), (64, 8), (1 << 31, 1), (m32, 1),
        (m32, 2), (m32, m32), (1 << 32, 1), ((1 << 32) + 1, 1),
        (1, m63), (m63, 1), (m63, 2), (m63, m63), (1 << 32, 1 << 31),
        (1 << 32, (1 << 31) - 1), (m32, (1 << 31) + 1),
        ((1 << 31) - 1, 2), (1 << 31, 2), (65535, 65535), (65536, 65536),
        (2, (1 << 32) + 1), ((1 << 32) + 1, 2), (1, 1 << 32),
        (3, m32 // 3), (3, m32 // 3 + 1), (7, m63 // 7), (7, m63 // 7 + 1),
        (1 << 48, 1), (1 << 48, 32767), (1 << 48, 32768), (m32, m32 - 1),
    ]


def guard_packet_and_reference(repeats):
    pairs = guard_pairs()
    payload = b'QWCHK1\0\0' + struct.pack('<I', repeats) + bytes(116)
    payload += b''.join(struct.pack('<qq', a, b) for a, b in pairs)
    expected = bytearray()
    for a, b in pairs:
        product = a * b
        signed_ok, extent_ok = product <= (1 << 63) - 1, product <= (1 << 32) - 1
        expected += struct.pack('<IIqQ', signed_ok, extent_ok, product if signed_ok else -123,
                                product if extent_ok else 123)
    return payload, bytes(expected)


def parse_response(data, repeats, guard=False):
    if len(data) != (128 + 32 * 24 if guard else OUTPUT_BYTES):
        raise ValueError('Actual packet extent differs')
    elapsed, actual, native_size, cycles = struct.unpack_from('<QIIQ', data)
    if not elapsed or actual != repeats or not cycles or any(data[24:128]):
        raise ValueError('Actual timing, repetitions, or reserved bytes differ')
    if native_size != (4 if guard else 0):
        raise ValueError('Actual DSP32 size_t or original packet ABI differs')
    return dict(elapsed_us=elapsed, repeats=actual, processor_cycles=cycles,
                native_size_t_bytes=native_size, effective_active_cycles_per_us=cycles / elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'catalog', 'receipt', 'workdir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('catalog', 'receipt'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--repeats', type=int, default=32768)
    parser.add_argument('--fixed512', action='store_true',
                        help='Explicit accepted-guarded baseline plus block-distinct aligned/unaligned 512-byte experiment')
    args = parser.parse_args()
    if (platform.machine() != 'aarch64' or sys.byteorder != 'little' or
            not args.output.is_absolute() or args.output.exists() or not 1 <= args.repeats <= 100000):
        raise ValueError('Actual ARM and fresh bounded output required')
    if sha(args.catalog) != args.catalog_sha256 or sha(args.receipt) != args.receipt_sha256:
        raise ValueError('Pinned diagnostic inputs changed')
    receipt = json.loads(args.receipt.read_text())
    catalog = json.loads(args.catalog.read_text())
    expected_status = ('fixed512_byte_copy_primitive_and_production_runtime_built_NOT_device_or_model_acceptance' if args.fixed512
                       else 'guarded_byte_copy_primitive_and_production_runtime_built_NOT_device_or_model_acceptance')
    if (receipt['status'] != expected_status
            or receipt['model_inference'] or receipt['FP_evaluation_changed'] or receipt['power_votes_changed']
            or not receipt['test_hooks_absent_from_production_runtime']):
        raise ValueError('Pinned counter-free production runtime required')
    if args.fixed512 and (receipt.get('baseline_kind') != 'actual_accepted_guarded_copy_runtime'
            or receipt.get('never_recursive_memcpy_call') is not True
            or receipt.get('original_guarded_runtime_object_identical') is not True
            or receipt['baseline_production_runtime']['sha256'] != '49d7e1d84745ea68875e83c1cdeccfc5451d88a6eda37ecaaa95fa6d6ff87137'):
        raise ValueError('Exact original accepted guarded runtime and nonrecursive fixed512 wrapper required')
    for row in receipt['sources']:
        if sha(args.receipt.parent / Path(row['path']).name) != row['sha256']:
            raise ValueError('Byte-copy probe source changed')
    libraries = {}
    for row in receipt['rows']:
        if row['returncode']:
            raise ValueError('Diagnostic compilation failed')
        if 'library' in row:
            path = args.receipt.parent / Path(row['library']['path']).name
            if sha(path) != row['library']['sha256']:
                raise ValueError('Diagnostic library changed')
            libraries[row['name'].removeprefix('link-')] = path
    if set(libraries) != {'original', 'candidate', 'candidate-checks'}:
        raise ValueError('Unwrapped ABBA pair and separate integer diagnostic required')
    sys.path[:0] = [str(args.sources / 'models/qwen3/runtime/arm'), str(args.sources)]
    from hexagon_runtime.hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import validate_workdir
    from hexagon_live_compile_mailbox import atomic_json
    validate_workdir(args.workdir)
    active = subprocess.run(['pgrep', '-x', 'buffer_stream'], capture_output=True, timeout=10)
    if active.returncode not in (0, 1) or active.stdout.strip():
        raise ValueError('DSP owner is still active')
    args.output.mkdir(parents=True)
    args.workdir.mkdir()
    (args.workdir / 'rfsa').mkdir()
    control = Path(catalog['control_deployment'])
    for row in catalog['runtime']['artifacts']:
        relative = Path(row['name']) if row['name'] == 'buffer_stream' else Path('rfsa') / row['name']
        if row['name'] not in ('buffer_stream', 'libhexagon_buffer_runner_skel.so') or sha(control / relative) != row['sha256']:
            raise ValueError('Pinned AP/DSP transport changed')
        shutil.copy2(control / relative, args.workdir / relative)
    staged = {}
    for name, path in libraries.items():
        target = args.workdir / (sha(path) + '.so')
        shutil.copy2(path, target)
        if sha(target) != sha(path):
            raise ValueError('Staged diagnostic library changed')
        staged[name] = target
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.workdir), ADSP_LIBRARY_PATH=str(args.workdir)
               + ';' + str(args.workdir / 'rfsa') + ';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    argv = ['flock', '-n', '/tmp/qwen3-qcs9075-probe.lock', str(args.workdir / 'buffer_stream'),
            str(staged['original']), str(INPUT_BYTES), str(OUTPUT_BYTES)]
    record = dict(status='actual_DSP_guarded_copy_running', source_sha256=sha(__file__),
                  receipt_sha256=args.receipt_sha256, catalog_sha256=args.catalog_sha256,
                  FP_evaluation_changed=False, power_votes_changed=False, model_inference_acceptance=False,
                  scope='standalone_descriptor_bytes_and_integer_guards_NOT_full_model_acceptance', rows=[], diagnostic_rows=[])
    if args.fixed512:
        record.update(fixed512=True, source_blocks_distinct=True, baseline_kind='actual_accepted_guarded_copy_runtime')
    def save():
        atomic_json(args.output / 'result.json', record)
    save()
    previous = Path.cwd()
    try:
        os.chdir(args.workdir)
        with HexagonSharedPacketStream(argv, env, args.output, input_bytes=INPUT_BYTES,
                output_bytes=OUTPUT_BYTES, library_path_prefix='/dev/shm/qwen3-', initial_library=str(staged['original']), timeout=120) as stream:
            record['ready'] = stream.start()
            save()
            for case in cases(args.fixed512):
                name = case[0]
                repeats = args.repeats if name.startswith(('contiguous','hot-512')) or name in ('scalar', 'empty') else 1024
                payload, expected = packet_and_reference(case, repeats, args.fixed512)
                (args.output / (name + '-input.bin')).write_bytes(payload)
                (args.output / (name + '-Python-integer-oracle.bin')).write_bytes(expected)
                for index, library in enumerate(('original', 'candidate', 'candidate', 'original')):
                    record['pending'] = dict(case=name, index=index, library=library)
                    save()
                    stream.select_library(str(staged[library]))
                    output, event = stream.invoke_packet(payload, OUTPUT_BYTES)
                    (args.output / (name + '-' + str(index) + '-DSP.bin')).write_bytes(output)
                    timing = parse_response(output, repeats)
                    if output[128:] != expected:
                        raise ValueError('Byte-copy destination or unwritten sentinel differs: ' + name)
                    record['rows'].append(dict(case=name, descriptor=case[1:], index=index, library=library,
                        diagnostic_hooks=False, all_4096_destination_bytes_and_unwritten_sentinel_exact=True,
                        input_sha256=hashlib.sha256(payload).hexdigest(), full_output_sha256=hashlib.sha256(output).hexdigest(),
                        value_sha256=hashlib.sha256(expected).hexdigest(), actual_invoke=event, **timing))
                    save()
            payload, expected = guard_packet_and_reference(1024)
            (args.output / 'integer-checks-input.bin').write_bytes(payload)
            (args.output / 'integer-checks-Python-integer-oracle.bin').write_bytes(expected)
            for index in range(2):
                record['pending'] = dict(case='integer-checks', index=index, library='candidate-checks')
                save()
                stream.select_library(str(staged['candidate-checks']))
                output, event = stream.invoke_packet(payload, 128 + 32 * 24)
                (args.output / ('integer-checks-' + str(index) + '-DSP.bin')).write_bytes(output)
                timing = parse_response(output, 1024, guard=True)
                if output[128:] != expected:
                    raise ValueError('Actual DSP32 integer guard classification or result differs')
                record['diagnostic_rows'].append(dict(index=index, library='candidate-checks', diagnostic_hooks=True,
                    integer_pairs=32, independent_Python_unbounded_integer_oracle_exact=True, actual_invoke=event, **timing))
                save()
        if stream.process_exit_code:
            raise ValueError('DSP owner exit failed')
        record.pop('pending', None)
        record.update(status='actual_DSP_guarded_copy_original_and_independent_bytes_and_DSP32_guard_queries_passed',
                      DSP_calls=len(record['rows']) + len(record['diagnostic_rows']), process_exit_code=stream.process_exit_code,
                      stream_events=stream.events)
        if args.fixed512:
            record['status'] = 'actual_DSP_fixed512_copy_accepted_baseline_distinct_blocks_and_DSP32_guards_passed'
        save()
        os.chdir(previous)
        shutil.rmtree(args.workdir)
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        save()
        raise
    finally:
        os.chdir(previous)
    print(record['status'])


if __name__ == '__main__':
    main()
