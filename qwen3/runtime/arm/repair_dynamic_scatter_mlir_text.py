"""Translate the one known torch-mlir scatter property syntax for Hexagon's reader.

The source is the pinned complete text archive from the verified bound q17 graph.
Only generic tm_tensor.scatter's property representation is changed: the new
dimension_map array becomes the old dialect's dimension_map attribute, with
unique_indices=false (the conservative setting ignored by our serial lowering).
No tensor values or operations are added or removed.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import traceback


PREFIX = b'"tm_tensor.scatter"('
SCATTER = re.compile(
    rb'"tm_tensor\.scatter"\(([^()\n]{1,256})\) '
    rb'<array<i64: ([0-9, ]{1,256})>> \(\{')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ScatterPropertyRepair:
    def __init__(self, emit):
        self.emit = emit
        self.pending = b''
        self.count = 0

    def write(self, data):
        self.pending += data
        while self.pending:
            start = self.pending.find(PREFIX)
            if start < 0:
                keep = len(PREFIX) - 1
                if len(self.pending) > keep:
                    self.emit(self.pending[:-keep])
                    self.pending = self.pending[-keep:]
                break
            if start:
                self.emit(self.pending[:start])
                self.pending = self.pending[start:]
            match = SCATTER.match(self.pending)
            if match is None:
                if len(self.pending) > 1024:
                    raise ValueError('Unexpected tm_tensor.scatter property syntax')
                break
            replacement = (PREFIX + match.group(1) + b') '
                + b'<{dimension_map = array<i64: ' + match.group(2)
                + b'>, unique_indices = false}> ({')
            self.emit(replacement)
            self.pending = self.pending[match.end():]
            self.count += 1

    def finish(self):
        if PREFIX in self.pending:
            raise ValueError('Incomplete tm_tensor.scatter property syntax')
        if self.pending:
            self.emit(self.pending)
            self.pending = b''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source_archive', 'optimizer'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
        parser.add_argument('--' + key.replace('_', '-') + '-sha256', required=True)
    parser.add_argument('--bytecode-output', type=Path, required=True)
    parser.add_argument('--record-output', type=Path, required=True)
    args = parser.parse_args()
    if args.bytecode_output.exists() or args.record_output.exists():
        raise FileExistsError('Scoped reader outputs must not exist')
    for key in ('source_archive', 'optimizer'):
        if sha(getattr(args, key)) != getattr(args, key + '_sha256'):
            raise ValueError('Pinned reader input drift: ' + key)
    if shutil.disk_usage(args.bytecode_output.parent).free < 1536 * 1024**2:
        raise ValueError('At least 1.5 GiB output space required')
    args.record_output.mkdir(exist_ok=False)
    result = dict(status='admitting_scatter_property_reader_compatibility',
        source_sha256=sha(__file__), source_archive=dict(path=str(args.source_archive),
            sha256=args.source_archive_sha256), optimizer=dict(path=str(args.optimizer),
            sha256=args.optimizer_sha256), DSP_calls=0, machine_code_compilations=0,
        model_execution=False)

    def save():
        (args.record_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    process = None
    timer = None
    expired = threading.Event()
    try:
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        source_bytes = output_bytes = 0
        with (args.record_output / 'hexagon-reader.log').open('xb') as log:
            process = subprocess.Popen([str(args.optimizer), '-', '--verify-each',
                '--emit-bytecode', '-o', str(args.bytecode_output)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)

            def expire():
                expired.set()
                if process.poll() is None:
                    process.kill()

            timer = threading.Timer(1800, expire)
            timer.daemon = True
            timer.start()

            def emit(data):
                nonlocal output_bytes
                process.stdin.write(data)
                output_digest.update(data)
                output_bytes += len(data)

            repair = ScatterPropertyRepair(emit)
            with gzip.open(args.source_archive, 'rb') as source:
                while chunk := source.read(1024 * 1024):
                    source_digest.update(chunk)
                    source_bytes += len(chunk)
                    repair.write(chunk)
            repair.finish()
            if repair.count != 56:
                raise ValueError(f'Expected 56 Qwen KV scatters, got {repair.count}')
            process.stdin.close()
            status = process.wait(timeout=1800)
            if expired.is_set() or status:
                raise RuntimeError(f'Hexagon reader failed: status={status}, timed_out={expired.is_set()}')
        result['interchange'] = dict(source_text_sha256=source_digest.hexdigest(),
            source_text_bytes=source_bytes, converted_text_sha256=output_digest.hexdigest(),
            converted_text_bytes=output_bytes, scatter_properties_rewritten=repair.count,
            tensor_constant_bytes_untouched=True)
        with args.bytecode_output.open('rb') as stream:
            if stream.read(4) != b'ML\xefR':
                raise ValueError('Hexagon reader did not emit bytecode')
        result.update(status='bound_Linalg_scatter_compatible_Hexagon_reader_NOT_machine_code_or_DSP',
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
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
