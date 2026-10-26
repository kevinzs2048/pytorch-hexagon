"""Bounded foreground compiler worker for one explicitly trusted ARM run.

No sockets, remote shell commands or graph recapture. Fixed compiler/configuration
come from local CLI arguments, never from request contents. Pickle is only for
this user's trusted board and pinned PyTorch build. Not a multi-user service.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from capture_qwen_symbolic_graph_carrier import sha
from hexagon_live_compile_mailbox import atomic_json
from hexagon_compile_library_reply import publish_library
from hexagon_compile_session_progress import SessionProgress


def run_compiler(command, log, timeout):
    """Cancel only this transaction's process group, including LLVM children."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
        if status:
            raise subprocess.CalledProcessError(status, command)
    except BaseException:
        # Killing only the Python driver can orphan its optimizer/compiler
        # children. The new session above makes this group exclusively ours.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        raise


def session_closed(mailbox, completed):
    marker = Path(mailbox)/'session-closed.json'
    if not marker.exists():
        return False
    if marker.is_symlink() or marker.stat().st_size > 16384:
        raise ValueError('Bounded session completion marker required')
    record = json.loads(marker.read_text())
    if (record.get('schema') != 1 or type(record.get('requests_submitted')) is not int
            or record['requests_submitted'] != completed
            or record.get('status') != 'ARM_native_dynamic_compiler_handoff_generate_completed'
            or (Path(mailbox)/f'capture-{completed:02d}'/'ready.json').exists()):
        raise ValueError('ARM run failed or compiler request accounting differs at completion')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mailbox', type=Path, required=True)
    parser.add_argument('--trusted-local-producer', action='store_true')
    lifetime = parser.add_mutually_exclusive_group(required=True)
    lifetime.add_argument('--requests', type=int, choices=range(1, 5),
                          help='Fixed-count diagnostic mode only')
    lifetime.add_argument('--until-session-close', action='store_true',
                          help='Follow actual framework capture requests; no predicted graph count')
    parser.add_argument('--timeout', type=int, default=1800)
    for name in ('context', 'catalog', 'frontend-cache', 'machine-cache', 'machine-recipe'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('catalog-sha256', 'machine-recipe-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if (not args.trusted_local_producer or not args.mailbox.is_absolute()
            or args.mailbox.is_symlink() or not args.mailbox.is_dir()
            or not 1 <= args.timeout <= 3600):
        parser.error('Explicit trusted private existing mailbox and bounded timeout required')
    limit = args.requests if args.requests else 4
    progress = SessionProgress()
    for index in range(limit+int(args.until_session_close)):
        folder = args.mailbox/f'capture-{index:02d}'
        ready = folder/'ready.json'
        deadline = time.monotonic()+args.timeout
        while True:
            if args.until_session_close and session_closed(args.mailbox, index):
                progress.observe(args.mailbox)
                print(f'ARM generation closed after {index} actual compiler requests; '
                      f'reported completed forwards={progress.completed_forwards}', flush=True)
                return
            if ready.exists():
                break
            if args.until_session_close and progress.observe(args.mailbox):
                deadline = time.monotonic()+args.timeout
            if time.monotonic() >= deadline:
                raise TimeoutError('No next ARM compilation request within deadline')
            time.sleep(0.25)
        if (folder.is_symlink() or ready.is_symlink() or ready.stat().st_size > 16384
                or (folder/'response.json').exists() or (folder/'compiled').exists()):
            raise ValueError('Fresh scoped compiler request required')
        admission = json.loads(ready.read_text())
        producer_sha = admission['producer_sha256']
        if admission['schema'] != 1 or sha(folder/'result.json') != producer_sha:
            raise ValueError('Published producer changed')
        response = dict(schema=1, producer_sha256=producer_sha, status='failed')
        try:
            if index >= limit:
                raise ValueError('Selected runtime admits at most four compiled graph entries')
            command = [sys.executable, str(Path(__file__).with_name('compile_qwen_symbolic_carrier.py')),
                '--producer', str(folder/'result.json'), '--producer-sha256', producer_sha,
                '--trusted-local-producer', '--output', str(folder/'compiled')]
            for name in ('context', 'catalog', 'catalog-sha256', 'frontend-cache',
                         'machine-cache', 'machine-recipe', 'machine-recipe-sha256'):
                command.extend(['--'+name, str(getattr(args, name.replace('-', '_')))])
            print(f'Compiling actual ARM capture {index}', flush=True)
            with (folder/'compiler.log').open('x') as log:
                run_compiler(command, log, args.timeout)
            result = folder/'compiled'/'result.json'
            compiled = json.loads(result.read_text())
            returned = publish_library(folder, compiled['compiled_entry'], args.machine_cache)
            response.update(status='compiled', result_sha256=sha(result), returned_library=returned)
        except Exception as error:
            response['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            atomic_json(folder/'response.json', response)
        print(f'Returned actual ARM capture {index}', flush=True)


if __name__ == '__main__':
    main()
