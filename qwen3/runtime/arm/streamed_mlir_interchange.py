"""Lossless text interchange across different MLIR bytecode dialect versions.

Print complete IR, without eliding constants, to an explicit compiler process
and a recoverable gzip archive. This is opt-in transport, not a numerical pass.
"""
import gzip
import hashlib
from pathlib import Path
import subprocess
import threading


def stream_mlir_text(operation, command, *, archive_path, log_path, timeout=600):
    result = stream_mlir_pipeline(operation, [command], archive_path=archive_path,
                                  log_path=log_path, timeout=timeout)
    result['command'] = result.pop('commands')[0]
    return result


def stream_mlir_pipeline(operation, commands, *, archive_path, log_path, timeout=600):
    """Stream verified compiler stages without materializing intermediate weights.

    Every stage must succeed. Only stdout is piped to the next stage; compiler
    diagnostics share the separate log. The deadline kills all owned children.
    """
    if timeout<=0:
        raise ValueError('timeout must be positive')
    commands = [list(command) for command in commands]
    if not commands or any(not command for command in commands):
        raise ValueError('At least one explicit compiler command required')
    archive_path=Path(archive_path);log_path=Path(log_path)
    if archive_path.exists() or log_path.exists():
        raise FileExistsError('Refusing to overwrite interchange evidence')
    digest=hashlib.sha256();size=0
    process=None;processes=[];timer=None;expired=threading.Event()
    try:
        with log_path.open('xb') as log,gzip.open(archive_path,'xb',compresslevel=1) as compressed:
            for index, command in enumerate(commands):
                previous = processes[-1] if processes else None
                child = subprocess.Popen(command,
                    stdin=previous.stdout if previous else subprocess.PIPE,
                    stdout=log if index == len(commands) - 1 else subprocess.PIPE,
                    stderr=log)
                processes.append(child)
                if previous is not None:
                    previous.stdout.close()
            process=processes[0]
            def expire():
                expired.set()
                for child in processes:
                    if child.poll() is None:child.kill()
            timer=threading.Timer(timeout,expire);timer.daemon=True;timer.start()
            class Sink:
                def write(self,raw):
                    nonlocal size
                    if not isinstance(raw,bytes):raise TypeError('MLIR printer must use binary=True')
                    digest.update(raw);size+=len(raw)
                    compressed.write(raw);process.stdin.write(raw)
                    return len(raw)
                def flush(self):
                    compressed.flush();process.stdin.flush()
            operation.print(file=Sink(),binary=True)
            process.stdin.close()
            statuses=[child.wait(timeout=timeout) for child in processes]
            if expired.is_set():raise TimeoutError('MLIR text interchange timed out')
            if any(statuses):raise RuntimeError(f'MLIR text interchange failed ({statuses}); inspect {log_path}')
        with gzip.open(archive_path,'rb') as stream:
            recovered=hashlib.file_digest(stream,'sha256').hexdigest()
        if recovered!=digest.hexdigest():raise RuntimeError('Interchange archive failed SHA256 check')
        return {'text_sha256':recovered,'text_bytes':size,'commands':commands,
                'archive_path':str(archive_path),'full_constants_preserved':True}
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('MLIR text interchange timed out') from error
    except BrokenPipeError as error:
        if expired.is_set():raise TimeoutError('MLIR text interchange timed out') from error
        raise RuntimeError(f'Compiler closed MLIR input; inspect {log_path}') from error
    finally:
        if timer is not None:timer.cancel()
        for child in processes:
            if child.poll() is None:child.kill()
        for child in processes:
            child.wait(timeout=30)
            if child.stdin is not None and not child.stdin.closed:
                try:child.stdin.close()
                except BrokenPipeError:pass
            if child.stdout is not None and not child.stdout.closed:
                child.stdout.close()
