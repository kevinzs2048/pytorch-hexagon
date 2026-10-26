"""Read an explicitly SHA-bound archived compiler input without materializing it."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
from hexagon_generate_reuse import sha


@contextmanager
def verified_graph_file(target, expected, archive=None):
    """Expose exact compiled graph bytes to a file-only parser; RAM is temporary."""
    if archive is None:
        if sha(target) != expected:
            raise ValueError('Compiled graph bytes drifted')
        yield str(target)
        return
    command = archived_graph_command(archive, target, expected)
    fd = os.memfd_create('verified-archived-hexagon-graph', os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, timeout=180)
        path = '/proc/self/fd/' + str(fd)
        if result.returncode or sha(path) != expected:
            raise ValueError('In-memory restored graph identity differs')
        yield path
    finally:
        os.close(fd)


def archived_graph_command(manifest, target, expected):
    record = json.loads(Path(manifest).read_text())
    if record.get('status') != 'complete':
        raise ValueError('Archive incomplete')
    matches = [x for x in record['files'] if Path(x['original_path']) == Path(target)]
    if len(matches) != 1:
        raise ValueError('Archive does not uniquely bind the requested graph')
    item = matches[0]
    if (item['original_sha256'] != expected or not item['restore_sha256_verified']
            or not item['original_removed']):
        raise ValueError('Archive does not bind the original compiled graph')
    for path, digest in ((item['reference_path'], item['reference_sha256']),
                         (item['patch_path'], item['patch_sha256'])):
        if sha(path) != digest:
            raise ValueError('Archive dictionary or patch drift')
    command = ['zstd', '-q', '-d', '--memory=2048MB', '--patch-from=' + item['reference_path'], item['patch_path'], '-c']
    digest = hashlib.sha256(); size = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        while raw := process.stdout.read(8 * 1024**2):
            digest.update(raw); size += len(raw)
        error = process.stderr.read()
        if process.wait(timeout=180):
            raise ValueError('Archived graph decoding failed: ' + error.decode(errors='replace'))
    if digest.hexdigest() != expected or size != item['original_bytes']:
        raise ValueError('Restored compiler input identity differs')
    return command


@contextmanager
def print_graph(optimizer, graph, flags, stderr, archive_command=None):
    decoder = subprocess.Popen(archive_command, stdout=subprocess.PIPE, stderr=stderr) if archive_command else None
    printer = subprocess.Popen([optimizer, '-' if decoder else str(graph), *flags],
                               stdin=decoder.stdout if decoder else None, stdout=subprocess.PIPE, stderr=stderr)
    if decoder:
        decoder.stdout.close()
    try:
        yield printer
        if printer.wait(timeout=180) != 0:
            raise ValueError('Graph printing failed')
        if decoder and decoder.wait(timeout=180) != 0:
            raise ValueError('Archive decoding failed during graph printing')
    finally:
        for process in (printer, decoder):
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=30)
