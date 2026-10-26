"""Return a bounded verified DSO, without exposing the compiler cache to ARM.

This is a trusted local experiment, not an authenticated multi-user transport.
The compiler result and actual capture still provide the artifact binding.
"""
import json
from pathlib import Path
import shutil

from capture_qwen_symbolic_graph_carrier import sha
from hexagon_live_compile_mailbox import checked_library_path

RELATIVE = 'compiled/returned-library.so'
MAX_BYTES = 64 * 1024**2


def reply_path(directory):
    root = Path(directory)
    parent = root/'compiled'
    if root.is_symlink() or parent.is_symlink() or not parent.is_dir():
        raise ValueError('Private real compiler reply directory required')
    return root/RELATIVE


def publish_library(directory, entry, machine_cache):
    """Called before the atomic successful response, never rewrite its result."""
    library = entry['library']
    source = checked_library_path(machine_cache, library['relative_to_machine_cache'],
                                  library['sha256'])
    size = source.stat().st_size
    if not 0 < size <= MAX_BYTES:
        raise ValueError('Bounded compiled library required')
    target = reply_path(directory)
    if target.exists() or target.is_symlink():
        raise ValueError('Fresh returned library required')
    temporary = target.with_suffix('.so.pending')
    with source.open('rb') as reader, temporary.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=1024**2)
    if temporary.stat().st_size != size or sha(temporary) != library['sha256']:
        raise ValueError('Returned library changed during copy')
    temporary.replace(target)
    return dict(path=RELATIVE, bytes=size, sha256=library['sha256'])


def checked_reply_library(directory, entry, *, response=None):
    """ARM only reads this request's reply; no compiler filesystem path needed."""
    if response is None:
        path = Path(directory)/'response.json'
        if path.is_symlink() or path.stat().st_size > 16384:
            raise ValueError('Bounded compiler response required')
        response = json.loads(path.read_text())
    item = response.get('returned_library', {})
    if (response.get('status') != 'compiled' or item.get('path') != RELATIVE
            or type(item.get('bytes')) is not int or not 0 < item['bytes'] <= MAX_BYTES
            or item.get('sha256') != entry['library']['sha256']):
        raise ValueError('Returned library is not bound to the compiled entry')
    target = reply_path(directory)
    if (target.is_symlink() or not target.is_file()
            or target.stat().st_size != item['bytes'] or sha(target) != item['sha256']):
        raise ValueError('Actual returned library bytes changed')
    return target
