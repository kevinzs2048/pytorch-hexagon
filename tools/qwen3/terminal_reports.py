"""Read completed board reports without SSHFS size or content caches."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def read_terminal_report(ssh, remote, snapshot):
    remote = Path(remote)
    if not remote.is_absolute():
        raise ValueError('Absolute board report path required')
    raw = subprocess.check_output([*ssh, shlex.join(['cat', '--', str(remote)])])
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError('Board report must be an object')
    Path(snapshot).write_bytes(raw)
    return report, hashlib.sha256(raw).hexdigest()
