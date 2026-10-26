"""Bounded reported model progress for a trusted single-run compiler mailbox.

Liveness only, NOT evidence of model accuracy or successful DSP execution.
Only a strictly advancing completed-forward count renews the idle deadline;
re-reading or rewriting the same record does not keep a stalled session alive.
"""
import json
from pathlib import Path

from hexagon_live_compile_mailbox import atomic_json


def progress_record(completed_forwards, requests_submitted):
    if (type(completed_forwards) is not int or not 0 <= completed_forwards <= 4096
            or type(requests_submitted) is not int or not 0 <= requests_submitted <= 4):
        raise ValueError('Bounded actual forward/request counters required')
    return dict(schema=1, completed_forwards=completed_forwards, requests_submitted=requests_submitted)


def publish_progress(mailbox, completed_forwards, requests_submitted):
    record = progress_record(completed_forwards, requests_submitted)
    atomic_json(Path(mailbox)/'session-progress.json', record)


class SessionProgress:
    def __init__(self):
        self.completed_forwards = 0
        self.requests_submitted = 0

    def observe(self, mailbox):
        path = Path(mailbox)/'session-progress.json'
        if path.is_symlink():
            raise ValueError('Progress marker must not be a symlink')
        if not path.exists():
            return False  # Backward compatibility with clients without telemetry.
        if path.stat().st_size > 16384:
            raise ValueError('Bounded progress marker required')
        record = json.loads(path.read_text())
        if (type(record) is not dict or type(record.get('schema')) is not int or record['schema'] != 1
                or set(record) != {'schema', 'completed_forwards', 'requests_submitted'}):
            raise ValueError('Explicit compiler-session progress schema required')
        progress_record(record['completed_forwards'], record['requests_submitted'])
        forwards, requests = record['completed_forwards'], record['requests_submitted']
        if forwards < self.completed_forwards or requests < self.requests_submitted:
            raise ValueError('Reported model progress went backwards')
        advanced = forwards > self.completed_forwards
        self.completed_forwards, self.requests_submitted = forwards, requests
        return advanced
