"""Synchronous command transport to one ARM/FastRPC process; no inference loop.

RPC result codes and the process exit status are deliberately separate. Failure
never creates a synthetic successful process exit or uses an earlier response.
"""
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import time


class PersistentRPCError(RuntimeError):
    pass


class HexagonRPCStream:
    def __init__(self, argv, env, directory, *, timeout=120, pass_fds=()):
        self.directory = Path(directory)
        self.stderr = (self.directory/'stream.stderr.txt').open('xb')
        self.stdout = (self.directory/'stream.stdout.txt').open('xb')
        try:
            self.process = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, pass_fds=pass_fds)
        except BaseException:
            self.stdout.close(); self.stderr.close(); raise
        self.buffer = b''; self.timeout = timeout; self.events = []; self.ordinal = 0
        self.ready = False; self.closed = False; self.process_exit_code = None
        self.resources_closed = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None: raise
            error.add_note('RPC process cleanup also failed: '+repr(cleanup_error))
        return False

    def event(self, stage):
        deadline = time.monotonic()+self.timeout
        while True:
            while b'\n' not in self.buffer:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise PersistentRPCError('Timed out waiting for '+stage)
                block = os.read(self.process.stdout.fileno(), 4096)
                if not block: raise PersistentRPCError('ARM process ended before '+stage)
                self.stdout.write(block); self.stdout.flush(); self.buffer += block
                if len(self.buffer) > 65536: raise PersistentRPCError('Unbounded protocol line')
            line, self.buffer = self.buffer.split(b'\n', 1)
            if not line.startswith(b'{'): continue  # SDK diagnostics remain in the raw log.
            event = json.loads(line)
            self.events.append(event)
            if event.get('stage') != stage: raise PersistentRPCError('Unexpected protocol stage: '+str(event))
            if stage != 'ready':
                seconds = event.get('seconds')
                if (type(event.get('rpc_result')) is not int or event['rpc_result'] != 0
                        or type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0):
                    raise PersistentRPCError('RPC failed or malformed: '+str(event))
            if event.get('loader_result', 0) != 0 or event.get('DSP_result', 0) != 0:
                raise PersistentRPCError('DSP load/entry failed: '+str(event))
            return event

    def start(self):
        self.event('open'); self.event('load')
        event = self.event('ready')
        if type(event.get('ARM_pid')) is not int or event['ARM_pid'] <= 0:
            raise PersistentRPCError('Missing ARM process identity')
        self.ready = True
        return event

    def invoke(self):
        return self._invoke(b'invoke\n')

    def _invoke(self, command):
        if (command not in (b'invoke\n', b'invoke-reuse\n')
                and re.fullmatch(rb'invoke [1-9][0-9]* [1-9][0-9]*\n', command) is None):
            raise PersistentRPCError('Unknown explicit invocation command')
        if not self.ready or self.closed: raise PersistentRPCError('Stream not ready')
        self.ordinal += 1
        self.process.stdin.write(command); self.process.stdin.flush()
        event = self.event('invoke')
        if event.get('ordinal') != self.ordinal: raise PersistentRPCError('Stale/out-of-order response')
        return event

    def _consume_close_events(self):
        self.event('unload'); self.event('close')

    def close(self):
        if self.resources_closed: return  # Idempotent cleanup, not a success claim.
        try:
            if self.ready:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.write(b'close\n'); self.process.stdin.flush()
                    except BrokenPipeError:
                        pass  # An entry failure may already have triggered cleanup.
                self._consume_close_events()
            self.process_exit_code = self.process.wait(timeout=self.timeout)
            if self.process_exit_code != 0:
                raise PersistentRPCError('ARM process exit='+str(self.process_exit_code))
            self.closed = True
        finally:
            if self.process.poll() is None:
                # Local transport termination is not evidence of clean DSP release.
                self.process.terminate()
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=10)
            if self.process_exit_code is None: self.process_exit_code = self.process.returncode
            # Retain cleanup/error output even when start/invoke failed before
            # all protocol events could be consumed. Never synthesize it.
            while select.select([self.process.stdout], [], [], 0)[0]:
                block = os.read(self.process.stdout.fileno(), 4096)
                if not block: break
                self.stdout.write(block)
            self.stdout.close(); self.stderr.close()
            self.process.stdin.close(); self.process.stdout.close()
            self.resources_closed = True
