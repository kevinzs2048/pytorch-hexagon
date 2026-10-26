"""Hash-bound relocatable two-shape deployment, not a general model compiler.

The manifest digest must come from the trusted exporter, not be discovered from
the bundle being checked. Hashes provide integrity, not a signature/sandbox.
Original compiler records retain provenance paths; execution uses relative
members only. No graph mismatch may select CPU execution or another artifact.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def member(root, relative):
    if type(relative) is not str or not relative or any(c in relative for c in '\x00\n\r\\'):
        raise ValueError('Explicit relative bundle member required')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or str(path) != relative or relative == '.':
        raise ValueError('Noncanonical or escaping bundle member')
    result = Path(root) / relative
    if Path(root).resolve() not in result.resolve().parents:
        raise ValueError('Bundle symlink escapes root')
    if not result.is_file(): raise ValueError('Bundle member is not a regular file')
    return result


class ArmBundle:
    def __init__(self, root, expected_sha256):
        self.root = Path(root).resolve()
        self.sha256 = sha(self.root / 'manifest.json')
        if self.sha256 != expected_sha256: raise ValueError('Deployment manifest digest differs')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if m.get('status') != 'ARM_pair256_bundle_prepared_not_executed' or m.get('schema') != 1:
            raise ValueError('Unsupported deployment manifest')
        self.files = {}
        for row in m['files']:
            if row['path'] in self.files: raise ValueError('Duplicate deployment member')
            if sha(member(self.root, row['path'])) != row['sha256']:
                raise ValueError('Deployment file bytes changed: ' + row['path'])
            self.files[row['path']] = row['sha256']
        self.captures = {n: self.read('single_capture.json' if n == 1 else 'pair_capture.json') for n in (1, 2)}
        compiled = self.read('compiled-storage.json')
        self.storage = {g['query_tokens']: g for g in self.read('storage-capture.json')['graphs']}
        self.updates = {g['query_tokens']: g['input_update_outputs'] for g in compiled['graphs']}
        if set(self.storage) != {1, 2} or set(self.updates) != {1, 2}:
            raise ValueError('Exact single/pair compiler inputs required')
        audit = self.read('provenance/boundary-audit.json')
        producer = self.read('provenance/boundary-result.json')
        if (audit.get('status') != 'native_frozen_pair256_cached_generate_independently_bitexact'
                or audit.get('result_sha256') != self.files['provenance/boundary-result.json']
                or audit.get('DSP_calls') != 275 or audit.get('original_native_generate_replayed') is not True):
            raise ValueError('Independent prior native DSP acceptance required')
        for key, relative in (('context', 'context.json'), ('single_capture', 'single_capture.json'),
                ('pair_capture', 'pair_capture.json'), ('runtime', 'provenance/runtime.json'),
                ('linked', 'provenance/linked.json'), ('dispatch_control', 'reference.json')):
            if producer['bindings'][key]['sha256'] != self.files[relative]:
                raise ValueError('Accepted producer binding differs: ' + key)
        linked = self.read('provenance/linked.json')
        if (linked['compiled']['sha256'] != self.files['compiled-storage.json']
                or compiled['capture']['sha256'] != self.files['storage-capture.json']
                or compiled['context']['sha256'] != self.files['context.json']):
            raise ValueError('Compiler input provenance chain differs')
        original = ([linked['dispatches']['native']] + [g['library'] for g in linked['graphs']]
                    + linked['constants'])
        if (len(original) != 5 or len(linked['constants']) != 2
                or linked['dispatches']['native']['layout']['diagnostic_header'] is not False
                or m['constant_partitions'] != 2 or m['layouts'] != linked['layouts']
                or [r['sha256'] for r in original] != [r['sha256'] for r in m['libraries']]):
            raise ValueError('Deployed model library identity/order changed')
        runtime = self.read('provenance/runtime.json')
        relocated = dict(runtime, artifacts=m['runtime']['artifacts'])
        if (relocated != m['runtime'] or [r['sha256'] for r in runtime['artifacts']]
                != [r['sha256'] for r in m['runtime']['artifacts']]):
            raise ValueError('Deployed runtime differs from accepted runtime')
        for row in m['libraries'] + m['runtime']['artifacts']: self.artifact(row)
        serializers = [[r['sha256'] for r in g['sources'] if Path(r['path']).name == 'hexagon_buffer_io.py']
                       for g in compiled['graphs']]
        if serializers != [[m['buffer_generator_sha256']]] * 2:
            raise ValueError('Native serializer does not match compiler sources')
        if self.files['runtime/hexagon_buffer_io.py'] != m['buffer_generator_sha256']:
            raise ValueError('Native serializer deployment changed')
        policy = m['decomposition_policy']
        if self.files[policy['path']] != policy['sha256']: raise ValueError('Decomposition policy changed')
        if m.get('natural_reference_included'):
            natural_audit = self.read('provenance/natural-audit.json')
            natural_run = self.read('provenance/natural-result.json')
            if (natural_audit.get('status') != 'native_frozen_pair256_cached_natural_independently_bitexact'
                    or natural_audit.get('result_sha256') != self.files['provenance/natural-result.json']
                    or natural_audit.get('DSP_calls') != 383
                    or natural_audit.get('original_native_generate_replayed') is not True
                    or natural_run['bindings']['natural_reference']['sha256'] != self.files['natural-reference.json']):
                raise ValueError('Independent prior natural DSP acceptance required')
            for key in ('context', 'single_capture', 'pair_capture', 'runtime', 'linked', 'dispatch_control'):
                if natural_run['bindings'][key] != producer['bindings'][key]:
                    raise ValueError('Natural acceptance belongs to a different model/runtime')

    def read(self, relative):
        if relative not in self.files: raise ValueError('Unbound JSON member')
        return json.loads(member(self.root, relative).read_text())

    def artifact(self, record):
        if self.files.get(record['path']) != record['sha256']:
            raise ValueError('Artifact not bound by deployment manifest')
        return dict(record, path=str(member(self.root, record['path'])))

    def rotary_control(self, n):
        if n not in (1, 2): raise ValueError('Unsupported shape')
        result = dict(self.captures[n]['rotary_CPU_binding' if n == 1 else 'CPU_rotary_control'])
        relative = f'rotary-{n}/result.json'
        if self.files[relative] != result['control_sha256']: raise ValueError('Rotary metadata changed')
        result['control_path'] = str(member(self.root, relative))
        return result
