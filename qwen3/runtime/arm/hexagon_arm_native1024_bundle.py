"""Relocatable capacity1024 deployment admission, separate from capacity256.

An external trusted manifest digest is mandatory. Compiler provenance keeps
original paths; execution resolves only explicitly bound relative members.
"""
import importlib.util
import json
from pathlib import Path

from hexagon_arm_bundle import ArmBundle, member, sha
from hexagon_native1024_cache_binding import check_graph_binding
from hexagon_extended_kv_layout import ExtendedKVLayout
from hexagon_arm_native1024_admission import admit_proofs


class ArmNative1024Bundle(ArmBundle):
    def __init__(self, root, expected_sha256):
        self.root = Path(root).resolve()
        self.sha256 = sha(self.root / 'manifest.json')
        if self.sha256 != expected_sha256:
            raise ValueError('Deployment manifest digest differs')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if (m.get('status') != 'ARM_native1024_bundle_prepared_not_executed'
                or m.get('schema') != 1 or m.get('capacity') != 1024
                or m.get('compiler_binaries_included') is not False):
            raise ValueError('Explicit compiler-free capacity1024 bundle required')
        self.files = {}
        for row in m['files']:
            if row['path'] in self.files or sha(member(self.root, row['path'])) != row['sha256']:
                raise ValueError('Duplicate or changed deployment member')
            self.files[row['path']] = row['sha256']
        fixed=self.read('provenance/fixed-result.json')
        fixed_audit=self.read('provenance/fixed-audit.json')
        wire=self.read('provenance/wire-result.json')
        wire_audit=self.read('provenance/wire-audit.json')
        runtime=self.read('provenance/runtime.json')
        digests={name:self.files[relative] for name,relative in (
            ('fixed_result','provenance/fixed-result.json'),('fixed_audit','provenance/fixed-audit.json'),
            ('fixed_auditor','provenance/fixed-auditor.py'),('wire_result','provenance/wire-result.json'),
            ('wire_auditor','provenance/wire-auditor.py'),('packet_auditor','provenance/packet-auditor.py'),
            ('linked','provenance/linked.json'),('runtime','provenance/runtime.json'))}
        self.admission=admit_proofs(fixed,fixed_audit,wire,wire_audit,runtime,digests=digests)
        inputs=self.read('provenance/capture-inputs.json')
        if (inputs.get('status')!='ARM_capacity1024_capture_inputs_prepared_not_executed'
                or inputs.get('pre_execution_capture_only') is not True
                or inputs.get('model_runtime_admitted') is not False):
            raise ValueError('Separate actual ARM capture preparation required')
        linked = self.read('provenance/linked.json')
        if (linked.get('capacity') != 1024 or linked.get('model_arithmetic_objects_unchanged') is not True
                or linked.get('status') != 'native1024_shared_entries_linked_NOT_executed'
                or len(linked['compiled']) != 2):
            raise ValueError('Original complete capacity1024 linked model required')
        codec = member(self.root, 'runtime/hexagon_buffer_io.py')
        if self.files['runtime/hexagon_buffer_io.py'] != m['buffer_generator_sha256']:
            raise ValueError('Native serializer deployment changed')
        spec = importlib.util.spec_from_file_location('_arm_native1024_buffer_io', codec)
        bio = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bio)
        self.captures, self.compiled = {}, {}
        for n in (1, 2):
            compiled = self.read(f'compiled-{n}.json')
            capture = self.read('single_capture.json' if n == 1 else 'pair_capture.json')
            if (linked['compiled'][n-1]['sha256'] != self.files[f'compiled-{n}.json']
                    or compiled['bindings']['capture']['sha256'] != self.files['single_capture.json' if n == 1 else 'pair_capture.json']
                    or compiled['bindings']['context']['sha256'] != self.files['context.json']):
                raise ValueError('Complete original compiled graph provenance differs')
            layout = bio.make_layout(*ExtendedKVLayout(1024, query_tokens=n).signature())
            check_graph_binding(compiled, capture, context=compiled['bindings']['context'], query_tokens=n, layout=layout)
            if m['layouts'].get(str(n)) != layout or linked['layouts'].get(str(n)) != layout:
                raise ValueError('Relocated native ABI differs from compiler input')
            serializers = [r['sha256'] for r in compiled['sources'] if Path(r['path']).name == 'hexagon_buffer_io.py']
            if serializers != [m['buffer_generator_sha256']]:
                raise ValueError('Native serializer differs from compiled source')
            if capture['rotary_control']['sha256'] != self.files['rotary/result.json']:
                raise ValueError('Original position-table metadata changed')
            arm=self.read(f'provenance/arm-shape-{n}.json')
            graph=capture['graphs'][0]
            if (arm.get('status')!='ARM_Qwen3_capacity1024_shape_matches_compiled_input_not_executed'
                    or arm.get('input_manifest_sha256')!=self.files['provenance/capture-inputs.json']
                    or arm.get('capacity')!=1024 or arm.get('query_tokens')!=n
                    or arm.get('expected_pre_execution_stop') is not True
                    or arm.get('full_model_executed') is not False or arm.get('DSP_executed') is not False
                    or arm.get('CPU_fallback') is not False or len(arm.get('graphs',[]))!=1):
                raise ValueError('Actual ARM stop-before-execution graph compatibility required')
            actual=arm['graphs'][0]
            if (actual['parameters']!=graph['parameters'] or actual['storage']!=graph['storage']
                    or actual['native_aten_sha256']!=graph['aten_sha256']
                    or actual['packed_aten_sha256']!=graph['packed_aten_sha256']
                    or self.files[f'provenance/arm-native-{n}.py']!=graph['aten_sha256']
                    or self.files[f'provenance/arm-packed-{n}.py']!=graph['packed_aten_sha256']):
                raise ValueError('ARM graph/parameters differ from actual compiled input')
            self.captures[n], self.compiled[n] = capture, compiled
        if (self.compiled[1]['parameters'] != self.compiled[2]['parameters']
                or self.compiled[1]['storage'] != self.compiled[2]['storage']):
            raise ValueError('Both shapes must share identical original weights and storage policy')
        selected = [linked['dispatches']['native']] + [g['library'] for g in linked['graphs']] + linked['constants']
        if (len(selected) != 5 or len(linked['constants']) != 2 or m['constant_partitions'] != 2
                or linked['dispatches']['native']['layout']['diagnostic_header'] is not False
                or [r['sha256'] for r in selected] != [r['sha256'] for r in m['libraries']]
                or [Path(r['path']).name for r in selected] != [Path(r['path']).name for r in m['libraries']]):
            raise ValueError('Relocated native library closure differs')
        runtime = self.read('provenance/runtime.json')
        if (dict(runtime, artifacts=m['runtime']['artifacts']) != m['runtime']
                or [r['sha256'] for r in runtime['artifacts']] != [r['sha256'] for r in m['runtime']['artifacts']]
                or runtime.get('resident_KV_protocol') != 'native-generate-pair-capacity1024-v4'):
            raise ValueError('Relocated actual version4 runtime differs')
        for row in m['libraries'] + m['runtime']['artifacts']:
            self.artifact(row)
        policy = m['decomposition_policy']
        if self.files[policy['path']] != policy['sha256']:
            raise ValueError('Original decomposition policy differs')

    def native_capture(self, n):
        if type(n) is not int or n not in (1, 2):
            raise ValueError('Explicit native shape required')
        result = dict(self.captures[n])
        result['rotary_control'] = dict(result['rotary_control'],
            path=str(member(self.root, 'rotary/result.json')))
        return result
