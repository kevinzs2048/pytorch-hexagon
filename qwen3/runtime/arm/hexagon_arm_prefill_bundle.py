"""Small isolated prefill extension of the accepted ARM bundle.

Old checkpoint and framework deployment are verified read-only through the
base bundle. New libraries/sources are separate relative members in this root.
No old manifest is rewritten; no generic model/backend support is implied.
"""
import importlib.util
import json
from pathlib import Path

from hexagon_arm_bundle import ArmBundle,member,sha
from hexagon_arm_native1024_bundle import ArmNative1024Bundle
from hexagon_prefill1024_layout_candidate import Prefill1024LayoutCandidate
from hexagon_prefill_cache_binding import check_graph_binding


class ArmPrefillBundle(ArmBundle):
    def __init__(self,root,expected_sha256,*,base_root=None):
        self.root=Path(root).resolve();self.sha256=sha(self.root/'manifest.json')
        if self.sha256!=expected_sha256:raise ValueError('Trusted prefill manifest differs')
        self.manifest=json.loads((self.root/'manifest.json').read_text());m=self.manifest
        if (m.get('schema')!=1 or m.get('status')!='ARM_prefill4_bundle_prepared_NOT_generate'
                or m.get('query_tokens')!=[1,2,3,4] or m.get('capacity')!=1024
                or m.get('compiler_binaries_included') is not False):
            raise ValueError('Explicit complete prefill bundle required')
        self.files={}
        for row in m['files']:
            if row['path'] in self.files or sha(member(self.root,row['path']))!=row['sha256']:
                raise ValueError('Duplicate or changed deployment file')
            self.files[row['path']]=row['sha256']
        # Exporter may verify the identical base through a host-side mount.
        # The board driver supplies no override; the trusted digest is identical.
        self.base=ArmNative1024Bundle(base_root or m['base']['root'],m['base']['sha256'])
        if self.files['context.json']!=self.base.files['context.json']:
            raise ValueError('Original model context differs')
        linked=self.read('provenance/linked.json')
        if (linked.get('status')!='prefill_shared_entries_linked_NOT_executed'
                or linked.get('query_tokens')!=[1,2,3,4]
                or linked.get('model_arithmetic_objects_unchanged') is not True):
            raise ValueError('Actual four-shape shared linkage required')
        wire=self.read('provenance/wire-result.json');proof=self.read('provenance/wire-audit.json')
        runtime=self.read('provenance/runtime.json')
        if (proof.get('status')!='prefill1024_variable_KV_synthetic_DSP_independently_audited_not_model'
                or proof.get('result_sha256')!=self.files['provenance/wire-result.json']
                or proof.get('successful_DSP_calls')!=12 or proof.get('expected_failed_DSP_calls')!=5
                or wire['runtime']['sha256']!=self.files['provenance/runtime.json']
                or runtime.get('resident_KV_protocol')!='native-generate-prefill-capacity1024-v5'
                or runtime.get('query_tokens')!=[1,2,3,4,8]):
            raise ValueError('Independently verified actual v5 runtime required')
        codec=member(self.root,'runtime/hexagon_buffer_io.py')
        if self.files['runtime/hexagon_buffer_io.py']!=m['buffer_generator_sha256']:
            raise ValueError('Native serializer differs')
        spec=importlib.util.spec_from_file_location('_arm_prefill_buffer_io',codec)
        bio=importlib.util.module_from_spec(spec);spec.loader.exec_module(bio)
        self.captures={};self.compiled={}
        for n in (1,2,3,4):
            compiled=self.read(f'compiled-{n}.json');capture=self.read(f'capture-{n}.json')
            if (linked['compiled'][n-1]['sha256']!=self.files[f'compiled-{n}.json']
                    or compiled['bindings']['capture']['sha256']!=self.files[f'capture-{n}.json']
                    or compiled['bindings']['context']['sha256']!=self.files['context.json']):
                raise ValueError('Compiled graph provenance differs')
            layout=bio.make_layout(*Prefill1024LayoutCandidate(query_tokens=n).signature())
            check_graph_binding(compiled,capture,context=compiled['bindings']['context'],query_tokens=n,layout=layout)
            if m['layouts'][str(n)]!=layout or linked['layouts'][str(n)]!=layout:
                raise ValueError('Shared entry geometry differs')
            if [r['sha256'] for r in compiled['sources'] if Path(r['path']).name=='hexagon_buffer_io.py']!=[m['buffer_generator_sha256']]:
                raise ValueError('Compiler serializer identity differs')
            if n in (3,4):
                fixed=self.read(f'provenance/fixed-{n}.json');audit=self.read(f'provenance/fixed-audit-{n}.json')
                if (audit.get('status')!='prefill_fixed_complete_DSP_forward_independently_bitexact_NOT_generate'
                        or audit.get('result_sha256')!=self.files[f'provenance/fixed-{n}.json']
                        or audit.get('query_tokens')!=n or audit.get('output_tensors')!=114
                        or fixed['compiled']['sha256']!=self.files[f'compiled-{n}.json']):
                    raise ValueError('New complete fixed DSP graph proof required before this generation experiment')
            self.captures[n]=capture;self.compiled[n]=compiled
        if any(self.compiled[n][key]!=self.compiled[1][key] for n in (2,3,4) for key in ('parameters','storage')):
            raise ValueError('All entries must share unchanged native weights')
        libraries=[linked['dispatches']['native']]+[g['library'] for g in linked['graphs']]+linked['constants']
        if (len(libraries)!=7 or len(linked['constants'])!=2 or m.get('constant_partitions')!=2
                or [r['sha256'] for r in libraries]!=[r['sha256'] for r in m['libraries']]
                or [Path(r['path']).name for r in libraries]!=[Path(r['path']).name for r in m['libraries']]
                or dict(runtime,artifacts=m['runtime']['artifacts'])!=m['runtime']
                or [r['sha256'] for r in runtime['artifacts']]!=[r['sha256'] for r in m['runtime']['artifacts']]):
            raise ValueError('Exact relocated model/runtime closure required')
        for row in m['libraries']+m['runtime']['artifacts']:self.artifact(row)
        policy=m['decomposition_policy']
        if self.files[policy['path']]!=policy['sha256'] or policy!=self.base.manifest['decomposition_policy']:
            raise ValueError('Original decomposition policy required')
        from hexagon_arm_prefill_plan import integration_plan
        reference=self.read('reference.json');integration_plan(reference)
        if reference['context']['sha256']!=self.files['context.json']:
            raise ValueError('Reference describes another model context')
        self.admission=dict(model_fixed_forward_audited=True,transport_audited=True,model_generation_accepted=False)

    def native_capture(self,n):
        if type(n) is not int or n not in self.captures:raise ValueError('Undeclared model shape')
        result=dict(self.captures[n]);relative='rotary-old/result.json' if n in (1,2) else 'rotary-prefill/result.json'
        if self.files[relative]!=result['rotary_control']['sha256']:raise ValueError('Positional table proof differs')
        result['rotary_control']=dict(result['rotary_control'],path=str(member(self.root,relative)))
        return result
