"""Independent ARM CPU replay on a first differing actual DSP packet."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def snapshot(path,payload):
    from hexagon_sparse_packet_storage import sparsify_verified_packet
    path.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest();sparsify_verified_packet(path,digest)
    return dict(path=str(path),sha256=digest,bytes=len(payload))


def make_replay(*,sources,plan,plan_sha256,context,hf,audit_output):
    a=SimpleNamespace(sources=sources,plan=plan,plan_sha256=plan_sha256,audit_output=audit_output)
    def execute(command,folder):
        (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (folder/'log.txt').open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                env=dict(os.environ,PYTHONPATH=str(a.sources/'qwen3/runtime/arm')+':'+str(a.sources)+':'+os.environ.get('PYTHONPATH',''),PYTHONDONTWRITEBYTECODE='1'))
    def replay(expected,incoming,outgoing,layout):
        folder=a.audit_output/f"difference-{expected['step']:04d}";folder.mkdir()
        evidence=dict(actual_input=snapshot(folder/'input.bin',incoming),actual_output=snapshot(folder/'output.bin',outgoing))
        refdir=folder/'independent-native'
        cmd=[sys.executable,str(a.sources/'tools/qwen3/reference_packet_trace_arm.py'),
            '--sources',str(a.sources),'--context',str(context),'--hf',str(hf),
            '--plan',str(a.plan),'--plan-sha256',a.plan_sha256,'--output',str(refdir),
            '--snapshot-step',str(expected['step'])]
        execute(cmd,folder)
        rr=json.loads((refdir/'result.json').read_text())
        if rr['status']!='independent_native_exact_snapshot_recorded' or rr['DSP_calls']!=0:
            raise ValueError('Independent native snapshot execution failed')
        entry=rr['forwards'][-1]
        result=dict(native_input=Path(entry['input']['snapshot']).read_bytes(),
            native_output=Path(entry['output']['snapshot']).read_bytes(),evidence=evidence)
        evidence['native_reference']=dict(path=str(refdir/'result.json'),sha256=sha(refdir/'result.json'))
        if evidence['actual_input']['sha256']!=expected['input']['sha256']:
            same=folder/'same-input';same.mkdir()
            command=[sys.executable,str(a.sources/'qwen3/runtime/arm/reference_qwen_dynamic_exact_packet_arm.py'),
                '--request',str(folder/'input.bin'),'--request-sha256',evidence['actual_input']['sha256'],
                '--query',str(expected['query']),'--start-position',str(expected['positions'][0]),
                '--hf',str(hf),'--context',str(context),'--context-sha256',sha(context),'--output',str(same/'reference')]
            execute(command,same);sr=json.loads((same/'reference/result.json').read_text())
            if sr['status']!='dynamic_exact_packet_ARM_eager_logits_KV_recorded' or not sr['inputs_exact'] or sr['DSP_calls']!=0:
                raise ValueError('Independent exact actual-input forward failed')
            tensors=[]
            for key,shape in [('last_logits',(1,1,151936)),('updated_KV',(56,1,8,expected['query'],128))]:
                f=Path(sr[key]['path'])
                if sha(f)!=sr[key]['sha256']:raise ValueError('Same-input reference tensor bytes changed')
                tensors.append(np.frombuffer(f.read_bytes(),dtype='<f4').reshape(shape))
            result.update(same_input_sha256=sr['request_sha256'],same_input_logits=tensors[0],same_input_KV=tensors[1])
            evidence['same_input_reference']=dict(path=str(same/'reference/result.json'),sha256=sha(same/'reference/result.json'))
        return result
    return replay
