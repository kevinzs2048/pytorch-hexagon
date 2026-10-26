"""Generate with the installed dynamic inference deployment."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt');inputs.add_argument('--prompts-json');inputs.add_argument('--requests-json')
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--chat-template',action='store_true')
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--deadline-unix',type=int,help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path)
    a=p.parse_args();deployment=json.loads(a.deployment.read_text())
    if deployment.get('schema')!=1 or deployment.get('status')!='accepted_dynamic_model_native_AOT_opt_in_deployment':
        p.error('An accepted dynamic inference deployment is required')
    mode=deployment.get('compiler_mode','original_live_handoff')
    if mode not in ('original_live_handoff','original_warm_catalog'):
        p.error('A supported dynamic compiler mode is required')
    artifact=None
    if mode=='original_warm_catalog':
        from native_warm_deployment import require_warm_deployment
        artifact=require_warm_deployment(deployment)
    elif any(key in deployment for key in ('artifact_catalog','warm_validation')):
        p.error('Artifact proofs require an explicit warm deployment')
    if not 300<=a.timeout<=14400:p.error('Generation timeout must be between 300 and 14400 seconds')
    if a.chat_template and a.requests_json is not None:p.error('Chat template requires text')
    root=Path(deployment['results_root'])
    if not root.is_absolute():p.error('Deployment output root must be absolute')
    root.mkdir(parents=True,exist_ok=True)
    name='qwen3-infer-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    output=a.output.absolute() if a.output is not None else root/name
    delegate=Path(__file__).with_name('generate_native_aot.py')
    args=[str(delegate),'--name',name,'--output',str(output),
        '--deadline-unix',str(a.deadline_unix if a.deadline_unix is not None else int(time.time())+a.timeout),
        '--max-new-tokens',str(a.max_new_tokens),'--chunk-size',str(a.chunk_size)]
    for flag,key in [('config','config'),('index','index'),('baseline-native','baseline_native'),
        ('cache-state304','cache_state304'),('cache-abba','cache_abba'),('cache-name','cache_name')]:
        args+=['--'+flag,str(deployment[key])]
    if artifact is not None:args+=['--artifact-catalog',str(artifact)]
    if a.requests_json is not None:args+=['--requests-json',a.requests_json]
    else:args+=['--prompts-json',json.dumps([a.prompt],ensure_ascii=False) if a.prompt is not None else a.prompts_json]
    if a.chat_template:args.append('--chat-template')
    previous=sys.argv
    try:
        sys.argv=args;runpy.run_path(str(delegate),run_name='__main__')
    finally:sys.argv=previous
    print('结果保存于 '+str(output))


if __name__=='__main__':main()
