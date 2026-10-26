"""Deliver a completed Q2 model393 without repeating numerical/model codegen work."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from build import ROOT,HERE,entry,write
from profile_model_memory import checked
from q2_dispatch_release import FULL_STATUS,REVIEW_STATUS
from revisit_q2_dispatch_preflight import actual_owner
from revisit_native_compile_session_abba import require_normal_owner
from native_warm_deployment import require_warm_deployment


def require_completed_model(closure_path,controller_path,review_path,review_controller_path):
    read = lambda path: json.loads(Path(path).read_text())
    closure = read(Path(closure_path)/'result.json'); review = read(review_path)
    actual_owner(controller_path,closure,'run_q2_dispatch_native393.py',Path(closure_path))
    owner = read(review_controller_path)
    require_normal_owner(review_controller_path,owner['authorized_deadline_unix'])
    source = checked(review['reviewer_source'])
    if (closure['status'] != FULL_STATUS or review['status'] != REVIEW_STATUS
            or closure['candidate_kind'] != 'ordered-q2-runtime-QK-dispatch'
            or closure.get('Q2_QK_dispatch_candidate') is not True or review.get('Q2_QK_dispatch_candidate') is not True
            or review.get('model_inference_acceptance') is not True
            or review.get('positive_prefill_native_gain') is not True
            or review.get('ordered_FMA_precision_and_owned_lifetimes_preserved') is not True
            or review['closure'] != entry(Path(closure_path)/'result.json') or review['controller'] != entry(controller_path)
            or owner['source_root'] != str(source.parents[2]) or owner['source_revision'] != review['reviewer_revision']
            or owner['command'][1] != str(source)
            or owner['command'][owner['command'].index('--output')+1] != str(review_path)):
        raise ValueError('Exact own complete independent Q2 model393 and normal owners required')
    for proof in (closure,review):
        if any(type(proof.get(k)) is not int or proof[k] != value for k,value in (
                ('DSP_calls',393),('candidate_DSP_calls',361),('paired_baseline_DSP_calls',32),
                ('newly_executed_DSP_calls',384),('reused_independently_verified_DSP_calls',9))):
            raise ValueError('Actual full original393 scope and separate new/reused calls required')
    return closure,review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','parent-index','model','model-controller','model-review','model-review-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--deadline-unix',type=int,required=True)
    a = p.parse_args()
    git = lambda *args: subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if (a.output.exists() or not a.output.is_absolute() or git('status','--porcelain')
            or a.deadline_unix-time.time()<32400 or not a.name
            or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for ch in a.name)):
        raise ValueError('Fresh committed delivery and original nine-hour admission required')
    closure,review = require_completed_model(a.model,a.model_controller,a.model_review,a.model_review_controller)
    revision = git('rev-parse','HEAD'); a.output.mkdir()
    record = dict(status='running',source_root=str(ROOT),source_revision=revision,source=entry(__file__),
                  deadline_unix=a.deadline_unix,steps=[],reused_model393_DSP_calls=393,
                  reused_model=entry(a.model/'result.json'),reused_model_controller=entry(a.model_controller),
                  model_review=entry(a.model_review),model_review_controller=entry(a.model_review_controller),
                  newly_executed_DSP_calls=0,model_PRIMARY_compilations=0,shared_default_changed=False)
    save = lambda: write(a.output/'result.json',record)
    def own(script,args,stem,minimum):
        if git('rev-parse','HEAD') != revision or git('status','--porcelain'):
            raise ValueError('Pinned delivery source changed')
        plan = a.output/(stem+'-plan.json'); controller = a.output/(stem+'-process.json')
        command = [sys.executable,str(HERE/script),*map(str,args)]
        write(plan,dict(source_root=str(ROOT),source_revision=revision,deadline_unix=a.deadline_unix,
                        minimum_remaining_seconds=minimum,command=command))
        record['current_stage']=stem;save()
        subprocess.run([sys.executable,str(HERE/'run_owned_command.py'),'--plan',str(plan),
                        '--controller',str(controller)],check=True)
        record['steps'].append(dict(stage=stem,controller=entry(controller),completed_unix=time.time()));save()
        return controller
    save()
    try:
        selected=a.output/'selection'
        own('select_q2_dispatch_release.py',['--parent-index',a.parent_index,'--closure',a.model,
            '--controller',a.model_controller,'--review',a.model_review,
            '--review-controller',a.model_review_controller,'--output',selected],'selection',300)
        warm=a.output/'warm800'
        own('run_native_warm_release.py',['--config',a.config,'--index',selected/'index.json',
            '--oracle',a.model/'warm16','--online-source',closure['numerical_source_root'],
            '--baseline-state-suite',a.model/'state304','--output',warm,'--name',a.name+'-delivery',
            '--deadline-unix',a.deadline_unix],'warm800',21600)
        result=json.loads((warm/'result.json').read_text())
        if result['status']!='qualified_dynamic_PV_warm_deployment_closed':
            raise ValueError('All original warm800 stages must close normally')
        deployment_path=warm/'warm-deployment.json'; deployment=json.loads(deployment_path.read_text())
        catalog=require_warm_deployment(deployment)
        artifact_owner=Path(deployment['warm_validation']['artifact_controller']['path'])
        board=json.loads((a.model/'warm16/commands.json').read_text())['board']
        prompts=board[board.index('--prompts-json')+1]; public=a.output/'public-cli16'
        public_owner=own('qwen3_native_infer.py',['--deployment',deployment_path,'--prompts-json',prompts,
            '--max-new-tokens','8','--chunk-size','32','--output',public,'--deadline-unix',a.deadline_unix],
            'public-cli16',1800)
        public_review=a.output/'public-cli16-review.json'
        own('revisit_native_artifact_generation.py',['--config',deployment['config'],'--index',deployment['index'],
            '--oracle',a.model/'warm16','--native',public,'--controller',public_owner,'--output',public_review,
            '--compiler-mode','original_warm_catalog','--artifact-catalog',catalog,
            '--artifact-controller',artifact_owner],'public-cli16-review',300)
        if json.loads(public_review.read_text())['DSP_calls']!=16:
            raise ValueError('Complete independent public CLI16 required')
        record.update(status='qualified_Q2_QK_dispatch_model393_and_warm800_public16_opt_in_delivery_closed',
                      delivery=entry(deployment_path),public_CLI16_review=entry(public_review),
                      newly_executed_DSP_calls=816,qualified_delivery_DSP_calls=816,
                      qualified_model_DSP_calls=393,completed_unix=time.time());save()
    except BaseException as exc:
        record.update(status='failed',error=repr(exc),completed_unix=time.time());save();raise


if __name__ == '__main__':
    main()
