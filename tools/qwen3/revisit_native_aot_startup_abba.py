"""Measure original/cache/cache/original from four independent native16 audits.

The host worker stays in session mode throughout. Two startup samples per
mode produce a diagnostic signal, never production or TUI acceptance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from build import ROOT,entry,sha,write
from revisit_native_compile_session_abba import require_normal_owner
from revisit_native_startup import startup_summary


def summarize(rows):
    if len(rows)!=4 or [r['aot_mode'] for r in rows]!=['original','cache','cache','original']:
        raise ValueError('Four original/cache/cache/original startup arms required')
    if any(r['worker_mode']!='session' or r['DSP_calls']!=16 or type(r['DSP_calls']) is not int for r in rows):
        raise ValueError('Exactly native16 with the same host session worker in every arm required')
    result={}
    for label,select in (
        ('bootstrap_seconds',lambda r:r['summary']['bootstrap_seconds']),
        ('first_token_seconds',lambda r:r['summary']['requests'][0]['first_token_seconds']),
        ('first_request_full_generate_seconds',lambda r:r['summary']['requests'][0]['full_generate_seconds']),
        ('AOT_prepare_seconds',lambda r:r['summary']['requests'][0]['capture_stages_seconds']['AOT_prepare_graph']),
        ('host_wait_seconds',lambda r:r['summary']['requests'][0]['capture_stages_seconds']['wait_for_compiler_host']),
        ('first_request_ARM_callback_seconds',lambda r:r['summary']['requests'][0]['ARM_forward_callback_seconds']),
        ('second_request_full_generate_seconds',lambda r:r['summary']['requests'][1]['full_generate_seconds'])):
        groups={mode:[select(r) for r in rows if r['aot_mode']==mode] for mode in ('original','cache')}
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for values in groups.values() for v in values):
            raise ValueError('Positive finite original native startup observations required')
        means={mode:statistics.mean(values) for mode,values in groups.items()}
        result[label]=dict(original_samples=groups['original'],cache_samples=groups['cache'],
            baseline=means['original'],candidate=means['cache'],ratio=means['cache']/means['original'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *words:subprocess.check_output(['git','-C',str(ROOT),*words],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain'):
        raise ValueError('Fresh diagnostic output and frozen independent reviewer required')
    manifest=read(a.arms_manifest);arms=manifest['arms'];deadline=manifest['deadline_unix'];rows=[];retained=[]
    if [r['order'] for r in arms]!=list(range(4)) or [r['aot_mode'] for r in arms]!=['original','cache','cache','original']:
        raise ValueError('Original four-arm order must be preserved')
    namespace=None;oracle=None;worker=None;previous_end=0
    for row in arms:
        for key in ('independent_review','reviewer_controller','controller'):
            if entry(row[key]['path'])!=row[key]:raise ValueError('Actual independent arm receipt changed')
        owner=require_normal_owner(row['controller']['path'],deadline)
        reviewer=require_normal_owner(row['reviewer_controller']['path'],deadline)
        proof=read(row['independent_review']['path']);folder=Path(row['native'])
        native=read(folder/'native-result.json');coordinator=read(folder/'result.json')
        protocol=read(folder/'AOT-protocol-controller.json');cache=read(folder/'AOT-cache-report.json')
        if (proof['status']!='native_AOT_protocol16_all_original_packets_full_guards_raw_FNV_and_normal_owners_independently_revisited_DIAGNOSTIC'
                or proof['cache_case']!=('original' if row['aot_mode']=='original' else 'hit')
                or proof['controller']!=row['controller'] or proof['native']!=entry(folder/'result.json')
                or proof['AOT_protocol']!=entry(folder/'AOT-protocol-controller.json')
                or proof['AOT_cache_report']!=entry(folder/'AOT-cache-report.json')
                or proof['DSP_calls']!=16 or proof['model_inference_acceptance'] is not False
                or proof['performance_acceptance'] is not False
                or sha(folder/'native-result.json')!=coordinator['native_sha256']
                or startup_summary(native)!=proof['summary'] or coordinator['worker_mode']!='session'
                or cache['aot_mode']!=row['aot_mode']
                or owner['start_unix']<previous_end or reviewer['start_unix']<owner['completed_unix']):
            raise ValueError('Complete independently reviewed native16 and ordered normal owners required')
        reviewer_path=Path(proof['reviewer_source']['path'])
        if (owner['source_root']!=str(ROOT) or reviewer['source_root']!=str(ROOT)
                or owner['command'][1]!=protocol['driver_source']['path']
                or owner['command'][owner['command'].index('--output')+1]!=str(folder)
                or reviewer['command'][1]!=str(reviewer_path)
                or reviewer['command'][reviewer['command'].index('--output')+1]!=row['independent_review']['path']
                or reviewer['source_revision']!=proof['reviewer_revision']):
            raise ValueError('Actual owners must execute the recorded producer and independent reviewer for this arm')
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',reviewer['source_revision']+':'+str(reviewer_path.relative_to(ROOT))])
        if hashlib.sha256(blob).hexdigest()!=proof['reviewer_source']['sha256'] or entry(reviewer_path)!=proof['reviewer_source']:
            raise ValueError('Actual independent reviewer executable differs')
        if namespace is None:
            namespace=protocol['manifest'];oracle=proof['oracle'];worker=coordinator['worker_source']
        if protocol['manifest']!=namespace or proof['oracle']!=oracle or coordinator['worker_source']!=worker:
            raise ValueError('All arms must share the original oracle, complete backend namespace and host worker')
        previous_end=reviewer['completed_unix']
        rows.append(dict(aot_mode=row['aot_mode'],worker_mode=coordinator['worker_mode'],DSP_calls=16,summary=proof['summary']))
        retained.append(dict(order=row['order'],aot_mode=row['aot_mode'],controller=row['controller'],
            reviewer_controller=row['reviewer_controller'],independent_review=row['independent_review'],
            native=entry(folder/'native-result.json')))
    measurements=summarize(rows)
    write(a.output,dict(status='native_AOT_ABBA64_same_session_worker_all_packets_guards_and_normal_owners_independently_revisited_DIAGNOSTIC',
        arms_manifest=entry(a.arms_manifest),arms=retained,measurements=measurements,DSP_calls=64,new_DSP_calls=0,
        startup_samples_per_mode=2,positive_early_startup_signal=measurements['first_token_seconds']['ratio']<.95,
        original_oracle=oracle,backend_namespace=namespace,host_worker_source=worker,
        model_inference_acceptance=False,performance_acceptance=False,production_configuration_changed=False,
        TUI_startup_claimed=False,reviewer_source=entry(__file__),reviewer_revision=git('rev-parse','HEAD'),completed_unix=time.time()))


if __name__=='__main__':main()
