"""Replay the exact numerically checked model, then add integer-guard counters."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from build import ROOT,HERE,entry,sha,write
from profile_model_memory import checked,instrument,linked_caller
from revisit_q2_dispatch_preflight import actual_owner
from native_model_acceptance import require_model_acceptance

SYMBOL='_mlir_ciface_qwen_q2_qk_positive_zero_extent'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','model-review','model-review-controller','original-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or git('status','--porcelain'):raise ValueError('Fresh frozen diagnostic required')
    review=read(a.model_review)
    actual_owner(a.model_review_controller,review,'revisit_q2_tail_model_comparison.py',a.model_review)
    if (review['status']!='actual_Q2_identical_tail_model_ABBA12_complete_logits_56KV_3FNV_and_normal_owners_independently_revisited_NOT_generation_acceptance'
            or review['DSP_calls']!=12 or review['model_inference_acceptance'] or review['generation_regression']
            or review['manifest']!=entry(a.original_manifest)):
        raise ValueError('Original independently checked same full model packets required')
    original=read(a.original_manifest);require_model_acceptance(read(checked(original['index'])))
    link=read(checked(original['link']));c=read(a.config)
    actual_owner(checked(original['link_controller']),link,'link_q2_tail_model.py',Path(original['link']['path']).parent)
    if original['libraries']!=dict(baseline=link['baseline_library'],candidate=link['library']):
        raise ValueError('Original exact candidate library binding required')
    for row in link['original_link_inputs']+link['guard_objects']:checked(row)
    caller=read(checked(original['caller']));baseline=read(checked(link['baseline_link']))
    command=link['commands'][-1]['argv'][:]
    if command[command.index('-o')+1]!=str(checked(link['library'])):
        raise ValueError('Actual original successful candidate link required')
    source=linked_caller(caller,baseline)
    if command.count(str(source))!=1:raise ValueError('One exact original packet caller required')
    header=checked(caller['runtime_header'])
    a.output.mkdir()
    commands=[]
    def run(argv,label):
        done=subprocess.run(argv,capture_output=True,timeout=180)
        (a.output/(label+'.stdout.txt')).write_bytes(done.stdout);(a.output/(label+'.stderr.txt')).write_bytes(done.stderr)
        commands.append(dict(argv=argv,returncode=done.returncode));done.check_returncode()
    replay=a.output/'uninstrumented-replay.so';argv=command[:];argv[argv.index('-o')+1]=str(replay);run(argv,'replay')
    if sha(replay)!=link['library']['sha256']:raise ValueError('Candidate must first replay byte-identically')
    wrapper=a.output/source.name;wrapper.write_text(instrument(source.read_text()))
    (a.output/header.name).write_bytes(header.read_bytes())
    if 'silu_effects'in caller:
        guard=checked(caller['silu_effects']['guard_header']);(a.output/guard.name).write_bytes(guard.read_bytes())
    runtime=compiler_tools()/'runtime/profile/q2_tail_guard_profile.c';obj=a.output/'guard-profile.o'
    compile=[str(Path(c['hexagon_tools'])/'bin/hexagon-clang'),'-O2','-Wall','-Wextra','-Werror','-fPIC','-fvisibility=hidden','-fno-builtin','-mv73','-mhvx','-G0',
        '-I',str(Path(c['hexagon_sdk'])/'incs'),'-I',str(Path(c['hexagon_sdk'])/'incs/stddef'),'-c',str(runtime),'-o',str(obj)]
    run(compile,'compile')
    library=a.output/'guard-profile.so';argv=command[:];argv[argv.index(str(source))]=str(wrapper);argv[argv.index('-o')+1]=str(library)
    argv[1:1]=[str(obj),'-Wl,--wrap='+SYMBOL];run(argv,'link')
    undefined=subprocess.check_output([str(Path(c['hexagon_tools'])/'bin/llvm-nm'),'-u',str(library)],text=True)
    if '__real_'in undefined or '__wrap_'in undefined:raise ValueError('Guard must delegate locally to original tested object')
    for row in link['original_link_inputs']+link['guard_objects']:checked(row)
    m=dict(original);m.update(diagnostic_instrumentation=True,model_review=entry(a.model_review),model_review_controller=entry(a.model_review_controller),
        original_manifest=entry(a.original_manifest),uninstrumented_candidate=link['library'],
        libraries=dict(baseline=link['baseline_library'],candidate=entry(library)),
        guard_profile=dict(runtime=entry(runtime),object=entry(obj),original_wrapper=entry(source),wrapper=entry(wrapper),
            model_primary=link['candidate_primary'],replay=entry(replay),commands=commands,wrapped_symbol=SYMBOL,packet_bytes=1024))
    for name,path in [('baseline.so',checked(link['baseline_library'])),('candidate.so',library)]:os.link(path,a.output/name)
    for row in m['packets']:
        for kind,label in [('input','input'),('output','reference')]:os.link(checked(row[kind]),a.output/(label+'-q'+str(row['query'])+'.bin'))
    write(a.output/'manifest.json',m)
    write(a.output/'result.json',dict(status='Q2_tail_guard_profile_prepared_NOT_executed',source_root=str(ROOT),
        source_revision=git('rev-parse','HEAD'),source=entry(__file__),manifest=entry(a.output/'manifest.json'),
        DSP_calls=0,model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False,completed_unix=time.time()))

if __name__=='__main__':main()
