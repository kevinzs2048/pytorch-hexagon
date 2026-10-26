"""Reuse the checked model PRIMARY and replace only the newly target-proved integer guard."""
from qwen3.compiler_dependency import compiler_root
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from build import ROOT,entry,sha,write
from profile_model_memory import checked
from revisit_q2_dispatch_preflight import actual_owner
from native_model_acceptance import require_model_acceptance

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-review','model-review-controller','row-build','row-build-controller','row-review','row-review-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists()or git('status','--porcelain'):raise ValueError('Fresh frozen production relink required')
    original_review=read(a.model_review);build=read(a.row_build);review=read(a.row_review)
    actual_owner(a.model_review_controller,original_review,'revisit_q2_tail_model_comparison.py',a.model_review)
    actual_owner(a.row_build_controller,build,'build_q2_tail_row_guard.py',a.row_build.parent)
    actual_owner(a.row_review_controller,review,'revisit_q2_tail_row_guard.py',a.row_review)
    if (original_review['status']!='actual_Q2_identical_tail_model_ABBA12_complete_logits_56KV_3FNV_and_normal_owners_independently_revisited_NOT_generation_acceptance'
            or original_review['DSP_calls']!=12 or original_review['model_inference_acceptance']
            or review['status']!='actual_Q2_row_major_guard_ABBA256_full_bits_four_rounding_FP_flags_and_extents_independently_revisited_NOT_model'
            or review['DSP_calls']!=256 or review['primitive_positive_gain']is not True or review['receipt']!=entry(a.row_build)
            or build['isolated_operator_compilations']or build['new_model_PRIMARY_compilations']):
        raise ValueError('Independent full model12 plus new actual primitive256 guard proof required')
    old_manifest=checked(original_review['manifest']);m=read(old_manifest)
    require_model_acceptance(read(checked(m['index'])))
    link=read(checked(m['link']))
    actual_owner(checked(m['link_controller']),link,'link_q2_tail_model.py',Path(m['link']['path']).parent)
    original_guard=next(checked(row)for row in link['guard_objects']if Path(row['path']).name=='q2_qk_positive_zero_extent.c.o')
    guard=checked(build['integer_guard_object'])
    if (build['integer_guard_variant']['sha256']!=sha(compiler_root()/'tools/qwen3/runtime/profile/q2_qk_positive_zero_extent_row_major.c')
            or build['commands'][-1]['argv'].count(str(guard))!=1):raise ValueError('The exact target-tested integer guard object required')
    for row in link['original_link_inputs']+link['guard_objects']:checked(row)
    command=link['commands'][-1]['argv'][:]
    if command.count(str(original_guard))!=1 or command[command.index('-o')+1]!=str(checked(link['library'])):
        raise ValueError('One exact original successful model link required')
    a.output.mkdir();commands=[]
    def run(argv,label):
        done=subprocess.run(argv,capture_output=True,timeout=180)
        (a.output/(label+'.stdout.txt')).write_bytes(done.stdout);(a.output/(label+'.stderr.txt')).write_bytes(done.stderr)
        commands.append(dict(argv=argv,returncode=done.returncode));done.check_returncode()
    replay=a.output/'original-model-replay.so';argv=command[:];argv[argv.index('-o')+1]=str(replay);run(argv,'replay')
    if sha(replay)!=link['library']['sha256']:raise ValueError('Original numerical candidate must fully byte-replay')
    library=a.output/'row-model.so';argv=command[:];argv[argv.index(str(original_guard))]=str(guard);argv[argv.index('-o')+1]=str(library);run(argv,'row-link')
    for row in link['original_link_inputs']+link['guard_objects']:checked(row)
    checked(link['candidate_primary']);checked(build['integer_guard_object'])
    m['libraries']=dict(baseline=link['baseline_library'],candidate=entry(library))
    m['row_guard']=dict(build=entry(a.row_build),build_controller=entry(a.row_build_controller),review=entry(a.row_review),review_controller=entry(a.row_review_controller),
        original_model_review=entry(a.model_review),original_model_review_controller=entry(a.model_review_controller),original_manifest=entry(old_manifest),
        primary=link['candidate_primary'],guard_object=entry(guard),replay=entry(replay),commands=commands)
    for name,path in [('baseline.so',checked(link['baseline_library'])),('candidate.so',library)]:os.link(path,a.output/name)
    for row in m['packets']:
        for kind,label in [('input','input'),('output','reference')]:os.link(checked(row[kind]),a.output/(label+'-q'+str(row['query'])+'.bin'))
    write(a.output/'manifest.json',m)
    write(a.output/'result.json',dict(status='Q2_row_major_guard_model_prepared_NOT_executed',source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),
        source=entry(__file__),manifest=entry(a.output/'manifest.json'),DSP_calls=0,model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False,completed_unix=time.time()))

if __name__=='__main__':main()
