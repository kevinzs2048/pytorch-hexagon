"""Change only integer guard traversal; byte-replay and reuse both tested operators."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build import ROOT,HERE,entry,sha,write
from profile_model_memory import checked
from revisit_q2_dispatch_preflight import actual_owner

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','original-build','original-controller','original-review','original-review-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists()or git('status','--porcelain'):raise ValueError('Fresh frozen guard-only build required')
    old=read(a.original_build);review=read(a.original_review)
    actual_owner(a.original_controller,old,'build_q2_tail_physical_probe.py',a.original_build.parent)
    actual_owner(a.original_review_controller,review,'revisit_q2_tail_probe.py',a.original_review)
    if (review['receipt']!=entry(a.original_build)or review['DSP_calls']!=256
            or review['all_runtime_guard_counts_and_independent_positive_zero_extents_equal']is not True
            or review['four_rounding_modes_original_target_bits_and_FP_flags_equal']is not True):
        raise ValueError('Actual original target256 proof required')
    for row in old['sources']:checked(row)
    a.output.mkdir();c=read(a.config);commands=[]
    record=dict(old,status='building_Q2_row_major_guard_ONLY_NOT_target_or_model',source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),
        source=entry(__file__),original_build=entry(a.original_build),original_controller=entry(a.original_controller),
        original_target_review=entry(a.original_review),original_target_review_controller=entry(a.original_review_controller),
        commands=commands,isolated_operator_compilations=0,new_model_PRIMARY_compilations=0,DSP_calls=0,
        original_operator_objects_unchanged=True,integer_guard_variant=entry(compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent_row_major.c'),
        variants={})
    save=lambda:write(a.output/'result.json',record)
    def run(argv,label):
        done=subprocess.run(argv,capture_output=True,timeout=180)
        (a.output/(label+'.stdout.txt')).write_bytes(done.stdout);(a.output/(label+'.stderr.txt')).write_bytes(done.stderr)
        commands.append(dict(argv=argv,returncode=done.returncode));save();done.check_returncode()
    # Reproduce both actually tested operators with every original link byte.
    original_commands=old['commands'][-2:]
    for kind,command in zip(('baseline','tail'),original_commands):
        argv=command['argv'][:];target=a.output/(kind+'-original-replay.so');argv[argv.index('-o')+1]=str(target);run(argv,kind+'-replay')
        if sha(target)!=old['variants'][kind]['library']['sha256']:raise ValueError('Original complete operator library replay differs')
    compile=next(row['argv']for row in old['commands']if str(a.original_build.parent/'q2_qk_positive_zero_extent.c.o')in row['argv']and '-c'in row['argv'])[:]
    original_guard=a.original_build.parent/'q2_qk_positive_zero_extent.c.o';new_guard=a.output/'row-major-guard.o'
    source=compiler_tools()/'runtime/profile/q2_qk_positive_zero_extent_row_major.c'
    compile[compile.index('-c')+1]=str(source);compile[compile.index('-o')+1]=str(new_guard);run(compile,'compile-row-guard')
    record['integer_guard_object']=entry(new_guard)
    for kind,command in zip(('baseline','tail'),original_commands):
        argv=command['argv'][:]
        if argv.count(str(original_guard))!=1:raise ValueError('Exactly one original integer guard object required')
        target=a.output/(kind+'.so');argv[argv.index(str(original_guard))]=str(new_guard);argv[argv.index('-o')+1]=str(target);run(argv,kind+'-link')
        record['variants'][kind]=dict(old['variants'][kind],library=entry(target))
    reference=next(checked(row)for row in old['sources']if Path(row['path']).name=='dynamic_qk_reference.c')
    shutil.copy2(reference,a.output/'dynamic_qk_reference.c')
    record['sources']=old['sources']+[entry(source)]
    record.update(status='actual_Q2_row_major_guard_same_operators_linked_NOT_target_or_model',completed_unix=time.time());save()

if __name__=='__main__':main()
