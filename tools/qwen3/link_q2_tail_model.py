"""Replace only the current q2 primary; reuse the exact target-tested guard."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from build import ROOT,entry,write
from profile_model_memory import checked
from revisit_model_integer_timeline import historical_owner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-machine','model-code','model-controller','physical-build','physical-controller','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if a.output.exists() or not a.output.is_absolute() or git('status','--porcelain'):
        raise ValueError('Fresh frozen link output required')
    code,physical=read(a.model_code),read(a.physical_build/'result.json');c=read(a.config)
    historical_owner(a.model_controller,'compile_q2_tail_model_candidate.py',a.model_code.parent,
        code['source']['sha256'],code['source_revision'])
    historical_owner(a.physical_controller,'build_q2_tail_physical_probe.py',a.physical_build,
        physical['source']['sha256'],physical['source_revision'])
    sdk=read(checked(code['SDK']));review=read(checked(sdk['target_review']))
    if (code['status']!='one_existing_Q2_identical_tail_PRIMARY_compiled_constants_identical_NOT_model_acceptance'
            or code['model_PRIMARY_compilations']!=1 or len(code['unchanged_constant_objects'])!=3
            or physical['status']!='actual_Q2_identical_tail_and_current_dispatch_objects_linked_NOT_target_or_model'
            or review['receipt']!=entry(a.physical_build/'result.json') or review['primitive_positive_gain']is not True):
        raise ValueError('Actual target-proved same guard and one complete strict model compilation required')
    link_path=a.baseline_machine/'linked/result.json';link=read(link_path)
    objects=read(a.baseline_machine/'objects/result.json');old=checked(objects['objects'][0]);primary=checked(code['primary'])
    if (len(objects['objects'])!=4 or link['query_specialization']is not None
            or link['output_bytes']!=235488768 or old==primary
            or entry(old)['sha256']==entry(primary)['sha256']
            or [r['sha256']for r in code['unchanged_constant_objects']]!=[r['sha256']for r in objects['objects'][1:]]):
        raise ValueError('Original dynamic ABI, changed primary and byte-identical constants required')
    for row in objects['objects']:checked(row)
    command=link['command'][:]
    if command.count(str(old))!=1 or command.count('-o')!=1:raise ValueError('Exactly one original primary link input required')
    a.output.mkdir();record=dict(status='linking_Q2_identical_tail_model_NOT_numerical_acceptance',
        source_root=str(ROOT),source_revision=git('rev-parse','HEAD'),source=entry(__file__),
        model_code=entry(a.model_code),model_controller=entry(a.model_controller),physical_build=entry(a.physical_build/'result.json'),
        physical_controller=entry(a.physical_controller),target_review=sdk['target_review'],baseline_link=entry(link_path),
        baseline_library=link['library'],original_primary=entry(old),candidate_primary=entry(primary),
        commands=[],new_model_PRIMARY_compilations=0,DSP_calls=0,model_inference_acceptance=False,
        performance_acceptance=False)
    save=lambda:write(a.output/'result.json',record)
    def run(argv,label):
        row=dict(argv=argv,started_unix=time.time());record['commands'].append(row);save()
        r=subprocess.run(argv,capture_output=True,timeout=180)
        (a.output/(label+'.stdout.txt')).write_bytes(r.stdout);(a.output/(label+'.stderr.txt')).write_bytes(r.stderr)
        row.update(returncode=r.returncode,completed_unix=time.time());save();r.check_returncode()
    save()
    try:
        checked(link['library']);originals=[]
        for word in command:
            path=Path(word)
            if path.is_absolute() and path.is_file() and str(path)!=link['library']['path']:originals.append(entry(path))
        replay=a.output/'baseline-replay.so';argv=command[:];argv[argv.index('-o')+1]=str(replay);run(argv,'baseline-replay')
        if entry(replay)['sha256']!=link['library']['sha256']:raise ValueError('Current accepted q2 model must replay byte-identically')
        target=physical['commands'][-1]['argv'][:]
        physical_replay=a.output/'physical-replay.so';target[target.index('-o')+1]=str(physical_replay)
        run(target,'physical-replay')
        if entry(physical_replay)['sha256']!=physical['variants']['tail']['library']['sha256']:
            raise ValueError('Exact tested helper/object bytes must reproduce the target-proved physical library')
        helpers=[a.physical_build/name for name in ('q2_qk_positive_zero_extent.c.o','zero_tail_hvx_guard.c.o')]
        for helper in helpers:
            if target.count(str(helper))!=1:raise ValueError('Exact tested guard helper input required')
        candidate=a.output/'model.so';argv=command[:];argv[argv.index(str(old))]=str(primary)
        argv[argv.index('-o')+1]=str(candidate);argv[1:1]=list(map(str,helpers));run(argv,'candidate-link')
        undefined=subprocess.check_output([str(Path(c['hexagon_tools'])/'bin/llvm-nm'),'-u',str(candidate)],text=True)
        if 'qwen_q2_qk_positive_zero_extent'in undefined or 'qwen_zero_words'in undefined:
            raise ValueError('Target-tested guard must resolve locally inside the candidate model')
        for row in originals:checked(row)
        checked(primary)
        record.update(status='Q2_identical_tail_current_model_replayed_new_primary_and_tested_guard_linked_NOT_numerical_acceptance',
            library=entry(candidate),original_link_inputs=originals,guard_objects=[entry(p)for p in helpers],
            current_baseline_byte_replayed=True,actual_tested_physical_library_byte_replayed=True,
            original_caller_runtime_FP_helpers_and_constants_unchanged=True,
            input_bytes=link['input_bytes'],output_bytes=link['output_bytes'],completed_unix=time.time());save()
    except BaseException as error:record.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
