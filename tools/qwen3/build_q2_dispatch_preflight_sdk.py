"""Replay the original composed SDK, replace only QK with a no-codegen dispatch."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
from build import ROOT, HERE, entry, write
from build_zero_tail_pv_preflight_sdk import relocate
from profile_model_memory import checked
from q2_dispatch_contract import require_proven_qk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'base-sdk', 'qk-review', 'qk-review-controller', 'fixture', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--deadline-unix', type=int, required=True)
    p.add_argument('--scratch-root',type=Path,
        help='Existing local /dev/shm directory for transient linker outputs; final SDK remains durable')
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    c, base = read(a.config), read(a.base_sdk)
    git = lambda *args: subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if (not a.output.is_absolute() or a.output.exists() or git('status','--porcelain')
            or a.deadline_unix-time.time() < 1800):
        raise ValueError('Fresh output, frozen committed source and admitted budget required')
    if a.scratch_root is not None and (not a.scratch_root.is_absolute()
            or not a.scratch_root.resolve().is_relative_to(Path('/dev/shm'))
            or not a.scratch_root.is_dir() or a.scratch_root.is_symlink()
            or shutil.disk_usage(a.scratch_root).free < 1024**3):
        raise ValueError('Existing local tmpfs scratch root with 1GiB free required')
    require_proven_qk(a.qk_review, a.qk_review_controller, base)
    if (base['status'] != 'isolated_combined_dynamic_QK_PV_register_model_candidate_SDK_built_NOT_DSP_or_model_acceptance'
            or not base['unmodified_SDK_replay_byte_identical']
            or not base['original_link_inputs_unchanged'] or not base['original_SDK_unchanged']
            or entry(c['compiler']) != base['accepted_sdk']):
        raise ValueError('Exact previous composed SDK and unchanged shared compiler required')
    checked(base['candidate'])
    for path, digest in base['link_inputs'].items():
        if entry(path)['sha256'] != digest:
            raise ValueError('Original link input changed: '+path)
    # Bind the actual original extracted recurrence, not a hand-written proxy.
    proof = read(a.qk_review)
    receipt = read(checked(proof['receipt']))
    lowering = read(checked(receipt['lowering']))
    if entry(a.fixture) != lowering['variants']['baseline']['input']:
        raise ValueError('Exact originally target-tested QK fixture required')
    old = next(row for row in base['commands'] if '-c' in row['argv']
        and row['argv'][row['argv'].index('-c')+1].endswith('DynamicQKRegisterCompose.cpp'))
    link = base['commands'][-1]
    old_object = old['argv'][old['argv'].index('-o')+1]
    if link['argv'].count(old_object) != 1:
        raise ValueError('Exactly one QK object may be replaced')
    retained = {str((Path(link['cwd'])/x).resolve()): entry(Path(link['cwd'])/x)['sha256']
        for x in link['argv'] if x.endswith('.o')}
    a.output.mkdir()
    record = dict(status='building_Q2_dispatch_structural_only', source_root=str(ROOT),
        source_revision=git('rev-parse','HEAD'), source=entry(__file__), base_sdk=entry(a.base_sdk),
        qk_review=entry(a.qk_review), qk_review_controller=entry(a.qk_review_controller),
        proven_recurrence=entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterProven.inc'),
        dispatch=entry(compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp'), fixture=entry(a.fixture),
        retained_objects=retained, original_shared_sdk=entry(c['compiler']), commands=[],
        model_PRIMARY_compilations=0, DSP_calls=0, model_acceptance=False,
        performance_acceptance=False, deadline_unix=a.deadline_unix)
    save = lambda: write(a.output/'result.json', record)
    def run(argv, cwd, label):
        row = dict(argv=argv,cwd=cwd,started_unix=time.time())
        record['commands'].append(row); save()
        with (a.output/(label+'.log')).open('xb') as log:
            result = subprocess.run(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,LD_LIBRARY_PATH=':'.join(c['library_path'])),timeout=300)
        row.update(returncode=result.returncode,completed_unix=time.time()); save()
        result.check_returncode()
    scratch = tempfile.TemporaryDirectory(prefix='qwen3-q2-sdk-',dir=a.scratch_root) if a.scratch_root else None
    work = Path(scratch.name) if scratch else a.output
    record['transient_linker_root'] = str(work)
    save()
    try:
        transient_replay = work/'base-replay.so'
        run(relocate(link['argv'], transient_replay),link['cwd'],'base-replay')
        if entry(transient_replay)['sha256'] != base['candidate']['sha256']:
            raise ValueError('Original composed SDK replay must be byte-identical')
        replay = a.output/'base-replay.so'
        temporary = a.output/'base-replay-dedup.tmp'
        os.link(base['candidate']['path'],temporary); os.replace(temporary,replay)
        obj = a.output/'DynamicQKRegisterDispatch.cpp.o'
        argv = relocate(old['argv'],obj,compiler_tools()/'buffer_reuse/DynamicQKRegisterDispatch.cpp')
        argv.append('-DQWEN_Q2_DISPATCH_PREFLIGHT=1')
        run(argv,old['cwd'],'dispatch-compile')
        transient_library = work/'libtriton.so'
        argv = relocate(link['argv'],transient_library)
        argv[argv.index(old_object)] = str(obj)
        run(argv,link['cwd'],'dispatch-sdk-link')
        library = a.output/'libtriton.so'
        if transient_library != library:
            expected = entry(transient_library)['sha256']
            shutil.copy2(transient_library,library)
            if entry(library)['sha256'] != expected:
                raise ValueError('Complete final SDK differs after durable copy')
        for path,digest in {**base['link_inputs'],**retained}.items():
            if entry(path)['sha256'] != digest:
                raise ValueError('Original input changed during build: '+path)
        if entry(c['compiler']) != record['original_shared_sdk']:
            raise ValueError('Shared SDK changed')
        record.update(status='Q2_dispatch_preflight_SDK_built_STOP_BEFORE_MODEL_CODEGEN',
            candidate=entry(library), original_replay=entry(replay),
            unmodified_SDK_replay_byte_identical=True, original_PV_objects_unchanged=True,
            original_link_inputs_unchanged=True, completed_unix=time.time())
    except BaseException as exc:
        record.update(status='failed',error=repr(exc)); raise
    finally:
        if scratch:
            scratch.cleanup()
            record['transient_linker_root_removed'] = not work.exists()
        save()


if __name__ == '__main__':
    main()
