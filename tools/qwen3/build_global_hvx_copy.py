"""Reuse accepted primaries/constants and exact HVX code for model-local memcpy."""
from qwen3.compiler_dependency import compiler_tools
import argparse
import json
from pathlib import Path
import subprocess
import time
from build import ROOT, HERE, entry, sha, write
from native_model_acceptance import require_model_acceptance, accepted_cold
from profile_model_memory import checked, preserve_wrapped_memcpy
from revisit_q2_dispatch_preflight import actual_owner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('delivery','delivery-controller','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if not a.output.is_absolute() or a.output.exists() or git('status','--porcelain'):
        raise ValueError('Fresh output and committed exact-copy candidate required')
    release=read(a.delivery/'result.json')
    actual_owner(a.delivery_controller,release,'run_q2_dispatch_delivery.py',a.delivery)
    if (release['status']!='qualified_Q2_QK_dispatch_model393_and_warm800_public16_opt_in_delivery_closed'
            or release['qualified_model_DSP_calls']!=393 or release['qualified_delivery_DSP_calls']!=816
            or release['shared_default_changed'] is not False):
        raise ValueError('Current independently accepted model393 and complete delivery816 required')
    index=read(a.delivery/'selection/index.json');model,closure=require_model_acceptance(index)
    cold=accepted_cold(model,closure);commands=read(cold/'commands.json');worker=commands['worker']
    cache=Path(worker[worker.index('--machine-cache')+1]);machines={}
    for folder in cache.iterdir():
        if not folder.is_dir() or not (folder/'linked/result.json').is_file():continue
        link=read(folder/'linked/result.json');bounds=(1,1)if link['query_specialization']==1 else(2,32)
        if bounds in machines:raise ValueError('Exactly two unambiguous accepted cache domains required')
        checked(link['library'])
        if link['library']['sha256']!=index['candidate_libraries'][str(bounds)]:
            raise ValueError('Actual current accepted library differs from its complete model proof')
        machines[bounds]=(folder,link)
    if set(machines)!={(1,1),(2,32)}:raise ValueError('Both actual accepted dynamic model libraries required')
    source=compiler_tools()/'runtime/profile/global_hvx_byte_copy.cpp';probe=source.with_name('global_copy_probe.cpp')
    compiler=Path(machines[(1,1)][1]['command'][0]);tools=compiler.parent
    if any(sha(link['command'][0])!=link['compiler_sha256']for _,link in machines.values()):
        raise ValueError('Actual accepted Hexagon compiler changed')
    a.output.mkdir();runtime=a.output/'global-hvx-copy.o'
    command=[str(compiler),'-O2','-fPIC','-fno-builtin','-fno-strict-aliasing','-mv73','-mhvx','-G0',
             '-c',str(source),'-o',str(runtime)]
    record=dict(status='building_model_local_exact_HVX_copy_NOT_device_acceptance',source_root=str(ROOT),
        source_revision=git('rev-parse','HEAD'),source=entry(__file__),runtime_source=entry(source),
        accepted_delivery=entry(a.delivery/'result.json'),accepted_delivery_controller=entry(a.delivery_controller),
        accepted_index=entry(a.delivery/'selection/index.json'),model=model,models=[],commands=[],
        DSP_calls=0,new_model_PRIMARY_compilations=0,model_inference_acceptance=False,performance_acceptance=False,
        shared_default_changed=False,started_unix=time.time())
    save=lambda:write(a.output/'result.json',record);save()
    def run(label,argv):
        with (a.output/(label+'.log')).open('xb')as log:
            done=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        record['commands'].append(dict(label=label,argv=argv,returncode=done.returncode));save()
        done.check_returncode()
    try:
        run('compile-dispatch',command);record['runtime']=entry(runtime)
        names=subprocess.check_output([str(tools/'llvm-nm'),'--defined-only',str(runtime)],text=True).split()
        if names!=['00000000','T','__wrap_memcpy']:raise ValueError('Only the model-local memcpy dispatch may be defined')
        for bounds,(folder,original)in sorted(machines.items()):
            label='q1'if bounds==(1,1)else'q2';out=a.output/label;out.mkdir()
            caller=read(folder/'caller/result.json');objects=read(folder/'objects/result.json')
            primary=checked(objects['objects'][0]);primary_before=entry(primary)
            copy=original['caller_byte_copy'];copy_runtime=checked(copy['runtime']);copy_source=checked(copy['source'])
            if (copy['contract']!='owned_caller_only_exact_HVX_byte_copy_v1'
                    or sha(copy_source)!=sha(compiler_tools()/'runtime/profile/caller_hvx_byte_copy.c')
                    or original['command'].count(str(copy_runtime))!=1):
                raise ValueError('Exactly the existing accepted HVX byte-copy implementation must be reused')
            constants=read(checked(index['machine_recipe']))['inputs']['constants']
            constant_rows=read(checked(constants))['libraries']
            for row in constant_rows:
                checked(row)
                if row['input_object_sha256']!=objects['objects'][row['index']]['sha256']:
                    raise ValueError('Every original constant member must remain byte-identical')
            inputs=[]
            for word in original['command']:
                path=Path(word)
                if path.is_absolute() and path.is_file() and str(path)!=original['library']['path']:
                    inputs.append(entry(path))
            checked(caller['wrapper']);checked(caller['runtime_header'])
            if 'silu_effects'in caller:checked(caller['silu_effects']['guard_header'])
            replay=out/'baseline-replay.so';replay_command=original['command'][:]
            replay_command[replay_command.index('-o')+1]=str(replay)
            run(label+'-replay',replay_command)
            if sha(replay)!=original['library']['sha256']:
                raise ValueError('Current accepted model must replay byte-identically before changing dispatch')
            candidate_command=original['command'][:]
            preserved=preserve_wrapped_memcpy(candidate_command,tools/'llvm-nm',tools/'hexagon-llvm-objcopy',out,
                delegate_symbol='qwen_global_copy_preserved_memcpy')
            if preserved is None:raise ValueError('The complete original fixed512 wrapper must remain the fallback')
            target=out/'global-hvx-copy.so';candidate_command[candidate_command.index('-o')+1]=str(target)
            candidate_command.insert(1,str(runtime));run(label+'-candidate',candidate_command)
            reloc=subprocess.check_output([str(tools/'llvm-readobj'),'--relocations',str(target)],text=True)
            (out/'relocations.txt').write_text(reloc)
            if any(name in reloc for name in ('__wrap_memcpy','qwen_caller_byte_copy','qwen_global_copy_preserved_memcpy')):
                raise ValueError('Copy dispatch, existing HVX helper and fallback must bind locally to this model')
            if entry(primary)!=primary_before:raise ValueError('Accepted model primary changed')
            for row in inputs+constant_rows:checked(row)
            record['models'].append(dict(query_bounds=list(bounds),original_machine=str(folder),
                original_link=entry(folder/'linked/result.json'),original_link_inputs=inputs,primary=primary_before,
                constants=constant_rows,baseline=original['library'],baseline_replay=entry(replay),
                candidate=entry(target),library=entry(target),command=candidate_command,delegate=preserved,
                accepted_HVX_runtime=copy['runtime'],accepted_HVX_source=copy['source'],
                all_copy_symbols_locally_bound=True,input_bytes=original['input_bytes'],output_bytes=original['output_bytes']))
            save()
        first=record['models'][0];original=machines[(1,1)][1]
        common=[str(compiler),'-O3','-mv73','-mhvx','-nostdlib++','-fno-builtin']
        common +=[x for x in original['command']if x.startswith(('-I','--ld-path='))]
        common +=['-Wl,-z,separate-loadable-segments','-shared','-fPIC','-G0']
        old=first['delegate']['original_object']['path'];preserved=first['delegate']['diagnostic_delegate_object']['path']
        record['primitive']={}
        for role,extra in [('baseline',[old]),('candidate',[str(runtime),preserved,first['accepted_HVX_runtime']['path']])]:
            target=a.output/(role+'-copy-probe.so');argv=common+extra+['-Wl,--wrap=memcpy',str(probe),'-o',str(target)]
            run('probe-'+role,argv);record['primitive'][role]=dict(library=entry(target),command=argv)
        record.update(status='model_local_global_HVX_copy_both_accepted_primaries_relinked_NOT_device_or_generation_acceptance',
            primitive_source=entry(probe),all_original_model_primaries_and_constants_unchanged=True,
            original_fixed512_text_unchanged=True,model_math_and_objects_unchanged=True,
            owned_input_and_returned_buffer_lifetimes_unchanged=True,completed_unix=time.time());save()
    except BaseException as error:record.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
