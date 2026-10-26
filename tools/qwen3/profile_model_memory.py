"""Relink an accepted q1 object with diagnostic external memory-call counters.

This does not compile or rewrite the model. Its output needs separate full-packet
device validation before its counters are evidence of model memory-call cost.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from build import ROOT, verify_dependencies
from inspect_integer_matmul import sha


def instrument(text,inspect_fmaf=False):
    entry='extern "C" int hexagon_forward_buffers('
    if text.count(entry)!=1:
        raise ValueError('Exactly one packet entry required')
    declarations='''extern "C" void qwen_memory_profile_begin();
extern "C" void qwen_memory_profile_end();
extern "C" int qwen_memory_profile_query(const unsigned char *,int,unsigned char *,int,uint64_t *);
'''
    if inspect_fmaf:
        declarations+='extern "C" int qwen_fmaf_effects_query(const unsigned char *,int,unsigned char *,int,uint64_t *);\n'
    text=text.replace(entry,declarations+entry,1)
    guard='  *kernel_us = 0;\n'
    position=text.index(guard,text.index(entry))
    query=('  int effects = qwen_fmaf_effects_query(input,input_len,output,output_len,kernel_us);\n'
           '  if (effects >= 0) return effects;\n') if inspect_fmaf else ''
    text=text[:position]+text[position:].replace(guard,guard+query+'''  int diagnostic = qwen_memory_profile_query(input,input_len,output,output_len,kernel_us);
  if (diagnostic >= 0) return diagnostic;
''',1)
    begin='  *kernel_us = benchmark_time_us(1, [&]() {\n'
    if text.count(begin)!=1:
        raise ValueError('One measured model invocation required')
    position=text.index(begin)+len(begin)
    end=text.index('\n  });',position)
    call=text[position:end]
    if call.count('_mlir_ciface_Qwen3DynamicBoundProbe(')!=1 or ';' not in call:
        raise ValueError('Original model call changed')
    return text[:position]+'    qwen_memory_profile_begin();\n'+call+'\n    qwen_memory_profile_end();'+text[end:]


def checked(row):
    path=Path(row['path'])
    if sha(path)!=row['sha256']:
        raise ValueError('Pinned input changed: '+str(path))
    return path


def linked_caller(caller, link):
    """Select the actual authenticated link input, including caller byte copies."""
    original = checked(caller['wrapper'])
    proof = link.get('caller_byte_copy')
    if proof is None:
        selected = original
    else:
        sys.path.insert(0, str(ROOT/'qwen3/runtime/arm'))
        from hexagon_caller_byte_copy import CONTRACT, rewrite_caller
        if (proof.get('contract') != CONTRACT or proof.get('tensor_transfer_calls') != 117
                or proof.get('original_model_object_and_FP_order_unchanged') is not True
                or proof.get('owned_buffer_lifetimes_unchanged') is not True
                or proof.get('original_wrapper') != caller['wrapper']):
            raise ValueError('Complete original caller byte-copy proof required')
        selected = checked(proof['candidate_wrapper'])
        for name in ('runtime', 'source', 'nm'):
            checked(proof[name])
        if selected.read_text() != rewrite_caller(original.read_text()):
            raise ValueError('Actually linked caller must preserve the exact original rewrite')
        if str(original) in link['command']:
            raise ValueError('Original and rewritten callers cannot both be linked')
    if link['command'].count(str(selected)) != 1:
        raise ValueError('Exactly one authenticated actual caller link input required')
    return selected


def preserve_wrapped_memcpy(command, nm, objcopy, output,
                            delegate_symbol='qwen_profile_preserved_memcpy'):
    # Shared with the model linker; imports no model/compiler tool state.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen3/runtime/arm'))
    from hexagon_preserved_memcpy import preserve_wrapped_memcpy as preserve
    return preserve(command,nm,objcopy,output,delegate_symbol=delegate_symbol)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','machine','recipe','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--external-helpers',action='store_true',
                        help='Also delegate and time pinned FP/INT external callees without rewriting the model')
    parser.add_argument('--allocation-helpers',action='store_true',
                        help='Aggregate allocator timings by function and requested size; preserve original errno')
    parser.add_argument('--inspect-fmaf',action='store_true',
                        help='Add an independent runtime fmaf provider/bytes/results/errno query; no replacement')
    parser.add_argument('--dynamic-query',action='store_true',
                        help='Diagnose the already accepted query2..32 graph without changing its object')
    parser.add_argument('--dynamic-multi-int8',action='store_true',
                        help='Delegate and time the original dynamic multi-matmul in category 10')
    parser.add_argument('--integer-timeline',action='store_true',
                        help='Timestamp only the original dynamic INT8 calls; no scalar/allocator wrappers')
    parser.add_argument('--attention-phases',action='store_true',
                        help='Also timestamp the first and last existing softmax exponentials per layer')
    args=parser.parse_args()
    if args.attention_phases and not args.integer_timeline:
        parser.error('Attention boundaries require the unchanged INT8 timeline')
    if args.integer_timeline and (not args.dynamic_multi_int8 or not args.external_helpers
            or args.allocation_helpers or args.inspect_fmaf):
        parser.error('Timeline requires dynamic multi-INT8 only, without allocation/effects profiling')
    if args.dynamic_multi_int8 and (not args.dynamic_query or not args.external_helpers):
        parser.error('Dynamic INT8 profiling requires the dynamic graph and external helpers')
    if args.allocation_helpers and not args.external_helpers:
        parser.error('Allocation diagnostics require --external-helpers')
    if not args.output.is_absolute() or args.output.exists():
        parser.error('Fresh absolute output required')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed diagnostic source required')
    config=json.loads(args.config.read_text())
    verified=verify_dependencies(config,json.loads(Path(__file__).with_name('dependencies.lock.json').read_text()))
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    link=json.loads((args.machine/'linked/result.json').read_text())
    caller=json.loads((args.machine/'caller/result.json').read_text())
    objects=json.loads((args.machine/'objects/result.json').read_text())
    recipe=json.loads(args.recipe.read_text())
    expected_query=None if args.dynamic_query else 1
    expected_bytes=235012880 if args.dynamic_query else 234885520
    if (link['query_specialization']!=expected_query or link['input_bytes']!=expected_bytes
            or link['output_bytes']!=235488768 or len(objects['objects'])!=4):
        raise ValueError('Frozen original model packet and four objects required')
    if args.dynamic_query:
        sys.path.insert(0,str(ROOT/'qwen3/runtime/arm'))
        from hexagon_qwen_shape_policy import contract_query_bounds
        if contract_query_bounds(caller['contract'])!=(2,32) or args.inspect_fmaf:
            raise ValueError('Only the original accepted dynamic graph may be diagnosed')
    original=linked_caller(caller,link);header=checked(caller['runtime_header'])
    obj=checked(objects['objects'][0]);checked(link['library'])
    constants_path=checked(recipe['inputs']['constants'])
    constants=json.loads(constants_path.read_text())
    for row in constants['libraries']:
        checked(row)
        if row['input_object_sha256']!=objects['objects'][row['index']]['sha256']:
            raise ValueError('Frozen model constants differ')
    command=link['command'][:]
    if sha(command[0])!=link['compiler_sha256'] or sha(config['lld'])!=link['linker_sha256']:
        raise ValueError('Frozen linker or wrapper compiler changed')
    if command.count(str(original))!=1 or command.count(str(obj))!=1 or command.count('-o')!=1:
        raise ValueError('Unexpected original link command')
    link_inputs=[]
    for word in command:
        path=Path(word)
        if path.is_absolute() and path.is_file() and word!=link['library']['path']:
            link_inputs.append(dict(path=str(path),sha256=sha(path)))
        if word.startswith('-I'):
            folder=Path(word[2:])
            if folder.name=='dsp-include':
                link_inputs.extend(dict(path=str(p),sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file())
    args.output.mkdir(parents=True)
    # Reproduce the accepted library before adding any diagnostic machinery.
    # The numerical primary, production wrapper and every runtime stay intact.
    replay_command=command[:]
    replay_path=args.output/'uninstrumented-replay.so'
    replay_command[replay_command.index('-o')+1]=str(replay_path)
    replay=subprocess.run(replay_command,capture_output=True,timeout=120)
    (args.output/'uninstrumented-replay.stdout.txt').write_bytes(replay.stdout)
    (args.output/'uninstrumented-replay.stderr.txt').write_bytes(replay.stderr)
    replay_proof=dict(command=replay_command,returncode=replay.returncode,
        expected_library=link['library'],library=(dict(path=str(replay_path),sha256=sha(replay_path))
        if replay_path.is_file() else None))
    replay_proof['byte_identical']=(replay.returncode==0 and replay_proof['library'] is not None
        and replay_proof['library']['sha256']==link['library']['sha256'])
    (args.output/'uninstrumented-replay.json').write_text(json.dumps(replay_proof,indent=2)+'\n')
    if not replay_proof['byte_identical']:
        raise ValueError('Uninstrumented accepted library must replay byte-identically before profiling')
    (args.output/header.name).write_bytes(header.read_bytes())
    effect_guard=None
    if 'silu_effects' in caller:
        guard=checked(caller['silu_effects']['guard_header'])
        effect_guard=args.output/guard.name
        effect_guard.write_bytes(guard.read_bytes())
    wrapper=args.output/original.name;wrapper.write_text(instrument(original.read_text(),args.inspect_fmaf))
    runtime=Path(__file__).parent/'runtime/profile/model_memory_profile.c'
    (args.output/runtime.name).write_bytes(runtime.read_bytes())
    if args.external_helpers:runtime=runtime.with_name('model_external_profile.c')
    if args.integer_timeline:runtime=runtime.with_name('model_integer_timeline.c')
    source=args.output/runtime.name;source.write_bytes(runtime.read_bytes())
    object_path=args.output/'memory-profile.o'
    tools=Path(config['hexagon_tools'])/'bin'
    preserved_memcpy=(None if args.integer_timeline else preserve_wrapped_memcpy(command,tools/'llvm-nm',
        tools/'hexagon-llvm-objcopy',args.output))
    compile_command=[str(Path(config['hexagon_tools'])/'bin/hexagon-clang'),'-O2','-Wall',
        '-Wextra','-Werror','-fPIC','-fvisibility=hidden','-fno-builtin','-mv73','-mhvx','-G0',
        '-I',str(Path(config['hexagon_sdk'])/'incs'),'-I',str(Path(config['hexagon_sdk'])/'incs/stddef'),
        '-c',str(source),'-o',str(object_path)]
    if preserved_memcpy:
        compile_command.insert(1,'-DQWEN_PROFILE_MEMCPY_TARGET=qwen_profile_preserved_memcpy')
    if args.allocation_helpers:
        compile_command.insert(1,'-DQWEN_PROFILE_ALLOCATIONS')
    if args.dynamic_multi_int8:
        compile_command.insert(1,'-DQWEN_PROFILE_MULTI_INT8')
    elif args.dynamic_query:
        compile_command.insert(1,'-DQWEN_PROFILE_NO_PACKED_INT8')
    if args.attention_phases:
        compile_command.insert(1,'-DQWEN_TIMELINE_ATTENTION_PHASES')
    library=args.output/'memory-profile.so'
    command[command.index(str(original))]=str(wrapper)
    command[command.index('-o')+1]=str(library)
    symbols=['memcpy','memset','__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes']
    if args.external_helpers:
        symbols+=['__hexagon_sqrtf','expf','_mlir_ciface___hexagon_silu_exp_compat',
            '_mlir_ciface___hexagon_softmax_exp_compat','_mlir_ciface___hexagon_arm_sin4_lane',
            '_mlir_ciface___hexagon_arm_cos4_lane','_mlir_ciface___qwen_i8_matmul',
            '__hexagon_divdi3','__hexagon_udivdi3']
    if args.dynamic_multi_int8:
        symbols[9]='_mlir_ciface___qwen_i8_multi_matmul'
    if args.allocation_helpers:
        symbols+=['malloc','calloc','posix_memalign','free']
    if args.integer_timeline:
        symbols=['_mlir_ciface___qwen_i8_multi_matmul']
        if args.attention_phases:symbols+=['_mlir_ciface___hexagon_softmax_exp_compat']
    command[1:1]=[str(object_path)]+['-Wl,--wrap='+symbol for symbol in symbols
        if '-Wl,--wrap='+symbol not in command]
    effects_command=None
    if args.inspect_fmaf:
        effects=Path(__file__).parent/'runtime/profile/model_fmaf_effects.c'
        effects_source=args.output/effects.name;effects_source.write_bytes(effects.read_bytes())
        effects_object=args.output/'fmaf-effects.o'
        effects_command=compile_command[:]
        effects_command[effects_command.index(str(source))]=str(effects_source)
        effects_command[effects_command.index('-o')+1]=str(effects_object)
        command.insert(1,str(effects_object))
    record=dict(status='building_diagnostic_NOT_device_acceptance',model_object_unchanged=True,
        model_graph_changed=False,power_votes_changed=False,source_revision=revision,
        source_sha256=sha(__file__),SDK_verified_files=len(verified),model_object=dict(path=str(obj),sha256=sha(obj)),
        original_link_receipt=dict(path=str(args.machine/'linked/result.json'),sha256=sha(args.machine/'linked/result.json')),
        original_wrapper=dict(path=str(original),sha256=sha(original)),
        original_caller_receipt_wrapper=caller['wrapper'],
        preserved_caller_byte_copy=link.get('caller_byte_copy'),
        uninstrumented_replay=replay_proof,
        diagnostic_wrapper=dict(path=str(wrapper),sha256=sha(wrapper)),
        profiler_source=dict(path=str(source),sha256=sha(source)),
        runtime_header=dict(path=str(args.output/header.name),sha256=sha(header)),
        original_link_inputs=link_inputs,
        constants=constants['libraries'],input_bytes=link['input_bytes'],output_bytes=link['output_bytes'],
        compile_command=compile_command,link_command=command,DSP_calls=0)
    record['external_profile_symbols']={str(index+1):symbol for index,symbol in enumerate(symbols)}
    record['profile_state_and_counter_wrappers_library_local']=True
    record['allocation_profile_scope']=('aggregate_per_function_and_requested_size_without_call_sites'
        if args.allocation_helpers else None)
    record['allocation_wrappers_locally_bound']=args.allocation_helpers
    record['preserved_existing_memcpy']=preserved_memcpy
    record['external_profile_kinds']=list(range(1,len(symbols)+1))
    record['required_profile_kinds']=[1,2,3]+([10] if args.external_helpers and (not args.dynamic_query or args.dynamic_multi_int8) else [])
    if args.dynamic_query:
        record['query_bounds']=[2,32]
        record['input_bytes_are_maximum_capacity']=True
    record['original_memory_profile_source']=dict(path=str(args.output/'model_memory_profile.c'),
        sha256=sha(args.output/'model_memory_profile.c'))
    record['inspect_fmaf']=args.inspect_fmaf
    if args.integer_timeline:
        record['integer_call_timeline']=True
        record['timeline_only_wrapped_symbol']=None if args.attention_phases else symbols[0]
        record['timeline_wrapped_symbols']=symbols
        record['all_original_link_objects_unchanged']=True
        record['external_profile_symbols']={}
        record['external_profile_kinds']=[]
        record['required_profile_kinds']=[]
        record['attention_phase_timeline']=args.attention_phases
    if args.dynamic_multi_int8:
        record['dynamic_multi_INT8_instrumented']=True
    if effect_guard:
        record['original_model_effect_guard']=dict(path=str(effect_guard),sha256=sha(effect_guard))
    if effects_command:
        record['fmaf_effects_source']=dict(path=str(effects_source),sha256=sha(effects_source))
        record['fmaf_effects_compile_command']=effects_command
    try:
        steps=[('compile',compile_command)]+([('compile-effects',effects_command)] if effects_command else [])+[('link',command)]
        for name,argv in steps:
            start=time.monotonic();result=subprocess.run(argv,capture_output=True,timeout=120)
            (args.output/(name+'.stdout.txt')).write_bytes(result.stdout)
            (args.output/(name+'.stderr.txt')).write_bytes(result.stderr)
            record[name+'_seconds']=time.monotonic()-start;record[name+'_returncode']=result.returncode
            result.check_returncode()
        unresolved=subprocess.check_output([str(Path(config['hexagon_tools'])/'bin/llvm-nm'),'-u',str(library)],text=True)
        (args.output/'undefined-symbols.txt').write_text(unresolved)
        if any(line.split()[-1:][0].startswith('__real_') for line in unresolved.splitlines() if line.split()):
            raise ValueError('Diagnostic wrapper targets a callee absent from the unchanged graph')
        relocations=subprocess.check_output([str(tools/'llvm-readobj'),
            '--relocations',str(library)],text=True)
        local_symbols=['qwen_memory_profile_'+name for name in ('begin','end','query')]
        local_symbols+=['__wrap_'+name for name in symbols]
        if any(name in relocations for name in local_symbols):
            raise ValueError('Profiler state and counters must bind locally, without preemptible relocations')
        record['profile_state_and_counters_no_preemptible_relocations']=True
        if args.allocation_helpers:
            record['allocator_counters_no_preemptible_relocations']=True
        record['all_wrapped_model_callees_resolve']=True
        record.update(status='diagnostic_model_memory_library_built_NOT_device_acceptance',
            library=dict(path=str(library),sha256=sha(library)))
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['status'])


if __name__=='__main__':main()
