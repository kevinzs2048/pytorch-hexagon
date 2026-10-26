"""Link captured code with its OWN generated C caller, not a fixed arity.

Reuse weight DSOs only if every compiled constant object matches the original.
No device connection, no graph substitution, no numerical acceptance.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from hexagon_compiled_artifact_cache import canonical, digest_file
from relink_captured_hexagon_object import replace_object


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('objects', 'caller', 'reference', 'manifest', 'constants'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guarded-copy-dynamic',action='store_true')
    parser.add_argument('--fixed512-copy-query1',action='store_true')
    for name in ('guarded-copy-runtime','guarded-copy-source'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('fixed512-copy-runtime','fixed512-copy-source','fixed512-copy-nm'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('caller-copy-runtime','caller-copy-source','caller-copy-nm'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    for name in ('zero-tail-runtime', 'zero-tail-link','qk-tail-runtime','qk-tail-link'):
        parser.add_argument('--'+name, type=Path)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    records = {}
    for name in ('objects', 'caller', 'reference', 'manifest', 'constants'):
        if digest_file(getattr(args, name)) != getattr(args, name+'_sha256'):
            raise ValueError('q1 link input changed: '+name)
        records[name] = json.loads(getattr(args, name).read_text())
    objects, caller, reference, manifest, constants = [records[name]
        for name in ('objects', 'caller', 'reference', 'manifest', 'constants')]
    from hexagon_qwen_shape_policy import contract_query_bounds
    bounds = contract_query_bounds(caller['contract'])
    specialization = bounds[0] if bounds[0] == bounds[1] else None
    q1 = bounds == (1, 1)
    guarded_copy=any((args.guarded_copy_runtime,args.guarded_copy_runtime_sha256,
                      args.guarded_copy_source,args.guarded_copy_source_sha256))
    copy_rows=None
    if args.guarded_copy_dynamic and not guarded_copy:
        raise ValueError('Explicit dynamic byte-copy selection requires its pinned runtime and source')
    if guarded_copy:
        permitted=(bounds==(2,32) if args.guarded_copy_dynamic else q1)
        if (not permitted or not all((args.guarded_copy_runtime,args.guarded_copy_runtime_sha256,
                args.guarded_copy_source,args.guarded_copy_source_sha256))
                or caller['input_bytes']!=(235012880 if args.guarded_copy_dynamic else 234885520)
                or caller['output_bytes']!=235488768):
            raise ValueError('Paired original query1 or explicit query2..32 byte-copy runtime and source required')
        from hexagon_guarded_memref_copy import CONTRACT,DYNAMIC_CONTRACT,validate_runtime
        copy_contract=DYNAMIC_CONTRACT if args.guarded_copy_dynamic else CONTRACT
        copy_rows=dict(runtime=dict(path=str(args.guarded_copy_runtime),sha256=args.guarded_copy_runtime_sha256),
                       source=dict(path=str(args.guarded_copy_source),sha256=args.guarded_copy_source_sha256))
        validate_runtime(copy_rows['runtime'],copy_rows['source'])
    fixed512=any(getattr(args,name+suffix) for name in
                 ('fixed512_copy_runtime','fixed512_copy_source','fixed512_copy_nm')
                 for suffix in ('','_sha256'))
    fixed512_rows=None
    if args.fixed512_copy_query1 and not fixed512:
        raise ValueError('Explicit query1 fixed512 selection requires the complete runtime tuple')
    if fixed512:
        selected=(bounds==(1,1) and not args.guarded_copy_dynamic if args.fixed512_copy_query1
                  else bounds==(2,32) and args.guarded_copy_dynamic)
        if (not selected or not guarded_copy
                or not all(getattr(args,name+suffix) for name in
                           ('fixed512_copy_runtime','fixed512_copy_source','fixed512_copy_nm')
                           for suffix in ('','_sha256'))):
            raise ValueError('Only the explicitly selected original guarded query domain admits the fixed512 runtime tuple')
        from hexagon_fixed512_memcpy import CONTRACT,QUERY1_CONTRACT,validate_runtime
        FIXED512_CONTRACT=QUERY1_CONTRACT if args.fixed512_copy_query1 else CONTRACT
        fixed512_rows={key:dict(path=str(getattr(args,'fixed512_copy_'+key)),
                               sha256=getattr(args,'fixed512_copy_'+key+'_sha256'))
                       for key in ('runtime','source','nm')}
        validate_runtime(fixed512_rows['runtime'],fixed512_rows['source'],fixed512_rows['nm'])
    expected_status = ('captured_q1_buffer_caller_prepared_NOT_linked_or_DSP' if q1 else
                       'captured_bounded_buffer_caller_prepared_NOT_linked_or_DSP')
    if (args.output.exists() or caller['status'] != expected_status
            or caller['query_specialization'] != specialization
            or caller['input_arguments'] != 60+caller['scalar_inputs']
            or not 2 <= caller['scalar_inputs'] <= 8 or caller['tensor_inputs'] != 60
            or caller['output_count'] != 57 or caller['reference_sha256'] != args.reference_sha256
            or objects['machine_code_compilations'] != 1 or len(objects['objects']) != 4
            or objects['status'] != 'dynamic-bound_q17_lifted_graph_Hexagon_code_candidate_constants_identical_NOT_DSP'
            or objects['inputs']['linalg']['sha256'] != caller['graph_sha256']
            or digest_file(caller['wrapper']['path']) != caller['wrapper']['sha256']
            or digest_file(caller['runtime_header']['path']) != caller['runtime_header']['sha256']
            or reference['constants_result_sha256'] != args.constants_sha256):
        raise ValueError('Captured q1 code and matching specialized C caller required')
    if 'silu_effects' in caller:
        from hexagon_silu_helper_effects import CONTRACT,DYNAMIC_CONTRACT,ORIGINAL_HEADER_SHA256
        effects=caller['silu_effects'];guard=effects['guard_header']
        expected_effect_contract=CONTRACT if q1 else DYNAMIC_CONTRACT
        if (bounds not in ((1,1),(2,32)) or effects['contract']!=expected_effect_contract
                or caller['input_bytes']!=(234885520 if q1 else 235012880)
                or caller['output_bytes']!=235488768 or effects['unchanged_FP_helper_sha256']!=ORIGINAL_HEADER_SHA256
                or caller['runtime_header']['sha256']!=ORIGINAL_HEADER_SHA256
                or effects['actual_fmaf_entire_code_bytes']!='82c100efc03f2050'
                or effects['runtime_mismatch_returncode']!=6
                or Path(guard['path'])!=Path(caller['wrapper']['path']).parent/'silu_effects_guard.h'
                or digest_file(guard['path'])!=guard['sha256']):
            raise ValueError('Original helper and guarded query1 scalar effect contract required')
    code = objects['objects'][0]
    if digest_file(code['path']) != code['sha256']:
        raise ValueError('Fresh q1 code object changed')
    argv = shlex.split(manifest['link_command'])
    if (argv[-1] != reference['library']['path']
            or argv.count(reference['wrapper']['path']) != 1
            or digest_file(reference['wrapper']['path']) != reference['wrapper']['sha256']):
        raise ValueError('Original link command/caller changed')
    directories = [Path(arg[2:]) for arg in argv if arg.startswith('-L')]
    weights = sorted((row for row in constants['libraries'] if row['index']), key=lambda row: row['index'])
    if [row['index'] for row in weights] != [1, 2, 3]:
        raise ValueError('Three original constant libraries required')
    for row in weights:
        path = Path(row['path'])
        candidates = [folder/path.name for folder in directories if (folder/path.name).is_file()]
        if (objects['objects'][row['index']]['sha256'] != row['input_object_sha256']
                or digest_file(path) != row['sha256'] or not candidates
                or candidates[0].resolve() != path.resolve()
                or argv.count('-l'+path.name[3:-3]) != 1):
            raise ValueError('q1 constant bytes differ or linked library is shadowed')
    linker = constants['inputs']['lld']
    if (argv.count('--ld-path='+linker['path']) != 1
            or digest_file(linker['path']) != linker['sha256']):
        raise ValueError('Pinned linker changed')
    args.output.mkdir()
    library = args.output/('libqwen_captured_q1.so' if q1 else 'libqwen_captured_buffer.so')
    argv = replace_object(argv, args.reference.parent/'bounded_rmsnorm_qwen.o', Path(code['path']), library)
    argv[argv.index(reference['wrapper']['path'])] = caller['wrapper']['path']
    if guarded_copy:
        from hexagon_guarded_memref_copy import add_link_runtime
        argv=add_link_runtime(argv,args.guarded_copy_runtime)
    if fixed512:
        from hexagon_fixed512_memcpy import add_link_runtime
        argv=add_link_runtime(argv,args.fixed512_copy_runtime,query1=args.fixed512_copy_query1)
    record = dict(status='linking_captured_q1_entry', source_sha256=digest_file(__file__),
        objects_sha256=args.objects_sha256, caller_sha256=args.caller_sha256,
        reference_sha256=args.reference_sha256, constants_sha256=args.constants_sha256,
        manifest_sha256=args.manifest_sha256, graph_sha256=caller['graph_sha256'],
        query_specialization=caller['query_specialization'], constant_objects_identical=True,
        input_arguments=caller['input_arguments'], scalar_inputs=caller['scalar_inputs'], tensor_inputs=60, output_count=57,
        input_bytes=caller['input_bytes'], output_bytes=caller['output_bytes'],
        compiler_sha256=digest_file(argv[0]), linker_sha256=linker['sha256'], command=argv,
        host_only_link=True, DSP_calls=0, numerical_acceptance=False)
    if 'silu_effects' in caller:record['silu_effects']=caller['silu_effects']
    if guarded_copy:record['guarded_copy']=dict(contract=copy_contract,**copy_rows,
        original_model_object_and_FP_order_unchanged=True,linker_wrap='memrefCopy')
    if fixed512:record['fixed512_copy']=dict(contract=FIXED512_CONTRACT,**fixed512_rows,
        original_model_object_and_FP_order_unchanged=True,linker_wrap='memcpy')
    caller_copy=any(getattr(args,name+suffix) for name in
        ('caller_copy_runtime','caller_copy_source','caller_copy_nm') for suffix in ('','_sha256'))
    if caller_copy:
        if not fixed512 or bounds not in ((1,1),(2,32)) or not all(getattr(args,name+suffix) for name in
                ('caller_copy_runtime','caller_copy_source','caller_copy_nm') for suffix in ('','_sha256')):
            raise ValueError('Complete original two-graph route and paired caller copy inputs required')
        from hexagon_caller_byte_copy import CONTRACT, rewrite_caller, validate_runtime, add_link_runtime
        rows={name:dict(path=str(getattr(args,'caller_copy_'+name)),sha256=getattr(args,'caller_copy_'+name+'_sha256'))
            for name in ('runtime','source','nm')}
        validate_runtime(rows['runtime'],rows['source'],rows['nm'])
        original=Path(caller['wrapper']['path']);candidate=args.output/('caller-q1.cpp' if q1 else 'caller-q2.cpp')
        candidate.write_text(rewrite_caller(original.read_text()))
        argv[argv.index(str(original))]=str(candidate)
        argv.insert(1,'-I'+str(original.parent))
        argv=add_link_runtime(argv,args.caller_copy_runtime)
        proof=dict(contract=CONTRACT,original_wrapper=caller['wrapper'],
            candidate_wrapper=dict(path=str(candidate),sha256=digest_file(candidate)),**rows,
            original_model_object_and_FP_order_unchanged=True,owned_buffer_lifetimes_unchanged=True,
            tensor_transfer_calls=117)
        (args.output/'caller-byte-copy-proof.json').write_bytes(canonical(proof)+b'\n')
        record['caller_byte_copy']=proof;record['command']=argv
    zero_tail = any(getattr(args, name+suffix) for name in ('zero_tail_runtime', 'zero_tail_link')
                    for suffix in ('', '_sha256'))
    qk_tail=any(getattr(args,name+suffix) for name in ('qk_tail_runtime','qk_tail_link')
                for suffix in ('','_sha256'))
    qk_code=None
    if qk_tail:
        if (not q1 or not zero_tail or not caller_copy or not all(getattr(args,name+suffix)
                for name in ('qk_tail_runtime','qk_tail_link') for suffix in ('','_sha256'))):
            raise ValueError('QK-tail requires both complete q1 runtimes and original owned caller')
        from hexagon_qk_tail import validate_link_runtime,CONTRACT as QK_CONTRACT
        qk_runtime=dict(path=str(args.qk_tail_runtime),sha256=args.qk_tail_runtime_sha256)
        qk_link=dict(path=str(args.qk_tail_link),sha256=args.qk_tail_link_sha256)
        pv_link=dict(path=str(args.zero_tail_link),sha256=args.zero_tail_link_sha256)
        qk_code,qk_producer=validate_link_runtime(qk_runtime,qk_link,pv_link)
        if (code['sha256']!=qk_code['primary']['sha256']
                or objects.get('qk_tail_producer')!=qk_producer['model_code']):
            raise ValueError('Exact target-tested QK primary and staged producer required')
    if zero_tail:
        if not q1 or not caller_copy or not all(getattr(args, name+suffix)
                for name in ('zero_tail_runtime', 'zero_tail_link') for suffix in ('', '_sha256')):
            raise ValueError('Only original q1 caller admits the paired zero-tail runtime')
        from hexagon_zero_tail import validate_link_runtime, CONTRACT
        runtime_row = dict(path=str(args.zero_tail_runtime), sha256=args.zero_tail_runtime_sha256)
        link_row = dict(path=str(args.zero_tail_link), sha256=args.zero_tail_link_sha256)
        producer = validate_link_runtime(runtime_row, link_row)
        if ((not qk_tail and code['sha256'] != producer['primary']['sha256'])
                or (qk_tail and objects.get('qk_tail_base_producer')!=json.loads(args.zero_tail_link.read_text())['model_code'])
                or objects.get('zero_tail_producer') != json.loads(args.zero_tail_link.read_text())['model_code']):
            raise ValueError('Linker object and exact reviewed zero-tail producer differ')
        argv.insert(1, str(args.zero_tail_runtime))
        record['zero_tail'] = dict(contract=CONTRACT, runtime=runtime_row, tested_link=link_row,
                                   model_acceptance=False)
        record['command'] = argv
    if qk_tail:
        argv.insert(1,str(args.qk_tail_runtime))
        record['qk_tail']=dict(contract=QK_CONTRACT,runtime=qk_runtime,tested_link=qk_link,
            unchanged_original_PV_runtime=True,model_acceptance=False)
        record['command']=argv
    try:
        with (args.output/'link.log').open('xb') as stream:
            subprocess.run(argv, cwd=args.reference.parent, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        with library.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('Expected Hexagon ELF library')
        record.update(status=('captured_q1_entry_linked_NOT_DSP_or_numerical' if q1 else
                              'captured_bounded_entry_linked_NOT_DSP_or_numerical'),
            library=dict(path=str(library), bytes=library.stat().st_size, sha256=digest_file(library)))
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (args.output/'result.json').write_bytes(canonical(record)+b'\n')
    print(record['status'])


if __name__ == '__main__':
    main()
