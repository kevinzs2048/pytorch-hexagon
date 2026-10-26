"""Model-local exact byte-copy relinking; no model lowering or numerical change."""
from qwen3.compiler_dependency import compiler_root
import hashlib
import json
from pathlib import Path
import subprocess
from hexagon_compiled_artifact_cache import canonical, digest_file
from hexagon_preserved_memcpy import preserve_wrapped_memcpy

CONTRACT='model_local_existing_HVX_large_byte_copy_original_fixed512_fallback_v1'
INPUTS={'global_copy_runtime','global_copy_source','global_copy_review','global_copy_review_controller',
        'global_copy_linker','global_copy_preserver','global_copy_objcopy'}
REVIEW_STATUS='model_local_global_HVX_copy_original_ARM_q1_q8_ABBA8_all_logits_56KV_complete_packets_3FNV_and_owners_independently_revisited_NOT_generation_acceptance'
ROOT=Path(__file__).resolve().parents[3]


def checked(row):
    path=Path(row['path'])
    if not path.is_absolute() or not path.is_file() or digest_file(path)!=row['sha256']:
        raise ValueError('Pinned global-copy input changed: '+str(path))
    return path


def require_owner(path,result,script,output):
    owner=json.loads(Path(path).read_text());root=Path(result['source_root'])
    if (owner['status']!='completed' or type(owner['returncode'])is not int or owner['returncode']!=0
            or owner['completed_unix']>=owner['authorized_deadline_unix']
            or any(Path('/proc',str(owner[k])).exists()for k in ('controller_pid','child_pid'))
            or owner['source_root']!=str(root) or owner['source_revision']!=result['source_revision']
            or owner['command'][1]!=str(root/'tools/qwen3'/script)
            or owner['command'][owner['command'].index('--output')+1]!=str(output)):
        raise ValueError('Actual normal historical global-copy proof producer required')
    blob=subprocess.check_output(['git','-C',str(root),'show',
        result['source_revision']+':tools/qwen3/'+script])
    if hashlib.sha256(blob).hexdigest()!=result['source']['sha256']:
        raise ValueError('Historical global-copy proof executable differs')
    if script.startswith('revisit_') and result['source']['sha256']!=digest_file(ROOT/'tools/qwen3'/script):
        raise ValueError('The actual independent reviewer implementation is required')


def require_model_scope(review):
    if (review['status']!=REVIEW_STATUS or review['positive_model_gain']is not True
            or any(type(review.get(k))is not int or review[k]!=n for k,n in
                   [('DSP_calls',8),('new_DSP_calls',0),('new_model_PRIMARY_compilations',0)])
            or review['model_inference_acceptance']is not False or review['performance_acceptance']is not False
            or len(review['proofs'])!=2 or {r['query']for r in review['proofs']}!={1,8}
            or any(len(r['outputs'])!=4 for r in review['proofs'])):
        raise ValueError('Own actual ARM-exact complete q1/q8 model8 and measured gain required')


def authenticate(inputs):
    if not INPUTS<=set(inputs):raise ValueError('Complete explicit global-copy proof and inputs required')
    for name in INPUTS:checked(inputs[name])
    if (Path(inputs['global_copy_linker']['path']).resolve()!=Path(__file__).resolve()
            or Path(inputs['global_copy_preserver']['path']).resolve()!=Path(__file__).with_name('hexagon_preserved_memcpy.py').resolve()
            or Path(inputs['global_copy_source']['path']).resolve()!=compiler_root()/'tools/qwen3/runtime/profile/global_hvx_byte_copy.cpp'):
        raise ValueError('Only the actual reviewed byte dispatcher and linker may be used')
    path=checked(inputs['global_copy_review']);review=json.loads(path.read_text())
    require_owner(checked(inputs['global_copy_review_controller']),review,'revisit_global_copy_model.py',path)
    require_model_scope(review)
    build_path=checked(review['build']);build=json.loads(build_path.read_text())
    require_owner(checked(review['build_controller']),build,'build_global_hvx_copy.py',build_path.parent)
    primitive_path=checked(review['primitive']);primitive=json.loads(primitive_path.read_text())
    require_owner(checked(review['primitive_controller']),primitive,'revisit_global_copy_physical.py',primitive_path)
    if (build['status']!='model_local_global_HVX_copy_both_accepted_primaries_relinked_NOT_device_or_generation_acceptance'
            or build['new_model_PRIMARY_compilations'] or build['DSP_calls']
            or build['all_original_model_primaries_and_constants_unchanged']is not True
            or build['original_fixed512_text_unchanged']is not True
            or primitive['status']!='actual_DSP_model_local_global_copy_498_cases_ABBA4_complete_bytes_canaries_3FNV_and_normal_owner_independently_revisited_NOT_model_acceptance'
            or primitive['case_count']!=498 or primitive['DSP_primitive_calls']!=4
            or primitive['DSP_model_calls'] or primitive['positive_primitive_gain']is not True
            or primitive['build']!=review['build']
            or inputs['global_copy_runtime']['sha256']!=build['runtime']['sha256']
            or inputs['global_copy_source']['sha256']!=build['runtime_source']['sha256']):
        raise ValueError('Exact target-tested runtime, original fallback and operator498 prerequisites required')
    return review,build


def relink(linked,bounds,inputs):
    """Preserve the observed base link and add only the proven local dispatcher."""
    review,build=authenticate(inputs);bounds=tuple(bounds)
    matches=[row for row in build['models']if tuple(row['query_bounds'])==bounds]
    if len(matches)!=1:raise ValueError('One of the two original accepted query domains required')
    tested=matches[0];path=linked/'result.json';base=json.loads(path.read_text());library=checked(base['library'])
    if (base['library']['sha256']!=tested['baseline']['sha256'] or 'global_copy'in base
            or base.get('caller_byte_copy',{}).get('runtime',{}).get('sha256')!=tested['accepted_HVX_runtime']['sha256']
            or base.get('fixed512_copy',{}).get('runtime',{}).get('sha256')!=tested['delegate']['original_object']['sha256']):
        raise ValueError('The entire newly rebuilt baseline must equal the actually tested original library')
    command=list(base['command']);nm=checked(inputs['fixed512_copy_nm']);objcopy=checked(inputs['global_copy_objcopy'])
    tools=tested['delegate']['tools']
    if digest_file(nm)!=tools[0]['sha256'] or digest_file(objcopy)!=tools[1]['sha256']:
        raise ValueError('Original recorded ELF symbol tools required')
    baseline_path=linked/'global-copy-base-result.json'
    if baseline_path.exists():raise ValueError('Fresh global-copy base receipt required')
    baseline_path.write_bytes(path.read_bytes())
    delegate=preserve_wrapped_memcpy(command,nm,objcopy,linked,
        delegate_symbol='qwen_global_copy_preserved_memcpy')
    if delegate is None or delegate['original_object']['sha256']!=tested['delegate']['original_object']['sha256']:
        raise ValueError('Exactly the original fixed512 object must be preserved')
    # This SDK appends the objcopy command (including absolute paths) to
    # .comment. Retain that fresh derivation as evidence, then link the exact
    # already tested renamed object, just as we reuse the tested HVX runtime.
    # No executable section is normalized or discarded to manufacture a hash.
    reviewed_delegate=checked(tested['delegate']['diagnostic_delegate_object'])
    if (delegate['original_text_sha256']!=tested['delegate']['original_text_sha256']
            or delegate['delegate_symbol']!=tested['delegate']['delegate_symbol']):
        raise ValueError('Fresh symbol preservation must retain the reviewed original instructions')
    owned_delegate=linked/'preserved-memcpy.o'
    derived_delegate=linked/'derived-preserved-memcpy.o'
    owned_delegate.rename(derived_delegate)
    delegate['diagnostic_delegate_object']['path']=str(derived_delegate)
    owned_delegate.write_bytes(reviewed_delegate.read_bytes())
    if digest_file(owned_delegate)!=tested['delegate']['diagnostic_delegate_object']['sha256']:
        raise ValueError('Owned reviewed delegate bytes changed')
    delegate['exact_tested_link_object']=dict(path=str(owned_delegate),sha256=digest_file(owned_delegate))
    runtime_folder=linked.parent/'runtime';runtime_folder.mkdir(exist_ok=True)
    runtime=runtime_folder/'global-hvx-copy.o'
    if runtime.exists():raise ValueError('Fresh owned global-copy runtime required')
    runtime.write_bytes(checked(inputs['global_copy_runtime']).read_bytes())
    if digest_file(runtime)!=inputs['global_copy_runtime']['sha256']:raise ValueError('Owned runtime copy differs')
    candidate=linked/('libqwen_global_copy_q1.so'if bounds==(1,1)else'libqwen_global_copy_q2.so')
    if candidate.exists():raise ValueError('Fresh global-copy candidate library required')
    command.insert(1,str(runtime));command[command.index('-o')+1]=str(candidate)
    with (linked/'global-copy-link.log').open('xb')as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    if digest_file(candidate)!=tested['candidate']['sha256']:
        raise ValueError('Public cache must reproduce the exact fully packet-tested candidate library')
    inspector=nm.with_name('llvm-readobj')
    reloc=subprocess.check_output([str(inspector),'--relocations',str(candidate)],text=True)
    if any(name in reloc for name in ('__wrap_memcpy','qwen_caller_byte_copy','qwen_global_copy_preserved_memcpy')):
        raise ValueError('Every copy function must bind locally without interposition')
    (linked/'global-copy-relocations.txt').write_text(reloc)
    result=dict(base,command=command,source_sha256=digest_file(__file__),
        base_linker_source_sha256=base['source_sha256'],
        library=dict(path=str(candidate),sha256=digest_file(candidate),bytes=candidate.stat().st_size))
    result['global_copy']=dict(contract=CONTRACT,base_link=dict(path=str(baseline_path),sha256=digest_file(baseline_path)),
        runtime=dict(path=str(runtime),sha256=digest_file(runtime)),source=inputs['global_copy_source'],
        tested_model=inputs['global_copy_review'],tested_model_controller=inputs['global_copy_review_controller'],
        linker=inputs['global_copy_linker'],preserver=inputs['global_copy_preserver'],delegate=delegate,
        original_model_primary_and_constants_unchanged=True,original_fixed512_text_unchanged=True,
        all_copy_symbols_locally_bound=True,model_acceptance=False)
    path.write_bytes(canonical(result)+b'\n')
    return ['linked/'+library.name,'linked/global-copy-base-result.json','linked/preserved-memcpy.o',
            'linked/derived-preserved-memcpy.o',
            'linked/global-copy-link.log','linked/global-copy-relocations.txt','runtime/global-hvx-copy.o']
