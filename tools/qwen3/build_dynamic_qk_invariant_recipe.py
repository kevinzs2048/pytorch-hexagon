"""Prepare distinct original two-graph QK reuse after positive model8 and precision32."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
from build import ROOT,entry,sha,write
from dynamic_qk_invariant_contract import KIND,require_reference,require_compiler
from revisit_key_tile_model import revisit
from run_integer_regressions import candidate_libraries
CONTRACT='original_Qwen_query2to32_invariant_Query_ordered_FMA_columns128_v1'


def require_precision(proof,run,compiler_sha):
    if (proof.get('status')!='dynamic_QK_precision32_full_FP32_raw_triple_FNV_and_normal_owners_independently_revisited'
            or proof.get('DSP_calls')!=32 or proof.get('new_DSP_calls')!=0
            or proof.get('all_original_scalar_and_candidate_words_bitexact_with_independent_ARM_fmaf') is not True
            or proof.get('model_inference_acceptance') is not False
            or proof.get('primitive_only') is not True
            or run.get('status')!='dynamic_QK_invariant_edges32_independent_ARM_fmaf_bitexact_NOT_model_acceptance'
            or run.get('DSP_calls')!=32 or run.get('verified_DSP_calls')!=32 or run.get('process_exit_code')!=0):
        raise ValueError('Own normally closed independently bitexact precision32 required')
    plan=json.loads(Path(proof['plan']['path']).read_text())
    file=next(row for row in plan['files'] if row['name']=='candidate-result.json')
    build=json.loads(Path(file['source']).read_text())
    if (sha(file['source'])!=file['sha256'] or build['compiler_sha256']!=compiler_sha
            or build['variant']!='invariant'):
        raise ValueError('Precision must test this actual compiled candidate SDK')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-cold','model-revisit','accepted-closure','accepted-model-audit','precision-revisit','numerical-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if (not a.output.is_absolute() or a.output.exists() or any(subprocess.check_output(
            ['git','-C',str(root),'status','--porcelain'],text=True).strip() for root in (ROOT,a.numerical_source))):
        raise ValueError('Fresh output and clean committed source required')
    closure=read(a.accepted_closure);reference=entry(a.accepted_closure)
    require_reference(closure,read(a.accepted_model_audit),reference)
    base,libraries=candidate_libraries(a.baseline_cold);original_path=Path(base['recipe']['path']);original=read(original_path)
    if (closure['candidate_libraries']!={str(k):v for k,v in libraries.items()}
            or len(original['inputs'])!=49 or original.get('empty_tensor_cse_query2_to32') is not True
            or original.get('gqa_key_tile_query1') is not True or not original['reuse_exact_prepared_linalg']
            or not original['reuse_exact_compiler_objects']):
        raise ValueError('Own complete accepted dynamic CSE two-category recipe required')
    proof=read(a.model_revisit)
    if (proof['candidate_kind']!=KIND or proof['DSP_calls']!=8 or not proof['positive_early_model_gain']
            or not proof['full_original_logits_and_56_KV_byteexact'] or proof['model_inference_acceptance'] is not False
            or Path(proof['accepted_cold']['path']).parent!=a.baseline_cold
            or revisit(Path(proof['run']['path']),Path(proof['plan']['path']),a.baseline_cold)!=proof):
        raise ValueError('Own positive full original CSE packet ABBA8 required')
    build=read(proof['build']['path']);sdk=read(build['compiler_receipt']['path']);code=read(build['objects_receipt']['path'])
    if (sha(proof['build']['path'])!=proof['build']['sha256'] or build['accepted_q2_baseline393']!=reference
            or build['original_model_library']['sha256']!=libraries[(2,32)]
            or sha(build['compiler_receipt']['path'])!=build['compiler_receipt']['sha256']):
        raise ValueError('Model8 and full accepted baseline lineage changed')
    require_compiler(sdk,Path(code['compiler_diagnostics']['path']).read_text())
    precision=read(a.precision_revisit);saved_run=precision['run'];run=read(saved_run['path'])
    if sha(saved_run['path'])!=saved_run['sha256']:raise ValueError('Actual independently verified precision run changed')
    require_precision(precision,run,sdk['candidate']['sha256'])
    for row in original['inputs'].values():
        if sha(row['path'])!=row['sha256']:raise ValueError('Original numerical dependency changed')
    recipe=copy.deepcopy(original);recipe.update(dynamic_qk_invariant_query2_to32=True,dynamic_qk_invariant_contract=CONTRACT)
    added=dict(query2_qk_invariant_compiler={k:sdk['candidate'][k] for k in ('path','sha256')},
        query2_qk_invariant_sdk_receipt=build['compiler_receipt'],query2_qk_invariant_objects_receipt=build['objects_receipt'],
        query2_qk_invariant_contract_source=entry(a.numerical_source/'tools/qwen3/dynamic_qk_invariant_contract.py'))
    replaced=dict(compiler_stage_reuse_source=entry(a.numerical_source/'tools/qwen3/reuse_accepted_compiler_objects.py'),
        prepared_stage_reuse_source=entry(a.numerical_source/'tools/qwen3/reuse_accepted_prepared_linalg.py'),compiler_stage_reuse_config=entry(a.config))
    recipe['inputs'].update(added,**replaced)
    if (len(recipe['inputs'])!=53 or any(recipe['inputs'][k]!=v for k,v in original['inputs'].items() if k not in replaced)):
        raise ValueError('All original q1, CSE provenance, arithmetic, constants and call dependencies must remain')
    a.output.mkdir();write(a.output/'machine-recipe.json',recipe)
    write(a.output/'result.json',dict(status='dynamic_QK_invariant_vector_exact_original_object_recipe_built_NOT_native_or_state_acceptance',
        candidate_kind=KIND,source_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        numerical_source=str(a.numerical_source),numerical_revision=subprocess.check_output(['git','-C',str(a.numerical_source),'rev-parse','HEAD'],text=True).strip(),
        base_recipe=entry(original_path),accepted_closure=reference,accepted_model_audit=entry(a.accepted_model_audit),model_revisit=entry(a.model_revisit),precision_revisit=entry(a.precision_revisit),
        recipe=entry(a.output/'machine-recipe.json'),all_original_FP_integer_query1_and_CSE_dependencies_unchanged=True,
        changed_graph_candidate_SDK_fallback_forbidden=True,new_primary_model_codegen=0,DSP_calls=0,minimum_original_full393_start_budget_hours=4,model_inference_acceptance=False))
    print('distinct dynamic QK original two-graph native recipe prepared; zero primary compilation or DSP')

if __name__=='__main__':main()
