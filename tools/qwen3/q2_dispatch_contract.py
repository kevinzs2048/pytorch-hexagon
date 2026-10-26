"""Bind the dispatch prototype to the previously target-reviewed recurrence."""
from qwen3.compiler_dependency import compiler_tools
import json
import subprocess
from pathlib import Path
from build import HERE, entry
from profile_model_memory import checked
from revisit_native_compile_session_abba import require_normal_owner


def require_proven_qk(proof_path, controller, base):
    read = lambda path: json.loads(Path(path).read_text())
    proof = read(proof_path)
    owner = require_normal_owner(controller, read(controller)['authorized_deadline_unix'])
    if (proof['status'] != 'actual_dynamic_QK_ABBA56_all_FP32_words_full_FNV_and_normal_owners_independently_revisited'
            or proof['DSP_calls'] != 56 or proof['new_DSP_calls']
            or proof['model_inference_acceptance'] or len(proof['cases']) != 14
            or owner['source_revision'] != proof['reviewer_revision']
            or owner['command'][owner['command'].index('--output')+1] != str(proof_path)):
        raise ValueError('Original normal independent QK56 reviewer required')
    checked(proof['reviewer_source'])
    receipt = read(checked(proof['receipt']))
    lowering = read(checked(receipt['lowering']))
    inspection = read(checked(lowering['inspection']))
    sdk = read(checked(inspection['SDK']))
    original = checked(sdk['loop_profile_source']).read_text()
    original = original.replace('mlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module) {',
        'mlir::LogicalResult retainOriginalDynamicPV(mlir::ModuleOp module);\nmlir::LogicalResult instrumentQwenLoopRegions(mlir::ModuleOp module) {')
    before = '  llvm::errs()<<"\\nQWEN_DYNAMIC_QK_REGISTER_AFTER_END\\nQWEN_DYNAMIC_QK_STOP_NO_MODEL_CODE\\n";\n  return failure();'
    after = '  llvm::errs()<<"\\nQWEN_DYNAMIC_QK_REGISTER_AFTER_END\\nQWEN_DYNAMIC_QK_REGISTER_MODEL_CONTINUE_NOT_DSP_ACCEPTANCE\\n";\n  return retainOriginalDynamicPV(module);'
    source = compiler_tools()/'buffer_reuse/DynamicQKRegisterProven.inc'
    if original.count(before) != 1 or original.replace(before, after) != source.read_text():
        raise ValueError('Entire reviewed recurrence must be verbatim except its prior PV terminal hook')
    if entry(source)['sha256'] != base['loop_profile_source']['sha256']:
        raise ValueError('Same QK recurrence as the original composed model SDK required')
    checked(base['loop_profile_source'])
    if base['qk_operator_proof'] != entry(proof_path) or base['qk_review_controller'] != entry(controller):
        raise ValueError('Original model SDK must reference this exact operator review')
    for ref in proof['controls']:
        path = checked(ref)
        require_normal_owner(path, read(path)['authorized_deadline_unix'])
    return proof


def require_qk_model_review(review_path, controller):
    read = lambda path: json.loads(Path(path).read_text())
    proof = read(review_path); owner = read(controller)
    require_normal_owner(controller,owner['authorized_deadline_unix'])
    source = checked(proof['reviewer_source']); reviewer = source.parents[2]
    if (proof['status'] != 'zero_tail_original393_independently_revisited'
            or proof['DSP_calls'] != 393 or proof['model_inference_acceptance'] is not True
            or proof.get('QK_guard_candidate') is not True
            or proof['newly_executed_DSP_calls'] != 384 or proof['reused_independently_verified_DSP_calls'] != 9
            or proof['positive_native_callback_gain'] is not True
            or proof['ordered_FMA_precision_and_owned_lifetimes_preserved'] is not True
            or owner['source_root'] != str(reviewer) or owner['source_revision'] != proof['reviewer_revision']
            or owner['command'][1] != str(source)
            or owner['command'][owner['command'].index('--output')+1] != str(review_path)):
        raise ValueError('Exact complete QK393 independent model prerequisite and normal reviewer required')
    blob = subprocess.check_output(['git','-C',str(reviewer),'show',
        owner['source_revision']+':'+str(source.relative_to(reviewer))])
    import hashlib
    if hashlib.sha256(blob).hexdigest() != proof['reviewer_source']['sha256']:
        raise ValueError('Actual model reviewer differs from its committed source')
    closure = checked(proof['closure']); result = read(closure)
    model_owner = read(checked(proof['controller']))
    require_normal_owner(proof['controller']['path'],model_owner['authorized_deadline_unix'])
    if (result['status'] != 'zero_tail_original393_complete_NOT_independently_accepted'
            or result['DSP_calls'] != 393 or result.get('QK_guard_candidate') is not True
            or model_owner['source_revision'] != result['source_revision']
            or model_owner['command'][model_owner['command'].index('--output')+1] != str(closure.parent)):
        raise ValueError('Complete original model producer and actual normal whole-queue owner required')
    return proof
