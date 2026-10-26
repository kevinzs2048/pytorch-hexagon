"""Bind the dispatch prototype to the previously target-reviewed recurrence."""
from qwen3.compiler_dependency import compiler_tools
import json
from pathlib import Path
from build import entry
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
