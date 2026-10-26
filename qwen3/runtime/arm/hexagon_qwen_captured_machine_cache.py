"""Cold-compile the actual captured Qwen graph, or reuse authenticated artifacts.

The recipe explicitly selects the current bounded W8A8 Qwen candidate. It is
not a model engine and does not own generation or KV semantics. Numerical
acceptance is deliberately separate from artifact identity.
"""
from qwen3.compiler_dependency import compiler_root
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hexagon_compiled_artifact_cache import (
    CompiledArtifactCache, canonical, digest_file)


class CapturedQwenMachineCache:
    def __init__(self, root, recipe_path, recipe_sha256, frontend_root):
        if digest_file(recipe_path) != recipe_sha256:
            raise ValueError('Explicit machine compiler recipe changed')
        self.recipe = json.loads(Path(recipe_path).read_text())
        self.recipe_path=Path(recipe_path)
        self.recipe_sha256=recipe_sha256
        if self.recipe.get('schema') != 1:
            raise ValueError('Unsupported captured Qwen compiler recipe')
        required = {'attention_optimizer', 'sums_optimizer', 'knorm_optimizer',
                    'compiler', 'options_result', 'expected_objects_result',
                    'reference', 'manifest', 'constants'}
        self.packed_i8_query1=self.recipe.get('packed_i8_query1',False)
        if type(self.packed_i8_query1) is not bool:
            raise ValueError('packed_i8_query1 must be an explicit boolean')
        if self.packed_i8_query1:
            required |= {'packed_i8_optimizer','packed_i8_runtime','packed_i8_nm'}
        self.packed_multi_i8_query2_to32=self.recipe.get('packed_multi_i8_query2_to32',False)
        if type(self.packed_multi_i8_query2_to32) is not bool:
            raise ValueError('packed_multi_i8_query2_to32 must be an explicit boolean')
        if self.packed_multi_i8_query2_to32:
            required |= {'packed_multi_i8_optimizer','packed_multi_i8_runtime','packed_multi_i8_nm'}
            from hexagon_multi_integer import CONTRACT
            if self.recipe.get('packed_multi_i8_contract')!=CONTRACT:
                raise ValueError('Explicit original query2..32 integer contraction contract required')
        self.silu_effects_query1=self.recipe.get('silu_effects_query1',False)
        if type(self.silu_effects_query1) is not bool:
            raise ValueError('silu_effects_query1 must be an explicit boolean')
        if self.silu_effects_query1:
            required |= {'silu_effects_optimizer','silu_effects_guard'}
            from hexagon_silu_helper_effects import CONTRACT
            if self.recipe.get('silu_effects_contract')!=CONTRACT:
                raise ValueError('Explicit original guarded SiLU effect contract required')
        self.guarded_copy_query1=self.recipe.get('guarded_copy_query1',False)
        if type(self.guarded_copy_query1) is not bool:
            raise ValueError('guarded_copy_query1 must be an explicit boolean')
        if self.guarded_copy_query1:
            required |= {'guarded_copy_runtime','guarded_copy_source'}
            from hexagon_guarded_memref_copy import CONTRACT,validate_runtime
            if self.recipe.get('guarded_copy_contract')!=CONTRACT or not self.packed_i8_query1:
                raise ValueError('Explicit original query1 integer/byte-copy contract required')
        self.guarded_copy_query2_to32=self.recipe.get('guarded_copy_query2_to32',False)
        if type(self.guarded_copy_query2_to32) is not bool:
            raise ValueError('guarded_copy_query2_to32 must be an explicit boolean')
        if self.guarded_copy_query2_to32:
            from hexagon_guarded_memref_copy import DYNAMIC_CONTRACT
            if not self.guarded_copy_query1 or self.recipe.get('guarded_copy_dynamic_contract')!=DYNAMIC_CONTRACT:
                raise ValueError('Explicit shared original query2..32 byte-copy contract required')
        self.silu_effects_query2_to32=self.recipe.get('silu_effects_query2_to32',False)
        if type(self.silu_effects_query2_to32) is not bool:
            raise ValueError('silu_effects_query2_to32 must be an explicit boolean')
        if self.silu_effects_query2_to32:
            from hexagon_silu_helper_effects import DYNAMIC_CONTRACT
            if (not self.silu_effects_query1 or not self.packed_multi_i8_query2_to32
                    or not self.guarded_copy_query2_to32
                    or self.recipe.get('silu_effects_dynamic_contract')!=DYNAMIC_CONTRACT):
                raise ValueError('Original integer/copy/query1 recipe plus explicit guarded query2..32 SiLU required')
            required.add('multi_silu_effects_optimizer')
        self.fixed512_copy_query2_to32=self.recipe.get('fixed512_copy_query2_to32',False)
        if type(self.fixed512_copy_query2_to32) is not bool:
            raise ValueError('fixed512_copy_query2_to32 must be an explicit boolean')
        if self.fixed512_copy_query2_to32:
            from hexagon_fixed512_memcpy import CONTRACT
            if (not self.silu_effects_query2_to32 or not self.packed_multi_i8_query2_to32
                    or not self.guarded_copy_query2_to32
                    or self.recipe.get('fixed512_copy_contract')!=CONTRACT):
                raise ValueError('Original multi-SiLU/integer/guarded-copy recipe plus exact512 byte-copy contract required')
            required|={'fixed512_copy_runtime','fixed512_copy_source','fixed512_copy_nm'}
        self.fixed512_copy_query1=self.recipe.get('fixed512_copy_query1',False)
        if type(self.fixed512_copy_query1) is not bool:
            raise ValueError('fixed512_copy_query1 must be an explicit boolean')
        if self.fixed512_copy_query1:
            from hexagon_fixed512_memcpy import QUERY1_CONTRACT
            if (not self.fixed512_copy_query2_to32 or not self.guarded_copy_query1
                    or self.recipe.get('fixed512_copy_query1_contract')!=QUERY1_CONTRACT):
                raise ValueError('Original fixed512 recipe plus explicit guarded query1 contract required')
        self.caller_hvx_byte_copy=self.recipe.get('caller_hvx_byte_copy',False)
        if type(self.caller_hvx_byte_copy) is not bool:
            raise ValueError('Caller HVX byte copy must be an explicit boolean')
        if self.caller_hvx_byte_copy:
            from hexagon_caller_byte_copy import CONTRACT
            if (not self.fixed512_copy_query1 or not self.fixed512_copy_query2_to32
                    or self.recipe.get('caller_hvx_byte_copy_contract')!=CONTRACT):
                raise ValueError('Both unchanged original fixed512 routes and exact caller transfer contract required')
            required|={'caller_copy_runtime','caller_copy_source'}
        self.integer_tiles_query1=self.recipe.get('integer_tiles_query1',False)
        if type(self.integer_tiles_query1) is not bool:
            raise ValueError('integer_tiles_query1 must be an explicit boolean')
        if self.integer_tiles_query1:
            from hexagon_q1_integer_tiles import CONTRACT
            if not self.packed_i8_query1 or self.recipe.get('integer_tiles_contract')!=CONTRACT:
                raise ValueError('Explicit original query1 INT32 ring tile contract required')
            required|={'integer_tiles_source','integer_tiles_header','integer_tiles_original_header'}
        self.reuse_exact_compiler_objects=self.recipe.get('reuse_exact_compiler_objects',False)
        if type(self.reuse_exact_compiler_objects) is not bool:
            raise ValueError('reuse_exact_compiler_objects must be an explicit boolean')
        if self.reuse_exact_compiler_objects:
            if not self.fixed512_copy_query2_to32:
                raise ValueError('Compiler-stage reuse currently requires the explicit guarded fixed512 recipe')
            required|={'compiler_stage_reuse_source','compiler_stage_reuse_cold','compiler_stage_reuse_config'}
        self.reuse_exact_prepared_linalg=self.recipe.get('reuse_exact_prepared_linalg',False)
        if type(self.reuse_exact_prepared_linalg) is not bool:
            raise ValueError('reuse_exact_prepared_linalg must be an explicit boolean')
        self.share_immutable_prepared_artifacts=self.recipe.get('share_immutable_prepared_artifacts',False)
        if (type(self.share_immutable_prepared_artifacts) is not bool or
                self.share_immutable_prepared_artifacts and not self.reuse_exact_prepared_linalg):
            raise ValueError('Immutable prepared sharing requires explicit exact preparation reuse')
        if self.reuse_exact_prepared_linalg:
            if not self.reuse_exact_compiler_objects:
                raise ValueError('Prepared-stage reuse requires the existing exact compiler-stage route')
            required.add('prepared_stage_reuse_source')
        self.empty_tensor_cse_query1=self.recipe.get('empty_tensor_cse_query1',False)
        if type(self.empty_tensor_cse_query1) is not bool:
            raise ValueError('empty_tensor_cse_query1 must be an explicit boolean')
        if self.empty_tensor_cse_query1:
            if (not self.reuse_exact_compiler_objects or not self.integer_tiles_query1
                    or self.recipe.get('empty_tensor_cse_contract')!='original_Qwen_query1_CSE_after_empty_tensor_v1'):
                raise ValueError('Original query1 recipe and explicit isolated CSE contract required')
            required|={'query1_cse_compiler','query1_cse_sdk_receipt','query1_cse_objects_receipt'}
        self.empty_tensor_cse_query2_to32=self.recipe.get('empty_tensor_cse_query2_to32',False)
        if type(self.empty_tensor_cse_query2_to32) is not bool:
            raise ValueError('Dynamic CSE selection must be explicit boolean')
        if self.empty_tensor_cse_query2_to32:
            if (not self.reuse_exact_compiler_objects or not self.reuse_exact_prepared_linalg or
                    self.recipe.get('empty_tensor_cse_dynamic_contract')!='original_Qwen_query2to32_CSE_after_empty_tensor_v1'):
                raise ValueError('Original two-graph reuse and isolated dynamic CSE required')
            required|={'query2_cse_compiler','query2_cse_sdk_receipt','query2_cse_objects_receipt'}
        self.dynamic_qk_invariant_query2_to32=self.recipe.get('dynamic_qk_invariant_query2_to32',False)
        if type(self.dynamic_qk_invariant_query2_to32) is not bool:
            raise ValueError('Dynamic QK selection must be explicit boolean')
        if self.dynamic_qk_invariant_query2_to32:
            if (not self.empty_tensor_cse_query2_to32 or
                    self.recipe.get('dynamic_qk_invariant_contract')!='original_Qwen_query2to32_invariant_Query_ordered_FMA_columns128_v1'):
                raise ValueError('Original accepted CSE route and distinct ordered QK contract required')
            required|={'query2_qk_invariant_compiler','query2_qk_invariant_sdk_receipt',
                'query2_qk_invariant_objects_receipt','query2_qk_invariant_contract_source'}
        self.gqa_value_view_query1=self.recipe.get('gqa_value_view_query1',False)
        if type(self.gqa_value_view_query1) is not bool:
            raise ValueError('gqa_value_view_query1 must be an explicit boolean')
        if self.gqa_value_view_query1:
            if (not self.empty_tensor_cse_query1 or not self.reuse_exact_prepared_linalg or
                    self.recipe.get('gqa_value_view_contract')!='original_Qwen_query1_identity_V_shape_view_v1'):
                raise ValueError('Existing CSE recipe plus proven identity V view contract required')
            required|={'query1_gqa_value_view_compiler','query1_gqa_value_view_sdk_receipt','query1_gqa_value_view_objects_receipt'}
        self.gqa_value_forward_query1=self.recipe.get('gqa_value_forward_query1',False)
        if type(self.gqa_value_forward_query1) is not bool:
            raise ValueError('gqa_value_forward_query1 must be an explicit boolean')
        if self.gqa_value_forward_query1:
            if (not self.gqa_value_view_query1 or
                    self.recipe.get('gqa_value_forward_contract')!='original_Qwen_query1_component_slices_through_proved_V_repeat_v1'):
                raise ValueError('Accepted identity V recipe plus explicit proved repeat-slice contract required')
            required|={'query1_gqa_value_forward_compiler','query1_gqa_value_forward_sdk_receipt','query1_gqa_value_forward_objects_receipt'}
        self.gqa_key_compact_query1=self.recipe.get('gqa_key_compact_query1',False)
        if type(self.gqa_key_compact_query1) is not bool:
            raise ValueError('gqa_key_compact_query1 must be an explicit boolean')
        if self.gqa_key_compact_query1:
            if (not self.gqa_value_forward_query1 or
                    self.recipe.get('gqa_key_compact_contract')!='original_Qwen_query1_eight_head_contiguous_K_SCF_copy_v1'):
                raise ValueError('Accepted V repeat-slice recipe plus explicit original SCF K copy contract required')
            required|={'query1_gqa_key_compact_compiler','query1_gqa_key_compact_sdk_receipt','query1_gqa_key_compact_objects_receipt'}
        self.gqa_dead_repeat_query1=self.recipe.get('gqa_dead_repeat_query1',False)
        if type(self.gqa_dead_repeat_query1) is not bool:
            raise ValueError('gqa_dead_repeat_query1 must be an explicit boolean')
        if self.gqa_dead_repeat_query1:
            if (not self.gqa_key_compact_query1 or
                    self.recipe.get('gqa_dead_repeat_contract')!='original_Qwen_query1_proved_closed_unused_V_repeat_after_CSE_v1'):
                raise ValueError('Original K recipe plus explicit late closed-use dead-repeat contract required')
            required|={'query1_gqa_dead_repeat_compiler','query1_gqa_dead_repeat_sdk_receipt','query1_gqa_dead_repeat_objects_receipt'}
        self.gqa_key_tile_query1=self.recipe.get('gqa_key_tile_query1',False)
        if type(self.gqa_key_tile_query1) is not bool:
            raise ValueError('gqa_key_tile_query1 must be an explicit boolean')
        if self.gqa_key_tile_query1:
            if (not self.gqa_dead_repeat_query1 or self.recipe.get('gqa_key_tile_contract') !=
                    'original_Qwen_query1_pure_compact_K_copy_partition_N32_v1'):
                raise ValueError('N32 K scheduling requires the complete original dead-repeat recipe and exact contract')
            required|={'query1_gqa_key_tile_contract_source'}
        self.ordered_pv_register_query1=self.recipe.get('ordered_pv_register_query1',False)
        if type(self.ordered_pv_register_query1) is not bool:
            raise ValueError('ordered_pv_register_query1 must be an explicit boolean')
        if self.ordered_pv_register_query1:
            if (not self.gqa_key_tile_query1 or not self.dynamic_qk_invariant_query2_to32
                    or self.recipe.get('ordered_pv_register_contract') !=
                    'original_Qwen_query1_ordered_K1024_PV_columns8_after_N32_GQA_v1'):
                raise ValueError('Ordered PV requires the complete current QK and N32 recipe')
            required|={'query1_ordered_pv_compiler','query1_ordered_pv_sdk_receipt',
                       'query1_ordered_pv_objects_receipt','query1_ordered_pv_contract_source'}
        self.gqa_key_scalar_forward_query1=self.recipe.get('gqa_key_scalar_forward_query1',False)
        self.ordered_pv_register_query2_to32=self.recipe.get('ordered_pv_register_query2_to32',False)
        if type(self.ordered_pv_register_query2_to32) is not bool:
            raise ValueError('Dynamic ordered PV selection must be an explicit boolean')
        if self.ordered_pv_register_query2_to32:
            if (not self.ordered_pv_register_query1 or not self.dynamic_qk_invariant_query2_to32
                    or self.recipe.get('ordered_pv_register_query2_contract')!=
                    'original_Qwen_query2to32_PV_partial352_336_336_ordered_register8_v1'):
                raise ValueError('Dynamic PV requires the complete original QK and ordered q1 PV recipe')
            required|={'query2_ordered_pv_compiler','query2_ordered_pv_sdk_receipt',
                'query2_ordered_pv_objects_receipt','query2_ordered_pv_contract_source'}
        self.zero_tail_query1 = self.recipe.get('zero_tail_query1', False)
        if type(self.zero_tail_query1) is not bool:
            raise ValueError('q1 zero-tail selection must be an explicit boolean')
        if self.zero_tail_query1:
            from hexagon_zero_tail import CONTRACT, INPUTS, authenticate
            if (not self.ordered_pv_register_query2_to32 or not self.caller_hvx_byte_copy
                    or not self.reuse_exact_compiler_objects or not self.reuse_exact_prepared_linalg
                    or self.recipe.get('zero_tail_contract') != CONTRACT):
                raise ValueError('Complete original two-graph recipe and exact q1 contract required')
            required |= INPUTS
            authenticate(self.recipe['inputs'])
        self.qk_tail_query1=self.recipe.get('qk_tail_query1',False)
        if type(self.qk_tail_query1) is not bool:
            raise ValueError('QK-tail selection must be an explicit boolean')
        if self.qk_tail_query1:
            from hexagon_qk_tail import CONTRACT,INPUTS,authenticate
            if not self.zero_tail_query1 or self.recipe.get('qk_tail_contract')!=CONTRACT:
                raise ValueError('QK-tail requires the complete unchanged PV-zero-tail recipe')
            required|=INPUTS
            authenticate(self.recipe['inputs'])
        self.q2_qk_dispatch=self.recipe.get('q2_qk_dispatch',False)
        if type(self.q2_qk_dispatch) is not bool:
            raise ValueError('Q2 QK dispatch selection must be an explicit boolean')
        if self.q2_qk_dispatch:
            from hexagon_q2_qk_dispatch import CONTRACT,INPUTS,authenticate
            if (not self.qk_tail_query1 or not self.ordered_pv_register_query2_to32
                    or not self.reuse_exact_compiler_objects or not self.reuse_exact_prepared_linalg
                    or self.recipe.get('q2_qk_dispatch_contract')!=CONTRACT):
                raise ValueError('Q2 dispatch requires the complete unchanged accepted QK/PV recipe')
            required|=INPUTS
            authenticate(self.recipe['inputs'])
        self.global_hvx_memcpy=self.recipe.get('global_hvx_memcpy',False)
        if type(self.global_hvx_memcpy)is not bool:
            raise ValueError('Model-local global HVX copying must be an explicit boolean')
        if self.global_hvx_memcpy:
            from hexagon_global_memcpy import CONTRACT,INPUTS,authenticate
            if (not self.q2_qk_dispatch or not self.caller_hvx_byte_copy
                    or not self.fixed512_copy_query1 or not self.fixed512_copy_query2_to32
                    or self.recipe.get('global_hvx_memcpy_contract')!=CONTRACT):
                raise ValueError('Both complete accepted query domains and exact global-copy contract required')
            required|=INPUTS
            authenticate(self.recipe['inputs'])
        self.q2_qk_identical_tail=self.recipe.get('q2_qk_identical_tail',False)
        if type(self.q2_qk_identical_tail)is not bool:
            raise ValueError('Q2 identical-tail selection must be an explicit boolean')
        if self.q2_qk_identical_tail:
            from hexagon_q2_identical_tail import CONTRACT,INPUTS,authenticate
            if (not self.global_hvx_memcpy or not self.q2_qk_dispatch or not self.reuse_exact_compiler_objects
                    or not self.reuse_exact_prepared_linalg or self.recipe.get('q2_qk_identical_tail_contract')!=CONTRACT):
                raise ValueError('Complete accepted global-copy recipe and exact QK tail contract required')
            required|=INPUTS
            authenticate(self.recipe['inputs'])
        if type(self.gqa_key_scalar_forward_query1) is not bool:
            raise ValueError('gqa_key_scalar_forward_query1 must be an explicit boolean')
        if self.gqa_key_scalar_forward_query1:
            if (not self.gqa_dead_repeat_query1 or self.gqa_key_tile_query1 or self.recipe.get('gqa_key_scalar_forward_contract') !=
                    'original_Qwen_query1_immutable_K_scalar_source_index_equivalence_v1'):
                raise ValueError('Exact immutable K source forwarding must have a distinct final recipe')
            required|={'query1_gqa_key_scalar_forward_contract_source'}
        if set(self.recipe['inputs']) != required:
            raise ValueError('Complete captured Qwen compiler recipe required')
        if self.guarded_copy_query1:
            validate_runtime(self.recipe['inputs']['guarded_copy_runtime'],self.recipe['inputs']['guarded_copy_source'])
        if self.packed_multi_i8_query2_to32:
            from hexagon_multi_integer import validate_runtime
            validate_runtime(self.recipe['inputs']['packed_multi_i8_runtime'],self.recipe['inputs']['packed_multi_i8_nm'])
        if self.fixed512_copy_query2_to32:
            from hexagon_fixed512_memcpy import validate_runtime
            validate_runtime(self.recipe['inputs']['fixed512_copy_runtime'],
                             self.recipe['inputs']['fixed512_copy_source'],self.recipe['inputs']['fixed512_copy_nm'])
        for row in self.recipe['inputs'].values():
            if not Path(row['path']).is_absolute() or digest_file(row['path']) != row['sha256']:
                raise ValueError('Compiler recipe dependency changed')
        if self.integer_tiles_query1:
            from hexagon_q1_integer_tiles import validate_runtime
            inputs=self.recipe['inputs']
            validate_runtime(inputs['packed_i8_runtime'],inputs['packed_i8_nm'],
                inputs['integer_tiles_source'],inputs['integer_tiles_header'],inputs['integer_tiles_original_header'])
        if self.reuse_exact_compiler_objects:
            owned=Path(__file__).resolve().parents[3]/'tools/qwen3/reuse_accepted_compiler_objects.py'
            if (Path(self.recipe['inputs']['compiler_stage_reuse_source']['path']).resolve()!=owned.resolve()
                    or digest_file(owned)!=self.recipe['inputs']['compiler_stage_reuse_source']['sha256']):
                raise ValueError('Owned exact compiler-stage reuse source differs from recipe')
        if self.reuse_exact_prepared_linalg:
            owned=Path(__file__).resolve().parents[3]/'tools/qwen3/reuse_accepted_prepared_linalg.py'
            if (Path(self.recipe['inputs']['prepared_stage_reuse_source']['path']).resolve()!=owned.resolve()
                    or digest_file(owned)!=self.recipe['inputs']['prepared_stage_reuse_source']['sha256']):
                raise ValueError('Owned exact preparation authentication source differs from recipe')
        if type(self.recipe.get('arm_rope_trig_compat', False)) is not bool:
            raise ValueError('arm_rope_trig_compat must be an explicit boolean')
        self.reference = json.loads(Path(self.recipe['inputs']['reference']['path']).read_text())
        constants = json.loads(Path(self.recipe['inputs']['constants']['path']).read_text())
        manifest = json.loads(Path(self.recipe['inputs']['manifest']['path']).read_text())
        link_command = shlex.split(manifest['link_command'])
        dependencies = [Path(arg) for arg in link_command
                        if arg.startswith('/') and Path(arg).is_file()]
        dependencies.append(Path(self.reference['wrapper']['path']).parent /
                            'qwen3_sleef_exp_candidate.h')
        python_backend = Path(self.recipe['triton_python'])/'triton/backends/qcom_hexagon_backend'
        dependencies += [python_backend/name for name in (
            'hexagon_options.py', 'torch_mlir_hexagon_launcher.py', 'hexagon_launcher_base.py',
            'utils.py', 'hexagon_executor.py', 'hexagon_link_options.py', 'compiler.py')]
        local_backend = compiler_root()/'qcom_hexagon_backend/backend'
        dependencies += [local_backend/name for name in ('hexagon_dynamic_buffer_io.py', 'hexagon_buffer_io.py')]
        self.link_dependencies = {str(path): digest_file(path) for path in dependencies}
        self.linker = constants['inputs']['lld']
        if digest_file(self.linker['path']) != self.linker['sha256']:
            raise ValueError('Captured compiler linker changed')
        self.root = Path(root)
        self.frontend_root = Path(frontend_root)
        self.cache = CompiledArtifactCache(root)
        self.scripts = Path(__file__).resolve().parent
        names = ['hexagon_qwen_captured_machine_cache.py', 'prepare_captured_qwen_hexagon.py',
                 'probe_hexagon_dynamic_linalg.py', 'relink_captured_hexagon_object.py',
                 'apply_dynamic_bound_qk_policy.py', 'repair_dynamic_scatter_mlir_text.py',
                 'streamed_mlir_interchange.py', 'hexagon_compiled_artifact_cache.py',
                 'hexagon_python_source.py', 'compile_qwen3_artifact.py',
                 'prepare_captured_qwen_q1_caller.py', 'prepare_qwen_static_query_entry.py',
                 'link_qwen_static_query_entry.py', 'prepare_qwen_dynamic_debug_caller.py',
                 'hexagon_qwen_shape_policy.py', 'hexagon_qwen_numerical_policy.py']
        if self.silu_effects_query1:
            names.append('hexagon_silu_helper_effects.py')
        if self.guarded_copy_query1:
            names.append('hexagon_guarded_memref_copy.py')
        if self.packed_multi_i8_query2_to32:
            names.append('hexagon_multi_integer.py')
        if self.fixed512_copy_query2_to32:
            names.append('hexagon_fixed512_memcpy.py')
        if self.caller_hvx_byte_copy:
            names.append('hexagon_caller_byte_copy.py')
        if self.global_hvx_memcpy:
            names += ['hexagon_global_memcpy.py','hexagon_preserved_memcpy.py']
        if self.zero_tail_query1:
            names.append('hexagon_zero_tail.py')
        if self.qk_tail_query1:
            names.append('hexagon_qk_tail.py')
        if self.q2_qk_dispatch:
            names.append('hexagon_q2_qk_dispatch.py')
        if self.integer_tiles_query1:
            names.append('hexagon_q1_integer_tiles.py')
        if self.q2_qk_identical_tail:
            names.append('hexagon_q2_identical_tail.py')
        self.sources = {name: digest_file(self.scripts/name) for name in names}
        if self.reuse_exact_compiler_objects:
            self.sources['tools/qwen3/run_regression.py']=digest_file(
                Path(__file__).resolve().parents[3]/'tools/qwen3/run_regression.py')
        if self.reuse_exact_prepared_linalg:
            for name in ('reuse_accepted_prepared_linalg.py','reuse_accepted_compiler_objects.py',
                         'build.py','dependencies.lock.json'):
                self.sources['tools/qwen3/'+name]=digest_file(
                    Path(__file__).resolve().parents[3]/'tools/qwen3'/name)

    def get_or_compile(self, frontend, *, runtime_contract=None):
        if frontend['status'] != 'captured_AOT_Linalg_cached_NOT_machine_code':
            raise ValueError('Actual captured frontend artifact required')
        entry = self.frontend_root/(frontend['key']+'.json')
        disk_entry = json.loads(entry.read_text())
        if any(disk_entry[name] != frontend[name] for name in
               ('key', 'request', 'artifacts', 'build', 'status')):
            raise ValueError('Frontend entry and live capture binding differ')
        identity = frontend['request']
        specialized = runtime_contract is not None
        if specialized:
            from hexagon_qwen_shape_policy import contract_query_bounds
            bounds=contract_query_bounds(runtime_contract)
        else:
            bounds=None
        if self.packed_i8_query1 and not specialized:
            raise ValueError('Integer query1 route requires an actual captured shape contract')
        if self.silu_effects_query1 and not specialized:
            raise ValueError('SiLU effect query1 route requires an actual captured shape contract')
        if self.guarded_copy_query1 and not specialized:
            raise ValueError('Guarded byte-copy query1 route requires an actual captured shape contract')
        if (self.packed_multi_i8_query2_to32 or self.guarded_copy_query2_to32 or self.silu_effects_query2_to32 or self.fixed512_copy_query2_to32) and not specialized:
            raise ValueError('Multi-token optimization requires an actual captured shape contract')
        packed_i8=self.packed_i8_query1 and bounds==(1,1)
        packed_multi=self.packed_multi_i8_query2_to32 and bounds==(2,32)
        silu_effects=self.silu_effects_query1 and bounds==(1,1)
        multi_silu_effects=self.silu_effects_query2_to32 and bounds==(2,32)
        fixed512_copy=(self.fixed512_copy_query2_to32 and bounds==(2,32)
                       or self.fixed512_copy_query1 and bounds==(1,1))
        caller_hvx_byte_copy=getattr(self,'caller_hvx_byte_copy',False)
        if caller_hvx_byte_copy and bounds not in ((1,1),(2,32)):
            raise ValueError('Caller copying requires one of the original two bounded graph contracts')
        reuse_selected=self.reuse_exact_compiler_objects and bounds in ((1,1),(2,32))
        cse_query1=self.empty_tensor_cse_query1 and bounds==(1,1)
        view_query1=self.gqa_value_view_query1 and bounds==(1,1)
        forward_query1=self.gqa_value_forward_query1 and bounds==(1,1)
        key_query1=self.gqa_key_compact_query1 and bounds==(1,1)
        dead_query1=self.gqa_dead_repeat_query1 and bounds==(1,1)
        cse_dynamic=self.empty_tensor_cse_query2_to32 and bounds==(2,32)
        qk_dynamic=self.dynamic_qk_invariant_query2_to32 and bounds==(2,32)
        pv_query1=self.ordered_pv_register_query1 and bounds==(1,1)
        pv_dynamic=getattr(self,'ordered_pv_register_query2_to32',False) and bounds==(2,32)
        compiler_key='query2_ordered_pv_compiler' if pv_dynamic else 'query1_ordered_pv_compiler' if pv_query1 else 'query2_qk_invariant_compiler' if qk_dynamic else 'query2_cse_compiler' if cse_dynamic else 'query1_gqa_dead_repeat_compiler' if dead_query1 else 'query1_gqa_key_compact_compiler' if key_query1 else 'query1_gqa_value_forward_compiler' if forward_query1 else 'query1_gqa_value_view_compiler' if view_query1 else 'query1_cse_compiler' if cse_query1 else 'compiler'
        selected_compiler=self.recipe['inputs'][compiler_key]
        guarded_copy=(self.guarded_copy_query1 and bounds==(1,1) or
                      self.guarded_copy_query2_to32 and bounds==(2,32))
        if specialized and (not 62 <= len(runtime_contract.get('inputs', [])) <= 68
                or len(runtime_contract.get('outputs', [])) != 57
                or len(identity['retained_input_indices']) != len(runtime_contract['inputs'])):
            raise ValueError('Generated machine caller requires actual bounded captured contract')
        request = dict(schema=1, graph_sha256=identity['graph']['sha256'],
            parameters_sha256=identity['parameters_sha256'],
            compiler_sha256=selected_compiler['sha256'],
            linker_sha256=self.linker['sha256'],
            lowering_policy_sha256=hashlib.sha256(canonical(
                dict(recipe=self.recipe, sources=self.sources,
                     link_dependencies=self.link_dependencies,
                     linalg=frontend['artifacts'][0]))).hexdigest(),
            runtime_abi_sha256=hashlib.sha256(canonical(dict(
                reference=self.recipe['inputs']['reference'],
                tensor_inputs=self.reference['tensor_inputs'],
                scalar_inputs=self.reference['scalar_inputs'],
                captured_runtime_contract=runtime_contract))).hexdigest(),
            framework_sha256=identity['frontend_sha256'], target='hexagon-v73',
            options=dict(function_name=identity['function_name'],
                         retained_input_indices=identity['retained_input_indices']))

        model_object_codegen_invocations=0
        def compile_actual(folder):
            nonlocal model_object_codegen_invocations
            def run(script, arguments, log_name):
                with (folder/log_name).open('xb') as stream:
                    subprocess.run([sys.executable, str(self.scripts/script), *arguments],
                        stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=2400)

            def pair(name, row):
                flag = '--' + name.replace('_', '-')
                return [flag, row['path'], flag+'-sha256', row['sha256']]

            inputs = self.recipe['inputs']
            prepared = folder/'prepared'
            arguments = pair('entry', dict(path=str(entry), sha256=digest_file(entry)))
            for name in ('attention_optimizer', 'sums_optimizer', 'knorm_optimizer'):
                arguments += pair(name, inputs[name])
            if self.recipe.get('arm_rope_trig_compat', False):
                arguments += ['--arm-rope-trig']
            if packed_i8:
                arguments += pair('packed_i8_optimizer',inputs['packed_i8_optimizer'])
            if packed_multi:
                arguments += pair('packed_multi_i8_optimizer',inputs['packed_multi_i8_optimizer'])
            if silu_effects:
                arguments += pair('silu_effects_optimizer',inputs['silu_effects_optimizer'])
            if multi_silu_effects:
                arguments += pair('multi_silu_effects_optimizer',inputs['multi_silu_effects_optimizer'])
            preparation_reused=False
            preparation_artifacts=[]
            if self.reuse_exact_prepared_linalg and bounds in ((1,1),(2,32)):
                reuse=inputs['prepared_stage_reuse_source']
                reuse_args=['--config',inputs['compiler_stage_reuse_config']['path'],
                    '--accepted-cold',str(Path(inputs['compiler_stage_reuse_cold']['path']).parent),
                    '--frontend-entry',str(entry),'--recipe',str(self.recipe_path),
                    '--recipe-sha256',self.recipe_sha256,'--query-bounds',str(bounds[0]),str(bounds[1]),
                    '--stage-output','--output',str(prepared)]
                if self.share_immutable_prepared_artifacts:reuse_args.append('--share-immutable-artifacts')
                with (folder/'prepare.log').open('xb') as log:
                    checked=subprocess.run([sys.executable,reuse['path'],*reuse_args],stdout=log,
                        stderr=subprocess.STDOUT,timeout=2400)
                receipt=json.loads((prepared/'reuse-result.json').read_text())
                if checked.returncode==10:
                    if receipt['status']!='prepared_stage_cache_miss' or receipt['DSP_calls']:
                        raise ValueError('Invalid exact preparation cache-miss receipt')
                    missed=folder/'prepared-reuse-miss';prepared.rename(missed)
                    preparation_artifacts.append('prepared-reuse-miss/reuse-result.json')
                elif checked.returncode:
                    raise ValueError('Exact preparation authentication failed; no silent rebuild')
                else:
                    expected_status=('exact_accepted_prepared_Linalg_staged_NOT_pipeline_or_DSP_acceptance'
                                     if self.share_immutable_prepared_artifacts else
                                     'exact_accepted_prepared_Linalg_copied_NOT_pipeline_or_DSP_acceptance')
                    if (receipt['status']!=expected_status
                            or receipt['DSP_calls'] or receipt['new_prepare_optimizer_invocations']
                            or receipt['current_recipe']['sha256']!=self.recipe_sha256
                            or receipt['query_bounds']!=list(bounds)):
                        raise ValueError('Exact preparation reuse identity changed')
                    if self.share_immutable_prepared_artifacts and (
                            receipt.get('shared_immutable_artifacts') is not True or
                            any(r['mode']=='hardlink' and Path(r['source']).name not in
                                {'hexagon-linalg.mlirbc','adapted-source.mlir.gz'}
                                for r in receipt['staged_artifacts'])):
                        raise ValueError('Mutable prepared receipt sharing is forbidden')
                    preparation_reused=True;preparation_artifacts.append('prepared/reuse-result.json')
            if not preparation_reused:
                run('prepare_captured_qwen_hexagon.py', arguments + ['--output', str(prepared)],
                    'prepare-fallback.log' if self.reuse_exact_prepared_linalg else 'prepare.log')
            lowered = json.loads((prepared/'result.json').read_text())
            if packed_i8 and lowered.get('packed_integer_policy',{}).get('coverage')!=197:
                raise ValueError('Query1 integer policy coverage missing')
            if packed_multi and lowered.get('packed_multi_integer_policy',{}).get('coverage')!=197:
                raise ValueError('Query2..32 integer policy coverage missing')
            if not packed_multi and 'packed_multi_integer_policy' in lowered:
                raise ValueError('Multi-row integer policy escaped query2..32')
            if silu_effects and lowered.get('silu_effects',{}).get('coverage')!=28:
                raise ValueError('Query1 scoped SiLU effect coverage missing')
            if not silu_effects and 'silu_effects' in lowered:
                raise ValueError('Scoped SiLU effects escaped query1')
            if multi_silu_effects and lowered.get('multi_silu_effects',{}).get('coverage')!=28:
                raise ValueError('Query2..32 scoped SiLU effect coverage missing')
            if not multi_silu_effects and 'multi_silu_effects' in lowered:
                raise ValueError('Scoped SiLU effects escaped query2..32')
            objects = folder/'objects'
            arguments = pair('linalg', lowered['output'])
            for name in ('compiler', 'options_result', 'expected_objects_result'):
                arguments += pair(name, selected_compiler if name=='compiler' else inputs[name])
            arguments += ['--triton-python', self.recipe['triton_python'], '--emit', 'obj',
                          '--code-only', '--allow-code-difference', '--graph-kind', 'dynamic-bound',
                          '--output', str(objects)]
            reused_model=False
            if reuse_selected:
                reuse=inputs['compiler_stage_reuse_source']
                reuse_args=['--config',inputs['compiler_stage_reuse_config']['path'],
                    '--accepted-cold',str(Path(inputs['compiler_stage_reuse_cold']['path']).parent),
                    '--linalg',lowered['output']['path'],'--compiler',selected_compiler['path'],
                    '--options-result',inputs['options_result']['path'],
                    '--expected-objects-result',inputs['expected_objects_result']['path'],
                    '--triton-python',self.recipe['triton_python'],
                    '--query-bounds',str(bounds[0]),str(bounds[1]),'--stage-output','--output',str(objects)]
                if pv_query1:
                    for name in ('sdk','objects'):
                        row=inputs['query1_ordered_pv_'+name+'_receipt']
                        reuse_args+=['--ordered-pv-'+name+'-receipt',row['path'],
                                     '--ordered-pv-'+name+'-receipt-sha256',row['sha256']]
                    reuse_args+=['--ordered-pv-contract-source-sha256',
                                 inputs['query1_ordered_pv_contract_source']['sha256']]
                elif cse_query1 or view_query1:
                    prefix='gqa_dead_repeat' if dead_query1 else 'gqa_key_compact' if key_query1 else 'gqa_value_forward' if forward_query1 else 'gqa_value_view' if view_query1 else 'cse'
                    for name in ('sdk','objects'):
                        row=inputs['query1_'+prefix+'_'+name+'_receipt']
                        flag=prefix.replace('_','-')
                        reuse_args += ['--'+flag+'-'+name+'-receipt',row['path'],
                                       '--'+flag+'-'+name+'-receipt-sha256',row['sha256']]
                if cse_dynamic:
                    reuse_args+=['--dynamic-pv-register','--dynamic-pv-register-contract-source-sha256',
                        inputs['query2_ordered_pv_contract_source']['sha256']] if pv_dynamic else (
                        ['--dynamic-qk-invariant','--dynamic-qk-invariant-contract-source-sha256',
                        inputs['query2_qk_invariant_contract_source']['sha256']] if qk_dynamic else ['--dynamic-cse'])
                    for name in ('sdk','objects'):
                        row=inputs[('query2_ordered_pv_' if pv_dynamic else 'query2_qk_invariant_' if qk_dynamic else 'query2_cse_')+name+'_receipt']
                        reuse_args+=['--cse-'+name+'-receipt',row['path'],'--cse-'+name+'-receipt-sha256',row['sha256']]
                if self.gqa_key_scalar_forward_query1 and bounds==(1,1):
                    reuse_args += ['--gqa-key-scalar-forward','--gqa-key-scalar-forward-contract-source-sha256',
                                   inputs['query1_gqa_key_scalar_forward_contract_source']['sha256']]
                if self.gqa_key_tile_query1 and bounds==(1,1) and not pv_query1:
                    reuse_args += ['--gqa-key-tile-n32','--gqa-key-tile-contract-source-sha256',
                                   inputs['query1_gqa_key_tile_contract_source']['sha256']]
                with (folder/'compile.log').open('xb') as stream:
                    reuse_process=subprocess.run([sys.executable,reuse['path'],*reuse_args],stdout=stream,
                        stderr=subprocess.STDOUT,timeout=2400)
                if reuse_process.returncode==10:
                    miss=json.loads((objects/'reuse-result.json').read_text())
                    if (miss['status']!='compiler_stage_cache_miss'
                            or miss['new_machine_code_compilations']!=0 or miss['DSP_calls']!=0):
                        raise ValueError('Invalid compiler-stage cache miss receipt')
                    objects.rename(folder/'objects-reuse-miss')
                    if view_query1 or cse_dynamic:
                        raise ValueError('GQA copies are scoped to the authenticated original dynamic graph; changed graphs require a separately accepted recipe')
                    run('probe_hexagon_dynamic_linalg.py',arguments,'compile-after-reuse-miss.log')
                    model_object_codegen_invocations=1
                elif reuse_process.returncode:
                    raise subprocess.CalledProcessError(reuse_process.returncode,reuse_args)
                else:
                    reused_model=True
            else:
                run('probe_hexagon_dynamic_linalg.py', arguments, 'compile.log')
                model_object_codegen_invocations=1
            linked = folder/'linked'
            extra_artifacts = []
            if reused_model:
                reused=json.loads((objects/'reuse-result.json').read_text())
                if (reused['new_machine_code_compilations']!=0
                        or reused['model_inference_acceptance'] is not False):
                    raise ValueError('Explicit historical object reuse and zero new backend codegen required')
                extra_artifacts+=['objects/producer-result.json','objects/reuse-result.json']
            elif reuse_selected:
                extra_artifacts.append('objects-reuse-miss/reuse-result.json')
            zero_tail = self.zero_tail_query1 and bounds == (1, 1)
            if zero_tail:
                if not reused_model:
                    raise ValueError('q1 zero-tail requires successful exact original compiler-stage reuse')
                from hexagon_zero_tail import stage
                stage(inputs, objects, bounds)
                extra_artifacts += ['objects/before-zero-tail-result.json',
                                    'objects/zero-tail-reuse-result.json']
            qk_tail=self.qk_tail_query1 and bounds==(1,1)
            if qk_tail:
                from hexagon_qk_tail import stage
                stage(inputs,objects,bounds)
                extra_artifacts+=['objects/before-qk-tail-result.json','objects/qk-tail-reuse-result.json']
            q2_qk_dispatch=self.q2_qk_dispatch and bounds==(2,32)
            if q2_qk_dispatch:
                if not reused_model:
                    raise ValueError('Q2 dispatch requires successful exact original compiler-stage reuse')
                from hexagon_q2_qk_dispatch import stage
                stage(inputs,objects,bounds)
                extra_artifacts+=['objects/before-q2-qk-dispatch-result.json',
                                  'objects/q2-qk-dispatch-reuse-result.json']
            multi_runtime=None
            if packed_multi:
                from hexagon_multi_integer import validate_runtime
                validate_runtime(inputs['packed_multi_i8_runtime'],inputs['packed_multi_i8_nm'])
                runtime_folder=folder/'runtime';runtime_folder.mkdir()
                owned=runtime_folder/'packed-multi-i8-runtime.o'
                owned.write_bytes(Path(inputs['packed_multi_i8_runtime']['path']).read_bytes())
                multi_runtime=dict(path=str(owned),sha256=digest_file(owned))
                if multi_runtime['sha256']!=inputs['packed_multi_i8_runtime']['sha256']:
                    raise ValueError('Cached owned integer runtime changed')
                extra_artifacts.append('runtime/packed-multi-i8-runtime.o')
            if silu_effects:
                extra_artifacts.append('prepared/silu-effects-optimizer.log')
            if multi_silu_effects:
                extra_artifacts.append('prepared/multi-silu-effects-optimizer.log')
            if packed_i8 or packed_multi:
                prefix='packed_multi_i8' if packed_multi else 'packed_i8'
                symbol='_mlir_ciface___qwen_i8_multi_matmul' if packed_multi else '_mlir_ciface___qwen_i8_matmul'
                nm=inputs[prefix+'_nm']
                if digest_file(nm['path'])!=nm['sha256']:
                    raise ValueError('Pinned integer call inspector changed')
                compiled=json.loads((objects/'result.json').read_text())
                symbols=subprocess.run([nm['path'],'-u',compiled['objects'][0]['path']],
                    capture_output=True,text=True,check=True,timeout=30).stdout
                proof=folder/'packed-import-proof.json'
                retained=any(line.split()[-1:]==[symbol]
                    for line in symbols.splitlines())
                proof.write_bytes(canonical(dict(runtime_symbol_retained=retained,
                    nm=nm,object_sha256=compiled['objects'][0]['sha256'],
                    undefined_symbols=symbols,numerical_acceptance=False))+b'\n')
                if not retained:
                    raise ValueError('Query1 compiled object lost the integer runtime call')
                extra_artifacts.append('packed-import-proof.json')
                extra_artifacts.append('prepared/packed-multi-optimizer.log' if packed_multi else 'prepared/packed-optimizer.log')
            if specialized:
                contract_path = folder/'captured-contract.json'
                contract_path.write_bytes(canonical(runtime_contract)+b'\n')
                caller = folder/'caller'
                arguments = pair('entry', dict(path=str(entry), sha256=digest_file(entry)))
                arguments += pair('contract', dict(path=str(contract_path), sha256=digest_file(contract_path)))
                arguments += pair('prepared_graph', dict(path=str(prepared/'result.json'),
                                                         sha256=digest_file(prepared/'result.json')))
                arguments += pair('reference', inputs['reference'])
                if silu_effects or multi_silu_effects:
                    arguments += pair('silu_effects_guard',inputs['silu_effects_guard'])
                    if multi_silu_effects:arguments += ['--silu-effects-dynamic']
                run('prepare_captured_qwen_q1_caller.py', arguments +
                    ['--triton-python', self.recipe['triton_python'], '--output', str(caller)],
                    'caller.log')
                extra_artifacts += ['captured-contract.json', 'caller/result.json',
                    'caller/qwen_captured_q1_wrapper.cpp', 'caller/qwen3_sleef_exp_candidate.h']
                if silu_effects or multi_silu_effects:
                    extra_artifacts.append('caller/silu_effects_guard.h')
            arguments = pair('objects', dict(path=str(objects/'result.json'),
                                              sha256=digest_file(objects/'result.json')))
            for name in ('reference', 'manifest', 'constants'):
                row=inputs[name]
                if name=='manifest' and (packed_i8 or packed_multi):
                    manifest=json.loads(Path(row['path']).read_text())
                    argv=shlex.split(manifest['link_command'])
                    if argv.count('-o')!=1 or argv[-2]!='-o':
                        raise ValueError('Pinned link output position changed')
                    argv.insert(len(argv)-2,multi_runtime['path'] if packed_multi else inputs['packed_i8_runtime']['path'])
                    manifest['link_command']=shlex.join(argv)
                    packed_manifest=folder/'packed-manifest.json'
                    packed_manifest.write_bytes(canonical(manifest)+b'\n')
                    row=dict(path=str(packed_manifest),sha256=digest_file(packed_manifest))
                    extra_artifacts.append('packed-manifest.json')
                arguments += pair(name, row)
            if specialized:
                arguments += pair('caller', dict(path=str(caller/'result.json'),
                                                  sha256=digest_file(caller/'result.json')))
            if guarded_copy:
                from hexagon_guarded_memref_copy import validate_runtime
                validate_runtime(inputs['guarded_copy_runtime'],inputs['guarded_copy_source'])
                runtime_folder=folder/'runtime';runtime_folder.mkdir(exist_ok=True)
                local_runtime=runtime_folder/'guarded-copy-runtime.o'
                local_runtime.write_bytes(Path(inputs['guarded_copy_runtime']['path']).read_bytes())
                runtime_row=dict(path=str(local_runtime),sha256=digest_file(local_runtime))
                if runtime_row['sha256']!=inputs['guarded_copy_runtime']['sha256']:
                    raise ValueError('Cached owned byte-copy runtime changed')
                arguments += pair('guarded_copy_runtime',runtime_row)
                arguments += pair('guarded_copy_source',inputs['guarded_copy_source'])
                if bounds==(2,32):
                    arguments += ['--guarded-copy-dynamic']
                extra_artifacts.append('runtime/guarded-copy-runtime.o')
            if fixed512_copy:
                from hexagon_fixed512_memcpy import validate_runtime
                validate_runtime(inputs['fixed512_copy_runtime'],inputs['fixed512_copy_source'],inputs['fixed512_copy_nm'])
                runtime_folder=folder/'runtime';runtime_folder.mkdir(exist_ok=True)
                local_runtime=runtime_folder/'fixed512-memcpy.o'
                local_runtime.write_bytes(Path(inputs['fixed512_copy_runtime']['path']).read_bytes())
                runtime_row=dict(path=str(local_runtime),sha256=digest_file(local_runtime))
                if runtime_row['sha256']!=inputs['fixed512_copy_runtime']['sha256']:
                    raise ValueError('Cached owned fixed512 byte-copy runtime changed')
                arguments+=pair('fixed512_copy_runtime',runtime_row)
                arguments+=pair('fixed512_copy_source',inputs['fixed512_copy_source'])
                arguments+=pair('fixed512_copy_nm',inputs['fixed512_copy_nm'])
                if bounds==(1,1): arguments+=['--fixed512-copy-query1']
                extra_artifacts.append('runtime/fixed512-memcpy.o')
            if caller_hvx_byte_copy:
                from hexagon_caller_byte_copy import validate_runtime
                validate_runtime(inputs['caller_copy_runtime'],inputs['caller_copy_source'],inputs['fixed512_copy_nm'])
                runtime_folder=folder/'runtime';runtime_folder.mkdir(exist_ok=True)
                owned=runtime_folder/'caller-byte-copy.o'
                owned.write_bytes(Path(inputs['caller_copy_runtime']['path']).read_bytes())
                runtime_row=dict(path=str(owned),sha256=digest_file(owned))
                if runtime_row['sha256']!=inputs['caller_copy_runtime']['sha256']:
                    raise ValueError('Cached caller copy object changed')
                arguments+=pair('caller_copy_runtime',runtime_row)
                arguments+=pair('caller_copy_source',inputs['caller_copy_source'])
                arguments+=pair('caller_copy_nm',inputs['fixed512_copy_nm'])
                extra_artifacts.append('runtime/caller-byte-copy.o')
                extra_artifacts+=[('linked/caller-q1.cpp' if bounds==(1,1) else 'linked/caller-q2.cpp'),
                    'linked/caller-byte-copy-proof.json']
            if zero_tail:
                owned = folder/'runtime/zero-tail-runtime.o'
                owned.write_bytes(Path(inputs['zero_tail_runtime']['path']).read_bytes())
                runtime_row = dict(path=str(owned), sha256=digest_file(owned))
                if runtime_row['sha256'] != inputs['zero_tail_runtime']['sha256']:
                    raise ValueError('Owned zero-tail integer runtime changed')
                arguments += pair('zero_tail_runtime', runtime_row)
                arguments += pair('zero_tail_link', inputs['zero_tail_link'])
                extra_artifacts.append('runtime/zero-tail-runtime.o')
            if qk_tail:
                owned=folder/'runtime/qk-tail-runtime.o'
                owned.write_bytes(Path(inputs['qk_tail_runtime']['path']).read_bytes())
                runtime_row=dict(path=str(owned),sha256=digest_file(owned))
                if runtime_row['sha256']!=inputs['qk_tail_runtime']['sha256']:
                    raise ValueError('Owned QK-tail runtime changed')
                arguments+=pair('qk_tail_runtime',runtime_row)
                arguments+=pair('qk_tail_link',inputs['qk_tail_link'])
                extra_artifacts.append('runtime/qk-tail-runtime.o')
            run('link_qwen_static_query_entry.py' if specialized else 'relink_captured_hexagon_object.py',
                arguments + ['--output', str(linked)],
                'relink.log')
            if getattr(self,'global_hvx_memcpy',False):
                from hexagon_global_memcpy import relink
                extra_artifacts+=relink(linked,bounds,inputs)
            if getattr(self,'q2_qk_identical_tail',False) and bounds==(2,32):
                from hexagon_q2_identical_tail import relink
                extra_artifacts+=relink(linked,bounds,inputs)
            library = json.loads((linked/'result.json').read_text())['library']
            relative = str(Path(library['path']).relative_to(folder))
            return ['prepared/result.json', 'prepared/hexagon-linalg.mlirbc',
                    'prepared/adapted-source.mlir.gz', 'objects/result.json',
                    'objects/dynamic-hexagon-0.o', 'objects/compiler-diagnostics.txt',
                    'linked/result.json', relative, *extra_artifacts, *preparation_artifacts]

        result = self.cache.get_or_compile(request, compile_actual)
        linked_path = Path(result['directory'])/'linked/result.json'
        result['linked_result'] = dict(path=str(linked_path), sha256=digest_file(linked_path))
        result['machine_code_compilations'] = result['compiler_invocations']
        # A pipeline miss still compiles a fresh SDK caller and links a fresh
        # DSO. Count primary Linalg model codegen separately when it is reused.
        result['model_object_codegen_invocations'] = model_object_codegen_invocations
        return result
