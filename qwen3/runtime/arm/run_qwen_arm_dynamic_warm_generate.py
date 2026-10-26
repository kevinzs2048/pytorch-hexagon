"""ARM-native generate with actual dynamic captures and verified warm DSP code.

Default warm-only profile: unknown graphs fail before DSP execution. Optional
trusted compiler mailbox requests compilation for every actual AOT capture.
Generation and KV semantics stay in Transformers/AOT. Full logical KV still
crosses ARM/DSP each step, but there is no per-forward x86/SSH transfer.
"""
from qwen3.compiler_dependency import compiler_root
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from capture_qwen_symbolic_graph_carrier import sha, CONTEXT_SHA, GENERATION_SHA, POLICY_SHA
from hexagon_arm_dynamic_catalog import select_actual_capture
from hexagon_buffer_transport import HexagonLibraryBufferStream, buffer_capacities_for_contract
from hexagon_library_lifecycle import audit_library_events
from hexagon_sparse_packet_storage import probe_sparse_packet_storage, sparsify_verified_packet
from hexagon_host_timing import ArmForwardTimer
from hexagon_qwen_generation_budget import cache_demand
from hexagon_live_compile_mailbox import atomic_json
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import pack_torch_request, unpack_torch_outputs

CONSTANT_SHA = (
    '0d1b1298eb7339ed0d4324884d0cf1e7727fb63c7f02dc52cae8c4bcd0bf867c',
    '1700a01446763de1fbee46de08cd8dccb8d35ce3ee93eaab79483e6cb145b6a9',
    '66be6d41ea56fcec9d82571551460a8c53caf4fa4200ef1d9c87fa956ed83bde')


def validate_requests(requests, max_new_tokens, chunk):
    if (not isinstance(requests, list) or not 1 <= len(requests) <= 8
            or type(max_new_tokens) is not int or max_new_tokens < 1
            or type(chunk) is not int or not 1 <= chunk <= 32
            or any(not isinstance(row, list) or not row or cache_demand(len(row),max_new_tokens) > 1024
                   or any(type(x) is not int or not 0 <= x < 151936 for x in row) for row in requests)):
        raise ValueError('Bounded batch-one native requests required')
    maximum_forwards = sum((len(row)+chunk-1)//chunk+max_new_tokens-1 for row in requests)
    if maximum_forwards > 4096:
        raise ValueError('Requested session exceeds the current runtime limit of 4096 forwards')


def retain_packet(path, payload, mode):
    """Optional diagnostics only; never changes bytes sent to/from the DSP."""
    if mode not in ('full', 'hashes') or not isinstance(payload, bytes) or not payload:
        raise ValueError('Explicit packet recording mode and nonempty bytes required')
    digest = hashlib.sha256(payload).hexdigest()
    if mode == 'hashes':
        return digest, dict(retained=False, logical_bytes=len(payload),
                            reason='hashes_only_NOT_independently_replayable')
    with path.open('xb') as stream:
        stream.write(payload)
    storage = sparsify_verified_packet(path, digest)
    return digest, dict(storage, retained=True)


def validate_local_transport(catalog, mode):
    if not isinstance(catalog.get('runtime'), dict):
        raise ValueError('Pinned runtime description required')
    shared = catalog['runtime'].get('shared_memory_packets', False)
    if (mode not in ('files', 'shared-memory') or type(shared) is not bool
            or shared != (mode == 'shared-memory')):
        raise ValueError('Explicit local packet transport must match the pinned runtime')


def update_compile_totals(record):
    """Observed invocations, not warm-only defaults or capability claims."""
    totals = {}
    for key in ('frontend_invocations', 'machine_code_compilations'):
        values = [row[key] for row in record['compiler_transactions']]
        if any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError('Each captured graph needs actual zero/one compiler invocation counts')
        totals[key] = sum(values)
    record.update(totals, automatic_cold_compilation=totals['machine_code_compilations'] > 0)


def tokenize_prompts(hf, prompts, *, chat=False):
    if (type(chat) is not bool or not isinstance(prompts, list) or not 1 <= len(prompts) <= 8
            or any(type(text) is not str or not text or len(text.encode('utf-8')) > 65536 for text in prompts)):
        raise ValueError('One to eight bounded nonempty text prompts required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(hf), local_files_only=True, trust_remote_code=False)
    if chat:
        rows = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False) for text in prompts]
    else:
        rows = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    files = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
    provenance = dict(kind='native_HuggingFace_tokenizer', prompts=prompts, chat_template=chat,
        enable_thinking=False if chat else None, tokenizer_class=type(tokenizer).__name__,
        files=[dict(name=name, sha256=sha(hf/name)) for name in files])
    return tokenizer, rows, provenance


def terminal_observation(generated, max_new_tokens, eos_ids):
    """Describe observed output; do not implement or replace stopping criteria."""
    if not generated or len(generated) > max_new_tokens:
        raise ValueError('Native generated length outside requested bound')
    return dict(last_token_is_native_eos=generated[-1] in eos_ids,
                reached_requested_token_limit=len(generated) == max_new_tokens,
                generated_tokens=len(generated))


def load_model(context_path, hf):
    import torch
    from transformers import Qwen3ForCausalLM
    from transformers.generation import utils as generation_utils
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    if sha(context_path) != CONTEXT_SHA or sha(generation_utils.__file__) != GENERATION_SHA:
        raise ValueError('Original context and previously checked generation source required')
    context = json.loads(context_path.read_text())
    if any(importlib.metadata.version(name) != context['versions'][name]
           for name in ('torch', 'torchao', 'transformers')):
        raise ValueError('Pinned native framework versions required')
    if sha(hf/'model.safetensors') != context['checkpoint']['sha256']:
        raise ValueError('Original checkpoint changed')
    model = Qwen3ForCausalLM.from_pretrained(str(hf), local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval().requires_grad_(False)
    actual, expected = json.loads(json.dumps(model.config.to_dict())), dict(context['model_config'])
    actual.pop('_name_or_path', None)
    expected.pop('_name_or_path', None)
    if actual != expected:
        raise ValueError('Native model configuration changed')
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        raw = value.detach().contiguous().view(torch.uint8).numpy()
        digest.update((name+':'+hashlib.sha256(memoryview(raw).cast('B')).hexdigest()+'\n').encode())
    if digest.hexdigest() != context['historical_parameter_sha256']:
        raise ValueError('Original named parameter bytes changed')
    quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
    return model


def validate_workdir(workdir):
    """Reject a bad deployment path before loading/tokenizing the model."""
    if (not workdir.is_absolute() or workdir.parent != Path('/dev/shm')
            or not workdir.name.startswith('qwen3-arm-dynamic-') or workdir.exists() or workdir.is_symlink()):
        raise ValueError('Fresh explicitly scoped ARM work directory required')


def stage(catalog, workdir, *, include_code=True, constant_load_split_proofs=None):
    # Recheck immediately before deployment in case the path appeared meanwhile.
    validate_workdir(workdir)
    control = Path(catalog['control_deployment'])
    if control.parent != Path('/dev/shm') or not control.name.startswith('qwen3-'):
        raise ValueError('Previously verified control deployment required')
    active = subprocess.run(['pgrep', '-x', 'buffer_stream'], capture_output=True, timeout=10)
    if active.returncode not in (0, 1) or active.stdout.strip():
        raise ValueError('Another DSP buffer runner is active')
    workdir.mkdir()
    (workdir/'rfsa').mkdir()
    for row in catalog['runtime']['artifacts']:
        name = row['name']
        if name not in ('buffer_stream', 'libhexagon_buffer_runner_skel.so'):
            raise ValueError('Unexpected runtime deployment file')
        relative = Path(name) if name == 'buffer_stream' else Path('rfsa')/name
        source = control/relative
        if sha(source) != row['sha256']:
            raise ValueError('Actual ARM runtime changed')
        shutil.copy2(source, workdir/relative)
        if sha(workdir/relative) != row['sha256']:
            raise ValueError('Staged runtime changed')
    constant_layout=[]
    for index, expected in enumerate(CONSTANT_SHA, 1):
        name = f'lib_mlir_ciface_Qwen3DynamicBoundProbe-consts-{index}.so'
        if sha(control/name) != expected:
            raise ValueError('Actual DSP constant library changed')
        source=control/name
        if constant_load_split_proofs:
            from tools.qwen3.split_constant_load_segments import verified_proof
            proof_path=constant_load_split_proofs/('proof-const'+str(index)+'.json')
            proof=json.loads(proof_path.read_text());candidate=Path(proof['candidate']['path'])
            if (candidate.parent.parent!=Path('/dev/shm')
                    or not candidate.parent.name.startswith('qwen3-arm-dynamic-')
                    or candidate.name!=name or candidate.is_symlink()):
                raise ValueError('Explicit bounded-loader constant deployment required')
            constant_layout.append(verified_proof(source,candidate,proof_path,expected))
            source=candidate
        (workdir/name).symlink_to(source)
    for entry in catalog['entries'] if include_code else ():
        library = entry['library']
        source = Path(library['remote_path'])
        if (source.parent.parent != Path('/dev/shm') or not source.parent.name.startswith('qwen3-')
                or source.name != library['sha256']+'.so' or sha(source) != library['sha256']):
            raise ValueError('Actual compiled control library changed')
        destination = workdir/(library['sha256']+'.so')
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha(destination) != library['sha256']:
            raise ValueError('Staged compiled library changed')
    return constant_layout


def run(args, catalog, requests, record, save):
    bootstrap=None
    if args.startup_timing:
        from tools.qwen3.startup_timing import BootstrapTimer,CaptureTimer
        bootstrap=BootstrapTimer()
        record['startup_timing']=dict(clock='perf_counter_ns',captures=[],
            scope='ARM_bootstrap_and_actual_capture_callback',
            nested_framework_spans_not_added=True)
    import torch
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from transformers import CompileConfig
    from hexagon_native1024_cache_setup import prepare_native1024_cache
    from hexagon_native_silu import preserve_native_silu
    from hexagon_runtime_decompositions import load_runtime_decompositions
    from hexagon_dynamic_module_constants import DynamicParameterBinding
    from hexagon_module_scalar_binding import bind_module_scalar_attributes
    from hexagon_captured_linalg_cache import captured_identity
    validate_local_transport(catalog, args.local_packet_transport)
    stream_type = HexagonLibraryBufferStream
    if args.local_packet_transport == 'shared-memory':
        from hexagon_shared_packet_transport import HexagonSharedPacketStream
        stream_type = HexagonSharedPacketStream
    decode_outputs = unpack_torch_outputs
    pack_inputs = pack_torch_request
    if args.input_packer == 'joined':
        from tools.qwen3.joined_input_packet import pack_torch_request_joined
        pack_inputs = pack_torch_request_joined
    if args.output_decoder == 'owned-once':
        from qcom_hexagon_backend.backend.hexagon_dynamic_tensor_runtime import unpack_owned_torch_outputs
        decode_outputs = unpack_owned_torch_outputs
    torch.set_num_threads(2)
    parameter_pool=None
    if args.parameter_check=='exact-bytes':
        from tools.qwen3.exact_parameter_bytes import ExactParameterBytePool
        parameter_pool=ExactParameterBytePool()
    record['parameter_validation']=dict(mode=args.parameter_check,
        all_logical_parameter_bytes_checked_each_forward=True,version_only_check=False,
        initial_compiler_SHA_pins_preserved=True)
    if bootstrap is not None:bootstrap.mark('import_and_configure')
    model = load_model(args.context, args.hf)
    if bootstrap is not None:bootstrap.mark('load_and_verify_native_model')
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
    record['generation_options'] = dict(max_new_tokens=args.max_new_tokens,
        prefill_chunk_size=args.chunk_size, do_sample=False, native_eos_token_ids=eos_ids,
        generation_config_sha256=sha(args.hf/'generation_config.json'))
    record['generation_options']['specialize_python_floats']=args.specialize_floats
    cache = prepare_native1024_cache(model)
    record['generation_options']['native_cache_capacity'] = cache.max_cache_len
    original = (model.forward.__func__, model.generate.__func__)
    table = preserve_native_silu(load_runtime_decompositions(args.decompositions,
        expected_sha256=POLICY_SHA, expected_torch_version='2.12.0+cpu',
        expected_torch_git='7661cd9c6b841b62b7f411aa52ec51f05457263b'))
    capacities = [buffer_capacities_for_contract(e['contract']) for e in catalog['entries']]
    caps = {key: max(row[key] for row in capacities) for key in ('input_bytes', 'output_bytes')}
    record.update(status='capturing_ARM_native_generate', buffer_capacities=caps,
                  torch_version=torch.__version__, torch_git=torch.version.git_version)
    save()
    # In live mode no precompiled model entry is predeployed. Only the fixed
    # runtime/weight bundle is staged; code must come from a compiler reply.
    if bootstrap is not None:bootstrap.mark('prepare_cache_and_decompositions')
    constant_layout=stage(catalog, args.workdir, include_code=not args.compiler_mailbox,
        constant_load_split_proofs=args.constant_load_split_proofs)
    if constant_layout:
        loader_mode=getattr(args,'constant_loader_mode','diagnostic')
        if loader_mode=='verified-payload-preserving' and any(row['max_load_bytes']!=256*1024**2 for row in constant_layout):
            raise ValueError('Verified payload policy requires the firmware-compatible 256MiB LOAD bound')
        record['constant_loader_layout']=dict(kind=('bounded_readonly_LOAD_payload_preserving_v1'
            if loader_mode=='verified-payload-preserving' else 'bounded_readonly_LOAD_diagnostic'),
            original_weights_code_and_symbol_values_preserved=True,libraries=constant_layout,
            loader_failure_fixed=False,performance_acceptance=False)
        save()
    if bootstrap is not None:
        bootstrap.mark('stage_runtime_and_constants')
        record['startup_timing']['bootstrap']=bootstrap.completed()
        save()
    env = dict(os.environ)
    env['ADSP_LIBRARY_PATH'] = (str(args.workdir)+';'+str(args.workdir/'rfsa')+
        ';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp')
    env['LD_LIBRARY_PATH'] = str(args.workdir)
    stream = None
    selected = None
    request_index = 0
    request_start = 0
    generate_started_ns = 0
    previous_cwd = Path.cwd()
    os.chdir(args.workdir)
    try:
        with ExitStack() as stack:
            def select(entry):
                nonlocal stream, selected
                expected = entry['library']['sha256']
                path = args.workdir/(expected+'.so')
                if selected == expected:
                    return
                if sha(path) != expected:
                    raise ValueError('Library changed before actual invocation')
                if stream is None:
                    argv = ['flock', '-n', '/tmp/qwen3-qcs9075-probe.lock',
                        str(args.workdir/'buffer_stream'), str(path), str(caps['input_bytes']), str(caps['output_bytes'])]
                    stream = stack.enter_context(stream_type(argv, env, args.output,
                        **caps, initial_library=str(path), timeout=240))
                    record['ready'] = stream.start()
                else:
                    stream.select_library(str(path))
                selected = expected

            def backend(dynamo_graph, dynamo_inputs):
                capture_timer=CaptureTimer() if args.startup_timing else None
                request_to_backend_ns=(capture_timer.start-generate_started_ns
                    if capture_timer is not None else None)
                prior_request_forwards=len(record['forwards'])-request_start
                def compiler(aten, examples):
                    if capture_timer is not None:capture_timer.mark('AOT_prepare_graph')
                    tracing = torch._guards.TracingContext.get()
                    binding = DynamicParameterBinding(dynamo_graph, dynamo_inputs, aten, examples,
                        input_info=tracing.fw_metadata.input_info, retain_snapshots=False)
                    bind_module_scalar_attributes(binding, dynamo_graph, dynamo_inputs, aten, examples,
                        input_info=tracing.fw_metadata.input_info, roots={'self': model},
                        allowed_attributes={'variance_epsilon', 'scaling', 'attention_scaling'},
                        record_kind='immutable_Qwen3_module_scalar')
                    if capture_timer is not None:capture_timer.mark('bind_immutable_inputs')
                    identity = captured_identity(aten)
                    if capture_timer is not None:capture_timer.mark('identify_live_graph')
                    constants_guard=None
                    if args.specialize_floats:
                        from tools.qwen3.guarded_module_constants import GuardedModuleConstants
                        constants_guard=GuardedModuleConstants(model)
                        record['parameter_validation']['guarded_python_float_constants']=dict(
                            count=len(constants_guard.records),FP64_bytes_per_constant=8,
                            ledger_sha256=constants_guard.signature,
                            owner_identity_and_all_bytes_checked_each_forward=True)
                    if args.compiler_mailbox:
                        from hexagon_live_compile_mailbox import (
                            publish_capture, wait_for_compilation, checked_library_path)
                        request_dir = args.compiler_mailbox/f"capture-{len(record['captures']):02d}"
                        request_sha = publish_capture(request_dir, model, dynamo_graph,
                            dynamo_inputs, aten, examples, binding, prior_dsp_calls=record['DSP_calls'],
                            guarded_constants=constants_guard)
                        record['compile_requests_submitted'] += 1
                        record['pending_compilation'] = dict(path=str(request_dir), sha256=request_sha)
                        save()
                        if capture_timer is not None:capture_timer.mark('publish_live_capture')
                        candidate, compiled = wait_for_compilation(request_dir, request_sha)
                        if capture_timer is not None:capture_timer.mark('wait_for_compiler_host')
                        # Rebind the response to this live graph and all native
                        # immutable inputs, not a query-length lookup.
                        entry = select_actual_capture({'entries': [candidate]}, identity, binding)
                        required = buffer_capacities_for_contract(entry['contract'])
                        if any(required[key] > caps[key] for key in caps):
                            raise ValueError('New compiled ABI exceeds this session capacity; no fallback')
                        library = entry['library']
                        if args.compiler_machine_cache:
                            # Explicit compatibility mode for prior compiler workers.
                            source = checked_library_path(args.compiler_machine_cache,
                                library['relative_to_machine_cache'], library['sha256'])
                        else:
                            from hexagon_compile_library_reply import checked_reply_library
                            source = checked_reply_library(request_dir, entry)
                        target = args.workdir/(library['sha256']+'.so')
                        if not target.exists():
                            shutil.copy2(source, target)
                        if sha(target) != library['sha256']:
                            raise ValueError('Returned compiler library changed in staging')
                        record['compiler_transactions'].append(dict(producer_sha256=request_sha,
                            frontend_cache_hit=compiled['frontend_cache']['cache_hit'],
                            machine_cache_hit=compiled['machine_cache']['cache_hit'],
                            frontend_invocations=compiled['frontend_invocations'],
                            machine_code_compilations=compiled['machine_code_compilations']))
                        update_compile_totals(record)
                        record.pop('pending_compilation', None)
                        if capture_timer is not None:capture_timer.mark('verify_and_stage_reply')
                    else:
                        entry = select_actual_capture(catalog, identity, binding)
                    contract = entry['contract']
                    capture_index = len(record['captures'])
                    record['captures'].append(dict(graph=identity, frontend_key=entry['frontend_key'],
                        machine_key=entry['machine_key'], DSO_sha256=entry['library']['sha256'],
                        retained_input_indices=binding.retained, contract=contract))
                    save()

                    parameter_guard=parameter_pool.bind(binding) if parameter_pool is not None else binding
                    if capture_timer is not None:
                        capture_timer.mark('record_and_bind_callback')
                        record['startup_timing']['captures'].append(dict(
                            capture_index=capture_index,request_index=request_index,
                            request_start_to_backend_enter_ns=request_to_backend_ns,
                            prior_forwards_in_this_request=prior_request_forwards,
                            sequential_callback=capture_timer.completed()))
                        save()
                    def call(*values):
                        timer = ArmForwardTimer()
                        if constants_guard is not None:
                            constants_guard.check()
                            validation=record['parameter_validation']
                            validation['guarded_python_float_forwards_checked']=validation.get(
                                'guarded_python_float_forwards_checked',0)+1
                        bound = parameter_guard.check(values)
                        if parameter_pool is not None:
                            record['parameter_validation']['owned_immutable_snapshot_bytes']=parameter_pool.snapshot_bytes
                        timer.mark('validate_frozen_parameters')
                        tensors = [value for value in bound if isinstance(value, torch.Tensor)]
                        packet, layout, dimensions = pack_inputs(contract, bound)
                        query = dimensions['query']
                        positions = tensors[1].reshape(-1).tolist()
                        start = sum(row['query'] for row in record['forwards'][request_start:])
                        if (query > args.chunk_size or positions != list(range(start, start+query))
                                or tensors[2].reshape(-1).tolist() != positions):
                            raise ValueError('Native cache position trace changed')
                        timer.mark('pack_and_validate_inputs')
                        select(entry)
                        timer.mark('select_compiled_library')
                        step = len(record['forwards'])
                        folder = args.output/f'forward-{step:02d}-q{query}'
                        if args.packet_recording == 'full':
                            folder.mkdir()
                        input_file = folder/'input.bin'
                        input_sha, input_storage = retain_packet(input_file, packet, args.packet_recording)
                        timer.mark('record_request')
                        if args.local_packet_transport == 'shared-memory':
                            response, event = stream.invoke_packet(packet, layout['output_bytes'])
                        else:
                            (args.workdir/'input.bin').write_bytes(packet)
                            if sha(args.workdir/'input.bin') != input_sha:
                                raise ValueError('Local ARM staging changed request bytes')
                            event = stream.invoke_sized(len(packet), layout['output_bytes'])
                        record['DSP_calls'] += 1
                        save()
                        if args.local_packet_transport == 'files':
                            response = (args.workdir/'output.bin').read_bytes()
                        if len(response) != layout['output_bytes']:
                            raise ValueError('DSP response length changed')
                        timer.mark('exchange_with_AP')
                        output_file = folder/'output.bin'
                        output_sha, output_storage = retain_packet(output_file, response, args.packet_recording)
                        sparse = dict(input=input_storage, output=output_storage)
                        timer.mark('record_response')
                        if args.local_packet_transport == 'files':
                            (args.workdir/'input.bin').unlink()
                            (args.workdir/'output.bin').unlink()
                        timer.mark('release_file_scratch')
                        outputs = decode_outputs(contract, dimensions, response)
                        if len(outputs) != len(contract['outputs']):
                            raise ValueError('Captured output ABI changed')
                        timer.mark('reconstruct_owned_outputs')
                        record['forwards'].append(dict(step=step, query=query, request_index=request_index,
                            AOT_capture_index=capture_index, DSP_library_slot=stream.current_library,
                            DSO_sha256=selected, input_sha256=input_sha, output_sha256=output_sha,
                            cache_positions=positions, input_token_ids=tensors[0].reshape(-1).tolist(),
                            logits_top_token=int(outputs[-1].reshape(-1).argmax()),
                            invoke=event, packet_storage=sparse, arm_timing=timer.completed()))
                        save()
                        return outputs
                    return make_boxed_func(call)
                def no_backward(*unused):
                    raise RuntimeError('Inference-only ARM profile')
                return aot_autograd(fw_compiler=compiler, inference_compiler=compiler,
                    bw_compiler=no_backward, decompositions=table)(dynamo_graph, dynamo_inputs)

            def no_eager(module, positional, kwargs):
                if not torch.compiler.is_compiling():
                    raise RuntimeError('Eager CPU model execution forbidden')
            hook = model.register_forward_pre_hook(no_eager, with_kwargs=True)
            try:
                cfg = CompileConfig(backend=backend, fullgraph=True, dynamic=True, mode='default')
                cfg._compile_all_devices = True
                for request_index, prompt in enumerate(requests):
                    request_start = len(record['forwards'])
                    ids = torch.tensor([prompt], dtype=torch.int64)
                    timing = None
                    timing_options = {}
                    from hexagon_qwen_shape_policy import compiled_query_scope
                    from tools.qwen3.capture_configuration import capture_configuration
                    with compiled_query_scope(model), torch.no_grad(), capture_configuration(specialize_floats=args.specialize_floats):
                        if args.generation_timing:
                            from tools.qwen3.generation_timing import NativeGenerationTimer
                            timing = NativeGenerationTimer()
                            timing_options['streamer'] = timing
                        if args.startup_timing:generate_started_ns=time.perf_counter_ns()
                        generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                            cache_implementation='static', compile_config=cfg, do_sample=False,
                            max_new_tokens=args.max_new_tokens, prefill_chunk_size=args.chunk_size,
                            return_dict_in_generate=True, output_logits=True, **timing_options)
                        if timing is not None:
                            timing.mark_return()
                    new = generated.sequences[0, len(prompt):].tolist()
                    if (not 1 <= len(new) <= args.max_new_tokens or model._cache is not cache
                            or sum(row['query'] for row in record['forwards'][request_start:]) != len(prompt)+len(new)-1
                            or original != (model.forward.__func__, model.generate.__func__)):
                        raise ValueError('Native generation/cache lifetime differs')
                    record['requests'].append(dict(index=request_index, prompt_ids=prompt, generated_ids=new,
                        first_forward=request_start, end_forward=len(record['forwards']),
                        terminal_observation=terminal_observation(new, args.max_new_tokens, eos_ids)))
                    if timing is not None:
                        record['requests'][-1]['generation_timing'] = timing.completed(prompt,new)
                    if args.tokenizer is not None:
                        record['requests'][-1]['generated_text'] = args.tokenizer.decode(new, skip_special_tokens=True)
                    save()
            finally:
                hook.remove()
        if stream is None or stream.process_exit_code != 0:
            raise ValueError('Expected completed local DSP session')
        record.update(status=('ARM_native_dynamic_compiler_handoff_generate_completed'
            if args.compiler_mailbox else 'ARM_native_dynamic_warm_generate_completed'), native_generate=True,
            stream_events=stream.events, process_exit_code=stream.process_exit_code,
            library_lifecycle=audit_library_events(stream.events,
                [row['DSP_library_slot'] for row in record['forwards']]))
        if args.startup_timing:
            from torch._dynamo.utils import compilation_time_metrics
            record['startup_timing']['framework_nested_seconds']={
                key:list(values) for key,values in compilation_time_metrics.items()}
    finally:
        os.chdir(previous_cwd)
        if stream is not None:
            record.update(stream_events=stream.events, process_exit_code=stream.process_exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--hf', type=Path, required=True)
    parser.add_argument('--decompositions', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--requests-json', help='Explicit token IDs for controlled regression')
    inputs.add_argument('--prompts-json', help='Text prompts tokenized by the native local model tokenizer')
    parser.add_argument('--chat-template', action='store_true',
                        help='Use the native user-chat template, with thinking disabled; text inputs only')
    parser.add_argument('--max-new-tokens', type=int, default=2,
                        help='Native generation budget; prompt plus budget minus one must fit KV capacity 1024')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='Prefill tokens per forward (1..32), independent of generation budget')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet-recording', choices=('full', 'hashes'), default='full',
                        help='full keeps replayable diagnostic packets; hashes avoids archiving KV but cannot pass full-packet audits')
    parser.add_argument('--local-packet-transport', choices=('files', 'shared-memory'), default='files',
                        help='Python/AP staging only; must match catalog runtime; FastRPC/KV payloads unchanged')
    parser.add_argument('--output-decoder', choices=('established', 'owned-once'), default='established',
                        help='Opt-in fewer host copies; both decoders return independently owned CPU tensors')
    parser.add_argument('--compiler-mailbox', type=Path,
                        help='Optional fresh trusted shared directory; request compilation for EVERY capture')
    parser.add_argument('--generation-timing', action='store_true',
                        help='Observe native selected-token streamer times including capture and AOT writeback')
    parser.add_argument('--input-packer', choices=('established', 'joined'), default='established',
                        help='Opt-in one-allocation host serialization of the unchanged request ABI')
    parser.add_argument('--parameter-check',choices=('sha256','exact-bytes'),default='sha256',
                        help='Both modes check all frozen parameter bytes every forward, including .data writes')
    parser.add_argument('--startup-timing',action='store_true',
                        help='Separate ARM capture/AOT/host wait; nested framework intervals are not added')
    parser.add_argument('--specialize-floats',action='store_true',
                        help='Opt-in scoped native Dynamo guards for Python float constants; tensor/query inputs stay dynamic')
    parser.add_argument('--compiler-machine-cache', type=Path,
                        help='Legacy compatibility only: ARM-visible cache; omit to use returned DSO')
    parser.add_argument('--constant-load-split-proofs',type=Path,
                        help='Verify all original constant bytes and bounded ELF LOAD proofs before staging')
    parser.add_argument('--constant-loader-mode',choices=('diagnostic','verified-payload-preserving'),default='diagnostic',
                        help='Explicit common payload-preserving loader policy; full model acceptance remains separate')
    args = parser.parse_args()
    if args.constant_loader_mode!='diagnostic' and not args.constant_load_split_proofs:
        parser.error('Verified payload loader policy requires actual constant proofs')
    if args.startup_timing and not args.compiler_mailbox:
        parser.error('Startup capture timing requires actual compiler handoff')
    validate_workdir(args.workdir)
    if args.chat_template and args.prompts_json is None:
        parser.error('Chat template requires text prompts')
    args.tokenizer = None
    if args.compiler_machine_cache and not args.compiler_mailbox:
        parser.error('Legacy machine cache requires compiler mailbox')
    if args.compiler_mailbox and (not args.compiler_mailbox.is_absolute()
            or args.compiler_mailbox.is_symlink() or not args.compiler_mailbox.is_dir()
            or any(args.compiler_mailbox.iterdir())):
        parser.error('Fresh private existing mailbox required')
    if args.compiler_machine_cache and (not args.compiler_machine_cache.is_absolute()
            or not args.compiler_machine_cache.is_dir()):
        parser.error('Explicit legacy machine cache must be an existing absolute directory')
    minimum_evidence_space = (1024 if args.packet_recording == 'full' else 128)*1024**2
    if (platform.machine() != 'aarch64' or sha(args.catalog) != args.catalog_sha256
            or not args.output.is_absolute() or args.output.exists() or args.output.is_symlink()
            or not args.output.parent.is_dir()
            or shutil.disk_usage(args.output.parent).free < minimum_evidence_space):
        raise ValueError('Actual ARM, pinned catalog and fresh evidence directory with recording-space margin required')
    catalog = json.loads(args.catalog.read_text())
    if catalog['status'] != 'warm_ARM_dynamic_catalog_NOT_executed' or catalog['schema'] != 1:
        raise ValueError('Explicit warm catalog required')
    args.output.mkdir()
    record = dict(status='admitting_ARM_warm_native_generate', source_sha256=sha(__file__),
        catalog=dict(path=str(args.catalog), sha256=args.catalog_sha256),
        architecture=platform.machine(), captures=[], forwards=[], requests=[], DSP_calls=0,
        native_generate=False, numerical_acceptance=False, CPU_model_fallback=False,
        automatic_cold_compilation=False, machine_code_compilations=0, frontend_invocations=0,
        live_compiler_handoff=bool(args.compiler_mailbox), compiler_transactions=[],
        compiler_library_transport=('legacy_shared_cache' if args.compiler_machine_cache
                                    else 'request_scoped_reply' if args.compiler_mailbox else None),
        compile_requests_submitted=0,
        packet_recording=args.packet_recording, complete_raw_packets_retained=args.packet_recording == 'full',
        local_packet_transport=args.local_packet_transport,
        output_decoder=args.output_decoder,
        input_packer=args.input_packer,
        input_packing_source_sha256=(sha(Path(__file__).resolve().parents[3]/
            'tools/qwen3/joined_input_packet.py') if args.input_packer == 'joined' else
            sha(compiler_root()/
                'qcom_hexagon_backend/backend/hexagon_dynamic_buffer_io.py')),
        arm_forward_timing=dict(schema='arm_local_forward_stage_timing_v1',
            includes_compilation_or_generate=False, includes_AOT_output_writeback=False),
        input_source={'kind': 'input_admission_pending'},
        per_forward_SSH=False, DSP_resident_KV=False, full_ARM_KV_mirror=True)
    last_progress = None
    def save():
        nonlocal last_progress
        atomic_json(args.output/'result.json',record)
        if args.compiler_mailbox:
            counters = (len(record['forwards']), record['compile_requests_submitted'])
            if counters != last_progress:
                from hexagon_compile_session_progress import publish_progress
                publish_progress(args.compiler_mailbox, *counters)
                last_progress = counters
    save()
    try:
        validate_local_transport(catalog, args.local_packet_transport)
        input_source = {'kind': 'explicit_token_ids'}
        if args.prompts_json is not None:
            args.tokenizer, requests, input_source = tokenize_prompts(
                args.hf, json.loads(args.prompts_json), chat=args.chat_template)
        else:
            requests = json.loads(args.requests_json)
        validate_requests(requests, args.max_new_tokens, args.chunk_size)
        record['input_source'] = input_source
        save()
        if args.packet_recording == 'full':
            record['sparse_storage_preflight'] = probe_sparse_packet_storage(args.output)
        run(args, catalog, requests, record, save)
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()
        if args.compiler_mailbox:
            atomic_json(args.compiler_mailbox/'session-closed.json', dict(schema=1,
                status=record['status'], requests_submitted=record['compile_requests_submitted']))
    print(record['status'], flush=True)


if __name__ == '__main__':
    main()
