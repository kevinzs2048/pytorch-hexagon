"""Native Transformers generation using an explicitly supplied resident bundle.

This candidate shares the public Qwen CLI and the verified base model/runtime.
Compiled generation can retain all KV tensors on the DSP. The generic PyTorch
and explicit cache-verification paths maintain an ARM DynamicCache using new KV rows.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from contextlib import nullcontext

from .bundle import ROOT, inside, sha, write


def normalize_unused_embedding_options(model):
    """Avoid an AOT scalar restart for Embedding's unused default norm option.

    With max_norm=None, embedding performs no renormalization. Keeping the
    default norm as integer 2 avoids tensorifying the unused Python float 2.0;
    active renormalization and non-default options are left untouched.
    """
    import torch
    embedding = model.get_input_embeddings()
    if (type(embedding) is torch.nn.Embedding and embedding.max_norm is None
            and type(embedding.norm_type) is float and embedding.norm_type == 2.0):
        embedding.norm_type = 2
        return True
    return False


def resident_cache_namespace(base_manifest, session):
    """Extend the existing cache protocol with resident code and bundle pins."""
    from .cli import implementation_digest
    namespace = dict(base_manifest['namespace'],
                     runtime_catalog_sha256=sha(session / 'session.json'))
    namespace['native_source_sha256'] = sha(
        ROOT / 'models/qwen3/runtime/arm/hexagon_resident_session.py')
    namespace['adapter_sha256'] = sha(
        ROOT / 'models/qwen3/runtime/support/native_aot_output.py')
    # The legacy execution manifest covers common runtime helpers. These two
    # resident-specific files are additional executable dependencies.
    recipe = (namespace['recipe_sha256'] + implementation_digest()
              + sha(Path(__file__))
              + sha(ROOT / 'models/qwen3/runtime/arm/hexagon_dynamic_cache_graph.py')
              + sha(ROOT / 'models/qwen3/runtime/arm/hexagon_resident_protocol.py'))
    namespace['recipe_sha256'] = hashlib.sha256(recipe.encode()).hexdigest()
    return namespace


Q8_0_LIBRARY = 'libqwen_q8_0_weights.so'
Q8_0_SESSION_KEYS = {'library', 'projections', 'shared_constant_libraries', 'weights', 'numerics'}


def load_session(directory, base_manifest):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'session.json').read_text())
    if manifest.get('schema') != 'qwen3-resident-session-v1':
        raise ValueError('Unsupported resident session bundle')
    modes = manifest.get('resident_forward_modes', [1])
    if (modes not in ([1], [1, 3], [1, 3, 4], [1, 3, 4, 5]) or any(type(mode) is not int for mode in modes)
            # Mode 4 (no logits) is implemented by the Q8_0 runtime's LM head;
            # mode 5 (a logical batch in one call) is a sequence of modes 4 and 3.
            or (4 in modes and manifest.get('q8_0_projections') is None)):
        raise ValueError('Unsupported resident forward modes')
    paired = manifest.get('resident_paired_chunks', False)
    if type(paired) is not bool or (paired and 5 not in modes):
        raise ValueError('Paired chunks require the batched forward mode')
    if manifest['checkpoint_sha256'] != base_manifest['files']['hf/model.safetensors']['sha256']:
        raise ValueError('Resident code and base checkpoint differ')
    for name, row in manifest['files'].items():
        path = inside(directory, name)
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Resident session artifact changed: ' + name)
    required = {'model.so', 'decompositions.py'}
    required.update(name+'-graph.txt' for name in ('initial', 'prefill', 'decode', 'single0', 'single1'))
    required.update(name+'-mapping.json' for name in ('initial', 'prefill', 'decode', 'single0', 'single1'))
    constants = manifest['constants']
    q8 = manifest.get('q8_0_projections')
    if q8 is None:
        shared = constants
        if len(set(shared)) != 4:
            raise ValueError('Incomplete resident code, graph or constant inventory')
    else:
        # Q8_0 projections drop the INT8 weights from every domain: fewer shared
        # constant libraries, then the Q8_0 weight library staged with them.
        if (not isinstance(q8, dict) or set(q8) != Q8_0_SESSION_KEYS
                or q8['library'] != Q8_0_LIBRARY or type(q8['projections']) is not int
                or type(q8['shared_constant_libraries']) is not int
                or not 1 <= q8['shared_constant_libraries'] <= 4
                or constants[-1:] != [Q8_0_LIBRARY]
                or len(constants) != q8['shared_constant_libraries'] + 1):
            raise ValueError('Inconsistent Q8_0 projection inventory')
        shared = constants[:-1]
    if (len(set(shared)) != len(shared) or any(name != f'libqwen_dynamic_kv_prefill_const_{i}.so'
                                             for i, name in enumerate(shared, 1))
            or not required.union(constants) <= manifest['files'].keys()):
        raise ValueError('Incomplete resident code, graph or constant inventory')
    return manifest


def generate(args):
    setup_started = time.perf_counter()
    execution = getattr(args, 'resident_execution', 'torch-compile')
    batch_size = getattr(args, 'batch_size', None)
    if execution == 'compiled':
        batch_size = 1024 if batch_size is None else batch_size
        if type(batch_size) is not int or not args.chunk_size <= batch_size <= 1024:
            raise ValueError('Require ubatch-size <= batch-size <= 1024')
    elif batch_size is not None:
        raise ValueError('batch-size currently requires compiled resident execution')
    else:
        batch_size = args.chunk_size
    # Keep the existing checkpoint, framework-source and transport checks.
    from .cli import doctor, runtime_paths
    doctor(args.bundle, execution=True)
    runtime_paths()
    base = args.bundle.resolve()
    base_manifest = json.loads((base / 'bundle.json').read_text())
    session = args.resident_session.resolve()
    manifest = load_session(session, base_manifest)
    resident_only = (execution == 'compiled'
                     and 3 in manifest.get('resident_forward_modes', [1])
                     and not args.verify_resident_cache)
    # Prompt chunks before the last one of a logical batch need no prediction.
    skip_unused_logits = resident_only and 4 in manifest.get('resident_forward_modes', [1])
    # ... and the chunks of a logical batch need no separate DSP calls.
    batched_forward = skip_unused_logits and 5 in manifest['resident_forward_modes']
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import DynamicCache
    from hexagon_runtime.cpu_wake_latency import CpuWakeLatencyHold
    from hexagon_runtime.hexagon_shared_packet_transport import HexagonSharedPacketStream
    from run_qwen_arm_dynamic_warm_generate import tokenize_prompts, validate_requests

    torch.set_num_threads(4)
    tokenizer = None
    if args.requests_json is not None:
        requests = json.loads(args.requests_json)
    else:
        prompts = json.loads(args.prompts_json) if args.prompts_json else [args.prompt]
        tokenizer, requests, _ = tokenize_prompts(base / 'hf', prompts, chat=args.chat_template)
    validate_requests(requests, args.max_new_tokens, args.chunk_size)
    normalized_embedding = False
    if execution == 'torch-compile':
        from transformers import Qwen3ForCausalLM
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
        model = Qwen3ForCausalLM.from_pretrained(
            base / 'hf', local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').eval().requires_grad_(False)
        normalized_embedding = normalize_unused_embedding_options(model)
        quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
    elif execution != 'compiled':
        raise ValueError('Unsupported resident execution mode')
    parameter_check = getattr(args, 'resident_parameter_check', 'every-forward')
    if parameter_check not in ('every-forward', 'owned-request'):
        raise ValueError('Unsupported resident parameter check policy')
    if execution == 'compiled':
        parameter_check = 'compiled-artifact'
    cache_enabled = getattr(args, 'resident_aot_cache', False)
    cache_events = []
    cache_scope = nullcontext(None)
    if cache_enabled:
        from models.qwen3.runtime.support.native_aot_output import native_cache_scope
        # Resolve before entering the temporary DSP working directory.
        cache_scope = native_cache_scope(resident_cache_namespace(base_manifest, session),
                                         args.cache_dir.absolute(), cache_events)
    from torch._dynamo.utils import counters
    cache_before = dict(counters['aot_autograd'])

    work = Path('/dev/shm/qwen3-resident-' + uuid.uuid4().hex)
    result = dict(schema='qwen3-resident-run-v1', status='running', native_generate=True,
                  CPU_model_fallback=False, machine_code_compilations=0,
                  execution=execution,
                  kv_cache_location='dsp' if resident_only else 'dsp_with_arm_mirror',
                  # Packet diagnostics record every chunk's top token, so they keep all logits.
                  unused_prefill_logits=('skipped' if skip_unused_logits and args.packet_recording == 'none'
                                         else 'computed'),
                  # Packet diagnostics also keep one DSP call per ubatch.
                  prefill_dsp_calls=('one_per_batch' if batched_forward and args.packet_recording == 'none'
                                     else 'one_per_ubatch'),
                  prefill_batch_size=batch_size, prefill_ubatch_size=args.chunk_size,
                  compiled_query_limit=32,
                  parameter_validation=('immutable_artifact_SHA256' if execution == 'compiled'
                                        else 'exact_bytes_initial_SHA256'),
                  parameter_check_policy=parameter_check,
                  unused_embedding_norm_normalized=normalized_embedding,
                  aot_cache=dict(enabled=cache_enabled, events=cache_events),
                  session_sha256=sha(session / 'session.json'),
                  numerical_policy=dict(
                      linear='W8A8', kv='FP32',
                      attention={key: value for key, value in manifest.get('attention', {}).items()
                                 if key == 'implementation' or 'precision' in key},
                      silu=manifest.get('silu_precision', 'unspecified')),
                  model_quality_accepted=manifest.get('model_quality_accepted',
                      manifest.get('attention', {}).get('model_quality_accepted')),
                  verify_resident_cache=args.verify_resident_cache,
                  packet_recording=args.packet_recording,
                  requests=[], forwards=[], captures=[])
    previous = Path.cwd()
    backend = None
    hook = None
    try:
        work.mkdir()
        (work / 'rfsa').mkdir()
        (work / 'buffer_stream').symlink_to(base / 'runtime/buffer_stream')
        (work / 'rfsa/libhexagon_buffer_runner_skel.so').symlink_to(
            base / 'runtime/libhexagon_buffer_runner_skel.so')
        for name in ['model.so', *manifest['constants']]:
            shutil.copy2(inside(session, name), work / name)
            if sha(work / name) != manifest['files'][name]['sha256']:
                raise ValueError('Staged resident artifact changed: ' + name)
        env = dict(os.environ, LD_LIBRARY_PATH=str(work),
                   ADSP_LIBRARY_PATH=(str(work)+';'+str(work / 'rfsa')+
                       ';/usr/lib/rfsa/adsp;/usr/share/qcom/sa8775p/Qualcomm/SA8775P-RIDE/dsp/cdsp'))
        # Snapshot mode is diagnostic only; normal packets carry one new chunk.
        input_cap = 64 + (1024 if batched_forward else 32)*8
        output_cap = 64 + (56*1024*1024 if args.verify_resident_cache
                          else 151936 if resident_only
                          else 151936+56*32*1024)*4
        argv = ['flock', '-n', '/tmp/qwen3-qcs9075-probe.lock', str(work / 'buffer_stream'),
                str(work / 'model.so'), str(input_cap), str(output_cap)]
        os.chdir(work)
        # Keep the ARM cores out of deep idle for the life of the DSP stream: each
        # FastRPC return otherwise waits for a core to wake up. Best effort.
        wake_hold = CpuWakeLatencyHold()
        result['cpu_wake_latency_hold'] = wake_hold.record()
        with wake_hold, cache_scope as cache_info, torch.inference_mode(), HexagonSharedPacketStream(
                argv, env, output, input_bytes=input_cap, output_bytes=output_cap,
                library_path_prefix='/dev/shm/qwen3-', initial_library=str(work / 'model.so'),
                timeout=240) as stream:
            if cache_info is not None:
                result['aot_cache'].update(cache_info)
            result['ready'] = stream.start()

            class RecordedStream:
                def invoke_packet(self, packet, size):
                    response, event = stream.invoke_packet(packet, size)
                    if args.packet_recording != 'none':
                        event = dict(event, input_sha256=hashlib.sha256(packet).hexdigest(),
                                     output_sha256=hashlib.sha256(response).hexdigest())
                    if args.packet_recording == 'full':
                        folder = output / ('packet-%04d' % event['ordinal'])
                        folder.mkdir()
                        (folder / 'input.bin').write_bytes(packet)
                        (folder / 'output.bin').write_bytes(response)
                    return response, event

            if execution == 'compiled':
                from transformers import AutoConfig, GenerationConfig
                from models.qwen3.runtime.arm.hexagon_compiled_generation import CompiledGeneration
                config = AutoConfig.from_pretrained(base / 'hf', local_files_only=True)
                config.dtype = torch.float32
                generation_config = GenerationConfig.from_pretrained(base / 'hf', local_files_only=True)
                backend = CompiledGeneration(config, generation_config, RecordedStream(),
                                             verify_cache=args.verify_resident_cache,
                                             diagnostics=args.packet_recording != 'none',
                                             batch_size=batch_size, ubatch_size=args.chunk_size,
                                             resident_only=resident_only,
                                             skip_unused_logits=skip_unused_logits,
                                             batched_forward=batched_forward)
                model = backend.model
            else:
                from hexagon_resident_session import ResidentSessionBackend
                backend = ResidentSessionBackend(model, RecordedStream(), session, output,
                                                 verify_cache=args.verify_resident_cache,
                                                 diagnostics=args.packet_recording != 'none')

                def no_eager(module, positional, kwargs):
                    if not torch.compiler.is_compiling():
                        raise RuntimeError('CPU model forward is forbidden')

                hook = model.register_forward_pre_hook(no_eager, with_kwargs=True)
                model.compile(backend=backend, fullgraph=True, dynamic=True)
            result['setup_seconds'] = time.perf_counter() - setup_started
            for request, tokens in enumerate(requests):
                backend.reset()
                begin = len(backend.calls)
                ids = torch.tensor([tokens], dtype=torch.int64)
                started = time.perf_counter()
                # This entry constructs and owns the model; it neither exposes
                # aliases nor accepts external hooks/callbacks. No weight writes
                # occur during generate. The generic backend remains fully checked.
                ownership = (backend.parameter_pool.owned_request()
                             if parameter_check == 'owned-request' else nullcontext())
                with ownership:
                    sequence = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        past_key_values=(backend.new_cache() if execution == 'compiled'
                                         else DynamicCache(config=model.config)),
                        max_new_tokens=args.max_new_tokens, prefill_chunk_size=batch_size,
                        do_sample=False, disable_compile=True, pad_token_id=0)
                elapsed = time.perf_counter() - started
                generated = sequence[0, len(tokens):].tolist()
                row = dict(index=request, prompt_tokens=len(tokens), generated_tokens=generated,
                           generation_seconds_including_framework_capture=elapsed,
                           forward_begin=begin, forward_end=len(backend.calls))
                if tokenizer is not None:
                    row['text'] = tokenizer.decode(generated, skip_special_tokens=True)
                result['requests'].append(row)
                result.update(forwards=backend.calls, captures=backend.captures)
                write(output / 'result.json', result)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            stream.close()
            result['process_exit_code'] = stream.process.returncode
        if result['process_exit_code'] != 0:
            raise RuntimeError('DSP transport did not exit cleanly')
        result['status'] = 'native_resident_generation_complete'
        result['run_seconds'] = time.perf_counter() - setup_started
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['aot_cache']['counters'] = {
            key: value - cache_before.get(key, 0)
            for key, value in counters['aot_autograd'].items()
            if value != cache_before.get(key, 0)}
        if hook is not None:
            hook.remove()
        if backend is not None:
            if backend.parameter_pool is not None:
                backend.parameter_pool.close()
            result.update(forwards=backend.calls, captures=backend.captures,
                          DSP_calls=len(backend.calls),
                          parameter_snapshot_bytes=(backend.parameter_pool.snapshot_bytes
                                                    if backend.parameter_pool is not None else 0))
        os.chdir(previous)
        write(output / 'result.json', result)
        if work.exists():
            shutil.rmtree(work)
    return result
