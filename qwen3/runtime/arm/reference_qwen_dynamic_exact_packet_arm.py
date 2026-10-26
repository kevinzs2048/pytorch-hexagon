"""QCS9075 ARM PyTorch eager control for a real bounded dynamic DSP request.

This replays one exact q-token packet, including all 56 KV inputs, without
running a DSP kernel.  It retains only last-token logits and updated KV slots.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import traceback


CHECKPOINT_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
KV_SHAPE = (1, 8, 1024, 128)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack_request(path, expected_q, start):
    import numpy as np
    import torch

    packet = path.read_bytes()
    if (packet[:8] != b'HXDY\x01\x00\x00\x00'
            or struct.unpack('<q', packet[8:16]) != (expected_q,)):
        raise ValueError('Dynamic Qwen request frame differs')
    q = expected_q
    shapes = [(1, q), (q,), (1, q), KV_SHAPE, KV_SHAPE,
              (1, 1, q, 1024)] + [KV_SHAPE] * 54
    dtypes = ['<i8'] * 3 + ['<f4'] * 57
    tensors = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        offset = (offset + 127) & ~127
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = packet[16 + offset:16 + offset + size]
        if len(raw) != size:
            raise ValueError('Dynamic Qwen request tensor truncated')
        tensors.append(torch.from_numpy(np.frombuffer(raw, dtype=dtype)
                                        .reshape(shape).copy()))
        offset += size
    if (len(tensors) != 60 or 16 + ((offset + 127) & ~127) != len(packet)
            or tensors[1].tolist() != list(range(start, start + q))
            or tensors[2].reshape(-1).tolist() != list(range(start, start + q))):
        raise ValueError('Dynamic Qwen request tensor/position ABI differs')
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--request-sha256', required=True)
    p.add_argument('--query', type=int, required=True)
    p.add_argument('--start-position', type=int, required=True)
    p.add_argument('--hf', type=Path, required=True)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--context-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--observe-module', action='append', default=[])
    p.add_argument('--baseline-result', type=Path)
    p.add_argument('--baseline-result-sha256')
    args = p.parse_args()
    if (bool(args.observe_module) != (args.baseline_result is not None)
            or bool(args.observe_module) != (args.baseline_result_sha256 is not None)):
        raise ValueError('Read-only observations require a pinned unobserved baseline')
    if (platform.machine() != 'aarch64' or args.output.exists() or not 1 <= args.query <= 32
            or not 0 <= args.start_position < 1024
            or args.start_position + args.query > 1024
            or sha(args.request) != args.request_sha256
            or sha(args.hf / 'model.safetensors') != CHECKPOINT_SHA
            or sha(args.context) != args.context_sha256):
        raise ValueError('Fresh output and pinned exact dynamic Qwen inputs required')
    context = json.loads(args.context.read_text())
    baseline = None
    if args.baseline_result is not None:
        if sha(args.baseline_result) != args.baseline_result_sha256:
            raise ValueError('Unobserved ARM baseline result changed')
        baseline = json.loads(args.baseline_result.read_text())
        if (baseline['status'] != 'dynamic_exact_packet_ARM_eager_logits_KV_recorded'
                or baseline['request_sha256'] != args.request_sha256
                or baseline['context_sha256'] != args.context_sha256
                or baseline['query'] != args.query
                or baseline['start_position'] != args.start_position
                or any(sha(baseline[key]['path']) != baseline[key]['sha256']
                       for key in ('last_logits', 'updated_KV'))):
            raise ValueError('Exact packet or unobserved ARM outputs differ')
    if (context['checkpoint']['sha256'] != CHECKPOINT_SHA
            or any(importlib.metadata.version(name) != context['versions'][name]
                   for name in ('torch', 'torchao', 'transformers'))):
        raise ValueError('ARM model or framework differs from pinned control')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='admitting_dynamic_exact_packet_ARM_eager_control',
                  source_sha256=sha(__file__), request_sha256=args.request_sha256,
                  checkpoint_sha256=CHECKPOINT_SHA,
                  context_sha256=args.context_sha256,
                  query=args.query, start_position=args.start_position,
                  architecture=platform.machine(),
                  DSP_calls=0, CPU_reference_execution=False,
                  numerical_acceptance=False)

    def save():
        (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')

    save()
    try:
        import torch
        from transformers import Qwen3ForCausalLM
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        from hexagon_native1024_cache_setup import prepare_native1024_cache
        from qwen_native_module_observation import NativeModuleObservation

        torch.set_num_threads(2)
        tensors = unpack_request(args.request, args.query, args.start_position)
        model = Qwen3ForCausalLM.from_pretrained(str(args.hf), local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').eval().requires_grad_(False)
        expected = dict(context['model_config'])
        actual = json.loads(json.dumps(model.config.to_dict()))
        expected.pop('_name_or_path', None)
        actual.pop('_name_or_path', None)
        if actual != expected:
            raise ValueError('ARM Qwen model configuration changed')
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            payload = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            digest.update((name + ':' + hashlib.sha256(payload).hexdigest() + '\n').encode())
        if digest.hexdigest() != context['historical_parameter_sha256']:
            raise ValueError('ARM Qwen original parameter bytes changed')
        quantize_(model, Int8DynamicActivationInt8WeightConfig(set_inductor_config=False))
        cache = prepare_native1024_cache(model)
        kv = [value for layer in cache.layers for value in (layer.keys, layer.values)]
        payload_kv = tensors[3:5] + tensors[6:]
        if (len(kv) != 56 or len(payload_kv) != 56
                or any(tuple(value.shape) != KV_SHAPE for value in kv)):
            raise ValueError('Native ARM KV cache ABI changed')
        with torch.no_grad():
            for target, source in zip(kv, payload_kv):
                target.copy_(source)
        if any(not torch.equal(left, right) for left, right in zip(kv, payload_kv)):
            raise ValueError('ARM eager KV differs from exact DSP request')
        record['inputs_exact'] = True
        record['CPU_reference_execution'] = True
        save()
        observations = NativeModuleObservation(model, args.observe_module, args.output)
        with torch.inference_mode(), observations:
            output = model.forward(input_ids=tensors[0], cache_position=tensors[1],
                position_ids=tensors[2], past_key_values=cache,
                attention_mask={'full_attention': tensors[5]},
                use_cache=True, return_dict=True)
        logits = output.logits[:, -1:, :].detach().contiguous()
        if (tuple(logits.shape) != (1, 1, 151936)
                or not torch.isfinite(logits).all()
                or output.past_key_values is not cache):
            raise ValueError('ARM exact dynamic Qwen output incomplete')
        logits_path = args.output / 'last-logits.bin'
        logits_path.write_bytes(logits.numpy().tobytes())
        kv_path = args.output / 'updated-kv.bin'
        with kv_path.open('wb') as stream:
            for value in kv:
                active = value[:, :, args.start_position:
                               args.start_position + args.query, :]
                stream.write(active.detach().contiguous().numpy().tobytes())
        record.update(status='dynamic_exact_packet_ARM_eager_logits_KV_recorded',
                      last_logits=dict(path=str(logits_path), sha256=sha(logits_path),
                                       top_token=int(logits.argmax())),
                      updated_KV=dict(path=str(kv_path), sha256=sha(kv_path),
                                      entries=56, active_slots=args.query))
        if baseline is not None:
            if any(record[key]['sha256'] != baseline[key]['sha256']
                   for key in ('last_logits', 'updated_KV')):
                raise ValueError('Observation hooks changed ordinary ARM logits or active KV')
            record.update(observations=observations.rows,
                          observation_source_sha256=sha(
                              Path(__file__).with_name('qwen_native_module_observation.py')),
                          baseline_result_sha256=args.baseline_result_sha256,
                          ordinary_outputs_bitexact_to_unobserved=True)
        save()
        print(json.dumps({key: record[key] for key in
                          ('status', 'query', 'start_position', 'last_logits',
                           'updated_KV')}, indent=2), flush=True)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__,
                      error=str(exc)[:1000])
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
