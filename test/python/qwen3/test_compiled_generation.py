"""Exercise native generation and failure boundaries without DSP hardware."""
import struct

import pytest
import torch
from transformers import DynamicCache, GenerationConfig, Qwen3Config

from models.qwen3.runtime.arm.hexagon_compiled_generation import CompiledGeneration
from models.qwen3.runtime.arm import hexagon_resident_protocol as protocol


class Stream:
    def __init__(self, stop_at=34):
        self.stop_at = stop_at
        self.calls = []
        self.output_sizes = []
        self.bad_response = False

    def invoke_packet(self, packet, size):
        mode, position, query = struct.unpack_from('<III', packet, 8)
        self.calls.append((mode, position, query))
        self.output_sizes.append(size)
        length = position + query if mode in (1, 3, 4) else 0
        response = b'QWMS001\0' + struct.pack('<6Q', length, 1024, 0, 0, 0, 1) + bytes(8)
        if mode in (1, 3):
            data = torch.zeros(protocol.VOCAB + (56 * query * 1024 if mode == 1 else 0))
            data[3 if length >= self.stop_at else 5] = 100
            response += data.numpy().tobytes()
        return (response[:-1] if self.bad_response else response), {}


def backend(stream=None, **kwargs):
    config = Qwen3Config(hidden_size=1024, intermediate_size=3072,
                        num_hidden_layers=28, num_attention_heads=16,
                        num_key_value_heads=8, head_dim=128, vocab_size=151936)
    generation = GenerationConfig(eos_token_id=3, pad_token_id=0, use_cache=True)
    return CompiledGeneration(config, generation, stream or Stream(), **kwargs)


@pytest.mark.parametrize('logical_batch', [32, 128])
@pytest.mark.parametrize('resident_only', [False, True])
def test_native_generate_chunks_prompt_and_stops_on_eos_without_parameters(logical_batch, resident_only):
    # The admitted ARM framework includes the pinned prefill-position fix.
    # Unpatched 5.0.0rc1 advances cache_position twice after chunked prefill.
    import hashlib
    import json
    from pathlib import Path
    from transformers.generation import utils
    lock = json.loads((Path(__file__).resolve().parents[3] / 'models/qwen3/dependencies.lock.json').read_text())
    if hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest() != lock['framework']['transformers_generation_sha256']:
        pytest.skip('Native generation requires the pinned framework; run this test on the ARM deployment')
    b = backend(resident_only=resident_only)
    assert list(b.model.parameters()) == []
    b.reset()
    ids = torch.full((1, 33), 7, dtype=torch.int64)
    cache = b.new_cache()
    with torch.inference_mode():
        result = b.model.generate(ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                                  max_new_tokens=4, prefill_chunk_size=logical_batch,
                                  do_sample=False, disable_compile=True)
    assert result[0, 33:].tolist() == [5, 3]
    mode = 3 if resident_only else 1
    assert b.stream.calls == [(0, 0, 0), (mode, 0, 32), (mode, 32, 1), (mode, 33, 1)]
    assert cache.get_seq_length() == b.position == 34
    assert b.captures == []


def inputs(b, cache=None, token=7):
    p = b.position
    return dict(input_ids=torch.tensor([[token]]), attention_mask=torch.ones((1, p + 1), dtype=torch.int64),
                position_ids=torch.tensor([[p]]), cache_position=torch.tensor([p]),
                past_key_values=cache if cache is not None else DynamicCache(config=b.model.config),
                logits_to_keep=1)


def test_reset_and_cache_ownership_are_explicit():
    b = backend()
    with pytest.raises(ValueError, match='Reset'):
        b.model(**inputs(b))
    b.reset()
    first = inputs(b)
    b.model(**first)
    other = DynamicCache(config=b.model.config)
    for layer in range(28):
        other.update(torch.zeros(1, 8, 1, 128), torch.zeros(1, 8, 1, 128), layer)
    with pytest.raises(ValueError, match='ownership'):
        b.model(**inputs(b, other))
    b.reset()
    b.model(**inputs(b))
    assert b.position == 1


def test_invalid_response_requires_reset_before_retry():
    b = backend()
    b.reset()
    b.stream.bad_response = True
    with pytest.raises(ValueError, match='response'):
        b.model(**inputs(b))
    b.stream.bad_response = False
    with pytest.raises(ValueError, match='Reset'):
        b.model(**inputs(b))
    b.reset()
    b.model(**inputs(b))


@pytest.mark.parametrize('token', [-1, protocol.VOCAB])
def test_invalid_token_never_reaches_transport(token):
    b = backend()
    b.reset()
    with pytest.raises(ValueError, match='vocabulary'):
        b.model(**inputs(b, token=token))
    assert b.stream.calls == [(0, 0, 0)]


def prompt_inputs(b, count, cache=None):
    p = b.position
    return dict(input_ids=torch.full((1, count), 7, dtype=torch.int64),
                attention_mask=torch.ones((1, p + count), dtype=torch.int64),
                position_ids=torch.arange(p, p + count).unsqueeze(0),
                cache_position=torch.arange(p, p + count),
                past_key_values=cache if cache is not None else DynamicCache(config=b.model.config))


def test_logical_prefill_reuses_one_cache_and_returns_last_microbatch_logits():
    b = backend(ubatch_size=16)
    b.reset()
    args = prompt_inputs(b, 35)
    out = b.model(**args)
    assert b.stream.calls == [(0, 0, 0), (1, 0, 16), (1, 16, 16), (1, 32, 3)]
    assert out.logits.argmax().item() == 3  # Earlier chunks predicted 5.
    assert out.past_key_values is args['past_key_values']
    assert out.past_key_values.get_seq_length() == b.position == 35
    assert [row['logical_query'] for row in b.calls] == [35, 35, 35]
    b.model(**inputs(b, out.past_key_values))
    assert b.stream.calls[-1] == (1, 35, 1)


def test_invalid_late_token_does_not_partially_execute_logical_prefill():
    b = backend()
    b.reset()
    args = prompt_inputs(b, 33)
    args['input_ids'][0, -1] = protocol.VOCAB
    with pytest.raises(ValueError, match='vocabulary'):
        b.model(**args)
    assert b.stream.calls == [(0, 0, 0)]
    assert b.position == args['past_key_values'].get_seq_length() == 0


def test_mid_batch_failure_requires_reset_and_fresh_cache():
    class FailingStream(Stream):
        def invoke_packet(self, packet, size):
            result = super().invoke_packet(packet, size)
            if struct.unpack_from('<III', packet, 8) == (1, 32, 1):
                raise RuntimeError('injected second microbatch failure')
            return result

    b = backend(FailingStream())
    b.reset()
    args = prompt_inputs(b, 33)
    with pytest.raises(RuntimeError, match='injected'):
        b.model(**args)
    assert b.position == args['past_key_values'].get_seq_length() == 32
    with pytest.raises(ValueError, match='Reset'):
        b.model(**inputs(b, args['past_key_values']))
    b.reset()
    b.model(**inputs(b))
    assert b.position == 1


@pytest.mark.parametrize('batch,micro', [(0, 1), (32, 33), (16, 32), (1025, 32)])
def test_execution_limits_checked_before_start(batch, micro):
    with pytest.raises(ValueError, match='ubatch-size'):
        backend(batch_size=batch, ubatch_size=micro)


def test_resident_handle_keeps_kv_off_arm_and_expires_on_reset():
    b = backend(resident_only=True, ubatch_size=16)
    b.reset()
    cache = b.new_cache()
    out = b.model(**prompt_inputs(b, 35, cache))
    assert out.past_key_values is cache
    assert cache.get_seq_length() == b.position == 35
    assert cache.layers == []  # No placeholder or mirrored KV tensors on ARM.
    assert b.stream.calls == [(0, 0, 0), (3, 0, 16), (3, 16, 16), (3, 32, 3)]
    assert b.stream.output_sizes[1:] == [64 + protocol.VOCAB * 4] * 3
    assert cache.get_mask_sizes(torch.arange(35, 37)) == (37, 0)
    b.model(**inputs(b, cache))
    assert cache.get_seq_length() == 36
    b.reset()
    with pytest.raises(ValueError, match='inactive request'):
        cache.get_seq_length()
    fresh = b.new_cache()
    b.model(**inputs(b, fresh))
    assert fresh.get_seq_length() == 1


@pytest.mark.parametrize('diagnostics', [False, True])
def test_only_the_last_chunk_of_a_logical_batch_requests_logits(diagnostics):
    b = backend(resident_only=True, skip_unused_logits=True, diagnostics=diagnostics)
    b.reset()
    cache = b.new_cache()
    out = b.model(**prompt_inputs(b, 70, cache))
    middle = 3 if diagnostics else 4  # diagnostics record every chunk's top token
    assert b.stream.calls == [(0, 0, 0), (middle, 0, 32), (middle, 32, 32), (3, 64, 6)]
    assert b.stream.output_sizes[1:] == [64 + (protocol.VOCAB * 4 if mode == 3 else 0)
                                         for mode, _, _ in b.stream.calls[1:]]
    assert out.logits.shape == (1, 1, protocol.VOCAB) and int(out.logits.argmax()) == 3
    assert cache.get_seq_length() == b.position == 70
    b.model(**inputs(b, cache))  # a single-chunk forward always returns logits
    assert b.stream.calls[-1] == (3, 70, 1)


def test_skipping_logits_requires_a_resident_only_session():
    with pytest.raises(ValueError, match='resident-only'):
        backend(skip_unused_logits=True)
    with pytest.raises(ValueError, match='resident-only'):
        backend(resident_only=True, skip_unused_logits=1)
    with pytest.raises(ValueError, match='omit its logits'):
        protocol.forward(Stream(), torch.zeros((1, 2), dtype=torch.int64), 0, logits=False)


@pytest.mark.parametrize('operation', ['update', 'reset', 'crop', 'reorder_cache',
                                       'batch_repeat_interleave', 'batch_select_indices',
                                       'offload', 'prefetch', '__iter__'])
def test_resident_handle_rejects_unsupported_cache_mutation(operation):
    b = backend(resident_only=True)
    b.reset()
    with pytest.raises(ValueError, match='DSP cache supports'):
        getattr(b.new_cache(), operation)()
    assert b.stream.calls == [(0, 0, 0)]


def test_resident_handle_rejects_foreign_sessions_and_invalid_response():
    b, other = backend(resident_only=True), backend(resident_only=True)
    b.reset()
    other.reset()
    with pytest.raises(ValueError, match='private request cache'):
        b.model(**inputs(b, other.new_cache()))
    assert b.stream.calls == [(0, 0, 0)]
    cache = b.new_cache()
    b.stream.bad_response = True
    with pytest.raises(ValueError, match='response'):
        b.model(**inputs(b, cache))
    with pytest.raises(ValueError, match='inactive request'):
        cache.get_seq_length()
    b.stream.bad_response = False
    b.reset()
    b.model(**inputs(b, b.new_cache()))
    assert b.position == 1


def test_cache_verification_requires_an_arm_mirror():
    with pytest.raises(ValueError, match='verification requires'):
        backend(resident_only=True, verify_cache=True)
