"""Transformers generation over an immutable, already verified Qwen DSP bundle.

Internal CLI adapter, not a backend for arbitrary PyTorch modules or supplied KV.
The caller owns a fresh cache for each request, exposes no mutable
aliases/hooks, and verifies the bundle/framework before constructing this object.
No CPU model parameters exist: changing model weights requires a new DSP bundle.
"""
import time

import torch
from transformers import Cache, DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.qwen3.runtime.arm import hexagon_resident_protocol as protocol


class ResidentCache(Cache):
    """Request handle for KV tensors owned by the DSP, not ARM tensor storage."""

    def __init__(self, owner):
        super().__init__(layers=[])
        self.owner = owner
        self.epoch = owner.epoch

    def get_seq_length(self, layer_idx=0):
        if not self.owner.active or self.epoch != self.owner.epoch:
            raise ValueError('DSP cache handle belongs to an inactive request')
        if not 0 <= layer_idx < protocol.LAYERS:
            raise ValueError('Invalid DSP cache layer')
        return self.owner.position

    def get_max_cache_shape(self, layer_idx=0):
        return protocol.MAX_LENGTH

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    @property
    def is_initialized(self):
        return self.get_seq_length() > 0

    @property
    def max_batch_size(self):
        return 1

    @property
    def max_cache_len(self):
        return protocol.MAX_LENGTH

    @property
    def is_sliding(self):
        return [False] * protocol.LAYERS

    def __len__(self):
        return protocol.LAYERS

    def _unsupported(self, *args, **kwargs):
        raise ValueError('DSP cache supports this session generation only; reset through the backend')

    update = reset = crop = reorder_cache = _unsupported
    batch_repeat_interleave = batch_select_indices = _unsupported
    offload = prefetch = __iter__ = _unsupported


class CompiledGeneration:
    parameter_pool = None

    def __init__(self, config, generation_config, stream, *, verify_cache=False, diagnostics=False,
                 batch_size=protocol.MAX_LENGTH, ubatch_size=protocol.MAX_QUERY, resident_only=False):
        expected = dict(model_type='qwen3', num_hidden_layers=28, num_attention_heads=16,
                        num_key_value_heads=8, head_dim=128, hidden_size=1024,
                        intermediate_size=3072, vocab_size=151936)
        if any(getattr(config, name, None) != value for name, value in expected.items()):
            raise ValueError('Configuration differs from the supported compiled Qwen ABI')
        if (type(batch_size) is not int or type(ubatch_size) is not int
                or not 1 <= ubatch_size <= min(batch_size, protocol.MAX_QUERY)
                or batch_size > protocol.MAX_LENGTH):
            raise ValueError('Require 1 <= ubatch-size <= min(batch-size, compiled query limit) '
                             'and batch-size <= cache limit')
        if type(resident_only) is not bool:
            raise ValueError("resident_only must be a boolean")
        self.resident_only = resident_only
        if resident_only and verify_cache:
            raise ValueError("Cache verification requires the ARM mirror")
        self.epoch = 0
        self.batch_size, self.ubatch_size = batch_size, ubatch_size
        self.stream, self.verify_cache, self.diagnostics = stream, verify_cache, diagnostics
        self.calls, self.captures = [], []
        self.position, self.owned_cache, self.active = 0, None, False
        owner = self

        class CompiledModel(PreTrainedModel, GenerationMixin):
            @property
            def device(self):
                return torch.device('cpu')

            @property
            def dtype(self):
                return torch.float32

            def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=True,
                        cache_position=None, logits_to_keep=1, **kwargs):
                return owner.forward(input_ids, attention_mask, position_ids, past_key_values,
                                     inputs_embeds, use_cache, cache_position, logits_to_keep, kwargs)

        self.model = CompiledModel(config).eval()
        self.model.generation_config = generation_config

    def reset(self):
        self.active = False
        self.epoch += 1
        raw, _ = self.stream.invoke_packet(protocol.header(0, 0), 64)
        if len(raw) != 64 or raw[:8] != b'QWMS001\0' or any(raw[8:16]):
            raise ValueError('Invalid resident reset response')
        self.position, self.owned_cache, self.active = 0, None, True

    def new_cache(self):
        return ResidentCache(self) if self.resident_only else DynamicCache(config=self.model.config)

    def forward(self, ids, mask, positions, cache, embeds, use_cache, cache_position, logits_to_keep, kwargs):
        started = time.perf_counter()
        if not self.active:
            raise ValueError('Reset the compiled request before generation')
        if (embeds is not None or use_cache is not True or type(logits_to_keep) is not int
                or logits_to_keep != 1 or kwargs.get('return_dict') is False
                or kwargs.get('output_attentions') or kwargs.get('output_hidden_states')
                or set(kwargs) - {'return_dict', 'output_attentions', 'output_hidden_states'}):
            raise ValueError('Unsupported compiled-generation input or output option')
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                or ids.dtype != torch.int64 or ids.device.type != 'cpu'):
            raise ValueError('Compiled generation requires batch-one CPU int64 tokens')
        query, position = ids.shape[1], self.position
        if not 1 <= query <= self.batch_size or position + query > protocol.MAX_LENGTH:
            raise ValueError('Compiled query/cache extent is unsupported')
        # Validate the entire logical batch before advancing either cache.
        if bool(((ids < 0) | (ids >= protocol.VOCAB)).any()):
            raise ValueError('Token ID is outside the compiled vocabulary')
        expected = torch.arange(position, position + query)
        if (not isinstance(cache_position, torch.Tensor) or not isinstance(positions, torch.Tensor)
                or not torch.equal(cache_position, expected) or not torch.equal(positions, expected.unsqueeze(0))):
            raise ValueError('Compiled generation positions differ from the resident cache')
        if not isinstance(mask, torch.Tensor) or mask.shape != (1, position + query) or not bool((mask == 1).all()):
            raise ValueError('Only the admitted unpadded causal request is supported')
        expected_cache = ResidentCache if self.resident_only else DynamicCache
        if type(cache) is not expected_cache or (self.resident_only and cache.owner is not self) or cache.get_seq_length() != position:
            raise ValueError('Compiled generation requires a matching private request cache')
        if self.owned_cache is not None and cache is not self.owned_cache:
            raise ValueError('Request cache ownership changed within a request')
        self.owned_cache = cache
        checked = time.perf_counter()
        try:
            # Like llama.cpp's logical batch / physical ubatch separation, a
            # framework forward may contain several DSP forwards. Keep one KV
            # session and return only the final position's logits.
            for offset in range(0, query, self.ubatch_size):
                chunk = ids[:, offset:offset + self.ubatch_size]
                chunk_position = position + offset
                logits, delta, stats, event, exchanged = protocol.forward(
                    self.stream, chunk, chunk_position, resident_only=self.resident_only)
                if not self.resident_only:
                    for layer in range(protocol.LAYERS):
                        cache.update(delta[2 * layer], delta[2 * layer + 1], layer)
                reconstructed = time.perf_counter()
                self.position = chunk_position + chunk.shape[1]
                if self.verify_cache:
                    planes = [value for layer in cache.layers for value in (layer.keys, layer.values)]
                    protocol.verify_cache(self.stream, self.position, planes)
                finished = time.perf_counter()
                row = dict(capture=None, position=chunk_position, query=chunk.shape[1],
                           logical_position=position, logical_query=query,
                           ids=chunk.tolist(), statistics=stats, invoke=event,
                           parameter_check_s=checked - started if offset == 0 else 0,
                           exchange_s=exchanged - checked, reconstruct_s=reconstructed - exchanged,
                           snapshot_check_s=finished - reconstructed)
                if self.diagnostics:
                    row['top_token'] = int(logits.reshape(-1).argmax())
                self.calls.append(row)
                checked = time.perf_counter()
        except BaseException:
            # A failed response/update may already have advanced DSP state.
            self.active = False
            raise
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
