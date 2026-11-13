"""The upstream DynamicCache append must lower with symbolic past/query lengths."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('torch_mlir')
from torch_mlir import fx
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from transformers import DynamicCache, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
from hexagon_tensor_literal_buffers import lower_aten_graph_owned_constants


def test_dynamic_cache_prefill_decode_lowering():
    torch._dynamo.reset()
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    captures = []

    def forward(q, k, v, cache):
        k, v = cache.update(k, v, 0)
        nq, nk = q.shape[-2], k.shape[-2]
        visible = torch.arange(nk)[None, :] <= (torch.arange(nq) + nk - nq)[:, None]
        mask = torch.zeros(nq, nk).masked_fill(~visible, float('-inf'))[None, None]
        return eager_attention_forward(module, q, k, v, mask, 8**-0.5)[0]

    def compiler(graph, unused):
        before = graph.code
        module, _ = lower_aten_graph_owned_constants(graph, 'dynamic_cache_attention')
        assert module.operation.verify()
        assert graph.code == before  # AOT still owns this graph.
        captures.append(str(module))
        return make_boxed_func(graph.forward)

    compiled = torch.compile(forward, backend=aot_autograd(
        fw_compiler=compiler, inference_compiler=compiler,
        decompositions=fx.get_decomposition_table()), fullgraph=True, dynamic=True)
    cache, reference = (DynamicCache(config=Qwen3Config(num_hidden_layers=1)) for _ in range(2))
    generator = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        for query in (3, 5, 1, 1):
            q = torch.randn(1, 4, query, 8, generator=generator)
            k = torch.randn(1, 2, query, 8, generator=generator)
            v = torch.randn(1, 2, query, 8, generator=generator)
            for tensor in (q, k, v):
                for axis in (0, 1, 3):
                    torch._dynamo.mark_static(tensor, axis)
            actual = compiled(q, k, v, cache)
            torch.testing.assert_close(actual, forward(q, k, v, reference))
            assert cache.get_seq_length() == reference.get_seq_length()
    assert len(captures) == 3  # Empty, nonempty prefill, singleton decode.
    assert 'tensor.concat' in captures[1]
    assert 'arith.addi' in captures[1]
    assert cache.get_seq_length() == 10
