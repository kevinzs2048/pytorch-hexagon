"""Per-domain fused-graph policy passes of the resident builder, in the order
the accepted candidates were prepared."""
from models.qwen3.build_model import FUSED_GRAPH_KEYS, fused_graph_passes

DQ = '--qwen-fuse-i8-dequant=expected-count=197 expected-residual=56 expected-product=28'
RP = '--qwen-fuse-parallel-rope=expected-count=56'
TR = '--qwen-fuse-parallel-trig=expected-count=2'
MK = '--qwen-fuse-causal-mask=expected-count=1'
KV = '--qwen-fp16-kv-storage=expected-calls=28'
ALL = dict(int8_dequant={}, parallel_rope={}, parallel_trig=True, causal_mask={}, kv_storage='fp16', kv_dma=True)


def test_no_features_select_no_passes():
    assert all(fused_graph_passes(d, {}) == [] for d in ('initial_single', 'initial', 'prefill', 'decode'))


def test_accepted_per_domain_order():
    assert fused_graph_passes('initial', ALL) == [DQ, RP, TR, KV]
    assert fused_graph_passes('prefill', ALL) == [DQ, RP, TR, MK, KV]
    assert fused_graph_passes('decode', ALL) == [DQ, RP, KV]
    assert fused_graph_passes('initial_single', ALL) == []


def test_q8_0_projections_fuse_the_single_token_domain_too():
    features = dict(ALL, q8_0_projections={'hf_checkpoint': 'model.safetensors'})
    assert fused_graph_passes('initial_single', features) == [DQ]
    assert fused_graph_passes('prefill', features) == [DQ, RP, TR, MK, KV]


def test_features_are_independent():
    assert fused_graph_passes('prefill', dict(causal_mask={})) == [MK]
    assert fused_graph_passes('decode', dict(kv_storage='fp16')) == [KV]
    assert set(ALL) | {'q8_0_projections'} == FUSED_GRAPH_KEYS
