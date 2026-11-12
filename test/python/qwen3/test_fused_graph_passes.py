"""Per-domain fused-graph policy passes of the resident builder, in the order
the accepted candidates were prepared."""
from models.qwen3.build_model import FUSED_GRAPH_KEYS, fused_graph_passes

DQ = '--qwen-fuse-i8-dequant=expected-count=197 expected-residual=56 expected-product=28'
RP = '--qwen-fuse-parallel-rope=expected-count=56'
HN = '--qwen-fuse-head-norm-rope=expected-count=56'
EM = '--qwen-embedding-rows=expected-count=1'
TR = '--qwen-fuse-parallel-trig=expected-count=2'
MK = '--qwen-fuse-causal-mask=expected-count=1'
KV = '--qwen-fp16-kv-storage=expected-calls=28'
ALL = dict(int8_dequant={}, parallel_rope={}, head_norm_rope={}, parallel_trig=True, causal_mask={}, kv_storage='fp16', kv_dma=True,
           embedding_rows=True)


def test_no_features_select_no_passes():
    assert all(fused_graph_passes(d, {}) == [] for d in ('initial_single', 'initial', 'prefill', 'decode'))


def test_accepted_per_domain_order():
    assert fused_graph_passes('initial', ALL) == [DQ, RP, HN, TR, KV, EM]
    assert fused_graph_passes('prefill', ALL) == [DQ, RP, HN, TR, MK, KV, EM]
    assert fused_graph_passes('decode', ALL) == [DQ, RP, KV, EM]
    assert fused_graph_passes('initial_single', ALL) == [EM]


def test_q8_0_projections_fuse_the_single_token_domain_too():
    features = dict(ALL, q8_0_projections={'hf_checkpoint': 'model.safetensors'})
    assert fused_graph_passes('initial_single', features) == [DQ, EM]
    assert fused_graph_passes('prefill', features) == [DQ, RP, HN, TR, MK, KV, EM]


def test_features_are_independent():
    assert fused_graph_passes('prefill', dict(causal_mask={})) == [MK]
    assert fused_graph_passes('decode', dict(kv_storage='fp16')) == [KV]
    assert set(ALL) | {'q8_0_projections'} == FUSED_GRAPH_KEYS
