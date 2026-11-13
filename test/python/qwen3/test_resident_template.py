"""The capture workaround cannot change embedding arithmetic or active options."""
from types import SimpleNamespace

import pytest
import torch

from models.qwen3.resident import normalize_unused_embedding_options


def model(embedding):
    return SimpleNamespace(get_input_embeddings=lambda: embedding)


def test_unused_default_norm_preserves_weights_and_embedding_outputs():
    embedding = torch.nn.Embedding(16, 8, padding_idx=0)
    ids = torch.tensor([[0, 1, 5, 15]])
    before = embedding(ids).detach().clone()
    weights = embedding.weight.detach().clone()
    assert normalize_unused_embedding_options(model(embedding))
    assert type(embedding.norm_type) is int and embedding.norm_type == 2
    assert torch.equal(embedding(ids).view(torch.int32), before.view(torch.int32))
    assert torch.equal(embedding.weight.view(torch.int32), weights.view(torch.int32))
    assert not normalize_unused_embedding_options(model(embedding))


@pytest.mark.parametrize('max_norm,norm_type', [(1.0, 2.0), (None, 1.5)])
def test_active_or_nondefault_options_remain_unchanged(max_norm, norm_type):
    embedding = torch.nn.Embedding(16, 8, max_norm=max_norm, norm_type=norm_type)
    assert not normalize_unused_embedding_options(model(embedding))
    assert embedding.max_norm == max_norm
    assert type(embedding.norm_type) is float and embedding.norm_type == norm_type


def test_custom_embedding_implementation_is_not_rewritten():
    class CustomEmbedding(torch.nn.Embedding):
        pass
    embedding = CustomEmbedding(16, 8)
    assert not normalize_unused_embedding_options(model(embedding))
    assert type(embedding.norm_type) is float


@pytest.mark.parametrize('recording,expected', [(None, 'none'), ('hashes', 'hashes'), ('full', 'full')])
def test_resident_diagnostic_selection_preserves_explicit_modes(monkeypatch, recording, expected):
    from models.qwen3 import cli, resident
    observed = []
    monkeypatch.setattr(resident, 'generate', lambda args: observed.append(args.packet_recording))
    cli.generate(SimpleNamespace(resident_session='session', packet_recording=recording))
    assert observed == [expected]


def test_no_recording_is_rejected_for_legacy_entry_before_loading():
    from models.qwen3 import cli
    with pytest.raises(ValueError, match='requires --resident-session'):
        cli.generate(SimpleNamespace(resident_session=None, packet_recording='none'))
