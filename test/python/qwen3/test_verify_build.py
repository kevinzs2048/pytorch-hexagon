"""Build comparisons must preserve native EOS and the requested generation limit."""
import json
from pathlib import Path

import pytest

from qwen3.verify_build import reference_generation_options


def test_early_eos_replays_the_requested_limit_not_observed_token_count():
    reference = dict(generation_options=dict(max_new_tokens=32, prefill_chunk_size=8, do_sample=False),
                     requests=[dict(generated_ids=[1, 2, 3]), dict(generated_ids=[4, 3])])
    assert reference_generation_options(reference) == (32, 8)


def test_compact_completed_reference_preserves_original_budget():
    path = Path(__file__).resolve().parents[3]/'qwen3/validation/reference16.json'
    assert reference_generation_options(json.loads(path.read_text())) == (8, 32)


def test_early_eos_without_original_budget_cannot_be_reconstructed():
    reference = dict(requests=[dict(generated_ids=[1, 2], terminal_observation=dict(
        reached_requested_token_limit=False, last_token_is_native_eos=True))])
    with pytest.raises(ValueError, match='original generation_options'):
        reference_generation_options(reference)
