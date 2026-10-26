"""Cache demand follows actual autoregressive forwards, not output length."""
import pytest
from hexagon_qwen_generation_budget import validate_cache_demand
from run_qwen_arm_dynamic_warm_generate import validate_requests


@pytest.mark.parametrize('prompt,budget',[(1024,1),(993,32),(1,1024)])
def test_last_physical_KV_slot_is_admitted(prompt,budget):
    assert validate_cache_demand(prompt,budget)==1024
    validate_requests([[1]*prompt],budget,32)


@pytest.mark.parametrize('prompt,budget',[(1025,1),(1024,2),(993,33),(1,1025)])
def test_first_unavailable_slot_is_rejected(prompt,budget):
    with pytest.raises(ValueError):validate_cache_demand(prompt,budget)
    with pytest.raises(ValueError):validate_requests([[1]*prompt],budget,32)


@pytest.mark.parametrize('prompt,budget',[(0,1),(1,0),(True,1),(1,False)])
def test_invalid_lengths_are_not_resource_budgets(prompt,budget):
    with pytest.raises(ValueError):validate_cache_demand(prompt,budget)

