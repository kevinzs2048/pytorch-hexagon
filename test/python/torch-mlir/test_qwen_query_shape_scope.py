"""Real Dynamo boundary tests; CPU execution here is a synthetic test reference."""
import pytest
import torch

from hexagon_qwen_shape_policy import (annotate_query_inputs, compiled_query_scope,
                                      query_bounds, contract_query_bounds)


def inputs(query, past=0):
    positions = torch.arange(past, past+query)
    return dict(input_ids=torch.arange(query).reshape(1, query),
                cache_position=positions, position_ids=positions.unsqueeze(0),
                attention_mask=torch.ones(1, past+query, dtype=torch.int64))


class NativeCall(torch.nn.Module):
    def __init__(self, captures):
        super().__init__()
        self.captures = captures

    def forward(self, input_ids, cache_position, position_ids, attention_mask):
        return input_ids+cache_position+position_ids+attention_mask[:, -input_ids.shape[1]:]

    def get_compiled_call(self, unused):
        if not hasattr(self, '_compiled_call'):
            def backend(graph, examples):
                self.captures.append(graph)
                return graph.forward
            self._compiled_call = torch.compile(self.__call__, backend=backend, fullgraph=True, dynamic=True)
        return self._compiled_call


def test_full_and_tail_queries_reuse_one_symbolic_capture_then_singleton():
    torch._dynamo.reset()
    captures = []
    model = NativeCall(captures)
    original = model.forward.__func__
    with compiled_query_scope(model):
        for query, past in [(8, 0), (6, 8), (5, 0)]:
            kwargs = inputs(query, past)
            expected = model.forward(**kwargs)
            assert torch.equal(model.get_compiled_call(None)(**kwargs), expected)
        assert len(captures) == 1
        kwargs = inputs(1, 14)
        assert torch.equal(model.get_compiled_call(None)(**kwargs), model.forward(**kwargs))
        assert len(captures) == 2
    assert model.forward.__func__ is original
    assert 'get_compiled_call' not in model.__dict__


def test_scope_restores_instance_getter_on_exception():
    model = NativeCall([])
    getter = lambda unused: model.forward
    model.get_compiled_call = getter
    with pytest.raises(RuntimeError, match='sentinel'):
        with compiled_query_scope(model):
            raise RuntimeError('sentinel')
    assert model.get_compiled_call is getter


@pytest.mark.parametrize('fault', ['position', 'capacity', 'query', 'rank'])
def test_bad_native_axes_rejected_before_compilation(fault):
    kwargs = inputs(8)
    if fault == 'position':
        kwargs['position_ids'] = torch.zeros(1, 7, dtype=torch.int64)
    elif fault == 'capacity':
        kwargs['attention_mask'] = torch.ones(1, 1025)
    elif fault == 'query':
        kwargs = inputs(33)
    else:
        kwargs['input_ids'] = kwargs['input_ids'].flatten()
    with pytest.raises(ValueError):
        annotate_query_inputs(kwargs)


@pytest.mark.parametrize('value', [0, 33, True])
def test_invalid_fixed_query_rejected(value):
    with pytest.raises(ValueError):
        query_bounds(value)


@pytest.mark.parametrize('variables', [{'query': [0, 8]}, {'query': [1, 33]},
                                      {'query': [True, 8]}, {'query': [8, 7]},
                                      {'query': [1, 8], 'other': [1, 1]}])
def test_caller_resource_policy_rejects_invalid_bounds(variables):
    with pytest.raises(ValueError):
        contract_query_bounds({'variables': variables})
