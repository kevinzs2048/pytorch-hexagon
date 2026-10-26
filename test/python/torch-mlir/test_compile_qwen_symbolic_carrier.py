"""Synthetic ABI derivation checks, not model compilation or numerical tests."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from compile_qwen_symbolic_carrier import derive_runtime_contract
from hexagon_qwen_shape_policy import contract_query_bounds, query_bounds
from qcom_hexagon_backend.backend.hexagon_captured_buffer_contract import derive_captured_contract


def graph_and_signature(*, singleton=False, extra=0, fixed_query=None, bounds=(2, 32),
                        query_scalar=True):
    env = ShapeEnv()
    fixed_query = 1 if singleton else fixed_query
    q = fixed_query if fixed_query is not None else env.create_unbacked_symint()
    if fixed_query is None:
        env.constrain_symbol_range(q.node.expr, *bounds)
    with FakeTensorMode(shape_env=env):
        tensors = [torch.empty(1, q, dtype=torch.int64)]
        tensors += [torch.empty(1, q) for _ in range(59)]
    graph = torch.fx.Graph()
    nodes = []
    scalars = [q]*(3+extra) if query_scalar else [1, 1024]
    for index, value in enumerate([*tensors, *scalars]):
        node = graph.placeholder(f'arg{index}')
        node.meta['val'] = value
        nodes.append(node)
    graph.output(tuple(nodes[1:58]))
    extent = str(fixed_query) if fixed_query is not None else '?'
    types = [f'tensor<1x{extent}xi64>']+[f'tensor<1x{extent}xf32>']*59
    signature = '('+', '.join(types+['i64']*len(scalars))+') -> ('+', '.join(types[1:58])+')'
    return torch.fx.GraphModule({}, graph), signature


def fixture(tmp_path):
    graph, signature = graph_and_signature()
    base = derive_captured_contract(graph, list(range(63)), signature,
        variables={'query': (2, 32)}, scalar_bindings={60: 'query', 61: 'query', 62: 'query'})
    base = json.loads(json.dumps(base))
    return dict(entries=[{'contract': base}]), {'directory': str(tmp_path)}


@pytest.mark.parametrize('singleton,extra,generated', [(False, 0, False), (True, 0, True), (False, 2, True)])
def test_contract_comes_from_actual_captured_metadata(tmp_path, singleton, extra, generated):
    catalog, frontend = fixture(tmp_path)
    graph, signature = graph_and_signature(singleton=singleton, extra=extra)
    (tmp_path/'lowering.json').write_text(json.dumps({'signatures': [signature]}))
    contract, actual = derive_runtime_contract(graph, SimpleNamespace(retained=list(range(63+extra))),
                                              frontend, catalog)
    assert actual is generated
    assert contract['variables'] == {'query': [1, 1] if singleton else [2, 32]}
    assert len(contract['inputs']) == 63+extra
    assert len(contract['outputs']) == 57


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_ambiguous_base_policy_rejected(tmp_path, count):
    catalog, frontend = fixture(tmp_path)
    catalog['entries'] *= count
    graph, _ = graph_and_signature()
    with pytest.raises(ValueError, match='One admitted base'):
        derive_runtime_contract(graph, SimpleNamespace(retained=list(range(63))), frontend, catalog)


@pytest.mark.parametrize('fixed_query', [1, 8, 32])
@pytest.mark.parametrize('query_scalar', [False, True])
def test_fixed_capture_keeps_actual_arity_and_fixed_query(tmp_path, fixed_query, query_scalar):
    catalog, frontend = fixture(tmp_path)
    graph, signature = graph_and_signature(fixed_query=fixed_query, query_scalar=query_scalar)
    (tmp_path/'lowering.json').write_text(json.dumps({'signatures': [signature]}))
    arity = 63 if query_scalar else 62
    contract, generated = derive_runtime_contract(graph, SimpleNamespace(retained=list(range(arity))),
                                                  frontend, catalog)
    assert generated
    assert contract_query_bounds(contract) == (fixed_query, fixed_query)
    assert len(contract['inputs']) == arity
    assert contract['inputs'][0]['shape'] == [1, fixed_query]
    assert all(type(row['value']) is int for row in contract['inputs'] if row['kind'] == 'scalar')


def test_symbolic_contract_respects_narrower_capture_guards(tmp_path):
    catalog, frontend = fixture(tmp_path)
    graph, signature = graph_and_signature(bounds=(4, 16))
    (tmp_path/'lowering.json').write_text(json.dumps({'signatures': [signature]}))
    contract, generated = derive_runtime_contract(graph, SimpleNamespace(retained=list(range(63))),
                                                  frontend, catalog)
    assert generated
    assert contract_query_bounds(contract) == (4, 16)
    assert contract['inputs'][0]['shape'] == [1, 'query']


@pytest.mark.parametrize('value', [0, 33, True])
def test_invalid_fixed_query_rejected(value):
    with pytest.raises(ValueError):
        query_bounds(value)


def test_symbolic_range_outside_budget_rejected():
    graph, _ = graph_and_signature(bounds=(33, 64))
    extent = next(iter(graph.graph.nodes)).meta['val'].shape[1]
    with pytest.raises(ValueError, match='does not intersect'):
        query_bounds(extent)


@pytest.mark.parametrize('variables', [{'query': [0, 8]}, {'query': [1, 33]},
                                      {'query': [True, 8]}, {'query': [8, 7]},
                                      {'query': [1, 8], 'other': [1, 1]}])
def test_caller_resource_policy_rejects_invalid_bounds(variables):
    with pytest.raises(ValueError):
        contract_query_bounds({'variables': variables})
