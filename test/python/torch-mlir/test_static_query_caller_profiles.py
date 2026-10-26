import pytest

from prepare_qwen_static_query_entry import captured_profiles, bounded_profiles


def test_static_query_abi_has_no_invented_leading_scalar():
    contract = dict(inputs=[dict(kind='tensor', dtype='int64', shape=[1, 1]),
                            dict(kind='scalar', value=1024)])
    runtime = [dict(kind='tensor', dtype='torch.int64', shape=[1, 1]), dict(kind='scalar', value=1024)]
    profiles = captured_profiles(contract, runtime)
    assert len(profiles) == 2 and profiles[0].input_type == 'tensor'
    assert profiles[1].idx == 0 and profiles[1].input_id == 1
    assert profiles[0].shape == ('{1, 1}', '{1, 1}')
    from_contract = captured_profiles(contract)
    assert [vars(row) for row in from_contract] == [vars(row) for row in profiles]
    runtime[0]['shape'] = [1, 2]
    with pytest.raises(ValueError, match='shape/dtype'):
        captured_profiles(contract, runtime)


def test_contract_cannot_invent_a_symbolic_q1_tensor():
    with pytest.raises(ValueError, match='shape/dtype'):
        captured_profiles(dict(inputs=[dict(kind='tensor', dtype='int64', shape=[1, 'query'])]))


def test_bounded_c_declaration_keeps_all_repeated_scalar_arguments():
    contract = dict(variables={'query': [2, 32]}, inputs=[
        dict(kind='scalar', value='query'), dict(kind='tensor', dtype='int64', shape=[1, 'query']),
        dict(kind='scalar', value='query'), dict(kind='scalar', value=1024)])
    profiles = bounded_profiles(contract)
    assert len(profiles) == 4 and [p.input_type for p in profiles] == ['scalar', 'tensor', 'scalar', 'scalar']
    assert profiles[0].idx == 0 and profiles[2].idx == 1 and profiles[3].idx == 2
    assert contract['inputs'][0]['value'] == contract['inputs'][2]['value'] == 'query'
