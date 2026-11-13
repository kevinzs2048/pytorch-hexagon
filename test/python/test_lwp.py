# SPDX-License-Identifier: BSD-3-Clause
import pytest
from hexagon_runtime.lwp import read_map, summarize


def test_nested_and_repeated_sites():
    rows = {r['id']:r for r in summarize([
        {'id':1,'cyc':0}, {'id':2,'cyc':2}, {'id':2,'cyc':5},
        {'id':2,'cyc':6}, {'id':2,'cyc':10}, {'id':1,'cyc':12}])}
    assert rows[1]['inclusive_cycles'] == 12
    assert rows[1]['exclusive_cycles'] == 5
    assert rows[2]['calls'] == 2
    assert rows[2]['inclusive_cycles'] == rows[2]['exclusive_cycles'] == 7
    assert rows[2]['parents'] == [1]


@pytest.mark.parametrize('entries', [
    [], [{'id':1,'cyc':0}],
    [{'id':1,'cyc':10},{'id':1,'cyc':9}],
    [{'id':1,'cyc':0},{'id':2,'cyc':1},{'id':1,'cyc':2}],
])
def test_reject_invalid_or_incomplete_capture(entries):
    with pytest.raises(ValueError): summarize(entries)


def test_reject_duplicate_site_map(tmp_path):
    path=tmp_path/'map.txt'
    path.write_text('Location 12 corresponds to ID 1 | Collected ops: \n' * 2)
    with pytest.raises(ValueError, match='Duplicate site ID'): read_map(path)
