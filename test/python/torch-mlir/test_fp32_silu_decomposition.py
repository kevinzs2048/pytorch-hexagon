import pytest
import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

from fp32_silu_decomposition import with_fp32_silu_division


class SiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


@pytest.mark.parametrize('shape', [(), (1,), (32,), (33,), (1, 4, 3072)])
def test_fp32_uses_direct_division_without_changing_caller(shape):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(17))
    raw = torch.export.export(SiLU(), (x,))
    before = str(raw.graph)
    table = fx.get_decomposition_table()
    old_silu = table[torch.ops.aten.silu.default]
    revised = with_fp32_silu_division(table)
    assert table[torch.ops.aten.silu.default] is old_silu
    assert all(revised[k] is v for k, v in table.items() if k != torch.ops.aten.silu.default)
    program = raw.run_decompositions(revised)
    assert str(raw.graph) == before
    value = program.module()(x)
    expected = x / (1.0 + torch.exp(-x))
    assert torch.equal(value.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    targets = [n.target for n in program.graph.nodes if n.op == 'call_function']
    assert torch.ops.aten.div.Tensor in targets
    assert torch.ops.aten.sigmoid.default not in targets
    assert torch.ops.aten.mul.Tensor not in targets


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
def test_other_dtypes_keep_the_original_decomposition(dtype):
    x = torch.tensor([-5., -0., 0., .25, 3.], dtype=dtype)
    table = fx.get_decomposition_table()
    revised = with_fp32_silu_division(table)
    original = torch.export.export(SiLU(), (x,))
    a = original.run_decompositions(table).module()(x)
    b = original.run_decompositions(revised).module()(x)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_explicit_multiply_sigmoid_is_not_reassociated():
    class Explicit(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
    x = torch.linspace(-5, 5, 1024)
    raw = torch.export.export(Explicit(), (x,))
    table = fx.get_decomposition_table()
    a = raw.run_decompositions(table).module()(x)
    b = raw.run_decompositions(with_fp32_silu_division(table)).module()(x)
    assert torch.equal(a, b)


def test_linalg_contains_direct_division_not_multiply_by_reciprocal():
    x = torch.ones(4, 128)
    table = with_fp32_silu_division(fx.get_decomposition_table())
    text = str(fx.export_and_import(SiLU(), x, decomposition_table=table,
                                   output_type=OutputType.LINALG_ON_TENSORS))
    assert text.count('math.exp ') == 1
    assert text.count('arith.divf ') == 1
    assert 'arith.mulf ' not in text


def test_missing_original_entry_is_rejected():
    with pytest.raises(ValueError, match='standard'):
        with_fp32_silu_division({})


def test_signed_zero_and_nonfinite_values_follow_the_explicit_formula():
    x = torch.tensor([-0., 0., float('-inf'), float('inf'), float('nan')])
    fn = with_fp32_silu_division(fx.get_decomposition_table())[torch.ops.aten.silu.default]
    a, b = fn(x), x / (1.0 + torch.exp(-x))
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    mask = ~torch.isnan(a)
    assert torch.equal(a[mask].view(torch.uint8), b[mask].view(torch.uint8))
