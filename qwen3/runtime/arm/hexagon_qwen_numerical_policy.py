"""Check that a prepared Qwen graph has not silently skipped required policies.

This is compiler structure validation, not a numerical proof or DSP acceptance.
Counts follow the pinned 28-layer Qwen model, independent of query length.
"""
import re


def policy_counts(inspection):
    """Read markers from verified C++ printer output with large constants elided.

    The host torch-mlir Python parser cannot read every dialect property emitted
    by the external optimizer. This inspection is never used as compiler input.
    """
    markers = {
        'hexagon.fp32_arm_static_rmsnorm_sum': 'norm_rows',
        'hexagon.fp32_arm_rmsnorm_sum': 'norm_rows',
        'hexagon.fp32_arm_k_norm_sum': 'norm_rows',
        'hexagon.fp32_prefill_dot_order': 'attention_contractions',
        'hexagon.fp32_arm_softmax_normalization': 'softmax_rows',
        'hexagon.fp32_arm_silu_division': 'silu_rows',
        'hexagon.fp32_silu_exp_compat': 'silu_exponents',
        'hexagon.fp32_softmax_exp_compat': 'softmax_exponents',
    }
    counts = {}
    for marker, name in markers.items():
        counts[name] = counts.get(name, 0)+inspection.count(marker)
    counts['rope_trig_attributes'] = inspection.count('hexagon.fp32_arm_rope_trig4')
    # Batch-one products (including RoPE frequency construction) are not attention.
    # Attention contractions have the admitted 16-head batch dimension.
    counts['unconverted_fp32_attention'] = len(re.findall(
        r'\blinalg\.batch_matmul\b[^\n]*\bins\([^\n]*tensor<16x[^\n]*xf32>', inspection))
    return counts


def audit_policy(inspection, *, arm_rope_trig):
    counts = policy_counts(inspection)
    expected = dict(norm_rows=4*28+1, attention_contractions=2*28,
                    softmax_rows=28, silu_rows=28, silu_exponents=28,
                    softmax_exponents=28, unconverted_fp32_attention=0)
    if arm_rope_trig:
        expected['rope_trig_attributes'] = 3  # One function opt-in and two transformed maps.
    differences = {name: dict(expected=value, actual=counts.get(name, 0))
                   for name, value in expected.items() if counts.get(name, 0) != value}
    if differences:
        raise ValueError('Prepared Qwen numerical policies incomplete: '+str(differences))
    return dict(status='prepared_operator_policy_coverage_verified', counts=counts,
                expected=expected, constant_values_elided_for_inspection=True,
                numerical_acceptance=False, DSP_calls=0)
