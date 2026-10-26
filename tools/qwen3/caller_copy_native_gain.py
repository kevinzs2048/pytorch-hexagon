"""Caller copying must improve callback time while preserving kernel timing."""
import math


def positive_native_gain(timing):
    if set(timing) != {'1', '13', '14'}:
        return False
    for query, metrics in timing.items():
        if set(metrics) != {'kernel_seconds', 'callback_seconds'}:
            return False
        for row in metrics.values():
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('baseline', 'candidate', 'ratio')):
                return False
            if not math.isclose(row['ratio'], row['candidate']/row['baseline'], rel_tol=1e-12):
                return False
        if metrics['kernel_seconds']['ratio'] > 1.02 or metrics['callback_seconds']['ratio'] > 1.02:
            return False
    return timing['1']['callback_seconds']['ratio'] < 0.995
