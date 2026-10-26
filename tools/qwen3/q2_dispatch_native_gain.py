"""Require native prefill gain while preserving the unchanged q1 decode path."""
import math


def positive_native_gain(timing):
    if set(timing) != {'1','13','14'}:
        return False
    for metrics in timing.values():
        if set(metrics) != {'kernel_seconds','callback_seconds'}:
            return False
        for row in metrics.values():
            if (any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k] <= 0
                    for k in ('baseline','candidate','ratio'))
                    or not math.isclose(row['ratio'],row['candidate']/row['baseline'],rel_tol=1e-12)):
                return False
    if any(row['ratio'] > 1.02 for row in timing['1'].values()):
        return False
    return all(row['ratio'] < 0.995 for q in ('13','14') for row in timing[q].values())
