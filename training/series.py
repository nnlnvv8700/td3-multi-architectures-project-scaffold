"""Align saved evaluations by training step before comparing random seeds."""

import numpy as np


def align_evaluations(records, metric_key):
    """Return common steps and per-run values; never average different checkpoints."""
    series = []
    for record in records:
        values = np.asarray(record.get(metric_key, []), dtype=float)
        if values.size == 0:
            continue
        steps = np.asarray(record.get("eval_steps", []), dtype=float)
        if steps.ndim != 1 or values.ndim != 1 or steps.size != values.size:
            raise ValueError(f"{metric_key}: eval_steps must exist and match the metric length")
        if not np.isfinite(steps).all() or np.any(np.diff(steps) <= 0):
            raise ValueError("eval_steps must be finite and strictly increasing")
        series.append((steps, values))
    if not series:
        return np.array([]), np.empty((0, 0))
    common = series[0][0]
    for steps, _ in series[1:]:
        common = np.intersect1d(common, steps)
    if not common.size:
        raise ValueError("Runs have no common evaluation steps")
    values = np.stack([y[np.searchsorted(x, common)] for x, y in series])
    return common / 10000.0, values
