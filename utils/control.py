"""Kinematic baseline for optional residual velocity control."""

import numpy as np


def damped_least_squares(jacobian, cartesian_velocity, damping=0.05):
    """Solve J^T (J J^T + lambda^2 I)^-1 v without an explicit inverse."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    velocity = np.asarray(cartesian_velocity, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != 3 or velocity.shape != (3,):
        raise ValueError("Expected a (3, n) Jacobian and a (3,) Cartesian velocity")
    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    if not np.isfinite(jacobian).all() or not np.isfinite(velocity).all():
        raise ValueError("Controller inputs must be finite")
    return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(3), velocity)
