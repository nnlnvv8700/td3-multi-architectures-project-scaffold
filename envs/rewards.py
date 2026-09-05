"""Versioned, dimensionless trajectory costs with no early-goal bonus."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingReward:
    position_scale: float = 0.10
    terminal_scale: float = 0.05
    position_weight: float = 1.0
    smooth_weight: float = 0.02
    terminal_weight: float = 2.0

    def __post_init__(self):
        for name in ("position_scale", "terminal_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("position_weight", "smooth_weight", "terminal_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def __call__(
        self, achieved, reference, goal, action, previous_action, action_limit, terminal=False
    ):
        error = float(np.linalg.norm(np.asarray(achieved) - reference))
        distance = float(np.linalg.norm(np.asarray(achieved) - goal))
        # Pseudo-Huber retains a useful large-error slope without squared-cost blowup.
        tracking = -(np.hypot(1.0, error / self.position_scale) - 1.0)
        smooth = -float(np.mean(((np.asarray(action) - previous_action) / action_limit) ** 2))
        endpoint = -(np.hypot(1.0, distance / self.terminal_scale) - 1.0) if terminal else 0.0
        components = {
            "tracking": float(self.position_weight * tracking),
            "smoothness": float(self.smooth_weight * smooth),
            "terminal": float(self.terminal_weight * endpoint),
        }
        return float(sum(components.values())), components
