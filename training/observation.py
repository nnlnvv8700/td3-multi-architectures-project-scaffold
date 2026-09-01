"""Observation helpers shared by training, evaluation and tests."""

from typing import Any, Tuple

import numpy as np


def flatten_obs(obs: Any) -> np.ndarray:
    """Flatten a GoalEnv-style observation in one canonical field order."""
    if isinstance(obs, dict):
        return np.concatenate(
            [obs["observation"], obs["achieved_goal"], obs["desired_goal"]],
            axis=0,
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def infer_dimensions(env) -> Tuple[int, int, float]:
    """Return state dimension, action dimension and symmetric action bound."""
    if hasattr(env.observation_space, "spaces") and "observation" in env.observation_space.spaces:
        obs_dim = int(np.prod(env.observation_space["observation"].shape))
        achieved_dim = int(np.prod(env.observation_space["achieved_goal"].shape))
        desired_dim = int(np.prod(env.observation_space["desired_goal"].shape))
        state_dim = obs_dim + achieved_dim + desired_dim
    else:
        state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(np.max(np.abs(env.action_space.high)))
    return state_dim, action_dim, max_action
