"""Single source of truth for experiment configuration."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


ARCHITECTURE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mlp": {"node_dim": None, "num_nodes": None, "use_state_encoder": False},
    "gnn": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
    "transformer": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
    "gnn_transformer": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
}

ENVIRONMENT_DEFAULTS = {
    "env_max_steps": 200,
    "sim_steps_per_action": 10,
    "joint_vel_limit": 1.5,
    "distance_threshold": 0.10,
    "reward_mode": "legacy",
    "control_mode": "direct",
    "residual_scale": 0.2,
    "controller_gain": 4.0,
    "controller_damping": 0.05,
    "tracking_position_scale": 0.10,
    "tracking_terminal_scale": 0.05,
    "tracking_smooth_weight": 0.02,
    "tracking_terminal_weight": 2.0,
    "joint_limit_margin": 0.0,
    "tracking_success_threshold": 0.05,
}


def environment_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replay the saved simulation settings, including Gym's outer time limit."""
    values = {key: config.get(key, default) for key, default in ENVIRONMENT_DEFAULTS.items()}
    horizon = values.pop("env_max_steps")
    return {"max_steps": horizon, "max_episode_steps": horizon, **values}


def get_default_args(arch: str = "mlp") -> Dict[str, Any]:
    if arch not in ARCHITECTURE_DEFAULTS:
        raise ValueError(f"Unknown actor_arch: {arch}")
    defaults = ARCHITECTURE_DEFAULTS[arch]
    return {"actor_arch": arch, **defaults}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TD3 on KUKA trajectory tracking")
    parser.add_argument(
        "--config", help="JSON or YAML defaults; explicit CLI options take precedence"
    )
    parser.add_argument("--env", default="KukaIiwa7Track-v0")
    parser.add_argument(
        "--actor_arch", default="gnn_transformer", choices=list(ARCHITECTURE_DEFAULTS)
    )
    parser.add_argument("--max_timesteps", type=int, default=500_000)
    parser.add_argument("--start_timesteps", type=int, default=25_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--expl_noise", type=float, default=0.01)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.003)
    parser.add_argument("--policy_noise", type=float, default=0.05)
    parser.add_argument("--noise_clip", type=float, default=0.2)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--node_dim", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=10_000)
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--observation_version", type=int, default=2, choices=[1, 2])
    parser.add_argument("--use_kuka_pe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_per", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--algorithm_version", choices=["legacy", "corrected"], default="legacy")
    parser.add_argument("--zero_init_actor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--critic_loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--expl_noise_final", type=float, default=None)
    parser.add_argument("--torch_threads", type=int, default=None)
    for key, default in ENVIRONMENT_DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(default), default=default)
    parser.add_argument(
        "--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    return parser


def finalize_config(config: Any) -> Any:
    # Programmatic callers receive the same validation/defaults as the CLI.
    for key, value in vars(build_parser().parse_args([])).items():
        if not hasattr(config, key):
            setattr(config, key, value)
    defaults = get_default_args(config.actor_arch)
    if config.node_dim is None:
        config.node_dim = defaults["node_dim"]
    if config.num_nodes is None:
        config.num_nodes = defaults["num_nodes"]
    config.use_state_encoder = defaults["use_state_encoder"]
    if config.algorithm_version not in ("legacy", "corrected") or config.critic_loss not in ("huber", "mse"):
        raise ValueError("Invalid algorithm_version or critic_loss")
    config.evaluation_protocol = "full_trajectory_rmse_and_endpoint_v2" if config.reward_mode == "tracking" else "legacy_any_goal_entry"
    if config.reward_mode not in ("legacy", "tracking") or config.control_mode not in ("direct", "residual"):
        raise ValueError("Invalid reward_mode or control_mode")
    if (config.reward_mode == "tracking" or config.control_mode == "residual") and config.observation_version != 2:
        raise ValueError("Tracking/residual modes require observation_version=2")
    if not 0 <= config.residual_scale <= 1 or not 0 <= config.joint_limit_margin < 1:
        raise ValueError("Invalid residual_scale or joint_limit_margin")
    for key in ("controller_gain", "controller_damping", "tracking_position_scale", "tracking_terminal_scale", "tracking_success_threshold"):
        if not math.isfinite(getattr(config, key)) or getattr(config, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("tracking_smooth_weight", "tracking_terminal_weight", "per_alpha"):
        if not math.isfinite(getattr(config, key)) or getattr(config, key) < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if not 0 <= config.per_beta_start <= 1:
        raise ValueError("per_beta_start must be in [0, 1]")
    if config.expl_noise_final is not None and (not math.isfinite(config.expl_noise_final) or config.expl_noise_final < 0):
        raise ValueError("expl_noise_final must be finite and non-negative")
    if config.torch_threads is not None and config.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    for key in (
        "max_timesteps",
        "start_timesteps",
        "batch_size",
        "buffer_size",
        "eval_freq",
        "eval_episodes",
        "policy_delay",
        "env_max_steps",
        "sim_steps_per_action",
        "observation_version",
    ):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    for key in ("actor_lr", "critic_lr", "joint_vel_limit", "distance_threshold"):
        value = getattr(config, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("expl_noise", "policy_noise", "noise_clip"):
        value = getattr(config, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    for key in ("gamma", "tau"):
        if not math.isfinite(getattr(config, key)) or not 0 <= getattr(config, key) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in ("policy_delay", "env_max_steps", "sim_steps_per_action"):
        if getattr(config, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("seed", "eval_seed"):
        value = getattr(config, key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32
        ):
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    if config.observation_version not in (1, 2):
        raise ValueError("observation_version must be 1 or 2")
    if config.actor_arch != "mlp":
        if not isinstance(config.num_nodes, int) or config.num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer")
        if not isinstance(config.node_dim, int) or config.node_dim <= 0:
            raise ValueError("node_dim must be a positive integer")
        if config.num_nodes == 7 and config.node_dim not in (3, 4, 5, 6):
            raise ValueError("The seven-joint state encoder supports node_dim 3, 4, 5 or 6")
    if config.max_timesteps <= 0 or config.start_timesteps < 0:
        raise ValueError(
            "Training timesteps must be non-negative and max_timesteps must be positive"
        )
    if config.batch_size <= 0 or config.buffer_size < config.batch_size:
        raise ValueError("buffer_size must be at least batch_size")
    if config.eval_freq <= 0 or config.eval_episodes <= 0:
        raise ValueError("eval_freq and eval_episodes must be positive")
    if config.start_timesteps > config.max_timesteps:
        raise ValueError("start_timesteps cannot exceed max_timesteps")
    return config


def parse_args(argv: Optional[Sequence[str]] = None) -> Any:
    parser = build_parser()
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        path = Path(preliminary.config)
        with path.open(encoding="utf-8") as fh:
            if path.suffix.lower() == ".json":
                defaults = json.load(fh)
            else:
                import yaml

                defaults = yaml.safe_load(fh)
        if not isinstance(defaults, dict):
            raise ValueError("Configuration must be a mapping of training option names")
        actions = {
            action.dest: action
            for action in parser._actions
            if action.dest not in ("help", "config")
        }
        unknown = set(defaults) - actions.keys()
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in defaults.items():
            action = actions[key]
            if isinstance(action, argparse.BooleanOptionalAction) and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            if action.type is not None and value is not None:
                if action.type is int and (isinstance(value, bool) or isinstance(value, float)):
                    raise ValueError(f"{key} must be an integer")
                defaults[key] = action.type(value)
            if action.choices and defaults[key] not in action.choices:
                raise ValueError(f"Invalid {key}: {value!r}")
        parser.set_defaults(**defaults)
    return finalize_config(parser.parse_args(argv))
