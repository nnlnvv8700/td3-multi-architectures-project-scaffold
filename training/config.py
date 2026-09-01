"""Single source of truth for experiment configuration."""

import argparse
from typing import Any, Dict, Optional, Sequence


ARCHITECTURE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mlp": {"node_dim": None, "num_nodes": None, "use_state_encoder": False},
    "gnn": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
    "transformer": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
    "gnn_transformer": {"node_dim": 6, "num_nodes": 7, "use_state_encoder": True},
}


def get_default_args(arch: str = "mlp") -> Dict[str, Any]:
    defaults = ARCHITECTURE_DEFAULTS.get(arch, ARCHITECTURE_DEFAULTS["mlp"])
    return {"actor_arch": arch, **defaults}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TD3 on KUKA trajectory tracking")
    parser.add_argument("--env", default="KukaIiwa7Track-v0")
    parser.add_argument("--actor_arch", default="gnn_transformer",
                        choices=list(ARCHITECTURE_DEFAULTS))
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
    return parser


def finalize_config(config: Any) -> Any:
    defaults = get_default_args(config.actor_arch)
    if config.node_dim is None:
        config.node_dim = defaults["node_dim"]
    if config.num_nodes is None:
        config.num_nodes = defaults["num_nodes"]
    config.use_state_encoder = defaults["use_state_encoder"]
    if config.max_timesteps <= 0 or config.start_timesteps < 0:
        raise ValueError("Training timesteps must be non-negative and max_timesteps must be positive")
    if config.batch_size <= 0 or config.buffer_size < config.batch_size:
        raise ValueError("buffer_size must be at least batch_size")
    if config.eval_freq <= 0 or config.eval_episodes <= 0:
        raise ValueError("eval_freq and eval_episodes must be positive")
    if config.start_timesteps > config.max_timesteps:
        raise ValueError("start_timesteps cannot exceed max_timesteps")
    return config


def parse_args(argv: Optional[Sequence[str]] = None) -> Any:
    return finalize_config(build_parser().parse_args(argv))
