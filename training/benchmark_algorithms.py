"""Paired algorithm ablations with separate validation and held-out goal seeds.

This is a bounded learning diagnostic, not evidence of convergence or superiority.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agents.td3_agent import TD3
from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from training.config import parse_args
from training.evaluator import PolicyEvaluator
from training.train_experiment import configure_logging, make_env, run_experiment

ROOT = Path(__file__).resolve().parents[1]


class ZeroPolicy:
    def select_action(self, state, deterministic=True):
        return np.zeros(7, dtype=np.float32)


def benchmark(output, steps=1000, seeds=(42, 123), architecture="mlp", episodes=5, threads=2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    records = []

    def record(label, policy, env, seed, checkpoint, run_dir=""):
        result = PolicyEvaluator(env, policy, distance_threshold=0.1, seed=20000).evaluate(episodes)
        row = {
            "variant": label,
            "seed": seed,
            "checkpoint": checkpoint,
            "split": "held_out_20000",
            "run_dir": str(run_dir),
            **result.to_dict(),
        }
        records.append(row)
        (output / "benchmark.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(records)
        print(
            f"[held-out] {label} seed={seed} checkpoint={checkpoint} "
            f"rmse={result.mean_rmse:.5f} endpoint={result.mean_endpoint_error:.5f} "
            f"tracking_success={result.tracking_success_rate:.1%}",
            flush=True,
        )

    for mode, label in [("direct", "zero_velocity"), ("residual", "dls_only")]:
        with KukaIiwa7TrackEnv(
            render_mode="rgb_array",
            reward_mode="tracking",
            control_mode=mode,
            joint_limit_margin=0.05,
        ) as env:
            record(label, ZeroPolicy(), env, None, "untrained")

    for seed in seeds:
        for variant in ("legacy", "reward_only", "corrected_direct", "corrected_residual"):
            argv = [
                "--actor_arch",
                architecture,
                "--max_timesteps",
                str(steps),
                "--start_timesteps",
                str(min(200, steps // 4)),
                "--batch_size",
                "64",
                "--buffer_size",
                str(max(1024, steps)),
                "--eval_freq",
                str(steps),
                "--eval_episodes",
                str(episodes),
                "--seed",
                str(seed),
                "--eval_seed",
                "10000",
                "--device",
                "cpu",
                "--torch_threads",
                str(threads),
                "--save_dir",
                str(output / variant),
            ]
            if variant in ("corrected_direct", "corrected_residual"):
                argv = ["--config", str(ROOT / "configs/algorithm_v2.yaml"), *argv]
                argv += [
                    "--control_mode",
                    "direct" if variant == "corrected_direct" else "residual",
                ]
            elif variant == "reward_only":
                argv += ["--reward_mode", "tracking"]
            config = parse_args(argv)
            run_dir = Path(run_experiment(config))
            with make_env(config, evaluation=True) as env:
                agent = TD3(31, 7, 1.5, config)
                agent.load_actor(run_dir / "final_model.pt")
                record(variant, agent, env, seed, "final", run_dir)
                if (run_dir / "best_model.pt").is_file():
                    agent.load_actor(run_dir / "best_model.pt")
                    record(variant, agent, env, seed, "validation_best", run_dir)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./results/algorithm_benchmark")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument(
        "--architecture", choices=["mlp", "gnn", "transformer", "gnn_transformer"], default="mlp"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 64 or args.episodes <= 0 or args.threads <= 0:
        parser.error("steps must be at least 64; episodes and threads must be positive")
    configure_logging()
    benchmark(args.output, args.steps, args.seeds, args.architecture, args.episodes, args.threads)


if __name__ == "__main__":
    main()
