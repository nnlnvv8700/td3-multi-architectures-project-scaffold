"""Experiment directory, checkpoint and summary persistence."""

import csv
import json
import os
import io
import platform
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np
from utils.persistence import artifact_lock, atomic_output


def _save_json(path, value) -> None:
    with atomic_output(path) as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))


def create_run_dir(config: Any) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_suffix = "" if config.seed is None else f"_seed{config.seed}"
    name = f"{config.env}_{config.actor_arch}_dense_{stamp}{seed_suffix}"
    run_dir = os.path.abspath(os.path.join(config.save_dir, name))
    os.makedirs(run_dir, exist_ok=False)
    config.save_dir = run_dir
    with atomic_output(os.path.join(os.path.dirname(run_dir), "LATEST_RUN.txt")) as fh:
        fh.write(run_dir.encode("utf-8"))
    save_config(config)
    packages = {}
    for name in ("torch", "numpy", "gymnasium", "pybullet", "pyyaml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    _save_json(os.path.join(run_dir, "runtime.json"), {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
    })
    return run_dir


def save_config(config: Any) -> None:
    _save_json(os.path.join(config.save_dir, "config.json"), vars(config))


def _write_episode_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with io.StringIO(newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with atomic_output(path) as stream:
            stream.write(fh.getvalue().encode("utf-8"))


def save_progress(
    config: Any,
    agent: Any,
    metrics: Dict[str, List],
    rewards: List[float],
    successes: List[int],
    episodes: List[Dict[str, Any]],
    save_model: bool = False,
) -> None:
    os.makedirs(config.save_dir, exist_ok=True)
    with atomic_output(os.path.join(config.save_dir, "rewards.npy")) as stream:
        np.save(stream, np.asarray(rewards, dtype=np.float32))
    with atomic_output(os.path.join(config.save_dir, "success.npy")) as stream:
        np.save(stream, np.asarray(successes, dtype=np.int8))
    _save_json(os.path.join(config.save_dir, "metrics.json"), metrics)
    _write_episode_csv(os.path.join(config.save_dir, "episodes.csv"), episodes)
    save_config(config)
    if save_model:
        agent.save(os.path.join(config.save_dir, "final_model"))


def append_summary(
    config: Any, metrics: Dict[str, List], rewards: List[float], successes: List[int]
) -> str:
    """Write a stable v2 schema instead of appending to the malformed legacy CSV."""
    path = os.path.join(os.path.dirname(config.save_dir), "training_summary_v2.csv")
    last = lambda key: metrics.get(key, [float("nan")])[-1] if metrics.get(key) else float("nan")
    record = {
        "run_name": os.path.basename(config.save_dir),
        "actor_arch": config.actor_arch,
        "seed": config.seed,
        "timesteps": getattr(config, "completed_timesteps", config.max_timesteps),
        "episodes": len(rewards),
        "train_reward_mean": float(np.mean(rewards[-20:])) if rewards else float("nan"),
        "train_success_mean": float(np.mean(successes[-20:])) if successes else float("nan"),
        "eval_reward": last("eval_rewards"),
        "eval_success": last("eval_success"),
        "eval_tts_steps": last("eval_tts"),
        "eval_min_distance": last("eval_min_distance"),
        "eval_rmse": last("eval_rmse"),
        "eval_endpoint_error": last("eval_end_err"),
    }
    with artifact_lock(path):
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
    return path
