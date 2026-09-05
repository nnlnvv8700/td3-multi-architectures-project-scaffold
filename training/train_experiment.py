"""Canonical TD3 training entry point.

All experiment configuration, observation handling, evaluation and persistence live
in dedicated modules. Run with:

    python -m training.train_experiment --actor_arch gnn_transformer
"""

import os
import logging
import random
import sys
from contextlib import ExitStack
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym
import numpy as np
import torch

import envs  # noqa: F401 - register custom environment
from agents.td3_agent import TD3, resolve_device
from training.artifacts import append_summary, create_run_dir, save_progress
from training.config import environment_kwargs, finalize_config, parse_args
from training.config import get_default_args as get_default_args  # legacy public import
from training.evaluator import PolicyEvaluator
from training.metrics import extract_distance, is_success_state
from training.observation import flatten_obs, infer_dimensions
from utils.gym_compat import reset_env, step_env
from utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(config: Any, evaluation: bool = False):
    kwargs = {
        "render_mode": "rgb_array",
        "dense_reward": True,
        "observation_version": config.observation_version,
        **environment_kwargs(vars(config)),
    }
    env = gym.make(config.env, **kwargs)
    seed = config.eval_seed if evaluation else config.seed
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def new_metrics() -> Dict[str, List]:
    return {
        "actor_loss": [],
        "actor_loss_t": [],
        "critic_loss": [],
        "critic_loss_t": [],
        "eval_steps": [],
        "eval_rewards": [],
        "eval_success": [],
        "eval_tts": [],
        "eval_min_distance": [],
        "eval_rmse": [],
        "eval_max_dev": [],
        "eval_end_err": [],
        "eval_path_len_exec": [],
        "eval_path_len_ref": [],
    }


def run_evaluation(agent: TD3, env, config: Any, step: int, metrics: Dict[str, List]) -> Tuple:
    evaluation = PolicyEvaluator(env, agent, distance_threshold=config.distance_threshold,
                                 seed=config.eval_seed).evaluate(config.eval_episodes)
    result = (evaluation.mean_reward, evaluation.success_rate, evaluation.mean_tts,
              evaluation.mean_min_distance, evaluation.mean_rmse, evaluation.mean_max_deviation,
              evaluation.mean_endpoint_error, evaluation.mean_path_length_exec, evaluation.mean_path_length_ref)
    for key in ("endpoint_success_rate", "tracking_success_rate", "mean_action_rate", "mean_jerk"):
        metrics.setdefault("eval_" + key, []).append(float(getattr(evaluation, key)))
    keys = (
        "eval_rewards", "eval_success", "eval_tts", "eval_min_distance",
        "eval_rmse", "eval_max_dev", "eval_end_err",
        "eval_path_len_exec", "eval_path_len_ref",
    )
    metrics["eval_steps"].append(int(step))
    for key, value in zip(keys, result):
        metrics[key].append(float(value))
    logger.info(
        f"[eval step={step:,}] reward={result[0]:.3f} "
        f"success={result[1]:.1%} tts={result[2]:.1f} steps "
        f"min_distance={result[3]:.4f}m rmse={result[4]:.4f}m"
    )
    if config.algorithm_version == "corrected":
        rank = [-evaluation.tracking_success_rate, evaluation.mean_rmse, evaluation.mean_endpoint_error]
        previous = getattr(config, "best_evaluation", None)
        if np.isfinite(rank).all() and (previous is None or rank < previous["rank"]):
            agent.save(os.path.join(config.save_dir, "best_model"))
            config.best_evaluation = {"step": step, "rank": rank,
                                      "selection_split": "validation", "seed": config.eval_seed}
    return result


def run_experiment(config: Any) -> str:
    """Validate before starting and close every created environment on all exits."""
    config = finalize_config(config)
    resolve_device(config.device)
    if config.seed is not None:
        set_global_seed(config.seed)
    with ExitStack() as resources:
        if config.torch_threads is not None:
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(config.torch_threads)
            resources.callback(torch.set_num_threads, previous_threads)
        env = make_env(config, evaluation=False)
        resources.callback(env.close)
        eval_env = make_env(config, evaluation=True)
        resources.callback(eval_env.close)
        create_run_dir(config)
        handler = logging.FileHandler(os.path.join(config.save_dir, "run.log"), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        previous_level = logger.level
        logger.setLevel(config.log_level)
        logger.addHandler(handler)
        resources.callback(logger.setLevel, previous_level)
        resources.callback(handler.close)
        resources.callback(logger.removeHandler, handler)
        return _run_experiment(config, env, eval_env)


def _run_experiment(config: Any, env, eval_env) -> str:
    run_dir = config.save_dir
    logger.info("[run] %s", run_dir)

    state_dim, action_dim, max_action = infer_dimensions(env)
    logger.info(
        f"[config] arch={config.actor_arch} state={state_dim} action={action_dim} "
        f"node_dim={config.node_dim} seed={config.seed}"
    )

    agent = TD3(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=config.buffer_size,
        her_prob=0.0,
        her_k=0,
        dense_reward=True,
        distance_threshold=float(env.unwrapped.distance_threshold),
        use_per=config.use_per,
        beta_frames=max(1, config.max_timesteps -
                        (max(config.start_timesteps, config.batch_size - 1)
                         if config.algorithm_version == "corrected" else config.start_timesteps)),
        alpha=config.per_alpha,
        beta_start=config.per_beta_start,
        per_mode="proportional" if config.algorithm_version == "corrected" else "legacy",
    )

    metrics = new_metrics()
    rewards: List[float] = []
    successes: List[int] = []
    episodes: List[Dict[str, Any]] = []
    obs, _ = reset_env(env, seed=config.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_success = False
    episode_index = 0
    completed_steps = 0
    interrupted = False
    episode_components = {}
    episode_squared_error = 0.0

    try:
        run_evaluation(agent, eval_env, config, 0, metrics)
        for t in range(config.max_timesteps):
            episode_steps += 1

            if t < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(flatten_obs(obs), deterministic=True)
                noise_std = exploration_noise(config, t)
                if noise_std > 0:
                    action = action + np.random.normal(0.0, noise_std, size=action_dim)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, reward, done, info = step_env(env, action)
            terminal = info["terminated"] if config.algorithm_version == "corrected" else done
            # Critic models policy commands; residual transformation is part of the environment.
            replay_buffer.add(obs, action, next_obs, reward, done, terminal=terminal)
            completed_steps = t + 1
            obs = next_obs
            episode_reward += float(reward)
            episode_squared_error += float(info.get("tracking_error", 0.0)) ** 2
            for key, value in info.get("reward_components", {}).items():
                episode_components[key] = episode_components.get(key, 0.0) + float(value)
            episode_success = episode_success or is_success_state(obs, info, env)

            if t >= config.start_timesteps and replay_buffer.size >= config.batch_size:
                losses = agent.train(replay_buffer, config.batch_size)
                if isinstance(losses, dict):
                    metrics["actor_loss"].append(float(losses["actor_loss"]))
                    metrics["actor_loss_t"].append(completed_steps)
                    metrics["critic_loss"].append(float(losses["critic_loss"]))
                    metrics["critic_loss_t"].append(completed_steps)
                else:
                    metrics["critic_loss"].append(float(losses))
                    metrics["critic_loss_t"].append(completed_steps)

            if done:
                distance = extract_distance(obs, info, env)
                # Include the zero-error initial reference point, matching evaluation RMSE.
                episode_rmse = float(np.sqrt(episode_squared_error / (episode_steps + 1)))
                if config.reward_mode == "tracking":
                    episode_success = (distance < config.distance_threshold and
                                       episode_rmse < config.tracking_success_threshold)
                rewards.append(episode_reward)
                successes.append(int(episode_success))
                episodes.append({
                    "episode": episode_index,
                    "global_step": completed_steps,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "success": int(episode_success),
                    "final_distance": distance,
                    "tracking_rmse": episode_rmse,
                    **{f"reward_{key}": value for key, value in episode_components.items()},
                })
                episode_index += 1
                obs, _ = reset_env(env)
                episode_reward = 0.0
                episode_steps = 0
                episode_success = False
                episode_squared_error = 0.0
                episode_components = {}

            if completed_steps % config.eval_freq == 0:
                if agent.last_diagnostics:
                    metrics.setdefault("diagnostics", []).append({"step": completed_steps, **agent.last_diagnostics})
                run_evaluation(agent, eval_env, config, completed_steps, metrics)
                save_progress(config, agent, metrics, rewards, successes, episodes)
                agent.save(os.path.join(config.save_dir, "checkpoint_latest"))

        if metrics["eval_steps"][-1] != completed_steps:
            run_evaluation(agent, eval_env, config, completed_steps, metrics)
        config.status = "completed"

    except KeyboardInterrupt:
        interrupted = True
        config.status = "interrupted"
        logger.warning("[interrupt] saving current progress")
    except Exception:
        config.status = "failed"
        logger.exception("Training failed after %s completed steps", completed_steps)
        raise
    finally:
        config.completed_timesteps = completed_steps
        config.interrupted = interrupted
        training_failed = sys.exc_info()[0] is not None
        try:
            save_progress(config, agent, metrics, rewards, successes, episodes, save_model=True)
            append_summary(config, metrics, rewards, successes)
        except Exception:
            logger.exception("Could not save final progress to %s", run_dir)
            if not training_failed:
                raise

    logger.info("[done] steps=%s episodes=%s run=%s", completed_steps, len(rewards), run_dir)
    return run_dir


def exploration_noise(config, step):
    """Linearly anneal physical policy-command noise after warm-up."""
    if config.expl_noise_final is None:
        return config.expl_noise
    progress = np.clip((step - config.start_timesteps) /
                       max(1, config.max_timesteps - config.start_timesteps - 1), 0.0, 1.0)
    return float(config.expl_noise + progress * (config.expl_noise_final - config.expl_noise))


def main() -> None:
    config = parse_args()
    configure_logging(config.log_level)
    run_experiment(config)


if __name__ == "__main__":
    main()
