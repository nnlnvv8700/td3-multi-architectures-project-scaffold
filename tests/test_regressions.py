"""Engineering regressions around the existing TD3 algorithm."""

import csv
import json
import tempfile
import unittest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from agents.td3_agent import TD3
from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from training.artifacts import append_summary
from training.config import parse_args
from training.train_experiment import run_experiment
from utils.persistence import atomic_output


def small_config(directory, architecture="mlp", per=True):
    return parse_args(
        [
            "--actor_arch",
            architecture,
            "--device",
            "cpu",
            "--max_timesteps",
            "4",
            "--start_timesteps",
            "0",
            "--batch_size",
            "2",
            "--buffer_size",
            "8",
            "--eval_freq",
            "3",
            "--eval_episodes",
            "1",
            "--env_max_steps",
            "2",
            "--save_dir",
            str(directory),
            "--use_per" if per else "--no-use_per",
        ]
    )


class ConfigRegressionTests(unittest.TestCase):
    def test_single_point_smoothing_window_keeps_raw_series(self):
        from training.evaluate import build_series

        record = {"eval_steps": [0, 100], "eval_rewards": [1, 2]}
        with patch("training.evaluate.load_metrics", return_value=record):
            series = build_series(["mlp"], "eval_rewards", [], ma_window=1)
            np.testing.assert_array_equal(series["MLP"][1], [1, 2])
            self.assertIsNone(series["MLP"][4])

    def test_invalid_parameters_fail_at_startup(self):
        for option, value in [
            ("policy_delay", "0"),
            ("gamma", "1.1"),
            ("tau", "nan"),
            ("actor_lr", "0"),
            ("expl_noise", "-1"),
            ("seed", "-1"),
            ("env_max_steps", "0"),
            ("node_dim", "8"),
        ]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                parse_args([f"--{option}", value])

    def test_file_defaults_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("actor_arch: mlp\nbatch_size: 64\nuse_per: false\n", encoding="utf-8")
            config = parse_args(["--config", str(path), "--batch_size", "16"])
            self.assertEqual(config.batch_size, 16)
            self.assertEqual(config.actor_arch, "mlp")
            self.assertFalse(config.use_per)
            path.write_text("actor_lrr: 0.01\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                parse_args(["--config", str(path)])

    def test_invalid_device_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            config.device = "cuda:10000"
            with patch("agents.td3_agent.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    run_experiment(config)
            self.assertEqual(list(Path(directory).iterdir()), [])

class NetworkRegressionTests(unittest.TestCase):
    def test_legacy_v1_checkpoints_still_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for arch in ("mlp", "gnn", "transformer", "gnn_transformer"):
                with self.subTest(architecture=arch):
                    config = small_config(directory, arch)
                    if arch != "mlp":
                        config.node_dim, config.num_nodes = 5, 4
                        config.use_state_encoder = False
                    agent = TD3(20, 7, 1.0, config)
                    state = np.zeros(20, dtype=np.float32)
                    expected = agent.select_action(state)
                    agent.save(Path(directory) / arch)
                    agent.load_actor(Path(directory) / f"{arch}.pt")
                    np.testing.assert_array_equal(expected, agent.select_action(state))

    def test_mismatched_checkpoint_does_not_mutate_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TD3(31, 7, 1.5, small_config(directory))
            state = np.zeros(31, dtype=np.float32)
            expected = agent.select_action(state)
            weights = agent.actor.state_dict()
            weights["net.0.weight"] = torch.zeros(1)
            path = Path(directory) / "bad.pt"
            torch.save(weights, path)
            with self.assertRaisesRegex(ValueError, "Checkpoint does not match"):
                agent.load_actor(path)
            np.testing.assert_array_equal(expected, agent.select_action(state))

    def test_tensor_dtype_and_mode_restoration(self):
        agent = TD3(31, 7, 1.5, small_config("unused"))
        expected = agent.select_action(np.zeros(31, dtype=np.float32))
        actual = agent.select_action(torch.zeros(31, dtype=torch.float64))
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(agent.actor.training)
        with patch.object(agent.actor, "forward", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                agent.select_action(np.zeros(31))
        self.assertTrue(agent.actor.training)
        with self.assertRaises(ValueError):
            agent.select_action(np.zeros(20))

    def test_all_architectures_update_and_reload(self):
        # Both PER loss and uniform loss paths must reach delayed actor updates.
        with tempfile.TemporaryDirectory() as directory:
            for arch in ("mlp", "gnn", "transformer", "gnn_transformer"):
                for per in (False, True):
                    with self.subTest(architecture=arch, per=per):
                        config = small_config(Path(directory) / f"{arch}_{per}", arch, per)
                        run_dir = Path(run_experiment(config))
                        metrics = json.loads((run_dir / "metrics.json").read_text())
                        self.assertEqual(metrics["eval_steps"], [0, 3, 4])
                        self.assertEqual(len(metrics["critic_loss"]), 3)
                        self.assertEqual(len(metrics["actor_loss"]), 1)
                        self.assertTrue(np.isfinite(metrics["actor_loss"]).all())
                        self.assertTrue(np.isfinite(metrics["critic_loss"]).all())
                        agent = TD3(31, 7, 1.5, config)
                        agent.load_actor(run_dir / "final_model.pt")
                        state = np.zeros(31, dtype=np.float32)
                        first = agent.select_action(state)
                        agent.save(run_dir / "roundtrip")
                        agent.load_actor(run_dir / "roundtrip.pt")
                        np.testing.assert_array_equal(first, agent.select_action(state))


class LifecycleRegressionTests(unittest.TestCase):
    def test_batch_runner_never_overlaps_jobs_on_same_gpu(self):
        from training.run_all_experiments import ExperimentRunner

        runner = ExperimentRunner(architectures=["mlp", "gnn"], num_runs_per_arch=2,
                                  parallel=True, gpus=[0, 1])
        active = set()
        lock = threading.Lock()

        def fake_run(arch, run_id, gpu):
            with lock:
                self.assertNotIn(gpu, active)
                active.add(gpu)
            time.sleep(0.02 if gpu == 0 else 0.005)
            with lock:
                active.remove(gpu)
            return {"success": True, "architecture": arch, "run_id": run_id}

        with patch.object(runner, "run_single_experiment", side_effect=fake_run), \
             patch.object(runner, "_print_summary"), patch.object(runner, "_save_experiment_log"):
            runner.run_all_parallel()
        self.assertEqual(len(runner.experiment_log), 4)

    def test_invalid_action_does_not_advance_simulation(self):
        with KukaIiwa7TrackEnv(render_mode="rgb_array") as env:
            env.reset(seed=42)
            for action in (np.zeros(6), np.full(7, np.nan), np.zeros((1, 7))):
                with self.assertRaises(ValueError):
                    env.step(action)
                self.assertEqual(env.step_counter, 0)

    def test_second_environment_creation_failure_closes_first(self):
        env = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "training.train_experiment.make_env", side_effect=[env, RuntimeError("creation")]
            ):
                with self.assertRaisesRegex(RuntimeError, "creation"):
                    run_experiment(small_config(directory))
        env.close.assert_called_once()

    def test_agent_initialization_failure_closes_environments(self):
        envs = [KukaIiwa7TrackEnv(render_mode="rgb_array") for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            with patch("training.train_experiment.make_env", side_effect=envs):
                with patch("training.train_experiment.TD3", side_effect=RuntimeError("model init")):
                    with patch.object(envs[0], "close", wraps=envs[0].close) as close1:
                        with patch.object(envs[1], "close", wraps=envs[1].close) as close2:
                            with self.assertRaisesRegex(RuntimeError, "model init"):
                                run_experiment(small_config(directory))
                            close1.assert_called_once()
                            close2.assert_called_once()

    def test_save_failure_preserves_training_exception_and_closes_clients(self):
        envs = [KukaIiwa7TrackEnv(render_mode="rgb_array") for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_experiment.make_env", side_effect=envs),
                patch(
                    "training.train_experiment.run_evaluation",
                    side_effect=RuntimeError("evaluation failed"),
                ),
                patch("training.train_experiment.save_progress", side_effect=OSError("disk full")),
            ):
                with self.assertLogs("training.train_experiment", level="ERROR"):
                    with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                        run_experiment(small_config(directory))
        self.assertTrue(all(env._p_client is None for env in envs))

    def test_atomic_save_failure_keeps_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"previous")
            with self.assertRaises(OSError):
                with atomic_output(path) as stream:
                    stream.write(b"partial")
                    raise OSError("disk full")
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_summary_uses_actual_completed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(Path(directory) / "run")
            config.completed_timesteps = 1
            path = append_summary(config, {}, [], [])
            with open(path, newline="", encoding="utf-8") as stream:
                self.assertEqual(next(csv.DictReader(stream))["timesteps"], "1")


if __name__ == "__main__":
    unittest.main()
