import json
import os
import tempfile
import unittest

from training.config import parse_args
from training.train_experiment import run_experiment


class TrainingSmokeTest(unittest.TestCase):
    def test_canonical_entry_runs_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="td3_training_test_") as output_dir:
            config = parse_args([
                "--actor_arch", "mlp",
                "--max_timesteps", "2",
                "--start_timesteps", "0",
                "--batch_size", "2",
                "--buffer_size", "8",
                "--eval_freq", "2",
                "--eval_episodes", "1",
                "--save_dir", output_dir,
                "--seed", "123",
                "--eval_seed", "456",
                "--no-use_per",
            ])
            run_dir = run_experiment(config)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "final_model.pt")))
            with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as fh:
                metrics = json.load(fh)
            self.assertEqual(metrics["eval_steps"], [0, 2])
            self.assertEqual(len(metrics["critic_loss"]), 1)
            with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
                saved_config = json.load(fh)
            self.assertEqual(saved_config["observation_version"], 2)
            self.assertEqual(saved_config["completed_timesteps"], 2)


if __name__ == "__main__":
    unittest.main()
