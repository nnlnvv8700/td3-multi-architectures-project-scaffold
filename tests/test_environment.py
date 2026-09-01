import unittest

import numpy as np
import pybullet as p

from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from training.observation import flatten_obs


class EnvironmentTests(unittest.TestCase):
    def test_clients_are_isolated(self):
        env1 = KukaIiwa7TrackEnv(render_mode="rgb_array")
        env2 = KukaIiwa7TrackEnv(render_mode="rgb_array")
        try:
            env1.reset(seed=1)
            env2.reset(seed=2)
            self.assertNotEqual(env1._p_client, env2._p_client)
            self.assertEqual(p.getNumBodies(physicsClientId=env1._p_client), 3)
            self.assertEqual(p.getNumBodies(physicsClientId=env2._p_client), 3)
            env1.step(np.zeros(7, dtype=np.float32))
            env2.step(np.zeros(7, dtype=np.float32))
        finally:
            env1.close()
            env2.close()

    def test_v2_observation_contains_trajectory_context(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array", max_steps=4, observation_version=2)
        try:
            obs, info = env.reset(seed=7)
            self.assertEqual(flatten_obs(obs).shape, (31,))
            self.assertEqual(obs["observation"].shape, (25,))
            self.assertAlmostEqual(float(info["phase"]), 0.0)
            action = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
            obs2, _, _, _, info2 = env.step(action)
            self.assertAlmostEqual(float(info2["phase"]), 0.25)
            np.testing.assert_allclose(obs2["observation"][18:25], action)
            np.testing.assert_allclose(info2["reference_point"], info2["ref_traj"][1])
        finally:
            env.close()

    def test_reset_seed_repeats_goal(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array")
        try:
            first, _ = env.reset(seed=123)
            second, _ = env.reset(seed=123)
            np.testing.assert_allclose(first["desired_goal"], second["desired_goal"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
