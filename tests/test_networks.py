import unittest
from types import SimpleNamespace

import numpy as np
import torch

from agents.state_encoder import KukaStateEncoder, extract_kuka_global_context
from agents.td3_agent import TD3


class NetworkTests(unittest.TestCase):
    def test_all_v2_actor_architectures_forward(self):
        state = np.zeros(31, dtype=np.float32)
        for architecture in ("mlp", "gnn", "transformer", "gnn_transformer"):
            with self.subTest(architecture=architecture):
                cfg = SimpleNamespace(
                    actor_arch=architecture,
                    node_dim=None if architecture == "mlp" else 6,
                    num_nodes=None if architecture == "mlp" else 7,
                    use_state_encoder=architecture != "mlp",
                    use_kuka_pe=True,
                    max_timesteps=20,
                    start_timesteps=2,
                    policy_delay=2,
                    device="cpu",
                )
                agent = TD3(31, 7, 1.5, cfg)
                action = agent.select_action(state, deterministic=True)
                self.assertEqual(action.shape, (7,))

    def test_six_dimensional_nodes_keep_goal_direction_and_previous_action(self):
        state = torch.zeros(1, 31)
        state[:, 18:25] = torch.arange(7)
        state[:, 25:28] = torch.tensor([[0.1, 0.2, 0.3]])
        state[:, 28:31] = torch.tensor([[0.4, 0.6, 0.8]])
        nodes = KukaStateEncoder(node_dim=6)(state).view(1, 7, 6)
        torch.testing.assert_close(nodes[0, :, 2], torch.arange(7, dtype=torch.float32))
        expected_direction = torch.tensor([0.3, 0.4, 0.5]).expand(7, 3)
        torch.testing.assert_close(nodes[0, :, 3:6], expected_direction)

    def test_global_context_has_phase_and_reference_error(self):
        state = torch.zeros(2, 31)
        state[:, 14] = 0.5
        state[:, 15:18] = 1.0
        state[:, 25:28] = 0.25
        context = extract_kuka_global_context(state)
        self.assertEqual(tuple(context.shape), (2, 4))
        torch.testing.assert_close(context[:, 0], torch.full((2,), 0.5))
        torch.testing.assert_close(context[:, 1:], torch.full((2, 3), 0.75))

    def test_action_scale_and_deterministic_dropout(self):
        cfg = SimpleNamespace(
            actor_arch="transformer",
            node_dim=6,
            num_nodes=7,
            use_state_encoder=True,
            use_kuka_pe=True,
            max_timesteps=100,
            start_timesteps=10,
            policy_delay=2,
            device="cpu",
        )
        agent = TD3(31, 7, 1.5, cfg)
        self.assertEqual(agent.actor.max_action, 1.5)
        state = np.linspace(-0.5, 0.5, 31, dtype=np.float32)
        first = agent.select_action(state, deterministic=True)
        second = agent.select_action(state, deterministic=True)
        np.testing.assert_allclose(first, second, atol=1e-7)
        self.assertLessEqual(float(np.max(np.abs(first))), 1.5)


if __name__ == "__main__":
    unittest.main()
