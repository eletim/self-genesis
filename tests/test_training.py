import math
import unittest
from dataclasses import replace

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import survival_policy_loss, train_episode


class TrainingTests(unittest.TestCase):
    def collector(self, message_length=3):
        torch.manual_seed(7)
        return RolloutCollector(
            ExperimentConfig(num_agents=2, appearance_dim=3,
                             initial_life=3, initial_points=0, device="cpu"),
            RecurrentPolicy(3, max_message_length=message_length))

    def test_exact_per_agent_credit_includes_lone_survival(self):
        collector = self.collector()
        collector.world.state.life[:] = torch.tensor([1, 3])
        rollout = collector.collect(3)
        logs = []
        experiences = []
        for items in rollout.experiences:
            decisions = []
            for decision in items[0].decisions:
                log_prob = torch.tensor(-0.5, requires_grad=True)
                logs.append(log_prob)
                decisions.append(replace(decision, log_prob=log_prob))
            experiences.append((replace(items[0], decisions=tuple(decisions)), *items[1:]))
        rollout = replace(rollout, experiences=tuple(experiences))
        loss = survival_policy_loss(rollout)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        self.assertEqual([x.grad.item() for x in logs], [-0.5, -0.5, -1.5, -1.5])

        # Changing agent 1's later survival cannot change agent 0's credit.
        longer = list(rollout.experiences)
        longer[1] = (*longer[1][:-1], replace(longer[1][-1], reward=2.0))
        for log in logs:
            log.grad = None
        survival_policy_loss(replace(rollout, experiences=tuple(longer))).backward()
        self.assertEqual([x.grad.item() for x in logs], [-0.5, -0.5, -2.0, -2.0])

    def test_finite_gradients_through_heads_and_recurrent_states(self):
        collector = self.collector()
        rollout = collector.collect(3)
        state = rollout.experiences[0][0].decisions[0].state_after
        state.memory.retain_grad()
        state.affect.retain_grad()
        loss = survival_policy_loss(rollout)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in collector.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for value in (state.memory, state.affect):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_repeated_optimizer_updates_and_disabled_channel(self):
        for length in (0, 3):
            collector = self.collector(length)
            optimizer = torch.optim.Adam(collector.network.parameters(), lr=0.01)
            for _ in range(2):
                before = {n: p.detach().clone() for n, p in collector.network.named_parameters()}
                result = train_episode(collector, optimizer)
                self.assertTrue(math.isfinite(result.loss))
                self.assertEqual(result.steps, 3)
                self.assertEqual(result.survival_returns, (3, 3))
                for name, parameter in collector.network.named_parameters():
                    if length == 0 and name.startswith("message_head"):
                        self.assertIsNone(parameter.grad)
                        self.assertTrue(torch.equal(parameter, before[name]))
                    else:
                        self.assertFalse(torch.equal(parameter, before[name]), name)
                for agent in collector.agents:
                    self.assertIsNone(agent.state.memory.grad_fn)
                    self.assertIsNone(agent.state.affect.grad_fn)
                    self.assertEqual(agent.log_probs, [])

    def test_rejects_incomplete_and_empty_episodes(self):
        collector = self.collector()
        with self.assertRaisesRegex(ValueError, "complete episode"):
            survival_policy_loss(collector.collect(1))
        with self.assertRaisesRegex(ValueError, "complete episode"):
            survival_policy_loss(collector.collect(2))
        with self.assertRaisesRegex(ValueError, "no sampled"):
            survival_policy_loss(collector.collect(1))

    def test_update_collects_to_extinction_with_giving(self):
        collector = self.collector()
        collector.config = replace(collector.config, initial_points=2)
        # Make gifts overwhelmingly likely while keeping finite log probabilities.
        with torch.no_grad():
            collector.network.action_head.bias[:] = torch.tensor([-10.0, 10.0])
        optimizer = torch.optim.Adam(collector.network.parameters(), lr=0.001)
        result = train_episode(collector, optimizer)
        self.assertTrue(math.isfinite(result.loss))
        self.assertEqual(result.survival_returns, (5, 5))
        self.assertEqual(result.steps, 5)
        self.assertFalse(collector.world.alive.any())


if __name__ == "__main__":
    unittest.main()
