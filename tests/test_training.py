import math
import unittest
from dataclasses import replace
from unittest.mock import patch

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
                decisions.append(replace(decision, log_prob=log_prob,
                                         value=torch.tensor(0.0), entropy=torch.tensor(0.0)))
            experiences.append((replace(items[0], decisions=tuple(decisions)), *items[1:]))
        rollout = replace(rollout, experiences=tuple(experiences))
        loss = survival_policy_loss(rollout)
        self.assertEqual(loss.item(), 7.0)
        loss.backward()
        self.assertEqual([x.grad.item() for x in logs], [-0.5, -0.5, -1.5, -1.5])

        # Changing agent 1's later survival cannot change agent 0's credit.
        longer = list(rollout.experiences)
        longer[1] = (*longer[1][:-1], replace(longer[1][-1], reward=2.0))
        for log in logs:
            log.grad = None
        survival_policy_loss(replace(rollout, experiences=tuple(longer))).backward()
        self.assertEqual([x.grad.item() for x in logs], [-0.5, -0.5, -2.0, -2.0])

    def test_actor_critic_arithmetic_and_gradient_isolation(self):
        collector = self.collector()
        collector.world.state.life[:] = torch.tensor([1, 3])
        rollout = collector.collect(3)
        logs = [torch.tensor(x, requires_grad=True) for x in (-0.2, -0.4, -0.6, -0.8)]
        values = [torch.tensor(x, requires_grad=True) for x in (0.25, 1.25, 1.5, 2.0)]
        entropies = [torch.tensor(x, requires_grad=True) for x in (0.1, 0.2, 0.3, 0.4)]
        experiences = []
        for agent, items in enumerate(rollout.experiences):
            decisions = tuple(
                replace(decision, log_prob=logs[2 * agent + channel],
                        value=values[2 * agent + channel],
                        entropy=entropies[2 * agent + channel])
                for channel, decision in enumerate(items[0].decisions))
            experiences.append((replace(items[0], decisions=decisions), *items[1:]))
        rollout = replace(rollout, experiences=tuple(experiences))
        loss = survival_policy_loss(
            rollout, value_loss_coefficient=0.7,
            action_entropy_coefficient=0.2, message_entropy_coefficient=0.3)
        # Returns are 1 and 3; each agent has one message and one action.
        # Actor = 1.75 / 2, value = 3.875 / 2, entropy bonus = 0.24 / 2.
        self.assertAlmostEqual(loss.item(), 2.11125, places=6)
        loss.backward()
        torch.testing.assert_close(
            torch.stack([x.grad for x in logs]), torch.tensor([-0.375, 0.125, -0.75, -0.5]))
        # Only regression reaches values: actor advantages must be detached.
        torch.testing.assert_close(
            torch.stack([x.grad for x in values]), torch.tensor([-0.525, 0.175, -1.05, -0.7]))
        torch.testing.assert_close(
            torch.stack([x.grad for x in entropies]), torch.tensor([-0.15, -0.1, -0.15, -0.1]))
        for coefficient, expected in (("action_entropy_coefficient", [-0.0, -0.2, -0.0, -0.2]),
                                      ("message_entropy_coefficient", [-0.2, -0.0, -0.2, -0.0])):
            for entropy in entropies:
                entropy.grad = None
            settings = dict(action_entropy_coefficient=0, message_entropy_coefficient=0)
            settings[coefficient] = 0.4
            survival_policy_loss(rollout, **settings).backward()
            torch.testing.assert_close(
                torch.stack([x.grad for x in entropies]), torch.tensor(expected))

    def test_unselected_step_rewards_contribute_to_returns(self):
        collector = self.collector()
        collector.config = replace(collector.config, num_agents=3)
        collector = RolloutCollector(collector.config, collector.network)
        rollout = collector.collect(3)
        logs, expected, experiences = [], [], []
        self.assertTrue(any(not item.decisions for items in rollout.experiences for item in items))
        for items in rollout.experiences:
            replaced = []
            for item in items:
                decisions = []
                for decision in item.decisions:
                    log = torch.tensor(-0.5, requires_grad=True)
                    logs.append(log)
                    expected.append(-(3 - item.step) / 3)
                    decisions.append(replace(decision, log_prob=log, value=torch.tensor(0.0),
                                             entropy=torch.tensor(0.0)))
                replaced.append(replace(item, decisions=tuple(decisions)))
            experiences.append(tuple(replaced))
        survival_policy_loss(replace(rollout, experiences=tuple(experiences))).backward()
        torch.testing.assert_close(torch.stack([x.grad for x in logs]), torch.tensor(expected))

    def test_training_uses_configured_loss_coefficients(self):
        collector = self.collector()
        collector.config = replace(
            collector.config, value_loss_coefficient=0.8,
            action_entropy_coefficient=0.03, message_entropy_coefficient=0)
        optimizer = torch.optim.Adam(collector.network.parameters(), lr=0.001)
        with patch("self_genesis.training.survival_policy_loss",
                   wraps=survival_policy_loss) as loss:
            train_episode(collector, optimizer)
        self.assertEqual(loss.call_args.kwargs, dict(
            value_loss_coefficient=0.8, action_entropy_coefficient=0.03,
            message_entropy_coefficient=0))

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

    def test_horizon_credit_includes_death_and_censored_lone_survivor(self):
        collector = self.collector()
        collector.config = replace(collector.config, survival_horizon=2)
        collector.world.state.life[:] = torch.tensor([1, 3])
        rollout = collector.collect(10)
        self.assertTrue(rollout.horizon_completed)
        self.assertFalse(rollout.terminated)
        self.assertFalse(rollout.truncated)
        logs = []
        experiences = []
        for items in rollout.experiences:
            decisions = []
            for decision in items[0].decisions:
                log = torch.tensor(-0.5, requires_grad=True)
                logs.append(log)
                decisions.append(replace(decision, log_prob=log,
                                         value=torch.tensor(0.0), entropy=torch.tensor(0.0)))
            experiences.append((replace(items[0], decisions=tuple(decisions)), *items[1:]))
        survival_policy_loss(replace(rollout, experiences=tuple(experiences))).backward()
        self.assertEqual([log.grad.item() for log in logs], [-0.5, -0.5, -1.0, -1.0])
        self.assertTrue(rollout.experiences[0][-1].terminated)
        self.assertFalse(rollout.experiences[1][-1].terminated)

    def test_renewable_horizon_updates_and_recurrent_gradients(self):
        collector = self.collector()
        collector.config = replace(
            collector.config, initial_life=4, initial_points=1,
            point_generation_probability_min=1, point_generation_probability_max=1,
            survival_horizon=12)
        with torch.no_grad():
            collector.network.action_head.bias[:] = torch.tensor([-4.0, 4.0])
        collector.reset()
        rollout = collector.collect(100)
        self.assertTrue(rollout.horizon_completed)
        state = rollout.experiences[0][0].decisions[0].state_after
        state.memory.retain_grad()
        state.affect.retain_grad()
        survival_policy_loss(rollout).backward()
        for value in (state.memory, state.affect):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        optimizer = torch.optim.Adam(collector.network.parameters(), lr=0.001)
        for _ in range(3):
            before = collector.network.action_head.weight.detach().clone()
            result = train_episode(collector, optimizer)
            self.assertEqual(result.steps, 12)
            self.assertEqual(result.survival_returns, (12, 12))
            self.assertTrue(result.horizon_completed)
            self.assertFalse(result.terminated)
            self.assertTrue(math.isfinite(result.loss))
            self.assertFalse(torch.equal(before, collector.network.action_head.weight))
            for name, parameter in collector.network.named_parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            for agent in collector.agents:
                self.assertIsNone(agent.state.memory.grad_fn)
                self.assertIsNone(agent.state.affect.grad_fn)
                self.assertEqual(agent.log_probs, [])

    def test_horizon_segments_and_extinction_precedence(self):
        collector = self.collector()
        collector.config = replace(collector.config, survival_horizon=2)
        first = collector.collect(1)
        self.assertTrue(first.truncated)
        self.assertFalse(first.horizon_completed)
        with self.assertRaisesRegex(ValueError, "complete episode"):
            survival_policy_loss(first)
        final = collector.collect(20)
        self.assertEqual(final.steps, 1)
        self.assertTrue(final.horizon_completed)
        with self.assertRaisesRegex(ValueError, "complete episode"):
            survival_policy_loss(final)
        self.assertEqual(collector.collect(1).steps, 0)
        for horizon in (3, 4):
            collector.config = replace(collector.config, survival_horizon=horizon)
            collector.reset()
            rollout = collector.collect(10)
            self.assertTrue(rollout.terminated)
            self.assertFalse(rollout.horizon_completed)
            self.assertFalse(rollout.truncated)
            survival_policy_loss(rollout)

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
