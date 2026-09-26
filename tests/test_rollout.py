import unittest

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.world import Action


class RolloutTests(unittest.TestCase):
    def collector(self, **overrides):
        config = ExperimentConfig(appearance_dim=3, **overrides)
        return RolloutCollector(config, RecurrentPolicy(3))

    def test_limits_continuation_and_separate_state(self):
        collector = self.collector(num_agents=4, initial_life=3, initial_points=0)
        initial = [agent.state for agent in collector.agents]
        first = collector.collect(1)
        self.assertTrue(first.truncated)
        self.assertFalse(first.terminated)
        self.assertEqual(first.steps, 1)
        selected = []
        for index, experiences in enumerate(first.experiences):
            self.assertEqual(len(experiences), 1)
            self.assertEqual(experiences[0].reward, 1)
            self.assertFalse(experiences[0].terminated)
            if experiences[0].decisions:
                selected.append(index)
                self.assertIs(experiences[0].decisions[0].state_before, initial[index])
            else:
                self.assertIs(collector.agents[index].state, initial[index])
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({a.state.memory.data_ptr() for a in collector.agents}), 4)
        states = [agent.state for agent in collector.agents]
        second = collector.collect(1)
        for index, experiences in enumerate(second.experiences):
            self.assertEqual(experiences[0].step, 1)
            if experiences[0].decisions:
                self.assertIs(experiences[0].decisions[0].state_before, states[index])
        final = collector.collect(1)
        self.assertTrue(final.terminated)
        self.assertFalse(final.truncated)  # Death on the exact time limit wins.
        self.assertTrue(all(items[-1].terminated for items in final.experiences))
        self.assertEqual(collector.collect(10).steps, 0)

    def test_death_and_lone_survivor(self):
        collector = self.collector(num_agents=2, initial_points=0)
        collector.world.state.life[:] = torch.tensor([1, 3])
        rollout = collector.collect(10)
        self.assertEqual(rollout.steps, 3)
        self.assertEqual([len(items) for items in rollout.experiences], [1, 3])
        self.assertEqual([sum(x.reward for x in items) for items in rollout.experiences], [1, 3])
        self.assertTrue(rollout.experiences[0][0].terminated)
        self.assertTrue(rollout.experiences[1][-1].terminated)
        self.assertEqual([len(x.decisions) for x in rollout.experiences[1]], [2, 0, 0])
        for agent in collector.agents:
            self.assertEqual(agent.state.memory.count_nonzero(), 0)
            self.assertEqual(agent.state.affect.count_nonzero(), 0)

    def test_decisions_train_from_later_survival_after_reset(self):
        collector = self.collector(num_agents=2, initial_life=3, initial_points=2)
        rollout = collector.collect(20)
        self.assertTrue(rollout.terminated)
        collector.reset()
        terms = []
        for experiences in rollout.experiences:
            reward_to_go = 0
            for experience in reversed(experiences):
                reward_to_go += experience.reward
                for decision in experience.decisions:
                    self.assertIsNotNone(decision.log_prob.grad_fn)
                    terms.append(-decision.log_prob * reward_to_go)
            message, action = experiences[0].decisions
            self.assertIsInstance(message.choice, tuple)
            self.assertIsInstance(action.choice, Action)
            self.assertIs(message.state_after, action.state_before)
            self.assertEqual(len(action.observation.received_message), 3)
        torch.stack(terms).sum().backward()
        for name, parameter in collector.network.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertEqual(collector.elapsed_steps, 0)
        self.assertTrue((collector.world.state.life == 3).all())
        self.assertTrue((collector.world.state.points == 2).all())
        self.assertTrue(all(not a.log_probs for a in collector.agents))

    def test_detach_and_disabled_channel(self):
        collector = RolloutCollector(
            ExperimentConfig(num_agents=2, appearance_dim=3),
            RecurrentPolicy(3, vocabulary_size=2, max_message_length=0))
        rollout = collector.collect(1)
        saved = [a.state.memory.clone() for a in collector.agents]
        collector.detach()
        for agent, memory in zip(collector.agents, saved):
            self.assertTrue(torch.equal(agent.state.memory, memory))
            self.assertIsNone(agent.state.memory.grad_fn)
            self.assertIsNone(agent.state.affect.grad_fn)
        for items in rollout.experiences:
            message, action = items[0].decisions
            self.assertEqual(message.choice, ())
            self.assertIsNone(message.log_prob)
            self.assertIsNotNone(action.log_prob.grad_fn)
        collector.collect(1)

    def test_reset_reproduces_episode_and_validates_budget(self):
        collector = self.collector(num_agents=2, initial_life=2, initial_points=0)
        first = collector.collect(10)
        appearances = collector.world.state.appearance.clone()
        collector.reset()
        self.assertTrue(torch.equal(appearances, collector.world.state.appearance))
        second = collector.collect(10)
        for left, right in zip(first.experiences, second.experiences):
            self.assertEqual([[d.choice for d in x.decisions] for x in left],
                             [[d.choice for d in x.decisions] for x in right])
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                collector.collect(budget)
        with self.assertRaises(ValueError):
            RolloutCollector(ExperimentConfig(appearance_dim=4), RecurrentPolicy(3))


if __name__ == "__main__":
    unittest.main()
