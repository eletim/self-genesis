import random
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
            if name.startswith("value_head"):
                self.assertIsNone(parameter.grad)
                continue
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
            self.assertIsNone(message.value)
            self.assertIsNone(message.entropy)
            self.assertIsNotNone(action.value.grad_fn)
            self.assertIsNotNone(action.entropy.grad_fn)
            self.assertIs(message.state_after, action.state_before)
            self.assertIsNotNone(action.log_prob.grad_fn)
        terms = [d.value.square() - d.entropy for items in rollout.experiences
                 for item in items for d in item.decisions if d.log_prob is not None]
        torch.stack(terms).sum().backward()
        self.assertIsNone(collector.network.message_head.weight.grad)
        self.assertIsNotNone(collector.network.value_head.weight.grad)
        collector.collect(1)

    def test_value_and_entropy_gradients_survive_collection_and_reset(self):
        for statistic, heads in (("value", ("value_head",)),
                                 ("entropy", ("message_head", "action_head"))):
            torch.manual_seed(7)
            collector = self.collector(num_agents=2, initial_life=4, initial_points=0)
            first = collector.collect(1)
            state = first.experiences[0][0].decisions[0].state_after
            state.memory.retain_grad()
            state.affect.retain_grad()
            state.entities[0].value.retain_grad()
            later = collector.collect(1)
            collector.detach()
            collector.reset()
            decisions = [decision for items in later.experiences
                         for item in items for decision in item.decisions]
            terms = [getattr(decision, statistic) for decision in decisions]
            loss = (torch.stack(terms).sub(3).square().sum() if statistic == "value"
                    else -torch.stack(terms).sum())
            loss.backward()
            for name, parameter in collector.network.named_parameters():
                if name.startswith((*heads, "thought", "memory_update", "affect_update",
                                    "entity_update", "encounter_update")):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                else:
                    self.assertIsNone(parameter.grad, name)
            for tensor in (state.memory, state.affect, state.entities[0].value):
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(tensor.grad.abs().sum().item(), 0)
            for agent in collector.agents:
                self.assertEqual(agent.state.entities, ())
                self.assertEqual(agent.values, [])
                self.assertEqual(agent.entropies, [])

    def test_reset_preserves_rng_streams_and_reproduces_runs(self):
        def run():
            torch.manual_seed(17)
            collector = self.collector(num_agents=6, initial_life=8,
                                       initial_points=0, device="cpu")
            appearances = collector.world.state.appearance.clone()
            episodes = []
            for _ in range(3):
                rollout = collector.collect(8)
                episodes.append(tuple(
                    tuple((item.step, tuple((d.observation.first, d.choice)
                                           for d in item.decisions))
                          for item in experiences)
                    for experiences in rollout.experiences))
                random.random()
                python_rng = random.getstate()
                torch_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state_all()
                encounter_rng = collector.protocol._random.getstate()
                collector.reset()
                self.assertEqual(random.getstate(), python_rng)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()):
                    self.assertTrue(torch.equal(before, after))
                self.assertEqual(collector.protocol._random.getstate(), encounter_rng)
                self.assertTrue(torch.equal(appearances, collector.world.state.appearance))
                self.assertTrue((collector.world.state.life == 8).all())
                self.assertTrue((collector.world.state.points == 0).all())
                self.assertEqual(collector.elapsed_steps, 0)
                for agent in collector.agents:
                    self.assertEqual(agent.state.memory.count_nonzero(), 0)
                    self.assertEqual(agent.state.affect.count_nonzero(), 0)
                    self.assertFalse(agent.log_probs)
            self.assertNotEqual(episodes[0], episodes[1])
            self.assertNotEqual(episodes[1], episodes[2])
            return episodes

        self.assertEqual(run(), run())

    def test_validates_budget_and_dimensions(self):
        collector = self.collector()
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                collector.collect(budget)
        with self.assertRaises(ValueError):
            RolloutCollector(ExperimentConfig(appearance_dim=4), RecurrentPolicy(3))


if __name__ == "__main__":
    unittest.main()
