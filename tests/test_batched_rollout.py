import unittest

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import AgentExperience, PolicyDecision, Rollout
from self_genesis.training import survival_loss_components, survival_policy_loss
from self_genesis.world import Action


class BatchedRolloutTests(unittest.TestCase):
    def collector(self, device='cpu', length=3, **kwargs):
        torch.manual_seed(7)
        config = ExperimentConfig(num_agents=3, appearance_dim=3, initial_life=4,
                                  initial_points=0, device=device, **kwargs)
        return BatchedRolloutCollector(
            config, RecurrentPolicy(3, max_message_length=length).to(device), seeds=[11, 22])

    def test_complete_objective_and_gradient_parity_with_scalar_loss(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for length in (0, 3):
                for method in ('reinforce', 'actor_critic'):
                    with self.subTest(device=device, length=length, method=method):
                        collector = self.collector(device, length)
                        collector.world.state.life[:] = torch.tensor(
                            [[1, 2, 4], [3, 3, 3]], device=device)
                        rollout = collector.collect(20)
                        self.assertEqual(rollout.end_steps.tolist(), [4, 3])
                        rewards = torch.stack([e.reward for e in rollout.experiences])
                        self.assertEqual(rewards.sum(0).tolist(), [[1, 2, 4], [3, 3, 3]])
                        self.assertEqual(rollout.experiences[-1].decisions, ())
                        reference = []
                        for row in range(2):
                            agents = [[] for _ in range(3)]
                            for step, experience in enumerate(rollout.experiences):
                                for agent in range(3):
                                    if not experience.reward[row, agent]:
                                        continue
                                    decisions = []
                                    for d in experience.decisions:
                                        if not d.active[row, agent]:
                                            continue
                                        choice = (tuple(d.choice[row, agent].tolist()) if d.communicating
                                                  else (Action.GIVE if d.choice[row, agent] else Action.NOTHING))
                                        stats = [None if t is None else t[row, agent]
                                                 for t in (d.log_prob, d.value, d.entropy)]
                                        decisions.append(PolicyDecision(None, choice, stats[0], None,
                                                                        None, stats[1], stats[2]))
                                    agents[agent].append(AgentExperience(
                                        step, float(experience.reward[row, agent]),
                                        bool(experience.died[row, agent]), tuple(decisions)))
                            reference.append(survival_loss_components(
                                Rollout(tuple(tuple(a) for a in agents), int(rollout.end_steps[row]),
                                        True, False), training_method=method))
                        actual = survival_loss_components(rollout, training_method=method)
                        for field in ('loss', 'actor_loss', 'value_loss', 'action_entropy', 'message_entropy'):
                            expected = sum(getattr(r, field) for r in reference) / 2
                            torch.testing.assert_close(getattr(actual, field), expected)
                        parameters = tuple(collector.network.parameters())
                        gradients = torch.autograd.grad(actual.loss, parameters, retain_graph=True, allow_unused=True)
                        expected = torch.autograd.grad(sum(r.loss for r in reference) / 2,
                                                       parameters, allow_unused=True)
                        for g, e in zip(gradients, expected):
                            if e is None:
                                self.assertIsNone(g)
                            else:
                                torch.testing.assert_close(g, e, atol=2e-6, rtol=2e-5)

    def test_budget_continuation_retains_full_recurrent_graph(self):
        collector = self.collector(survival_horizon=3)
        first = collector.collect(1)
        state = collector.state
        for tensor in (state.memory, state.affect, state.entities.values):
            tensor.retain_grad()
        with self.assertRaises(ValueError):
            survival_policy_loss(first)
        with self.assertRaises(ValueError):
            collector.detach()
        tail = collector.collect(10)
        with self.assertRaises(ValueError):
            survival_policy_loss(tail)
        complete = first.extend(tail)
        self.assertEqual(complete.terminated.tolist(), [False, False])
        self.assertEqual(complete.horizon_completed.tolist(), [True, True])
        collector.detach()
        collector.reset(0, seed=33)
        survival_policy_loss(complete).backward()
        for tensor in (state.memory, state.affect, state.entities.values):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum(), 0)
        self.assertGreater(collector.network.encounter_update.weight_ih.grad.abs().sum(), 0)
        self.assertEqual(collector.state.memory[0].count_nonzero(), 0)
        self.assertIsNone(collector.state.memory.grad_fn)

    def test_split_collection_matches_uninterrupted_worlds(self):
        collector = self.collector(survival_horizon=3)
        reference = self.collector(survival_horizon=3)
        complete = reference.collect(10)
        split = collector.collect(1).extend(collector.collect(1)).extend(collector.collect(10))
        for actual, expected in zip(split.experiences, complete.experiences):
            torch.testing.assert_close(actual.reward, expected.reward)
            torch.testing.assert_close(actual.died, expected.died)
            for a, e in zip(actual.decisions, expected.decisions):
                torch.testing.assert_close(a.choice, e.choice)
                torch.testing.assert_close(a.log_prob, e.log_prob)
        survival_policy_loss(split).backward()
        survival_policy_loss(complete).backward()
        for actual, expected in zip(collector.network.parameters(), reference.network.parameters()):
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad)

    def test_independent_reset_freezing_and_invalid_joins(self):
        collector = self.collector()
        collector.world.state.life[0] = 1
        first = collector.collect(1)
        self.assertEqual(first.terminated.tolist(), [True, False])
        previous = collector.state.memory
        previous.retain_grad()
        memory = collector.state.memory[1].clone()
        collector.reset(0, seed=33)
        torch.testing.assert_close(collector.state.memory[1], memory)
        next_segment = collector.collect(1)
        later_values = [d.value[1].sum() for d in next_segment.experiences[0].decisions]
        sum(later_values).backward()
        self.assertEqual(previous.grad[0].count_nonzero(), 0)
        self.assertGreater(previous.grad[1].abs().sum(), 0)
        with self.assertRaises(ValueError):
            first.extend(next_segment)
        with self.assertRaises(ValueError):
            next_segment.extend(first)
        with self.assertRaises(ValueError):
            first.extend(self.collector().collect(1))
        self.assertEqual(first.terminated.tolist(), [True, False])
        final = collector.collect(20)
        self.assertTrue(final.terminated.all())
        self.assertEqual(collector.collect(1).experiences, ())
        for budget in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                collector.collect(budget)


if __name__ == '__main__':
    unittest.main()
