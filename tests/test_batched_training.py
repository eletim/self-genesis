from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import survival_loss_components, train_batch


class BatchedTrainingTests(unittest.TestCase):
    def test_updates_match_sequential_complete_episode_reference(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for length in (0, 3):
                for horizon in (None, 3):
                    with self.subTest(device=device, length=length, horizon=horizon):
                        torch.manual_seed(8)
                        config = ExperimentConfig(
                            num_agents=3, appearance_dim=3, initial_life=4,
                            initial_points=0, survival_horizon=horizon, device=device,
                            value_loss_coefficient=0.7, action_entropy_coefficient=0.03,
                            message_entropy_coefficient=0.09)
                        network = RecurrentPolicy(3, max_message_length=length).to(device)
                        reference = RecurrentPolicy(3, max_message_length=length).to(device)
                        reference.load_state_dict(network.state_dict())
                        collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
                        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
                        expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
                        lives = ([1, 2, 4], [2, 2, 2])
                        # Repeat partners so entity writes affect later decisions.
                        draws = torch.full((2, 4 + 2 * length), 0.6, device=device)
                        draws[:, :2] = torch.tensor([[0.8, 0.8], [0.1, 0.1]], device=device)
                        reset = collector.reset

                        def reset_lives(row, *, seed):
                            reset(row, seed=seed)
                            collector.world.state.life[row] = torch.tensor(lives[row], device=device)

                        # Repeated updates exercise detach, reset, and optimizer state.
                        for _ in range(2):
                            before = {n: p.detach().clone() for n, p in network.named_parameters()}
                            recorded, early_states = [], []
                            step = collector.protocol.step

                            def record_step(state):
                                for name, parameter in network.named_parameters():
                                    self.assertTrue(torch.equal(parameter, before[name]), name)
                                result = step(state)
                                if not recorded:
                                    for tensor in (result.state.memory, result.state.affect,
                                                   result.state.entities.values):
                                        tensor.retain_grad()
                                        early_states.append(tensor)
                                recorded.append(result)
                                return result

                            with patch.object(collector.protocol, '_uniforms', return_value=draws), patch.object(
                                    collector, 'reset', side_effect=reset_lives), patch.object(
                                    collector.protocol, 'step', side_effect=record_step), patch.object(
                                    optimizer, 'step', wraps=optimizer.step) as update:
                                actual = train_batch(collector, optimizer, seeds=[11, 22])
                            self.assertEqual(update.call_count, 1)
                            self.assertTrue(collector.world.done.all())
                            self.assertEqual(actual.steps, (horizon or 4, 2))
                            self.assertEqual(actual.survival_returns,
                                             ((1., 2., float(horizon or 4)), (2., 2., 2.)))
                            self.assertEqual(actual.terminated, (horizon is None, True))
                            self.assertEqual(actual.horizon_completed, (horizon is not None, False))
                            for tensor in early_states:
                                self.assertTrue(torch.isfinite(tensor.grad).all())
                                self.assertGreater(tensor.grad.abs().sum(), 0)
                            for tensor in (collector.state.memory, collector.state.affect,
                                           collector.state.entities.values):
                                self.assertIsNone(tensor.grad_fn)

                            losses = []
                            for row, seed in enumerate((11, 22)):
                                sequential = RolloutCollector(replace(config, seed=seed), reference)
                                sequential.world.state.life[:] = torch.tensor(lives[row], device=device)
                                sequential_step = sequential.protocol.step
                                cursor = 0

                                def replay(policies):
                                    nonlocal cursor
                                    result = recorded[cursor]
                                    cursor += 1
                                    pair = result.pairs[row].tolist()
                                    samples = [d.choice[row, pair[i % 2]]
                                               for i, d in enumerate(result.decisions)
                                               if pair[0] >= 0 and (not d.communicating or length)]
                                    with patch.object(sequential.protocol._random, 'sample', return_value=pair), patch(
                                            'self_genesis.policy.Categorical.sample', side_effect=samples):
                                        return sequential_step(policies)

                                with patch.object(sequential.protocol, 'step', side_effect=replay):
                                    rollout = sequential.collect(10)
                                self.assertEqual(tuple(sum(e.reward for e in items)
                                                       for items in rollout.experiences),
                                                 actual.survival_returns[row])
                                losses.append(survival_loss_components(
                                    rollout, value_loss_coefficient=config.value_loss_coefficient,
                                    action_entropy_coefficient=config.action_entropy_coefficient,
                                    message_entropy_coefficient=config.message_entropy_coefficient))
                            for field in ('loss', 'actor_loss', 'value_loss', 'action_entropy', 'message_entropy'):
                                expected = sum(getattr(loss, field) for loss in losses) / 2
                                self.assertAlmostEqual(getattr(actual, field), expected.item(), places=5)
                            expected_optimizer.zero_grad(set_to_none=True)
                            (sum(loss.loss for loss in losses) / 2).backward()
                            expected_optimizer.step()
                            for (name, parameter), (_, expected) in zip(
                                    network.named_parameters(), reference.named_parameters()):
                                if expected.grad is None:
                                    self.assertIsNone(parameter.grad, name)
                                    self.assertTrue(torch.equal(parameter, before[name]), name)
                                else:
                                    torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-4)
                                    self.assertGreater(parameter.grad.abs().sum(), 0, name)
                                    self.assertFalse(torch.equal(parameter, before[name]), name)
                                torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-4)

    def test_renewable_horizon_and_validation_before_reset(self):
        config = ExperimentConfig(num_agents=2, appearance_dim=3, initial_life=4,
                                  point_generation_probability_min=1,
                                  point_generation_probability_max=1)
        network = RecurrentPolicy(3)
        collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
        optimizer = torch.optim.Adam(network.parameters())
        with patch.object(collector, 'reset', wraps=collector.reset) as reset:
            with self.assertRaisesRegex(ValueError, 'survival_horizon'):
                train_batch(collector, optimizer, seeds=[11, 22])
            reset.assert_not_called()
        config = replace(config, survival_horizon=3)
        collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
        for seeds in ([11], [11, -1], [11, True], [11, 2**63]):
            with patch.object(collector, 'reset', wraps=collector.reset) as reset:
                with self.assertRaisesRegex(ValueError, 'seed'):
                    train_batch(collector, optimizer, seeds=seeds)
                reset.assert_not_called()
        result = train_batch(collector, optimizer, seeds=[11, 22])
        self.assertEqual(result.steps, (3, 3))
        self.assertEqual(result.survival_returns, ((3., 3.), (3., 3.)))
        self.assertEqual(result.horizon_completed, (True, True))


if __name__ == '__main__':
    unittest.main()
