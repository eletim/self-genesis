from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.experiment import ExperimentState
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import survival_loss_components, train_batch


class BatchedTrainingTests(unittest.TestCase):
    def test_renewable_training_matches_independently_scripted_reference(self):
        # FP64 policy forwards isolate routing; training still reduces in FP32.
        # Neither scalar pairs nor samples are read back from batched results.
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for length in (0, 3):
                for method in ('reinforce', 'actor_critic'):
                    with self.subTest(device=device, length=length, method=method):
                        torch.manual_seed(19)
                        config = ExperimentConfig(
                            num_agents=3, appearance_dim=3, initial_points=1,
                            survival_horizon=5, device=device, training_method=method,
                            point_generation_probability_min=0.5,
                            point_generation_probability_max=0.5,
                            value_loss_coefficient=0.7, action_entropy_coefficient=0.03,
                            message_entropy_coefficient=0.09)
                        network = RecurrentPolicy(3, max_message_length=length).to(device).double()
                        reference = RecurrentPolicy(3, max_message_length=length).to(device).double()
                        reference.load_state_dict(network.state_dict())
                        collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
                        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
                        expected_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
                        actions = ((1, 1), (0, 0), (1, 0), (0, 1), (1, 1))
                        lives = ((1, 1, 1), (4, 4, 1))
                        points = ((1, 1, 1), (0, 1, 1))
                        draws, generation = [], []
                        for step, choices in enumerate(actions):
                            uniforms = torch.zeros(2, 4 + 2 * length, device=device,
                                                   dtype=torch.float64)
                            uniforms[1, 0] = 0.5  # [1, 0], before and after agent 2 dies
                            # Uniform endpoints select first/last token and NOTHING/GIVE.
                            uniforms[:, 2 + length:2 + 2 * length] = 1 - 1e-12
                            uniforms[1, -2:] = torch.tensor(choices, device=device, dtype=torch.float64) * (1 - 1e-12)
                            draws.append(uniforms)
                            generation.append(torch.tensor(
                                [[0.1, 0.9, 0.1], [0.1, 0.9, 0.1] if step % 2 == 0
                                 else [0.9, 0.1, 0.9]], device=device))
                        reset, step_batch = collector.reset, collector.protocol.step
                        recorded, snapshots = [], []

                        def reset_resources(row, *, seed):
                            reset(row, seed=seed)
                            collector.world.state.life[row] = torch.tensor(lives[row], device=device)
                            collector.world.state.points[row] = torch.tensor(points[row], device=device)

                        def record(state):
                            result = step_batch(state)
                            recorded.append(result)
                            snapshots.append({f.name: getattr(collector.world.state, f.name).clone()
                                              for f in fields(ExperimentState)})
                            return result

                        with patch.object(collector, 'reset', side_effect=reset_resources), patch.object(
                                collector.protocol, '_uniforms', side_effect=draws), patch.object(
                                collector.world._generation_rng, 'uniform', side_effect=generation), patch.object(
                                collector.protocol, 'step', side_effect=record):
                            actual = train_batch(collector, optimizer, seeds=[11, 22])
                        self.assertEqual(actual.steps, (1, 5))
                        self.assertEqual(actual.terminated, (True, False))
                        self.assertEqual(actual.horizon_completed, (False, True))
                        self.assertEqual(actual.survival_returns, ((1., 1., 1.), (5., 5., 1.)))
                        self.assertEqual(recorded[0].world.successful_transfers[1].tolist(),
                                         [False, True, False])
                        self.assertEqual(recorded[0].world.generated_points[1].tolist(), [1, 0, 0])
                        self.assertFalse(recorded[1].world.successful_transfers.any())
                        for step in range(1, 5):
                            self.assertEqual(recorded[step].pairs[0].tolist(), [-1, -1])
                            for name in ('reward', 'died', 'generated_points', 'successful_transfers'):
                                self.assertFalse(getattr(recorded[step].world, name)[0].any(), name)
                            for name in snapshots[0]:
                                self.assertTrue(torch.equal(snapshots[step][name][0], snapshots[0][name][0]), name)
                        losses = []
                        for row, seed in enumerate((11, 22)):
                            scalar = RolloutCollector(replace(config, seed=seed), reference)
                            scalar.world.state.life[:] = torch.tensor(lives[row], device=device)
                            scalar.world.state.points[:] = torch.tensor(points[row], device=device)
                            scalar_step = scalar.protocol.step
                            cursor = 0

                            def scripted(policies):
                                nonlocal cursor
                                step = cursor
                                cursor += 1
                                pair = [0, 1] if row == 0 else [1, 0]
                                choices = (0, 0) if row == 0 else actions[step]
                                samples = ([torch.zeros(length, dtype=torch.long, device=device),
                                            torch.full((length,), network.vocabulary_size - 1,
                                                       dtype=torch.long, device=device)] if length else [])
                                samples += [torch.tensor(c, device=device) for c in choices]
                                with patch.object(scalar.protocol._random, 'sample', return_value=pair), patch(
                                        'self_genesis.policy.Categorical.sample', side_effect=samples), patch(
                                        'self_genesis.world.torch.rand', side_effect=lambda *a, **kw:
                                        generation[step][row, scalar.world.alive].cpu()):
                                    expected = scalar_step(policies)
                                result = recorded[step]
                                self.assertEqual(result.pairs[row].tolist(), pair)
                                for phase, token in enumerate((0, network.vocabulary_size - 1)):
                                    self.assertEqual(result.decisions[phase].choice[row, pair[phase]].tolist(),
                                                     [token] * length)
                                for phase, choice in enumerate(choices, start=2):
                                    self.assertEqual(int(result.decisions[phase].choice[row, pair[phase % 2]]),
                                                     choice)
                                for name in ('reward', 'died', 'generated_points'):
                                    self.assertTrue(torch.equal(getattr(result.world, name)[row],
                                                                getattr(expected, name)), name)
                                transfers = tuple((donor, 1 - donor) for donor in
                                                  result.world.successful_transfers[row].nonzero().flatten().tolist())
                                self.assertEqual(transfers, expected.successful_transfers)
                                for name, snapshot in snapshots[step].items():
                                    self.assertTrue(torch.equal(snapshot[row], getattr(scalar.world.state, name)), name)
                                return expected

                            with patch.object(scalar.protocol, 'step', side_effect=scripted):
                                rollout = scalar.collect(10)
                            self.assertEqual(tuple(sum(e.reward for e in agent) for agent in rollout.experiences),
                                             actual.survival_returns[row])
                            losses.append(survival_loss_components(
                                rollout, training_method=method,
                                value_loss_coefficient=config.value_loss_coefficient,
                                action_entropy_coefficient=config.action_entropy_coefficient,
                                message_entropy_coefficient=config.message_entropy_coefficient))
                        for name in ('loss', 'actor_loss', 'value_loss', 'action_entropy', 'message_entropy'):
                            expected = sum(getattr(loss, name) for loss in losses) / 2
                            torch.testing.assert_close(torch.tensor(getattr(actual, name), dtype=torch.float64),
                                                       expected.detach().cpu(), atol=2e-6, rtol=2e-5)
                        expected_optimizer.zero_grad(set_to_none=True)
                        (sum(loss.loss for loss in losses) / 2).backward()
                        expected_optimizer.step()
                        for (name, parameter), (_, expected) in zip(
                                network.named_parameters(), reference.named_parameters()):
                            if expected.grad is None:
                                self.assertIsNone(parameter.grad, name)
                            else:
                                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                                torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-4)
                            torch.testing.assert_close(parameter, expected, atol=2e-6, rtol=2e-4)

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
