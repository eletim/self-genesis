"""BF16 must preserve stable statistics and agree with a matched FP32 rollout."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.training import run_training, survival_loss_components, train_batch


class MixedPrecisionTests(unittest.TestCase):
    def test_configuration_and_cpu_rejection_before_output_or_reset(self):
        self.assertEqual(ExperimentConfig().mixed_precision, 'fp32')
        for precision in ('fp16', '', None, True):
            with self.assertRaisesRegex(ValueError, 'mixed_precision'):
                ExperimentConfig(batched=True, mixed_precision=precision)
        with self.assertRaisesRegex(ValueError, 'batched'):
            ExperimentConfig(mixed_precision='bf16')
        config = ExperimentConfig(batched=True, mixed_precision='bf16', initial_life=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                run_training(config, output)
            self.assertFalse(output.exists())
        network = RecurrentPolicy(config.appearance_dim)
        collector = BatchedRolloutCollector(config, network, seeds=[1])
        with patch.object(collector, 'reset') as reset:
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                train_batch(collector, torch.optim.Adam(network.parameters()), seeds=[1])
            reset.assert_not_called()

    def test_cli_fp32_override_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.toml'
            config.write_text('batched = true\nmixed_precision = "bf16"\nnum_worlds = 2\n'
                              'initial_life = 1\ninitial_points = 0\nepisodes = 1\n')
            output = Path(directory) / 'run.jsonl'
            result = subprocess.run(
                [sys.executable, '-m', 'self_genesis', 'train', '--config', str(config),
                 '--mixed-precision', 'fp32', '--output', str(output)],
                check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(result.stdout)['config']['mixed_precision'], 'fp32')
            self.assertEqual(json.loads(output.read_text().splitlines()[0])['config']
                             ['mixed_precision'], 'fp32')

    def test_nonfinite_loss_and_gradients_skip_optimizer(self):
        for bad_gradient in (False, True):
            with self.subTest(bad_gradient=bad_gradient):
                config = ExperimentConfig(batched=True, initial_life=2, initial_points=0)
                network = RecurrentPolicy(config.appearance_dim)
                collector = BatchedRolloutCollector(config, network, seeds=[1])
                optimizer = torch.optim.Adam(network.parameters())
                before = [p.detach().clone() for p in network.parameters()]
                if bad_gradient:
                    network.action_head.weight.register_hook(lambda grad: grad * float('nan'))

                def loss(*args, **kwargs):
                    components = survival_loss_components(*args, **kwargs)
                    return components if bad_gradient else replace(
                        components, loss=components.loss * float('nan'))

                with patch('self_genesis.training.survival_loss_components', side_effect=loss), patch.object(
                        optimizer, 'step', wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(ValueError, 'Non-finite'):
                        train_batch(collector, optimizer, seeds=[1])
                    step.assert_not_called()
                for parameter, original in zip(network.parameters(), before):
                    torch.testing.assert_close(parameter, original)
                    self.assertIsNone(parameter.grad)
                self.assertIsNone(collector.state.memory.grad_fn)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         'CUDA BF16 hardware unavailable')
    def test_cuda_bf16_losses_gradients_and_repeated_updates_against_fp32(self):
        for method in ('actor_critic', 'reinforce'):
            for length, horizon, entity_dim in ((0, 4, 0), (3, 32, 16)):
                with self.subTest(method=method, length=length, horizon=horizon):
                    torch.manual_seed(18)
                    config = ExperimentConfig(
                        batched=True, device='cuda', num_agents=3, appearance_dim=3,
                        initial_life=40, initial_points=0, survival_horizon=horizon,
                        max_message_length=length, entity_memory_dim=entity_dim,
                        training_method=method)
                    reference = RecurrentPolicy(3, max_message_length=length,
                                                entity_memory_dim=entity_dim).cuda()
                    network = RecurrentPolicy(3, max_message_length=length,
                                              entity_memory_dim=entity_dim).cuda()
                    network.load_state_dict(reference.state_dict())
                    collectors = [BatchedRolloutCollector(c, n, seeds=[11, 22]) for c, n in (
                        (config, reference), (replace(config, mixed_precision='bf16'), network))]
                    optimizers = [torch.optim.Adam(n.parameters(), lr=1e-4)
                                  for n in (reference, network)]
                    # Fixed interior draws keep discrete decisions identical, isolating
                    # arithmetic error from policy sampling divergence.
                    draws = torch.full((2, 4 + 2 * length), 0.01, device='cuda')
                    head_dtypes, value_dtypes = [], []
                    handles = [network.action_head.register_forward_hook(
                        lambda module, inputs, output: head_dtypes.append(output.dtype)),
                        network.value_head.register_forward_hook(
                        lambda module, inputs, output: value_dtypes.append(output.dtype))]
                    try:
                        for update in range(2):
                            results, choices = [], []
                            for collector, optimizer in zip(collectors, optimizers):
                                collect = collector.collect

                                def checked_collect(budget):
                                    rollout = collect(budget)
                                    choices.append([d.choice.detach().clone()
                                                    for e in rollout.experiences for d in e.decisions])
                                    for experience in rollout.experiences:
                                        self.assertEqual(experience.reward.dtype, torch.float32)
                                        for decision in experience.decisions:
                                            for tensor in (decision.log_prob, decision.value, decision.entropy):
                                                if tensor is not None:
                                                    self.assertEqual(tensor.dtype, torch.float32)
                                                    self.assertTrue(torch.isfinite(tensor).all())
                                    return rollout

                                with patch.object(collector.protocol, '_uniforms', return_value=draws), patch.object(
                                        collector, 'collect', side_effect=checked_collect):
                                    results.append(train_batch(collector, optimizer, seeds=[11, 22]))
                                for tensor in (collector.state.memory, collector.state.affect,
                                               collector.state.entities.values):
                                    self.assertEqual(tensor.dtype, torch.float32)
                                    self.assertIsNone(tensor.grad_fn)
                            for actual, expected in zip(choices[1], choices[0]):
                                self.assertTrue(torch.equal(actual, expected))
                            self.assertEqual(results[0].survival_returns, results[1].survival_returns)
                            for field in ('loss', 'actor_loss', 'value_loss', 'action_entropy', 'message_entropy'):
                                expected, actual = getattr(results[0], field), getattr(results[1], field)
                                self.assertAlmostEqual(actual, expected, delta=0.02 * abs(expected) + 0.002)
                            for (name, expected), actual in zip(
                                    reference.named_parameters(), network.parameters()):
                                self.assertEqual(actual.dtype, torch.float32)
                                self.assertTrue(torch.isfinite(actual).all())
                                if expected.grad is None:
                                    self.assertIsNone(actual.grad)
                                else:
                                    self.assertEqual(actual.grad.dtype, torch.float32)
                                    self.assertTrue(torch.isfinite(actual.grad).all())
                                    error = (actual.grad - expected.grad).norm()
                                    self.assertLessEqual(error, 0.03 * expected.grad.norm() + 0.002,
                                                         msg=f'{name}, update {update + 1}')
                        self.assertEqual(set(head_dtypes), {torch.bfloat16})
                        self.assertEqual(set(value_dtypes), {torch.float32})
                    finally:
                        for handle in handles:
                            handle.remove()

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         'CUDA BF16 hardware unavailable')
    def test_bf16_thought_preserves_every_step_and_callback_gradient(self):
        for steps in (16, 32):
            with self.subTest(steps=steps):
                torch.manual_seed(18)
                network = RecurrentPolicy(3, think_steps=steps).cuda()
                state = network.initial_batch_state(1, 2, slots=2)
                inputs = torch.ones(2, network.affect_update.in_features - 2 * network.memory_dim,
                                    device='cuda')
                retrieved = torch.zeros(2, 16, device='cuda')
                # Yielding a diagnostic step must restore the caller's autocast.
                context = torch.cat((inputs, state.memory[0], state.affect[0], retrieved), dim=-1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for thought in network._think(context, state.memory[0]):
                        self.assertEqual(thought.dtype, torch.float32)
                        self.assertEqual(network.action_head(thought).dtype, torch.bfloat16)
                outputs = []

                def capture(module, args, output):
                    self.assertEqual(output.dtype, torch.float32)
                    output.retain_grad()
                    outputs.append(output)

                handle = network.thought.register_forward_hook(capture)
                try:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        _, memory, affect = network._recur(
                            inputs, state.memory[0], state.affect[0], retrieved)
                        memory.retain_grad()
                        affect.retain_grad()
                        _, final_memory, _ = network._recur(inputs, memory, affect, retrieved)
                        logits = network.action_head(final_memory)
                        self.assertEqual(logits.dtype, torch.bfloat16)
                    logits.float().sum().backward()
                    self.assertEqual(len(outputs), 2 * steps)
                    for tensor in (*outputs, memory, affect):
                        self.assertIsNotNone(tensor.grad)
                        self.assertTrue(torch.isfinite(tensor.grad).all())
                        self.assertGreater(tensor.grad.abs().sum().item(), 0)
                finally:
                    handle.remove()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_unsupported_cuda_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'torch.cuda.is_bf16_supported', return_value=False):
            output = Path(directory) / 'run.jsonl'
            with self.assertRaisesRegex(ValueError, 'not supported'):
                run_training(ExperimentConfig(batched=True, device='cuda',
                                              mixed_precision='bf16'), output)
            self.assertFalse(output.exists())
