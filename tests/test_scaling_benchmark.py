"""Ensure throughput counts encounters rather than agent decisions or steps."""
import importlib.util
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import json
import subprocess
import tempfile
from unittest.mock import Mock, patch

import torch

from self_genesis.world import Action

spec = importlib.util.spec_from_file_location(
    'benchmark_rtx5090', Path(__file__).resolve().parents[1] / 'examples/benchmark_rtx5090.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class EncounterCountTests(unittest.TestCase):
    def test_no_encounters(self):
        self.assertEqual(benchmark.encounter_count(NS(experiences=[NS(decisions=[])]), True), 0)

    def test_scalar_messages_and_lone_survivor_do_not_count(self):
        rollout = NS(experiences=[
            [NS(decisions=[NS(choice=(1, 2)), NS(choice=Action.GIVE)]), NS(decisions=[])],
            [NS(decisions=[NS(choice=()), NS(choice=Action.NOTHING)])]])
        self.assertEqual(benchmark.encounter_count(rollout, False), 1)

    def test_batch_ignores_inactive_worlds_and_message_phases(self):
        decisions = [NS(communicating=communicating, active=torch.tensor(mask))
                     for communicating, mask in [
                         (True, [[True, False], [False, True], [False, False]]),
                         (False, [[True, False], [False, True], [False, False]]),
                         (False, [[False, True], [True, False], [False, False]])]]
        rollout = NS(experiences=[NS(decisions=decisions), NS(decisions=[])])
        self.assertEqual(benchmark.encounter_count(rollout, True), 2)


class SweepTests(unittest.TestCase):
    def test_density_changes_only_worlds_and_pairs(self):
        args = NS(sweep='density', capacity='medium', precision='fp32',
                  world_counts=[64, 128, 256], seed=42, horizon=16, warmup=1, updates=20)
        cases = benchmark.sweep_cases(args)
        self.assertEqual(len(cases), 12)
        self.assertEqual({case[4] for case in cases}, {1, 4, 8, 16})
        configs = []
        for capacity, precision, worlds, agents, pairs in cases:
            config = benchmark.benchmark_config(NS(**vars(args), worlds=worlds,
                                                  num_agents=agents, encounter_count=pairs))
            self.assertEqual(config.num_agents, 32)
            self.assertTrue(config.batched)
            self.assertEqual(config.encounter_pairs(32), pairs)
            values = benchmark.asdict(config)
            del values['num_worlds'], values['encounter_count']
            configs.append(values)
        self.assertTrue(all(config == configs[0] for config in configs))
        self.assertEqual(configs[0]['memory_dim'], 128)
        self.assertEqual(configs[0]['learning_rate'], 0.001)
        self.assertEqual(configs[0]['point_generation_probability_min'], 0.1)

    def test_recurrent_depth_preserves_full_episode_conditions(self):
        args = NS(sweep='recurrent', capacity='small', precision='fp32',
                  world_counts=[64, 128], num_agents=32, encounter_count=16,
                  seed=42, horizon=16, warmup=1, updates=20)
        self.assertEqual(benchmark.sweep_cases(args),
                         [('small', 'fp32', n, 32, 16) for n in (64, 128)])
        configs = []
        for depth in (16, 32, 64):
            config = benchmark.benchmark_config(NS(**vars(args), worlds=64, think_steps=depth))
            self.assertEqual(config.think_steps, depth)
            self.assertEqual(config.thought_mode, 'recurrent')
            self.assertEqual(config.survival_horizon, 16)
            values = benchmark.asdict(config)
            del values['think_steps']
            configs.append(values)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])

    def test_recurrent_commands_and_failures_never_retry_shorter(self):
        for optional, depths in (([], [16, 32]), (['--include-64'], [16, 32, 64])):
            with self.subTest(depths=depths), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'results'
                argv = ['benchmark', '--output', str(output), '--sweep', 'recurrent',
                        '--world-counts', '64', '--horizon', '16', *optional]
                failures = [subprocess.CompletedProcess([], 1, '', 'CUDA out of memory'),
                            subprocess.TimeoutExpired('worker', 300)]
                if optional:
                    failures.append(subprocess.CompletedProcess([], 1, '', 'Non-finite gradients'))
                with patch('sys.argv', argv), patch.object(benchmark.subprocess, 'run',
                                                          side_effect=failures) as run:
                    benchmark.main()
                rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
                self.assertEqual(run.call_count, len(depths))
                self.assertEqual([row['status'] for row in rows],
                                 ['failed', 'timeout'] + (['failed'] if optional else []))
                for row, depth in zip(rows, depths):
                    command = row['command']
                    self.assertEqual(command[command.index('--think-steps') + 1], str(depth))
                    self.assertEqual(command[command.index('--horizon') + 1], '16')
                    self.assertIn(f'think{depth}', row['case'])
                    self.assertTrue(row['error'])

    def test_original_scaling_matrix_is_preserved(self):
        cases = benchmark.sweep_cases(NS(sweep='scaling'))
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(set(cases)), 33)
        self.assertTrue(all(agents == 4 and pairs == 1
                            for _, _, _, agents, pairs in cases))
        self.assertEqual(sum(worlds == 0 for _, _, worlds, _, _ in cases), 3)


class WorkerTests(unittest.TestCase):
    def test_worker_uses_depth_and_retains_evidence_on_failure(self):
        args = NS(capacity='small', precision='fp32', seed=42, worlds=1,
                  num_agents=2, encounter_count=1, horizon=1, warmup=0,
                  updates=2, think_steps=32)
        config = replace(benchmark.benchmark_config(args), device='cpu')
        network = benchmark.RecurrentPolicy(config.appearance_dim,
                                           memory_dim=config.memory_dim,
                                           affect_dim=config.affect_dim,
                                           entity_memory_dim=config.entity_memory_dim,
                                           think_steps=32)
        telemetry = Mock(samples=[], errors=[])
        telemetry.thread.ident = None
        train_batch = benchmark.train_batch
        calls = []

        def train(collector, optimizer, *, seeds):
            calls.append(collector.network.think_steps)
            if len(calls) == 2:
                raise ValueError('Non-finite batched training gradients; optimizer update skipped')
            return train_batch(collector, optimizer, seeds=seeds)

        measurement = benchmark.UpdateMeasurement
        with ExitStack() as stack:
            for target, kwargs in [
                ('benchmark_config', dict(return_value=config)),
                ('RecurrentPolicy', dict(return_value=network)),
                ('Telemetry', dict(return_value=telemetry)),
                ('train_batch', dict(side_effect=train)),
                ('UpdateMeasurement', dict(side_effect=lambda device: measurement(torch.device('cpu')))),
            ]:
                stack.enter_context(patch.object(benchmark, target, **kwargs))
            stack.enter_context(patch.object(network, 'cuda', return_value=network))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='RTX 5090'))
            stack.enter_context(patch.object(benchmark.subprocess, 'check_output', return_value='test metadata'))
            result = benchmark.worker(args)
            benchmark.RecurrentPolicy.assert_called_once_with(
                config.appearance_dim, memory_dim=config.memory_dim,
                affect_dim=config.affect_dim, entity_memory_dim=config.entity_memory_dim,
                thought_mode='recurrent', think_steps=32)
        self.assertEqual(calls, [32, 32])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_update'], 1)
        self.assertIn('Non-finite', result['error'])
        self.assertEqual(len(result['records']), 1)
        self.assertTrue(result['records'][0]['gradients_finite'])
        self.assertGreater(result['records'][0]['gradient_tensor_count'], 0)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
