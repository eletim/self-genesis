from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.config import ExperimentConfig, load_config
from self_genesis.training import run_training


class BatchedTrainingCommandTests(unittest.TestCase):
    def test_settings_and_presets(self):
        for name in ('batched', 'deterministic'):
            for value in (1, 'true', None):
                with self.assertRaisesRegex(ValueError, name):
                    ExperimentConfig(**{name: value})
        for value in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'num_worlds'):
                ExperimentConfig(num_worlds=value)
        with self.assertRaisesRegex(ValueError, 'capacity_preset'):
            load_config(capacity_preset='huge')
        config = load_config(capacity_preset='medium', memory_dim=11)
        self.assertEqual((config.memory_dim, config.affect_dim, config.entity_memory_dim),
                         (11, 32, 64))

    def test_cli_reproducible_updates_config_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / 'run.toml'
            config.write_text('batched = true\nnum_worlds = 64\ncapacity_preset = "medium"\n'
                              'deterministic = true\nepisodes = 2\ninitial_life = 2\n'
                              'initial_points = 0\npoint_generation_probability_max = 0.5\n'
                              'survival_horizon = 2\n')
            command = [sys.executable, '-m', 'self_genesis', 'train', '--config', str(config),
                       '--num-worlds', '3', '--capacity-preset', 'small', '--seed', '42',
                       '--memory-dim', '9', '--device', 'cpu']
            records = []
            for filename in ('a.jsonl', 'b.jsonl'):
                output = path / filename
                completed = subprocess.run(command + ['--output', str(output)], check=True,
                                           text=True, capture_output=True)
                summary = json.loads(completed.stdout)
                self.assertEqual(summary['updates'], 2)
                self.assertEqual(summary['episodes'], 6)
                self.assertEqual(summary['total_steps'], 12)
                self.assertEqual(summary['config']['memory_dim'], 9)
                self.assertEqual(summary['config']['affect_dim'], 4)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(rows[0]['random_algorithm'], 'philox4x32-10')
                self.assertEqual(rows[0]['config'], summary['config'])
                self.assertEqual(rows[1]['seeds'], [42, 43, 44])
                self.assertEqual(rows[2]['seeds'], [45, 46, 47])
                self.assertEqual(len(rows[1]['survival_returns']), 3)
                self.assertNotEqual(rows[1]['loss'], rows[2]['loss'])
                for row in rows:
                    row.pop("measurement", None)
                records.append(rows)
            self.assertEqual(*records)
            original = (path / 'a.jsonl').read_bytes()
            failed = subprocess.run(command + ['--output', str(path / 'a.jsonl')],
                                    text=True, capture_output=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn('File exists', failed.stderr)
            self.assertEqual(original, (path / 'a.jsonl').read_bytes())

    def check_sizes(self, device):
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(batched=True, deterministic=True, episodes=1,
                                      device=device, num_agents=2, initial_life=1,
                                      initial_points=0, memory_dim=4, affect_dim=2,
                                      entity_memory_dim=0, max_message_length=0)
            for count in (64, 256, 512, 1024):
                with self.subTest(device=device, worlds=count):
                    result = run_training(replace(config, num_worlds=count),
                                          Path(directory) / f'{count}.jsonl')
                    self.assertEqual(result['total_steps'], count)
                    self.assertEqual(result['last_training']['steps'], (1,) * count)
                    self.assertTrue(all(result['last_training']['terminated']))

    def test_world_counts_without_fixed_barrier(self):
        self.check_sizes('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_world_counts(self):
        self.check_sizes('cuda')

    def test_already_initialized_cuda_requires_workspace_before_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            output = Path(directory) / 'bad.jsonl'
            with patch('torch.cuda.is_available', return_value=True), patch(
                    'torch.cuda.is_initialized', return_value=True):
                with self.assertRaisesRegex(ValueError, 'before CUDA initialization'):
                    run_training(ExperimentConfig(batched=True, deterministic=True,
                                                  device='cuda'), output)
            self.assertFalse(output.exists())

    def test_determinism_settings_restored_after_validation_failure(self):
        previous = torch.are_deterministic_algorithms_enabled()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bad.jsonl'
            with self.assertRaisesRegex(ValueError, 'explicit survival_horizon'):
                run_training(ExperimentConfig(batched=True, deterministic=not previous,
                                              point_generation_probability_max=1), output)
            self.assertFalse(output.exists())
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous)
