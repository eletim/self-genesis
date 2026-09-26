import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.config import ExperimentConfig, load_config
from self_genesis.training import run_training


class TrainingCommandTests(unittest.TestCase):
    def test_invalid_training_settings(self):
        for name in ('vocabulary_size', 'max_message_length', 'memory_dim',
                     'affect_dim', 'episodes'):
            for value in (-1, True, 1.5, '2'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExperimentConfig(**{name: value})
            if name != 'max_message_length':
                with self.assertRaises(ValueError):
                    ExperimentConfig(**{name: 0})
        for value in (0, -1, True, '0.1', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExperimentConfig(learning_rate=value)
        with self.assertRaisesRegex(ValueError, 'Unknown configuration keys'):
            load_config(typo=1)

    def test_cli_config_overrides_saved_updates_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / 'experiment.toml'
            config.write_text('seed = 19\nnum_agents = 3\nappearance_dim = 2\n'
                              'initial_life = 2\ninitial_points = 0\n'
                              'vocabulary_size = 5\nmax_message_length = 2\n'
                              'memory_dim = 7\naffect_dim = 3\nepisodes = 1\n'
                              'learning_rate = 0.02\n')
            command = [sys.executable, '-m', 'self_genesis', 'train',
                       '--config', str(config), '--device', 'cpu', '--seed', '42',
                       '--episodes', '2', '--max-message-length', '0',
                       '--learning-rate', '0.005']
            outputs = []
            for name in ('first.jsonl', 'second.jsonl'):
                output = path / name
                completed = subprocess.run(command + ['--output', str(output)],
                                           check=True, capture_output=True, text=True)
                summary = json.loads(completed.stdout)
                self.assertEqual(summary['output'], str(output.resolve()))
                self.assertEqual(summary['episodes'], 2)
                self.assertEqual(summary['total_steps'], 4)
                self.assertEqual(summary['resolved_device'], 'cpu')
                records = [json.loads(line) for line in output.read_text().splitlines()]
                outputs.append(records)
                starts = [r for r in records if r['type'] == 'episode_start']
                self.assertEqual(len(starts), summary['episodes'])
                self.assertEqual([r['episode'] for r in starts], [0, 1])
                for kind in ('summary', 'training'):
                    self.assertEqual(
                        [r['episode'] for r in records if r['type'] == kind],
                        [r['episode'] for r in starts])
                self.assertEqual(starts[0]['settings']['seed'], 42)
                self.assertEqual(starts[0]['policy_settings'], {
                    'appearance_dim': 2, 'vocabulary_size': 5,
                    'max_message_length': 0, 'memory_dim': 7, 'affect_dim': 3})
                updates = [r for r in records if r['type'] == 'training']
                self.assertEqual(len(updates), 2)
                self.assertEqual(updates[0]['optimizer_settings'][0]['lr'], 0.005)
                self.assertEqual(updates[0]['survival_returns'], [2, 2, 2])
                self.assertNotEqual(updates[0]['loss'], updates[1]['loss'])
                self.assertTrue(all(r['terminated'] and not r['truncated']
                                    for r in records if r['type'] == 'summary'))
            self.assertEqual(*outputs)
            original = (path / 'first.jsonl').read_bytes()
            for extra, message in (([], 'requires --output'),
                                   (['--output', str(path / 'first.jsonl')], 'File exists'),
                                   (['--output', str(path / 'invalid.jsonl'),
                                     '--episodes', '0'], 'episodes must be')):
                failed = subprocess.run(command + extra, capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
                self.assertIn(message, failed.stderr)
                self.assertNotIn('Traceback', failed.stderr)
            self.assertEqual(original, (path / 'first.jsonl').read_bytes())
            self.assertFalse((path / 'invalid.jsonl').exists())

    def test_unavailable_cuda_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            with patch('torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(ValueError, 'CUDA was requested'):
                    run_training(ExperimentConfig(device='cuda'), output)
            self.assertFalse(output.exists())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_training(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_training(ExperimentConfig(
                device='cuda', num_agents=2, initial_life=1, initial_points=0,
                episodes=1), Path(directory) / 'run.jsonl')
            self.assertEqual(summary['resolved_device'], 'cuda')
            self.assertEqual(summary['total_steps'], 1)


if __name__ == '__main__':
    unittest.main()
