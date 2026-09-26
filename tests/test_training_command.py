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
                     'affect_dim', 'episodes', 'survival_horizon'):
            for value in (-1, True, 1.5, '2'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExperimentConfig(**{name: value})
            if name != 'max_message_length':
                with self.assertRaises(ValueError):
                    ExperimentConfig(**{name: 0})
        for value in (0, -1, True, '0.1', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExperimentConfig(learning_rate=value)
        for name in ('value_loss_coefficient', 'action_entropy_coefficient',
                     'message_entropy_coefficient'):
            for value in (-1, True, '0.1', float('nan'), float('inf'), -float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExperimentConfig(**{name: value})
        with self.assertRaises(ValueError):
            ExperimentConfig(value_loss_coefficient=0)
        ExperimentConfig(action_entropy_coefficient=0, message_entropy_coefficient=0)
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
                              'learning_rate = 0.02\nvalue_loss_coefficient = 0.7\n'
                              'action_entropy_coefficient = 0.2\nmessage_entropy_coefficient = 0.3\n')
            command = [sys.executable, '-m', 'self_genesis', 'train',
                       '--config', str(config), '--device', 'cpu', '--seed', '42',
                       '--episodes', '2', '--max-message-length', '0',
                       '--learning-rate', '0.005', '--value-loss-coefficient', '0.8',
                       '--action-entropy-coefficient', '0', '--message-entropy-coefficient', '0.04']
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
                for name, expected in (('value_loss_coefficient', 0.8),
                                       ('action_entropy_coefficient', 0),
                                       ('message_entropy_coefficient', 0.04)):
                    self.assertEqual(starts[0]['settings'][name], expected)
                    self.assertEqual(summary['config'][name], expected)
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

    def test_finite_horizon_cli_and_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            config = Path(directory) / 'run.toml'
            config.write_text('survival_horizon = 7\n')
            command = [sys.executable, '-m', 'self_genesis', 'train',
                       '--config', str(config), '--survival-horizon', '2',
                       '--initial-life', '10', '--episodes', '2',
                       '--point-generation-probability-max', '1', '--output', str(output)]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout)
            self.assertEqual(result['config']['survival_horizon'], 2)
            self.assertEqual(result['total_steps'], 4)
            self.assertTrue(result['last_training']['horizon_completed'])
            records = [json.loads(line) for line in output.read_text().splitlines()]
            for record in records:
                if record['type'] == 'summary':
                    self.assertTrue(record['horizon_completed'])
                    self.assertFalse(record['terminated'])
                    self.assertFalse(record['truncated'])
                    self.assertEqual(record['deaths'], 0)
                    self.assertIsNone(record['mean_survival_time'])
                    self.assertTrue(all(item['censored'] for item in record['lifetimes']))
            analyzed = subprocess.run(
                [sys.executable, 'examples/analyze_run.py', str(output)],
                check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in analyzed.stdout.splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                events = [r for r in records if r['episode'] == row['episode']]
                start = events[0]
                steps = [r for r in events if r['type'] == 'step']
                actions = row['relationship_actions']
                self.assertEqual(len(actions), sum(row['action_counts'].values()))
                for action in actions:
                    agent, partner = action['agent'], action['partner']
                    earlier = [s for s in steps if s['step'] < action['step']]
                    current = next(s for s in steps if s['step'] == action['step'])
                    self.assertEqual(action['partner_appearance'], start['appearance'][partner])
                    self.assertEqual(action['partner_generation_probability'],
                                     start['point_generation_probability'][partner])
                    self.assertEqual(action['partner_prior_generated_points'],
                                     sum(s['generated_points'][partner] for s in earlier))
                    self.assertEqual(action['prior']['encounters'],
                                     sum(set(s['participants']) == {agent, partner}
                                         for s in earlier))
                    self.assertEqual(action['successful_aid'],
                                     dict(donor=agent, recipient=partner)
                                     in current['successful_transfers'])

    def test_renewable_training_requires_horizon_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            with self.assertRaisesRegex(ValueError, 'explicit survival_horizon'):
                run_training(ExperimentConfig(point_generation_probability_max=0.1), output)
            self.assertFalse(output.exists())

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
