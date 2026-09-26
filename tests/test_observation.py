import json
from pathlib import Path
import tempfile
import unittest

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.observation import RunRecorder
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode


class ObservationTests(unittest.TestCase):
    def collector(self, recorder=None, length=3):
        torch.manual_seed(12)
        config = ExperimentConfig(num_agents=3, appearance_dim=2,
                                  initial_life=3, initial_points=0, seed=7)
        return RolloutCollector(config, RecurrentPolicy(2, max_message_length=length),
                                recorder=recorder)

    def read(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_continuation_censoring_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            with RunRecorder(path) as recorder:
                collector = self.collector(recorder)
                collector.collect(1)
                rows = self.read(path)  # Flushed before close.
                summary = rows[-1]
                self.assertIsNone(summary['mean_survival_time'])
                self.assertEqual(summary['deaths'], 0)
                self.assertTrue(all(x['censored'] for x in summary['lifetimes']))
                collector.collect(10)
                collector.collect(1)  # Extinction must not count twice.
            rows = self.read(path)
            self.assertEqual(rows[0]['settings']['seed'], 7)
            self.assertEqual(rows[0]['policy_settings']['memory_dim'], 16)
            steps = [row for row in rows if row['type'] == 'step']
            self.assertEqual([row['step'] for row in steps], [0, 1, 2])
            summary = rows[-1]
            self.assertEqual(summary['deaths'], 3)
            self.assertEqual(summary['mean_survival_time'], 3)
            self.assertFalse(any(x['censored'] for x in summary['lifetimes']))
            self.assertEqual(summary['life'], [0, 0, 0])
            self.assertEqual(summary['points'], [0, 0, 0])
            self.assertEqual(sum(summary['token_counts']), 18)
            self.assertEqual(sum(summary['action_counts'].values()), 6)
            self.assertAlmostEqual(sum(summary['action_ratios'].values()), 1)
            counts = {'GIVE': 0, 'NOTHING': 0}
            for step in steps:
                self.assertEqual(step['rewards'], [1, 1, 1])
                self.assertEqual([x['phase'] for x in step['callbacks']],
                                 ['message', 'message', 'action', 'action'])
                first, second = step['participants']
                self.assertEqual([x['agent'] for x in step['callbacks']],
                                 [first, second, first, second])
                for callback in step['callbacks']:
                    self.assertNotIn('agent', callback['observation'])
                    self.assertEqual(len(callback['state_after']['working_memory']), 16)
                    self.assertEqual(len(callback['state_after']['affect']), 4)
                    if callback['phase'] == 'action':
                        counts[callback['choice']] += 1
            self.assertEqual(summary['action_counts'], counts)
            with self.assertRaises(FileExistsError):
                RunRecorder(path)

    def test_logging_preserves_policy_and_training(self):
        plain = self.collector()
        optimizer = torch.optim.Adam(plain.network.parameters(), lr=0.001)
        expected = train_episode(plain, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            with RunRecorder(path) as recorder:
                logged = self.collector(recorder)
                actual = train_episode(logged, torch.optim.Adam(
                    logged.network.parameters(), lr=0.001))
                self.assertEqual(expected, actual)
                for a, b in zip(plain.network.parameters(), logged.network.parameters()):
                    self.assertTrue(torch.equal(a, b))
                train_episode(logged, torch.optim.Adam(
                    logged.network.parameters(), lr=torch.tensor(0.001)))
            rows = self.read(path)
            updates = [row for row in rows if row['type'] == 'training']
            self.assertEqual([row['episode'] for row in updates], [1, 2])
            self.assertEqual(updates[0]['loss'], actual.loss)
            self.assertEqual(updates[0]['optimizer_settings'][0]['lr'], 0.001)

    def test_disabled_channel_and_lone_survivor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            with RunRecorder(path) as recorder:
                collector = self.collector(recorder, length=0)
                collector.world.state.life[:] = torch.tensor([1, 1, 3])
                collector.collect(2)
                summary = self.read(path)[-1]
                self.assertEqual(summary['deaths'], 2)
                self.assertEqual(summary['mean_completed_lifetime'], 1)
                self.assertIsNone(summary['mean_survival_time'])
                collector.collect(2)
            rows = self.read(path)
            steps = [row for row in rows if row['type'] == 'step']
            self.assertEqual(steps[1]['participants'], [])
            self.assertEqual(steps[1]['callbacks'], [])
            self.assertEqual(rows[-1]['token_counts'], [0, 0, 0, 0])
            self.assertEqual(rows[-1]['mean_survival_time'], 5 / 3)

    def test_reset_preserves_censored_summary_and_empty_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.jsonl'
            with RunRecorder(path) as recorder:
                collector = self.collector(recorder)
                collector.world.state.life[:] = torch.tensor([0, 0, 3])
                collector.collect(1)
                collector.reset()
                collector.collect(1)
            rows = self.read(path)
            summaries = [row for row in rows if row['type'] == 'summary']
            self.assertEqual([row['episode'] for row in summaries], [0, 1])
            self.assertEqual(summaries[0]['action_ratios'],
                             {'GIVE': None, 'NOTHING': None})
            self.assertTrue(summaries[0]['lifetimes'][2]['censored'])
            self.assertEqual(summaries[1]['mean_observed_lifetime'], 1)
