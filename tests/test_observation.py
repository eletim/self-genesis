import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.observation import RunRecorder
from self_genesis.policy import AgentPolicy, RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode
from self_genesis.world import Action


class ObservationTests(unittest.TestCase):
    def collector(self, recorder=None, length=3):
        torch.manual_seed(12)
        config = ExperimentConfig(num_agents=3, appearance_dim=2,
                                  initial_life=3, initial_points=0, seed=7)
        return RolloutCollector(config, RecurrentPolicy(2, max_message_length=length),
                                recorder=recorder)

    def read(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_resource_reconstruction_and_horizon_censoring(self):
        for probability in (0, 1):
            with self.subTest(probability=probability), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'run.jsonl'
                config = ExperimentConfig(
                    num_agents=3, appearance_dim=2, initial_life=2, initial_points=0,
                    point_generation_probability_min=probability,
                    point_generation_probability_max=probability, survival_horizon=4)
                with RunRecorder(path) as recorder, patch.object(
                        AgentPolicy, 'act', return_value=Action.GIVE):
                    collector = RolloutCollector(config, RecurrentPolicy(2), recorder=recorder)
                    collector.collect(1)
                    collector.collect(10)
                    collector.collect(1)  # No duplicate events after either ending.
                    collector.reset()
                    collector.collect(1)
                rows = self.read(path)
                episode = [row for row in rows if row['episode'] == 0]
                start = episode[0]
                self.assertEqual(start['point_generation_probability'], [probability] * 3)
                life, points = start['life'][:], start['points'][:]
                steps = [row for row in episode if row['type'] == 'step']
                self.assertEqual(len(steps), 4 if probability else 2)
                self.assertEqual(steps[0]['successful_transfers'], [])
                self.assertEqual(steps[0]['generated_points'], [probability] * 3)
                self.assertEqual(sum(c['choice'] == 'GIVE' for c in
                                     steps[0]['callbacks']), 2)
                transfers = 0
                for step in steps:
                    alive = [value > 0 for value in life]
                    incoming, outgoing = [0] * 3, [0] * 3
                    for event in step['successful_transfers']:
                        donor, recipient = event['donor'], event['recipient']
                        self.assertNotEqual(donor, recipient)
                        self.assertTrue(alive[donor] and alive[recipient])
                        self.assertGreater(points[donor], 0)
                        self.assertEqual(set((donor, recipient)), set(step['participants']))
                        outgoing[donor] += 1
                        incoming[recipient] += 1
                        transfers += 1
                    life = [max(0, value + incoming[i] - alive[i])
                            for i, value in enumerate(life)]
                    self.assertEqual(step['generated_points'],
                                     [probability if value > 0 else 0 for value in life])
                    points = [value - outgoing[i] + step['generated_points'][i]
                              for i, value in enumerate(points)]
                    self.assertEqual(step['life'], life)
                    self.assertEqual(step['points'], points)
                    for callback in step['callbacks']:
                        self.assertEqual(set(callback['observation']), {
                            'life', 'points', 'partner_life', 'partner_points',
                            'partner_appearance', 'first', 'received_message', 'partner_action'})
                summaries = [row for row in episode if row['type'] == 'summary']
                self.assertTrue(summaries[0]['truncated'])
                summary = summaries[-1]
                self.assertEqual(summary['collection_steps'], 0)
                self.assertEqual(summary['horizon_completed'], bool(probability))
                self.assertEqual(summary['terminated'], not probability)
                self.assertFalse(summary['truncated'])
                self.assertEqual(summary['action_counts']['GIVE'], 2 * len(steps))
                self.assertLess(transfers, summary['action_counts']['GIVE'])
                self.assertEqual([x['censored'] for x in summary['lifetimes']],
                                 [value > 0 for value in life])
                if probability:
                    self.assertEqual(transfers, 6)
                    self.assertEqual(summary['deaths'], 1)
                    self.assertEqual(summary['mean_completed_lifetime'], 2)
                    self.assertIsNone(summary['mean_survival_time'])
                else:
                    self.assertEqual(summary['mean_survival_time'], 2)
                reset_start = next(row for row in rows
                                   if row['episode'] == 1 and row['type'] == 'episode_start')
                self.assertEqual(reset_start['point_generation_probability'],
                                 start['point_generation_probability'])
                self.assertEqual(reset_start['appearance'], start['appearance'])
                self.assertEqual(rows[-1]['action_counts']['GIVE'], 2)

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
