"""Real CLI training, temporal analysis, and matched baseline evaluations."""

from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


class RenewableIntegrationTests(unittest.TestCase):
    def run_workflow(self, device):
        # Short horizon guarantees censoring; the longer budget exercises deaths.
        for horizon in (3, 30):
            with self.subTest(device=device, horizon=horizon), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                options = ['--config', str(ROOT / 'configs/default.toml'),
                           '--device', device, '--seed', '42', '--episodes', '2',
                           '--initial-life', '4', '--initial-points', '1',
                           '--survival-horizon', str(horizon)]
                log = directory / 'train.jsonl'
                subprocess.run([sys.executable, '-m', 'self_genesis', 'train',
                                *options, '--output', str(log)],
                               check=True, capture_output=True, text=True)
                analyzed = subprocess.run([
                    sys.executable, str(ROOT / 'examples/analyze_run.py'), str(log)
                ], check=True, capture_output=True, text=True)
                rows = [json.loads(line) for line in analyzed.stdout.splitlines()]
                records = [json.loads(line) for line in log.read_text().splitlines()]
                updates = []
                for episode, row in enumerate(rows):
                    events = [r for r in records if r['episode'] == episode]
                    start, summary, update = events[0], events[-2], events[-1]
                    updates.append(update)
                    steps = events[1:-2]
                    self.assertTrue(all(s['type'] == 'step' for s in steps))
                    self.assertLessEqual(len(steps), horizon)
                    self.assertFalse(summary['truncated'])
                    self.assertTrue(summary['terminated'] or summary['horizon_completed'])
                    self.assertTrue(math.isfinite(row['loss']))
                    abilities = start['point_generation_probability']
                    self.assertTrue(all(0.1 <= p <= 0.3 for p in abilities))
                    generated = [0] * 4
                    life, points = start['life'], start['points']
                    actions = iter(row['relationship_actions'])
                    history = {}
                    for step in steps:
                        donors = Counter(t['donor'] for t in step['successful_transfers'])
                        recipients = Counter(t['recipient'] for t in step['successful_transfers'])
                        for i in range(4):
                            self.assertEqual(step['rewards'][i], int(life[i] > 0))
                            self.assertEqual(step['life'][i], life[i] + recipients[i] - step['rewards'][i])
                            self.assertEqual(step['points'][i], points[i] - donors[i] + step['generated_points'][i])
                            self.assertIn(step['generated_points'][i], (0, 1))
                            if step['life'][i] == 0:
                                self.assertEqual(step['generated_points'][i], 0)
                        for callback in step['callbacks']:
                            # Hidden ability and logging identities never enter observations.
                            self.assertEqual(set(callback['observation']), {
                                'life', 'points', 'partner_life', 'partner_points',
                                'partner_appearance', 'first', 'received_message', 'partner_action'})
                            if callback['phase'] != 'action':
                                continue
                            action = next(actions)
                            agent, partner = action['agent'], action['partner']
                            prior = history.get((agent, partner), [])
                            self.assertEqual(action['prior']['encounters'], len(prior))
                            self.assertEqual(action['prior']['received_aid'], sum(prior))
                            self.assertEqual(action['partner_prior_generated_points'], generated[partner])
                            self.assertEqual(action['partner_generation_probability'], abilities[partner])
                            self.assertEqual(action['partner_appearance'], start['appearance'][partner])
                            self.assertEqual(action['action'], callback['choice'])
                            self.assertEqual(action['successful_aid'], bool(donors[agent]))
                        # Update only after checking both actions, independently of analyzer.
                        for agent in step['participants']:
                            partner, = [i for i in step['participants'] if i != agent]
                            history.setdefault((agent, partner), []).append(bool(donors[partner]))
                        generated = [a + b for a, b in zip(generated, step['generated_points'])]
                        life, points = step['life'], step['points']
                    self.assertIsNone(next(actions, None))
                    returns = [sum(s['rewards'][i] for s in steps) for i in range(4)]
                    self.assertEqual(row['survival_returns'], returns)
                    for i, lifetime in enumerate(summary['lifetimes']):
                        self.assertEqual(lifetime['observed_steps'], returns[i])
                        self.assertEqual(lifetime['censored'], life[i] > 0)
                        self.assertEqual(lifetime['death_step'], None if life[i] > 0 else returns[i] - 1)
                    if horizon == 3:
                        self.assertTrue(summary['horizon_completed'])
                        self.assertEqual(row['deaths'], 0)
                        self.assertIsNone(row['mean_survival_time'])
                    else:
                        self.assertTrue(summary['terminated'])
                        self.assertEqual(row['mean_survival_time'], sum(returns) / 4)
                self.assertEqual(len(rows), 2)
                reports = []
                for name in ('comparison.json', 'repeat.json'):
                    output = directory / name
                    subprocess.run([sys.executable, '-m', 'self_genesis', 'compare',
                                    *options, '--evaluation-seeds', '101', '102',
                                    '--output', str(output)],
                                   check=True, capture_output=True, text=True)
                    reports.append(json.loads(output.read_text()))
                self.assertEqual(*reports)
                for actual, expected in zip(reports[0]['training'], updates):
                    for key, value in actual.items():
                        self.assertEqual(value, expected[key])
                for seed in (101, 102):
                    evaluations = [r for r in reports[0]['evaluations'] if r['seed'] == seed]
                    self.assertEqual([r['policy'] for r in evaluations],
                                     ['learned', 'always-GIVE', 'always-NOTHING'])
                    for result in evaluations:
                        self.assertEqual(result['initial'], evaluations[0]['initial'])
                        self.assertEqual(result['config'], evaluations[0]['config'])
                        self.assertLessEqual(result['steps'], horizon)
                        self.assertEqual(result['successful_aid'], sum(
                            r['successful_aid'] for r in result['relationship_actions']))
                    nothing = evaluations[-1]
                    self.assertEqual(nothing['survival_returns'], [min(4, horizon)] * 4)
                    self.assertEqual(nothing['successful_aid'], 0)
                    self.assertEqual(nothing['action_counts']['GIVE'], 0)
                    self.assertEqual(evaluations[1]['action_counts']['NOTHING'], 0)

    def test_cpu_renewable_workflow(self):
        self.run_workflow('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_renewable_workflow(self):
        self.run_workflow('cuda')
