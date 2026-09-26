"""Exercise the documented training-to-analysis path across real processes."""

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


class ExperimentIntegrationTests(unittest.TestCase):
    def check_experiment(self, device):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            completed = subprocess.run([
                sys.executable, '-m', 'self_genesis', 'train',
                '--config', str(ROOT / 'configs/default.toml'),
                '--seed', '42', '--device', device, '--episodes', '2',
                '--initial-life', '3', '--initial-points', '1',
                '--point-generation-probability-min', '0',
                '--point-generation-probability-max', '0',
                '--output', str(output),
            ], check=True, capture_output=True, text=True)
            run = json.loads(completed.stdout)
            analyzed = subprocess.run([
                sys.executable, str(ROOT / 'examples/analyze_run.py'), str(output)
            ], check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in analyzed.stdout.splitlines()]
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            total_steps = 0
            for episode, row in enumerate(rows):
                with self.subTest(device=device, episode=episode):
                    events = [r for r in records if r['episode'] == episode]
                    self.assertEqual(events[0]['type'], 'episode_start')
                    self.assertEqual([r['type'] for r in events[-2:]], ['summary', 'training'])
                    start, summary, update = events[0], events[-2], events[-1]
                    steps = [r for r in events if r['type'] == 'step']
                    total_steps += len(steps)
                    self.assertEqual([s['step'] for s in steps], list(range(len(steps))))
                    self.assertLessEqual(len(steps), 7)
                    self.assertEqual(torch.device(row['device']).type, device)
                    self.assertTrue(math.isfinite(row['loss']))
                    returns = [sum(s['rewards'][i] for s in steps) for i in range(4)]
                    self.assertEqual(row['survival_returns'], returns)
                    self.assertEqual(row['mean_survival_time'], sum(returns) / 4)
                    self.assertEqual(row['deaths'], 4)
                    self.assertEqual(row['final_life'], [0] * 4)
                    self.assertEqual(row['final_points'], steps[-1]['points'])
                    actions = Counter({'GIVE': 0, 'NOTHING': 0})
                    tokens = [0] * 4
                    loss = 0.0
                    previous_life = start['life']
                    previous_points = start['points']
                    for s in steps:
                        # Every spent Point restores one Life before living agents decay.
                        spent = sum(previous_points) - sum(s['points'])
                        self.assertGreaterEqual(spent, 0)
                        self.assertEqual(sum(s['life']),
                                         sum(previous_life) + spent - sum(s['rewards']))
                        previous_life, previous_points = s['life'], s['points']
                        for callback in s['callbacks']:
                            agent = callback['agent']
                            future = sum(t['rewards'][agent] for t in steps[s['step']:])
                            loss -= callback['log_probability'] * future / 4
                            if callback['phase'] == 'action':
                                actions[callback['choice']] += 1
                            else:
                                for token in callback['choice']:
                                    tokens[token] += 1
                            state = callback['state_after']
                            for name, dimension in [('working_memory', 16), ('affect', 4)]:
                                self.assertEqual(len(state[name]), dimension)
                                self.assertTrue(all(math.isfinite(v) for v in state[name]))
                    self.assertTrue(math.isclose(loss, row['loss'], rel_tol=1e-5))
                    self.assertEqual(row['action_counts'], dict(actions))
                    self.assertEqual(row['token_counts'], tokens)
                    self.assertGreater(sum(tokens), 0)
                    for action, count in actions.items():
                        self.assertEqual(row['action_ratios'][action], count / sum(actions.values()))
                    for i, lifetime in enumerate(summary['lifetimes']):
                        death = [s['step'] for s in steps if s['died'][i]]
                        self.assertEqual(death, [lifetime['death_step']])
                        self.assertFalse(lifetime['censored'])
                        self.assertEqual(lifetime['observed_steps'], returns[i])
                    self.assertEqual(update['steps'], len(steps))
                    self.assertEqual(start['settings']['device'], device)
            self.assertEqual(run['total_steps'], total_steps)
            self.assertEqual(run['last_training']['loss'], rows[-1]['loss'])
            # A flushed but interrupted last episode must not look complete.
            output.write_text('\n'.join(json.dumps(r) for r in records[:-1]) + '\n')
            failed = subprocess.run([
                sys.executable, str(ROOT / 'examples/analyze_run.py'), str(output)
            ], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Incomplete training run', failed.stderr)
            self.assertEqual(failed.stdout, '')

    def test_cpu_training_and_analysis(self):
        self.check_experiment('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_training_and_analysis(self):
        self.check_experiment('cuda')
