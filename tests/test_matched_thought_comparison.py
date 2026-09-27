"""Matched Thought matrix retains behavioral evidence and observational probes."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.analysis import evaluation_summary
from self_genesis.comparison import (INTERVENTIONS, THOUGHT_CONDITIONS,
                                     evaluate_policy, run_comparison)
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy


class MatchedThoughtComparisonTests(unittest.TestCase):
    def config(self, **kwargs):
        return ExperimentConfig(num_agents=4, initial_life=4, survival_horizon=3,
                                episodes=2, num_worlds=2, memory_dim=4, affect_dim=2,
                                entity_memory_dim=2, max_message_length=1,
                                encounter_count=2, deterministic=True, **kwargs)

    def test_matrix_matching_frozen_weights_metrics_and_reproducibility(self):
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as directory:
                config = self.config(batched=batched)
                snapshots = {}

                def checked(settings, policy, **kwargs):
                    if isinstance(policy, RecurrentPolicy):
                        self.assertFalse(policy.training)
                        self.assertTrue(all(not p.requires_grad and p.grad is None
                                            for p in policy.parameters()))
                        before = snapshots.setdefault(policy, {k: v.clone() for k, v
                                                               in policy.state_dict().items()})
                        self.assertEqual(settings.thought_mode, policy.thought_mode)
                        self.assertEqual(settings.think_steps, policy.think_steps)
                    result = evaluate_policy(settings, policy, **kwargs)
                    if isinstance(policy, RecurrentPolicy):
                        for key, value in policy.state_dict().items():
                            self.assertTrue(torch.equal(before[key], value))
                    return result

                reports = []
                for repeat in range(2):
                    output = Path(directory) / f'{repeat}.json'
                    with patch('self_genesis.comparison.evaluate_policy', side_effect=checked):
                        result = run_comparison(config, output, compare_thought=True,
                                                training_seeds=[10, 20], evaluation_seeds=[100, 101])
                    self.assertEqual(result['training_episodes'], 12 * (2 if batched else 1))
                    reports.append(json.loads(output.read_text()))
                self.assertEqual(*reports)
                report = reports[0]
                self.assertEqual(len(report['training_runs']), 6)
                self.assertEqual(len(report['evaluations']), 72)
                self.assertEqual(report['interventions'], list(INTERVENTIONS))
                for run in report['training_runs']:
                    settings = dict(run['config'])
                    settings.update(seed=config.seed, thought_mode=config.thought_mode,
                                    think_steps=config.think_steps)
                    self.assertEqual(settings, json.loads(json.dumps(config.__dict__)))
                    self.assertEqual(len(run['updates']), config.episodes)
                    if batched:
                        self.assertEqual(run['world_seeds'], [[run['seed'], run['seed'] + 1],
                                                              [run['seed'] + 2, run['seed'] + 3]])
                counts = {r['policy']: r['parameter_count'] for r in report['training_runs']}
                self.assertEqual(counts['recurrent-16'], counts['recurrent-32'])
                self.assertLess(counts['shallow'], counts['recurrent-16'])
                for seed in (100, 101):
                    rows = [r for r in report['evaluations'] if r['seed'] == seed]
                    self.assertTrue(all(r['initial'] == rows[0]['initial'] for r in rows))
                for row in report['evaluations']:
                    if row['policy'] in ('shallow', 'recurrent-16', 'recurrent-32'):
                        self.assertEqual(len(row['thought_samples']), 6)
                        for sample in row['thought_samples']:
                            self.assertEqual(len(sample['dynamics']),
                                             1 if row['policy'] == 'shallow' else row['config']['think_steps'])
                    else:
                        self.assertEqual(row['thought_samples'], [])
                for summary in report['summaries']:
                    rows = [r for r in report['evaluations'] if all(
                        r[key] == summary[key] for key in ('training_seed', 'policy', 'intervention'))]
                    expected = evaluation_summary(rows, config.vocabulary_size)
                    for key, value in expected.items():
                        self.assertEqual(summary[key], json.loads(json.dumps(value)))
                    for key in ('give_collapse', 'partner_history_metrics', 'communication',
                                'mean_observed_lifetime'):
                        self.assertIn(key, summary)
                self.assertEqual(len(report['intervention_effects']), 48)
                for effect in report['intervention_effects']:
                    rows = [r for r in report['evaluations']
                            if r['seed'] == effect['evaluation_seed']
                            and r['training_seed'] == effect['training_seed']
                            and r['policy'] == effect['policy']]
                    before = next(r for r in rows if r['intervention'] is None)
                    after = next(r for r in rows if r['intervention'] == effect['intervention'])
                    self.assertEqual(effect['mean_observed_lifetime_delta'],
                                     after['mean_observed_lifetime'] - before['mean_observed_lifetime'])

    def test_sampling_is_bounded_and_does_not_change_evaluation(self):
        for _, mode, steps in THOUGHT_CONDITIONS:
            config = replace(self.config(thought_mode=mode, think_steps=steps),
                             initial_life=6, survival_horizon=5)
            network = RecurrentPolicy(config.appearance_dim, thought_mode=mode, think_steps=steps,
                                      max_message_length=config.max_message_length)
            for intervention in (None, *INTERVENTIONS):
                plain = evaluate_policy(config, network, intervention=intervention)
                sampled = evaluate_policy(config, network, intervention=intervention, sample_thought=True)
                samples = sampled.pop('thought_samples')
                self.assertEqual(plain, sampled)
                self.assertEqual(len(samples), 6)
        empty = evaluate_policy(replace(config, encounter_count=0), network, sample_thought=True)
        self.assertEqual(empty['thought_samples'], [])

    def test_holdout_rejects_overlap_before_creating_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bad.json'
            for batched, starts, seeds in ((False, [10, 20], None),
                                          (False, [10, 20], [20]),
                                          (True, [10, 20], [23]),
                                          (True, [2**63 - 2], [1])):
                with self.assertRaisesRegex(ValueError, 'held.out'):
                    run_comparison(self.config(batched=batched), output, compare_thought=True,
                                   training_seeds=starts, evaluation_seeds=seeds)
                self.assertFalse(output.exists())

    def test_cli_matrix_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            command = [sys.executable, '-m', 'self_genesis', 'compare', '--compare-thought',
                       '--episodes', '1', '--survival-horizon', '1', '--evaluation-seeds', '100',
                       '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)['schema_version'], 4)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(output.read_bytes(), original)
            invalid = subprocess.run([sys.executable, '-m', 'self_genesis', 'init',
                                      '--compare-thought'], capture_output=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
