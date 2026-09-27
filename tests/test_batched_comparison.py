"""Batched training hands its actual frozen weights to held-out density evaluation."""
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
from self_genesis.comparison import INTERVENTIONS, evaluate_policy, run_comparison
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.training import run_training


class BatchedComparisonTests(unittest.TestCase):
    def config(self, **kwargs):
        return ExperimentConfig(batched=True, deterministic=True, num_worlds=2,
                                num_agents=4, episodes=2, initial_life=3,
                                survival_horizon=4, memory_dim=4, affect_dim=2,
                                entity_memory_dim=2, max_message_length=1, **kwargs)

    def test_replay_training_parity_frozen_weights_and_matched_effects(self):
        config = self.config(encounter_fraction=1.0)
        snapshots = {}

        def checked(settings, policy, **kwargs):
            if isinstance(policy, RecurrentPolicy):
                self.assertFalse(policy.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in policy.parameters()))
                before = snapshots.setdefault(policy, {k: v.clone()
                                                        for k, v in policy.state_dict().items()})
                for key, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value))
            self.assertFalse(settings.batched)
            return evaluate_policy(settings, policy, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for index in range(2):
                with patch('self_genesis.comparison.evaluate_policy', side_effect=checked):
                    result = run_comparison(config, root / f'{index}.json',
                                            evaluation_seeds=[101, 102],
                                            interventions=INTERVENTIONS)
                self.assertEqual(result['training_episodes'], 8)
                reports.append(json.loads((root / f'{index}.json').read_text()))
            self.assertEqual(*reports)
            run_training(config, root / 'train.jsonl')
            trained = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
            report = reports[0]
            for actual, expected in zip(report['training_runs'][0]['updates'], trained[1:]):
                self.assertEqual(actual, {key: expected[key] for key in actual})
        self.assertEqual(report['training_runs'][0]['world_seeds'], [[0, 1], [2, 3]])
        self.assertEqual(len(report['evaluations']), 26)
        self.assertEqual(report['evaluation_execution'], 'sequential-fp32')
        self.assertEqual(report['training_execution'], 'batched')
        for seed in (101, 102):
            rows = [r for r in report['evaluations'] if r['seed'] == seed]
            self.assertTrue(all(r['initial'] == rows[0]['initial'] for r in rows))
            self.assertTrue(all(r['config']['encounter_fraction'] == 1 for r in rows))
        for effect in report['intervention_effects']:
            matched = [r for r in report['evaluations'] if r['seed'] == effect['evaluation_seed']
                       and r['policy'] == effect['policy']]
            before = evaluation_summary([next(r for r in matched if r['intervention'] is None)], 4)
            after = evaluation_summary([next(r for r in matched
                                            if r['intervention'] == effect['intervention'])], 4)
            for field in ('mean_observed_lifetime', 'prior_aid_give_difference'):
                expected = (None if before[field] is None or after[field] is None
                            else after[field] - before[field])
                self.assertEqual(effect[field + '_delta'], expected)
            self.assertIn('encounter_exposure', before)
            field = 'repeat_minus_first_producer_difference'
            a, b = after['partner_history_metrics'][field], before['partner_history_metrics'][field]
            self.assertEqual(effect[field + '_delta'], None if a is None or b is None else a - b)
        for network, before in snapshots.items():
            for key, value in network.state_dict().items():
                self.assertTrue(torch.equal(before[key], value))

    def test_holdout_checks_all_worlds_updates_and_wraparound_before_output(self):
        previous = torch.are_deterministic_algorithms_enabled()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bad.json'
            for start, seeds in ((0, None), (0, [0]), (0, [3]), (2**63 - 2, [1])):
                with self.assertRaisesRegex(ValueError, 'held.out'):
                    run_comparison(replace(self.config(), seed=start), output,
                                   evaluation_seeds=seeds)
                self.assertFalse(output.exists())
                self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous)

    def test_cli_count_density_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            command = [sys.executable, '-m', 'self_genesis', 'compare', '--batched',
                       '--deterministic', '--num-worlds', '2', '--episodes', '1',
                       '--survival-horizon', '2', '--encounter-count', '2',
                       '--entity-memory-dim', '0', '--evaluation-seeds', '100',
                       '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            original = output.read_bytes()
            report = json.loads(original)
            baseline = next(r for r in report['evaluations'] if r['policy'] == 'always-GIVE')
            self.assertEqual(baseline['action_callbacks'], 8)
            failed = subprocess.run(command, capture_output=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(output.read_bytes(), original)
