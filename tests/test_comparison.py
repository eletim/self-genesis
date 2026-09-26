"""Fixed-policy outcomes and matched, reproducible comparison reports."""

from dataclasses import fields, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from self_genesis.comparison import ProducerOracle, evaluate_policy, run_comparison
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.policy import RecurrentPolicy
from self_genesis.world import Action, World


class ComparisonTests(unittest.TestCase):
    def test_oracle_selects_producers_and_sends_empty_messages(self):
        config = ExperimentConfig(num_agents=4, point_generation_probability_max=1,
                                  survival_horizon=3)
        world = World(config)
        world.state.point_generation_probability[:] = torch.tensor([0, 0.25, 0.5, 1.0])
        oracle = ProducerOracle(world, config)
        for i, expected in enumerate((Action.NOTHING, Action.NOTHING,
                                      Action.GIVE, Action.GIVE)):
            for points in (0, 1):
                observation = Observation(2, points, 2, 0, world.state.appearance[i],
                                          True, (), None)
                self.assertEqual(oracle.communicate(observation), ())
                self.assertIs(oracle.act(observation), expected)
        self.assertEqual({field.name for field in fields(Observation)}, {
            'life', 'points', 'partner_life', 'partner_points', 'partner_appearance',
            'first', 'received_message', 'partner_action'})

    def test_oracle_uses_shared_transfer_decay_and_generation_rules(self):
        config = ExperimentConfig(num_agents=2, initial_life=2, initial_points=1,
                                  point_generation_probability_max=1, survival_horizon=5)
        world = World(config)
        world.state.point_generation_probability[:] = torch.tensor([0.0, 1.0])
        with patch('self_genesis.comparison.World', return_value=world):
            report = evaluate_policy(config, ProducerOracle)
        self.assertEqual(report['survival_returns'], [2, 3])
        self.assertEqual(report['successful_aid'], 1)
        self.assertEqual(report['action_counts'], {'GIVE': 2, 'NOTHING': 2})
        self.assertEqual(report['final_points'], [0, 3])
        self.assertTrue(report['terminated'])
        for row in report['relationship_actions']:
            self.assertEqual(row['action'], 'GIVE' if row['partner'] == 1 else 'NOTHING')
        zero = replace(config, point_generation_probability_max=0)
        self.assertEqual(evaluate_policy(zero, ProducerOracle),
                         evaluate_policy(zero, Action.NOTHING))
        for probability in (0.7, 1.0):
            all_producers = replace(config, initial_points=0,
                                    point_generation_probability_min=probability,
                                    point_generation_probability_max=probability)
            self.assertEqual(evaluate_policy(all_producers, ProducerOracle),
                             evaluate_policy(all_producers, Action.GIVE))

    def test_full_comparison_preserves_trained_weights(self):
        snapshots = []

        def checked_evaluation(config, policy):
            if isinstance(policy, RecurrentPolicy):
                snapshots.append((policy, {key: value.clone() for key, value
                                           in policy.state_dict().items()}))
                self.assertFalse(policy.training)
            result = evaluate_policy(config, policy)
            for network, before in snapshots:
                for key, value in network.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch('self_genesis.comparison.evaluate_policy',
                       side_effect=checked_evaluation) as evaluation:
                run_comparison(ExperimentConfig(num_agents=2, initial_life=3,
                                               episodes=1, survival_horizon=2),
                               Path(directory) / 'report.json', evaluation_seeds=[101, 102])
            self.assertEqual(evaluation.call_count, 8)
            self.assertEqual(len(snapshots), 2)

    def test_zero_generation_extinction_and_attempts(self):
        config = ExperimentConfig(num_agents=2, initial_life=2, initial_points=1)
        give = evaluate_policy(config, Action.GIVE)
        nothing = evaluate_policy(config, Action.NOTHING)
        self.assertEqual(give['survival_returns'], [3, 3])
        self.assertEqual(nothing['survival_returns'], [2, 2])
        self.assertEqual(give['action_counts'], {'GIVE': 6, 'NOTHING': 0})
        self.assertEqual(give['successful_aid'], 2)
        self.assertEqual(nothing['successful_aid'], 0)
        self.assertEqual(give['relationship_metrics']['unseen_partner']['action_callbacks'], 2)
        self.assertEqual(give['relationship_metrics']['previously_received_aid']['action_callbacks'], 4)
        self.assertIsNone(nothing['relationship_metrics']['previously_received_aid']['action_ratios']['GIVE'])
        self.assertEqual(nothing['relationship_metrics']['encountered_without_received_aid']['action_callbacks'], 2)
        for report in (give, nothing):
            self.assertTrue(report['terminated'])
            self.assertFalse(report['horizon_completed'])
            self.assertEqual(report['deaths'], 2)
            self.assertTrue(all(not row['censored'] for row in report['lifetimes']))

    def test_renewal_timing_censoring_and_extinction_on_horizon(self):
        config = ExperimentConfig(num_agents=2, initial_life=2, initial_points=0,
                                  point_generation_probability_min=1,
                                  point_generation_probability_max=1, survival_horizon=5)
        give = evaluate_policy(config, Action.GIVE)
        self.assertEqual(give['survival_returns'], [5, 5])
        self.assertEqual(give['successful_aid'], 8)  # No Points on the first step.
        self.assertTrue(give['horizon_completed'])
        self.assertFalse(give['terminated'])
        self.assertIsNone(give['mean_survival_time'])
        self.assertTrue(all(row['censored'] for row in give['lifetimes']))
        nothing = evaluate_policy(replace(config, survival_horizon=2), Action.NOTHING)
        self.assertTrue(nothing['terminated'])
        self.assertFalse(nothing['horizon_completed'])
        self.assertEqual(nothing['final_points'], [1, 1])
        exhausted = evaluate_policy(replace(config, initial_life=1), Action.GIVE)
        self.assertEqual(exhausted['survival_returns'], [1, 1])
        self.assertEqual(exhausted['successful_aid'], 0)

    def test_lone_survivor_has_no_action_callbacks(self):
        report = evaluate_policy(ExperimentConfig(num_agents=3, initial_life=2,
                                                 initial_points=1), Action.GIVE)
        self.assertEqual(report['deaths'], 3)
        self.assertEqual(report['steps'], 4)
        self.assertEqual(report['action_callbacks'], 6)
        self.assertEqual(sorted(report['survival_returns']), [2, 3, 4])
        self.assertEqual(max(r['step'] for r in report['relationship_actions']), 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            run_comparison(ExperimentConfig(device='cuda', num_agents=2,
                                           initial_life=2, initial_points=1,
                                           episodes=1, survival_horizon=2), output)
            rows = json.loads(output.read_text())['evaluations']
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row['resolved_device'], 'cuda:0')
                self.assertEqual(row['survival_returns'], [2, 2])

    def test_evaluation_releases_decision_statistics_each_step(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for length in (0, 3):
                with self.subTest(device=device, message_length=length):
                    config = ExperimentConfig(
                        device=device, num_agents=4, initial_life=10,
                        survival_horizon=5, max_message_length=length)
                    network = RecurrentPolicy(
                        config.appearance_dim, max_message_length=length).to(device)
                    agents = []
                    original_step = EncounterProtocol.step

                    def checked_step(protocol, policies):
                        agents[:] = [wrapper.policy for wrapper in policies]
                        for agent in agents:
                            self.assertEqual(agent.log_probs, [])
                            self.assertEqual(agent.values, [])
                            self.assertEqual(agent.entropies, [])
                        result = original_step(protocol, policies)
                        # Ensure the evaluation really generated statistics.
                        self.assertEqual(sum(len(a.values) for a in agents),
                                         2 if length == 0 else 4)
                        return result

                    with patch.object(EncounterProtocol, "step", autospec=True,
                                      side_effect=checked_step) as step:
                        report = evaluate_policy(config, network)
                    self.assertEqual(step.call_count, 5)
                    self.assertTrue(report['horizon_completed'])
                    self.assertEqual(report['deaths'], 0)
                    for agent in agents:
                        self.assertEqual(agent.log_probs, [])
                        self.assertEqual(agent.values, [])
                        self.assertEqual(agent.entropies, [])

    def test_evaluation_restarts_sampling_and_preserves_weights(self):
        config = ExperimentConfig(initial_life=4, survival_horizon=3)
        network = RecurrentPolicy(config.appearance_dim)
        before = {key: value.clone() for key, value in network.state_dict().items()}
        first = evaluate_policy(config, network)
        evaluate_policy(config, Action.GIVE)
        evaluate_policy(config, ProducerOracle)
        self.assertEqual(first, evaluate_policy(config, network))
        for key, value in network.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(p.grad is None for p in network.parameters()))

    def test_cli_matched_seeds_reproducibility_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            command = [sys.executable, '-m', 'self_genesis', 'compare', '--device', 'cpu',
                       '--num-agents', '3', '--initial-life', '3', '--initial-points', '1',
                       '--episodes', '1', '--survival-horizon', '2', '--seed', '42',
                       '--point-generation-probability-max', '0.5',
                       '--evaluation-seeds', '7', '8']
            for name in ('first.json', 'second.json'):
                output = Path(directory) / name
                result = subprocess.run(command + ['--output', str(output)],
                                        check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(result.stdout)['evaluations'], 8)
                reports.append(json.loads(output.read_text()))
            self.assertEqual(*reports)
            report = reports[0]
            self.assertEqual(len(report['training']), 1)
            for seed in (7, 8):
                rows = [r for r in report['evaluations'] if r['seed'] == seed]
                self.assertEqual([r['policy'] for r in rows],
                                 ['learned', 'always-GIVE', 'always-NOTHING',
                                  'producer-oracle'])
                for row in rows:
                    self.assertEqual(row['initial'], rows[0]['initial'])
                    self.assertEqual(row['config'], rows[0]['config'])
                    self.assertEqual(row['survival_returns'], [2, 2, 2])
                    self.assertEqual([(r['step'], r['agent'], r['partner'])
                                      for r in row['relationship_actions']],
                                     [(r['step'], r['agent'], r['partner'])
                                      for r in rows[0]['relationship_actions']])
            original = output.read_bytes()
            failed = subprocess.run(command + ['--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(original, output.read_bytes())

    def test_invalid_comparisons_create_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            for config, seeds in (
                (ExperimentConfig(point_generation_probability_max=0.1), [1]),
                (ExperimentConfig(), []), (ExperimentConfig(), [-1]),
            ):
                with self.assertRaises(ValueError):
                    run_comparison(config, output, evaluation_seeds=seeds)
                self.assertFalse(output.exists())

    def test_training_and_comparison_cli_use_identical_reproducible_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for method in ('actor_critic', 'reinforce'):
                with self.subTest(method=method):
                    config = path / f'{method}.toml'
                    config.write_text(f'training_method = "{method}"\n')
                    command = [sys.executable, '-m', 'self_genesis']
                    settings = ['--config', str(config), '--device', 'cpu', '--seed', '17',
                                '--num-agents', '3', '--initial-life', '3', '--initial-points', '1',
                                '--episodes', '2', '--survival-horizon', '2',
                                '--value-loss-coefficient', '0.7',
                                '--action-entropy-coefficient', '0.2',
                                '--message-entropy-coefficient', '0.3']
                    training = path / f'{method}.jsonl'
                    subprocess.run(command + ['train', *settings, '--output', str(training)],
                                   check=True, capture_output=True, text=True)
                    updates = [json.loads(line) for line in training.read_text().splitlines()
                               if json.loads(line)['type'] == 'training']
                    reports = []
                    for repeat in range(2):
                        output = path / f'{method}-{repeat}.json'
                        subprocess.run(command + ['compare', *settings, '--output', str(output)],
                                       check=True, capture_output=True, text=True)
                        reports.append(json.loads(output.read_text()))
                    self.assertEqual(*reports)
                    self.assertEqual(reports[0]['config']['training_method'], method)
                    for update, recorded in zip(reports[0]['training'], updates, strict=True):
                        self.assertEqual(update, {key: recorded[key] for key in update})
                        self.assertEqual(update['training_method'], method)
                        self.assertAlmostEqual(
                            update['loss'], update['actor_loss'] + 0.7 * update['value_loss']
                            - 0.2 * update['action_entropy'] - 0.3 * update['message_entropy'],
                            places=5)
                        if method == 'reinforce':
                            self.assertEqual(update['value_loss'], 0)
                            self.assertEqual(update['action_entropy'], 0)
                            self.assertEqual(update['message_entropy'], 0)
                    invalid = path / f'{method}-invalid.json'
                    failed = subprocess.run(
                        command + ['compare', *settings, '--training-method', 'typo',
                                   '--output', str(invalid)], capture_output=True, text=True)
                    self.assertEqual(failed.returncode, 2)
                    self.assertFalse(invalid.exists())
