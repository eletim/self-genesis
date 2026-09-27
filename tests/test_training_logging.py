from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from examples.analyze_run import analyze
from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import ExperimentConfig, load_config
from self_genesis.observation import BatchedTraceRecorder, RunRecorder
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import run_training, survival_loss_components
from self_genesis.training_metrics import rollout_metrics
from self_genesis.world import Action


class TrainingLoggingTests(unittest.TestCase):
    def config(self, **kwargs):
        return ExperimentConfig(num_agents=2, num_worlds=3, initial_life=4,
                                initial_points=0, survival_horizon=3, episodes=3,
                                memory_dim=4, affect_dim=2, entity_memory_dim=3, **kwargs)

    def run_records(self, config, path):
        summary = run_training(config, path)
        return summary, [json.loads(line) for line in path.read_text().splitlines()]

    def test_sampling_validation_and_toml(self):
        for options in (dict(trace_worlds=[1]), dict(trace_worlds=[True]),
                        dict(trace_worlds=[0, 0]), dict(trace_worlds='0'),
                        dict(batched=True, num_worlds=2, trace_worlds=[2]),
                        dict(trace_step_interval=0), dict(trace_update_interval=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExperimentConfig(**options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.toml'
            path.write_text('batched=true\ntrace_worlds=[0, 2]\ntrace_step_interval=3\n')
            self.assertEqual(load_config(path).trace_worlds, (0, 2))

    def test_sampling_does_not_change_updates_or_aggregate_metrics(self):
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as directory:
                config = self.config(batched=batched)
                summary, quiet = self.run_records(config, Path(directory) / 'quiet.jsonl')
                sampled_summary, sampled = self.run_records(replace(
                    config, trace_worlds=(1,) if batched else (0,),
                    trace_step_interval=2, trace_update_interval=2),
                    Path(directory) / 'sampled.jsonl')
                self.assertEqual(summary['last_training'], sampled_summary['last_training'])
                kind = 'batch_training' if batched else 'training'
                quiet_updates = [r for r in quiet if r['type'] == kind]
                sampled_updates = [r for r in sampled if r['type'] == kind]
                for a, b in zip(quiet_updates, sampled_updates):
                    a.pop('measurement', None)
                    b.pop('measurement', None)
                    self.assertEqual(a, b)
                    metrics = a['metrics']
                    self.assertEqual(metrics['mean_observed_survival'], 3)
                    self.assertTrue(0 <= metrics['give_rate'] <= 1)
                    self.assertGreater(metrics['gradient_norm'], 0)
                    self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
                trace_kind = 'batch_trace' if batched else 'step'
                self.assertFalse(any(r['type'] == trace_kind for r in quiet))
                traces = [r for r in sampled if r['type'] == trace_kind]
                self.assertEqual([(r['update' if batched else 'episode'], r['step'])
                                  for r in traces], [(0, 0), (0, 2), (2, 0), (2, 2)])
                if not batched:
                    with self.assertRaisesRegex(ValueError, 'complete traces'):
                        list(analyze(Path(directory) / 'sampled.jsonl'))
                if batched:
                    self.assertTrue(all(r['world'] == 1 for r in traces))
                    self.assertEqual(len(traces[0]['entity_memory']['values']), 2)
                else:
                    self.assertEqual(len(traces[0]['callbacks']), 4)

    def test_batched_trace_sampling_after_independent_world_reset(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                config = self.config(batched=True, device=device, trace_worlds=(0, 1),
                                     trace_step_interval=2)
                collector = BatchedRolloutCollector(
                    config, RecurrentPolicy(config.appearance_dim).to(device), seeds=[0, 1, 2])
                records = []
                collector.trace_recorder = BatchedTraceRecorder(
                    config, lambda kind, **values: records.append(values))
                collector.world.state.life[0] = 1
                collector.collect(1)
                self.assertEqual([(r['world'], r['step']) for r in records], [(0, 0), (1, 0)])
                records.clear()

                collector.reset(0, seed=3)
                collector.collect(1)
                collector.collect(1)
                collector.collect(2)
                self.assertEqual([(r['world'], r['step']) for r in records],
                                 [(0, 0), (1, 2), (0, 2)])
                records.clear()

                # Other worlds stay finished throughout the next episode.
                collector.reset(0, seed=4)
                collector.collect(5)
                self.assertEqual([(r['world'], r['step']) for r in records], [(0, 0), (0, 2)])

    def test_unsampled_scalar_never_serializes_memory_and_evaluation_stays_full(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('self_genesis.observation._state', side_effect=AssertionError), patch(
                    'self_genesis.rollout.entity_memory_record', side_effect=AssertionError):
                self.run_records(self.config(), Path(directory) / 'training.jsonl')
            with RunRecorder(Path(directory) / 'evaluation.jsonl') as recorder:
                config = self.config()
                collector = RolloutCollector(config, RecurrentPolicy(config.appearance_dim),
                                             recorder=recorder)
                collector.collect(3)
            records = [json.loads(line) for line in
                       (Path(directory) / 'evaluation.jsonl').read_text().splitlines()]
            self.assertTrue(records[0]['trace_complete'])
            self.assertEqual(len([r for r in records if r['type'] == 'step']), 3)

    def test_metric_denominators_and_pre_update_gradient_norm(self):
        config = self.config(max_message_length=0)
        network = RecurrentPolicy(config.appearance_dim, max_message_length=0)
        collector = RolloutCollector(config, network)
        rollout = collector.collect(3)
        loss = survival_loss_components(rollout)
        loss.loss.backward()
        metrics = rollout_metrics(rollout, network)
        decisions = [d for items in rollout.experiences for e in items
                     for d in e.decisions if isinstance(d.choice, Action)]
        self.assertAlmostEqual(metrics['value_error'],
                               loss.value_loss.item() * 2 / len(decisions), places=5)
        self.assertAlmostEqual(metrics['mean_action_entropy'],
                               loss.action_entropy.item() * 2 / len(decisions), places=5)
        self.assertEqual(metrics['give_rate'],
                         sum(d.choice == Action.GIVE for d in decisions) / len(decisions))
        self.assertIsNone(metrics['mean_message_entropy'])
        expected = math.sqrt(sum(p.grad.double().square().sum().item()
                                 for p in network.parameters() if p.grad is not None))
        self.assertAlmostEqual(metrics['gradient_norm'], expected, places=4)

    def check_measurements(self, device):
        for batched in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                _, rows = self.run_records(self.config(device=device, batched=batched),
                                           Path(directory) / 'run.jsonl')
                if batched:
                    metadata = rows[0]['measurement_metadata']
                    measurements = [r['measurement'] for r in rows if r['type'] == 'batch_training']
                else:
                    measurements = [r for r in rows if r['type'] == 'measurement']
                    metadata = measurements[0]['metadata']
                self.assertEqual(metadata['cuda_synchronized'], device == 'cuda')
                for measurement in measurements:
                    self.assertGreater(measurement['elapsed_seconds'], 0)
                    self.assertAlmostEqual(measurement['world_steps_per_second'] *
                                           measurement['elapsed_seconds'], 9 if batched else 3)
                    if device == 'cuda':
                        self.assertGreater(measurement['cuda_peak_allocated_bytes'], 0)
                        self.assertTrue(metadata['gpu_name'])
                    else:
                        self.assertIsNone(measurement['cuda_peak_allocated_bytes'])
                        self.assertIsNone(metadata['gpu_name'])

    def test_cpu_measurements(self):
        self.check_measurements('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_measurements(self):
        self.check_measurements('cuda')
