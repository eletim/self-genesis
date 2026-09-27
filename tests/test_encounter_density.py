from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from self_genesis.batched_encounter import BatchedEncounterProtocol
from self_genesis.batched_world import BatchedWorld
from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.comparison import evaluate_policy
from self_genesis.config import CAPACITY_PRESETS, ExperimentConfig, load_config
from self_genesis.encounter import EncounterProtocol
from self_genesis.policy import RecurrentPolicy
from self_genesis.world import Action, World
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode, train_batch


class CountingPolicy:
    def __init__(self):
        self.observations = []
        self.completions = []

    def communicate(self, observation):
        return ()

    def act(self, observation):
        self.observations.append(observation)
        return Action.GIVE

    def complete_encounter(self, experience):
        self.completions.append(experience)


class EncounterDensityTests(unittest.TestCase):
    def test_validation_resolution_and_serialization(self):
        self.assertEqual(ExperimentConfig().encounter_count, 1)
        for count in (0, 1, 4, 8, 16, 10**30):
            config = ExperimentConfig(encounter_count=count)
            for living in (0, 1, 3, 7, 32):
                self.assertEqual(config.encounter_pairs(living), min(count, living // 2))
            self.assertEqual(ExperimentConfig(**asdict(config)), config)
        for fraction in (0, 0.1, 0.5, 1):
            config = ExperimentConfig(encounter_fraction=fraction)
            self.assertIsNone(config.encounter_count)
            for living in (0, 1, 3, 7, 32):
                self.assertEqual(config.encounter_pairs(living), int(fraction * living / 2))
            self.assertEqual(ExperimentConfig(**asdict(config)), config)
        for values in (
            {'encounter_count': -1}, {'encounter_count': 1.5},
            {'encounter_count': True}, {'encounter_count': '1'},
            *({'encounter_fraction': value} for value in
              (-0.1, 1.1, float('nan'), float('inf'), -float('inf'), True, '0.5')),
            {'encounter_count': 0, 'encounter_fraction': 0},
            {'encounter_count': 1, 'encounter_fraction': 0.5},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ExperimentConfig(**values)

    def test_toml_overrides_conflicts_and_preserved_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.toml'
            path.write_text('encounter_count = 4\n')
            self.assertEqual(load_config(path, encounter_count=8).encounter_count, 8)
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                load_config(path, encounter_fraction=0.5)
            path.write_text('encounter_fraction = 0.5\n')
            self.assertEqual(load_config(path).encounter_pairs(7), 1)
            with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                load_config(path, encounter_count=1)
        for preset, dimensions in CAPACITY_PRESETS.items():
            for count in (1, 4, 8, 16):
                config = load_config(capacity_preset=preset, num_agents=32, encounter_count=count)
                self.assertEqual((config.memory_dim, config.affect_dim, config.entity_memory_dim),
                                 dimensions)
                self.assertEqual((config.value_loss_coefficient, config.action_entropy_coefficient,
                                  config.message_entropy_coefficient), (0.5, 0.01, 0.01))

    def test_both_executors_use_current_population_and_advance_once(self):
        for density in ({'encounter_count': 0}, {'encounter_count': 1},
                        {'encounter_count': 4}, {'encounter_count': 8},
                        {'encounter_count': 16}, {'encounter_fraction': 0.1},
                        {'encounter_fraction': 0.5}, {'encounter_fraction': 1}):
            config = ExperimentConfig(num_agents=32, initial_life=4, initial_points=3,
                                      max_message_length=0, **density)
            for living in (32, 7, 3, 1, 0):
                with self.subTest(density=density, living=living):
                    world = World(config)
                    world.state.life[living:] = 0
                    policies = [CountingPolicy() for _ in range(32)]
                    result = EncounterProtocol(world, seed=42).step(policies)
                    count = config.encounter_pairs(living)
                    self.assertEqual(sum(len(p.observations) for p in policies), 2 * count)
                    self.assertTrue(all(len(p.observations) <= 1 for p in policies))
                    self.assertEqual(sum(len(p.completions) for p in policies), 2 * count)
                    self.assertEqual(len(result.successful_transfers), 2 * count)
                    self.assertEqual(int(result.reward.sum()), living)
                    for policy in policies:
                        for observation in policy.observations:
                            self.assertEqual((observation.life, observation.partner_life), (4, 4))
                    self.assertEqual(int(world.state.life.sum()), 3 * living + 2 * count)

                    network = RecurrentPolicy(config.appearance_dim, max_message_length=0)
                    batch = BatchedWorld(config, seeds=[42, 43])
                    batch.state.life[:, living:] = 0
                    protocol = BatchedEncounterProtocol(batch, network, seeds=[42, 43])
                    state = network.initial_batch_state(2, 32, slots=32)
                    actual = protocol.step(state)
                    for row in range(2):
                        selected = actual.pairs[row][actual.pairs[row] >= 0].tolist()
                        self.assertEqual(len(selected), 2 * count)
                        self.assertEqual(len(set(selected)), 2 * count)
                        self.assertTrue(all(agent < living for agent in selected))
                    actions = [d for d in actual.decisions if not d.communicating]
                    self.assertEqual(sum(int(d.active.sum()) for d in actions), 4 * count)
                    self.assertEqual(actual.world.reward.sum(dim=1).tolist(), [living] * 2)
                    self.assertEqual(batch.steps.tolist(), [1, 1])
                    self.assertEqual(batch.state.life.sum(dim=1).tolist(),
                                     (3 * living + actual.world.successful_transfers.sum(dim=1)).tolist())
                    if count == 0:
                        self.assertEqual(actual.decisions, ())
                        torch.testing.assert_close(actual.state.memory, state.memory)

    def test_comparison_tracks_each_partner_at_all_requested_densities(self):
        for count in (1, 4, 8, 16):
            config = ExperimentConfig(num_agents=32, encounter_count=count,
                                      initial_life=3, survival_horizon=2)
            first = evaluate_policy(config, Action.GIVE)
            self.assertEqual(first, evaluate_policy(config, Action.GIVE))
            rows = first['relationship_actions']
            self.assertEqual(len(rows), 4 * count)
            for step in (0, 1):
                actions = {row['agent']: row for row in rows if row['step'] == step}
                self.assertEqual(len(actions), 2 * count)
                for agent, row in actions.items():
                    self.assertEqual(actions[row['partner']]['partner'], agent)

    def test_zero_density_leaves_parameters_and_optimizer_untouched(self):
        config = ExperimentConfig(num_agents=4, encounter_count=0, initial_life=2)
        for batched in (False, True):
            network = RecurrentPolicy(config.appearance_dim)
            before = {key: value.clone() for key, value in network.state_dict().items()}
            optimizer = torch.optim.AdamW(network.parameters(), weight_decay=0.1)
            if batched:
                collector = BatchedRolloutCollector(config, network, seeds=[1, 2])
                result = train_batch(collector, optimizer, seeds=[1, 2])
            else:
                collector = RolloutCollector(config, network)
                result = train_episode(collector, optimizer)
            self.assertEqual(result.loss, 0)
            self.assertEqual(result.metrics['gradient_norm'], 0)
            self.assertEqual(result.metrics['encounters'], 0)
            self.assertEqual(result.metrics['action_callbacks'], 0)
            self.assertEqual(result.metrics['mean_observed_survival'], 2)
            self.assertEqual(optimizer.state, {})
            for key, value in network.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_density_training(self):
        for density in ({'encounter_count': 0}, {'encounter_count': 4},
                        {'encounter_count': 8}, {'encounter_count': 16},
                        {'encounter_fraction': 0.5}):
            config = ExperimentConfig(device='cuda', num_agents=32, initial_life=2,
                                      survival_horizon=2, **density)
            network = RecurrentPolicy(config.appearance_dim).cuda()
            collector = BatchedRolloutCollector(config, network, seeds=[1, 2])
            optimizer = torch.optim.Adam(network.parameters())
            result = train_batch(collector, optimizer, seeds=[1, 2])
            self.assertEqual(result.steps, (2, 2))
            self.assertTrue(torch.isfinite(torch.tensor(result.loss)))
            self.assertEqual(result.metrics['gradient_norm'] == 0,
                             config.encounter_pairs(32) == 0)

    def test_cli_records_density_in_both_training_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            for batched in (False, True):
                for flag, value, count, fraction in (
                    ('--encounter-count', '0', 0, None),
                    ('--encounter-count', '16', 16, None),
                    ('--encounter-fraction', '0.5', None, 0.5),
                ):
                    output = Path(directory) / f'{batched}-{value}.jsonl'
                    command = [sys.executable, '-m', 'self_genesis', 'train', '--output', str(output),
                               '--num-agents', '32', '--initial-life', '1', '--initial-points', '0',
                               '--episodes', '1', '--num-worlds', '2', flag, value]
                    if batched:
                        command.append('--batched')
                    completed = subprocess.run(command, capture_output=True, text=True, check=True)
                    summary = json.loads(completed.stdout)
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    settings = [summary['config']]
                    settings.extend(row[key] for row in rows for key in ('config', 'settings') if key in row)
                    self.assertGreater(len(settings), 1)
                    for config in settings:
                        self.assertEqual(config['encounter_count'], count)
                        self.assertEqual(config['encounter_fraction'], fraction)
            for flags in (['--encounter-count', '-1'], ['--encounter-count', '1.5'],
                          ['--encounter-fraction', 'nan'], ['--encounter-fraction', '1.1'],
                          ['--encounter-count', '1', '--encounter-fraction', '0.5']):
                failed = subprocess.run([sys.executable, '-m', 'self_genesis', *flags],
                                        capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
                self.assertNotIn('Traceback', failed.stderr)
