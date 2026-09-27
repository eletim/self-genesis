"""Ensure throughput counts encounters rather than agent decisions or steps."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from self_genesis.world import Action

spec = importlib.util.spec_from_file_location(
    'benchmark_rtx5090', Path(__file__).resolve().parents[1] / 'examples/benchmark_rtx5090.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class EncounterCountTests(unittest.TestCase):
    def test_no_encounters(self):
        self.assertEqual(benchmark.encounter_count(NS(experiences=[NS(decisions=[])]), True), 0)

    def test_scalar_messages_and_lone_survivor_do_not_count(self):
        rollout = NS(experiences=[
            [NS(decisions=[NS(choice=(1, 2)), NS(choice=Action.GIVE)]), NS(decisions=[])],
            [NS(decisions=[NS(choice=()), NS(choice=Action.NOTHING)])]])
        self.assertEqual(benchmark.encounter_count(rollout, False), 1)

    def test_batch_ignores_inactive_worlds_and_message_phases(self):
        decisions = [NS(communicating=communicating, active=torch.tensor(mask))
                     for communicating, mask in [
                         (True, [[True, False], [False, True], [False, False]]),
                         (False, [[True, False], [False, True], [False, False]]),
                         (False, [[False, True], [True, False], [False, False]])]]
        rollout = NS(experiences=[NS(decisions=decisions), NS(decisions=[])])
        self.assertEqual(benchmark.encounter_count(rollout, True), 2)


class SweepTests(unittest.TestCase):
    def test_density_changes_only_worlds_and_pairs(self):
        args = NS(sweep='density', capacity='medium', precision='fp32',
                  world_counts=[64, 128, 256], seed=42, horizon=16, warmup=1, updates=20)
        cases = benchmark.sweep_cases(args)
        self.assertEqual(len(cases), 12)
        self.assertEqual({case[4] for case in cases}, {1, 4, 8, 16})
        configs = []
        for capacity, precision, worlds, agents, pairs in cases:
            config = benchmark.benchmark_config(NS(**vars(args), worlds=worlds,
                                                  num_agents=agents, encounter_count=pairs))
            self.assertEqual(config.num_agents, 32)
            self.assertTrue(config.batched)
            self.assertEqual(config.encounter_pairs(32), pairs)
            values = benchmark.asdict(config)
            del values['num_worlds'], values['encounter_count']
            configs.append(values)
        self.assertTrue(all(config == configs[0] for config in configs))
        self.assertEqual(configs[0]['memory_dim'], 128)
        self.assertEqual(configs[0]['learning_rate'], 0.001)
        self.assertEqual(configs[0]['point_generation_probability_min'], 0.1)

    def test_original_scaling_matrix_is_preserved(self):
        cases = benchmark.sweep_cases(NS(sweep='scaling'))
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(set(cases)), 33)
        self.assertTrue(all(agents == 4 and pairs == 1
                            for _, _, _, agents, pairs in cases))
        self.assertEqual(sum(worlds == 0 for _, _, worlds, _, _ in cases), 3)


if __name__ == '__main__':
    unittest.main()
