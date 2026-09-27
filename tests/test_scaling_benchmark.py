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


if __name__ == '__main__':
    unittest.main()
