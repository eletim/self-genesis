from dataclasses import fields, replace
import random
import unittest

import torch

from self_genesis.batched_world import BatchedWorld
from self_genesis.config import ExperimentConfig
from self_genesis.experiment import ExperimentState
from self_genesis.world import Action, Decision, World


class BatchedWorldTests(unittest.TestCase):
    def assert_state_equal(self, batch, row, world):
        for field in fields(ExperimentState):
            self.assertTrue(torch.equal(getattr(batch.state, field.name)[row].cpu(),
                                        getattr(world.state, field.name).cpu()), field.name)

    def check_reference(self, device):
        config = ExperimentConfig(num_agents=4, initial_life=4, initial_points=2,
                                  point_generation_probability_min=0.2,
                                  point_generation_probability_max=0.8,
                                  survival_horizon=7, device=device)
        seeds = [0, 73, 918]
        batch = BatchedWorld(config, seeds=seeds)
        references = [World(replace(config, seed=seed), seed_rng=False) for seed in seeds]
        for row, world in enumerate(references):
            self.assert_state_equal(batch, row, world)
        # Distinct death schedules exercise independent survivor-only RNG streams.
        batch.state.life[0] = torch.tensor([1, 1, 1, 1], device=device)
        batch.state.life[1] = torch.tensor([0, 1, 3, 4], device=device)
        for row, world in enumerate(references):
            world.state.life.copy_(batch.state.life[row])
        schedule = [[1, 0, 1, -1], [2, 2, -1, 2], [-1, -1, -1, -1]]
        totals = torch.zeros_like(batch.state.life, dtype=torch.float32)
        expected_totals = torch.zeros_like(totals)
        finished = [False] * len(seeds)
        for step in range(10):
            targets = torch.tensor([schedule[(step + row) % 3] for row in range(3)],
                                   device=device)
            result = batch.step(targets)
            totals += result.reward
            for row, world in enumerate(references):
                if finished[row]:
                    self.assertFalse(result.reward[row].any())
                    self.assertFalse(result.died[row].any())
                    self.assertFalse(result.generated_points[row].any())
                    self.assertFalse(result.successful_transfers[row].any())
                else:
                    decisions = [Decision() if target == -1 else Decision(Action.GIVE, target)
                                 for target in targets[row].tolist()]
                    expected = world.step(decisions)
                    expected_totals[row] += expected.reward
                    for name in ('reward', 'died', 'generated_points'):
                        self.assertTrue(torch.equal(getattr(result, name)[row],
                                                    getattr(expected, name)), name)
                    transfers = tuple((donor, int(targets[row, donor])) for donor in
                                      result.successful_transfers[row].nonzero().flatten().tolist())
                    self.assertEqual(transfers, expected.successful_transfers)
                    horizon = step + 1 == config.survival_horizon and not expected.done
                    self.assertEqual(bool(result.terminated[row]), expected.done)
                    self.assertEqual(bool(result.horizon_completed[row]), horizon)
                    finished[row] = expected.done or horizon
                    self.assertEqual(bool(result.done[row]), finished[row])
                self.assert_state_equal(batch, row, world)
        self.assertTrue(torch.equal(totals, expected_totals))

    def test_matches_sequential_reference(self):
        self.check_reference('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_matches_cpu_reference(self):
        # Check sequential CUDA behavior, then exact CPU/CUDA equivalence.
        self.check_reference('cuda')
        config = ExperimentConfig(num_agents=3, initial_life=4,
                                  point_generation_probability_min=0.1,
                                  point_generation_probability_max=0.9)
        cpu = BatchedWorld(config, seeds=[2, 7])
        cuda = BatchedWorld(replace(config, device='cuda'), seeds=[2, 7])
        for _ in range(8):
            targets = torch.tensor([[1, 2, 0], [2, 0, 1]])
            a, b = cpu.step(targets), cuda.step(targets.cuda())
            for field in fields(a):
                self.assertTrue(torch.equal(getattr(a, field.name), getattr(b, field.name).cpu()))
            for field in fields(ExperimentState):
                self.assertTrue(torch.equal(getattr(cpu.state, field.name),
                                            getattr(cuda.state, field.name).cpu()))

    def test_simultaneous_rescue_and_next_step_regeneration(self):
        config = ExperimentConfig(num_agents=3, initial_life=1, initial_points=1,
                                  point_generation_probability_min=1,
                                  point_generation_probability_max=1)
        batch = BatchedWorld(config, seeds=[1, 2])
        batch.state.points[1] = 0
        batch.state.life[1] = 2
        first = batch.step(torch.tensor([[1, 0, 1], [1, 0, -1]]))
        self.assertEqual(batch.state.life.tolist(), [[1, 2, 0], [1, 1, 1]])
        self.assertEqual(first.generated_points.tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(first.successful_transfers.tolist(), [[True, True, True], [False]*3])
        second = batch.step(torch.tensor([[2, 0, 1], [1, 0, -1]]))
        self.assertEqual(second.successful_transfers.tolist(), [[False, True, False], [True, True, False]])
        self.assertEqual(batch.state.life.tolist(), [[1, 1, 0], [1, 1, 0]])
        self.assertEqual(second.reward.tolist(), [[1, 1, 0], [1, 1, 1]])

    def test_horizon_death_precedence_freeze_and_result_snapshot(self):
        batch = BatchedWorld(ExperimentConfig(num_agents=2, initial_life=1,
                                             survival_horizon=1), seeds=[1, 2])
        result = batch.step(torch.tensor([[-1, -1], [1, 0]]))
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(result.horizon_completed.tolist(), [False, True])
        self.assertEqual(result.died.tolist(), [[True, True], [False, False]])
        state = batch.state.life.clone()
        for _ in range(3):
            frozen = batch.step(torch.tensor([[1, 0], [1, 0]]))
            self.assertFalse(frozen.reward.any())
            self.assertFalse(frozen.died.any())
            self.assertTrue(torch.equal(state, batch.state.life))
        self.assertEqual(batch.steps.tolist(), [1, 1])
        batch.reset(0, seed=42)
        self.assertEqual(result.done.tolist(), [True, True])
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(batch.done.tolist(), [False, True])

    def test_reset_and_batch_composition_do_not_change_other_streams(self):
        config = ExperimentConfig(num_agents=2, initial_life=20,
                                  point_generation_probability_min=0.5,
                                  point_generation_probability_max=0.5)
        torch_state, python_state = torch.get_rng_state(), random.getstate()
        batch = BatchedWorld(config, seeds=[3, 91])
        single = BatchedWorld(config, seeds=[91])
        permuted = BatchedWorld(config, seeds=[91, 6, 3])
        for step in range(6):
            batch.reset(0, seed=step)
            batch.step(torch.full((2, 2), -1))
            single.step(torch.full((1, 2), -1))
            permuted.step(torch.full((3, 2), -1))
            for field in fields(ExperimentState):
                self.assertTrue(torch.equal(getattr(batch.state, field.name)[1],
                                            getattr(single.state, field.name)[0]))
                self.assertTrue(torch.equal(getattr(batch.state, field.name)[1],
                                            getattr(permuted.state, field.name)[0]))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        batch.reset(1, seed=91)
        self.assert_state_equal(batch, 1, World(replace(config, seed=91), seed_rng=False))
        self.assertEqual(batch.steps.tolist(), [1, 0])

    def test_lone_survivor_decays_without_horizon(self):
        batch = BatchedWorld(ExperimentConfig(num_agents=2, initial_life=2), seeds=[4])
        batch.state.life[0, 0] = 0
        first = batch.step(torch.tensor([[1, 0]]))
        self.assertEqual(first.reward.tolist(), [[0, 1]])
        self.assertFalse(first.successful_transfers.any())
        self.assertFalse(first.done.any())
        second = batch.step(torch.tensor([[-1, -1]]))
        self.assertEqual(second.died.tolist(), [[False, True]])
        self.assertTrue(second.terminated.all())
        self.assertFalse(second.horizon_completed.any())
        self.assertEqual(batch.state.points.tolist(), [[3, 3]])

    def test_invalid_input_is_atomic_including_rng(self):
        config = ExperimentConfig(num_agents=2, point_generation_probability_max=1)
        batch, control = BatchedWorld(config, seeds=[1, 2]), BatchedWorld(config, seeds=[1, 2])
        for targets in (None, torch.zeros(2, dtype=torch.int64), torch.zeros(2, 2),
                        torch.tensor([[1, 0], [0, -1]]), torch.tensor([[1, 0], [2, -1]]),
                        torch.tensor([[1, 0], [-2, -1]])):
            with self.assertRaises(ValueError):
                batch.step(targets)
        for world, seed in ((-1, 1), (2, 1), (True, 1), (0, -1), (0, True)):
            with self.assertRaises(ValueError):
                batch.reset(world, seed=seed)
        self.assertEqual(batch.steps.tolist(), [0, 0])
        targets = torch.tensor([[1, 0], [1, 0]])
        a, b = batch.step(targets), control.step(targets)
        for field in fields(a):
            self.assertTrue(torch.equal(getattr(a, field.name), getattr(b, field.name)))
        for field in fields(ExperimentState):
            self.assertTrue(torch.equal(getattr(batch.state, field.name),
                                        getattr(control.state, field.name)))
        for seeds in ([], [-1], [True], [2**63]):
            with self.assertRaises(ValueError):
                BatchedWorld(config, seeds=seeds)


if __name__ == '__main__':
    unittest.main()
