import unittest

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.world import Action, Decision, World


NOTHING = Decision()


def give(target):
    return Decision(Action.GIVE, target)


class WorldTests(unittest.TestCase):
    def test_decay_death_and_survival_rewards(self):
        world = World(ExperimentConfig(num_agents=3, initial_life=3))
        world.state.life[:] = torch.tensor([1, 2, 3])
        total = torch.zeros(3)
        for expected_life, expected_deaths in (
            ([0, 1, 2], [True, False, False]),
            ([0, 0, 1], [False, True, False]),
            ([0, 0, 0], [False, False, True]),
        ):
            result = world.step([NOTHING] * 3)
            total += result.reward
            self.assertEqual(world.state.life.tolist(), expected_life)
            self.assertEqual(result.died.tolist(), expected_deaths)
            self.assertEqual(result.done, not any(expected_life))
        self.assertEqual(total.tolist(), [1, 2, 3])
        self.assertEqual(world.state.points.tolist(), [3, 3, 3])
        result = world.step([NOTHING] * 3)
        self.assertEqual(result.reward.tolist(), [0, 0, 0])
        self.assertFalse(result.died.any())
        self.assertTrue(result.done)

    def test_give_spends_only_donor_points_and_has_no_social_bonus(self):
        config = ExperimentConfig(num_agents=3, initial_life=2, initial_points=1)
        world, baseline = World(config), World(config)
        result = world.step([give(1), NOTHING, NOTHING])
        no_gifts = baseline.step([NOTHING] * 3)
        self.assertEqual(world.state.life.tolist(), [1, 2, 1])
        self.assertEqual(world.state.points.tolist(), [0, 1, 1])
        self.assertTrue(torch.equal(result.reward, no_gifts.reward))
        # An exhausted donor cannot restore Life or spend another agent's Points.
        world.step([give(1), NOTHING, NOTHING])
        self.assertEqual(world.state.life.tolist(), [0, 1, 0])
        self.assertEqual(world.state.points.tolist(), [0, 1, 1])

    def test_dead_donors_and_recipients_have_no_effect(self):
        world = World(ExperimentConfig(num_agents=3, initial_life=1))
        world.step([give(1), NOTHING, NOTHING])
        self.assertEqual(world.alive.tolist(), [False, True, False])
        points = world.state.points.clone()
        result = world.step([give(1), give(2), give(1)])
        self.assertTrue(result.done)
        self.assertEqual(result.reward.tolist(), [0, 1, 0])
        self.assertEqual(world.state.life.tolist(), [0, 0, 0])
        self.assertTrue(torch.equal(world.state.points, points))

    def test_simultaneous_gifts_can_save_agents_on_their_last_life(self):
        world = World(ExperimentConfig(num_agents=2, initial_life=1, initial_points=1))
        result = world.step([give(1), give(0)])
        self.assertEqual(world.state.life.tolist(), [1, 1])
        self.assertEqual(world.state.points.tolist(), [0, 0])
        self.assertFalse(result.died.any())
        self.assertFalse(result.done)
        result = world.step([give(1), give(0)])
        self.assertTrue(result.done)
        self.assertEqual(result.died.tolist(), [True, True])

    def test_multiple_donors_restore_same_recipient(self):
        world = World(ExperimentConfig(num_agents=3, initial_life=2, initial_points=1))
        world.step([give(2), give(2), NOTHING])
        self.assertEqual(world.state.life.tolist(), [1, 1, 3])
        self.assertEqual(world.state.points.tolist(), [0, 0, 1])

    def test_invalid_decisions_do_not_partially_advance_world(self):
        world = World(ExperimentConfig(num_agents=2))
        for decisions in (
            [], [NOTHING], [NOTHING] * 3,
            [give(1), give(1)], [give(1), give(-1)],
            [give(1), give(2)], [give(1), give(None)],
            [give(1), give(True)], [give(1), give(0.0)],
            [give(1), Decision(Action.NOTHING, 0)],
            [give(1), Decision("GIVE", 0)], [give(1), None],
        ):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                world.step(decisions)
            self.assertEqual(world.state.life.tolist(), [10, 10])
            self.assertEqual(world.state.points.tolist(), [3, 3])

    def test_scarcity_eventually_ends_world_and_appearance_stays_fixed(self):
        for budget in (0, 3):
            with self.subTest(budget=budget):
                world = World(ExperimentConfig(num_agents=2, initial_life=2,
                                               initial_points=budget))
                appearance = world.state.appearance.clone()
                total = torch.zeros(2)
                for _ in range(2 + budget):
                    result = world.step([give(1), give(0)])
                    total += result.reward
                    self.assertTrue(torch.equal(world.state.appearance, appearance))
                    self.assertTrue((world.state.points >= 0).all())
                self.assertTrue(result.done)
                self.assertEqual(world.state.points.tolist(), [0, 0])
                self.assertEqual(total.tolist(), [2 + budget] * 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_cuda_steps_match_cpu(self):
        cpu = World(ExperimentConfig(num_agents=3, initial_life=2))
        cuda = World(ExperimentConfig(num_agents=3, initial_life=2, device="cuda"))
        for _ in range(5):
            decisions = [give(1), give(0), give(1)]
            cpu_result, cuda_result = cpu.step(decisions), cuda.step(decisions)
            for name in ("life", "points", "appearance"):
                actual = getattr(cuda.state, name)
                self.assertEqual(actual.device.type, "cuda")
                self.assertTrue(torch.equal(getattr(cpu.state, name), actual.cpu()))
            for name in ("reward", "died"):
                actual = getattr(cuda_result, name)
                self.assertEqual(actual.device.type, "cuda")
                self.assertTrue(torch.equal(getattr(cpu_result, name), actual.cpu()))
            self.assertEqual(cpu_result.done, cuda_result.done)


if __name__ == "__main__":
    unittest.main()
