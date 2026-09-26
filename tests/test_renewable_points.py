from dataclasses import fields, replace
from pathlib import Path
import random
import unittest

import torch

from self_genesis.config import ExperimentConfig, load_config
from self_genesis.encounter import EncounterProtocol, Observation
from self_genesis.experiment import initialize
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.world import Action, Decision, World


class GivingPolicy:
    def __init__(self):
        self.observations = []

    def communicate(self, observation):
        self.observations.append(observation)
        return ()

    def act(self, observation):
        self.observations.append(observation)
        return Action.GIVE


class RenewablePointsTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(ExperimentConfig(point_generation_probability_min=1,
                                        point_generation_probability_max=1), **kwargs)

    def test_config_validation_and_preset(self):
        for name in ('point_generation_probability_min', 'point_generation_probability_max'):
            for value in (-0.1, 1.1, float('nan'), float('inf'), True, '0.5'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.config(**{name: value})
        with self.assertRaises(ValueError):
            self.config(point_generation_probability_min=0.8,
                        point_generation_probability_max=0.2)
        config = load_config(Path('configs/renewable.toml'))
        self.assertEqual((config.point_generation_probability_min,
                          config.point_generation_probability_max), (0.1, 0.3))

    def test_generation_is_next_step_only_and_outside_encounters(self):
        world = World(self.config(num_agents=3, initial_life=3, initial_points=0))
        protocol = EncounterProtocol(world, seed=1)
        policies = [GivingPolicy() for _ in range(3)]
        first = protocol.step(policies)
        self.assertEqual(world.state.life.tolist(), [2, 2, 2])
        self.assertEqual(world.state.points.tolist(), [1, 1, 1])
        self.assertEqual(first.generated_points.tolist(), [1, 1, 1])
        for policy in policies:
            for observation in policy.observations:
                self.assertEqual((observation.points, observation.partner_points), (0, 0))
            policy.observations.clear()
        second = protocol.step(policies)
        self.assertEqual(sorted(world.state.life.tolist()), [1, 2, 2])
        self.assertEqual(sorted(world.state.points.tolist()), [1, 1, 2])
        self.assertEqual(second.reward.tolist(), [1, 1, 1])
        for policy in policies:
            for observation in policy.observations:
                self.assertEqual((observation.points, observation.partner_points), (1, 1))
        self.assertEqual({field.name for field in fields(Observation)}, {
            'life', 'points', 'partner_life', 'partner_points', 'partner_appearance',
            'first', 'received_message', 'partner_action'})

    def test_death_rescue_lone_survivor_and_accounting(self):
        world = World(self.config(num_agents=3, initial_life=1, initial_points=1))
        before = world.state.points.clone()
        result = world.step([Decision(Action.GIVE, 1), Decision(), Decision()])
        self.assertEqual(result.generated_points.tolist(), [0, 1, 0])
        self.assertEqual(world.state.points.tolist(),
                         (before - torch.tensor([1, 0, 0]) + result.generated_points).tolist())
        self.assertEqual(result.died.tolist(), [True, False, True])
        world.state.life[1] = 2
        protocol = EncounterProtocol(world, seed=0)
        policies = [GivingPolicy() for _ in range(3)]
        result = protocol.step(policies)
        self.assertEqual(result.generated_points.tolist(), [0, 1, 0])
        self.assertTrue(all(not policy.observations for policy in policies))
        self.assertTrue(protocol.step(policies).done)
        self.assertEqual(protocol.step(policies).generated_points.tolist(), [0, 0, 0])

    def test_abilities_independent_fixed_and_reproducible(self):
        config = self.config(num_agents=20, initial_life=20,
                             point_generation_probability_min=0.1,
                             point_generation_probability_max=0.7)
        first, second = World(config), World(config)
        ability = first.state.point_generation_probability.clone()
        appearance = first.state.appearance.clone()
        self.assertGreater(ability.unique().numel(), 1)
        self.assertTrue(((ability >= 0.1) & (ability <= 0.7)).all())
        changed = initialize(replace(config, appearance_dim=1))
        self.assertTrue(torch.equal(ability, changed.point_generation_probability))
        self.assertTrue(torch.equal(appearance, initialize(replace(config,
            point_generation_probability_min=0)).appearance))
        self.assertFalse(torch.equal(ability, initialize(replace(config, seed=1)).point_generation_probability))
        for _ in range(10):
            a = first.step([Decision()] * 20)
            torch.rand(9)
            random.random()
            b = second.step([Decision()] * 20)
            self.assertTrue(torch.equal(a.generated_points, b.generated_points))
            self.assertTrue(((a.generated_points == 0) | (a.generated_points == 1)).all())
            self.assertTrue(torch.equal(first.state.point_generation_probability, ability))
            self.assertTrue(torch.equal(first.state.appearance, appearance))

    def test_reset_restores_resources_preserves_stream_and_replays_run(self):
        def run():
            config = self.config(initial_life=10, point_generation_probability_min=0.3,
                                 point_generation_probability_max=0.7)
            collector = RolloutCollector(config, RecurrentPolicy(config.appearance_dim))
            ability = collector.world.state.point_generation_probability.clone()
            results = []
            for _ in range(3):
                results.append(collector.world.step([Decision()] * 4).generated_points.tolist())
                stream = collector.world._generation_rng.get_state().clone()
                global_stream = torch.get_rng_state().clone()
                collector.reset()
                self.assertTrue(torch.equal(stream, collector.world._generation_rng.get_state()))
                self.assertTrue(torch.equal(global_stream, torch.get_rng_state()))
                self.assertTrue(torch.equal(ability, collector.world.state.point_generation_probability))
                self.assertEqual(collector.world.state.points.tolist(), [3] * 4)
                self.assertEqual(collector.world.state.life.tolist(), [10] * 4)
            return results
        self.assertEqual(run(), run())

    def test_zero_generation_and_self_spending(self):
        world = World(ExperimentConfig(num_agents=2, initial_life=2, initial_points=1))
        for _ in range(3):
            result = world.step([Decision(Action.GIVE, 1), Decision(Action.GIVE, 0)])
            self.assertEqual(result.generated_points.tolist(), [0, 0])
        self.assertTrue(result.done)
        renewable = World(self.config(num_agents=2))
        with self.assertRaises(ValueError):
            renewable.step([Decision(Action.GIVE, 0), Decision()])
        self.assertEqual(renewable.state.points.tolist(), [3, 3])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cpu_cuda_resource_draws_match(self):
        config = self.config(point_generation_probability_min=0.1,
                             point_generation_probability_max=0.8)
        cpu, cuda = World(config), World(replace(config, device='cuda'))
        for _ in range(10):
            a, b = cpu.step([Decision()] * 4), cuda.step([Decision()] * 4)
            self.assertTrue(torch.equal(a.generated_points, b.generated_points.cpu()))
            self.assertTrue(torch.equal(cpu.state.points, cuda.state.points.cpu()))
