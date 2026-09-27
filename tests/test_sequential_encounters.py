"""Multi-pair regressions for docs/design-principles.md's shared world clock."""

from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterExperience, EncounterProtocol
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.world import Action, World


class RecordingPolicy:
    def __init__(self, index, events, action=Action.GIVE):
        self.index = index
        self.events = events
        self.action = action
        self.message = (index,)

    def communicate(self, observation):
        self.events.append((self.index, 'message', observation))
        return self.message

    def act(self, observation):
        self.events.append((self.index, 'action', observation))
        return self.action

    def complete_encounter(self, experience):
        self.events.append((self.index, 'complete', experience))


class SequentialEncounterTests(unittest.TestCase):
    def make_world(self):
        world = World(ExperimentConfig(
            num_agents=5, encounter_count=2,
            point_generation_probability_min=1,
            point_generation_probability_max=1))
        world.state.life[:] = torch.tensor([1, 1, 2, 1, 3])
        world.state.points[:] = torch.tensor([1, 1, 0, 1, 0])
        protocol = EncounterProtocol(world, seed=17, vocabulary_size=5)
        events = []
        policies = [RecordingPolicy(i, events) for i in range(5)]
        policies[3].action = Action.NOTHING
        return world, protocol, events, policies

    def test_pair_order_preserves_resources_and_local_completion(self):
        # Mutual GIVE saves both Life-1 participants. The other pair has an
        # exhausted donor and a dying non-donor; agent 4 never participates.
        for selected in ([0, 1, 2, 3], [2, 3, 0, 1]):
            with self.subTest(selected=selected):
                world, protocol, events, policies = self.make_world()
                life, points = world.state.life.clone(), world.state.points.clone()
                original_step = world.step

                def resolve(decisions):
                    self.assertEqual(len(events), 8)
                    torch.testing.assert_close(world.state.life, life)
                    torch.testing.assert_close(world.state.points, points)
                    result = original_step(decisions)
                    events.append((None, 'resolve', result))
                    return result

                with patch.object(protocol._random, 'sample', return_value=selected), \
                        patch.object(world, 'step', side_effect=resolve) as step:
                    result = protocol.step(policies)
                step.assert_called_once()
                pairs = list(zip(selected[::2], selected[1::2]))
                self.assertEqual([(i, phase) for i, phase, _ in events],
                                 [(i, phase) for a, b in pairs
                                  for i, phase in ((a, 'message'), (b, 'message'),
                                                   (a, 'action'), (b, 'action'))]
                                 + [(None, 'resolve')]
                                 + [(i, 'complete') for i in selected])
                for offset, (first, second) in enumerate(pairs):
                    observations = events[4 * offset:4 * offset + 4]
                    self.assertEqual([o.received_message for _, _, o in observations],
                                     [(), (first,), (second,), (first,)])
                    self.assertEqual([o.partner_action for _, _, o in observations],
                                     [None, None, None, policies[first].action])
                    for agent, _, observation in observations:
                        partner = second if agent == first else first
                        self.assertEqual(observation.first, agent == first)
                        self.assertEqual((observation.life, observation.points,
                                          observation.partner_life, observation.partner_points),
                                         (int(life[agent]), int(points[agent]),
                                          int(life[partner]), int(points[partner])))
                        torch.testing.assert_close(observation.partner_appearance,
                                                   world.state.appearance[partner])
                self.assertEqual(result.successful_transfers, ((0, 1), (1, 0)))
                self.assertEqual(world.state.life.tolist(), [1, 1, 1, 0, 2])
                self.assertEqual(world.state.points.tolist(), [1, 1, 1, 1, 1])
                self.assertEqual(result.generated_points.tolist(), [1, 1, 1, 0, 1])
                self.assertEqual(result.died.tolist(), [False, False, False, True, False])
                self.assertEqual(result.reward.tolist(), [1] * 5)
                self.assertFalse(result.done)
                for agent, _, experience in events[9:]:
                    before = next(o for i, phase, o in events[:8]
                                  if i == agent and phase == 'action')
                    partner = next(b if a == agent else a for a, b in pairs
                                   if agent in (a, b))
                    self.assertEqual(experience.action, policies[agent].action)
                    self.assertEqual(experience.observation.partner_action, policies[partner].action)
                    self.assertEqual((experience.gave, experience.received),
                                     (agent in (0, 1), agent in (0, 1)))
                    # Completion retains the observed resources and Appearance,
                    # including for the participant that died this step.
                    for field in fields(before):
                        if field.name != 'partner_action':
                            actual = getattr(experience.observation, field.name)
                            expected = getattr(before, field.name)
                            if isinstance(actual, torch.Tensor):
                                torch.testing.assert_close(actual, expected)
                            else:
                                self.assertEqual(actual, expected)
                self.assertEqual({f.name for f in fields(EncounterExperience)},
                                 {'observation', 'action', 'gave', 'received'})

    def test_invalid_later_pair_does_not_resolve_or_complete_earlier_pair(self):
        for invalid in ('message', 'action'):
            with self.subTest(invalid=invalid):
                world, protocol, events, policies = self.make_world()
                if invalid == 'message':
                    policies[3].message = (5,)
                else:
                    policies[3].action = 'GIVE'
                life, points = world.state.life.clone(), world.state.points.clone()
                rng = world._generation_rng.get_state().clone()
                with patch.object(protocol._random, 'sample', return_value=[0, 1, 2, 3]), \
                        patch.object(world, 'step', wraps=world.step) as step:
                    with self.assertRaises(ValueError):
                        protocol.step(policies)
                step.assert_not_called()
                self.assertFalse(any(phase == 'complete' for _, phase, _ in events))
                torch.testing.assert_close(world.state.life, life)
                torch.testing.assert_close(world.state.points, points)
                torch.testing.assert_close(world._generation_rng.get_state(), rng)

    def test_shrinking_population_recomputes_matching_and_rewards(self):
        for density, sizes in (({'encounter_count': 9}, [2, 2, 1, 1, 0]),
                               ({'encounter_fraction': 0.8}, [2, 1, 1, 0, 0]),
                               ({'encounter_fraction': 0.5}, [1, 1, 0, 0, 0])):
            with self.subTest(density=density):
                world = World(ExperimentConfig(num_agents=5, initial_points=0, **density))
                world.state.life[:] = torch.arange(1, 6)
                protocol = EncounterProtocol(world, seed=17, vocabulary_size=5)
                events = []
                policies = [RecordingPolicy(i, events, Action.NOTHING) for i in range(5)]
                for step, pair_count in enumerate(sizes):
                    events.clear()
                    result = protocol.step(policies)
                    selected = [i for pair in protocol.last_pairs for i in pair]
                    self.assertEqual(len(protocol.last_pairs), pair_count)
                    self.assertEqual(len(set(selected)), 2 * pair_count)
                    self.assertTrue(all(i >= step for i in selected))
                    self.assertEqual(len(events), 6 * pair_count)
                    self.assertEqual(result.reward.tolist(), [int(i >= step) for i in range(5)])
                    self.assertEqual(result.died.tolist(), [i == step for i in range(5)])
                    self.assertEqual(result.done, step == 4)
                events.clear()
                self.assertEqual(protocol.step(policies).reward.tolist(), [0] * 5)
                self.assertEqual(events, [])

    def test_multi_pair_memory_survives_collection_and_clears_on_death(self):
        config = ExperimentConfig(num_agents=5, initial_points=0,
                                  encounter_count=2, survival_horizon=2)
        network = RecurrentPolicy(config.appearance_dim)
        collector = RolloutCollector(config, network)
        collector.world.state.life[:] = torch.tensor([1, 3, 3, 3, 4])
        unselected = collector.agents[4].state
        with patch.object(collector.protocol._random, 'sample',
                          side_effect=[[0, 1, 2, 3], [1, 2, 3, 4]]), \
                patch.object(network, 'complete_encounter',
                             wraps=network.complete_encounter) as complete:
            first = collector.collect(1)
            self.assertEqual(complete.call_count, 4)
            self.assertTrue(first.experiences[0][0].terminated)
            self.assertEqual(collector.agents[0].state.entities, ())
            self.assertIs(collector.agents[4].state, unselected)
            survivors = [agent.state for agent in collector.agents]
            second = collector.collect(1)
            self.assertEqual(complete.call_count, 8)
        self.assertTrue(second.horizon_completed)
        self.assertFalse(second.terminated)
        self.assertEqual(second.experiences[0], ())
        for index in range(1, 5):
            self.assertIs(second.experiences[index][0].decisions[0].state_before,
                          survivors[index])
            self.assertEqual(second.experiences[index][0].reward, 1)
        # The second pair of the first step remembers only its own partner.
        for agent, partner in ((1, 0), (2, 3), (3, 2)):
            self.assertEqual(len(survivors[agent].entities), 1)
            torch.testing.assert_close(survivors[agent].entities[0].appearance,
                                       collector.world.state.appearance[partner])
        states = [agent.state for agent in collector.agents]
        self.assertEqual(collector.collect(1).steps, 0)
        for agent, state in zip(collector.agents, states):
            self.assertIs(agent.state, state)

    def test_zero_pairs_preserve_memory_and_advance_to_horizon_once(self):
        for density in ({'encounter_count': 0}, {'encounter_fraction': 0.3}):
            with self.subTest(density=density):
                config = ExperimentConfig(num_agents=5, initial_life=4,
                                          survival_horizon=2, initial_points=0,
                                          point_generation_probability_min=1,
                                          point_generation_probability_max=1, **density)
                network = RecurrentPolicy(config.appearance_dim)
                collector = RolloutCollector(config, network)
                states = [agent.state for agent in collector.agents]
                with patch.object(network, 'complete_encounter',
                                  wraps=network.complete_encounter) as complete:
                    rollout = collector.collect(10)
                complete.assert_not_called()
                self.assertEqual((rollout.steps, collector.elapsed_steps), (2, 2))
                self.assertTrue(rollout.horizon_completed)
                self.assertFalse(rollout.terminated)
                for agent, state, experiences in zip(collector.agents, states, rollout.experiences):
                    self.assertIs(agent.state, state)
                    self.assertEqual([e.reward for e in experiences], [1, 1])
                    self.assertTrue(all(e.decisions == () and not e.terminated for e in experiences))
                self.assertEqual(collector.world.state.life.tolist(), [2] * 5)
                self.assertEqual(collector.world.state.points.tolist(), [2] * 5)
                rng = collector.world._generation_rng.get_state().clone()
                self.assertEqual(collector.collect(10).steps, 0)
                self.assertEqual(collector.world.state.life.tolist(), [2] * 5)
                self.assertEqual(collector.world.state.points.tolist(), [2] * 5)
                torch.testing.assert_close(collector.world._generation_rng.get_state(), rng)
