from dataclasses import fields, replace
import unittest
from unittest.mock import patch

import torch

from self_genesis.comparison import _AppearanceShuffleProtocol, evaluate_policy
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterExperience, Observation
from self_genesis.policy import AgentPolicy, RecurrentPolicy
from self_genesis.rollout import Rollout, RolloutCollector
from self_genesis.training import survival_policy_loss, train_episode
from self_genesis.world import Action, World


class EncounterMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.config = ExperimentConfig(num_agents=2, appearance_dim=3, initial_life=4,
                                       initial_points=0, survival_horizon=3, device='cpu')
        self.network = RecurrentPolicy(3)

    def test_completion_is_local_resolved_and_uses_observed_appearance(self):
        world = World(replace(self.config, num_agents=3,
                              point_generation_probability_min=1,
                              point_generation_probability_max=1))
        world.state.points[:] = torch.tensor([1, 0, 0])
        events = []

        class Participant:
            def communicate(self, observation):
                return (2,)

            def act(self, observation):
                events.append(observation)
                return Action.GIVE

            def complete_encounter(self, experience):
                # Exactly one world step, including renewal, has already resolved.
                self_test.assertEqual(world.state.points.tolist(), [1, 1, 1])
                events.append(experience)

        self_test = self
        protocol = _AppearanceShuffleProtocol(world, seed=42)
        with patch.object(protocol._random, 'sample', return_value=[0, 1]):
            result = protocol.step([Participant() for _ in range(3)])
        self.assertEqual(len(events), 4)
        self.assertIsNone(events[0].partner_action)
        self.assertEqual(events[1].partner_action, Action.GIVE)
        self.assertEqual(result.successful_transfers, ((0, 1),))
        self.assertEqual(world.state.life.tolist(), [3, 4, 3])
        self.assertEqual(result.reward.tolist(), [1, 1, 1])
        for before, completed in zip(events[:2], events[2:]):
            self.assertTrue(torch.equal(before.partner_appearance,
                                        completed.observation.partner_appearance))
            self.assertEqual(completed.observation.partner_action, Action.GIVE)
            self.assertEqual(completed.observation.received_message, (2,))
            self.assertEqual(completed.observation.life, 4)
        self.assertEqual((events[2].gave, events[2].received), (True, False))
        self.assertEqual((events[3].gave, events[3].received), (False, True))
        self.assertEqual({f.name for f in fields(EncounterExperience)},
                         {'observation', 'action', 'gave', 'received'})
        self.assertIs(type(events[2].observation), Observation)

    def test_outcomes_change_only_entity_state_and_disabled_mode_is_noop(self):
        observation = Observation(4, 0, 4, 0, torch.ones(3), True, (), Action.GIVE)
        agent = AgentPolicy(self.network)
        agent.act(observation)
        before = agent.state
        experience = EncounterExperience(observation, Action.NOTHING, False, True)
        agent.complete_encounter(experience)
        self.assertIs(agent.state.memory, before.memory)
        self.assertIs(agent.state.affect, before.affect)
        self.assertEqual(len(agent.log_probs), 1)
        other = self.network.complete_encounter(replace(experience, received=False), before)
        self.assertFalse(torch.equal(agent.state.entities[0].value, other.entities[0].value))
        self.assertFalse(torch.equal(agent.state.entities[0].value, before.entities[0].value))
        disabled = RecurrentPolicy(3, entity_memory_dim=0)
        state = disabled.initial_state()
        self.assertIs(disabled.complete_encounter(experience, state), state)

    def test_collection_boundary_preserves_completion_graph_and_reset_clears_it(self):
        collector = RolloutCollector(self.config, self.network)
        first = collector.collect(1)
        values = [agent.state.entities[0].value for agent in collector.agents]
        for value in values:
            value.retain_grad()
        rest = collector.collect(2)
        for index, items in enumerate(rest.experiences):
            self.assertIs(items[0].decisions[0].state_before.entities[0].value, values[index])
        complete = Rollout(tuple(a + b for a, b in zip(first.experiences, rest.experiences)),
                           3, False, False, True)
        survival_policy_loss(complete).backward()
        for value in values:
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for parameter in self.network.encounter_update.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        saved = collector.agents[0].state.entities[0].value
        collector.detach()
        self.assertIsNone(collector.agents[0].state.entities[0].value.grad_fn)
        torch.testing.assert_close(saved, collector.agents[0].state.entities[0].value)
        collector.reset()
        self.assertTrue(all(agent.state.entities == () for agent in collector.agents))

    def test_death_clears_state_and_lone_survivor_gets_no_completion(self):
        collector = RolloutCollector(self.config, self.network)
        collector.world.state.life[:] = torch.tensor([1, 4])
        with patch.object(self.network, 'complete_encounter',
                          wraps=self.network.complete_encounter) as complete:
            first = collector.collect(1)
            self.assertEqual(complete.call_count, 2)
            self.assertTrue(first.experiences[0][0].terminated)
            self.assertEqual(collector.agents[0].state.entities, ())
            survivor = collector.agents[1].state
            collector.collect(2)
            self.assertEqual(complete.call_count, 2)
            self.assertIs(collector.agents[1].state, survivor)

    def test_unselected_agent_keeps_state_and_horizon_does_not_repeat_completion(self):
        collector = RolloutCollector(replace(self.config, num_agents=3, survival_horizon=1),
                                     self.network)
        unselected = collector.agents[2].state
        with patch.object(collector.protocol._random, 'sample', return_value=[0, 1]):
            rollout = collector.collect(1)
        self.assertTrue(rollout.horizon_completed)
        self.assertIs(collector.agents[2].state, unselected)
        states = [agent.state for agent in collector.agents]
        self.assertEqual(collector.collect(1).steps, 0)
        for agent, state in zip(collector.agents, states):
            self.assertIs(agent.state, state)

    def test_training_updates_completion_weights_and_evaluation_uses_fresh_state(self):
        collector = RolloutCollector(self.config, self.network)
        before = self.network.encounter_update.weight_ih.detach().clone()
        train_episode(collector, torch.optim.Adam(self.network.parameters(), lr=0.001))
        self.assertFalse(torch.equal(before, self.network.encounter_update.weight_ih))
        weights = {name: value.clone() for name, value in self.network.state_dict().items()}
        original = self.network.complete_encounter
        calls = []

        def capture(experience, state):
            updated = original(experience, state)
            self.assertFalse(updated.entities[0].value.requires_grad)
            key = experience.observation.partner_appearance
            self.assertTrue(any(torch.equal(entry.appearance, key) for entry in updated.entities))
            calls.append(updated)
            return updated

        for intervention in (None, 'appearance-shuffle', 'working-memory-reset'):
            with patch.object(self.network, 'complete_encounter', side_effect=capture):
                first = evaluate_policy(self.config, self.network, intervention=intervention)
                second = evaluate_policy(self.config, self.network, intervention=intervention)
            self.assertEqual(first, second)
        self.assertEqual(len(calls), 36)
        for name, value in self.network.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]))
