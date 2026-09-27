from dataclasses import fields, replace
import random
import unittest
from unittest.mock import patch

import torch

from self_genesis.batched_encounter import BatchedEncounterProtocol
from self_genesis.batched_policy import BatchedObservation
from self_genesis.batched_world import BatchedWorld
from self_genesis.config import ExperimentConfig
from self_genesis.encounter import EncounterProtocol
from self_genesis.policy import AgentPolicy, RecurrentPolicy
from self_genesis.rollout import _RecordingPolicy
from self_genesis.world import Action, World


class BatchedEncounterTests(unittest.TestCase):
    def config(self, **kwargs):
        return ExperimentConfig(num_agents=4, appearance_dim=3, initial_life=5,
                                initial_points=1, device='cpu', **kwargs)

    def test_sequential_routing_communication_completion_and_gradient_parity(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for length in (0, 3):
                with self.subTest(device=device, length=length):
                    torch.manual_seed(8)
                    config = replace(self.config(survival_horizon=3), device=device)
                    network = RecurrentPolicy(3, max_message_length=length).to(device).double()
                    reference = RecurrentPolicy(3, max_message_length=length).to(device).double()
                    reference.load_state_dict(network.state_dict())
                    world = BatchedWorld(config, seeds=[11, 22])
                    protocol = BatchedEncounterProtocol(world, network, seeds=[31, 42])
                    state = network.initial_batch_state(2, 4, slots=4)
                    worlds = [World(replace(config, seed=seed), seed_rng=False) for seed in (11, 22)]
                    protocols = [EncounterProtocol(w, seed=0, max_message_length=length) for w in worlds]
                    agents = [[AgentPolicy(reference) for _ in range(4)] for _ in worlds]
                    loss, expected_loss = [], []
                    # Repeat partners so completion must feed later decisions.
                    draws = torch.tensor([[.1, .1] + [.2, .7, .4][:length] * 2 + [.99, .99],
                                          [.8, .5] + [.8, .1, .6][:length] * 2 + [.99, .1]],
                                         device=device, dtype=torch.float64)
                    with patch.object(protocol, '_uniforms', return_value=draws), patch.object(
                            network, 'forward', side_effect=AssertionError('per-agent forward')):
                        for step in range(3):
                            result = protocol.step(state)
                            state = result.state
                            self.assertEqual(len(result.decisions), 4)
                            for decision in result.decisions:
                                if decision.log_prob is not None:
                                    loss.append((decision.log_prob + decision.value
                                                 + decision.entropy).sum())
                            for row, (single, sequential) in enumerate(zip(worlds, protocols)):
                                pair = result.pairs[row].tolist()
                                policies = [_RecordingPolicy(a) for a in agents[row]]
                                samples = [d.choice[row, pair[i % 2]] for i, d in enumerate(result.decisions)
                                           if not d.communicating or length]
                                with patch.object(sequential._random, 'sample', return_value=pair), patch(
                                        'self_genesis.policy.Categorical.sample', side_effect=samples):
                                    expected = sequential.step(policies)
                                torch.testing.assert_close(world.state.life[row], single.state.life)
                                torch.testing.assert_close(world.state.points[row], single.state.points)
                                torch.testing.assert_close(result.world.reward[row], expected.reward)
                                for phase, decision in enumerate(result.decisions):
                                    agent = pair[phase % 2]
                                    recorded = policies[agent].decisions[phase // 2]
                                    obs = decision.observation
                                    seen = recorded.observation
                                    self.assertEqual(obs.resources[row, agent].tolist(),
                                                     [seen.life, seen.points, seen.partner_life, seen.partner_points])
                                    self.assertEqual(bool(obs.first[row, agent]), seen.first)
                                    tokens = tuple(t for t in obs.received_message[row, agent].tolist()
                                                   if t != network.vocabulary_size)
                                    self.assertEqual(tokens, seen.received_message)
                                    self.assertEqual(int(obs.partner_action[row, agent]),
                                                     {None: 0, Action.NOTHING: 1, Action.GIVE: 2}[seen.partner_action])
                                    torch.testing.assert_close(obs.partner_appearance[row, agent],
                                                               seen.partner_appearance.to(network.thought.weight))
                                    if recorded.log_prob is not None:
                                        for actual, expected_stat in ((decision.log_prob, recorded.log_prob),
                                                                      (decision.value, recorded.value),
                                                                      (decision.entropy, recorded.entropy)):
                                            torch.testing.assert_close(actual[row, agent], expected_stat)
                                        expected_loss.append(recorded.log_prob + recorded.value + recorded.entropy)
                                    else:
                                        self.assertIsNone(decision.log_prob)
                                for agent, policy in enumerate(agents[row]):
                                    torch.testing.assert_close(state.memory[row, agent], policy.state.memory)
                                    torch.testing.assert_close(state.affect[row, agent], policy.state.affect)
                                    for partner in range(4):
                                        key = world.state.appearance[:, partner:partner + 1].expand(-1, 4, -1).to(
                                            network.thought.weight)
                                        torch.testing.assert_close(state.entities.retrieve(key)[row, agent],
                                                                   reference.retrieve_entity(key[row, agent], policy.state))
                                    policy.clear_decisions()
                    sum(loss).backward()
                    sum(expected_loss).backward()
                    for (name, parameter), (_, expected) in zip(network.named_parameters(), reference.named_parameters()):
                        if expected.grad is None:
                            self.assertIsNone(parameter.grad, name)
                        else:
                            torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-7, atol=1e-9)
                    self.assertGreater(network.encounter_update.weight_ih.grad.abs().sum(), 0)
                    self.assertEqual({f.name for f in fields(BatchedObservation)},
                                     {'resources', 'partner_appearance', 'first', 'received_message', 'partner_action'})

    def test_masks_no_callbacks_and_participant_visible_completion_after_death(self):
        network = RecurrentPolicy(3)
        world = BatchedWorld(self.config(point_generation_probability_min=1,
                                        point_generation_probability_max=1), seeds=[1, 2, 3, 4])
        world.state.life[:] = torch.tensor([[0, 1, 4, 0], [0, 0, 4, 0], [0, 0, 0, 0], [4, 4, 4, 4]])
        world.state.points.zero_()
        world.state.points[0, 1] = 1
        world.horizon_completed[3] = True
        protocol = BatchedEncounterProtocol(world, network, seeds=[1, 2, 3, 4])
        state = network.initial_batch_state(4, 4, slots=4)
        with patch.object(network, 'forward', side_effect=AssertionError('scalar callback')), patch.object(
                network, 'complete_encounter', side_effect=AssertionError('scalar completion')), patch.object(
                network, 'complete_encounter_batch', wraps=network.complete_encounter_batch) as complete:
            # Force both participants to GIVE: second cannot afford it before regeneration.
            with torch.no_grad():
                network.action_head.weight.zero_()
                network.action_head.bias.copy_(torch.tensor([-100., 100.]))
            result = protocol.step(state)
        self.assertEqual(set(result.pairs[0].tolist()), {1, 2})
        self.assertEqual(result.pairs[1:].tolist(), [[-1, -1]] * 3)
        self.assertEqual(result.world.successful_transfers[0].tolist(), [False, True, False, False])
        self.assertTrue(result.world.died[0, 1])
        self.assertEqual(world.state.life[0].tolist(), [0, 0, 4, 0])
        args, outcomes = complete.call_args
        self.assertEqual(args[0].resources[0, 1].tolist(), [1, 1, 4, 0])
        self.assertEqual(args[0].partner_action[0, 1:3].tolist(), [2, 2])
        self.assertEqual(outcomes['received'][0, 1:3].tolist(), [False, True])
        self.assertEqual(outcomes['active'].sum(), 2)
        for tensor in (result.state.memory, result.state.affect, result.state.entities.values):
            self.assertEqual(tensor[1:].count_nonzero(), 0)
            self.assertEqual(tensor[0, [0, 3]].count_nonzero(), 0)
        with patch.object(network, 'forward_batch', side_effect=AssertionError('no encounter')), patch.object(
                network, 'complete_encounter_batch', side_effect=AssertionError('no completion')):
            next_result = protocol.step(result.state)
        self.assertEqual(next_result.decisions, ())
        self.assertIs(next_result.state, result.state)
        self.assertEqual(world.steps.tolist(), [2, 2, 1, 0])

    def test_selection_is_uniform_ordered_living_and_streams_are_independent(self):
        living = torch.tensor([[True, False, True, True]]).expand(6000, -1)
        draws = torch.rand(6000, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        pairs = BatchedEncounterProtocol._select(living, draws)
        for a, b in ((0, 2), (0, 3), (2, 0), (2, 3), (3, 0), (3, 2)):
            count = ((pairs[:, 0] == a) & (pairs[:, 1] == b)).sum()
            self.assertTrue(850 < count < 1150, (a, b, count))
        network = RecurrentPolicy(3)
        config = self.config()
        world = BatchedWorld(config, seeds=[8, 9])
        single = BatchedWorld(config, seeds=[9])
        batch = BatchedEncounterProtocol(world, network, seeds=[18, 19])
        reference = BatchedEncounterProtocol(single, network, seeds=[19])
        state = network.initial_batch_state(2, 4, slots=4)
        other = network.initial_batch_state(1, 4, slots=4)
        torch_rng, python_rng = torch.get_rng_state(), random.getstate()
        with torch.no_grad():
            for step in range(3):
                if step == 1:
                    world.reset(0, seed=98)
                    state = state.reset(torch.tensor([True, False]))
                if step == 2:
                    world.horizon_completed[0] = True
                result, expected = batch.step(state), reference.step(other)
                state, other = result.state, expected.state
                torch.testing.assert_close(result.pairs[1], expected.pairs[0])
                for actual, wanted in zip(result.decisions, expected.decisions):
                    torch.testing.assert_close(actual.choice[1], wanted.choice[0])
                torch.testing.assert_close(world.state.life[1], single.state.life[0])
                torch.testing.assert_close(state.memory[1], other.memory[0])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        with self.assertRaises(ValueError):
            BatchedEncounterProtocol(world, network, seeds=[1])
        with self.assertRaises(ValueError):
            BatchedEncounterProtocol(world, RecurrentPolicy(4), seeds=[1, 2])


if __name__ == '__main__':
    unittest.main()
