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
            for length, count in ((0, 1), (3, 1), (0, 2), (3, 2)):
                with self.subTest(device=device, length=length, count=count):
                    torch.manual_seed(8)
                    config = replace(self.config(survival_horizon=3, encounter_count=count), device=device)
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
                                samples = [d.choice[row, pair[offset + i % 2]]
                                           for offset in range(0, len(pair), 2)
                                           for i, d in enumerate(result.decisions)
                                           if not d.communicating or length]
                                with patch.object(sequential._random, 'sample', return_value=pair), patch(
                                        'self_genesis.policy.Categorical.sample', side_effect=samples):
                                    expected = sequential.step(policies)
                                torch.testing.assert_close(world.state.life[row], single.state.life)
                                torch.testing.assert_close(world.state.points[row], single.state.points)
                                torch.testing.assert_close(result.world.reward[row], expected.reward)
                                for offset in range(0, len(pair), 2):
                                    for phase, decision in enumerate(result.decisions):
                                        agent = pair[offset + phase % 2]
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

    def test_multi_pair_phases_sampling_masks_and_completion_boundaries(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for density in ({'encounter_count': 3}, {'encounter_fraction': 1},
                            {'encounter_fraction': 0.5}):
                for length in (0, 2):
                    with self.subTest(device=device, density=density, length=length):
                        config = ExperimentConfig(
                            num_agents=7, appearance_dim=3, initial_life=4,
                            initial_points=1, survival_horizon=1, device=device,
                            point_generation_probability_min=1,
                            point_generation_probability_max=1, **density)
                        world = BatchedWorld(config, seeds=list(range(6)))
                        populations = (7, 5, 2, 1, 0, 7)
                        for row, population in enumerate(populations):
                            world.state.life[row, population:] = 0
                        world.state.life[0, 0] = 1
                        world.state.points[:, 1::2] = 0
                        world.horizon_completed[-1] = True
                        network = RecurrentPolicy(3, max_message_length=length).to(device)
                        # Uniform categorical probabilities make the expected draws exact.
                        with torch.no_grad():
                            for head in (network.message_head, network.action_head):
                                head.weight.zero_()
                                head.bias.zero_()
                        protocol = BatchedEncounterProtocol(world, network, seeds=list(range(6)))
                        state = network.initial_batch_state(6, 7, slots=7)
                        state.memory.fill_(0.25)
                        state.affect.fill_(0.5)
                        before_life = world.state.life.clone()
                        before_points = world.state.points.clone()
                        draws = [torch.tensor(
                            [[0., 0.] + [u] * length + [1 - u] * length + [u, 1 - u]],
                            device=device, dtype=torch.float64).expand(6, -1)
                            for u in (0.1, 0.6, 0.9)]
                        with patch.object(protocol, '_uniforms', side_effect=draws), patch.object(
                                network, 'forward_batch', wraps=network.forward_batch) as forward, patch.object(
                                network, 'complete_encounter_batch',
                                wraps=network.complete_encounter_batch) as complete:
                            result = protocol.step(state)
                        self.assertEqual(forward.call_count, 4)
                        self.assertEqual(complete.call_count, 1)
                        self.assertEqual(len(result.decisions), 4)
                        participants = torch.zeros_like(world.alive)
                        for row, population in enumerate(populations):
                            count = config.encounter_pairs(population) if row != 5 else 0
                            self.assertEqual(result.pairs[row, :2 * count].tolist(), list(range(2 * count)))
                            self.assertTrue((result.pairs[row, 2 * count:] == -1).all())
                            participants[row, :2 * count] = True
                            for slot in range(count):
                                for phase, decision in enumerate(result.decisions):
                                    agent = 2 * slot + phase % 2
                                    self.assertTrue(decision.active[row, agent])
                                    u = draws[slot][row, 2 + (phase % 2) * length] if phase < 2 and length else (
                                        draws[slot][row, -2 + phase % 2])
                                    if decision.communicating:
                                        self.assertEqual(decision.choice[row, agent].tolist(),
                                                         [int(u * network.vocabulary_size)] * length)
                                    else:
                                        self.assertEqual(int(decision.choice[row, agent]), int(u * 2))
                        for decision in result.decisions:
                            self.assertEqual(int(decision.active.sum()), int(participants.sum()) // 2)
                        observation = complete.call_args.args[0]
                        outcomes = complete.call_args.kwargs
                        torch.testing.assert_close(outcomes['active'], participants)
                        torch.testing.assert_close(observation.resources[..., 0], before_life)
                        torch.testing.assert_close(observation.resources[..., 1], before_points)
                        self.assertTrue(result.world.died[0, 0])
                        self.assertTrue(outcomes['active'][0, 0])
                        # The first pair's second agent chose GIVE but had no point
                        # before regeneration; completion must report failure.
                        self.assertEqual(int(outcomes['action'][0, 1]), 1)
                        self.assertFalse(outcomes['gave'][0, 1])
                        self.assertFalse(outcomes['received'][0, 0])
                        for actual, previous in ((result.state.memory, state.memory),
                                                 (result.state.affect, state.affect),
                                                 (result.state.entities.values, state.entities.values)):
                            torch.testing.assert_close(actual[~participants], previous[~participants])
                        self.assertTrue(world.done.all())
                        self.assertEqual(world.steps.tolist(), [1, 1, 1, 1, 1, 0])
                        with patch.object(network, 'forward_batch', side_effect=AssertionError('completed')):
                            frozen = protocol.step(result.state)
                        self.assertEqual(frozen.decisions, ())
                        self.assertIs(frozen.state, result.state)
                        self.assertTrue((frozen.pairs == -1).all())

    def test_selection_is_uniform_ordered_living_and_streams_are_independent(self):
        torch.manual_seed(8)
        living = torch.tensor([[True, False, True, True]]).expand(6000, -1)
        draws = torch.rand(6000, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        pairs = BatchedEncounterProtocol._select(living, draws)
        for a, b in ((0, 2), (0, 3), (2, 0), (2, 3), (3, 0), (3, 2)):
            count = ((pairs[:, 0] == a) & (pairs[:, 1] == b)).sum()
            self.assertTrue(850 < count < 1150, (a, b, count))
        network = RecurrentPolicy(3)
        config = self.config(encounter_count=2)
        world = BatchedWorld(config, seeds=[8, 9])
        single = BatchedWorld(config, seeds=[9])
        batch = BatchedEncounterProtocol(world, network, seeds=[18, 19])
        reference = BatchedEncounterProtocol(single, network, seeds=[19])
        # Equal keys across worlds must still address private observer tables.
        world.state.appearance[0].copy_(world.state.appearance[1])
        world.state.life[0, 2:] = 0
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
                for actual, expected in ((state.memory, other.memory), (state.affect, other.affect),
                                         (state.entities.values, other.entities.values)):
                    torch.testing.assert_close(actual[1], expected[0], atol=2e-6, rtol=2e-5)
                self.assertTrue(torch.equal(state.entities.keys[1], other.entities.keys[0]))
                self.assertTrue(torch.equal(state.entities.occupied[1], other.entities.occupied[0]))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        with self.assertRaises(ValueError):
            BatchedEncounterProtocol(world, network, seeds=[1])
        with self.assertRaises(ValueError):
            BatchedEncounterProtocol(world, RecurrentPolicy(4), seeds=[1, 2])


if __name__ == '__main__':
    unittest.main()
