from dataclasses import replace
from pathlib import Path
import unittest

import torch

from self_genesis.batched_policy import BatchedObservation
from self_genesis.config import load_config
from self_genesis.encounter import EncounterExperience, Observation
from self_genesis.policy import RecurrentPolicy
from self_genesis.world import Action


class BatchedPolicyTests(unittest.TestCase):
    def observations(self, network, step=0):
        parameter = next(network.parameters())
        resources = parameter.new_tensor([[[5, 3, 4, 2], [4, 2, 5, 3]],
                                          [[7, 1, 6, 0], [6, 0, 7, 1]]])
        appearance = parameter.new_zeros(2, 2, network.appearance_dim)
        appearance[..., 0] = step % 2  # Revisit observed keys after intervening encounters.
        message = torch.full((2, 2, network.max_message_length), network.vocabulary_size,
                             dtype=torch.int64, device=parameter.device)
        if network.max_message_length:
            message[0, 1, 0] = step % network.vocabulary_size
        return BatchedObservation(
            resources, appearance,
            torch.tensor([[True, False], [False, True]], device=parameter.device),
            message, torch.tensor([[0, 1], [2, 0]], device=parameter.device))

    @staticmethod
    def sequential(observation, world, observer, vocabulary):
        message = tuple(token for token in observation.received_message[world, observer].tolist()
                        if token != vocabulary)
        return Observation(*observation.resources[world, observer].tolist(),
                           observation.partner_appearance[world, observer],
                           bool(observation.first[world, observer]), message,
                           (None, Action.NOTHING, Action.GIVE)[
                               observation.partner_action[world, observer].item()])

    def test_sequential_output_state_completion_and_gradient_parity(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            for entity_dim in (0, 5):
                torch.manual_seed(9)
                network = RecurrentPolicy(3, entity_memory_dim=entity_dim).to(device).double()
                reference = RecurrentPolicy(3, entity_memory_dim=entity_dim).to(device).double()
                reference.load_state_dict(network.state_dict())
                state = network.initial_batch_state(2, 2, slots=2)
                states = [[reference.initial_state() for _ in range(2)] for _ in range(2)]
                terms, reference_terms = [], []
                for step in range(6):
                    observation = self.observations(network, step)
                    active = torch.tensor([[True, step != 2], [step != 3, False]], device=device)
                    communicating = step % 2 == 0
                    old = state
                    logits, values, state = network.forward_batch(
                        observation, state, communicating=communicating, active=active)
                    terms.extend((logits.square().sum(), values.square().sum()))
                    for w in range(2):
                        for a in range(2):
                            if active[w, a]:
                                expected, states[w][a] = reference(
                                    self.sequential(observation, w, a, network.vocabulary_size),
                                    states[w][a], communicating=communicating)
                                value = reference.value_head(states[w][a].memory).squeeze(-1)
                                reference_terms.extend((expected.square().sum(), value.square()))
                                torch.testing.assert_close(logits[w, a], expected, rtol=1e-9, atol=1e-10)
                                torch.testing.assert_close(values[w, a], value, rtol=1e-9, atol=1e-10)
                            else:
                                self.assertEqual(logits[w, a].count_nonzero(), 0)
                                self.assertEqual(values[w, a], 0)
                                torch.testing.assert_close(state.memory[w, a], old.memory[w, a])
                            torch.testing.assert_close(state.memory[w, a], states[w][a].memory)
                            torch.testing.assert_close(state.affect[w, a], states[w][a].affect)
                    actions = torch.tensor([[1, 0], [1, 0]], device=device)
                    gave = actions.bool()
                    received = ~gave
                    before_completion = state
                    state = network.complete_encounter_batch(
                        observation, state, action=actions, gave=gave, received=received, active=active)
                    self.assertIs(state.memory, before_completion.memory)
                    self.assertIs(state.affect, before_completion.affect)
                    retrieved = state.entities.retrieve(observation.partner_appearance)
                    for w in range(2):
                        for a in range(2):
                            if active[w, a]:
                                states[w][a] = reference.complete_encounter(EncounterExperience(
                                    self.sequential(observation, w, a, network.vocabulary_size),
                                    (Action.NOTHING, Action.GIVE)[actions[w, a].item()],
                                    bool(gave[w, a]), bool(received[w, a])), states[w][a])
                            torch.testing.assert_close(retrieved[w, a], reference.retrieve_entity(
                                observation.partner_appearance[w, a], states[w][a]),
                                rtol=1e-9, atol=1e-10)
                sum(terms).backward()
                sum(reference_terms).backward()
                for (name, parameter), (_, expected) in zip(
                        network.named_parameters(), reference.named_parameters()):
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-8, atol=1e-10)

    def test_private_gradients_affect_feedback_reset_and_detach(self):
        network = RecurrentPolicy(3)
        state = network.initial_batch_state(2, 2, slots=2)
        active = torch.ones(2, 2, dtype=torch.bool)
        observation = self.observations(network)
        _, _, state = network.forward_batch(observation, state, communicating=True, active=active)
        state.memory.retain_grad()
        state.affect.retain_grad()
        state.entities.values.retain_grad()
        snapshot = state.memory.clone()
        logits, values, next_state = network.forward_batch(
            observation, state, communicating=False, active=active)
        changed, _, _ = network.forward_batch(
            observation, replace(state, affect=state.affect + 1), communicating=False, active=active)
        self.assertFalse(torch.allclose(changed, logits))
        (logits[1, 0].sum() + values[1, 0]).backward()
        for tensor in (state.memory, state.affect, state.entities.values):
            self.assertGreater(tensor.grad[1, 0].abs().sum().item(), 0)
            self.assertEqual(tensor.grad[0].count_nonzero(), 0)
            self.assertEqual(tensor.grad[1, 1].count_nonzero(), 0)
        torch.testing.assert_close(state.memory, snapshot)
        reset = next_state.reset(torch.tensor([True, False]))
        for tensor in (reset.memory, reset.affect, reset.entities.values):
            self.assertEqual(tensor[0].count_nonzero(), 0)
        self.assertFalse(reset.entities.occupied[0].any())
        torch.testing.assert_close(reset.memory[1], next_state.memory[1])
        torch.testing.assert_close(reset.affect[1], next_state.affect[1])
        torch.testing.assert_close(reset.entities.values[1], next_state.entities.values[1])
        for tensor in (reset.detach().memory, reset.detach().affect, reset.detach().entities.values):
            self.assertIsNone(tensor.grad_fn)

    def test_empty_channel_inactive_padding_and_validation(self):
        for length in (0, 3):
            network = RecurrentPolicy(3, max_message_length=length)
            state = network.initial_batch_state(2, 2, slots=1)
            observation = self.observations(network)
            inactive = torch.zeros(2, 2, dtype=torch.bool)
            padding = replace(observation, resources=torch.full((2, 2, 4), float('nan')),
                              partner_action=torch.full((2, 2), -9, dtype=torch.int64),
                              received_message=torch.full((2, 2, length), -9, dtype=torch.int64))
            for communicating in (True, False):
                logits, values, result = network.forward_batch(
                    padding, state, communicating=communicating, active=inactive)
                self.assertEqual(logits.count_nonzero(), 0)
                self.assertEqual(values.count_nonzero(), 0)
                torch.testing.assert_close(result.memory, state.memory)
                torch.testing.assert_close(result.affect, state.affect)
                self.assertFalse(result.entities.occupied.any())
                with self.assertRaises(ValueError):
                    network.forward_batch(padding, state, communicating=communicating, active=~inactive)
                logits, _, _ = network.forward_batch(
                    observation, state, communicating=communicating, active=~inactive)
                self.assertEqual(logits.shape, (2, 2, 4 if communicating else 2))
            for invalid in (replace(observation, first=observation.first.float()),
                            replace(observation, resources=observation.resources[..., :3]),
                            replace(observation, partner_action=observation.partner_action + 3)):
                with self.assertRaises(ValueError):
                    network.forward_batch(invalid, state, communicating=False, active=~inactive)
            with self.assertRaises(ValueError):
                network.forward_batch(observation, state, communicating=False, active=inactive.float())
            if length:
                malformed = observation.received_message.clone()
                malformed[0, 0, 1] = 0
                with self.assertRaisesRegex(ValueError, 'suffix'):
                    network.forward_batch(replace(observation, received_message=malformed),
                                          state, communicating=True, active=~inactive)

    def test_capacity_presets_exact_counts_and_large_backward(self):
        for name, expected in (("small", 10987), ("medium", 278823), ("large", 4211847)):
            config = load_config(Path(__file__).resolve().parents[1] / 'configs' / f'policy-{name}.toml')
            network = RecurrentPolicy.from_preset(config.appearance_dim, name)
            self.assertEqual((network.memory_dim, network.affect_dim, network.entity_memory_dim),
                             (config.memory_dim, config.affect_dim, config.entity_memory_dim))
            self.assertEqual(network.parameter_count, expected)
            self.assertEqual(network.parameter_count, sum(p.numel() for p in network.parameters()))
            state = network.initial_batch_state(2, 2, slots=2)
            observation = self.observations(network)
            active = torch.ones(2, 2, dtype=torch.bool)
            for communicating in (True, False):
                logits, values, state = network.forward_batch(
                    observation, state, communicating=communicating, active=active)
            (logits.square().sum() + values.square().sum()).backward()
            self.assertGreater(network.thought.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(network.value_head.weight.grad).all())
        with self.assertRaises(ValueError):
            RecurrentPolicy.from_preset(8, 'unknown')
        custom = RecurrentPolicy.from_preset(8, 'large', memory_dim=32, entity_memory_dim=0)
        self.assertEqual(custom.memory_dim, 32)
        self.assertEqual(custom.entity_memory_dim, 0)


if __name__ == '__main__':
    unittest.main()
