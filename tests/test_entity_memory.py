from dataclasses import replace
import unittest

import torch
from torch import nn

from self_genesis.config import ExperimentConfig
from self_genesis.encounter import Observation
from self_genesis.policy import AgentPolicy, PolicyState, RecurrentPolicy


class EntityMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.network = RecurrentPolicy(3, entity_memory_dim=5)
        self.a = Observation(5, 3, 4, 2, torch.ones(3), True, (), None)
        self.b = replace(self.a, partner_appearance=torch.zeros(3))

    def test_retrieval_survives_other_partners_and_collisions_share_entry(self):
        initial = self.network.initial_state()
        self.assertEqual(self.network.retrieve_entity(self.a.partner_appearance, initial).count_nonzero(), 0)
        _, first = self.network(self.a, initial, communicating=True)
        saved = first.entities[0].value.clone()
        _, second = self.network(self.b, first, communicating=False)
        torch.testing.assert_close(self.network.retrieve_entity(self.a.partner_appearance, second), saved)
        self.assertEqual(len(second.entities), 2)
        # A distinct observation with an identical Appearance has the same key.
        collision = replace(self.a, partner_appearance=self.a.partner_appearance.clone(), partner_life=2)
        _, third = self.network(collision, second, communicating=False)
        self.assertEqual(len(third.entities), 2)
        self.assertFalse(torch.equal(third.entities[0].value, saved))
        torch.testing.assert_close(first.entities[0].value, saved)
        self.assertEqual(initial.entities, ())
        self.a.partner_appearance.zero_()
        torch.testing.assert_close(first.entities[0].appearance, torch.ones(3))

    def test_retrieved_values_drive_thought_actions_communication_and_gradients(self):
        for communicating in (True, False):
            _, written = self.network(self.a, self.network.initial_state(), communicating=False)
            _, written = self.network(self.a, written, communicating=True)
            value = written.entities[0].value
            value.retain_grad()
            # Remove Working Memory/affect history to isolate the Entity Memory path.
            state = PolicyState(torch.zeros_like(written.memory), torch.zeros_like(written.affect), written.entities)
            thoughts = []
            hook = self.network.thought.register_forward_hook(lambda module, args, output: thoughts.append(output))
            logits, _ = self.network(self.a, state, communicating=communicating)
            missing, _ = self.network(self.a, replace(state, entities=()), communicating=communicating)
            hook.remove()
            self.assertFalse(torch.equal(thoughts[0], thoughts[1]))
            self.assertFalse(torch.equal(logits, missing))
            self.network.zero_grad()
            # A policy-gradient term, with a scalar survival return and no latent labels.
            (-torch.log_softmax(logits, dim=0)[0] * 3).backward()
            self.assertGreater(value.grad.abs().sum().item(), 0)
            for parameter in self.network.entity_update.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_agent_ownership_detach_reset_and_device(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            network = self.network.to(device)
            first, second = AgentPolicy(network), AgentPolicy(network)
            first.act(self.a)
            self.assertEqual(second.state.entities, ())
            second.act(self.a)
            self.assertNotEqual(first.state.entities[0].value.data_ptr(), second.state.entities[0].value.data_ptr())
            saved = first.state.entities[0].value.clone()
            first.clear_decisions()
            self.assertIsNotNone(first.state.entities[0].value.grad_fn)
            first.detach()
            self.assertIsNone(first.state.entities[0].value.grad_fn)
            torch.testing.assert_close(first.state.entities[0].value, saved)
            self.assertEqual(first.state.entities[0].value.device.type, device)
            first.act(self.b)
            self.assertEqual(len(second.state.entities), 1)
            first.reset()
            self.assertEqual(first.state.entities, ())
            self.assertEqual(first.state.memory.count_nonzero(), 0)
            self.assertEqual(first.state.affect.count_nonzero(), 0)

    def test_disabled_mode_matches_v005_parameters_and_recurrence(self):
        # Original v0.0.5 layer order and dimensions also fix RNG consumption.
        torch.manual_seed(29)
        legacy = nn.Module()
        legacy.thought = nn.Linear(9 + 3 + 3 * 5 + 16 + 4, 16)
        legacy.memory_update = nn.GRUCell(16 + 4, 16)
        legacy.affect_update = nn.Linear(9 + 3 + 3 * 5 + 32, 4)
        legacy.message_head = nn.Linear(16, 4)
        legacy.action_head = nn.Linear(16, 2)
        legacy.value_head = nn.Linear(16, 1)
        rng = torch.get_rng_state()
        torch.manual_seed(29)
        network = RecurrentPolicy(3, entity_memory_dim=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(legacy.state_dict().keys(), network.state_dict().keys())
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, network.state_dict()[name]), name)
        state = network.initial_state()
        for observation, communicating in ((self.a, True), (self.b, False), (self.a, False)):
            inputs = network._encode(observation, communicating)
            thought = torch.tanh(legacy.thought(torch.cat((inputs, state.memory, state.affect))))
            memory = legacy.memory_update(torch.cat((thought, state.affect)), state.memory)
            affect = torch.tanh(legacy.affect_update(torch.cat((inputs, thought, memory))))
            expected = (legacy.message_head if communicating else legacy.action_head)(memory)
            logits, state = network(observation, state, communicating=communicating)
            self.assertTrue(torch.equal(logits, expected))
            self.assertTrue(torch.equal(state.memory, memory))
            self.assertTrue(torch.equal(state.affect, affect))
            self.assertEqual(state.entities, ())

    def test_dimension_validation(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                RecurrentPolicy(3, entity_memory_dim=value)
            with self.assertRaises(ValueError):
                ExperimentConfig(entity_memory_dim=value)
        self.assertEqual(ExperimentConfig(entity_memory_dim=0).entity_memory_dim, 0)
