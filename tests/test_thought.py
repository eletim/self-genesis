"""Callback-local Thought recurrence, shared weights and gradient lifetime."""

import unittest

import torch

from self_genesis.batched_policy import BatchedObservation
from self_genesis.encounter import Observation
from self_genesis.policy import RecurrentPolicy


class ThoughtTests(unittest.TestCase):
    def test_fixed_context_shared_core_and_callback_boundaries(self):
        for batched in (False, True):
            for steps in (16, 32, 64):
                with self.subTest(batched=batched, steps=steps):
                    network = RecurrentPolicy(2, memory_dim=4, think_steps=steps).double()
                    # Positive, contractive recurrence makes every step and its
                    # gradient observable without depending on random ReLU gates.
                    with torch.no_grad():
                        network.thought.weight.fill_(0.01)
                        network.thought.weight[:, -4:] = torch.eye(4) * 0.5
                        network.thought.bias.fill_(0.1)
                    if batched:
                        state = network.initial_batch_state(2, 2, slots=1)
                        observation = BatchedObservation(
                            torch.ones(2, 2, 4, dtype=torch.double),
                            torch.ones(2, 2, 2, dtype=torch.double),
                            torch.ones(2, 2, dtype=torch.bool),
                            torch.full((2, 2, 3), 4, dtype=torch.long),
                            torch.zeros(2, 2, dtype=torch.long))
                        active = torch.tensor([[True, False], [True, True]])
                    else:
                        state = network.initial_state()
                        observation = Observation(1, 1, 1, 1, torch.ones(2), True, (), None)
                    calls, events, handles = [], [], []

                    def capture(module, args, output):
                        output.retain_grad()
                        calls.append((module, args[0], output))
                        events.append('thought')

                    handles.append(network.thought.register_forward_hook(capture))
                    for name in ('memory_update', 'affect_update', 'entity_update'):
                        handles.append(getattr(network, name).register_forward_hook(
                            lambda module, args, output, name=name: events.append(name)))
                    prior_state = None
                    for communicating in (True, False):
                        calls.clear()
                        events.clear()
                        if batched:
                            logits, values, state = network.forward_batch(
                                observation, state, communicating=communicating, active=active)
                        else:
                            logits, state = network(observation, state, communicating=communicating)
                            values = network.value_head(state.memory)
                        self.assertEqual(events, ['thought'] * steps +
                                         ['memory_update', 'affect_update', 'entity_update'])
                        context = calls[0][1][..., :-4]
                        self.assertEqual(calls[0][1][..., -4:].count_nonzero(), 0)
                        expected = torch.zeros_like(calls[0][2])
                        for module, inputs, output in calls:
                            self.assertIs(module, network.thought)
                            torch.testing.assert_close(inputs[..., :-4], context)
                            torch.testing.assert_close(inputs[..., -4:], expected)
                            expected = torch.relu(torch.nn.functional.linear(
                                torch.cat((context, expected), dim=-1),
                                network.thought.weight, network.thought.bias))
                            torch.testing.assert_close(torch.relu(output), expected)
                        self.assertFalse(torch.equal(calls[0][2], calls[-1][2]))
                        if prior_state is None:
                            prior_state = (state.memory, state.affect,
                                           state.entities.values if batched else state.entities[0].value)
                            for tensor in prior_state:
                                tensor.retain_grad()
                    (logits.sum() + values.sum()).backward()
                    # No detach between steps, or across the previous callback's
                    # Working Memory, Affect and Entity Memory graphs.
                    for _, _, output in calls:
                        self.assertGreater(output.grad.abs().sum().item(), 0)
                    for tensor in prior_state:
                        self.assertGreater(tensor.grad.abs().sum().item(), 0)
                    self.assertGreater(network.thought.weight.grad[:, -4:].abs().sum().item(), 0)
                    for handle in handles:
                        handle.remove()

    def test_step_count_does_not_add_parameters_and_is_validated(self):
        reference = RecurrentPolicy(2)
        self.assertEqual(reference.think_steps, 16)
        for steps in (32, 64):
            network = RecurrentPolicy(2, think_steps=steps)
            self.assertEqual(network.parameter_count, reference.parameter_count)
            network.load_state_dict(reference.state_dict(), strict=True)
        for steps in (0, 1, 15, 16.0, True, None):
            with self.assertRaisesRegex(ValueError, 'think_steps'):
                RecurrentPolicy(2, think_steps=steps)
        with self.assertRaisesRegex(ValueError, 'thought_mode'):
            RecurrentPolicy(2, thought_mode='unknown')

    def test_core_rectifies_negative_activations(self):
        network = RecurrentPolicy(2)
        with torch.no_grad():
            network.thought.weight.zero_()
            network.thought.bias.fill_(-1)
        thought, _, _ = network._recur(
            torch.ones(26), torch.zeros(16), torch.zeros(4), torch.zeros(16))
        self.assertEqual(thought.count_nonzero(), 0)

    def test_shallow_mode_preserves_single_tanh_transform(self):
        network = RecurrentPolicy(2, thought_mode='shallow')
        for shape in ((), (3,)):
            inputs = torch.randn(*shape, 26)
            memory = torch.randn(*shape, 16)
            affect = torch.randn(*shape, 4)
            retrieved = torch.randn(*shape, 16)
            expected = torch.tanh(network.thought(torch.cat(
                (inputs, memory, affect, retrieved), dim=-1)))
            thought, updated, new_affect = network._recur(inputs, memory, affect, retrieved)
            torch.testing.assert_close(thought, expected)
            torch.testing.assert_close(updated, network.memory_update(
                torch.cat((expected, affect), dim=-1), memory))
            torch.testing.assert_close(new_affect, torch.tanh(network.affect_update(
                torch.cat((inputs, expected, updated), dim=-1))))


if __name__ == '__main__':
    unittest.main()
