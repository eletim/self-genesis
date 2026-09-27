from dataclasses import replace
import unittest

import torch

from self_genesis.entity_memory import BatchedEntityMemory
from self_genesis.policy import RecurrentPolicy


class BatchedEntityMemoryTests(unittest.TestCase):
    def test_matches_sequential_reference_with_collisions_and_masking(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            memory = BatchedEntityMemory.empty(2, 3, 3, 2, 4, device=device)
            network = RecurrentPolicy(2, entity_memory_dim=4).to(device)
            references = [[network.initial_state() for _ in range(3)] for _ in range(2)]
            for step in range(6):
                # Different insertion order in each observer; repeated keys collide.
                appearance = torch.tensor([
                    [[step % 2, 0], [(step + 1) % 2, 0], [0, 0]],
                    [[0, 0], [step % 3, 0], [1, 0]],
                ], dtype=torch.float32, device=device)
                active = torch.tensor([[True, step != 2, False], [step != 4, True, True]],
                                      device=device)
                actual = memory.retrieve(appearance, active=active)
                for world in range(2):
                    for observer in range(3):
                        expected = network.retrieve_entity(
                            appearance[world, observer], references[world][observer])
                        torch.testing.assert_close(actual[world, observer],
                                                   expected if active[world, observer]
                                                   else torch.zeros_like(expected))
                value = torch.arange(24., device=device).reshape(2, 3, 4) + step
                previous = memory
                snapshot = previous.values.clone()
                memory = memory.write(appearance, value, active=active)
                for world in range(2):
                    for observer in range(3):
                        if active[world, observer]:
                            state = references[world][observer]
                            references[world][observer] = replace(state, entities=network._store_entity(
                                appearance[world, observer], value[world, observer], state.entities))
                        self.assertEqual(memory.occupied[world, observer].sum().item(),
                                         len(references[world][observer].entities))
                torch.testing.assert_close(previous.values, snapshot)
            self.assertEqual(memory.values.shape, (2, 3, 3, 4))
            torch.testing.assert_close(memory.retrieve(appearance, active=active),
                                       torch.where(active[..., None], value, 0))

    def test_world_observer_separation_reset_and_gradient_flow(self):
        torch.manual_seed(8)
        memory = BatchedEntityMemory.empty(2, 2, 2, 3, 5)
        appearance = torch.ones(2, 2, 3, requires_grad=True)
        network = RecurrentPolicy(3, entity_memory_dim=5)
        inputs = torch.randn(4, network.entity_update.input_size, requires_grad=True)
        first = network.entity_update(inputs, memory.retrieve(appearance).reshape(4, 5))
        first.retain_grad()
        memory = memory.write(appearance, first.reshape(2, 2, 5))
        # Preserve the first Appearance across an intervening encounter.
        memory = memory.write(torch.zeros_like(appearance), torch.zeros(2, 2, 5))
        second = network.entity_update(inputs, memory.retrieve(appearance).reshape(4, 5))
        memory = memory.write(appearance.clone(), second.reshape(2, 2, 5))
        reset = memory.reset(torch.tensor([True, False]))
        self.assertFalse(reset.occupied[0].any())
        self.assertEqual(reset.keys[0].count_nonzero(), 0)
        self.assertEqual(reset.values[0].count_nonzero(), 0)
        self.assertTrue(memory.occupied.all())
        torch.testing.assert_close(reset.values[1], memory.values[1])
        retrieved = reset.retrieve(appearance)
        self.assertEqual(retrieved[0].count_nonzero(), 0)
        retrieved[1, 0].sum().backward()
        self.assertGreater(first.grad[2].abs().sum().item(), 0)
        self.assertEqual(first.grad[[0, 1, 3]].count_nonzero(), 0)
        self.assertGreater(inputs.grad[2].abs().sum().item(), 0)
        self.assertEqual(inputs.grad[[0, 1, 3]].count_nonzero(), 0)
        self.assertIsNone(appearance.grad)
        for parameter in network.entity_update.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(reset.detach().values.grad_fn)
        torch.testing.assert_close(reset.detach().values, reset.values)
        # A fresh episode can allocate new keys, even after the old table was full.
        fresh = reset.write(appearance + 2, torch.ones(2, 2, 5),
                            active=torch.tensor([[True, True], [False, False]]))
        self.assertTrue((fresh.occupied[0].sum(dim=-1) == 1).all())
        torch.testing.assert_close(fresh.values[1], memory.values[1])

    def test_capacity_masked_writes_and_copied_keys(self):
        memory = BatchedEntityMemory.empty(1, 2, 1, 1, 2)
        appearance = torch.zeros(1, 2, 1)
        value = torch.ones(1, 2, 2)
        written = memory.write(appearance, value)
        appearance.add_(1)
        self.assertEqual(written.keys.count_nonzero(), 0)
        self.assertEqual(written.retrieve(appearance).count_nonzero(), 0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            written.write(appearance, value)
        unchanged = written.write(appearance, value, active=torch.zeros(1, 2, dtype=torch.bool))
        torch.testing.assert_close(unchanged.values, written.values)
        updated = written.write(torch.zeros_like(appearance), value * 2)
        torch.testing.assert_close(updated.retrieve(torch.zeros_like(appearance)), value * 2)
        self.assertFalse(memory.occupied.any())
        self.assertEqual(memory.values.count_nonzero(), 0)

    def test_validation_and_disabled_memory(self):
        for dims in ((0, 2, 2, 3, 4), (1, 2, 0, 3, 4), (1, 2, 2, 3, -1),
                     (True, 2, 2, 3, 4)):
            with self.assertRaises(ValueError):
                BatchedEntityMemory.empty(*dims)
        memory = BatchedEntityMemory.empty(1, 2, 2, 3, 4, dtype=torch.float64)
        appearance = torch.zeros(1, 2, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            memory.retrieve(appearance.float())
        with self.assertRaises(ValueError):
            memory.retrieve(appearance[0])
        with self.assertRaises(ValueError):
            memory.retrieve(appearance, active=torch.ones(1, 2))
        with self.assertRaises(ValueError):
            memory.write(appearance, torch.zeros(1, 2, 5, dtype=torch.float64))
        with self.assertRaises(ValueError):
            memory.reset(torch.tensor([True, False]))
        disabled = BatchedEntityMemory.empty(1, 2, 1, 3, 0)
        self.assertEqual(disabled.retrieve(appearance.float()).shape, (1, 2, 0))
        self.assertIs(disabled.write(appearance.float(), torch.empty(1, 2, 0)), disabled)
        self.assertFalse(disabled.occupied.any())
