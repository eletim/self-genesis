import unittest

import torch

from self_genesis.batched_random import BatchedRandom


class BatchedRandomTests(unittest.TestCase):
    def test_philox_reference_vector(self):
        rng = BatchedRandom([0], device='cpu')
        values = rng.uniform(4, torch.tensor([True])) * 2**32
        self.assertEqual(values.long().tolist(),
                         [[0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]])

    def test_independent_streams_freezing_reset_and_seed_high_bits(self):
        batch = BatchedRandom([7, 2**62 + 7, 19], device='cpu')
        single = BatchedRandom([19], device='cpu')
        active = torch.tensor([True, False, True])
        initial = torch.get_rng_state()
        for step in range(5):
            batch.reset(0, step)
            draws = batch.uniform(13, active)
            self.assertTrue(torch.equal(draws[2], single.uniform(13, active[2:])[0]))
            self.assertFalse(draws[1].any())
        self.assertEqual(batch.counters.tolist(), [4, 0, 20])
        batch.reset(0, 7)
        draws = batch.uniform(13, torch.ones(3, dtype=torch.bool))
        self.assertFalse(torch.equal(draws[0], draws[1]))
        self.assertTrue(torch.equal(initial, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cpu_cuda_uniforms_match(self):
        seeds = [0, 2**63 - 1, 12345]
        cpu = BatchedRandom(seeds, device='cpu', stream=1)
        cuda = BatchedRandom(seeds, device='cuda', stream=1)
        for active in ([True, True, True], [False, True, False], [True, True, True]):
            mask = torch.tensor(active)
            self.assertTrue(torch.equal(cpu.uniform(17, mask),
                                        cuda.uniform(17, mask.cuda()).cpu()))
