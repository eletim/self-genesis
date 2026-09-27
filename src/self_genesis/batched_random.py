"""Device-side Philox4x32-10 streams, keyed independently by world seed.

Integer arithmetic gives identical uniforms on CPU and CUDA. Each call consumes
whole four-value blocks only for active rows; batch order and inactive neighbors
cannot change a world's stream. This is separate from PyTorch's RNG sequence.
"""

import torch

_MASK = 2**32 - 1


def _multiply(value, multiplier):
    # Split products to avoid signed int64 overflow on either device.
    low_product = (value & 65535) * multiplier
    high_product = (value >> 16) * multiplier + (low_product >> 16)
    return high_product >> 16, ((high_product & 65535) << 16) | (low_product & 65535)


class BatchedRandom:
    def __init__(self, seeds, *, device, stream=0):
        self.keys = torch.tensor(seeds, dtype=torch.int64, device=device)
        self.counters = torch.zeros_like(self.keys)
        self.stream = stream

    def reset(self, row, seed):
        self.keys[row] = seed
        self.counters[row] = 0

    def uniform(self, size, active):
        blocks = (size + 3) // 4
        counter = self.counters[:, None] + torch.arange(blocks, device=self.keys.device)
        c0, c1 = counter & _MASK, counter >> 32
        c2, c3 = torch.full_like(counter, self.stream), torch.zeros_like(counter)
        k0, k1 = self.keys[:, None] & _MASK, self.keys[:, None] >> 32
        for _ in range(10):
            hi0, lo0 = _multiply(c0, 0xD2511F53)
            hi1, lo1 = _multiply(c2, 0xCD9E8D57)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            k0, k1 = (k0 + 0x9E3779B9) & _MASK, (k1 + 0xBB67AE85) & _MASK
        self.counters.add_(active.to(torch.int64) * blocks)
        values = torch.stack((c0, c1, c2, c3), dim=-1).flatten(1)[:, :size]
        return (values.to(torch.float64) / 2**32) * active[:, None]
