"""Appearance-keyed tensor storage for independent observers and worlds."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchedEntityMemory:
    """Copy-on-write episode state, separate from shared policy weights.

    Keys are [world, observer, slot, appearance], values are
    [world, observer, slot, value], and occupied is [world, observer, slot].
    Each call reads or writes one observed Appearance per observer. Callers mask
    nonparticipants, dead agents and finished worlds with ``active``. Slots are
    allocated on first observation, never by partner ID; equal Appearances share
    a slot. Reserve at least the number of possible distinct partners as capacity.
    No values are evicted or detached during an episode. At episode boundaries,
    call ``reset`` with the completed-world mask before collecting new episodes.
    """

    keys: torch.Tensor
    values: torch.Tensor
    occupied: torch.Tensor

    @classmethod
    def empty(cls, worlds: int, observers: int, slots: int, appearance_dim: int,
              value_dim: int, *, device=None, dtype=torch.float32):
        for name, size in (("worlds", worlds), ("observers", observers),
                           ("slots", slots), ("appearance_dim", appearance_dim),
                           ("value_dim", value_dim)):
            if type(size) is not int or size < (0 if name == "value_dim" else 1):
                raise ValueError(f"Invalid {name}")
        if not dtype.is_floating_point:
            raise ValueError("Memory dtype must be floating point")
        shape = (worlds, observers, slots)
        return cls(torch.zeros((*shape, appearance_dim), device=device, dtype=dtype),
                   torch.zeros((*shape, value_dim), device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=torch.bool))

    def _mask(self, mask, shape):
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=self.values.device)
        if (mask.shape != shape or mask.dtype != torch.bool
                or mask.device != self.values.device):
            raise ValueError("Mask must be boolean with the expected shape and device")
        return mask

    def _matches(self, appearance):
        if (appearance.shape != self.keys.shape[:2] + self.keys.shape[-1:]
                or appearance.device != self.keys.device
                or appearance.dtype != self.keys.dtype):
            raise ValueError("Appearance must match the memory shape, device and dtype")
        return self.occupied & (self.keys == appearance.unsqueeze(2)).all(dim=-1)

    def retrieve(self, appearance: torch.Tensor, *, active=None) -> torch.Tensor:
        """Return [world, observer, value]; unseen or inactive queries return zero."""
        matches = self._matches(appearance)
        active = self._mask(active, self.values.shape[:2])
        index = matches.to(torch.int64).argmax(dim=2)
        value = self.values.gather(
            2, index[..., None, None].expand(*index.shape, 1, self.values.shape[-1])
        ).squeeze(2)
        return torch.where((active & matches.any(dim=2))[..., None], value, 0)

    def write(self, appearance: torch.Tensor, value: torch.Tensor, *, active=None):
        """Store learned values without breaking their graph or modifying old states.

        Full observers may still update existing keys. A new key in a full
        observer raises rather than silently forgetting an earlier experience.
        Keys are observed data and are copied without an autograd history.
        """
        matches = self._matches(appearance)
        active = self._mask(active, self.values.shape[:2])
        if (value.shape != self.values.shape[:2] + self.values.shape[-1:]
                or value.device != self.values.device or value.dtype != self.values.dtype):
            raise ValueError("Value must match the memory shape, device and dtype")
        if self.values.shape[-1] == 0:
            return self
        found = matches.any(dim=2)
        if bool((active & ~found & self.occupied.all(dim=2)).any()):
            raise ValueError("Entity Memory capacity exhausted")
        index = torch.where(found, matches.to(torch.int64).argmax(dim=2),
                            (~self.occupied).to(torch.int64).argmax(dim=2))
        selected = (torch.arange(self.values.shape[2], device=value.device)
                    == index[..., None]) & active[..., None]
        return BatchedEntityMemory(
            torch.where(selected[..., None], appearance.detach().unsqueeze(2), self.keys),
            torch.where(selected[..., None], value.unsqueeze(2), self.values),
            self.occupied | selected)

    def reset(self, worlds: torch.Tensor):
        """Clear selected episodes, preserving other worlds' values and gradients."""
        worlds = self._mask(worlds, self.values.shape[:1])
        return BatchedEntityMemory(
            torch.where(worlds[:, None, None, None], 0, self.keys),
            torch.where(worlds[:, None, None, None], 0, self.values),
            self.occupied & ~worlds[:, None, None])

    def detach(self):
        """Explicit graph boundary for use after an optimizer update."""
        return BatchedEntityMemory(self.keys, self.values.detach(), self.occupied)
