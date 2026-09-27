"""Tensor-only policy observations and per-world, per-observer recurrent state."""

from dataclasses import dataclass

import torch

from self_genesis.entity_memory import BatchedEntityMemory


@dataclass(frozen=True)
class BatchedObservation:
    """Permitted inputs, with leading [world, observer] axes.

    resources ends in four entries: own Life/Points, partner Life/Points.
    partner_action is int64: 0 unknown, 1 NOTHING, 2 GIVE. received_message
    ends in max_message_length ordered slots, padded with vocabulary_size.
    first is boolean; partner_appearance ends in appearance_dim. Routing IDs,
    generation ability, future actions and analysis histories are not inputs.
    All tensors must be on the policy device. Inactive values are ignored.
    """

    resources: torch.Tensor
    partner_appearance: torch.Tensor
    first: torch.Tensor
    received_message: torch.Tensor
    partner_action: torch.Tensor


@dataclass(frozen=True)
class BatchedPolicyState:
    memory: torch.Tensor
    affect: torch.Tensor
    entities: BatchedEntityMemory

    def reset(self, worlds: torch.Tensor):
        """Clear only completed worlds, preserving other episodes' graphs."""
        entities = self.entities.reset(worlds)
        return BatchedPolicyState(
            torch.where(worlds[:, None, None], 0, self.memory),
            torch.where(worlds[:, None, None], 0, self.affect), entities)

    def detach(self):
        """Explicit optimizer boundary; never called implicitly by forwards."""
        return BatchedPolicyState(self.memory.detach(), self.affect.detach(),
                                  self.entities.detach())
