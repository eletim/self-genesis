"""Tensor trajectories across independently ending survival worlds."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from self_genesis.batched_encounter import BatchedDecision, BatchedEncounterProtocol
from self_genesis.batched_world import BatchedWorld
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy


@dataclass(frozen=True)
class BatchedExperience:
    # All resource/mask tensors are [world, agent]; decisions retain autograd.
    reward: torch.Tensor
    died: torch.Tensor
    decisions: tuple[BatchedDecision, ...]


@dataclass(frozen=True)
class BatchedRollout:
    experiences: tuple[BatchedExperience, ...]
    start_steps: torch.Tensor
    end_steps: torch.Tensor
    terminated: torch.Tensor
    horizon_completed: torch.Tensor
    episode_ids: torch.Tensor
    source: object

    @property
    def truncated(self) -> torch.Tensor:
        return ~(self.terminated | self.horizon_completed)

    def extend(self, following: "BatchedRollout") -> "BatchedRollout":
        """Join contiguous budgets from the same episodes without detaching."""
        if (self.source is not following.source
                or not torch.equal(self.episode_ids, following.episode_ids)
                or not torch.equal(self.end_steps, following.start_steps)):
            raise ValueError("Segments must be contiguous and from the same episodes")
        return BatchedRollout(
            self.experiences + following.experiences, self.start_steps,
            following.end_steps, following.terminated, following.horizon_completed,
            self.episode_ids, self.source)


class BatchedRolloutCollector:
    """Collect budgets without resetting completed rows or cutting recurrence.

    Join successive segments with ``extend`` before computing survival loss.
    Finished rows freeze while the remaining worlds finish their objectives.
    Consume the loss before detach/optimizer updates; reset explicitly starts
    a new episode in one row. Returned segments survive reset and detach.
    """

    def __init__(self, config: ExperimentConfig, network: RecurrentPolicy, *,
                 seeds: Sequence[int]):
        seeds = tuple(seeds)
        self.config = config
        self.network = network
        self.world = BatchedWorld(config, seeds=seeds)
        self.protocol = BatchedEncounterProtocol(self.world, network, seeds=seeds)
        self.state = network.initial_batch_state(len(seeds), config.num_agents,
                                                 slots=config.num_agents)
        self.episode_ids = torch.zeros_like(self.world.steps)
        self._source = object()
        self.trace_recorder = None

    def reset(self, world: int, *, seed: int) -> None:
        """Reset one world and its recurrent state, retaining other rows' graphs."""
        self.world.reset(world, seed=seed)
        mask = torch.zeros_like(self.world.done)
        mask[world] = True
        self.state = self.state.reset(mask)
        self.episode_ids[world] += 1

    def detach(self) -> None:
        """Release collector graph references only after complete objectives."""
        if not bool(self.world.done.all()):
            raise ValueError("Cannot detach incomplete survival episodes")
        self.state = self.state.detach()

    def collect(self, max_steps: int) -> BatchedRollout:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        start = self.world.steps.clone()
        trace_start = int(start.max()) if self.trace_recorder is not None else 0
        experiences = []
        for _ in range(max_steps):
            if bool(self.world.done.all()):
                break
            result = self.protocol.step(self.state)
            self.state = result.state
            if self.trace_recorder is not None:
                self.trace_recorder.record_step(self, result, trace_start + len(experiences))
            experiences.append(BatchedExperience(
                result.world.reward, result.world.died, result.decisions))
        return BatchedRollout(
            tuple(experiences), start, self.world.steps.clone(),
            self.world.terminated.clone(), self.world.horizon_completed.clone(),
            self.episode_ids.clone(), self._source)
