"""Independent survival worlds with tensor-based resource transitions."""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import torch

from self_genesis.batched_random import BatchedRandom
from self_genesis.config import ExperimentConfig
from self_genesis.experiment import ExperimentState, initialize


@dataclass(frozen=True)
class BatchedStepResult:
    reward: torch.Tensor
    died: torch.Tensor
    terminated: torch.Tensor
    horizon_completed: torch.Tensor
    done: torch.Tensor
    generated_points: torch.Tensor
    # Boolean [world, donor] mask; recipients are the corresponding input targets.
    successful_transfers: torch.Tensor


class BatchedWorld:
    """Fixed-size worlds sharing conditions but owning separate seeded streams.

    State resources have shape [world, agent], Appearance [world, agent, feature].
    ``step`` accepts int64 targets [world, agent]: -1 means NOTHING, otherwise
    GIVE to that agent in the same world. Completed rows freeze until explicitly
    reset. No global RNG is seeded or consumed, including during reset.
    """

    def __init__(self, config: ExperimentConfig, *, seeds: Sequence[int]):
        self.config = config
        seeds = tuple(seeds)
        if not seeds:
            raise ValueError("Provide at least one world seed")
        states = [initialize(replace(config, seed=seed), seed_rng=False) for seed in seeds]
        self.state = ExperimentState(**{
            field.name: torch.stack([getattr(state, field.name) for state in states])
            for field in fields(ExperimentState)
        })
        device = self.state.life.device
        self._generation_rng = BatchedRandom(seeds, device=device, stream=1)
        self.steps = torch.zeros(len(seeds), dtype=torch.int64, device=device)
        self.terminated = torch.zeros(len(seeds), dtype=torch.bool, device=device)
        self.horizon_completed = torch.zeros_like(self.terminated)

    @property
    def alive(self) -> torch.Tensor:
        return self.state.life > 0

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.horizon_completed

    def reset(self, world: int, *, seed: int) -> None:
        """Start one row's new episode without advancing or resetting other rows."""
        if type(world) is not int or not 0 <= world < self.steps.numel():
            raise ValueError("world must index a row in this batch")
        state = initialize(replace(self.config, seed=seed), seed_rng=False)
        for field in fields(ExperimentState):
            getattr(self.state, field.name)[world].copy_(getattr(state, field.name))
        self._generation_rng.reset(world, seed)
        self.steps[world] = 0
        self.terminated[world] = False
        self.horizon_completed[world] = False

    def step(self, targets: torch.Tensor) -> BatchedStepResult:
        """Resolve GIVE simultaneously, decay Life, then regenerate for survivors.

        Validate the entire input before mutation, even for finished rows. Dead
        or exhausted donors and dead recipients cannot transfer. Rewards count
        agents alive at step start, including their final living step. Extinction
        takes precedence over horizon completion on the same step.
        """
        state = self.state
        if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64
                or targets.shape != state.life.shape or targets.device != state.life.device):
            raise ValueError("targets must be int64 with the state shape and device")
        donors = torch.arange(state.life.shape[1], device=state.life.device)
        if bool(((targets < -1) | (targets >= state.life.shape[1])
                 | (targets == donors)).any()):
            raise ValueError("Each target must be -1 or another agent in this world")

        active = ~self.done
        alive = self.alive & active[:, None]
        recipients = targets.clamp(min=0)
        transfers = ((targets >= 0) & alive & (state.points > 0)
                     & alive.gather(1, recipients))
        restored = torch.zeros_like(state.life)
        restored.scatter_add_(1, recipients, transfers.to(state.life.dtype))
        state.points.sub_(transfers.to(state.points.dtype))
        state.life.add_(restored).sub_(alive.to(state.life.dtype)).clamp_(min=0)
        survivors = self.alive & active[:, None]
        draws = self._generation_rng.uniform(state.life.shape[1], active)
        generated = (survivors & (draws < state.point_generation_probability)).to(state.points.dtype)
        state.points.add_(generated)
        self.steps.add_(active.to(self.steps.dtype))
        self.terminated |= active & ~self.alive.any(dim=1)
        if self.config.survival_horizon is not None:
            self.horizon_completed |= (active & ~self.terminated
                                       & (self.steps >= self.config.survival_horizon))
        return BatchedStepResult(
            reward=alive.to(torch.float32),
            died=alive & ~survivors,
            terminated=self.terminated.clone(),
            horizon_completed=self.horizon_completed.clone(),
            done=self.done,
            generated_points=generated,
            successful_transfers=transfers,
        )
