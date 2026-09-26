"""Synchronous survival dynamics shared by all agents."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch

from self_genesis.config import ExperimentConfig
from self_genesis.experiment import initialize


class Action(Enum):
    NOTHING = "NOTHING"
    GIVE = "GIVE"


@dataclass(frozen=True)
class Decision:
    action: Action = Action.NOTHING
    target: int | None = None


@dataclass(frozen=True)
class StepResult:
    reward: torch.Tensor
    died: torch.Tensor
    done: bool
    generated_points: torch.Tensor


class World:
    """One survival world, with one decision per agent per time step.

    A valid GIVE spends one Point to restore one Life to another agent.
    Gifts resolve simultaneously using eligibility at the start of the step,
    before every living agent loses one Life. Dead agents never act or revive.
    Each agent alive at the start earns one survival reward, including its
    final time step. Summed rewards therefore measure time lived. Survivors
    then generate at most one Point, available for use on the next step.
    """

    def __init__(self, config: ExperimentConfig, *, seed_rng: bool = True):
        self.state = initialize(config, seed_rng=seed_rng)
        # CPU draws match across devices without consuming other sampling streams.
        self._generation_rng = torch.Generator(device="cpu").manual_seed(config.seed + 2)

    @property
    def alive(self) -> torch.Tensor:
        return self.state.life > 0

    def step(self, decisions: Sequence[Decision]) -> StepResult:
        """Advance all agents together, rejecting malformed decisions atomically.

        Self-directed or out-of-range GIVE targets are invalid. A well-formed
        GIVE from a dead or exhausted donor, or to a dead recipient, has no
        effect and costs nothing. NOTHING must have no target. Calls after
        extinction return zero rewards and no new deaths.
        """
        state = self.state
        count = state.life.numel()
        if len(decisions) != count:
            raise ValueError("Provide exactly one decision per agent")
        for donor, decision in enumerate(decisions):
            if not isinstance(decision, Decision) or not isinstance(decision.action, Action):
                raise ValueError("Each decision must specify an Action")
            if decision.action is Action.NOTHING:
                if decision.target is not None:
                    raise ValueError("NOTHING must have no target")
            elif (type(decision.target) is not int
                  or not 0 <= decision.target < count
                  or decision.target == donor):
                raise ValueError("GIVE must target another agent in this world")

        alive = self.alive
        restored = torch.zeros_like(state.life)
        for donor, decision in enumerate(decisions):
            if (decision.action is Action.GIVE and alive[donor]
                    and alive[decision.target] and state.points[donor] > 0):
                state.points[donor] -= 1
                restored[decision.target] += 1
        state.life.add_(restored).sub_(alive.to(state.life.dtype)).clamp_(min=0)
        survivors = self.alive
        generated = torch.zeros_like(state.points)
        indices = survivors.nonzero().flatten()
        draws = torch.rand(indices.numel(), generator=self._generation_rng).to(state.life.device)
        generated[indices] = (draws < state.point_generation_probability[indices]).to(generated.dtype)
        state.points.add_(generated)
        return StepResult(
            reward=alive.to(torch.float32),
            died=alive & ~survivors,
            done=not bool(survivors.any()),
            generated_points=generated,
        )
