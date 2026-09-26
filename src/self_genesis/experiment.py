"""Initialize independent agent state for a shared survival world."""

from dataclasses import dataclass
import random

import torch

from self_genesis.config import ExperimentConfig


@dataclass
class ExperimentState:
    life: torch.Tensor
    points: torch.Tensor
    appearance: torch.Tensor
    point_generation_probability: torch.Tensor


def resolve_device(requested: str) -> torch.device:
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device(requested)


def initialize(config: ExperimentConfig, *, seed_rng: bool = True) -> ExperimentState:
    """Create configured state, optionally seeding global sampling streams."""
    device = resolve_device(config.device)
    if seed_rng:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
    # Generate on CPU so initial appearances match across CPU and CUDA runs.
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    appearance = torch.rand(
        (config.num_agents, config.appearance_dim), generator=generator
    ).to(device)
    # Separate seeded streams keep ability independent of Appearance dimensions
    # and preserve global policy sampling streams on episode reset.
    ability_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    probability = torch.rand(config.num_agents, generator=ability_generator)
    probability = (config.point_generation_probability_min + probability *
                   (config.point_generation_probability_max -
                    config.point_generation_probability_min))
    return ExperimentState(
        life=torch.full((config.num_agents,), config.initial_life, device=device),
        points=torch.full((config.num_agents,), config.initial_points, device=device),
        appearance=appearance,
        point_generation_probability=probability.to(device),
    )
