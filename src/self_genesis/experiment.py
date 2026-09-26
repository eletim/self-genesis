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


def resolve_device(requested: str) -> torch.device:
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device(requested)


def initialize(config: ExperimentConfig) -> ExperimentState:
    device = resolve_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    # Generate on CPU so initial appearances match across CPU and CUDA runs.
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    appearance = torch.rand(
        (config.num_agents, config.appearance_dim), generator=generator
    ).to(device)
    return ExperimentState(
        life=torch.full((config.num_agents,), config.initial_life, device=device),
        points=torch.full((config.num_agents,), config.initial_points, device=device),
        appearance=appearance,
    )
