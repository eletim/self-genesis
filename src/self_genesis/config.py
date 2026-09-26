"""Small, validated experiment conditions."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    num_agents: int = 4
    appearance_dim: int = 8
    initial_life: int = 10
    initial_points: int = 3
    point_generation_probability_min: float = 0.0
    point_generation_probability_max: float = 0.0
    vocabulary_size: int = 4
    max_message_length: int = 3
    memory_dim: int = 16
    affect_dim: int = 4
    entity_memory_dim: int = 16
    survival_horizon: int | None = None
    episodes: int = 3
    learning_rate: float = 0.001
    training_method: str = "actor_critic"
    value_loss_coefficient: float = 0.5
    action_entropy_coefficient: float = 0.01
    message_entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        if self.training_method not in ("actor_critic", "reinforce"):
            raise ValueError("training_method must be actor_critic or reinforce")
        for name, minimum in (
            ("seed", 0), ("num_agents", 2), ("appearance_dim", 1),
            ("initial_life", 1), ("initial_points", 0),
            ("vocabulary_size", 1), ("max_message_length", 0),
            ("memory_dim", 1), ("affect_dim", 1), ("episodes", 1),
            ("entity_memory_dim", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.survival_horizon is not None and (
                type(self.survival_horizon) is not int or self.survival_horizon < 1):
            raise ValueError("survival_horizon must be a positive integer or None")
        for name in ("point_generation_probability_min", "point_generation_probability_max"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if self.point_generation_probability_min > self.point_generation_probability_max:
            raise ValueError("Point generation probability minimum must not exceed maximum")
        if (type(self.learning_rate) not in (int, float)
                or not 0 < self.learning_rate < math.inf):
            raise ValueError("learning_rate must be a finite positive number")
        for name in ("value_loss_coefficient", "action_entropy_coefficient",
                     "message_entropy_coefficient"):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (name == "value_loss_coefficient" and value == 0)):
                constraint = "positive" if name == "value_loss_coefficient" else "nonnegative"
                raise ValueError(f"{name} must be a finite {constraint} number")
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")


def load_config(path: Path | None = None, **overrides: object) -> ExperimentConfig:
    values = {} if path is None else tomllib.loads(path.read_text())
    values.update({key: value for key, value in overrides.items() if value is not None})
    unknown = values.keys() - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    return ExperimentConfig(**values)
