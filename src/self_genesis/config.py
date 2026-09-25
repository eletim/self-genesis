"""Small, validated experiment conditions."""

from dataclasses import dataclass, fields
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

    def __post_init__(self) -> None:
        for name, minimum in (
            ("seed", 0), ("num_agents", 2), ("appearance_dim", 1),
            ("initial_life", 1), ("initial_points", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")


def load_config(path: Path | None = None, **overrides: object) -> ExperimentConfig:
    values = {} if path is None else tomllib.loads(path.read_text())
    unknown = values.keys() - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**values)
