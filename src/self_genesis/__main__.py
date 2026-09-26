"""Initialize agents or run bounded survival training experiments."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from self_genesis.config import load_config
from self_genesis.experiment import initialize
from self_genesis.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("init", "train"), default="init")
    parser.add_argument("--output", type=Path, help="New JSONL results file (required for train)")
    for name in ("num_agents", "appearance_dim", "initial_life", "initial_points",
                 "vocabulary_size", "max_message_length", "memory_dim", "affect_dim",
                 "episodes"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--point-generation-probability-min", type=float)
    parser.add_argument("--point-generation-probability-max", type=float)
    parser.add_argument("--config", type=Path, help="TOML experiment conditions")
    parser.add_argument("--seed", type=int, help="Override the configured seed")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    if args.command == "train" and args.output is None:
        parser.error("train requires --output pointing to a new JSONL file")
    if args.command == "init" and args.output is not None:
        parser.error("--output is only supported for train")
    overrides = vars(args).copy()
    for key in ("command", "output", "config"):
        overrides.pop(key)
    try:
        config = load_config(args.config, **overrides)
        if args.command == "train":
            result = run_training(config, args.output)
            print(json.dumps(result, sort_keys=True))
            return
        state = initialize(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "config": asdict(config),
        "resolved_device": str(state.life.device),
        "life": state.life.tolist(),
        "points": state.points.tolist(),
        "appearance": state.appearance.tolist(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
