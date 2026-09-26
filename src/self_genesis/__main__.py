"""Initialize agents, train survival policies, or compare fixed baselines."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from self_genesis.config import load_config
from self_genesis.comparison import INTERVENTIONS, run_comparison
from self_genesis.experiment import initialize
from self_genesis.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("init", "train", "compare"), default="init")
    parser.add_argument("--output", type=Path,
                        help="New results file (train: JSONL; compare: JSON)")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+",
                        help="Matched comparison seeds (default: configured seed)")
    parser.add_argument("--training-seeds", type=int, nargs="+",
                        help="Independent comparison training seeds (default: configured seed)")
    parser.add_argument("--interventions", choices=INTERVENTIONS, nargs="+",
                        help="Additional frozen learned-policy evaluation treatments")
    for name in ("num_agents", "appearance_dim", "initial_life", "initial_points",
                 "vocabulary_size", "max_message_length", "memory_dim", "affect_dim",
                 "episodes", "survival_horizon"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--training-method", choices=("actor_critic", "reinforce"))
    parser.add_argument("--value-loss-coefficient", type=float)
    parser.add_argument("--action-entropy-coefficient", type=float)
    parser.add_argument("--message-entropy-coefficient", type=float)
    parser.add_argument("--point-generation-probability-min", type=float)
    parser.add_argument("--point-generation-probability-max", type=float)
    parser.add_argument("--config", type=Path, help="TOML experiment conditions")
    parser.add_argument("--seed", type=int, help="Override the configured seed")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    if args.command in ("train", "compare") and args.output is None:
        parser.error(f"{args.command} requires --output pointing to a new results file")
    if args.command == "init" and args.output is not None:
        parser.error("--output is only supported for train or compare")
    for option in ("evaluation_seeds", "training_seeds", "interventions"):
        if args.command != "compare" and getattr(args, option) is not None:
            parser.error(f"--{option.replace('_', '-')} is only supported for compare")
    overrides = vars(args).copy()
    for key in ("command", "output", "config", "evaluation_seeds", "training_seeds", "interventions"):
        overrides.pop(key)
    try:
        config = load_config(args.config, **overrides)
        if args.command == "compare":
            result = run_comparison(config, args.output, evaluation_seeds=args.evaluation_seeds,
                                    training_seeds=args.training_seeds,
                                    interventions=args.interventions or ())
            print(json.dumps(result, sort_keys=True))
            return
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
