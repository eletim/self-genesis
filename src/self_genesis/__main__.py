"""Run a reproducible initialization smoke experiment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from self_genesis.config import load_config
from self_genesis.experiment import initialize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML experiment conditions")
    parser.add_argument("--seed", type=int, help="Override the configured seed")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    try:
        config = load_config(args.config, seed=args.seed, device=args.device)
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
