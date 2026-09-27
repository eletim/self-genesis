"""Run the predeclared bounded density experiment; never overwrite evidence."""
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from self_genesis.comparison import INTERVENTIONS, run_comparison
from self_genesis.config import load_config

TRAINING_SEEDS = (10000, 20000, 30000)
EVALUATION_SEEDS = (100000, 100001, 100002)
DENSITIES = (1, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=DENSITIES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(Path('configs/controlled-density.toml'))
    if args.worker is not None:
        run_comparison(replace(config, encounter_count=args.worker), args.output,
                       training_seeds=TRAINING_SEEDS, evaluation_seeds=EVALUATION_SEEDS,
                       interventions=INTERVENTIONS)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config=asdict(config), training_seeds=TRAINING_SEEDS,
                    evaluation_seeds=EVALUATION_SEEDS, densities=DENSITIES,
                    timeout_seconds_per_density=900, python=platform.python_version(),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cases=[])
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for density in DENSITIES:
        output = args.output / f'pairs-{density}.json'
        command = [sys.executable, __file__, '--worker', str(density), '--output', str(output)]
        case = dict(density=density, command=command)
        start = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
            case.update(returncode=completed.returncode, stdout=completed.stdout,
                        stderr=completed.stderr)
            if completed.returncode:
                raise RuntimeError(f'Density {density} failed: {completed.stderr}')
            raw = output.read_bytes()
            with output.with_suffix('.json.gz').open('xb') as destination:
                destination.write(gzip.compress(raw, mtime=0))
            output.unlink()
            case['report_sha256'] = hashlib.sha256(raw).hexdigest()
        except Exception as error:
            case['error'] = str(error)
            raise
        finally:
            case['elapsed_seconds_including_evaluation'] = time.monotonic() - start
            manifest['cases'].append(case)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Completed {density} pairs', flush=True)


if __name__ == '__main__':
    main()
