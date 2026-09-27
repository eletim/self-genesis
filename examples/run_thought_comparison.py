"""Run the predeclared bounded Thought experiment; never overwrite evidence."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from self_genesis.comparison import run_comparison
from benchmark_rtx5090 import Telemetry
from self_genesis.config import load_config

TRAINING_SEEDS = (10000, 20000, 30000)
EVALUATION_SEEDS = (100000, 100001, 100002)
TIMEOUT_SECONDS = 1200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(Path('configs/controlled-thought.toml'))
    if args.worker is not None:
        run_comparison(config, args.output, training_seeds=[args.worker],
                       evaluation_seeds=EVALUATION_SEEDS, compare_thought=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(config=asdict(config), training_seeds=TRAINING_SEEDS,
                    evaluation_seeds=EVALUATION_SEEDS,
                    timeout_seconds_per_seed=TIMEOUT_SECONDS, python=platform.python_version(),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0),
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    telemetry_scope='Whole worker: training plus evaluation; device-wide.',
                    cases=[])
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    for seed in TRAINING_SEEDS:
        output = args.output / f'seed-{seed}.json'
        command = [sys.executable, __file__, '--worker', str(seed), '--output', str(output)]
        case = dict(seed=seed, command=command)
        telemetry = Telemetry()
        start = time.monotonic()
        telemetry.thread.start()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            case.update(returncode=completed.returncode, stdout=completed.stdout,
                        stderr=completed.stderr)
            if completed.returncode:
                raise RuntimeError(f'Seed {seed} failed: {completed.stderr}')
            raw = output.read_bytes()
            with output.with_suffix('.json.gz').open('xb') as destination:
                destination.write(gzip.compress(raw, mtime=0))
            output.unlink()
            case['report_sha256'] = hashlib.sha256(raw).hexdigest()
        except Exception as error:
            case['error'] = str(error)
        finally:
            case['elapsed_seconds_including_evaluation'] = time.monotonic() - start
            telemetry.stop.set()
            telemetry.thread.join()
            case.update(telemetry=telemetry.samples, telemetry_errors=telemetry.errors)
            manifest['cases'].append(case)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f"Seed {seed}: {'failed' if 'error' in case else 'completed'}", flush=True)
    if any('error' in case for case in manifest['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
