"""Isolated RTX 5090 training sweep; run from the repository with PYTHONPATH=src."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import torch

from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.config import load_config
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_batch, train_episode
from self_genesis.training_metrics import UpdateMeasurement
from self_genesis.world import Action


def encounter_count(rollout, batched):
    """Each encounter has exactly two action decisions, including NOTHING."""
    if batched:
        counts = [d.active.sum() for e in rollout.experiences
                  for d in e.decisions if not d.communicating]
        return int(torch.stack(counts).sum().item()) // 2 if counts else 0
    return sum(isinstance(d.choice, Action) for agent in rollout.experiences
               for e in agent for d in e.decisions) // 2


class Telemetry:
    def __init__(self):
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll)

    def poll(self):
        while not self.stop.is_set():
            try:
                output = subprocess.check_output([
                    'nvidia-smi', '--id=0',
                    '--query-gpu=utilization.gpu,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=5)
                utilization, memory, power = map(float, output.strip().split(','))
                self.samples.append(dict(monotonic_seconds=time.perf_counter(),
                                         utilization_percent=utilization,
                                         memory_used_mib=memory, power_watts=power))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
            self.stop.wait(0.2)


def benchmark_config(args):
    """Resolve a worker workload without changing its training or resource rules."""
    return load_config(None, capacity_preset=args.capacity, device='cuda',
                       seed=args.seed, batched=args.worlds > 0,
                       thought_mode="recurrent", think_steps=getattr(args, "think_steps", 16),
                       num_worlds=max(1, args.worlds), mixed_precision=args.precision,
                       num_agents=args.num_agents, encounter_count=args.encounter_count,
                       survival_horizon=args.horizon, episodes=args.warmup + args.updates,
                       point_generation_probability_min=0.1,
                       point_generation_probability_max=0.3)


def sweep_cases(args):
    if args.sweep == 'recurrent':
        return [(args.capacity, args.precision, worlds, args.num_agents, args.encounter_count)
                for worlds in args.world_counts]
    if args.sweep == 'density':
        # Capacity and precision stay fixed within this 32-agent comparison.
        return [(args.capacity, args.precision, worlds, 32, pairs)
                for pairs in (1, 4, 8, 16) for worlds in args.world_counts]
    return [(capacity, precision, worlds, 4, 1)
            for capacity in ('small', 'medium', 'large')
            for precision, worlds in ([('fp32', 0)] + [
                (precision, worlds) for precision in ('fp32', 'bf16')
                for worlds in (64, 128, 256, 512, 1024)])]


def worker(args):
    if not torch.cuda.is_available() or 'RTX 5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This evidence runner requires an actual RTX 5090 at CUDA device 0')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = benchmark_config(args)
    network = RecurrentPolicy(config.appearance_dim, memory_dim=config.memory_dim,
                              affect_dim=config.affect_dim,
                              entity_memory_dim=config.entity_memory_dim,
                              thought_mode=config.thought_mode, think_steps=config.think_steps).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    seeds = [args.seed + row for row in range(config.num_worlds)]
    collector = (BatchedRolloutCollector(config, network, seeds=seeds) if config.batched
                 else RolloutCollector(config, network))
    collect = collector.collect
    counts = []

    def counted_collect(budget):
        rollout = collect(budget)
        counts.append(encounter_count(rollout, config.batched))
        return rollout

    collector.collect = counted_collect
    records = []
    telemetry = Telemetry()
    started = time.perf_counter()
    status, error = 'ok', None
    try:
        for update in range(args.warmup + args.updates):
            if update == args.warmup:
                telemetry.thread.start()
            measurement = UpdateMeasurement(torch.device('cuda'))
            result = (train_batch(collector, optimizer, seeds=[seed + update * config.num_worlds
                                                               for seed in seeds])
                      if config.batched else train_episode(collector, optimizer))
            steps = sum(result.steps) if config.batched else result.steps
            measured = measurement.finish(steps)
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            gradients_finite = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
            finite = all(torch.isfinite(p).all().item() for p in network.parameters())
            if not gradients_finite or not finite or not math.isfinite(result.loss) or not math.isfinite(result.metrics['gradient_norm']):
                raise RuntimeError('Non-finite loss, gradient norm, or updated parameters')
            records.append(dict(update=update, warmup=update < args.warmup,
                                steps=steps, encounters=counts[-1], episodes=config.num_worlds if config.batched else 1,
                                loss=result.loss, metrics=result.metrics, parameters_finite=finite,
                                gradients_finite=gradients_finite, gradient_tensor_count=len(gradients),
                                **measured))
    except Exception as failure:
        status, error = 'failed', f'{type(failure).__name__}: {failure}'
    finally:
        telemetry.stop.set()
        if telemetry.thread.ident is not None:
            telemetry.thread.join()
    measured = [record for record in records if not record['warmup']]
    seconds = sum(record['elapsed_seconds'] for record in measured)
    return dict(status=status, error=error, failed_update=update if error else None,
                capacity=args.capacity, config=asdict(config),
                parameter_count=network.parameter_count, records=records,
                measured_seconds=seconds, worker_seconds=time.perf_counter() - started,
                rates={name + '_per_second': sum(row[name] for row in measured) / seconds
                       for name in ('steps', 'encounters', 'episodes')} if seconds else {},
                telemetry=telemetry.samples, telemetry_errors=telemetry.errors,
                warmup_updates=args.warmup, measured_updates=args.updates,
                source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                environment=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                                 nvidia_smi=subprocess.check_output([
                                     'nvidia-smi', '--id=0',
                                     '--query-gpu=uuid,driver_version,memory.total,power.limit',
                                     '--format=csv'], text=True).strip(),
                                 tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                 cuda=torch.version.cuda, python=platform.python_version(),
                                 platform=platform.platform(), cpu_threads=torch.get_num_threads()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--sweep', choices=['scaling', 'density', 'recurrent'], default='scaling')
    parser.add_argument('--world-counts', type=int, nargs='+', default=[64, 128, 256],
                        help='positive world counts for density or recurrent sweeps')
    parser.add_argument('--think-steps', type=int, choices=[16, 32, 64], default=16,
                        help='worker recurrence depth')
    parser.add_argument('--include-64', action='store_true',
                        help='also run 64 think steps in the recurrent sweep')
    parser.add_argument('--num-agents', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--encounter-count', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--capacity', choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--worlds', type=int, default=64, help='0 selects sequential FP32')
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if min(args.updates, args.horizon, args.timeout) < 1 or min(args.warmup, args.worlds) < 0:
        parser.error('updates, horizon, timeout must be positive; warmup and worlds nonnegative')
    if any(world < 1 for world in args.world_counts) or len(set(args.world_counts)) != len(args.world_counts):
        parser.error('world-counts must be positive and unique')
    if args.worker:
        result = worker(args)
        with args.output.open('x') as file:
            json.dump(result, file, allow_nan=False, sort_keys=True)
        if result['status'] != 'ok':
            raise SystemExit(result['error'])
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [(case, depth) for case in sweep_cases(args)
             for depth in ((16, 32, 64) if args.include_64 else (16, 32))
             ] if args.sweep == 'recurrent' else [(case, args.think_steps) for case in sweep_cases(args)]
    for (capacity, precision, worlds, agents, pairs), depth in cases:
        name = f'{capacity}-{precision}-{worlds}'
        if args.sweep in ('density', 'recurrent'):
            name += f'-agents{agents}-pairs{pairs}'
        if args.sweep == 'recurrent':
            name += f'-think{depth}'
        command = [sys.executable, __file__, '--worker', '--output', str(args.output / (name + '.json')),
                   '--capacity', capacity, '--precision', precision, '--worlds', str(worlds),
                   '--num-agents', str(agents), '--encounter-count', str(pairs),
                   '--updates', str(args.updates), '--warmup', str(args.warmup),
                   '--horizon', str(args.horizon), '--seed', str(args.seed), '--think-steps', str(depth)]
        start = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            status = 'ok' if completed.returncode == 0 else 'failed'
            error = completed.stderr
        except subprocess.TimeoutExpired:
            status, error = 'timeout', f'Exceeded {args.timeout} seconds'
        manifest = dict(case=name, status=status, process_seconds=time.perf_counter() - start,
                        command=command, error=error)
        with (args.output / 'manifest.jsonl').open('a') as file:
            file.write(json.dumps(manifest) + '\n')
        print(name, status, round(manifest['process_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
