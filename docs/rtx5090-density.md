# RTX 5090: 32-agent encounter density

Measured on 2026-09-28 with the existing training implementation from
`dev/v0.0.8` (`7f16f09`) and benchmark extension at `a24147a`.
[Raw evidence](evidence/rtx5090-density/results.json.gz) contains all per-update
records, resolved configurations, telemetry, worker commands, source revision,
runner SHA-256, software metadata, failures/timeouts and validation output.
The earlier [four-agent capacity sweep](rtx5090-scaling.md) is a different workload.

## Reproduce

Use a CUDA-capable PyTorch environment from the repository root:

```sh
PYTHONPATH=src python examples/benchmark_rtx5090.py --sweep density \
  --capacity small --precision fp32 --world-counts 64 128 256 \
  --output /tmp/rtx5090-density-32 --updates 20 --warmup 1 --horizon 16 --seed 42
PYTHONPATH=src python examples/benchmark_rtx5090.py --worker \
  --capacity small --precision fp32 --worlds 256 --num-agents 32 --encounter-count 16 \
  --output /tmp/rtx5090-density-soak.json --updates 100 --warmup 1 --horizon 16 --seed 43
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

Output paths must be new. Each matrix case runs in an isolated process with a
300-second timeout, logged in the manifest; the direct worker has no parent
timeout. The original default `--sweep scaling` matrix remains available.
Density mode fixes 32 agents and counts 1/4/8/16, using the requested capacity
and precision for every case. World counts must be positive and unique.

Hardware: NVIDIA GeForce RTX 5090, 32,607 MiB VRAM, 575 W power limit,
driver 595.91.07; AMD Ryzen 9 9950X3D. Python 3.12.14, PyTorch 2.7.1+cu128,
CUDA runtime 12.8, one PyTorch CPU thread, TF32 matmul disabled, default
nondeterministic execution. The interpreter was `/tmp/self-genesis-verify-gpu/bin/python`.
GPU numbering was unmodified. The desktop remained active; no other benchmark
or test workload was launched during measurement.

## Fixed conditions and measurement scope

Every case uses 32 agents, small capacity (memory/affect/entity dimensions
16/4/16; 10,987 parameters), FP32, Appearance dimension 8, vocabulary four,
maximum message length three, initial Life 10, initial Points 3, renewable
Point probabilities 0.1–0.3 and horizon 16. Actor-Critic uses Adam at 0.001,
value coefficient 0.5 and action/message entropy coefficients 0.01 each.
Capacity, learning coefficients, precision and resource rules do not change
within the matrix. No application defaults or training algorithms are changed.

Count density requests `min(count, living // 2)` disjoint pairs each world step:
1/4/8/16 are maxima, not guaranteed realized encounters after deaths. All cases
start with seed 42; batched world seeds are `42 + row + update * world_count`.
The independent stability run uses seed 43. Different densities and batch sizes
change trajectories, episode objectives and optimizer update frequency per world;
throughput is not a measure of equivalent learning progress or policy quality.

Each case has one warmup and 20 measured complete-episode optimizer updates.
Rates divide total observed world steps, actual encounters or completed world
episodes by summed CUDA-synchronized update wall time. The timer includes reset,
collection, encounter counting, backward, training diagnostics and Adam; updated
parameter finiteness checks are outside the timer. Encounters count two action
decisions per pair, excluding communication and inactive worlds. Horizon-censored
episodes count as completed objectives. Warmup is excluded from rates and peaks.

`nvidia-smi` polls at 0.2 seconds plus command latency during the measured phase,
including between-update checks. Utilization is the arithmetic sample mean/max,
not kernel occupancy or time-integrated utilization. Sampled device memory includes
the desktop and CUDA context. Allocator peaks include existing process allocations
and reset each update. GiB means 2^30 bytes. Sampling overhead, NVIDIA's sampling
window (possibly overlapping warmup), fixed case order and desktop activity limit
fine-grained comparisons. No profiler or energy integral was used.

## Results

All 12 cases completed, with no OOMs, timeouts, telemetry errors or non-finite
losses, gradient norms or updated parameters across 240 measured updates
(35,840 world episodes).

| Requested pairs | Worlds | World steps/s | Encounters/s | Episodes/s | GPU % mean/max | Device GiB peak | Allocated/reserved GiB peak | Samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 3779.8 | 3527.8 | 292.8 | 38.9/44 | 1.32 | 0.36/0.38 | 21 |
| 1 | 128 | 7200.5 | 6721.3 | 558.9 | 40.5/47 | 1.82 | 0.71/0.87 | 22 |
| 1 | 256 | 13031.1 | 12182.2 | 1011.4 | 43.6/48 | 2.38 | 1.41/1.44 | 24 |
| 4 | 64 | 3617.4 | 12712.7 | 229.7 | 44.5/48 | 1.36 | 0.40/0.41 | 26 |
| 4 | 128 | 6986.6 | 24478.9 | 444.8 | 47.0/53 | 1.88 | 0.78/0.94 | 27 |
| 4 | 256 | 12881.1 | 45043.7 | 820.8 | 46.5/55 | 2.51 | 1.54/1.56 | 29 |
| 8 | 64 | 2914.2 | 21429.1 | 182.1 | 46.1/51 | 1.40 | 0.44/0.46 | 33 |
| 8 | 128 | 5665.3 | 41561.7 | 354.1 | 48.0/51 | 1.97 | 0.87/1.03 | 34 |
| 8 | 256 | 10166.3 | 74635.8 | 635.4 | 46.9/58 | 2.73 | 1.72/1.79 | 38 |
| 16 | 64 | 2093.8 | 30714.4 | 130.9 | 48.3/52 | 1.49 | 0.53/0.54 | 46 |
| 16 | 128 | 4057.9 | 59520.3 | 253.6 | 51.6/58 | 2.15 | 1.04/1.21 | 47 |
| 16 | 256 | 7560.0 | 110779.3 | 472.5 | 50.7/59 | 3.11 | 2.08/2.17 | 51 |

## Production choice and stability limits

For this small/FP32/horizon-16 workload, use **256 worlds** as the measured
production starting point at the density required by the experiment. It is the
fastest tested world count at every density, about 3.5–3.6 times the throughput
of 64 worlds, with at most 3.11 GiB sampled device use. Retain 128 or 64 worlds
when lower per-update latency or a smaller optimization batch is required.
There is substantial VRAM headroom, but counts above 256 were not measured here;
this is not a maximum-capacity or globally optimal batch-size claim.

Do not reduce density solely to increase world steps/s: at 256 worlds, moving
from one to 16 requested pairs lowers world steps/s from 13,031 to 7,560 while
raising actual encounters/s from 12,182 to 110,779. The latter performs much more
interaction work. Density is an experimental condition, not a free performance
knob. These results do not select which density learns the best policy.

The independent seed-43 run at 16 pairs and 256 worlds completed another 100
measured updates (25,600 episodes) over 54.50 measured seconds: 7,515 world
steps/s, 105,662 encounters/s and 469.7 episodes/s. All losses, gradient norms
and updated parameters remained finite. Loss ranged 838.94–1673.97 and gradient
norm 347.11–576.24; those magnitudes are diagnostics, not convergence evidence.
Its 253 telemetry samples averaged 50.4% GPU utilization (61% maximum), with
3.08 GiB peak sampled device memory and no telemetry errors. Allocated peaks
were 2.069 GiB on the first measured update, 2.049 on the last and 2.069 maximum;
reserved peaks were 2.139 / 2.141 / 2.141 GiB respectively. This shows bounded
allocator usage over this run, not a proof of leak freedom or long-duration
thermal stability.

The matrix plus stability run covers 340 measured updates and 61,440 episodes.
No multi-seed confidence intervals, long-horizon runs, BF16 comparisons or larger
capacities were measured. Rebenchmark those workloads before applying this
recommendation. GPU utilization below saturation is consistent with dispatch and
small-kernel overhead, but this measurement does not isolate the bottleneck.

## Validation

All 188 project tests passed with CUDA enabled and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (76.79 seconds). `compileall` and
`git diff --check` passed. Benchmark regression tests cover encounter counting,
the unchanged 33-case legacy matrix and the 12-case density matrix's fixed
32-agent configuration and invariant settings. Evidence checks recomputed every
rate from measured records, verified telemetry presence/error absence, checked
configuration equality except world/pair count (and seed/update budget for the
stability run), and matched the recorded runner SHA-256 to the source.
