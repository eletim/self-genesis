# RTX 5090 training scaling

Measured on an actual NVIDIA GeForce RTX 5090 on 2026-09-27 using the
`dev/v0.0.7` training implementation at `8faffc553a01b8102ad3f93753e3278f4c329192`.
The benchmark adds instrumentation only; it does not change training behavior.
Raw per-update results, sampled telemetry, commands, process wall times and
software metadata are in [the compressed evidence](evidence/rtx5090/results.json.gz).

## Reproduce

Install the project with a CUDA-capable PyTorch build, then run from the repository:

```sh
PYTHONPATH=src python examples/benchmark_rtx5090.py --output /tmp/rtx5090-scaling --updates 20 --warmup 1 --horizon 16 --seed 42
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

The output directory must be new. Each of the 33 cases uses a fresh process;
failures and 300-second timeouts are recorded in `manifest.jsonl` and the sweep
continues. GPU 0 must be an RTX 5090; leave CUDA device numbering unmodified so
PyTorch and `nvidia-smi --id=0` measure the same device. No parallel benchmark or
test workloads ran during the final sweep. The GPU also drove the desktop.

Hardware: 32,607 MiB reported VRAM, 575 W power limit, driver 595.91.07,
AMD Ryzen 9 9950X3D CPU. Software: Python 3.12.14, PyTorch 2.7.1+cu128,
CUDA runtime 12.8. One PyTorch CPU thread; default nondeterministic execution;
TF32 matrix multiplication disabled. No compilation, profiler, or detailed traces.

## Workload and measurement scope

Every case uses Actor-Critic, four agents, Appearance dimension 8, three-token
messages, vocabulary size four, enabled Entity Memory, initial Life 10, initial
Points 3, renewable Point probabilities 0.1–0.3, and survival horizon 16.
Small/medium/large use the existing capacity presets. FP32 is the baseline;
BF16 uses the existing autocast path with FP32 parameters, recurrent storage,
critic, reductions and optimizer. The sequential API supports FP32 only.

Each case executes one warmup and 20 measured optimizer updates. Every batched
update trains one complete episode per world; sequential updates train one
complete episode. Sampling streams, trajectories, loss averaging and optimizer
update frequency differ across execution modes: rates compare execution throughput,
not equivalent learning progress or time to convergence. Each case starts at seed
42; there are no independent-seed replicates or confidence intervals.

Rates divide total measured work by summed CUDA-synchronized update wall time,
including reset, collection, encounter counting, backward, diagnostics and Adam.
Steps mean completed **world** steps, not agent steps or batch ticks. Encounters
are counted from the two actual action decisions per encounter (including NOTHING);
lone-survivor steps do not count. Episodes mean complete world objectives, including
horizon-censored episodes. A counting wrapper adds one batched host reduction per
collection and scalar decision traversal. Full updated-parameter finiteness checks
run after the timed region; loss and gradient diagnostics come from training.

GPU utilization, device memory used and power are polled with `nvidia-smi` every
0.2 seconds plus command latency throughout the measured phase, including gaps
between updates and parameter checks. These are device-wide sampled readings,
not per-process attribution, integrated energy or instantaneous kernel occupancy.
The polling thread has overhead and NVIDIA's own utilization sampling window.
Allocator peaks include preexisting process allocations and are reset per update.
Warmup is excluded from rates and table peaks. Process wall time includes Python,
CUDA/network/optimizer initialization, warmup, checks, polling and serialization;
it is deliberately reported separately from measured training wall time.

## Results

All 33 cases completed 20 measured updates after one warmup without non-finite
losses, gradient norms or updated parameters. No OOMs or timeouts occurred.
This demonstrates bounded repeated-update stability at 64, 128 and 256 worlds,
and successful probes at 512 and 1024, for all three capacities and both precisions.

### Throughput and wall time

Speedup is world steps/s divided by the same-capacity sequential FP32 rate.

| Capacity | Precision | Worlds | Steps/s | Encounters/s | Episodes/s | Speedup | Training s | Process s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | fp32 | sequential | 248.9 | 226.5 | 16.6 | 1.0× | 1.21 | 2.93 |
| small | fp32 | 64 | 4425.8 | 4066.5 | 291.1 | 17.8× | 4.40 | 6.27 |
| small | fp32 | 128 | 8479.9 | 7814.3 | 557.2 | 34.1× | 4.59 | 6.54 |
| small | fp32 | 256 | 15706.9 | 14424.7 | 1032.4 | 63.1× | 4.96 | 6.94 |
| small | fp32 | 512 | 27284.3 | 25054.2 | 1795.0 | 109.6× | 5.70 | 7.66 |
| small | fp32 | 1024 | 44524.2 | 40927.6 | 2923.3 | 178.9× | 7.01 | 9.04 |
| small | bf16 | 64 | 4314.6 | 3964.1 | 283.8 | 17.3× | 4.51 | 6.55 |
| small | bf16 | 128 | 7939.4 | 7315.7 | 521.5 | 31.9× | 4.91 | 6.92 |
| small | bf16 | 256 | 15306.7 | 14056.9 | 1005.9 | 61.5× | 5.09 | 7.09 |
| small | bf16 | 512 | 26863.9 | 24666.7 | 1767.6 | 107.9× | 5.79 | 7.79 |
| small | bf16 | 1024 | 43809.8 | 40265.1 | 2876.4 | 176.0× | 7.12 | 9.19 |
| medium | fp32 | sequential | 250.0 | 229.0 | 19.9 | 1.0× | 1.00 | 2.76 |
| medium | fp32 | 64 | 4470.2 | 4158.8 | 291.6 | 17.9× | 4.39 | 6.26 |
| medium | fp32 | 128 | 8464.9 | 7782.1 | 557.6 | 33.9× | 4.59 | 6.56 |
| medium | fp32 | 256 | 15389.4 | 14088.9 | 1023.4 | 61.6× | 5.00 | 6.99 |
| medium | fp32 | 512 | 26827.2 | 24590.5 | 1779.7 | 107.3× | 5.75 | 7.79 |
| medium | fp32 | 1024 | 44491.9 | 41404.5 | 2874.3 | 178.0× | 7.13 | 9.26 |
| medium | bf16 | 64 | 4328.9 | 4026.9 | 282.4 | 17.3× | 4.53 | 6.45 |
| medium | bf16 | 128 | 8201.0 | 7553.4 | 540.0 | 32.8× | 4.74 | 6.75 |
| medium | bf16 | 256 | 15063.7 | 13802.4 | 1000.8 | 60.3× | 5.12 | 7.08 |
| medium | bf16 | 512 | 26273.9 | 24041.7 | 1759.8 | 105.1× | 5.82 | 7.84 |
| medium | bf16 | 1024 | 44317.7 | 41260.5 | 2861.3 | 177.3× | 7.16 | 9.28 |
| large | fp32 | sequential | 239.8 | 221.5 | 18.3 | 1.0× | 1.09 | 2.78 |
| large | fp32 | 64 | 4548.2 | 4335.3 | 285.9 | 19.0× | 4.48 | 6.33 |
| large | fp32 | 128 | 7595.3 | 7073.4 | 607.5 | 31.7× | 4.21 | 6.27 |
| large | fp32 | 256 | 15574.1 | 14825.6 | 984.9 | 64.9× | 5.20 | 7.20 |
| large | fp32 | 512 | 26966.1 | 25868.4 | 1694.8 | 112.4× | 6.04 | 8.05 |
| large | fp32 | 1024 | 41722.6 | 39974.2 | 2622.6 | 174.0× | 7.81 | 9.95 |
| large | bf16 | 64 | 4452.1 | 4246.6 | 279.9 | 18.6× | 4.57 | 6.53 |
| large | bf16 | 128 | 7622.9 | 7026.4 | 586.0 | 31.8× | 4.37 | 6.33 |
| large | bf16 | 256 | 15707.2 | 14951.4 | 992.8 | 65.5× | 5.16 | 7.17 |
| large | bf16 | 512 | 27864.0 | 26704.9 | 1751.7 | 116.2× | 5.85 | 7.90 |
| large | bf16 | 1024 | 43024.3 | 41296.5 | 2703.4 | 179.4× | 7.58 | 9.81 |

### GPU resource observations

Utilization and power show arithmetic sample mean / maximum. VRAM columns
show maximum sampled device usage and maximum measured-update allocator peaks.
GiB uses 2^30 bytes; device readings include desktop and CUDA context overhead.

| Capacity | Precision | Worlds | GPU % mean/max | Power W mean/max | Device GiB | Allocated GiB | Reserved GiB | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | fp32 | sequential | 18.0/30 | 89.2/110.7 | 0.97 | 0.02 | 0.02 | 6 |
| small | fp32 | 64 | 38.7/45 | 111.3/116.9 | 0.98 | 0.03 | 0.04 | 21 |
| small | fp32 | 128 | 41.5/47 | 114.4/117.2 | 1.00 | 0.05 | 0.05 | 22 |
| small | fp32 | 256 | 37.1/41 | 114.9/117.3 | 1.03 | 0.08 | 0.09 | 24 |
| small | fp32 | 512 | 39.9/50 | 115.8/118.0 | 1.10 | 0.15 | 0.16 | 27 |
| small | fp32 | 1024 | 37.5/48 | 117.1/118.9 | 1.24 | 0.29 | 0.29 | 33 |
| small | bf16 | 64 | 38.9/44 | 114.7/117.4 | 0.98 | 0.03 | 0.03 | 22 |
| small | bf16 | 128 | 33.7/39 | 113.0/116.7 | 0.99 | 0.04 | 0.04 | 23 |
| small | bf16 | 256 | 37.4/42 | 114.3/117.4 | 1.01 | 0.06 | 0.07 | 24 |
| small | bf16 | 512 | 40.3/46 | 115.5/118.0 | 1.06 | 0.11 | 0.12 | 27 |
| small | bf16 | 1024 | 36.7/47 | 116.4/118.7 | 1.16 | 0.21 | 0.21 | 34 |
| medium | fp32 | sequential | 17.2/28 | 98.3/107.2 | 0.97 | 0.02 | 0.03 | 5 |
| medium | fp32 | 64 | 38.8/47 | 113.1/118.5 | 1.05 | 0.10 | 0.11 | 21 |
| medium | fp32 | 128 | 41.7/49 | 115.9/118.8 | 1.13 | 0.18 | 0.18 | 22 |
| medium | fp32 | 256 | 37.9/43 | 117.5/120.3 | 1.28 | 0.33 | 0.34 | 24 |
| medium | fp32 | 512 | 41.6/50 | 121.4/124.3 | 1.60 | 0.65 | 0.66 | 27 |
| medium | fp32 | 1024 | 35.9/46 | 129.3/133.3 | 2.33 | 1.36 | 1.38 | 34 |
| medium | bf16 | 64 | 37.6/45 | 115.1/118.0 | 1.02 | 0.07 | 0.08 | 22 |
| medium | bf16 | 128 | 38.7/46 | 115.2/118.1 | 1.07 | 0.13 | 0.13 | 23 |
| medium | bf16 | 256 | 44.2/48 | 116.3/119.2 | 1.18 | 0.23 | 0.23 | 24 |
| medium | bf16 | 512 | 41.1/48 | 118.8/121.8 | 1.39 | 0.43 | 0.45 | 28 |
| medium | bf16 | 1024 | 39.8/51 | 124.7/127.7 | 1.87 | 0.90 | 0.92 | 34 |
| large | fp32 | sequential | 20.8/31 | 104.7/119.3 | 1.06 | 0.10 | 0.12 | 6 |
| large | fp32 | 64 | 43.8/48 | 123.8/131.0 | 1.37 | 0.40 | 0.43 | 21 |
| large | fp32 | 128 | 40.2/47 | 130.8/136.9 | 1.69 | 0.70 | 0.74 | 20 |
| large | fp32 | 256 | 46.7/55 | 147.0/154.8 | 2.40 | 1.40 | 1.45 | 25 |
| large | fp32 | 512 | 49.7/62 | 170.0/181.9 | 3.75 | 2.73 | 2.81 | 28 |
| large | fp32 | 1024 | 56.6/68 | 214.2/227.4 | 6.35 | 5.26 | 5.40 | 37 |
| large | bf16 | 64 | 39.8/47 | 120.4/123.9 | 1.27 | 0.30 | 0.32 | 22 |
| large | bf16 | 128 | 40.4/45 | 122.1/126.8 | 1.46 | 0.49 | 0.52 | 21 |
| large | bf16 | 256 | 42.2/50 | 131.0/136.0 | 1.92 | 0.95 | 0.97 | 24 |
| large | bf16 | 512 | 39.6/51 | 143.5/150.1 | 2.81 | 1.83 | 1.87 | 28 |
| large | bf16 | 1024 | 50.8/60 | 167.7/175.7 | 4.56 | 3.55 | 3.62 | 36 |

## Interpretation and measured limitations

The presets contain 10,987 / 278,823 / 4,211,847 parameters. Across the 33 cases,
660 measured optimizer updates completed 238,140 world episodes; the sum of
case process wall times was 232.30 seconds. Every capacity/precision combination
completed 1,280 / 2,560 / 5,120 measured episodes at 64 / 128 / 256 worlds.
The corresponding 512 / 1024 probes completed 10,240 / 20,480 episodes each.
These are finite-update stability observations, not convergence results.

At 64 worlds, FP32 throughput was 4,426–4,548 world steps/s, or 17.8–19.0×
the sequential baseline. At 256 it was 15,389–15,707 steps/s (61.6–64.9×).
At 1024 it reached 41,723–44,524 steps/s (174.0–178.9×). Increasing world count
16-fold from 64 to 1024 yielded only about 9–10× throughput: scaling is already
sublinear in this workload, although throughput still improves through 1024.

BF16 did not improve small or medium throughput in this sweep. Large capacity
at 1024 worlds gained about 3.1% throughput, with peak allocated memory falling
from 5.26 to 3.55 GiB and mean sampled power from 214.2 to 167.7 W. This single
fixed-order sweep cannot distinguish small speed differences from noise; it does
show the memory reduction. It does not justify changing the FP32 default.

The largest observed device-memory sample was 6.35 GiB, far below physical VRAM;
there is no measured OOM boundary here. Large FP32 at 1024 worlds averaged 56.6%
GPU utilization (68% maximum), and maximum sampled power was 227.4 W versus the
575 W limit. Small/medium capacity throughput was similar despite different
parameter counts. Together these observations are consistent with substantial
host dispatch, reset, synchronization and small-kernel overhead, but no profiler
was used to isolate a bottleneck. They do not prove a particular CPU or GPU limit.

Only four agents and horizon 16 were measured. Complete recurrent graphs retain
state until backward; larger populations, longer lifetimes/horizons, more Entity
Memory slots, tracing and different workloads may substantially increase VRAM and
wall time. BF16 retains much of the state in FP32. No long-duration thermal test,
energy integral, repeated-seed performance experiment, or beyond-1024-world run
was performed. Telemetry has only 5–6 samples for sequential cases and 20–37 for
batched cases; utilization can include the preceding warmup sampling window.
Fixed case order, desktop activity, polling overhead and changing trajectories
limit fine-grained precision/capacity comparisons. The report makes no claim of
improved cooperation, survival, policy quality or equivalent optimizer progress.

## Validation

All 163 project tests passed with CUDA enabled using
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; the full log is embedded in the evidence archive.
The initial CUDA-suite invocation without that required setting failed four
configuration checks and was rerun correctly. The CPU-only run passed 162 tests
(13 skipped); the subsequently added empty-encounter case also passed in the
final CUDA suite. `compileall` and `git diff --check` passed. Benchmark-specific
tests verify encounter counts exclude messages, inactive worlds and lone-survivor
steps, and handle batches without decisions. Evidence validation recomputed every
rate from per-update counts and times, checked all 33 cases and telemetry streams,
and verified the runner SHA-256 against the committed source file.
