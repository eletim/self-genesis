# RTX 5090 recurrent Thought benchmark

Measured on 2026-09-28 against `dev/v0.0.9` at `98ce206`, with the benchmark
extension in this change. [Raw evidence](evidence/rtx5090-recurrent-thought/results.json.gz)
retains resolved configurations, per-update results, telemetry, commands, timings,
source revision, runner SHA-256 and validation output. The recorded revision is
the training implementation's base; the hash identifies the extended runner.

## Reproduce

From the repository root, using a CUDA-capable PyTorch environment:

```sh
PYTHONPATH=src /tmp/self-genesis-verify-gpu/bin/python examples/benchmark_rtx5090.py \
  --sweep recurrent --capacity small --precision fp32 --world-counts 64 128 256 \
  --num-agents 32 --encounter-count 16 --updates 20 --warmup 1 --horizon 16 \
  --seed 42 --timeout 600 --output /tmp/rtx5090-recurrent-thought
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /tmp/self-genesis-verify-gpu/bin/python \
  -m unittest discover -s tests -v
/tmp/self-genesis-verify-gpu/bin/python -m compileall -q src tests examples
git diff --check
```

Use a new output directory. The sweep always runs 16 and 32 think steps for each
world count; `--include-64` additionally runs 64. A single worker accepts
`--worker --think-steps 64 --worlds 256` and the same workload options, with a
new JSON output path instead of a directory. The optional 64-step performance
case was not measured here. Scaling and density matrices remain available.

Hardware/software: RTX 5090, 32,607 MiB VRAM, 575 W power limit, driver
595.91.07; Python 3.12.14, PyTorch 2.7.1+cu128, CUDA runtime 12.8, one PyTorch
CPU thread, TF32 matmul disabled. Execution uses the default nondeterministic
kernels. CUDA device numbering was unmodified. The desktop was active; no other
training/test workload was run during measurement. Tests ran after the sweep.

## Fixed workload and gradient lifetime

All cases use small capacity (memory/affect/entity dimensions 16/4/16), 11,243
shared parameters, FP32, 32 agents, 16 requested disjoint encounter pairs per
world step, and horizon 16. Actual pairs decline with deaths. Initial Life is
10, Points 3, renewable Point probability 0.1–0.3, Appearance dimension 8,
vocabulary four and maximum message length three. Actor-Critic uses Adam at
0.001, value coefficient 0.5 and action/message entropy coefficients 0.01.
Only world count and think steps vary. Each isolated worker seeds PyTorch with
42 and world rows with `42 + row + update * world_count`.

Each case performs one warmup and 20 measured optimizer updates. Every update
collects complete episodes to death or the configured horizon, retains the full
recurrence and episode graph through backward, and only then detaches episode
state. The existing `train_batch` checks loss and every present gradient for
finiteness before Adam. The runner also requires nonempty, finite gradients,
finite gradient norm and finite updated parameters, including during warmup.
There is no clipping, recurrence truncation, mid-episode detach, retry with fewer
think steps, or reduced horizon. Training code is unchanged.

Failures during updates retain completed records, the failed update index and
exception in worker JSON, and exit unsuccessfully. The parent manifest records
stderr, exit failure or timeout and process wall time for each requested case.
Setup failures and hard timeouts may have only a manifest record, without partial
worker measurements. No failed case is reinterpreted as a successful shorter run.

## Measurements

Rates divide total world steps, actual encounters and completed world episodes by
summed CUDA-synchronized update wall time. Timing includes reset, full collection,
encounter counting, backward, diagnostics and Adam; the runner's extra gradient
and parameter checks are outside that timer. Worker wall time includes warmup,
those extra checks and telemetry shutdown, but excludes initial model/collector
setup. Full subprocess wall time is retained in the manifest.

Allocator peaks reset each update and include existing allocations; reported
peaks exclude warmup. `nvidia-smi` polls at 0.2 seconds plus command latency.
Its arithmetic utilization mean/max and sampled device VRAM include context,
desktop and between-update checks; utilization is not kernel occupancy. GiB is
2^30 bytes. NVIDIA's sampling window, fixed case order, nondeterministic kernels
and desktop activity limit precise comparisons.

| Think steps | Worlds | World steps/s | Encounters/s | Episodes/s | Measured/worker seconds | GPU % mean/max | Device GiB peak | Allocated/reserved GiB peak | Gradient norm range |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 1859.4 | 26833.2 | 116.2 | 11.01/11.86 | 52.0/71 | 1.82 | 0.861/0.879 | 389.98–460.05 |
| 32 | 64 | 1605.3 | 23165.3 | 100.3 | 12.76/13.68 | 52.8/69 | 2.18 | 1.213/1.234 | 389.98–460.05 |
| 16 | 128 | 3576.6 | 51584.4 | 223.5 | 11.45/12.33 | 49.9/71 | 2.82 | 1.706/1.877 | 390.34–457.79 |
| 32 | 128 | 3147.5 | 45395.8 | 196.7 | 13.01/13.98 | 53.5/81 | 3.52 | 2.411/2.574 | 390.34–457.79 |
| 16 | 256 | 6653.9 | 96070.1 | 415.9 | 12.31/13.23 | 55.2/77 | 4.47 | 3.442/3.529 | 391.28–458.57 |
| 32 | 256 | 5734.0 | 82788.0 | 358.4 | 14.29/15.31 | 58.3/85 | 5.91 | 4.872/4.971 | 391.28–458.57 |

All six cases completed: 120 measured updates, 17,920 world episodes, plus six
warmup updates. There were no OOMs, timeouts, telemetry errors or non-finite
losses, gradients or parameters. Each update had 22 populated gradient tensors.
Measured loss ranged from 1476.46 to 1693.70 and gradient norm from 389.98 to
460.05. Near-identical loss/norm values across depths at a fixed world count are
observations for this seed and workload, not evidence that recurrence was skipped
or that additional thinking improves learning.

At 256 worlds, 32 think steps delivered **5734.0 world steps/s**, versus 6653.9
at 16 steps, with allocated peaks of 4.872 versus 3.442 GiB. The 256-world cases
were fastest among tested batch sizes at each depth. Finite gradients over these
short runs establish numerical stability only for these conditions, not useful
credit assignment at every step, convergence, or long-duration stability. Other
seeds, capacities, BF16, longer horizons and 64-step throughput were not measured.
Existing Thought execution tests independently verify gradients through all
16/32/64 steps and across episode callbacks on CPU and CUDA.

## Validation

Benchmark regression tests cover depth configuration, matched workload settings,
unchanged legacy matrices, exact subprocess recurrence arguments, OOM/non-finite/
timeout reporting without fallback, and preservation of earlier update evidence
when a later update fails. Raw-evidence checks verify all rates from update
records, finite-gradient flags, telemetry presence, and the runner hash.

The CUDA-enabled full suite ran 210 tests in 95.25 seconds, with two failing
subtests in the existing BF16-versus-FP32 gradient tolerance test (Actor-Critic
and REINFORCE, message length three, horizon 32). Both failures reproduced with
identical observed error magnitudes against an untouched archive of base
`98ce206` (five mixed-precision tests, 5.35 seconds). They are retained in the
evidence and were not resolved by weakening tolerance or changing training.
All new benchmark tests and CPU/CUDA Thought gradient-lifetime tests passed.
`compileall` and `git diff --check` passed.

The final CPU-only suite passed all 210 tests (14 CUDA skips). The CPU-only
benchmark regression suite passed all eight tests, including the worker failure
fixture with CUDA runtime calls mocked out. The evidence retains the initial
fixture failure and final passing output.
