# Controlled shallow / recurrent Thought comparison

## Predeclared protocol

Fixed before execution on 2026-09-28: use the existing matched Thought comparison
with shallow, recurrent-16 and recurrent-32, each trained from seeds
10000/20000/30000. Every run receives 50 complete-batch updates of 128 worlds,
32 agents, 16 disjoint encounter pairs per step and horizon 32. Small capacity
16/4/16, FP32, deterministic execution, Actor-Critic coefficients, renewable
resources and survival-only reward are fixed in `configs/controlled-thought.toml`.
Only Thought mode/depth varies. Dimensions are matched; shallow and recurrent
parameter counts need not match. Recurrent-16/32 share the same architecture
and initialization seed, with weights shared across internal steps.

World seeds are `training_seed + update * 128 + row`: disjoint training ranges
10000–16399, 20000–26399, 30000–36399, paired across modes. Held-out evaluation
seeds **100000/100001/100002** are fixed and outside all training ranges.
Frozen sequential FP32 evaluation restarts worlds and sampling streams for
untreated policies, Entity Memory reset, Appearance shuffle, Appearance
replacement and Working Memory reset. Always-GIVE, always-NOTHING and producer
oracle are paired fixed references, repeated per seed, not independent replicates.

Budget: nine training runs, 450 updates, 57600 training world episodes, at most
1843200 training world steps; 162 evaluation episodes, at most 5184 evaluation
world steps. One subprocess per training seed, each capped at 1200 seconds
including all three modes and evaluation; total worker budget 3600 seconds.
No warmup training, retries, seed selection, coefficient tuning, shortened runs
or outcome-dependent extensions. Failures are retained in the manifest and the
remaining declared seeds still run. A failed worker may leave only an incomplete
report; it is never counted as a completed run.

Report survival/censoring, GIVE/NOTHING collapse, successful aid, encounter/repeat
exposure, all direct/third-party history bins, producer first/repeat contrasts,
all matched intervention effects and communication usage, including null bins.
The existing deterministic probe rule samples the first communication and action
callbacks at world steps 0/1/2, including interventions (at most six per episode).
Retain every internal step's norm, change, cosine, relative change, convergence,
saturation, action logits/changes and value/changes. Representative narrative
samples will use the first untreated held-out world and first action callback
for each mode and training seed, regardless of outcomes. Convergence tolerance
is 0.001, not a stopping rule. Neither convergence nor channel usage implies utility.

Primary comparisons use equal updates/episodes, not equal compute. Whole-worker
wall time and device-wide utilization/VRAM telemetry include sequential evaluation
and must not be interpreted as training throughput. The separate
[recurrent throughput benchmark](rtx5090-recurrent-thought.md) measures GPU
training at horizon 16 and nondeterministic settings; it cannot establish learning
gains here. Report actual training steps/encounters separately. No untrained
checkpoint comparison is planned: final differences cannot isolate improvement
from initialization. Three training seeds and three evaluation worlds provide
bounded descriptive evidence, not robust statistical significance.

Reproduce from the repository root in a CUDA PyTorch environment:

```sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /tmp/self-genesis-verify-gpu/bin/python \
  examples/run_thought_comparison.py --output /tmp/controlled-thought
```

Output must be a new directory. The runner writes the resolved protocol manifest
before workers start, retains compressed reports with SHA-256 digests, records
source revision and runner hash, and samples NVIDIA device telemetry every
0.2 seconds plus command latency. Training and world rules are unchanged.
