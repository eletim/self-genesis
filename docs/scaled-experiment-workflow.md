# v0.0.7 capacity, throughput, and behavior workflow

v0.0.7 validates scalable execution of the existing survival experiment. It does
not establish that larger policies learn cooperation, reciprocity, useful
communication, or partner recognition. Keep three questions separate: numerical
reference parity, execution throughput, and held-out learned behavior. The
[reference checks](../README.md#batched-policy-forwards-and-capacity-presets)
and [RTX 5090 evidence](rtx5090-scaling.md) address the first two. The
[bounded v0.0.6 behavioral evidence](entity-memory-experiment.md) remains historical;
it is not a held-out result for large or batched v0.0.7 training.

## Record conditions and verify capacity

Use the [environment setup](minimal-experiment.md), run from the repository root,
and choose a new output directory. Record `git rev-parse HEAD`, Python/PyTorch/CUDA
versions, GPU/driver, device, precision, thread count, all commands, seeds,
configuration, and output checksums with each experiment. Pin the same software
for repeats; identical seed labels do not guarantee cross-device or cross-shape
trajectories. Leave benchmark GPU 0 numbering unchanged.

```sh
export PYTHONPATH=src
python - <<'PY'
from self_genesis.policy import RecurrentPolicy
for preset in ('small', 'medium', 'large'):
    policy = RecurrentPolicy.from_preset(8, preset)
    print(preset, policy.parameter_count)
PY
```

| Preset | Working Memory / thought | Affect | Entity Memory | Shared parameters |
| --- | ---: | ---: | ---: | ---: |
| small | 16 | 4 | 16 | 10,987 |
| medium | 128 | 32 | 64 | 278,823 |
| large | 512 | 128 | 256 | 4,211,847 |

These counts use Appearance dimension 8, vocabulary 4 and message length 3,
including the critic and both Entity Memory update cells. Private per-world,
per-agent recurrent state is not a parameter. Changing dimensions or disabling
Entity Memory changes the count; use the actual recorded `parameter_count`.
A preset changes widths, not observations, survival rewards or encounter rules.
Explicit dimensions override presets: avoid `configs/default.toml` for capacity
sweeps because its explicit small dimensions override the preset. The commands
below specify the renewable environment without that file.

## Compare execution throughput

Reproduce all 33 sequential/batched, capacity, FP32/BF16 cases using the existing
runner, which records warmup separately, failures/timeouts, update measurements,
telemetry and commands:

```sh
python examples/benchmark_rtx5090.py --output /tmp/scaled-throughput \
  --updates 20 --warmup 1 --horizon 16 --seed 42
```

This runner requires an RTX 5090. Inspect every `manifest.jsonl` status before
comparing results. Use the [measurement definitions and full tables](rtx5090-scaling.md)
for rates, allocator peaks, utilization and power. On the recorded four-agent,
horizon-16 workload, FP32 measured:

| Capacity | Sequential world steps/s | 64 worlds steps/s | 256 worlds steps/s | 1024 worlds steps/s |
| --- | ---: | ---: | ---: | ---: |
| small | 248.9 | 4,425.8 | 15,706.9 | 44,524.2 |
| medium | 250.0 | 4,470.2 | 15,389.4 | 44,491.9 |
| large | 239.8 | 4,548.2 | 15,574.1 | 41,722.6 |

Use summed completed world steps divided by summed measured update seconds,
not an unweighted mean of per-update rates. Report encounters/s, episodes/s,
training and process wall times, peak allocated/reserved memory, and sampled
utilization/power alongside throughput. Speedup uses the same-capacity sequential
FP32 denominator. More worlds per update changes the amount of experience per
optimizer step: equal updates do not mean equal episodes, compute, or learning.

The retained sweep used one warmup and 20 measured updates, one seed and fixed
case order. It found no OOM through 1024 worlds; this is not an OOM boundary.
BF16 did not improve small/medium throughput; large at 1024 gained about 3.1%
and reduced peak allocated memory from 5.26 to 3.55 GiB. FP32 remains the default.
Repeat sweeps with independent seeds and varied case order before interpreting
small performance differences. Larger populations, horizons and tracing need
new measurements; these short runs do not establish long-run stability.

For a portable execution smoke test, use CPU FP32 and a small world count:

```sh
python -m self_genesis train --batched --num-worlds 2 --capacity-preset small \
  --device cpu --seed 42 --deterministic --episodes 2 \
  --initial-life 3 --initial-points 1 --survival-horizon 8 \
  --point-generation-probability-min 0.1 --point-generation-probability-max 0.3 \
  --trace-worlds 0 --output /tmp/scaled-smoke.jsonl
```

For larger training, explicitly select capacity, device, world count, horizon,
precision and update budget. Increase one budget at a time. Aggregate metrics
include observed survival, GIVE choice fraction, value error, entropy and gradient
norm. Timing includes diagnostics and sampled trace I/O; match tracing settings
when comparing rates. `batch_trace` snapshots are not complete relationship
histories and cannot be passed to `examples/analyze_run.py`.

## Held-out capacity and Entity Memory comparison

The `compare` command defaults to sequential FP32 training. Add `--batched`
and `--num-worlds` to train batched policies and evaluate their actual frozen
weights through the same sequential FP32 evaluator. Batched comparisons require
explicit held-out evaluation seeds disjoint from all training world seeds. See
[the density evaluation command](../README.md#batched-trained-density-evaluation)
for a reproducible example. Training JSONL does not save a loadable checkpoint;
comparison reruns training in memory. The procedure below uses sequential capacity
variants, not the policies from the throughput sweep.

Choose budgets and disjoint training/evaluation seeds before viewing results.
This small two-update command is a smoke test, not a convergence experiment:

```sh
mkdir /tmp/scaled-behavior
for capacity in small medium large; do
  python -m self_genesis compare --capacity-preset "$capacity" \
    --device cpu --no-batched --mixed-precision fp32 --episodes 2 \
    --num-agents 4 --appearance-dim 8 --initial-life 10 --initial-points 3 \
    --vocabulary-size 4 --max-message-length 3 --survival-horizon 100 \
    --point-generation-probability-min 0.1 --point-generation-probability-max 0.3 \
    --training-method actor_critic --learning-rate 0.001 \
    --value-loss-coefficient 0.5 --action-entropy-coefficient 0.01 \
    --message-entropy-coefficient 0.01 --training-seeds 41 42 43 \
    --evaluation-seeds 101 102 103 \
    --interventions appearance-shuffle appearance-replacement working-memory-reset entity-memory-reset \
    --output "/tmp/scaled-behavior/$capacity.json"
done
```

For a longer preregistered comparison, replace `--episodes 2` with the same
budget in every condition (the historical experiment used 100). Each capacity
trains enabled and disabled Entity Memory independently per seed: 12 training
episodes per capacity in this smoke test, and 117 evaluation episodes. Report
actual world steps and runtime as well as updates and episodes. Disabled memory
changes parameter count and initialization, so it compares architectures rather
than isolating memory with equal capacity. Equal seeds do not pair initial weights.
Keep held-out seeds out of tuning; use a separate validation set if tuning.
Always-GIVE, always-NOTHING and the privileged producer oracle share the evaluation
world conditions. Repeated fixed baselines across training seeds are not independent
samples. Preserve censored survivors and compare per-training-seed results before
pooling; dependent agents/callbacks are not independent replicates.

## Analyze partner-specific behavior and memory

Read schema-v3 comparison JSON directly. Retain `training_runs`, `evaluations`,
`summaries`, `intervention_effects` and their conditions. For every seed and
condition report survival, censoring, GIVE attempts/action callbacks, successful
transfers and baseline differences. Empty bins are null, not zero.

1. Use `relationship_actions` and `partner_history_metrics` to separate first
   meetings from repeats, and repeat partners with/without earlier received or
   outgoing help. Separate GIVE attempts from successful aid and include bin
   denominators. Histories use strictly earlier steps, even for the second actor.
2. Compare partner production and third-party aid bins separately from direct
   history. True generation ability, realized production and actual partner IDs
   are analysis-only. Resources, roles, survival and encounter opportunities can
   confound these associations; they do not prove reciprocity or producer recognition.
3. Inspect `entity_memory_events` by episode, observer, observed Appearance and
   callback phase, including `completion`. Check zero retrieval for unseen keys,
   persistence at repeat meetings, collision sharing, and completion writes after
   outcomes. Compare retrieved values and before/after entries with subsequent
   choices. Slot numbers and latent coordinates have no prescribed social meaning.
4. Pair each frozen learned condition's untreated evaluation with each separate
   Appearance shuffle/replacement, Working Memory reset and Entity Memory reset
   treatment at the same training/evaluation seeds. Use `intervention_effects`
   for survival and GIVE differences and inspect memory/message changes too.
   Treatments preserve weights and rewards; trajectories can diverge after actions.
   Resets occur before steps, allow within-encounter updates and retain other
   state. Disabled Entity Memory reset is a useful no-op control.

Appearance replacement can grow the number of memory entries and full reports
can become large. Use short analysis runs before increasing horizons. Null
interventions do not prove unused memory; nonzero effects do not alone establish
partner-specific social function. Token frequencies do not establish meaning or
causal communication utility. Report null, mixed and collapsed behavior as valid
outcomes. The [historical analysis and verifier](entity-memory-experiment.md)
provide a worked example, not expected results for the new capacity sweep.

## Validation before scaling

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m unittest discover -s tests -v
python -m compileall -q src tests examples
python -m pip check
```

Reference tests cover world rules, ordered encounters, recurrent/Entity Memory
state, loss, gradients and optimizer updates under controlled trajectories.
CPU-only passes leave CUDA/BF16 cases unvalidated; run on the target GPU too.
Numerical parity is tolerance-based and distinct from seeded sampling equivalence,
throughput or learned behavior. Keep validation logs with experiment artifacts.
