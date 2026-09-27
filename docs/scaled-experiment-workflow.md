# Capacity, encounter density, Thought, throughput, and behavior workflow

v0.0.7 validates scalable execution of the existing survival experiment. It does
not establish that larger policies learn cooperation, reciprocity, useful
communication, or partner recognition. Keep three questions separate: numerical
reference parity, execution throughput, and held-out learned behavior. The
[reference checks](../README.md#batched-policy-forwards-and-capacity-presets)
and [RTX 5090 evidence](rtx5090-scaling.md) address the first two. The
[bounded v0.0.6 behavioral evidence](entity-memory-experiment.md) remains historical;
it is not a held-out result for large or batched v0.0.7 training.

For v0.0.8, [Issue #58](https://github.com/eletim/self-genesis/issues/58) supplies
the motivation: sparse encounters may limit the pressure to learn partner-specific
memory. Its description of earlier large-scale learning and memory use is not
validated by the repository's v0.0.7 throughput evidence. The retained
[controlled density results](controlled-density-comparison.md) below provide a
separate, bounded behavioral test. Encounter density is the sole experimental
change across densities within each learned architecture: keep network widths,
Actor-Critic coefficients, survival-only reward, renewable world, recurrent state
structures and communication fixed. Hidden traits and identity labels remain
analysis/oracle-only; no social reward or supervised target is added.

v0.0.9 delivers recurrent Thought in sequential and batched execution. The
[design contract](design-principles.md) and
[representative scenarios](representative-scenarios.md) describe callback-local
zero initialization, fixed observation/memory context, shared Linear → ReLU
iterations and frozen world time. Working Memory and affect update once after
the loop; Entity Memory retains callback/completion writes. Persistent states
remain private and reset at episode boundaries. Complete-episode gradients
cross all internal steps; thinking adds no decisions, rewards or world steps.

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

| Preset | Working Memory / thought | Affect | Entity Memory | Recurrent parameters | Shallow parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 16 | 4 | 16 | 11,243 | 10,987 |
| medium | 128 | 32 | 64 | 295,207 | 278,823 |
| large | 512 | 128 | 256 | 4,473,991 | 4,211,847 |

The script prints current recurrent counts; pass `thought_mode="shallow"` to
`from_preset` for historical shallow counts. Increasing recurrent depth adds no
parameters. These counts use Appearance dimension 8, vocabulary 4 and message length 3,
including the critic and both Entity Memory update cells. Private per-world,
per-agent recurrent state is not a parameter. Changing dimensions or disabling
Entity Memory changes the count; use the actual recorded `parameter_count`.
A preset changes widths, not observations, survival rewards or encounter rules.
Explicit dimensions override presets: avoid `configs/default.toml` for capacity
sweeps because its explicit small dimensions override the preset. The commands
below specify the renewable environment without that file.

## Reproduce the Thought experiments

After setup above, use an activated Python/PyTorch environment and fresh output
paths. TOML keys `thought_mode = "recurrent"` and `think_steps = 16` are the
current defaults. CLI `--thought-mode` and `--think-steps` override them for
initialization, training and comparison, independently of capacity presets.
Integer counts of at least 16 are valid (including 32/64); booleans, fractions
and smaller counts are rejected. Shallow mode uses one Linear → tanh transform
and ignores the validated count. Neither mode changes the world rules.

A portable CPU smoke comparison exercises shallow/16/32 and all four memory and
Appearance interventions with frozen sequential evaluation of trained weights:

```sh
PYTHONPATH=src python -m self_genesis compare --compare-thought --batched \
  --deterministic --device cpu --num-worlds 2 --episodes 2 \
  --num-agents 4 --encounter-fraction 1.0 --survival-horizon 8 \
  --point-generation-probability-max 0.5 \
  --training-seeds 7 17 --evaluation-seeds 101 102 \
  --output /tmp/thought-smoke.json
```

The matrix overrides individual mode/depth flags and retains Entity Memory in
all three architectures. This tiny budget checks execution, not learning.
Training seeds must be disjoint from evaluation seeds across the full schedule
`training_seed + update * num_worlds + row`. Reports contain configs, parameter
counts, world seeds, updates, held-out summaries, matched intervention effects
and sampled Thought diagnostics; training JSONL is not a checkpoint.

Verify the retained [three-seed evidence](controlled-thought-comparison.md) on
CPU without retraining:

```sh
PYTHONPATH=src python examples/verify_thought_comparison.py \
  > /tmp/thought-retained-summary.json
```

Reproduce the predeclared [configuration](../configs/controlled-thought.toml)
in a CUDA-capable environment, then verify the new reports:

```sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  examples/run_thought_comparison.py --output /tmp/controlled-thought-replay
PYTHONPATH=src python examples/verify_thought_comparison.py \
  /tmp/controlled-thought-replay > /tmp/thought-replay-summary.json
```

The runner fixes deterministic FP32, one CPU thread, small capacity, 32 agents,
16 pairs, horizon 32, 50 updates of 128 worlds per mode and training seeds
10000/20000/30000. Evaluation seeds are 100000/100001/100002. This is nine training
runs, 450 updates and 57,600 episodes, with a 1,200-second worker cap per training
seed (all three modes plus evaluation). Preserve manifests, failed cases, hashes
and compressed reports. The verifier checks budgets, seeds, finite diagnostics,
evaluation worlds, recomputed summaries/deltas and sampled internal steps; it
does not promise exact replay across software or devices.

Both recurrent depths yielded seed-level mean lifetimes
15.4688 / 15.4375 / 14.1146 versus shallow's 14.2292 / 14.9167 / 15.2500.
All stayed below always-GIVE (15.8021); 32 steps improved no measured survival,
aid or selection metric over 16. Entity Memory reset had zero recurrent lifetime
effect, while Appearance/Working Memory effects were small and mixed. Keep null
bins, censoring and denominators, and report each seed before pooling. Three
training seeds and three evaluation worlds are descriptive evidence; equal
episodes do not match compute or actual encounters. Shallow differs in activation,
initialization and parameter count, and no untrained control isolates learning.
Mixed actions and positive history associations do not prove cooperation or
useful memory; token usage does not prove causal communication utility.

Read [preselected stepwise dynamics](controlled-thought-dynamics.md) alongside
`thought_samples`. Probes replay fixed context without changing state or RNG;
intermediate logit/value readouts are hypothetical. The relative-change threshold
0.001 is diagnostic, never early stopping. Recurrent-32 representative probes
first met it at steps 19/13/14; this is neither universal convergence nor evidence
of useful deliberation. Normal training does not retain internal-step traces.

Measure full-episode GPU cost separately on an RTX 5090:

```sh
PYTHONPATH=src python examples/benchmark_rtx5090.py --sweep recurrent \
  --capacity small --precision fp32 --world-counts 64 128 256 \
  --num-agents 32 --encounter-count 16 --updates 20 --warmup 1 --horizon 16 \
  --seed 42 --timeout 600 --output /tmp/thought-throughput
```

The sweep runs 16/32 steps; `--include-64` adds an unmeasured optional depth.
[All six retained cases](rtx5090-recurrent-thought.md) had finite full-episode
losses/gradients/parameters. At 256 worlds, 16/32 measured 6,653.9 / 5,734.0 world
steps/s and 3.442 / 4.872 GiB peak allocated memory. This was the fastest tested
batch at both depths, not an optimum. One seed, short horizon, nondeterministic
kernels and desktop telemetry limit generalization. The benchmark's full CUDA
suite also retained two pre-existing BF16 gradient-tolerance failures reproduced
on its base revision; FP32 Thought checks passed. No BF16 fix or long-run
stability claim follows from these measurements. Worker time in the behavioral
comparison includes evaluation/compression and is not GPU training throughput.

## Compare execution throughput

Run the 33-case sequential/batched, capacity, FP32/BF16 matrix using the existing
runner, which records warmup separately, failures/timeouts, update measurements,
telemetry and commands. On this branch it uses recurrent Thought (16 steps by
default); the table below is historical shallow v0.0.7 evidence, not the expected
output of this current command:

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

## Reproduce the controlled density experiment

After environment setup above, use a CUDA-capable PyTorch interpreter and fresh
output directories. The retained run used Python 3.12.14, PyTorch 2.7.1+cu128 and
CUDA 12.8 on an RTX 5090. This section documents historical shallow evidence;
run its verifier and reproduction commands from evidence commit `9bb64ec`,
which adds the verifier to the implementation recorded in the
[retained manifest](evidence/controlled-density/manifest.json). Its verifier
compares reports against historical configuration defaults, which differ from
this branch. Verify existing evidence on CPU without retraining:

```sh
PYTHONPATH=src:. python examples/verify_density_comparison.py \
  > /tmp/density-retained-summary.json
```

The commands below are the historical v0.0.8 reproduction procedure.
On this branch the density runner inherits recurrent Thought defaults, so it
does not reproduce the shallow evidence or its parameter counts. For the
current matched architecture experiment use the Thought procedure above.
The historical procedure uses the fixed [configuration](../configs/controlled-density.toml):

```sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  examples/run_density_comparison.py --output /tmp/controlled-density-replay
PYTHONPATH=src:. python examples/verify_density_comparison.py \
  /tmp/controlled-density-replay > /tmp/density-replay-summary.json
```

The runner fixes one CPU thread, deterministic FP32, small capacity, 32 agents,
128 worlds, 50 complete-batch optimizer updates and horizon 32. It varies only
requested pairs (1/4/8/16), independently training enabled and disabled Entity
Memory at seeds 10000/20000/30000. Compare densities within each architecture;
disabling memory changes parameter count (10,987 to 3,051). World seeds are
`seed + update * 128 + row`; held-out seeds 100000/100001/100002 are disjoint.
All four memory/Appearance interventions and always-GIVE, always-NOTHING and
producer-oracle baselines use matched evaluation worlds. Actual frozen weights
are evaluated sequentially in FP32; training logs are not checkpoints.

The declared budget is 24 training runs, 1,200 updates, 153,600 world episodes
and 468 evaluations, with a 900-second limit per density. Equal episodes do not
mean equal encounters or compute. Preserve the manifest, failures, compressed
reports and hashes; the verifier checks settings, seeds, exposure counts,
recomputed summaries and matched intervention deltas. It validates report
consistency, not exact cross-version numerical replay.

For current operating measurements, run the separate shorter-horizon matrix.
It now uses recurrent Thought; the retained measurements below used shallow:

```sh
python examples/benchmark_rtx5090.py --sweep density \
  --capacity small --precision fp32 --world-counts 64 128 256 \
  --output /tmp/density-throughput --updates 20 --warmup 1 --horizon 16 --seed 42
```

The [12-case density benchmark and 100-update stability run](rtx5090-density.md)
support **256 worlds for small/FP32/horizon 16** as the fastest tested batch at
every density, not a global optimum. At 16 pairs, measured throughput was 7,560
world steps/s and 110,779 encounters/s, sampled GPU utilization 50.7% mean/59%
maximum and peak device memory 3.11 GiB (allocated/reserved 2.08/2.17 GiB).
All measured updates remained finite with no OOM or timeout; the stability run
showed bounded allocator use, not long-duration stability. Keep 128 worlds for
the declared horizon-32 behavioral reproduction. Larger capacities, BF16, longer
horizons and tracing require their own measurements.

The [retained behavioral reports and per-seed tables](controlled-density-comparison.md)
show main learned mean lifetime rising from 10.3646 at one pair to 14.7986 at
16 pairs, with more repeat encounters. Every main learned seed stayed below
always-GIVE at its density. Both architectures remained mixed-GIVE; Entity
Memory reset gave zero lifetime delta in 11 of 12 main learned seed/density
summaries and +0.0104 in the other. Appearance and Working Memory effects were
mixed, with no consistent producer/history selection. Density changes both
training and evaluation opportunities, so these gains do not isolate learning
from additional aid opportunities. Three training seeds, sparse history bins,
within-encounter updates under reset and trajectory divergence under interventions
limit interpretation. Token entropy measures usage, not useful communication.

Report null and collapsed outcomes without selecting favorable seeds or extending
the budget after viewing results. GIVE fractions at or below 0.05 are
near-always-NOTHING, at or above 0.95 near-always-GIVE; otherwise mixed. No action
callbacks and absent history bins are missing evidence, not zero selection.
Retain denominators, successful aid, censoring, first/repeat producer contrasts,
prior-aid contrasts and every intervention delta. Mixed actions alone do not
establish learned selection, and null interventions do not prove memory is unused.

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
