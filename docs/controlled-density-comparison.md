# Controlled encounter-density comparison

The bounded comparison completed all four densities and all three training seeds.
**More encounters improved survival and repeat exposure, but did not establish
useful partner-specific memory.** Main learned mean lifetime across seeds rose
from 10.3646 to 11.5035, 12.8472 and 14.7986 steps at 1/4/8/16 pairs. Every main
learned seed remained below always-GIVE at its density (10.8438, 13.2083, 15.0729,
15.8021). Both learned architectures remained in the mixed-GIVE category.

At 16 pairs, main learned agents averaged 14.0000–15.1042 encounters and
2.8125–3.1250 repeats per episode, versus 0.6875–0.8125 encounters and
0.0417–0.1042 repeats at one pair. Entity Memory reset had zero lifetime effect
in 11 of 12 density/training-seed summaries and **increased** lifetime by 0.0104
in the remaining one. Identity disruption had small, mixed effects. Working
Memory reset also had mixed effects, without a consistent survival penalty.
Repeat-minus-first producer selection at 16 pairs was +0.0112 / +0.0285 / −0.0074;
prior-aid GIVE differences were +0.0317 / −0.0310 / +0.0938. These do not show
consistent density-driven selection. Sparse one-pair history bins make their
large contrasts especially unreliable.

This is evidence of greater interaction opportunity and survival under denser
worlds, not evidence that partner-specific memory caused the gain. Null or small
intervention effects do not prove memory cannot become useful with another
budget. No longer training, seed selection, coefficient tuning or reward changes
were introduced after observing these results.

## Predeclared protocol

Declared before execution on 2026-09-28. Compare requested maxima of 1/4/8/16
pairs per world step, with 32 agents and identical small capacity, FP32,
Actor-Critic, survival-only reward, and renewable resource world. Full settings
are in `configs/controlled-density.toml`; no tuning or outcome-based stopping.

Each density trains independently from initialization seeds 10000/20000/30000,
for exactly 50 complete-batch updates of 128 worlds, horizon 32. Each seed uses
world seeds `seed + update * 128 + row`: disjoint ranges 10000–16399,
20000–26399, 30000–36399. The same initialization/world seeds are paired across
densities. Evaluation seeds 100000/100001/100002 are outside every training
range. They are reused across frozen policies and interventions, with fresh
world and policy state. Evaluation uses the sequential FP32 reference at the
training density and horizon. Sampling streams restart; trajectories can diverge.

The existing comparison also independently trains the no-Entity-Memory control
at each seed. Its capacity differs by design; compare densities within each
architecture. Main learned capacity stays 16/4/16 (Working Memory/affect/Entity
Memory); the control is 16/4/0. Fixed always-GIVE, always-NOTHING and producer
oracle baselines receive the same evaluation worlds. Run all four interventions:
Entity Memory reset, Appearance shuffle, Appearance replacement, Working Memory
reset, on both learned architectures.

Budget: 24 training runs, 1200 optimizer updates, 153600 training world episodes,
at most 4915200 world steps; 468 evaluation episodes (including repeated fixed
baselines), at most 14976 evaluation world steps. Each density has a 900-second
wall timeout including evaluation; total worker budget 3600 seconds. Failures
are recorded, never silently replaced with shorter runs. Equal updates/episodes
are the primary budget, not equal interaction exposure or compute. Report actual
world steps and encounters from training, plus held-out exposure distributions.

Primary interpretation: increased survival alone does not establish memory use.
Look for increased repeats, history-conditioned GIVE, repeat-versus-first producer
selection, and matched degradation under Entity Memory and identity interventions.
Retain empty-bin nulls and callback counts. Three training seeds are descriptive
replicates, not enough to claim robust statistical significance. Horizon survivors
are right censored. Communication entropy measures usage, not utility.

Reproduce from the repository root in a CUDA PyTorch environment:

```sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /tmp/self-genesis-verify-gpu/bin/python \
  examples/run_density_comparison.py --output /tmp/controlled-density
```

Output directory must be new. The manifest is written before any worker starts.
Raw compressed reports preserve every update, evaluation, exposure distribution,
behavior summary, matched intervention effect and intervention semantics. Wall
measurement includes training and sequential evaluation; it is not GPU training
throughput. See [the density benchmark](rtx5090-density.md) for separately measured
training throughput, utilization and VRAM under its explicitly shorter horizon.

## Evidence and scope

The predeclaration is committed at `094102c`; the executed source revision is
`bd4f3e9`, which adds actual action-callback and encounter totals to existing
training diagnostics. This adds no policy inputs, rewards, optimizer changes,
or world-rule changes. The enabled network has 10987 parameters; the independently
trained disabled-memory control has 3051. Both counts stay fixed across densities.

Retained reports: [1 pair](evidence/controlled-density/pairs-1.json.gz),
[4 pairs](evidence/controlled-density/pairs-4.json.gz),
[8 pairs](evidence/controlled-density/pairs-8.json.gz),
[16 pairs](evidence/controlled-density/pairs-16.json.gz), and
[execution manifest](evidence/controlled-density/manifest.json).
Decompress with `gzip -dc` to inspect JSON. Each report includes:

- `training_runs`: resolved per-run configuration, initialization seed, complete
  world-seed schedule, parameter count and every update's losses, entropy,
  gradient norm, world steps, endings, returns, actual encounters/action callbacks.
- `evaluations`: initial worlds, lifetimes/deaths/censoring, final resources,
  attempted actions and successful aid, messages/token usage, Entity Memory
  events, directed actual-partner action histories and per-agent exposure.
- `summaries`: every policy/intervention/training-seed combination, pooled exposure
  and same-partner distributions, GIVE collapse, prior direct and third-party
  aid/attempt bins, first/repeat high/low producer selection, communication usage.
- `intervention_effects` and `intervention_semantics`: all matched survival,
  censoring, GIVE, prior-aid and repeat-minus-first producer-selection deltas,
  with missing-bin nulls preserved and exact intervention definitions.

Density changes the opportunities to transfer resources even for fixed policies.
Training and evaluation both use the selected density; this is not a cross-density
transfer test, nor a comparison against untrained checkpoints. It therefore does
not isolate learning gains from density's direct mechanical survival benefit.
The disabled-memory control is a different architecture, not an equal-parameter
ablation. Within-step Entity Memory updates remain possible during its reset
intervention, and other recurrent state can retain history. Identity interventions
can perturb observations and trajectories independently of memory utility.

## Results

Each vector below is ordered by training seed **10000 / 20000 / 30000**. Each
seed summary pools the same three held-out worlds (96 agent-episodes). Rates
pool callbacks within each seed; they are not averages of episode rates.

### Actual training exposure

| Pairs | Policy | Actual world steps | Actual encounters |
| --- | --- | --- | --- |
| 1 | learned | 79308 / 78573 / 83122 | 74601 / 73905 / 77655 |
| 1 | learned-no-entity-memory | 85630 / 83697 / 81344 | 79792 / 78150 / 76264 |
| 4 | learned | 98535 / 97251 / 104448 | 335616 / 331223 / 351684 |
| 4 | learned-no-entity-memory | 111701 / 106868 / 100784 | 375456 / 359362 / 341250 |
| 8 | learned | 112236 / 110476 / 121541 | 718882 / 704928 / 775310 |
| 8 | learned-no-entity-memory | 125012 / 123062 / 116450 | 801869 / 788654 / 742030 |
| 16 | learned | 124809 / 124191 / 131017 | 1469813 / 1453526 / 1558204 |
| 16 | learned-no-entity-memory | 132270 / 131422 / 129809 | 1581917 / 1570046 / 1536433 |

Every row represents 19200 world episodes and 150 updates. An encounter has
two action callbacks; communication callbacks are excluded from this count.

### Held-out survival and behavior

| Pairs | Policy | Mean lifetime | GIVE fraction | Successful aid | Censored / 96 |
| --- | --- | --- | --- | --- | --- |
| 1 | learned | 10.2292 / 10.3333 / 10.5312 | 0.3333 / 0.4848 / 0.6538 | 22 / 32 / 51 | 0 / 0 / 0 |
| 1 | learned-no-entity-memory | 10.5833 / 10.5312 / 10.3750 | 0.7125 / 0.6538 / 0.5294 | 56 / 51 / 36 | 0 / 0 / 0 |
| 1 | always-GIVE | 10.8438 / 10.8438 / 10.8438 | 1.0000 / 1.0000 / 1.0000 | 81 / 81 / 81 | 0 / 0 / 0 |
| 1 | always-NOTHING | 10.0000 / 10.0000 / 10.0000 | 0.0000 / 0.0000 / 0.0000 | 0 / 0 / 0 | 0 / 0 / 0 |
| 1 | producer-oracle | 10.6979 / 10.6979 / 10.6979 | 0.6731 / 0.6731 / 0.6731 | 67 / 67 / 67 | 0 / 0 / 0 |
| 4 | learned | 11.1146 / 11.6146 / 11.7812 | 0.3497 / 0.4787 / 0.5210 | 107 / 155 / 171 | 0 / 0 / 0 |
| 4 | learned-no-entity-memory | 12.6250 / 12.1042 / 11.3438 | 0.7833 / 0.6170 / 0.4172 | 252 / 202 / 129 | 0 / 0 / 0 |
| 4 | always-GIVE | 13.2083 / 13.2083 / 13.2083 | 1.0000 / 1.0000 / 1.0000 | 308 / 308 / 308 | 0 / 0 / 0 |
| 4 | always-NOTHING | 10.0000 / 10.0000 / 10.0000 | 0.0000 / 0.0000 / 0.0000 | 0 / 0 / 0 | 0 / 0 / 0 |
| 4 | producer-oracle | 12.6458 / 12.6458 / 12.6458 | 0.6795 / 0.6795 / 0.6795 | 254 / 254 / 254 | 0 / 0 / 0 |
| 8 | learned | 12.3438 / 12.5312 / 13.6667 | 0.3704 / 0.4142 / 0.5620 | 225 / 243 / 352 | 0 / 0 / 0 |
| 8 | learned-no-entity-memory | 14.3021 / 13.9375 / 12.9375 | 0.7337 / 0.6257 / 0.4666 | 413 / 378 / 282 | 0 / 0 / 0 |
| 8 | always-GIVE | 15.0729 / 15.0729 / 15.0729 | 1.0000 / 1.0000 / 1.0000 | 487 / 487 / 487 | 0 / 0 / 0 |
| 8 | always-NOTHING | 10.0000 / 10.0000 / 10.0000 | 0.0000 / 0.0000 / 0.0000 | 0 / 0 / 0 | 0 / 0 / 0 |
| 8 | producer-oracle | 14.6146 / 14.6146 / 14.6146 | 0.6800 / 0.6800 / 0.6800 | 443 / 443 / 443 | 0 / 0 / 0 |
| 16 | learned | 14.2292 / 14.9167 / 15.2500 | 0.3713 / 0.4475 / 0.4979 | 406 / 472 / 504 | 0 / 0 / 0 |
| 16 | learned-no-entity-memory | 15.6458 / 15.4688 / 14.9792 | 0.7016 / 0.6562 / 0.4676 | 542 / 525 / 478 | 0 / 0 / 0 |
| 16 | always-GIVE | 15.8021 / 15.8021 / 15.8021 | 1.0000 / 1.0000 / 1.0000 | 557 / 557 / 557 | 0 / 0 / 0 |
| 16 | always-NOTHING | 10.0000 / 10.0000 / 10.0000 | 0.0000 / 0.0000 / 0.0000 | 0 / 0 / 0 | 0 / 0 / 0 |
| 16 | producer-oracle | 15.5417 / 15.5417 / 15.5417 | 0.6306 / 0.6306 / 0.6306 | 532 / 532 / 532 | 0 / 0 / 0 |

Fixed baseline entries repeat across training seeds because their policies do
not train; these are paired references, not additional independent samples.

### Exposure and selection for the main learned architecture

| Pairs | Seed | Encounters/agent | Repeats/agent | Repeat fraction | Prior-aid GIVE Δ | First producer Δ | Repeat producer Δ | Repeat − first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10000 | 0.6875 | 0.0417 | 0.0606 | -1.0000 | -0.0230 | 0.0000 | 0.0230 |
| 1 | 20000 | 0.6875 | 0.0417 | 0.0606 | -0.6667 | -0.0021 | 0.0000 | 0.0021 |
| 1 | 30000 | 0.8125 | 0.1042 | 0.1282 | -0.2222 | 0.0667 | 0.2857 | 0.2190 |
| 4 | 10000 | 3.1875 | 0.2500 | 0.0784 | 0.2889 | -0.0765 | -0.1111 | -0.0346 |
| 4 | 20000 | 3.4167 | 0.2083 | 0.0610 | 0.1667 | -0.0684 | -0.0330 | 0.0354 |
| 4 | 30000 | 3.4792 | 0.3125 | 0.0898 | 0.2381 | -0.0614 | 0.0679 | 0.1293 |
| 8 | 10000 | 6.7500 | 0.7500 | 0.1111 | 0.1181 | 0.0437 | 0.0151 | -0.0287 |
| 8 | 20000 | 6.9167 | 0.6667 | 0.0964 | 0.1808 | 0.0186 | -0.1064 | -0.1250 |
| 8 | 30000 | 7.7292 | 1.1458 | 0.1482 | 0.0667 | -0.0346 | 0.1076 | 0.1422 |
| 16 | 10000 | 14.0000 | 2.8125 | 0.2009 | 0.0317 | -0.0189 | -0.0077 | 0.0112 |
| 16 | 20000 | 14.6875 | 2.8750 | 0.1957 | -0.0310 | -0.0042 | 0.0243 | 0.0285 |
| 16 | 30000 | 15.1042 | 3.1250 | 0.2069 | 0.0938 | 0.0069 | -0.0005 | -0.0074 |

Producer differences are high-minus-low GIVE fractions. Prior-aid differences
compare previously aided with previously encountered but unaided partners.
All are observational and can reflect resources, survival selection and small
bins. Full bin counts, direct/third-party histories, per-agent encounter/repeat
distributions and same-partner count distributions (including zeros) are in
the linked raw reports; null denotes an unsupported contrast.

### Matched interventions

Deltas are intervention minus untreated; negative survival deltas mean harm.
Lifetime vectors average the three matched evaluation-seed effects for each
training seed. Full per-evaluation GIVE, censoring and selection deltas remain
in `intervention_effects` in each report.

| Pairs | Policy | Intervention | Lifetime Δ |
| --- | --- | --- | --- |
| 1 | learned | appearance-shuffle | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned | appearance-replacement | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned | working-memory-reset | 0.0000 / -0.0104 / -0.0208 |
| 1 | learned-no-entity-memory | appearance-shuffle | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned-no-entity-memory | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned-no-entity-memory | appearance-replacement | 0.0000 / 0.0000 / 0.0000 |
| 1 | learned-no-entity-memory | working-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 4 | learned | appearance-shuffle | 0.0000 / -0.0208 / 0.0000 |
| 4 | learned | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 4 | learned | appearance-replacement | 0.0000 / 0.0000 / 0.0000 |
| 4 | learned | working-memory-reset | 0.1979 / -0.1562 / 0.0000 |
| 4 | learned-no-entity-memory | appearance-shuffle | 0.0104 / 0.0000 / 0.0208 |
| 4 | learned-no-entity-memory | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 4 | learned-no-entity-memory | appearance-replacement | -0.0104 / 0.0000 / 0.0000 |
| 4 | learned-no-entity-memory | working-memory-reset | -0.1250 / -0.0625 / 0.0104 |
| 8 | learned | appearance-shuffle | -0.0104 / 0.0312 / -0.0417 |
| 8 | learned | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 8 | learned | appearance-replacement | -0.0208 / -0.0312 / -0.0417 |
| 8 | learned | working-memory-reset | 0.1354 / -0.0938 / 0.0000 |
| 8 | learned-no-entity-memory | appearance-shuffle | 0.0312 / 0.0000 / -0.0104 |
| 8 | learned-no-entity-memory | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 8 | learned-no-entity-memory | appearance-replacement | 0.0729 / 0.0000 / 0.0000 |
| 8 | learned-no-entity-memory | working-memory-reset | -0.2812 / -0.1875 / 0.0208 |
| 16 | learned | appearance-shuffle | 0.0000 / 0.0312 / 0.0000 |
| 16 | learned | entity-memory-reset | 0.0000 / 0.0000 / 0.0104 |
| 16 | learned | appearance-replacement | 0.0000 / 0.0000 / 0.0104 |
| 16 | learned | working-memory-reset | 0.3021 / -0.3958 / -0.0417 |
| 16 | learned-no-entity-memory | appearance-shuffle | 0.0104 / 0.0000 / 0.0208 |
| 16 | learned-no-entity-memory | entity-memory-reset | 0.0000 / 0.0000 / 0.0000 |
| 16 | learned-no-entity-memory | appearance-replacement | -0.0417 / 0.0000 / 0.1250 |
| 16 | learned-no-entity-memory | working-memory-reset | -0.1875 / 0.0000 / 0.2292 |

### Communication and collapse

| Pairs | Policy | GIVE collapse | Mean message length | Token entropy (bits) |
| --- | --- | --- | --- | --- |
| 1 | learned | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.9730 / 1.9659 / 1.7068 |
| 1 | learned-no-entity-memory | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.8679 / 1.8619 / 1.9439 |
| 4 | learned | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.9332 / 1.9567 / 1.4579 |
| 4 | learned-no-entity-memory | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.9627 / 1.8468 / 1.9569 |
| 8 | learned | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.8648 / 1.9203 / 1.5717 |
| 8 | learned-no-entity-memory | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.9509 / 1.9068 / 1.9197 |
| 16 | learned | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.8529 / 1.9431 / 1.4937 |
| 16 | learned-no-entity-memory | mixed / mixed / mixed | 3.0000 / 3.0000 / 3.0000 | 1.9151 / 1.8677 / 1.8778 |


## Execution and validation

All 24 training runs completed the declared 50 updates: 1200 updates and 153600
world episodes in total, with **2593556 actual world steps and 16256570 actual
encounters** (32513140 action callbacks). All 468 evaluations completed. All
untreated policy summaries had zero censored lifetimes; intervention-specific
endings and censoring are retained in the reports. No density timed out or failed.
End-to-end case times, including evaluation and compression, were 102.9 / 167.6 /
245.9 / 400.7 seconds, below the declared per-worker limits. The CPU test suite
ran during part of the experiment; these times are execution records, not isolated
performance benchmarks. Hardware/software: RTX 5090, Python 3.12.14, PyTorch
2.7.1+cu128, CUDA 12.8, one worker CPU thread, deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. No new GPU utilization or allocator-throughput
claim is made; the separate density benchmark supplies those measurements.

Validation commands:

```sh
PYTHONPATH=src:. /tmp/self-genesis-verify-gpu/bin/python \
  examples/verify_density_comparison.py > /tmp/density-verified-summary.json
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /tmp/self-genesis-verify-gpu/bin/python -m unittest discover -s tests -v
PYTHONPATH=src:tests CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /tmp/self-genesis-verify-gpu/bin/python -m unittest \
  test_density_training_parity test_training_logging test_encounter_density -v
/tmp/self-genesis-verify-gpu/bin/python -m compileall -q src tests examples
git diff --check
```

The full CPU suite ran 188 tests successfully (14 CUDA-only skips). All 16
focused tests passed with CUDA available, covering actual multi-pair counts,
scalar counts, zero encounters, training/trace parity and training logging.
[CPU log](evidence/controlled-density/tests-cpu.log) and
[CUDA log](evidence/controlled-density/tests-cuda.log) retain the test output.
The evidence verifier passed: report hashes, runner hash, fixed configurations,
parameter counts, complete disjoint seed schedules, budgets, actual exposure
bounds, matched initial worlds, no-memory reset invariance, recomputed summaries
and every reported intervention delta. `compileall` and `git diff --check` passed.
