# Bounded v0.0.5 versus v0.0.6 Entity Memory validation

Recorded 2026-09-27 on Linux x86_64, Python 3.12.14, PyTorch 2.7.1+cpu.
Implementation: `865c61c5f5d9c6e6703d5371c3c015802258598f` (v0.0.6), with an
actual v0.0.5 run at `7292fc47735f1dca1e75b2ff11edd735c88a4947`.
No training or environment implementation changed for this experiment.

Entity Memory did not consistently improve held-out survival: paired seed
changes were -0.0833, +2.3333, and -1.9167 steps. Neither learned condition beat
always-GIVE. Entity Memory reset and both Appearance interventions had null
survival/GIVE effects. These results validate the procedure without establishing
improved cooperation, survival, reciprocity, or useful memory.

## Conditions and budget

Both conditions use survival-only Actor-Critic, Adam learning rate 0.001,
value coefficient 0.5 and action/message entropy coefficients 0.01. Four agents
start with Life 10 and Points 3, fixed generation abilities uniform in [0.1, 0.3],
Appearance dimension 8, vocabulary 4, message length 3, Working Memory 16,
affect 4, and horizon 100. Each trains **100 complete updates** for each of
**41, 42, 43**, then freezes weights and evaluates on **101, 102, 103**.
No tuning, seed selection, or performance stopping rule was used.

v0.0.6 separately trains Entity Memory dimensions 16 (`learned`) and zero
(`learned-no-entity-memory`). The latter reproduces v0.0.5: all 300 training
updates, including losses, and all shared evaluation behavioral fields match the
actual old implementation exactly. Architecture dimensions and parameter counts
differ between enabled and disabled conditions, so equal seeds do not mean equal
initial weights. This compares complete architectures, not an isolated causal
effect of memory capacity. Training resets reuse the configured population while
continuing random streams; held-out worlds have fresh populations and state.

Each learned condition receives untreated evaluation plus four **separate**
treatments: Appearance shuffle, Appearance replacement, Working Memory reset,
and Entity Memory reset. Three fixed/oracle baselines run untreated. Thus one
v0.0.6 report has 117 evaluations (3 training seeds × 3 worlds × 13 conditions).
The old implementation supports only shuffle and Working Memory reset and has
54 evaluations. The current disabled condition covers the two new treatments.

Each architecture has a training upper bound of 30,000 world steps. One current
run is bounded by 60,000 training and 11,700 evaluation steps; one historical run
by 30,000 and 5,400. Both are independently repeated once, giving total bounds
of **180,000 training and 34,200 evaluation steps**. Equal updates/horizons do not
mean equal realized steps or compute. Actual training steps:

| Condition | Seed 41 | Seed 42 | Seed 43 |
| --- | ---: | ---: | ---: |
| Entity Memory 16 | 1,486 | 1,619 | 1,380 |
| Disabled / actual v0.0.5 | 1,426 | 1,307 | 1,646 |

Evaluations restart matched worlds and sampling streams. Subsequent actions,
extinction, encounters, and draws may diverge. Fixed baselines repeat identically
across training seeds and versions; their repetitions are not independent evidence.
The oracle uses privileged true partner ability >=0.2 and positive, sending empty
messages. It is a heuristic, not an optimal policy or survival upper bound.

## Reproduction and retained evidence

Use the [pinned CPU environment recipe](minimal-experiment.md#small-cpu-run),
then run from this repository with that Python. Use a fresh output directory.
The archive avoids switching branches or modifying the current checkout.

```sh
export PYTHONPATH=src
mkdir /tmp/entity-memory-reproduction
mkdir /tmp/entity-memory-reproduction/v005
git archive 7292fc47735f1dca1e75b2ff11edd735c88a4947 | \
  tar -x -C /tmp/entity-memory-reproduction/v005
for repeat in 1 2; do
  python -m self_genesis compare --config configs/default.toml --device cpu \
    --training-method actor_critic --entity-memory-dim 16 \
    --episodes 100 --survival-horizon 100 --training-seeds 41 42 43 \
    --evaluation-seeds 101 102 103 \
    --interventions appearance-shuffle appearance-replacement working-memory-reset entity-memory-reset \
    --output "/tmp/entity-memory-reproduction/v006-$repeat.json"
  PYTHONPATH=/tmp/entity-memory-reproduction/v005/src python -m self_genesis compare \
    --config /tmp/entity-memory-reproduction/v005/configs/default.toml --device cpu \
    --training-method actor_critic --episodes 100 --survival-horizon 100 \
    --training-seeds 41 42 43 --evaluation-seeds 101 102 103 \
    --interventions appearance-shuffle working-memory-reset \
    --output "/tmp/entity-memory-reproduction/v005-$repeat.json"
done
for version in v005 v006; do
  cmp "/tmp/entity-memory-reproduction/$version-1.json" \
      "/tmp/entity-memory-reproduction/$version-2.json"
done
gzip -dc docs/evidence/entity-memory/v006.json.gz | \
  cmp - /tmp/entity-memory-reproduction/v006-1.json
gzip -dc docs/evidence/matched-learning/actor_critic.json.gz | \
  cmp - /tmp/entity-memory-reproduction/v005-1.json
python examples/verify_entity_memory_experiment.py \
  /tmp/entity-memory-reproduction/v005-1.json \
  /tmp/entity-memory-reproduction/v006-1.json > /tmp/entity-memory-reproduction/summary.json
```

Both independent repeats were byte-identical. The v0.0.5 rerun also exactly
matches the [existing historical report](evidence/matched-learning/actor_critic.json.gz),
so that evidence is reused. The [full v0.0.6 report](evidence/entity-memory/v006.json.gz)
retains all updates, evaluations, memory events, actions, messages, history bins,
initial populations, summaries, and paired effects. SHA-256 of decompressed JSON
bytes, including the trailing newline:

| Report | SHA-256 |
| --- | --- |
| v0.0.5 | `1bae8534acc38c4b8d5301c2d2bc85e49cdcaeb38e16bf2bee31962be135534c` |
| v0.0.6 | `aa6cb5ee411cf39926db14d1431386f2575b9807e9cb8532c985da412169fc92` |

Without retraining, run `PYTHONPATH=src python examples/verify_entity_memory_experiment.py`.
It checks matched settings/populations, training budgets, old-version behavioral
parity, fixed-baseline invariance, disabled Entity Memory reset as a no-op, and
recomputes every current summary from evaluation rows. It prints all pooled
history counts/rates/nulls and per-world intervention effects as readable JSON.
Do not use Python's `-O`, which disables these validation assertions.

## Held-out outcomes

Each learned row pools three held-out worlds, or 12 dependent agent lifetimes.
All 117 current evaluations ended in extinction, latest at step 22; none were
horizon-censored. Mean observed lifetime therefore equals completed lifetime here.
GIVE counts include failed attempts; aid counts only successful transfers.

| Condition | Training seed | Mean lifetime | GIVE / callbacks | Successful aid |
| --- | ---: | ---: | ---: | ---: |
| Memory 16 | 41 | 12.1667 | 27 / 74 | 26 |
| Disabled | 41 | 12.2500 | 28 / 74 | 27 |
| Memory 16 | 42 | 13.5833 | 53 / 82 | 43 |
| Disabled | 42 | 11.2500 | 16 / 70 | 15 |
| Memory 16 | 43 | 11.6667 | 21 / 70 | 20 |
| Disabled | 43 | 13.5833 | 52 / 82 | 43 |
| Always GIVE | — | 14.4167 | 92 / 92 | 53 |
| Always NOTHING | — | 10.0000 | 0 / 60 | 0 |
| Producer oracle | — | 13.4167 | 51 / 86 | 41 |

Equally weighted training-seed means are 12.4722 enabled and 12.3611 disabled,
a difference of +0.1111 steps. The signs vary across seeds; this tiny sample
does not establish an improvement. Every learned summary, treated or untreated,
is `mixed` under the <=0.05 / >=0.95 GIVE collapse thresholds. This is an
evaluation diagnostic, not proof against training collapse.

## Interventions and partner history

For both architectures, Appearance shuffle, Appearance replacement, and Entity
Memory reset leave lifetime and GIVE counts unchanged **at every matched world**.
All four treatments have null survival/GIVE effects on the disabled architecture.
Working Memory reset on the enabled architecture has pooled changes:

| Training seed | Lifetime delta | GIVE fraction delta (percentage points) |
| --- | ---: | ---: |
| 41 | 0 | 0 |
| 42 | 0 | -1.2195 |
| 43 | +0.0833 | +1.4286 |

These are treatment minus untreated, with GIVE differences computed from pooled
counts. The report also stores each world's ratio differences. State and message
changes can occur despite null action counts. Shuffle can have fixed points;
replacement samples fresh Appearance each step. Both preserve world identities.
Resets occur before each step, retain other state, and permit within-encounter
updates. Disabled Entity Memory reset is verified as an exact behavioral no-op.
See [the intervention semantics](../README.md#matched-policy-comparisons).

The following direct-history entries give positive-minus-zero GIVE percentage
points, with positive/zero callback counts in parentheses. Only repeat encounters
and successful aid from strictly earlier steps enter these two contrasts.
Producer columns give high-minus-low GIVE at first and repeat meetings, and then
repeat minus first, in percentage points.

| Condition / seed | Received aid | Outgoing aid | Producer first | Producer repeat | Repeat − first |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memory 16 / 41 | +3.19 (15/23) | -18.84 (15/23) | 0 | -25.49 | -25.49 |
| Disabled / 41 | -1.70 (16/22) | -12.50 (16/22) | +5.56 | -25.49 | -31.05 |
| Memory 16 / 42 | +24.94 (33/13) | +3.50 (33/13) | +11.11 | -32.95 | -44.07 |
| Disabled / 42 | -31.03 (5/29) | +39.31 (5/29) | -5.56 | -26.39 | -20.83 |
| Memory 16 / 43 | +5.38 (10/26) | +5.38 (10/26) | 0 | -13.75 | -13.75 |
| Disabled / 43 | +20.98 (32/14) | +0.45 (32/14) | +5.56 | -32.95 | -38.51 |

All untreated learned repeat-minus-first producer contrasts are negative. High
means strictly above the **sample population's** ability midrange (not the
oracle's configured threshold). These associations do not establish recognition
of ability or learning to favor producers. Direct aid signs vary, and bins can
be small (five positive callbacks for disabled seed 42).

Third-party history excludes both current participants and uses all encounters.
For memory-enabled seeds 41/42/43, prior partner outgoing successful aid to others
has GIVE differences -24.32/-4.89/-26.63 percentage points, with positive/zero
counts 37/37, 54/28, 24/46. Disabled differences are -23.68/-8.33/-6.75, counts
38/36, 18/52, 54/28. The retained summaries additionally cover incoming aid,
the focal agent's third-party history, and distinct attempt versus transfer bins.
All bins include denominators, unknown counts, missing-bin flags, and null rates
when empty; never interpret null as zero. Actual directed identities are used
only in analysis and routing, not supplied to learned policies.

## Checks and limits

The CPU suite ran 114 tests successfully, with seven CUDA skips. The evidence
verifier (including rejection of altered seeds, populations, and summaries),
independent repeat comparisons, compileall, `uv pip check --python <cpu-python>`, and whitespace
checks passed. The numerical experiment and tests here are CPU-only; no new CUDA
validation is claimed. Reproduction is scoped to the pinned software/device.

Three training seeds and three worlds in one resource regime, only 100 updates,
and extinction by step 22 do not establish convergence, long-run survival, or a
general architecture ranking. Equal seed labels do not pair initial parameters
across different architectures. Agents and callbacks within worlds are dependent;
no confidence or significance claim is made. Resources, roles, encounters, and
survival confound history associations. Null interventions do not prove state is
unused; other memory/affect and within-encounter updates remain available, and
sampled actions can mask distribution changes. Messages are fixed-length channel
usage, not evidence of semantic content or causal utility. Future horizon
survivors must remain censored; this sample cannot characterize their behavior.
No zero-generation control or learning-method sweep was rerun for this work item.
