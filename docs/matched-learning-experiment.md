# Validated v0.0.5 matched learning comparison

Recorded 2026-09-26 using implementation commit
`8487da27cad2010c4eef6787d7bb32cce584c305` on `dev/v0.0.5`.
Actor-Critic did not improve held-out survival over legacy REINFORCE in this
sample. Both had mixed GIVE rates for every training seed; this is a descriptive
validation, with no requirement for improved survival or reduced collapse.

## Conditions and bounds

The experiment reuses `compare` without changing training, evaluation, world
rules, or metrics. Both methods retain the v0.0.4 environment via `configs/default.toml`:
four agents, Life 10, Points 3, independent fixed generation abilities uniform
in [0.1, 0.3], Appearance dimension 8, vocabulary 4, message length 3, memory
16, affect 4, and survival horizon 100. Adam uses learning rate 0.001.

Each method trains independently with seeds **41, 42, 43**, for **100 complete
updates per seed**, then freezes the network and evaluates on held-out seeds
**101, 102, 103**. The same architecture and initialization procedure are used
for both objectives. Actor-Critic uses detached value advantages, value-loss
coefficient 0.5, and action/message entropy coefficients 0.01 each. Legacy
REINFORCE uses survival returns directly and ignores those three coefficients;
it runs in the current implementation, not an old binary. No social reward,
ability input, or identity label is supplied to either learner.

The budget is matched by updates and horizon, not realized steps or wall time:
policy-dependent extinction changes episode lengths. Each method is bounded by
30,000 training world steps and 5,400 evaluation world steps (54 evaluations:
three training seeds × three held-out seeds × six conditions). A complete
independent repeat of both methods checks reproducibility. Total bounds across
both methods and both repeats are 120,000 training and 21,600 evaluation steps.
Actual training steps per seed were:

| Method | 41 | 42 | 43 |
| --- | ---: | ---: | ---: |
| Actor-Critic | 1,426 | 1,307 | 1,646 |
| REINFORCE | 1,486 | 1,573 | 1,561 |

The six evaluation conditions are untreated learned, always-GIVE, always-NOTHING,
producer oracle, learned with Appearance shuffle, and learned with Working Memory
reset. All start with fresh recurrent state and matched worlds and sampling
streams. Changed actions and survival can subsequently change encounters and
random draws. Baselines repeat identically across training seeds and methods;
they are shown once below and are not independent training-seed replicates.

The oracle attempts GIVE to partners with true generation probability at least
0.2 (the positive range midpoint); it sends empty messages. This privileged
heuristic is not an optimal policy. Appearance shuffle permutes all original
Appearance vectors before each world step, with an isolated seed+3 RNG; actual
partner routing is retained. Working Memory reset zeros memory before each step,
but retains affect and allows within-encounter updates. Treatments are separate,
only at evaluation, and do not retrain weights. Full semantics are also stored
in the reports and described in [the comparison guide](../README.md#matched-policy-comparisons).

## Reproduction and retained evidence

Recorded on Linux x86_64, Python **3.12.14**, PyTorch **2.7.1+cpu**. Use the
[pinned CPU environment recipe](minimal-experiment.md#small-cpu-run), then run
from this checkout with `PYTHONPATH=src`. The two full schema-v2 reports are
retained as gzip files, including every training update, evaluation, action
history, message, summary, initial population, and paired intervention effect:

- [Actor-Critic report](evidence/matched-learning/actor_critic.json.gz)
- [REINFORCE report](evidence/matched-learning/reinforce.json.gz)

Use a fresh directory; the command refuses existing output files. Explicit
method flags are essential because the default is now Actor-Critic.

```sh
export PYTHONPATH=src
mkdir /tmp/matched-learning-reproduction
for repeat in 1 2; do
  for method in actor_critic reinforce; do
    python -m self_genesis compare --config configs/default.toml --device cpu \
      --training-method "$method" --value-loss-coefficient 0.5 \
      --action-entropy-coefficient 0.01 --message-entropy-coefficient 0.01 \
      --episodes 100 --survival-horizon 100 \
      --training-seeds 41 42 43 --evaluation-seeds 101 102 103 \
      --interventions appearance-shuffle working-memory-reset \
      --output "/tmp/matched-learning-reproduction/$method-$repeat.json"
  done
done
for method in actor_critic reinforce; do
  cmp "/tmp/matched-learning-reproduction/$method-1.json" \
      "/tmp/matched-learning-reproduction/$method-2.json"
  gzip -dc "docs/evidence/matched-learning/$method.json.gz" | \
    cmp - "/tmp/matched-learning-reproduction/$method-1.json"
done
```

Both independent repeats were byte-identical. SHA-256 of the **decompressed JSON
bytes**, including the trailing newline:

| Method | SHA-256 |
| --- | --- |
| Actor-Critic | `1bae8534acc38c4b8d5301c2d2bc85e49cdcaeb38e16bf2bee31962be135534c` |
| REINFORCE | `1cdfaa944da138a6e4a22ac8dd54c94da121e61b2ca2d61d71aa735ed8ee8216` |

To inspect the retained summaries without training again:

```sh
python - <<'PY'
import gzip, json
from pathlib import Path
for path in sorted(Path('docs/evidence/matched-learning').glob('*.json.gz')):
    report = json.loads(gzip.decompress(path.read_bytes()))
    print(path.name)
    for row in report['summaries']:
        print(row['training_seed'], row['policy'], row['intervention'],
              row['mean_observed_lifetime'], row['action_counts'],
              row['give_collapse'], row['prior_aid_give_difference'],
              row['communication']['token_entropy_bits'])
PY
```

## Untreated evaluation results

Each row pools three held-out worlds (12 agent lifetimes) within one training
seed. GIVE denominators count encounter action callbacks, including failed
attempts; aid counts successful transfers. All 54 evaluations per method went
extinct, latest at step 22: there were no horizon-censored lifetimes. Thus the
observed means here are also completed lifetimes. These are dependent agents
within worlds, not 12 independent experimental replicates.

| Method / policy | Training seed | Mean lifetime | GIVE / callbacks | Successful aid |
| --- | ---: | ---: | ---: | ---: |
| Actor-Critic | 41 | 12.2500 | 28 / 74 | 27 |
| Actor-Critic | 42 | 11.2500 | 16 / 70 | 15 |
| Actor-Critic | 43 | 13.5833 | 52 / 82 | 43 |
| REINFORCE | 41 | 12.9167 | 37 / 82 | 35 |
| REINFORCE | 42 | 13.4167 | 47 / 80 | 41 |
| REINFORCE | 43 | 13.6667 | 51 / 82 | 44 |
| Always GIVE | — | 14.4167 | 92 / 92 | 53 |
| Always NOTHING | — | 10.0000 | 0 / 60 | 0 |
| Producer oracle | — | 13.4167 | 51 / 86 | 41 |

The equally weighted training-seed means are 12.3611 for Actor-Critic and
13.3333 for REINFORCE, a paired difference of -0.9722 world steps. Per-seed
differences are -0.6667, -2.1667, and -0.0833. Neither method exceeds always-GIVE
for any training seed. The oracle is between the two fixed baselines and is
not a survival upper bound.

All six untreated learned summaries and all twelve treated learned summaries
are `mixed`. The existing collapse thresholds are GIVE fraction <=0.05 for
near-always-NOTHING and >=0.95 for near-always-GIVE; no callbacks would yield
`no_actions`. Neither learned method exhibits this diagnostic collapse here.
That does not establish reduced collapse risk or characterize training dynamics.

## Interventions, history, and communication

Below, lifetime changes are intervention minus untreated, averaged over the
three matched held-out seeds. GIVE changes are differences of pooled callback
fractions in percentage points; these are not averages of the per-world ratio
changes stored in `intervention_effects`. Actor-Critic changes are zero for both
treatments at every matched evaluation seed, not merely zero on average.

| Method | Training seed | Shuffle lifetime Δ | Shuffle GIVE Δ (pp) | Memory reset lifetime Δ | Memory reset GIVE Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor-Critic | 41 | 0 | 0 | 0 | 0 |
| Actor-Critic | 42 | 0 | 0 | 0 | 0 |
| Actor-Critic | 43 | 0 | 0 | 0 | 0 |
| REINFORCE | 41 | 0 | 0 | 0 | 0 |
| REINFORCE | 42 | +0.0833 | -0.2134 | -0.0833 | -2.6524 |
| REINFORCE | 43 | -0.0833 | -1.2195 | 0 | 0 |

Strictly prior successful aid dependence is the GIVE fraction toward previously
aiding partners minus that toward encountered-but-unaided partners, excluding
unseen partners. Untreated differences (percentage points) are -1.70, -31.03,
+20.98 for Actor-Critic seeds 41–43 and -4.35, +12.50, +7.59 for REINFORCE.
The reports retain the counts and null handling for each relationship group.
These small observational groups do not establish reciprocity, ability inference,
or stable partner selection; resources, role, time, and survival can confound them.

Learned messages always have three tokens by architecture. Untreated empirical
token entropies (bits) are 1.8878, 1.8093, 1.6403 for Actor-Critic and 1.9400,
1.9951, 1.9449 for REINFORCE. All fixed/oracle messages are empty with null token
entropy. Some interventions change token counts even when survival and GIVE
counts stay identical (for example Actor-Critic seed 43 memory reset raises
entropy from 1.6403 to 1.8909). Channel usage does not demonstrate useful
communication or token meaning; neither intervention directly removes messages.

## Validation and limitations

Checks in the recorded environment:

- Both method reports replayed byte-for-byte. Conditions differ only in training
  method; seed lists, initial populations, update counts, and horizon bounds
  were checked. Fixed/oracle outcomes match across methods and training seeds.
  An additional run of each method through the documented CLI also matched the
  retained bytes (the original runs used the same `run_comparison` Python API).
- CPU suite: 91 tests, seven CUDA skips, all remaining tests passed.
- CUDA suite (PyTorch 2.7.1+cu128, RTX 5090): 91 tests passed, no skips.
- `python -m compileall -q src tests examples`, dependency checks for both
  environments, and `git diff --check` passed.

Only the test suite ran on CUDA; the numerical experiment is CPU-only.
Reproduction is limited to the same software/device environment. Three training
seeds, three held-out seeds, one environment regime, and 100 updates are a small
bounded sample without hyperparameter search or convergence analysis. No
zero-generation control was rerun for this work item; the earlier
[v0.0.4 experiment](renewable-experiment.md) records that separate comparison.
Equal update budgets do not imply equal trajectory counts, compute, or optimizer
loss scales. This compares the complete Actor-Critic objective, including its
entropy regularization, with legacy REINFORCE; it does not isolate each component.

Intervention null effects do not prove the corresponding input/state is unused.
Appearance permutations can have fixed points; memory reset retains affect and
within-encounter recurrence. Stochastic actions can mask small distributional
changes, and diverging trajectories complicate downstream interpretation.
There is no claim of improved survival, reduced collapse, causal communication,
long-run survival, or a general ranking of learning methods. Future censored
runs must keep observed lifetime separate from eventual survival; this sample's
early extinction supplies no evidence about behavior near the horizon.
