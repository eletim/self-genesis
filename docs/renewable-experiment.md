# Historical v0.0.4 renewable experiment

This record describes v0.0.4 REINFORCE validation at commit
[`ac50ee0`](https://github.com/eletim/self-genesis/commit/ac50ee01a6f96c93410ef6d10b254a278886a647)
(the renewable validation change). Its numerical results, report hashes,
three-policy report format, and test counts are historical. The commands below
belong to that revision. On v0.0.5, the default objective is Actor-Critic and
comparison reports also include the producer oracle and additional diagnostics;
simply replaying these commands will not reproduce the historical report bytes.
Use `--training-method reinforce` for a current-code legacy-objective comparison,
or follow the [matched v0.0.5 procedure](matched-learning-experiment.md) for both
methods and retained evidence. That experiment did not rerun the zero-generation
control. The v0.0.4 world and survival-only reward contract remain unchanged.

The historical runnable default, `configs/default.toml`, selected generation probabilities
0.1–0.3 and a survival horizon of 100. `configs/renewable.toml` is the equivalent
minimal preset. Both use four agents, initial Life 10, initial Points 3, and
three updates unless overridden. No-config CLI and `ExperimentConfig()` defaults
retain zero generation and no horizon for compatibility; pass the file explicitly.

## What changed from v0.0.3

v0.0.3 had only initial Points and trained to extinction. v0.0.4 samples an
independent, lifetime-fixed generation probability per agent and generates at
most one Point per survivor after simultaneous transfers, Life decay, and death.
Generated Points are usable next step, including for agents outside the encounter.
Appearance does not encode ability; ability and generation history are analysis
fields, never policy inputs. Points still only restore another agent's Life.

The recurrent network, communication channel, Adam update, and per-agent
survival-only REINFORCE objective are unchanged. Renewable training requires an
explicit horizon: rewards count living steps through that horizon, without a
bootstrap, social bonus, or fabricated terminal death. Episode resets restore
abilities/resources but continue sampling streams. A new run replays those streams.
The design principles and representative scenarios describe these same rules.

Logs now include generation abilities, actual generation, and directed successful
transfers. Analysis distinguishes prior encounters, received GIVE attempts,
received NOTHING, successful aid, and outgoing help, keyed by partner Appearance.
Fixed-policy comparisons use the same world rules and matched evaluation seeds.
These fields let us ask about partner selection without making selection or
cooperation a test requirement.

## Reproduce the bounded comparison

Use the CPU environment in [the run guide](minimal-experiment.md), with Python
3.12.14 and PyTorch 2.7.1+cpu for the recorded results. Run from the repository
root. Use a fresh output directory (the commands intentionally refuse overwrite).

```sh
mkdir /tmp/self-genesis-renewable-comparison
for repeat in 1 2; do
  python -m self_genesis compare --config configs/default.toml --device cpu \
    --seed 42 --episodes 100 --survival-horizon 100 \
    --evaluation-seeds 101 102 103 \
    --output /tmp/self-genesis-renewable-comparison/renewable-$repeat.json
  python -m self_genesis compare --config configs/default.toml --device cpu \
    --seed 42 --episodes 100 --survival-horizon 100 \
    --point-generation-probability-min 0 --point-generation-probability-max 0 \
    --evaluation-seeds 101 102 103 \
    --output /tmp/self-genesis-renewable-comparison/zero-$repeat.json
done
cmp /tmp/self-genesis-renewable-comparison/renewable-{1,2}.json
cmp /tmp/self-genesis-renewable-comparison/zero-{1,2}.json
```

Each report trains one network, then freezes it for nine evaluations: three
policies times three seeds. The bound is 100 training updates of at most 100
steps plus nine evaluations of at most 100 steps. Networks are trained separately
in each resource regime. This historical zero-generation control used the v0.0.4 code and
the same horizon, rather than replaying an old v0.0.3 binary or its historical
pre-sampling-fix results. Full episode graphs are bounded by the horizon.

Each policy gets fresh memory, the same initial population/abilities, and restarted
sampling streams for each seed. Realized encounters and generation can diverge
when survival differs. Fixed policies send empty messages; learned messages and
actions remain sampled. The JSON report is separate from training JSONL and must
not be passed to `examples/analyze_run.py`.

To inspect training and temporal histories, run the CPU/CUDA recipes in the run
guide, or `train` with the same configuration/seed/episode count and a fresh
`--output` JSONL path, then run `python examples/analyze_run.py PATH`. The comparison
and training commands perform the same updates under matching settings. The
integration test checks this directly. Raw `step` records retain generation and
transfer histories; analyzer `relationship_actions` joins only earlier history
to each current action. Summary lifetimes preserve censoring information.

## Observed results

Recorded on 2026-09-26, Linux x86_64, Python 3.12.14, PyTorch 2.7.1+cpu.
Both repetitions of each report were byte-identical. Renewable training took
1,573 world steps; zero-generation training took 1,415. Mean lifetime below pools
12 agents (four per evaluation seed). GIVE fractions pool encounter action
callbacks, including failed attempts; aid counts only successful transfers.

| Regime | Policy | Mean lifetime | GIVE / callbacks | Successful aid | Deaths / agents |
| --- | --- | ---: | ---: | ---: | ---: |
| Renewable | Learned | 13.4167 | 47 / 80 | 41 | 12 / 12 |
| Renewable | Always GIVE | 14.4167 | 92 / 92 | 53 | 12 / 12 |
| Renewable | Always NOTHING | 10.0000 | 0 / 60 | 0 | 12 / 12 |
| Zero generation | Learned | 12.6667 | 38 / 76 | 32 | 12 / 12 |
| Zero generation | Always GIVE | 13.0000 | 80 / 80 | 36 | 12 / 12 |
| Zero generation | Always NOTHING | 10.0000 | 0 / 60 | 0 | 12 / 12 |

Learned GIVE counts by strictly prior relationship history:

| Regime | Unseen partner | Previously received successful aid | Encountered without received aid |
| --- | ---: | ---: | ---: |
| Renewable | 16 / 36 (44.4%) | 21 / 28 (75.0%) | 10 / 16 (62.5%) |
| Zero generation | 13 / 34 (38.2%) | 12 / 22 (54.5%) | 13 / 20 (65.0%) |

The renewable learned policy gave more often after prior successful aid in this
sample, but these are small, dependent, observational groups. Episode time,
resources, encounter role, and survival selection can explain associations.
There is no demonstrated causal use of Appearance, memory, or partner ability;
that would require repeated training seeds and controlled ablations. The learned
policy fell below always-GIVE in both regimes. Neither near-always-GIVE nor
near-always-NOTHING collapse appeared in these held-out evaluations (58.75% and
50% GIVE); this does not rule out collapse with other seeds or longer training.
No cooperation, token semantics, or stable partner selection is established.

For exact artifact verification, SHA-256 of the report bytes:

- Renewable: `dcc1f5a0980c08256a50a2a2531882c7ff055f1c143278e8db12adafbd9b8cd5`
- Zero generation: `c8c9870ed57a1fb0ab68b25c6c6c067068dac6568f38688702dadc6a1049dda8`

## Horizon censoring and validation

All comparison populations above went extinct before step 100 (latest step 18),
so their means are completed lifetimes. This does not validate long-run survival.
When any agent survives to the horizon, `horizon_completed=true`,
`terminated=false`, and `truncated=false`; survivors have `censored=true` and no
death step. `mean_survival_time` is null. `mean_observed_lifetime` and per-agent
returns describe survival within the observation window, not eventual lifetimes.
Extinction on the final horizon step takes precedence. An interrupted collection
is truncated, not a complete training objective. Never mix completed and censored
means as though both estimate eventual survival.

The focused integration tests use horizons 3 and 30 to exercise surviving and
extinct populations on both CPU and CUDA. They independently reconstruct Life,
Points, survival returns, deaths, and prior relationship history from raw logs;
check hidden abilities stay out of observations; compare training updates across
CLI paths; and verify repeated matched baseline reports. They impose no desired
learned action ratio. Existing unit tests cover deterministic renewal timing,
failed GIVE attempts, horizon/death precedence, gradients, and frozen evaluation.

Validation on the same date:

- CPU, PyTorch 2.7.1+cpu: 70 tests run, seven CUDA cases skipped, all others passed.
- NVIDIA GeForce RTX 5090, PyTorch 2.7.1+cu128: all 70 tests passed, no skips.
- Both documented small renewable training-to-analysis recipes passed.
- `compileall` for `src tests examples`, `git diff --check`, and dependency checks
  passed. Existing environments were reused with `PYTHONPATH=src` to select this
  checkout; dependency checks used `uv pip check --python ENV/bin/python` because
  those environments do not include pip.

The 100-update numerical comparison was CPU-only. No CPU/CUDA trajectory equality,
scientific convergence, long-run throughput, or peak memory capacity is claimed.
