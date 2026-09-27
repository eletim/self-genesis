# Controlled shallow / recurrent Thought comparison

All nine declared training runs completed. **32 steps did not improve measured
survival, GIVE, aid or partner-selection metrics over 16 steps in any seed.**
Both recurrent depths had mean lifetimes 15.4688 / 15.4375 / 14.1146 for training
seeds 10000 / 20000 / 30000, versus shallow's 14.2292 / 14.9167 / 15.2500.
Recurrent policies outperformed shallow in two seeds and underperformed in one;
every learned policy remained below always-GIVE (15.8021). All learned policies
were mixed-GIVE, with no near-always-GIVE/NOTHING collapse and no held-out censoring.
Communication token counts differed slightly between 16/32 in the first two
seeds, so the policies are not claimed to be numerically identical.

Recurrent repeat-minus-first producer contrasts were positive in all seeds
(+0.0756 / +0.0451 / +0.0602), as were prior-aid GIVE contrasts
(+0.0587 / +0.0926 / +0.1369). However, Entity Memory reset had **zero survival
effect in every recurrent seed**, and Appearance interventions had small, mixed
effects. Working Memory reset changed recurrent survival by −0.0625 / −0.0104 /
+0.0104 steps. The descriptive history associations therefore do not establish
useful partner-specific memory or a consistent learning benefit from more
internal computation. Empty-bin estimates remain null in the evidence, including
always-NOTHING's prior-aid contrast; null does not mean zero.

The preselected recurrent-32 probes reached the convergence threshold at steps
19 / 13 / 14. Extra iterations approached a fixed point with very small final
logit/value changes; none of these states was an all-zero ReLU collapse.
The 16-step seed-10000 probe had not yet met the threshold. No outcomes were
used to extend the budget or select seeds. These bounded results do not rule out
useful recurrence under different capacities, contexts or longer training.

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

Timing clarification: case wall time and telemetry also include subprocess startup
and report compression. The 1200-second timeout applies to the worker subprocess,
not compression; the 3600-second budget is the sum of worker caps.

## Evidence and reproduction

The protocol and runner were committed at `c8fb61fa0c5d9daf1aa6714b3a1025e13f60cb2b`
before execution or outcome inspection, on base `aaf5e13`. This is also the
executed source revision; subsequent changes only validate and report evidence.
No training, policy, reward, world or evaluation implementation was changed.
Shallow has 10,987 parameters; recurrent-16 and recurrent-32 each have 11,243.
Matching dimensions does not isolate the recurrent architecture from its extra
parameters, changed activation or initialization relative to shallow.

Retained compressed JSON reports: [seed 10000](evidence/controlled-thought/seed-10000.json.gz),
[seed 20000](evidence/controlled-thought/seed-20000.json.gz),
[seed 30000](evidence/controlled-thought/seed-30000.json.gz).
The [manifest](evidence/controlled-thought/manifest.json) records commands,
resolved configuration, timings, telemetry, source revision and report hashes;
[environment](evidence/controlled-thought/environment.txt) records hardware and
software. Decompress reports with `gzip -dc` to inspect JSON.

Each report retains all updates and world-seed schedules, initial evaluation
worlds, lifetimes/censoring, action and aid records, direct and third-party
histories, per-agent encounter/repeat exposure, same-partner count distributions,
communication messages, Entity Memory events, every sampled Thought step,
all 18 per-seed policy/intervention summaries and 36 matched effects.
`metric_semantics`, `intervention_semantics` and `thought_comparison` define
sampling, missing bins and interventions. Memory reset does not preclude
within-step memory writes; other state can retain history. Appearance changes
can perturb observations independently of partner-specific memory utility.

Validate report hashes, fixed budgets and schedules, finite training diagnostics,
equal initial evaluation worlds, all summary metrics, every intervention delta
and all retained Thought samples:

```sh
PYTHONPATH=src /tmp/self-genesis-verify-gpu/bin/python \
  examples/verify_thought_comparison.py > /tmp/thought-verified.json
```

## Results

All three declared seeds completed all three modes. The following tables pool the three held-out worlds per training seed (96 agent-episodes). Rates pool callbacks; fixed references repeat across training seeds and are shown once. `null` always means an unsupported estimate, never zero.

### Held-out survival and behavior

| Seed | Policy | Mean lifetime | GIVE fraction | Collapse | Successful aid | Deaths / censored |
| --- | --- | --- | --- | --- | --- | --- |
| 10000 | shallow | 14.2292 | 0.3713 | mixed | 406 | 96 / 0 |
| 10000 | recurrent-16 | 15.4688 | 0.6447 | mixed | 525 | 96 / 0 |
| 10000 | recurrent-32 | 15.4688 | 0.6447 | mixed | 525 | 96 / 0 |
| 10000 | always-GIVE | 15.8021 | 1.0000 | near_always_GIVE | 557 | 96 / 0 |
| 10000 | always-NOTHING | 10.0000 | 0.0000 | near_always_NOTHING | 0 | 96 / 0 |
| 10000 | producer-oracle | 15.5417 | 0.6306 | mixed | 532 | 96 / 0 |
| 20000 | shallow | 14.9167 | 0.4475 | mixed | 472 | 96 / 0 |
| 20000 | recurrent-16 | 15.4375 | 0.6102 | mixed | 522 | 96 / 0 |
| 20000 | recurrent-32 | 15.4375 | 0.6102 | mixed | 522 | 96 / 0 |
| 30000 | shallow | 15.2500 | 0.4979 | mixed | 504 | 96 / 0 |
| 30000 | recurrent-16 | 14.1146 | 0.3554 | mixed | 395 | 96 / 0 |
| 30000 | recurrent-32 | 14.1146 | 0.3554 | mixed | 395 | 96 / 0 |

NOTHING fraction is one minus GIVE. Observed lifetime includes horizon censoring.

### Exposure and producer selection

| Seed | Policy | Encounters/agent | Repeats/agent | Repeat fraction | Prior-aid GIVE Δ | First high−low | Repeat high−low | Repeat−first |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | shallow | 14.0000 | 2.8125 | 0.2009 | 0.0317 | -0.0189 | -0.0077 | 0.0112 |
| 10000 | recurrent-16 | 15.3333 | 3.2917 | 0.2147 | 0.0587 | -0.0222 | 0.0535 | 0.0756 |
| 10000 | recurrent-32 | 15.3333 | 3.2917 | 0.2147 | 0.0587 | -0.0222 | 0.0535 | 0.0756 |
| 20000 | shallow | 14.6875 | 2.8750 | 0.1957 | -0.0310 | -0.0042 | 0.0243 | 0.0285 |
| 20000 | recurrent-16 | 15.3125 | 3.3333 | 0.2177 | 0.0926 | -0.0157 | 0.0294 | 0.0451 |
| 20000 | recurrent-32 | 15.3125 | 3.3333 | 0.2177 | 0.0926 | -0.0157 | 0.0294 | 0.0451 |
| 30000 | shallow | 15.1042 | 3.1250 | 0.2069 | 0.0938 | 0.0069 | -0.0005 | -0.0074 |
| 30000 | recurrent-16 | 13.9792 | 2.7708 | 0.1982 | 0.1369 | -0.0080 | 0.0522 | 0.0602 |
| 30000 | recurrent-32 | 13.9792 | 2.7708 | 0.1982 | 0.1369 | -0.0080 | 0.0522 | 0.0602 |

Encounter and same-partner count distributions, including zero exposure, are retained in every raw summary.

### History-conditioned GIVE

Each cell below is positive-history GIVE / zero-history GIVE (callback counts). Direct history uses repeat encounters; third-party history excludes both participants from the third-party set and includes first encounters. Attempts and successful aid are distinct. Unknown bins and their null rates remain in the raw evidence.

#### Seed 10000

| History | shallow | recurrent-16 | recurrent-32 |
| --- | --- | --- | --- |
| prior.outgoing_aid | 0.4167 (120) / 0.4000 (150) | 0.6780 (177) / 0.6835 (139) | 0.6780 (177) / 0.6835 (139) |
| prior.outgoing_give_attempts | 0.4015 (132) / 0.4130 (138) | 0.6653 (236) / 0.7250 (80) | 0.6653 (236) / 0.7250 (80) |
| prior.received_aid | 0.4250 (120) / 0.3933 (150) | 0.7062 (177) / 0.6475 (139) | 0.7062 (177) / 0.6475 (139) |
| prior.received_give_attempts | 0.4242 (132) / 0.3913 (138) | 0.7034 (236) / 0.6125 (80) | 0.7034 (236) / 0.6125 (80) |
| prior_third_party.agent_outgoing_aid | 0.3683 (1078) / 0.3835 (266) | 0.6415 (1325) / 0.6735 (147) | 0.6415 (1325) / 0.6735 (147) |
| prior_third_party.agent_outgoing_give_attempts | 0.3683 (1078) / 0.3835 (266) | 0.6415 (1325) / 0.6735 (147) | 0.6415 (1325) / 0.6735 (147) |
| prior_third_party.agent_received_aid | 0.3619 (1072) / 0.4081 (272) | 0.6429 (1316) / 0.6603 (156) | 0.6429 (1316) / 0.6603 (156) |
| prior_third_party.agent_received_give_attempts | 0.3623 (1082) / 0.4084 (262) | 0.6419 (1318) / 0.6688 (154) | 0.6419 (1318) / 0.6688 (154) |
| prior_third_party.partner_outgoing_aid | 0.3571 (1078) / 0.4286 (266) | 0.6392 (1325) / 0.6939 (147) | 0.6392 (1325) / 0.6939 (147) |
| prior_third_party.partner_outgoing_give_attempts | 0.3571 (1078) / 0.4286 (266) | 0.6392 (1325) / 0.6939 (147) | 0.6392 (1325) / 0.6939 (147) |
| prior_third_party.partner_received_aid | 0.3713 (1072) / 0.3713 (272) | 0.6451 (1316) / 0.6410 (156) | 0.6451 (1316) / 0.6410 (156) |
| prior_third_party.partner_received_give_attempts | 0.3678 (1082) / 0.3855 (262) | 0.6457 (1318) / 0.6364 (154) | 0.6457 (1318) / 0.6364 (154) |

#### Seed 20000

| History | shallow | recurrent-16 | recurrent-32 |
| --- | --- | --- | --- |
| prior.outgoing_aid | 0.4074 (135) / 0.4965 (141) | 0.5955 (178) / 0.5915 (142) | 0.5955 (178) / 0.5915 (142) |
| prior.outgoing_give_attempts | 0.3987 (158) / 0.5254 (118) | 0.6026 (229) / 0.5714 (91) | 0.6026 (229) / 0.5714 (91) |
| prior.received_aid | 0.4370 (135) / 0.4681 (141) | 0.6348 (178) / 0.5423 (142) | 0.6348 (178) / 0.5423 (142) |
| prior.received_give_attempts | 0.4430 (158) / 0.4661 (118) | 0.6288 (229) / 0.5055 (91) | 0.6288 (229) / 0.5055 (91) |
| prior_third_party.agent_outgoing_aid | 0.4483 (1180) / 0.4435 (230) | 0.6059 (1317) / 0.6471 (153) | 0.6059 (1317) / 0.6471 (153) |
| prior_third_party.agent_outgoing_give_attempts | 0.4483 (1180) / 0.4435 (230) | 0.6059 (1317) / 0.6471 (153) | 0.6059 (1317) / 0.6471 (153) |
| prior_third_party.agent_received_aid | 0.4453 (1152) / 0.4574 (258) | 0.6084 (1305) / 0.6242 (165) | 0.6084 (1305) / 0.6242 (165) |
| prior_third_party.agent_received_give_attempts | 0.4463 (1163) / 0.4534 (247) | 0.6075 (1307) / 0.6319 (163) | 0.6075 (1307) / 0.6319 (163) |
| prior_third_party.partner_outgoing_aid | 0.4458 (1180) / 0.4565 (230) | 0.6029 (1317) / 0.6732 (153) | 0.6029 (1317) / 0.6732 (153) |
| prior_third_party.partner_outgoing_give_attempts | 0.4458 (1180) / 0.4565 (230) | 0.6029 (1317) / 0.6732 (153) | 0.6029 (1317) / 0.6732 (153) |
| prior_third_party.partner_received_aid | 0.4575 (1152) / 0.4031 (258) | 0.6100 (1305) / 0.6121 (165) | 0.6100 (1305) / 0.6121 (165) |
| prior_third_party.partner_received_give_attempts | 0.4549 (1163) / 0.4130 (247) | 0.6106 (1307) / 0.6074 (163) | 0.6106 (1307) / 0.6074 (163) |

#### Seed 30000

| History | shallow | recurrent-16 | recurrent-32 |
| --- | --- | --- | --- |
| prior.outgoing_aid | 0.5000 (154) / 0.4863 (146) | 0.3874 (111) / 0.3742 (155) | 0.3874 (111) / 0.3742 (155) |
| prior.outgoing_give_attempts | 0.4804 (179) / 0.5124 (121) | 0.3932 (117) / 0.3691 (149) | 0.3932 (117) / 0.3691 (149) |
| prior.received_aid | 0.5390 (154) / 0.4452 (146) | 0.4595 (111) / 0.3226 (155) | 0.4595 (111) / 0.3226 (155) |
| prior.received_give_attempts | 0.5196 (179) / 0.4545 (121) | 0.4701 (117) / 0.3087 (149) | 0.4701 (117) / 0.3087 (149) |
| prior_third_party.agent_outgoing_aid | 0.4929 (1260) / 0.5316 (190) | 0.3622 (1038) / 0.3322 (304) | 0.3622 (1038) / 0.3322 (304) |
| prior_third_party.agent_outgoing_give_attempts | 0.4929 (1260) / 0.5316 (190) | 0.3622 (1038) / 0.3322 (304) | 0.3622 (1038) / 0.3322 (304) |
| prior_third_party.agent_received_aid | 0.4940 (1247) / 0.5222 (203) | 0.3526 (1055) / 0.3659 (287) | 0.3526 (1055) / 0.3659 (287) |
| prior_third_party.agent_received_give_attempts | 0.4940 (1249) / 0.5224 (201) | 0.3521 (1065) / 0.3682 (277) | 0.3521 (1065) / 0.3682 (277) |
| prior_third_party.partner_outgoing_aid | 0.4881 (1260) / 0.5632 (190) | 0.3449 (1038) / 0.3914 (304) | 0.3449 (1038) / 0.3914 (304) |
| prior_third_party.partner_outgoing_give_attempts | 0.4881 (1260) / 0.5632 (190) | 0.3449 (1038) / 0.3914 (304) | 0.3449 (1038) / 0.3914 (304) |
| prior_third_party.partner_received_aid | 0.4964 (1247) / 0.5074 (203) | 0.3564 (1055) / 0.3519 (287) | 0.3564 (1055) / 0.3519 (287) |
| prior_third_party.partner_received_give_attempts | 0.4964 (1249) / 0.5075 (201) | 0.3531 (1065) / 0.3646 (277) | 0.3531 (1065) / 0.3646 (277) |

### Matched interventions

Lifetime deltas are intervention minus untreated, averaged over the three paired evaluation worlds; censored deltas are count differences across 96 agent-episodes. GIVE, prior-aid and producer-selection deltas here use pooled per-seed summaries; the raw report also retains every per-world paired delta, including nulls. Sampling streams restart, but trajectories may diverge.

| Seed | Policy | Intervention | Lifetime Δ | Censored Δ | GIVE Δ | Prior-aid Δ | Repeat−first Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | shallow | appearance-shuffle | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | shallow | entity-memory-reset | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | shallow | appearance-replacement | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | shallow | working-memory-reset | 0.3021 | 0 | 0.0283 | -0.0138 | 0.0271 |
| 10000 | recurrent-16 | appearance-shuffle | 0.0000 | 0 | 0.0002 | 0.0311 | 0.0263 |
| 10000 | recurrent-16 | entity-memory-reset | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | recurrent-16 | appearance-replacement | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | recurrent-16 | working-memory-reset | -0.0625 | 0 | 0.0077 | -0.0012 | -0.0631 |
| 10000 | recurrent-32 | appearance-shuffle | 0.0000 | 0 | 0.0002 | 0.0311 | 0.0263 |
| 10000 | recurrent-32 | entity-memory-reset | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | recurrent-32 | appearance-replacement | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 10000 | recurrent-32 | working-memory-reset | -0.0625 | 0 | 0.0077 | -0.0012 | -0.0631 |
| 20000 | shallow | appearance-shuffle | 0.0312 | 0 | 0.0001 | 0.0029 | -0.0029 |
| 20000 | shallow | entity-memory-reset | 0.0000 | 0 | -0.0007 | 0.0000 | -0.0017 |
| 20000 | shallow | appearance-replacement | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 20000 | shallow | working-memory-reset | -0.3958 | 0 | -0.0449 | 0.0712 | 0.0491 |
| 20000 | recurrent-16 | appearance-shuffle | -0.0312 | 0 | 0.0002 | 0.0184 | 0.0180 |
| 20000 | recurrent-16 | entity-memory-reset | 0.0000 | 0 | -0.0007 | 0.0070 | 0.0059 |
| 20000 | recurrent-16 | appearance-replacement | 0.0000 | 0 | -0.0007 | 0.0070 | 0.0059 |
| 20000 | recurrent-16 | working-memory-reset | -0.0104 | 0 | -0.0062 | -0.0490 | 0.0042 |
| 20000 | recurrent-32 | appearance-shuffle | -0.0312 | 0 | 0.0002 | 0.0184 | 0.0180 |
| 20000 | recurrent-32 | entity-memory-reset | 0.0000 | 0 | -0.0007 | 0.0070 | 0.0059 |
| 20000 | recurrent-32 | appearance-replacement | 0.0000 | 0 | -0.0007 | 0.0070 | 0.0059 |
| 20000 | recurrent-32 | working-memory-reset | -0.0104 | 0 | -0.0062 | -0.0490 | 0.0042 |
| 30000 | shallow | appearance-shuffle | 0.0000 | 0 | -0.0007 | 0.0282 | 0.0053 |
| 30000 | shallow | entity-memory-reset | 0.0104 | 0 | 0.0014 | 0.0358 | 0.0311 |
| 30000 | shallow | appearance-replacement | 0.0104 | 0 | 0.0021 | 0.0287 | 0.0328 |
| 30000 | shallow | working-memory-reset | -0.0417 | 0 | 0.0159 | -0.0302 | -0.0647 |
| 30000 | recurrent-16 | appearance-shuffle | 0.0000 | 0 | -0.0007 | 0.0000 | -0.0018 |
| 30000 | recurrent-16 | entity-memory-reset | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 30000 | recurrent-16 | appearance-replacement | 0.0312 | 0 | 0.0002 | -0.0132 | -0.0019 |
| 30000 | recurrent-16 | working-memory-reset | 0.0104 | 0 | 0.0003 | 0.0085 | -0.0339 |
| 30000 | recurrent-32 | appearance-shuffle | 0.0000 | 0 | -0.0007 | 0.0000 | -0.0018 |
| 30000 | recurrent-32 | entity-memory-reset | 0.0000 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 30000 | recurrent-32 | appearance-replacement | 0.0312 | 0 | 0.0002 | -0.0132 | -0.0019 |
| 30000 | recurrent-32 | working-memory-reset | 0.0104 | 0 | 0.0003 | 0.0085 | -0.0339 |

### Communication

Channel usage is not evidence of causal utility. All interventions retain these communication metrics in the raw summaries.

| Seed | Policy | Callbacks | Nonempty | Tokens | Mean length | Token entropy (bits) | Token counts 0/1/2/3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | shallow | 1344 | 1344 | 4032 | 3.0000 | 1.8529 | [1205, 1255, 1264, 308] |
| 10000 | recurrent-16 | 1472 | 1472 | 4416 | 3.0000 | 1.8492 | [924, 1102, 494, 1896] |
| 10000 | recurrent-32 | 1472 | 1472 | 4416 | 3.0000 | 1.8479 | [924, 1102, 491, 1899] |
| 20000 | shallow | 1410 | 1410 | 4230 | 3.0000 | 1.9431 | [1097, 691, 1511, 931] |
| 20000 | recurrent-16 | 1470 | 1470 | 4410 | 3.0000 | 1.9684 | [1299, 727, 1199, 1185] |
| 20000 | recurrent-32 | 1470 | 1470 | 4410 | 3.0000 | 1.9683 | [1299, 726, 1200, 1185] |
| 30000 | shallow | 1450 | 1450 | 4350 | 3.0000 | 1.4937 | [268, 979, 430, 2673] |
| 30000 | recurrent-16 | 1342 | 1342 | 4026 | 3.0000 | 1.8230 | [1853, 569, 1005, 599] |
| 30000 | recurrent-32 | 1342 | 1342 | 4026 | 3.0000 | 1.8230 | [1853, 569, 1005, 599] |

### Training exposure and compute cost

| Seed | Policy | World steps | Encounters | Gradient norm min/max |
| --- | --- | --- | --- | --- |
| 10000 | shallow | 124809 | 1469813 | 515.0057 / 636.4229 |
| 10000 | recurrent-16 | 131555 | 1573849 | 379.3476 / 757.4964 |
| 10000 | recurrent-32 | 131565 | 1573815 | 379.3476 / 757.1533 |
| 20000 | shallow | 124191 | 1453526 | 440.3042 / 581.4676 |
| 20000 | recurrent-16 | 130412 | 1548043 | 433.3813 / 753.0270 |
| 20000 | recurrent-32 | 130412 | 1548043 | 433.3813 / 753.0308 |
| 30000 | shallow | 131017 | 1558204 | 497.9611 / 662.6516 |
| 30000 | recurrent-16 | 123884 | 1457015 | 460.5286 / 652.8314 |
| 30000 | recurrent-32 | 123884 | 1457015 | 460.5286 / 652.8315 |

| Seed | Case seconds (training + evaluation + compression) | Device GPU % mean/max | Device VRAM peak GiB |
| --- | --- | --- | --- |
| 10000 | 279.0432 | 36.53 / 70 | 4.1953 |
| 20000 | 280.5649 | 36.11 / 68 | 4.1797 |
| 30000 | 269.6884 | 37.04 / 69 | 4.1660 |

Each mode receives 6400 world episodes per seed. All recorded losses and gradient norms are finite. Case timing and telemetry cover process startup, all modes, evaluation and evidence compression; telemetry includes the desktop and polling overhead, and is not per-mode GPU training throughput or allocator VRAM. The earlier horizon-16 benchmark measures training-only costs separately. These runs match learning opportunities by episodes, not compute or realized encounters.

### Representative Thought dynamics

[Every step of the nine preselected action probes](controlled-thought-dynamics.md) records norms, changes, cosine, convergence, saturation, action logits/changes and value/changes. All probes, including communication and interventions, are retained in raw reports.

## Validation

The [full CPU suite](evidence/controlled-thought/tests-cpu.log) passed all 212 tests
(14 CUDA skips) in 74.66 seconds. The [CUDA-enabled Thought suite](evidence/controlled-thought/tests-cuda.log)
passed all 15 tests in 10.55 seconds, including full-episode gradient lifetime and
observational probe checks. Both suites ran after the experiment. Runner tests
exercise pre-execution manifest creation, fixed settings, timeout/nonzero-exit
retention, continuing to every declared seed, and refusing to overwrite evidence.
The verifier recomputed all 54 summaries and 108 intervention effects, checked
all 450 updates and 810 sampled callbacks, and passed. `compileall` and
`git diff --check` passed. Commands are retained in
[validation.txt](evidence/controlled-thought/validation.txt).

Actual training exposure totaled 1,151,729 world steps and 13,639,323 encounters
across the declared 57,600 episodes. Case times totaled 829.30 seconds including
evaluation and compression; no timeout, worker error or telemetry error occurred.
