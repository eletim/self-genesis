# self-genesis

[Canonical design principles](docs/design-principles.md)

[Measured RTX 5090 training scaling](docs/rtx5090-scaling.md)

[Runnable CPU and RTX 5090 experiments, analysis, and validation](docs/minimal-experiment.md)

[Historical v0.0.4 renewable comparison and observed behavior](docs/renewable-experiment.md)

[Validated v0.0.5 matched Actor-Critic versus REINFORCE evidence](docs/matched-learning-experiment.md)

[Validated v0.0.6 Entity Memory comparison and reproduction](docs/entity-memory-experiment.md)

## v0.0.7 scaled experiments

Use the [scaled experiment workflow](docs/scaled-experiment-workflow.md) to
reproduce capacity counts, compare throughput, and run held-out partner-history
and Entity Memory analyses. Small, medium and large presets have **10,987**,
**278,823** and **4,211,847** shared parameters at the default Appearance/channel
dimensions. Batched training preserves the survival-only objective and private
agent state; sequential FP32 training remains the default.

The retained RTX 5090 sweep completed all 33 cases across three capacities,
sequential execution and 64–1024 batched worlds, with FP32/BF16 batched training.
At 256 worlds, FP32 measured 15,389–15,707 world steps/s versus 240–250 sequentially
on the four-agent, horizon-16 workload. These are bounded execution measurements,
not evidence of improved learning or social behavior. BF16 remains optional.
`compare` trains and evaluates sequential policies; it does not load batched
training JSONL. The behavioral evidence below predates the scaling work.

## v0.0.6 Entity Memory evidence

The current default adds 16-dimensional Entity Memory keyed by perceived
Appearance; `--entity-memory-dim 0` retains the v0.0.5 architecture. The
[matched CPU experiment](docs/entity-memory-experiment.md) trained each condition
for 100 updates with three training seeds and three held-out seeds. Mean lifetime
was 12.4722 steps enabled versus 12.3611 disabled, with paired seed differences
of -0.0833, +2.3333, and -1.9167. Neither beat always-GIVE (14.4167).
Entity Memory reset and both Appearance interventions left survival and GIVE
counts unchanged. These mixed and null findings do not establish improved
cooperation, survival, useful memory, or producer identification. Both full runs
replayed exactly; disabled-memory updates and shared evaluation outcomes match
the actual v0.0.5 implementation. The older results below are historical.

## v0.0.5 learning and evidence

v0.0.5 defaults to Actor-Critic with a scalar value baseline, detached advantages,
and separate action/message entropy controls; `--training-method reinforce`
selects the legacy objective in the current implementation. The v0.0.4 renewable
world, encounter order, fixed Appearance, observations, and recurrent dimensions
remain unchanged. Generation abilities and identity labels stay out of learned
policy inputs. As required by the [design principles](docs/design-principles.md),
reward remains each agent's own survival only: entropy regularizes the loss,
with no GIVE, cooperation, or communication bonuses. Finite-horizon survivors
remain censored, with no bootstrap or terminal bonus.

The [validated matched experiment](docs/matched-learning-experiment.md) used
three training seeds, three held-out seeds, and 100 updates per training seed.
Mean held-out lifetime averaged over training seeds was 12.3611 steps for
Actor-Critic versus 13.3333 for REINFORCE; neither beat always-GIVE (14.4167).
All learned seed-level GIVE summaries were mixed. Appearance shuffle and Working
Memory reset left Actor-Critic survival and GIVE counts unchanged in this sample;
that does not prove these inputs/states are unused. History associations and token
usage establish neither causal reciprocity nor useful communication. These bounded
CPU results do not establish improved survival, reduced collapse, or convergence.
The separate [v0.0.4 record](docs/renewable-experiment.md) is historical evidence,
including a zero-generation control that was not rerun for v0.0.5.

## Experiment foundation

Requires Python 3.11+ and PyTorch. Create a virtual environment and install:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m self_genesis --config configs/default.toml --seed 42 --device cpu
```

The `self-genesis` command provides the same entry point. For a CPU-only
installation, install PyTorch from its CPU wheel index before installing the
project: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
CUDA execution requires a CUDA-capable PyTorch installation and compatible GPU.

Edit [configs/default.toml](configs/default.toml) to configure the seed, device,
number of agents (at least two), appearance dimension, initial Life, and initial
Point budget, renewable generation bounds, and survival horizon. Life and
appearance dimension must be positive integers; Points may be zero. Seeds are integers from 0 through `2**63 - 1`. Without `--config`,
the API defaults retain zero generation and no horizon for compatibility. The
sample default file uses generation probabilities 0.1–0.3 and horizon 100.
`--seed` and `--device` override file settings.
Unknown configuration keys and invalid values fail with an error.

Device choices are `cpu` (default), `cuda` (fails if unavailable), and `auto`
(CUDA when available, otherwise CPU). The run seeds Python and PyTorch and
initializes separate Life, Point, and fixed Appearance values for multiple
agents. Appearance is sampled on CPU and transferred to the selected device,
so the same seed produces the same initial state on CPU and CUDA within the
same software version. This does not promise deterministic future training or
identical results across PyTorch versions.

Each run prints JSON containing the effective conditions, resolved device, and
initial agent state. Use the `train` command below to run learning experiments.
Shared world dynamics and encounters are available through the Python API below.
A trainable recurrent policy is available through the Python API below.

## Shared survival world

```python
from self_genesis.config import ExperimentConfig
from self_genesis.world import Action, Decision, World

world = World(ExperimentConfig(num_agents=2))
result = world.step([Decision(Action.GIVE, target=1), Decision(Action.NOTHING)])
print(world.state.life, world.state.points, result.reward, result.done)
```

Each step takes exactly one decision per agent in state order. GIVE spends one
of the donor's Points and restores one Life to another living agent. Gifts
resolve simultaneously before all living agents lose one Life. This lets a
gift save a recipient with one Life remaining. Multiple gifts to the same
recipient add together; Life has no upper cap. NOTHING only allows time to pass.

Life reaching zero means permanent death. Dead agents cannot act or receive
Life. A GIVE involving a dead agent or a donor without Points does nothing and
costs nothing. Self-directed and invalid targets, malformed decisions, and
incorrect decision counts raise `ValueError` before changing the world.
Points never transfer to recipients, and Appearance never changes. With positive
generation probabilities, survivors may generate a Point after decay (see below).

`StepResult.reward` gives each agent alive at the start of the step one reward,
including its final step: accumulated reward is its lifetime in world steps.
There are no GIVE, receipt, cooperation, or other social bonuses.
`StepResult.died` marks new deaths, and `done` becomes true when everyone is dead.
Further steps return zero rewards and no new deaths. With zero generation, finite
initial Points bound restored Life; renewable training instead requires a finite horizon.

## Encounter communication

```python
from self_genesis.encounter import EncounterProtocol

class QuietPolicy:
    def communicate(self, observation):
        return ()

    def act(self, observation):
        return Action.NOTHING

protocol = EncounterProtocol(world, seed=42, vocabulary_size=4, max_message_length=3)
result = protocol.step([QuietPolicy() for _ in range(2)])
```

Provide one policy per agent in state order. Each step uniformly samples two
living agents without replacement; their sampled order assigns first/second
roles. The protocol owns its seeded random generator, independent of global
random draws. First sends one message, second observes it and replies, then
first and second choose GIVE or NOTHING in that order. Both see the other's
message when acting; second also sees first's chosen action. GIVE automatically
targets the encounter partner. Gifts resolve simultaneously through the shared
world, followed by one Life decay for every living agent, including those not
selected. Fewer than two survivors means no communication or action callbacks;
time still advances.

Policy observations contain own Life/Points, partner Life/Points, a copy of the
partner's fixed Appearance, the first/second role, the received message, and the
partner's action when available. They contain no agent indices or identity
labels. Policy positions are used only to route callbacks.

Messages are sequences of integer tokens in `range(vocabulary_size)`, at most
`max_message_length` long. Empty messages are allowed, and a zero length limit
disables token transmission. Tokens carry no predefined meaning, reward, or
direct world effect. Invalid messages or actions raise `ValueError` before
world state changes (policy callbacks and random sampling are not rolled back).

Encounter density is configurable in both sequential and batched execution with
`--encounter-count` (TOML: `encounter_count`) or `--encounter-fraction`
(TOML: `encounter_fraction`). Count must be a nonnegative integer; fraction must
be finite and in `[0, 1]`. Supplying both is an error, including across TOML and
CLI; an override of the same setting replaces its configured value. If neither
is supplied, count defaults to 1. For the current living population `L`, the
number of disjoint pairs is `min(count, L // 2)` or
`floor(fraction * L / 2)`. Zero pairs still advance survival time; an episode
with no decisions records zero loss and skips the optimizer update.

For 32-agent comparisons, keep all other settings fixed and run with
`--num-agents 32 --encounter-count 1`, then counts `4`, `8`, and `16`.
These options combine with existing `--capacity-preset small|medium|large`
and leave Actor-Critic coefficients unchanged. Resolved density settings are
saved in the existing configuration records and summaries. Sequential traces
include explicit `pairs` for relationship analysis; batched trace participants
are consecutive first/second pairs, padded with `-1` for unused slots.

Run the checks from the repository root:

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

CUDA initialization and world steps are tested when CUDA is available; device
selection and the unavailable-CUDA error path are also tested on CPU hosts.


## Recurrent agent policy

```python
from self_genesis.policy import AgentPolicy, RecurrentPolicy

world = World(ExperimentConfig(num_agents=2))
network = RecurrentPolicy(world.state.appearance.shape[1]).to(world.state.life.device)
agents = [AgentPolicy(network) for _ in range(2)]
protocol = EncounterProtocol(world, seed=42)
result = protocol.step(agents)
```

Use a distinct `AgentPolicy` for each agent. Adapters may share a network, but
own separate Working Memory, affect tensors, Entity Memory, and sampled-decision log
probabilities. The network itself holds only weights. Its `forward` method
also accepts and returns explicit `PolicyState` tensors for inspection.

Each communication or action callback encodes resources, partner Appearance,
role, received message positions, available partner action, and callback phase.
A thought layer consumes this observation plus previous memory, affect, and
the retrieved Entity Memory value;
a GRU updates memory using thought and previous affect. New affect is generated
from the observation, thought, and updated memory, then feeds the next callback.
Affect dimensions have no predefined meanings or supervised targets. No agent
indices, self labels, or auxiliary classification objectives are added.

Entity Memory stores one unlabeled latent value per distinct observed Appearance,
using exact tensor equality for retrieval. An unseen Appearance retrieves zeros;
identical Appearances share an entry, without hidden IDs. After each callback, a
learned GRU updates that entry from the observation, thought, updated Working
Memory, and affect. Retrieval feeds thought and thus both action and Communication
heads and the critic. After world resolution, an additional learned GRU writes
completed Encounter experience: the same observed Appearance, pre-step resource
observation, received message, both chosen actions, and whether each directed
gift succeeded. Only participants receive this feedback; hidden IDs, generation
abilities/draws, and analysis histories are excluded. Completion updates only
Entity Memory, without sampling another decision or adding a loss/reward. The
same feedback path runs during training and frozen-weight evaluation, including
Appearance shuffling. Values have no assigned meanings or auxiliary targets.
Entries persist across encounters and collection boundaries, reset on death and
at episode boundaries, and detach with other recurrent state only when explicitly
requested (training does so after complete-episode backpropagation). Storage grows
with the distinct Appearances observed during an episode; there is no eviction or
approximate match.

`entity_memory_dim` defaults to 16 in the network and experiment configuration.
Set it to `0` in TOML or pass `--entity-memory-dim 0` to disable Entity Memory and
recover the v0.0.5 layer shapes, initialization, and forward computation. The
existing `working-memory-reset` intervention resets only Working Memory.

Training JSONL states include `entity_memory` entries with observed `appearance`
and latent `value`. Each callback's `entity_memory` records its `retrieved` value
and entry lists in `state_before` and `state_after`; step-level
`entity_memory_completions` records the additional resolved-encounter writes,
before death resets. Comparison evaluations expose the same snapshots in
`entity_memory_events`, tagged by step, logging agent index, and phase (`message`,
`action`, or `completion`). Fixed/oracle policies have no memory events; disabled
Entity Memory records empty retrievals and entry lists. These additive fields are
detached JSON values, separate from differentiable rollout states. Logging IDs,
hidden generation abilities, and analysis histories never feed policy inputs or
memory writes. Existing action, survival, communication, and training outputs
retain their meanings. Trace size grows with observed Appearances and callbacks.

The communication head samples independent tokens from one categorical
distribution for a fixed-length message; the action head samples GIVE or
NOTHING. Match the network's `vocabulary_size` and `max_message_length` to the
encounter protocol (defaults match). A zero message length disables transmission
while still updating internal state. Unselected agents retain their state.

`agent.log_probs` retains differentiable log probabilities (one sum per message,
one per action) for policy-gradient training with each agent's survival
reward-to-go. Matching `agent.values` and `agent.entropies` retain scalar value
predictions and categorical entropy for each sampled decision. The value readout
uses the existing updated memory before sampling; message entropy sums across
independent token slots. Rollout decisions expose these as `value` and `entropy`,
with `None` for both on empty messages, which still update recurrent state.
Actor-Critic training uses these statistics for a detached advantage baseline,
value regression, and entropy regularization. Discrete samples themselves are not
differentiable. The tests exercise a complete episode and optimizer update using only world survival
rewards; this is a trainability check, not evidence of learned cooperation.
The `train` command below exposes these updates through the CLI.

Call `agent.reset()` at episode boundaries to clear state and experience. For
truncated backpropagation, consume the pending loss before calling
`agent.detach()` to preserve state values while dropping their graph and clearing
all captured decision statistics. Reset or detach before collecting another segment after an
optimizer update. Construct adapters after moving the network to its device;
use `torch.no_grad()` for inference and `agent.clear_decisions()` to release
accumulated statistics as needed. Sampling uses PyTorch's RNG (`torch.manual_seed` controls it).

## Bounded multi-agent rollouts

```python
from self_genesis.rollout import RolloutCollector

config = ExperimentConfig(num_agents=4, appearance_dim=8)
network = RecurrentPolicy(config.appearance_dim)
collector = RolloutCollector(config, network)
segment = collector.collect(max_steps=32)
```

The collector connects the shared world, encounter protocol, and policy, using
one independent adapter per agent. Channel bounds come from the network.
`segment.experiences[index]` is that agent's trajectory, with an absolute episode
step, survival reward, death flag (`terminated`), and ordered policy decisions.
Each decision retains its observation, sampled message or action, differentiable
log probability, and recurrent states before and after the callback. An empty
channel still records communication and its state update, with no log probability.
These indices route experience only; they are never fed to the policy.

Living agents receive an experience on every world step, including steps when
they are not selected or are the lone survivor. Death includes the final survival
reward, stops subsequent experience for that agent, and clears its live recurrent
state. Returned decisions keep their graphs even after death or explicit reset.

`segment.terminated` means extinction. `segment.horizon_completed` means the
configured survival horizon was reached with survivors. `segment.truncated` means the collection budget ran out before
either episode ending; extinction on the last allowed step takes
precedence. Calling `collect` again continues the same episode, step counter,
and recurrent graph until the horizon. After horizon completion, reset explicitly
to begin another episode. Calling it after extinction returns an empty terminated
segment. Budgets must be positive integers. `collector.reset()` explicitly
restores configured resources and appearances and clears all live agent state.
Network weights and the Python, PyTorch, and encounter RNG streams are preserved,
so subsequent episodes draw fresh samples. Collector construction seeds sampling;
reproduce a run by also seeding before constructing its network.

For survival learning, accumulate each agent's rewards backward through its
trajectory and weight each message/action log probability by that agent's
reward-to-go, including later steps without encounters. Join consecutive segments
for complete episode returns from step zero through extinction or the configured
survival horizon. Training rejects truncated rollouts and uses no truncation
bootstrap; a collection budget boundary must not be treated as death. No social
reward or learning objective is added by collection.

Consume the pending loss before `collector.detach()` and an optimizer update.
Detach preserves recurrent values while cutting history for subsequent segments;
reset starts fresh instead. Returned records retain their graphs until released,
so discard consumed segments to free memory. Collection never silently detaches
at a budget boundary. Use `torch.no_grad()` for inference. The `train` command below runs complete learning episodes.

## Survival policy training

```python
import torch
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy
from self_genesis.rollout import RolloutCollector
from self_genesis.training import train_episode

config = ExperimentConfig(num_agents=4, appearance_dim=8, device="cpu")
torch.manual_seed(config.seed)
network = RecurrentPolicy(config.appearance_dim)
collector = RolloutCollector(config, network)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
for _ in range(3):
    result = train_episode(collector, optimizer)
    print(result.loss, result.survival_returns)
```

`train_episode` explicitly resets the collector, collects a complete episode,
backpropagates the survival policy loss, detaches live state, and steps the
supplied optimizer. Set `survival_horizon` to a positive integer (TOML or
`--survival-horizon`) to finish after that many world steps, or earlier extinction.
The objective is each agent's survival reward through that horizon, with no
bootstrap or terminal bonus. Without a horizon (the API default), initial Life plus
total initial Points bounds collection to extinction; renewable training requires
an explicit horizon. Sampling streams continue across resets. Results report a
scalar loss, world steps, ending flags, and separate agent survival totals
without retaining training graphs. Use an optimizer over the collector's network parameters.

`survival_policy_loss(rollout)` is also available for complete episodes collected
from step zero. Every sampled message and action uses its owner's undiscounted
survival reward-to-go, including subsequent steps without encounters and the final
living step. The actor uses the detached advantage `return - value`; the critic
minimizes squared error to that return. Losses are summed over decisions and
averaged over the initial number of agents. The total loss is actor loss plus
`value_loss_coefficient * value_loss`, minus separate action and message entropy
bonuses. `value_loss_coefficient` defaults to 0.5 and must be positive;
`action_entropy_coefficient` and `message_entropy_coefficient` each default to 0.01
and must be nonnegative (zero disables the corresponding bonus). All coefficients
must be finite and can be set in TOML or through the matching CLI flags, such as
`--message-entropy-coefficient 0`. They are saved with the experiment settings.
Select `training_method = "actor_critic"` (the default) or `"reinforce"` in TOML,
or use `--training-method actor_critic|reinforce` with `train` or `compare`.
Legacy REINFORCE uses `-log_prob * survival_return`, without a value baseline,
value regression, or entropy bonuses; the existing coefficient settings are
validated and saved but ignored by that method. Both methods use the same network,
environment defaults, sampling streams, and complete survival objectives.
Training results, JSONL training records, and comparison updates include
`training_method`, total `loss`, and the unweighted, per-agent averaged components
`actor_loss`, `value_loss`, `action_entropy`, and `message_entropy`.
REINFORCE records zero for the three unused components; a disabled message
channel records zero message entropy. The configured coefficients reconstruct
the total Actor-Critic loss from these components.
Entropy regularizes the loss without changing survival rewards. Returns are never
pooled across agents; there are no communication,
GIVE, cooperation, or internal-state rewards. Memory, thought, and affect learn
through recurrent gradients from the same objective. A disabled channel has no
message loss but retains its internal-state update.

Incomplete episodes, truncated segments, and episodes without sampled decisions
are rejected by the loss. This minimal update uses full episode graphs and has
no truncation bootstrap. CPU tests verify finite losses,
per-agent credit, nonzero gradients and parameter updates, including memory and
affect feedback; they do not establish learned cooperation or communication.

## Persisted experiment observations

```python
from self_genesis.observation import RunRecorder

# Use a fresh path; existing files are never overwritten.
with RunRecorder("run.jsonl") as recorder:
    collector = RolloutCollector(config, network, recorder=recorder)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    result = train_episode(collector, optimizer)
```

The optional recorder writes versioned JSON Lines and flushes each record. It
also works with `collector.collect(max_steps=...)` for bounded inference or
collection. Each reset starts a numbered episode; construction records episode
zero, so `train_episode`'s explicit reset starts episode one. The file is the run
identifier. Use one recorder per collector and close it with the context manager.
Records contain ordinary numbers and lists, without retaining autograd graphs.

- `episode_start`: effective world settings (including seed), resolved device,
  policy/channel dimensions, initial Appearance, Life, Points, Working Memory,
  and affect latents, plus each agent's fixed `point_generation_probability`.
  Seed covers collection; callers still control initial
  network weights and must seed before network construction for repeatability.
- `step`: zero-based episode step, ordered encounter participants, both messages
  followed by both actions, each callback's observation, sampled choice, log
  probability, and memory/affect before and after. Rewards and new deaths cover
  every agent. Life/Point arrays are post-step population distributions in agent
  order; state arrays are captured before death clears the live adapter. No
  encounter produces empty participant and callback lists. `generated_points`
  records actual post-decay generation for every agent, including nonparticipants.
  `successful_transfers` lists directed `{donor, recipient}` events: each spends
  one donor Point and restores one recipient Life before decay. Empty lists mean
  no successful transfers. Together with initial resources, these fields permit
  per-agent reconstruction of Life and Points at every step.
- `summary`: cumulative episode deaths, observed lifetimes, zero-based death
  steps, token counts by token number, and sampled GIVE/NOTHING counts and
  fractions. The denominator is encounter action callbacks, excluding automatic
  NOTHING for unselected agents. GIVE counts include attempts with no Points;
  successful transfer events distinguish realized effects from those attempts.
  Fractions are null with no callbacks.
  Collection budget and termination/truncation flags describe this segment.
- `training`: the survival policy loss used for the completed update, per-agent
  survival returns, optimizer class, and parameter-group settings.

A surviving lifetime is explicitly right-censored at each collection boundary.
`mean_survival_time` is null until extinction; `mean_completed_lifetime` averages
only deaths (null when none), and `mean_observed_lifetime` includes the observed
ages of survivors. Continued segments update cumulative summaries; do not sum
summaries across segments. Raw step rewards permit independent reconstruction.
A reset or closing the file does not turn unfinished lifetimes into deaths.
Survivors at a completed survival horizon remain censored; horizon completion
does not imply extinction or make `mean_survival_time` available.
Agent and episode numbers are logging keys only and never enter policy inputs.
Generation abilities, generation outcomes, and transfer events are analysis-only
fields and are not added to policy observations.
The `train` command records aggregate diagnostics automatically; use
`--trace-worlds 0` to include full scalar observations.

## Configurable training command

```sh
python -m self_genesis train --config configs/default.toml --device cpu \
  --seed 42 --episodes 2 --learning-rate 0.001 --output run.jsonl
```

The default command (or explicit `init`) still prints initialization JSON.
`train` builds one shared recurrent network and Adam optimizer and performs
exactly `episodes` complete survival-policy updates. With zero generation, each episode is bounded by
`initial_life + num_agents * initial_points` world steps, so there is no
truncation bootstrap or open-ended loop. Full episode graphs are held in memory;
keep resource budgets small for exploratory runs.

All flat TOML settings can also be overridden with hyphenated CLI flags:
`--num-agents`, `--appearance-dim`, `--initial-life`, `--initial-points`,
`--vocabulary-size`, `--max-message-length`, `--memory-dim`, `--affect-dim`, `--entity-memory-dim`,
`--episodes`, `--learning-rate`, `--seed`, `--device`, `--survival-horizon`,
`--point-generation-probability-min`, `--point-generation-probability-max`,
`--training-method`, `--value-loss-coefficient`, `--action-entropy-coefficient`,
and `--message-entropy-coefficient`.
Vocabulary, memory, affect dimensions, and episode count must be positive
integers. Message length must be a nonnegative integer; zero disables messages.
Learning rate must be finite and positive. The runnable renewable settings are
listed in the sample configuration; API defaults keep zero generation and no
horizon. The objective remains undiscounted per-agent survival return.

Training seeds network initialization before moving weights to CPU/CUDA and seeds
collection once, preserving sampling streams across episode resets. Repeat runs
on the same device and software are reproducible subject to PyTorch backend
determinism; CPU and CUDA training need not match.

`--output` is required for training and must name a new file in an existing
directory. Existing results are never overwritten. Stdout reports JSON with the
absolute results path, effective settings, resolved device, completed episode
count, total steps, and last update. The JSONL file contains all episode settings,
summaries and learning metrics described above, plus observations when tracing is enabled. Training attaches
the recorder after collector construction, so only the requested episodes are
recorded, numbered from zero, each with a summary and training result. Read it with
`json.loads(line)` for each line. It is observation data, not a model checkpoint.
If interrupted, flushed records remain accessible but the run may be incomplete.

### Renewable Points

`configs/renewable.toml` enables scarce renewable Points for bounded
`RolloutCollector.collect(max_steps=...)` experiments and finite-horizon training:

```sh
python -m self_genesis train --config configs/renewable.toml --survival-horizon 100 --output renewable.jsonl
```

Each agent samples a lifetime-fixed probability uniformly between `point_generation_probability_min`
and `point_generation_probability_max` (inclusive bounds in [0, 1]). Equal bounds
set a constant probability; both zero reproduce the initial-Points-only world.
The fields also have matching CLI flags. API defaults remain zero for compatibility;
`configs/default.toml` and `configs/renewable.toml` both select 0.1–0.3 and horizon 100.
Horizon completion sets `horizon_completed=true`, `terminated=false`, and
`truncated=false` in the rollout and summary. Death on the horizon takes
precedence. Survivors retain their Life and are logged as censored, with no
fabricated death; `mean_survival_time` remains null. Interrupted collections
remain truncated and cannot be used as complete training episodes.

After simultaneous GIVE, Life decay, and death resolution, every survivor draws
0 or 1 new Point, including agents outside the Encounter and lone survivors.
`StepResult.generated_points` reports actual generation. New Points can only be
observed or spent next step and can only restore another agent's Life. Abilities
are world state, never policy inputs. Appearance, ability, and generation use
separate seeded streams with matching CPU/CUDA resource draws. Collector resets
restore the configured population and abilities while continuing generation,
encounter, and policy sampling streams; recreating a collector replays the run.

## Matched policy comparisons

```sh
python -m self_genesis compare --config configs/renewable.toml --device cpu \
  --episodes 2 --survival-horizon 100 --seed 42 --evaluation-seeds 101 102 \
  --output comparison.json
```

`compare` separately trains Entity Memory enabled (`learned`, configured positive
`entity_memory_dim`) and disabled (`learned-no-entity-memory`, dimension zero)
shared networks per training seed for the configured number of episodes using
only the existing survival objective. It then freezes the weights and evaluates
both learned conditions, always-GIVE, always-NOTHING, and producer-oracle populations separately
for every `--evaluation-seeds` value (default: the configured seed). Each evaluation starts
with fresh agent memory and a fresh world under identical resources, generation
probabilities, channel limits, horizon, and seed. Fixed policies send empty
messages; GIVE is attempted on every encounter, even without Points. All policies
use the existing encounter sampling, simultaneous transfer, decay, death, and
post-decay generation rules. Sampling streams restart for each policy; realized
encounters and generation draws can diverge as survival populations diverge.
Learned actions remain sampled, with no learning during evaluation.

The `producer-oracle` baseline sends empty messages and attempts GIVE exactly
when the partner's true generation probability is positive and at least the
midpoint of the configured generation-probability range; otherwise it chooses
NOTHING. Equality at a positive midpoint qualifies; zero-generation populations
always choose NOTHING. Like always-GIVE, it may attempt aid without Points and
relies on the world to enforce eligibility. The encounter protocol routes the actual
partner's generation probability through an oracle-only observation, so duplicate
Appearances cannot conflate abilities. This field is available only to this
evaluation baseline. Learned observations, training inputs, and rewards receive
no generation knowledge. This is a privileged heuristic, not an optimal policy
or a guarantee of improved survival. Use evaluation seeds distinct from the
training seed for held-out results, as in the example above.

The new JSON output file contains training settings and update results, evaluation
seeds, initial Appearances and generation abilities, and one result per policy and
seed. Results include per-agent survival returns and censored lifetimes, deaths,
final resources, action counts/fractions, successful aid counts, and the existing
prior-relationship action rows. Comparison history is keyed by the actual partner,
including when Appearances collide or are shuffled; identity and history never
enter learned observations. `relationship_metrics` groups action counts,
fractions, and successful aid by unseen partners, previously received aid, and
previously encountered partners without received aid. Only earlier steps determine
these groups; empty groups have null fractions. Action denominators count encounter
callbacks, including failed GIVE attempts, and exclude nonparticipants. Survivors
at the horizon remain censored; mean survival time is reported only at extinction.
These are descriptive comparisons, with no prescribed learned behavior or added
rewards. Repeatability applies within the same device and software environment.

For the initial-Points-only comparison, use the same command with
`--point-generation-probability-min 0 --point-generation-probability-max 0` and a
new output filename. Keep the other settings and seeds unchanged when comparing
resource regimes. Zero generation also supports omission of the horizon, running
to extinction; renewable comparison requires an explicit horizon. Existing output
files are never overwritten. The comparison JSON is separate from training JSONL
and does not go through `examples/analyze_run.py`.


For a small comparison of both learning methods, independent training seeds, and
all four evaluation interventions (use fresh output filenames):

```sh
for method in actor_critic reinforce; do
  python -m self_genesis compare --config configs/renewable.toml --device cpu \
    --training-method "$method" --value-loss-coefficient 0.5 \
    --action-entropy-coefficient 0.01 --message-entropy-coefficient 0.01 \
    --episodes 2 --survival-horizon 100 --training-seeds 41 42 43 \
    --evaluation-seeds 101 102 \
    --interventions appearance-shuffle appearance-replacement working-memory-reset entity-memory-reset \
    --output "$method-comparison.json"
done
```

This two-update smoke run is not the recorded 100-update experiment. Use the
[Entity Memory reproduction procedure and retained report](docs/entity-memory-experiment.md)
for the historical v0.0.6 validated findings. The historical
[learning-method comparison](docs/matched-learning-experiment.md#reproduction-and-retained-evidence)
requires its recorded checkout. Keep all conditions except `--training-method` matched;
REINFORCE ignores the value/entropy coefficients.

`--training-seeds` defaults to the configured seed. Each seed initializes and trains
each condition independently with the same training budget, optimizer settings,
world settings and seed. Dimensions differ, so equal seeds do not imply identical
initial weights or training trajectories. No trained weights are shared between
conditions. Explicit `--entity-memory-dim 0` retains a disabled-only `learned` run;
use a positive dimension (default 16) for the paired comparison. Evaluation seeds default to the
configured seed independently of this list; choose disjoint lists for held-out
comparisons. Interventions are optional and applied separately to the frozen
learned conditions, each alongside its own untreated evaluation and the three baselines.
The total training budget is twice the per-condition budget for paired runs.

- `appearance-shuffle`: before each world step, permute the original Appearance
  vectors of **all** agents using an isolated `random.Random(evaluation_seed + 3)`
  stream. The actual partner's index selects the presented vector. The permutation
  stays fixed across the encounter's communication and action callbacks, then is
  resampled. Fixed points and Appearances belonging to dead agents are allowed.
  This disrupts stable perceived identity across encounters; it does not mutate
  world Appearance, resources, abilities, encounter routing, or generation RNG.
- `working-memory-reset`: zero every agent's Working Memory before each world
  step, retaining Entity Memory, affect and all world state. Memory updates normally within the
  encounter. Affect can still carry history, so this is not a complete removal of
  recurrent information.
- `entity-memory-reset`: clear every agent's Entity Memory before each world step,
  preserving Working Memory, affect, weights and world state. Memory can be written
  and retrieved normally during communication, actions and encounter completion.
  Working Memory and affect can still carry history. On the separately trained
  disabled condition this is a no-op.
- `appearance-replacement`: before each world step, sample fresh independent
  uniform `[0, 1)` vectors of the configured Appearance dimension for all agents,
  using an isolated CPU `torch.Generator` seeded with `evaluation_seed + 4`.
  Present the partner-indexed vector consistently through communication, actions
  and completion, then resample next step. This uses the world's Appearance
  distribution but does not permute existing identities. Both Appearance
  interventions preserve all policy state, world state, and other RNG streams;
  subsequent policy updates use the presented Appearance as the Entity Memory key.

No intervention updates weights or changes rewards.

Comparison schema version 3 records `training_runs` (seed, policy, full condition
config and updates), `training_seeds`, treatment labels, recorded `communication_messages`,
`intervention_semantics`, `metric_semantics`, `summaries`, and
`intervention_effects`. The original `training` list remains available for a
single training seed and refers to the configured `learned` condition only; it is
null for multiple seeds. Each evaluation records its condition config, and each
intervention effect includes its learned policy label. Each summary pools evaluation
samples only within one training seed, policy, and intervention:

- GIVE collapse is a descriptive evaluation diagnostic: a GIVE fraction at least
  0.95 is `near_always_GIVE`, at most 0.05 is `near_always_NOTHING`, otherwise
  `mixed`. With no action callbacks it is `no_actions` and fractions are null.
  Counts accompany fractions; this label does not establish training-time collapse.
- Survival summaries retain death and censoring counts. `mean_observed_lifetime`
  includes horizon-censored lifetimes; `mean_survival_time` is null if any lifetime
  is censored. A horizon survivor is never counted as a death.
- `prior_aid_give_difference` is the GIVE fraction for previously aided partners
  minus that for encountered-but-unaided partners. Unseen partners are excluded,
  and the difference is null if either group has no samples. Only earlier steps
  enter history, even for the second action callback. This is descriptive dependence,
  not a causal estimate; actual identity, abilities, and histories stay analysis-only.
- `partner_history_metrics` conditions subsequent GIVE separately on earlier
  received and outgoing aid, with distinct attempt and successful-transfer bins.
  Direct conditions use repeat encounters. Third-party conditions use all
  encounters and count each participant's earlier incoming/outgoing aid with
  agents outside the current pair. Histories use actual directed identities,
  reset each episode, and exclude all current-step events.
  Each condition has `positive`, `zero`, and `unknown` bins and a
  `positive_minus_zero_give` difference. Missing transfer records propagate
  unknown success counts, while attempts remain countable.
- `partner_history_metrics.partner_producers` compares GIVE **to** high versus
  low producers separately at `first` and `repeat` encounters. High means the
  partner's fixed generation probability is strictly above the midpoint of the
  episode population's minimum and maximum probabilities; low includes ties.
  The row records this `producer_midpoint` and `partner_producer_bin`.
  Equal abilities leave high empty; missing abilities form an `unknown` bin.
  `high_minus_low_give` is reported for each encounter group, followed by
  `repeat_minus_first_producer_difference`. These are descriptive associations,
  not causal estimates or learning targets.
- All new bins include `action_callbacks` sample counts, action counts and
  fractions, successful aid totals, `successful_aid_known_samples`, and an
  explicit `missing_bin`. Empty fractions and differences lacking either
  comparison bin are null. Samples are callbacks, not independent agents.
  Metrics appear per evaluation, in pooled summaries, and in offline analysis.
- Communication reports callback and nonempty-message counts, token counts, mean
  length, and empirical token entropy in bits. Entropy is null with no tokens;
  mean length is null with no callbacks. Empty messages are counted, including a
  disabled channel. Fixed-length learned messages may show usage without utility.

`intervention_effects` reports intervention-minus-untreated differences in observed
lifetime, censoring count, and GIVE fraction for each matched training/evaluation
seed pair and learned condition. The detailed rows and summaries support history and Communication
comparisons. World initialization and sampling streams restart identically for
each evaluation; changed actions can subsequently change resources, survivors,
encounters, and generation draws. Reports contain the treatment and metric
semantics needed to interpret these differences, without adding policy inputs.

The resource-only `BatchedWorld` API runs independent worlds with tensor state;
policy, encounter, and training collection continue to use the existing APIs.
Supply an explicit seed per world. Life, Points, and generation probabilities
have shape `[world, agent]`; Appearance has shape `[world, agent, feature]`.

```python
import torch
from self_genesis.batched_world import BatchedWorld
from self_genesis.config import ExperimentConfig

worlds = BatchedWorld(
    ExperimentConfig(num_agents=2, survival_horizon=10), seeds=[41, 42]
)
# -1 is NOTHING; other entries are same-world GIVE recipient indices.
result = worlds.step(torch.tensor([[1, 0], [-1, -1]], device=worlds.state.life.device))
worlds.reset(0, seed=43)  # reset only this world's episode and random streams
```

`step` requires int64 targets on the state device and validates them before
mutation. Gifts resolve simultaneously before decay and survivor-only Point
regeneration. Rewards are per-agent living steps, without a GIVE bonus.
`terminated` marks extinction; `horizon_completed` marks a horizon reached with
survivors. Extinction on the horizon takes precedence. Completed rows freeze,
return zero new rewards/transfers/generation, and retain their completion flags
until reset. `successful_transfers` is a boolean `[world, donor]` mask whose
recipients are the input targets. Initialization matches each seed's sequential
`World`. Device-side Philox streams supply regeneration uniforms independently
of batch ordering or other worlds' deaths and resets. The batched generator
consumes a full agent row per active step, masking out dead agents; its draws
do not match the sequential survivor-only PyTorch RNG. Controlled-uniform tests
verify the shared resource semantics.

`BatchedEntityMemory` in `self_genesis.entity_memory` provides the corresponding
Appearance-keyed memory storage primitive. Keys have shape
`[world, observer, slot, appearance]`, values `[world, observer, slot, value]`,
and occupancy `[world, observer, slot]`. It is used by the batched policy API
below, the batched encounter executor, and the tensor trajectory collector.

Create it with `BatchedEntityMemory.empty(worlds, observers, slots, appearance_dim,
value_dim, device=..., dtype=...)`, reserving enough slots for each observer's
distinct partners. `retrieve(appearance, active=mask)` and
`write(appearance, value, active=mask)` accept one observed Appearance per
world/observer, with an optional boolean `[world, observer]` participation mask.
Unseen/inactive reads return zero; inactive writes preserve state. Equal
Appearances share one entry, irrespective of partner identity. New keys in a
full table raise an error; existing entries can still be updated. A zero value
dimension disables storage.

Writes return new state and preserve gradients through learned values and
previous encounters. Assign their result back to the memory variable. At each
episode boundary, also assign `memory = memory.reset(completed_worlds)` using a
boolean `[world]` mask alongside the world reset. This clears selected worlds'
keys and values while preserving ongoing worlds and their gradient history.
Call `detach()` only at an intentional training graph boundary.

### Batched policy forwards and capacity presets

`RecurrentPolicy` shares the same weights and recurrent computation between
sequential callbacks and `forward_batch`. The named capacities keep the existing
architecture: one thought Linear, one Working Memory GRUCell, one circulating
affect Linear, message/action/value Linear heads, and two Entity Memory GRUCells
(callback and resolved-encounter updates). Thought width equals Working Memory
width. All presets retain all components; no extra observations or labels are
introduced. The scalar critic reads the updated private Working Memory.

Exact shared parameter counts at `appearance_dim=8`, `vocabulary_size=4`,
`max_message_length=3` are:

| Preset | Working Memory / thought | Affect | Entity Memory value | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Small | 16 | 4 | 16 | 10,987 |
| Medium | 128 | 32 | 64 | 278,823 |
| Large | 512 | 128 | 256 | 4,211,847 |

Use `RecurrentPolicy.from_preset(8, "large")` or the configuration files
`configs/policy-small.toml`, `configs/policy-medium.toml`, and
`configs/policy-large.toml`. These files set only the three capacity dimensions;
other settings use the existing experiment defaults. Existing CLI dimension
flags override them, for example:

```bash
python -m self_genesis train --config configs/policy-large.toml \
  --device auto --episodes 1 --survival-horizon 10 --output large.jsonl
```

Python preset construction also accepts explicit dimension/channel overrides.
`network.parameter_count` computes the actual count after those overrides,
excluding per-agent state. Training summaries, recorded episode starts, and
comparison training runs report that count alongside the configured dimensions.
Changing Appearance/channel dimensions or disabling Entity Memory changes the
count. Capacity selection does not change the existing sequential trainer into
a batched trainer.

```python
import torch
from self_genesis.batched_policy import BatchedObservation
from self_genesis.policy import RecurrentPolicy

network = RecurrentPolicy.from_preset(8, "large")
state = network.initial_batch_state(2, 4, slots=4)
observation = BatchedObservation(
    resources=torch.ones(2, 4, 4),  # own Life/Points, partner Life/Points
    partner_appearance=torch.zeros(2, 4, 8),  # observed Appearance, never an ID
    first=torch.ones(2, 4, dtype=torch.bool),
    received_message=torch.full((2, 4, 3), 4, dtype=torch.int64),
    partner_action=torch.zeros(2, 4, dtype=torch.int64),
)
active = torch.tensor([[True, False, False, False], [False, True, False, False]])
logits, values, state = network.forward_batch(
    observation, state, communicating=True, active=active)
print(network.parameter_count)  # 4211847
```

All observations and masks must be on the policy device; Appearance and state
use its floating dtype. `partner_action` uses 0=unknown, 1=NOTHING, 2=GIVE.
Message tensors contain ordered tokens with a suffix padded by `vocabulary_size`.
Each call batches only the current callback phase across `[world, observer]`.
The caller must preserve communication and first/second action ordering, exposing
only actions already visible at that phase. World/observer indices route state;
they are never network features or Entity Memory keys.

Logits have shape `[world, observer, vocabulary_size]` during communication and
`[world, observer, 2]` during action selection; values have shape
`[world, observer]`. Inactive rows return zero outputs and preserve all state.
Sample only active rows; a zero-length channel produces no message decision,
although its callback still updates recurrent state, as in the sequential API.

After gifts resolve, call `complete_encounter_batch(observation, state,
action=actions, gave=gave, received=received, active=active)` and retain its result.
Here `actions` is int64 (0=NOTHING, 1=GIVE), gift outcomes are boolean, and all three
have shape `[world, observer]`. The observation may now reveal the partner's final
action. Completion updates only Appearance memory, without advancing Working
Memory or affect. Equal Appearances share entries. Reserve sufficient entity
slots for all distinct partners; overflow raises rather than evicting memories.

Use `state = state.reset(completed_worlds)` with a boolean `[world]` mask at episode
boundaries, and `state = state.detach()` only at an explicit optimizer boundary. Updates
preserve old states and recurrent gradients across encounters and callbacks.
CPU/CUDA parity tests compare batched outputs, completion writes and parameter
gradients with sequential callbacks; they also check masks, private gradients,
reset/detach, disabled memory and empty channels. No throughput or learning
improvement is claimed by these policy primitives.

Deterministic reference checks can be run with:

```bash
python -m unittest discover -s tests -p 'test_batched*.py' -v
```

Disjoint encounter pairs share four policy calls per world step: first messages,
replies, first actions, then second actions. One completion call follows simultaneous
world resolution, including participants that died or reached the horizon on that
step. Decisions are routed by `[world, observer]` with active masks; the pair list
retains consecutive first/second entries and `-1` padding. Each pair retains its
own random draws from its world's stream, and each observer retains private
recurrent and Entity Memory state. Pair selection consumes the existing per-pair
random blocks; batching phases changes neither network capacity nor learning rules.

The small-configuration reference coverage is split by responsibility:

| Tests | Reference comparisons |
| --- | --- |
| `test_batched_world.py` | Scalar `World` resource updates, simultaneous gifts, renewable Points, death, horizon precedence, frozen completed rows, and cumulative survival rewards. |
| `test_batched_encounter.py` | Scalar `EncounterProtocol` observations, ordered messages/replies and GIVE/NOTHING actions, completion writes, recurrent state, and gradients; batch composition/reset isolation includes affect and all Entity Memory tensors. |
| `test_batched_policy.py`, `test_batched_entity_memory.py` | Scalar policy and Appearance memory lookup/write results, repeated/colliding keys, inactive observers, disabled memory, reset/detach, and private recurrent/Entity Memory gradients. |
| `test_batched_rollout.py` | Scalar loss reduction for both training methods, reward accounting for unselected/lone survivors, continuation, and gradients. |
| `test_batched_training.py` | Independent scalar complete episodes, loss components, parameter gradients, and Adam updates. Covers extinction and finite horizons, empty/three-token channels, repeated updates, and renewable resources with successful/unaffordable GIVE and NOTHING. |
| `test_density_training_parity.py` | Independently scripted multi-pair episodes at fixed and fractional density: resources, routing, Communication, completion memory, scalar/batched traces, encounter counters, episode-normalized Actor-Critic losses, gradients and Adam updates across death and horizon boundaries. |

Batched traces include explicit `pairs` of first/second participants, excluding
unused slots, alongside the existing flat, `-1`-padded `participants` list.
Encounter counters count pairs of action decisions, including NOTHING; messages,
padding, finished worlds and steps without encounters do not add encounters.
Increasing density adds decision terms to the complete-episode objective without
renormalizing by encounters: losses still average over starting agents and worlds.

Policy initialization uses fixed seeds in the numerical comparisons. World tests
replay controlled generation uniforms for scalar survivors because the scalar
and batched random streams deliberately differ. Encounter comparisons replay
choices to isolate execution from sampling; the renewable training comparison
also scripts scalar pairs and token/action samples independently of batched
outputs. Endpoint uniforms select known tokens/actions, and alternating generation
uniforms exercise both renewal outcomes. One world dies at step one while another
reaches step five, checking normalization and rewards across unequal episodes.

Resources, routing, discrete choices, masks, and integer-valued survival rewards
must agree exactly. FP64 policy output/completion checks use `rtol=1e-9,
atol=1e-10`; policy gradients use `1e-8, 1e-10` and Encounter gradients use
`1e-7, 1e-9`. The renewable training comparison uses `rtol=2e-5, atol=2e-6`
for loss components and `rtol=2e-4, atol=2e-6` for gradients/Adam parameters,
matching the existing FP32 update tolerance: batched training intentionally
accumulates losses in FP32 even with FP64 policy weights. Batch-composition state
checks use `rtol=2e-5, atol=2e-6` for FP32 kernel roundoff. These are numerical
equivalence checks, not requirements for bitwise equality across execution shapes.
Device loops repeat reference comparisons on CUDA when available; CPU-only runs
do not establish CUDA or BF16 parity (see the separate mixed-precision tests).


### Batched Encounter execution

`BatchedEncounterProtocol` connects `BatchedWorld` and `RecurrentPolicy` without
per-agent Python policy callbacks:

```python
from self_genesis.batched_encounter import BatchedEncounterProtocol
from self_genesis.batched_world import BatchedWorld
from self_genesis.config import ExperimentConfig
from self_genesis.policy import RecurrentPolicy

config = ExperimentConfig(device="cpu")
world = BatchedWorld(config, seeds=[10, 20])
network = RecurrentPolicy(config.appearance_dim)
protocol = BatchedEncounterProtocol(world, network, seeds=[30, 40])
state = network.initial_batch_state(2, config.num_agents, slots=config.num_agents)
result = protocol.step(state)
state = result.state
```

Each unfinished world with at least two survivors selects one uniform ordered
living pair. Four batched phases preserve message, reply, first action and second
action order. Only the second action sees the first action; completion reveals
both final actions and successful transfers to the participants. Completion uses
pre-step resources, messages and observed Appearance, including for participants
who die at resolution. Unselected agents retain their state. Lone survivors
advance time without policy decisions; completed worlds remain frozen.

`result.pairs` contains routing indices (`-1` for no encounter), never policy
features. `result.decisions` holds the four phase observations, active masks,
choices and differentiable log probabilities, values and entropies; an entirely
inactive batch returns no phases. Empty channels still update recurrence but
have no sampled statistics. `result.world` contains the resource transition.
These records retain graphs without accumulating history inside the executor.

Per-world seeded device-side Philox streams feed tensor selection and sampling
on the world device, without consuming global RNG state. Changing or resetting another
world does not change a world's stream. These streams reproduce batched runs,
not the legacy Python encounter RNG sequence. Reset world resources with
`world.reset(row, seed=...)` and recurrent/entity state with `state = state.reset(mask)`;
encounter streams continue across resets. Use `state = state.detach()` only at an optimizer
boundary. This executor does not change the existing sequential training CLI.


### Batched survival trajectories

`BatchedRolloutCollector` owns the tensor world, encounter executor, and independent
recurrent state for each world and agent. Collection budgets return segments;
join consecutive segments to retain the entire survival objective and graph:

```python
from self_genesis.batched_rollout import BatchedRolloutCollector
from self_genesis.training import survival_policy_loss

collector = BatchedRolloutCollector(config, network, seeds=[11, 22])
rollout = collector.collect(32)
while bool(rollout.truncated.any()):
    rollout = rollout.extend(collector.collect(32))
optimizer.zero_grad(set_to_none=True)
survival_policy_loss(rollout).backward()
collector.detach()
optimizer.step()
```

Use a finite `survival_horizon` for renewable worlds. Finished rows freeze while
other worlds continue. Rewards include unselected agents and lone survivors;
horizon completion is distinct from death. The existing survival loss functions
accept tensor rollouts and average the per-world objectives, each normalized by
its starting agent count. They reject incomplete budgets and segments missing
step zero. Joining requires contiguous segments from the same collector and
episodes. Empty channels have no message loss but still advance recurrence.

After consuming the objective, `reset(world, seed=...)` clears one world's
resources and recurrent state while preserving other rows and their graphs.
Encounter sampling streams persist through reset. No collection call implicitly
detaches state or updates weights, and `detach()` rejects unfinished objectives.
Reset every row before collecting the next full batch for a shared-weight update.
This API does not change the existing scalar training command.

### Batched optimizer updates

`train_batch` resets all rows, collects complete episodes under unchanged shared
weights, then performs one optimizer update using the existing batched survival
loss. Pass one reset seed per world; encounter sampling streams persist across
calls. Renewable training requires an explicit `survival_horizon`.

```python
from self_genesis.training import train_batch

optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
for _ in range(config.episodes):
    result = train_batch(collector, optimizer, seeds=[11, 22])
```

Actor-Critic uses each agent's undiscounted survival return, detached advantages,
value regression, and separately configured action and message entropy bonuses.
Each world's loss is normalized by its starting agent count and averaged across
worlds, including worlds that finish early. REINFORCE remains supported through
`config.training_method`. Recurrent and entity-memory graphs span the complete
objective and are detached before weights change. Results contain plain Python
values: aggregate loss components, steps and ending flags by world, and survival
returns indexed by world then agent. The training CLI can select this batched
path as described below.


### Batched training command

```sh
python -m self_genesis train --batched --num-worlds 64 --capacity-preset small \
  --device auto --seed 42 --deterministic --episodes 3 \
  --survival-horizon 8 --output batched-run.jsonl
```

Optional CUDA BF16 autocast is available with `--mixed-precision bf16` (or
`mixed_precision = "bf16"` in TOML) together with `--batched --device cuda`.
`--device auto` also works when it resolves to a BF16-capable CUDA device;
CPU and unsupported CUDA devices are rejected before creating output.
FP32 remains the default (`--mixed-precision fp32`). Eligible policy operations
use BF16 while parameters, optimizer state, recurrent/Entity Memory storage,
value-head evaluation, probability/entropy calculations, reward accumulation,
advantages and loss reductions remain FP32. Backward and optimizer updates run
outside autocast; non-finite losses or gradients abort the update. BF16 uses no
FP16 loss scaler. The selected precision is recorded in the run configuration.

The mixed-precision tests compare finite losses and parameter gradients against
FP32 on matched discrete trajectories, including repeated updates, Entity Memory,
zero-length messages and 32-step survival horizons. Run them on CUDA hardware
with `python -m unittest discover -s tests -p test_mixed_precision.py -v`.
These bounded comparisons do not establish stability for all capacities or long
survival horizons; validate representative runs against FP32 before considering
a change to the default. Identical seeds need not produce identical trajectories
across precisions when rounding changes a sampled decision.

Use `configs/batched.toml` for equivalent reusable settings. Sequential training
remains the default; `--no-batched` selects it explicitly. In batched mode,
`episodes` counts complete batch updates: three updates with 64 worlds train
192 episodes. Each update averages the complete world objectives and updates
one shared policy. `num_worlds` defaults to 64 and accepts any positive integer;
64–256 are practical starting counts, with no fixed-size limit at 512–1024.
Actual memory use depends on policy capacity, agent count and full episode length.

`capacity_preset` in TOML or `--capacity-preset small|medium|large` selects the
existing policy widths. A CLI preset overrides the TOML preset; explicit width
settings in TOML override preset defaults, and CLI widths override both.
`seed`, `device`, `deterministic`, `batched`, and `num_worlds` are also TOML settings.
`--deterministic` enables PyTorch deterministic algorithms during training and
sets the CUDA workspace configuration when unset before CUDA initialization;
unsupported deterministic operations raise an error. For an existing Python
process that has already used CUDA, start the process with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in its environment. `--no-deterministic` disables this requirement.
Backend algorithm settings are restored after the run.

Each world starts with seed `(seed + world_index) % 2**63`. Update `u` resets
resources with `(seed + u * num_worlds + world_index) % 2**63`; encounter/policy
streams persist across updates. Regeneration uses a separate Philox stream.
The Philox4x32-10 uniforms are generated in batched tensor operations on the
selected device, consume four-value blocks, and match across CPU/CUDA. Inactive
worlds do not advance their streams. Policy floating-point results and training
need not match across devices or software versions.

Batched JSONL contains one `batch_run` record with resolved settings, device,
parameter count, RNG algorithm and initial encounter seeds, followed by one
`batch_training` record per update. Each update records reset seeds, losses,
steps and completion flags indexed by world, and survival returns indexed by
world then agent. By default this compact format does not contain detailed encounter traces
and is not input for the scalar `examples/analyze_run.py` tool. The final CLI
JSON reports update count, total world episodes and total world steps.

### Training diagnostics and sampled traces

Training defaults to aggregate records without detailed traces. Every update
includes loss components and `metrics`: mean observed survival (including
horizon-censored agents), GIVE choices divided by all action choices (not
successful transfers), mean squared value error against undiscounted survival
reward-to-go per sampled decision, mean action/message entropy in nats per
sampled decision, and the global L2 gradient norm before the optimizer step.
Message entropy is for the whole message. An empty message channel has a null
mean entropy. REINFORCE still reports these diagnostics, although its value
head is untrained and its unused loss components remain zero.

Use `--trace-worlds 0 3 --trace-update-interval 10 --trace-step-interval 5`
with batched training to record only worlds 0 and 3, updates 0, 10, 20, ...,
and steps 0, 5, 10, .... TOML equivalents are `trace_worlds = [0, 3]`,
`trace_update_interval = 10`, and `trace_step_interval = 5`. Defaults are an
empty world list and intervals of 1; scalar training accepts only world 0.
Sampled records contain detailed choices, observations, resources, latent
states and Entity Memory. Batched snapshots show post-step state, with
`occupied` marking valid Entity Memory slots. Completed worlds are not
repeated. Sampling does not change policy inputs, rewards or aggregate metrics.

Each update also records elapsed wall seconds, completed world steps per
second, and CUDA peak allocated/reserved bytes (null on CPU). Measurement
metadata names the device/GPU, PyTorch/CUDA versions, timing scope and allocator
scope. CUDA synchronizes at measurement boundaries; CPU uses wall timing.
Timing includes reset, rollout, backward, diagnostics, optimizer work and
sampled trace I/O and result construction (including scalar training record I/O),
but excludes construction and final measurement serialization.
Allocator peaks include existing process allocations, not just this update's
new tensors. GPU utilization is explicitly not sampled. These measurements
vary across otherwise reproducible runs and are not benchmark claims.

Evaluation `RunRecorder` and comparison relationship analysis retain full
recording independent of training sampling settings. For full scalar training
analysis with `examples/analyze_run.py`, use `--trace-worlds 0` with both
intervals set to 1. The analyzer rejects incomplete sampled traces rather than
reporting partial relationship histories as complete. Batched traces use a
separate `batch_trace` format and are not input to that scalar analyzer.
