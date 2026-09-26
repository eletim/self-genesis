# self-genesis

[Canonical design principles](docs/design-principles.md)

[Runnable CPU and RTX 5090 experiments, analysis, and validation](docs/minimal-experiment.md)

[Historical v0.0.4 renewable comparison and observed behavior](docs/renewable-experiment.md)

[Validated v0.0.5 matched Actor-Critic versus REINFORCE evidence](docs/matched-learning-experiment.md)

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
own separate Working Memory, affect tensors, and sampled-decision log
probabilities. The network itself holds only weights. Its `forward` method
also accepts and returns explicit `PolicyState` tensors for inspection.

Each communication or action callback encodes resources, partner Appearance,
role, received message positions, available partner action, and callback phase.
A thought layer consumes this observation plus previous memory and affect;
a GRU updates memory using thought and previous affect. New affect is generated
from the observation, thought, and updated memory, then feeds the next callback.
Affect dimensions have no predefined meanings or supervised targets. No agent
indices, self labels, or auxiliary classification objectives are added.

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
The `train` command also records these observations automatically.

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
`--vocabulary-size`, `--max-message-length`, `--memory-dim`, `--affect-dim`,
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
observations, summaries, and learning metrics described above. Training attaches
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

`compare` trains one shared network per training seed for the configured number of episodes using
only the existing survival objective. It then freezes the weights and evaluates
learned, always-GIVE, always-NOTHING, and producer-oracle populations separately
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
both evaluation interventions (use fresh output filenames):

```sh
for method in actor_critic reinforce; do
  python -m self_genesis compare --config configs/renewable.toml --device cpu \
    --training-method "$method" --value-loss-coefficient 0.5 \
    --action-entropy-coefficient 0.01 --message-entropy-coefficient 0.01 \
    --episodes 2 --survival-horizon 100 --training-seeds 41 42 43 \
    --evaluation-seeds 101 102 --interventions appearance-shuffle working-memory-reset \
    --output "$method-comparison.json"
done
```

This two-update smoke run is not the recorded 100-update experiment. Use the
[full reproduction procedure and retained reports](docs/matched-learning-experiment.md#reproduction-and-retained-evidence)
for the validated findings. Keep all conditions except `--training-method` matched;
REINFORCE ignores the value/entropy coefficients.

`--training-seeds` defaults to the configured seed. Each seed initializes and trains
its own network with the same training budget. Evaluation seeds default to the
configured seed independently of this list; choose disjoint lists for held-out
comparisons. Interventions are optional and applied separately to the frozen
learned policy, alongside its untreated evaluation and the three baselines.

- `appearance-shuffle`: before each world step, permute the original Appearance
  vectors of **all** agents using an isolated `random.Random(evaluation_seed + 3)`
  stream. The actual partner's index selects the presented vector. The permutation
  stays fixed across the encounter's communication and action callbacks, then is
  resampled. Fixed points and Appearances belonging to dead agents are allowed.
  This disrupts stable perceived identity across encounters; it does not mutate
  world Appearance, resources, abilities, encounter routing, or generation RNG.
- `working-memory-reset`: zero every agent's Working Memory before each world
  step, retaining affect and all world state. Memory updates normally within the
  encounter. Affect can still carry history, so this is not a complete removal of
  recurrent information. Neither treatment updates weights or changes rewards.

Comparison schema version 2 adds `training_runs` (seed and updates),
`training_seeds`, treatment labels, recorded `communication_messages`,
`intervention_semantics`, `metric_semantics`, `summaries`, and
`intervention_effects`. The original `training` list remains available for a
single training seed; it is null for multiple seeds. Each summary pools evaluation
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
- Communication reports callback and nonempty-message counts, token counts, mean
  length, and empirical token entropy in bits. Entropy is null with no tokens;
  mean length is null with no callbacks. Empty messages are counted, including a
  disabled channel. Fixed-length learned messages may show usage without utility.

`intervention_effects` reports intervention-minus-untreated differences in observed
lifetime, censoring count, and GIVE fraction for each matched training/evaluation
seed pair. The detailed rows and summaries support history and Communication
comparisons. World initialization and sampling streams restart identically for
each evaluation; changed actions can subsequently change resources, survivors,
encounters, and generation draws. Reports contain the treatment and metric
semantics needed to interpret these differences, without adding policy inputs.
