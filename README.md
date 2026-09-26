# self-genesis

[Canonical design principles](docs/design-principles.md)

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
Point budget. Life and appearance dimension must be positive integers; Points
may be zero. Seeds are integers from 0 through `2**63 - 1`. Without `--config`,
the same defaults are used. `--seed` and `--device` override file settings.
Unknown configuration keys and invalid values fail with an error.

Device choices are `cpu` (default), `cuda` (fails if unavailable), and `auto`
(CUDA when available, otherwise CPU). The run seeds Python and PyTorch and
initializes separate Life, Point, and fixed Appearance values for multiple
agents. Appearance is sampled on CPU and transferred to the selected device,
so the same seed produces the same initial state on CPU and CUDA within the
same software version. This does not promise deterministic future training or
identical results across PyTorch versions.

Each run prints JSON containing the effective conditions, resolved device, and
initial agent state. The CLI remains an initialization smoke experiment.
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
Points never regenerate or transfer to recipients, and Appearance never changes.

`StepResult.reward` gives each agent alive at the start of the step one reward,
including its final step: accumulated reward is its lifetime in world steps.
There are no GIVE, receipt, cooperation, or other social bonuses.
`StepResult.died` marks new deaths, and `done` becomes true when everyone is dead.
Further steps return zero rewards and no new deaths. Finite initial Points
bound how much Life can be restored, so even mutual giving cannot sustain the
world indefinitely.

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
reward-to-go. Discrete samples themselves are not differentiable. The tests
exercise a complete episode and optimizer update using only world survival
rewards; this is a trainability check, not evidence of learned cooperation.
The CLI remains an initialization smoke experiment, not a training runner.

Call `agent.reset()` at episode boundaries to clear state and experience. For
truncated backpropagation, consume the pending loss before calling
`agent.detach()` to preserve state values while dropping their graph and clearing
log probabilities. Reset or detach before collecting another segment after an
optimizer update. Construct adapters after moving the network to its device;
use `torch.no_grad()` for inference and clear accumulated log probabilities as
needed. Sampling uses PyTorch's RNG (`torch.manual_seed` controls it).

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

`segment.terminated` means extinction. `segment.truncated` means the collection
budget ran out while survivors remain; extinction on the last allowed step takes
precedence. Calling `collect` again continues the same episode, step counter,
and recurrent graph. Calling it after extinction returns an empty terminated
segment. Budgets must be positive integers. `collector.reset()` explicitly
restores the configured world and seed and clears all live agent state; it does
not change network weights. Like world initialization, reset reseeds PyTorch's
global sampler, so identical weights reproduce the configured episode.

For survival learning, accumulate each agent's rewards backward through its
trajectory and weight each message/action log probability by that agent's
reward-to-go, including later steps without encounters. Join consecutive segments
for complete episode returns, or supply an appropriate estimated future return
at a truncation boundary; a time limit must not be treated as death. No social
reward or learning objective is added by collection.

Consume the pending loss before `collector.detach()` and an optimizer update.
Detach preserves recurrent values while cutting history for subsequent segments;
reset starts fresh instead. Returned records retain their graphs until released,
so discard consumed segments to free memory. Collection never silently detaches
at a budget boundary. Use `torch.no_grad()` for inference. The CLI remains an
initialization smoke experiment.

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
supplied optimizer. Initial Life plus the world's total initial Points bounds
collection to extinction. Each reset uses the configured seed. Results report a
scalar loss, world steps, and separate agent survival totals without retaining
training graphs. Use an optimizer over the collector's network parameters.

`survival_policy_loss(rollout)` is also available for complete episodes collected
from step zero. It weights every sampled message and action by its owner's
undiscounted survival reward-to-go, including subsequent steps without encounters
and the final living step. Losses are summed over decisions and averaged over
agents. Returns are never pooled across agents; there are no communication,
GIVE, cooperation, or internal-state rewards. Memory, thought, and affect learn
through recurrent gradients from the same objective. A disabled channel has no
message loss but retains its internal-state update.

Incomplete episodes, truncated segments, and episodes without sampled decisions
are rejected by the loss. This minimal update uses full episode graphs and has
no value baseline or truncation bootstrap. CPU tests verify finite losses,
per-agent credit, nonzero gradients and parameter updates, including memory and
affect feedback; they do not establish learned cooperation or communication.
