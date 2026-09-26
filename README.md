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
Shared world dynamics are available through the Python API below; encounters,
communication, neural networks, and learning are not implemented yet.

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

Run the checks from the repository root:

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

CUDA initialization and world steps are tested when CUDA is available; device
selection and the unavailable-CUDA error path are also tested on CPU hosts.
