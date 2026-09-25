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
initial agent state. This is an initialization smoke experiment; encounters,
actions, rewards, neural networks, and learning are not implemented yet. It
introduces no self labels, communication meanings, or social reward bonuses.

Run the checks from the repository root:

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

CUDA initialization is tested when CUDA is available; device selection and the
unavailable-CUDA error path are also tested on CPU hosts.
