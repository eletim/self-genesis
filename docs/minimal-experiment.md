# Run and inspect the minimal experiment

Run these commands from the repository root on Linux with Python 3.12. The
examples use one shared recurrent policy, independent agent memory/affect,
a bounded token channel, finite Points, and survival-only REINFORCE. They do
not add social rewards, identity labels, or auxiliary objectives.

## Small CPU run

Use a fresh environment so an existing CUDA or nightly installation cannot
silently satisfy the dependency. The package supports a wider PyTorch range;
these recipes pin a common version for repeatable setup.

```sh
python3.12 -m venv /tmp/self-genesis-cpu-env
. /tmp/self-genesis-cpu-env/bin/activate
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'numpy==2.2.6' -e .
python -m pip check
mkdir -p /tmp/self-genesis-runs
python -m self_genesis train --config configs/default.toml \
  --device cpu --seed 42 --episodes 2 --initial-life 3 --initial-points 1 \
  --output /tmp/self-genesis-runs/cpu.jsonl
python examples/analyze_run.py /tmp/self-genesis-runs/cpu.jsonl
```

Choose a fresh output filename on every run: existing files are rejected.
This uses four agents, eight Appearance dimensions, four tokens, three tokens
per message, 16 memory dimensions, four affect dimensions, and Adam at 0.001.
Each of the two updates lasts at most `3 + 4 * 1 = 7` world steps. Nothing is
saved in the repository by these run commands.

## Single RTX 5090 run

The RTX 5090 requires a Blackwell-capable wheel. PyTorch introduced Blackwell
support and CUDA 12.8 wheels in [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/).
Use the official [2.7.1 CUDA 12.8 wheel recipe](https://pytorch.org/get-started/previous-versions/)
with a compatible NVIDIA driver. The CPU wheel and older CUDA builds are not
suitable for this GPU. Torchvision and torchaudio are unnecessary here.

```sh
python3.12 -m venv /tmp/self-genesis-cu128-env
. /tmp/self-genesis-cu128-env/bin/activate
python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy==2.2.6' -e .
python -m pip check
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable; check wheel and driver'
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
x = torch.ones(4, device='cuda', requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
print('CUDA forward/backward:', x.grad.tolist())
PY
mkdir -p /tmp/self-genesis-runs
python -m self_genesis train --config configs/default.toml \
  --device cuda --seed 42 --episodes 2 --initial-life 3 --initial-points 1 \
  --output /tmp/self-genesis-runs/rtx5090.jsonl
python examples/analyze_run.py /tmp/self-genesis-runs/rtx5090.jsonl
```

Use explicit `cuda` to detect unavailable hardware; `auto` may fall back to CPU.
The driver CUDA version shown by `nvidia-smi` differs from the wheel runtime
reported by `torch.version.cuda`. An availability check alone does not prove
kernel or backward compatibility; run the training integration tests too.

This deliberately uses the same small configuration on both devices. Full
episode autograd graphs and detailed JSONL state records grow with episode
length and policy size. Python encounter dispatch and GPU-to-CPU logging can
dominate this small workload; GPU speedup is not an acceptance criterion.
Increase one budget at a time only after inspecting the small run.

## Analysis and verification

`examples/analyze_run.py` is a standard-library example for complete CLI training
files, schema version 1. It prints one JSON row per update with loss, individual
survival returns, mean survival, deaths, action counts/ratios, token counts, and
final resources. It rejects unfinished training runs; it is not an analyzer for
arbitrary truncated API rollouts. It streams raw records and retains only the
small per-episode output rows before printing.

For deeper inspection, load each line with `json.loads`. `step.participants`
links encounter order to agent logging indices; `callbacks` holds ordered
messages/actions, observations, and memory/affect before and after each callback.
`step.life` and `step.points` are post-step population arrays. `episode_start`
contains initial Appearance and effective settings. Agent indices are analysis
keys only, never policy inputs. Each `summary` is cumulative: do not sum repeated
summaries from continued API collection. Surviving lifetimes at truncation are
censored, not completed deaths. GIVE counts are attempts, including attempts
without Points; inspect resource changes to identify actual transfers.

```sh
python -m unittest discover -s tests -p 'test_experiment_integration.py' -v
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
python -m pip check
```

The integration tests run training and analysis in separate processes, reconstruct
per-agent returns, the survival policy loss, deaths, and message/action counts
from raw steps, check finite latent states, and reject interrupted output.
Existing training tests verify gradients and actual parameter updates, including
memory and affect feedback. CUDA cases skip when CUDA is unavailable; a skipped
case is not hardware validation. Do not compare exact CPU and CUDA trajectories:
backend sampling and numerical results can differ. These tiny runs establish
execution and observable training, not learned cooperation, communication
semantics, or self-representation.

## Validation record

On 2026-09-26, Linux x86_64, Python 3.12.14, NumPy 2.2.6:

- PyTorch 2.7.1+cpu: all 44 tests completed successfully (four CUDA tests skipped).
  The two-episode CPU recipe completed eight world steps, with survival returns
  `[4, 3, 4, 3]` and mean lifetime 3.5 for each episode. Final losses were finite;
  the second update's loss was approximately 21.4918. These are observed outputs
  for this software/seed, not acceptance thresholds for other backends.
- PyTorch 2.7.1+cu128 (CUDA 12.8), one NVIDIA GeForce RTX 5090,
  capability 12.0, driver 595.91.07: all 44 tests passed with no skips. CUDA
  forward/backward and the two-episode training/analysis recipe also passed. The run completed ten world steps with returns
  `[5, 4, 3, 3]` and mean lifetime 3.75 per episode; the second loss was approximately
  25.8135. The recorder reports the concrete device `cuda:0` while CLI stdout
  reports the requested resolved device type `cuda`.
- Byte compilation and dependency consistency checks passed in both environments.

Large population/long episode throughput, peak VRAM limits, multi-GPU execution,
other operating systems, older drivers, and scientific convergence remain
unverified. A successful tiny experiment does not establish those limits.
