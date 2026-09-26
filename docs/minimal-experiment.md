# Run and inspect the minimal experiment

Run these commands from the repository root on Linux with Python 3.12. The
examples use one shared recurrent policy, independent agent memory/affect,
a bounded token channel, scarce renewable Points, and survival-only REINFORCE. They do
not add social rewards, identity labels, or auxiliary objectives.

For the matched v0.0.4 comparison, reproduction commands, and observed policy
behavior, see [the renewable experiment record](renewable-experiment.md).

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
  --survival-horizon 12 \
  --output /tmp/self-genesis-runs/cpu.jsonl
python examples/analyze_run.py /tmp/self-genesis-runs/cpu.jsonl
```

Choose a fresh output filename on every run: existing files are rejected.
This uses four agents, eight Appearance dimensions, four tokens, three tokens
per message, 16 memory dimensions, four affect dimensions, and Adam at 0.001.
Generation abilities are sampled in [0.1, 0.3]. Each of the two updates lasts
at most 12 world steps, ending earlier at extinction. Nothing is
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
  --survival-horizon 12 \
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
arbitrary truncated API rollouts. It streams raw records and retains the
per-episode output, including action analysis rows, before printing.

Each update also includes `relationship_actions`, one row per sampled action.
Rows link the current `action` (GIVE/NOTHING) and `successful_aid` to the actor's
and partner's generation probabilities and cumulative `*_prior_generated_points`.
The `prior` object counts encounters, received GIVE attempts, received NOTHING,
received successful aid, outgoing GIVE attempts, and outgoing successful aid.
Histories are directed, keyed by observer and exact partner Appearance, and reset
each episode. Identical Appearances are consequently indistinguishable in these
histories; agent/partner indices remain available as logging references.
Both actions use only earlier steps: the current partner action, successful
transfers, and newly generated Points enter history after both rows are emitted.
Non-increasing step numbers are rejected.

For temporal comparisons, group rows by `prior.encounters == 0` (unseen) versus
`> 0` (repeated), or compare subsequent GIVE fractions for rows with
`prior.received_give_attempts > 0` against rows with prior encounters but no
received GIVE attempts. `prior.received_nothing` also identifies explicit past
non-GIVE, including partners with mixed histories. Compare `prior.received_aid`
separately to distinguish successful help from attempts, and
`prior.outgoing_give_attempts`/`prior.outgoing_aid` for previous outgoing help.
Generation probabilities and prior generation totals support trait/history
comparisons without leaking the current generation outcome into a predictor.
These are descriptive associations, not evidence of causal reciprocity or policy
access to hidden generation traits. For older schema-1 logs lacking generation
or transfer records, unavailable values are `null`, not assumed zero/successful;
attempt and encounter histories remain available.

For deeper inspection, load each line with `json.loads`. `step.participants`
links encounter order to agent logging indices; `callbacks` holds ordered
messages/actions, observations, and memory/affect before and after each callback.
`step.life` and `step.points` are post-step population arrays. `episode_start`
contains initial Appearance and effective settings. Agent indices are analysis
keys only, never policy inputs. Each `summary` is cumulative: do not sum repeated
summaries from continued API collection. Surviving lifetimes at truncation are
censored, not completed deaths. The same applies at a completed survival horizon:
`mean_survival_time` is null while anyone survives. Inspect summary `lifetimes`,
`mean_observed_lifetime`, `terminated`, and `horizon_completed` to distinguish
censored finite-horizon returns from completed lifetimes. GIVE counts are attempts, including attempts
without Points; `successful_transfers` identifies actual directed aid.

```sh
python -m unittest discover -s tests -p 'test*integration.py' -v
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
python -m pip check
```

The integration tests run training and analysis in separate processes, reconstruct
per-agent returns, the survival policy loss, deaths, and message/action counts
from raw steps, check finite latent states, and reject interrupted output.
Renewable integration tests additionally reconstruct per-agent transfers and
generation, verify prior-only relationship rows and hidden-ability isolation,
exercise horizon censoring and extinction, and match CLI training results to
repeated learned/fixed-policy comparisons. Existing training tests verify
gradients and actual parameter updates, including memory and affect feedback. CUDA cases skip when CUDA is unavailable; a skipped
case is not hardware validation. Do not compare exact CPU and CUDA trajectories:
backend sampling and numerical results can differ. These tiny runs establish
execution and observable training, not learned cooperation, communication
semantics, or self-representation.

## Historical v0.0.3 validation record

The following historical results predate the fix that preserves sampling streams
across episode resets. Exact trajectories and losses change with that fix.

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
