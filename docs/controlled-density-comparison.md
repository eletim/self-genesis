# Controlled encounter-density comparison

## Predeclared protocol

Declared before execution on 2026-09-28. Compare requested maxima of 1/4/8/16
pairs per world step, with 32 agents and identical small capacity, FP32,
Actor-Critic, survival-only reward, and renewable resource world. Full settings
are in `configs/controlled-density.toml`; no tuning or outcome-based stopping.

Each density trains independently from initialization seeds 10000/20000/30000,
for exactly 50 complete-batch updates of 128 worlds, horizon 32. Each seed uses
world seeds `seed + update * 128 + row`: disjoint ranges 10000–16399,
20000–26399, 30000–36399. The same initialization/world seeds are paired across
densities. Evaluation seeds 100000/100001/100002 are outside every training
range. They are reused across frozen policies and interventions, with fresh
world and policy state. Evaluation uses the sequential FP32 reference at the
training density and horizon. Sampling streams restart; trajectories can diverge.

The existing comparison also independently trains the no-Entity-Memory control
at each seed. Its capacity differs by design; compare densities within each
architecture. Main learned capacity stays 16/4/16 (Working Memory/affect/Entity
Memory); the control is 16/4/0. Fixed always-GIVE, always-NOTHING and producer
oracle baselines receive the same evaluation worlds. Run all four interventions:
Entity Memory reset, Appearance shuffle, Appearance replacement, Working Memory
reset, on both learned architectures.

Budget: 24 training runs, 1200 optimizer updates, 153600 training world episodes,
at most 4915200 world steps; 468 evaluation episodes (including repeated fixed
baselines), at most 14976 evaluation world steps. Each density has a 900-second
wall timeout including evaluation; total worker budget 3600 seconds. Failures
are recorded, never silently replaced with shorter runs. Equal updates/episodes
are the primary budget, not equal interaction exposure or compute. Report actual
world steps and encounters from training, plus held-out exposure distributions.

Primary interpretation: increased survival alone does not establish memory use.
Look for increased repeats, history-conditioned GIVE, repeat-versus-first producer
selection, and matched degradation under Entity Memory and identity interventions.
Retain empty-bin nulls and callback counts. Three training seeds are descriptive
replicates, not enough to claim robust statistical significance. Horizon survivors
are right censored. Communication entropy measures usage, not utility.

Reproduce from the repository root in a CUDA PyTorch environment:

```sh
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 /tmp/self-genesis-verify-gpu/bin/python \
  examples/run_density_comparison.py --output /tmp/controlled-density
```

Output directory must be new. The manifest is written before any worker starts.
Raw compressed reports preserve every update, evaluation, exposure distribution,
behavior summary, matched intervention effect and intervention semantics. Wall
measurement includes training and sequential evaluation; it is not GPU training
throughput. See [the density benchmark](rtx5090-density.md) for separately measured
training throughput, utilization and VRAM under its explicitly shorter horizon.
